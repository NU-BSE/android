#!/usr/bin/env python3
"""Calibrate GUI-Owl layer/channel/head saliency before structural pruning.

The output is deliberately a compact JSON manifest rather than a checkpoint.
It records activation statistics from the trained 2B teacher and deterministic
ranked selections used by the progressive depth builder and the 0.5B structured
width-pruning builder.  Width pruning is marked ready only when image examples
actually exercised the vision tower.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from android_drlqn.gui_owl_student import (
    derive_width_selections,
    load_student_plan,
    plan_student_specs,
    resolve_peft_adapter_dir,
    TEXT_TAIL_PROTECTED_LAYERS,
    select_salient_layer_ids,
)


@dataclass(frozen=True)
class CalibrationRecord:
    mode: str
    messages: list[object]


def _load_jsonl(path: Path, mode: str, limit: int | None) -> list[CalibrationRecord]:
    output: list[CalibrationRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        messages = value.get("messages") if isinstance(value, Mapping) else None
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError(f"{path}:{line_number}: messages must be an array")
        if len(messages) < 2 or not isinstance(messages[-1], Mapping):
            raise ValueError(f"{path}:{line_number}: assistant completion is missing")
        if str(messages[-1].get("role") or "") != "assistant":
            raise ValueError(f"{path}:{line_number}: final message must be assistant")
        output.append(CalibrationRecord(mode=mode, messages=list(messages)))
        if limit is not None and len(output) >= limit:
            break
    if not output:
        raise ValueError(f"calibration corpus is empty: {path}")
    return output


def _records(plan: Mapping[str, object], limit: int | None) -> list[CalibrationRecord]:
    corpora = plan.get("specialized_corpora")
    if not isinstance(corpora, Mapping):
        corpora = plan.get("corpora")
    if not isinstance(corpora, Mapping):
        raise ValueError("student plan must define specialized_corpora")
    result: list[CalibrationRecord] = []
    for mode in ("gui_policy", "chat"):
        raw = corpora.get(mode)
        if not isinstance(raw, Mapping):
            raise ValueError(f"student plan is missing {mode} corpus")
        path = Path(str(raw.get("path") or ""))
        if not path.is_file():
            raise FileNotFoundError(f"calibration corpus not found: {path}")
        result.extend(_load_jsonl(path, mode, limit))
    return result


def _to_device(batch: Mapping[str, object], device: str) -> dict[str, object]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _encode(processor, messages: Sequence[object], device: str) -> tuple[dict[str, object], int]:
    prompt = processor.apply_chat_template(
        list(messages[:-1]),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full = processor.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    prompt = _to_device(prompt, device)
    full = _to_device(full, device)
    prompt_length = min(int(prompt["input_ids"].shape[1]), int(full["input_ids"].shape[1]))
    full.pop("labels", None)
    return full, prompt_length


def _teacher_adapters(plan: Mapping[str, object]) -> dict[str, Path]:
    value = plan.get("teacher_adapters")
    if not isinstance(value, Mapping):
        raise ValueError("student plan must define teacher_adapters")
    return {
        mode: resolve_peft_adapter_dir(Path(str(value.get(mode) or "")), mode)
        for mode in ("gui_policy", "chat")
    }


def _load_teacher(
    *,
    model_path: str,
    plan: Mapping[str, object],
    unified: bool,
    dtype,
    device: str,
    model_class,
    peft_model_class,
):
    try:
        base = model_class.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except TypeError:
        base = model_class.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    if unified:
        teacher = base
    else:
        adapters = _teacher_adapters(plan)
        teacher = peft_model_class.from_pretrained(
            base,
            str(adapters["gui_policy"]),
            adapter_name="gui_policy",
            is_trainable=False,
        )
        teacher.load_adapter(
            str(adapters["chat"]),
            adapter_name="chat",
            is_trainable=False,
        )
    teacher.to(device)
    teacher.eval()
    teacher.config.use_cache = False
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


class _Accumulator:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.vectors: dict[str, object] = {}
        self.counts: dict[str, int] = {}
        self.scalars: dict[str, float] = {}
        self.scalar_counts: dict[str, int] = {}

    def add_vector(self, name: str, value) -> None:
        tensor = value.detach().float().cpu()
        if tensor.ndim != 1:
            raise ValueError(f"calibration vector {name} must be one-dimensional")
        current = self.vectors.get(name)
        if current is None:
            self.vectors[name] = tensor.clone()
            self.counts[name] = 1
        else:
            if tuple(current.shape) != tuple(tensor.shape):
                raise RuntimeError(f"calibration vector shape changed for {name}")
            current.add_(tensor)
            self.counts[name] += 1

    def add_vector_max(self, name: str, value) -> None:
        """Running element-wise maximum; peak activations mark super-weight neurons."""
        tensor = value.detach().float().cpu()
        if tensor.ndim != 1:
            raise ValueError(f"calibration vector {name} must be one-dimensional")
        current = self.vectors.get(name)
        if current is None:
            self.vectors[name] = tensor.clone()
            self.counts[name] = 0
        else:
            if tuple(current.shape) != tuple(tensor.shape):
                raise RuntimeError(f"calibration vector shape changed for {name}")
            self.torch.maximum(current, tensor, out=current)

    def add_scalar(self, name: str, value: float) -> None:
        self.scalars[name] = self.scalars.get(name, 0.0) + float(value)
        self.scalar_counts[name] = self.scalar_counts.get(name, 0) + 1

    def vector(self, name: str) -> list[float]:
        if name not in self.vectors:
            raise KeyError(name)
        count = max(1, self.counts[name])
        return [float(value) for value in (self.vectors[name] / count).tolist()]

    def scalar(self, name: str) -> float:
        if name not in self.scalars:
            raise KeyError(name)
        return self.scalars[name] / max(1, self.scalar_counts[name])


def _vector_energy(tensor):
    dims = tuple(range(tensor.ndim - 1))
    return tensor.detach().float().pow(2).mean(dim=dims)


def _core_model(model):
    getter = getattr(model, "get_base_model", None)
    return getter() if callable(getter) else model


def calibrate(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
    except ImportError as exc:
        raise RuntimeError("install the training extra: python -m pip install -e '.[train]'") from exc

    plan = load_student_plan(args.config)
    records = _records(plan, args.max_records_per_corpus)
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    teacher_model = args.teacher_model or str(plan.get("initialization_teacher") or "")
    if not teacher_model:
        raise ValueError("calibration teacher model is missing")

    teacher = _load_teacher(
        model_path=teacher_model,
        plan=plan,
        unified=args.teacher_unified,
        dtype=dtype,
        device=device,
        model_class=Qwen3VLForConditionalGeneration,
        peft_model_class=PeftModel,
    )
    processor = AutoProcessor.from_pretrained(teacher_model)
    core = _core_model(teacher)
    language_layers = core.model.language_model.layers
    vision_blocks = core.model.visual.blocks
    text_heads = int(core.config.text_config.num_attention_heads)
    text_kv_heads = int(core.config.text_config.num_key_value_heads)
    text_head_dim = int(core.config.text_config.head_dim)
    vision_heads = int(core.config.vision_config.num_heads)
    vision_head_dim = int(core.config.vision_config.hidden_size) // vision_heads

    acc = _Accumulator(torch)
    handles = []

    for layer_index, layer in enumerate(language_layers):
        def q_hook(_module, _inputs, output, index=layer_index):
            shaped = output.detach().float().reshape(-1, text_heads, text_head_dim)
            acc.add_vector(f"text_q_heads:{index}", shaped.pow(2).mean(dim=(0, 2)))

        def k_hook(_module, _inputs, output, index=layer_index):
            shaped = output.detach().float().reshape(-1, text_kv_heads, text_head_dim)
            acc.add_vector(f"text_kv_heads:{index}", shaped.pow(2).mean(dim=(0, 2)))

        def mlp_hook(_module, _inputs, output, index=layer_index):
            acc.add_vector(f"text_mlp:{index}", _vector_energy(output))

        def mlp_activation_hook(_module, inputs, index=layer_index):
            # What a neuron actually contributes is the gated activation
            # act(gate) * up that enters down_proj, not the gate pre-activation.
            if inputs:
                activation = inputs[0].detach().float()
                acc.add_vector(f"text_mlp_activation:{index}", _vector_energy(activation))
                peak = activation.abs().reshape(-1, activation.shape[-1]).max(dim=0).values
                acc.add_vector_max(f"text_mlp_activation_peak:{index}", peak)

        handles.append(layer.self_attn.q_proj.register_forward_hook(q_hook))
        handles.append(layer.self_attn.k_proj.register_forward_hook(k_hook))
        handles.append(layer.mlp.gate_proj.register_forward_hook(mlp_hook))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(mlp_activation_hook))

    for block_index, block in enumerate(vision_blocks):
        def block_hook(_module, inputs, output, index=block_index):
            if not inputs:
                return
            source = inputs[0].detach().float()
            target = output.detach().float()
            denominator = source.pow(2).mean().clamp_min(1e-8)
            residual = (target - source).pow(2).mean() / denominator
            acc.add_scalar(f"vision_layer:{index}", float(residual.cpu()))
            acc.add_vector("vision_hidden", _vector_energy(target))

        def qkv_hook(_module, _inputs, output, index=block_index):
            shaped = output.detach().float().reshape(-1, 3, vision_heads, vision_head_dim)
            acc.add_vector(f"vision_heads:{index}", shaped.pow(2).mean(dim=(0, 1, 3)))

        def vision_mlp_hook(_module, _inputs, output, index=block_index):
            acc.add_vector(f"vision_mlp:{index}", _vector_energy(output))

        def vision_mlp_activation_hook(_module, inputs, index=block_index):
            # Post-GELU activation entering linear_fc2, for the same reason as
            # the text down_proj input above.
            if inputs:
                acc.add_vector(f"vision_mlp_activation:{index}", _vector_energy(inputs[0]))

        handles.append(block.register_forward_hook(block_hook))
        handles.append(block.attn.qkv.register_forward_hook(qkv_hook))
        handles.append(block.mlp.linear_fc1.register_forward_hook(vision_mlp_hook))
        handles.append(block.mlp.linear_fc2.register_forward_pre_hook(vision_mlp_activation_hook))

    mode_counts = {"gui_policy": 0, "chat": 0}
    completion_tokens = 0
    try:
        with torch.no_grad():
            for record in records:
                if not args.teacher_unified:
                    teacher.set_adapter(record.mode)
                model_inputs, prompt_length = _encode(processor, record.messages, device)
                output = teacher(
                    **model_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
                hidden_states = output.hidden_states
                if hidden_states is None or len(hidden_states) != len(language_layers) + 1:
                    raise RuntimeError("teacher did not return the expected text hidden-state stack")
                sequence_length = int(hidden_states[-1].shape[1])
                start = min(prompt_length, max(0, sequence_length - 1))
                completion_tokens += max(0, sequence_length - start)
                mode_counts[record.mode] += 1

                for layer_index in range(len(language_layers)):
                    source = hidden_states[layer_index][:, start:, :].detach().float()
                    target = hidden_states[layer_index + 1][:, start:, :].detach().float()
                    denominator = source.pow(2).mean().clamp_min(1e-8)
                    residual = (target - source).pow(2).mean() / denominator
                    acc.add_scalar(f"text_layer:{layer_index}", float(residual.cpu()))
                stacked = torch.stack(
                    [state[:, start:, :].detach().float().pow(2).mean(dim=(0, 1)) for state in hidden_states],
                    dim=0,
                ).mean(dim=0)
                acc.add_vector("text_hidden", stacked)
                del output
    finally:
        for handle in handles:
            handle.remove()

    text_layer_scores = [acc.scalar(f"text_layer:{index}") for index in range(len(language_layers))]
    text_hidden_scores = acc.vector("text_hidden")
    text_mlp_scores = {
        str(index): acc.vector(f"text_mlp:{index}")
        for index in range(len(language_layers))
    }
    text_mlp_activation_scores = {
        str(index): acc.vector(f"text_mlp_activation:{index}")
        for index in range(len(language_layers))
    }
    text_mlp_activation_peak_scores = {
        str(index): acc.vector(f"text_mlp_activation_peak:{index}")
        for index in range(len(language_layers))
    }
    text_q_head_scores = {
        str(index): acc.vector(f"text_q_heads:{index}")
        for index in range(len(language_layers))
    }
    text_kv_head_scores = {
        str(index): acc.vector(f"text_kv_heads:{index}")
        for index in range(len(language_layers))
    }

    vision_ready = "vision_hidden" in acc.vectors
    vision_layer_scores: list[float] = []
    vision_hidden_scores: list[float] = []
    vision_mlp_scores: dict[str, list[float]] = {}
    vision_mlp_activation_scores: dict[str, list[float]] = {}
    vision_head_scores: dict[str, list[float]] = {}
    if vision_ready:
        vision_layer_scores = [acc.scalar(f"vision_layer:{index}") for index in range(len(vision_blocks))]
        vision_hidden_scores = acc.vector("vision_hidden")
        vision_mlp_scores = {
            str(index): acc.vector(f"vision_mlp:{index}")
            for index in range(len(vision_blocks))
        }
        vision_mlp_activation_scores = {
            str(index): acc.vector(f"vision_mlp_activation:{index}")
            for index in range(len(vision_blocks))
        }
        vision_head_scores = {
            str(index): acc.vector(f"vision_heads:{index}")
            for index in range(len(vision_blocks))
        }

    # Produce nested depth selections so every later stage can be initialized
    # exactly from the previous recovered checkpoint while retaining stable
    # original teacher-layer lineage IDs.
    selected = tuple(range(len(language_layers)))
    depth_layers: dict[str, list[int]] = {}
    for depth in (24, 20, 16, 12, 9, 6):
        if depth > len(selected):
            continue
        selected = select_salient_layer_ids(
            text_layer_scores, selected, depth, protect_tail=TEXT_TAIL_PROTECTED_LAYERS
        )
        depth_layers[str(depth)] = list(selected)

    recommended: dict[str, object] = {
        "text_layers_by_depth": depth_layers,
    }
    if vision_ready:
        recommended["vision_layers_by_depth"] = {
            "12": list(
                select_salient_layer_ids(
                    vision_layer_scores,
                    tuple(range(len(vision_blocks))),
                    12,
                )
            )
        }

    manifest = {
        "schema_version": 2,
        "teacher_model": teacher_model,
        "teacher_unified": bool(args.teacher_unified),
        "teacher_adapters": None
        if args.teacher_unified
        else {key: str(value) for key, value in _teacher_adapters(plan).items()},
        "records_seen_by_mode": mode_counts,
        "completion_tokens_observed": completion_tokens,
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "text": {
            "layers": len(language_layers),
            "hidden_size": len(text_hidden_scores),
            "attention_heads": text_heads,
            "key_value_heads": text_kv_heads,
            "head_dim": text_head_dim,
            "layer_scores": text_layer_scores,
            "hidden_scores": text_hidden_scores,
            "mlp_scores": text_mlp_scores,
            "mlp_activation_scores": text_mlp_activation_scores,
            "mlp_activation_peak_scores": text_mlp_activation_peak_scores,
            "q_head_scores": text_q_head_scores,
            "kv_head_scores": text_kv_head_scores,
        },
        "vision": {
            "ready": vision_ready,
            "layers": len(vision_blocks),
            # The tower shape is recorded even when no image exercised it, so a
            # vision-narrowing build can be refused for the right reason.
            "hidden_size": int(core.config.vision_config.hidden_size),
            "attention_heads": vision_heads,
            "head_dim": vision_head_dim,
            "layer_scores": vision_layer_scores,
            "hidden_scores": vision_hidden_scores,
            "mlp_scores": vision_mlp_scores,
            "mlp_activation_scores": vision_mlp_activation_scores,
            "head_scores": vision_head_scores,
        },
        "recommended": recommended,
        "width_pruning_ready": bool(vision_ready and text_hidden_scores),
        "selection_method": "activation_energy_and_residual_saliency",
        "prefix_dimension_truncation_used": False,
        "credentials_written": False,
    }

    # Width selections for every student the plan declares, derived from the
    # raw scores above. Only narrowed axes are emitted, and GQA groups are kept
    # whole, so the runner's normalization step finds nothing to repair.
    mlp_sources: dict[str, str] = {}
    for spec in plan_student_specs(plan):
        try:
            derived = derive_width_selections(manifest, spec)
        except ValueError:
            continue
        source = derived.pop("text_mlp_score_source", None)
        derived.pop("vision_mlp_score_source", None)
        if isinstance(source, str):
            mlp_sources[spec.name] = source
        recommended.update(derived)
    if mlp_sources:
        manifest["text_mlp_score_source"] = mlp_sources
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gui_owl_students.json")
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-unified", action="store_true")
    parser.add_argument("--max-records-per-corpus", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="artifacts/calibration/gui-owl-compression.json",
    )
    args = parser.parse_args(argv)
    print(json.dumps(calibrate(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
