#!/usr/bin/env python3
"""Build one calibrated progressive GUI-Owl compression stage.

Depth-only stages copy exact tensors from a recovered predecessor and select
blocks by calibration saliency.  The 0.5B width stage requires activation-
ranked residual, MLP and attention-head selections and applies those indices
consistently across Qwen3-VL text, vision and patch-merger tensors.  There is no
prefix-dimension fallback: an unrecognized shape change fails the build.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from android_drlqn.gui_owl_student import (
    build_student_config_dict,
    expand_head_indices,
    expand_spatial_merge_indices,
    load_calibration_manifest,
    load_student_plan,
    relative_parameter_error,
    select_salient_layer_ids,
    source_positions_for_layer_ids,
    spec_from_plan,
    teacher_key_for_student,
)


_TEXT_LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_VISION_LAYER = re.compile(r"^model\.visual\.blocks\.(\d+)\.(.+)$")


def _student_entry(plan: Mapping[str, object], name: str) -> dict[str, object]:
    students = plan.get("students")
    if not isinstance(students, Mapping):
        raise ValueError("student plan must define students")
    raw = students.get(name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"student plan does not define {name}")
    return dict(raw)


def _torch_dtype(torch_module, name: str):
    values = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest must be an object: {path}")
    return dict(value)


def _source_lineage(source_model: str, source_config: Mapping[str, object]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    text_config = source_config.get("text_config")
    vision_config = source_config.get("vision_config")
    if not isinstance(text_config, Mapping) or not isinstance(vision_config, Mapping):
        raise ValueError("source model config lacks text/vision sub-configs")
    text_layers = int(text_config.get("num_hidden_layers") or 0)
    vision_layers = int(vision_config.get("depth") or 0)
    if min(text_layers, vision_layers) < 1:
        raise ValueError("source model has invalid text/vision depth")

    source = Path(source_model)
    candidates = (
        source / "student-build-manifest.json",
        source / "compression-lineage.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        manifest = _read_json(path)
        text = tuple(int(value) for value in manifest.get("text_original_layer_ids") or ())
        vision = tuple(int(value) for value in manifest.get("vision_original_layer_ids") or ())
        if len(text) == text_layers and len(vision) == vision_layers:
            return text, vision
    return tuple(range(text_layers)), tuple(range(vision_layers))


def _calibration_scores(calibration: Mapping[str, object], section: str, field: str) -> object:
    value = calibration.get(section)
    if not isinstance(value, Mapping) or field not in value:
        raise ValueError(f"calibration manifest lacks {section}.{field}")
    return value[field]


def _recommended(calibration: Mapping[str, object], key: str) -> object:
    value = calibration.get("recommended")
    if not isinstance(value, Mapping) or key not in value:
        raise ValueError(f"calibration manifest lacks recommended.{key}")
    return value[key]


def _select_text_layers(
    calibration: Mapping[str, object], source_ids: Sequence[int], target_depth: int
) -> tuple[int, ...]:
    if target_depth == len(source_ids):
        return tuple(int(value) for value in source_ids)
    recommended = _recommended(calibration, "text_layers_by_depth")
    if isinstance(recommended, Mapping):
        candidate = recommended.get(str(target_depth))
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            selected = tuple(int(value) for value in candidate)
            if len(selected) == target_depth and set(selected).issubset(set(source_ids)):
                return selected
    scores = _calibration_scores(calibration, "text", "layer_scores")
    return select_salient_layer_ids(scores, source_ids, target_depth)


def _select_vision_layers(
    calibration: Mapping[str, object], source_ids: Sequence[int], target_depth: int
) -> tuple[int, ...]:
    if target_depth == len(source_ids):
        return tuple(int(value) for value in source_ids)
    recommended = _recommended(calibration, "vision_layers_by_depth")
    if isinstance(recommended, Mapping):
        candidate = recommended.get(str(target_depth))
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            selected = tuple(int(value) for value in candidate)
            if len(selected) == target_depth and set(selected).issubset(set(source_ids)):
                return selected
    scores = _calibration_scores(calibration, "vision", "layer_scores")
    return select_salient_layer_ids(scores, source_ids, target_depth)


def _indices_by_original_layer(value: object, original_id: int, count: int) -> tuple[int, ...]:
    if not isinstance(value, Mapping):
        raise ValueError("per-layer calibration selection must be an object")
    raw = value.get(str(original_id))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"calibration selection missing original layer {original_id}")
    result = tuple(int(item) for item in raw)
    if len(result) != count or len(set(result)) != len(result):
        raise ValueError(f"calibration selection for layer {original_id} has wrong size")
    return result


def _index_select(tensor, dim: int, indices: Sequence[int], torch_module):
    index = torch_module.tensor(tuple(int(value) for value in indices), device=tensor.device, dtype=torch_module.long)
    return tensor.index_select(dim, index)


def _slice_two_dims(tensor, rows: Sequence[int], cols: Sequence[int], torch_module):
    return _index_select(_index_select(tensor, 0, rows, torch_module), 1, cols, torch_module)


class _WidthSelectors:
    def __init__(
        self,
        *,
        calibration: Mapping[str, object],
        source_config: Mapping[str, object],
        target_spec,
        text_original_ids: Sequence[int],
        vision_original_ids: Sequence[int],
    ) -> None:
        if not bool(calibration.get("width_pruning_ready")):
            raise RuntimeError(
                "0.5B structured width pruning requires a calibration run that exercised the vision tower"
            )
        recommended = calibration.get("recommended")
        if not isinstance(recommended, Mapping):
            raise ValueError("calibration manifest lacks recommended selections")
        text_cfg = source_config["text_config"]
        vision_cfg = source_config["vision_config"]
        self.source_text_hidden = int(text_cfg["hidden_size"])
        self.source_text_heads = int(text_cfg["num_attention_heads"])
        self.source_text_kv_heads = int(text_cfg["num_key_value_heads"])
        self.text_head_dim = int(text_cfg.get("head_dim") or self.source_text_hidden // self.source_text_heads)
        self.source_vision_hidden = int(vision_cfg["hidden_size"])
        self.source_vision_heads = int(vision_cfg["num_heads"])
        self.vision_head_dim = self.source_vision_hidden // self.source_vision_heads
        self.merge_units = int(vision_cfg.get("spatial_merge_size") or 1) ** 2

        hidden = recommended.get(f"text_hidden_indices_{target_spec.text_hidden_size}")
        if not isinstance(hidden, Sequence) or isinstance(hidden, (str, bytes)):
            raise ValueError("calibration manifest lacks text residual-channel selection")
        self.text_hidden = tuple(int(value) for value in hidden)
        if len(self.text_hidden) != target_spec.text_hidden_size:
            raise ValueError("text hidden selection has wrong size")

        vision_hidden = recommended.get(f"vision_hidden_indices_{target_spec.vision_hidden_size}")
        if not isinstance(vision_hidden, Sequence) or isinstance(vision_hidden, (str, bytes)):
            raise ValueError("calibration manifest lacks vision residual-channel selection")
        self.vision_hidden = tuple(int(value) for value in vision_hidden)
        if len(self.vision_hidden) != target_spec.vision_hidden_size:
            raise ValueError("vision hidden selection has wrong size")
        self.vision_merge = expand_spatial_merge_indices(
            self.vision_hidden,
            self.source_vision_hidden,
            self.merge_units,
        )

        self.text_q_heads_raw = recommended.get(f"text_q_heads_{target_spec.text_attention_heads}")
        self.text_kv_heads_raw = recommended.get(f"text_kv_heads_{target_spec.text_key_value_heads}")
        self.text_mlp_raw = recommended.get(f"text_mlp_indices_{target_spec.text_intermediate_size}")
        self.vision_heads_raw = recommended.get(f"vision_heads_{target_spec.vision_heads}")
        self.vision_mlp_raw = recommended.get(f"vision_mlp_indices_{target_spec.vision_intermediate_size}")
        self.target_spec = target_spec
        self.text_original_ids = tuple(int(value) for value in text_original_ids)
        self.vision_original_ids = tuple(int(value) for value in vision_original_ids)

    def text_q(self, target_layer: int) -> tuple[int, ...]:
        original = self.text_original_ids[target_layer]
        heads = _indices_by_original_layer(
            self.text_q_heads_raw, original, self.target_spec.text_attention_heads
        )
        return expand_head_indices(heads, self.text_head_dim)

    def text_kv(self, target_layer: int) -> tuple[int, ...]:
        original = self.text_original_ids[target_layer]
        heads = _indices_by_original_layer(
            self.text_kv_heads_raw, original, self.target_spec.text_key_value_heads
        )
        return expand_head_indices(heads, self.text_head_dim)

    def text_mlp(self, target_layer: int) -> tuple[int, ...]:
        original = self.text_original_ids[target_layer]
        return _indices_by_original_layer(
            self.text_mlp_raw, original, self.target_spec.text_intermediate_size
        )

    def vision_attn(self, target_layer: int) -> tuple[int, ...]:
        original = self.vision_original_ids[target_layer]
        heads = _indices_by_original_layer(
            self.vision_heads_raw, original, self.target_spec.vision_heads
        )
        return expand_head_indices(heads, self.vision_head_dim)

    def vision_qkv(self, target_layer: int) -> tuple[int, ...]:
        attention = self.vision_attn(target_layer)
        hidden = self.source_vision_hidden
        return tuple(attention) + tuple(value + hidden for value in attention) + tuple(
            value + 2 * hidden for value in attention
        )

    def vision_mlp(self, target_layer: int) -> tuple[int, ...]:
        original = self.vision_original_ids[target_layer]
        return _indices_by_original_layer(
            self.vision_mlp_raw, original, self.target_spec.vision_intermediate_size
        )


def _structured_width_tensor(key: str, source, selectors: _WidthSelectors, torch_module):
    """Slice a known Qwen3-VL tensor according to semantic channel selections."""

    text_match = _TEXT_LAYER.match(key)
    if key in {"model.language_model.embed_tokens.weight", "lm_head.weight"}:
        return _index_select(source, 1, selectors.text_hidden, torch_module)
    if key == "model.language_model.norm.weight":
        return _index_select(source, 0, selectors.text_hidden, torch_module)

    if text_match:
        layer = int(text_match.group(1))
        suffix = text_match.group(2)
        q = selectors.text_q(layer)
        kv = selectors.text_kv(layer)
        mlp = selectors.text_mlp(layer)
        hidden = selectors.text_hidden
        if suffix in {"input_layernorm.weight", "post_attention_layernorm.weight"}:
            return _index_select(source, 0, hidden, torch_module)
        if suffix == "self_attn.q_proj.weight":
            return _slice_two_dims(source, q, hidden, torch_module)
        if suffix in {"self_attn.k_proj.weight", "self_attn.v_proj.weight"}:
            return _slice_two_dims(source, kv, hidden, torch_module)
        if suffix == "self_attn.o_proj.weight":
            return _slice_two_dims(source, hidden, q, torch_module)
        if suffix == "self_attn.q_proj.bias":
            return _index_select(source, 0, q, torch_module)
        if suffix in {"self_attn.k_proj.bias", "self_attn.v_proj.bias"}:
            return _index_select(source, 0, kv, torch_module)
        if suffix == "self_attn.o_proj.bias":
            return _index_select(source, 0, hidden, torch_module)
        if suffix in {"mlp.gate_proj.weight", "mlp.up_proj.weight"}:
            return _slice_two_dims(source, mlp, hidden, torch_module)
        if suffix == "mlp.down_proj.weight":
            return _slice_two_dims(source, hidden, mlp, torch_module)
        if suffix in {"mlp.gate_proj.bias", "mlp.up_proj.bias"}:
            return _index_select(source, 0, mlp, torch_module)
        if suffix == "mlp.down_proj.bias":
            return _index_select(source, 0, hidden, torch_module)
        # q_norm/k_norm are per-head-dimension and remain exact-width tensors.
        if suffix in {"self_attn.q_norm.weight", "self_attn.k_norm.weight"}:
            return source

    if key == "model.visual.patch_embed.proj.weight":
        return _index_select(source, 0, selectors.vision_hidden, torch_module)
    if key == "model.visual.patch_embed.proj.bias":
        return _index_select(source, 0, selectors.vision_hidden, torch_module)
    if key == "model.visual.pos_embed.weight":
        return _index_select(source, 1, selectors.vision_hidden, torch_module)

    vision_match = _VISION_LAYER.match(key)
    if vision_match:
        layer = int(vision_match.group(1))
        suffix = vision_match.group(2)
        hidden = selectors.vision_hidden
        attn = selectors.vision_attn(layer)
        qkv = selectors.vision_qkv(layer)
        mlp = selectors.vision_mlp(layer)
        if suffix in {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"}:
            return _index_select(source, 0, hidden, torch_module)
        if suffix == "attn.qkv.weight":
            return _slice_two_dims(source, qkv, hidden, torch_module)
        if suffix == "attn.qkv.bias":
            return _index_select(source, 0, qkv, torch_module)
        if suffix == "attn.proj.weight":
            return _slice_two_dims(source, hidden, attn, torch_module)
        if suffix == "attn.proj.bias":
            return _index_select(source, 0, hidden, torch_module)
        if suffix == "mlp.linear_fc1.weight":
            return _slice_two_dims(source, mlp, hidden, torch_module)
        if suffix == "mlp.linear_fc1.bias":
            return _index_select(source, 0, mlp, torch_module)
        if suffix == "mlp.linear_fc2.weight":
            return _slice_two_dims(source, hidden, mlp, torch_module)
        if suffix == "mlp.linear_fc2.bias":
            return _index_select(source, 0, hidden, torch_module)

    if key.startswith("model.visual.merger.") or key.startswith("model.visual.deepstack_merger_list."):
        merge = selectors.vision_merge
        hidden = selectors.vision_hidden
        text = selectors.text_hidden
        if key.endswith("linear_fc1.weight"):
            return _slice_two_dims(source, merge, merge, torch_module)
        if key.endswith("linear_fc1.bias"):
            return _index_select(source, 0, merge, torch_module)
        if key.endswith("linear_fc2.weight"):
            return _slice_two_dims(source, text, merge, torch_module)
        if key.endswith("linear_fc2.bias"):
            return _index_select(source, 0, text, torch_module)
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            if int(source.shape[0]) == selectors.source_vision_hidden:
                return _index_select(source, 0, hidden, torch_module)
            if int(source.shape[0]) == selectors.source_vision_hidden * selectors.merge_units:
                return _index_select(source, 0, merge, torch_module)

    return None


def _copy_initial_weights(
    *,
    student,
    source_model,
    text_layer_map: Sequence[int],
    vision_layer_map: Sequence[int],
    text_original_ids: Sequence[int],
    vision_original_ids: Sequence[int],
    width_selectors: _WidthSelectors | None,
    torch_module,
) -> dict[str, object]:
    source_state = source_model.state_dict()
    exact_tensors = structured_tensors = copied_elements = total_elements = 0
    unmatched: list[str] = []
    with torch_module.no_grad():
        for key, destination in student.state_dict().items():
            total_elements += destination.numel()
            source_key = teacher_key_for_student(key, text_layer_map, vision_layer_map)
            source = source_state.get(source_key)
            if source is None:
                unmatched.append(f"{key}: source key missing ({source_key})")
                continue
            if tuple(source.shape) == tuple(destination.shape):
                destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
                exact_tensors += 1
                copied_elements += destination.numel()
                continue
            if width_selectors is None:
                unmatched.append(
                    f"{key}: shape {tuple(source.shape)} -> {tuple(destination.shape)} without width selectors"
                )
                continue
            sliced = _structured_width_tensor(key, source, width_selectors, torch_module)
            if sliced is None or tuple(sliced.shape) != tuple(destination.shape):
                shape = None if sliced is None else tuple(sliced.shape)
                unmatched.append(
                    f"{key}: unsupported structured slice {tuple(source.shape)} -> {tuple(destination.shape)}; got {shape}"
                )
                continue
            destination.copy_(sliced.to(device=destination.device, dtype=destination.dtype))
            structured_tensors += 1
            copied_elements += destination.numel()
    return {
        "exact_tensors": exact_tensors,
        "structured_tensors": structured_tensors,
        "copied_elements": copied_elements,
        "total_elements": total_elements,
        "copy_fraction": copied_elements / max(1, total_elements),
        "unmatched": unmatched,
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("install the training extra: python -m pip install -e '.[train]'") from exc

    plan = load_student_plan(args.config)
    spec = spec_from_plan(plan, args.student)
    entry = _student_entry(plan, args.student)
    source_model_name = args.source_model or str(plan.get("initialization_teacher") or "")
    if not source_model_name:
        raise ValueError("compression source model is missing")
    calibration_path = Path(args.calibration_manifest)
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"compression calibration manifest not found: {calibration_path}; run calibrate_gui_owl_compression.py first"
        )
    calibration = load_calibration_manifest(calibration_path)

    output = Path(
        args.output
        or str(entry.get("initialized") or f"artifacts/students/gui-owl-{args.student}-init")
    )
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"student output is not empty: {output}; pass --overwrite to replace")
        shutil.rmtree(output)

    dtype = _torch_dtype(torch, args.dtype)
    source_config_obj = AutoConfig.from_pretrained(source_model_name)
    source_config = source_config_obj.to_dict()
    source_text_ids, source_vision_ids = _source_lineage(source_model_name, source_config)
    target_text_depth = int(args.target_text_layers or spec.text_layers)
    if target_text_depth > len(source_text_ids):
        raise ValueError("target text depth cannot exceed source depth")
    if args.student == "0.5b" and target_text_depth != spec.text_layers:
        raise ValueError("0.5B width pruning must use its final text depth")

    selected_text_ids = _select_text_layers(calibration, source_text_ids, target_text_depth)
    text_layer_map = source_positions_for_layer_ids(source_text_ids, selected_text_ids)
    target_vision_depth = spec.vision_depth
    if target_vision_depth > len(source_vision_ids):
        raise ValueError("target vision depth cannot exceed source depth")
    selected_vision_ids = _select_vision_layers(calibration, source_vision_ids, target_vision_depth)
    vision_layer_map = source_positions_for_layer_ids(source_vision_ids, selected_vision_ids)

    target_config_dict = build_student_config_dict(
        source_config,
        spec,
        text_layers=target_text_depth,
    )
    source_text_cfg = source_config.get("text_config")
    if isinstance(source_text_cfg, Mapping):
        layer_types = source_text_cfg.get("layer_types")
        if isinstance(layer_types, Sequence) and not isinstance(layer_types, (str, bytes)):
            target_config_dict["text_config"]["layer_types"] = [
                layer_types[index] for index in text_layer_map
            ]
    target_config = Qwen3VLConfig.from_dict(target_config_dict)

    previous_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        student = Qwen3VLForConditionalGeneration(target_config)
    finally:
        torch.set_default_dtype(previous_default)

    try:
        source_model = Qwen3VLForConditionalGeneration.from_pretrained(
            source_model_name,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except TypeError:
        source_model = Qwen3VLForConditionalGeneration.from_pretrained(
            source_model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    source_model.eval()

    source_text_hidden = int(source_config["text_config"]["hidden_size"])
    source_vision_hidden = int(source_config["vision_config"]["hidden_size"])
    width_pruned = (
        source_text_hidden != spec.text_hidden_size
        or source_vision_hidden != spec.vision_hidden_size
        or int(source_config["text_config"]["intermediate_size"]) != spec.text_intermediate_size
        or int(source_config["vision_config"]["intermediate_size"]) != spec.vision_intermediate_size
    )
    width_selectors = None
    if width_pruned:
        width_selectors = _WidthSelectors(
            calibration=calibration,
            source_config=source_config,
            target_spec=spec,
            text_original_ids=selected_text_ids,
            vision_original_ids=selected_vision_ids,
        )

    copy_stats = _copy_initial_weights(
        student=student,
        source_model=source_model,
        text_layer_map=text_layer_map,
        vision_layer_map=vision_layer_map,
        text_original_ids=selected_text_ids,
        vision_original_ids=selected_vision_ids,
        width_selectors=width_selectors,
        torch_module=torch,
    )
    del source_model
    if copy_stats["unmatched"]:
        preview = "\n".join(str(value) for value in copy_stats["unmatched"][:20])
        raise RuntimeError(
            "compression build refused to leave shape-changed tensors randomly initialized. "
            f"Unmatched tensors ({len(copy_stats['unmatched'])}):\n{preview}"
        )

    actual_parameters = sum(parameter.numel() for parameter in student.parameters())
    is_final_architecture = (
        target_text_depth == spec.text_layers
        and int(target_config.vision_config.depth) == spec.vision_depth
        and int(target_config.text_config.hidden_size) == spec.text_hidden_size
    )
    parameter_error = relative_parameter_error(actual_parameters, spec.target_parameters)
    if (
        is_final_architecture
        and parameter_error > spec.max_relative_parameter_error
        and not args.allow_parameter_miss
    ):
        raise RuntimeError(
            f"{args.student} architecture has {actual_parameters:,} parameters, "
            f"{parameter_error:.1%} from target {spec.target_parameters:,}; "
            "adjust config or pass --allow-parameter-miss for an exploratory build"
        )

    output.mkdir(parents=True, exist_ok=True)
    student.config.use_cache = False
    student.save_pretrained(output, safe_serialization=True)
    processor = AutoProcessor.from_pretrained(source_model_name)
    processor.save_pretrained(output)

    manifest = {
        "schema_version": 2,
        "student": args.student,
        "compression_source": source_model_name,
        "calibration_manifest": str(calibration_path),
        "target_parameters": spec.target_parameters,
        "actual_parameters": actual_parameters,
        "relative_parameter_error": parameter_error,
        "final_architecture": is_final_architecture,
        "architecture": asdict(spec),
        "target_text_layers": target_text_depth,
        "source_text_original_layer_ids": list(source_text_ids),
        "text_original_layer_ids": list(selected_text_ids),
        "text_source_positions": list(text_layer_map),
        "source_vision_original_layer_ids": list(source_vision_ids),
        "vision_original_layer_ids": list(selected_vision_ids),
        "vision_source_positions": list(vision_layer_map),
        "width_pruned": width_pruned,
        "width_selection_method": "activation_ranked_structured_indices" if width_pruned else None,
        "prefix_dimension_truncation_used": False,
        "copy": copy_stats,
        "dtype": args.dtype,
        "full_student_checkpoint": True,
        "credentials_written": False,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (output / "student-build-manifest.json").write_text(manifest_text, encoding="utf-8")
    (output / "compression-lineage.json").write_text(manifest_text, encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, choices=("0.5b", "1b", "1.5b"))
    parser.add_argument("--config", default="config/gui_owl_students.json")
    parser.add_argument("--source-model")
    parser.add_argument("--target-text-layers", type=int)
    parser.add_argument(
        "--calibration-manifest",
        default="artifacts/calibration/gui-owl-compression.json",
    )
    parser.add_argument("--output")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-parameter-miss", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
