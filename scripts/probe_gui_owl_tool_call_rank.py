#!/usr/bin/env python3
"""Probe teacher-forced `<tool_call>` token geometry across student checkpoints.

Greedy parse rate answers "does the student emit a tool call"; it cannot say how
far the grammar token is from selection when it does not.  This probe teacher-
forces the real assistant target and reports the rank, probability and
top-1 logit margin of `<tool_call>` at the exact position the target requires
it, so the same measurement can be compared across compression depths.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence


def _to_device(batch: Mapping[str, object], device: str) -> dict[str, object]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _load_gui_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        messages = value.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError(f"{path}:{line_number}: messages must be an array")
        if len(messages) < 2 or not isinstance(messages[-1], Mapping):
            raise ValueError(f"{path}:{line_number}: assistant target is missing")
        if str(messages[-1].get("role") or "") != "assistant":
            raise ValueError(f"{path}:{line_number}: final message must be assistant")
        records.append(dict(value))
    if not records:
        raise ValueError(f"GUI policy corpus is empty: {path}")
    return records


def _select_records(
    records: Sequence[dict[str, object]], *, seed: int, holdout_fraction: float, max_records: int
) -> list[dict[str, object]]:
    values = list(records)
    random.Random(seed).shuffle(values)
    holdout_count = max(1, int(round(len(values) * holdout_fraction)))
    holdout_count = min(holdout_count, len(values))
    return values[:holdout_count][:max_records]


def _rewrite_images(
    messages: Sequence[Mapping[str, object]], *, source_prefix: str, target_prefix: str
) -> list[dict[str, object]]:
    """Repoint corpus image paths recorded on the training host at a local root."""
    if not source_prefix:
        return [dict(message) for message in messages]
    result: list[dict[str, object]] = []
    for message in messages:
        value = dict(message)
        content = value.get("content")
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            parts: list[object] = []
            for part in content:
                if isinstance(part, Mapping) and str(part.get("type") or "") == "image":
                    part = dict(part)
                    image = str(part.get("image") or "")
                    if image.startswith(source_prefix):
                        part["image"] = target_prefix + image[len(source_prefix) :]
                parts.append(part)
            value["content"] = parts
        result.append(value)
    return result


def _resolve_token_id(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{token!r} is not a single vocabulary item: {ids}")
    return int(ids[0])


def load_checkpoint(
    checkpoint: Path,
    *,
    device: str,
    dtype_name: str,
    adapter: str = "",
    adapter_name: str = "gui_policy",
    processor_source: str = "",
):
    """Load a student checkpoint for measurement; shared with the ablation tool.

    `adapter` attaches a PEFT adapter, which is how the 2B teacher carries its
    GUI grammar: probing the base weights alone would understate it.
    """
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("install the training extra: python -m pip install -e '.[train]'") from exc

    # The teacher is loaded by hub id, so only local sources are checked here.
    source = str(checkpoint)
    looks_local = source.startswith((".", "/", "~")) or Path(source).exists()
    if looks_local and not Path(source).is_dir():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_name == "auto":
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    else:
        dtype = dtypes[dtype_name]

    processor = AutoProcessor.from_pretrained(str(processor_source or checkpoint))
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(checkpoint), dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
        )
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(checkpoint), torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
        )
    if adapter:
        from peft import PeftModel

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from android_drlqn.gui_owl_student import resolve_peft_adapter_dir

        resolved = resolve_peft_adapter_dir(Path(adapter), adapter_name)
        model = PeftModel.from_pretrained(model, str(resolved))
    model.to(device)
    model.eval()
    return model, processor


def measure_model(
    *,
    model,
    processor,
    records: Sequence[dict[str, object]],
    device: str,
    token: str,
    image_prefix: str,
    image_root: str,
    label: str,
) -> dict[str, object]:
    """Teacher-force `records` and report where `token` sits in the distribution."""
    import torch

    token_id = _resolve_token_id(processor.tokenizer, token)
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or ())}
    if not eos_ids:
        raise ValueError("tokenizer has no EOS token id")

    rows: list[dict[str, object]] = []
    ranks: list[int] = []
    probabilities: list[float] = []
    top1_margins: list[float] = []
    top1_probabilities: list[float] = []
    entropies: list[float] = []
    first_ranks: list[int] = []
    first_probabilities: list[float] = []
    first_eos_margins: list[float] = []
    top1_is_token = 0
    missing_target_token = 0

    with torch.no_grad():
        for index, record in enumerate(records):
            messages = _rewrite_images(
                list(record["messages"]), source_prefix=image_prefix, target_prefix=image_root
            )
            prompt = processor.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            full = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            total_length = int(full["input_ids"].shape[1])
            prompt_length = min(int(prompt["input_ids"].shape[1]), total_length)
            completion_length = total_length - prompt_length
            if completion_length < 1:
                raise ValueError("assistant completion produced no target tokens")

            target_ids = [int(value) for value in full["input_ids"][0, prompt_length:].tolist()]
            try:
                offset = target_ids.index(token_id)
            except ValueError:
                offset = -1

            full = _to_device(full, device)
            full.pop("labels", None)
            output = model(**full, use_cache=False, logits_to_keep=completion_length + 1)
            logits = output.logits[0].float()

            first_logits = logits[0]
            first_target = target_ids[0]
            first_rank = int((first_logits > first_logits[first_target]).sum().item()) + 1
            first_prob = float(torch.softmax(first_logits, dim=-1)[first_target].cpu())
            eos_id = max(eos_ids, key=lambda value: float(first_logits[value].cpu()))
            first_eos_margin = float(first_logits[first_target].cpu()) - float(
                first_logits[eos_id].cpu()
            )
            first_ranks.append(first_rank)
            first_probabilities.append(first_prob)
            first_eos_margins.append(first_eos_margin)

            row: dict[str, object] = {
                "index": index,
                "id": record.get("id"),
                "episode_id": record.get("episode_id"),
                "completion_tokens": completion_length,
                "first_target_token": processor.tokenizer.decode([first_target]),
                "first_target_rank": first_rank,
                "first_target_probability": first_prob,
                "first_target_minus_eos_logit": first_eos_margin,
                "target_token_offset": offset,
            }

            if offset < 0:
                missing_target_token += 1
                row["token_present_in_target"] = False
            else:
                position = logits[offset]
                distribution = torch.softmax(position, dim=-1)
                rank = int((position > position[token_id]).sum().item()) + 1
                probability = float(distribution[token_id].cpu())
                top1_id = int(torch.argmax(position).item())
                margin = float(position[token_id].cpu()) - float(position[top1_id].cpu())
                # Removing layers without retraining flattens the distribution,
                # which can lift a token's rank while its probability collapses.
                # Entropy and the top-1 mass separate "moved up" from "noise".
                top1_probability = float(distribution[top1_id].cpu())
                entropy = float(
                    -(distribution * distribution.clamp_min(1e-12).log()).sum().cpu()
                )
                if top1_id == token_id:
                    top1_is_token += 1
                ranks.append(rank)
                probabilities.append(probability)
                top1_margins.append(margin)
                top1_probabilities.append(top1_probability)
                entropies.append(entropy)
                row.update(
                    {
                        "token_present_in_target": True,
                        "token_rank": rank,
                        "token_probability": probability,
                        "token_minus_top1_logit": margin,
                        "top1_token_id": top1_id,
                        "top1_token": processor.tokenizer.decode([top1_id]),
                        "top1_probability": top1_probability,
                        "entropy": entropy,
                    }
                )
            rows.append(row)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    measured = len(ranks)
    summary: dict[str, object] = {
        "checkpoint": label,
        "records": len(rows),
        "token": token,
        "token_id": token_id,
        "records_missing_token": missing_target_token,
        "records_measured": measured,
        "median_first_target_rank": float(statistics.median(first_ranks)),
        "mean_first_target_probability": sum(first_probabilities) / len(first_probabilities),
        "median_first_target_minus_eos_logit": float(statistics.median(first_eos_margins)),
    }
    if measured:
        summary.update(
            {
                "median_token_rank": float(statistics.median(ranks)),
                "mean_token_rank": sum(ranks) / measured,
                "mean_token_probability": sum(probabilities) / measured,
                "median_token_probability": float(statistics.median(probabilities)),
                "median_token_minus_top1_logit": float(statistics.median(top1_margins)),
                "mean_top1_probability": sum(top1_probabilities) / measured,
                "mean_entropy": sum(entropies) / measured,
                "top1_is_token_rate": top1_is_token / measured,
            }
        )
    summary["rows"] = rows
    return summary


def probe_checkpoint(
    *,
    checkpoint: Path,
    records: Sequence[dict[str, object]],
    device: str,
    dtype_name: str,
    token: str,
    image_prefix: str,
    image_root: str,
    adapter: str = "",
    adapter_name: str = "gui_policy",
    processor_source: str = "",
) -> dict[str, object]:
    model, processor = load_checkpoint(
        checkpoint,
        device=device,
        dtype_name=dtype_name,
        adapter=adapter,
        adapter_name=adapter_name,
        processor_source=processor_source,
    )
    try:
        return measure_model(
            model=model,
            processor=processor,
            records=records,
            device=device,
            token=token,
            image_prefix=image_prefix,
            image_root=image_root,
            label=str(checkpoint),
        )
    finally:
        del model


def probe(args: argparse.Namespace) -> dict[str, object]:
    corpus = Path(args.corpus)
    records = _select_records(
        _load_gui_records(corpus),
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        max_records=args.max_records,
    )
    results = [
        probe_checkpoint(
            checkpoint=Path(value),
            records=records,
            device=args.device,
            dtype_name=args.dtype,
            token=args.token,
            image_prefix=args.image_prefix,
            image_root=args.image_root,
            adapter=args.adapter,
            adapter_name=args.adapter_name,
            processor_source=args.processor,
        )
        for value in args.checkpoint
    ]
    result = {
        "corpus": str(corpus),
        "records": len(records),
        "checkpoints": [
            {key: value for key, value in item.items() if key != "rows"} for item in results
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {**result, "rows_by_checkpoint": {item["checkpoint"]: item["rows"] for item in results}},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--corpus", default="artifacts/training/gui-policy-unigui-openmobile.jsonl"
    )
    parser.add_argument("--output", default="artifacts/diagnostics/gui-tool-call-rank.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16")
    )
    parser.add_argument("--token", default="<tool_call>")
    parser.add_argument(
        "--adapter", default="", help="PEFT adapter root to attach (teacher probing)"
    )
    parser.add_argument("--adapter-name", default="gui_policy")
    parser.add_argument(
        "--processor", default="", help="processor source when it differs from the weights"
    )
    parser.add_argument(
        "--image-prefix",
        default="",
        help="corpus image path prefix recorded on the training host",
    )
    parser.add_argument(
        "--image-root", default="", help="local replacement for --image-prefix"
    )
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    if not 0 < args.holdout_fraction <= 1:
        parser.error("--holdout-fraction must be in (0, 1]")
    if args.image_prefix and not args.image_root:
        parser.error("--image-prefix requires --image-root")
    print(json.dumps(probe(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
