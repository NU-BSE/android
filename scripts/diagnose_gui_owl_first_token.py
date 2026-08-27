#!/usr/bin/env python3
"""Compare first-token GUI probabilities between two student checkpoints.

Greedy generation is intentionally coarse: a checkpoint can remain 100% EOS
while training is still moving probability mass away from EOS. This diagnostic
looks directly at the next-token distribution on the same held-out GUI prompts
and reports the ground-truth first completion token versus EOS.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
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


def _manifest_summary(checkpoint: Path) -> dict[str, object] | None:
    path = checkpoint / "distillation-manifest.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        return None
    keys = (
        "stage",
        "optimizer_updates",
        "mean_loss",
        "mean_supervised_loss",
        "mean_forward_logit_kl",
        "mean_k3_reverse_kl",
        "learning_rate",
        "accepted",
    )
    return {key: value.get(key) for key in keys if key in value}


def _evaluate_checkpoint(
    *,
    checkpoint: Path,
    records: Sequence[dict[str, object]],
    device: str,
) -> dict[str, object]:
    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("install training dependencies before running diagnostics") from exc

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(str(checkpoint))
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(checkpoint), dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
        )
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(checkpoint), torch_dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
        )
    model.to(device)
    model.eval()

    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or ())}
    if not eos_ids:
        raise ValueError("tokenizer has no EOS token id")

    rows: list[dict[str, object]] = []
    eos_probs: list[float] = []
    target_probs: list[float] = []
    target_log_probs: list[float] = []
    target_ranks: list[int] = []
    margins: list[float] = []
    top1_eos = 0

    with torch.no_grad():
        for index, record in enumerate(records):
            messages = list(record["messages"])
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
            prompt_length = min(
                int(prompt["input_ids"].shape[1]), int(full["input_ids"].shape[1])
            )
            if prompt_length >= int(full["input_ids"].shape[1]):
                raise ValueError("assistant completion produced no first target token")
            target_id = int(full["input_ids"][0, prompt_length])
            prompt = _to_device(prompt, device)
            prompt.pop("labels", None)
            output = model(**prompt, use_cache=False, logits_to_keep=1)
            logits = output.logits[0, -1].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            target_logit = float(logits[target_id].cpu())
            target_prob = float(probs[target_id].cpu())
            target_log_prob = float(log_probs[target_id].cpu())
            target_rank = int((logits > logits[target_id]).sum().item()) + 1
            eos_id = max(eos_ids, key=lambda token_id: float(logits[token_id].cpu()))
            eos_prob = float(probs[eos_id].cpu())
            eos_logit = float(logits[eos_id].cpu())
            top1_id = int(torch.argmax(logits).item())
            if top1_id in eos_ids:
                top1_eos += 1

            eos_probs.append(eos_prob)
            target_probs.append(target_prob)
            target_log_probs.append(target_log_prob)
            target_ranks.append(target_rank)
            margins.append(target_logit - eos_logit)
            rows.append(
                {
                    "index": index,
                    "id": record.get("id"),
                    "target_token_id": target_id,
                    "target_token": processor.tokenizer.decode([target_id]),
                    "target_probability": target_prob,
                    "target_log_probability": target_log_prob,
                    "target_rank": target_rank,
                    "eos_probability": eos_prob,
                    "target_minus_eos_logit": target_logit - eos_logit,
                    "top1_token_id": top1_id,
                    "top1_token": processor.tokenizer.decode([top1_id]),
                    "top1_is_eos": top1_id in eos_ids,
                }
            )

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    total = len(rows)
    return {
        "checkpoint": str(checkpoint),
        "records": total,
        "manifest": _manifest_summary(checkpoint),
        "top1_eos_rate": top1_eos / total,
        "mean_eos_probability": sum(eos_probs) / total,
        "median_eos_probability": float(statistics.median(eos_probs)),
        "mean_target_probability": sum(target_probs) / total,
        "median_target_probability": float(statistics.median(target_probs)),
        "mean_target_log_probability": sum(target_log_probs) / total,
        "median_target_rank": float(statistics.median(target_ranks)),
        "mean_target_minus_eos_logit": sum(margins) / total,
        "median_target_minus_eos_logit": float(statistics.median(margins)),
        "rows": rows,
    }


def _compact(value: Mapping[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != "rows"}


def compare(args: argparse.Namespace) -> dict[str, object]:
    corpus = Path(args.corpus)
    records = _select_records(
        _load_gui_records(corpus),
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        max_records=args.max_records,
    )
    before = _evaluate_checkpoint(
        checkpoint=Path(args.before), records=records, device=args.device
    )
    after = _evaluate_checkpoint(
        checkpoint=Path(args.after), records=records, device=args.device
    )

    def number(value: Mapping[str, object], name: str) -> float:
        return float(value[name])

    deltas = {
        "mean_eos_probability": number(after, "mean_eos_probability")
        - number(before, "mean_eos_probability"),
        "median_eos_probability": number(after, "median_eos_probability")
        - number(before, "median_eos_probability"),
        "mean_target_probability": number(after, "mean_target_probability")
        - number(before, "mean_target_probability"),
        "mean_target_log_probability": number(after, "mean_target_log_probability")
        - number(before, "mean_target_log_probability"),
        "mean_target_minus_eos_logit": number(after, "mean_target_minus_eos_logit")
        - number(before, "mean_target_minus_eos_logit"),
        "median_target_minus_eos_logit": number(after, "median_target_minus_eos_logit")
        - number(before, "median_target_minus_eos_logit"),
        "median_target_rank": number(after, "median_target_rank")
        - number(before, "median_target_rank"),
    }
    probability_progress = (
        deltas["mean_eos_probability"] < -1e-4
        or deltas["mean_target_log_probability"] > 1e-3
        or deltas["mean_target_minus_eos_logit"] > 1e-3
        or deltas["median_target_rank"] < 0
    )
    result = {
        "corpus": str(corpus),
        "before": _compact(before),
        "after": _compact(after),
        "delta_after_minus_before": deltas,
        "probability_progress": probability_progress,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                **result,
                "before_rows": before["rows"],
                "after_rows": after["rows"],
            },
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
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument(
        "--corpus", default="artifacts/training/gui-policy-unigui-openmobile.jsonl"
    )
    parser.add_argument("--output", default="artifacts/diagnostics/gui-first-token.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    if not 0 < args.holdout_fraction <= 1:
        parser.error("--holdout-fraction must be in (0, 1]")
    print(json.dumps(compare(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
