#!/usr/bin/env python3
"""LoRA contract adaptation for an unpruned Qwen3.5 backbone on GUI-policy data.

Stage 1 of docs/qwen35_08b_training_plan.md. The backbone grounds well and has
one action: measured on the 192-record baseline slice it emits mobile.click for
165 of 192 records, scores 0.922 recall on click and 0.000 on system_button,
answer, long_press and wait, for a macro recall of 0.172 over seven classes
against a uniform-chance 0.143. What it needs is action coverage. What it must
not lose is aim, and this corpus's supervised coordinates are exactly the signal
that pulled every pruned student to the corpus median.

So the coordinate objective is a flag, not a decision, and the two arms are run
against the same holdout:

    --coordinate-loss weighted   coordinate literals weighted --coordinate-weight
    --coordinate-loss masked     coordinate literals excluded from the loss

Under `masked` the structural tokens ("coordinate", brackets, commas) stay
supervised, so format is taught and aim is left to the pretrained weights.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.gui_owl_contract import normalize_gui_messages  # noqa: E402
from android_drlqn.gui_owl_metrics import assistant_content_text  # noqa: E402

# Qwen3.5 is a hybrid stack: 18 linear-attention layers carry in_proj_qkv /
# in_proj_z / out_proj, 6 full-attention layers carry q/k/v/o_proj. The GUI-Owl
# list (q,k,v,o + MLP) attaches to 96 modules and reaches none of the 18. With
# the three linear-attention projections it reaches 150 modules / 20.4M
# parameters. in_proj_a, in_proj_b, conv1d, A_log and dt_bias are deliberately
# excluded: they parameterise the recurrence's gating and decay, and a low-rank
# perturbation of a decay rate is not a low-rank perturbation of a function.
QWEN35_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
)

_COORDINATE = re.compile(r'"coordinate2?"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]')
_ACTION_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _load(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{number}: record must be an object")
        record = dict(value)
        record["messages"] = normalize_gui_messages(record["messages"])
        records.append(record)
    if not records:
        raise ValueError(f"corpus is empty: {path}")
    return records


def _action_name(record: Mapping[str, object]) -> str:
    messages = record.get("messages") or ()
    target = assistant_content_text(messages[-1].get("content")) if messages else ""
    found = _ACTION_NAME.search(target)
    return found.group(1) if found else "unknown"


def token_char_spans(tokenizer, ids: Sequence[int]) -> list[tuple[int, int]] | None:
    """Character span of each token inside the decoded sequence.

    Cumulative decoding rather than offset mapping, because the completion is
    reached through a chat template and its tokens never pass through the
    tokenizer's own offset-tracking path. Returns None when decoding is not
    monotonic, which happens when a token is a partial UTF-8 sequence; the
    caller then leaves that record unmasked rather than masking the wrong span.
    """
    spans: list[tuple[int, int]] = []
    previous = ""
    for index in range(len(ids)):
        current = tokenizer.decode(list(ids[: index + 1]), skip_special_tokens=False)
        if not current.startswith(previous):
            return None
        spans.append((len(previous), len(current)))
        previous = current
    return spans


def coordinate_token_flags(tokenizer, ids: Sequence[int]) -> list[bool] | None:
    """Flag the tokens holding the integer literals of a coordinate pair."""
    spans = token_char_spans(tokenizer, ids)
    if spans is None:
        return None
    text = tokenizer.decode(list(ids), skip_special_tokens=False)
    literal_spans: list[tuple[int, int]] = []
    for match in _COORDINATE.finditer(text):
        literal_spans.extend((match.span(1), match.span(2)))
    if not literal_spans:
        return [False] * len(ids)
    flags: list[bool] = []
    for start, end in spans:
        flags.append(
            any(start < stop and begin < end for begin, stop in literal_spans)
        )
    return flags


def _sample_weights(records: Sequence[Mapping[str, object]], mode: str) -> list[float]:
    if mode == "none":
        return [1.0] * len(records)
    counts = collections.Counter(_action_name(record) for record in records)
    if mode == "balanced":
        return [1.0 / counts[_action_name(record)] for record in records]
    if mode == "sqrt":
        return [1.0 / math.sqrt(counts[_action_name(record)]) for record in records]
    raise ValueError(f"unknown --action-balance: {mode}")


def train(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    records = _load(args.train)
    weights = _sample_weights(records, args.action_balance)
    classes = collections.Counter(_action_name(record) for record in records)

    processor = AutoProcessor.from_pretrained(args.model)
    if args.max_image_pixels:
        size = dict(getattr(processor.image_processor, "size", {}) or {})
        size["longest_edge"] = int(args.max_image_pixels)
        processor.image_processor.size = size

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
    )
    model.to(device)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(QWEN35_LORA_TARGETS),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    adapted = sum(1 for name, _ in model.named_modules() if name.endswith("lora_A.default"))
    if adapted < len(QWEN35_LORA_TARGETS):
        raise RuntimeError(
            f"LoRA attached to only {adapted} modules; the target list does not "
            f"match {args.model}"
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    warmup = max(1, int(args.max_updates * args.warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.max_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    history: list[dict[str, object]] = []
    unmaskable = skipped = 0
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for update in range(args.max_updates):
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        supervised = 0
        for _ in range(args.gradient_accumulation):
            record = rng.choices(records, weights=weights, k=1)[0]
            messages = list(record["messages"])
            try:
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
            except (OSError, ValueError):
                skipped += 1
                continue
            prompt_length = int(prompt["input_ids"].shape[1])
            ids = full["input_ids"]
            total = int(ids.shape[1])
            completion = total - prompt_length
            if completion < 1:
                skipped += 1
                continue

            completion_ids = [int(value) for value in ids[0, prompt_length:].tolist()]
            flags = coordinate_token_flags(processor.tokenizer, completion_ids)
            if flags is None:
                unmaskable += 1
                flags = [False] * completion
            if args.coordinate_loss == "masked":
                token_weight = [0.0 if flag else 1.0 for flag in flags]
            else:
                token_weight = [args.coordinate_weight if flag else 1.0 for flag in flags]
            if not any(token_weight):
                skipped += 1
                continue

            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in full.items()
            }
            batch.pop("labels", None)
            keep = completion + 1
            try:
                out = model(**batch, logits_to_keep=keep)
                logits = out.logits[:, :-1, :]
            except TypeError:
                out = model(**batch)
                logits = out.logits[:, prompt_length - 1 : total - 1, :]

            targets = ids[0, prompt_length:].to(device)
            weight_tensor = torch.tensor(token_weight, device=device, dtype=torch.float32)
            token_loss = F.cross_entropy(
                logits[0].float(), targets, reduction="none"
            )
            denominator = weight_tensor.sum().clamp(min=1e-6)
            loss = (token_loss * weight_tensor).sum() / denominator
            (loss / args.gradient_accumulation).backward()
            batch_loss += float(loss.detach())
            supervised += 1

        if not supervised:
            continue
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if (update + 1) % args.log_interval == 0 or update == 0:
            entry = {
                "update": update + 1,
                "loss": batch_loss / supervised,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - started, 1),
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)

    model.save_pretrained(str(output))
    processor.save_pretrained(str(output))

    merged_output = ""
    if args.merged_output:
        # The evaluator loads a checkpoint with AutoModelForImageTextToText and
        # cannot read an adapter directory, so write a standalone checkpoint
        # while the weights are still resident.
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(str(args.merged_output), safe_serialization=True)
        processor.save_pretrained(str(args.merged_output))
        merged_output = str(args.merged_output)

    manifest = {
        "schema_version": 1,
        "base_model": args.model,
        "train_corpus": str(args.train),
        "train_records": len(records),
        "train_action_classes": dict(classes.most_common()),
        "arm": args.coordinate_loss,
        "coordinate_weight": args.coordinate_weight if args.coordinate_loss == "weighted" else 0.0,
        "action_balance": args.action_balance,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(QWEN35_LORA_TARGETS),
            "adapted_modules": adapted,
            "trainable_parameters": sum(p.numel() for p in trainable),
        },
        "optimisation": {
            "learning_rate": args.learning_rate,
            "max_updates": args.max_updates,
            "gradient_accumulation": args.gradient_accumulation,
            "warmup_updates": warmup,
            "grad_clip": args.grad_clip,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "records_skipped": skipped,
        "records_without_coordinate_spans": unmaskable,
        "max_image_pixels": args.max_image_pixels,
        "elapsed_s": round(time.time() - started, 1),
        "history": history,
        "output": str(output),
        "merged_output": merged_output,
    }
    (output / "training-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "history"}, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--merged-output",
        type=Path,
        help="also write a standalone merged checkpoint the evaluator can load",
    )
    parser.add_argument(
        "--coordinate-loss", choices=("weighted", "masked"), required=True
    )
    parser.add_argument("--coordinate-weight", type=float, default=3.0)
    parser.add_argument(
        "--action-balance", choices=("none", "sqrt", "balanced"), default="sqrt"
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-updates", type=int, default=300)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-image-pixels", type=int)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.coordinate_weight <= 0:
        parser.error("--coordinate-weight must be positive")
    if args.max_updates < 1 or args.gradient_accumulation < 1:
        parser.error("--max-updates and --gradient-accumulation must be positive")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
