#!/usr/bin/env python3
"""Evaluate the 2B GUI-Owl teacher with its GUI-policy PEFT adapter.

This is a paid-run preflight: the compact student must not be asked to imitate a
teacher adapter that itself immediately emits EOS or fails the repository's
Action:/<tool_call> grammar.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from android_drlqn.gui_owl_student import resolve_peft_adapter_dir  # noqa: E402
from project_gui_owl_train import parse_gui_owl_output  # noqa: E402


def _to_device(batch: Mapping[str, object], device: str) -> dict[str, object]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _trim(ids: Sequence[int], eos_ids: set[int], pad_id: int | None) -> list[int]:
    result: list[int] = []
    for raw in ids:
        token = int(raw)
        if token in eos_ids or (pad_id is not None and token == pad_id):
            break
        result.append(token)
    return result


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


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("install the training extra: python -m pip install -e '.[train]'") from exc

    corpus = Path(args.corpus)
    records = _load_gui_records(corpus)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    holdout_count = max(1, int(round(len(records) * args.holdout_fraction)))
    holdout_count = min(holdout_count, len(records))
    selected = records[:holdout_count][: args.max_records]
    if not selected:
        raise ValueError("no GUI policy records selected")

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(args.base_model)
    adapter_dir = resolve_peft_adapter_dir(args.adapter_root, args.adapter_name)

    try:
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except TypeError:
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    model = PeftModel.from_pretrained(
        base,
        str(adapter_dir),
        adapter_name=args.adapter_name,
        is_trainable=False,
    )
    model.set_adapter(args.adapter_name)
    model.to(device)
    model.eval()

    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or ())}
    pad_id = processor.tokenizer.pad_token_id

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    first_eos = empty = valid = 0
    lengths: list[int] = []

    with output.open("w", encoding="utf-8") as handle, torch.no_grad():
        for index, record in enumerate(selected):
            messages = list(record["messages"])
            prompt = processor.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt = _to_device(prompt, device)
            prompt.pop("labels", None)
            sequence = model.generate(
                **prompt,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            prompt_tokens = int(prompt["input_ids"].shape[1])
            generated = [int(value) for value in sequence[0, prompt_tokens:].tolist()]
            is_first_eos = bool(generated and generated[0] in eos_ids)
            if is_first_eos:
                first_eos += 1
            trimmed = _trim(generated, eos_ids, pad_id)
            lengths.append(len(trimmed))
            text = processor.tokenizer.decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not text:
                empty += 1
            parsed = bool(text and parse_gui_owl_output(text) is not None)
            if parsed:
                valid += 1
            target = messages[-1].get("content") if isinstance(messages[-1], Mapping) else None
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "id": record.get("id"),
                        "generated_tokens": len(trimmed),
                        "first_token_eos": is_first_eos,
                        "empty": not bool(text),
                        "gui_parse_valid": parsed,
                        "generation": text,
                        "target": target,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(selected)
    summary = {
        "base_model": args.base_model,
        "adapter": str(adapter_dir),
        "corpus": str(corpus),
        "records": total,
        "first_token_eos_rate": first_eos / total,
        "empty_generation_rate": empty / total,
        "median_generated_tokens": float(statistics.median(lengths)),
        "gui_parse_rate": valid / total,
        "output": str(output),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="mPLUG/GUI-Owl-1.5-2B-Instruct")
    parser.add_argument("--adapter-root", default="artifacts/adapters/gui_policy")
    parser.add_argument("--adapter-name", default="gui_policy")
    parser.add_argument(
        "--corpus", default="artifacts/training/gui-policy-unigui-openmobile.jsonl"
    )
    parser.add_argument("--output", default="artifacts/diagnostics/gui-teacher-generations.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--holdout-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_records < 1 or args.max_new_tokens < 1:
        parser.error("record/token limits must be positive")
    if not 0 < args.holdout_fraction <= 1:
        parser.error("--holdout-fraction must be in (0, 1]")
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
