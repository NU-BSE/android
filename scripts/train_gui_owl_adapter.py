#!/usr/bin/env python3
"""Train one GUI-Owl LoRA from a prepared GUI-policy or chat JSONL corpus."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


def load_adapter_config(path: str | Path, adapter_name: str) -> tuple[str, dict[str, object]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("dual-LoRA config must be an object")
    base_model = str(value.get("base_model") or "")
    adapters = value.get("adapters")
    if not base_model or not isinstance(adapters, Mapping):
        raise ValueError("dual-LoRA config must define base_model and adapters")
    raw = adapters.get(adapter_name)
    if not isinstance(raw, Mapping):
        raise ValueError(f"unknown adapter: {adapter_name}")
    return base_model, dict(raw)


def iter_messages(path: str | Path, *, max_records: int | None = None) -> Iterator[list[object]]:
    source = Path(path)
    count = 0
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        messages = value.get("messages") if isinstance(value, Mapping) else None
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError(f"{source}:{line_number}: messages must be an array")
        if len(messages) < 2 or not isinstance(messages[-1], Mapping):
            raise ValueError(f"{source}:{line_number}: assistant completion is missing")
        if str(messages[-1].get("role") or "") != "assistant":
            raise ValueError(f"{source}:{line_number}: final message must be assistant")
        yield list(messages)
        count += 1
        if max_records is not None and count >= max_records:
            break


def _default_device(torch_module: object) -> str:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and bool(cuda.is_available()):
        return "cuda"
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and bool(mps.is_available()):
        return "mps"
    return "cpu"


def _dtype_for_device(torch_module: object, device: str):
    if device.startswith("cuda"):
        return torch_module.bfloat16
    if device.startswith("mps"):
        return torch_module.float16
    return torch_module.float32


def _to_device(batch: Mapping[str, object], device: str) -> dict[str, object]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _encode_completion(processor, messages: Sequence[object], device: str) -> dict[str, object]:
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
    prompt, full = _to_device(prompt, device), _to_device(full, device)
    labels = full["input_ids"].clone()
    prompt_length = min(int(prompt["input_ids"].shape[1]), int(labels.shape[1]))
    labels[:, :prompt_length] = -100
    full["labels"] = labels
    return full


def train(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is missing; install the training extra: "
            "python -m pip install -e '.[train]'"
        ) from exc
    try:
        import torchvision  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "torchvision is missing or incompatible with torch. Reinstall the "
            "training extra together: python -m pip install --upgrade --force-reinstall "
            "torch torchvision && python -m pip install -e '.[train]'"
        ) from exc
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "install the training extra: python -m pip install -e '.[train]'"
        ) from exc

    base_model, adapter = load_adapter_config(args.config, args.adapter)
    data_path = Path(args.data or str(adapter.get("prepared_jsonl") or ""))
    if not data_path.is_file():
        raise FileNotFoundError(f"prepared adapter data not found: {data_path}")
    lora = adapter.get("lora")
    if not isinstance(lora, Mapping):
        raise ValueError("adapter config lacks lora settings")

    set_seed(args.seed)
    device = args.device or _default_device(torch)
    dtype = _dtype_for_device(torch, device)
    processor = AutoProcessor.from_pretrained(base_model)
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model, dtype=dtype, attn_implementation="sdpa"
        )
    except TypeError:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model, torch_dtype=dtype, attn_implementation="sdpa"
        )
    model = get_peft_model(
        model.to(device),
        LoraConfig(
            r=int(lora.get("rank", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.0)),
            target_modules=[str(item) for item in lora.get("target_modules") or ()],
            bias="none",
            task_type="CAUSAL_LM",
        ),
        adapter_name=args.adapter,
    )
    model.config.use_cache = False
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    records = list(iter_messages(data_path, max_records=args.max_records))
    if not records:
        raise ValueError("prepared adapter corpus is empty")
    total_updates = math.ceil(len(records) * args.epochs / args.gradient_accumulation)
    optimizer.zero_grad(set_to_none=True)
    micro_step = updates = 0
    final_loss = 0.0
    for _ in range(args.epochs):
        for messages in records:
            batch = _encode_completion(processor, messages, device)
            loss = model(**batch).loss / args.gradient_accumulation
            loss.backward()
            final_loss = float(loss.detach().float().cpu()) * args.gradient_accumulation
            micro_step += 1
            if micro_step % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
    if micro_step % args.gradient_accumulation:
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1

    output = Path(args.output or f"artifacts/adapters/{args.adapter}")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, selected_adapters=[args.adapter])
    processor.save_pretrained(output)
    manifest = {
        "adapter": args.adapter,
        "base_model": base_model,
        "data": str(data_path),
        "device": device,
        "dtype": str(dtype).removeprefix("torch."),
        "records": len(records),
        "epochs": args.epochs,
        "updates": updates,
        "planned_updates": total_updates,
        "final_loss": final_loss,
        "api_credentials_written": False,
    }
    (output / "training-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, choices=("gui_policy", "chat"))
    parser.add_argument("--config", default="config/gui_owl_dual_lora_data.json")
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.gradient_accumulation < 1:
        parser.error("epochs and gradient accumulation must be positive")
    print(json.dumps(train(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
