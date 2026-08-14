#!/usr/bin/env python3
"""Measure what removing text layers does to `<tool_call>` token geometry.

A progressive depth step is a destructive edit applied to a known parent, so its
effect can be measured without training: drop exactly the layers the next depth
would drop and re-probe the same held-out records.  This isolates the structural
contribution to GUI grammar loss from anything the following recovery or
specialization stage later does to the weights.

Layers are named by their *original* teacher ids, the way the compression
lineage records them, and mapped onto current stack positions via
`compression-lineage.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from probe_gui_owl_tool_call_rank import (  # noqa: E402
    _load_gui_records,
    _select_records,
    load_checkpoint,
    measure_model,
)


def _original_layer_ids(checkpoint: Path, depth: int) -> list[int]:
    path = checkpoint / "compression-lineage.json"
    if not path.is_file():
        return list(range(depth))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        return list(range(depth))
    raw = value.get("text_original_layer_ids")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return list(range(depth))
    ids = [int(item) for item in raw]
    if len(ids) != depth:
        raise ValueError(
            f"compression lineage lists {len(ids)} text layers but the checkpoint has {depth}"
        )
    return ids


def _text_model(model):
    return model.model.language_model


def _apply_depth(model, keep_positions: Sequence[int]) -> None:
    """Rebuild the text stack from the retained decoder layers, in place."""
    import torch

    text_model = _text_model(model)
    retained = [text_model.layers[position] for position in keep_positions]
    for index, layer in enumerate(retained):
        # layer_idx addresses the KV cache; a renumbered stack must stay contiguous.
        layer.self_attn.layer_idx = index
    text_model.layers = torch.nn.ModuleList(retained)
    for config in (getattr(model.config, "text_config", None), getattr(text_model, "config", None)):
        if config is not None and hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = len(retained)


def _parse_variants(values: Sequence[str]) -> list[tuple[str, tuple[int, ...]]]:
    variants: list[tuple[str, tuple[int, ...]]] = []
    for raw in values:
        name, separator, spec = raw.partition("=")
        if not separator:
            name, spec = f"drop-{raw}", raw
        dropped = tuple(int(item) for item in spec.split(",") if item.strip())
        if not dropped:
            raise ValueError(f"variant {raw!r} drops no layer")
        variants.append((name, dropped))
    return variants


def ablate(args: argparse.Namespace) -> dict[str, object]:
    import torch

    checkpoint = Path(args.checkpoint)
    records = _select_records(
        _load_gui_records(Path(args.corpus)),
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        max_records=args.max_records,
    )
    variants = _parse_variants(args.drop)

    model, processor = load_checkpoint(checkpoint, device=args.device, dtype_name=args.dtype)
    text_model = _text_model(model)
    full_layers = list(text_model.layers)
    depth = len(full_layers)
    original_ids = _original_layer_ids(checkpoint, depth)
    position_of = {layer_id: index for index, layer_id in enumerate(original_ids)}

    def restore() -> None:
        for index, layer in enumerate(full_layers):
            layer.self_attn.layer_idx = index
        text_model.layers = torch.nn.ModuleList(full_layers)
        for config in (
            getattr(model.config, "text_config", None),
            getattr(text_model, "config", None),
        ):
            if config is not None and hasattr(config, "num_hidden_layers"):
                config.num_hidden_layers = depth

    def measure(label: str) -> dict[str, object]:
        return measure_model(
            model=model,
            processor=processor,
            records=records,
            device=args.device,
            token=args.token,
            image_prefix=args.image_prefix,
            image_root=args.image_root,
            label=label,
        )

    results: list[dict[str, object]] = []
    baseline = measure(f"baseline-depth-{depth}")
    baseline["dropped_original_layers"] = []
    baseline["depth"] = depth
    results.append(baseline)

    try:
        for name, dropped in variants:
            unknown = [layer_id for layer_id in dropped if layer_id not in position_of]
            if unknown:
                raise ValueError(
                    f"variant {name}: original layer ids not present in this checkpoint: {unknown}"
                )
            drop_positions = {position_of[layer_id] for layer_id in dropped}
            keep = [index for index in range(depth) if index not in drop_positions]
            _apply_depth(model, keep)
            try:
                summary = measure(name)
            finally:
                restore()
            summary["dropped_original_layers"] = list(dropped)
            summary["dropped_positions"] = sorted(drop_positions)
            summary["depth"] = len(keep)
            results.append(summary)
    finally:
        restore()
        del model

    def compact(value: Mapping[str, object]) -> dict[str, object]:
        return {key: item for key, item in value.items() if key != "rows"}

    reference = float(baseline.get("median_token_rank", 0.0) or 0.0)
    comparison = []
    for summary in results:
        rank = summary.get("median_token_rank")
        entry = compact(summary)
        if rank is not None and reference:
            entry["median_rank_ratio_vs_baseline"] = float(rank) / reference
        comparison.append(entry)

    result = {
        "checkpoint": str(checkpoint),
        "corpus": str(args.corpus),
        "records": len(records),
        "token": args.token,
        "baseline_depth": depth,
        "original_layer_ids": original_ids,
        "variants": comparison,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {**result, "rows_by_variant": {item["checkpoint"]: item["rows"] for item in results}},
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--drop",
        action="append",
        required=True,
        help="original layer ids to remove, e.g. '11,15,22,24' or 'tail=22,24'",
    )
    parser.add_argument(
        "--corpus", default="artifacts/training/gui-policy-unigui-openmobile.jsonl"
    )
    parser.add_argument("--output", default="artifacts/diagnostics/gui-layer-ablation.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16")
    )
    parser.add_argument("--token", default="<tool_call>")
    parser.add_argument("--image-prefix", default="")
    parser.add_argument("--image-root", default="")
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
    print(json.dumps(ablate(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
