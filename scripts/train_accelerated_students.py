"""Compare accelerator-backed student architectures on grouped held-out games."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drlqn.accelerator.models import build_accelerator_model, require_torch
from drlqn.train.dense_sft import prepare_dense_example
from drlqn.train.duck_dataset import DuckExample, load_duck_traces


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        choices=("large-drlqn", "recurrent-visual", "tile-transformer", "drlqn-no-jvp"),
        default=[],
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--held-out-game", action="append", default=[])
    parser.add_argument("--held-out-game-count", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints/accelerated"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/accelerated_student_ablation.json"),
    )
    return parser.parse_args()


def _prepared_pairs(examples: list[DuckExample]):
    pairs = []
    for example in examples:
        prepared = prepare_dense_example(example)
        if prepared is not None:
            pairs.append((example, prepared))
    return pairs


def _effect(example: DuckExample) -> np.ndarray:
    return np.asarray(
        [
            float(example.action) / 9.0,
            float(bool(example.visual_changed)),
            float(bool(example.level_advanced)),
            float(example.level_before or 0) / 128.0,
            float(example.x if example.x is not None else -1) / 64.0,
            float(example.y if example.y is not None else -1) / 64.0,
        ],
        dtype=np.float32,
    )


def _tensor_batch(torch, batch, device):
    examples = [item[0] for item in batch]
    prepared = [item[1] for item in batch]
    grids = torch.as_tensor(
        np.stack([item.grid for item in examples]),
        device=device,
        dtype=torch.long,
    )
    next_grids = torch.as_tensor(
        np.stack(
            [
                item.next_grid if item.next_grid is not None else item.grid
                for item in examples
            ]
        ),
        device=device,
        dtype=torch.long,
    )
    labels = torch.as_tensor(
        [item.label for item in prepared], device=device, dtype=torch.long
    )
    weights = torch.as_tensor(
        [item.weight for item in prepared], device=device, dtype=torch.float32
    )
    changed = torch.as_tensor(
        [max(0, item.visual_changed) for item in prepared],
        device=device,
        dtype=torch.float32,
    )
    advanced = torch.as_tensor(
        [max(0, item.level_advanced) for item in prepared],
        device=device,
        dtype=torch.float32,
    )
    effects = torch.as_tensor(
        np.stack([_effect(item) for item in examples]), device=device
    )
    return grids, next_grids, labels, weights, changed, advanced, effects


def _iterate(items, batch_size, rng):
    order = rng.permutation(len(items))
    for start in range(0, len(order), batch_size):
        yield [items[int(index)] for index in order[start : start + batch_size]]


def _evaluate(torch, model, items, batch_size, device):
    model.eval()
    correct = 0
    total = 0
    advances = []
    change_correct = []
    per_game: dict[str, list[bool]] = {}
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            grids, _next, labels, _weights, changed, advanced, effects = _tensor_batch(
                torch, batch, device
            )
            output = model(grids, effects)
            prediction = output["logits"].argmax(dim=-1)
            matches = (prediction == labels).cpu().tolist()
            correct += int(sum(matches))
            total += len(batch)
            for pair, matched in zip(batch, matches):
                per_game.setdefault(pair[0].game_id, []).append(bool(matched))
            positive = advanced > 0.5
            if bool(positive.any()):
                advances.extend(
                    (output["level_advance"][positive] >= 0).cpu().tolist()
                )
            change_correct.extend(
                ((output["frame_change"] >= 0) == (changed > 0.5)).cpu().tolist()
            )
    return {
        "count": total,
        "action_accuracy": correct / max(total, 1),
        "level_advance_recall": float(np.mean(advances)) if advances else 0.0,
        "frame_change_accuracy": (
            float(np.mean(change_correct)) if change_correct else 0.0
        ),
        "per_game_action_accuracy": {
            game: float(np.mean(values)) for game, values in sorted(per_game.items())
        },
    }


def main() -> None:
    args = _parse_args()
    torch, _nn = require_torch()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("epochs and batch size must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    pairs = _prepared_pairs(load_duck_traces(args.datasets))
    if not pairs:
        raise SystemExit("no usable examples")
    games = sorted({item[0].game_id for item in pairs})
    requested = [game for game in args.held_out_game if game in games]
    count = min(args.held_out_game_count, max(1, len(games) - 1))
    held_games = set(requested or games[-count:])
    train = [item for item in pairs if item[0].game_id not in held_games]
    held = [item for item in pairs if item[0].game_id in held_games]
    if not train or not held:
        raise SystemExit("grouped held-out split must contain train and held-out examples")

    mps = getattr(torch.backends, "mps", None)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if mps and mps.is_available()
        else "cpu"
    )
    names = args.model or [
        "large-drlqn",
        "recurrent-visual",
        "tile-transformer",
        "drlqn-no-jvp",
    ]
    rng = np.random.default_rng(args.seed)
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        model = build_accelerator_model(name).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=1e-4
        )
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in _iterate(train, args.batch_size, rng):
                (
                    grids,
                    next_grids,
                    labels,
                    weights,
                    changed,
                    advanced,
                    effects,
                ) = _tensor_batch(torch, batch, device)
                output = model(grids, effects, selected_action=labels)
                with torch.no_grad():
                    next_target = model(next_grids, effects)["embedding"]
                action = torch.nn.functional.cross_entropy(
                    output["logits"], labels, reduction="none"
                )
                action = (action * weights).sum() / weights.sum().clamp_min(1e-6)
                change = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["frame_change"], changed
                )
                advance = torch.nn.functional.binary_cross_entropy_with_logits(
                    output["level_advance"], advanced
                )
                next_loss = torch.nn.functional.mse_loss(
                    output["next_embedding"], next_target
                )
                loss = action + 0.3 * change + 0.5 * advance + 0.2 * next_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            metrics = _evaluate(torch, model, held, args.batch_size, device)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "held_out": metrics,
                }
            )
            print(json.dumps({"model": name, **history[-1]}, sort_keys=True))
        checkpoint = args.output_dir / f"{name}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": model.config.__dict__,
                "held_out_games": sorted(held_games),
            },
            checkpoint,
        )
        results.append(
            {
                "model": name,
                "checkpoint": str(checkpoint),
                "differential_runtime": bool(model.config.differential_runtime),
                "parameters": int(
                    sum(parameter.numel() for parameter in model.parameters())
                ),
                "history": history,
                "final": history[-1]["held_out"],
            }
        )

    payload = {
        "device": str(device),
        "train_examples": len(train),
        "held_out_examples": len(held),
        "held_out_games": sorted(held_games),
        "models": results,
        "selection_rule": (
            "retain sparse executor/JVP only if the official environment benchmark "
            "improves held-out completion or action efficiency"
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"metrics: {args.metrics}")


if __name__ == "__main__":
    main()
