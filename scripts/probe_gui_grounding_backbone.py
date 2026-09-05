#!/usr/bin/env python3
"""Measure raw screenshot grounding of a candidate backbone, independent of output format.

`evaluate_gui_owl_policy_checkpoint.py` scores a checkpoint under the shipping
GUI-Owl contract prompt, which asks for an `Action:` line plus a
`<tool_call>` JSON action. That is the right measurement for a GUI-Owl
descendant and the wrong one for a general backbone, which has never seen the
format: it would score zero on syntax and tell us nothing about whether the
model can see the target.

Format is trainable, and the contract bootstrap stage exists to teach it.
Grounding is what depth and width pruning were shown to destroy. This probe
therefore isolates grounding: it asks for a tap point in whatever form the
model likes, parses leniently, and scores the point against the annotated
target box. It selects the holdout exactly as the production evaluator does,
so the two runs cover the same records.

Coordinate space is the trap here. The corpus annotates boxes on a 0..1000
scale, GUI-Owl emits 0..1000, and most general VLMs emit absolute pixels. Any
prediction outside 0..1000 is therefore rescaled by the real screenshot size,
and the report says how many predictions needed it.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COORDINATE_SCALE = 1000.0

DEFAULT_PROMPT = (
    "Look at this Android screenshot.\n"
    "Task: {goal}\n\n"
    "Where should I tap next to make progress on this task?\n"
    "Answer with only the point, as JSON: {{\"point\": [x, y]}}\n"
    "Use integer coordinates on a 0-1000 scale, where (0,0) is the top-left "
    "corner and (1000,1000) is the bottom-right corner of the screenshot."
)

_NUMBER = r"-?\d+(?:\.\d+)?"
_PAIR_PATTERNS = (
    re.compile(r'"point"\s*:\s*\[\s*(%s)\s*,\s*(%s)\s*\]' % (_NUMBER, _NUMBER)),
    re.compile(r'"coordinate"\s*:\s*\[\s*(%s)\s*,\s*(%s)\s*\]' % (_NUMBER, _NUMBER)),
    re.compile(r'\[\s*(%s)\s*,\s*(%s)\s*\]' % (_NUMBER, _NUMBER)),
    re.compile(r'\(\s*(%s)\s*,\s*(%s)\s*\)' % (_NUMBER, _NUMBER)),
)
_BOX_PATTERN = re.compile(
    r'\[\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*,\s*(%s)\s*\]' % ((_NUMBER,) * 4)
)


def parse_point(text: str) -> tuple[float, float] | None:
    """Return the first point the model emitted, accepting common shapes."""

    if not text:
        return None
    # A 4-number box is a grounding answer too; collapse it to its centre
    # before the 2-number patterns can match its first half.
    box = _BOX_PATTERN.search(text)
    if box:
        x0, y0, x1, y1 = (float(v) for v in box.groups())
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    for pattern in _PAIR_PATTERNS:
        match = pattern.search(text)
        if match:
            return (float(match.group(1)), float(match.group(2)))
    return None


def to_corpus_scale(
    point: tuple[float, float], size: tuple[int, int] | None
) -> tuple[tuple[float, float], bool]:
    """Map a prediction onto the corpus 0..1000 scale, rescaling pixel space."""

    x, y = point
    if x <= COORDINATE_SCALE and y <= COORDINATE_SCALE:
        return (x, y), False
    if not size:
        return (x, y), False
    width, height = size
    if width <= 0 or height <= 0:
        return (x, y), False
    return (x / width * COORDINATE_SCALE, y / height * COORDINATE_SCALE), True


def box_distance(point: tuple[float, float], box: Sequence[float]) -> float:
    x, y = point
    x0, y0, x1, y1 = (float(v) for v in box)
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


def _image_and_goal(record: Mapping[str, object]) -> tuple[str | None, str]:
    image = None
    for message in record.get("messages") or ():
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "image":
                image = str(item.get("image") or "") or None
    return image, str(record.get("goal") or "")


def select_holdout(
    records: list[dict], *, seed: int, holdout_fraction: float, max_records: int
) -> list[dict]:
    """Replicate the production evaluator's holdout selection exactly."""

    rng = random.Random(seed)
    rng.shuffle(records)
    count = max(1, int(round(len(records) * holdout_fraction)))
    count = min(count, len(records))
    return records[:count][:max_records]


def probe(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    records = [
        json.loads(line)
        for line in Path(args.corpus).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = select_holdout(
        records,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        max_records=args.max_records,
    )
    selected = [r for r in selected if r.get("target_bbox")]
    if not selected:
        raise ValueError("no holdout records carry target_bbox")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
    )
    model.to(args.device)
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    in_box = within_50 = parsed = rescaled = 0
    distances: list[float] = []
    with output.open("w", encoding="utf-8") as handle, torch.no_grad():
        for index, record in enumerate(selected):
            image_path, goal = _image_and_goal(record)
            if not image_path or not Path(image_path).is_file():
                continue
            size = Image.open(image_path).size
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": args.prompt.format(goal=goal)},
                    ],
                }
            ]
            batch = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            batch.pop("labels", None)
            batch = {
                key: (value.to(args.device) if hasattr(value, "to") else value)
                for key, value in batch.items()
            }
            sequence = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            prompt_tokens = int(batch["input_ids"].shape[1])
            text = processor.tokenizer.decode(
                sequence[0, prompt_tokens:], skip_special_tokens=True
            ).strip()

            box = [float(v) for v in record["target_bbox"]]
            raw = parse_point(text)
            point = None
            was_rescaled = False
            hit = False
            distance = None
            if raw is not None:
                parsed += 1
                point, was_rescaled = to_corpus_scale(raw, size)
                rescaled += int(was_rescaled)
                distance = box_distance(point, box)
                hit = distance == 0.0
                in_box += int(hit)
                centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
                if math.dist(point, centre) <= 50.0:
                    within_50 += 1
                distances.append(distance)
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "id": record.get("id"),
                        "target_bbox": box,
                        "raw_point": raw,
                        "point_0_1000": point,
                        "rescaled_from_pixels": was_rescaled,
                        "image_size": list(size),
                        "in_box": hit,
                        "distance_to_box": distance,
                        "generation": text[:600],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(selected)
    summary = {
        "model": args.model,
        "corpus": args.corpus,
        "records": total,
        "parsed_point_rate": parsed / total if total else 0.0,
        "rescaled_from_pixel_space": rescaled,
        # Scored over all selected records: an unparseable answer counts as a
        # miss, so this cannot be inflated by refusing to answer.
        "point_in_box_rate": in_box / total if total else 0.0,
        "within_50_of_box_centre_rate": within_50 / total if total else 0.0,
        "median_distance_to_box": (
            float(statistics.median(distances)) if distances else None
        ),
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
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--corpus", default="artifacts/training/gui-policy-unigui-openmobile.jsonl"
    )
    parser.add_argument("--output", default="artifacts/diagnostics/grounding-probe.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args(argv)
    if args.max_records < 1 or args.max_new_tokens < 1:
        parser.error("record/token limits must be positive")
    if not 0 < args.holdout_fraction <= 1:
        parser.error("--holdout-fraction must be in (0, 1]")
    print(json.dumps(probe(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
