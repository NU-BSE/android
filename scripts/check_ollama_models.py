#!/usr/bin/env python3
"""Validate local Qwen3-VL and UI-TARS models on Ollama port 11434."""
from __future__ import annotations

import argparse
import json

from android_drlqn.ollama_runtime import (
    OllamaRuntimeBundle,
    smoke_image_base64,
)

_SMOKE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ollama_models.json")
    parser.add_argument("--smoke", action="store_true", help="run one multimodal request per model")
    args = parser.parse_args()

    bundle = OllamaRuntimeBundle.from_json(args.config)
    qwen, ui_tars = bundle.preflight()
    result = {
        "base_url": bundle.config.server.normalized_base_url,
        "models": [qwen.__dict__, ui_tars.__dict__],
        "smoke": {},
    }
    if args.smoke:
        image = smoke_image_base64()
        qwen_response = bundle.client.chat(
            bundle.config.qwen3_vl,
            (
                {"role": "system", "content": "Return JSON matching the schema."},
                {"role": "user", "content": "Inspect the image and return {\"ok\":true}."},
            ),
            images=(image,),
            response_format=_SMOKE_SCHEMA,
            max_new_tokens=32,
            think=False,
        )
        ui_response = bundle.client.chat(
            bundle.config.ui_tars,
            ({"role": "user", "content": "Inspect the image. Return one UI-TARS Action_Summary and WAIT() action."},),
            images=(image,),
            max_new_tokens=80,
            think=False,
        )
        result["smoke"] = {
            "qwen3_vl": bundle.client.content(qwen_response),
            "ui_tars": bundle.client.content(ui_response),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
