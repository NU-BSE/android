# MobileForge GUI policy and Qwen screen-chat LoRAs

GUI-Owl uses two independent PEFT adapters over the same frozen
`mPLUG/GUI-Owl-1.5-2B-Instruct` base model:

```text
GUI-Owl base
├── gui_policy LoRA  <- MobileForge Android Settings exploration trajectories
└── chat LoRA        <- Qwen3.7-Plus generated screen-search conversations
```

The adapters target the language-transformer attention and MLP projections. The
shared token embedding / LM-head matrix remains frozen.

## GUI-policy dataset

The source is `lgy0404/mobileforge-exploration-trajectories`. Preparation is
restricted to the parsed `final/com.android.settings` JSON/JSONL artifacts and
screen images. The release's compressed pickle objects are never deserialized;
they are executable Python serialization and are outside this pipeline's trust
boundary.

Install the data extra and pin an immutable dataset revision:

```bash
python -m pip install -e '.[data]'

python scripts/prepare_mobileforge_gui_policy.py \
  --revision <DATASET_COMMIT_SHA> \
  --output artifacts/training/gui-policy-mobileforge.jsonl
```

The converter maps only bounded click, swipe, Back, and wait actions to the
GUI-Owl tool grammar. It rejects ADB, shell, root, AccessibilityService, Settings
Provider, and other executable strings. Coordinates are normalized to GUI-Owl's
integer `0..1000` scale. The manifest records the dataset revision and states
that no pickle content was loaded.

## Chat dataset generation

The prepared GUI-policy JSONL also carries screen identity, goal, visible labels,
and optional route metadata. The chat generator sends those facts—not Android
control tools—to Alibaba Model Studio's OpenAI-compatible chat endpoint.

```bash
export DASHSCOPE_API_KEY='...'
# Prefer a workspace-dedicated Model Studio URL when available.
export DASHSCOPE_BASE_URL='https://dashscope-intl.aliyuncs.com/api/v1'

python scripts/generate_qwen_screen_chat.py \
  --input artifacts/training/gui-policy-mobileforge.jsonl \
  --output artifacts/training/chat-screen-qa-qwen.jsonl \
  --model qwen3.7-plus \
  --questions-per-screen 4 \
  --resume
```

Requests use JSON-object structured output and disable thinking. Responses must
match a closed schema with `question`, `answer`, and one of the intents `find`,
`open`, `understand`, or `confirm`. The generator rejects privileged or
executable guidance, retries bounded transient failures, and never writes the
API key to disk.

The resulting chat records contain ordinary system/user/assistant completions,
without GUI tool calls. This prevents natural chat from collapsing into the
action grammar and keeps navigation policy responses concise.

## Train the separate adapters

Install the training extra, then train each named adapter from its own grammar:

```bash
python -m pip install -e '.[train]'

python scripts/train_gui_owl_adapter.py \
  --adapter gui_policy \
  --output artifacts/adapters/gui_policy

python scripts/train_gui_owl_adapter.py \
  --adapter chat \
  --output artifacts/adapters/chat
```

Training masks every system/user prompt and applies completion loss only to the
final assistant turn. GUI records therefore supervise GUI-Owl tool calls, while
chat records supervise plain natural-language answers. Each output directory
contains only its PEFT adapter, processor metadata, and a credential-free
training manifest.

Configuration for datasets, adapter names, and LoRA targets is in
`config/gui_owl_dual_lora_data.json`. The repository does not commit third-party
data, generated Q&A, API credentials, model weights, or adapter checkpoints.
