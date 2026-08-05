# Uni-GUI-OpenMobile, Qwen screen chat, and UI-MOPD

GUI-Owl uses two independent PEFT adapters over the same frozen
`mPLUG/GUI-Owl-1.5-2B-Instruct` base:

```text
GUI-Owl base
├── gui_policy LoRA  <- UI-MOPD/Uni-GUI-OpenMobile reviewed Android trajectories
└── chat LoRA        <- Qwen3.7-Plus generated screen-search conversations
```

Stage 1 is completion-only supervised fine-tuning. Stage 2 exposes the
UI-MOPD loss primitives needed by an on-policy rollout engine: platform-routed
teachers, K3 reverse-KL distillation, adaptive KL masking, group-relative
advantages, structured action reward, and asymmetric clipped policy updates.

## GUI-policy dataset

The source is `UI-MOPD/Uni-GUI-OpenMobile` (Apache-2.0). Each episode contains a
normalized `task.json` and `screenshot_stepN.png` files. The converter consumes
only reviewed `mobile_use` plans and local screenshots. Raw generation evidence,
remote screenshot URLs, and `code` fields are ignored.

Install the data extra and pin an immutable dataset revision:

```bash
python -m pip install -e '.[data]'

python scripts/prepare_unigui_gui_policy.py \
  --revision <DATASET_COMMIT_SHA> \
  --app settings \
  --output artifacts/training/gui-policy-unigui-openmobile.jsonl
```

Download is two-phase. First, all small `task.json` files are fetched. The app
and optional package filters are applied locally, then only screenshots for the
selected episodes are fetched. This avoids downloading the full dataset when the
initial policy corpus is restricted to Android Settings.

The converter maps the normalized dataset actions into GUI-Owl's bounded mobile
tool grammar:

```text
click/tap      -> mobile.click
long_press     -> mobile.long_press
swipe          -> mobile.swipe
type           -> mobile.type
system_button  -> mobile.system_button
wait           -> mobile.wait
answer         -> mobile.answer
terminate      -> mobile.terminate (opt-in for SFT export)
```

Coordinates remain on the integer `0..1000` scale. Pixel coordinates above that
range are normalized using the episode resolution. Unreviewed, deleted, unused,
unsupported, or image-missing steps are rejected. The manifest records the
source revision and explicitly states that raw code fields were not used.

## Chat dataset generation

The prepared GUI-policy JSONL carries the task goal, app/package, screen identity,
and labels inferred from the reviewed action trace. The chat generator sends only
those facts to Alibaba Model Studio; it exposes no Android execution tools.

```bash
export DASHSCOPE_API_KEY='...'
# International/Singapore legacy OpenAI-compatible endpoint. For production,
# prefer the workspace-dedicated compatible-mode URL shown in Model Studio.
export DASHSCOPE_BASE_URL='https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

python scripts/generate_qwen_screen_chat.py \
  --input artifacts/training/gui-policy-unigui-openmobile.jsonl \
  --output artifacts/training/chat-screen-qa-qwen.jsonl \
  --model qwen3.7-plus \
  --questions-per-screen 4 \
  --resume
```

The `/api/v1` DashScope endpoint is not interchangeable with this script's
OpenAI-compatible Chat Completions transport. The configured base URL must end
at a compatible-mode API root; the client appends `/chat/completions`.

Requests use JSON-object structured output and disable thinking. Each response
must match a closed schema containing `question`, `answer`, and one of the
intents `find`, `open`, `understand`, or `confirm`. Executable or privileged
guidance is rejected, retries are bounded, and the API key is never serialized.

## Stage 1: train the separate adapters

```bash
python -m pip install -e '.[train]'

python scripts/train_gui_owl_adapter.py \
  --adapter gui_policy \
  --output artifacts/adapters/gui_policy

python scripts/train_gui_owl_adapter.py \
  --adapter chat \
  --output artifacts/adapters/chat
```

The trainer masks the system/user prompt and applies loss only to the final
assistant completion. GUI records supervise action conclusions and tool calls;
chat records supervise ordinary natural-language answers.

## Stage 2: UI-MOPD mathematics

The reference implementation is `android_drlqn/ui_mopd.py`. It is model-agnostic
and can be called by the existing on-policy rollout/update layer.

For student-sampled token `y_t` under context `h_t`, route the rollout to the
teacher matching its platform and define:

```text
delta_t = log pi_ref(y_t | h_t) - log pi_theta(y_t | h_t)
rho_t   = exp(delta_t)
K3_t    = rho_t - delta_t - 1
```

`K3_t` is the non-negative sampled-token estimate of the reverse KL used by
UI-MOPD. Prompt and padding tokens are masked out. A prompt group disables the
teacher penalty once its mean outcome reward exceeds the configured threshold:

```text
mu_i = 0  when mean_group_reward > tau_KL
mu_i = 1  otherwise
```

The structured action reward is:

```text
1.0   all required action dimensions match
-0.5  parsable action with one or more mismatched dimensions
-1.0  unparsable or invalid action
```

Group-relative advantage is `R_i - mean(R_group)`. The policy term uses an
asymmetric clipped ratio with defaults `epsilon_low=0.20` and
`epsilon_high=0.28`. The minimized objective is:

```text
L = L_policy + beta * L_MOPD
```

with default `beta=0.01`, eight rollouts per prompt, and K3 delta clipping for
numerical stability. These defaults are recorded in
`config/gui_owl_dual_lora_data.json`.

`torch_ui_mopd_loss` is differentiable and expects current-policy, frozen-old-
policy, and routed-teacher sampled-token log probabilities plus response masks,
advantages, and adaptive KL masks. The repository does not pretend that this
alone reproduces the paper's complete 64-H100 `verl`/Megatron/SGLang training
stack; it provides the exact loss components and routing contract needed to wire
that stage into the project's own rollout infrastructure.

No third-party dataset files, generated Q&A, API credentials, model weights, or
adapter checkpoints are committed.
