# Stage 3 runbook: on-policy on the emulator

Stage 3 of `qwen35_08b_training_plan.md`. The student acts on a live emulator,
the environment verifies transitions, a local teacher corrects what the
environment cannot judge, and the policy is updated from grouped rollouts.

Before writing a plan for it, the loop was run. **Three of its four links were
broken**, all in ways that would have produced a training run that looked
healthy and taught the wrong thing. Each is fixed and each fix is measured on
the live emulator, not argued.

## First: a correction to the Stage 2 handoff

Stage 2 moved `mobile.answer` here on the grounds that "terminal verification is
exactly the signal Stage 3 supplies". **That was wrong.** `mobile.answer` maps
to `ProposalKind.EXPLAIN`, and the environment's first branch is:

```python
if kind in (ProposalKind.EXPLAIN, ProposalKind.ASK_CONFIRMATION):
    return EnvironmentOutcome(reward=0.0, terminal=True, verified=False,
        next_observation=None,
        reason="language turn; verification delegated to the teacher")
```

Reward 0.0, `verified=False`, delegated to the teacher — and the teacher is the
merged 2B, which scores **0/9 on `answer`** and never emits it. So `answer` is
unsupervised by the environment *and* unsupervised by the teacher. It is not
solved by Stage 3 as built.

Supervising it needs one of: an environment predicate that checks the answer
string against the task's expected output (the corpus has it — `answer` is the
final step of 80 of its 88 episodes, so episode termination is a usable label),
or a teacher that is not the 2B. Neither is in scope here. Stage 3 excludes
`answer` from its gate and says so, rather than carrying a claim it cannot pay.

## What was broken

### 1. The reward function punished correct navigation

`_looks_like_settings_search_screen` tested every visible label for the
substring `"search"`. The API 29 Settings homepage carries a permanent
`com.android.settings:id/search_bar` whose title reads **"Search settings"**, so
the guard classified the untouched homepage as the search screen. Every episode
starts there. The first action of every rollout was reverted with a BACK
keypress, scored **-0.25**, and marked unverified.

Measured on the live emulator, one tap on "Network & internet":

| | reward | verified | reason |
|---|---|---|---|
| before | **-0.25** | False | "Settings Search was opened; action reverted" |
| after | **+1.0** | True | "expected setting verified after a UI transition" |

The real discriminator was measured, not guessed: search runs in its own
package, `com.android.settings.intelligence/...search.SearchActivity`, against
the homepage's `com.android.settings/...homepage.SettingsHomepageActivity`. The
guard now tests the focused component. Three regression tests cover the
homepage, the real search screen, and a sub-screen with a search affordance.

### 2. The rollout ran a stale second copy of the contract prompt

`project_gui_owl_train.py` carried its own 637-character system prompt; the
corpus, the gates and every measurement in this project use the 1152-character
`GUI_OWL_SYSTEM_PROMPT`. They were not the same text, and the rollout copy still
had the defect fixed in `600f48d` — it wrote the action space as a literal
alternation inside a worked tool call.

The merged 2B, on a real screenshot, under the old prompt:

```
<tool_call>{"name": "mobile.click|mobile.system_button|mobile.wait|mobile.answer|mobile.interact",
            "arguments": {"text": "I clicked on \"Network & internet\" ..."}}
```

It copied the alternation verbatim as the tool name. That parses to
`ask_confirmation` with no coordinate — so every rollout step would have been
graded against a nonsense action. At the configured 96 new tokens it did not
even finish the tool call, and degraded to the WAIT fallback.

Same model, same screenshot, under the shipping contract:

```
<tool_call>{"name":"mobile.click","arguments":{"coordinate":[340,670]}}
```

`project_gui_owl_train.py` now imports the contract prompt. There is one
contract, not two. This matters beyond the parse: the findings doc measures
prompt framing as worth ~16 points to the teacher and states that no
cross-prompt comparison is meaningful, so a rollout under a private prompt is
not comparable to anything Stages 0-2 measured.

### 3. There was no teacher

`config/distillation.json` configures `flux3_early_access` with
`early_access_acknowledged: false` and `endpoint: null` — unusable by
construction. The alternatives the entrypoint can build are an OpenAI-compatible
HTTP endpoint and `replay_teacher`, which returns confidence 1.0, all scores
1.0, and `should_update=True` for **every** rollout. Training against it is
training against a constant.

`project_gui_owl_teacher.py` adds `LocalGuiOwlTeacher`: it runs a local
checkpoint over the same observation the student saw and returns its own action
as the correction target. Same `teacher_id` / `evaluate` signature as
`StrictJsonTeacher`, so it is a drop-in, and unlike the HTTP teacher it sees the
screenshot rather than a label list. `scripts/train_on_policy.py` gains
`--teacher-plugin`, which outranks the configured profile.

Two design points worth keeping:

- **A parse failure is not a WAIT.** `propose` degrades an unparseable
  generation to `ProposalKind.WAIT`. Grading against that endorses the student
  whenever the student also waited — the first version of this teacher scored a
  wrong WAIT as a perfect 1.00 match for exactly that reason. The fallback is
  now a module constant, `UNPARSEABLE_FALLBACK_MESSAGE`, and the teacher returns
  confidence 0.0 so the sample is dropped rather than learned from.
- **The environment outranks the teacher.** When `outcome.verified` is true the
  transition happened; the reference model's opinion about it is discarded.

The teacher's reliability is the 2B's measured per-class reliability: click
0.860, swipe 0.867, type 0.875, wait 1.000, long_press 0.200, `system_button`
0.000, `answer` 0.000. `--language-confidence` therefore defaults to **0.0**:
a judgement from a model that never emits `mobile.answer` carries no
information, so those samples are dropped by `min_teacher_confidence` instead of
being learned from a guess.

### 4. The student class was Qwen3-VL-only

`GUIOwlStudent` hardcoded `Qwen3VLForConditionalGeneration`, and its LoRA target
list reaches none of Qwen3.5's 18 linear-attention layers. Both are fixed: the
class resolves through `AutoModelForImageTextToText`, and the target list gains
`in_proj_qkv` / `in_proj_z` / `out_proj` when the checkpoint is `qwen3_5` — the
same list verified against the real checkpoint in Stage 1.

## What is verified

On `emulator-5554` (AVD `pixel_2`, API 29, profile `aosp10-low-phone`):

- `reset` → Settings homepage, screenshot captured, 30+ accessibility labels,
  focused activity parsed
- `step(TAP "Network & internet")` → input injected, transition verified,
  reward 1.0, episode terminal
- `step(EXPLAIN)` → reward 0.0, unverified, terminal, delegated
- `LocalGuiOwlTeacher.evaluate` → merged 2B loads, proposes
  `mobile.click [340,670]`, grades a wrong WAIT as ground 0.0 / plan 0.0 /
  `should_update=True`, and returns a TAP correction at bbox centre
  (0.340, 0.670)

## Running it

```bash
./android_drlqn_avd_matrix_apple_silicon.sh boot aosp10-low-phone
export ANDROID_DRLQN_ADB_SERIAL_AOSP10_LOW_PHONE=emulator-5554
export ANDROID_DRLQN_ALLOW_EMULATOR_INPUT=1

python scripts/train_on_policy.py \
  --documents <normalized ORM JSON array> \
  --environment-plugin project_lab:create_environment \
  --student-plugin project_gui_owl_mrpo_train:create_student \
  --teacher-plugin project_gui_owl_teacher:create_teacher \
  --mrpo-group-size 2 --mrpo-oversample-factor 2 \
  --checkpoint-reward-weight 0.5 \
  --output artifacts/distillation/stage3-$(date -u +%Y%m%dT%H%M%SZ)
```

Point `config/distillation.json`'s `student.model_id` at the Stage 1 winner's
merged checkpoint and set `GUI_OWL_TEACHER_MODEL` (or the plugin's `model_id`)
to `artifacts/teachers/gui-owl-2b-merged`. Run `--dry-run` first; it writes
`plan.json` with the resolved plugins and task counts and touches no GPU.

**Both models are resident at once** — a 0.85B student in bf16 plus a 2.13B
teacher is ~4.4 GB of weights before activations, so one 24 GB card holds them.
If it does not, relabel offline: the teacher's judgement depends only on the
observation, so a rollout can be recorded and graded in a second pass.

## Before this is a paid run

- [ ] **More than one profile.** `aosp10-low-phone` is the only AVD that exists;
      the other eight report MISSING. API 29 at 1080x1794 is one point of a
      distribution the corpus draws from 19 apps and several form factors. Run
      `prepare-stock` first.
- [ ] **The task source.** `--documents` wants a normalized ORM array, and the
      curriculum is built from documentation sections, not from the Uni-GUI
      corpus. Stage 3 therefore trains on a different task distribution than
      Stages 1-2 measured. Decide deliberately whether that is intended.
- [ ] **A tap-resolution check.** `_resolve_tap` prefers the proposal's bbox and
      falls back to substring-matching the target label against node labels. The
      one teacher sample measured here landed at y=0.670 against a row spanning
      0.679-0.786 — just outside. One sample is not a finding, but a
      resolution-accuracy pass over a few dozen screens is cheap and this reward
      function is all-or-nothing.
- [ ] **`answer` stays out of the gate**, with the reason above recorded.

The pattern in this project's history is a stage that passed a gate measuring
the wrong thing. Three of the four links in this loop were of exactly that
shape, and none of them would have raised an error.
