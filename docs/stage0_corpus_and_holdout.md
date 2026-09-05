# Stage 0 runbook: corpus and holdout

Stage 0 of `qwen35_08b_training_plan.md`. No training happens here. It produces
the four artifacts every later stage is measured against, and it must finish
before a single GPU-hour is spent, because two of them invalidate numbers the
pipeline currently treats as settled.

Outputs:

| artifact | what it is |
|---|---|
| `artifacts/training/gui-policy-unigui-all.jsonl` | all 19 apps, relative image paths |
| `artifacts/training/gui-policy-holdout-v2-episode.json` | the pinned settings holdout, 67 episodes |
| `artifacts/training/split-all/{train,holdout}.jsonl` + manifest | the episode-disjoint split trained against |
| `artifacts/diagnostics/stage0/*.summary.json` | 0.8B and 2B re-baselined on the honest split |

## Finding 1: the current holdout leaks, and every gate is calibrated on it

`evaluate_gui_owl_policy_checkpoint.py` splits by shuffling **records**:
`random.Random(seed).shuffle(records)`, then the first `round(N x fraction)`.
The corpus is 2,729 records drawn from 303 episodes, median 8 steps each, and
steps inside an episode share a goal, an app and a near-identical screenshot.

Measured on the shipping split (seed 0, fraction 0.2):

- 540 of 546 holdout records (**98.9%**) come from an episode whose other steps
  are in the training split.
- **192 of 192** records in the baseline slice do.

This does not invalidate the Qwen3.5-0.8B baseline — that model has seen none of
the corpus, so leakage cannot help it. It does invalidate the **teacher's**
numbers, and therefore the gates: GUI-Owl-2B was fine-tuned on this corpus's
training split, and its 0.439 within50 / 128.3px is measured on episodes it
trained on. `SEMANTIC_ACTION_MIN`, `SEMANTIC_WITHIN_50_MIN` and
`SEMANTIC_COORD_MAX` are all 85% of those leaked figures.

Fixed by splitting on `episode_id`. `scripts/split_gui_policy_corpus.py` does
that, refuses to emit a split that shares an episode or a record id, and pins an
existing holdout forward so the numbers stay comparable when the corpus widens.

## Finding 2: settings is 11% of the dataset that is already on disk

Every episode's `task.json` is already downloaded — `download_unigui_openmobile`
fetches all metadata first and only then filters screenshots. Tallied from
`artifacts/uni-gui-openmobile/*/task.json`, all 2,640 episodes, all `device: mobile`:

| app | episodes | raw steps | | app | episodes | raw steps |
|---|---|---|---|---|---|---|
| settings | 305 | 2,951 | | audio recorder | 54 | 437 |
| simple calendar pro | 251 | 2,976 | | contacts | 58 | 585 |
| markor | 223 | 2,258 | | camera | 45 | 359 |
| simple sms messenger | 210 | 2,265 | | simple draw pro | 43 | 455 |
| joplin | 207 | 1,997 | | vlc | 72 | 514 |
| pro expense | 203 | 1,998 | | clock | 81 | 772 |
| tasks | 197 | 1,964 | | chrome | 101 | 759 |
| broccoli-recipe app | 181 | 1,702 | | osmand | 107 | 1,170 |
| retro music | 155 | 1,461 | | opentracks | 20 | 273 |
| files | 127 | 1,008 | | **total** | **2,640** | **25,904** |

Settings alone yielded 2,729 records from 2,951 raw steps — 92.5% survive the
reviewed/executable/image filters. At the same yield the full set is **roughly
23,900 records, an 8.8x increase**, across 19 apps instead of one. Only the
screenshots are missing, ~380 KB per step, so **about 9.8 GB to download**.

Widening changes the distribution the gates describe, so the settings holdout
stays pinned as the comparability set and an all-apps holdout is added beside
it. Report both, always.

## The Vast.ai box

**Disk is the thing to get right.** Two production runs on this project died of
ENOSPC. Provision **150 GB** and preflight before every download.

| | |
|---|---|
| GPU | one 24 GB card (RTX 4090 / A5000 / L4). Stage 0 needs it only for the re-baseline |
| disk | 150 GB. ~10 GB dataset, ~10 GB model cache, ~25 GB image, rest headroom |
| image | a CUDA PyTorch template; the repo's `Dockerfile.train` is **not** usable as-is |
| repo root | `/workspace/android` — keep it, the recorded image paths and prior logs assume it |

The pinned stack in `Dockerfile.train` is torch 2.5.1 / transformers>=4.57.1,
which predates this backbone. Qwen3.5-0.8B is verified on **transformers 5.14.1,
torch 2.13.0, peft 0.20.0** (that is where `AutoModelForImageTextToText` resolves
`Qwen3_5ForConditionalGeneration` and where PEFT attaches to `in_proj_qkv`).
Pin those; do not trust the `>=`. The two linear-attention kernel packages are
needed before any timing measurement is meaningful — without them transformers
prints *the fast path is not available* and runs the recurrence in Python.

```bash
# on the instance, from /workspace/android
python -m pip install --no-cache-dir \
  'transformers==5.14.1' 'peft==0.20.0' 'huggingface-hub>=1.26' \
  'datasets>=3.0' 'safetensors>=0.4' 'pillow>=10.0'
python -m pip install --no-cache-dir causal-conv1d flash-linear-attention
python -c "import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__);print(torch.cuda.get_device_name(0))"
```

If `flash-linear-attention` refuses to build against the instance's torch, note
it and continue — Stage 0 does not need speed. Do not start Stage 1 without it.

## Steps

Run everything from `/workspace/android` with `PYTHONPATH=$PWD`.

### 1. Preflight

```bash
df -h /workspace
python - <<'EOF'
import shutil, sys
free = shutil.disk_usage("/workspace").free / 2**30
print(f"{free:.1f} GiB free")
sys.exit(0 if free > 40 else "refusing to start: need >40 GiB free for the download")
EOF
export HF_TOKEN=...            # the second snapshot_download call does not pass a token explicitly
export HF_HOME=/workspace/.hf  # keep the model cache on the big volume, not the 10 GB root
```

### 2. Download every episode's screenshots

Separate from corpus preparation, so preparation can then be re-run cheaply and
with **relative** image paths. The path a record carries is
`source_dir / episode / screenshot`, so a relative `--source-dir` yields a corpus
that is portable between the Vast box and the Mac — which is what the prefix
rewrite in `gui_owl_grounding_findings.md` was working around.

```bash
python - <<'EOF'
from android_drlqn.unigui_openmobile import download_unigui_openmobile
print(download_unigui_openmobile("artifacts/uni-gui-openmobile", app=None))
EOF
du -sh artifacts/uni-gui-openmobile   # expect ~11 GB
```

Expect roughly an hour. It is resumable: re-running skips what is present.

### 3. Prepare the widened corpus

```bash
python scripts/prepare_unigui_gui_policy.py \
  --source-dir artifacts/uni-gui-openmobile \
  --app "" \
  --output artifacts/training/gui-policy-unigui-all.jsonl
```

`--app ""` disables the filter (`_matches_episode` short-circuits on a falsy
app). `--source-dir` skips the download entirely. Expect ~23,900 records; the
printed manifest states the true count.

```bash
python - <<'EOF'
import json, collections
c = collections.Counter(); b = 0
for line in open("artifacts/training/gui-policy-unigui-all.jsonl", encoding="utf-8"):
    r = json.loads(line); c[r["app"]] += 1; b += bool(r.get("target_bbox"))
print(sum(c.values()), "records,", len(c), "apps,", b, "with target_bbox")
print(c.most_common())
EOF
```

### 4. Build the splits

The settings holdout is already pinned in the repo, built from the shipping
corpus so it is reproducible without any download. Carry it forward:

```bash
python scripts/split_gui_policy_corpus.py \
  --corpus artifacts/training/gui-policy-unigui-all.jsonl \
  --pin-holdout artifacts/training/gui-policy-holdout-v2-episode.json \
  --holdout-fraction 0.2 --seed 0 \
  --train-out artifacts/training/split-all/train.jsonl \
  --holdout-out artifacts/training/split-all/holdout.jsonl \
  --manifest artifacts/training/split-all/split-manifest.json
```

It verifies that every referenced screenshot resolves before writing anything,
so a bad `--source-dir` fails here rather than 400 updates into Stage 1. It
fails loudly if any episode or record id lands in both splits.

Also build the settings-only split, which is the set every number in the plan's
baseline table can be re-measured on:

```bash
python scripts/split_gui_policy_corpus.py \
  --corpus artifacts/training/gui-policy-unigui-openmobile.jsonl \
  --pin-holdout artifacts/training/gui-policy-holdout-v2-episode.json \
  --holdout-fraction 0.2 --seed 0 \
  --train-out artifacts/training/split-settings/train.jsonl \
  --holdout-out artifacts/training/split-settings/holdout.jsonl \
  --manifest artifacts/training/split-settings/split-manifest.json \
  --skip-image-check
```

That one reproduces exactly: 236 episodes / 2,181 records train, 67 / 548 holdout.

### 5. Re-baseline both models on the honest split

The only GPU work in Stage 0, and it is short. Four runs: two models x two
holdouts. `--holdout-fraction 1.0` because the file already *is* the holdout — both the
evaluator and the probe default to 0.2 and would silently take a fifth of it.
`--max-records 512` is held constant across all four runs so the confidence
intervals are comparable; the settings holdout is 548 records, the all-apps one
roughly 4,800.

```bash
for model in Qwen/Qwen3.5-0.8B artifacts/teachers/gui-owl-2b-merged; do
  for split in settings all; do
    tag="$(basename "$model")-$split"
    python scripts/evaluate_gui_owl_policy_production.py \
      --checkpoint "$model" \
      --corpus "artifacts/training/split-$split/holdout.jsonl" \
      --device cuda --holdout-fraction 1.0 --max-records 512 \
      --max-new-tokens 96 --seed 0 \
      --output "artifacts/diagnostics/stage0/contract-$tag.jsonl"
  done
done

python scripts/probe_gui_grounding_backbone.py --model Qwen/Qwen3.5-0.8B \
  --corpus artifacts/training/split-all/holdout.jsonl --device cuda \
  --holdout-fraction 1.0 --max-records 512 --max-new-tokens 64 --seed 0 \
  --output artifacts/diagnostics/stage0/probe-qwen35-08b-all.jsonl
```

The teacher's merged checkpoint must exist first; the recipe is in
`gui_owl_grounding_findings.md`.

### 6. Recalibrate the gates

`SEMANTIC_ACTION_MIN` 0.65, `SEMANTIC_WITHIN_50_MIN` 0.38 and
`SEMANTIC_COORD_MAX` 200.0 are 85% retention of a teacher measured on a leaked
split. Recompute them as 85% of the teacher's **settings-holdout** numbers from
step 5 and write the new values, with the run that produced them, into
`scripts/run_gui_owl_1_5b_production.py`'s comment block and
`config/gui_owl_students.json`. Add `gui_point_in_box_rate_min`; the findings
doc shows a 50px radius is stricter than the median 224x64 box it checks.

Expect the gates to move **down**. That is the leak coming out, not a
concession.

### 7. Sync back

```bash
# on the Mac
rsync -avP --partial -e "ssh -p $VAST_PORT" \
  "$VAST_USER@$VAST_HOST:/workspace/android/artifacts/training/split-all/" \
  artifacts/training/split-all/
rsync -avP --partial -e "ssh -p $VAST_PORT" \
  "$VAST_USER@$VAST_HOST:/workspace/android/artifacts/diagnostics/stage0/" \
  artifacts/diagnostics/stage0/
```

Do not pull the 11 GB of screenshots. Commit the split manifests and the
diagnostics summaries; leave the corpora and the images untracked, as the
existing `artifacts/training/*.jsonl` are.

## Acceptance

Stage 0 is done when all of these hold:

- [ ] `gui-policy-unigui-all.jsonl` has >20,000 records across 19 apps, and every
      image path is relative and resolves from the repo root
- [ ] both split manifests report `"episode_disjoint": true` and
      `"images_verified": true`
- [ ] the settings split reproduces 236/2,181 and 67/548
- [ ] four `contract-*.summary.json` files exist, and the teacher's settings
      numbers are recorded as the new gate basis
- [ ] the gate constants in the runner and the config match those numbers at 85%
      retention, with the diagnostics path named in the comment
- [ ] `fla` and `causal-conv1d` import, or their absence is written down

Do not begin Stage 1 with any box unticked. The failure mode this project has
already paid for twice is a stage that passed a gate measuring the wrong thing.
