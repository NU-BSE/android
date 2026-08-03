# GUI-Owl-1.5-2B student and MRPO settings training

This project now uses `mPLUG/GUI-Owl-1.5-2B-Instruct` as the default real
student for Android settings rollouts. The legacy `project_train` Qwen2.5 text
plugin remains available for reproducibility, but it is no longer selected by
`scripts/train_on_policy_ollama.py` unless passed explicitly.

The design follows *Mobile-Agent-v3.5: Multi-platform Fundamental GUI Agents*
(arXiv:2602.16855) while preserving this repository's evidence, confirmation,
and verified-receipt boundaries.

## Interaction contract

`project_gui_owl_train.py` implements the native GUI-Owl model/format and
`project_gui_owl_mrpo_train.py` is the default grouped-rollout factory:

1. The system message defines a bounded mobile action/tool space.
2. The current user turn contains the task, device/OS scope, current screenshot,
   accessibility labels, normalized settings evidence, and documented UI path.
3. The latest four multimodal turns are retained in full.
4. Older turns are compressed to their visible `Action:` conclusions.
5. The model emits one concise conclusion and one tool call:

```text
Action: Open Network & internet to continue toward the requested setting.
<tool_call>{"name":"mobile.click","arguments":{"coordinate":[512,318],"target_label":"Network & internet"}}</tool_call>
```

The 2B Instruct model is not asked to expose chain-of-thought. Its conclusion is
short, user-visible, and suitable for audit logs. Coordinates use GUI-Owl's
integer `0..1000` convention and are converted to normalized boxes only at the
repository action boundary.

The student action subset maps to the existing safe coordinator actions:

| GUI-Owl tool | Repository proposal |
| --- | --- |
| `mobile.click` | `tap` |
| `mobile.swipe` | `scroll` |
| `mobile.system_button(Back)` | `back` |
| `mobile.wait` | `wait` |
| `mobile.interact` | `ask_confirmation` |
| `mobile.answer` / `terminate` | `explain` |

No tool call can directly invoke ADB, shell, root, accessibility execution,
DevicePolicyManager, or a Settings Provider write.

## Checkpoint-prefix evaluation

A settings task with documented route

```text
Settings > Network & internet > Internet
```

is treated as a path `p = (v_1, ..., v_K)`. For each checkpoint, the evaluator
defines a visible-label predicate `phi_k(o_t)`. Progress is:

```text
c_k(tau) = max_t phi_k(o_t)
m(tau) = max { m : c_k(tau) = 1 for every k <= m }
```

The shaped progress reward is the newly completed prefix fraction:

```text
r_progress,t = lambda_checkpoint * max(0, m_t - m_{t-1}) / max(1, K)
```

Terminal environment verification remains authoritative. A failed trajectory is
truncated to the action that produced its last verified checkpoint. If no route
checkpoint was verified, only the final teacher correction is kept for SFT and
its policy advantage is zero.

## MRPO grouped rollouts

For each task and emulator profile, the distiller freezes the current policy and
collects a pool of `k*n` trajectories. The default is `n=2`, `k=2`, configurable
through:

```bash
--mrpo-group-size 2 \
--mrpo-oversample-factor 2 \
--checkpoint-reward-weight 0.5
```

Let `Z(tau)` be binary terminal success. A group is collapsed when all selected
rollouts have the same outcome. The probability that the oversampled pool has
both outcomes is:

```text
P(A) = 1 - p^(kn) - (1-p)^(kn)
```

The trainer uniformly samples `n` pool members. If the subgroup is collapsed but
the pool is diverse, it swaps one member with an opposite-outcome member from
the same current-policy pool. If the entire pool is collapsed, samples still
receive supervised, grounding, language, and KL losses, but their policy
advantages are zero.

For an informative selected group, binary terminal outcomes receive centered,
variance-normalized group-relative advantages. A small discounted within-
trajectory term is added for step credit assignment, then clipped by the
existing `advantage_clip` setting.

Every selected group is kept inside one optimizer update. This is important:
splitting a group across optimizer steps would make its later members stale.

## Token-ID and screenshot alignment

The inference continuation token IDs and the exact multimodal prompt messages
are retained in an in-memory rollout sidecar keyed by the exact proposal object.
The advantage-weighted policy loss reuses those IDs rather than decoding and
re-tokenizing the sampled action. Each mutable lab screenshot is copied to an
immutable content-addressed rollout path, preventing later MRPO rollouts from
overwriting an earlier prompt image. Consumed trace sidecars are deleted after
the grouped optimizer update.

## Run

```bash
python -m pip install -e '.[train,test]'

python scripts/train_on_policy_ollama.py \
  --documents artifacts/normalized-documents.json \
  --environment-plugin project_lab:create_environment
```

The student plugin defaults to
`project_gui_owl_mrpo_train:create_student`. To run the legacy text-only
baseline, pass `--student-plugin project_train:create_student`.
