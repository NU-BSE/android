#!/usr/bin/env python3
"""Run the production 1.5B GUI-Owl compression/recovery pipeline.

The failed 24-layer experiments established that ordinary teacher-forced
recovery can lower CE/KL while free generation stops after the visible Action
line. This runner therefore treats output-contract recovery as a first-class
stage after every structural depth cut:

    build -> contract bootstrap -> grounded mixed recovery

for both 28->24 and 24->20. Only then does it run on-policy alignment and
specialization. Every transition is gated, periodic best checkpoints can be
rescued if a later update regresses, and promotion happens only after an
independent 64-record semantic evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from android_drlqn.gui_owl_student import resolve_peft_adapter_dir  # noqa: E402

TEACHER = "mPLUG/GUI-Owl-1.5-2B-Instruct"
DEFAULT_CONFIG = Path("config/gui_owl_students.json")
DEFAULT_CALIBRATION = Path("artifacts/calibration/gui-owl-compression.json")
CANONICAL = Path("artifacts/students/v2/gui-owl-1.5b-distilled")
TRAINER = "scripts/distill_gui_owl_production.py"
BUILDER = "scripts/build_gui_owl_student_grounded.py"
EVALUATOR = "scripts/evaluate_gui_owl_policy_production.py"

SEMANTIC_PARSE_MIN = 0.70
SEMANTIC_ACTION_MIN = 0.70
SEMANTIC_COORD_MAX = 100.0
CONTRACT_OPEN_MIN = 0.90
CONTRACT_CLOSE_MIN = 0.80
CONTRACT_PARSE_MIN = 0.65
EOS_RATE_MAX = 0.05
EMPTY_RATE_MAX = 0.05
MEDIAN_TOKENS_MIN = 8.0


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object expected: {path}")
    return dict(value)


def _stream(command: Sequence[str], *, env: Mapping[str, str] | None, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    if env:
        merged.update({str(key): str(value) for key, value in env.items()})
    print("+ " + " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n+ " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            list(command),
            cwd=str(ROOT),
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, list(command))


def _require_inputs(config: Path, calibration: Path) -> None:
    required = (
        config,
        calibration,
        Path("artifacts/training/gui-owl-recovery.jsonl"),
        Path("artifacts/training/gui-policy-unigui-openmobile.jsonl"),
        Path("artifacts/training/chat-screen-qa-qwen.jsonl"),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing production input(s):\n- " + "\n- ".join(missing))

    plan = _read_json(config)
    adapters = plan.get("teacher_adapters")
    if not isinstance(adapters, Mapping):
        raise ValueError("GUI-Owl config lacks teacher_adapters")
    for mode in ("gui_policy", "chat"):
        resolve_peft_adapter_dir(Path(str(adapters.get(mode) or "")), mode)


def _production_config(base_path: Path, output: Path, *, bootstrap: bool) -> dict[str, object]:
    value = _read_json(base_path)
    stages = value.get("stages")
    students = value.get("students")
    if not isinstance(stages, dict) or not isinstance(students, dict):
        raise ValueError("GUI-Owl config lacks stages/students")
    if not isinstance(students.get("1.5b"), dict):
        raise ValueError("GUI-Owl config lacks 1.5b student")
    students["1.5b"]["recovery_include_specialized"] = True
    students["1.5b"]["distilled"] = str(output)

    if bootstrap:
        recovery = stages.get("recovery")
        if not isinstance(recovery, dict):
            raise ValueError("GUI-Owl config lacks recovery stage")
        recovery.update(
            {
                "supervised_weight": 1.0,
                "forward_kl_weight": 0.25,
                "hidden_state_weight": 0.0,
                "k3_weight": 0.0,
                "learning_rate": 0.000001,
                "gradient_accumulation": 8,
                "freeze_embeddings": True,
                "freeze_lm_head": False,
                "freeze_vision": True,
                "holdout_fraction": 0.05,
                "eval_interval_updates": 100,
                "eval_records": 64,
                "eval_max_new_tokens": 64,
            }
        )
        recovery["generation_gate"] = {
            "first_token_eos_rate_max": EOS_RATE_MAX,
            "empty_generation_rate_max": EMPTY_RATE_MAX,
            "median_generated_tokens_min": MEDIAN_TOKENS_MIN,
            "gui_parse_rate_min": SEMANTIC_PARSE_MIN,
            "gui_action_name_match_rate_min": SEMANTIC_ACTION_MIN,
            "gui_median_coordinate_error_max": SEMANTIC_COORD_MAX,
        }
    else:
        recovery = stages.get("recovery")
        on_policy = stages.get("on_policy")
        specialize = stages.get("specialize")
        if not all(isinstance(stage, dict) for stage in (recovery, on_policy, specialize)):
            raise ValueError("GUI-Owl config lacks required production stages")
        recovery.update(
            {
                "supervised_weight": 0.65,
                "forward_kl_weight": 0.8,
                "hidden_state_weight": 0.10,
                "k3_weight": 0.0,
                "learning_rate": 0.0000015,
                "gradient_accumulation": 8,
                "freeze_embeddings": True,
                "freeze_lm_head": False,
                "freeze_vision": True,
                "eval_interval_updates": 100,
                "eval_records": 64,
                "eval_max_new_tokens": 64,
            }
        )
        on_policy.update(
            {
                "supervised_weight": 0.50,
                "forward_kl_weight": 0.40,
                "hidden_state_weight": 0.0,
                "k3_weight": 0.35,
                "learning_rate": 0.000001,
                "gradient_accumulation": 8,
                "freeze_embeddings": True,
                "freeze_lm_head": True,
                "freeze_vision": True,
                "eval_interval_updates": 100,
                "eval_records": 64,
                "eval_max_new_tokens": 64,
            }
        )
        specialize.update(
            {
                "supervised_weight": 0.65,
                "forward_kl_weight": 0.50,
                "hidden_state_weight": 0.05,
                "k3_weight": 0.10,
                "learning_rate": 0.0000007,
                "gradient_accumulation": 8,
                "freeze_embeddings": True,
                "freeze_lm_head": True,
                "freeze_vision": True,
                "eval_interval_updates": 100,
                "eval_records": 64,
                "eval_max_new_tokens": 64,
            }
        )
        gate = {
            "first_token_eos_rate_max": EOS_RATE_MAX,
            "empty_generation_rate_max": EMPTY_RATE_MAX,
            "median_generated_tokens_min": MEDIAN_TOKENS_MIN,
            "gui_parse_rate_min": SEMANTIC_PARSE_MIN,
            "gui_action_name_match_rate_min": SEMANTIC_ACTION_MIN,
            "gui_median_coordinate_error_max": SEMANTIC_COORD_MAX,
        }
        recovery["generation_gate"] = dict(gate)
        on_policy["generation_gate"] = dict(gate)
        specialize["generation_gate"] = dict(gate)

    value["production_revision"] = {
        "name": "contract-first-grounding-v2",
        "canonical_gui_prompt": True,
        "contract_bootstrap_after_each_depth_cut": True,
        "action_balancing": True,
        "lm_head_unfrozen_during_bootstrap_and_recovery": True,
        "semantic_gate": {
            "parse_min": SEMANTIC_PARSE_MIN,
            "action_min": SEMANTIC_ACTION_MIN,
            "coordinate_median_max": SEMANTIC_COORD_MAX,
        },
    }
    return value


def _write_configs(base: Path, work: Path, candidate: Path) -> tuple[Path, Path]:
    production = work / "production-config.json"
    bootstrap = work / "bootstrap-config.json"
    production.write_text(
        json.dumps(_production_config(base, candidate, bootstrap=False), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bootstrap.write_text(
        json.dumps(_production_config(base, candidate, bootstrap=True), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return production, bootstrap


def _checkpoint_metrics(checkpoint: Path) -> dict[str, object] | None:
    manifest = checkpoint / "distillation-manifest.json"
    if manifest.is_file():
        value = _read_json(manifest)
        metrics = value.get("final_evaluation")
        if isinstance(metrics, Mapping):
            return dict(metrics)
    periodic = checkpoint / "checkpoint-metrics.json"
    if periodic.is_file():
        value = _read_json(periodic)
        metrics = value.get("metrics")
        if isinstance(metrics, Mapping):
            return dict(metrics)
    return None


def _semantic_metrics_pass(metrics: Mapping[str, object]) -> bool:
    try:
        return (
            float(metrics.get("first_token_eos_rate", 1.0)) <= EOS_RATE_MAX
            and float(metrics.get("empty_generation_rate", 1.0)) <= EMPTY_RATE_MAX
            and float(metrics.get("median_generated_tokens", 0.0)) >= MEDIAN_TOKENS_MIN
            and float(metrics.get("gui_parse_rate", 0.0)) >= SEMANTIC_PARSE_MIN
            and float(metrics.get("gui_action_name_match_rate", 0.0)) >= SEMANTIC_ACTION_MIN
            and float(metrics.get("gui_coordinate_records", 0.0)) > 0
            and float(metrics.get("gui_median_coordinate_error", float("inf"))) <= SEMANTIC_COORD_MAX
        )
    except (TypeError, ValueError):
        return False


def _periodic_checkpoint_pass(checkpoint: Path) -> bool:
    periodic = checkpoint / "checkpoint-metrics.json"
    if not periodic.is_file():
        return False
    value = _read_json(periodic)
    failures = value.get("gate_failures")
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes)) or len(failures) != 0:
        return False
    metrics = value.get("metrics")
    return isinstance(metrics, Mapping) and _semantic_metrics_pass(metrics)


def _semantic_checkpoint(output: Path) -> Path | None:
    manifest = output / "distillation-manifest.json"
    if manifest.is_file():
        value = _read_json(manifest)
        metrics = value.get("final_evaluation")
        if value.get("accepted") is True and isinstance(metrics, Mapping) and _semantic_metrics_pass(metrics):
            return output
        best = str(value.get("best_checkpoint") or "").strip()
        if best:
            best_path = Path(best)
            if _periodic_checkpoint_pass(best_path):
                return best_path
    return None


def _contract_pass(metrics: Mapping[str, object]) -> bool:
    try:
        return (
            float(metrics.get("gui_tool_call_open_rate", 0.0)) >= CONTRACT_OPEN_MIN
            and float(metrics.get("gui_tool_call_close_rate", 0.0)) >= CONTRACT_CLOSE_MIN
            and float(metrics.get("gui_parse_rate", 0.0)) >= CONTRACT_PARSE_MIN
        )
    except (TypeError, ValueError):
        return False


def _stage_env(
    *,
    gui_only: bool,
    unfreeze_lm_head: bool,
    prefix: float,
    suffix: float,
    focus: float,
    first: float,
    require_marker: bool,
) -> dict[str, str]:
    return {
        "GUI_OWL_ONLY_GUI": "1" if gui_only else "0",
        "GUI_OWL_UNFREEZE_LM_HEAD": "1" if unfreeze_lm_head else "0",
        "GUI_OWL_CONTRACT_PREFIX_WEIGHT": str(prefix),
        "GUI_OWL_CONTRACT_SUFFIX_WEIGHT": str(suffix),
        "GUI_OWL_CONTRACT_FOCUS_WEIGHT": str(focus),
        "GUI_OWL_FIRST_TOKEN_WEIGHT": str(first),
        "GUI_OWL_REQUIRE_TOOL_MARKER": "1" if require_marker else "0",
    }


def _build(
    *,
    source: str,
    depth: int,
    output: Path,
    calibration: Path,
    config: Path,
    log: Path,
    resume: bool,
) -> None:
    if resume and (output / "student-build-manifest.json").is_file():
        print(f"==> resume: keeping structural build {output}")
        return
    _stream(
        [
            sys.executable,
            BUILDER,
            "--student",
            "1.5b",
            "--source-model",
            source,
            "--target-text-layers",
            str(depth),
            "--calibration-manifest",
            str(calibration),
            "--output",
            str(output),
            "--config",
            str(config),
            "--overwrite",
        ],
        env=None,
        log=log,
    )


def _bootstrap_contract(
    *,
    initial: Path,
    root: Path,
    config: Path,
    device: str,
    log: Path,
    chunk_updates: int,
    max_chunks: int,
    resume: bool,
    max_records: int | None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = initial
    start_chunk = 1

    if resume:
        for chunk in range(1, max_chunks + 1):
            checkpoint = root / f"chunk-{chunk}"
            metrics = _checkpoint_metrics(checkpoint)
            if metrics is None:
                break
            print(
                f"==> resume: contract chunk {chunk}: "
                f"open={float(metrics.get('gui_tool_call_open_rate', 0)):.1%} "
                f"close={float(metrics.get('gui_tool_call_close_rate', 0)):.1%} "
                f"parse={float(metrics.get('gui_parse_rate', 0)):.1%}"
            )
            if _contract_pass(metrics):
                return checkpoint
            source = checkpoint
            start_chunk = chunk + 1

    for chunk in range(start_chunk, max_chunks + 1):
        checkpoint = root / f"chunk-{chunk}"
        command = [
            sys.executable,
            TRAINER,
            "--student",
            "1.5b",
            "--stage",
            "recovery",
            "--student-model",
            str(source),
            "--teacher-model",
            TEACHER,
            "--output",
            str(checkpoint),
            "--device",
            device,
            "--max-updates",
            str(chunk_updates),
            "--learning-rate",
            "0.000001",
            "--config",
            str(config),
            "--allow-gate-failure",
        ]
        if max_records is not None:
            command += ["--max-records-per-corpus", str(max_records)]
        _stream(
            command,
            env=_stage_env(
                gui_only=True,
                unfreeze_lm_head=True,
                prefix=0.25,
                suffix=10.0,
                focus=20.0,
                first=1.0,
                require_marker=True,
            ),
            log=log,
        )
        metrics = _checkpoint_metrics(checkpoint)
        if metrics is None:
            raise RuntimeError(f"contract checkpoint has no final metrics: {checkpoint}")
        open_rate = float(metrics.get("gui_tool_call_open_rate", 0.0))
        close_rate = float(metrics.get("gui_tool_call_close_rate", 0.0))
        parse_rate = float(metrics.get("gui_parse_rate", 0.0))
        print(
            f"==> contract chunk {chunk}: open={open_rate:.1%} "
            f"close={close_rate:.1%} parse={parse_rate:.1%}"
        )
        if _contract_pass(metrics):
            return checkpoint
        if chunk >= 2 and open_rate < 0.05:
            raise RuntimeError(
                "contract bootstrap failed closed: <tool_call> open rate is still below 5% "
                "after two focused chunks; refusing to spend the rest of the training budget"
            )
        source = checkpoint

    raise RuntimeError(
        f"contract bootstrap did not reach open>={CONTRACT_OPEN_MIN:.0%}, "
        f"close>={CONTRACT_CLOSE_MIN:.0%}, parse>={CONTRACT_PARSE_MIN:.0%}"
    )


def _distill_stage(
    *,
    stage: str,
    source: Path,
    output: Path,
    config: Path,
    device: str,
    updates: int,
    log: Path,
    resume: bool,
    env: Mapping[str, str],
    max_records: int | None,
) -> Path:
    if resume:
        selected = _semantic_checkpoint(output)
        if selected is not None:
            print(f"==> resume: keeping accepted {stage} checkpoint {selected}")
            return selected
    command = [
        sys.executable,
        TRAINER,
        "--student",
        "1.5b",
        "--stage",
        stage,
        "--student-model",
        str(source),
        "--teacher-model",
        TEACHER,
        "--output",
        str(output),
        "--device",
        device,
        "--max-updates",
        str(updates),
        "--config",
        str(config),
        "--allow-gate-failure",
    ]
    if max_records is not None:
        command += ["--max-records-per-corpus", str(max_records)]
    _stream(command, env=env, log=log)
    selected = _semantic_checkpoint(output)
    if selected is None:
        metrics = _checkpoint_metrics(output) or {}
        raise RuntimeError(
            f"{stage} produced no final or periodic checkpoint satisfying the grounding gate: "
            f"parse={metrics.get('gui_parse_rate')} action={metrics.get('gui_action_name_match_rate')} "
            f"coord={metrics.get('gui_median_coordinate_error')}"
        )
    if selected != output:
        print(f"==> {stage}: final update regressed; using periodic best checkpoint {selected}")
    return selected


def _evaluate_final(candidate: Path, work: Path, device: str, log: Path) -> dict[str, object]:
    output = work / "final-64.jsonl"
    _stream(
        [
            sys.executable,
            EVALUATOR,
            "--checkpoint",
            str(candidate),
            "--device",
            device,
            "--max-records",
            "64",
            "--max-new-tokens",
            "64",
            "--holdout-fraction",
            "0.05",
            "--seed",
            "0",
            "--output",
            str(output),
            "--min-action-name-match-rate",
            str(SEMANTIC_ACTION_MIN),
            "--max-median-coordinate-error",
            str(SEMANTIC_COORD_MAX),
        ],
        env=None,
        log=log,
    )
    summary = _read_json(Path(str(output) + ".summary.json"))
    if not _semantic_metrics_pass(summary):
        raise RuntimeError(
            "independent final generation gate failed: "
            f"eos={summary.get('first_token_eos_rate')} empty={summary.get('empty_generation_rate')} "
            f"tokens={summary.get('median_generated_tokens')} parse={summary.get('gui_parse_rate')} "
            f"action={summary.get('gui_action_name_match_rate')} coord={summary.get('gui_median_coordinate_error')}"
        )
    return summary


def _record_state(work: Path, payload: Mapping[str, object]) -> None:
    (work / "production-run-manifest.json").write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    config = Path(args.config)
    calibration = Path(args.calibration_manifest)
    _require_inputs(config, calibration)

    tag = args.run_tag or datetime.now(timezone.utc).strftime("prod-%Y%m%dT%H%M%SZ")
    work = Path(args.work_root or f"artifacts/students/v2/production-1.5b/{tag}")
    work.mkdir(parents=True, exist_ok=True)
    logs = work / "logs"
    candidate_output = work / "final"
    production_config, bootstrap_config = _write_configs(config, work, candidate_output)
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_head = None

    state: dict[str, object] = {
        "schema_version": 1,
        "revision": "contract-first-grounding-v2",
        "git_head": git_head,
        "run_tag": tag,
        "work_root": str(work),
        "teacher": TEACHER,
        "production_config": str(production_config),
        "bootstrap_config": str(bootstrap_config),
        "stages": [],
    }

    depth24_init = work / "depth-24-init"
    _build(
        source=TEACHER,
        depth=24,
        output=depth24_init,
        calibration=calibration,
        config=production_config,
        log=logs / "01-build-depth24.log",
        resume=args.resume,
    )
    depth24_contract = _bootstrap_contract(
        initial=depth24_init,
        root=work / "depth-24-contract",
        config=bootstrap_config,
        device=args.device,
        log=logs / "02-contract-depth24.log",
        chunk_updates=args.contract_chunk_updates,
        max_chunks=args.contract_max_chunks,
        resume=args.resume,
        max_records=args.max_records_per_corpus,
    )
    state["stages"].append({"stage": "depth24_contract", "checkpoint": str(depth24_contract)})
    _record_state(work, state)

    depth24_recovered = _distill_stage(
        stage="recovery",
        source=depth24_contract,
        output=work / "depth-24-recovered",
        config=production_config,
        device=args.device,
        updates=args.depth24_recovery_updates,
        log=logs / "03-recover-depth24.log",
        resume=args.resume,
        env=_stage_env(
            gui_only=False,
            unfreeze_lm_head=True,
            prefix=0.75,
            suffix=4.0,
            focus=8.0,
            first=1.5,
            require_marker=False,
        ),
        max_records=args.max_records_per_corpus,
    )
    state["stages"].append({"stage": "depth24_recovery", "checkpoint": str(depth24_recovered)})
    _record_state(work, state)

    depth20_init = work / "depth-20-init"
    _build(
        source=str(depth24_recovered),
        depth=20,
        output=depth20_init,
        calibration=calibration,
        config=production_config,
        log=logs / "04-build-depth20.log",
        resume=args.resume,
    )
    depth20_contract = _bootstrap_contract(
        initial=depth20_init,
        root=work / "depth-20-contract",
        config=bootstrap_config,
        device=args.device,
        log=logs / "05-contract-depth20.log",
        chunk_updates=args.contract_chunk_updates,
        max_chunks=args.contract_max_chunks,
        resume=args.resume,
        max_records=args.max_records_per_corpus,
    )
    state["stages"].append({"stage": "depth20_contract", "checkpoint": str(depth20_contract)})
    _record_state(work, state)

    depth20_recovered = _distill_stage(
        stage="recovery",
        source=depth20_contract,
        output=work / "depth-20-recovered",
        config=production_config,
        device=args.device,
        updates=args.depth20_recovery_updates,
        log=logs / "06-recover-depth20.log",
        resume=args.resume,
        env=_stage_env(
            gui_only=False,
            unfreeze_lm_head=True,
            prefix=0.75,
            suffix=4.0,
            focus=8.0,
            first=1.5,
            require_marker=False,
        ),
        max_records=args.max_records_per_corpus,
    )
    state["stages"].append({"stage": "depth20_recovery", "checkpoint": str(depth20_recovered)})
    _record_state(work, state)

    on_policy = _distill_stage(
        stage="on_policy",
        source=depth20_recovered,
        output=work / "on-policy",
        config=production_config,
        device=args.device,
        updates=args.on_policy_updates,
        log=logs / "07-on-policy.log",
        resume=args.resume,
        env=_stage_env(
            gui_only=False,
            unfreeze_lm_head=False,
            prefix=1.0,
            suffix=2.5,
            focus=5.0,
            first=1.5,
            require_marker=False,
        ),
        max_records=args.max_records_per_corpus,
    )
    state["stages"].append({"stage": "on_policy", "checkpoint": str(on_policy)})
    _record_state(work, state)

    final_checkpoint = _distill_stage(
        stage="specialize",
        source=on_policy,
        output=candidate_output,
        config=production_config,
        device=args.device,
        updates=args.specialize_updates,
        log=logs / "08-specialize.log",
        resume=args.resume,
        env=_stage_env(
            gui_only=False,
            unfreeze_lm_head=False,
            prefix=1.0,
            suffix=2.5,
            focus=5.0,
            first=1.5,
            require_marker=False,
        ),
        max_records=args.max_records_per_corpus,
    )
    final_summary = _evaluate_final(final_checkpoint, work, args.device, logs / "09-final-eval.log")
    state["stages"].append({"stage": "specialize", "checkpoint": str(final_checkpoint)})
    state["final_evaluation"] = final_summary

    if args.promote:
        canonical = Path(args.canonical)
        backup = canonical.with_name(canonical.name + f".pre-{tag}")
        if canonical.exists():
            if backup.exists():
                raise FileExistsError(f"promotion backup already exists: {backup}")
            shutil.move(str(canonical), str(backup))
            state["previous_canonical_backup"] = str(backup)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(final_checkpoint), str(canonical))
        state["promoted_checkpoint"] = str(canonical)
    else:
        state["promoted_checkpoint"] = None
        state["candidate_checkpoint"] = str(final_checkpoint)

    _record_state(work, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--calibration-manifest", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-tag")
    parser.add_argument("--work-root")
    parser.add_argument("--canonical", default=str(CANONICAL))
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contract-chunk-updates", type=int, default=100)
    parser.add_argument("--contract-max-chunks", type=int, default=6)
    parser.add_argument("--depth24-recovery-updates", type=int, default=1800)
    parser.add_argument("--depth20-recovery-updates", type=int, default=2800)
    parser.add_argument("--on-policy-updates", type=int, default=1000)
    parser.add_argument("--specialize-updates", type=int, default=900)
    parser.add_argument("--max-records-per-corpus", type=int)
    args = parser.parse_args(argv)
    positive = (
        args.contract_chunk_updates,
        args.contract_max_chunks,
        args.depth24_recovery_updates,
        args.depth20_recovery_updates,
        args.on_policy_updates,
        args.specialize_updates,
    )
    if min(positive) <= 0:
        parser.error("all update/chunk counts must be positive")
    if args.max_records_per_corpus is not None and args.max_records_per_corpus <= 0:
        parser.error("--max-records-per-corpus must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
