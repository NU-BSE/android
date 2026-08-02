from __future__ import annotations

from pathlib import Path

import pytest

import project_train
from android_drlqn import ollama_text_teacher


def test_windows_cuda_defaults_to_float16():
    device, dtype = project_train.resolve_student_runtime(
        {}, cuda_available=True, windows_host=True, environ={}
    )
    assert device == "cuda"
    assert dtype == "float16"


def test_non_windows_cuda_keeps_existing_automatic_dtype():
    device, dtype = project_train.resolve_student_runtime(
        {}, cuda_available=True, windows_host=False, environ={}
    )
    assert device == "cuda"
    assert dtype is None


def test_explicit_student_config_wins_over_environment():
    device, dtype = project_train.resolve_student_runtime(
        {"device": "cpu", "dtype": "float32"},
        cuda_available=True,
        windows_host=True,
        environ={
            project_train.ENV_STUDENT_DEVICE: "cuda",
            project_train.ENV_STUDENT_DTYPE: "float16",
        },
    )
    assert (device, dtype) == ("cpu", "float32")


def test_student_environment_can_select_cuda_and_dtype():
    device, dtype = project_train.resolve_student_runtime(
        {},
        cuda_available=True,
        windows_host=True,
        environ={
            project_train.ENV_STUDENT_DEVICE: "cuda:0",
            project_train.ENV_STUDENT_DTYPE: "fp16",
        },
    )
    assert (device, dtype) == ("cuda:0", "fp16")


def test_explicit_cuda_fails_early_without_cuda_pytorch():
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        project_train.resolve_student_runtime(
            {"device": "cuda"},
            cuda_available=False,
            windows_host=True,
            environ={},
        )


def test_ollama_teacher_hard_unloads_models_after_requests():
    source = Path(ollama_text_teacher.__file__).read_text(encoding="utf-8")
    assert '"keep_alive": "30m"' not in source
    assert source.count('"keep_alive": "0"') >= 2
