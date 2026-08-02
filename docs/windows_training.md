# Windows training with an NVIDIA GPU

This lane targets Windows 10/11 x86-64, Android Emulator x86_64 images and an
NVIDIA CUDA GPU such as the GeForce RTX 4060.

## Prerequisites

- Android Studio or Android command-line tools under `%LOCALAPPDATA%\Android\Sdk`;
- Android Emulator and Windows Hypervisor Platform enabled;
- a current NVIDIA driver;
- a CUDA-enabled PyTorch wheel;
- Ollama with the Qwen-VL perception and text-only UI-TARS critic models.

Verify the host before creating AVDs:

```powershell
.\android_drlqn_avd_matrix_windows.ps1 doctor
```

## Prepare and boot the stock matrix

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\android_drlqn_avd_matrix_windows.ps1 prepare-stock
.\android_drlqn_avd_matrix_windows.ps1 boot aosp10-low-phone
. .\artifacts\emulator-serials.ps1
```

The Windows matrix keeps the same profile IDs and fixed ADB ports as the Apple
Silicon matrix, but installs `x86_64` system images. Custom GrapheneOS, CalyxOS
and /e/OS AVDs must be installed separately and are never synthesized from an
AOSP image.

To let `project_lab.py` manage matching AVDs sequentially instead:

```powershell
Invoke-Expression (& .\android_drlqn_avd_matrix_windows.ps1 print-autoboot-env | Out-String)
```

## Student CUDA defaults

On Windows, when CUDA is visible and no explicit student device or dtype is
configured, the student resolves to:

```text
device=cuda
dtype=float16
```

Explicit `student.device` / `student.dtype` configuration wins. The equivalent
environment overrides are:

```powershell
$env:ANDROID_DRLQN_STUDENT_DEVICE = "cuda"
$env:ANDROID_DRLQN_STUDENT_DTYPE = "float16"
```

The plugin fails early when CUDA is explicitly requested but the installed
PyTorch build cannot see a CUDA device.

## VRAM behavior

Both Qwen-VL perception and UI-TARS critic Ollama requests use
`keep_alive: "0"`. Ollama unloads each model after its response, reducing VRAM
contention with the LoRA student at the cost of model reload latency.

Monitor training in another terminal:

```powershell
Get-Content .\artifacts\training-progress.jsonl -Wait -Tail 20
```
