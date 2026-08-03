#Requires -Version 5.1
<#
.SYNOPSIS
  Prepare and run the Android DRLQN emulator matrix on Windows x86-64.

.DESCRIPTION
  Stock AOSP AVDs are created from x86_64 system images. Genuine GrapheneOS,
  CalyxOS and /e/OS AVDs are never replaced by AOSP images and must already be
  installed. Fixed emulator ports preserve the profile-specific ADB serials
  expected by project_lab.py.

.EXAMPLE
  .\android_drlqn_avd_matrix_windows.ps1 doctor
  .\android_drlqn_avd_matrix_windows.ps1 prepare-stock
  .\android_drlqn_avd_matrix_windows.ps1 boot aosp10-low-phone
  . .\artifacts\emulator-serials.ps1
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Command = "help",

    [Parameter(Position = 1)]
    [string]$Profile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-AndroidSdkRoot {
    foreach ($candidate in @(
        $env:ANDROID_SDK_ROOT,
        $env:ANDROID_HOME,
        (Join-Path $env:LOCALAPPDATA "Android\Sdk")
    )) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    # Keep help/print commands usable before SDK installation; tool-requiring
    # commands produce a precise missing-file error in Require-Tools.
    return (Join-Path $env:LOCALAPPDATA "Android\Sdk")
}

function Resolve-CommandLineTool([string]$Name) {
    $latest = Join-Path $script:AndroidSdk "cmdline-tools\latest\bin\$Name"
    if (Test-Path -LiteralPath $latest) {
        return $latest
    }
    $root = Join-Path $script:AndroidSdk "cmdline-tools"
    if (Test-Path -LiteralPath $root) {
        $candidate = Get-ChildItem -LiteralPath $root -Directory |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "bin\$Name" } |
            Where-Object { Test-Path -LiteralPath $_ } |
            Select-Object -First 1
        if ($candidate) { return $candidate }
    }
    return $latest
}

$script:AndroidSdk = Resolve-AndroidSdkRoot
$env:ANDROID_SDK_ROOT = $script:AndroidSdk
$env:ANDROID_HOME = $script:AndroidSdk
$script:Adb = if ($env:ANDROID_DRLQN_ADB) {
    $env:ANDROID_DRLQN_ADB
} else {
    Join-Path $script:AndroidSdk "platform-tools\adb.exe"
}
$script:Emulator = if ($env:ANDROID_DRLQN_EMULATOR) {
    $env:ANDROID_DRLQN_EMULATOR
} else {
    Join-Path $script:AndroidSdk "emulator\emulator.exe"
}
$script:SdkManager = if ($env:ANDROID_DRLQN_SDKMANAGER) {
    $env:ANDROID_DRLQN_SDKMANAGER
} else {
    Resolve-CommandLineTool "sdkmanager.bat"
}
$script:AvdManager = if ($env:ANDROID_DRLQN_AVDMANAGER) {
    $env:ANDROID_DRLQN_AVDMANAGER
} else {
    Resolve-CommandLineTool "avdmanager.bat"
}
$script:BootTimeout = if ($env:ANDROID_DRLQN_BOOT_TIMEOUT) {
    [int]$env:ANDROID_DRLQN_BOOT_TIMEOUT
} else {
    420
}
$script:ArtifactRoot = Join-Path (Get-Location) "artifacts"
$script:LogRoot = if ($env:ANDROID_DRLQN_EMULATOR_LOG_DIR) {
    $env:ANDROID_DRLQN_EMULATOR_LOG_DIR
} else {
    Join-Path $script:ArtifactRoot "emulator-logs"
}
$script:EnvironmentFile = if ($env:ANDROID_DRLQN_EMULATOR_ENV_FILE) {
    $env:ANDROID_DRLQN_EMULATOR_ENV_FILE
} else {
    Join-Path $script:ArtifactRoot "emulator-serials.ps1"
}

$script:Profiles = @(
    [pscustomobject]@{ Id = "aosp10-low-phone"; Os = "android-aosp"; Api = 29; Avd = "pixel_2"; Env = "ANDROID_DRLQN_ADB_SERIAL_AOSP10_LOW_PHONE"; Device = "pixel_2"; Stock = $true; Port = 5554; Ram = 2048 },
    [pscustomobject]@{ Id = "aosp12l-mid-resizable"; Os = "android-aosp"; Api = 32; Avd = "resizable"; Env = "ANDROID_DRLQN_ADB_SERIAL_AOSP12L_MID_RESIZABLE"; Device = "resizable"; Stock = $true; Port = 5556; Ram = 3072 },
    [pscustomobject]@{ Id = "aosp14-main-phone"; Os = "android-aosp"; Api = 34; Avd = "pixel_6"; Env = "ANDROID_DRLQN_ADB_SERIAL_AOSP14_MAIN_PHONE"; Device = "pixel_6"; Stock = $true; Port = 5558; Ram = 4096 },
    [pscustomobject]@{ Id = "aosp15-high-phone"; Os = "android-aosp"; Api = 35; Avd = "pixel_8"; Env = "ANDROID_DRLQN_ADB_SERIAL_AOSP15_HIGH_PHONE"; Device = "pixel_8"; Stock = $true; Port = 5560; Ram = 4096 },
    [pscustomobject]@{ Id = "aosp16-tablet"; Os = "android-aosp"; Api = 36; Avd = "pixel_tablet"; Env = "ANDROID_DRLQN_ADB_SERIAL_AOSP16_TABLET"; Device = "pixel_tablet"; Stock = $true; Port = 5562; Ram = 6144 },
    [pscustomobject]@{ Id = "aosp17-foldable"; Os = "android-aosp"; Api = 37; Avd = "pixel_fold"; Env = "ANDROID_DRLQN_ADB_SERIAL_AOSP17_FOLDABLE"; Device = "pixel_fold"; Stock = $true; Port = 5564; Ram = 6144 },
    [pscustomobject]@{ Id = "grapheneos17-sdk-phone64"; Os = "grapheneos"; Api = 37; Avd = "grapheneos17-sdk-phone64"; Env = "ANDROID_DRLQN_ADB_SERIAL_GRAPHENEOS17_SDK_PHONE64"; Device = "pixel_8"; Stock = $false; Port = 5566; Ram = 4096 },
    [pscustomobject]@{ Id = "calyxos-emulator-current"; Os = "calyxos"; Api = 36; Avd = "calyxos-emulator-current"; Env = "ANDROID_DRLQN_ADB_SERIAL_CALYXOS_EMULATOR_CURRENT"; Device = "pixel_8"; Stock = $false; Port = 5568; Ram = 4096 },
    [pscustomobject]@{ Id = "eos13-nexus5x-emulator"; Os = "eos"; Api = 33; Avd = "Nexus 5X"; Env = "ANDROID_DRLQN_ADB_SERIAL_EOS13_NEXUS5X_EMULATOR"; Device = "Nexus 5X"; Stock = $false; Port = 5570; Ram = 3072 }
)

function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
}

function Require-Tools {
    Require-File $script:Adb "adb.exe"
    Require-File $script:Emulator "emulator.exe"
    Require-File $script:SdkManager "sdkmanager.bat"
    Require-File $script:AvdManager "avdmanager.bat"
}

function Get-Profile([string]$Id) {
    $item = $script:Profiles | Where-Object Id -EQ $Id | Select-Object -First 1
    if (-not $item) {
        throw "Unknown profile: $Id"
    }
    return $item
}

function Get-InstalledAvds {
    Require-File $script:Emulator "emulator.exe"
    @(& $script:Emulator -list-avds 2>$null) | Where-Object { $_ -and $_.Trim() }
}

function Test-AvdInstalled([string]$Name) {
    return [bool](Get-InstalledAvds | Where-Object { $_.Trim() -eq $Name })
}

function Invoke-Checked([string]$FilePath, [string[]]$Arguments) {
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

function Get-SystemImagePackage([int]$Api) {
    $overrideName = "ANDROID_DRLQN_IMAGE_PACKAGE_API_$Api"
    $override = [Environment]::GetEnvironmentVariable($overrideName)
    if ($override) {
        return $override
    }

    $previousErrorActionPreference = $ErrorActionPreference

    try {
        $ErrorActionPreference = "Continue"

        # Redirect stderr inside cmd.exe so PowerShell does not convert
        # sdkmanager warnings into terminating NativeCommandError records.
        $listing = (
            & $env:ComSpec /d /s /c `
                "`"$script:SdkManager`" --list 2>&1" |
            Out-String
        )

        $sdkManagerExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($sdkManagerExitCode -ne 0) {
        throw "sdkmanager --list failed with exit code $sdkManagerExitCode`n$listing"
    }
    $apiNames = if ($Api -eq 37) { @("37.0", "37") } else { @([string]$Api) }
    $tags = @("google_apis_ps16k", "google_apis", "default")
    foreach ($apiName in $apiNames) {
        foreach ($tag in $tags) {
            $candidate = "system-images;android-$apiName;$tag;x86_64"
            if ($listing.Contains($candidate)) {
                return $candidate
            }
        }
    }
    throw "No x86_64 system image found for Android API $Api"
}

function Set-IniValue([string]$Path, [string]$Name, [string]$Value) {
    $lines = if (Test-Path -LiteralPath $Path) {
        [System.Collections.Generic.List[string]](Get-Content -LiteralPath $Path)
    } else {
        [System.Collections.Generic.List[string]]::new()
    }
    $prefix = "$Name="
    $found = $false
    for ($index = 0; $index -lt $lines.Count; $index++) {
        if ($lines[$index].StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            $lines[$index] = "$Name=$Value"
            $found = $true
            break
        }
    }
    if (-not $found) {
        $lines.Add("$Name=$Value")
    }
    Set-Content -LiteralPath $Path -Value $lines -Encoding ASCII
}

function Configure-Avd($Item) {
    $config = Join-Path $env:USERPROFILE ".android\avd\$($Item.Avd).avd\config.ini"
    if (-not (Test-Path -LiteralPath $config)) {
        Write-Warning "AVD config not found: $config"
        return
    }
    Set-IniValue $config "hw.ramSize" ([string]$Item.Ram)
    Set-IniValue $config "hw.gpu.enabled" "yes"
    Set-IniValue $config "hw.gpu.mode" "host"
    Set-IniValue $config "fastboot.forceColdBoot" "yes"
    Set-IniValue $config "fastboot.forceFastBoot" "no"
    Set-IniValue $config "disk.dataPartition.size" "8G"
}

function Prepare-StockProfile($Item) {
    if (-not $Item.Stock) {
        throw "$($Item.Id) is a custom-OS profile and cannot be created from AOSP"
    }
    if (Test-AvdInstalled $Item.Avd) {
        Write-Host "[READY] $($Item.Id): AVD $($Item.Avd) already exists"
        Configure-Avd $Item
        return
    }

    $package = Get-SystemImagePackage $Item.Api
    Write-Host "[SDK] Installing $package"
    Invoke-Checked $script:SdkManager @("platform-tools", "emulator", $package)

    $devices = @($Item.Device, "pixel_8", "pixel_6", "pixel", "Nexus 5X") |
        Select-Object -Unique
    $created = $false
    foreach ($device in $devices) {
        Write-Host "[AVD] Creating $($Item.Avd) with hardware profile $device"
        "no" | & $script:AvdManager create avd --force --name $Item.Avd --package $package --device $device
        if ($LASTEXITCODE -eq 0) {
            $created = $true
            break
        }
    }
    if (-not $created) {
        throw "Unable to create AVD $($Item.Avd)"
    }
    Configure-Avd $Item
}

function Get-Serial($Item) {
    "emulator-$($Item.Port)"
}

function Get-AdbState([string]$Serial) {
    $lines = @(& $script:Adb devices 2>$null)
    foreach ($line in $lines) {
        if ($line -match "^$([regex]::Escape($Serial))\s+(\S+)") {
            return $Matches[1]
        }
    }
    return "missing"
}

function Wait-ForBoot($Item) {
    $serial = Get-Serial $Item
    $deadline = [DateTime]::UtcNow.AddSeconds($script:BootTimeout)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ((Get-AdbState $serial) -eq "device") {
            $completed = (& $script:Adb -s $serial shell getprop sys.boot_completed 2>$null | Out-String).Trim()
            if ($completed -eq "1") {
                $api = (& $script:Adb -s $serial shell getprop ro.build.version.sdk 2>$null | Out-String).Trim()
                if ($api -ne [string]$Item.Api) {
                    throw "$serial booted API $api, expected API $($Item.Api)"
                }
                Write-Host "[BOOTED] $($Item.Id) serial=$serial API=$api"
                return
            }
        }
        Start-Sleep -Seconds 2
    }
    throw "Timed out waiting for $($Item.Id) on $serial"
}

function Write-SerialEnvironment {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $script:EnvironmentFile) | Out-Null
    $lines = @(
        '# Generated by android_drlqn_avd_matrix_windows.ps1',
        "`$env:ANDROID_DRLQN_ADB = '$($script:Adb.Replace("'", "''"))'",
        "`$env:ANDROID_DRLQN_EMULATOR = '$($script:Emulator.Replace("'", "''"))'",
        '`$env:ANDROID_DRLQN_ALLOW_EMULATOR_INPUT = "1"',
        '`$env:ANDROID_DRLQN_STUDENT_DEVICE = "cuda"',
        '`$env:ANDROID_DRLQN_STUDENT_DTYPE = "float16"'
    )
    foreach ($item in $script:Profiles) {
        $serial = Get-Serial $item
        if ((Get-AdbState $serial) -eq "device") {
            $lines += "`$env:$($item.Env) = '$serial'"
        }
    }
    Set-Content -LiteralPath $script:EnvironmentFile -Value $lines -Encoding UTF8
    Write-Host "[ENV] Wrote $script:EnvironmentFile"
}

function Boot-Profile($Item) {
    Require-Tools
    if (-not (Test-AvdInstalled $Item.Avd)) {
        if ($Item.Stock) {
            Prepare-StockProfile $Item
        } else {
            throw "Custom AVD $($Item.Avd) is not installed"
        }
    }
    $serial = Get-Serial $Item
    if ((Get-AdbState $serial) -eq "device") {
        Write-Host "[READY] $($Item.Id) already runs as $serial"
        Write-SerialEnvironment
        return
    }

    New-Item -ItemType Directory -Force -Path $script:LogRoot | Out-Null
    $stdout = Join-Path $script:LogRoot "$($Item.Id).stdout.log"
    $stderr = Join-Path $script:LogRoot "$($Item.Id).stderr.log"
    $arguments = @(
        "-avd", $Item.Avd,
        "-port", [string]$Item.Port,
        "-memory", [string]$Item.Ram,
        "-gpu", "host",
        "-accel", "auto",
        "-no-snapshot-load",
        "-no-boot-anim",
        "-netfast"
    )
    Write-Host "[BOOT] $($Item.Id) AVD=$($Item.Avd) serial=$serial"
    Start-Process -FilePath $script:Emulator -ArgumentList $arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr | Out-Null
    Wait-ForBoot $Item
    Write-SerialEnvironment
}

function Stop-Profile($Item) {
    $serial = Get-Serial $Item
    if ((Get-AdbState $serial) -ne "missing") {
        & $script:Adb -s $serial emu kill | Out-Null
        Write-Host "[STOP] $($Item.Id) $serial"
    }
}

function Show-Status {
    Require-File $script:Adb "adb.exe"
    foreach ($item in $script:Profiles) {
        $serial = Get-Serial $item
        $state = Get-AdbState $serial
        $api = "-"
        $avd = "-"
        if ($state -eq "device") {
            $api = (& $script:Adb -s $serial shell getprop ro.build.version.sdk 2>$null | Out-String).Trim()
            $avdLines = @(& $script:Adb -s $serial emu avd name 2>$null)
            $avd = ($avdLines | Where-Object { $_ -and $_.Trim() -ne "OK" } | Select-Object -First 1)
        }
        "{0,-32} {1,-14} state={2,-8} api={3,-3} avd={4}" -f $item.Id, $serial, $state, $api, $avd
    }
}

function Check-CustomAvds {
    $missing = @()
    foreach ($item in $script:Profiles | Where-Object { -not $_.Stock }) {
        if (Test-AvdInstalled $item.Avd) {
            Write-Host "[READY] $($Item.Id): $($Item.Avd)"
        } else {
            Write-Warning "Missing genuine custom AVD: $($Item.Avd)"
            $missing += $item.Id
        }
    }
    if ($missing.Count -gt 0) {
        throw "Missing custom AVD profiles: $($missing -join ', ')"
    }
}

function Quote-PowerShell([string]$Value) {
    "'" + $Value.Replace("'", "''") + "'"
}

function Print-Environment([bool]$AutoBoot) {
    "`$env:ANDROID_HOME = $(Quote-PowerShell $script:AndroidSdk)"
    "`$env:ANDROID_SDK_ROOT = $(Quote-PowerShell $script:AndroidSdk)"
    "`$env:ANDROID_DRLQN_ADB = $(Quote-PowerShell $script:Adb)"
    "`$env:ANDROID_DRLQN_EMULATOR = $(Quote-PowerShell $script:Emulator)"
    '`$env:ANDROID_DRLQN_ALLOW_EMULATOR_INPUT = "1"'
    '`$env:ANDROID_DRLQN_STUDENT_DEVICE = "cuda"'
    '`$env:ANDROID_DRLQN_STUDENT_DTYPE = "float16"'
    '`$env:OLLAMA_MAX_LOADED_MODELS = "1"'
    if ($AutoBoot) {
        '`$env:ANDROID_DRLQN_LAB_AUTOBOOT = "1"'
    }
    foreach ($item in $script:Profiles) {
        $serial = Get-Serial $item
        if ((Get-AdbState $serial) -eq "device") {
            "`$env:$($item.Env) = '$serial'"
        }
    }
}

function Doctor {
    Require-Tools
    Write-Host "Windows:          $([Environment]::OSVersion.VersionString)"
    Write-Host "Android SDK:      $script:AndroidSdk"
    Write-Host "adb:              $script:Adb"
    Write-Host "emulator:         $script:Emulator"
    Write-Host "sdkmanager:       $script:SdkManager"
    Write-Host "avdmanager:       $script:AvdManager"
    Write-Host ""
    Write-Host "Installed AVDs:"
    Get-InstalledAvds | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
    if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
        & nvidia-smi.exe --query-gpu=name,driver_version,memory.total --format=csv,noheader
    } else {
        Write-Warning "nvidia-smi.exe is not on PATH"
    }
    $python = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($python) {
        & $python.Source -3 -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('Torch CUDA:', torch.version.cuda)"
    } else {
        Write-Warning "py.exe is not on PATH"
    }
}

function Usage {
@'
Usage:
  .\android_drlqn_avd_matrix_windows.ps1 doctor
  .\android_drlqn_avd_matrix_windows.ps1 list
  .\android_drlqn_avd_matrix_windows.ps1 prepare-stock
  .\android_drlqn_avd_matrix_windows.ps1 prepare-profile <profile>
  .\android_drlqn_avd_matrix_windows.ps1 check-custom
  .\android_drlqn_avd_matrix_windows.ps1 boot <profile>
  .\android_drlqn_avd_matrix_windows.ps1 boot-stock
  .\android_drlqn_avd_matrix_windows.ps1 boot-all
  .\android_drlqn_avd_matrix_windows.ps1 status
  .\android_drlqn_avd_matrix_windows.ps1 print-env
  .\android_drlqn_avd_matrix_windows.ps1 print-autoboot-env
  .\android_drlqn_avd_matrix_windows.ps1 stop <profile|all>

Recommended Windows workflow:
  1. .\android_drlqn_avd_matrix_windows.ps1 doctor
  2. .\android_drlqn_avd_matrix_windows.ps1 prepare-stock
  3. .\android_drlqn_avd_matrix_windows.ps1 boot aosp10-low-phone
  4. . .\artifacts\emulator-serials.ps1
  5. Run scripts\train_on_policy_ollama.py
'@
}

switch ($Command.ToLowerInvariant()) {
    "doctor" { Doctor; break }
    "list" { $script:Profiles | Format-Table Id, Os, Api, Avd, Port, Stock -AutoSize; break }
    "prepare-stock" {
        Require-Tools
        $failures = @()
        foreach ($item in $script:Profiles | Where-Object Stock) {
            try { Prepare-StockProfile $item } catch { Write-Warning $_; $failures += $item.Id }
        }
        if ($failures.Count) { throw "Failed profiles: $($failures -join ', ')" }
        break
    }
    "prepare-profile" {
        if (-not $Profile) { throw "prepare-profile requires a profile" }
        Prepare-StockProfile (Get-Profile $Profile)
        break
    }
    "check-custom" { Require-Tools; Check-CustomAvds; break }
    "boot" {
        if (-not $Profile) { throw "boot requires a profile" }
        Boot-Profile (Get-Profile $Profile)
        break
    }
    "boot-stock" {
        foreach ($item in $script:Profiles | Where-Object Stock) { Boot-Profile $item }
        break
    }
    "boot-all" {
        if ($env:ANDROID_DRLQN_BOOT_ALL -ne "1") {
            throw "Set ANDROID_DRLQN_BOOT_ALL=1 before boot-all"
        }
        foreach ($item in $script:Profiles) { Boot-Profile $item }
        break
    }
    "status" { Show-Status; break }
    "print-env" { Print-Environment $false; break }
    "print-autoboot-env" { Print-Environment $true; break }
    "stop" {
        if (-not $Profile) { throw "stop requires a profile or all" }
        if ($Profile -eq "all") {
            foreach ($item in $script:Profiles) { Stop-Profile $item }
        } else {
            Stop-Profile (Get-Profile $Profile)
        }
        Write-SerialEnvironment
        break
    }
    { $_ -in @("help", "-h", "--help", "") } { Usage; break }
    default { Usage; throw "Unknown command: $Command" }
}
