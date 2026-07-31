#!/usr/bin/env bash
#
# android_drlqn_avd_matrix_apple_silicon.sh
#
# Prepare, boot and map the Android DRLQN emulator matrix on Apple Silicon.
#
# Commands:
#   ./android_drlqn_avd_matrix_apple_silicon.sh doctor
#   ./android_drlqn_avd_matrix_apple_silicon.sh prepare-stock
#   ./android_drlqn_avd_matrix_apple_silicon.sh check-custom
#   ./android_drlqn_avd_matrix_apple_silicon.sh boot <profile>
#   ./android_drlqn_avd_matrix_apple_silicon.sh boot-stock
#   ANDROID_DRLQN_BOOT_ALL=1 ./android_drlqn_avd_matrix_apple_silicon.sh boot-all
#   ./android_drlqn_avd_matrix_apple_silicon.sh status
#   ./android_drlqn_avd_matrix_apple_silicon.sh print-env
#   ./android_drlqn_avd_matrix_apple_silicon.sh print-autoboot-env
#   ./android_drlqn_avd_matrix_apple_silicon.sh stop <profile|all>
#
# The script automatically installs/creates stock AOSP ARM64 AVDs.
# Genuine GrapheneOS, CalyxOS and /e/OS AVDs must already be installed.
#
set -Eeuo pipefail

ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$ANDROID_HOME}"
ANDROID_AVD_HOME="${ANDROID_AVD_HOME:-$HOME/.android/avd}"

SDKMANAGER="${ANDROID_DRLQN_SDKMANAGER:-$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager}"
AVDMANAGER="${ANDROID_DRLQN_AVDMANAGER:-$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager}"
EMULATOR="${ANDROID_DRLQN_EMULATOR:-$ANDROID_HOME/emulator/emulator}"
ADB="${ANDROID_DRLQN_ADB:-$ANDROID_HOME/platform-tools/adb}"

SDK_CHANNEL="${ANDROID_DRLQN_SDK_CHANNEL:-3}"
BOOT_TIMEOUT="${ANDROID_DRLQN_BOOT_TIMEOUT:-420}"
LOG_DIR="${ANDROID_DRLQN_EMULATOR_LOG_DIR:-$PWD/artifacts/emulator-logs}"
ENV_FILE="${ANDROID_DRLQN_EMULATOR_ENV_FILE:-$PWD/artifacts/emulator-serials.env}"

export ANDROID_HOME ANDROID_SDK_ROOT ANDROID_AVD_HOME
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"

# profile<TAB>os<TAB>api<TAB>expected_avd<TAB>serial_env<TAB>hardware_id<TAB>kind<TAB>port<TAB>ram_mb
matrix() {
  cat <<'EOF'
aosp10-low-phone	android-aosp	29	pixel_2	ANDROID_DRLQN_ADB_SERIAL_AOSP10_LOW_PHONE	pixel_2	stock	5554	2048
aosp12l-mid-resizable	android-aosp	32	resizable	ANDROID_DRLQN_ADB_SERIAL_AOSP12L_MID_RESIZABLE	resizable	stock	5556	3072
aosp14-main-phone	android-aosp	34	pixel_6	ANDROID_DRLQN_ADB_SERIAL_AOSP14_MAIN_PHONE	pixel_6	stock	5558	4096
aosp15-high-phone	android-aosp	35	pixel_8	ANDROID_DRLQN_ADB_SERIAL_AOSP15_HIGH_PHONE	pixel_8	stock	5560	4096
aosp16-tablet	android-aosp	36	pixel_tablet	ANDROID_DRLQN_ADB_SERIAL_AOSP16_TABLET	pixel_tablet	stock	5562	6144
aosp17-foldable	android-aosp	37	pixel_fold	ANDROID_DRLQN_ADB_SERIAL_AOSP17_FOLDABLE	pixel_fold	stock	5564	6144
grapheneos17-sdk-phone64	grapheneos	37	grapheneos17-sdk-phone64	ANDROID_DRLQN_ADB_SERIAL_GRAPHENEOS17_SDK_PHONE64	pixel_8	custom	5566	4096
calyxos-emulator-current	calyxos	36	calyxos-emulator-current	ANDROID_DRLQN_ADB_SERIAL_CALYXOS_EMULATOR_CURRENT	pixel_8	custom	5568	4096
eos13-nexus5x-emulator	eos	33	Nexus 5X	ANDROID_DRLQN_ADB_SERIAL_EOS13_NEXUS5X_EMULATOR	Nexus 5X	custom	5570	3072
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

info() {
  printf '==> %s\n' "$*"
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -x "$path" ]] || die "$label not found or not executable: $path"
}

require_apple_silicon() {
  local machine
  machine="$(uname -m)"
  [[ "$machine" == "arm64" ]] ||
    die "this script targets Apple Silicon (arm64); detected: $machine"
}

require_tools() {
  require_apple_silicon
  require_file "$SDKMANAGER" sdkmanager
  require_file "$AVDMANAGER" avdmanager
  require_file "$EMULATOR" emulator
  require_file "$ADB" adb
}

row_for_profile() {
  local wanted="$1"
  matrix | awk -F '\t' -v wanted="$wanted" '$1 == wanted { print; found=1 } END { if (!found) exit 1 }'
}

list_profiles() {
  matrix | awk -F '\t' '{
    printf "profile=%s\n  os=%s\n  api=%s\n  expected_avd=%s\n  serial_env=%s\n", $1, $2, $3, $4, $5
  }'
}

available_avds() {
  "$EMULATOR" -list-avds 2>/dev/null || true
}

normalize_name() {
  printf '%s' "$1" |
    tr '[:upper:]' '[:lower:]' |
    sed -E 's/[^a-z0-9]+//g'
}

avd_metadata_valid() {
  local avd="$1"
  local ini="$ANDROID_AVD_HOME/${avd}.ini"
  local config="$ANDROID_AVD_HOME/${avd}.avd/config.ini"
  local image_path

  [[ -f "$ini" && -f "$config" ]] || return 1

  image_path="$(sed -nE 's/^image\.sysdir\.1=(.*)$/\1/p' "$config" | head -n 1)"
  [[ -n "$image_path" ]] || return 1

  if [[ "$image_path" == /* ]]; then
    [[ -d "$image_path" ]]
  else
    [[ -d "$ANDROID_HOME/$image_path" ]]
  fi
}

resolve_existing_avd() {
  local requested="$1"
  local requested_norm
  local candidate

  requested_norm="$(normalize_name "$requested")"

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    if [[ "$candidate" == "$requested" ]] && avd_metadata_valid "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(available_avds)

  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] || continue
    if [[ "$(normalize_name "$candidate")" == "$requested_norm" ]] && avd_metadata_valid "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(available_avds)

  return 1
}

avd_exists() {
  resolve_existing_avd "$1" >/dev/null 2>&1
}

# Current command-line tools may try to read an optional devices.xml from
# every installed system image while listing/creating AVDs. Stock images do
# not consistently ship that file, so avoid the global avdmanager scan and
# create the minimal AVD metadata directly (the same approach used by AOSP
# aadevtools for image packages).
system_image_relative_dir() {
  local package="$1"
  local root platform tag abi extra

  IFS=';' read -r root platform tag abi extra <<<"$package"
  [[ "$root" == "system-images" && -n "$platform" && -n "$tag" && -n "$abi" && -z "${extra:-}" ]] ||
    die "invalid system image package id: $package"

  printf 'system-images/%s/%s/%s\n' "$platform" "$tag" "$abi"
}

hardware_geometry() {
  local hardware="$1"

  case "$hardware" in
    pixel_2)
      printf '1080 1920 420 portrait\n'
      ;;
    resizable|pixel_6|pixel_8)
      printf '1080 2400 420 portrait\n'
      ;;
    pixel_tablet)
      printf '2560 1600 320 landscape\n'
      ;;
    pixel_fold)
      printf '2208 1840 420 portrait\n'
      ;;
    *)
      printf '1080 2400 420 portrait\n'
      ;;
  esac
}

create_avd_metadata() {
  local profile="$1"
  local api="$2"
  local avd="$3"
  local hardware="$4"
  local ram="$5"
  local package="$6"
  local relative_image_dir
  local image_dir
  local avd_dir
  local avd_ini
  local platform
  local tag
  local abi
  local width height density orientation

  relative_image_dir="$(system_image_relative_dir "$package")"
  image_dir="$ANDROID_HOME/$relative_image_dir"
  avd_dir="$ANDROID_AVD_HOME/${avd}.avd"
  avd_ini="$ANDROID_AVD_HOME/${avd}.ini"

  IFS=';' read -r _ platform tag abi <<<"$package"
  read -r width height density orientation <<<"$(hardware_geometry "$hardware")"

  [[ -d "$image_dir" ]] ||
    die "installed system image directory is missing: $image_dir"
  [[ -f "$image_dir/userdata.img" ]] ||
    die "system image is incomplete (userdata.img missing): $image_dir"
  [[ -f "$image_dir/system.img" || -f "$image_dir/system-qemu.img" ]] ||
    die "system image is incomplete (system image missing): $image_dir"

  mkdir -p "$ANDROID_AVD_HOME"

  # Clean only a stale partial AVD with the exact requested name. A valid AVD
  # has already returned from avd_exists() before this function is called.
  if [[ -e "$avd_dir" || -e "$avd_ini" ]]; then
    warn "removing partial AVD metadata for '$avd'"
    rm -rf "$avd_dir" "$avd_ini"
  fi

  mkdir -p "$avd_dir"
  cp -f "$image_dir/userdata.img" "$avd_dir/userdata.img"

  cat >"$avd_ini" <<EOF
avd.ini.encoding=UTF-8
path=$avd_dir
path.rel=avd/${avd}.avd
target=$platform
EOF

  cat >"$avd_dir/config.ini" <<EOF
AvdId=$avd
PlayStore.enabled=false
abi.type=$abi
avd.ini.displayname=$avd
avd.ini.encoding=UTF-8
disk.dataPartition.size=8G
fastboot.forceColdBoot=yes
fastboot.forceFastBoot=no
hw.arc=false
hw.audioInput=yes
hw.battery=yes
hw.camera.back=emulated
hw.camera.front=emulated
hw.cpu.arch=$abi
hw.cpu.ncore=4
hw.device.manufacturer=Google
hw.device.name=$hardware
hw.gps=yes
hw.gpu.enabled=yes
hw.gpu.mode=auto
hw.initialOrientation=$orientation
hw.keyboard=yes
hw.lcd.density=$density
hw.lcd.height=$height
hw.lcd.width=$width
hw.mainKeys=no
hw.ramSize=$ram
hw.sensors.orientation=yes
hw.sensors.proximity=yes
image.sysdir.1=${relative_image_dir}/
runtime.network.latency=none
runtime.network.speed=full
showDeviceFrame=no
skin.dynamic=yes
skin.name=${width}x${height}
skin.path=${width}x${height}
snapshot.present=no
tag.display=$tag
tag.id=$tag
vm.heapSize=512
EOF

  info "created AVD '$avd' without avdmanager device scanning"
}

sdk_list_cache=""

refresh_sdk_list() {
  mkdir -p "$LOG_DIR"
  sdk_list_cache="$LOG_DIR/sdkmanager-list.txt"
  info "refreshing SDK package list (channel $SDK_CHANNEL)"
  "$SDKMANAGER" --list --channel="$SDK_CHANNEL" >"$sdk_list_cache"
}

ensure_sdk_list() {
  [[ -n "$sdk_list_cache" && -s "$sdk_list_cache" ]] || refresh_sdk_list
}

package_available() {
  local package="$1"
  ensure_sdk_list
  grep -Fq "$package" "$sdk_list_cache"
}

select_stock_image_package() {
  local api="$1"
  local override_var="ANDROID_DRLQN_IMAGE_PACKAGE_API_${api}"
  local override="${!override_var:-}"
  local candidate

  if [[ -n "$override" ]]; then
    package_available "$override" ||
      die "$override_var points to an unavailable SDK package: $override"
    printf '%s\n' "$override"
    return 0
  fi

  # Prefer a pure AOSP image, then Google APIs. Never silently select x86_64.
  # Android 17 currently publishes its emulator image under android-37.0 and
  # google_apis_ps16k, while the compile platform remains platforms;android-37.
  if [[ "$api" == "37" ]]; then
    for candidate in \
      "system-images;android-37.0;google_apis_ps16k;arm64-v8a" \
      "system-images;android-37.0;google_apis;arm64-v8a" \
      "system-images;android-37;google_apis_ps16k;arm64-v8a" \
      "system-images;android-37;google_apis;arm64-v8a"
    do
      if package_available "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
  fi

  for candidate in \
    "system-images;android-${api};default;arm64-v8a" \
    "system-images;android-${api};google_apis;arm64-v8a" \
    "system-images;android-${api};google_apis_playstore;arm64-v8a"
  do
    if package_available "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

install_base_sdk() {
  info "accepting Android SDK licenses"
  yes | "$SDKMANAGER" --licenses >/dev/null 2>&1 || true

  info "installing Android SDK command-line dependencies"
  "$SDKMANAGER" --channel="$SDK_CHANNEL" \
    "platform-tools" \
    "emulator" \
    "cmdline-tools;latest"
}

install_stock_profile() {
  local profile="$1"
  local os_family="$2"
  local api="$3"
  local avd="$4"
  local serial_env="$5"
  local hardware="$6"
  local kind="$7"
  local port="$8"
  local ram="$9"
  local package

  [[ "$kind" == "stock" ]] || return 0

  if ! package="$(select_stock_image_package "$api")"; then
    warn "no ARM64 system image found for API $api (profile=$profile)"
    warn "for preview APIs, update Android Studio/SDK tools or set ANDROID_DRLQN_IMAGE_PACKAGE_API_${api}"
    return 1
  fi

  info "installing $profile image: $package"
  "$SDKMANAGER" --channel="$SDK_CHANNEL" \
    "platforms;android-${api}" \
    "$package"
}

set_ini_property() {
  local file="$1"
  local key="$2"
  local value="$3"

  if grep -Eq "^${key}=" "$file"; then
    sed -i.bak -E "s|^${key}=.*|${key}=${value}|" "$file"
    rm -f "${file}.bak"
  else
    printf '%s=%s\n' "$key" "$value" >>"$file"
  fi
}

create_stock_profile() {
  local profile="$1"
  local os_family="$2"
  local api="$3"
  local avd="$4"
  local serial_env="$5"
  local hardware="$6"
  local kind="$7"
  local port="$8"
  local ram="$9"
  local package
  local config

  [[ "$kind" == "stock" ]] || return 0

  if avd_exists "$avd"; then
    info "AVD already exists: $avd"
    return 0
  fi

  package="$(select_stock_image_package "$api")" ||
    die "cannot create $profile: no ARM64 system image for API $api"

  info "creating AVD '$avd' for $profile with $package and hardware '$hardware'"
  create_avd_metadata "$profile" "$api" "$avd" "$hardware" "$ram" "$package"

  config="$ANDROID_AVD_HOME/${avd}.avd/config.ini"
  [[ -f "$config" ]] || die "AVD created but config.ini was not found: $config"

  set_ini_property "$config" "hw.keyboard" "yes"
  set_ini_property "$config" "hw.gpu.enabled" "yes"
  set_ini_property "$config" "hw.gpu.mode" "auto"
  set_ini_property "$config" "hw.ramSize" "$ram"
  set_ini_property "$config" "vm.heapSize" "512"
  set_ini_property "$config" "disk.dataPartition.size" "8G"
  set_ini_property "$config" "fastboot.forceColdBoot" "yes"
  set_ini_property "$config" "fastboot.forceFastBoot" "no"
}

prepare_one_stock_profile() {
  local profile="$1"
  local os_family="$2"
  local api="$3"
  local avd="$4"
  local serial_env="$5"
  local hardware="$6"
  local kind="$7"
  local port="$8"
  local ram="$9"

  [[ "$kind" == "stock" ]] ||
    die "$profile is a custom-OS profile and cannot be created from an AOSP image"

  install_stock_profile \
    "$profile" "$os_family" "$api" "$avd" "$serial_env" \
    "$hardware" "$kind" "$port" "$ram" || return 1

  create_stock_profile \
    "$profile" "$os_family" "$api" "$avd" "$serial_env" \
    "$hardware" "$kind" "$port" "$ram"
}

prepare_profile() {
  local wanted="$1"
  local row
  local profile os_family api avd serial_env hardware kind port ram

  require_tools
  install_base_sdk
  refresh_sdk_list

  row="$(row_for_profile "$wanted")" || die "unknown profile: $wanted"
  IFS=$'\t' read -r profile os_family api avd serial_env hardware kind port ram <<<"$row"

  prepare_one_stock_profile \
    "$profile" "$os_family" "$api" "$avd" "$serial_env" \
    "$hardware" "$kind" "$port" "$ram"
}

prepare_stock() {
  local failures=0
  local prepared=0
  local failed_profiles=()

  require_tools
  install_base_sdk
  refresh_sdk_list

  while IFS=$'\t' read -r profile os_family api avd serial_env hardware kind port ram; do
    [[ "$kind" == "stock" ]] || continue

    if prepare_one_stock_profile \
      "$profile" "$os_family" "$api" "$avd" "$serial_env" \
      "$hardware" "$kind" "$port" "$ram"
    then
      prepared=$((prepared + 1))
    else
      failures=$((failures + 1))
      failed_profiles+=("$profile")
      warn "skipping unavailable stock profile: $profile (API $api)"
    fi
  done < <(matrix)

  info "$prepared stock AVD profile(s) are ready"
  check_custom || true

  if (( failures > 0 )); then
    warn "$failures stock profile(s) were skipped: ${failed_profiles[*]}"
    warn "available profiles were still installed and created successfully"
    return 1
  fi
}

serial_running() {
  local serial="$1"
  [[ "$("$ADB" -s "$serial" get-state 2>/dev/null || true)" == "device" ]]
}

wait_for_boot() {
  local serial="$1"
  local expected_api="$2"
  local profile="$3"
  local elapsed=0
  local api
  local completed

  info "waiting for $profile on $serial (timeout ${BOOT_TIMEOUT}s)"

  while (( elapsed < BOOT_TIMEOUT )); do
    if serial_running "$serial"; then
      completed="$("$ADB" -s "$serial" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || true)"
      if [[ "$completed" == "1" ]]; then
        api="$("$ADB" -s "$serial" shell getprop ro.build.version.sdk 2>/dev/null | tr -d '\r')"
        [[ "$api" == "$expected_api" ]] ||
          die "$profile booted with API $api, expected API $expected_api"
        "$ADB" -s "$serial" shell input keyevent 82 >/dev/null 2>&1 || true
        info "$profile ready on $serial (API $api)"
        return 0
      fi
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done

  "$ADB" -s "$serial" emu kill >/dev/null 2>&1 || true
  die "$profile did not finish booting within ${BOOT_TIMEOUT}s; inspect $LOG_DIR/${profile}.log"
}

append_env_mapping() {
  local serial_env="$1"
  local serial="$2"

  mkdir -p "$(dirname "$ENV_FILE")"
  touch "$ENV_FILE"

  grep -Ev "^export ${serial_env}=" "$ENV_FILE" >"${ENV_FILE}.tmp" || true
  mv "${ENV_FILE}.tmp" "$ENV_FILE"
  printf 'export %s=%q\n' "$serial_env" "$serial" >>"$ENV_FILE"
}

boot_profile() {
  local wanted="$1"
  local row
  local profile os_family api expected_avd serial_env hardware kind port ram
  local actual_avd
  local serial
  local log

  require_tools
  row="$(row_for_profile "$wanted")" || die "unknown profile: $wanted"
  IFS=$'\t' read -r profile os_family api expected_avd serial_env hardware kind port ram <<<"$row"

  if ! actual_avd="$(resolve_existing_avd "$expected_avd")"; then
    if [[ "$kind" == "stock" ]]; then
      die "AVD '$expected_avd' is missing; run: $0 prepare-stock"
    fi
    die "genuine custom AVD '$expected_avd' is missing; install it first, then rerun"
  fi

  serial="emulator-${port}"
  log="$LOG_DIR/${profile}.log"
  mkdir -p "$LOG_DIR"

  if serial_running "$serial"; then
    info "$profile is already running on $serial"
  else
    info "booting $profile using AVD '$actual_avd' on port $port"
    nohup "$EMULATOR" \
      -avd "$actual_avd" \
      -port "$port" \
      -no-snapshot-load \
      -no-boot-anim \
      -gpu auto \
      -no-audio \
      >"$log" 2>&1 &
  fi

  wait_for_boot "$serial" "$api" "$profile"
  append_env_mapping "$serial_env" "$serial"

  printf '\nRun in the current shell:\n'
  printf '  export %s=%q\n' "$serial_env" "$serial"
  printf '  export ANDROID_DRLQN_ALLOW_EMULATOR_INPUT=1\n'
}

boot_stock() {
  local profile os_family api avd serial_env hardware kind port ram

  while IFS=$'\t' read -r profile os_family api avd serial_env hardware kind port ram; do
    [[ "$kind" == "stock" ]] || continue
    boot_profile "$profile"
  done < <(matrix)
}

boot_all() {
  local profile os_family api avd serial_env hardware kind port ram
  local memory_bytes
  local memory_gib

  [[ "${ANDROID_DRLQN_BOOT_ALL:-0}" == "1" ]] ||
    die "booting nine emulators is resource-intensive; rerun with ANDROID_DRLQN_BOOT_ALL=1"

  memory_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
  memory_gib=$((memory_bytes / 1024 / 1024 / 1024))
  if (( memory_gib < 48 )); then
    warn "only ${memory_gib} GiB RAM detected; boot-all may be unstable"
    warn "prefer print-autoboot-env or boot one profile at a time"
  fi

  while IFS=$'\t' read -r profile os_family api avd serial_env hardware kind port ram; do
    boot_profile "$profile"
  done < <(matrix)
}

stop_profile() {
  local wanted="$1"
  local row profile os_family api avd serial_env hardware kind port ram
  local serial

  row="$(row_for_profile "$wanted")" || die "unknown profile: $wanted"
  IFS=$'\t' read -r profile os_family api avd serial_env hardware kind port ram <<<"$row"
  serial="emulator-${port}"

  if serial_running "$serial"; then
    info "stopping $profile on $serial"
    "$ADB" -s "$serial" emu kill >/dev/null 2>&1 || true
  else
    info "$profile is not running on $serial"
  fi
}

stop_all() {
  local profile os_family api avd serial_env hardware kind port ram

  while IFS=$'\t' read -r profile os_family api avd serial_env hardware kind port ram; do
    stop_profile "$profile"
  done < <(matrix)
}

status() {
  local profile os_family api expected_avd serial_env hardware kind port ram
  local serial actual_avd state actual_api fingerprint

  require_tools
  printf '%-32s %-10s %-22s %-8s %-6s %s\n' \
    PROFILE KIND AVD SERIAL API STATE

  while IFS=$'\t' read -r profile os_family api expected_avd serial_env hardware kind port ram; do
    serial="emulator-${port}"
    actual_avd="$(resolve_existing_avd "$expected_avd" 2>/dev/null || printf '%s' MISSING)"

    if serial_running "$serial"; then
      state=running
      actual_api="$("$ADB" -s "$serial" shell getprop ro.build.version.sdk 2>/dev/null | tr -d '\r')"
    else
      state=stopped
      actual_api="-"
    fi

    printf '%-32s %-10s %-22s %-8s %-6s %s\n' \
      "$profile" "$kind" "$actual_avd" "$serial" "$actual_api" "$state"
  done < <(matrix)
}

check_custom() {
  local missing=0
  local profile os_family api expected_avd serial_env hardware kind port ram
  local actual

  printf '\nCustom OS AVD check:\n'
  while IFS=$'\t' read -r profile os_family api expected_avd serial_env hardware kind port ram; do
    [[ "$kind" == "custom" ]] || continue
    if actual="$(resolve_existing_avd "$expected_avd")"; then
      printf '  OK      %-32s -> %s\n' "$profile" "$actual"
    else
      printf '  MISSING %-32s expected AVD: %s\n' "$profile" "$expected_avd"
      missing=$((missing + 1))
    fi
  done < <(matrix)

  if (( missing > 0 )); then
    warn "$missing genuine custom-OS AVD(s) are missing"
    warn "do not create AOSP stand-ins under those names; that would corrupt OS-family labels"
    return 1
  fi
}

print_env() {
  local profile os_family api avd serial_env hardware kind port ram
  local serial

  printf 'export ANDROID_HOME=%q\n' "$ANDROID_HOME"
  printf 'export ANDROID_SDK_ROOT=%q\n' "$ANDROID_SDK_ROOT"
  printf 'export ANDROID_DRLQN_ADB=%q\n' "$ADB"
  printf 'export ANDROID_DRLQN_EMULATOR=%q\n' "$EMULATOR"
  printf 'export ANDROID_DRLQN_ALLOW_EMULATOR_INPUT=1\n'

  while IFS=$'\t' read -r profile os_family api avd serial_env hardware kind port ram; do
    serial="emulator-${port}"
    if serial_running "$serial"; then
      printf 'export %s=%q\n' "$serial_env" "$serial"
    fi
  done < <(matrix)
}

print_autoboot_env() {
  printf 'export ANDROID_HOME=%q\n' "$ANDROID_HOME"
  printf 'export ANDROID_SDK_ROOT=%q\n' "$ANDROID_SDK_ROOT"
  printf 'export ANDROID_DRLQN_ADB=%q\n' "$ADB"
  printf 'export ANDROID_DRLQN_EMULATOR=%q\n' "$EMULATOR"
  printf 'export ANDROID_DRLQN_LAB_AUTOBOOT=1\n'
  printf 'export ANDROID_DRLQN_ALLOW_EMULATOR_INPUT=1\n'
}

doctor() {
  require_tools

  printf 'Host architecture: %s\n' "$(uname -m)"
  printf 'ANDROID_HOME:      %s\n' "$ANDROID_HOME"
  printf 'ANDROID_AVD_HOME:  %s\n' "$ANDROID_AVD_HOME"
  printf 'sdkmanager:        %s\n' "$SDKMANAGER"
  printf 'avdmanager:        %s\n' "$AVDMANAGER"
  printf 'emulator:          %s\n' "$EMULATOR"
  printf 'adb:               %s\n' "$ADB"
  printf 'SDK channel:       %s\n' "$SDK_CHANNEL"
  printf '\nInstalled AVDs:\n'
  available_avds | sed 's/^/  /'

  printf '\nConfigured profiles:\n'
  list_profiles | sed 's/^/  /'

  check_custom || true
}

usage() {
  cat <<EOF
Usage:
  $0 doctor
  $0 list
  $0 prepare-stock
  $0 prepare-profile <profile>
  $0 check-custom
  $0 boot <profile>
  $0 boot-stock
  ANDROID_DRLQN_BOOT_ALL=1 $0 boot-all
  $0 status
  $0 print-env
  $0 print-autoboot-env
  $0 stop <profile|all>

Recommended workflow on Apple Silicon:
  1. $0 prepare-stock
  2. $0 check-custom
  3. source <($0 print-autoboot-env)
  4. Run train_on_policy_ollama.py

To override a stock image package:
  export ANDROID_DRLQN_IMAGE_PACKAGE_API_37='system-images;android-37;google_apis;arm64-v8a'
EOF
}

main() {
  local command="${1:-}"
  case "$command" in
    doctor)
      doctor
      ;;
    list)
      list_profiles
      ;;
    prepare-stock)
      prepare_stock
      ;;
    prepare-profile)
      [[ $# -eq 2 ]] || die "prepare-profile requires a profile name"
      prepare_profile "$2"
      ;;
    check-custom)
      require_tools
      check_custom
      ;;
    boot)
      [[ $# -eq 2 ]] || die "boot requires a profile name"
      boot_profile "$2"
      ;;
    boot-stock)
      boot_stock
      ;;
    boot-all)
      boot_all
      ;;
    status)
      status
      ;;
    print-env)
      require_tools
      print_env
      ;;
    print-autoboot-env)
      require_tools
      print_autoboot_env
      ;;
    stop)
      [[ $# -eq 2 ]] || die "stop requires a profile name or 'all'"
      if [[ "$2" == "all" ]]; then
        stop_all
      else
        stop_profile "$2"
      fi
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      usage >&2
      die "unknown command: $command"
      ;;
  esac
}

main "$@"
