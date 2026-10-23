#!/usr/bin/env bash
# Build, install, and run the on-device MLX smoke test on a connected iPhone.
#
# Answers the two questions no build can settle (docs/ios-epics.md, E3):
#   S3.3 -- does the bundled metallib resolve inside the app sandbox?
#   R11  -- are the cross-compiled Metal kernels numerically correct?
#
# Usage:  scripts/ios/run_smoke.sh [--debug] [--media|--zimage] [--unadmitted]
#             [--device <udid> | --simulator [<udid>|<name>]] [--memory-entitlement on|off]
#
# Device:    a paired device with Developer Mode on, xcodegen, and a signing identity.
# Simulator: xcodegen and an installed iOS simulator runtime; no signing, no device. The simulator
#            is a BUILD/packaging/correctness check only: it has no jetsam per-app cap (the report
#            says so) and no ANE, so no memory or performance claim may rest on it.
#
# --memory-entitlement (device only; default on) selects whether the app is signed with
# com.apple.developer.kernel.increased-memory-limit. The app logs os_proc_available_memory at
# launch (Documents/launch-memory.txt, also pulled to /tmp/ios-smoke-launch-memory.txt), so one run
# with `on` and one with `off` is the per-app cap measurement (sc-16780).
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$(pwd)

PROFILE=release
CARGO_PROFILE_FLAG=--release
DEVICE=""
SIMULATOR=""
MEMORY_ENTITLEMENT=on
CARGO_FEATURES=""
INSTALL_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --debug)  PROFILE=debug; CARGO_PROFILE_FLAG=""; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    # Optional argument: a simulator UDID or device-type name; default is the first booted
    # simulator, else the first available iPhone.
    --simulator)
      if [ $# -gt 1 ] && [ "${2#--}" = "$2" ]; then SIMULATOR="$2"; shift 2; else SIMULATOR=auto; shift; fi ;;
    --memory-entitlement)
      case "${2:-}" in on|off) MEMORY_ENTITLEMENT="$2" ;; *) echo "--memory-entitlement takes on|off" >&2; exit 2 ;; esac
      shift 2 ;;
    # E5: also run the SANA image-generation check. Needs a SANA snapshot pushed to
    # Documents/<any-subdir>/ (scripts/ios/push_model.sh <dir> sana); without one the check
    # reports "skipped" rather than failing.
    --media)  CARGO_FEATURES="--features media"; shift ;;
    # One bounded CAPTURE of a Z-Image Turbo campaign case (IOS_SMOKE_ZIMAGE_CASE, default
    # z_image_turbo/z_image_turbo/q4/1024/text/4/s_t512_o64_c67108864_w1_dit/1) through the capture
    # permit (sc-26433); off-device or without a provisioned slot it skips; needs the
    # campaign slot (scripts/ios/push_model.sh <tier-dir> --campaign-route z_image_turbo --tier q4).
    --zimage) CARGO_FEATURES="--features media,zimage"; shift ;;
    # The measurement lane: LLM loads have no footprint estimate, so iOS admission
    # refuses them by default (the report says "refused before load"). This runs them UNADMITTED to
    # produce that estimate, and the report's "iOS admission lane" line says so.
    --unadmitted) export IOS_SMOKE_UNADMITTED=1; shift ;;
    --campaign-case)
      [ $# -ge 2 ] || { echo "--campaign-case needs provider/route/tier/edge/mode/rung/selection/steps" >&2; exit 2; }
      export IOS_CAMPAIGN_CASE="$2"; CARGO_FEATURES="--features media"; shift 2 ;;
    --install-only) INSTALL_ONLY=1; shift ;;
    --campaign-timeout) export IOS_CAMPAIGN_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --campaign-list) export IOS_CAMPAIGN_LIST=1; CARGO_FEATURES="--features media"; shift ;;
    --campaign-diagnostic) export IOS_CAMPAIGN_DIAGNOSTIC=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -n "${IOS_CAMPAIGN_CASE:-}${IOS_CAMPAIGN_LIST:-}" ]; then
  [ -z "${IOS_SMOKE_UNADMITTED:-}${IOS_SMOKE_SANA_CAPTURE:-}" ] || { echo "campaign cannot inherit legacy unadmitted/capture bypass flags" >&2; exit 2; }
  [ -z "${IOS_CAMPAIGN_CASE:-}" ] || [ -z "${IOS_CAMPAIGN_LIST:-}" ] || { echo "select a case OR listing" >&2; exit 2; }
  [ -z "$SIMULATOR" ] || [ -n "${IOS_CAMPAIGN_LIST:-}" ] || { echo "campaign execution requires a real device; simulator is build/list only" >&2; exit 2; }
fi
[ -z "${IOS_CAMPAIGN_DIAGNOSTIC:-}" ] || [ -n "${IOS_CAMPAIGN_CASE:-}" ] || { echo "--campaign-diagnostic needs --campaign-case" >&2; exit 2; }

# Campaign flags always use the bounded session driver. They must never reach this script's
# legacy reinstall/terminate-existing smoke path.
if [ -n "${IOS_CAMPAIGN_LIST:-}" ]; then
  exec python3 "$ROOT/scripts/ios/campaign.py" list
fi
if [ -n "${IOS_CAMPAIGN_CASE:-}" ] && [ "$INSTALL_ONLY" -eq 0 ]; then
  [ -n "$DEVICE" ] || { echo "campaign requires explicit --device" >&2; exit 2; }
  OUTPUT=${IOS_CAMPAIGN_OUTPUT:-/tmp/ios-campaign-$(date +%s)-$$.jsonl}
  CAMPAIGN_ARGS=(run --case "$IOS_CAMPAIGN_CASE" --device "$DEVICE" --output "$OUTPUT" --timeout-seconds "${IOS_CAMPAIGN_TIMEOUT_SECONDS:-1800}" --memory-entitlement "$MEMORY_ENTITLEMENT")
  if [ "${IOS_CAMPAIGN_DIAGNOSTIC:-}" = 1 ]; then CAMPAIGN_ARGS+=(--diagnostic); fi
  exec python3 "$ROOT/scripts/ios/campaign.py" "${CAMPAIGN_ARGS[@]}"
fi

say() { printf '\n=== %s\n' "$1"; }

# The bundle id, for the same reason the team is resolved rather than committed: it belongs to
# whoever signs. The default is what this repo has always used and needs no action; a developer whose
# team cannot claim `com.idkplay.*` exports SMOKE_BUNDLE_ID and every script here follows it.
#
# Changing it gives the app a FRESH Documents container, so a pushed model snapshot has to be pushed
# again — several GB. Change it only if signing actually fails, not defensively.
SMOKE_BUNDLE_ID=${SMOKE_BUNDLE_ID:-com.idkplay.SceneWorksSmoke}
export SMOKE_BUNDLE_ID

if [ -n "$SIMULATOR" ]; then
  [ -z "$DEVICE" ] || { echo "--device and --simulator are mutually exclusive" >&2; exit 2; }
  ON_SIMULATOR=1
else
  ON_SIMULATOR=0
fi

# --- simulator --------------------------------------------------------------------------------
# Everything the device path does except signing, devicectl and the entitlement A/B: build the
# aarch64-apple-ios-sim staticlib, assert its metallib is an iOS-simulator one, build the app for
# the iphonesimulator SDK, install, launch with the same env knobs, and read the report straight
# out of the simulator's data container.
if [ "$ON_SIMULATOR" -eq 1 ]; then
  if [ "$SIMULATOR" = auto ]; then
    SIMULATOR=$(xcrun simctl list devices booted 2>/dev/null \
      | grep -oE '[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}' | head -1 || true)
    if [ -z "$SIMULATOR" ]; then
      SIMULATOR=$(xcrun simctl list devices available 2>/dev/null | grep -E '^\s+iPhone' \
        | grep -oE '[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}' | head -1 || true)
    fi
  elif ! echo "$SIMULATOR" | grep -qE '^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-'; then
    NAME="$SIMULATOR"
    SIMULATOR=$(xcrun simctl list devices available 2>/dev/null | grep -F "    $NAME (" \
      | grep -oE '[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}' | head -1 || true)
  fi
  [ -n "$SIMULATOR" ] || { echo "no iOS simulator found (xcrun simctl list devices available)" >&2; exit 1; }
  say "simulator $SIMULATOR"
  [ "$MEMORY_ENTITLEMENT" = on ] || say "NOTE: --memory-entitlement is ignored on the simulator (unsigned; no jetsam cap)"

  SIM_TRIPLE=aarch64-apple-ios-sim
  SMOKE_TARGET="$ROOT/ios-host/smoke/target"
  LIB="$SMOKE_TARGET/$SIM_TRIPLE/$PROFILE/libios_smoke.a"
  say "building ios-smoke for $SIM_TRIPLE ($PROFILE${CARGO_FEATURES:+, $CARGO_FEATURES})"
  # shellcheck disable=SC2086  # both flag vars are deliberately word-split
  ( cd "$ROOT/ios-host/smoke" && cargo build --target "$SIM_TRIPLE" $CARGO_PROFILE_FLAG $CARGO_FEATURES )
  test -f "$LIB" || { echo "expected staticlib at $LIB" >&2; exit 1; }

  say "verifying the metallib targets the iOS simulator"
  METALLIB=$(find "$SMOKE_TARGET/$SIM_TRIPLE/$PROFILE" -name mlx.metallib -print -quit)
  test -n "$METALLIB" || { echo "no mlx.metallib was built" >&2; exit 1; }
  METAL_TRIPLE=$(strings "$METALLIB" | grep -oE 'apple-(ios|macos)[0-9.]*(-simulator)?' | sort -u | tr '\n' ' ')
  case "$METAL_TRIPLE" in
    *apple-macos*) echo "$METALLIB carries macOS kernels -- the cross-compile regressed" >&2; exit 1 ;;
    *-simulator*)  echo "metallib targets $METAL_TRIPLE" ;;
    *apple-ios*)   echo "$METALLIB carries iOS DEVICE kernels, not simulator ones" >&2; exit 1 ;;
    *)             echo "$METALLIB has no recognizable target triple" >&2; exit 1 ;;
  esac

  say "generating and building the app (iphonesimulator)"
  # No team: the simulator build is unsigned. Exported empty so xcodegen's ${DEVELOPMENT_TEAM}
  # interpolation resolves to nothing rather than to a stale value from the caller's shell.
  export DEVELOPMENT_TEAM=""
  ( cd ios-host && xcodegen generate )
  CONFIG=Release
  [ "$PROFILE" = debug ] && CONFIG=Debug
  DERIVED="$ROOT/target/ios-derived-sim"
  xcodebuild \
    -project ios-host/SceneWorksSmoke.xcodeproj \
    -scheme SceneWorksSmoke \
    -configuration "$CONFIG" \
    -sdk iphonesimulator \
    -destination "id=$SIMULATOR" \
    -derivedDataPath "$DERIVED" \
    CODE_SIGNING_ALLOWED=NO \
    build 2>&1 | tail -20
  APP=$(find "$DERIVED/Build/Products/$CONFIG-iphonesimulator" -name "SceneWorksSmoke.app" -print -quit)
  test -n "$APP" || { echo "no .app was produced" >&2; exit 1; }
  test -f "$APP/mlx.metallib" \
    || { echo "mlx.metallib is missing from the bundle; the app would fail at first Metal use" >&2; exit 1; }

  say "booting + installing"
  xcrun simctl boot "$SIMULATOR" 2>/dev/null || true
  xcrun simctl bootstatus "$SIMULATOR" -b >/dev/null
  xcrun simctl install "$SIMULATOR" "$APP"
  DATA=$(xcrun simctl get_app_container "$SIMULATOR" "$SMOKE_BUNDLE_ID" data)
  mkdir -p "$DATA/Documents"
  say "data container $DATA (push a SANA snapshot with: scripts/ios/push_model.sh <dir> sana --simulator $SIMULATOR)"

  # simctl forwards SIMCTL_CHILD_<VAR> from the calling environment as <VAR> in the app.
  for var in IOS_CAMPAIGN_TIMEOUT_SECONDS IOS_CAMPAIGN_BUILD_SHA IOS_CAMPAIGN_CASE IOS_CAMPAIGN_LIST IOS_CAMPAIGN_DIAGNOSTIC IOS_CAMPAIGN_REFUSAL_ONLY IOS_SMOKE_UNADMITTED IOS_SMOKE_SANA_CAPTURE IOS_SMOKE_IMAGE_ONLY IOS_SMOKE_ONLY IOS_SMOKE_ZIMAGE_SIZE IOS_SMOKE_ZIMAGE_TILE IOS_SMOKE_ZIMAGE_RUNG4 IOS_SMOKE_SOAK_SECS MLX_GEN_SANA_DECODE_TILE; do
    val=$(printenv "$var" || true)
    if [ -n "$val" ]; then export "SIMCTL_CHILD_$var=$val"; say "forwarding $var=$val"; fi
  done
  REPORT_SRC="$DATA/Documents/smoke-report.txt"
  rm -f "$REPORT_SRC" "$DATA/Documents/launch-memory.txt"
  say "launching"
  xcrun simctl launch --terminate-running-process "$SIMULATOR" "$SMOKE_BUNDLE_ID"

  POLL_TRIES=${POLL_TRIES:-60}
  if [ -n "$CARGO_FEATURES" ]; then POLL_TRIES=${POLL_TRIES_MEDIA:-200}; fi
  say "waiting for the report (up to ~$((POLL_TRIES * 3))s)"
  for _ in $(seq 1 "$POLL_TRIES"); do
    [ -f "$REPORT_SRC" ] && break
    sleep 3
  done
  [ -f "$DATA/Documents/launch-memory.txt" ] && cat "$DATA/Documents/launch-memory.txt"
  if [ ! -f "$REPORT_SRC" ]; then
    echo "no report was produced" >&2
    [ -f "$DATA/Documents/sana-progress.txt" ] && sed 's/^/  /' "$DATA/Documents/sana-progress.txt" >&2
    exit 1
  fi
  REPORT=/tmp/ios-smoke-report.txt
  cp "$REPORT_SRC" "$REPORT"
  echo
  cat "$REPORT"
  ARTIFACT_DIR=${ARTIFACT_DIR:-$HOME/Desktop/sana-ios}
  mkdir -p "$ARTIFACT_DIR"
  for png in "$DATA"/Documents/*.png; do
    [ -f "$png" ] && cp "$png" "$ARTIFACT_DIR/sim-$(basename "$png")" && echo "  pulled $(basename "$png") -> $ARTIFACT_DIR/sim-$(basename "$png")"
  done
  if [ "${IOS_CAMPAIGN_LIST:-}" = 1 ] && head -1 "$REPORT" | grep -q '^CAMPAIGN: LIST$'; then say "catalog only (not execution evidence)"; exit 0; fi
  if ! head -1 "$REPORT" | grep -q "SMOKE: PASS"; then say "FAIL"; exit 1; fi
  say "PASS (simulator: packaging + correctness only; no memory/perf thresholds apply)"
  exit 0
fi

# --- device -----------------------------------------------------------------------------------
if [ -z "$DEVICE" ]; then
  # Matched by UUID SHAPE, not by column position. The previous `$(NF-3)` counted back from the end
  # of the line, which put it inside the Model column as soon as that column had a different word
  # count -- "iPhone 17 Pro Max (iPhone18,2)" yields NF=11 and `$(NF-3)` is the literal "17". Every
  # subsequent devicectl call then failed with "The specified device was not found. (Name: 17)".
  #
  # Any usable STATE, not just "available (paired)". devicectl also reports "connected", and a
  # device flips between the two on its own -- so a filter on one of them was a coin toss. Worse,
  # it failed SILENTLY: `grep` exiting 1 made the pipeline fail, `set -o pipefail` propagated it,
  # the assignment inherited it, and `set -e` killed the script BEFORE its first line of output.
  # Twice that produced an empty log and a bare exit 1, which reads as "the script did nothing"
  # rather than "no device matched". The `|| true` and the explicit check below are what make the
  # failure legible.
  DEVICE=$(xcrun devicectl list devices 2>/dev/null \
    | grep -Ei 'available|connected' \
    | grep -oE '[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}' \
    | head -1 || true)
fi
if [ -z "$DEVICE" ]; then
  echo "no paired device found; connect an iPhone with Developer Mode enabled" >&2
  exit 1
fi
say "device $DEVICE"

# --- signing ----------------------------------------------------------------------------------
# Resolved locally rather than committed: project.yml interpolates DEVELOPMENT_TEAM, so nothing
# team-specific lives in the repo.
#
# Read the team from Xcode's signed-in ACCOUNTS, not from the keychain. A codesigning
# certificate's OU is a team id, but not necessarily one this Xcode has an account for -- and
# automatic signing fails with "No Account for Team" when they differ, which is exactly what
# happened here (the cert said 4UVJ2P7FW5, the account is 3FMAPDQDGP).
if [ -z "${DEVELOPMENT_TEAM:-}" ]; then
  DEVELOPMENT_TEAM=$(defaults read com.apple.dt.Xcode IDEProvisioningTeamByIdentifier 2>/dev/null \
    | sed -n 's/.*teamID = \([A-Z0-9]*\);.*/\1/p' | head -1)
fi
if [ -z "$DEVELOPMENT_TEAM" ]; then
  echo "no Xcode developer account found; sign in via Xcode > Settings > Accounts," >&2
  echo "or set DEVELOPMENT_TEAM=<team id> explicitly" >&2
  exit 1
fi
export DEVELOPMENT_TEAM
say "signing team $DEVELOPMENT_TEAM"

say "bundle id $SMOKE_BUNDLE_ID"

# The increased-memory-limit entitlement A/B (sc-16780). Both builds keep extended virtual
# addressing; `off` drops ONLY increased-memory-limit, and the Info.plist marker moves with it so the
# app's launch log names the build it is. Passed as xcodebuild overrides of project.yml's defaults.
if [ "$MEMORY_ENTITLEMENT" = on ]; then
  ENTITLEMENT_OVERRIDES=(CODE_SIGN_ENTITLEMENTS=App/SceneWorksSmoke.entitlements SMOKE_INCREASED_MEMORY_LIMIT=YES)
else
  ENTITLEMENT_OVERRIDES=(CODE_SIGN_ENTITLEMENTS=App/SceneWorksSmoke-NoIncreasedLimit.entitlements SMOKE_INCREASED_MEMORY_LIMIT=NO)
fi
say "increased-memory-limit entitlement: $MEMORY_ENTITLEMENT (${ENTITLEMENT_OVERRIDES[*]})"

# Stamp the exact compiled checkout, not a caller label on a stale installed app.
export IOS_CAMPAIGN_BUILD_SHA=$(git rev-parse HEAD)

# --- rust staticlib ---------------------------------------------------------------------------
# No IPHONEOS_DEPLOYMENT_TARGET here on purpose: .cargo/config.toml floors the iOS triples (per-target
# rustflags, not [env] -- see the note there), and this script is
# also a check that a clean invocation needs no environment help.
# ios-host/smoke is excluded from the workspace, so it has its OWN target directory rather than
# the root one. The metallib it builds lives there too.
SMOKE_TARGET="$ROOT/ios-host/smoke/target"
LIB="$SMOKE_TARGET/aarch64-apple-ios/$PROFILE/libios_smoke.a"

say "building ios-smoke for aarch64-apple-ios ($PROFILE${CARGO_FEATURES:+, $CARGO_FEATURES})"
# shellcheck disable=SC2086  # both flag vars are deliberately word-split
( cd "$ROOT/ios-host/smoke" && cargo build --target aarch64-apple-ios $CARGO_PROFILE_FLAG $CARGO_FEATURES )

# Guard against a stale staticlib. Xcode links whatever .a is on disk, so a Rust source edit that
# did not get rebuilt produces a green run reporting the PREVIOUS build's results -- which looks
# like a passing test rather than a build mistake, and cost three confusing runs before it was
# caught. Compare mtimes and fail loudly instead.
# Only sources that can actually reach the staticlib. `tests/`, `examples/` and `benches/` are
# separate compilation units that never link into it, so editing one and re-running would otherwise
# abort a perfectly valid device run — which is exactly what happened after adding a SANA
# real-weights test. A guard that cries wolf gets disabled, so keep it precise.
NEWEST_SRC=$(find "$ROOT/ios-host/smoke/src" "$ROOT/crates" -name '*.rs' -newer "$LIB" \
  -not -path '*/tests/*' -not -path '*/examples/*' -not -path '*/benches/*' \
  -print -quit 2>/dev/null || true)
if [ -n "$NEWEST_SRC" ]; then
  echo "stale staticlib: $NEWEST_SRC is newer than $LIB" >&2
  echo "(the build above should have caught this -- check for a cargo failure)" >&2
  exit 1
fi
test -f "$LIB" || { echo "expected staticlib at $LIB" >&2; exit 1; }

# Fail here rather than at first Metal op on the phone.
say "verifying the metallib targets iOS"
METALLIB=$(find "$SMOKE_TARGET/aarch64-apple-ios" -name mlx.metallib -print -quit)
test -n "$METALLIB" || { echo "no mlx.metallib was built" >&2; exit 1; }
METAL_TRIPLE=$(strings "$METALLIB" | grep -oE 'apple-(ios|macos)[0-9.]*(-simulator)?' | sort -u | tr '\n' ' ')
case "$METAL_TRIPLE" in
  *apple-macos*) echo "$METALLIB carries macOS kernels -- the cross-compile regressed" >&2; exit 1 ;;
  *-simulator*)  echo "$METALLIB carries iOS-simulator kernels, not device ones" >&2; exit 1 ;;
  *apple-ios*)   echo "metallib targets $METAL_TRIPLE" ;;
  *)             echo "$METALLIB has no recognizable target triple" >&2; exit 1 ;;
esac

# --- app --------------------------------------------------------------------------------------
say "generating and building the app"
( cd ios-host && xcodegen generate )

CONFIG=Release
[ "$PROFILE" = debug ] && CONFIG=Debug
DERIVED="$ROOT/target/ios-derived"

xcodebuild \
  -project ios-host/SceneWorksSmoke.xcodeproj \
  -scheme SceneWorksSmoke \
  -configuration "$CONFIG" \
  -destination "id=$DEVICE" \
  -derivedDataPath "$DERIVED" \
  -allowProvisioningUpdates \
  "${ENTITLEMENT_OVERRIDES[@]}" \
  build 2>&1 | tail -20

APP=$(find "$DERIVED/Build/Products" -name "SceneWorksSmoke.app" -print -quit)
test -n "$APP" || { echo "no .app was produced" >&2; exit 1; }

# The whole point of the packaging step -- assert it landed before installing.
test -f "$APP/mlx.metallib" \
  || { echo "mlx.metallib is missing from the bundle; the app would fail at first Metal use" >&2; exit 1; }
say "bundled metallib: $(du -h "$APP/mlx.metallib" | cut -f1)"

say "installing"
INSTALL_STATUS=0
INSTALL_LOG=$(xcrun devicectl device install app --device "$DEVICE" "$APP" 2>&1) || INSTALL_STATUS=$?
echo "$INSTALL_LOG" | tail -5

# A team change makes the install fail with MismatchedApplicationIdentifierEntitlement, because
# iOS refuses to upgrade across app-identifier prefixes. The raw error names both prefixes but not
# the remedy, and the remedy is destructive (uninstalling drops the app's Documents container, i.e.
# every pushed snapshot), so say so explicitly rather than letting it read as a signing bug.
if echo "$INSTALL_LOG" | grep -q "MismatchedApplicationIdentifierEntitlement"; then
  INSTALLED_TEAM=$(echo "$INSTALL_LOG" \
    | grep -oE "installed application's application-identifier string \([A-Z0-9]+\." \
    | grep -oE '\([A-Z0-9]+' | tr -d '(' | head -1)
  echo >&2
  echo "the installed app was signed by team ${INSTALLED_TEAM:-<unknown>}; this build uses $DEVELOPMENT_TEAM." >&2
  echo "iOS cannot upgrade across teams. Uninstall and reinstall:" >&2
  echo >&2
  echo "  xcrun devicectl device uninstall app --device $DEVICE $SMOKE_BUNDLE_ID" >&2
  echo >&2
  echo "NOTE: that DELETES the app's Documents container, including every pushed snapshot." >&2
  echo "Re-provision afterwards with scripts/ios/push_model.sh (see docs/ios-epics.md E5)." >&2
  exit 1
fi

# Campaigns install once, then use the Python session's non-terminating per-case launcher.
[ "$INSTALL_STATUS" -eq 0 ] || { echo "install failed (exit $INSTALL_STATUS)" >&2; exit "$INSTALL_STATUS"; }
if [ "$INSTALL_ONLY" -eq 1 ]; then
  say "installed only; no app launch requested"
  exit 0
fi

# The device must be UNLOCKED to launch an app: SpringBoard denies the request with
# FBSOpenApplicationErrorDomain 7 otherwise, after a successful build and install. Say so plainly
# rather than letting a 40-line CoreDevice error explain it.
# Forward the harness's own env gates to the app. `devicectl` takes them as a JSON dict; without
# this an on-device run can only ever exercise the default configuration, which is why the
# image-only attribution experiment (IOS_SMOKE_IMAGE_ONLY) had no way to reach the device.
LAUNCH_ENV="{}"
# A knob missing from this list does not fail — it silently runs the DEFAULT configuration and
# reports it under whatever label the caller believed. `IOS_SMOKE_ZIMAGE_RUNG4=0` was absent here and
# produced a 216.3 s "rung 4 disabled" run whose MLX peak (2901 MiB) was byte-identical to the
# rung-4-ENABLED run, when the host A/B moves that peak by 1662 MiB. The measurement was a no-op
# wearing the name of an experiment. If you add a knob to the harness, add it here in the same edit.
for var in IOS_CAMPAIGN_TIMEOUT_SECONDS IOS_CAMPAIGN_BUILD_SHA IOS_CAMPAIGN_CASE IOS_CAMPAIGN_LIST IOS_CAMPAIGN_DIAGNOSTIC IOS_CAMPAIGN_REFUSAL_ONLY IOS_SMOKE_UNADMITTED IOS_SMOKE_SANA_CAPTURE IOS_SMOKE_IMAGE_ONLY IOS_SMOKE_ONLY IOS_SMOKE_ZIMAGE_SIZE IOS_SMOKE_ZIMAGE_TILE IOS_SMOKE_ZIMAGE_RUNG4 IOS_SMOKE_SOAK_SECS MLX_GEN_SANA_DECODE_TILE; do
  val=$(printenv "$var" || true)
  if [ -n "$val" ]; then
    LAUNCH_ENV=$(python3 -c "
import json,sys
d = json.loads(sys.argv[1]); d[sys.argv[2]] = sys.argv[3]; print(json.dumps(d))" "$LAUNCH_ENV" "$var" "$val")
  fi
done
LAUNCHED_AT=$(date +%s)
say "launching (unlock the device if prompted; console output follows)"
# Two explicit invocations rather than an argument array. devicectl rejects an empty `{}` with
# "Unable to decode environment variables" (which reads like a quoting bug and is not one), and
# macOS ships bash 3.2, where expanding an EMPTY array under `set -u` is itself an error. Between
# them those two cost two device runs; a branch has neither problem.
if [ "$LAUNCH_ENV" = "{}" ]; then
  xcrun devicectl device process launch \
    --device "$DEVICE" --terminate-existing \
    "$SMOKE_BUNDLE_ID" 2>&1 | tail -5
else
  say "forwarding env: $LAUNCH_ENV"
  xcrun devicectl device process launch \
    --device "$DEVICE" --terminate-existing \
    --environment-variables "$LAUNCH_ENV" \
    "$SMOKE_BUNDLE_ID" 2>&1 | tail -5
fi

# The app writes its report to Documents and keeps running (it is a GUI app). Poll for the file
# rather than trying to capture stdout: --console does not reliably capture a GUI app's output,
# and `log collect` needs root.
# Poll for a report written AFTER the launch. Accepting the first file that copies is wrong: a
# previous run's report is still sitting in the container, so a crashed or slow app yields a
# confident-looking stale result -- which is exactly how a fixed bug appeared to persist.
# 180 s covers the LLM checks. The image checks add a 4.73 GB snapshot read from device storage
# plus two generations, so --media gets considerably longer — a timeout here is indistinguishable
# from a jetsam kill in the output, and mistaking "still working" for "died" would send the next
# hour after the wrong problem.
POLL_TRIES=${POLL_TRIES:-60}
# `if`, not `[ … ] && …`: under `set -e` a bare `test && assign` exits the script when the test is
# false, which is the same trap as the unguarded `from_check` above.
# 100 tries ≈ 10 min of wall clock (see the note at the poll), which covers a 4.73 GB snapshot read
# plus two tiled generations with room to spare. It was 200, which is 20 minutes of waiting for a
# run that has already died.
if [ -n "$CARGO_FEATURES" ]; then POLL_TRIES=${POLL_TRIES_MEDIA:-100}; fi

# Each iteration is `sleep 3` PLUS a devicectl copy that itself takes ~3s (tunnel + disk-image
# services), so the wall-clock budget is roughly double the naive sleep total. Say so, rather than
# advertising 600s and running for twenty minutes.
say "waiting for the on-device report (up to ~$((POLL_TRIES * 6))s)"
REPORT=/tmp/ios-smoke-report.txt
rm -f "$REPORT"
POLL_END=$((LAUNCHED_AT + ${IOS_CAMPAIGN_TIMEOUT_SECONDS:-$((POLL_TRIES * 6))}))
while [ "$(date +%s)" -lt "$POLL_END" ]; do
  sleep 3
  rm -f "$REPORT"
  xcrun devicectl device copy from --device "$DEVICE" \
    --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
    --source Documents/smoke-report.txt --destination "$REPORT" >/dev/null 2>&1 || continue
  # `copy from` preserves the device-side mtime, so this compares when the APP wrote it.
  [ "$(stat -f%m "$REPORT" 2>/dev/null || echo 0)" -ge "$LAUNCHED_AT" ] && break
  rm -f "$REPORT"
done

# The launch-time cap reading, pulled separately: it is written before the report runs anything, so
# it survives a jetsam kill of the report and is the cap measurement even when nothing else is.
LAUNCH_MEMORY=/tmp/ios-smoke-launch-memory.txt
rm -f "$LAUNCH_MEMORY"
EXPECTED_LIMIT=$( [ "$MEMORY_ENTITLEMENT" = on ] && echo YES || echo NO )
if xcrun devicectl device copy from --device "$DEVICE" \
    --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
    --source Documents/launch-memory.txt --destination "$LAUNCH_MEMORY" >/dev/null 2>&1; then
  # Same freshness guard as the report: a file left by the OTHER entitlement build's launch would
  # otherwise be printed as this run's cap measurement.
  if [ "$(stat -f%m "$LAUNCH_MEMORY" 2>/dev/null || echo 0)" -lt "$LAUNCHED_AT" ]; then
    echo "launch-memory.txt predates this launch (stale, from an earlier run); no cap reading" >&2
    exit 1
  fi
  say "$(cat "$LAUNCH_MEMORY") (expected increased-memory-limit = $EXPECTED_LIMIT)"
  if ! grep -q "increased-memory-limit = $EXPECTED_LIMIT\$" "$LAUNCH_MEMORY"; then
    echo "launch-memory.txt does not report increased-memory-limit = $EXPECTED_LIMIT;" \
      "the installed build is not the --memory-entitlement $MEMORY_ENTITLEMENT one" >&2
    exit 1
  fi
else
  if [ -n "${IOS_CAMPAIGN_CASE:-}" ]; then
    echo "campaign requires fresh launch-memory.txt entitlement verification" >&2
    exit 1
  fi
fi

if [ ! -f "$REPORT" ]; then
  echo "no report was produced; is the device unlocked?" >&2
  # A jetsam kill and a launch failure look identical from here: in both cases nothing was written.
  # The image check leaves a per-configuration breadcrumb precisely so the two can be told apart —
  # if it exists, the app ran and died partway rather than never starting.
  BREADCRUMB=/tmp/ios-sana-progress.txt
  rm -f "$BREADCRUMB"
  if xcrun devicectl device copy from --device "$DEVICE" \
      --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
      --source Documents/sana-progress.txt --destination "$BREADCRUMB" >/dev/null 2>&1; then
    echo >&2
    echo "the app DID run -- image-generation breadcrumbs found, so this is a mid-run death" >&2
    echo "(termination cause unconfirmed; retrieve the matching device diagnostic):" >&2
    sed 's/^/  /' "$BREADCRUMB" >&2
    echo >&2
    echo "the last breadcrumbs identify progress, not the termination cause" >&2
  fi
  exit 1
fi

echo
cat "$REPORT"

if [ "${IOS_CAMPAIGN_LIST:-}" = 1 ] && head -1 "$REPORT" | grep -q '^CAMPAIGN: LIST$'; then say "catalog only (not execution evidence)"; exit 0; fi
if ! head -1 "$REPORT" | grep -q "SMOKE: PASS"; then
  say "FAIL"
  exit 1
fi

# --- generated artifacts ------------------------------------------------------------------------
# Pull the renders. A memory number says the model FIT; only the image says it worked, and these are
# the artifacts worth keeping from a device session. Written to the desktop rather than /tmp so they
# survive and can actually be looked at.
ARTIFACT_DIR=${ARTIFACT_DIR:-$HOME/Desktop/sana-ios}
mkdir -p "$ARTIFACT_DIR"
# By the harness's own manifest (Documents/renders.txt, one PNG name per line, truncated at launch):
# the names derive from per-configuration labels, and a hand-copied list here drifted from them.
RENDERS=/tmp/ios-smoke-renders.txt
rm -f "$RENDERS"
if xcrun devicectl device copy from --device "$DEVICE" \
    --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
    --source Documents/renders.txt --destination "$RENDERS" >/dev/null 2>&1 \
    && [ "$(stat -f%m "$RENDERS" 2>/dev/null || echo 0)" -ge "$LAUNCHED_AT" ]; then
  while IFS= read -r png; do
    [ -n "$png" ] || continue
    if xcrun devicectl device copy from --device "$DEVICE" \
        --domain-type appDataContainer --domain-identifier "$SMOKE_BUNDLE_ID" \
        --source "Documents/$png" --destination "$ARTIFACT_DIR/device-$png" >/dev/null 2>&1; then
      echo "  pulled $png -> $ARTIFACT_DIR/device-$png"
    else
      echo "  could not pull $png (listed in renders.txt)" >&2
    fi
  done < "$RENDERS"
fi

# --- performance regression thresholds (S4.6) ---------------------------------------------------
# The report carries measurements the checks themselves do not assert on: a threshold invented
# from one device in one thermal state would be noise, and a check that fails on a warm phone
# teaches people to ignore it. So the numbers are asserted HERE, deliberately loose -- these catch
# a regression (a lost fast path, a leak, thermal collapse), not a slow afternoon.
#
# Baselines measured on an iPhone 17 Pro Max / iOS 26.5.2 with Qwen3-4B Q4 (docs/ios-epics.md E4):
#   steady throughput ~20.6 tok/s   peak RSS ~2.9 GiB   sustained RSS growth 0 MiB
# Re-baseline deliberately when the model, quantization, or device class changes.
THRESHOLD_MIN_TPS=${THRESHOLD_MIN_TPS:-12}
THRESHOLD_MAX_RSS_MIB=${THRESHOLD_MAX_RSS_MIB:-4096}
THRESHOLD_MAX_RSS_GROWTH_MIB=${THRESHOLD_MAX_RSS_GROWTH_MIB:-256}

fail_threshold() { echo "threshold: $1" >&2; THRESHOLD_FAILED=1; }
THRESHOLD_FAILED=0

# Pull a number from ONE named check's line, not from the whole report.
#
# These greps used to scan the file and take the last match, which silently depends on the set of
# checks and their order. Adding the E5 image-generation check broke exactly that: its detail
# carries "MLX peak ... MiB" and "process RSS peak ... MiB", so a bare `grep 'peak [0-9]* MiB' |
# tail -1` began reading SANA's number under the LLM's threshold. Anchor to the check name.
from_check() { # <check-name-fragment> <grep-pattern> <extract-pattern>
  # The trailing `|| true` is load-bearing under `set -e`. A skipped check (no snapshot pushed yet)
  # matches nothing, grep exits 1, and `var=$(...)` takes that as the assignment's status — which
  # killed the script AFTER it had printed "SMOKE: PASS", giving a green report and a red exit code.
  grep -- "$1" "$REPORT" 2>/dev/null | grep -o -- "$2" | grep -o -- "$3" | tail -1 || true
}

# Steady-state throughput. Sustained decode's LAST segment is the honest figure: the first is
# depressed by lazy weight faulting, so using it would mask a real slowdown.
tps=$(from_check 'sustained decode' 'last [0-9.]* tok/s' '[0-9.]*')
if [ -n "$tps" ]; then
  awk -v v="$tps" -v m="$THRESHOLD_MIN_TPS" 'BEGIN { exit !(v < m) }' \
    && fail_threshold "throughput ${tps} tok/s is below ${THRESHOLD_MIN_TPS} tok/s" \
    || echo "  throughput ${tps} tok/s (floor ${THRESHOLD_MIN_TPS})"
fi

# Peak RSS. 4 GiB is chosen against the ~4 GB cap of an 8 GB device, not the ~6 GB of this one --
# so the lane fails BEFORE a broader-device release would (docs/architecture/ios-project-spec.md
# §0.1), rather than after.
rss=$(from_check 'runtime-ios generation' 'peak [0-9]* MiB' '[0-9]*')
if [ -n "$rss" ] && [ "$rss" -gt "$THRESHOLD_MAX_RSS_MIB" ]; then
  fail_threshold "peak RSS ${rss} MiB exceeds ${THRESHOLD_MAX_RSS_MIB} MiB"
elif [ -n "$rss" ]; then
  echo "  peak RSS ${rss} MiB (ceiling ${THRESHOLD_MAX_RSS_MIB})"
fi

# Memory growth across repeated generations. Measured 0; a nonzero trend means a leak across
# calls, which on a memory-capped device ends as a jetsam kill with no crash log.
growth=$(from_check 'sustained decode' 'growth [0-9-]*' '[0-9-]*')
if [ -n "$growth" ] && [ "$growth" -gt "$THRESHOLD_MAX_RSS_GROWTH_MIB" ]; then
  fail_threshold "RSS grew ${growth} MiB across segments (limit ${THRESHOLD_MAX_RSS_GROWTH_MIB})"
elif [ -n "$growth" ]; then
  echo "  RSS growth ${growth} MiB (limit ${THRESHOLD_MAX_RSS_GROWTH_MIB})"
fi

# E5 image generation, when --media ran it. The gauge is MLX's own peak, not RSS: on the host
# `getrusage` was measured BELOW MLX's peak (2961 vs 4773 MiB), so it is not seeing Metal buffer
# allocations, and a ceiling read off it would be vacuous.
#
# 5120 MiB is set against the ~6 GB cap of a 12 GB device, deliberately NOT the 4 GB of an 8 GB one.
# The LLM ceiling above is the strict 8 GB line because the LLM is the guardrail product; SANA is
# measured at 3294-4773 MiB and the 8 GB case is genuinely tight, so a device run on this hardware
# cannot honestly assert it. Re-baseline downward when 8 GB hardware is available (E5, S5.2).
THRESHOLD_MAX_IMAGE_PEAK_MIB=${THRESHOLD_MAX_IMAGE_PEAK_MIB:-5120}
# The MAXIMUM across configurations, not the last one. `from_check`'s `tail -1` is right where a
# line carries one value, but the image check reports several (512 untiled, 1024 tiled) on one line
# and the ceiling must be read against the worst of them — which is 512 untiled at 4773 MiB, while
# the last is 1024 tiled at 3294. Taking the last would hide a regression in every earlier config.
img_peak=$(grep -- 'SANA image generation' "$REPORT" 2>/dev/null \
  | grep -o -- 'MLX peak [0-9]* MiB' | grep -o -- '[0-9]*' | sort -n | tail -1 || true)
if [ -n "$img_peak" ] && [ "$img_peak" -gt "$THRESHOLD_MAX_IMAGE_PEAK_MIB" ]; then
  fail_threshold "SANA MLX peak ${img_peak} MiB exceeds ${THRESHOLD_MAX_IMAGE_PEAK_MIB} MiB"
elif [ -n "$img_peak" ]; then
  echo "  SANA MLX peak ${img_peak} MiB (ceiling ${THRESHOLD_MAX_IMAGE_PEAK_MIB})"
fi

if [ "$THRESHOLD_FAILED" -ne 0 ]; then
  say "FAIL (performance regression)"
  exit 1
fi

say "PASS"
