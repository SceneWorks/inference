#!/usr/bin/env bash
# Build, install, and run the on-device MLX smoke test on a connected iPhone.
#
# Answers the two questions no build can settle (docs/ios-epics.md, E3):
#   S3.3 -- does the bundled metallib resolve inside the app sandbox?
#   R11  -- are the cross-compiled Metal kernels numerically correct?
#
# Usage:  scripts/ios/run_smoke.sh [--debug] [--device <udid>]
#
# Requires: a paired device with Developer Mode on, xcodegen, and a signing identity.
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$(pwd)

PROFILE=release
CARGO_PROFILE_FLAG=--release
DEVICE=""
CARGO_FEATURES=""
while [ $# -gt 0 ]; do
  case "$1" in
    --debug)  PROFILE=debug; CARGO_PROFILE_FLAG=""; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    # E5: also run the SANA image-generation check. Needs a SANA snapshot pushed to
    # Documents/<any-subdir>/ (scripts/ios/push_model.sh <dir> sana); without one the check
    # reports "skipped" rather than failing.
    --media)  CARGO_FEATURES="--features media"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n=== %s\n' "$1"; }

# --- device -----------------------------------------------------------------------------------
if [ -z "$DEVICE" ]; then
  # First device listed as available+paired. Explicit --device when more than one is attached.
  #
  # Matched by UUID SHAPE, not by column position. The previous `$(NF-3)` counted back from the end
  # of the line, which put it inside the Model column as soon as that column had a different word
  # count -- "iPhone 17 Pro Max (iPhone18,2)" yields NF=11 and `$(NF-3)` is the literal "17". Every
  # subsequent devicectl call then failed with "The specified device was not found. (Name: 17)".
  DEVICE=$(xcrun devicectl list devices 2>/dev/null \
    | awk '/available \(paired\)/ {print; exit}' \
    | grep -oE '[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}' \
    | head -1)
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

# The memory entitlements, on by default. Set SMOKE_ENTITLEMENTS= (empty) to build the unentitled
# control: the report prints `os_proc_available_memory`, so the pair of runs measures what the
# entitlement is worth on this device instead of assuming it is worth something.
SMOKE_ENTITLEMENTS=${SMOKE_ENTITLEMENTS-App/SceneWorksSmoke.entitlements}
export SMOKE_ENTITLEMENTS
if [ -n "$SMOKE_ENTITLEMENTS" ]; then
  say "entitlements $SMOKE_ENTITLEMENTS"
else
  say "entitlements NONE (unentitled control build)"
fi

# --- rust staticlib ---------------------------------------------------------------------------
# No IPHONEOS_DEPLOYMENT_TARGET here on purpose: .cargo/config.toml pins it, and this script is
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
METAL_TRIPLE=$(strings "$METALLIB" | grep -oE 'apple-(ios|macos)[0-9.]*' | sort -u | tr '\n' ' ')
case "$METAL_TRIPLE" in
  *apple-macos*) echo "$METALLIB carries macOS kernels -- the cross-compile regressed" >&2; exit 1 ;;
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
  build 2>&1 | tail -20

APP=$(find "$DERIVED/Build/Products" -name "SceneWorksSmoke.app" -print -quit)
test -n "$APP" || { echo "no .app was produced" >&2; exit 1; }

# The whole point of the packaging step -- assert it landed before installing.
test -f "$APP/mlx.metallib" \
  || { echo "mlx.metallib is missing from the bundle; the app would fail at first Metal use" >&2; exit 1; }
say "bundled metallib: $(du -h "$APP/mlx.metallib" | cut -f1)"

say "installing"
INSTALL_LOG=$(xcrun devicectl device install app --device "$DEVICE" "$APP" 2>&1) || true
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
  echo "  xcrun devicectl device uninstall app --device $DEVICE com.idkplay.SceneWorksSmoke" >&2
  echo >&2
  echo "NOTE: that DELETES the app's Documents container, including every pushed snapshot." >&2
  echo "Re-provision afterwards with scripts/ios/push_model.sh (see docs/ios-epics.md E5)." >&2
  exit 1
fi

# The device must be UNLOCKED to launch an app: SpringBoard denies the request with
# FBSOpenApplicationErrorDomain 7 otherwise, after a successful build and install. Say so plainly
# rather than letting a 40-line CoreDevice error explain it.
say "launching (unlock the device if prompted; console output follows)"
xcrun devicectl device process launch \
  --device "$DEVICE" --terminate-existing \
  com.idkplay.SceneWorksSmoke 2>&1 | tail -5

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
if [ -n "$CARGO_FEATURES" ]; then POLL_TRIES=${POLL_TRIES_MEDIA:-200}; fi

say "waiting for the on-device report (up to $((POLL_TRIES * 3))s)"
REPORT=/tmp/ios-smoke-report.txt
rm -f "$REPORT"
LAUNCHED_AT=$(date +%s)
for _ in $(seq 1 "$POLL_TRIES"); do
  sleep 3
  rm -f "$REPORT"
  xcrun devicectl device copy from --device "$DEVICE" \
    --domain-type appDataContainer --domain-identifier com.idkplay.SceneWorksSmoke \
    --source Documents/smoke-report.txt --destination "$REPORT" >/dev/null 2>&1 || continue
  # `copy from` preserves the device-side mtime, so this compares when the APP wrote it.
  [ "$(stat -f%m "$REPORT" 2>/dev/null || echo 0)" -ge "$LAUNCHED_AT" ] && break
  rm -f "$REPORT"
done

if [ ! -f "$REPORT" ]; then
  echo "no report was produced; is the device unlocked?" >&2
  # A jetsam kill and a launch failure look identical from here: in both cases nothing was written.
  # The image check leaves a per-configuration breadcrumb precisely so the two can be told apart —
  # if it exists, the app ran and died partway rather than never starting.
  BREADCRUMB=/tmp/ios-sana-progress.txt
  rm -f "$BREADCRUMB"
  if xcrun devicectl device copy from --device "$DEVICE" \
      --domain-type appDataContainer --domain-identifier com.idkplay.SceneWorksSmoke \
      --source Documents/sana-progress.txt --destination "$BREADCRUMB" >/dev/null 2>&1; then
    echo >&2
    echo "the app DID run -- image-generation breadcrumbs found, so this is a mid-run death" >&2
    echo "(most likely jetsam: no increased-memory-limit entitlement, see E5/S4.7):" >&2
    sed 's/^/  /' "$BREADCRUMB" >&2
    echo >&2
    echo "the configuration AFTER the last line above is the one that exceeded the cap" >&2
  fi
  exit 1
fi

echo
cat "$REPORT"

if ! head -1 "$REPORT" | grep -q "SMOKE: PASS"; then
  say "FAIL"
  exit 1
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
