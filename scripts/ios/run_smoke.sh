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
while [ $# -gt 0 ]; do
  case "$1" in
    --debug)  PROFILE=debug; CARGO_PROFILE_FLAG=""; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n=== %s\n' "$1"; }

# --- device -----------------------------------------------------------------------------------
if [ -z "$DEVICE" ]; then
  # First device listed as available+paired. Explicit --device when more than one is attached.
  DEVICE=$(xcrun devicectl list devices 2>/dev/null \
    | awk '/available \(paired\)/ {print $(NF-3); exit}')
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

# --- rust staticlib ---------------------------------------------------------------------------
# No IPHONEOS_DEPLOYMENT_TARGET here on purpose: .cargo/config.toml pins it, and this script is
# also a check that a clean invocation needs no environment help.
# ios-host/smoke is excluded from the workspace, so it has its OWN target directory rather than
# the root one. The metallib it builds lives there too.
SMOKE_TARGET="$ROOT/ios-host/smoke/target"
LIB="$SMOKE_TARGET/aarch64-apple-ios/$PROFILE/libios_smoke.a"

say "building ios-smoke for aarch64-apple-ios ($PROFILE)"
( cd "$ROOT/ios-host/smoke" && cargo build --target aarch64-apple-ios $CARGO_PROFILE_FLAG )

# Guard against a stale staticlib. Xcode links whatever .a is on disk, so a Rust source edit that
# did not get rebuilt produces a green run reporting the PREVIOUS build's results -- which looks
# like a passing test rather than a build mistake, and cost three confusing runs before it was
# caught. Compare mtimes and fail loudly instead.
NEWEST_SRC=$(find "$ROOT/ios-host/smoke/src" "$ROOT/crates" -name '*.rs' -newer "$LIB" -print -quit 2>/dev/null || true)
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
xcrun devicectl device install app --device "$DEVICE" "$APP" 2>&1 | tail -5

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
say "waiting for the on-device report"
REPORT=/tmp/ios-smoke-report.txt
rm -f "$REPORT"
LAUNCHED_AT=$(date +%s)
for _ in $(seq 1 60); do
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
  exit 1
fi

echo
cat "$REPORT"

if head -1 "$REPORT" | grep -q "SMOKE: PASS"; then
  say "PASS"
else
  say "FAIL"
  exit 1
fi
