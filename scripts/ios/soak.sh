#!/usr/bin/env bash
# Run a sustained on-device soak while capturing power on the host (E4/S4.4).
#
# The 512-token sustained check in run_smoke.sh takes ~30 s — enough to show throughput does not
# collapse immediately, not enough for a passively-cooled phone to actually get hot. Thermal
# throttling takes minutes, so a short run measures the best case and reports it as the steady
# state. This runs long enough for that to be false if it is going to be.
#
# Usage:  scripts/ios/soak.sh [--secs 300] [--device <udid>]
#
# Produces:
#   /tmp/ios-soak-report.txt      the on-device report (throughput per minute, retention, RSS)
#   /tmp/ios-soak.trace           an Instruments trace, if xctrace could attach
#
# On energy: Instruments' "Energy Log" template is GUI-only, so this uses the headless
# `Power Profiler` instrument instead. That gives CPU/GPU power counters rather than the
# mWh-per-100-tokens figure S4.4 names — enough to see whether the GPU is pinned and how thermal
# state moves, not enough to quote an energy number. Open the .trace in Instruments for that.
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT=$(pwd)

SECS=300
DEVICE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --secs)   SECS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n=== %s\n' "$1"; }

if [ -z "$DEVICE" ]; then
  DEVICE=$(xcrun devicectl list devices 2>/dev/null \
    | awk '/available \(paired\)/ {print $(NF-3); exit}')
fi
[ -n "$DEVICE" ] || { echo "no paired device" >&2; exit 1; }

BUNDLE=com.idkplay.SceneWorksSmoke
TRACE=/tmp/ios-soak.trace
REPORT=/tmp/ios-soak-report.txt

say "soaking for ${SECS}s on $DEVICE"
echo "  the device must stay UNLOCKED and plugged in for the whole run"

# The app reads this to decide whether to soak and for how long. Passed through the launch
# environment rather than baked in, so the same build serves both the quick and the long check.
# An .trace is a DIRECTORY, so `rm -f` fails on it and (under `set -e`) kills the run before it
# starts -- which is exactly what happened the first time this ran twice in a row.
rm -rf "$TRACE"
rm -f "$REPORT"

# Start the trace first so it covers the whole run. Best-effort: xctrace's device support varies by
# Xcode/iOS pairing, and a failed capture must not cost the soak measurement itself.
TRACE_PID=""
if xcrun xctrace record --template "Power Profiler" --device "$DEVICE" \
     --output "$TRACE" --time-limit "$((SECS + 60))s" --attach "$BUNDLE" >/tmp/xctrace.log 2>&1 & then
  TRACE_PID=$!
  sleep 3
  if ! kill -0 "$TRACE_PID" 2>/dev/null; then
    TRACE_PID=""
    say "xctrace did not attach (see /tmp/xctrace.log) -- continuing without a trace"
  else
    say "xctrace recording to $TRACE"
  fi
fi

# devicectl forwards variables prefixed DEVICECTL_CHILD_ from the CALLING environment into the
# launched process, stripping the prefix -- not via a flag.
say "launching with IOS_SMOKE_SOAK_SECS=$SECS"
DEVICECTL_CHILD_IOS_SMOKE_SOAK_SECS="$SECS" \
xcrun devicectl device process launch \
  --device "$DEVICE" --terminate-existing \
  "$BUNDLE" 2>&1 | tail -3

# Poll for a report written after launch. The soak itself takes SECS, so allow generous slack.
say "waiting for the report (~$((SECS / 60)) min)"
LAUNCHED_AT=$(date +%s)
DEADLINE=$((SECS + 300))
while [ $(( $(date +%s) - LAUNCHED_AT )) -lt "$DEADLINE" ]; do
  sleep 15
  rm -f "$REPORT"
  xcrun devicectl device copy from --device "$DEVICE" \
    --domain-type appDataContainer --domain-identifier "$BUNDLE" \
    --source Documents/smoke-report.txt --destination "$REPORT" >/dev/null 2>&1 || continue
  # copy from preserves the device-side mtime, so this is when the APP wrote it.
  if [ "$(stat -f%m "$REPORT" 2>/dev/null || echo 0)" -ge "$LAUNCHED_AT" ] \
     && grep -q "thermal soak" "$REPORT" 2>/dev/null \
     && ! grep -q "thermal soak -- skipped" "$REPORT" 2>/dev/null; then
    break
  fi
  rm -f "$REPORT"
done

[ -n "$TRACE_PID" ] && kill "$TRACE_PID" 2>/dev/null || true
wait 2>/dev/null || true

if [ ! -f "$REPORT" ]; then
  echo "no soak report -- was the device locked, or the app killed?" >&2
  exit 1
fi

echo
cat "$REPORT"
[ -d "$TRACE" ] && say "trace: $TRACE (open in Instruments for power detail)"

head -1 "$REPORT" | grep -q "SMOKE: PASS" || { say "FAIL"; exit 1; }
say "PASS"
