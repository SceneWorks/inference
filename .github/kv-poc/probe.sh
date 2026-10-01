#!/usr/bin/env bash
# Report-only host probe for kv-poc-campaign.yml. Launches nothing, kills nothing, and exits 0 on
# every measured outcome: the per-phase prechecks are what gate the campaign.
set -uo pipefail
# shellcheck source=.github/kv-poc/common.sh
source "$(dirname "$0")/common.sh"

row() { summary "| $1 | $2 |"; printf '%-34s %s\n' "$1" "$2"; }
code() { printf '`%s`' "$(printf '%s' "$1" | tr '\n|' ' /')"; }

chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
memsize="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
macos="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
build="$(sw_vers -buildVersion 2>/dev/null || echo unknown)"
model="$(sysctl -n hw.model 2>/dev/null || echo unknown)"
page="$(sysctl -n hw.pagesize)"
free_gib="$(available_gib)" || free_gib=0
vm="$(vm_stat)"
pages() { printf '%s\n' "$vm" | awk -v k="$1" 'index($0, k) == 1 {gsub("\\.","",$NF); print $NF+0}'; }
inactive_gib=$(( $(pages "Pages inactive") * page / 1073741824 ))
purgeable_gib=$(( $(pages "Pages purgeable") * page / 1073741824 ))
compressed_gib=$(( $(pages "Pages occupied by compressor") * page / 1073741824 ))

# NAX: MLX's own is_nax_available() is only reachable from a built MLX binary, which this probe
# deliberately does not build. HEURISTIC, mirroring the conditions that function checks: Apple
# GPU generation >= the M5 family (applegpu_g17) AND macOS >= 26.2.
chip_gen="$(printf '%s' "$chip" | sed -n 's/^Apple M\([0-9][0-9]*\).*/\1/p')"
macos_ok="$(printf '%s\n' "$macos" | awk -F. '{ if ($1 > 26 || ($1 == 26 && $2 + 0 >= 2)) print 1; else print 0 }')"
if [ -n "$chip_gen" ] && [ "$chip_gen" -ge 5 ] && [ "$macos_ok" = 1 ]; then
  nax="likely available (HEURISTIC: M${chip_gen} + macOS ${macos} >= 26.2)"
else
  nax="likely UNAVAILABLE (HEURISTIC: chip gen '${chip_gen:-?}', macOS ${macos}; needs >= M5 and >= 26.2)"
fi

lms_ok=1; lms_idle || lms_ok=0
busy="$(busy_processes)"
listeners="$(runner_listeners)"
listener_count="$(printf '%s' "$listeners" | grep -c . || true)"
py312="$(command -v python3.12 2>/dev/null && python3.12 --version 2>&1 || echo MISSING)"
first_line() { local out; out="$("$@" 2>/dev/null)" || { echo MISSING; return; }; printf '%s\n' "$out" | sed -n 1p; }
xcode="$(first_line xcodebuild -version)"
rustup_v="$(first_line rustup --version)"
zstd_v="$(first_line zstd --version)"
hf_v="$(command -v hf 2>/dev/null || echo "not on PATH (build installs the hash-locked huggingface_hub)")"
therm="$(pmset -g therm 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')"
if [ -d "$KV_HF_HUB" ]; then
  if [ -w "$KV_HF_HUB" ]; then hub_state="present, writable"; else hub_state="present, NOT writable by $(id -un)"; fi
else
  hub_state="MISSING"
fi

summary "## Host probe: ${RUNNER_NAME:-unknown runner}"
summary ""
summary "| check | value |"
summary "|---|---|"
row "runner" "$(code "${RUNNER_NAME:-?}") ($(id -un))"
row "chip" "$(code "$chip") ($(code "$model"))"
row "hw.memsize" "$(( memsize / 1073741824 )) GiB ($memsize bytes)"
row "macOS" "$(code "$macos ($build)")"
row "NAX" "$nax"
row "available RAM" "${free_gib} GiB (free + speculative + purgeable + file cache; W1 precheck needs >= 84, W2 >= its policy's cap + reserve)"
row "inactive / purgeable / compressed" "${inactive_gib} / ${purgeable_gib} / ${compressed_gib} GiB"
row "LM Studio" "$(code "$LMS_STATE")"
row "GPU/build processes" "$(code "${busy:-none}")"
row "Runner.Listener processes" "${listener_count} $(code "${listeners:-none}")"
row "python3.12" "$(code "$py312")"
row "xcodebuild" "$(code "$xcode")"
row "rustup" "$(code "$rustup_v")"
row "zstd (prebuilt MLX)" "$(code "$zstd_v")"
row "hf CLI" "$(code "$hf_v")"
row "thermal" "$(code "${therm:-unknown}")"
row "HF hub $KV_HF_HUB" "$hub_state"
for d in "$HOME" /Volumes/Models; do
  if [ -d "$d" ]; then row "disk free $d" "$(df -h "$d" | awk 'NR==2 {print $4 " free of " $2 " (" $5 " used)"}')"; fi
done

for snap in "$LQ" "$LB" "$QQ" "$QB"; do
  name="$(printf '%s' "$snap" | sed -n 's|.*/models--\([^/]*\)/snapshots/\(.......\).*|\1@\2|p')"
  if [ -d "$snap" ]; then
    size="$(du -shL "$snap" 2>/dev/null | awk '{print $1}')"
    row "model $name" "present ($size)"
  else
    row "model $name" "absent (the build job downloads it)"
  fi
done

row "campaign root $KV_ROOT" "$( [ -d "$KV_ROOT" ] && ls "$KV_ROOT" | tr '\n' ' ' || echo absent)"
if [ -f "$F/SHA256SUMS" ]; then
  if (cd "$F" && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1); then row "frozen dir" "sealed, verifies"; else row "frozen dir" "sealed but DOES NOT verify"; fi
else
  row "frozen dir" "not built"
fi
if [ -d "$R" ]; then row "runs dir $R" "$(code "$(ls "$R" "$R/evidence" 2>/dev/null | tr '\n' ' ')")"; else row "runs dir" "none yet"; fi

need_gib=84
if [ "${KV_MODE:-}" = w2 ]; then
  need_gib="$(policy_need_gib "$KV_DIR/policies/$W2_POLICY")" || need_gib=80
  if [ -f "$F2/SHA256SUMS" ]; then
    if (cd "$F2" && shasum -a 256 -c SHA256SUMS >/dev/null 2>&1); then row "W2 frozen dir" "sealed, verifies"; else row "W2 frozen dir" "sealed but DOES NOT verify"; fi
  else
    row "W2 frozen dir" "not built"
  fi
  for route in wan_vace wan_vace_fun; do
    if problem="$(verify_w2_vace "$route")"; then row "W2 assembled $route" "verifies"; else row "W2 assembled $route" "$(code "$problem") (the w2-assets job assembles it)"; fi
  done
  row "W2 precheck RAM" "cap + reserve of $W2_POLICY = ${need_gib} GiB"
  summary ""
  summary "#### W2 pinned assets (\`models-w2.tsv\`) vs free disk (report only; the w2-assets job fails on a shortfall)"
  summary '```'
  summary "$(python3.12 "$KV_DIR/models.py" --report --hub "$KV_HF_HUB" --pins "$KV_DIR/models-w2.tsv" --reserve-gib "${KV_W2_DISK_RESERVE_GIB:-10}" 2>&1 | head -120)"
  summary '```'
fi

go="GO"
[ "$free_gib" -ge "$need_gib" ] || go="NO-GO (RAM)"
[ -z "$busy" ] || go="NO-GO (processes)"
[ "$lms_ok" = 1 ] || go="NO-GO (LM Studio)"
summary ""
summary "Precheck as of now: **$go** (report only; each campaign job re-checks before launching)."
summary ""
summary "Safe stop for a running campaign job (it exits 75 after the in-flight row): \`$(stop_command)\`"
echo "precheck as of now: $go"
exit 0
