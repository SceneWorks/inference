# shellcheck shell=bash disable=SC2034  # every variable here is consumed by the sourcing script
# Shared paths and helpers for .github/workflows/kv-poc-campaign.yml (epic 20669, W1).
#
# Sourced by every macOS job. The runners' /bin/bash is 3.2: no mapfile, no associative arrays,
# no ${x,,}, and an EMPTY array expanded under `set -u` is an error -- keep it that way.
#
# Everything the campaign keeps lives under $KV_ROOT (default $HOME/kv-poc), OUTSIDE the Actions
# workspace, because actions/checkout wipes the workspace between jobs and the campaign binaries
# bind their inference tree at COMPILE time (CARGO_MANIFEST_DIR) and read `git rev-parse HEAD` of
# it, and of $SCENEWORKS_ROOT, at RUN time. Both trees must therefore sit at fixed paths, at the
# exact frozen SHAs, clean, for as long as any phase of the campaign can still run or resume.
#
#   $KV_ROOT/<inference_sha>/inference    stable inference clone (the bins' CARGO_MANIFEST_DIR)
#   $KV_ROOT/<inference_sha>/SceneWorks   SCENEWORKS_ROOT
#   $KV_ROOT/<inference_sha>/frozen       bins + mlx.metallib + policies + prompt, SHA256SUMS, a-w
#   $KV_ROOT/<inference_sha>-runs         resume dirs + evidence (a re-dispatch resumes here)
#   $KV_ROOT/cargo-target                 persistent CARGO_TARGET_DIR
#   $KV_ROOT/tools                        hash-locked huggingface_hub install

: "${INFERENCE_SHA:?INFERENCE_SHA is required}"
: "${SCENEWORKS_SHA:?SCENEWORKS_SHA is required}"

KV_ROOT="${KV_POC_ROOT:-$HOME/kv-poc}"
KV_HF_HUB="${KV_POC_HF_HUB:-/Volumes/Models/huggingface/hub}"
INF="$KV_ROOT/$INFERENCE_SHA/inference"
SW="$KV_ROOT/$INFERENCE_SHA/SceneWorks"
F="$KV_ROOT/$INFERENCE_SHA/frozen"
R="$KV_ROOT/$INFERENCE_SHA-runs"
KV_TARGET="$KV_ROOT/cargo-target"
KV_TOOLS="$KV_ROOT/tools"
INFERENCE_URL="https://github.com/SceneWorks/inference"
SCENEWORKS_URL="https://github.com/SceneWorks/SceneWorks"
BINS="sc20671_kv_baseline sc20676_packed_evidence sc20677_capture_kv sc20677_kv_candidates"

# The four pinned snapshots the W1 commands name (hub-cache layout on both Macs).
LQ="$KV_HF_HUB/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"
LB="$KV_HF_HUB/models--mlx-community--Llama-3.2-3B-Instruct-bf16/snapshots/6d88ba43024fef71b10e52e101c7cd4598322601"
QQ="$KV_HF_HUB/models--mlx-community--Qwen3-1.7B-4bit/snapshots/3b1b1768f8f8cf8351c712464f906e86c2b8269e"
QB="$KV_HF_HUB/models--mlx-community--Qwen3-1.7B-bf16/snapshots/9cd6692855d3e06772228e9a962b2606359b2d24"

KV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

summary() { if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then printf '%s\n' "$*" >> "$GITHUB_STEP_SUMMARY"; fi; }
output() { if [ -n "${GITHUB_OUTPUT:-}" ]; then printf '%s=%s\n' "$1" "$2" >> "$GITHUB_OUTPUT"; fi; }

stop_ref() { printf 'refs/heads/kv-poc-stop/%s' "$GITHUB_RUN_ID"; }
stop_command() {
  printf 'gh api -X POST repos/%s/git/refs -f ref=%s -f sha=%s' \
    "${GITHUB_REPOSITORY:-SceneWorks/inference}" "$(stop_ref)" "$INFERENCE_SHA"
}
stop_branch_present() {
  git ls-remote --exit-code "https://github.com/${GITHUB_REPOSITORY:-SceneWorks/inference}" "$(stop_ref)" >/dev/null 2>&1
}

# The macOS host-memory ADMISSION measure, identical to the campaign parents' pre-spawn check
# (crates/llm/mlx-llm campaign_supervisor::HostMemory, scripts/media_campaign_supervisor.py
# darwin_host_memory; all three are pinned by crates/llm/mlx-llm/testdata/
# darwin-host-memory-cases.json):
#   available = (free + speculative + purgeable + R) * page size   (must stay below 2^63)
#   R = min(inactive - purgeable, file-backed - speculative, inactive + throttled - anonymous),
#       each floored at zero: a provable lower bound on inactive file-backed pages, because
#       File-backed + Anonymous = active + inactive + speculative + throttled and throttled
#       pages are anonymous (darwin-vm-stat-available-v2).
# Reads vm_stat on stdin; prints "available page free speculative purgeable inactive file-backed
# anonymous throttled R" (bytes, then page size, then pages) or prints nothing and fails when the
# banner or any counter is missing, duplicated or malformed, or available reaches 2^63.
host_memory_from_vm_stat() {
  awk '
    NR == 1 {
      if (match($0, /page size of [0-9]+ bytes/)) page = substr($0, RSTART + 13, RLENGTH - 19) + 0
      next
    }
    {
      line = $0; sub(/^[ \t]+/, "", line); key = line; sub(/:.*/, "", key)
      if (key != "Pages free" && key != "Pages speculative" && key != "Pages purgeable" \
          && key != "Pages inactive" && key != "File-backed pages" \
          && key != "Anonymous pages" && key != "Pages throttled") next
      val = line; sub(/^[^:]*:[ \t]*/, "", val); sub(/[ \t]+$/, "", val)
      if (val !~ /^[0-9]+\.$/ || (key in v)) bad = 1
      sub(/\.$/, "", val); v[key] = val + 0; n++
    }
    END {
      p = page; while (p > 1 && p % 2 == 0) p /= 2
      if (bad || n != 7 || page < 4096 || p != 1) exit 1
      i = v["Pages inactive"] - v["Pages purgeable"]; if (i < 0) i = 0
      f = v["File-backed pages"] - v["Pages speculative"]; if (f < 0) f = 0
      a = v["Pages inactive"] + v["Pages throttled"] - v["Anonymous pages"]; if (a < 0) a = 0
      r = (i < f) ? i : f; r = (a < r) ? a : r
      avail = (v["Pages free"] + v["Pages speculative"] + v["Pages purgeable"] + r) * page
      if (avail >= 9223372036854775808) exit 1
      printf "%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n", avail, page, \
        v["Pages free"], v["Pages speculative"], v["Pages purgeable"], v["Pages inactive"], \
        v["File-backed pages"], v["Anonymous pages"], v["Pages throttled"], r
    }'
}

# Admission-available host RAM in whole GiB (floored, so ">= 84" matches the parents' byte
# comparison against 84 GiB exactly); fails when vm_stat cannot be measured.
available_gib() {
  local measured
  measured="$(vm_stat | host_memory_from_vm_stat)" && [ -n "$measured" ] || return 1
  echo $(( ${measured%% *} / 1073741824 ))
}

lms_bin() {
  if command -v lms >/dev/null 2>&1; then command -v lms
  elif [ -x "$HOME/.lmstudio/bin/lms" ]; then echo "$HOME/.lmstudio/bin/lms"
  fi
}

# Returns 0 when LM Studio holds no loaded model, 1 otherwise; sets LMS_STATE for the log.
# `lms ps` WAKES the LM Studio service when it is not running (it prints "Waking up LM Studio
# service..." before the JSON -- the first probe on nax-macos-2 did exactly that), and a service
# that is not running holds no model, so it is only asked when one of its processes exists.
lms_idle() {
  local lms out json n
  lms="$(lms_bin)"
  if [ -z "$lms" ]; then LMS_STATE="lms not installed"; return 0; fi
  # A here-string, not a pipe: `grep -q` exiting early would SIGPIPE `ps`, and under pipefail
  # that reads as "not running".
  if ! grep -qiE 'lm studio|lmstudio|llmster' <<< "$(ps -axo comm=)"; then LMS_STATE="LM Studio not running"; return 0; fi
  out="$("$lms" ps --json 2>&1)" || { LMS_STATE="lms ps --json failed: $out"; return 1; }
  json="$(printf '%s\n' "$out" | sed -n '/^[[:space:]]*\[/,$p')"
  n="$(printf '%s' "$json" | python3.12 -c 'import json, sys; print(len(json.load(sys.stdin)))' 2>/dev/null)" \
    || { LMS_STATE="unparseable lms ps --json output: $out"; return 1; }
  if [ "$n" = 0 ]; then LMS_STATE="running, no model loaded"; return 0; fi
  LMS_STATE="$n model(s) loaded: $(printf '%s' "$json" | tr '\n' ' ' | cut -c1-200)"
  return 1
}

# Processes that would share the GPU or the unified memory with a campaign row: the W1-PRECHECK.sh
# set, minus its Runner.Listener/Runner.Worker check (on this box the runner IS the launcher, and
# it runs one job at a time).
#
# NOT `pgrep -f`: on macOS it matches against, and `-l` prints, the process ENVIRONMENT as well as
# argv. Every process whose PATH holds ~/.cargo/bin then matches `cargo` (this job's own shell
# included, so the check could never pass), and the output leaks other processes' env -- tokens
# included -- into the job log. `ps -o comm` / `-o args` read the executable and argv only.
busy_processes() {
  ps -axo pid=,comm= | awk '{ pid = $1; sub(/^ *[0-9]+ +/, ""); sub(/.*\//, "");
    if ($0 ~ /^(cargo|rustc|sc2067.*|sc2068.*|krea-integration.*)$/) print pid " " $0 }'
  # A Python interpreter (argv[0]) with MLX anywhere in its argv, e.g. `python -m mlx_lm ...`. Keyed
  # on argv[0] so a shell whose command line merely mentions python and mlx is not a match. Prints
  # the pid and interpreter only; argv can carry things that do not belong in a job log.
  ps -axww -o pid=,args= | awk '{ n = split($2, a, "/"); if (a[n] ~ /^[Pp]ython/ && $0 ~ /mlx/) print $1 " " a[n] " (argv mentions mlx)" }'
}

runner_listeners() {
  ps -axo pid=,comm= | awk '/[R]unner\.Listener$/ { print }'
}

# Verify one stable clone is exactly <sha>, clean, with the origin the campaign bins demand.
verify_tree() {
  local dir="$1" url="$2" sha="$3" head
  [ -d "$dir/.git" ] || { echo "::error title=missing tree::$dir is not a git checkout (run the build job)"; return 1; }
  [ "$(git -C "$dir" config --get remote.origin.url)" = "$url" ] \
    || { echo "::error title=wrong origin::$dir origin is not $url"; return 1; }
  head="$(git -C "$dir" rev-parse HEAD)"
  [ "$head" = "$sha" ] || { echo "::error title=wrong revision::$dir HEAD is $head, expected $sha"; return 1; }
  [ -z "$(git -C "$dir" status --porcelain)" ] || { echo "::error title=dirty tree::$dir has local changes"; git -C "$dir" status --short | head -20; return 1; }
}

verify_frozen() {
  [ -f "$F/SHA256SUMS" ] || { echo "::error title=frozen dir not sealed::$F/SHA256SUMS is missing (run the build job)"; return 1; }
  (cd "$F" && shasum -a 256 -c SHA256SUMS >/dev/null) \
    || { echo "::error title=frozen dir mismatch::$F does not match its SHA256SUMS"; return 1; }
}
