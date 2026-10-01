# shellcheck shell=bash disable=SC2034  # every variable here is consumed by the sourcing script
# Shared paths and helpers for .github/workflows/kv-poc-campaign.yml (epic 20669, W1 + W2).
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
#   $KV_ROOT/<inference_sha>/frozen       W1 bins + mlx.metallib + policies + prompt, SHA256SUMS, a-w
#   $KV_ROOT/<inference_sha>/frozen-w2    W2 bins (sc20686_wan, sc20686_flux2_edit, krea-integration)
#                                         + mlx.metallib + media policy + reference images, sealed alike
#   $KV_ROOT/<inference_sha>-runs         resume dirs + evidence (a re-dispatch resumes here)
#   $KV_ROOT/cargo-target                 persistent CARGO_TARGET_DIR
#   $KV_ROOT/tools                        hash-locked huggingface_hub install
#   $KV_ROOT/w2-inputs                    W2 assembled VACE snapshots (per-file links into the hub
#                                         cache) + the generated VACE control/mask fixtures

: "${INFERENCE_SHA:?INFERENCE_SHA is required}"
: "${SCENEWORKS_SHA:?SCENEWORKS_SHA is required}"

# Host affinity. Each job reuses what the previous one left in THIS host's $KV_ROOT, so a run whose
# jobs split across the two Macs fails later with a missing tree or frozen dir on the other box
# (run 36816509385). config.sh resolves the one runner the run must stay on; refuse anything else
# before touching state. Unset outside the workflow (local tests source this file).
if [ -n "${KV_EXPECTED_RUNNER:-}" ] && [ "${RUNNER_NAME:-}" != "$KV_EXPECTED_RUNNER" ]; then
  echo "::error title=wrong runner::this job landed on '${RUNNER_NAME:-?}' ($(hostname)), but the campaign's state lives on '$KV_EXPECTED_RUNNER'; its runs-on labels no longer select that runner alone (check the org runner labels, kv-poc/config.sh)"
  exit 1
fi

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
F2="$KV_ROOT/$INFERENCE_SHA/frozen-w2"
W2_INPUTS="$KV_ROOT/w2-inputs"
W2_BINS="sc20686_wan sc20686_flux2_edit krea-integration"

# The four pinned snapshots the W1 commands name (hub-cache layout on both Macs).
LQ="$KV_HF_HUB/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"
LB="$KV_HF_HUB/models--mlx-community--Llama-3.2-3B-Instruct-bf16/snapshots/6d88ba43024fef71b10e52e101c7cd4598322601"
QQ="$KV_HF_HUB/models--mlx-community--Qwen3-1.7B-4bit/snapshots/3b1b1768f8f8cf8351c712464f906e86c2b8269e"
QB="$KV_HF_HUB/models--mlx-community--Qwen3-1.7B-bf16/snapshots/9cd6692855d3e06772228e9a962b2606359b2d24"

# The W2 (media) snapshots, pinned file-by-file in models-w2.tsv. The q4 tier roots are passed as
# is; the two VACE routes take the worker-assembled layout under $W2_INPUTS (build.sh w2-inputs).
hub_snapshot() { printf '%s/models--%s/snapshots/%s' "$KV_HF_HUB" "$(printf '%s' "$1" | sed 's|/|--|g')" "$2"; }
W2_KREA="$(hub_snapshot SceneWorks/krea-realtime-14b-mlx e68e9a3d98187fdf6936838ffcf6df5aa48d6626)/q4"
W2_FLUX="$(hub_snapshot SceneWorks/flux2-klein-9b-mlx 1902693279fcfb828919370dfac2b8922d99499a)/q4"
W2_FLUX_KV="$(hub_snapshot SceneWorks/flux2-klein-9b-kv-mlx bbf22de8d654789de3b177632d2e283cc4f77729)/q4"
W2_TI2V="$(hub_snapshot SceneWorks/wan2.2-ti2v-5b-mlx bb1b055249614cf9d7cf4373fbdbc184b77dee88)/q4"
W2_T2V="$(hub_snapshot SceneWorks/wan2.2-t2v-a14b-mlx 991eb255c544bbb2e1f1e07da4355c2f0a5337b7)/q4"
W2_I2V="$(hub_snapshot SceneWorks/wan2.2-i2v-a14b-mlx c6c786170031eccc3a1fac0f98f1ad4ff988271e)/q4"
W2_LIGHTNING="$(hub_snapshot lightx2v/Wan2.2-Lightning 18bccf8884ec0a078eed79785eb4ef13ea16ce1e)"
W2_VACE_REVISION=ec4d2cb062b548996b179d493fdd05340de702a1
W2_VACE_SRC="$(hub_snapshot Wan-AI/Wan2.1-VACE-1.3B-diffusers "$W2_VACE_REVISION")"
W2_VACE_FUN_REVISION=1abfb95801b7bd8f952083ebf80b93448ddb0ce4
W2_VACE_FUN_SRC="$(hub_snapshot linoyts/Wan2.2-VACE-Fun-14B-diffusers "$W2_VACE_FUN_REVISION")"
W2_VACE="$W2_INPUTS/assembled/wan_vace"
W2_VACE_FUN="$W2_INPUTS/assembled/wan_vace_fun"
W2_FIXTURES="$W2_INPUTS/fixtures/sc20686"
W2_POLICY=media-64.json
W2_KREA_OBSERVER="generate_smoke::sc20684_packed_campaign_observer"
# The two reference images the SC-20686 coordinates read (FLUX --reference/--reference2, Wan I2V
# --image, VACE --reference), copied from the frozen inference tree into $F2/inputs: path:sha256.
W2_REFERENCES="crates/media/mlx-gen/_vendor/mage_flow/assets/dog.jpg:164d8dfe707fb854e288ad2eea65c2db87e90af11f689c85502860eeaf3f4794
docs/migration/evidence/sc-16956/pulid-reference.png:3995f2e856346748588e76a5557516d0218f44f5663701c5a50139d50c86a7be"

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
#   R = max(0, file-backed - speculative)   (darwin-vm-stat-available-v3)
# Activity Monitor's "Cached Files" model: all file-backed page cache counts, anonymous pages
# never do. Known limitation: file pages another process has mapped count as available; the
# campaign child's own mapped weights are bounded by its phys_footprint cap.
# Reads vm_stat on stdin; prints "available page free speculative purgeable inactive file-backed
# anonymous throttled active R" (bytes, then page size, then pages; inactive, anonymous,
# throttled and active are audit-only) or prints nothing and fails when the banner or any counter
# is missing, duplicated or malformed, or available reaches 2^63.
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
          && key != "Anonymous pages" && key != "Pages throttled" && key != "Pages active") next
      val = line; sub(/^[^:]*:[ \t]*/, "", val); sub(/[ \t]+$/, "", val)
      if (val !~ /^[0-9]+\.$/ || (key in v)) bad = 1
      sub(/\.$/, "", val); v[key] = val + 0; n++
    }
    END {
      p = page; while (p > 1 && p % 2 == 0) p /= 2
      if (bad || n != 8 || page < 4096 || p != 1) exit 1
      r = v["File-backed pages"] - v["Pages speculative"]; if (r < 0) r = 0
      avail = (v["Pages free"] + v["Pages speculative"] + v["Pages purgeable"] + r) * page
      if (avail >= 9223372036854775808) exit 1
      printf "%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n", avail, page, \
        v["Pages free"], v["Pages speculative"], v["Pages purgeable"], v["Pages inactive"], \
        v["File-backed pages"], v["Anonymous pages"], v["Pages throttled"], v["Pages active"], r
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

verify_frozen() { # [dir] (default: the W1 frozen dir)
  local dir="${1:-$F}"
  [ -f "$dir/SHA256SUMS" ] || { echo "::error title=frozen dir not sealed::$dir/SHA256SUMS is missing (run the build job)"; return 1; }
  (cd "$dir" && shasum -a 256 -c SHA256SUMS >/dev/null) \
    || { echo "::error title=frozen dir mismatch::$dir does not match its SHA256SUMS"; return 1; }
}

# Cap + reserve of a darwin-mlx media safety policy, in whole GiB (the parents admit each arm only
# when available RAM covers both).
policy_need_gib() {
  python3.12 -c 'import json, sys; p = json.load(open(sys.argv[1])); print(-(-(p["childFootprintCapBytes"] + p["hostFreeReserveBytes"]) // 1073741824))' "$1"
}

# The worker-assembled VACE snapshot layout (mlx_gen_wan::convert::assemble_wan_vace[_fun]_snapshot,
# sceneworks-worker video_jobs/vace.rs): each transformer dir of the VACE repo plus the base-Wan
# T2V-A14B q4 tier's UMT5, z16 VAE and tokenizer, as REAL directories of per-file symlinks into the
# hub cache (the adapter's identity walk would also follow directory links, but per-file links keep
# the tree exactly the pinned file set), plus a .snapshot-revision holding the VACE revision.
# Prints "<relative path> <link target>" for every entry of <route> (wan_vace | wan_vace_fun).
w2_vace_layout() {
  local route="$1" repo src rev
  case "$route" in
    wan_vace) repo=Wan-AI/Wan2.1-VACE-1.3B-diffusers; src="$W2_VACE_SRC"; rev="$W2_VACE_REVISION" ;;
    wan_vace_fun) repo=linoyts/Wan2.2-VACE-Fun-14B-diffusers; src="$W2_VACE_FUN_SRC"; rev="$W2_VACE_FUN_REVISION" ;;
    *) return 2 ;;
  esac
  awk -F '\t' -v repo="$repo" -v rev="$rev" '$1 == repo && $2 == rev && $3 ~ /^transformer(_2)?\// { print $3 }' \
    "$KV_DIR/models-w2.tsv" | while read -r rel; do printf '%s %s\n' "$rel" "$src/$rel"; done
  for name in t5_encoder.safetensors vae.safetensors tokenizer.json; do printf '%s %s\n' "$name" "$W2_T2V/$name"; done
}

# Verify an assembled VACE snapshot: exactly the layout's links, each resolving to its hub file,
# and the .snapshot-revision marker. Prints the problems; returns 1 when there are any.
verify_w2_vace() {
  local route="$1" dir rev rel target expected=0 found
  case "$route" in
    wan_vace) dir="$W2_VACE"; rev="$W2_VACE_REVISION" ;;
    wan_vace_fun) dir="$W2_VACE_FUN"; rev="$W2_VACE_FUN_REVISION" ;;
    *) return 2 ;;
  esac
  [ -d "$dir" ] || { echo "$dir is missing"; return 1; }
  [ "$(cat "$dir/.snapshot-revision" 2>/dev/null)" = "$rev" ] || { echo "$dir/.snapshot-revision is not $rev"; return 1; }
  while read -r rel target; do
    expected=$((expected + 1))
    [ -L "$dir/$rel" ] && [ -f "$dir/$rel" ] && [ "$(realpath "$dir/$rel")" = "$(realpath "$target")" ] \
      || { echo "$dir/$rel does not link to $target"; return 1; }
  done < <(w2_vace_layout "$route")
  found="$(find "$dir" ! -type d ! -path "$dir/.snapshot-revision" | wc -l | tr -d ' ')"
  [ "$found" = "$expected" ] || { echo "$dir holds $found entries, the layout has $expected links"; return 1; }
}

