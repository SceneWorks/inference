#!/usr/bin/env bash
# Report-only record of what the box serving this job has free, and which of the job's
# snapshots are already resident on it.
#
# WHY THIS EXISTS (sc-17271, sc-17283). The macOS real-weight pool is two boxes that cannot
# see each other -- `nax-macos-2` is neither SSH-reachable nor resolvable from `nax-macos` --
# and `gh api /orgs/SceneWorks/actions/runners` is 403 without `admin:org`. So nobody has a
# shell on either machine, and a lane's disk cost is knowable only by measuring it there:
# `release/real-weight-models.toml` pins revisions but records `size_class`, never bytes.
#
# A snapshot variable is `~/`-relative (see `scripts/release/resolve_snapshot_paths.py`), so
# it names the same location on both Macs and the cost lands on whichever box claimed the
# job. `RUNNER_NAME` and `$HOME` are printed alongside the figures purely so the reader of a
# log knows which box these numbers describe without leaving the log. That is a convenience,
# NOT a new capability: `runner_name` is already a field on the repo-scoped jobs API
# (`gh api /repos/SceneWorks/inference/actions/runs/<id>/jobs -q '.jobs[].runner_name'`),
# needs only the `repo` scope, and works retroactively for jobs that ran before this step
# existed.
#
# THIS DOES NOT ANSWER WHICH LABELS A RUNNER CARRIES. It reports which box *won* a job, and
# a box that merely loses every race is indistinguishable here from one that dropped the
# label. sc-17271's H1 -- "has `nax-macos` really given up `rw-llm`/`rw-audio`?" -- needs the
# org runner settings, and no amount of placement evidence substitutes for it.
#
# WHAT IT IS FOR. A lane's first run on a box that does not already hold its snapshots pays
# their full Hub transfer into the `~/.cache/huggingface` that every other `rw-*` lane on
# that runner reads from, and an ENOSPC partway through leaves partial blobs behind rather
# than failing cleanly. This records what the box had before that transfer started.
#
# A RECORD, NOT A PRE-DISPATCH GATE. By the time this runs the runner is already claimed, so
# it cannot stop a doomed transfer -- it explains one afterwards. No pre-dispatch check is
# available at all; that would need a view of the pool from outside a job.
#
# REPORT-ONLY, DELIBERATELY. This exits 0 on every path -- unset variable, absent directory,
# `du` error. The calling steps additionally set `continue-on-error: true`, because the step
# can fail for reasons the script never sees (a lost exec bit, a wrong working directory),
# and a step that only produces a record must not be able to kill a multi-hour lane. A hard
# floor is not available to write either: `release/real-weight-models.toml` records
# `size_class` and no byte sizes, so any threshold here would be an invented per-model
# constant, and failing a lane on an invented constant is worse than it running out of disk
# with a record of why.

set -u

echo "runner: ${RUNNER_NAME:-<unset>}  home: ${HOME:-<unset>}"

# `$HOME` and `$RUNNER_TEMP` are one volume on both Macs, so `df` would print the same
# filesystem twice in a step whose only purpose is to be read. `RUNNER_TEMP` is defaulted
# rather than assumed: unset, `df` would exit non-zero on the empty operand.
df -h "${HOME:-/}" "${RUNNER_TEMP:-${HOME:-/}}" 2>/dev/null | awk 'NR == 1 || !seen[$0]++'

# Two variables in the same lane can name the SAME directory -- `candle-audio-chatterbox`
# resolves both `CHATTERBOX_SNAPSHOT` and `CHATTERBOX_VE_SNAPSHOT` to one snapshot. Sizing
# each would make a reader summing the lane's cost double-count it, in the one step that
# exists for byte accounting. Report the repeat, do not re-measure it.
# Indexed arrays, not a space-joined string: `MAGE_ORACLE_SEED_DIR` really does contain a
# space (`~/Library/Application Support/...`), so word-splitting a joined string would both
# mis-compare and mis-attribute. macOS ships bash 3.2, which has no associative arrays.
seen_paths=()
seen_owners=()

for var in "$@"; do
  path="${!var:-}"
  if [[ -z "$path" ]]; then
    echo "$var: <unset> — not passed to this job"
    continue
  fi

  duplicate_of=""
  index=0
  while [[ "$index" -lt "${#seen_paths[@]}" ]]; do
    if [[ "${seen_paths[$index]}" == "$path" ]]; then
      duplicate_of="${seen_owners[$index]}"
      break
    fi
    index=$((index + 1))
  done
  if [[ -n "$duplicate_of" ]]; then
    echo "$var: same path as $duplicate_of — counted once"
    continue
  fi
  seen_paths[${#seen_paths[@]}]="$path"
  seen_owners[${#seen_owners[@]}]="$var"

  if [[ -f "$path" ]]; then
    # A SINGLE FILE, not a directory: `MINIMAX_H3_VIDEO_VAE_REFERENCE` (sc-18932) is the ~840 KB
    # operator-produced diffusers decode reference. Without this arm the `! -d` test below reported
    # a file that is present, readable and about to be consumed as "ABSENT — not present on this
    # runner", which is worse than not reporting it: the one asset in that lane that cannot
    # self-heal would look missing on every run that has it.
    echo "$var: resident, $(du -shL "$path" 2>/dev/null | awk '{print $1}') (single file)"
  elif [[ ! -d "$path" ]]; then
    # Only a Hugging Face cache path is self-healing. `MAGE_ORACLE_SEED_DIR` is the frozen
    # Torch oracle bundle that `real-weights.yml` says exists on no Hub and that this
    # workflow deliberately refuses to regenerate, so "materializes from the Hub" would be
    # an outright false statement about the one asset that pins `rw-mage` to its box.
    if [[ "$path" == */models--* ]]; then
      echo "$var: ABSENT — this run materializes it from the Hub"
    else
      echo "$var: ABSENT — not present on this runner, and not Hub-materializable"
    fi
  elif compgen -G "$path/models--*" > /dev/null; then
    # The Hugging Face cache ROOT itself (`MLX_GEN_MODELS_ROOT`), not one model in it.
    # NEVER `du` this: it is every model the box has ever pulled -- 2.1 TB on `nax-macos`,
    # minutes of I/O in a step that exists to be cheap. The `df` above already reports the
    # volume it sits on, and every other variable in the list names a repo inside it.
    echo "$var: cache root, $(find "$path" -maxdepth 1 -name 'models--*' | wc -l | tr -d ' ') repos (not sized — see df above)"
  elif [[ "$(basename "$(dirname "$path")")" == snapshots ]]; then
    # Size the whole `models--*` repo dir, not `snapshots/<rev>`: revisions share `blobs/`,
    # so a per-revision figure understates what the cache costs on disk. Plain `du` here --
    # `-L` would follow the snapshot symlinks back down into `blobs/` and count every blob
    # a second time.
    repo="$(dirname "$(dirname "$path")")"
    echo "$var: resident, $(du -sh "$repo" 2>/dev/null | awk '{print $1}') ($(basename "$repo"))"
  else
    # Not a `models--*/snapshots/<rev>` layout (e.g. the operator-staged Mage oracle seed
    # directory), so there are no shared blobs to double-count and the path itself is the
    # thing to measure. `-L` because a hand-staged directory may be symlinked.
    echo "$var: resident, $(du -shL "$path" 2>/dev/null | awk '{print $1}')"
  fi
done

exit 0
