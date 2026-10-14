# sc-25690 — the terminal MLX memory measurement of Iris FULL fine-tuning (batch 1, 2 optimizer
# steps + export), one process per shape under the external phys_footprint guard
# (scripts/ci/iris_restore_footprint_guard.py: cap 115 GB, 10 min wall each). Each probe starts only
# when `memory_pressure` reports >= 120 GB free (free % x total RAM, decimal GB) right before it.
#
# Every shape runs TWICE, because the process footprint and the shape's need are different things:
#   cache=default  MLX's own buffer-cache limit, as the product runs. The trainer does not clear the
#                  cache per step, so the footprint climbs to the allocator's GC limit and stays there.
#   cache=none     the test sets the cache limit to 0 (the trainer is untouched), so the footprint
#                  tracks the live bytes.
# In both, MLX's own `get_peak_memory` (reset before the load) is the shape's live-byte peak. Each
# row is classified by scripts/ci/iris_memory_probe_row.py (all figures decimal GB): MEASURED,
# CACHE-SATURATED (the footprint is the allocator ceiling, not the shape), GUARD-STOP (lower bound:
# the footprint crossed 115 GB) or FAILED. Order: the 1024² shapes the product floors are declared
# for first, then 512² (the largest-that-fits basis), then the activation-checkpointed depth and
# restoration shapes (`ac=full`: `gradient_checkpointing`, sc-25690) next to the `ac=none` rows they
# compare with. Report-only: nothing is gated on a number; the job is red only for a FAILED / NOT
# RUN / missing row.
bin="$(cat "$RUNNER_TEMP/iris-memory-probe-test-binary")"
mem_bytes="$(sysctl -n hw.memsize)"
failed=0
names=()
# The `iris_memory_mlx_steady_steps` dispatch input (empty = the default matrix): N optimizer steps,
# cache off only, without the shape the guard stops — to see whether the per-step active peak has
# settled. The value is an untrusted dispatch input: digits only, never interpolated.
steps=2
caches=(default none)
steady="${IRIS_PROBE_STEADY_STEPS:-}"
if [[ -n "$steady" ]]; then
  if [[ ! "$steady" =~ ^[1-9][0-9]?$ ]]; then
    echo "::error::iris_memory_mlx_steady_steps must be 1-99"
    exit 1
  fi
  steps="$steady"
  caches=(none)
fi
# The `iris_memory_mlx_emulated_steps` dispatch input: N optimizer steps under a SMALLER Mac's MLX
# allocator limits, cache on (as the product runs). It emulates the allocator only, never a smaller
# machine's real memory pressure. MLX (metal/allocator.cpp): gc_limit = min(0.95 x recommended
# working set, memory_limit); `set_memory_limit` re-derives gc_limit = min(limit, 0.95 x THIS
# device's recommended working set), so the probe is handed the smaller machine's GC limit as both
# memory and cache limit. The recommended working set of a 64 / 96 GiB Mac was not measured: two
# fractions of RAM are run, 0.84 (observed here: 115.45 of 137.44 GB) and a conservative 0.75.
#   64 GiB = 68.72 GB: GC limit 54.84 (0.84) / 48.96 (0.75)
#   96 GiB = 103.08 GB: GC limit 82.26 (0.84) / 73.44 (0.75)
emulated="${IRIS_PROBE_EMULATED_STEPS:-}"
if [[ -n "$emulated" ]]; then
  if [[ -n "$steady" || ! "$emulated" =~ ^[1-9][0-9]?$ ]]; then
    echo "::error::iris_memory_mlx_emulated_steps must be 1-99 and excludes iris_memory_mlx_steady_steps"
    exit 1
  fi
  steps="$emulated"
fi
probe() { # $1 task, $2 size, $3 ac, $4 cache, [$5 emulated limit GB, $6 name suffix]
  local name="$1-$2-ac-$3-cache-$4${6:-}" log free_pct free_gb
  names+=("$name")
  log="$IRIS_OUT/$name.log"
  # The previous probe's ~100 GB (a guard-stopped one especially) takes the kernel a while to
  # reclaim, so the admission waits up to 3 minutes for the box to settle before refusing.
  local waited=0
  while :; do
    free_pct="$(/usr/bin/memory_pressure -Q 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
    free_gb=$(( mem_bytes / 1000000000 * ${free_pct:-0} / 100 ))
    if [[ -n "$free_pct" && "$free_gb" -ge 120 ]] || (( waited >= 180 )); then break; fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "admission for $name: ${free_gb} GB free after waiting ${waited}s"
  if [[ -z "$free_pct" || "$free_gb" -lt 120 ]]; then
    echo "[[IRIS_MEMORY_PROBE_MLX]] $name NOT RUN: ${free_gb} GB free (< 120 GB) after waiting ${waited}s" | tee -a "$IRIS_OUT/summary.txt"
    failed=1
    return
  fi
  local rc=0
  IRIS_PROBE_TASK="$1" IRIS_PROBE_SIZE="$2" IRIS_PROBE_AC="$3" IRIS_PROBE_MLX_CACHE="$4" \
    IRIS_PROBE_MLX_LIMIT_GB="${5:-default}" IRIS_PROBE_STEPS="$steps" TMPDIR="$RUNNER_TEMP" \
    python3.12 scripts/ci/iris_restore_footprint_guard.py 115 600 \
    "$bin" train_memory_probe::real_training_memory_probe --ignored --exact --nocapture --test-threads 1 \
    > "$log" 2>&1 || rc=$?
  cat "$log"
  python3.12 scripts/ci/iris_memory_probe_row.py mlx --name "$name" --log "$log" --rc "$rc" \
    --summary "$IRIS_OUT/summary.txt" || failed=1
}
if [[ -n "$emulated" ]]; then
  caches=()
  for assumption in "rec84 54.84 82.26" "rec75 48.96 73.44"; do
    read -r tag gc64 gc96 <<< "$assumption"
    probe depth 256 none default "$gc64" "-emu64-$tag"
    probe depth 512 none default "$gc64" "-emu64-$tag"
    probe generation 512 none default "$gc96" "-emu96-$tag"
    probe restoration 512 none default "$gc96" "-emu96-$tag"
    probe generation 1024 full default "$gc96" "-emu96-$tag"
    # Needs more than a 96 GiB Mac's limit: recorded for what MLX then does.
    probe depth 1024 none default "$gc96" "-emu96-$tag"
  done
fi
for cache in "${caches[@]}"; do
  probe generation 1024 full "$cache"
  probe generation 1024 none "$cache"
  probe depth 1024 none "$cache"
  probe generation 512 none "$cache"
  probe depth 512 none "$cache"
  probe restoration 512 none "$cache"
  probe depth 1024 full "$cache"
  probe restoration 512 full "$cache"
  # The 1024² restoration full fine-tune: checkpointed it fits the 115 GB guard ...
  probe restoration 1024 full "$cache"
  # ... and plain it is the one shape the guard stops (last in each pass).
  [[ -n "$steady" ]] || probe restoration 1024 none "$cache"
done
cat "$IRIS_OUT/summary.txt"
python3.12 scripts/ci/iris_memory_probe_row.py expect --backend mlx \
  --summary "$IRIS_OUT/summary.txt" "${names[@]}" || failed=1
exit "$failed"
