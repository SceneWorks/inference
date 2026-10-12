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
# for first, then 512² (the largest-that-fits basis). The depth and restoration trainers refuse
# activation checkpointing, so only generation has an `ac=full` row. Report-only: nothing is gated on
# a number; the job is red only for a FAILED / NOT RUN / missing row.
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
probe() { # $1 task, $2 size, $3 ac, $4 cache
  local name="$1-$2-ac-$3-cache-$4" log free_pct free_gb
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
    IRIS_PROBE_STEPS="$steps" TMPDIR="$RUNNER_TEMP" \
    python3.12 scripts/ci/iris_restore_footprint_guard.py 115 600 \
    "$bin" train_memory_probe::real_training_memory_probe --ignored --exact --nocapture --test-threads 1 \
    > "$log" 2>&1 || rc=$?
  cat "$log"
  python3.12 scripts/ci/iris_memory_probe_row.py mlx --name "$name" --log "$log" --rc "$rc" \
    --summary "$IRIS_OUT/summary.txt" || failed=1
}
for cache in "${caches[@]}"; do
  probe generation 1024 full "$cache"
  probe generation 1024 none "$cache"
  probe depth 1024 none "$cache"
  probe generation 512 none "$cache"
  probe depth 512 none "$cache"
  probe restoration 512 none "$cache"
  # Last in each pass: the one shape the guard stops.
  [[ -n "$steady" ]] || probe restoration 1024 none "$cache"
done
cat "$IRIS_OUT/summary.txt"
python3.12 scripts/ci/iris_memory_probe_row.py expect --backend mlx \
  --summary "$IRIS_OUT/summary.txt" "${names[@]}" || failed=1
exit "$failed"
