# sc-25690 — the terminal MLX memory measurement of Iris FULL fine-tuning (batch 1, 2 optimizer
# steps + export), one process per shape under the external phys_footprint guard
# (scripts/ci/iris_restore_footprint_guard.py: cap 115 GB, 10 min wall each). Each probe starts only
# when `memory_pressure` reports >= 120 GB free (free % x total RAM, decimal GB) right before it.
# A guard stop is a MEASURED LOWER BOUND (the shape needs > 115 GB here), never a peak. Order: the
# 1024² shapes the product floors are declared for first, then 512² (the largest-that-fits basis).
# The depth and restoration trainers refuse activation checkpointing, so only generation has an
# `ac=full` row. Report-only: nothing is gated on a number.
bin="$(cat "$RUNNER_TEMP/iris-memory-probe-test-binary")"
mem_bytes="$(sysctl -n hw.memsize)"
failed=0
probe() { # $1 task, $2 size, $3 ac
  local name="$1-$2-ac-$3" log free_pct free_gb
  log="$IRIS_OUT/$name.log"
  free_pct="$(/usr/bin/memory_pressure -Q 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
  free_gb=$(( mem_bytes / 1000000000 * ${free_pct:-0} / 100 ))
  if [[ -z "$free_pct" || "$free_gb" -lt 120 ]]; then
    echo "[[IRIS_MEMORY_PROBE_MLX]] $name NOT RUN: ${free_gb} GB free (< 120 GB)" | tee -a "$IRIS_OUT/summary.txt"
    failed=1
    return
  fi
  local rc=0 guard
  IRIS_PROBE_TASK="$1" IRIS_PROBE_SIZE="$2" IRIS_PROBE_AC="$3" IRIS_PROBE_STEPS=2 TMPDIR="$RUNNER_TEMP" \
    python3.12 scripts/ci/iris_restore_footprint_guard.py 115 600 \
    "$bin" train_memory_probe::real_training_memory_probe --ignored --exact --nocapture --test-threads 1 \
    > "$log" 2>&1 || rc=$?
  cat "$log"
  guard="$(grep '^\[guard\] exit=' "$log" | tail -1 || true)"
  if grep -q "test result: ok. 1 passed" "$log"; then
    echo "[[IRIS_MEMORY_PROBE_MLX]] $name MEASURED $guard" | tee -a "$IRIS_OUT/summary.txt"
  elif grep -q '^\[guard\] SIGTERM: footprint' "$log"; then
    echo "[[IRIS_MEMORY_PROBE_MLX]] $name LOWER-BOUND (> 115 GB) $guard" | tee -a "$IRIS_OUT/summary.txt"
  else
    echo "[[IRIS_MEMORY_PROBE_MLX]] $name FAILED rc=$rc $guard" | tee -a "$IRIS_OUT/summary.txt"
    failed=1
  fi
}
probe generation 1024 full
probe generation 1024 none
probe depth 1024 none
probe restoration 1024 none
probe generation 512 none
probe depth 512 none
probe restoration 512 none
cat "$IRIS_OUT/summary.txt"
exit "$failed"
