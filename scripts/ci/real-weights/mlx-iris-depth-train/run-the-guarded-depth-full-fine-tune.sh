mkdir -p "$IRIS_OUT"
# Build with readable diagnostics first; the JSON pass below only resolves the cached binary.
cargo test --locked --release -p mlx-gen-iris --test integration --no-run
bin=$(cargo test --locked --release -p mlx-gen-iris --test integration --no-run --message-format=json \
  | python3.12 -c 'import json,sys
for line in sys.stdin:
    m = json.loads(line)
    if m.get("reason") == "compiler-artifact" and m.get("executable") and m["target"]["name"] == "integration":
        print(m["executable"])')
test -x "$bin"
cap_kb=$((80 * 1024 * 1024))
limit_s=1200
start=$(date +%s)
"$bin" depth_train_real_weights::real_depth_full_finetune_trains_exports_and_estimates \
  --ignored --exact --nocapture --test-threads=1 > "$IRIS_OUT/depth-train.log" 2>&1 &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
  fp=$(footprint -p "$pid" 2>/dev/null | awk '/phys_footprint:/ {v=$2; u=$3; if (u=="GB") v*=1048576; else if (u=="MB") v*=1024; print int(v); exit}')
  if [ -n "$fp" ] && [ "$fp" -gt "$peak" ]; then peak=$fp; fi
  if [ -n "$fp" ] && [ "$fp" -gt "$cap_kb" ]; then
    echo "GUARD: phys_footprint ${fp} KB > 80 GB cap, SIGTERM"
    kill -TERM "$pid"
  fi
  if [ $(( $(date +%s) - start )) -gt "$limit_s" ]; then
    echo "GUARD: wall time over ${limit_s}s, SIGTERM"
    kill -TERM "$pid"
  fi
  sleep 1
done
set +e
wait "$pid"
rc=$?
set -e
cat "$IRIS_OUT/depth-train.log"
echo "[[IRIS_DEPTH_TRAIN_MLX]] exit $rc; peak phys_footprint $((peak / 1024)) MB; wall $(( $(date +%s) - start ))s" | tee "$IRIS_OUT/guard.txt"
grep -q "test result: ok. 1 passed" "$IRIS_OUT/depth-train.log"
exit "$rc"
