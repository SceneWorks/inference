# sc-26214 — the on-device Veda checks: every `#[ignore]`d test under `veda::device_gate::` and
# `veda_fixture::` in the crate's lib unit-test binary (exact block-sparse mask vs the dense
# reference on Metal, and the per-layer bench, which prints `VEDA_BENCH {json}` lines); libtest
# unions the two positional filters, so both run in ONE process under the external
# phys_footprint guard (cap 110 GB, 60 min wall). A filter that matches nothing is RED, not green:
# libtest exits 0 on `0 passed; ... N filtered out`, so the result line is checked too.
bin="$(cat "$RUNNER_TEMP/veda-mlx-lib-test-binary")"
log="$VEDA_OUT/device_gate.log"
rc=0
TMPDIR="$RUNNER_TEMP" python3.12 scripts/ci/iris_restore_footprint_guard.py 110 3600 \
  "$bin" veda::device_gate:: veda_fixture:: --ignored --nocapture --test-threads=1 > "$log" 2>&1 || rc=$?
cat "$log"
grep '^VEDA_BENCH ' "$log" | sed 's/^VEDA_BENCH //' > "$VEDA_OUT/bench.jsonl" || true
echo "bench lines: $(wc -l < "$VEDA_OUT/bench.jsonl")"
if (( rc != 0 )); then
  echo "::error::the Veda device gate exited $rc"
  exit 1
fi
if grep -q '^\[guard\] SIGTERM' "$log"; then
  echo "::error::the Veda device gate was stopped by the footprint guard"
  exit 1
fi
passed="$(sed -n 's/^test result: ok\. \([0-9]*\) passed; 0 failed.*/\1/p' "$log" | tail -1)"
if [[ -z "$passed" || "$passed" -eq 0 ]]; then
  echo "::error::no veda::device_gate:: / veda_fixture:: test ran (filter matched nothing, or no ok result line)"
  exit 1
fi
echo "veda::device_gate:: + veda_fixture:: $passed passed"
