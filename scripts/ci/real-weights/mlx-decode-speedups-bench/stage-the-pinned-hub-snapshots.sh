# The matrix's `hub:` pins (if any), staged and authenticated where the plan bound them.
if python3.12 -I -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("hub_snapshots") else 3)' "$DECODE_BENCH_OUT/plan.json"; then
  python3.12 -I scripts/ci/stage_pinned_hub_snapshots.py --plan "$DECODE_BENCH_OUT/plan.json"
  cp "$RUNNER_TEMP/model-proof-hub-snapshots.json" "$DECODE_BENCH_OUT/hub-snapshots-receipt.json"
else
  test $? -eq 3
  echo "the matrix names no pinned Hub snapshot"
fi
