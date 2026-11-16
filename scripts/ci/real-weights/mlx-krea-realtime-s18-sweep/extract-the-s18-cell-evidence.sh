touch "$RUNNER_TEMP/s18-sweep.log"
grep -E "^(S18CELL|  VERDICT:)" "$RUNNER_TEMP/s18-sweep.log" > "$RUNNER_TEMP/s18-cells.tsv" || true
echo "captured $(grep -c '^S18CELL' "$RUNNER_TEMP/s18-cells.tsv" || true) S18CELL rows"
