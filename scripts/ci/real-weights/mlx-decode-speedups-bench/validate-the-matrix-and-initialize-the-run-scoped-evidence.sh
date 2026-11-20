DECODE_BENCH_OUT="$RUNNER_TEMP/decode-speedups-bench-mlx-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
if [[ -e "$DECODE_BENCH_OUT" ]]; then echo "::error::$DECODE_BENCH_OUT already exists"; exit 1; fi
mkdir -p "$DECODE_BENCH_OUT"
echo "DECODE_BENCH_OUT=$DECODE_BENCH_OUT" >> "$GITHUB_ENV"
python3.12 -I scripts/release/speculative_bench_campaign.py plan --lane mlx --output "$DECODE_BENCH_OUT/plan.json"
