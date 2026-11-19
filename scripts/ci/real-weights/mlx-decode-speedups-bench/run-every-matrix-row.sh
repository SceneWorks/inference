# Report-only: the host the loan ran on and its memory before the first build.
{ sysctl -n machdep.cpu.brand_string; sysctl -n hw.memsize; sw_vers -productVersion; } > "$DECODE_BENCH_OUT/host.txt" || true
python3.12 -I scripts/release/speculative_bench_campaign.py run --plan "$DECODE_BENCH_OUT/plan.json" --epic-root . --epic-sha "$GITHUB_SHA" --output "$DECODE_BENCH_OUT" --stop-starting-after-minutes "$DECODE_BENCH_BUDGET_MINUTES"
