call "%VCVARS%"
"%REVIEWED_PYTHON%" scripts/release/speculative_bench_campaign.py run --plan "%DECODE_BENCH_OUT%\plan.json" --epic-root . --epic-sha "%GITHUB_SHA%" --pre-epic-root ..\inference-pre-epic --pre-epic-target-dir "%DECODE_BENCH_PRE_EPIC_TARGET%" --output "%DECODE_BENCH_OUT%" || exit /b 1
