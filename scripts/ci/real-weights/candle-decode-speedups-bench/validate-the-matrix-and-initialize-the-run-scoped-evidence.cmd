set "DECODE_BENCH_OUT=%RUNNER_TEMP%\decode-speedups-bench-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%"
if exist "%DECODE_BENCH_OUT%" (echo ::error::%DECODE_BENCH_OUT% already exists & exit /b 1)
mkdir "%DECODE_BENCH_OUT%" || exit /b 1
echo DECODE_BENCH_OUT=%DECODE_BENCH_OUT%>>"%GITHUB_ENV%"
"%REVIEWED_PYTHON%" scripts/release/speculative_bench_campaign.py plan --output "%DECODE_BENCH_OUT%\plan.json" || exit /b 1
