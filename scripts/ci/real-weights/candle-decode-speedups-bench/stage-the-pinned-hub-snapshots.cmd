"%REVIEWED_PYTHON%" -I -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1], encoding='utf-8')).get('hub_snapshots') else 3)" "%DECODE_BENCH_OUT%\plan.json"
if errorlevel 4 exit /b 1
if errorlevel 3 (echo the matrix names no pinned Hub snapshot& exit /b 0)
if errorlevel 1 exit /b 1
"%REVIEWED_PYTHON%" -I scripts\ci\stage_pinned_hub_snapshots.py --plan "%DECODE_BENCH_OUT%\plan.json" || exit /b 1
copy /Y "%RUNNER_TEMP%\model-proof-hub-snapshots.json" "%DECODE_BENCH_OUT%\hub-snapshots-receipt.json" || exit /b 1
