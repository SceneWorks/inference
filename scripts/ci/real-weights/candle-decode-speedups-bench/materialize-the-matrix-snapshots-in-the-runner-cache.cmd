rem Each snapshot alias a matrix row names: verified, else downloaded into the runner's hub cache.
"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
set "USES=import json,os,sys; sys.exit(0 if any(row.get('snapshot') == sys.argv[1] for row in json.loads(os.environ['DECODE_BENCH_MATRIX'])['rows']) else 3)"
"%REVIEWED_PYTHON%" -c "%USES%" qwen38
set "RC=%ERRORLEVEL%"
rem 0 = the matrix uses this alias, 3 = it does not; anything else is a failed check.
if "%RC%"=="0" ("%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model bonsai-qwen38-parent --snapshot "%DECODE_BENCH_SNAPSHOT_QWEN38%" || exit /b 1) else if not "%RC%"=="3" exit /b 1
"%REVIEWED_PYTHON%" -c "%USES%" bonsai-mlx
set "RC=%ERRORLEVEL%"
rem 0 = the matrix uses this alias, 3 = it does not; anything else is a failed check.
if "%RC%"=="0" ("%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model bonsai-mlx-2bit --snapshot "%DECODE_BENCH_SNAPSHOT_BONSAI_MLX%" || exit /b 1) else if not "%RC%"=="3" exit /b 1
exit /b 0
