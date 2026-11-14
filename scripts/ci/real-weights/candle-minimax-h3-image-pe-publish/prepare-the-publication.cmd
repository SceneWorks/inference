rem SW-7104: the hash-locked Hugging Face client, the runner's own Hugging Face login (the identity
rem publish_hf_tiers.py uploads as; the token is never printed), and this run's scratch and reports.
"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
"%REVIEWED_PYTHON%" -c "from huggingface_hub import HfApi; n = HfApi().whoami().get('name'); assert n, 'no Hugging Face login on this runner'; print('Hugging Face identity:', n)" || exit /b 1
set "PUBLISH_REPORTS=%RUNNER_TEMP%\sw-7104-publication-reports"
if exist "%PUBLISH_REPORTS%" rmdir /s /q "%PUBLISH_REPORTS%"
mkdir "%PUBLISH_REPORTS%" || exit /b 1
>>"%GITHUB_ENV%" echo PUBLISH_REPORTS=%PUBLISH_REPORTS%
if not exist "%PUBLISH_SCRATCH%" mkdir "%PUBLISH_SCRATCH%" || exit /b 1
rem The bf16 intermediate (54.7 GB) plus Q8 and Q4, and at least 100 GB left free beyond that.
"%REVIEWED_PYTHON%" -c "import shutil, sys; free = shutil.disk_usage(sys.argv[1]).free // 10**9; need = 105 + 100; print(free, 'GB free,', need, 'GB needed'); sys.exit(0 if free >= need else 1)" "%PUBLISH_SCRATCH%" || (echo ::error::not enough free disk for the PE tiers & exit /b 1)
exit /b 0
