set "HF_OUTER=%RUNNER_TEMP%\inference-ci-w4a8-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%"
if not exist "%HF_OUTER%" mkdir "%HF_OUTER%"
if errorlevel 1 exit /b 1
echo HF_HOME=%HF_OUTER%\hf-home>>"%GITHUB_ENV%"
echo HF_HUB_CACHE=%HF_OUTER%\hub>>"%GITHUB_ENV%"
echo HUGGINGFACE_HUB_CACHE=%HF_OUTER%\hub>>"%GITHUB_ENV%"
