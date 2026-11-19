@echo off
setlocal
if not defined REVIEWED_PYTHON exit /b 1
set "SENSENOVA_HUB_LIB=%RUNNER_TEMP%\sensenova-hub-%GITHUB_RUN_ID%-%GITHUB_RUN_ATTEMPT%"
set "PYTHONPATH=%SENSENOVA_HUB_LIB%"
"%REVIEWED_PYTHON%" scripts\ci\sensenova_loop_publish.py || exit /b 1
