if "%CANDLE_MAGE_SNAPSHOT%"=="" (
  echo Missing repository variable CANDLE_MAGE_SNAPSHOT for the Windows CUDA runner.
  exit /b 1
)
if "%CANDLE_MAGE_EDIT_SNAPSHOT%"=="" exit /b 1
if "%CANDLE_MAGE_EDIT_BASE_SNAPSHOT%"=="" exit /b 1
if "%CANDLE_MAGE_EDIT_TURBO_SNAPSHOT%"=="" exit /b 1
echo MAGE_GOLDEN_DIR=%RUNNER_TEMP%\mage-flow-oracles>>"%GITHUB_ENV%"
