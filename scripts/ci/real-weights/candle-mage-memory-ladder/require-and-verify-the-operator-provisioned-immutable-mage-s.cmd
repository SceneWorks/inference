if "%CANDLE_MAGE_SNAPSHOT%"=="" (
  echo Missing repository variable CANDLE_MAGE_SNAPSHOT for the Windows CUDA runner.
  exit /b 1
)
"%REVIEWED_PYTHON%" --version || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model mage-flow --snapshot "%CANDLE_MAGE_SNAPSHOT%" || exit /b 1
