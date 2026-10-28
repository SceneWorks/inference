call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %ERRORLEVEL%
if not defined REVIEWED_PYTHON (
  echo A reviewed absolute Python 3.12 interpreter must be provisioned.
  exit /b 1
)
set "RUSTC_WRAPPER="
"%REVIEWED_PYTHON%" -I scripts\ci\model_inference_proofs.py --backend candle-cuda --request-env MODEL_PROOF_REQUEST --expected-sha "%GITHUB_SHA%" --output "%MODEL_PROOF_EVIDENCE%"
