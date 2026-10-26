call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\w4a8-cuda.log"
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration w4a8_compare:: -- --ignored --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 2 -ExpectedNames "w4a8_compare::native_forward_matches_the_external_reference|w4a8_compare::mixed_w4a8_and_convrot_layers_keep_their_own_arithmetic"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"w4a8_compare::native_forward_matches_the_external_reference: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"w4a8_compare::mixed_w4a8_and_convrot_layers_keep_their_own_arithmetic: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration w4a8_compare:: -- --ignored --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 2 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test w4a8_compare::native_forward_matches_the_external_reference ..." "%LOG%" >nul || exit /b 1
findstr /C:"test w4a8_compare::mixed_w4a8_and_convrot_layers_keep_their_own_arithmetic ..." "%LOG%" >nul || exit /b 1
