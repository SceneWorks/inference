call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-convrot-cuda.log"
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration convrot_int8:: -- --ignored --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 1 -ExpectedNames "convrot_int8::native_projections_match_pinned_kitchen"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"convrot_int8::native_projections_match_pinned_kitchen: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration convrot_int8:: -- --ignored --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 1 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test convrot_int8::native_projections_match_pinned_kitchen ..." "%LOG%" >nul || exit /b 1
