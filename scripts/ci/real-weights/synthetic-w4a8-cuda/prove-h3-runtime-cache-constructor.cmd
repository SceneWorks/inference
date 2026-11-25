call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-runtime-cache-constructor.log"
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --lib weights::tests:: -- --ignored --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 1 -ExpectedNames "weights::tests::runtime_cache_transformer_constructor_reuses_both_video_partitions"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"weights::tests::runtime_cache_transformer_constructor_reuses_both_video_partitions: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --lib weights::tests:: -- --ignored --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 1 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test weights::tests::runtime_cache_transformer_constructor_reuses_both_video_partitions ..." "%LOG%" >nul || exit /b 1
