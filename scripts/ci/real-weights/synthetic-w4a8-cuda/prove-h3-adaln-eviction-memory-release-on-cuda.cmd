call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-adaln-evict-cuda.log"
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration adaln_evict_memory:: -- --ignored --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 2 -ExpectedNames "adaln_evict_memory::the_evicted_adaln_weights_leave_memory|adaln_evict_memory::the_evicted_adaln_weights_leave_the_cuda_pool"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"adaln_evict_memory::the_evicted_adaln_weights_leave_memory: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"adaln_evict_memory::the_evicted_adaln_weights_leave_the_cuda_pool: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration adaln_evict_memory:: -- --ignored --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 2 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test adaln_evict_memory::the_evicted_adaln_weights_leave_memory ..." "%LOG%" >nul || exit /b 1
findstr /C:"test adaln_evict_memory::the_evicted_adaln_weights_leave_the_cuda_pool ..." "%LOG%" >nul || exit /b 1
