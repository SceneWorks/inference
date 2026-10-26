call "%VCVARS%"
set "LOG=%RUNNER_TEMP%\h3-dit-io-cuda.log"
set "LIST=%LOG%.list"
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration dit_io:: -- --ignored --list --color never >"%LIST%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LIST%"
if not "%RC%"=="0" exit /b %RC%
powershell -NoProfile -NonInteractive -File scripts\ci\h3-native-test-list.ps1 -ListPath "%LIST%" -ExpectedCount 4 -ExpectedNames "dit_io::the_model_embeds_the_context_before_refining_it|dit_io::the_whole_model_reproduces_the_reference_velocity|dit_io::forward_packed_rejects_an_out_of_range_adaln_index|dit_io::the_cached_modulation_path_reproduces_the_resident_one"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" exit /b %RC%
findstr /L /C:"dit_io::the_model_embeds_the_context_before_refining_it: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_io::the_whole_model_reproduces_the_reference_velocity: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_io::forward_packed_rejects_an_out_of_range_adaln_index: test" "%LIST%" >nul || exit /b 1
findstr /L /C:"dit_io::the_cached_modulation_path_reproduces_the_resident_one: test" "%LIST%" >nul || exit /b 1
cargo test --locked -p candle-gen-minimax-h3 --features cuda --test integration dit_io:: -- --ignored --nocapture --test-threads=1 >"%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
if not "%RC%"=="0" exit /b %RC%
findstr /C:"test result: ok. 4 passed; 0 failed; 0 ignored" "%LOG%" >nul || exit /b 1
findstr /C:"test dit_io::the_model_embeds_the_context_before_refining_it ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_io::the_whole_model_reproduces_the_reference_velocity ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_io::forward_packed_rejects_an_out_of_range_adaln_index ..." "%LOG%" >nul || exit /b 1
findstr /C:"test dit_io::the_cached_modulation_path_reproduces_the_resident_one ..." "%LOG%" >nul || exit /b 1
