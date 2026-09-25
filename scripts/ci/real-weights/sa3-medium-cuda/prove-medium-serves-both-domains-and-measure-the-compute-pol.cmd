call "%VCVARS%"
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test variant_quality -- --ignored --nocapture || exit /b 1
REM The F16-vs-F32 measurement behind the shipped F32 policy, taken on the backend
REM upstream would half-cast. sc-14545 could only measure it on Metal.
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test dtype_policy -- --ignored --nocapture || exit /b 1
