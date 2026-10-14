@echo off
call "%VCVARS%" || exit /b 1
cargo clippy --locked -j2 -p candle-gen-minimax-h3 -p candle-quant-kernels --features cuda --all-targets -- -D warnings
exit /b %errorlevel%
