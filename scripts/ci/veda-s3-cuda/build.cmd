@echo off
call "%VCVARS%" || exit /b 1
cargo test --locked --release -j2 -p candle-gen-minimax-h3 --features cuda --lib --no-run --message-format=json
exit /b %errorlevel%
