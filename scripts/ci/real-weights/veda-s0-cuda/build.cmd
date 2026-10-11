@echo off
call "%VCVARS%" || exit /b 1
cargo build --locked --release -j4 -p candle-gen-minimax-h3 --features cuda,veda-s0-profile --example veda_s0_cuda
exit /b %errorlevel%
