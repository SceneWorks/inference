set "DECODE_BENCH_SEEN_UUID="
for /f "tokens=1,2 delims=, " %%A in ('nvidia-smi --query-gpu=index,uuid --format=csv,noheader') do if "%%A"=="1" set "DECODE_BENCH_SEEN_UUID=%%B"
if /I not "%DECODE_BENCH_SEEN_UUID%"=="%DECODE_BENCH_CUDA_UUID%" (echo ::error::GPU index 1 is "%DECODE_BENCH_SEEN_UUID%", not the campaign card %DECODE_BENCH_CUDA_UUID%& exit /b 1)
echo campaign card at index 1: %DECODE_BENCH_SEEN_UUID%
