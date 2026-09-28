cmd /c "call `"$env:VCVARS`" && set" |
  ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item -Force "env:$($matches[1])" $matches[2] }
  }
$env:SA3_TEST_DURATION = "30"
$env:SA3_TEST_STEPS = "8"
$env:SA3_MEDIUM_WAV_OUT = Join-Path $env:RUNNER_TEMP "sa3-medium-cuda.wav"
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda `
  --test provider connected_medium_generation_is_stereo_finite_and_exact_length `
  -- --ignored --nocapture
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Get-FileHash -Algorithm SHA256 $env:SA3_MEDIUM_WAV_OUT
