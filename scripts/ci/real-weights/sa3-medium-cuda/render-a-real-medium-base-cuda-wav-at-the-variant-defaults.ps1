cmd /c "call `"$env:VCVARS`" && set" |
  ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item -Force "env:$($matches[1])" $matches[2] }
  }
$env:SA3_MEDIUM_BASE_WAV_OUT = Join-Path $env:RUNNER_TEMP "sa3-medium-base-cuda.wav"
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda `
  --test provider connected_medium_base_generation_at_its_own_defaults_is_stereo_finite_and_exact_length `
  -- --ignored --nocapture
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Get-FileHash -Algorithm SHA256 $env:SA3_MEDIUM_BASE_WAV_OUT
