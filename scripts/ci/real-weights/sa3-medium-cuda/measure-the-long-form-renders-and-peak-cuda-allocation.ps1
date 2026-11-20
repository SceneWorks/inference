cmd /c "call `"$env:VCVARS`" && set" |
  ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item -Force "env:$($matches[1])" $matches[2] }
  }
$env:SA3_MEDIUM_LONG_SECONDS = "120,300,380"
$env:SA3_MEDIUM_LONG_WAV_DIR = $env:RUNNER_TEMP
cargo test --locked --release --no-run -p candle-audio-stable-audio-3 --features cuda `
  --test provider
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$stdout = Join-Path $env:RUNNER_TEMP "sa3-medium-long-form.stdout"
$stderr = Join-Path $env:RUNNER_TEMP "sa3-medium-long-form.stderr"
$stop = Join-Path $env:RUNNER_TEMP "sa3-medium-long-form.stop"
Remove-Item $stop -ErrorAction SilentlyContinue
$arguments = @(
  "test", "--locked", "--release",
  "-p", "candle-audio-stable-audio-3", "--features", "cuda",
  "--test", "provider", "medium_long_form_renders_are_exact_and_timed",
  "--", "--ignored", "--nocapture"
)
# Every GPU is sampled; the render's card is the one whose memory rose (see the helper).
. (Join-Path $PSScriptRoot "..\..\cuda_peak_sampler.ps1")
$measured = Measure-CudaPeak $arguments $stdout $stderr $stop
$process = $measured.Process
Get-Content $stdout
Get-Content $stderr
Write-Host ("SA3_MEDIUM_CUDA_MEMORY baseline_total_mib=$($measured.BaselineMiB) " +
  "peak_total_mib=$($measured.PeakMiB) peak_delta_mib=$($measured.DeltaMiB) " +
  "gpu=$($measured.Gpu) per_gpu=$($measured.PerGpu)")
if ($process.ExitCode -ne 0) {
  throw "medium long-form render failed with exit code $($process.ExitCode)"
}
# A real medium render allocates gigabytes, so no GPU rising above its baseline means the probe
# is broken, not that the render was free.
if ($measured.DeltaMiB -le 0) {
  throw ("nvidia-smi observed no allocation above baseline on any GPU " +
    "($($measured.PerGpu)); the peak probe measured nothing")
}
foreach ($seconds in 120, 300, 380) {
  Get-FileHash -Algorithm SHA256 (Join-Path $env:RUNNER_TEMP "sa3-medium-${seconds}s.wav")
}
# Start-Process does not update LASTEXITCODE, and the vcvars bootstrap above can leave a
# stale non-zero value that GitHub's PowerShell wrapper would propagate.
exit 0
