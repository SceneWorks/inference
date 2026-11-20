cmd /c "call `"$env:VCVARS`" && set" |
  ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item -Force "env:$($matches[1])" $matches[2] }
  }
. (Join-Path $PSScriptRoot "..\..\cuda_peak_sampler.ps1")
function Invoke-SameLProbe([string]$label, [string]$samples) {
  if ($samples) {
    $env:SA3_SAME_L_RESOURCE_SAMPLES = $samples
  } else {
    Remove-Item env:SA3_SAME_L_RESOURCE_SAMPLES -ErrorAction SilentlyContinue
  }
  $stdout = Join-Path $env:RUNNER_TEMP "same-l-resource-$label.stdout"
  $stderr = Join-Path $env:RUNNER_TEMP "same-l-resource-$label.stderr"
  $stop = Join-Path $env:RUNNER_TEMP "same-l-resource-$label.stop"
  Remove-Item $stop -ErrorAction SilentlyContinue
  $arguments = @(
    "test", "--locked", "--release",
    "-p", "candle-audio-stable-audio-3", "--features", "cuda",
    "same_l_long_duration_roundtrip_resource_probe", "--", "--ignored", "--nocapture"
  )
  # Every GPU is sampled; the probe's card is the one whose memory rose (see the helper).
  $measured = Measure-CudaPeak $arguments $stdout $stderr $stop
  $process = $measured.Process
  Get-Content $stdout
  Get-Content $stderr
  Write-Host ("SA3_SAME_L_CUDA_MEMORY case=$label baseline_total_mib=$($measured.BaselineMiB) " +
    "peak_total_mib=$($measured.PeakMiB) peak_delta_mib=$($measured.DeltaMiB) " +
    "gpu=$($measured.Gpu) per_gpu=$($measured.PerGpu)")
  if ($measured.DeltaMiB -le 0 -and $process.ExitCode -eq 0) {
    throw ("SAME-L resource probe $label observed no allocation above baseline on any GPU " +
      "($($measured.PerGpu)); the peak probe measured nothing")
  }
  if ($process.ExitCode -ne 0) { throw "SAME-L resource probe $label failed with exit code $($process.ExitCode)" }
}
Invoke-SameLProbe "literal-380s" ""
Invoke-SameLProbe "exact-max" "16777216"
# Start-Process does not update LASTEXITCODE. The vcvars bootstrap can
# leave it non-zero even after both foreground probes succeed, and
# GitHub's PowerShell wrapper propagates that stale value.
exit 0
