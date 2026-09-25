cmd /c "call `"$env:VCVARS`" && set" |
  ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") { Set-Item -Force "env:$($matches[1])" $matches[2] }
  }
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
  $baselineOutput = @(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi baseline query failed with exit code $LASTEXITCODE"
  }
  $baselineMiB = 0
  $baselineText = [string]($baselineOutput | Select-Object -First 1)
  if (-not [int]::TryParse($baselineText.Trim(), [ref]$baselineMiB)) {
    throw "nvidia-smi baseline query returned an invalid value: '$baselineText'"
  }
  $sampler = Start-Job -ArgumentList $stop, $baselineMiB -ScriptBlock {
    param([string]$stopPath, [int]$peakMiB)
    while (-not (Test-Path $stopPath)) {
      $sampleOutput = @(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
      if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi sampler query failed with exit code $LASTEXITCODE"
      }
      $usedMiB = 0
      $sampleText = [string]($sampleOutput | Select-Object -First 1)
      if (-not [int]::TryParse($sampleText.Trim(), [ref]$usedMiB)) {
        throw "nvidia-smi sampler query returned an invalid value: '$sampleText'"
      }
      $peakMiB = [Math]::Max($peakMiB, $usedMiB)
      Start-Sleep -Milliseconds 250
    }
    $peakMiB
  }
  try {
    $process = Start-Process cargo -ArgumentList $arguments -Wait -PassThru `
      -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  } finally {
    New-Item $stop -ItemType File -Force | Out-Null
  }
  Wait-Job $sampler | Out-Null
  $samplerState = $sampler.State
  try {
    $samplerOutput = @(Receive-Job $sampler -ErrorAction Stop)
  } finally {
    Remove-Job $sampler -Force
  }
  if ($samplerState -ne "Completed") {
    throw "nvidia-smi sampler job ended in state $samplerState"
  }
  if ($samplerOutput.Count -ne 1) {
    throw "nvidia-smi sampler returned $($samplerOutput.Count) values; expected one"
  }
  $peakMiB = 0
  if (-not [int]::TryParse(([string]$samplerOutput[0]).Trim(), [ref]$peakMiB)) {
    throw "nvidia-smi sampler returned an invalid peak: '$($samplerOutput[0])'"
  }
  Get-Content $stdout
  Get-Content $stderr
  Write-Host "SA3_SAME_L_CUDA_MEMORY case=$label baseline_total_mib=$baselineMiB peak_total_mib=$peakMiB peak_delta_mib=$($peakMiB - $baselineMiB)"
  if ($process.ExitCode -ne 0) { throw "SAME-L resource probe $label failed with exit code $($process.ExitCode)" }
}
Invoke-SameLProbe "literal-380s" ""
Invoke-SameLProbe "exact-max" "16777216"
# Start-Process does not update LASTEXITCODE. The vcvars bootstrap can
# leave it non-zero even after both foreground probes succeed, and
# GitHub's PowerShell wrapper propagates that stale value.
exit 0
