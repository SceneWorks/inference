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
# Resolve an absolute path in the parent scope and hand it to the job. `Start-Job` runs the
# scriptblock in a separate runspace that resolves commands from its own `PATH`, which is
# not guaranteed to be the one the vcvars import above installed into this scope; a bare
# `nvidia-smi` inside the job would surface as a CommandNotFound terminating error rather
# than a measurement. The bare form does in fact work on this runner today — run
# 30196371019's `SAME-L real weights (CUDA)` job logged
# `SA3_SAME_L_CUDA_MEMORY case=literal-380s baseline_total_mib=12252 peak_total_mib=28069`
# from the identical `Invoke-SameLProbe` sampler, which is proof the job runspace both
# resolved the tool and observed a real in-render peak. This binds it explicitly anyway so
# the failure mode is a named throw in the parent rather than a dead sampler, and so the
# resolved path is in the log. `Invoke-SameLProbe` above is deliberately left as-is: it is
# already green in a recorded run and this PR does not re-open it.
$nvidiaSmi = (Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue |
  Select-Object -First 1)
if (-not $nvidiaSmi) {
  throw "nvidia-smi is not resolvable on PATH; cannot measure peak CUDA allocation"
}
$nvidiaSmiPath = $nvidiaSmi.Source
Write-Host "nvidia-smi resolved to $nvidiaSmiPath"
$baselineOutput = @(& $nvidiaSmiPath --query-gpu=memory.used --format=csv,noheader,nounits)
if ($LASTEXITCODE -ne 0) {
  throw "nvidia-smi baseline query failed with exit code $LASTEXITCODE"
}
$baselineMiB = 0
$baselineText = [string]($baselineOutput | Select-Object -First 1)
if (-not [int]::TryParse($baselineText.Trim(), [ref]$baselineMiB)) {
  throw "nvidia-smi baseline query returned an invalid value: '$baselineText'"
}
$sampler = Start-Job -ArgumentList $nvidiaSmiPath, $stop, $baselineMiB -ScriptBlock {
  param([string]$nvidiaSmiPath, [string]$stopPath, [int]$peakMiB)
  while (-not (Test-Path $stopPath)) {
    $sampleOutput = @(& $nvidiaSmiPath --query-gpu=memory.used --format=csv,noheader,nounits)
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
Write-Host "SA3_MEDIUM_CUDA_MEMORY baseline_total_mib=$baselineMiB peak_total_mib=$peakMiB peak_delta_mib=$($peakMiB - $baselineMiB)"
if ($process.ExitCode -ne 0) {
  throw "medium long-form render failed with exit code $($process.ExitCode)"
}
# The sampler seeds its running peak with the baseline, so a sampler that observed nothing
# still returns the baseline and reports peak_delta_mib=0 — indistinguishable from "the
# render allocated no VRAM". A real medium render allocates gigabytes, so a zero delta
# means the probe is broken, not that the render was free.
if ($peakMiB -le $baselineMiB) {
  throw ("nvidia-smi sampler observed no allocation above the $baselineMiB MiB baseline " +
    "(peak $peakMiB MiB); the peak probe measured nothing")
}
foreach ($seconds in 120, 300, 380) {
  Get-FileHash -Algorithm SHA256 (Join-Path $env:RUNNER_TEMP "sa3-medium-${seconds}s.wav")
}
# Start-Process does not update LASTEXITCODE, and the vcvars bootstrap above can leave a
# stale non-zero value that GitHub's PowerShell wrapper would propagate.
exit 0
