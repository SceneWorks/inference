# Peak CUDA allocation of one cargo test run, sampled with nvidia-smi on EVERY GPU by index.
#
# The Windows CUDA box has two RTX PRO 6000s, and CUDA's default device order is not
# nvidia-smi's index order, so the card a test calls `cuda:0` need not be nvidia-smi GPU 0. A
# sampler that read only the first nvidia-smi line watched an idle card while the render ran on
# the other one: run 44635073560 logged baseline=peak=19 MiB for the SAME-L and medium probes.
# This samples every GPU, reports each one, and attributes the render to the GPU whose memory
# rose most. A run in which no GPU rose means the probe measured nothing and is refused.
#
# Windows PowerShell 5.1 compatible (the steps run under `shell: powershell`).

function Get-GpuMemoryUsedMiB([string]$NvidiaSmi) {
  $lines = @(& $NvidiaSmi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi memory query failed with exit code $LASTEXITCODE"
  }
  $used = @{}
  foreach ($line in $lines) {
    $parts = ([string]$line).Split(",")
    $index = 0
    $mib = 0
    if ($parts.Count -ne 2 -or -not [int]::TryParse($parts[0].Trim(), [ref]$index) -or
        -not [int]::TryParse($parts[1].Trim(), [ref]$mib)) {
      throw "nvidia-smi memory query returned an invalid line: '$line'"
    }
    $used[$index] = $mib
  }
  if ($used.Count -eq 0) {
    throw "nvidia-smi memory query reported no GPU"
  }
  $used
}

function Measure-CudaPeak([string[]]$Arguments, [string]$Stdout, [string]$Stderr, [string]$Stop) {
  # An absolute path: `Start-Job` runs in a runspace that resolves commands from its own PATH,
  # which need not be the one the vcvars import installed into the caller's scope.
  $resolved = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
  if (-not $resolved) {
    throw "nvidia-smi is not resolvable on PATH; cannot measure peak CUDA allocation"
  }
  $nvidiaSmi = $resolved.Source
  Write-Host "nvidia-smi resolved to $nvidiaSmi"
  $baseline = Get-GpuMemoryUsedMiB $nvidiaSmi
  Remove-Item $Stop -ErrorAction SilentlyContinue
  $reader = ${function:Get-GpuMemoryUsedMiB}.ToString()
  $sampler = Start-Job -ArgumentList $nvidiaSmi, $Stop, $reader -ScriptBlock {
    param([string]$nvidiaSmi, [string]$stopPath, [string]$reader)
    Set-Item -Path function:Get-GpuMemoryUsedMiB -Value ([ScriptBlock]::Create($reader))
    $peak = @{}
    while (-not (Test-Path $stopPath)) {
      $used = Get-GpuMemoryUsedMiB $nvidiaSmi
      foreach ($index in $used.Keys) {
        if (-not $peak.ContainsKey($index) -or $used[$index] -gt $peak[$index]) {
          $peak[$index] = $used[$index]
        }
      }
      Start-Sleep -Milliseconds 250
    }
    foreach ($index in $peak.Keys) { "$index,$($peak[$index])" }
  }
  try {
    $process = Start-Process cargo -ArgumentList $Arguments -Wait -PassThru `
      -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
  } finally {
    New-Item $Stop -ItemType File -Force | Out-Null
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
  $peak = @{}
  foreach ($line in $samplerOutput) {
    $parts = ([string]$line).Split(",")
    $index = 0
    $mib = 0
    if ($parts.Count -ne 2 -or -not [int]::TryParse($parts[0], [ref]$index) -or
        -not [int]::TryParse($parts[1], [ref]$mib)) {
      throw "nvidia-smi sampler returned an invalid peak line: '$line'"
    }
    $peak[$index] = $mib
  }
  # No sample (cargo exited before the job's runspace started) leaves every GPU at its baseline,
  # so the caller still prints cargo's logs and checks its exit code before judging the probe.
  $gpu = -1
  $delta = 0
  $per = @()
  foreach ($index in ($baseline.Keys | Sort-Object)) {
    $high = $baseline[$index]
    if ($peak.ContainsKey($index)) { $high = [Math]::Max($high, $peak[$index]) }
    $per += "gpu$index=$($baseline[$index])->$high"
    if ($high - $baseline[$index] -gt $delta) {
      $delta = $high - $baseline[$index]
      $gpu = $index
    }
  }
  $baselineMiB = 0
  $peakMiB = 0
  if ($gpu -ge 0) {
    $baselineMiB = $baseline[$gpu]
    $peakMiB = $baselineMiB + $delta
  }
  [pscustomobject]@{
    Process = $process
    Gpu = $gpu
    BaselineMiB = $baselineMiB
    PeakMiB = $peakMiB
    DeltaMiB = $delta
    PerGpu = ($per -join " ")
  }
}
