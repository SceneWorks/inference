param([switch]$WaitForSpace)
$ErrorActionPreference = 'Stop'

$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $env:GITHUB_WORKSPACE 'target' }
$root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($target))
$volume = [IO.DriveInfo]::new($root)

# Completed automatic CUDA jobs left up to 86.5 GB (80.6 GiB) in target
# before cleanup. Allow about 19 GiB for transient writes above that observed
# size. Other workloads can still allocate space after this check.
$minimum = 100GB
$elapsed = [Diagnostics.Stopwatch]::StartNew()
do {
    $available = $volume.AvailableFreeSpace
    Write-Host "CUDA build volume $root available_bytes=$available total_bytes=$($volume.TotalSize)"
    if (-not $WaitForSpace -or $available -ge $minimum) { return }
    if ($elapsed.Elapsed.TotalMinutes -ge 10) {
        throw "CUDA build volume $root has only $available bytes free after waiting 10 minutes. Free at least $minimum bytes before restoring the build cache."
    }
    Start-Sleep -Seconds 30
} while ($true)
