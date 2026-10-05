# Read-only follow-up for the one completed YuE2 app run. The stopped run
# root and worker ID arrive through environment, never this collector's argv.
param([Parameter(Mandatory = $true)][string]$OutputDirectory)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) { throw 'evidence directory missing' }
if ($env:YUE2_RELEASE_OLD_ROOT -cne 'E:\sceneworks-terminal\sc-23002-yue2-precision\37295993157-1' -or
    $env:YUE2_RELEASE_WORKER_ID -cne 'yue2-acceptance-ed59cf8e2088') { throw 'stopped run target is not the reviewed App8 run' }

function Hash-Text([string]$Value) {
    if ($null -eq $Value) { return $null }
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value)))).Replace('-', '').ToLowerInvariant() }
    finally { $hasher.Dispose() }
}
function Has-Old-Root([string]$Value) {
    if ($null -eq $Value) { return $false }
    $Value = $Value.Replace('/', '\')
    $start = 0
    while ($start -lt $Value.Length) {
        $found = $Value.IndexOf($env:YUE2_RELEASE_OLD_ROOT, $start, [StringComparison]::OrdinalIgnoreCase)
        if ($found -lt 0) { return $false }
        $after = $found + $env:YUE2_RELEASE_OLD_ROOT.Length
        if ($after -eq $Value.Length -or $Value[$after] -eq '\' -or $Value[$after] -eq '/' -or
            $Value[$after] -eq '"' -or $Value[$after] -eq "'" -or [char]::IsWhiteSpace($Value[$after])) { return $true }
        $start = $found + 1
    }
    return $false
}

function Save-Snapshot([string]$Name) {
    $at = (Get-Date).ToUniversalTime().ToString('o')
    try {
        $processes = @(Get-CimInstance -ClassName Win32_Process -OperationTimeoutSec 15 -ErrorAction Stop)
        $rows = @($processes | ForEach-Object {
            $created = if ($_.CreationDate) { $_.CreationDate.ToUniversalTime().ToString('o') } else { $null }
            $command = $_.CommandLine
            $exe = $_.ExecutablePath
            $length = if ($null -eq $command) { $null } else { $command.Length }
            $digest = Hash-Text $command
            $rootMatch = Has-Old-Root $command
            $exeRootMatch = Has-Old-Root $exe
            $workerMatch = $null -ne $command -and $command.IndexOf($env:YUE2_RELEASE_WORKER_ID, [StringComparison]::OrdinalIgnoreCase) -ge 0
            $relevant = $_.Name -match '^(?:sceneworks-(?:rust-api|api|worker)|candle[^.]*|node|python(?:3(?:\.\d+)?)?|powershell|pwsh|cmd|cargo|rustc|ffmpeg|nvidia-smi)\.exe$'
            if ($relevant -or $rootMatch -or $exeRootMatch -or $workerMatch -or $_.ProcessId -eq $PID) {
                @{ pid = $_.ProcessId; parentPid = $_.ParentProcessId; name = $_.Name;
                   createdUtc = $created; executablePath = $exe;
                   commandLineAvailable = ($null -ne $command -and -not [string]::IsNullOrWhiteSpace($command));
                   commandLineLength = $length;
                   commandLineSha256 = $digest;
                   oldRootInCommandLine = $rootMatch;
                   oldRootInExecutable = $exeRootMatch;
                   workerIdInCommandLine = $workerMatch }
            }
        })
        $result = @{ queriedUtc = $at; completedUtc = (Get-Date).ToUniversalTime().ToString('o');
                     complete = $true; collectorPid = $PID; totalCimCount = $processes.Count; rows = $rows }
    } catch {
        $result = @{ queriedUtc = $at; completedUtc = (Get-Date).ToUniversalTime().ToString('o');
                     complete = $false; error = $_.Exception.Message; rows = @() }
    }
    $result | ConvertTo-Json -Depth 8 | Out-File -LiteralPath (Join-Path $OutputDirectory "$Name.json") -Encoding utf8
    if (-not $result.complete) { throw "process snapshot $Name failed" }
}

for ($pair = 1; $pair -le 3; $pair++) {
    $stem = if ($pair -eq 1) { 'process-snapshot' } else { "process-snapshot-$pair" }
    Save-Snapshot "$stem-before"
    Start-Sleep -Seconds 3
    Save-Snapshot "$stem-after"
    Set-Content -LiteralPath (Join-Path $OutputDirectory "process-pair-$pair.ready") `
        -Value ([string]$pair) -NoNewline -Encoding ascii
    if ($pair -eq 3) { break }
    $decisionPath = Join-Path $OutputDirectory "process-pair-$pair.decision"
    $waited = 0
    while (-not (Test-Path -LiteralPath $decisionPath -PathType Leaf) -and $waited -lt 400) {
        Start-Sleep -Milliseconds 100
        $waited++
    }
    if (-not (Test-Path -LiteralPath $decisionPath -PathType Leaf)) { throw 'pair decision timed out' }
    $decision = Get-Content -LiteralPath $decisionPath -Raw -Encoding ascii
    if ($decision -ceq 'stop') { break }
    if ($decision -cne 'continue') { throw 'pair decision invalid' }
    Start-Sleep -Seconds 3
}
