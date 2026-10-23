# sc-26325: fixed native Rust acceptance only. No model acquisition or Python execution.
param([switch]$SelfTest)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Require($condition, [string]$message) { if (-not $condition) { throw $message } }
function Integer($value) { ($value -is [int] -or $value -is [long]) }
function Exact-Fields($value, [string[]]$expected) {
    $actual = @($value.PSObject.Properties.Name | Sort-Object)
    Require (($actual -join '|') -ceq (($expected | Sort-Object) -join '|')) 'Unexpected control fields'
}
function Plain-Path([string]$path) {
    Require ([IO.Path]::IsPathRooted($path) -and $path -notmatch '^\\\\') 'Absolute local path required'
    $probe = Get-Item -LiteralPath $path -Force
    while ($null -ne $probe) {
        Require (($probe.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) 'Reparse point refused'
        if ($probe -is [IO.DirectoryInfo]) { $probe = $probe.Parent } else { $probe = $probe.Directory }
    }
}
function Hash([string]$path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Creation-Key([DateTime]$created) {
    # Canonicalize CIM timestamps only. A retained handle has its own full-precision key;
    # comparing timestamps from different providers is not proof of process ownership.
    $ticks=$created.ToUniversalTime().Ticks
    [DateTime]::new(($ticks-($ticks%10)),[DateTimeKind]::Utc).ToString('o')
}
function Ancestry-Order([DateTime]$child,[DateTime]$parent) {
    $child.ToUniversalTime().Ticks -ge $parent.ToUniversalTime().Ticks
}
function Identity($pidValue) {
    Require ((Integer $pidValue) -and $pidValue -gt 0) 'Invalid PID'
    $pidValue=[int]$pidValue
    $row = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue"
    Require ($null -ne $row -and $null -ne $row.CreationDate) "Unresolved PID identity $pidValue"
    [pscustomobject]@{ pid = $pidValue; name = $row.Name; created_utc = Creation-Key $row.CreationDate }
}
function Same-Identity($a, $b) {
    $a.pid -eq $b.pid -and $a.name -ieq $b.name -and [DateTimeOffset]::Parse($a.created_utc) -eq [DateTimeOffset]::Parse($b.created_utc)
}
function Pid-Subset([int[]]$actual, [int[]]$allowed) {
    foreach ($id in $actual) { Require ($allowed -contains $id) "Foreign GPU0 PID $id; never terminate it" }
}
function Check-Pin([string]$root, [string]$expected) {
    Require ($expected -cmatch '^[0-9a-f]{40}$') 'Exact inference SHA required'
    $toml = [IO.File]::ReadAllText((Join-Path $root 'Cargo.toml'))
    $pins = [regex]::Matches($toml, 'git = "https://github\.com/SceneWorks/inference", rev = "([0-9a-f]{40})"')
    Require ($pins.Count -gt 0) 'Missing product inference pin'
    foreach ($pin in $pins) { Require ($pin.Groups[1].Value -ceq $expected) 'Product Cargo pin mismatch' }
    $lock = [IO.File]::ReadAllText((Join-Path $root 'Cargo.lock'))
    $pins = [regex]::Matches($lock, 'source = "git\+https://github\.com/SceneWorks/inference\?rev=([0-9a-f]{40})#([0-9a-f]{40})"')
    Require ($pins.Count -gt 0) 'Missing product inference lock'
    foreach ($pin in $pins) { Require ($pin.Groups[1].Value -ceq $expected -and $pin.Groups[2].Value -ceq $expected) 'Product lock mismatch' }
    $constant = [IO.File]::ReadAllText((Join-Path $root 'crates/sceneworks-memory-adapter/src/lib.rs'))
    Require ($constant -match ('INFERENCE_PIN: &str = "' + $expected + '"')) 'Generated product pin mismatch'
}
function Product-Revisions([string]$checkout,[string]$feature,[string]$expected) {
    Require ($expected -cmatch '^[0-9a-f]{40}$') 'Exact frozen product feature SHA required'
    Require ($checkout -ceq $expected) 'Product checkout mismatch'
    Require ($feature -ceq $expected) 'Product must equal frozen origin/feature/sc-26321-moss-image-evaluation head'
}
function Main-Source([string]$checkout,[string]$expected,[bool]$ancestor) {
    Require ($expected -cmatch '^[0-9a-f]{40}$' -and $checkout -ceq $expected) 'Exact pinned inference checkout mismatch'
    Require $ancestor 'Pinned inference source is not reachable from origin/main'
}
function Cargo-Dependency-Setup([scriptblock]$fetch) {
    $env:CARGO_NET_OFFLINE='false'
    try { & $fetch } finally { $env:CARGO_NET_OFFLINE='true' }
}
function Native([string]$exe, [string[]]$arguments) {
    $value = & $exe @arguments
    Require ($LASTEXITCODE -eq 0) "Native command failed: $exe"
    $value
}
function Import-MsvcEnvironment([string]$batch) {
    Plain-Path $batch
    # Windows PowerShell 5.1 rewrites nested quotes passed through & cmd.exe.
    # Pass cmd's complete /s /c command line directly, preserving both quote layers.
    $start=[Diagnostics.ProcessStartInfo]::new()
    $start.FileName=$env:ComSpec
    $start.Arguments='/d /s /c ""'+$batch+'" >nul && set"'
    $start.UseShellExecute=$false
    $start.CreateNoWindow=$true
    $start.RedirectStandardOutput=$true
    $start.RedirectStandardError=$true
    $process=[Diagnostics.Process]::new(); $process.StartInfo=$start; $started=$false
    try {
        $started=$process.Start()
        Require $started 'MSVC environment process did not start'
        $null=$process.Handle # Retain the exact process object for timeout cleanup.
        $stdout=$process.StandardOutput.ReadToEndAsync()
        $stderr=$process.StandardError.ReadToEndAsync()
        Require ($process.WaitForExit(60000)) 'MSVC environment import timed out'
        Require ([Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]@($stdout,$stderr),5000)) 'MSVC environment output drain timed out'
        $lines=$stdout.GetAwaiter().GetResult()
        $null=$stderr.GetAwaiter().GetResult()
        Require ($process.ExitCode -eq 0) "MSVC environment import failed: exit=$($process.ExitCode)"
        foreach ($line in ($lines -split "`r?`n")) {
            if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1],$Matches[2],'Process') }
        }
    } finally {
        try {
            if ($started -and -not $process.HasExited) {
                $process.Kill()
                Require ($process.WaitForExit(5000)) 'MSVC environment process cleanup timed out'
            }
        } finally { $process.Dispose() }
    }
}
function Snapshot-Identity($row) {
    Require ($null -ne $row.CreationDate) 'Unresolved process creation identity'
    [pscustomobject]@{pid=[int]$row.ProcessId;name=$row.Name;created_utc=Creation-Key $row.CreationDate}
}
function Owned-Entry-Live($entry) {
    try {
        $entry.process.Refresh()
        (-not $entry.process.HasExited) -and $entry.process.Id -eq $entry.identity.pid -and
            $entry.process.StartTime.ToUniversalTime().ToString('o') -ceq $entry.handle_created_utc
    } catch { $false }
}
function Owned-Gpu-Identity([int]$id,[hashtable]$owned) {
    Require ($owned.ContainsKey($id)) "Unretained GPU PID $id"
    $entry=$owned[$id]
    # Query failures are never absence. A successful empty CIM read is permitted only
    # after the exact retained kernel process object proves it has recently exited.
    $row=Get-CimInstance Win32_Process -Filter "ProcessId=$id" -ErrorAction Stop
    if ($null -ne $row) {
        Require (Same-Identity $entry.identity (Snapshot-Identity $row)) "GPU PID identity changed: $id"
    }
    $entry.process.Refresh()
    Require ($entry.process.Id -eq $id -and $entry.identity.pid -eq $id -and
        $entry.process.StartTime.ToUniversalTime().ToString('o') -ceq $entry.handle_created_utc) "GPU PID retained handle changed: $id"
    if ($entry.process.HasExited) {
        $age=([DateTime]::UtcNow-$entry.process.ExitTime.ToUniversalTime()).TotalSeconds
        Require ($age -ge 0 -and $age -le 30) "Exited GPU PID telemetry did not clear within 30 seconds: $id"
        return [pscustomobject]@{pid=$id;name=$entry.identity.name;created_utc=$entry.identity.created_utc;exited=$true}
    }
    Require ($null -ne $row) "Live GPU PID has no CIM identity: $id"
    $entry.identity
}
function Capture-Owned($process,$expected=$null) {
    # The caller already holds the exact process handle. Authenticate the CIM read by
    # bracketing it with liveness checks on that handle, not a launch-path basename or
    # an assumed CIM/GetProcessTimes conversion. Never change either identity later.
    if ($process.HasExited) { return $null }
    $entry=[pscustomobject]@{identity=[pscustomobject]@{pid=[int]$process.Id};process=$process;handle_created_utc=$process.StartTime.ToUniversalTime().ToString('o')}
    if (-not (Owned-Entry-Live $entry)) { return $null }
    try { $identity=Identity $process.Id } catch { if ($process.HasExited) { return $null }; throw }
    if (-not (Owned-Entry-Live $entry)) { return $null }
    if ($null -ne $expected -and -not (Same-Identity $expected $identity)) { return $null }
    [pscustomobject]@{identity=$identity;process=$process;handle_created_utc=$entry.handle_created_utc}
}
function Retain-Child($row,$parent) {
    $process=$null
    try {
        $identity=Snapshot-Identity $row
        $process=[Diagnostics.Process]::GetProcessById($identity.pid)
        $null=$process.Handle # Hold this exact kernel process object through final cleanup.
        $entry=Capture-Owned $process $identity
        # The snapshot's parent identity must still be live after capturing the child handle.
        # An exited/reused parent PID never authenticates a new edge (sc-26325 / F-277).
        if ($null -eq $entry -or -not (Owned-Entry-Live $parent) -or
            -not (Same-Identity $parent.identity (Identity $parent.identity.pid)) -or
            -not (Owned-Entry-Live $parent) -or
            -not (Owned-Entry-Live $entry) -or
            -not (Ancestry-Order $process.StartTime $parent.process.StartTime) -or
            -not (Same-Identity $identity (Identity $identity.pid)) -or
            -not (Owned-Entry-Live $entry)) { return $null }
        $process=$null # Ownership transfers to the retained entry only after every check passes.
        $entry
    } catch { $null } finally { if ($null -ne $process) { $process.Dispose() } }
}
function Discover-Owned($rows,[hashtable]$owned,[scriptblock]$retain) {
    $snapshot=@{}
    foreach ($row in $rows) { $snapshot[[int]$row.ProcessId]=$row }
    foreach ($entry in @($owned.Values)) {
        if ($snapshot.ContainsKey($entry.identity.pid)) {
            $observed=Snapshot-Identity $snapshot[$entry.identity.pid]
            if (-not (Same-Identity $entry.identity $observed)) {
                $detail=@{expected=$entry.identity;observed=$observed}|ConvertTo-Json -Compress -Depth 4
                throw "Observed owned PID was reused; immutable ownership retained: $detail"
            }
        }
    }
    do {
        $added=$false
        foreach ($row in $rows) {
            $id=[int]$row.ProcessId; $parentId=[int]$row.ParentProcessId
            if ($owned.ContainsKey($id) -or -not $owned.ContainsKey($parentId) -or -not $snapshot.ContainsKey($parentId)) { continue }
            $parent=$owned[$parentId]
            if (-not (Same-Identity $parent.identity (Snapshot-Identity $snapshot[$parentId]))) { continue }
            $child=Snapshot-Identity $row
            if (-not (Ancestry-Order $row.CreationDate $snapshot[$parentId].CreationDate)) { continue }
            $entry=& $retain $row $parent
            if ($null -ne $entry) {
                Require (Same-Identity $child $entry.identity) 'Child changed while retaining process handle'
                $owned[$id]=$entry; $added=$true
            }
        }
    } while ($added)
}
function Stop-Owned([hashtable]$owned,[scriptblock]$live,[scriptblock]$terminate) {
    foreach ($entry in @($owned.Values | Sort-Object { $_.identity.created_utc } -Descending)) {
        if (& $live $entry) { & $terminate $entry }
    }
}
function Owned-ExitCode($process) {
    $code=$process.get_ExitCode()
    Require (Integer $code) 'Owned child exit status unavailable'
    $code
}
function Owned-Run([string]$exe, [string[]]$arguments, [string]$name, [int]$seconds, [scriptblock]$check, [scriptblock]$drain=$null) {
    # All arguments are fixed tool/test arguments, never raw control strings.
    $out = Join-Path $script:evidence ($name + '.stdout.log')
    $err = Join-Path $script:evidence ($name + '.stderr.log')
    $start=[Diagnostics.ProcessStartInfo]::new()
    $start.FileName=$exe
    # Match Start-Process's existing join contract; callers already quote fixed paths.
    $start.Arguments=[string]::Join(' ',$arguments)
    $start.WorkingDirectory=$ExecutionContext.SessionState.Path.CurrentFileSystemLocation.Path
    $start.UseShellExecute=$false; $start.CreateNoWindow=$true
    $start.RedirectStandardOutput=$true; $start.RedirectStandardError=$true
    $p=[Diagnostics.Process]::new(); $p.StartInfo=$start
    $started=$false; $stdout=$null; $stderr=$null; $copies=@(); $failure=$null
    $owned = @{}
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {
        $stdout=[IO.File]::Open($out,[IO.FileMode]::Create,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        $stderr=[IO.File]::Open($err,[IO.FileMode]::Create,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        $started=$p.Start()
        Require $started 'Owned child did not start'
        $handle=$p.get_Handle()
        Require ($null -ne $handle -and $handle -ne [IntPtr]::Zero) 'Owned child handle unavailable'
        $copies+= $p.StandardOutput.BaseStream.CopyToAsync($stdout,65536)
        $copies+= $p.StandardError.BaseStream.CopyToAsync($stderr,65536)
        $root=Capture-Owned $p
        if ($null -ne $root) {
            $owned[[int]$p.Id]=$root
            @{launch_path=$exe;identity=$root.identity;handle_created_utc=$root.handle_created_utc}|
                ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $script:evidence ($name+'.ownership.json')) -Encoding UTF8
        }
        while (-not $p.HasExited) {
            foreach ($copy in $copies) { if ($copy.IsFaulted) { $null=$copy.GetAwaiter().GetResult() } }
            Require ($null -ne $root) 'Live root has no authenticated ownership identity'
            $all = @(Get-CimInstance Win32_Process)
            Discover-Owned $all $owned {param($row,$parent) Retain-Child $row $parent}
            $ids=@($owned.Values | Where-Object { Owned-Entry-Live $_ } | ForEach-Object { [int]$_.identity.pid })
            & $check $ids $owned
            Require ($watch.Elapsed.TotalSeconds -lt $seconds) "Owned child timeout: $name"
            Start-Sleep -Milliseconds 500
            $p.Refresh()
        }
        Require ($p.WaitForExit(5000)) "Owned child exit wait timed out: $name"
        $code=Owned-ExitCode $p
        Require ($code -eq 0) "Owned child failed: $name exit=$code"
    } catch { $failure=$_ } finally {
        try {
            Stop-Owned $owned {param($entry) Owned-Entry-Live $entry} {
                param($entry)
                # Kill uses the retained Process with its already-held handle; never reopen by PID.
                try { $entry.process.Kill() } catch {
                    if (Owned-Entry-Live $entry) { throw }
                }
            }
        } catch { if ($null -eq $failure) { $failure=$_ } else { Write-Warning $_.Exception.Message } }
        try {
            # Also cover identity-capture failure: Start gave us this exact object,
            # even when no CIM identity could be admitted. Never look it up again by PID.
            if ($started -and -not $p.HasExited) { try { $p.Kill() } catch { if (-not $p.HasExited) { throw } } }
            if ($started) {
                $cleanup=[Diagnostics.Stopwatch]::StartNew()
                $processes=@($p)+@($owned.Values | Where-Object { $_.process -ne $p } | ForEach-Object { $_.process })
                foreach ($process in $processes) {
                    $remaining=[Math]::Max(0,5000-[int]$cleanup.ElapsedMilliseconds)
                    Require ($process.WaitForExit($remaining)) "Owned child cleanup timed out: $name"
                }
            }
        } catch { if ($null -eq $failure) { $failure=$_ } else { Write-Warning $_.Exception.Message } }
        try {
            # Keep every authenticated handle through the existing device drain.
            if ($started -and $null -ne $drain) { & $drain $owned }
        } catch { if ($null -eq $failure) { $failure=$_ } else { Write-Warning $_.Exception.Message } }
        try {
            # Descendants can inherit the pipes: terminate them before waiting for EOF.
            if ($copies.Count -gt 0) {
                Require ([Threading.Tasks.Task]::WaitAll([Threading.Tasks.Task[]]$copies,5000)) "Owned child output drain timed out: $name"
                foreach ($copy in $copies) { $null=$copy.GetAwaiter().GetResult() }
            }
        } catch { if ($null -eq $failure) { $failure=$_ } else { Write-Warning $_.Exception.Message } }
        finally {
            $resources=@($stdout,$stderr)+@($owned.Values | Where-Object { $_.process -ne $p } | ForEach-Object { $_.process })
            if ($started) { $resources=@($p.StandardOutput,$p.StandardError)+$resources }
            foreach ($resource in $resources) {
                if ($null -ne $resource) {
                    try { $resource.Dispose() } catch { if ($null -eq $failure) { $failure=$_ } else { Write-Warning $_.Exception.Message } }
                }
            }
            try { $p.Dispose() } catch { if ($null -eq $failure) { $failure=$_ } else { Write-Warning $_.Exception.Message } }
        }
    }
    if ($null -ne $failure) { throw $failure }
}
function Check-Device([int[]]$allowed=@(),[bool]$quiescing=$false,[hashtable]$owned=@{}) {
    Require ([DateTimeOffset]::UtcNow -lt $script:deadline) 'Fresh lease expired'
    $gpu=@(Native nvidia-smi @('--query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits') | ConvertFrom-Csv -Header index,uuid,name,memory,utilization,driver | Where-Object { $_.index.Trim() -eq '0' })
    Require ($gpu.Count -eq 1 -and $gpu[0].uuid.Trim() -ceq $script:control.gpu_uuid) 'Physical GPU0 UUID mapping changed'
    $rows=@(Native nvidia-smi @('--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits') | ConvertFrom-Csv -Header uuid,pid | Where-Object { $_.uuid.Trim() -ceq $script:control.gpu_uuid })
    $baseline=@()
    foreach ($b in @($script:control.desktop_baseline)) {
        Exact-Fields $b @('pid','name','created_utc')
        Require (Same-Identity $b (Identity $b.pid)) 'Baseline PID disappeared, changed or became inaccessible'
        $baseline += [int]$b.pid
    }
    $pids=@(); foreach ($row in $rows) { Require ($row.pid.Trim() -cmatch '^[0-9]+$') 'Unresolved GPU process'; $pids += [int]$row.pid.Trim() }
    Pid-Subset $pids @($allowed+$baseline+@($owned.Keys))
    if ($allowed.Count -eq 0 -and $owned.Count -eq 0 -and -not $quiescing) {
        Require ([double]$gpu[0].memory -le [double]$script:control.baseline_max_memory_mib -and [double]$gpu[0].utilization -le [double]$script:control.baseline_max_utilization_percent) 'GPU0 not within admitted idle baseline'
    }
    $owners=@($pids | ForEach-Object {
        if ($baseline -contains $_) { Identity $_ } else { Owned-Gpu-Identity ([int]$_) $owned }
    })
    $gpu[0] | Add-Member -NotePropertyName owned_pids -NotePropertyValue @($pids | Where-Object {$baseline -notcontains $_})
    $key=($owners | ConvertTo-Json -Compress)
    if ($script:lastOwners -cne $key) {
        @{utc=[DateTime]::UtcNow.ToString('o');gpu0=$gpu[0];owners=$owners;baseline=$script:control.desktop_baseline} | ConvertTo-Json -Depth 8 -Compress | Add-Content -LiteralPath (Join-Path $script:evidence 'ownership-observations.jsonl') -Encoding UTF8
        $script:lastOwners=$key
    }
    $gpu
}
function Wait-DeviceIdle([hashtable]$owned=@{}) {
    $script:deviceIdleConfirmed=$false
    # NVML utilization is sampled over a window. Wait for the window to decay after an owned
    # process exits; foreign PIDs and UUID/identity changes still fail immediately on each poll.
    for ($attempt=0; $attempt -lt 60; $attempt++) {
        $g=@(Check-Device @() $true $owned)
        if ($g[0].owned_pids.Count -eq 0 -and [double]$g[0].memory -le [double]$script:control.baseline_max_memory_mib -and [double]$g[0].utilization -le [double]$script:control.baseline_max_utilization_percent) { $script:deviceIdleConfirmed=$true; return $g }
        Start-Sleep -Milliseconds 500
    }
    throw 'GPU0 did not return to the fresh admitted idle baseline'
}
function Evidence-Files([string]$root) {
    foreach ($directory in @($root,(Join-Path $root 'product-worker'))) {
        if (Test-Path -LiteralPath $directory) {
            Get-ChildItem -LiteralPath $directory -File | Where-Object { $_.Extension -in @('.json','.jsonl','.log','.csv','.png') }
        }
    }
}
function Test-Binary([string]$jsonLog,[string]$target) {
    $bins=@(Get-Content -LiteralPath $jsonLog | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.reason -eq 'compiler-artifact' -and $_.target.name -ceq $target -and $_.profile.test -and $null -ne $_.executable })
    Require ($bins.Count -eq 1) "Ambiguous Cargo executable: $target"
    $exe=[string]$bins[0].executable; Plain-Path $exe
    $exe
}
function Receipt([string]$root,[string]$file,$expected) {
    Require ($file -cmatch '^[A-Za-z0-9_-]+\.receipt\.json$') 'Unsafe receipt filename'
    $path=Join-Path $root $file; Plain-Path $path
    $r=Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
    Require ($r.execution_device -ceq 'cuda' -and $r.resolved.variant -ceq $expected.variant -and $r.resolved.acceleration -ceq $expected.acceleration) 'Actual native receipt mode/device mismatch'
    Require ($r.output_kind -ceq $expected.output_kind -and @($r.samples).Count -eq $expected.samples) 'Actual output/sample count mismatch'
    Require ($r.resolved.steps -eq $expected.steps) 'Actual receipt step count mismatch'
    $adapters=0; if ($expected.acceleration -ceq 'unmerged_bf16') { $adapters=1 }
    Require (@($r.adapter_reports).Count -eq $adapters) 'Actual adapter policy mismatch'
    Require (@($r.assets).Count -gt 0) 'Missing consumed asset provenance'
    $r
}
function Validate-Backend([string]$root,$protocol) {
    $rows=@(Get-Content -LiteralPath (Join-Path $root 'backend-cases.jsonl') | ForEach-Object { $_|ConvertFrom-Json })
    Require ($rows.Count -eq $protocol.backend.Count) 'Incomplete backend native case protocol'
    foreach ($expected in $protocol.backend) {
        $found=@($rows|Where-Object { $_.case -ceq $expected.case })
        Require ($found.Count -eq 1 -and $found[0].status -ceq $expected.status) 'Missing/duplicate/failed native backend case'
        $row=$found[0]
        if ($expected.status -ceq 'canceled') {
            Require ($row.typed_cancelled -eq $true -and $row.receipt_cleared -eq $true) 'Backend typed cancellation not proved'
        } else {
            Require ($row.pngs_skipped -eq $true) 'Duplicated backend PNG generation forbidden by combined protocol'
            Receipt $root $row.receipt $expected | Out-Null
        }
    }
}
function Validate-Product([string]$root,$protocol) {
    $summary=Get-Content -Raw -LiteralPath (Join-Path $root 'case-summary.json')|ConvertFrom-Json
    Require ($summary.protocol -ceq 'sc-26325-product-worker-v1' -and $summary.backend -ceq 'candle') 'Product worker protocol identity mismatch'
    Require ($summary.cases.Count -eq $protocol.product.Count) 'Incomplete product worker case protocol'
    foreach ($expected in $protocol.product) {
        $found=@($summary.cases|Where-Object { $_.case -ceq $expected.case })
        Require ($found.Count -eq 1 -and $found[0].status -ceq $expected.status) 'Missing/duplicate/failed product worker case'
        $row=$found[0]; $name=[string]$expected.case
        $events=@(Get-Content -Raw -LiteralPath (Join-Path $root ($name+'.progress.json'))|ConvertFrom-Json)
        Require ($events.Count -gt 0) 'Missing actual worker progress events'
        if ($expected.status -ceq 'canceled') {
            Require ($row.images -eq 0 -and $row.receipts -eq 0 -and $row.warmReentry -eq $true) 'Warm product cancellation leaked outputs/receipt'
            Require (-not (Test-Path -LiteralPath (Join-Path $root ($name+'.png'))) -and -not (Test-Path -LiteralPath (Join-Path $root ($name+'.receipt.json')))) 'Canceled product artifact leaked'
        } else {
            Require ($row.model -ceq ('moss_image_'+$expected.variant) -and $row.acceleration -ceq $expected.acceleration) 'Product requested model/mode mismatch'
            Require ($row.images -eq 1 -and $row.receipts -eq 1 -and $row.png -ceq ($name+'.png') -and $row.receipt -ceq ($name+'.receipt.json')) 'Product actual artifact declaration mismatch'
            $receiptExpected=[pscustomobject]@{variant=$expected.variant;acceleration=$expected.acceleration;steps=1;output_kind='images';samples=1}
            Receipt $root $row.receipt $receiptExpected|Out-Null
            Plain-Path (Join-Path $root $row.png)
        }
    }
}
function Test-Capture-Fixture {
    # Deliberately disagree across providers while keeping each provider immutable.
    # This rejects a mutation that compares the held timestamp to the CIM timestamp.
    $created=[DateTime]::new(2026,1,1,0,0,0,[DateTimeKind]::Utc).AddTicks(7)
    $script:fixtureCaptureProcess=[pscustomobject]@{Id=100;HasExited=$false;StartTime=$created}
    $script:fixtureCaptureProcess|Add-Member -MemberType ScriptMethod -Name Refresh -Value {}
    $script:fixtureCaptureRace='none'
    function Identity($pidValue) {
        if ($script:fixtureCaptureRace -eq 'unresolved') { throw 'fixture identity unavailable' }
        if ($script:fixtureCaptureRace -eq 'exit') { $script:fixtureCaptureProcess.HasExited=$true }
        if ($script:fixtureCaptureRace -eq 'creation') { $script:fixtureCaptureProcess.StartTime=$script:fixtureCaptureProcess.StartTime.AddTicks(1) }
        if ($script:fixtureCaptureRace -eq 'pid') { $script:fixtureCaptureProcess.Id=101 }
        [pscustomobject]@{pid=[int]$pidValue;name='actual-image.exe';created_utc='2026-01-01T00:00:00.0000010Z'}
    }
    try {
        $entry=Capture-Owned $script:fixtureCaptureProcess
        Require ($null -ne $entry -and (Owned-Entry-Live $entry)) 'Independent immutable CIM/handle identities rejected'
        Require ($entry.identity.created_utc -cne (Creation-Key $created)) 'Cross-provider precision fixture did not discriminate'
        foreach ($field in @('name','created_utc','pid')) {
            $changed=$entry.identity.PSObject.Copy()
            if ($field -eq 'name') { $changed.name='foreign.exe' }
            if ($field -eq 'created_utc') { $changed.created_utc=Creation-Key ([DateTime]::Parse($changed.created_utc).AddTicks(10)) }
            if ($field -eq 'pid') { $changed.pid=101 }
            Require ($null -eq (Capture-Owned $script:fixtureCaptureProcess $changed)) "Identity capture admitted changed $field"
        }
        $script:fixtureCaptureRace='unresolved'; $rejected=$false
        try { Capture-Owned $script:fixtureCaptureProcess|Out-Null } catch { $rejected=$_.Exception.Message -ceq 'fixture identity unavailable' }
        Require $rejected 'Unresolved live identity accepted'
        foreach ($race in @('exit','creation','pid')) {
            $script:fixtureCaptureProcess.Id=100; $script:fixtureCaptureProcess.HasExited=$false; $script:fixtureCaptureProcess.StartTime=$created
            $script:fixtureCaptureRace=$race
            Require ($null -eq (Capture-Owned $script:fixtureCaptureProcess)) "Identity capture admitted $race race"
        }
    } finally { Remove-Variable fixtureCaptureProcess,fixtureCaptureRace -Scope Script }
}
function Test-Ownership-Fixture {
    # sc-26325 / F-277: no real processes are killed by this discriminating CPU fixture.
    $row={param($id,$parent,$second) [pscustomobject]@{ProcessId=$id;ParentProcessId=$parent;Name='fixture.exe';CreationDate=[DateTime]::new(2026,1,1,0,0,$second,[DateTimeKind]::Utc)}}
    $root=& $row 100 9 10; $child=& $row 200 100 11; $grandchild=& $row 201 200 12
    Require ((Creation-Key ($root.CreationDate.AddTicks(-1))) -cne (Creation-Key $root.CreationDate)) 'Timestamp normalization fixture malformed'
    $exactParent=$root.CreationDate.AddTicks(5); $earlierChild=$root.CreationDate.AddTicks(4)
    Require ((Creation-Key $earlierChild) -ceq (Creation-Key $exactParent) -and (-not (Ancestry-Order $earlierChild $exactParent))) 'Held creation ordering lost sub-microsecond precision'
    Require (Ancestry-Order ($exactParent.AddTicks(1)) $exactParent) 'Valid held creation order rejected'
    $survivor=& $row 300 100 1; $foreignChild=& $row 301 300 2
    $rows=@($grandchild,$survivor,$foreignChild,$child,$root)
    $script:fixtureLive=@{}; foreach ($r in $rows) { $script:fixtureLive[[int]$r.ProcessId]=$r }
    $owned=@{100=[pscustomobject]@{identity=Snapshot-Identity $root;process=$null}}
    $retain={
        param($candidate,$parent)
        if (-not $script:fixtureLive.ContainsKey($parent.identity.pid) -or
            -not (Same-Identity $parent.identity (Snapshot-Identity $script:fixtureLive[$parent.identity.pid]))) { return $null }
        [pscustomobject]@{identity=Snapshot-Identity $candidate;process=$null}
    }
    $live={param($entry) $script:fixtureLive.ContainsKey($entry.identity.pid) -and (Same-Identity $entry.identity (Snapshot-Identity $script:fixtureLive[$entry.identity.pid]))}
    $terminate={param($entry) $script:fixtureTerminated+=@($entry.identity.pid)}
    Discover-Owned $rows $owned $retain
    Require ($owned.Count -eq 3 -and $owned.ContainsKey(200) -and $owned.ContainsKey(201)) 'Genuine child/grandchild rejected'
    Require (-not $owned.ContainsKey(300) -and -not $owned.ContainsKey(301)) 'Stale-parent foreign survivor admitted'
    foreach ($field in @('name','created_utc','pid')) {
        $changed=Snapshot-Identity $root
        if ($field -eq 'name') { $changed.name='foreign.exe' }
        if ($field -eq 'created_utc') { $changed.created_utc=Creation-Key ($root.CreationDate.AddTicks(10)) }
        if ($field -eq 'pid') { $changed.pid=101 }
        Require (-not (Same-Identity (Snapshot-Identity $root) $changed)) "Changed $field accepted"
    }
    $renamed=& $row 100 9 10; $renamed.Name='foreign.exe'
    $rejected=$false
    try { Discover-Owned @($renamed,$child) $owned $retain } catch {
        $rejected=$_.Exception.Message -like 'Observed owned PID was reused*' -and $_.Exception.Message -like '*"observed"*'
    }
    Require ($rejected -and $owned[100].identity.name -ceq 'fixture.exe') 'Changed name replaced immutable ownership or lacked identity evidence'
    $script:fixtureTerminated=@(); Stop-Owned $owned $live $terminate
    Require ($script:fixtureTerminated.Count -eq 3 -and $script:fixtureTerminated -notcontains 300 -and $script:fixtureTerminated -notcontains 301) 'Foreign survivor terminated or genuine cleanup omitted'

    # Reuse between snapshots must neither overwrite an owned identity nor authenticate its child.
    $reused=& $row 200 100 20; $foreignNew=& $row 202 200 21
    $script:fixtureLive[200]=$reused; $script:fixtureLive[202]=$foreignNew
    $rejected=$false
    try { Discover-Owned @($root,$reused,$grandchild,$foreignNew) $owned $retain } catch { $rejected=$_.Exception.Message -like 'Observed owned PID was reused*' }
    Require ($rejected -and -not $owned.ContainsKey(202) -and (Same-Identity $owned[200].identity (Snapshot-Identity $child))) 'Reused PID changed immutable ownership'
    $script:fixtureTerminated=@(); Stop-Owned $owned $live $terminate
    Require ($script:fixtureTerminated -notcontains 200 -and $script:fixtureTerminated -notcontains 202) 'Reused foreign identity terminated'

    # A parent that changes after the snapshot but before handle retention cannot admit a child.
    $owned=@{100=[pscustomobject]@{identity=Snapshot-Identity $root;process=$null}}
    $raceChild=& $row 400 100 11; $script:fixtureLive[100]=& $row 100 9 30
    Discover-Owned @($root,$raceChild) $owned $retain
    Require (-not $owned.ContainsKey(400)) 'Parent reuse during capture admitted a foreign child'
    $script:fixtureTerminated=@(); Stop-Owned $owned $live $terminate
    Require ($script:fixtureTerminated.Count -eq 0) 'Reused root identity terminated'
    $script:fixtureLive[100]=$root
    Discover-Owned @($raceChild) $owned $retain
    Require (-not $owned.ContainsKey(400)) 'Missing snapshot parent authenticated a new edge'
    Remove-Variable fixtureLive,fixtureTerminated -Scope Script
}
function Test-Device-Exit-Fixture([string]$temp) {
    # Exercise the actual device callback and drain with deterministic NVML/CIM races.
    function Reset-DeviceFixture([string]$mode) {
        $script:fixtureGpuMode=$mode; $script:fixtureGpuPoll=0
        $script:fixtureGpuProcess=[pscustomobject]@{Id=100;HasExited=$false;StartTime=[DateTime]::Parse('2026-01-01T00:00:00.0000007Z');ExitTime=[DateTime]::UtcNow}
        $script:fixtureGpuProcess | Add-Member -MemberType ScriptMethod -Name Refresh -Value {
            if ($script:fixtureGpuMode -eq 'handle-query-error') { throw 'fixture handle inaccessible' }
            if ($script:fixtureGpuMode -eq 'exit-during-refresh') { $this.HasExited=$true }
        }
        $script:fixtureGpuRow=[pscustomobject]@{ProcessId=100;Name='owned.exe';CreationDate=[DateTime]::Parse('2026-01-01T00:00:00.0000010Z')}
        $script:fixtureGpuOwned=@{100=[pscustomobject]@{identity=Snapshot-Identity $script:fixtureGpuRow;process=$script:fixtureGpuProcess;handle_created_utc=$script:fixtureGpuProcess.StartTime.ToUniversalTime().ToString('o')}}
        if ($mode -in @('exited-present','exited-missing','stale-expired','future-exit','drain','stuck-row','stuck-utilization','cim-error')) { $script:fixtureGpuProcess.HasExited=$true }
        if ($mode -eq 'stale-expired') { $script:fixtureGpuProcess.ExitTime=[DateTime]::UtcNow.AddSeconds(-31) }
        if ($mode -eq 'future-exit') { $script:fixtureGpuProcess.ExitTime=[DateTime]::UtcNow.AddSeconds(60) }
        if ($mode -eq 'wrong-handle-pid') { $script:fixtureGpuProcess.Id=101 }
        if ($mode -eq 'wrong-handle-time') { $script:fixtureGpuProcess.StartTime=$script:fixtureGpuProcess.StartTime.AddTicks(1) }
        $script:control=[pscustomobject]@{gpu_uuid='GPU-fixture';desktop_baseline=@([pscustomobject]@{pid=20;name='desktop.exe';created_utc='2026-01-01T00:00:00.0000000Z'});baseline_max_memory_mib=512;baseline_max_utilization_percent=0}
        $script:deadline=[DateTimeOffset]::UtcNow.AddMinutes(1)
        $script:evidence=$temp; $script:lastOwners='unobserved'; $script:deviceIdleConfirmed=$false
    }
    function Native([string]$exe,[string[]]$arguments) {
        if ($script:fixtureGpuMode -eq 'nvml-error') { throw 'fixture NVML inaccessible' }
        if ($arguments[0] -like '--query-gpu=*') {
            $uuid='GPU-fixture'; if ($script:fixtureGpuMode -eq 'uuid') { $uuid='GPU-foreign' }
            $util=0
            if ($script:fixtureGpuMode -eq 'stuck-utilization' -or ($script:fixtureGpuMode -eq 'drain' -and $script:fixtureGpuPoll -lt 2)) { $util=61 }
            return "0, $uuid, fixture, 282, $util, fixture-driver"
        }
        'GPU-fixture, 20'
        if ($script:fixtureGpuMode -eq 'stuck-utilization' -or ($script:fixtureGpuMode -eq 'drain' -and $script:fixtureGpuPoll -ge 1)) { return }
        if ($script:fixtureGpuMode -eq 'unknown-pid') { 'GPU-fixture, 999' } else { 'GPU-fixture, 100' }
    }
    function Identity($pidValue) {
        Require ($pidValue -eq 20) 'Fixture queried unexpected desktop identity'
        if ($script:fixtureGpuMode -eq 'baseline') { throw 'fixture desktop inaccessible' }
        $script:control.desktop_baseline[0]
    }
    function Get-CimInstance($class,[string]$Filter,[string]$ErrorAction) {
        Require ($class -ceq 'Win32_Process' -and $Filter -ceq 'ProcessId=100' -and $ErrorAction -ceq 'Stop') 'Fixture CIM query contract changed'
        if ($script:fixtureGpuMode -eq 'cim-error') { throw 'fixture CIM inaccessible' }
        if ($script:fixtureGpuMode -in @('exited-missing','missing-live','drain','stuck-row')) { return $null }
        $row=$script:fixtureGpuRow.PSObject.Copy()
        if ($script:fixtureGpuMode -eq 'wrong-pid') { $row.ProcessId=101 }
        if ($script:fixtureGpuMode -eq 'changed-name') { $row.Name='foreign.exe' }
        if ($script:fixtureGpuMode -eq 'changed-creation') { $row.CreationDate=$row.CreationDate.AddTicks(10) }
        if ($script:fixtureGpuMode -eq 'missing-creation') { $row.CreationDate=$null }
        $row
    }
    function Start-Sleep([int]$Milliseconds) { $script:fixtureGpuPoll++ }
    try {
        foreach ($mode in @('live','exit-during-refresh','exited-present','exited-missing')) {
            Reset-DeviceFixture $mode
            # Empty live-ID list reproduces a callback entered after the root exits.
            $g=@(Check-Device @() $false $script:fixtureGpuOwned)
            Require ($g.Count -eq 1 -and $g[0].owned_pids.Count -eq 1) "Owned exit race rejected: $mode"
        }
        Reset-DeviceFixture 'exit-during-refresh'
        $legacyAccepted=(Same-Identity $script:fixtureGpuOwned[100].identity (Snapshot-Identity $script:fixtureGpuRow)) -and (Owned-Entry-Live $script:fixtureGpuOwned[100])
        Require (-not $legacyAccepted) 'Legacy exit-race fixture did not discriminate'
        Check-Device @() $false $script:fixtureGpuOwned | Out-Null
        $failures=@{
            'wrong-pid'='GPU PID identity changed*';'changed-name'='GPU PID identity changed*';'changed-creation'='GPU PID identity changed*'
            'missing-creation'='Unresolved process creation identity';'wrong-handle-pid'='GPU PID retained handle changed*';'wrong-handle-time'='GPU PID retained handle changed*'
            'handle-query-error'='*fixture handle inaccessible*';'cim-error'='fixture CIM inaccessible';'missing-live'='Live GPU PID has no CIM identity*'
            'stale-expired'='Exited GPU PID telemetry did not clear*';'future-exit'='Exited GPU PID telemetry did not clear*'
            'unknown-pid'='Foreign GPU0 PID*';'unretained-allowed'='Unretained GPU PID*';'uuid'='Physical GPU0 UUID mapping changed'
            'baseline'='fixture desktop inaccessible';'nvml-error'='fixture NVML inaccessible';'deadline'='Fresh lease expired'
        }
        foreach ($mode in $failures.Keys) {
            Reset-DeviceFixture $mode
            if ($mode -eq 'unretained-allowed') { $script:fixtureGpuOwned=@{} }
            if ($mode -eq 'deadline') { $script:deadline=[DateTimeOffset]::UtcNow.AddMinutes(-1) }
            $rejected=$false
            try { Check-Device @(100) $false $script:fixtureGpuOwned | Out-Null } catch { $rejected=$_.Exception.Message -like $failures[$mode] }
            Require $rejected "Device guard admitted $mode or failed for the wrong reason"
        }
        # Reuse remains forbidden even when the held process has exited.
        foreach ($mode in @('wrong-pid','changed-name','changed-creation')) {
            Reset-DeviceFixture $mode; $script:fixtureGpuProcess.HasExited=$true
            $rejected=$false
            try { Check-Device @() $true $script:fixtureGpuOwned | Out-Null } catch { $rejected=$_.Exception.Message -like 'GPU PID identity changed*' }
            Require $rejected "Exited handle admitted reused $mode"
        }
        Reset-DeviceFixture 'drain'
        Wait-DeviceIdle $script:fixtureGpuOwned | Out-Null
        Require ($script:deviceIdleConfirmed -and $script:fixtureGpuPoll -eq 2) 'Drain skipped stale NVML rows or sampled utilization'
        foreach ($mode in @('stuck-row','stuck-utilization')) {
            Reset-DeviceFixture $mode; $rejected=$false
            try { Wait-DeviceIdle $script:fixtureGpuOwned | Out-Null } catch { $rejected=$_.Exception.Message -ceq 'GPU0 did not return to the fresh admitted idle baseline' }
            Require ($rejected -and -not $script:deviceIdleConfirmed -and $script:fixtureGpuPoll -eq 60) "Unbounded or unsafe drain: $mode"
        }
    } finally {
        Remove-Variable fixtureGpuMode,fixtureGpuPoll,fixtureGpuProcess,fixtureGpuRow,fixtureGpuOwned,control,deadline,lastOwners,deviceIdleConfirmed -Scope Script
    }
}
if ($SelfTest) {
    $temp = Join-Path ([IO.Path]::GetTempPath()) ('moss-transport-test-' + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $temp | Out-Null
    try {
        $msvc=Join-Path $temp 'MSVC fixture with spaces (x86)'
        New-Item -ItemType Directory -Path $msvc|Out-Null
        $batch=Join-Path $msvc 'vcvars64.bat'
        [IO.File]::WriteAllText($batch,"@echo off`r`nset `"MOSS_MSVC_FIXTURE=spaces & equals=preserved`"`r`nexit /b 0`r`n")
        $previous=[Environment]::GetEnvironmentVariable('MOSS_MSVC_FIXTURE','Process')
        try {
            # Reproduce the actual failed transport on Windows PowerShell 5.1 before
            # proving the replacement imports a space-containing path successfully.
            if ($PSVersionTable.PSVersion.Major -eq 5) {
                $rejected=$false
                try { Native $env:ComSpec @('/d','/c',('""'+$batch+'" >nul && set"'))|Out-Null } catch { $rejected=$true }
                Require $rejected 'Legacy MSVC quote regression did not discriminate'
            }
            Import-MsvcEnvironment $batch
            Require ($env:MOSS_MSVC_FIXTURE -ceq 'spaces & equals=preserved') 'MSVC path/environment quoting failed'
            [IO.File]::WriteAllText($batch,"@echo off`r`nset `"MOSS_MSVC_FIXTURE=must-not-import`"`r`nexit /b 7`r`n")
            $rejected=$false
            try { Import-MsvcEnvironment $batch } catch { $rejected=$_.Exception.Message -ceq 'MSVC environment import failed: exit=7' }
            Require ($rejected -and $env:MOSS_MSVC_FIXTURE -ceq 'spaces & equals=preserved') 'Failed MSVC batch imported partial environment'
        } finally { [Environment]::SetEnvironmentVariable('MOSS_MSVC_FIXTURE',$previous,'Process') }
        Test-Capture-Fixture
        Test-Ownership-Fixture
        Test-Device-Exit-Fixture $temp
        Cargo-Dependency-Setup {Require ($env:CARGO_NET_OFFLINE -ceq 'false') 'Dependency preparation was not scoped online'}
        Require ($env:CARGO_NET_OFFLINE -ceq 'true') 'Offline mode not restored after dependency setup'
        $rejected=$false
        try {Cargo-Dependency-Setup {throw 'fixture dependency failure'}} catch {$rejected=$_.Exception.Message -ceq 'fixture dependency failure'}
        Require ($rejected -and $env:CARGO_NET_OFFLINE -ceq 'true') 'Dependency failure masked or left Cargo online'
        # sc-26325 / F-278: pre-main frozen feature acceptance, never latest-main identity.
        Product-Revisions ('a'*40) ('a'*40) ('a'*40)
        $rejected=$false;try {Product-Revisions ('a'*40) ('b'*40) ('a'*40)} catch {$rejected=$true};Require $rejected 'Unfrozen product feature accepted'
        Main-Source ('a'*40) ('a'*40) $true # Main may now have unrelated newer commits.
        $rejected=$false;try {Main-Source ('a'*40) ('a'*40) $false} catch {$rejected=$true};Require $rejected 'Off-main inference source accepted'
        $rejected=$false;try {Main-Source ('b'*40) ('a'*40) $true} catch {$rejected=$true};Require $rejected 'Unpinned inference checkout accepted'
        [IO.File]::WriteAllText((Join-Path $temp 'Cargo.toml'), 'dep = { git = "https://github.com/SceneWorks/inference", rev = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }')
        [IO.File]::WriteAllText((Join-Path $temp 'Cargo.lock'), 'source = "git+https://github.com/SceneWorks/inference?rev=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"')
        New-Item -ItemType Directory -Path (Join-Path $temp 'crates/sceneworks-memory-adapter/src') | Out-Null
        [IO.File]::WriteAllText((Join-Path $temp 'crates/sceneworks-memory-adapter/src/lib.rs'), 'pub const INFERENCE_PIN: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";')
        $expected=[pscustomobject]@{variant='sft';acceleration='off';steps=1;output_kind='images';samples=1}
        $synthetic=@{execution_device='cuda';resolved=@{variant='sft';acceleration='off';steps=1};output_kind='images';samples=@(@{});adapter_reports=@();assets=@(@{})}
        $synthetic|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $temp 'fake.receipt.json') -Encoding UTF8
        Receipt $temp 'fake.receipt.json' $expected|Out-Null
        $expected.acceleration='full_fp32'
        $rejected=$false;try {Receipt $temp 'fake.receipt.json' $expected|Out-Null} catch {$rejected=$true};Require $rejected 'False receipt mode accepted'
        Check-Pin $temp ('a'*40)
        $rejected=$false; try { Check-Pin $temp ('b'*40) } catch { $rejected=$true }; Require $rejected 'Pin mismatch accepted'
        Require (Integer ([long]12)) 'JSON Int64 rejected'
        Require (-not (Integer 1.2)) 'Fractional PID accepted'
        Pid-Subset @(12) @(12,13)
        $rejected=$false; try { Pid-Subset @(14) @(12,13) } catch {$rejected=$true}; Require $rejected 'Foreign PID accepted'
        Exact-Fields ([pscustomobject]@{a=1;b=2}) @('a','b')
        $rejected=$false; try { Exact-Fields ([pscustomobject]@{a=1;b=2;c=3}) @('a','b') } catch {$rejected=$true}; Require $rejected 'Unknown field accepted'
        $a=[pscustomobject]@{pid=12;name='test';created_utc='2026-01-01T00:00:00Z'}
        $b=[pscustomobject]@{pid=12;name='test';created_utc='2026-01-01T00:00:01Z'}
        Require (-not (Same-Identity $a $b)) 'Reused PID accepted'
        Plain-Path $temp
        Require ((Hash (Join-Path $temp 'Cargo.lock')).Length -eq 64) 'Missing file hash'
        $script:evidence=$temp
        $missingExit=[pscustomobject]@{}
        $missingExit|Add-Member -MemberType ScriptMethod -Name get_ExitCode -Value {$null}
        $rejected=$false
        try { Owned-ExitCode $missingExit } catch { $rejected=$_.Exception.Message -ceq 'Owned child exit status unavailable' }
        Require $rejected 'Missing owned exit status accepted'
        $missingExit|Add-Member -MemberType ScriptMethod -Name get_ExitCode -Value {throw 'fixture exit getter failed'} -Force
        $rejected=$false
        try { Owned-ExitCode $missingExit } catch { $rejected=$_.Exception.Message -like '*fixture exit getter failed*' }
        Require $rejected 'Throwing owned exit getter accepted'
        Owned-Run $env:ComSpec @('/d','/c','exit 0') 'fake-success' 10 {param($ids)}
        $script:fixtureFailureDrained=$false; $rejected=$false
        try { Owned-Run $env:ComSpec @('/d','/c','exit 7') 'fake-failure' 10 {param($ids)} {param($owned) $script:fixtureFailureDrained=$true} } catch {$rejected=$_.Exception.Message -ceq 'Owned child failed: fake-failure exit=7'}
        Require ($rejected -and $script:fixtureFailureDrained) 'Child failure accepted or cleanup drain omitted'
        Remove-Variable fixtureFailureDrained -Scope Script
        Require (Test-Path -LiteralPath (Join-Path $temp 'fake-failure.stderr.log')) 'Failure log not retained'
        # F-277: a real harmless powershell child/grandchild chain exercises CIM versus held
        # Process creation times. The release-file protocol, not elapsed speed, proves admission.
        $grand=@'
$limit=[DateTime]::UtcNow.AddSeconds(60)
while (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'release-fixture'))) {
    if ([DateTime]::UtcNow -gt $limit) { exit 91 }
    Start-Sleep -Milliseconds 25
}
exit 0
'@
        $parent=@'
$child=Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -ArgumentList @('-NoProfile','-NonInteractive','-File',('"'+(Join-Path $PSScriptRoot 'grand-fixture.ps1')+'"')) -NoNewWindow -PassThru
$null=$child.Handle
try { $child.WaitForExit();exit $child.ExitCode } finally {$child.Dispose()}
'@
        $rootScript=$parent.Replace('grand-fixture.ps1','parent-fixture.ps1')
        [IO.File]::WriteAllText((Join-Path $temp 'grand-fixture.ps1'),$grand)
        [IO.File]::WriteAllText((Join-Path $temp 'parent-fixture.ps1'),$parent)
        [IO.File]::WriteAllText((Join-Path $temp 'root-fixture.ps1'),$rootScript)
        $script:fixtureRelease=Join-Path $temp 'release-fixture';$script:fixtureChainObserved=$false; $script:fixtureDrainObserved=$false
        # Deliberately launch via an extensionless command alias. The old root used
        # GetFileName(exe)='powershell', which cannot match CIM's 'powershell.exe'.
        # Keep the child chain alive until it is observed so a quick exit cannot pass.
        Owned-Run 'powershell' @('-NoProfile','-NonInteractive','-File',('"'+(Join-Path $temp 'root-fixture.ps1')+'"')) 'real-owned-chain' 60 {
            param($ids,$owned)
            $chain=@($owned.Values|Where-Object {$_.identity.name -ieq 'powershell.exe' -and (Owned-Entry-Live $_)})
            if ($chain.Count -ge 3) {
                foreach ($entry in $chain) {
                    Require (Same-Identity $entry.identity (Identity $entry.identity.pid)) 'Real held identity changed'
                    foreach ($field in @('name','created_utc','pid')) {
                        $changed=$entry.identity.PSObject.Copy()
                        if ($field -eq 'name') { $changed.name='foreign.exe' }
                        if ($field -eq 'created_utc') { $changed.created_utc=Creation-Key ([DateTime]::Parse($changed.created_utc).AddTicks(10)) }
                        if ($field -eq 'pid') { $changed.pid=-1 }
                        Require ($null -eq (Capture-Owned $entry.process $changed)) "Real changed $field admitted"
                    }
                    $changedHandle=[pscustomobject]@{identity=$entry.identity;process=$entry.process;handle_created_utc=([DateTime]::Parse($entry.handle_created_utc).AddTicks(1).ToUniversalTime().ToString('o'))}
                    Require (-not (Owned-Entry-Live $changedHandle)) 'Held identity lost 100ns precision'
                }
                $script:fixtureChainObserved=$true
                [IO.File]::WriteAllText($script:fixtureRelease,'release')
            }
        } {
            param($owned)
            Require ($owned.Count -ge 3) 'Drain lost retained ownership'
            foreach ($entry in $owned.Values) {
                Require ($entry.process.WaitForExit(5000)) 'Drain saw an unterminated owned process'
                Require (-not (Owned-Entry-Live $entry)) 'Exited-process fixture was still live'
                $identity=Owned-Gpu-Identity $entry.identity.pid $owned
                Require ($identity.exited -and (Same-Identity $entry.identity $identity)) 'Retained exited Windows handle rejected'
            }
            $script:fixtureDrainObserved=$true
        }
        Require ($script:fixtureChainObserved -and $script:fixtureDrainObserved) 'Real child ancestry or retained-handle drain never ran'
        $observed=Get-Content -Raw -LiteralPath (Join-Path $temp 'real-owned-chain.ownership.json')|ConvertFrom-Json
        Require ($observed.launch_path -ceq 'powershell' -and $observed.identity.name -ieq 'powershell.exe') 'Alias regression did not distinguish launch path from process identity'
        $legacy=[pscustomobject]@{pid=$observed.identity.pid;name=[IO.Path]::GetFileName($observed.launch_path);created_utc=Creation-Key ([DateTime]::Parse($observed.handle_created_utc))}
        Require (-not (Same-Identity $legacy $observed.identity)) 'Legacy root identity unexpectedly passed the alias regression'
        Write-Output ('Owned alias identity: '+($observed|ConvertTo-Json -Compress -Depth 4))
        Remove-Variable fixtureRelease,fixtureChainObserved,fixtureDrainObserved -Scope Script
        # Root exits after the exact intended child has an authenticated retained handle.
        # Output EOF must be awaited only after that pipe-holding child has stopped.
        [IO.File]::WriteAllText((Join-Path $temp 'pipe-child-fixture.ps1'),$grand.Replace('release-fixture','release-pipe-child'))
        $pipeRoot=@'
$child=Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -ArgumentList @('-NoProfile','-NonInteractive','-File',('"'+(Join-Path $PSScriptRoot 'pipe-child-fixture.ps1')+'"')) -NoNewWindow -PassThru
$null=$child.Handle
try {
    $ready=Join-Path $PSScriptRoot 'pipe-child-ready.json'
    [IO.File]::WriteAllText(($ready+'.tmp'),(@{pid=$child.Id;handle_created_utc=$child.StartTime.ToUniversalTime().ToString('o')}|ConvertTo-Json))
    [IO.File]::Move(($ready+'.tmp'),$ready)
    $limit=[DateTime]::UtcNow.AddSeconds(60)
    while (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'release-pipe-root'))) {
        if ([DateTime]::UtcNow -gt $limit) { exit 91 }
        Start-Sleep -Milliseconds 25
    }
    exit 0
} finally { $child.Dispose() }
'@
        [IO.File]::WriteAllText((Join-Path $temp 'pipe-root-fixture.ps1'),$pipeRoot)
        $script:fixturePipeObserved=$false; $script:fixturePipeDrained=$false
        $script:fixturePipeReady=Join-Path $temp 'pipe-child-ready.json'
        $script:fixturePipeRelease=Join-Path $temp 'release-pipe-root'
        $script:fixturePipeChildRelease=Join-Path $temp 'release-pipe-child'
        $script:fixturePipeChild=$null; $script:fixturePipeClock=[Diagnostics.Stopwatch]::StartNew()
        try {
            Owned-Run 'powershell' @('-NoProfile','-NonInteractive','-File',('"'+(Join-Path $temp 'pipe-root-fixture.ps1')+'"')) 'root-exits-before-pipe-child' 10 {
                param($ids,$owned)
                if (-not $script:fixturePipeObserved -and (Test-Path -LiteralPath $script:fixturePipeReady)) {
                    $ready=Get-Content -Raw -LiteralPath $script:fixturePipeReady|ConvertFrom-Json
                    Require (Integer $ready.pid) 'Pipe fixture child PID unavailable'
                    $id=[int]$ready.pid
                    if ($owned.ContainsKey($id) -and (Owned-Entry-Live $owned[$id])) {
                        $entry=$owned[$id]
                        Require ($entry.identity.name -ieq 'powershell.exe' -and $entry.handle_created_utc -ceq $ready.handle_created_utc) 'Pipe fixture retained the wrong child'
                        $script:fixturePipeChild=$entry.identity.PSObject.Copy()
                        $script:fixturePipeObserved=$true
                        Write-Output ('Pipe fixture retained: '+(@{elapsed_ms=$script:fixturePipeClock.ElapsedMilliseconds;child=$script:fixturePipeChild}|ConvertTo-Json -Compress -Depth 4))
                        [IO.File]::WriteAllText($script:fixturePipeRelease,'release')
                    }
                }
            } {
                param($owned)
                try {
                    Require ($null -ne $script:fixturePipeChild -and $owned.ContainsKey($script:fixturePipeChild.pid)) 'Pipe child was not retained'
                    $entry=$owned[$script:fixturePipeChild.pid]
                    Require (Same-Identity $entry.identity $script:fixturePipeChild) 'Pipe fixture retained child changed'
                    Require ($entry.process.HasExited) 'Pipe child survived owned cleanup'
                    foreach ($member in $owned.Values) { Require ($member.process.HasExited) 'Pipe fixture descendant survived owned cleanup' }
                    $script:fixturePipeDrained=$true
                    Write-Output ('Pipe fixture retained child exited: '+$script:fixturePipeClock.ElapsedMilliseconds)
                } finally {
                    # Release only this fixture's waiter even when an assertion failed.
                    [IO.File]::WriteAllText($script:fixturePipeChildRelease,'release')
                    Write-Output ('Pipe fixture cleanup release: '+$script:fixturePipeClock.ElapsedMilliseconds)
                }
            }
            Require ($script:fixturePipeObserved -and $script:fixturePipeDrained) 'Root-exit pipe cleanup never ran'
        } finally {
            [IO.File]::WriteAllText($script:fixturePipeChildRelease,'release')
            Write-Output ('Pipe fixture returned/finally: '+$script:fixturePipeClock.ElapsedMilliseconds)
            Remove-Variable fixturePipeReady,fixturePipeRelease,fixturePipeChildRelease,fixturePipeChild,fixturePipeClock,fixturePipeObserved,fixturePipeDrained -Scope Script
        }
        # Exercise the installed Cargo/rustup launcher without Rust compilation,
        # dependencies, network, or GPU work. Cargo waits for this temporary external
        # subcommand until the supervisor has authenticated the live process chain.
        $helper=Join-Path $temp 'cargo-moss-identity-fixture.exe'
        Add-Type -OutputAssembly $helper -OutputType ConsoleApplication -TypeDefinition @'
using System;
using System.IO;
using System.Threading;
public static class MossIdentityFixture {
    public static int Main(string[] args) {
        if (args[0] == "--io-fixture") {
            byte[] block = new byte[65536];
            for (int i = 0; i < block.Length; i++) block[i] = (byte)'O';
            var stdout = Console.OpenStandardOutput();
            stdout.Write(block, 0, block.Length); stdout.Write(block, 0, block.Length);
            for (int i = 0; i < block.Length; i++) block[i] = (byte)'E';
            var stderr = Console.OpenStandardError();
            stderr.Write(block, 0, block.Length); stderr.Write(block, 0, block.Length);
        }
        var deadline = DateTime.UtcNow.AddSeconds(60);
        while (!File.Exists(args[args.Length - 1])) {
            if (DateTime.UtcNow > deadline) return 91;
            Thread.Sleep(25);
        }
        return 0;
    }
}
'@
        $spaceHelper=Join-Path $temp 'native fixture (x86).exe'
        Copy-Item -LiteralPath $helper -Destination $spaceHelper
        $script:fixtureIoRelease=Join-Path $temp 'release io fixture (x86)'
        try {
            Owned-Run $spaceHelper @('--io-fixture',('"'+$script:fixtureIoRelease+'"')) 'streamed-io' 10 {
                param($ids)
                [IO.File]::WriteAllText($script:fixtureIoRelease,'release')
            }
            Require ([IO.File]::ReadAllText((Join-Path $temp 'streamed-io.stdout.log')) -ceq ('O'*131072)) 'Owned stdout lost or changed bytes'
            Require ([IO.File]::ReadAllText((Join-Path $temp 'streamed-io.stderr.log')) -ceq ('E'*131072)) 'Owned stderr lost or changed bytes'
        } finally { Remove-Variable fixtureIoRelease -Scope Script }
        $previousPath=$env:PATH
        $script:fixtureRelease=Join-Path $temp 'release-cargo-fixture'; $script:fixtureCargoObserved=$false
        try {
            $env:PATH=$temp+';'+$previousPath
            Owned-Run (Get-Command cargo).Source @('moss-identity-fixture',('"'+$script:fixtureRelease+'"')) 'real-cargo-chain' 60 {
                param($ids,$owned)
                $helpers=@($owned.Values|Where-Object {$_.identity.name -ieq 'cargo-moss-identity-fixture.exe' -and (Owned-Entry-Live $_)})
                if ($helpers.Count -eq 1 -and $ids.Count -ge 2) {
                    $script:fixtureCargoObserved=$true
                    [IO.File]::WriteAllText($script:fixtureRelease,'release')
                }
            }
            Require $script:fixtureCargoObserved 'Actual Cargo child was never authenticated'
            Write-Output ('Owned Cargo identity: '+(Get-Content -Raw -LiteralPath (Join-Path $temp 'real-cargo-chain.ownership.json')))
        } finally {
            $env:PATH=$previousPath
            Remove-Variable fixtureRelease,fixtureCargoObserved -Scope Script
        }
        $foreign=Start-Process -FilePath $helper -ArgumentList ('"'+(Join-Path $temp 'never-release-foreign')+'"') -PassThru -NoNewWindow
        $script:fixtureTimeoutHandle=$null
        try {
            $null=$foreign.Handle
            $rejected=$false
            try {
                Owned-Run $helper @(('"'+(Join-Path $temp 'never-release-owned')+'"')) 'fake-timeout' 1 {
                    param($ids,$owned)
                    if ($null -eq $script:fixtureTimeoutHandle) {
                        $entry=@($owned.Values)[0]
                        $script:fixtureTimeoutHandle=[Diagnostics.Process]::GetProcessById($entry.identity.pid)
                        $null=$script:fixtureTimeoutHandle.Handle
                        Require ($script:fixtureTimeoutHandle.StartTime.ToUniversalTime().ToString('o') -ceq $entry.handle_created_utc) 'Timeout observation captured a different process'
                    }
                }
            } catch { $rejected=$_.Exception.Message -like 'Owned child timeout*' }
            Require $rejected 'Owned-child timeout/cleanup failed'
            Require ($null -ne $script:fixtureTimeoutHandle -and $script:fixtureTimeoutHandle.WaitForExit(5000)) 'Timed-out owned process survived cleanup'
            $foreign.Refresh(); Require (-not $foreign.HasExited) 'Foreign same-name process was terminated'
        } finally {
            try { if (-not $foreign.HasExited) { $foreign.Kill(); $null=$foreign.WaitForExit(5000) } } finally { $foreign.Dispose() }
            if ($null -ne $script:fixtureTimeoutHandle) { $script:fixtureTimeoutHandle.Dispose() }
            Remove-Variable fixtureTimeoutHandle -Scope Script
        }
        $lockPath=Join-Path $temp 'lease.lock'
        $held=[IO.File]::Open($lockPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
        try {
            $rejected=$false; try { $other=[IO.File]::Open($lockPath,[IO.FileMode]::CreateNew); $other.Dispose() } catch {$rejected=$true}; Require $rejected 'Conflicting lease accepted'
        } finally { $held.Dispose() }
        Write-Output 'Native terminal CPU self-tests PASS'
    } finally { Remove-Item -LiteralPath $temp -Recurse -Force }
    exit 0
}
$script:evidence = $env:MOSS_TRANSPORT_ARTIFACT_ROOT
Require (-not [string]::IsNullOrEmpty($script:evidence)) 'Missing artifact root'
New-Item -ItemType Directory -Force -Path $script:evidence | Out-Null
$status = @{status='failed'; run=$env:GITHUB_RUN_ID; attempt=$env:GITHUB_RUN_ATTEMPT; host=$env:COMPUTERNAME; workflow_revision=$env:GITHUB_SHA; controller_path=$PSCommandPath; controller_sha256=(Hash $PSCommandPath); inference=$null; sceneworks=$env:MOSS_SCENEWORKS_REVISION; started_utc=[DateTime]::UtcNow.ToString('o')}
$leaseHandle=$null; $leasePath=$null; $leaseHash=$null; $persistent=$null; $persistentCreated=$false
$script:deviceIdleConfirmed=$false
try {
    Require ($env:GITHUB_REF -ceq 'refs/heads/main') 'Terminal requires final inference main, not a feature/story ref'
    $c = $env:MOSS_NATIVE_CONTROL | ConvertFrom-Json
    Exact-Fields $c @('campaign','inference_revision','snapshot_base','snapshot_shared','snapshot_dmd','gpu_uuid','physical_gpu_index','deadline_utc','timeout_seconds','owner','desktop_baseline','baseline_max_memory_mib','baseline_max_utilization_percent','external_owners_reconciled')
    Require ($c.campaign -cmatch '^sc-26325-native-[a-z0-9-]{1,48}$') 'Invalid fresh campaign'
    Require ($c.inference_revision -cmatch '^[0-9a-f]{40}$') 'Exact pinned inference SHA required'
    $status.inference=$c.inference_revision
    Require ($env:MOSS_SCENEWORKS_REVISION -cmatch '^[0-9a-f]{40}$') 'Exact SceneWorks revision required'
    Require ((Integer $c.physical_gpu_index) -and $c.physical_gpu_index -eq 0 -and $c.gpu_uuid -cmatch '^GPU-[0-9a-fA-F-]{16,64}$') 'Explicit physical GPU0 UUID required'
    Require (-not [string]::IsNullOrWhiteSpace($c.owner) -and $c.external_owners_reconciled -is [bool] -and $c.external_owners_reconciled -eq $true) 'Fresh cross-repository/external-owner reconciliation required'
    Require ($c.desktop_baseline -is [System.Array]) 'Explicit baseline identity array required'
    foreach ($number in @($c.baseline_max_memory_mib,$c.baseline_max_utilization_percent)) {
        Require ($number -is [ValueType] -and (-not [double]::IsNaN([double]$number)) -and (-not [double]::IsInfinity([double]$number)) -and [double]$number -ge 0) 'Finite nonnegative baseline limits required'
    }
    Require ([double]$c.baseline_max_utilization_percent -le 100) 'Invalid utilization baseline'
    Require ((Integer $c.timeout_seconds) -and $c.timeout_seconds -ge 1 -and $c.timeout_seconds -le 7200) 'Invalid child timeout'
    Require ($c.deadline_utc -match '(Z|[+-][0-9]{2}:[0-9]{2})$') 'Explicit UTC offset required'
    $script:deadline=[DateTimeOffset]::Parse($c.deadline_utc)
    Require ($script:deadline.Offset -eq [TimeSpan]::Zero) 'UTC deadline required'
    Require ($script:deadline -gt [DateTimeOffset]::UtcNow -and $script:deadline -le [DateTimeOffset]::UtcNow.AddHours(6)) 'Fresh deadline within six hours required'
    Require ($env:COMPUTERNAME -ieq 'MICHAEL-TRX50' -and $env:RUNNER_NAME -eq 'cuda-windows-2') 'Unexpected physical host/listener'
    $source = (Get-Location).Path
    $executionHead=(Native git @('rev-parse','HEAD')).Trim()
    & git merge-base --is-ancestor $c.inference_revision origin/main
    $mainAncestor=$LASTEXITCODE -eq 0
    Main-Source $executionHead $c.inference_revision $mainAncestor
    $dirty=@(Native git @('status','--porcelain') | Where-Object { $_ -cne '?? .terminal-sceneworks/' })
    Require ($dirty.Count -eq 0) 'Dirty inference source'
    $product = Join-Path $source '.terminal-sceneworks'
    Product-Revisions ((Native git @('-C',$product,'rev-parse','HEAD')).Trim()) ((Native git @('-C',$product,'rev-parse','origin/feature/sc-26321-moss-image-evaluation')).Trim()) $env:MOSS_SCENEWORKS_REVISION
    Require (@(Native git @('-C',$product,'status','--porcelain')).Count -eq 0) 'Dirty product source'
    Check-Pin $product $c.inference_revision
    $base='E:\sceneworks-evaluation\moss'
    Plain-Path $base
    $persistent=Join-Path $base $c.campaign
    Require (-not (Test-Path -LiteralPath $persistent)) 'Campaign already exists; preserve it and never retry automatically'
    New-Item -ItemType Directory -Path $persistent | Out-Null
    $persistentCreated=$true
    foreach ($pair in @(@('MOSS_BASE_SNAPSHOT',$c.snapshot_base),@('MOSS_SHARED_SNAPSHOT',$c.snapshot_shared),@('MOSS_DMD_SNAPSHOT',$c.snapshot_dmd))) {
        $path=[IO.Path]::GetFullPath($pair[1])
        Require ($path.StartsWith($base+'\',[StringComparison]::OrdinalIgnoreCase)) 'Snapshot outside explicit owned staging root'
        Plain-Path $path
        [Environment]::SetEnvironmentVariable($pair[0],$path,'Process')
    }
    $env:CARGO_TARGET_DIR=Join-Path $persistent 'target'
    $env:MOSS_NATIVE_OUTPUT_DIR=$script:evidence
    # Import only the installed, fixed MSVC environment. No input is inserted into cmd source.
    $vcvars='C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
    Require (Test-Path -LiteralPath $vcvars) 'Missing installed VS2022 BuildTools'
    Import-MsvcEnvironment $vcvars
    $env:NVCC_CCBIN=Join-Path $env:VCToolsInstallDir 'bin\Hostx64\x64'
    $status.rustc=(Native rustc @('-Vv')) -join "`n"; $status.cargo=(Native cargo @('-V')) -join "`n"; $status.nvcc=(Native nvcc @('--version')) -join "`n"
    # CPU source-dependency preparation is supervised and precedes all builds and the GPU lease.
    # This acquires only normal Cargo dependencies; model roots remain explicit/offline.
    $productLockHash=Hash (Join-Path $product 'Cargo.lock')
    $productManifestHash=Hash (Join-Path $product 'Cargo.toml')
    Push-Location $product
    try {
        Cargo-Dependency-Setup {
            Owned-Run (Get-Command cargo).Source @('fetch','--locked','--target','x86_64-pc-windows-msvc') 'product-fetch' 10800 {param($ids) Require ([DateTimeOffset]::UtcNow -lt $script:deadline) 'Deadline expired during Cargo dependency preparation'}
        }
    } finally { Pop-Location }
    Require ((Hash (Join-Path $product 'Cargo.lock')) -ceq $productLockHash -and (Hash (Join-Path $product 'Cargo.toml')) -ceq $productManifestHash) 'Dependency preparation modified immutable product pin/lock'
    Product-Revisions ((Native git @('-C',$product,'rev-parse','HEAD')).Trim()) ((Native git @('-C',$product,'rev-parse','origin/feature/sc-26321-moss-image-evaluation')).Trim()) $env:MOSS_SCENEWORKS_REVISION
    Check-Pin $product $c.inference_revision
    Require (@(Native git @('-C',$product,'status','--porcelain')).Count -eq 0) 'Dependency preparation dirtied product source'
    $status.product_lock_sha256=$productLockHash
    $status.product_dependency_setup='locked fetch completed; offline source verification follows'
    Owned-Run (Get-Command cargo).Source @('test','--locked','--release','-p','candle-gen-moss-image','--features','cuda','--test','integration','--no-run','--message-format=json') 'build' 10800 {param($ids) Require ([DateTimeOffset]::UtcNow -lt $script:deadline) 'Deadline expired during compile'}
    Owned-Run (Get-Command cargo).Source @('run','--locked','--release','-p','candle-gen-moss-image','--features','cuda','--example','native_inventory') 'assets' 10800 {param($ids) Require ([DateTimeOffset]::UtcNow -lt $script:deadline) 'Deadline expired during inventory'}
    $assets=@(Get-Content -LiteralPath (Join-Path $script:evidence 'assets.stdout.log') | ForEach-Object { $_ | ConvertFrom-Json })
    Require ($assets.Count -eq 41) 'Incomplete frozen native asset inventory'
    foreach ($a in $assets) { Plain-Path $a.path }
    $bins=@(Get-Content -LiteralPath (Join-Path $script:evidence 'build.stdout.log') | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.reason -eq 'compiler-artifact' -and $_.target.name -eq 'integration' -and $_.profile.test -and $null -ne $_.executable })
    Require ($bins.Count -eq 1) 'Ambiguous Cargo test executable'
    $exe=[string]$bins[0].executable; Plain-Path $exe; $status.binary_sha256=Hash $exe
    $status.cargo_lock_sha256=Hash (Join-Path $source 'Cargo.lock')
    Push-Location $product
    try {
        # The supervised CPU fetch has primed the exact final cache. This immutable offline check
        # must resolve the pinned source before any device lease, without patching the lockfile.
        Owned-Run (Get-Command cargo).Source @('metadata','--locked','--offline','--format-version','1','--filter-platform','x86_64-pc-windows-msvc') 'product-metadata' 600 {param($ids) Require ([DateTimeOffset]::UtcNow -lt $script:deadline) 'Deadline expired during offline dependency check'}
        $metadata=Get-Content -Raw -LiteralPath (Join-Path $script:evidence 'product-metadata.stdout.log')|ConvertFrom-Json
        $inferenceSources=@($metadata.packages|Where-Object {$null -ne $_.source -and $_.source -like 'git+https://github.com/SceneWorks/inference*'})
        Require ($inferenceSources.Count -gt 0) 'Offline product inference cache unresolved after locked dependency preparation; inspect product-fetch logs'
        $expectedSource='git+https://github.com/SceneWorks/inference?rev='+$c.inference_revision+'#'+$c.inference_revision
        foreach ($package in $inferenceSources) { Require ($package.source -ceq $expectedSource) 'Offline product inference cache revision mismatch' }
        Owned-Run (Get-Command cargo).Source @('test','--locked','--release','-p','sceneworks-worker','--features','backend-candle','--lib','--no-run','--message-format=json') 'product-build' 10800 {param($ids) Require ([DateTimeOffset]::UtcNow -lt $script:deadline) 'Deadline expired during product compile'}
    } finally { Pop-Location }
    $productExe=Test-Binary (Join-Path $script:evidence 'product-build.stdout.log') 'sceneworks_worker'
    $status.product_binary_sha256=Hash $productExe
    $protocolPath=Join-Path $source 'scripts/ci/moss-native-terminal-protocol.json'
    $protocol=Get-Content -Raw -LiteralPath $protocolPath|ConvertFrom-Json
    Require ($protocol.protocol -ceq 'sc-26325-native-terminal-v1') 'Native case protocol mismatch'
    Copy-Item -LiteralPath $protocolPath -Destination (Join-Path $script:evidence 'protocol.json')

    $status.assets=$assets
    $script:control=$c
    $script:lastOwners='unobserved'
    $status.before=@(Wait-DeviceIdle)
    $locks=Join-Path $base 'device-locks'; if (-not (Test-Path -LiteralPath $locks)) { New-Item -ItemType Directory -Path $locks | Out-Null }; Plain-Path $locks
    $leasePath=Join-Path $locks ($c.gpu_uuid+'.lock')
    $leaseHandle=[IO.File]::Open($leasePath,[IO.FileMode]::CreateNew,[IO.FileAccess]::ReadWrite,[IO.FileShare]::Read)
    $lease=[ordered]@{campaign=$c.campaign;owner=$c.owner;owner_pid=$PID;created_utc=(Identity $PID).created_utc;gpu_uuid=$c.gpu_uuid;physical_gpu_index=0;deadline_utc=$c.deadline_utc;inference=$c.inference_revision;run=$env:GITHUB_RUN_ID;attempt=$env:GITHUB_RUN_ATTEMPT}
    $bytes=[Text.Encoding]::UTF8.GetBytes(($lease|ConvertTo-Json)); $leaseHandle.Write($bytes,0,$bytes.Length); $leaseHandle.Flush()
    $leaseHandle.Dispose(); $leaseHandle=$null
    $hasher=[Security.Cryptography.SHA256]::Create()
    try { $leaseHash=([BitConverter]::ToString($hasher.ComputeHash($bytes))).Replace('-','').ToLowerInvariant() } finally { $hasher.Dispose() }
    $leaseHandle=[IO.File]::Open($leasePath,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
    Require ((Hash $leasePath) -ceq $leaseHash) 'Lease changed before read lock'
    $status.lease=$lease
    $env:MOSS_NATIVE_TERMINAL_ADMITTED='sc-26325-physical-gpu0'; $env:MOSS_PHYSICAL_GPU_INDEX='0'; $env:MOSS_EXPECTED_CUDA_UUID=$c.gpu_uuid
    $env:CUDA_DEVICE_ORDER='PCI_BUS_ID'; $env:CUDA_VISIBLE_DEVICES=$c.gpu_uuid; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
    $env:MOSS_NATIVE_SKIP_PNG='1'; $env:SCENEWORKS_GPU_ID='0'
    Check-Device @() | Out-Null
    $script:deviceIdleConfirmed=$false
    Owned-Run $exe @('real_weights::native_complete_surface_campaign','--ignored','--exact','--nocapture','--test-threads','1') 'native' $c.timeout_seconds {
        param($ids,$owned)
        Require ((Hash $script:leasePath) -ceq $script:leaseHash) 'Held UUID lease changed'
        Check-Device $ids $false $owned | Out-Null
    } {param($owned) Wait-DeviceIdle $owned | Out-Null}
    $status.native_exit_code=0
    Validate-Backend $script:evidence $protocol
    Wait-DeviceIdle|Out-Null
    Push-Location $product
    try {
        $script:deviceIdleConfirmed=$false
        Owned-Run $productExe @('--exact','image_jobs::moss_consumer_tests::moss_real_native_product_worker_campaign','--ignored','--nocapture','--test-threads=1') 'product-native' $c.timeout_seconds {
            param($ids,$owned)
            Require ((Hash $script:leasePath) -ceq $script:leaseHash) 'Held UUID lease changed'
            Check-Device $ids $false $owned|Out-Null
        } {param($owned) Wait-DeviceIdle $owned | Out-Null}
    } finally { Pop-Location }
    $status.product_exit_code=0
    Validate-Product (Join-Path $script:evidence 'product-worker') $protocol
    $status.after=@(Wait-DeviceIdle)
    $status.status='passed'
} catch {
    $status.error=$_.Exception.Message
    throw
} finally {
    try {
        Native nvidia-smi @('--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv') | Set-Content -LiteralPath (Join-Path $script:evidence 'gpu-final.csv')
        Native nvidia-smi @('--query-compute-apps=gpu_uuid,pid,process_name','--format=csv') | Set-Content -LiteralPath (Join-Path $script:evidence 'compute-final.csv')
    } catch { $status.cleanup_query_error=$_.Exception.Message }
    if ($null -ne $leaseHandle) {
        try {
            $leaseHandle.Dispose()
            if ($script:deviceIdleConfirmed -and (Test-Path -LiteralPath $leasePath) -and (Hash $leasePath) -ceq $leaseHash) { Remove-Item -LiteralPath $leasePath } else { $status.lock_retained='device idle unconfirmed or identity changed; coordinator reconciliation required' }
        } catch { $status.lock_cleanup_error=$_.Exception.Message } finally { $leaseHandle.Dispose() }
    }
    $status.finished_utc=[DateTime]::UtcNow.ToString('o')
    $status | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $script:evidence 'transport.json') -Encoding UTF8
    $files=@(Evidence-Files $script:evidence | ForEach-Object { @{path=$_.FullName.Substring($script:evidence.Length+1);bytes=$_.Length;sha256=(Hash $_.FullName)} })
    $files|ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $script:evidence 'artifact-manifest.json') -Encoding UTF8
    if ($persistentCreated) {
        $saved=Join-Path $persistent 'evidence'; New-Item -ItemType Directory -Path $saved|Out-Null
        foreach ($file in @(Evidence-Files $script:evidence)) {
            $relative=$file.FullName.Substring($script:evidence.Length+1)
            $destination=Join-Path $saved $relative
            New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($destination))|Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $destination
        }
    }
}
