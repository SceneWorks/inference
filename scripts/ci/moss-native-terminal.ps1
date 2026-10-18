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
function Identity($pidValue) {
    Require ((Integer $pidValue) -and $pidValue -gt 0) 'Invalid PID'
    $pidValue=[int]$pidValue
    $row = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue"
    Require ($null -ne $row -and $null -ne $row.CreationDate) "Unresolved PID identity $pidValue"
    [pscustomobject]@{ pid = $pidValue; name = $row.Name; created_utc = $row.CreationDate.ToUniversalTime().ToString('o') }
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
function Native([string]$exe, [string[]]$arguments) {
    $value = & $exe @arguments
    Require ($LASTEXITCODE -eq 0) "Native command failed: $exe"
    $value
}
function Owned-Run([string]$exe, [string[]]$arguments, [string]$name, [int]$seconds, [scriptblock]$check) {
    # All arguments are fixed tool/test arguments, never raw control strings.
    $out = Join-Path $script:evidence ($name + '.stdout.log')
    $err = Join-Path $script:evidence ($name + '.stderr.log')
    $p = Start-Process -FilePath $exe -ArgumentList $arguments -PassThru -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError $err
    $null=$p.Handle
    $owned = @{}
    $parentCreated=$p.StartTime.ToUniversalTime()
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {
        while (-not $p.HasExited) {
            $all = @(Get-CimInstance Win32_Process)
            $ids = @([int]$p.Id)
            do {
                $before = $ids.Count
                $ids += @($all | Where-Object { $ids -contains [int]$_.ParentProcessId -and $ids -notcontains [int]$_.ProcessId } | ForEach-Object { [int]$_.ProcessId })
            } while ($ids.Count -gt $before)
            foreach ($id in $ids) {
                $row = $all | Where-Object ProcessId -eq $id | Select-Object -First 1
                if ($null -ne $row) {
                    Require ($null -ne $row.CreationDate) 'Unresolved owned process identity'
                    $identity=[pscustomobject]@{pid=$id;name=$row.Name;created_utc=$row.CreationDate.ToUniversalTime().ToString('o')}
                    if ($id -eq $p.Id) { Require ([Math]::Abs(($row.CreationDate.ToUniversalTime()-$parentCreated).TotalMilliseconds) -lt 1) 'Owned parent PID was reused' }
                    if ($owned.ContainsKey($id)) { Require (Same-Identity $owned[$id] $identity) 'Observed child PID was reused' } else { $owned[$id]=$identity }
                }
            }
            & $check $ids
            Require ($watch.Elapsed.TotalSeconds -lt $seconds) "Owned child timeout: $name"
            Start-Sleep -Milliseconds 500
            $p.Refresh()
        }
        $p.WaitForExit()
        Require ($p.ExitCode -eq 0) "Owned child failed: $name exit=$($p.ExitCode)"
    } finally {
        foreach ($entry in @($owned.Values | Sort-Object pid -Descending)) {
            $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($entry.pid)"
            if ($null -ne $current -and $null -ne $current.CreationDate) {
                $identity = [pscustomobject]@{pid=[int]$current.ProcessId;name=$current.Name;created_utc=$current.CreationDate.ToUniversalTime().ToString('o')}
                if (Same-Identity $entry $identity) { Stop-Process -Id $entry.pid -ErrorAction SilentlyContinue }
            }
        }
        $p.Dispose()
    }
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
if ($SelfTest) {
    $temp = Join-Path ([IO.Path]::GetTempPath()) ('moss-transport-test-' + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $temp | Out-Null
    try {
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
        Owned-Run $env:ComSpec @('/d','/c','exit 0') 'fake-success' 10 {param($ids)}
        $rejected=$false; try { Owned-Run $env:ComSpec @('/d','/c','exit 7') 'fake-failure' 10 {param($ids)} } catch {$rejected=$true}; Require $rejected 'Child failure accepted'
        Require (Test-Path -LiteralPath (Join-Path $temp 'fake-failure.stderr.log')) 'Failure log not retained'
        $rejected=$false
        try { Owned-Run (Get-Command powershell).Source @('-NoProfile','-NonInteractive','-Command','Start-Sleep -Seconds 20') 'fake-timeout' 1 {param($ids)} } catch { $rejected=$_.Exception.Message -like 'Owned child timeout*' }
        Require $rejected 'Owned-child timeout/cleanup failed'
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
$status = @{status='failed'; run=$env:GITHUB_RUN_ID; attempt=$env:GITHUB_RUN_ATTEMPT; host=$env:COMPUTERNAME; inference=$env:GITHUB_SHA; sceneworks=$env:MOSS_SCENEWORKS_REVISION; started_utc=[DateTime]::UtcNow.ToString('o')}
$leaseHandle=$null; $leasePath=$null; $leaseHash=$null; $persistent=$null; $persistentCreated=$false
try {
    Require ($env:GITHUB_REF -ceq 'refs/heads/main') 'Terminal requires final inference main, not a feature/story ref'
    $c = $env:MOSS_NATIVE_CONTROL | ConvertFrom-Json
    Exact-Fields $c @('campaign','inference_revision','snapshot_base','snapshot_shared','snapshot_dmd','gpu_uuid','physical_gpu_index','deadline_utc','timeout_seconds','owner','desktop_baseline','baseline_max_memory_mib','baseline_max_utilization_percent','external_owners_reconciled')
    Require ($c.campaign -cmatch '^sc-26325-native-[a-z0-9-]{1,48}$') 'Invalid fresh campaign'
    Require ($c.inference_revision -ceq $env:GITHUB_SHA -and $c.inference_revision -cmatch '^[0-9a-f]{40}$') 'Inference dispatch SHA mismatch'
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
    Require ((Native git @('rev-parse','HEAD')).Trim() -ceq $c.inference_revision) 'Inference checkout mismatch'
    $dirty=@(Native git @('status','--porcelain') | Where-Object { $_ -cne '?? .terminal-sceneworks/' })
    Require ($dirty.Count -eq 0) 'Dirty inference source'
    $product = Join-Path $source '.terminal-sceneworks'
    Require ((Native git @('-C',$product,'rev-parse','HEAD')).Trim() -ceq $env:MOSS_SCENEWORKS_REVISION) 'Product checkout mismatch'
    Native git @('-C',$product,'merge-base','--is-ancestor',$env:MOSS_SCENEWORKS_REVISION,'origin/main') | Out-Null
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
    $lines = Native $env:ComSpec @('/d','/c',('""'+$vcvars+'" >nul && set"'))
    foreach ($line in $lines) { if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1],$Matches[2],'Process') } }
    $env:NVCC_CCBIN=Join-Path $env:VCToolsInstallDir 'bin\Hostx64\x64'
    $status.rustc=(Native rustc @('-Vv')) -join "`n"; $status.cargo=(Native cargo @('-V')) -join "`n"; $status.nvcc=(Native nvcc @('--version')) -join "`n"
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
        # Coordinator primes the exact final git cache separately. This immutable offline check
        # must resolve the pinned source before any device lease, without patching the lockfile.
        Owned-Run (Get-Command cargo).Source @('metadata','--locked','--offline','--format-version','1','--filter-platform','x86_64-pc-windows-msvc') 'product-metadata' 600 {param($ids) Require ([DateTimeOffset]::UtcNow -lt $script:deadline) 'Deadline expired during offline dependency check'}
        $metadata=Get-Content -Raw -LiteralPath (Join-Path $script:evidence 'product-metadata.stdout.log')|ConvertFrom-Json
        $inferenceSources=@($metadata.packages|Where-Object {$null -ne $_.source -and $_.source -like 'git+https://github.com/SceneWorks/inference*'})
        Require ($inferenceSources.Count -gt 0) 'Offline product inference cache unresolved; coordinator must prime exact final pin before dispatch'
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
    function Check-Device([int[]]$allowed=@(),[bool]$quiescing=$false) {
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
        Pid-Subset $pids @($allowed+$baseline)
        if ($allowed.Count -eq 0 -and -not $quiescing) {
            Require ([double]$gpu[0].memory -le [double]$script:control.baseline_max_memory_mib -and [double]$gpu[0].utilization -le [double]$script:control.baseline_max_utilization_percent) 'GPU0 not within admitted idle baseline'
        }
        $owners=@($pids | ForEach-Object { Identity $_ })
        $key=($owners | ConvertTo-Json -Compress)
        if ($script:lastOwners -cne $key) {
            @{utc=[DateTime]::UtcNow.ToString('o');gpu0=$gpu[0];owners=$owners;baseline=$script:control.desktop_baseline} | ConvertTo-Json -Depth 8 -Compress | Add-Content -LiteralPath (Join-Path $script:evidence 'ownership-observations.jsonl') -Encoding UTF8
            $script:lastOwners=$key
        }
        $gpu
    }
    function Wait-DeviceIdle {
        # NVML utilization is sampled over a window. Wait for the window to decay after an owned
        # process exits; foreign PIDs and UUID/identity changes still fail immediately on each poll.
        for ($attempt=0; $attempt -lt 60; $attempt++) {
            $g=@(Check-Device @() $true)
            if ([double]$g[0].memory -le [double]$script:control.baseline_max_memory_mib -and [double]$g[0].utilization -le [double]$script:control.baseline_max_utilization_percent) { return $g }
            Start-Sleep -Milliseconds 500
        }
        throw 'GPU0 did not return to the fresh admitted idle baseline'
    }
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
    Owned-Run $exe @('real_weights::native_complete_surface_campaign','--ignored','--exact','--nocapture','--test-threads','1') 'native' $c.timeout_seconds {
        param($ids)
        Require ((Hash $script:leasePath) -ceq $script:leaseHash) 'Held UUID lease changed'
        Check-Device $ids | Out-Null
    }
    $status.native_exit_code=0
    Validate-Backend $script:evidence $protocol
    Wait-DeviceIdle|Out-Null
    Push-Location $product
    try {
        Owned-Run $productExe @('--exact','image_jobs::moss_consumer_tests::moss_real_native_product_worker_campaign','--ignored','--nocapture','--test-threads=1') 'product-native' $c.timeout_seconds {
            param($ids)
            Require ((Hash $script:leasePath) -ceq $script:leaseHash) 'Held UUID lease changed'
            Check-Device $ids|Out-Null
        }
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
            if ((Test-Path -LiteralPath $leasePath) -and (Hash $leasePath) -ceq $leaseHash) { Remove-Item -LiteralPath $leasePath } else { $status.lock_retained='identity changed; coordinator reconciliation required' }
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
