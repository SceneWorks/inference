# Evidence collection only. No idle verdict, process signals, CUDA or D3D device creation.
param([Parameter(Mandatory = $true)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = [IO.Path]::GetFullPath($OutputDirectory)
$tempRoot = [IO.Path]::GetFullPath($env:RUNNER_TEMP).TrimEnd('\') + '\'
if (-not $root.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase) -or (Test-Path $root)) {
    throw 'Diagnostic output must be a fresh directory under RUNNER_TEMP'
}
New-Item -ItemType Directory -Path $root | Out-Null
$issues = [Collections.Generic.List[string]]::new()
$record = [ordered]@{
    schemaVersion = 1; purpose = 'read-only GPU1 evidence; not admission'
    host = [Environment]::MachineName; runner = $env:RUNNER_NAME
    run = $env:GITHUB_RUN_ID; attempt = $env:GITHUB_RUN_ATTEMPT; source = $env:GITHUB_SHA
    startedUtc = [DateTime]::UtcNow.ToString('o'); completed = $false
    admission = 'not_evaluated'; applicationJobOwnership = 'unknown'
    physicalGpu = 1; mapping = $null; samples = @(); queries = @(); errors = $issues
}
function Save-Record {
    $json = $record | ConvertTo-Json -Depth 16
    if ([Text.Encoding]::UTF8.GetByteCount($json) -gt 4MB) { throw 'Diagnostic JSON exceeded 4 MiB bound' }
    $json | Set-Content -Encoding UTF8 (Join-Path $root 'evidence.json')
}
function Smi([string[]]$Arguments) {
    $output = @(& nvidia-smi @Arguments 2>&1 | ForEach-Object { $_.ToString() })
    $code = $LASTEXITCODE
    $record.queries += @{ utc = [DateTime]::UtcNow.ToString('o'); arguments = $Arguments; exitCode = $code }
    if ($code -ne 0) { throw "nvidia-smi query failed with exit code $code" }
    $text = $output -join "`n"
    if ($text.Length -gt 1MB) { throw 'nvidia-smi response exceeded bound' }
    return $text
}
# Pure predicates are covered by CPU expression/structure tests; Windows parsing is runtime-only.
function Test-Device-Numbers([double]$Total, [double]$Used, [double]$Utilization) {
    return (-not [double]::IsNaN($Total) -and -not [double]::IsInfinity($Total) -and $Total -gt 0 -and -not [double]::IsNaN($Used) -and -not [double]::IsInfinity($Used) -and $Used -ge 0 -and $Used -le $Total -and -not [double]::IsNaN($Utilization) -and -not [double]::IsInfinity($Utilization) -and $Utilization -ge 0 -and $Utilization -le 100)
}
function Device-Numbers($Row) {
    $parsed = @{}
    foreach ($field in @('totalMiB', 'usedMiB', 'utilization')) {
        $value = 0.0
        if (-not [double]::TryParse([string]$Row.$field, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$value)) {
            $issues.Add("Device measurement unknown: $field")
            return @{ status = 'unknown' }
        }
        $parsed[$field] = $value
    }
    if (-not (Test-Device-Numbers $parsed.totalMiB $parsed.usedMiB $parsed.utilization)) {
        $issues.Add('Device measurements nonfinite, out of range or inconsistent')
        return @{ status = 'unknown' }
    }
    return @{ status = 'valid'; totalMiB = $parsed.totalMiB; usedMiB = $parsed.usedMiB; utilization = $parsed.utilization }
}
# PDH CStatus is distinct from query success: VALID_DATA=0 and NEW_DATA=1 are both valid.
# https://learn.microsoft.com/windows/win32/perfctrs/pdh-error-codes
# https://learn.microsoft.com/windows/win32/perfctrs/checking-pdh-interface-return-values
function Test-Counter-Value([uint32]$Status, [double]$Value) {
    return (($Status -eq 0 -or $Status -eq 1) -and -not [double]::IsNaN($Value) -and -not [double]::IsInfinity($Value) -and $Value -ge 0)
}
function Identity([string]$Selector) {
    $text = Smi @("--id=$Selector", '--query-gpu=index,uuid,name,pci.bus_id,driver_version,driver_model.current,display_active,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits')
    $rows = @($text | ConvertFrom-Csv -Header index,uuid,name,pci,driver,driverModel,displayActive,totalMiB,usedMiB,utilization)
    if ($rows.Count -ne 1 -or $rows[0].index.Trim() -ne '1' -or $rows[0].uuid.Trim() -notmatch '^GPU-[0-9a-fA-F-]+$') {
        throw 'Physical GPU1 identity missing or ambiguous'
    }
    if ($rows[0].driverModel.Trim() -notin @('WDDM', 'TCC')) { $issues.Add('Driver model unknown') }
    $rows[0] | Add-Member -NotePropertyName measurements -NotePropertyValue (Device-Numbers $rows[0])
    return $rows[0]
}
function Process-Identity([int[]]$ProcessIds) {
    $result = @()
    foreach ($processIdValue in @($ProcessIds | Sort-Object -Unique)) {
        try {
            $row = Get-CimInstance Win32_Process -Filter "ProcessId=$processIdValue" -Property ProcessId,Name,CreationDate,ParentProcessId,ExecutablePath -ErrorAction Stop
            if ($null -eq $row -or $null -eq $row.CreationDate -or -not $row.Name) { throw 'Process identity unavailable or exited' }
            if (-not $row.ExecutablePath) { $issues.Add("Executable path unavailable for PID $processIdValue") }
            $result += @{ status = 'resolved'; pathAvailable = [bool]$row.ExecutablePath; pid = $row.ProcessId; name = $row.Name; createdUtc = $row.CreationDate.ToUniversalTime().ToString('o'); parentPid = $row.ParentProcessId; path = $row.ExecutablePath }
        } catch {
            $issues.Add("CIM identity unknown for PID $processIdValue")
            $result += @{ pid = $processIdValue; status = 'unknown' }
        }
    }
    return $result
}
function Census([string]$Uuid) {
    $settings = [Xml.XmlReaderSettings]::new()
    $settings.DtdProcessing = [Xml.DtdProcessing]::Ignore
    $settings.XmlResolver = $null
    $reader = [Xml.XmlReader]::Create([IO.StringReader]::new((Smi @("--id=$Uuid", '-q', '-x'))), $settings)
    try { $doc = [Xml.XmlDocument]::new(); $doc.XmlResolver = $null; $doc.Load($reader) } finally { $reader.Dispose() }
    $gpus = @($doc.SelectNodes('/nvidia_smi_log/gpu'))
    if ($gpus.Count -ne 1 -or $gpus[0].uuid -ne $Uuid) { throw 'XML census GPU identity mismatch' }
    $containers = @($doc.SelectNodes('/nvidia_smi_log/gpu/processes'))
    if ($containers.Count -ne 1) { throw 'Process census structure unavailable' }
    $nodes = @($containers[0].SelectNodes('process_info'))
    if ($nodes.Count -eq 0 -and $containers[0].InnerText.Trim() -notin @('', 'None')) { throw 'Empty process census has unsupported status' }
    if ($nodes.Count -gt 256) { throw 'Process census exceeded 256-row bound' }
    $result = @()
    foreach ($node in $nodes) {
        $processIdValue = 0
        if (-not [int]::TryParse([string]$node.pid, [ref]$processIdValue) -or $processIdValue -le 0) { throw 'Malformed census PID' }
        $kind = [string]$node.type
        if ($kind -notin @('C','G','C+G','M','M+C','O')) { $issues.Add("Unknown process type for PID $processIdValue") }
        $result += @{ pid = $processIdValue; type = $kind; name = [string]$node.process_name; memory = [string]$node.used_memory }
    }
    return $result
}
function Read-Mapped-Counters([string]$Luid, [int]$Count, [bool]$ValidateValues) {
    if ($Count -notin @(1, 2)) { throw 'Counter sample count outside bound' }
    $result = @()
    # phys_N is a Windows adapter node, never an NVIDIA physical ordinal.
    foreach ($counter in @('\GPU Engine(*)\Utilization Percentage', '\GPU Process Memory(*)\Dedicated Usage', '\GPU Process Memory(*)\Shared Usage', '\GPU Adapter Memory(*)\Dedicated Usage')) {
        try {
            $sets = @(Get-Counter -Counter $counter -SampleInterval 1 -MaxSamples $Count -ErrorAction Stop)
            $rows = @($sets | ForEach-Object { $_.CounterSamples } | Where-Object { $_.InstanceName -match ('(^|_)' + [regex]::Escape($Luid) + '(_|$)') })
            if ($rows.Count -eq 0 -or $rows.Count -gt 2048) { throw 'Mapped counter rows missing or exceed bound' }
            foreach ($row in $rows) {
                $pidValue = $null
                if ($row.InstanceName -match '(^|_)pid_(\d+)(_|$)') {
                    $parsedPid = 0
                    if (-not [int]::TryParse($Matches[2], [ref]$parsedPid) -or $parsedPid -le 0) { throw 'Malformed mapped counter PID' }
                    $pidValue = $parsedPid
                } elseif ($counter -notlike '*GPU Adapter Memory*') { throw 'Mapped process counter PID unavailable' }
                $valid = Test-Counter-Value $row.Status $row.CookedValue
                if (-not $ValidateValues -and $row.Status -notin @(0, 1)) { $issues.Add("Counter discovery status unknown: $($row.Status)") }
                if ($ValidateValues -and -not $valid) { $issues.Add("Invalid mapped counter data: $counter, status $($row.Status)") }
                $value = $null
                if ($ValidateValues -and $valid) { $value = $row.CookedValue }
                $result += @{ instance = $row.InstanceName; counter = $counter; utc = $row.Timestamp.ToUniversalTime().ToString('o'); value = $value; rawValue = $row.CookedValue.ToString('R', [Globalization.CultureInfo]::InvariantCulture); status = $row.Status; pid = $pidValue; valuesValidated = $ValidateValues; valid = $valid }
            }
        } catch { $issues.Add("Mapped counter incomplete: $counter; $($_.Exception.Message)") }
    }
    return $result
}
Save-Record
try {
    $identity = Identity '1'
    $uuid = $identity.uuid.Trim(); $pci = $identity.pci.Trim()
    $record['identity'] = $identity
    if ($pci -notmatch '^00000000:([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])$') { throw 'Unsupported PCI domain or address syntax' }
    $pciArgs = @($Matches[1], $Matches[2], $Matches[3])
    try {
        $vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
        if (-not (Test-Path -PathType Leaf $vcvars)) { throw 'Installed MSVC environment unavailable' }
        $build = Join-Path $root 'host-helper'; New-Item -ItemType Directory -Path $build | Out-Null
        $source = Join-Path $PSScriptRoot 'windows-gpu1-luid.cpp'
        Copy-Item $source (Join-Path $build 'mapping.cpp')
        @('@echo off', "call `"$vcvars`" >nul", 'if errorlevel 1 exit /b 1', 'cl /nologo /std:c++17 /EHsc /W4 /D_WIN32_WINNT=0x0A00 mapping.cpp /Fe:mapping.exe /Fo:mapping.obj /link gdi32.lib', 'exit /b %errorlevel%') | Set-Content -Encoding ASCII (Join-Path $build 'build.cmd')
        Push-Location $build
        try { $compile = @(& cmd /d /c build.cmd 2>&1 | ForEach-Object { $_.ToString() }); $code = $LASTEXITCODE } finally { Pop-Location }
        $record['helperCompile'] = @{ exitCode = $code; output = @($compile | Select-Object -First 100); sourceSha256 = (Get-FileHash $source -Algorithm SHA256).Hash }
        if ($code -ne 0) { throw 'Host SDK mapper compile failed' }
        $mapped = @(& (Join-Path $build 'mapping.exe') @pciArgs)
        $mapCode = $LASTEXITCODE
        $record['helperQuery'] = @{ exitCode = $mapCode; output = @($mapped | Select-Object -First 4) }
        if ($mapCode -ne 0) { throw 'Host SDK mapper failed' }
        $map = ($mapped -join '') | ConvertFrom-Json
        if ($map.status -ne 'mapped' -or $map.counterLuid -notmatch '^luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}$') { throw 'Mapper did not return an exact LUID' }
        $record.mapping = $map
    } catch { $issues.Add('PCI to LUID mapping incomplete'); $record['mappingError'] = $_.Exception.Message }
    for ($sampleIndex = 0; $sampleIndex -lt 3; $sampleIndex++) {
        $sample = [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); counters = @() }
        try {
            $current = Identity $uuid
            if ($current.uuid.Trim() -ne $uuid -or $current.pci.Trim() -ne $pci) { throw 'GPU1 identity changed' }
            $sample['device'] = $current
            if ($identity.measurements.status -eq 'valid' -and $current.measurements.status -eq 'valid' -and $identity.measurements.totalMiB -ne $current.measurements.totalMiB) { $issues.Add('Device total memory changed during diagnostic') }
            $census = @(Census $uuid)
            $sample['processes'] = $census
            $discovery = @()
            if ($null -eq $record.mapping) { $issues.Add('Windows counter attribution unavailable') } else {
                # Discovery identifies the entire mapped cohort; it is not the measurement window.
                $discovery = @(Read-Mapped-Counters $record.mapping.counterLuid 1 $false)
            }
            $sample['counterDiscovery'] = $discovery
            $cohort = @(@($census | ForEach-Object { $_.pid }) + @($discovery | Where-Object { $null -ne $_.pid } | ForEach-Object { $_.pid }) | Sort-Object -Unique)
            if ($cohort.Count -gt 256) { throw 'Process identity cohort exceeded 256-PID bound' }
            $sample['identitiesBefore'] = @(Process-Identity $cohort)
            $sample['windowStartedUtc'] = [DateTime]::UtcNow.ToString('o')
            if ($null -ne $record.mapping) {
                $sample.counters = @(Read-Mapped-Counters $record.mapping.counterLuid 2 $true)
            }
            $sample['windowEndedUtc'] = [DateTime]::UtcNow.ToString('o')
            $afterCensus = @(Census $uuid)
            $sample['processesAfter'] = $afterCensus
            $observed = @(@($afterCensus | ForEach-Object { $_.pid }) + @($sample.counters | Where-Object { $null -ne $_.pid } | ForEach-Object { $_.pid }) | Sort-Object -Unique)
            foreach ($observedPid in $observed) {
                if ($observedPid -notin $cohort) { $issues.Add("New PID during counter window without pre-window identity: $observedPid") }
            }
            $processIds = @(@($cohort) + @($observed) | Sort-Object -Unique)
            if ($processIds.Count -gt 256) { throw 'Post-window identity cohort exceeded 256-PID bound' }
            $sample['identitiesAfter'] = @(Process-Identity $processIds)
            $beforeSet = @($census | ForEach-Object { "$($_.pid):$($_.type)" } | Sort-Object)
            $afterSet = @($afterCensus | ForEach-Object { "$($_.pid):$($_.type)" } | Sort-Object)
            if (($beforeSet -join ',') -ne ($afterSet -join ',')) { $issues.Add('GPU process census changed during sample') }
            foreach ($before in $sample.identitiesBefore) {
                $after = @($sample.identitiesAfter | Where-Object { $_.pid -eq $before.pid })
                if ($before.status -ne 'resolved' -or $after.Count -ne 1 -or $after[0].status -ne 'resolved') { continue }
                if ($before.createdUtc -ne $after[0].createdUtc -or $before.name -ne $after[0].name -or $before.path -ne $after[0].path) {
                    $issues.Add("Process identity changed for PID $($before.pid)")
                }
            }
        } catch { $issues.Add('GPU1 sample incomplete'); $sample['error'] = $_.Exception.Message }
        $record.samples += $sample
        Save-Record
    }
} catch { $issues.Add('GPU1 diagnostic incomplete'); $record['error'] = $_.Exception.Message }
$record['completedUtc'] = [DateTime]::UtcNow.ToString('o')
$record.completed = $true
$record['telemetryComplete'] = ($issues.Count -eq 0)
Save-Record
Write-Host "GPU1 evidence recorded; admission not evaluated; incomplete flags: $($issues.Count)"
if ($issues.Count -ne 0) { exit 1 }
