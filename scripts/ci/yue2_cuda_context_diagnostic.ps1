# Read-only Windows/NVIDIA evidence for an ambiguous C+G desktop context.
# This script does not decide whether the host is idle and must never be used
# as the ownership gate. cuInit below initializes the CUDA driver only; no
# context acquisition, allocation, kernel, model, or test is performed.
param(
    # Zero is the process-free GPU0 route. It still collects the same 29 raw
    # files, including unfiltered Windows counters for the mapped adapter.
    [Parameter(Mandatory = $true)][ValidateRange(0, 2147483647)][int]$TargetPid,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [Parameter(Mandatory = $true)][string]$EngineSha,
    [Parameter(Mandatory = $true)][string]$ControlSha
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -PathType Container $OutputDirectory)) { throw 'output directory does not exist' }
if ($EngineSha -cnotmatch '^[0-9a-f]{40}$' -or $ControlSha -cnotmatch '^[0-9a-f]{40}$') { throw 'invalid source SHA' }

function Save-Json($Name, $Value) {
    $Value | ConvertTo-Json -Depth 12 | Out-File -FilePath (Join-Path $OutputDirectory $Name) -Encoding utf8
}
function Invoke-Smi($Name, [string[]]$Arguments) {
    $at = (Get-Date).ToUniversalTime().ToString('o')
    try {
        $lines = & nvidia-smi @Arguments 2>&1 | ForEach-Object { $_.ToString() }
        $code = $LASTEXITCODE
        Save-Json "$Name.json" @{ utc = $at; argv = @('nvidia-smi') + $Arguments; exitCode = $code; output = @($lines) }
    } catch {
        Save-Json "$Name.json" @{ utc = $at; argv = @('nvidia-smi') + $Arguments; error = $_.Exception.Message }
    }
}
function Save-ProcessIdentity($Name) {
    $at = (Get-Date).ToUniversalTime().ToString('o')
    try {
        if ($TargetPid -eq 0) { Save-Json "$Name.json" @{ utc = $at; pid = 0; status = 'no-target-process' }; return }
        $item = Get-CimInstance Win32_Process -Filter "ProcessId = $TargetPid" -ErrorAction Stop
        if ($null -eq $item) { Save-Json "$Name.json" @{ utc = $at; pid = $TargetPid; status = 'not_found' }; return }
        $signature = $null
        if ($item.ExecutablePath) {
            try {
                $sig = Get-AuthenticodeSignature -FilePath $item.ExecutablePath -ErrorAction Stop
                $signature = @{ status = [string]$sig.Status; signerSubject = $sig.SignerCertificate.Subject; signerThumbprint = $sig.SignerCertificate.Thumbprint }
            } catch { $signature = @{ error = $_.Exception.Message } }
        }
        Save-Json "$Name.json" @{ utc = $at; pid = $TargetPid; name = $item.Name; executablePath = $item.ExecutablePath; creationDate = [string]$item.CreationDate; signature = $signature }
    } catch { Save-Json "$Name.json" @{ utc = $at; pid = $TargetPid; error = $_.Exception.Message } }
}
function Save-Counters($Name) {
    $at = (Get-Date).ToUniversalTime().ToString('o')
    $patterns = @('\GPU Engine(*)\Utilization Percentage',
                  '\GPU Process Memory(*)\Dedicated Usage', '\GPU Process Memory(*)\Shared Usage',
                  '\GPU Process Memory(*)\Local Usage', '\GPU Process Memory(*)\Non Local Usage',
                  '\GPU Process Memory(*)\Total Committed', '\GPU Adapter Memory(*)\Dedicated Usage',
                  '\GPU Adapter Memory(*)\Shared Usage', '\GPU Adapter Memory(*)\Total Committed')
    $results = @()
    foreach ($pattern in $patterns) {
        try {
            $set = Get-Counter -Counter $pattern -SampleInterval 1 -MaxSamples 1 -ErrorAction Stop
            $samples = @($set.CounterSamples | Where-Object {
                $TargetPid -eq 0 -or $pattern -like '*GPU Adapter Memory*' -or $_.InstanceName -match "(^|_)pid_$TargetPid(_|$)"
            } | ForEach-Object { @{ path = $_.Path; instance = $_.InstanceName; cookedValue = $_.CookedValue; status = [string]$_.Status } })
            $results += @{ counter = $pattern; timestamp = [string]$set.Timestamp; samples = $samples }
        } catch { $results += @{ counter = $pattern; error = $_.Exception.Message } }
    }
    Save-Json "$Name.json" @{ utc = $at; targetPid = $TargetPid; counters = $results }
}
function Save-CounterCatalog {
    $at = (Get-Date).ToUniversalTime().ToString('o')
    $sets = @()
    foreach ($name in @('GPU Engine', 'GPU Process Memory', 'GPU Adapter Memory')) {
        try {
            $set = Get-Counter -ListSet $name -ErrorAction Stop
            $instances = @($set.PathsWithInstances | Where-Object {
                $TargetPid -eq 0 -or $name -eq 'GPU Adapter Memory' -or $_ -match "(^|_)pid_$TargetPid(_|$)"
            })
            $sets += @{ name = $name; paths = @($set.Paths); targetOrAdapterInstances = $instances }
        } catch { $sets += @{ name = $name; error = $_.Exception.Message } }
    }
    Save-Json 'windows-counter-catalog.json' @{ utc = $at; targetPid = $TargetPid; sets = $sets }
}

# CUDA Driver property APIs expose PCI bus ID and Windows adapter LUID for the
# same CUdevice. CUDAAPI uses stdcall on Windows. Retain raw values and errors;
# no missing property or counter is interpreted as an idle device.
function Save-CudaAdapterMap {
    $at = (Get-Date).ToUniversalTime().ToString('o')
    try {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class Yue2CudaAdapterProperties {
    [DllImport("nvcuda.dll", CallingConvention = CallingConvention.StdCall)] public static extern int cuInit(uint flags);
    [DllImport("nvcuda.dll", CallingConvention = CallingConvention.StdCall)] public static extern int cuDeviceGetCount(out int count);
    [DllImport("nvcuda.dll", CallingConvention = CallingConvention.StdCall)] public static extern int cuDeviceGet(out int device, int ordinal);
    [DllImport("nvcuda.dll", CallingConvention = CallingConvention.StdCall)] public static extern int cuDeviceGetPCIBusId(StringBuilder pciBusId, int length, int device);
    [DllImport("nvcuda.dll", CallingConvention = CallingConvention.StdCall)] public static extern int cuDeviceGetLuid([Out] byte[] luid, out uint deviceNodeMask, int device);
}
'@ -ErrorAction Stop
        $init = [Yue2CudaAdapterProperties]::cuInit(0)
        $result = @{ utc = $at; cuInit = $init; driverInitializationOnly = $true; devices = @() }
        if ($init -eq 0) {
            $count = 0
            $result.cuDeviceGetCount = [Yue2CudaAdapterProperties]::cuDeviceGetCount([ref]$count)
            if ($result.cuDeviceGetCount -eq 0) {
                for ($ordinal = 0; $ordinal -lt $count; $ordinal++) {
                    $device = 0
                    $get = [Yue2CudaAdapterProperties]::cuDeviceGet([ref]$device, $ordinal)
                    $row = @{ ordinal = $ordinal; cuDeviceGet = $get }
                    if ($get -eq 0) {
                        $bus = New-Object System.Text.StringBuilder 32
                        $luid = New-Object byte[] 8
                        $mask = [uint32]0
                        $row.cuDeviceGetPCIBusId = [Yue2CudaAdapterProperties]::cuDeviceGetPCIBusId($bus, 32, $device)
                        $row.cuDeviceGetLuid = [Yue2CudaAdapterProperties]::cuDeviceGetLuid($luid, [ref]$mask, $device)
                        if ($row.cuDeviceGetPCIBusId -eq 0) { $row.pciBusId = $bus.ToString() }
                        if ($row.cuDeviceGetLuid -eq 0) { $row.luidBytes = [BitConverter]::ToString($luid); $row.nodeMask = $mask }
                    }
                    $result.devices += $row
                }
            }
        }
        Save-Json 'cuda-adapter-map.json' $result
    } catch { Save-Json 'cuda-adapter-map.json' @{ utc = $at; error = $_.Exception.Message; driverInitializationOnly = $true } }
}

$started = (Get-Date).ToUniversalTime().ToString('o')
Save-Json 'manifest.json' @{ schemaVersion = 1; purpose = 'diagnostic only, no idle verdict'; engineSha = $EngineSha; controlSha = $ControlSha; runner = $env:RUNNER_NAME; targetPid = $TargetPid; startedUtc = $started; completed = $false }
Save-ProcessIdentity 'process-before'
Save-CounterCatalog
for ($i = 0; $i -lt 3; $i++) {
    Invoke-Smi "gpu-sample-$i" @('--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory', '--format=csv,noheader,nounits')
    Invoke-Smi "driver-mode-$i" @('--query-gpu=index,uuid,driver_model.current,display_active', '--format=csv,noheader')
    foreach ($gpu in @(0, 1)) {
        Invoke-Smi "pmon-$gpu-$i" @('pmon', '-i', [string]$gpu, '-c', '1', '-s', 'um')
        Invoke-Smi "compute-apps-$gpu-$i" @('-i', [string]$gpu, '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader')
    }
    Save-Counters "windows-counters-$i"
    if ($i -lt 2) { Start-Sleep -Seconds 2 }
}
Invoke-Smi 'gpu-before-cuda-properties' @('--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu', '--format=csv,noheader,nounits')
Save-CudaAdapterMap
Invoke-Smi 'gpu-after-cuda-properties' @('--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu', '--format=csv,noheader,nounits')
Invoke-Smi 'pmon-0-final' @('pmon', '-i', '0', '-c', '1', '-s', 'um')
Save-ProcessIdentity 'process-after'
Save-Json 'manifest.json' @{ schemaVersion = 1; purpose = 'diagnostic only, no idle verdict'; engineSha = $EngineSha; controlSha = $ControlSha; runner = $env:RUNNER_NAME; targetPid = $TargetPid; startedUtc = $started; completedUtc = (Get-Date).ToUniversalTime().ToString('o'); completed = $true }
