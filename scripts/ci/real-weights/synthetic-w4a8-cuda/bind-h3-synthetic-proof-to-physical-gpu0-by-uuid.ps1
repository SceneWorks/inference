$ErrorActionPreference = 'Stop'
$identity = @(& nvidia-smi -i 0 --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used --format=csv,noheader,nounits)
if ($LASTEXITCODE -ne 0) { throw 'Failed to query physical GPU0 identity' }
$identity = @($identity | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($identity.Count -ne 1) { throw 'Physical GPU0 identity is missing or ambiguous' }
$fields = @($identity[0].Split(',') | ForEach-Object { $_.Trim() })
if ($fields.Count -ne 7 -or $fields[0] -ne '0' -or $fields[1] -notmatch '^GPU-[0-9a-fA-F-]+$') {
  throw "Invalid physical GPU0 identity: $($identity[0])"
}
$gpuUuid = $fields[1]
Write-Host "H3 physical GPU0 identity (index, UUID, name, PCI bus, driver, total MiB, used MiB): $($identity[0])"
"CUDA_VISIBLE_DEVICES=$gpuUuid" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
'CUDA_DEVICE_ORDER=PCI_BUS_ID' | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
'SCENEWORKS_GPU_ID=0' | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
Write-Host "H3 logical CUDA device 0 maps only to physical GPU0 UUID $gpuUuid"
