# Size of the weight files under a snapshot directory, in whole MiB (rounded up). Links (the HF
# cache's snapshot entries) are followed to the file they point at.
param([Parameter(Mandatory=$true)][string]$Dir)
$sum = 0
Get-ChildItem -LiteralPath $Dir -Recurse -File | ForEach-Object {
  $t = @($_.Target)[0]
  if ($t) { $sum += (Get-Item -LiteralPath ([IO.Path]::GetFullPath([IO.Path]::Combine($_.DirectoryName, $t)))).Length } else { $sum += $_.Length }
}
[math]::Ceiling($sum / 1MB)
