$ErrorActionPreference = 'Stop'
$target = Join-Path $env:RUNNER_TEMP "moss-edit-target-$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT"
if (Test-Path -LiteralPath $target) { throw 'Target directory already exists' }
New-Item -ItemType Directory -Path $target | Out-Null
"CARGO_TARGET_DIR=$target" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
