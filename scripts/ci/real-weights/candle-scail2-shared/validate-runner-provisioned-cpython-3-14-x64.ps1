$log = Join-Path $env:RUNNER_TEMP "scail2-shared-cuda.log"
$python = python -c "import platform,struct,sys; print('|'.join((sys.executable,sys.implementation.name,platform.python_version(),platform.machine(),str(struct.calcsize('P') * 8))))"
if ($LASTEXITCODE -ne 0 -or $python -notmatch '\|cpython\|3\.14\.\d+\|AMD64\|64$') {
  "python_validation=failed value=$python" | Add-Content -LiteralPath $log -Encoding utf8
  throw "expected runner-provisioned CPython 3.14 x64, got $python"
}
"python_validation=complete value=$python" | Add-Content -LiteralPath $log -Encoding utf8
