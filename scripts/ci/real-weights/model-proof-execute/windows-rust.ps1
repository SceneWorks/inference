$channel = (Select-String -Path rust-toolchain.toml -Pattern '^\s*channel\s*=\s*"([^"]+)"$' | Select-Object -First 1).Matches.Groups[1].Value
if (-not $channel) { throw 'Repository Rust toolchain pin is missing' }
$proofRustupHome = if ($env:RUSTUP_HOME) { $env:RUSTUP_HOME } else { Join-Path $env:USERPROFILE '.rustup' }
$bin = Join-Path $proofRustupHome "toolchains\${channel}-x86_64-pc-windows-msvc\bin"
foreach ($tool in @('cargo', 'rustc')) {
  if (-not (Test-Path -LiteralPath (Join-Path $bin "$tool.exe") -PathType Leaf)) {
    throw "Real pinned Windows toolchain is missing $tool.exe under $bin"
  }
}
Add-Content -Path $env:GITHUB_ENV -Value "MODEL_PROOF_RUST_BIN=$bin"
Add-Content -Path $env:GITHUB_PATH -Value $bin
