if [[ "$RUNNER_OS" == Windows ]]; then
  printf 'Rust bin: %s\nCargo cache: %s\n' "$MODEL_PROOF_RUST_BIN" "$CARGO_HOME"
  command -v cargo
  command -v rustc
else
  rustup show
fi
cargo --version --verbose
rustc -vV
hostname
df -h .
