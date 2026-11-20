RUSTUP_BIN="${CARGO_HOME:-$HOME/.cargo}/bin"
# GitHub-hosted macOS images install rustup with Homebrew (actions/runner-images
# install-rust.sh), so its proxies live in `brew --prefix rustup`/bin, not ~/.cargo/bin.
# The proxies still honour rust-toolchain.toml, so the pinned toolchain is unchanged.
if [[ ! -x "$RUSTUP_BIN/rustup" && "${RUNNER_OS:-}" == macOS
      && "${MODEL_PROOF_RUNNER_ENVIRONMENT:-}" == github-hosted ]]; then
  RUSTUP_BIN="$(brew --prefix rustup)/bin"
fi
for tool in cargo rustc rustup; do test -x "$RUSTUP_BIN/$tool"; done
printf '%s\n' "$RUSTUP_BIN" >> "$GITHUB_PATH"
