RUSTUP_BIN="${CARGO_HOME:-$HOME/.cargo}/bin"
for tool in cargo rustc rustup; do test -x "$RUSTUP_BIN/$tool"; done
printf '%s\n' "$RUSTUP_BIN" >> "$GITHUB_PATH"
