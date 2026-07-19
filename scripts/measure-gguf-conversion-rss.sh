#!/usr/bin/env bash
# Reproducible peak-RSS probe for the tensor-at-a-time GGUF converter.
# macOS: maximum resident set size is reported in bytes by /usr/bin/time -l.
# Linux: /usr/bin/time -v reports "Maximum resident set size" in KiB.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <model.gguf> <empty-output-dir> [convert_gguf options...]" >&2
  exit 2
fi

gguf=$1
out=$2
shift 2

if [[ -e "$out" ]]; then
  echo "output must not already exist: $out" >&2
  exit 2
fi

case "$(uname -s)" in
  Darwin) exec /usr/bin/time -l cargo run --release --locked -p mlx-llm --example convert_gguf -- "$gguf" "$out" "$@" ;;
  Linux) exec /usr/bin/time -v cargo run --release --locked -p mlx-llm --example convert_gguf -- "$gguf" "$out" "$@" ;;
  *) echo "unsupported platform for RSS measurement: $(uname -s)" >&2; exit 2 ;;
esac
