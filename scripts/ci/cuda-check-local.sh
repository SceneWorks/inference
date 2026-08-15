#!/usr/bin/env bash
# Reproduce the "Candle CUDA compile check (Windows)" lane on a machine with NO CUDA toolkit
# and NO NVIDIA GPU — including a Mac.
#
# WHY THIS EXISTS (sc-19556). Three of that lane's steps compile `--features cuda`, and a large
# amount of test code is gated behind it — whole files open with `#![cfg(feature = "cuda")]`
# (`candle-gen-svd/tests/real_weights_smoke.rs`, `candle-gen/tests/nvfp4_w4a4_ondevice_gpu.rs`,
# `candle-gen-sana/tests/depthwise_conv_gpu.rs`, and others). A local `cargo clippy` on macOS
# cannot pass `--features cuda`, so every one of those files is INVISIBLE to it: an edit there
# can be "verified" locally, look completely clean, and still fail the Windows lane.
#
# That is not hypothetical. It has now happened at least twice:
#   * sc-12379 — a `print_literal` in candle-gen-krea's cuda-gated `nvfp4_krea_dit_gpu.rs` sat red
#     on main until someone found it by hand.
#   * sc-19556 — deleting a degenerate assertion in `real_weights_smoke.rs` left
#     `ProfileOutput.wall` with no reader, which is `-D dead-code`. Six crates were clippied on
#     macOS before the push and none of them compiled the file the error was in.
#
# WHY IT WORKS WITHOUT A GPU. The CI lane itself is deliberately GPU-free: it runs
# `cargo test --no-run`, `cargo clippy`, and `cargo doc`, none of which create a CUDA context or
# allocate VRAM. They need nvcc only because dependency BUILD SCRIPTS shell out to it. Since
# nothing here links a kernel or runs one, a stub `nvcc` satisfies all three blockers:
#
#   1. `cudarc` build.rs        -> panics unless `nvcc --version` succeeds.
#   2. `candle-kernels` build.rs -> `ComputeCapDetectionFailed` unless CUDA_COMPUTE_CAP is set
#                                   (it otherwise shells `nvidia-smi`).
#   3. `candle-kernels`          -> shells `nvcc --ptx` per kernel, then `include_str!`s the result,
#                                   so the .ptx files must exist even if they are empty.
#
# The `--version` output below is load-bearing and cannot be reworded freely: cudarc's build script
# reads LINE INDEX 3 (the fourth line), splits it on ", " and takes [1], then splits THAT on " "
# and takes [1]. So the fourth line must parse to a release in cudarc's SUPPORTED_CUDA_VERSIONS.
#
# WHAT THIS DOES NOT DO. It type-checks; it does not RUN anything. `cargo test` (without
# --no-run) would link, and linking needs a real libcuda. Executing the cuda-gated `#[test]`s
# still requires the manual `windows-cuda` lane or a real device. That is fine — this script
# targets the compile/lint/doc failures, which are the ones that waste a shared-runner slot.
#
# GOTCHA that will cost you 20 minutes: if a candle-kernels build already failed in this target
# dir, cargo CACHES the build-script failure and replays stale "missing .ptx" errors without ever
# re-invoking nvcc — so the stub looks broken when it is working. This script clears those build
# dirs itself. If you run the cargo commands by hand instead, do that clearing yourself.
#
# USAGE
#   scripts/ci/cuda-check-local.sh              # clippy only (the fast, most common check)
#   scripts/ci/cuda-check-local.sh all          # clippy + compile (--no-run) + rustdoc, as CI runs them
set -euo pipefail

mode="${1:-clippy}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

stub_dir="$(mktemp -d)"
trap 'rm -rf "$stub_dir"' EXIT

cat > "$stub_dir/nvcc" <<'STUB'
#!/usr/bin/env bash
# Stub nvcc. Satisfies build scripts for check/clippy/doc, which never link or run a kernel.
for arg in "$@"; do
  if [ "$arg" = "--version" ]; then
    echo "nvcc: NVIDIA (R) Cuda compiler driver"
    echo "Copyright (c) 2005-2025 NVIDIA Corporation"
    echo "Built on Tue_May_27_02:21:03_PDT_2025"
    echo "Cuda compilation tools, release 12.9, V12.9.86"
    echo "Build cuda_12.9.r12.9/compiler.36037321_0"
    exit 0
  fi
done

out=""; outdir=""; src=""; prev=""
for arg in "$@"; do
  case "$prev" in
    -o) out="$arg" ;;
    --output-directory) outdir="$arg" ;;
  esac
  case "$arg" in *.cu) src="$arg" ;; esac
  prev="$arg"
done

# Emit empty artifacts so the later include_str!/link steps find a file.
if [ -n "$out" ]; then
  mkdir -p "$(dirname "$out")"
  : > "$out"
fi
if [ -n "$outdir" ] && [ -n "$src" ]; then
  base="$(basename "$src" .cu)"
  mkdir -p "$outdir"
  : > "$outdir/$base.ptx"
  : > "$outdir/$base.o"
fi
exit 0
STUB
chmod +x "$stub_dir/nvcc"

export PATH="$stub_dir:$PATH"
export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-90}"

# See the cached-build-script-failure gotcha above.
target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
rm -rf "$target_dir"/*/build/candle-kernels-* "$target_dir"/build/candle-kernels-* 2>/dev/null || true

# Keep this package set aligned with the three steps in ci.yml's `windows-cuda-check` job.
pkgs=(-p candle-llm -p "candle-gen*" -p "candle-audio*" -p runtime-cuda)

echo "==> clippy --features cuda (the lint twin of the Windows lane)"
cargo clippy --locked --all-targets "${pkgs[@]}" --features cuda -- -D warnings

if [ "$mode" = "all" ]; then
  echo "==> test --no-run --features cuda (compile only; never links a kernel)"
  cargo test --locked --lib --tests "${pkgs[@]}" --features cuda --no-run

  echo "==> doc --features cuda"
  RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps "${pkgs[@]}" --features cuda
fi

echo "OK: the cuda-gated code compiles and lints clean locally."
