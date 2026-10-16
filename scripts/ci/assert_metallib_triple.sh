#!/usr/bin/env bash
# Fail unless every mlx.metallib under a cargo target directory was compiled for one expected
# Apple Metal target triple (sc-16779, epic 16774).
#
# Usage: scripts/ci/assert_metallib_triple.sh <target-dir> <expected-triple>
#   e.g. scripts/ci/assert_metallib_triple.sh target/aarch64-apple-ios air64-apple-ios18.0.0
#
# The AIR version infix (`air64_v28-...`, emitted by newer Metal toolchains; Xcode 16's is bare
# `air64-...`) is normalized away before comparing: it names the toolchain, not the platform.
#
# A green iOS cross-build is NOT evidence of an iOS artifact: the failure mode is a build that
# succeeds while MLX's kernels were compiled with `xcrun -sdk macosx`, which only surfaces at
# `newLibraryWithURL` on a device. Device and simulator differ only in the `-simulator` suffix.
#
# The check is the one the mlx-rs fork's `.github/workflows/cross-apple.yml` runs on its own
# pmetal-mlx-sys build: the AIR objects record their full target triple and it survives into the
# metallib. Its pattern is widened from (ios|tvos) to any Apple platform so a macOS kernel is
# NAMED in the error instead of reported as "no triple", and every metallib found is checked, not
# just the first, so a stale second build-script output cannot hide behind a good one.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <target-dir> <expected-triple>" >&2
  exit 2
fi
target_dir=$1
expected=$2

metallibs=()
while IFS= read -r -d '' path; do
  metallibs+=("$path")
done < <(find "$target_dir" -name mlx.metallib -path '*/build/lib/*' -print0 2>/dev/null)

if [ ${#metallibs[@]} -eq 0 ]; then
  echo "::error::no mlx.metallib was produced under $target_dir"
  exit 1
fi

status=0
for metallib in "${metallibs[@]}"; do
  # `|| true` so an empty result is reported by the comparison below rather than as a bare grep
  # exit code under `pipefail`.
  found=$(LC_ALL=C strings -a "$metallib" \
          | grep -oE 'air64(_v[0-9]+)?-apple-[a-z]+[0-9]+(\.[0-9]+){2}(-simulator)?' \
          | sed -E 's/^air64_v[0-9]+-/air64-/' \
          | sort -u | tr '\n' ' ' | sed 's/ $//' || true)
  echo "metallib: $metallib"
  echo "triples:  ${found:-<none>}"
  if [ "$found" != "$expected" ]; then
    echo "::error::$metallib targets '${found:-<none>}', expected exactly '$expected'"
    status=1
  fi
done
exit "$status"
