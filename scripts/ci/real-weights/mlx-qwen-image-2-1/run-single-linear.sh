#!/usr/bin/env bash
set -euo pipefail
[[ "$RUNNER_NAME" == nax-macos-2 && "$GITHUB_RUN_ATTEMPT" == 1 ]]
[[ ! -e "$QWEN_SINGLE_LINEAR_ROOT" && ! -e "$CARGO_TARGET_DIR" ]]
[[ "$(git rev-parse HEAD)" == "$GITHUB_SHA" && -z "$(git status --porcelain)" ]]
mkdir "$QWEN_SINGLE_LINEAR_ROOT"
base=004cee41f380ff263796ad90277e5f6224d73ae3
# Full source diff is retained. Protected production and patch paths must equal exact M004.
git diff --name-status "$base" HEAD > "$QWEN_SINGLE_LINEAR_ROOT/source-diff.txt"
git diff --exit-code "$base" HEAD -- crates/media/mlx-gen/src crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/adapters.rs crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/transformer.rs crates/media/mlx-gen/mlx-gen-qwen-image-2-1/src/training.rs Cargo.toml Cargo.lock patches
git rev-parse HEAD HEAD^{tree} > "$QWEN_SINGLE_LINEAR_ROOT/source-head-tree.txt"
script=scripts/ci/real-weights/mlx-qwen-image-2-1/single_linear.py
fixture=crates/media/mlx-gen/mlx-gen-qwen-image-2-1/tests/fixtures/single-linear-candidate2
snapshot=/Users/MTrefry/sceneworks-rw-weights/hub/models--SceneWorks--qwen-image-2-1-mlx/snapshots/1691de01c24a070131e0a28bf4c065fd027f4fe9
# Only selected header/index and three exact payload ranges are read. No downloads/model load.
python3.12 "$script" prepare --fixture "$fixture" --snapshot "$snapshot" --out "$QWEN_SINGLE_LINEAR_ROOT/prepared" > "$QWEN_SINGLE_LINEAR_ROOT/prepared.sha256"
# No prebuilt override: actual compiled source must be present and hash-bound before first tensor.
unset PMETAL_MLX_PREBUILT_DIR MLX_ENABLE_TF32
cargo test --locked --release -p mlx-gen-qwen-image-2-1 --lib --no-run --message-format=json -j2 2>&1 | tee "$QWEN_SINGLE_LINEAR_ROOT/build.jsonl"
python3.12 scripts/ci/qwen21_mlx_build_identity.py --messages "$QWEN_SINGLE_LINEAR_ROOT/build.jsonl" --test-target lib --out "$QWEN_SINGLE_LINEAR_ROOT/build-identity.json"
python3.12 "$script" verify-build --target "$CARGO_TARGET_DIR" --out "$QWEN_SINGLE_LINEAR_ROOT/compiled-source.json" > "$QWEN_SINGLE_LINEAR_ROOT/compiled-source.sha256"
binary=$(python3.12 -c 'import json,os;print(json.load(open(os.environ["QWEN_SINGLE_LINEAR_ROOT"]+"/build-identity.json"))["libTestExecutable"]["path"])')
archive=$(python3.12 -c 'import json,os;print(next(x["path"] for x in json.load(open(os.environ["QWEN_SINGLE_LINEAR_ROOT"]+"/build-identity.json"))["archives"] if x["path"].endswith("/libmlx.a")))')
nm -g "$archive" > "$QWEN_SINGLE_LINEAR_ROOT/linked-symbols.txt"
# Rust cfg(test) binding names this actual existing private C++ symbol; no production API change.
grep -E ' T __ZN3mlx4core5metal16is_nax_availableEv$' "$QWEN_SINGLE_LINEAR_ROOT/linked-symbols.txt"
export QWEN_SINGLE_LINEAR_INPUT="$QWEN_SINGLE_LINEAR_ROOT/prepared"
export QWEN_SINGLE_LINEAR_PREPARED_SHA=$(cat "$QWEN_SINGLE_LINEAR_ROOT/prepared.sha256")
export QWEN_SINGLE_LINEAR_BUILD_PROOF="$QWEN_SINGLE_LINEAR_ROOT/compiled-source.json"
export QWEN_SINGLE_LINEAR_BUILD_PROOF_SHA=$(cat "$QWEN_SINGLE_LINEAR_ROOT/compiled-source.sha256")
selector=single_linear_diagnostic::diagnostic_current_q4_single_linear
# Each process independently admits <=1GiB active plus measured overhead and <=64MiB free cache,
# with unchanged full physical/host pressure guards and an additional <=4GiB diagnostic bound.
for mode in default strict; do
  export QWEN_SINGLE_LINEAR_MODE="$mode"
  export QWEN_SINGLE_LINEAR_OUTPUT="$QWEN_SINGLE_LINEAR_ROOT/$mode"
  if [[ "$mode" == strict ]]; then export MLX_ENABLE_TF32=0; else unset MLX_ENABLE_TF32; fi
  set +e
  "$binary" --ignored --exact "$selector" --nocapture --test-threads=1 2>&1 | tee "$QWEN_SINGLE_LINEAR_ROOT/$mode.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$QWEN_SINGLE_LINEAR_ROOT/$mode.exit-code"
  [[ "$rc" == 0 ]]
done
# CPU oracle runs on the independently collected packet; no numpy install or model work on Mac2.
