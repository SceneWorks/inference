#!/usr/bin/env bash
# Build job stages for kv-poc-campaign.yml: `trees`, `models`, `binaries`.
#
# Reproduces the dev-Mac W1 prep (prep-w1-env.sh + the P7 seal) on the runner: prebuilt MLX
# Release, `cargo build --locked --release -p mlx-llm` for the four campaign bins, bins +
# mlx.metallib + the exact policy/prompt bytes copied into a frozen dir, SHA256SUMS, chmod a-w.
# A frozen dir that already verifies is REUSED, never rebuilt: resume identities bind the
# executable sha256, so a rebuild would orphan every resume dir already started.
set -euo pipefail
# shellcheck source=.github/kv-poc/common.sh
source "$(dirname "$0")/common.sh"

# The dev-Mac frozen W1 build's metallib (prebuilt 105a72fd7840 Release). A mismatch is a notice
# for the reader, not a failure: fetch-prebuilt-mlx.sh already verifies the asset's own sha256.
REFERENCE_METALLIB_SHA256=09336da65b3e8fc60e9d07e9ecec1115df638e42b8b20ce8706d83946539879d

# Put <dir> at exactly <sha> from <url>: a persistent shallow clone, fetched only when missing.
ensure_clone() {
  local url="$1" dir="$2" sha="$3" attempt
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" config --get remote.origin.url 2>/dev/null)" = "$url" ]; then
    echo "reusing clone $dir"
  else
    [ ! -e "$dir" ] || { chmod -R u+w "$dir"; rm -rf "$dir"; }
    mkdir -p "$dir"
    git init --quiet "$dir"
    git -C "$dir" remote add origin "$url"
  fi
  if ! git -C "$dir" cat-file -e "${sha}^{commit}" 2>/dev/null; then
    for attempt in 1 2 3; do
      git -C "$dir" fetch --quiet --depth 1 --no-tags origin "$sha" && break
      [ "$attempt" = 3 ] && { echo "::error title=fetch failed::$url $sha"; return 1; }
      sleep $(( attempt * 15 ))
    done
  fi
  if [ "$(git -C "$dir" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$sha" ] \
    || [ -n "$(git -C "$dir" status --porcelain)" ]; then
    git -C "$dir" -c advice.detachedHead=false checkout --quiet --force --detach "$sha"
    git -C "$dir" clean -ffdxq
  fi
  verify_tree "$dir" "$url" "$sha"
  echo "$dir = $sha (clean)"
}

stage_trees() {
  mkdir -p "$KV_ROOT/$INFERENCE_SHA"
  ensure_clone "$INFERENCE_URL" "$INF" "$INFERENCE_SHA"
  ensure_clone "$SCENEWORKS_URL" "$SW" "$SCENEWORKS_SHA"
  summary "- inference \`$INFERENCE_SHA\` at \`$INF\`; SceneWorks \`$SCENEWORKS_SHA\` at \`$SW\` (both clean)"
}

stage_models() {
  local lock="$GITHUB_WORKSPACE/.github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt" tools
  tools="$KV_TOOLS/hub-$(shasum -a 256 "$lock" | cut -c1-12)"
  if [ ! -f "$tools/.installed" ]; then
    rm -rf "$tools"
    python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$tools" -r "$lock"
    touch "$tools/.installed"
  fi
  [ -d "$KV_HF_HUB" ] || { echo "::error title=no hub cache::$KV_HF_HUB does not exist on $(hostname)"; return 1; }
  HF_HUB_CACHE="$KV_HF_HUB" HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH="$tools" \
    python3.12 "$KV_DIR/models.py" --hub "$KV_HF_HUB" --pins "$KV_DIR/models.tsv"
  summary "- four pinned snapshots present in \`$KV_HF_HUB\` at their pinned byte sizes"
}

stage_binaries() {
  local b metallib_sha
  verify_tree "$INF" "$INFERENCE_URL" "$INFERENCE_SHA"
  (cd "$KV_DIR" && shasum -a 256 -c inputs.sha256)
  if [ -f "$F/SHA256SUMS" ]; then
    verify_frozen
    for b in $BINS; do [ -x "$F/$b" ] || { echo "::error title=frozen dir incomplete::$F/$b missing"; return 1; }; done
    echo "reusing sealed frozen dir $F"
    summary "- reused the sealed frozen dir \`$F\` (no rebuild; resume identities stay valid)"
    return 0
  fi
  if [ -e "$F" ]; then echo "discarding unsealed partial $F"; chmod -R u+w "$F"; rm -rf "$F"; fi

  local prebuilt rc=0
  prebuilt="$(cd "$INF" && scripts/fetch-prebuilt-mlx.sh --build-type Release)" || rc=$?
  [ "$rc" = 0 ] || { echo "::error title=prebuilt MLX unavailable::fetch-prebuilt-mlx.sh exited $rc; the frozen campaign build requires the published Release prebuilt"; return 1; }
  eval "$prebuilt"
  export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH
  export CARGO_TARGET_DIR="$KV_TARGET"
  local bin_args=""
  for b in $BINS; do bin_args="$bin_args --bin $b"; done
  # shellcheck disable=SC2086  # deliberate word splitting of the --bin list
  (cd "$INF" && rustc --version && cargo --version && cargo build --locked --release -p mlx-llm $bin_args)
  verify_tree "$INF" "$INFERENCE_URL" "$INFERENCE_SHA"

  mkdir -p "$F/policies" "$F/inputs"
  for b in $BINS; do cp "$CARGO_TARGET_DIR/release/$b" "$F/$b"; done
  cp "$PMETAL_METALLIB_PATH" "$F/mlx.metallib"
  cp "$KV_DIR/policies/llm.json" "$KV_DIR/policies/capture.json" "$F/policies/"
  cp "$KV_DIR/inputs/prompt.txt" "$F/inputs/"
  metallib_sha="$(shasum -a 256 "$F/mlx.metallib" | cut -d' ' -f1)"
  if [ "$metallib_sha" = "$REFERENCE_METALLIB_SHA256" ]; then
    echo "::notice title=metallib::identical to the dev-Mac frozen W1 build's mlx.metallib"
  else
    echo "::warning title=metallib differs::$metallib_sha != dev-Mac W1 $REFERENCE_METALLIB_SHA256"
  fi
  {
    echo "inference=$INFERENCE_SHA"
    echo "sceneworks=$SCENEWORKS_SHA"
    echo "inference_tree=$INF"
    echo "prebuilt=$PMETAL_MLX_PREBUILT_DIR"
    echo "metallib_sha256=$metallib_sha"
    echo "rustc=$(cd "$INF" && rustc --version)"
    echo "runner=${RUNNER_NAME:-?}"
    echo "workflow_run=${GITHUB_RUN_ID:-?}"
    echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$F/BUILD-INFO.txt"
  (cd "$F" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS)
  chmod -R a-w "$F"
  verify_frozen
  cat "$F/SHA256SUMS"
  summary "- built and sealed \`$F\` (metallib \`${metallib_sha:0:12}\`)"
  summary '```'
  summary "$(cat "$F/SHA256SUMS")"
  summary '```'
}

case "${1:-}" in
  trees) stage_trees ;;
  models) stage_models ;;
  binaries) stage_binaries ;;
  *) echo "usage: build.sh trees|models|binaries" >&2; exit 2 ;;
esac
