#!/usr/bin/env bash
# Build job stages for kv-poc-campaign.yml: W1 `trees`, `models`, `binaries`; W2 `w2-models`,
# `w2-inputs`, `w2-binaries` (`trees` is shared).
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

# The hash-locked huggingface_hub install (printed path), shared by W1 and W2.
hub_tools() {
  local lock="$GITHUB_WORKSPACE/.github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt" tools
  tools="$KV_TOOLS/hub-$(shasum -a 256 "$lock" | cut -c1-12)"
  if [ ! -f "$tools/.installed" ]; then
    rm -rf "$tools"
    # `|| return`: a command substitution does not inherit errexit.
    python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$tools" -r "$lock" >&2 \
      || return 1
    touch "$tools/.installed" || return 1
  fi
  printf '%s' "$tools"
}

stage_models() {
  local tools
  tools="$(hub_tools)"
  [ -d "$KV_HF_HUB" ] || { echo "::error title=no hub cache::$KV_HF_HUB does not exist on $(hostname)"; return 1; }
  HF_HUB_CACHE="$KV_HF_HUB" HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH="$tools" \
    python3.12 "$KV_DIR/models.py" --hub "$KV_HF_HUB" --pins "$KV_DIR/models.tsv"
  summary "- eight pinned snapshots present in \`$KV_HF_HUB\` at their pinned byte sizes"
}

# W2: every file in models-w2.tsv. Totals the missing bytes against the hub volume's free space
# FIRST and fails with the per-file list (downloading nothing) when they do not fit with
# $KV_W2_DISK_RESERVE_GIB (default 10) to spare; otherwise fetches only the missing pinned files,
# then verifies every pinned file's size AND sha256.
stage_w2_models() {
  local tools rc=0 report
  [ -d "$KV_HF_HUB" ] || { echo "::error title=no hub cache::$KV_HF_HUB does not exist on $(hostname)"; return 1; }
  [ -w "$KV_HF_HUB" ] || { echo "::error title=hub cache not writable::$KV_HF_HUB is not writable by $(id -un)"; return 1; }
  tools="$(hub_tools)"
  report="$(mktemp)"
  HF_HUB_CACHE="$KV_HF_HUB" HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH="$tools" \
    python3.12 "$KV_DIR/models.py" --hub "$KV_HF_HUB" --pins "$KV_DIR/models-w2.tsv" --only-pinned \
      --reserve-gib "${KV_W2_DISK_RESERVE_GIB:-10}" --verify-sha256 2>&1 | tee "$report" || rc=$?
  summary "### W2 model assets (\`models-w2.tsv\` on $(hostname))"
  summary '```'
  summary "$(grep -v '^::' "$report" | head -120)"
  summary '```'
  rm -f "$report"
  return "$rc"
}

# W2: the two worker-assembled VACE snapshots, the VACE control/mask fixtures, both verified.
stage_w2_inputs() {
  local route dir rel target problem
  python3.12 "$KV_DIR/models.py" --check-only --hub "$KV_HF_HUB" --pins "$KV_DIR/models-w2.tsv" \
    || { echo "::error title=W2 models missing::run the w2-models stage first"; return 1; }
  for route in wan_vace wan_vace_fun; do
    if problem="$(verify_w2_vace "$route")"; then
      echo "assembled $route verifies; reused"
      continue
    fi
    echo "(re)assembling $route: $problem"
    dir="$W2_VACE"; [ "$route" = wan_vace ] || dir="$W2_VACE_FUN"
    rm -rf "$dir.staging" "$dir"
    mkdir -p "$dir.staging"
    while read -r rel target; do
      mkdir -p "$(dirname "$dir.staging/$rel")"
      ln -s "$(realpath "$target")" "$dir.staging/$rel"
    done < <(w2_vace_layout "$route")
    if [ "$route" = wan_vace ]; then echo "$W2_VACE_REVISION"; else echo "$W2_VACE_FUN_REVISION"; fi > "$dir.staging/.snapshot-revision"
    mv "$dir.staging" "$dir"
    problem="$(verify_w2_vace "$route")" || { echo "::error title=VACE assembly failed::$route: $problem"; return 1; }
  done
  python3.12 "$KV_DIR/fixtures.py" generate --out "$W2_FIXTURES" --pins "$KV_DIR/fixtures-w2.tsv"
  summary "- assembled \`$W2_VACE\` + \`$W2_VACE_FUN\` (per-file links, \`.snapshot-revision\`); fixtures \`$W2_FIXTURES\` verified"
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

# W2: the two SC-20686 MLX entrypoints and the SC-20684 Krea observer (the crate's single
# `integration` test binary), built release against the prebuilt MLX Release, then sealed in $F2 with
# the metallib, the media policy and the reference images. A sealed $F2 that verifies is REUSED:
# both campaign parents bind the executable bytes into their resume identities.
stage_w2_binaries() {
  local b entry path sha metallib_sha krea
  # Its own job: (re)establish the tree idempotently (reused when present, no fetch) instead of
  # assuming w2-assets' clone is still there.
  ensure_clone "$INFERENCE_URL" "$INF" "$INFERENCE_SHA"
  (cd "$KV_DIR" && shasum -a 256 -c inputs.sha256)
  if [ -f "$F2/SHA256SUMS" ]; then
    verify_frozen "$F2"
    for b in $W2_BINS; do [ -x "$F2/$b" ] || { echo "::error title=frozen dir incomplete::$F2/$b missing"; return 1; }; done
    echo "reusing sealed W2 frozen dir $F2"
    summary "- reused the sealed W2 frozen dir \`$F2\` (no rebuild; resume identities stay valid)"
    return 0
  fi
  if [ -e "$F2" ]; then echo "discarding unsealed partial $F2"; chmod -R u+w "$F2"; rm -rf "$F2"; fi

  local prebuilt rc=0
  prebuilt="$(cd "$INF" && scripts/fetch-prebuilt-mlx.sh --build-type Release)" || rc=$?
  [ "$rc" = 0 ] || { echo "::error title=prebuilt MLX unavailable::fetch-prebuilt-mlx.sh exited $rc; the frozen campaign build requires the published Release prebuilt"; return 1; }
  eval "$prebuilt"
  export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH
  export CARGO_TARGET_DIR="$KV_TARGET"
  (cd "$INF" && rustc --version && cargo --version \
    && cargo build --locked --release -p mlx-gen-wan --example sc20686_wan -p mlx-gen-flux2 --example sc20686_flux2_edit)
  # `--no-run` prints the test executable on the compiler-artifact message of the `integration` target.
  krea="$(cd "$INF" && cargo test --locked --release -p mlx-gen-krea-realtime --test integration --no-run --message-format=json \
    | python3.12 -c 'import json, sys
found = [m["executable"] for m in map(json.loads, filter(str.strip, sys.stdin))
         if m.get("reason") == "compiler-artifact" and m.get("target", {}).get("name") == "integration"
         and m.get("profile", {}).get("test") and m.get("executable")]
print(found[-1] if len(found) == 1 else "")')"
  [ -n "$krea" ] && [ -x "$krea" ] || { echo "::error title=krea observer not found::cargo test --no-run printed no single integration executable"; return 1; }
  verify_tree "$INF" "$INFERENCE_URL" "$INFERENCE_SHA"

  mkdir -p "$F2/policies" "$F2/inputs"
  cp "$CARGO_TARGET_DIR/release/examples/sc20686_wan" "$CARGO_TARGET_DIR/release/examples/sc20686_flux2_edit" "$F2/"
  cp "$krea" "$F2/krea-integration"
  # A here-string, not a pipe: `grep -q` exiting early would SIGPIPE the lister under pipefail.
  grep -qx "$W2_KREA_OBSERVER: test" <<< "$("$F2/krea-integration" --list --ignored)" \
    || { echo "::error title=krea observer missing::$krea does not list $W2_KREA_OBSERVER as an ignored test"; return 1; }
  cp "$PMETAL_METALLIB_PATH" "$F2/mlx.metallib"
  cp "$KV_DIR/policies/$W2_POLICY" "$F2/policies/"
  while IFS= read -r entry; do
    path="${entry%%:*}"; sha="${entry##*:}"
    cp "$INF/$path" "$F2/inputs/"
    [ "$(shasum -a 256 "$F2/inputs/$(basename "$path")" | cut -d' ' -f1)" = "$sha" ] \
      || { echo "::error title=reference image drift::$path is not sha256 $sha at $INFERENCE_SHA"; return 1; }
  done <<< "$W2_REFERENCES"
  metallib_sha="$(shasum -a 256 "$F2/mlx.metallib" | cut -d' ' -f1)"
  {
    echo "inference=$INFERENCE_SHA"
    echo "inference_tree=$INF"
    echo "krea_observer_source=$krea"
    echo "prebuilt=$PMETAL_MLX_PREBUILT_DIR"
    echo "metallib_sha256=$metallib_sha"
    echo "rustc=$(cd "$INF" && rustc --version)"
    echo "runner=${RUNNER_NAME:-?}"
    echo "workflow_run=${GITHUB_RUN_ID:-?}"
    echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$F2/BUILD-INFO.txt"
  (cd "$F2" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS)
  chmod -R a-w "$F2"
  verify_frozen "$F2"
  cat "$F2/SHA256SUMS"
  summary "- built and sealed \`$F2\` (metallib \`${metallib_sha:0:12}\`)"
  summary '```'
  summary "$(cat "$F2/SHA256SUMS")"
  summary '```'
}

case "${1:-}" in
  trees) stage_trees ;;
  models) stage_models ;;
  binaries) stage_binaries ;;
  w2-models) stage_w2_models ;;
  w2-inputs) stage_w2_inputs ;;
  w2-binaries) stage_w2_binaries ;;
  *) echo "usage: build.sh trees|models|binaries|w2-models|w2-inputs|w2-binaries" >&2; exit 2 ;;
esac
