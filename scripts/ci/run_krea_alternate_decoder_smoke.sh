#!/usr/bin/env bash
# SC-18315 correctness-only Metal smoke for Krea's native and Wan z16 terminal decoders.
#
# This deliberately emits only images, immutable-input provenance, and content hashes. It does not
# sample runtime memory, latency, GPU counters, or calibration data. The caller must provide the
# runner-local Hugging Face cache root and a fresh output directory.

set -euo pipefail

: "${MLX_GEN_MODELS_ROOT:?set MLX_GEN_MODELS_ROOT to the absolute Hugging Face hub cache}"
: "${KREA_ALTERNATE_DECODER_OUTPUT_DIR:?set KREA_ALTERNATE_DECODER_OUTPUT_DIR}"
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"

if [[ "$MLX_GEN_MODELS_ROOT" != /* ]]; then
  echo "MLX_GEN_MODELS_ROOT must be absolute, got: $MLX_GEN_MODELS_ROOT" >&2
  exit 1
fi
if [[ "$KREA_ALTERNATE_DECODER_OUTPUT_DIR" != /* ]]; then
  echo "KREA_ALTERNATE_DECODER_OUTPUT_DIR must be absolute, got: $KREA_ALTERNATE_DECODER_OUTPUT_DIR" >&2
  exit 1
fi
if [[ "$KREA_ALTERNATE_DECODER_OUTPUT_DIR" != "$RUNNER_TEMP"/* ]]; then
  echo "KREA_ALTERNATE_DECODER_OUTPUT_DIR must be inside RUNNER_TEMP" >&2
  exit 1
fi

readonly KREA_REVISION="d009674080cc1bccf2b629d834c34bf5eccdb723"
readonly WAN_REVISION="e68e9a3d98187fdf6936838ffcf6df5aa48d6626"
readonly WAN_VAE_SHA256="42159a8b571dbeb3ea40327b88a6161a5342c0511202af7c031360629757163d"
readonly KREA_SNAPSHOT="$MLX_GEN_MODELS_ROOT/models--SceneWorks--krea-2-turbo-mlx/snapshots/$KREA_REVISION"
readonly WAN_SNAPSHOT="$MLX_GEN_MODELS_ROOT/models--SceneWorks--krea-realtime-14b-mlx/snapshots/$WAN_REVISION"
readonly KREA_TIER="$KREA_SNAPSHOT/q4"
readonly WAN_VAE="$WAN_SNAPSHOT/q8/vae.safetensors"

if [[ "${KREA_TURBO_DIR:?workflow must export the q4 manifest projection}" != */model-snapshots/krea-2-turbo-mlx-q4/$KREA_REVISION ]]; then
  echo "KREA_TURBO_DIR was not exported from the pinned q4 manifest row: $KREA_TURBO_DIR" >&2
  exit 1
fi
if [[ \
  "${KREA_ALTERNATE_DECODER_WAN_VAE_SNAPSHOT:?workflow must export the Wan donor manifest projection}" \
  != */model-snapshots/krea-realtime-14b-mlx-wan-z16-vae-q8/$WAN_REVISION \
]]; then
  echo "Wan donor path was not exported from its pinned manifest row: $KREA_ALTERNATE_DECODER_WAN_VAE_SNAPSHOT" >&2
  exit 1
fi
# The manifest exporter binds the selected key/revision to the workflow policy. This smoke consumes
# the immutable copy already resident in the canonical HF cache rather than materializing a second
# ~20 GiB q4 tier into RUNNER_TEMP.
export KREA_TURBO_DIR="$KREA_TIER"

# The manifest checks bind the complete q4 base and the workflow-provisioned standalone q8 donor to
# their immutable repository revisions and required file sets. Every video tier carries the same
# unquantized VAE; the digest below additionally binds the exact file used by the component seam.
python3.12 scripts/release/verify_model_snapshot.py \
  --model krea-2-turbo-mlx-q4 \
  --snapshot "$KREA_SNAPSHOT"
python3.12 scripts/release/verify_model_snapshot.py \
  --model krea-realtime-14b-mlx-wan-z16-vae-q8 \
  --snapshot "$WAN_SNAPSHOT"
if [[ ! -f "$WAN_VAE" ]]; then
  echo "pinned Wan z16 decoder is absent: $WAN_VAE" >&2
  exit 1
fi
actual_wan_sha="$(shasum -a 256 "$WAN_VAE" | awk '{print $1}')"
if [[ "$actual_wan_sha" != "$WAN_VAE_SHA256" ]]; then
  echo "Wan z16 decoder digest mismatch: $actual_wan_sha != $WAN_VAE_SHA256" >&2
  exit 1
fi

if [[ -L "$KREA_ALTERNATE_DECODER_OUTPUT_DIR" ]]; then
  echo "output directory must not be a symlink: $KREA_ALTERNATE_DECODER_OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$KREA_ALTERNATE_DECODER_OUTPUT_DIR"
if [[ -n "$(find "$KREA_ALTERNATE_DECODER_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output directory must start empty: $KREA_ALTERNATE_DECODER_OUTPUT_DIR" >&2
  exit 1
fi

run_characterization() {
  local size="$1"
  local tiled="$2"
  KREA_TURBO_DIR="$KREA_TIER" \
    WAN21_VAE_FILE="$WAN_VAE" \
    KREA_AB_OUTPUT_DIR="$KREA_ALTERNATE_DECODER_OUTPUT_DIR" \
    KREA_AB_SIZE="$size" \
    KREA_AB_TILED="$tiled" \
    cargo run --locked --release -p mlx-gen-krea --example alternate_decoder_characterization
}

# One ordinary render proves the no-override/native path and a distinct coherent Wan result. The
# second forces more than one 512 px decode tile for both decoders at the same fixed seed and prompt.
run_characterization 512 0
run_characterization 768 1

expected=(
  native-untiled-512.png
  wan21-untiled-512.png
  native-tiled-768.png
  wan21-tiled-768.png
)
for name in "${expected[@]}"; do
  if [[ ! -s "$KREA_ALTERNATE_DECODER_OUTPUT_DIR/$name" ]]; then
    echo "missing or empty correctness artifact: $name" >&2
    exit 1
  fi
done
actual_count="$(find "$KREA_ALTERNATE_DECODER_OUTPUT_DIR" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d '[:space:]')"
if [[ "$actual_count" != "${#expected[@]}" ]]; then
  echo "expected ${#expected[@]} PNGs, found $actual_count" >&2
  exit 1
fi

hash_receipt="$KREA_ALTERNATE_DECODER_OUTPUT_DIR/sha256.txt"
(
  cd "$KREA_ALTERNATE_DECODER_OUTPUT_DIR"
  shasum -a 256 "${expected[@]}"
) > "$hash_receipt"
unique_hashes="$(awk '{print $1}' "$hash_receipt" | sort -u | wc -l | tr -d '[:space:]')"
if [[ "$unique_hashes" != "${#expected[@]}" ]]; then
  echo "correctness artifacts are not four distinct images" >&2
  exit 1
fi

{
  printf 'inference_sha=%s\n' "${GITHUB_SHA:-local}"
  printf 'krea_repository=SceneWorks/krea-2-turbo-mlx\n'
  printf 'krea_revision=%s\n' "$KREA_REVISION"
  printf 'krea_tier=q4\n'
  printf 'wan_repository=SceneWorks/krea-realtime-14b-mlx\n'
  printf 'wan_revision=%s\n' "$WAN_REVISION"
  printf 'wan_vae_path=q8/vae.safetensors\n'
  printf 'wan_vae_sha256=%s\n' "$WAN_VAE_SHA256"
  printf 'seed=7\n'
  printf 'untiled_geometry=512x512\n'
  printf 'tiled_geometry=768x768\n'
  printf 'tile_edge=512\n'
  printf 'tile_overlap=64\n'
} > "$KREA_ALTERNATE_DECODER_OUTPUT_DIR/provenance.txt"

echo "SC-18315 correctness smoke produced ${#expected[@]} distinct PNGs"
