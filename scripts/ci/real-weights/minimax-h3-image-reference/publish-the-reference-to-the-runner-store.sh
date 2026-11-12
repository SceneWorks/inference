# Publish the freshly produced reference into this runner's store: copy into a partial directory
# inside the store root, write the SHA-256 manifest, then rename it to `<key>/`, so no run ever sees
# a partial entry. Then keep only the newest three `h3img-ref-v*` entries.
set -o pipefail
python3.12 -I scripts/ci/minimax_h3_image_reference_store.py publish --root "$H3I_REFERENCE_STORE_ROOT" --key "$H3I_REFERENCE_KEY" --dir "$GITHUB_WORKSPACE/minimax-h3-image-vae-reference"
