# Restore the diffusers reference from this runner's task-owned store, keyed by the content hash.
# `actions/cache` is branch-scoped, so a new story branch never saw a sibling branch's entry; this
# store is shared by every branch that runs on the box. A hit is verified file by file against the
# SHA-256 manifest written at publish time; a failing entry is deleted and treated as a miss.
set -o pipefail
lane_root="$HOME/sceneworks-rw-weights"
if [[ -L "$lane_root" ]]; then
  echo "task-owned weights root must be an internal directory: $lane_root" >&2; exit 1
fi
store_root="$lane_root/minimax-h3-image-reference"
mkdir -p "$store_root"
printf "H3I_REFERENCE_STORE_ROOT=%s\nH3I_REFERENCE_KEY=%s\n" "$store_root" "$H3I_REFERENCE_KEY" >> "$GITHUB_ENV"
echo "reference store: $store_root/$H3I_REFERENCE_KEY"
python3.12 -I scripts/ci/minimax_h3_image_reference_store.py restore --root "$store_root" --key "$H3I_REFERENCE_KEY" --dir "$GITHUB_WORKSPACE/minimax-h3-image-vae-reference"
