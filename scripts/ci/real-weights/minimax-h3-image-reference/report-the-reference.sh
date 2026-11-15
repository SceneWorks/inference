# Record which reference this run's gates use: the store key, whether the runner store supplied it,
# the store path, and the SHA-256 of each file. Fails when a file is missing, so a partial reference
# never reaches a gate or the store.
set -o pipefail
{
  echo "reference store key: $H3I_REFERENCE_KEY"
  echo "reference store hit: ${H3I_REFERENCE_STORE_HIT:-false}"
  echo "reference store path: $H3I_REFERENCE_STORE_ROOT/$H3I_REFERENCE_KEY"
  shasum -a 256 minimax-h3-image-vae-reference/{reference,dit-reference,edit-reference}.safetensors
} | tee -a "$GITHUB_STEP_SUMMARY"
