# Record which reference this run's gates use: the cache key, whether the cache supplied it, and the
# SHA-256 of each file. Fails when a file is missing, so a partial cache entry never reaches a gate.
set -o pipefail
{
  echo "reference cache key: $H3I_REFERENCE_KEY"
  echo "reference cache hit: ${H3I_REFERENCE_CACHE_HIT:-false}"
  shasum -a 256 minimax-h3-image-vae-reference/{reference,dit-reference,edit-reference}.safetensors
} | tee -a "$GITHUB_STEP_SUMMARY"
