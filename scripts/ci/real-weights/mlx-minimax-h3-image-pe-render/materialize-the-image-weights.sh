# SW-7104 (AT9) — the MiniMax-H3-Image inputs of the PE-expansion renders, materialized exactly as
# the `mlx-minimax-h3-image` job does (without its diffusers references): `vae_diffusers/`,
# `transformer_diffusers/` and `tokenizer/` of MiniMaxAI/MiniMax-H3-Image @492fd8c4 and MiniMax-H3's
# text encoder shards 1-12, into the runner's Hugging Face cache (plain HTTPS, two retries; a
# pinned, verified snapshot is a no-op). Exports both snapshot paths for the render step.
set -o pipefail
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
export HF_HUB_DISABLE_XET=1
hub="$(python3.12 scripts/release/runner_hub_cache.py)"
export MINIMAX_H3_IMAGE_SNAPSHOT="$hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots/492fd8c4eca666f1085f57e30738c1e0d86b3439"
export MINIMAX_H3_SNAPSHOT="$hub/models--MiniMaxAI--MiniMax-H3/snapshots/939557dc319dd91227e30195a763f272ba7f8765"
{
  echo "MINIMAX_H3_IMAGE_SNAPSHOT=$MINIMAX_H3_IMAGE_SNAPSHOT"
  echo "MINIMAX_H3_SNAPSHOT=$MINIMAX_H3_SNAPSHOT"
} >> "$GITHUB_ENV"
for model in "minimax-h3-image:$MINIMAX_H3_IMAGE_SNAPSHOT" "minimax-h3-image-dit:$MINIMAX_H3_IMAGE_SNAPSHOT" "minimax-h3-text-encoder:$MINIMAX_H3_SNAPSHOT"; do
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "${model%%:*}" --snapshot "${model#*:}" && break
    [[ $attempt == 3 ]] && exit 1
    echo "::warning::${model%%:*} materialization attempt $attempt failed"
    sleep 30
  done
done
