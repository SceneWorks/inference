python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model smollm2-135m-instruct --snapshot "$MLX_LLM_TEST_MODEL"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model qwen3-0.6b --snapshot "$MLX_LLM_QWEN3_MODEL"
