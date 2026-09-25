python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model krea-realtime-14b-mlx --snapshot "$KREA_REALTIME_SNAPSHOT"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model wan-origami-style-lora --snapshot "$KREA_STYLE_LORA_SNAPSHOT"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model wan-lightx2v-step-distill-lora --snapshot "$KREA_DISTILL_LORA_SNAPSHOT"
