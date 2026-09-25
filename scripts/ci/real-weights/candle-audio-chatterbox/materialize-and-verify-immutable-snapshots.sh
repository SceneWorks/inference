python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model kokoro-82m --snapshot "$KOKORO_SNAPSHOT"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model chatterbox --snapshot "$CHATTERBOX_SNAPSHOT"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model chatterbox-perth --snapshot "$CHATTERBOX_PERTH_SNAPSHOT"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model chatterbox-ve --snapshot "$CHATTERBOX_VE_SNAPSHOT"
