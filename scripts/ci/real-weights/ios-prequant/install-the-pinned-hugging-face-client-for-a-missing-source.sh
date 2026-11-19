python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
echo "IOS_PREQUANT_HF_HUB_PATH=$RUNNER_TEMP/huggingface-hub" >> "$GITHUB_ENV"
