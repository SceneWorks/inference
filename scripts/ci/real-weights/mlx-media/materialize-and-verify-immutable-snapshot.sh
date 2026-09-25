python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model z-image-turbo --snapshot "$ZIMAGE_SNAPSHOT"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model lens-turbo --snapshot "$LENS_SNAPSHOT"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model mage-flow --snapshot "$MAGE_SNAPSHOT" --require-materialization-provenance
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model mage-flow-edit --snapshot "$MAGE_EDIT_SNAPSHOT" --require-materialization-provenance
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model mage-flow-edit-base --snapshot "$MAGE_EDIT_BASE_SNAPSHOT" --require-materialization-provenance
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model mage-flow-edit-turbo --snapshot "$MAGE_EDIT_TURBO_SNAPSHOT" --require-materialization-provenance
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model sdxl-base-mlx-vae-bf16 --snapshot "$SDXL_N1_SNAPSHOT"
