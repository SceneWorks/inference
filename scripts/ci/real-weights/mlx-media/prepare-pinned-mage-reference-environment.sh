if [[ ! -x "$RUNNER_TEMP/mage-reference/bin/python" ]]; then
  python -m venv "$RUNNER_TEMP/mage-reference"
  "$RUNNER_TEMP/mage-reference/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes -r crates/media/mlx-gen/_vendor/mage_flow/requirements-oracles.txt
fi
