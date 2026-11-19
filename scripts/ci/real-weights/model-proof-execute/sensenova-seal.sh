set -euo pipefail
if [[ "$RUNNER_OS" == macOS ]]; then python=python3.12; else python="$REVIEWED_PYTHON"; fi
"$python" scripts/ci/sensenova_loop_seal.py
