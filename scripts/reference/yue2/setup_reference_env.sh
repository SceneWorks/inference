#!/usr/bin/env bash
# Build (or re-verify) the shared YuE2 Python reference environment for epic sc-22988.
#
# Every YuE2 fixture generator and reference comparison runs against ONE pinned upstream: the YuE
# GitHub source at commit 92a73cc7652fcc1f937855e4b765e0a0edd7ff2e, installed with the exact
# dependency pins its own pyproject.toml declares. The environment lives OUTSIDE every repository
# (default ~/.cache/sceneworks-yue2-ref, override with YUE2_REF_DIR) and is never committed.
#
# Idempotent: re-running re-verifies the checkout (HEAD == pinned commit, clean tree) and the venv
# and only installs what is missing. See scripts/reference/yue2/README.md.
set -euo pipefail

YUE2_COMMIT="92a73cc7652fcc1f937855e4b765e0a0edd7ff2e"
YUE2_REPO_URL="https://github.com/multimodal-art-projection/YuE"
PYTHON="${YUE2_PYTHON:-python3.12}"
REF_DIR="${YUE2_REF_DIR:-$HOME/.cache/sceneworks-yue2-ref}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"

mkdir -p "$REF_DIR"
REF_DIR="$(cd "$REF_DIR" && pwd)"
case "$REF_DIR/" in
"$repo_root"/*)
    echo "error: YUE2_REF_DIR ($REF_DIR) is inside the repository ($repo_root); the reference" >&2
    echo "       environment must live outside it" >&2
    exit 2
    ;;
esac

src="$REF_DIR/YuE"
if [ ! -d "$src/.git" ]; then
    echo "==> cloning $YUE2_REPO_URL into $src"
    git clone --no-checkout "$YUE2_REPO_URL" "$src"
fi
if ! git -C "$src" cat-file -e "$YUE2_COMMIT^{commit}" 2>/dev/null; then
    echo "==> fetching $YUE2_COMMIT"
    git -C "$src" fetch origin "$YUE2_COMMIT"
fi
git -C "$src" -c advice.detachedHead=false checkout --detach "$YUE2_COMMIT"

head="$(git -C "$src" rev-parse HEAD)"
if [ "$head" != "$YUE2_COMMIT" ]; then
    echo "error: $src is at $head, expected $YUE2_COMMIT" >&2
    exit 1
fi
if [ -n "$(git -C "$src" status --porcelain --untracked-files=all)" ]; then
    echo "error: $src has local modifications; the reference must be the pinned tree exactly" >&2
    git -C "$src" status --short >&2
    exit 1
fi
echo "==> upstream source verified at $YUE2_COMMIT (clean tree)"

venv="$REF_DIR/venv"
if [ ! -x "$venv/bin/python" ]; then
    echo "==> creating venv with $PYTHON"
    "$PYTHON" -m venv "$venv"
fi
# Install the pinned upstream package itself (non-editable) with exactly the dependency pins its
# pyproject.toml declares (torch==2.10.0, transformers==4.57.6, safetensors==0.7.0,
# tiktoken==0.12.0, ...). `--upgrade-strategy only-if-needed` keeps a re-run from drifting.
if ! "$venv/bin/python" -c "import yue2" 2>/dev/null; then
    echo "==> installing yue2-infer from the pinned checkout"
    "$venv/bin/python" -m pip install --quiet --upgrade pip
    "$venv/bin/python" -m pip install --quiet "$src"
fi
"$venv/bin/python" -m pip check

# Record what was built, so a fixture's provenance can name the exact stack.
"$venv/bin/python" - "$REF_DIR/ENVIRONMENT.json" "$YUE2_COMMIT" <<'PY'
import importlib.metadata as md
import json
import platform
import sys

out, commit = sys.argv[1], sys.argv[2]
packages = {d.metadata["Name"].lower(): d.version for d in md.distributions()}
json.dump(
    {
        "yue2_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
    },
    open(out, "w"),
    indent=2,
    sort_keys=False,
)
print(f"==> wrote {out}")
PY
echo "==> reference environment ready: $venv/bin/python (upstream at $src)"
