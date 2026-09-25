python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model qwen-image-mlx --snapshot "$QWEN_IMAGE_MLX_SNAPSHOT"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model qwen-image-edit-2511-mlx --snapshot "$QWEN_IMAGE_EDIT_MLX_SNAPSHOT"
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model qwen-image-2512-fun-controlnet-union --snapshot "$QWEN_CONTROL_UNION_SNAPSHOT"
# The Lightning revisions come from the manifest, never from a second copy written here.
# Assigned into a variable and shape-checked FIRST: `$(revision …)` interpolated straight
# into an argument is a command substitution, and `set -e` does not fire on those — a
# failing helper would silently produce `…/snapshots/` with an empty revision. Same
# `^[0-9a-f]{40}$` guard the Mage oracle key step above uses, for the same reason.
revision() { python3.12 -c "import sys,tomllib,pathlib;m=tomllib.loads(pathlib.Path('release/real-weight-models.toml').read_text());print(next(x['revision'] for x in m['models'] if x['key']==sys.argv[1]))" "$1"; }
for key in qwen-image-lightning qwen-image-edit-2511-lightning; do
  rev="$(revision "$key")"
  if [[ ! "$rev" =~ ^[0-9a-f]{40}$ ]]; then
    echo "manifest revision for $key is missing or malformed: '$rev'" >&2
    exit 1
  fi
  case "$key" in
    qwen-image-lightning) var=QWEN_LIGHTNING_SNAPSHOT; repo=models--lightx2v--Qwen-Image-Lightning ;;
    *)                    var=QWEN_EDIT_LIGHTNING_SNAPSHOT; repo=models--lightx2v--Qwen-Image-Edit-2511-Lightning ;;
  esac
  dir="$MLX_GEN_MODELS_ROOT/$repo/snapshots/$rev"
  echo "$var=$dir" >> "$GITHUB_ENV"
  PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "$key" --snapshot "$dir"
done
