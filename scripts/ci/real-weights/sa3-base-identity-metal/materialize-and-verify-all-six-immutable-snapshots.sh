# The three `-base` entries carry `download_files` allow-lists, so `svd_bases.pt` — 1.27 GB
# per small base and 3.84 GB for medium-base, 6.4 GB in total, and unreachable from
# `SnapshotLayout::from_dir` — is not fetched.
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
for model in stable-audio-3-small-music stable-audio-3-small-sfx stable-audio-3-medium \
             stable-audio-3-small-music-base stable-audio-3-small-sfx-base \
             stable-audio-3-medium-base same-l; do
  case "$model" in
    stable-audio-3-small-music) path="$SA3_SMALL_MUSIC_SNAPSHOT" ;;
    stable-audio-3-small-sfx) path="$SA3_SMALL_SFX_SNAPSHOT" ;;
    stable-audio-3-medium) path="$SA3_MEDIUM_SNAPSHOT" ;;
    stable-audio-3-small-music-base) path="$SA3_SMALL_MUSIC_BASE_SNAPSHOT" ;;
    stable-audio-3-small-sfx-base) path="$SA3_SMALL_SFX_BASE_SNAPSHOT" ;;
    stable-audio-3-medium-base) path="$SA3_MEDIUM_BASE_SNAPSHOT" ;;
    same-l) path="$SA3_SAME_L_SNAPSHOT" ;;
  esac
  PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "$model" --snapshot "$path"
done
