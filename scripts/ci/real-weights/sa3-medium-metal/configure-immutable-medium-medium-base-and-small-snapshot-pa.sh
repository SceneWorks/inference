# The cross-variant mutation gates compare medium against both smalls and against its own
# base sibling, and the two-domain comparison renders medium beside each specialist, so
# every snapshot they read has to be on this runner.
python3.12 scripts/release/export_model_snapshot_paths.py \
  --model stable-audio-3-medium --model stable-audio-3-medium-base \
  --model stable-audio-3-small-music --model stable-audio-3-small-sfx \
  --model same-l
