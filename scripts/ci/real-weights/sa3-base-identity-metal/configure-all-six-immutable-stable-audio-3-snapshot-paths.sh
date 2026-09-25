# sc-14546 made `variant_binding` a six-way matrix: each post-trained checkpoint against
# its `-base` sibling in both directions, plus the two small bases against each other and
# against medium-base. Every one of those needs both sides materialized, so this job owns
# the full set and the per-variant jobs no longer run the target.
python3.12 scripts/release/export_model_snapshot_paths.py \
  --model stable-audio-3-small-music --model stable-audio-3-small-sfx \
  --model stable-audio-3-medium --model stable-audio-3-small-music-base \
  --model stable-audio-3-small-sfx-base --model stable-audio-3-medium-base \
  --model same-l
