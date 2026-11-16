# sc-14544's identity and divergence gates compare the two checkpoints against each other,
# so both snapshots are required on this runner.
python3.12 scripts/release/export_model_snapshot_paths.py \
  --model stable-audio-3-small-sfx --model stable-audio-3-small-music
