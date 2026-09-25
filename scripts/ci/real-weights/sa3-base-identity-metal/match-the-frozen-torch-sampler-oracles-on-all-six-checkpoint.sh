# Both cases ran in **no** lane before sc-14546. `all_six_real_p0_pingpong_trajectories`
# needs every snapshot, and `real_sampler_cfg_apg_scale_phi` is the hard fixed-guidance
# parity gate this story's own CFG acceptance rests on — it replays vanilla CFG, APG and
# blended+rescaled guidance at `cfg_scale = 2.5` against
# `docs/migration/sa3-sampler-reference/guidance.safetensors` on `small-music` **and**
# `small-music-base`.
#
# Named rather than `--test sampler_oracle -- --ignored`: the third ignored case in that
# target, `real_default_sampler_resource_probe`, is an operator probe that requires
# SA3_RESOURCE_SNAPSHOT / SA3_RESOURCE_P0 / SA3_RESOURCE_SECONDS and fails without them.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test sampler_oracle all_six_real_p0_pingpong_trajectories_match_stepwise -- --ignored --nocapture
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test sampler_oracle real_sampler_cfg_apg_scale_phi_matches_frozen_upstream -- --ignored --nocapture
