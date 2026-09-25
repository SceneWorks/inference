call "%VCVARS%"
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test variant_binding -- --ignored --nocapture || exit /b 1
rem Named rather than `--test sampler_oracle -- --ignored`: the third ignored case in that
rem target is an operator resource probe that requires SA3_RESOURCE_* and fails without it.
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test sampler_oracle all_six_real_p0_pingpong_trajectories_match_stepwise -- --ignored --nocapture || exit /b 1
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test sampler_oracle real_sampler_cfg_apg_scale_phi_matches_frozen_upstream -- --ignored --nocapture || exit /b 1
