call "%VCVARS%"
rem `base_guidance`'s divergence assertion proves the negative prompt reaches the model,
rem not that the batch-2 path is wired correctly. This case is the only one that separates
rem absent negative conditioning — which zeroes the whole cross context — from an explicit
rem all-invalid negative prompt, which keeps its conditioned duration row. It ran in no
rem lane before sc-14546. Named rather than `--test dit_oracle -- --ignored`: the other
rem ignored cases need all six snapshots on CPU or an explicit SA3_DIT_CASE_ENV selection,
rem and wiring the rest of `dit_oracle` is sc-15235.
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test dit_oracle real_weights_detect_conditioning_mutations_and_exercise_cfg_apg -- --ignored --nocapture || exit /b 1
