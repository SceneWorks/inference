# `base_guidance`'s divergence assertion proves the negative prompt reaches the model; it
# cannot prove the batch-2 path is wired *correctly*, because a swapped-halves or masked-
# wrong negative branch diverges from the no-negative render just as loudly. Two gates
# close that, and neither ran in any lane before sc-14546:
#
#   * `real_weights_detect_conditioning_mutations_and_exercise_cfg_apg` (dit_oracle) is
#     the only case that separates absent negative conditioning — which zeroes the whole
#     cross context — from an explicit all-invalid negative prompt, which keeps its
#     conditioned duration row. It is a pure self-comparison, so it runs on this backend.
#   * `the_guided_latents_are_exactly_the_cfg_recomposition_of_their_own_two_branches`
#     (base_guidance, above) pins the guided latents to `L(N) + g*(L(P) - L(N))` and
#     rejects three named mis-wirings of the same two branches.
#
# Named rather than `--test dit_oracle -- --ignored`: the other ignored cases in that
# target need all six snapshots on CPU (multi-gigabyte) or an explicit SA3_DIT_CASE_ENV /
# SA3_DIT_RESOURCE_LENGTH selection. Wiring the rest of `dit_oracle` is sc-15235.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test dit_oracle real_weights_detect_conditioning_mutations_and_exercise_cfg_apg \
  -- --ignored --nocapture
