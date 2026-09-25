# sc-14547. This job is the only lane that provisions all six snapshots, and the story's
# acceptance is explicitly "every registered variant", so the target lands here rather
# than on a per-variant job.
#
# Both ignored cases in the target run, and both are wanted:
#
#   * `real_reference_restyle_is_bounded_and_ordered_on_all_six_variants` asserts *both*
#     bounds the story requires — measurable retained structure at full retention and a
#     divergence floor at zero retention — and, in the same sweep, the strength
#     *direction* through the full graph. The weight-free sign gate named in `ci.yml`
#     proves the mapping; this proves the mapping is the one the model responds to.
#   * `real_initial_sampler_noise_precedes_the_source_encode` is the only observation that
#     separates "initial noise first, then the source encode" from the reverse. Encoding
#     first would move every later draw, so the same seed would sound different merely for
#     having a clip attached.
#
# `--nocapture` keeps the measured per-variant correlations in the log, so the floors are
# auditable from the run rather than only from the source.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test reference_audio -- --ignored --nocapture
