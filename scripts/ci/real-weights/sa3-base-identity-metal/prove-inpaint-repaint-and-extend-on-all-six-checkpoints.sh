# sc-14548. Same reasoning as the restyle step above: this job and its CUDA twin are the
# only lanes provisioning all six snapshots, and the story's acceptance is every
# registered variant.
#
# All four ignored cases in the target run, and each covers something the others cannot:
#
#   * `real_inpaint_preserves_the_outside_exactly_and_changes_the_inside` asserts BOTH
#     halves. Preservation alone is satisfied by a no-op, so the interior divergence is
#     measured in the same pass, against the source's own energy rather than an absolute.
#     Outside the region the assertion is exact equality, not a bound — that span is
#     written by `stitch_outside_region` from a buffer the test prepares itself, so a
#     tolerance would pass a stitch that had slipped a frame.
#   * `real_repaint_is_byte_identical_to_inpaint` pins the alias. gen-core documents the
#     two modes as different (that text is written against ACE-Step); Stable Audio 3 has
#     one inpaint mechanism, so any divergence is a bug.
#   * `real_extend_keeps_the_source_prefix_and_bridges_the_seam` is the story's pinned
#     10 s -> 18 s case: exact 441_000-frame prefix, a non-silent tail, and a seam step
#     bounded by the material's own 99.9th-percentile step rather than by a chosen number.
#   * `real_edit_initial_sampler_noise_precedes_the_source_encode` pins the draw order.
#     Only `medium` / `medium_base` can falsify it — SAME-S consumes zero draws on encode
#     — which is why it runs all six and separately requires at least one drawing encode.
#
# These four are also the ONLY lane for three weights-only edit sites the PR lane cannot
# reach: the stitch invocation, the local-conditioning handoff into `sample`, and the
# `conditioning_is_forwarded` call in `generate`. See
# `docs/migration/SC_14548_AUDIO_EDIT_INPAINT.md`.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test audio_edit -- --ignored --nocapture
