# sc-14550. This job and its CUDA twin are the only lanes provisioning all six snapshots.
#
# ⚠ These cases do NOT validate a real adapter artifact. **None exists** — there is no
# Stable Audio 3 adapter of any type in any cache, and no published community one
# (sc-15347). Every adapter here is synthesized by the test against the real checkpoint's
# own safetensors header, in the format `src/adapters.rs` declares. What they prove is
# that the resolved plan reaches the real backend — including Metal's packed-buffer
# `SimpleBackend`, which is a different object from CPU/CUDA's mmap and is exercised in no
# other lane — and that the fold changes the rendered audio.
#
# The gates are the two exactly-signed ones, never a threshold:
#   * a `scale == 0.0` request renders **byte-identical** audio to a request with no
#     adapters at all. Exactly zero difference; a fold that ran at all breaks it.
#   * two adapters differing only in their factor values render **different** audio.
# "Adapted differs from un-adapted" appears only as a liveness check, because a
# *misapplied* adapter differs too — that is sc-14548's wrong-sign lesson applied up front.
#
# `real_lora_xs_folds_through_the_host_svd` is deliberately scoped to the conditioner's
# single [768, 256] Linear. Candle has no `linalg.svd`, so the `-xs` family computes one on
# the host in f64 (never on the accelerator, which is what makes it cross-platform
# deterministic by construction). Measured: ~1.9 s for [768, 256], ~113 s for
# [1024, 1024]. Covering the DiT's attention stack would be a multi-hour cold start; the
# math is identical at either scale and only the wall clock is not.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test adapters -- --ignored --nocapture
