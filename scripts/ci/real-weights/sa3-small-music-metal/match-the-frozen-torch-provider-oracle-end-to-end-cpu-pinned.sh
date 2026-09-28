# sc-14543's end-to-end parity oracle against the vendored
# `docs/migration/sa3-small-music-provider-reference` artifact. It ran in no lane until
# sc-14545, which is how its `from_layout` call site could be re-pointed from a bare
# `HUB_REPO` string to `Variant::SmallMusic.geometry()` with nothing but a compile check
# behind it. `validate_layout` rejects a wrong variant before a tensor is read, so this is
# the executable proof that the rename kept the checkpoint binding. Deterministic
# (`Device::Cpu`, portable LCG noise, fixed seed) and measured at 20.6 s on an M-series
# Mac against this exact pinned snapshot.
#
# This job sets `SA3_TEST_METAL`, but this step deliberately ignores it and every other
# `SA3_TEST_*` selector: its bounds are exact-reproduction-grade against a frozen Torch
# CPU-f32 artifact and are calibrated on CPU, so this step certifies frozen-Torch parity
# and **not** the Metal backend. The reasoning is on the test itself
# (`tests/provider_oracle.rs`). Metal small-music coverage comes from this job's
# conformance, stereo-width and render steps, which resolve their device through
# `candle_audio::default_device()`. This is why there is no CUDA counterpart.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test provider_oracle -- --ignored --nocapture
