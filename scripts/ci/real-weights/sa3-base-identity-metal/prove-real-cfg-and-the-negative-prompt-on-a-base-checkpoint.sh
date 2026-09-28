# The divergence floor is not a chosen tolerance: the test measures this implementation's
# own largest per-element disagreement with the frozen-Torch guidance oracle on this exact
# checkpoint, device and dtype, and then requires the negative-prompt divergence to exceed
# it. It also asserts the honest control — at `cfg_scale = 1.0` the negative prompt must be
# a bit-for-bit no-op, because that is the one value at which the DiT skips the negative
# branch entirely. Also pins the shipped 50/7.0 defaults against each snapshot's own
# `training.demo` block.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test base_guidance -- --ignored --nocapture
