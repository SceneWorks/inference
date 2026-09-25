# sc-14546: both cases in this target now run on the lane's device. The runtime half
# always did (it goes through `provider_registry().load(…)` ->
# `candle_audio::default_device()`); the single-step DiT half hardcoded `Device::Cpu` and
# now honours the shared `SA3_TEST_METAL`/`SA3_TEST_CUDA` selector. Its ±0.02 absolute
# cosine band sits far above accelerator reduced-precision noise, so nothing in it was
# CPU-calibrated.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test variant_divergence -- --ignored --nocapture
