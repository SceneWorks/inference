# Named tests: the shared `conformance` binary also carries the sc-14544 small-sfx cases,
# which need their own snapshot and run in the sa3-small-sfx job.
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test conformance registered_provider_passes_full_audio_conformance -- --ignored --nocapture
cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
  --test conformance concurrent_requests_are_deterministic_and_do_not_share_rng_state -- --ignored --nocapture
