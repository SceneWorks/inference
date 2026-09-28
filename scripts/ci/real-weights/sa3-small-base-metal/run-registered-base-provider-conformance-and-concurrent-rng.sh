for test in registered_music_base_provider_passes_full_audio_conformance \
            concurrent_music_base_requests_are_deterministic_and_do_not_share_rng_state \
            registered_sfx_base_provider_passes_full_audio_conformance \
            concurrent_sfx_base_requests_are_deterministic_and_do_not_share_rng_state; do
  cargo test --locked --release -p candle-audio-stable-audio-3 --features metal \
    --test conformance "$test" -- --ignored --nocapture
done
