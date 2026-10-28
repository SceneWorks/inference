call "%VCVARS%"
rem Standalone encoder conformance (sc-13474): the Synchformer sync encoder and the DFN5B-CLIP
rem visual+text tower — the two component encoders the video->audio pipeline below composes.
cargo test --locked -p candle-audio-mmaudio --features cuda --release --test conformance -- --ignored --nocapture || exit /b 1
cargo test --locked -p candle-audio-mmaudio --features cuda --release --test clip_conformance -- --ignored --nocapture || exit /b 1
cargo test --locked -p candle-audio-mmaudio --features cuda --release --test conformance_video_to_audio -- --ignored --nocapture || exit /b 1
cargo test --locked -p candle-audio-mmaudio --features cuda --release --test conformance_video_to_audio_44k -- --ignored --nocapture || exit /b 1
