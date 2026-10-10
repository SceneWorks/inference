call "%VCVARS%"
rem sc-14547. This job and its Metal twin are the only lanes provisioning all six
rem snapshots, and the story's acceptance is "every registered variant". Both ignored
rem cases run: the sweep asserts retained structure, a divergence floor, and the strength
rem direction through the full graph, and the draw-order case is the only observation that
rem separates "initial noise first, then the source encode" from the reverse.
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test reference_audio -- --ignored --nocapture || exit /b 1
