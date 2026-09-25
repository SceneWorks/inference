call "%VCVARS%"
rem sc-14550. This job and its Metal twin are the only lanes provisioning all six
rem snapshots. These cases do NOT validate a real adapter artifact -- none exists
rem (sc-15347); every adapter is synthesized by the test against the real checkpoint's own
rem safetensors header. What they prove is that the resolved plan reaches the real backend
rem and that the fold changes the rendered audio. The two gates are exactly signed: a
rem scale-0.0 request renders byte-identical audio to a request with no adapters, and two
rem adapters differing only in their factors render different audio. "Adapted differs from
rem un-adapted" is liveness only, because a misapplied adapter differs too.
rem The -xs case is scoped to the conditioner's [768, 256] Linear: the deterministic host
rem f64 SVD costs ~1.9 s there and ~113 s at [1024, 1024].
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test adapters -- --ignored --nocapture || exit /b 1
