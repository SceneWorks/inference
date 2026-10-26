call "%VCVARS%"
rem sc-14548. This job and its Metal twin are the only lanes provisioning all six
rem snapshots. All four ignored cases run: exact outside-region preservation plus a
rem measured interior divergence (preservation alone passes for a no-op), Repaint/Inpaint
rem byte equality, the pinned 10s -> 18s extend with an exact prefix and a seam bound
rem derived from the material, and the draw order — which only medium/medium-base can
rem falsify, since SAME-S consumes zero draws on encode. This is also the only lane that
rem reaches the three weights-only edit sites named in
rem docs/migration/SC_14548_AUDIO_EDIT_INPAINT.md.
cargo test --locked --release -p candle-audio-stable-audio-3 --features cuda --test audio_edit -- --ignored --nocapture || exit /b 1
