# The 380 s published cap is included: it is the only configuration that exercises the
# full 4,096-latent adapted geometry and the 43-chunk SAME-L decode plan, which is the
# entire reason this variant is registered separately from the smalls.
#
# `/usr/bin/time -l` reports `ru_maxrss` over the timed process *and its whole reaped
# child tree*, so timing `cargo test` conflates the release build — rustc and the linker
# peak in the multi-GB range on their own — with the render. Build first with `--no-run`,
# resolve the target's own executable from the JSON artifact stream, then time only that
# binary. The reported peak is then the render's, and nothing else's.
cargo test --locked --release --no-run -p candle-audio-stable-audio-3 --features metal \
  --test provider
binary="$(
  cargo test --locked --release --no-run --message-format=json \
    -p candle-audio-stable-audio-3 --features metal --test provider |
    python3.12 -c "import json, sys; print([m['executable'] for m in map(json.loads, sys.stdin) if m.get('reason') == 'compiler-artifact' and m.get('executable') and m.get('target', {}).get('name') == 'provider'][-1])"
)"
test -x "$binary" || { echo "could not resolve the provider test binary"; exit 1; }
echo "timing $binary"
SA3_MEDIUM_LONG_SECONDS=120,300,380 \
SA3_MEDIUM_LONG_WAV_DIR="$RUNNER_TEMP" \
  /usr/bin/time -l "$binary" medium_long_form_renders_are_exact_and_timed \
    --exact --ignored --nocapture
for seconds in 120 300 380; do
  shasum -a 256 "$RUNNER_TEMP/sa3-medium-${seconds}s.wav"
done
