set -o pipefail
resource_log="$RUNNER_TEMP/sa3-chunked-resource.log"
: > "$resource_log"
for case in same_s same_l; do
  for mode in direct chunked; do
    for operation in encode decode; do
      SA3_CHUNKED_CASE="$case" \
      SA3_CHUNKED_RESOURCE_MODE="$mode" \
      SA3_CHUNKED_RESOURCE_OPERATION="$operation" \
        cargo test --locked --release -p candle-audio-stable-audio-3 \
          --features metal --test chunked_oracle chunked_same_resource_probe \
          -- --ignored --nocapture 2>&1 | tee -a "$resource_log"
    done
  done
done
python3.12 scripts/reference/verify_sa3_chunked_resource_log.py \
  "$resource_log" --json | tee "$RUNNER_TEMP/sa3-chunked-resource.json"
