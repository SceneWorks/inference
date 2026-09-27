set -o pipefail
run_one() {
  local name="$1"
  local out
  out="$(cargo test --locked --release -p mlx-llm --test integration \
    joycaption::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    return 1
  fi
}
# The registered quantize-prepare of the pinned 16 GB dense source into a Q4 text-stack
# snapshot (vision tower + projector stay dense), reloaded and run end-to-end as a VLM.
run_one prepared_q4_snapshot_runs_full_vlm
# Greedy decoding is deterministic, so the converted VLM must reproduce the reference
# engine's caption TOKEN-FOR-TOKEN. A divergence means the vision tower / projector /
# splice / decode port drifted numerically from the reference.
run_one joycaption_model_matches_golden_tokens
# The same golden through the resize path — the preprocessing arm the caption gate skips.
run_one joycaption_resize_path_matches_golden
# The provider-contract arm: the caption streams through the registered core-llm surface,
# not just the direct model call the golden tests use.
run_one joycaption_provider_streams_caption_through_contract
