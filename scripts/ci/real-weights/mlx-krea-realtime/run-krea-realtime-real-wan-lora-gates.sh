set -o pipefail
run_one() {
  local name="$1"
  local out
  out="$(KREA_REALTIME_SNAPSHOT_DIR="$KREA_REALTIME_SNAPSHOT/q4" \
    KREA_STYLE_LORA="$KREA_STYLE_LORA_SNAPSHOT/origami_000000500.safetensors" \
    KREA_DISTILL_LORA="$KREA_DISTILL_LORA_SNAPSHOT/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors" \
    cargo test --locked --release -p mlx-gen-krea-realtime --test integration \
    style_lora_real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    return 1
  fi
}
run_one real_wan_style_lora_loads_and_changes_the_render
run_one real_wan_step_distill_lora_installs_over_the_widened_globals
