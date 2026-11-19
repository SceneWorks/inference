# SW-7104 — feature SW-F363's acceptance renders on MLX (tests/h3_image_acceptance_real.rs): AT1
# (2048², 4096², 6272x2688 at the admitted strategy, one fresh process each), AT3 (a 3-reference
# edit that must draw on each reference), AT6 (the refusals), AT8 (16 MP with two community LoRAs
# through bounded transformer residency and bounded decode) and AT11 (every community LoRA
# applies). Runs after the earlier steps, so the image snapshot, MiniMax-H3's text encoder, the
# pinned huggingface-hub install, the run-scoped CARGO_TARGET_DIR and the prebuilt libmlx are in
# place. Materializes the community LoRAs (the `minimax-h3-image-community-lora-*` rows) into the
# runner's Hugging Face cache. Each gate is name-selected with `--exact`, built with the registered
# `model-inference-proof` feature, run with MODEL_INFERENCE_CASE_FQN set and checked by
# verify_model_inference_log.py, so a rename cannot go vacuously green. Every gate runs and reports
# even when an earlier one fails; the step fails at the end if any did. The PNGs and logs go to
# $RUNNER_TEMP/minimax-h3-image-acceptance, uploaded as minimax-h3-image-acceptance-mlx.
set -o pipefail
out="$RUNNER_TEMP/minimax-h3-image-acceptance"
mkdir -p "$out"
export MINIMAX_H3_IMAGE_ACCEPTANCE_OUT="$out"
export HF_HUB_DISABLE_XET=1
hub="$(python3.12 scripts/release/runner_hub_cache.py)"
loras=""
# name:row:snapshot variable:repository:revision:file - each row's `environment` variable is
# exported with its snapshot path.
for entry in \
  "tanaami:tanaami:MINIMAX_H3_IMAGE_LORA_TANAAMI_SNAPSHOT:syntaxdiffusion/tanaami-style-h3-image-lora:3804b381526a3e537c7961942ba4861a831b2d39:tanaami_h3_image_lora_2000.safetensors" \
  "domokun:domokun:MINIMAX_H3_IMAGE_LORA_DOMOKUN_SNAPSHOT:SimpleTuner/minimax-h3-image-domokun-2048:57214626f457eac87077c7035bb1b64d5264ebf3:pytorch_lora_weights.safetensors" \
  "fizgig:fizgig:MINIMAX_H3_IMAGE_LORA_FIZGIG_SNAPSHOT:ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter:f1a4547596faeb98bae568dd28da296c49fbe0a1:fizgig_minimax_h3_image_training_adapter.safetensors" \
  "pdmd:pdmd:MINIMAX_H3_IMAGE_LORA_PDMD_SNAPSHOT:linoyts/MiniMax-H3-Image-pDMD-2steps:c07d96e5635311ed007d44654c2f98cc8d6e0a1e:pytorch_lora_weights.safetensors" \
  "character-swap:character-swap:MINIMAX_H3_IMAGE_LORA_CHARACTER_SWAP_SNAPSHOT:akhaliq/MiniMax-H3-Image-Character-Swap:0b7f918092f9e23c01ca62b63d7c17e4fce6faa9:lora/h3_image_character_swap_diffusers.safetensors"; do
  IFS=: read -r name row var repo rev file <<<"$entry"
  snapshot="$hub/models--${repo//\//--}/snapshots/$rev"
  export "$var=$snapshot"
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py \
      --model "minimax-h3-image-community-lora-$row" --snapshot "$snapshot" && break
    [[ $attempt == 3 ]] && exit 1
    echo "::warning::minimax-h3-image-community-lora-$row materialization attempt $attempt failed"
    sleep 30
  done
  loras+="$name=$snapshot/$file;"
done
export MINIMAX_H3_IMAGE_COMMUNITY_LORAS="$loras"
export MINIMAX_H3_IMAGE_AT8_LORAS="tanaami,domokun"
failed=()
run_gate() {
  local name="h3_image_acceptance_real::$1" status=0 log="$out/$1.log" learned=()
  [[ $1 == h3_image_refuses_the_unsupported_requests ]] && learned=(--no-learned-work)
  MODEL_INFERENCE_CASE_FQN="$name" cargo test --locked --release -p mlx-gen-minimax-h3 \
    --features model-inference-proof --test integration \
    "$name" -- --exact --ignored --nocapture 2>&1 | tee "$log" >&2 || status=$?
  if ((status)) || ! python3.12 -I scripts/ci/verify_model_inference_log.py --package mlx-gen-minimax-h3 \
    --backend mlx --fqn "$name" "${learned[@]}" --log "$log"; then
    echo "::error::'$name' failed or lacks its registered proof evidence (exit $status)" >&2
    failed+=("$1")
  fi
}
for name in h3_image_t2i_2048_at_the_admitted_strategy h3_image_t2i_4096_at_the_admitted_strategy \
  h3_image_t2i_6272x2688_at_the_admitted_strategy h3_image_edit_draws_on_each_of_three_references \
  h3_image_refuses_the_unsupported_requests h3_image_16mp_with_two_loras_streams_and_tiles \
  h3_image_community_loras_apply; do
  run_gate "$name"
done
grep -h "^RUN RECORD\|^AT[0-9]" "$out"/*.log > "$out/run-records.txt" || true
cat "$out/run-records.txt"
if ((${#failed[@]})); then
  echo "::error::failed acceptance gates: ${failed[*]}" >&2
  exit 1
fi
