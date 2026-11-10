# SW-7090 — MiniMax-H3-Image DiT on MLX: the published 644-tensor map, step-0 / step-3 parity
# against the diffusers reference, and a 2048² bf16 text-to-image render through the registered
# generator. Name-selected with `--exact` plus a pass-count assertion, so a rename cannot go
# vacuously green; one process per test, so each heavy test starts from a reaped allocator.
# SW-7096 adds the LoRA gates: three community layouts fold identically on the real DiT, ten
# stacked files keep their scales, and a block streamed through DitBlockStream carries the LoRA.
set -o pipefail
export MINIMAX_H3_IMAGE_RENDER_OUT="$RUNNER_TEMP/minimax-h3-image-2048.png"
for name in h3_image_dit_names_match_the_published_index h3_image_dit_first_step_matches_diffusers h3_image_t2i_renders_2048_bf16; do
  out="$(cargo test --locked --release -p mlx-gen-minimax-h3 --test integration \
    h3_image_real::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    exit 1
  fi
done
for name in h3_image_lora_layouts_fold_identically_on_the_real_dit h3_image_streamed_blocks_carry_the_lora_like_resident_ones; do
  out="$(cargo test --locked --release -p mlx-gen-minimax-h3 --test integration \
    h3_image_lora_real::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    exit 1
  fi
done
