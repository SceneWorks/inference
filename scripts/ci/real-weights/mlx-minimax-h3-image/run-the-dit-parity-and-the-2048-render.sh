# SW-7090 — MiniMax-H3-Image DiT on MLX: the published 644-tensor map, step-0 / step-3 parity
# against the diffusers reference, and a 2048² bf16 text-to-image render through the registered
# generator. Name-selected with `--exact` plus a pass-count assertion, so a rename cannot go
# vacuously green; one process per test, so each heavy test starts from a reaped allocator.
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
# SW-7093 -- MiniMax-H3-Image q8 / q4 DiT tiers on MLX: convert from the dense snapshot, audit the
# artifact, and render 2048x2048 through the registered generator at each tier.
#
# Runs at the end of this script (the DiT parity and the bf16 render above), so the task-owned hub, the pinned
# huggingface-hub install, the image snapshot, the run-scoped CARGO_TARGET_DIR and the prebuilt libmlx
# are all in place. The converted tiers go to the runner's own weights root (never to Hugging Face:
# publication is a separate step) and are removed after each tier's render to bound the disk footprint.
# The text encoder at each tier is MiniMax-H3's existing `SceneWorks/minimax-h3-mlx` artifact.
lane_root="$HOME/sceneworks-rw-weights"
export MINIMAX_H3_TIER_SNAPSHOT="$lane_root/hub/models--SceneWorks--minimax-h3-mlx/snapshots/137ce668c55a20bc0935fd1cf2a3de8448abb7f4"
export MINIMAX_H3_IMAGE_TIERS_ROOT="$lane_root/minimax-h3-image-tiers"
for tier in q8 q4; do
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "minimax-h3-mlx-te-$tier" --snapshot "$MINIMAX_H3_TIER_SNAPSHOT" && break
    [[ $attempt == 3 ]] && exit 1
    echo "::warning::minimax-h3-mlx-te-$tier materialization attempt $attempt failed"
    sleep 30
  done
done
export MINIMAX_H3_IMAGE_RENDER_OUT="$RUNNER_TEMP/minimax-h3-image-2048"
rm -rf "$MINIMAX_H3_IMAGE_TIERS_ROOT"
mkdir -p "$MINIMAX_H3_IMAGE_TIERS_ROOT"
for tier in q8 q4; do
  MINIMAX_H3_SRC="$MINIMAX_H3_IMAGE_SNAPSHOT" MINIMAX_H3_OUT="$MINIMAX_H3_IMAGE_TIERS_ROOT/$tier" \
    MINIMAX_H3_TIER="$tier" MINIMAX_H3_COMPONENT=image_dit \
    cargo run --locked --release -p mlx-gen-minimax-h3 --example minimax_h3_prequant
  name="h3_image_t2i_renders_2048_$tier"
  out="$(cargo test --locked --release -p mlx-gen-minimax-h3 --test integration \
    h3_image_real::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    exit 1
  fi
  rm -rf "${MINIMAX_H3_IMAGE_TIERS_ROOT:?}/$tier"
done
