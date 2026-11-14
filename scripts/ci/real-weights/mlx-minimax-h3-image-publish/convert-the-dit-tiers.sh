# mlx-gen-minimax-h3's converter (SW-7093): one tier directory per width, holding
# `transformer_diffusers/` and the byte-copied `pdd_fused_heads.safetensors`. Both backends load
# this MLX affine pack.
for tier in q8 q4; do
  MINIMAX_H3_SRC="$MINIMAX_H3_IMAGE_SNAPSHOT" MINIMAX_H3_OUT="$PUBLISH_SCRATCH/dit/$tier" \
    MINIMAX_H3_TIER="$tier" MINIMAX_H3_COMPONENT=image_dit \
    "$CARGO_TARGET_DIR/release/examples/minimax_h3_prequant"
done
