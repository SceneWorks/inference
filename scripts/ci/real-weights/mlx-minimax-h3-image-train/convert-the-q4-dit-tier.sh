rm -rf "$MINIMAX_H3_IMAGE_TIERS_ROOT"
mkdir -p "$MINIMAX_H3_IMAGE_TIERS_ROOT"
MINIMAX_H3_SRC="$MINIMAX_H3_IMAGE_SNAPSHOT" MINIMAX_H3_OUT="$MINIMAX_H3_IMAGE_TIERS_ROOT/q4" \
  MINIMAX_H3_TIER=q4 MINIMAX_H3_COMPONENT=image_dit \
  cargo run --locked --release -p mlx-gen-minimax-h3 --example minimax_h3_prequant
