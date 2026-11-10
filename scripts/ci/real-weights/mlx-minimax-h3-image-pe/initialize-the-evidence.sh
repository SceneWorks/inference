MINIMAX_PE_OUT="$RUNNER_TEMP/minimax-pe-mlx-evidence"
rm -rf "$MINIMAX_PE_OUT" "$RUNNER_TEMP/minimax-pe-cuda-evidence"
mkdir -p "$MINIMAX_PE_OUT"
# The converted tiers of the pinned revision, in the lane-owned internal weights root; rewritten by
# every run.
MINIMAX_PE_TIERS="$HOME/sceneworks-rw-weights/minimax-h3-image-pe-tiers/35603d4b63949afa12cb098227fffefedf081afa"
{
  echo "MINIMAX_PE_OUT=$MINIMAX_PE_OUT"
  echo "MINIMAX_PE_TIERS=$MINIMAX_PE_TIERS"
} >> "$GITHUB_ENV"
{
  echo "runner=$RUNNER_NAME"
  sysctl -n hw.model hw.memsize machdep.cpu.brand_string
  sw_vers
} | tee "$MINIMAX_PE_OUT/host.txt"
