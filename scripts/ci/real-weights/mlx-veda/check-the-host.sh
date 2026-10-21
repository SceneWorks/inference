# sc-26214: the MLX Veda device gate and dense-vs-sparse A/B run only on nax-macos-2 with >= 128 GB
# unified memory. Explicitly dispatched, so a wrong or too-small box FAILS here (before any build or
# GPU work) instead of reporting a green skip. The two dispatch inputs are untrusted and reach this
# step only as environment variables; both are validated here so a bad value fails before the build.
VEDA_OUT="$RUNNER_TEMP/veda-mlx-evidence"
mkdir -p "$VEDA_OUT"
echo "VEDA_OUT=$VEDA_OUT" >> "$GITHUB_ENV"
case "${VEDA_MLX_PHASE:-}" in
  device|ab|all) ;;
  *) echo "::error::veda_mlx_phase must be device, ab or all"; exit 1 ;;
esac
if [[ ! "${VEDA_MLX_CELLS:-}" =~ ^[1-3](,[1-3])*$ ]]; then
  echo "::error::veda_mlx_cells must be a comma-separated subset of 1,2,3 (digits and commas only)"
  exit 1
fi
mem_bytes="$(sysctl -n hw.memsize)"
mem_gb=$((mem_bytes / 1000000000))
pressure="$(/usr/bin/memory_pressure -Q 2>/dev/null || true)"
free_pct="$(printf '%s\n' "$pressure" | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
{
  echo "runner=$RUNNER_NAME phase=$VEDA_MLX_PHASE cells=$VEDA_MLX_CELLS"
  sysctl -n hw.model machdep.cpu.brand_string
  sw_vers
  echo "unified memory: $mem_gb GB; memory_pressure free ${free_pct:-?}%"
  echo "$pressure"
} | tee "$VEDA_OUT/host.txt"
if [[ "$RUNNER_NAME" != "nax-macos-2" ]]; then
  echo "::error::the MLX Veda lane runs only on nax-macos-2 (landed on $RUNNER_NAME); nothing was started"
  exit 1
fi
if (( mem_gb < 128 )); then
  echo "::error::only $mem_gb GB of unified memory (< 128 GB)"
  exit 1
fi
