# sc-26234: the distilled FLUX 3 Image Klein MLX evidence (text-to-image through the catalog +
# the component parity cases) runs on the `rw-mage` weight set (nax-macos-2). The bf16 package is ~20 GB resident (11 GB DiT +
# 8 GB Qwen3-VL tower + 0.3 GB autoencoder) plus the 1024-squared activations, so the host needs
# >= 48 GB of unified memory and no heavy memory pressure; anywhere else it records why and skips
# (FLUX3_SKIP=1), it never queues work onto a busy box. The size / steps are job env fixed to the
# release geometry (the dispatch form sits at GitHub's 25-input cap); both are validated here anyway.
FLUX3_OUT="$RUNNER_TEMP/flux3-klein-mlx-evidence"
mkdir -p "$FLUX3_OUT"
echo "FLUX3_OUT=$FLUX3_OUT" >> "$GITHUB_ENV"
if [[ ! "${FLUX3_REAL_SIZE:-1024}" =~ ^[0-9]{2,4}$ ]] || (( FLUX3_REAL_SIZE % 16 != 0 || FLUX3_REAL_SIZE < 64 || FLUX3_REAL_SIZE > 2048 )); then
  echo "::error::FLUX3_REAL_SIZE must be a multiple of 16 in 64..=2048"
  exit 1
fi
if [[ ! "${FLUX3_REAL_STEPS:-16}" =~ ^[1-9][0-9]?$ ]]; then
  echo "::error::FLUX3_REAL_STEPS must be 1-99"
  exit 1
fi
mem_bytes="$(sysctl -n hw.memsize)"
mem_gb=$((mem_bytes / 1000000000))
pressure="$(/usr/bin/memory_pressure -Q 2>/dev/null || true)"
free_pct="$(printf '%s\n' "$pressure" | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
{
  echo "runner=$RUNNER_NAME size=${FLUX3_REAL_SIZE:-1024} steps=${FLUX3_REAL_STEPS:-16}"
  sysctl -n hw.model machdep.cpu.brand_string
  sw_vers
  echo "unified memory: $mem_gb GB; memory_pressure free ${free_pct:-?}%"
  echo "$pressure"
} | tee "$FLUX3_OUT/host.txt"
skip=""
if (( mem_gb < 48 )); then
  skip="only $mem_gb GB of unified memory (< 48 GB)"
elif [[ -z "$free_pct" || "$free_pct" -lt 30 ]]; then
  skip="memory pressure too high (free ${free_pct:-unknown}% < 30%)"
fi
if [[ -n "$skip" ]]; then
  echo "::warning::skipping the sc-26234 FLUX 3 MLX evidence: $skip"
  echo "FLUX3_SKIP=1" >> "$GITHUB_ENV"
else
  echo "FLUX3_SKIP=0" >> "$GITHUB_ENV"
fi
