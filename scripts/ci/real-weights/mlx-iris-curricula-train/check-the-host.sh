# sc-25689: this lane runs only on nax-macos-2 with >= 128 GB unified memory and no heavy memory
# pressure; anywhere else it records why and skips (IRIS_CURRICULA_SKIP=1), it never queues work
# onto a busy box.
IRIS_OUT="$RUNNER_TEMP/iris-curricula-mlx-evidence"
mkdir -p "$IRIS_OUT"
echo "IRIS_OUT=$IRIS_OUT" >> "$GITHUB_ENV"
mem_bytes="$(sysctl -n hw.memsize)"
mem_gb=$((mem_bytes / 1000000000))
pressure="$(/usr/bin/memory_pressure -Q 2>/dev/null || true)"
free_pct="$(printf '%s\n' "$pressure" | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
free_gb=$(( mem_gb * ${free_pct:-0} / 100 ))
{
  echo "runner=$RUNNER_NAME"
  sysctl -n hw.model machdep.cpu.brand_string
  echo "unified memory: $mem_gb GB; memory_pressure free ${free_pct:-?}% = ${free_gb} GB"
  echo "$pressure"
  vm_stat | head -12
} | tee "$IRIS_OUT/host.txt"
skip=""
if [[ "$RUNNER_NAME" != "nax-macos-2" ]]; then
  skip="runner $RUNNER_NAME is not nax-macos-2"
elif (( mem_gb < 128 )); then
  skip="only $mem_gb GB of unified memory (< 128 GB)"
elif [[ -z "$free_pct" || "$free_pct" -lt 40 ]]; then
  skip="memory pressure too high (free ${free_pct:-unknown}% < 40%)"
fi
if [[ -n "$skip" ]]; then
  echo "::warning::skipping the sc-25689 MLX curricula evidence: $skip"
  echo "IRIS_CURRICULA_SKIP=1" >> "$GITHUB_ENV"
else
  echo "IRIS_CURRICULA_SKIP=0" >> "$GITHUB_ENV"
fi
