# sc-25690: the Iris full-fine-tune memory probes run only on a Mac with >= 128 GB unified memory.
# Explicitly dispatched, so a too-small box FAILS here (before any build or GPU work) instead of
# reporting a green skip. Only the total RAM is checked here; the runner name is provenance;
# `memory_pressure` is recorded for the evidence, and a BUSY box is refused per probe by
# run-the-guarded-probes.sh (>= 120 GB free right before each one).
IRIS_OUT="$RUNNER_TEMP/iris-memory-probe-mlx-evidence"
mkdir -p "$IRIS_OUT"
echo "IRIS_OUT=$IRIS_OUT" >> "$GITHUB_ENV"
mem_bytes="$(sysctl -n hw.memsize)"
mem_gb=$((mem_bytes / 1000000000))
pressure="$(/usr/bin/memory_pressure -Q 2>/dev/null || true)"
free_pct="$(printf '%s\n' "$pressure" | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
{
  echo "runner=$RUNNER_NAME"
  sysctl -n hw.model machdep.cpu.brand_string
  echo "unified memory: $mem_gb GB; memory_pressure free ${free_pct:-?}%"
  echo "$pressure"
} | tee "$IRIS_OUT/host.txt"
if (( mem_gb < 128 )); then
  echo "::error::only $mem_gb GB of unified memory (< 128 GB)"
  exit 1
fi
