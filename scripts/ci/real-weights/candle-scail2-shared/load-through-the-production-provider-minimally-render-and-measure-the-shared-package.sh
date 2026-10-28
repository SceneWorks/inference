set -o pipefail
evidence_log="$(cygpath -u "$RUNNER_TEMP")/scail2-shared-cuda.log"
export PATH="$(cygpath -u "$CUDA_PATH")/bin:$PATH"
profile_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
profile_gpu="${profile_gpu:-0}"
[[ "$profile_gpu" =~ ^[0-9]+$ ]] || {
  echo "CUDA_VISIBLE_DEVICES must start with a physical GPU ordinal, got ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  exit 1
}
{
  echo "[[CUDA_IDLE_RAW]] profileGpu=$profile_gpu visibleDevices=${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi -i "$profile_gpu" \
    --query-gpu=index,name,driver_version,pstate,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader
  for _ in 1 2 3 4 5 6; do
    nvidia-smi -i "$profile_gpu" --query-gpu=index,memory.used \
      --format=csv,noheader,nounits
    sleep 0.2
  done
  nvidia-smi pmon -i "$profile_gpu" -c 1 -s um
} 2>&1 | tee -a "$evidence_log"
cargo test --locked --release -p candle-gen-scail2 --features cuda \
  pipeline::tests::shared_bf16_real_weights_cuda_loads_and_renders_with_measured_peak \
  -- --ignored --exact --nocapture 2>&1 | tee -a "$evidence_log"
grep -Fq '[[SCAIL2_CUDA_VRAM]]' "$evidence_log" || {
  echo 'the real-weight test passed without emitting its machine-readable VRAM record' >&2
  exit 1
}
