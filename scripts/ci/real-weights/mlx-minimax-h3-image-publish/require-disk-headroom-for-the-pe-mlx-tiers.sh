# The PE MLX q8 and q4 tiers write 49 GB (q8 30.7 GB + q4 17.9 GB) beside the bf16 intermediate.
# The PE stage guard already reserved the full 105 GB peak (bf16 + q8 + q4) with 100 GB free before
# it, so this re-check asks only that 25 GB stay free after the write: it catches another writer on
# the host using the reserved space, without counting the reserved bytes twice.
set -- 49
df -h "$PUBLISH_SCRATCH"
free_gb=$(( $(df -k "$PUBLISH_SCRATCH" | awk 'NR==2 {print $4}') / 1048576 ))
if (( free_gb - $1 < 25 )); then
  echo "::error::$free_gb GB free under $PUBLISH_SCRATCH; the $1 GB write would leave less than 25 GB" >&2
  exit 1
fi
echo "$free_gb GB free; $(( free_gb - $1 )) GB will remain after the $1 GB write"
