# The PE MLX q8 and q4 tiers write 49 GB (q8 30.7 GB + q4 17.9 GB) beside the bf16 intermediate. Re-checked
# here because other work on the host can use the space between steps: at least 100 GB free before the
# write and at least 25 GB left after it.
set -- 49
df -h "$PUBLISH_SCRATCH"
free_gb=$(( $(df -k "$PUBLISH_SCRATCH" | awk 'NR==2 {print $4}') / 1048576 ))
if (( free_gb < 100 || free_gb - $1 < 25 )); then
  echo "::error::$free_gb GB free under $PUBLISH_SCRATCH; the conversion writes $1 GB and needs at least 100 GB free before it and 25 GB left after it" >&2
  exit 1
fi
echo "$free_gb GB free; $(( free_gb - $1 )) GB will remain after the $1 GB write"
