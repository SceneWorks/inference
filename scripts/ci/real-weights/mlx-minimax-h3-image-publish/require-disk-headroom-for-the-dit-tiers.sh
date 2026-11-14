# The DiT tiers write 55 GB at peak (q8 35.3 GB + q4 18.8 GB). The volume must have at least 100 GB free
# before the write, and keep at least 25 GB free after it so the box never runs out of space.
set -- 55
df -h "$PUBLISH_SCRATCH"
free_gb=$(( $(df -k "$PUBLISH_SCRATCH" | awk 'NR==2 {print $4}') / 1048576 ))
if (( free_gb < 100 || free_gb - $1 < 25 )); then
  echo "::error::$free_gb GB free under $PUBLISH_SCRATCH; the conversion writes $1 GB and needs at least 100 GB free before it and 25 GB left after it" >&2
  exit 1
fi
echo "$free_gb GB free; $(( free_gb - $1 )) GB will remain after the $1 GB write"
