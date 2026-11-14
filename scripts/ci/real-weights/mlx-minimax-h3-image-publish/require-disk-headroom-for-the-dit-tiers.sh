# The DiT tiers write 55 GB (q8 35.3 GB + q4 18.8 GB); the volume must keep 100 GB free beyond that.
set -- 55
df -h "$PUBLISH_SCRATCH"
free_gb=$(( $(df -k "$PUBLISH_SCRATCH" | awk 'NR==2 {print $4}') / 1048576 ))
need=$(( $1 + 100 ))
if (( free_gb < need )); then
  echo "::error::$free_gb GB free under $PUBLISH_SCRATCH; the conversion needs $need GB (its $1 GB plus 100 GB headroom)" >&2
  exit 1
fi
echo "$free_gb GB free, $need GB needed"
