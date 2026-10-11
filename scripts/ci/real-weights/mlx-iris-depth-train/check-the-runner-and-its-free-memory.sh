echo "runner: $RUNNER_NAME"
total_bytes=$(sysctl -n hw.memsize)
total_gb=$((total_bytes / 1073741824))
page=$(sysctl -n hw.pagesize)
used_gb=$(vm_stat | awk -v page="$page" '
  /Pages active/ {a=$3} /Pages wired down/ {w=$4} /Pages occupied by compressor/ {c=$5}
  END {gsub(/\./,"",a); gsub(/\./,"",w); gsub(/\./,"",c); printf "%d", (a + w + c) * page / 1073741824}')
echo "RAM total ${total_gb} GB; in use by other processes (active + wired + compressor) ${used_gb} GB"
memory_pressure | tail -n 1 || true
ok=true
if [ "$RUNNER_NAME" != "nax-macos-2" ]; then
  echo "::warning::SKIP: the depth full fine-tune is routed to nax-macos-2 only; this job landed on $RUNNER_NAME"
  ok=false
elif [ "$total_gb" -lt 128 ]; then
  echo "::warning::SKIP: ${total_gb} GB RAM is under the 128 GB a full 3B fine-tune (~60 GB, capped at 80 GB) needs"
  ok=false
elif [ "$used_gb" -gt 20 ]; then
  echo "::warning::SKIP: ${used_gb} GB is already in use by other processes (limit 20 GB)"
  ok=false
fi
echo "ok=$ok" >> "$GITHUB_OUTPUT"
