echo "runner: $RUNNER_NAME"
total_bytes=$(sysctl -n hw.memsize)
total_gb=$((total_bytes / 1073741824))
page=$(sysctl -n hw.pagesize)
used_gb=$(vm_stat | awk -v page="$page" '
  /Pages active/ {a=$3} /Pages wired down/ {w=$4} /Pages occupied by compressor/ {c=$5}
  END {gsub(/\./,"",a); gsub(/\./,"",w); gsub(/\./,"",c); printf "%d", (a + w + c) * page / 1073741824}')
free_pct=$(memory_pressure | awk -F': ' '/System-wide memory free percentage/ {gsub(/%/,"",$2); print int($2)}')
free_gb=$((total_gb * ${free_pct:-0} / 100))
echo "RAM total ${total_gb} GB; memory_pressure free ${free_pct}% = ${free_gb} GB; active + wired + compressor ${used_gb} GB (informational)"
ok=true
if [ "$RUNNER_NAME" != "nax-macos-2" ]; then
  echo "::warning::SKIP: the depth full fine-tune is routed to nax-macos-2 only; this job landed on $RUNNER_NAME"
  ok=false
elif [ "$total_gb" -lt 128 ]; then
  echo "::warning::SKIP: ${total_gb} GB RAM is under the 128 GB a full 3B fine-tune (~65 GB, capped at 80 GB) needs"
  ok=false
elif [ -z "$free_pct" ] || [ "$free_gb" -lt 85 ]; then
  echo "::warning::SKIP: ${free_gb} GB free (memory_pressure ${free_pct:-?}%) is under the 85 GB needed (~65 GB peak + 20 GB headroom)"
  ok=false
fi
echo "ok=$ok" >> "$GITHUB_OUTPUT"
