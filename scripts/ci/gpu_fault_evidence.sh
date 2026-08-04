#!/usr/bin/env bash
# Record what the machine's GPU and memory were doing across a macOS real-weight lane, so a
# Metal command-buffer failure is diagnosable from the run log instead of only announced by it.
#
# WHY THIS EXISTS (sc-17355). Run 30869410054 failed the Krea Realtime LoRA gate with
#
#   [METAL] Command buffer execution failed: Ignored (for causing prior/excessive GPU errors)
#   (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)
#
# `SubmissionsIgnored` is a CASCADE: this command buffer was dropped because an EARLIER
# submission faulted. The earlier fault is what names a cause, and it is not in the run log --
# so the failure is un-diagnosable after the fact, and a re-dispatch (which passed) destroys the
# machine state that would have explained it. Evidence has to be collected DURING the run that
# fails; there is no retro-capture. This is that collection, in place before the recurrence
# rather than added after it.
#
# NEITHER HALF OF THE STORY'S PROPOSED COMMAND SURVIVED MEASUREMENT, WHICH IS WHY BOTH CHANGED.
# sc-17355 suggested `log stream --predicate 'subsystem == "com.apple.gpu"'`. Both halves fail,
# independently, on nax-macos (macOS 25.5.0):
#
#   * The PREDICATE matches NOTHING. IOGPU faults are emitted with an EMPTY `subsystem` and an
#     empty `category`; the only field that identifies them is `senderImagePath` =
#     /System/Library/PrivateFrameworks/IOGPU.framework/Versions/A/IOGPU (verified by dumping
#     `log show --style ndjson` fields for real IOGPU messages). Over a 3-day window the
#     subsystem form returned zero events while sender-based matching returned real faults.
#
#   * `log stream` DROPS MESSAGES under load, and load is the only condition that matters here.
#     During a local run that ended in a memory kill, the stream emitted 20+ "Messages dropped
#     during live streaming (use `log show` to see what they were)" markers and NONE of the
#     events themselves; `log show` over the identical window returned the IOGPU event the
#     stream had dropped. A live stream is the one collection method guaranteed to fail exactly
#     when the machine is in the state being investigated. Hence `report` reads the log AFTER
#     the fact, bounded to the job's window, rather than tailing it during.
#
# So the story's command would have produced an empty file for two independent reasons. Anyone
# reaching for it at the recurrence should read this block first.
#
# `memorystatus` IS IN THE PREDICATE ON PURPOSE. The story's hypothesis is memory pressure, and
# the kernel writes the most direct possible evidence for it -- during the local kill it logged
# `memorystatus: System is unhealthy`, `{"compressor_exhausted": 1, ...}` and
# `killing due to "vm-compressor-space-shortage"`, naming the mechanism outright. None of that
# contains "IOGPU", "AGX" or "command buffer", so a GPU-only predicate silently discards the
# single most informative record on the box.
#
# WHAT IS PROVEN AND WHAT IS NOT. The memory half is measured: `vm_stat` sampling reproduced the
# trajectory of a real resource death (compressed memory 1.6 -> 106 GiB across one render), and
# the `memorystatus` clause is confirmed against a real kernel kill. What is UNPROVEN is whether
# the driver logs anything for `kIOGPUCommandBufferCallbackError*` specifically -- nax-macos has
# recorded no such event in the retained window, so there was no positive control, and MLX's own
# `mlx_pmetal_test_inject_command_buffer_error` hook fakes the error above the driver and emits
# no driver log line either. The predicate is deliberately wider than the target message for
# that reason: on a fault run the useful artefact may be the DIFF against a clean run's events,
# not a single matching line. If a recurrence yields a memory trajectory and no GPU events, that
# is itself an answer -- and only if this ran.
#
# REPORT-ONLY, DELIBERATELY. Every path exits 0, and the calling steps additionally set
# `continue-on-error: true`. A step whose only product is a record must never be able to fail a
# multi-hour real-weight lane -- the same rule as `report_runner_disk_headroom.sh`.

set -u

STATE_DIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/gpu-fault-evidence"
SAMPLER_PID_FILE="$STATE_DIR/sampler.pid"
MEM_CSV="$STATE_DIR/memory.csv"
START_FILE="$STATE_DIR/started-at"

# `log show --start` wants a local-time stamp in this exact shape; `date -u` would silently
# offset the window by the box's UTC offset and return events from the wrong hours.
now_local() { date '+%Y-%m-%d %H:%M:%S'; }

start() {
  # Seconds of coverage. Defaults to the shorter lane's cap; a longer job passes its own.
  # `[0-9]` guard so a typo'd argument degrades to the default instead of `[: bad number`.
  max_seconds="${1:-7200}"
  case "$max_seconds" in
    "" | *[!0-9]*) max_seconds=7200 ;;
  esac
  mkdir -p "$STATE_DIR" 2>/dev/null || return 0
  # Idempotent: a re-run of this step (a retried job reuses `$RUNNER_TEMP`) would otherwise leave
  # the first sampler running and orphaned -- two writers on one CSV, and only one of them killable
  # by `report`.
  if [ -f "$SAMPLER_PID_FILE" ]; then
    kill "$(cat "$SAMPLER_PID_FILE" 2>/dev/null)" 2>/dev/null
    rm -f "$SAMPLER_PID_FILE"
  fi
  now_local > "$START_FILE" 2>/dev/null || return 0

  # 1 Hz `vm_stat` is ~nothing next to a 14B render, and it is the only record of the machine
  # state at the moment of a failure: `get_peak_memory()` inside a test dies with the test, and
  # a post-hoc reading is taken after the pressure has already drained away.
  #
  # SELF-TERMINATING, AND THE BOUND MUST MATCH THE CALLING JOB. `report` may never run -- the job
  # can be cancelled, or the runner can lose the step -- and a sampler that outlives its job would
  # still be writing during the NEXT lane on this box. So the loop is bounded. But the bound is a
  # ceiling on COVERAGE as much as on leakage: the two callers are not the same length, and the
  # regression lane's `timeout-minutes: 120` is the short one. The S18 sweep is `timeout-minutes:
  # 480` and really does run ~4.3 hours, so a fixed 2h bound would have stopped recording less than
  # halfway through the longer of the two lanes -- silently, and precisely in the job most likely
  # to hit a pressure fault. Hence a per-caller argument rather than a constant.
  #
  # ALL THREE FDS REDIRECTED, and that is load-bearing rather than tidy. The Actions runner ends a
  # step when the step's stdout/stderr pipes close, so a background child that inherits them keeps
  # the step open until it exits -- here, for the whole `max_seconds`. A record-keeping step that
  # can hang a lane is far worse than no record. stdin is closed for the same reason.
  #
  # `compressed` and `swap_used` are in here deliberately, not for completeness. Unified memory
  # means a 14B render's working set and the machine's page cache are the same pool, and the state
  # that precedes a resource death is not "free is low" -- free is always low on macOS -- it is the
  # compressor filling and swap being written. Measured on nax-macos while a real-weight lane ran:
  # 62 GiB compressed and 29 GiB of 30 GiB swap in use. A record without those two columns cannot
  # distinguish a box with headroom from one that is out of it.
  (
    echo "iso,active_gib,wired_gib,compressed_gib,free_gib,swap_used_gib,max_rss_gib"
    i=0
    while [ "$i" -lt "$max_seconds" ]; do
      vm_stat | awk -v stamp="$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
                    -v rss="$(ps -Ao rss= | sort -rn | head -1)" \
                    -v swap="$(sysctl -n vm.swapusage 2>/dev/null)" -F: '
        /page size of/ { gsub(/[^0-9]/, "", $0); page = $0 }
        /Pages free/ { gsub(/[^0-9]/, "", $2); free = $2 }
        /Pages active/ { gsub(/[^0-9]/, "", $2); active = $2 }
        /Pages wired down/ { gsub(/[^0-9]/, "", $2); wired = $2 }
        /Pages occupied by compressor/ { gsub(/[^0-9]/, "", $2); comp = $2 }
        END {
          gib = page / 1073741824
          # `vm.swapusage` reads "total = 30720.00M  used = 29016.00M  free = 1704.00M". The unit
          # suffix is not fixed -- it is whatever scale the figure lands in -- so read it rather
          # than assume M, or a box that reports G silently logs a 1024x understatement.
          used_gib = 0
          n = split(swap, tok, /[ \t]+/)
          for (j = 1; j <= n; j++) {
            if (tok[j] == "used" && j + 2 <= n) {
              value = tok[j + 2]
              unit = substr(value, length(value), 1)
              sub(/[A-Za-z]$/, "", value)
              scale = (unit == "G") ? 1 : (unit == "M") ? 1024 : (unit == "K") ? 1048576 : 1
              used_gib = (value + 0) / scale
            }
          }
          printf "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
            stamp, active * gib, wired * gib, comp * gib, free * gib, used_gib, (rss + 0) / 1048576
        }'
      i=$((i + 1))
      sleep 1
    done
  ) > "$MEM_CSV" 2>/dev/null < /dev/null &
  echo $! > "$SAMPLER_PID_FILE" 2>/dev/null

  echo "gpu-fault evidence: sampling from $(cat "$START_FILE" 2>/dev/null) for up to ${max_seconds}s (pid $(cat "$SAMPLER_PID_FILE" 2>/dev/null))"
  return 0
}

report() {
  if [ -f "$SAMPLER_PID_FILE" ]; then
    kill "$(cat "$SAMPLER_PID_FILE")" 2>/dev/null
    rm -f "$SAMPLER_PID_FILE"
  fi

  echo "== memory trajectory =="
  if [ -s "$MEM_CSV" ]; then
    # The peaks are the claim under test (sc-17355's hypothesis is memory pressure), and the
    # tail is where a failure landed. Printing the whole file would bury both in ~7000 lines.
    awk -F, 'NR > 1 {
        if ($2 > pa) pa = $2
        if ($3 > pw) pw = $3
        if ($4 > pc) pc = $4
        if ($6 > ps) ps = $6
        if ($7 > pr) pr = $7
        n++
      }
      END {
        if (n) printf "samples %d  peak active %.2f GiB  peak wired %.2f GiB  peak compressed %.2f GiB  peak swap-used %.2f GiB  peak single-process RSS %.2f GiB\n", n, pa, pw, pc, ps, pr
        else print "no samples recorded"
      }' "$MEM_CSV"
    echo "-- last 30 samples --"
    tail -30 "$MEM_CSV"
  else
    echo "no memory samples — 'start' did not run, or the sampler could not write to $MEM_CSV"
  fi

  echo
  echo "== GPU / Metal events for this job's window =="
  if [ -f "$START_FILE" ]; then
    # See the header: sender- and message-based, NOT `subsystem == "com.apple.gpu"`, which
    # matches nothing. `MTLCompilerService` XPC churn and per-kernel "Metal Compiling Shader"
    # lines are excluded by construction -- they are thousands of lines of routine noise and
    # would make this unreadable in exactly the run where it has to be read.
    #
    # The `/usr/bin/log` exclusion is not tidiness. `log` records its own non-interactive
    # invocation INCLUDING its argument vector, and that argument vector contains the string
    # "IOGPU" -- so without this clause the query matches itself and every run reports one
    # spurious hit. A reader scanning for "did anything GPU-ish happen" would find something
    # every time, which is the same as finding nothing.
    events="$STATE_DIR/gpu-events.txt"
    if /usr/bin/log show \
      --start "$(cat "$START_FILE")" \
      --style compact \
      --predicate '(senderImagePath CONTAINS "IOGPU" OR eventMessage CONTAINS "IOGPU" OR eventMessage CONTAINS "kIOGPU" OR eventMessage CONTAINS[c] "command buffer" OR eventMessage CONTAINS[c] "gpu restart" OR eventMessage CONTAINS[c] "memorystatus" OR eventMessage CONTAINS[c] "jetsam" OR (senderImagePath CONTAINS "AGX" AND messageType == "Error")) AND processImagePath != "/usr/bin/log"' \
      > "$events" 2>/dev/null
    then
      total="$(wc -l < "$events" | tr -d ' ')"
      echo "$total matching event line(s)"
      # BOTH ends, never `head` alone. `log show` prints chronologically, and the two things worth
      # reading sit at opposite ends: the FIRST fault (which is the whole point -- the cascade names
      # no cause) is early, while the failure being explained is at the very end of a ~28-minute
      # job. Truncating from the top would keep the earliest events and discard the failure; from
      # the bottom, the reverse. This box logs unrelated IOGPU device-open faults from system
      # daemons several times a day, so overflowing a cap is realistic, not theoretical.
      if [ "$total" -le 300 ]; then
        cat "$events"
      else
        head -120 "$events"
        echo "... $((total - 240)) line(s) elided ..."
        tail -120 "$events"
      fi
    else
      echo "log show failed — no GPU event record for this run"
    fi
  else
    echo "no start marker — 'start' did not run"
  fi

  return 0
}

case "${1:-}" in
  start) start "${2:-}" ;;
  report) report ;;
  *) echo "usage: $0 start [max_seconds] | report" ;;
esac

exit 0
