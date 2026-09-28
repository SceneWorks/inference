set -o pipefail
name=long_clip_coherence_under_the_bounded_window
# Same split as the preflight above, so the guard and the run cannot disagree about which
# geometry — or, since sc-17807, which KV cache tier — is being measured.
geom="${KREA_S18_GEOMETRY%%@*}"
if [[ "$KREA_S18_GEOMETRY" == *@q* ]]; then
  export KREA_S18_KV_BITS="${KREA_S18_GEOMETRY##*@q}"
  echo "KV cache tier: q$KREA_S18_KV_BITS/g64 (sc-17807)"
fi
export KREA_SMOKE_W="${geom%x*}" KREA_SMOKE_H="${geom#*x}"
# The run-count assertion below is rows x seeds, and BOTH now come from dispatch inputs
# (sc-17655), so validate them here rather than letting a typo become a confusing count
# mismatch 4 hours later. `ABCDFEZ` is the row alphabet; a repeated letter selects one row
# but would inflate the expected count, so reject duplicates too.
if [[ ! "$KREA_S18_ROWS" =~ ^[ABCDFEZ]+$ ]]; then
  echo "::error::KREA_S18_ROWS='$KREA_S18_ROWS' is not a non-empty subset of ABCDFEZ" >&2
  exit 1
fi
if [[ -n "$(tr -d '[:space:]' <<<"$KREA_S18_ROWS" | fold -w1 | sort | uniq -d)" ]]; then
  echo "::error::KREA_S18_ROWS='$KREA_S18_ROWS' repeats a row" >&2
  exit 1
fi
# Count seeds the way the Rust side parses them: comma-separated, whitespace tolerated.
# The `|| true` is load-bearing: `grep -c` exits 1 when it counts zero, and under this
# step's `bash -e` plus `set -o pipefail` that kills the shell AT THE ASSIGNMENT — so
# without it the `-lt 1` branch below is unreachable and a bad seed list fails the step
# with no ::error:: line at all. Verified against `KREA_S18_SEEDS=abc`.
n_rows=${#KREA_S18_ROWS}
# EVERY non-blank field must be countable, not merely one of them — this shell and Rust's
# `u64::from_str` must agree on the seed COUNT or the assertion at the end of the run is
# measuring a different sweep than the one that ran. They disagree on more than junk:
# `+7` parses in Rust and is rejected here, and a value above u64::MAX is rejected there
# and accepted by a naive `[0-9]+`. So count the fields and the valid seeds separately and
# require they match, rather than silently skipping what this regex cannot read.
# `{1,19}` keeps every accepted value inside u64 (u64::MAX has 20 digits).
n_fields="$(tr ',' '\n' <<<"$KREA_S18_SEEDS" | grep -cE '[^[:blank:]]' || true)"
n_seeds="$(tr ',' '\n' <<<"$KREA_S18_SEEDS" | grep -cE '^[[:blank:]]*[0-9]{1,19}[[:blank:]]*$' || true)"
if [[ "$n_seeds" -lt 1 ]]; then
  echo "::error::KREA_S18_SEEDS='$KREA_S18_SEEDS' parsed to no seeds" >&2
  exit 1
fi
if [[ "$n_seeds" -ne "$n_fields" ]]; then
  echo "::error::KREA_S18_SEEDS='$KREA_S18_SEEDS' has $n_fields values but only $n_seeds this job can count as seeds (plain decimal, at most 19 digits) — the cell-count assertion would not match what the sweep runs" >&2
  exit 1
fi
# Seeds get the same duplicate check as rows: `7,7` measures one configuration twice, which
# counts as two cells here but reaches the verdict rule as a duplicate (row, seed) and is
# rejected there — better to say so before the run than after it.
# `[:blank:]` (space and tab), NOT `[:space:]` — the latter includes the newlines that
# separate the seeds, so it collapses `7,7` to a single `77` and the duplicate check
# silently never fires. Verified.
# Leading zeros are stripped first: `07` and `7` are the same seed to Rust, so comparing
# them textually would let a duplicate through to a late ladder panic.
if [[ -n "$(tr ',' '\n' <<<"$KREA_S18_SEEDS" | tr -d '[:blank:]' | sed -E 's/^0+([0-9])/\1/' | sort | uniq -d)" ]]; then
  echo "::error::KREA_S18_SEEDS='$KREA_S18_SEEDS' repeats a seed" >&2
  exit 1
fi
expected_cells=$(( n_rows * n_seeds ))
echo "expecting $expected_cells cells ($n_rows rows x $n_seeds seeds)"
# Streamed to a FILE as well as the log, because the evidence extraction below has to survive
# this command failing. `bash -e` kills the step at this assignment on a non-zero cargo exit,
# and the sweep's own verdict rule panics on evidence it judges unresolvable — precisely the
# run whose measured cells someone needs to read. Extraction and upload are separate `always()`
# steps for the same reason.
out="$(KREA_REALTIME_SNAPSHOT_DIR="$KREA_REALTIME_SNAPSHOT/q4" \
  cargo test --locked --release -p mlx-gen-krea-realtime --test integration \
  generate_smoke::"$name" -- --exact --ignored --nocapture 2>&1 \
  | tee "$RUNNER_TEMP/s18-sweep.log" /dev/stderr)"
if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
  echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
  exit 1
fi
cells="$(grep -c '^S18CELL' "$RUNNER_TEMP/s18-sweep.log" || true)"
if [[ "$cells" -ne "$expected_cells" ]]; then
  echo "::error::expected $expected_cells S18CELL rows (${KREA_S18_ROWS} x ${KREA_S18_SEEDS}), captured $cells — the sweep did not measure what this job claims" >&2
  exit 1
fi
