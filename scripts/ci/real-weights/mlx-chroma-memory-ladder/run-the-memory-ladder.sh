set -euo pipefail
tests="$CHROMA_LADDER_TESTS"
if [[ -z "${tests// }" ]]; then
  tests="$CHROMA_LADDER_ALL_TESTS"
fi
ran=0
for name in $tests; do
  echo "::group::$name"
  # `--exact` AFTER `--`: it is a libtest flag, and libtest is also what consumes
  # `--ignored`. The filter itself is cargo's positional test-name argument.
  out="$(cargo test --locked --release -p mlx-gen-chroma \
    --test integration \
    memory_ladder_real_weights::"$name" -- --exact --ignored --test-threads=1 --nocapture 2>&1 \
    | tee "$RUNNER_TEMP/chroma-ladder-$name.log" /dev/stderr)"
  echo "::endgroup::"
  # A rename, a typo, or a lost `#[ignore]` all produce "0 passed" and exit 0. This is
  # the guard that makes the lane's green mean something (sc-15520 review round 2).
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename or a filter typo would make this step vacuously green" >&2
    exit 1
  fi
  ran=$((ran + 1))
done
echo "ran $ran ladder test(s) for $CHROMA_MODEL"
if [[ "$ran" -eq 0 ]]; then
  echo "::error::the ladder ran no tests at all" >&2
  exit 1
fi
