# The three `#[ignore]`d real-weight tests on the GPU (the default MLX device), one invocation per
# name with `--exact`. Each must report exactly one passing test: a rename would otherwise make the
# step vacuously green.
bin="$(cat "$RUNNER_TEMP/supra2-mlx-test-binary")"
failed=0
for name in components_match_torch_reference trajectories_match_torch_reference registry_renders_published_defaults; do
  log="$SUPRA2_OUT/$name.log"
  rc=0
  "$bin" "real_weights::$name" --ignored --exact --nocapture --test-threads 1 > "$log" 2>&1 || rc=$?
  cat "$log"
  if (( rc != 0 )); then
    echo "::error::real_weights::$name failed (exit $rc)"
    failed=1
  elif ! grep -q "test result: ok. 1 passed" "$log"; then
    echo "::error::real_weights::$name did not run exactly one passing test - a rename would make this step vacuously green"
    failed=1
  fi
done
if (( failed )); then
  exit 1
fi
