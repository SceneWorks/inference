SUPRA2_OUT="$RUNNER_TEMP/supra2-mlx-evidence"
rm -rf "$SUPRA2_OUT" "$RUNNER_TEMP/supra2-bundle"
mkdir -p "$SUPRA2_OUT"
{
  echo "SUPRA2_OUT=$SUPRA2_OUT"
  echo "SUPRA2_OUTPUT_DIR=$SUPRA2_OUT/renders"
  echo "SUPRA2_BUNDLE=$RUNNER_TEMP/supra2-bundle"
} >> "$GITHUB_ENV"
{
  echo "runner=$RUNNER_NAME"
  sysctl -n hw.model machdep.cpu.brand_string
  sw_vers
} | tee "$SUPRA2_OUT/host.txt"
