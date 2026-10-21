# sc-26214 — the MLX Veda dense-vs-sparse A/B (port of the dev-Mac `run_all.sh` / `run_cell.sh`
# queue, cells c1-c3). Each selected cell renders TWICE through the integration binary's
# `veda_ab::veda_ab_render_one_cell`, `VEDA_AB_VEDA=off` then `on`, one process per render, each under
# the external phys_footprint guard (scripts/ci/iris_restore_footprint_guard.py: cap 110 GB, 90 min
# wall). Same seed / frames / geometry / tier / recipes / LoRAs / prompts as the dev-Mac queue.
# Cells 2 (fl2va) and 3 (r2va) condition on frames of cell 1's DENSE render, extracted from its raw
# `frames.rgb` (no ffmpeg needed), so they require cell 1 in the same run.
# Report-only: no speed threshold gates the job. It is red when a render fails or is guard-stopped,
# or a selected cell lacks both result.json files.
if [[ ! "${VEDA_MLX_CELLS:-}" =~ ^[1-3](,[1-3])*$ ]]; then
  echo "::error::veda_mlx_cells must be a comma-separated subset of 1,2,3"
  exit 1
fi
want=",$VEDA_MLX_CELLS,"
if [[ "$want" != *,1,* && ( "$want" == *,2,* || "$want" == *,3,* ) ]]; then
  echo "::error::cells 2 and 3 condition on cell 1's dense render; select cell 1 in the same run (e.g. 1,2,3)"
  exit 1
fi
bin="$(cat "$RUNNER_TEMP/veda-mlx-integration-test-binary")"
T="$VEDA_AB_TURBO_SNAPSHOT"
inputs="$VEDA_OUT/inputs"
export VEDA_AB_SEED=17147 VEDA_AB_FRAMES=124 VEDA_AB_TIER=q4 VEDA_AB_WIDTH=1344 VEDA_AB_HEIGHT=768
failed=0
cells=()
render() { # $1 cell dir, $2 arm (off|on), rest: the cell's VEDA_AB_* assignments
  local dir="$1/$2" arm="$2" rc=0 extra=(VEDA_AB_VEDA=off)
  shift 2
  mkdir -p "$dir"
  [[ "$arm" == on ]] && extra=(VEDA_AB_VEDA=on "VEDA_AB_PREDICTOR=$VEDA_PREDICTOR")
  echo "render $dir: $(/usr/bin/memory_pressure -Q 2>/dev/null | sed -n 's/.*\(free percentage: [0-9]*%\).*/\1/p')"
  env "$@" "${extra[@]}" "VEDA_AB_OUT=$dir" TMPDIR="$RUNNER_TEMP" \
    python3.12 scripts/ci/iris_restore_footprint_guard.py 110 5400 \
    "$bin" veda_ab::veda_ab_render_one_cell --ignored --exact --nocapture --test-threads 1 \
    > "$dir/log.txt" 2>&1 || rc=$?
  tail -n 40 "$dir/log.txt"
  if grep -q '^\[guard\] SIGTERM' "$dir/log.txt"; then
    echo "::error::$dir was stopped by the footprint guard"
    failed=1
  elif (( rc != 0 )); then
    echo "::error::$dir exited $rc"
    failed=1
  elif ! grep -q '^test result: ok\. 1 passed; 0 failed' "$dir/log.txt"; then
    echo "::error::$dir ran no test (veda_ab::veda_ab_render_one_cell not selected)"
    failed=1
  fi
}
cell() { # $1 cell name, rest: the cell's VEDA_AB_* assignments
  local name="$1" out="$VEDA_OUT/$1"
  shift
  cells+=("$name")
  render "$out" off "$@"
  render "$out" on "$@"
  if [[ -f "$out/off/result.json" && -f "$out/on/result.json" ]]; then
    python3.12 scripts/ci/veda_ab_summary.py cell "$out" || failed=1
    if command -v ffmpeg > /dev/null; then
      python3.12 scripts/veda_ab_compare.py "$out/off" "$out/on" --out "$out/side_by_side.mp4" \
        > "$out/compare.json" 2> "$out/compare.err" || echo "::warning::$name compare failed (see compare.err)"
    else
      echo "::notice::ffmpeg is not on PATH; $name PSNR/SNR compare and side-by-side skipped"
    fi
  else
    echo "::error::$name lacks off/on result.json"
    failed=1
  fi
}
c1="c1_t2va_t8_q4_1344x768"
if [[ "$want" == *,1,* ]]; then
  cell "$c1" VEDA_AB_MODE=t2va VEDA_AB_RECIPE=minimax_h3_turbo_8step \
    "VEDA_AB_LORAS=$T/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors" \
    "VEDA_AB_PROMPT=A red fox trots through fresh snow in a sunlit pine forest, breath steaming, camera tracking alongside; soft crunch of footsteps and distant birdsong."
fi
if [[ "$want" == *,2,* || "$want" == *,3,* ]]; then
  dense="$VEDA_OUT/$c1/off"
  if [[ -f "$dense/result.json" && -f "$dense/frames.rgb" ]]; then
    mkdir -p "$inputs"
    # run_all.sh's ffmpeg selections: frame 0, the last frame, frame 62.
    for pick in "0 first" "-1 last" "62 ref"; do
      read -r index png <<< "$pick"
      python3.12 scripts/ci/veda_extract_frame_png.py "$dense" "$index" "$inputs/$png.png" || failed=1
    done
  else
    echo "::error::cell 1's dense render left no frames; cells 2 and 3 cannot be conditioned"
    failed=1
  fi
fi
if [[ "$want" == *,2,* && -f "$inputs/first.png" && -f "$inputs/last.png" ]]; then
  cell c2_fl2va_t4768_q4_1344x768 VEDA_AB_MODE=fl2va \
    "VEDA_AB_FIRST_IMAGE=$inputs/first.png" "VEDA_AB_LAST_IMAGE=$inputs/last.png" \
    VEDA_AB_RECIPE=minimax_h3_turbo_4step_768p \
    "VEDA_AB_LORAS=$T/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors" \
    "VEDA_AB_PROMPT=The fox turns and bounds away between the pines, snow spraying; footsteps and wind."
elif [[ "$want" == *,2,* ]]; then
  cells+=(c2_fl2va_t4768_q4_1344x768)
fi
if [[ "$want" == *,3,* && -f "$inputs/ref.png" ]]; then
  cell c3_r2va_ref2v_t4_q4_1344x768 VEDA_AB_MODE=r2va "VEDA_AB_REF_IMAGES=$inputs/ref.png" \
    VEDA_AB_RECIPE=minimax_h3_ref2v_turbo_4step \
    "VEDA_AB_LORAS=$T/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors" \
    "VEDA_AB_PROMPT=The same red fox sits on a snowy log at dusk and yawns; quiet wind and a distant owl."
elif [[ "$want" == *,3,* ]]; then
  cells+=(c3_r2va_ref2v_t4_q4_1344x768)
fi
python3.12 scripts/ci/veda_ab_summary.py aggregate "$VEDA_OUT" "${cells[@]}" || failed=1
cat "$VEDA_OUT/ab_summary.json"
exit "$failed"
