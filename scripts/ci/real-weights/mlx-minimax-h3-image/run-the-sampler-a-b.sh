# SW-7106 — the H3-Image sampler A/B on MLX: the registered generator renders two prompts at two
# seeds, 2048² bf16, with the released velocity Euler and with ComfyUI's stochastic LCM (s_noise
# 1.1 -> 0.9). The gate asserts every render is non-degenerate (pixel std above 10) and that each
# LCM render differs from its Euler twin; the eight PNGs and the log are uploaded as
# minimax-h3-image-sampler-ab-mlx. Name-selected with `--exact` plus a pass-count assertion.
set -o pipefail
export MINIMAX_H3_IMAGE_SAMPLER_AB_OUT="$RUNNER_TEMP/minimax-h3-image-sampler-ab"
mkdir -p "$MINIMAX_H3_IMAGE_SAMPLER_AB_OUT"
name=h3_image_sampler_real::h3_image_sampler_ab_renders_2048_bf16
status=0
out="$(cargo test --locked --release -p mlx-gen-minimax-h3 --test integration \
  "$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)" || status=$?
printf '%s\n' "$out" > "$MINIMAX_H3_IMAGE_SAMPLER_AB_OUT/sampler-ab.log"
if ((status)) || ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
  echo "::error::'$name' failed or did not run exactly one passing test (exit $status)" >&2
  exit 1
fi
