# SW-7090 — MiniMax-H3-Image DiT on MLX: the published 644-tensor map, step-0 / step-3 parity
# against the diffusers reference, and a 2048² bf16 text-to-image render through the registered
# generator. Name-selected with `--exact` plus a pass-count assertion, so a rename cannot go
# vacuously green; one process per test, so each heavy test starts from a reaped allocator. Every
# gate runs and reports even when an earlier one fails; the step fails at the end if any did.
set -o pipefail
export MINIMAX_H3_IMAGE_RENDER_OUT="$RUNNER_TEMP/minimax-h3-image-2048.png"
failed=()
for name in h3_image_dit_names_match_the_published_index h3_image_dit_first_step_matches_diffusers h3_image_t2i_renders_2048_bf16; do
  status=0
  out="$(cargo test --locked --release -p mlx-gen-minimax-h3 --test integration \
    h3_image_real::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)" || status=$?
  if ((status)) || ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' failed or did not run exactly one passing test (exit $status) — a rename would make this step vacuously green" >&2
    failed+=("$name")
  fi
done
if ((${#failed[@]})); then
  echo "::error::failed gates: ${failed[*]}" >&2
  exit 1
fi
