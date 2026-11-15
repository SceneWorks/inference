# SW-7104 (AT9) — MLX: render a short t2i prompt and the PE's expansion of it at one seed
# (h3_image_acceptance_real::h3_image_pe_expanded_prompt_renders_more_detail). The expansion is case
# `t2i-0` of the `mlx-minimax-h3-image-pe` job's q8 tier run (mlx-q8.json, downloaded beside this
# job); its short prompt is core-llm's `minimax_h3_image_pe::quality::T2I_PROMPTS[0]`, which the
# test refuses to accept unless the expansion is longer. The gate is name-selected with `--exact`,
# built with the registered `model-inference-proof` feature, run with MODEL_INFERENCE_CASE_FQN set
# and checked by verify_model_inference_log.py. The PNGs and the log go to
# $RUNNER_TEMP/minimax-h3-image-acceptance.
set -o pipefail
out="$RUNNER_TEMP/minimax-h3-image-acceptance"
mkdir -p "$out"
export MINIMAX_H3_IMAGE_ACCEPTANCE_OUT="$out"
evidence="$RUNNER_TEMP/minimax-pe-mlx-evidence/mlx-q8.json"
test -s "$evidence" || { echo "::error::the PE job's mlx-q8.json is missing"; exit 1; }
export MINIMAX_H3_IMAGE_PE_SHORT_PROMPT="a red bicycle leaning against a brick wall"
export MINIMAX_H3_IMAGE_PE_EXPANDED_FILE="$out/pe-expanded-t2i-0.txt"
MINIMAX_H3_IMAGE_PE_LABEL="$(python3.12 -I scripts/ci/h3_image_pe_expansion.py "$evidence" t2i-0 "$MINIMAX_H3_IMAGE_PE_EXPANDED_FILE")"
export MINIMAX_H3_IMAGE_PE_LABEL
name=h3_image_acceptance_real::h3_image_pe_expanded_prompt_renders_more_detail
log="$out/h3_image_pe_expanded_prompt_renders_more_detail.log"
status=0
MODEL_INFERENCE_CASE_FQN="$name" cargo test --locked --release -p mlx-gen-minimax-h3 \
  --features model-inference-proof --test integration \
  "$name" -- --exact --ignored --nocapture 2>&1 | tee "$log" >&2 || status=$?
if ((status)); then
  echo "::error::'$name' failed (exit $status)" >&2
  exit 1
fi
python3.12 -I scripts/ci/verify_model_inference_log.py --package mlx-gen-minimax-h3 --backend mlx --fqn "$name" --log "$log"
