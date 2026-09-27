if python3.12 scripts/release/qwen38_bonsai_terminal.py preflight --model-key bonsai-mlx-2bit --load-profile mlx-unified --output "$QWEN_BONSAI_OUTPUT_DIR/functional-mlx-bonsai-mlx-preflight.json"; then
  echo "bonsai=true" >> "$GITHUB_OUTPUT"
else
  echo "bonsai=false" >> "$GITHUB_OUTPUT"
fi
for language in pq2 ptq1; do
  for vision in bf16 q8; do
    if python3.12 scripts/release/qwen38_bonsai_terminal.py preflight --model-key bonsai-gguf --language-variant "$language" --vision-variant "$vision" --load-profile mlx-unified --output "$QWEN_BONSAI_OUTPUT_DIR/functional-mlx-${language}-${vision}-preflight.json"; then
      echo "${language}_${vision}=true" >> "$GITHUB_OUTPUT"
    else
      echo "${language}_${vision}=false" >> "$GITHUB_OUTPUT"
    fi
  done
done
exit 0
