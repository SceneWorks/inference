mkdir -p "$RUNNER_TEMP/qwen38-bonsai-aggregate"
"$REVIEWED_PYTHON" scripts/release/qwen38_bonsai_artifacts.py select --role matrix --runtime-sha "$GITHUB_SHA" --run-id "$GITHUB_RUN_ID" --repository "$GITHUB_REPOSITORY" --api-url "$GITHUB_API_URL" --output "$RUNNER_TEMP/qwen38-bonsai-aggregate/selected-artifacts.json" --github-output "$GITHUB_OUTPUT"
