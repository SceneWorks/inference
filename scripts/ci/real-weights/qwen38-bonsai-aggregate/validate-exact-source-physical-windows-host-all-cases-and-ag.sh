"$REVIEWED_PYTHON" scripts/release/qwen38_bonsai_artifacts.py aggregate --downloads "$RUNNER_TEMP/qwen38-bonsai-aggregate/downloads" --selection "$RUNNER_TEMP/qwen38-bonsai-aggregate/selected-artifacts.json" --output "$RUNNER_TEMP/qwen38-bonsai-aggregate/reports" --runtime-sha "$GITHUB_SHA" --run-id "$GITHUB_RUN_ID"
cat "$RUNNER_TEMP/qwen38-bonsai-aggregate/reports/matrix-report.md"
