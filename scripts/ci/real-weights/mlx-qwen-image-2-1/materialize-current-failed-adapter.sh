set -euo pipefail

if [[ "${QWEN_IMAGE_2_1_LORA_PHASE:-full}" != current-diagnostic ]]; then
  echo "current failed adapter materializer is restricted to current-diagnostic" >&2
  exit 1
fi
if [[ "${GITHUB_REPOSITORY:-}" != SceneWorks/inference || "${GITHUB_JOB:-}" != mlx-qwen-image-2-1 ]]; then
  echo "current diagnostic requires the exact inference workflow job context" >&2
  exit 1
fi
if [[ ! "${GITHUB_RUN_ID:-}" =~ ^[1-9][0-9]*$ || ! "${GITHUB_RUN_ATTEMPT:-}" =~ ^[1-9][0-9]*$ ||
      ! "${GITHUB_SHA:-}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "current diagnostic requires exact live run/attempt/source context" >&2
  exit 1
fi

out="$QWEN_IMAGE_2_1_RENDER_OUT"
api="$out/current-diagnostic/api"
mkdir -p "$api"
python3.12 -m scripts.ci.qwen21_current_failed_adapter init --output "$out"
refuse() {
  rm -f "$api/source-artifact.zip"
  python3.12 -m scripts.ci.qwen21_current_failed_adapter refuse \
    --output "$out" --reason github_api_read_failed || true
}
trap refuse ERR

# Historical failed input. Artifact metadata cannot bind a producing job, so the exact terminal
# job and its immutable upload receipt log are fetched and checked separately.
gh api repos/SceneWorks/inference/actions/runs/37392084691 > "$api/source-run.json"
gh api repos/SceneWorks/inference/actions/runs/37392084691/attempts/1 > "$api/source-attempt.json"
gh api 'repos/SceneWorks/inference/actions/runs/37392084691/attempts/1/jobs?per_page=100' > "$api/source-jobs.json"
gh api repos/SceneWorks/inference/actions/jobs/112039296411/logs > "$api/source-job.log"
gh api repos/SceneWorks/inference/actions/artifacts/11383988900 > "$api/source-artifact.json"
gh api repos/SceneWorks/inference/actions/artifacts/11383988900/zip > "$api/source-artifact.zip"

# Future execution identity. This intentionally does not reuse the historical failed run/job.
gh api "repos/SceneWorks/inference/actions/runs/$GITHUB_RUN_ID" > "$api/live-run.json"
gh api "repos/SceneWorks/inference/actions/runs/$GITHUB_RUN_ID/attempts/$GITHUB_RUN_ATTEMPT/jobs?per_page=100" > "$api/live-jobs.json"
trap - ERR
refuse() {
  rm -f "$api/source-artifact.zip"
  python3.12 -m scripts.ci.qwen21_current_failed_adapter refuse \
    --output "$out" --reason materialization_refused || true
}
trap refuse ERR
python3.12 -m scripts.ci.qwen21_current_failed_adapter capture-hardware --output "$api/hardware.json"

manifest="$(python3.12 -m scripts.ci.qwen21_current_failed_adapter materialize \
  --source-run "$api/source-run.json" \
  --source-attempt "$api/source-attempt.json" \
  --source-jobs "$api/source-jobs.json" \
  --source-job-log "$api/source-job.log" \
  --artifact "$api/source-artifact.json" \
  --artifact-zip "$api/source-artifact.zip" \
  --live-run "$api/live-run.json" \
  --live-jobs "$api/live-jobs.json" \
  --hardware "$api/hardware.json" \
  --build-identity "$out/mlx-lib-test-build-identity.json" \
  --output "$out" \
  --repository-root "$GITHUB_WORKSPACE" \
  --snapshots "$MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT" "$MLX_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT")"
trap - ERR
rm -f "$api/source-artifact.zip"
echo "QWEN_IMAGE_2_1_CURRENT_VELOCITY_MANIFEST=$manifest" >> "$GITHUB_ENV"
