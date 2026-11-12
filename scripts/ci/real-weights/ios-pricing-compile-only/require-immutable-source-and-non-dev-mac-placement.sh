[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] && [[ "$SOURCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$SOURCE_SHA" == "$GITHUB_SHA" ]]
[[ "$COMPILE_HOST" == second-mac ]]
[[ "$COMPILE_PHASE" == compile-only || "$COMPILE_PHASE" == compile-host || "$COMPILE_PHASE" == prerequisite-probe ]]
[[ "$RUNNER_NAME" == nax-macos-2 ]]
command -v python3.12 >/dev/null || { echo "::error::Preinstalled Python3.12 required; no provisioning"; exit 1; }
