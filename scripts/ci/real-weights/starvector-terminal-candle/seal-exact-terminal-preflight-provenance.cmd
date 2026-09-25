node scripts/release/starvector_terminal_preflight.mjs assemble --root "%RUNNER_TEMP%\\starvector-terminal-preflight" --head-sha "%GITHUB_SHA%" --workflow-run-id "%GITHUB_RUN_ID%" --workflow-run-attempt "%GITHUB_RUN_ATTEMPT%" || exit /b 1
type "%RUNNER_TEMP%\\starvector-terminal-preflight\\starvector-terminal-preflight.json"
