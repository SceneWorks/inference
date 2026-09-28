if not exist "%RUNNER_TEMP%\\starvector-terminal-mlx\\inventory\\starvector-1b-inventory.json" exit /b 1
if not exist "%RUNNER_TEMP%\\starvector-terminal-mlx\\inventory\\starvector-8b-inventory.json" exit /b 1
if not exist "%RUNNER_TEMP%\\starvector-terminal-mlx\\hooks\\mlx-starvector-1b.log" exit /b 1
if not exist "%RUNNER_TEMP%\\starvector-terminal-mlx\\hooks\\mlx-starvector-8b.log" exit /b 1
if not exist "%RUNNER_TEMP%\\starvector-terminal-preflight\\inventory" mkdir "%RUNNER_TEMP%\\starvector-terminal-preflight\\inventory" || exit /b 1
if not exist "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks" mkdir "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks" || exit /b 1
copy /y "%RUNNER_TEMP%\\starvector-terminal-mlx\\inventory\\starvector-1b-inventory.json" "%RUNNER_TEMP%\\starvector-terminal-preflight\\inventory\\starvector-1b-inventory.json" >nul || exit /b 1
copy /y "%RUNNER_TEMP%\\starvector-terminal-mlx\\inventory\\starvector-8b-inventory.json" "%RUNNER_TEMP%\\starvector-terminal-preflight\\inventory\\starvector-8b-inventory.json" >nul || exit /b 1
copy /y "%RUNNER_TEMP%\\starvector-terminal-mlx\\hooks\\mlx-starvector-1b.log" "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks\\mlx-starvector-1b.log" >nul || exit /b 1
copy /y "%RUNNER_TEMP%\\starvector-terminal-mlx\\hooks\\mlx-starvector-8b.log" "%RUNNER_TEMP%\\starvector-terminal-preflight\\hooks\\mlx-starvector-8b.log" >nul || exit /b 1
node scripts/release/starvector_terminal_evidence.mjs validate-plan --corpus release/starvector-terminal-corpus-v1.json || exit /b 1
