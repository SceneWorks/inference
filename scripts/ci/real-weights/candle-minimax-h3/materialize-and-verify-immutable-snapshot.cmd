"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
rem MiniMaxAI/MiniMax-H3 is a ~196 GB repo; the `minimax-h3` row carries a `download_files`
rem allow-list so only the 11.638 GB the tests below read is fetched.
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3 --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
rem sc-19558 — the DiT component, on its OWN key so the shared row above stays 11.640 GB
rem for every lane that does not read it. 132.6 GB for both partitions on a cold box;
rem sc-18677 provisioned the full upstream snapshot on THIS runner, so here it verifies
rem rather than fetches. It is what makes the two DiT header gates below reachable at all —
rem they read headers only, and were unreachable purely because the bytes were in no row.
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model minimax-h3-dit --snapshot "%MINIMAX_H3_SNAPSHOT%" || exit /b 1
