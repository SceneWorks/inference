"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
rem The three `-base` entries carry `download_files` allow-lists, so the 6.4 GB of
rem `svd_bases.pt` training pickles across them are not fetched.
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model stable-audio-3-small-music --snapshot "%SA3_SMALL_MUSIC_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model stable-audio-3-small-sfx --snapshot "%SA3_SMALL_SFX_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model stable-audio-3-medium --snapshot "%SA3_MEDIUM_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model stable-audio-3-small-music-base --snapshot "%SA3_SMALL_MUSIC_BASE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model stable-audio-3-small-sfx-base --snapshot "%SA3_SMALL_SFX_BASE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model stable-audio-3-medium-base --snapshot "%SA3_MEDIUM_BASE_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model same-l --snapshot "%SA3_SAME_L_SNAPSHOT%" || exit /b 1
