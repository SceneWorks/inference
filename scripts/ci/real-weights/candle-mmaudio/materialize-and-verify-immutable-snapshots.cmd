"%REVIEWED_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "%RUNNER_TEMP%\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py312.txt || exit /b 1
set "PYTHONPATH=%RUNNER_TEMP%\huggingface-hub"
rem hkchengrex/MMAudio is a 54 GB repo (~46 GB of training checkpoints + weight variants the
rem inference stack never loads); each catalog key carries a `download_files` allow-list so
rem only the pinned checkpoint is fetched (~8 GB across the six components into one dir).
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model synchformer --snapshot "%MMAUDIO_MMAUDIO_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model mmaudio-small-16k --snapshot "%MMAUDIO_MMAUDIO_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model mmaudio-large-44k-v2 --snapshot "%MMAUDIO_MMAUDIO_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model mmaudio-vae-16k --snapshot "%MMAUDIO_MMAUDIO_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model mmaudio-vae-44k --snapshot "%MMAUDIO_MMAUDIO_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model mmaudio-bigvgan-16k --snapshot "%MMAUDIO_MMAUDIO_SNAPSHOT%" || exit /b 1
rem The stable catalog key still ends in 384 because MMAudio feeds 384px; its repository is
rem the canonical -378 checkpoint declared in release/real-weight-models.toml.
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model dfn5b-clip-vit-h14-384 --snapshot "%MMAUDIO_CLIP_SNAPSHOT%" || exit /b 1
"%REVIEWED_PYTHON%" scripts/release/ensure_model_snapshot.py --model nvidia-bigvgan-v2-44khz-128band-512x --snapshot "%MMAUDIO_BIGVGAN_V2_SNAPSHOT%" || exit /b 1
