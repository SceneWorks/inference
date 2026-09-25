$log = Join-Path $env:RUNNER_TEMP "scail2-shared-cuda.log"
python -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$env:RUNNER_TEMP\huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-windows-x64-py314.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PYTHONPATH = "$env:RUNNER_TEMP\huggingface-hub"
@'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo = os.environ["SCAIL2_REPOSITORY"]
revision = os.environ["SCAIL2_REVISION"]
root = Path(snapshot_download(
    repo_id=repo,
    repo_type="model",
    revision=revision,
    allow_patterns=["bf16/**"],
    cache_dir=Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub",
    token=False,
    max_workers=4,
)).resolve() / "bf16"
expected = (
    Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub" /
    "models--SceneWorks--scail2-mlx" / "snapshots" / revision / "bf16"
).resolve()
if root != expected:
    raise SystemExit(f"resolved unexpected SCAIL package path: {root} != {expected}")
required = {
    "config.json", "dit.safetensors", "t5_encoder.safetensors", "tokenizer.json",
    "clip.safetensors", "vae.safetensors",
}
actual = {p.name for p in root.iterdir() if p.is_file()}
missing = sorted(required - actual)
if missing:
    raise SystemExit(f"exact SCAIL bf16 package is incomplete: {missing}")
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as out:
    out.write(f"SCAIL2_SHARED_BF16_DIR={root}\n")
print(f"Exact shared package ready: {repo}@{revision}/bf16 -> {root}")
'@ | python - 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
"provision_status=complete" | Add-Content -LiteralPath $log -Encoding utf8
