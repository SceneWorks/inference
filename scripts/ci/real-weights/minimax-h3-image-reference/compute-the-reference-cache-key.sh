# The cache key of the MiniMax-H3-Image diffusers references (reference.safetensors,
# dit-reference.safetensors, edit-reference.safetensors). It hashes, as whole files or rows:
# * this script and the two materialize-and-produce scripts (the pinned snapshot revision);
# * every `crates/media/mlx-gen/...` file those scripts name: the three hash-pinned producers, whose
#   constants pin the diffusers commit, the upstream module SHA-256s, the seeds, the edit prompt and
#   the sizes, and the hash-locked Mage reference environment;
# * every `release/real-weight-models.toml` row they materialize (repository, revision, files);
# * the interpreter that builds the venv and the host OS/arch that runs CPU torch.
# The script fails when it finds fewer producers or rows than it expects, so a reworded call cannot
# silently drop an input. Bump `salt` to force a fresh reference with every input unchanged.
set -o pipefail
salt=1
key="$(python3.12 -I - "$salt" scripts/ci/real-weights/minimax-h3-image-reference <<'PY'
import hashlib, json, platform, re, sys, tomllib
from pathlib import Path

salt, directory = sys.argv[1], Path(sys.argv[2])
scripts = [directory / "compute-the-reference-cache-key.sh",
           directory / "materialize-and-produce-the-image-vae-reference.sh",
           directory / "materialize-the-dit-and-produce-its-reference.sh"]
text = "".join(path.read_text(encoding="utf-8") for path in scripts[1:])
named = sorted(set(re.findall(r"crates/media/mlx-gen/[\w./-]+\.(?:py|txt)", text)))
producers = [name for name in named if name.endswith(".py")]
rows = sorted(set(re.findall(r"--model ([a-z0-9-]+)", text)))
if len(producers) < 3 or not any(name.endswith(".txt") for name in named) or len(rows) < 2:
    sys.exit(f"reference key inputs are incomplete: producers={producers} files={named} rows={rows}")
manifest = {row["key"]: row for row in tomllib.loads(Path("release/real-weight-models.toml").read_text(encoding="utf-8"))["models"]}
inputs = {"salt": salt, "python": sys.version, "host": [platform.system(), platform.machine(), platform.mac_ver()[0]]}
for path in [*map(str, scripts), *named]:
    inputs[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
for row in rows:
    inputs[f"manifest:{row}"] = hashlib.sha256(json.dumps(manifest[row], sort_keys=True).encode()).hexdigest()
for name, value in inputs.items():
    print(f"  {name}: {value}", file=sys.stderr)
print("h3img-ref-v1-" + hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest())
PY
)"
echo "reference cache key: $key"
echo "key=$key" >> "$GITHUB_OUTPUT"
