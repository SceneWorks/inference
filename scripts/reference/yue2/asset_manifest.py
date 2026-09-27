#!/usr/bin/env python3
"""Regenerate the YuE2 conversion manifests (sc-22989, epic sc-22988).

For every pinned YuE2 component this reads the **pinned original** snapshot from a local Hugging
Face hub cache (offline — nothing is downloaded) and writes
``crates/audio/candle-audio-yue2/manifests/<component>.json``:

* ``source`` — repo, revision, and ``bytes`` / ``sha256`` of every closure file, streamed from the
  snapshot. For the YuE2 repositories the weights are additionally checked by **upstream's own**
  ``yue2.storage.model_identity(path, verify=True)`` against the repository's
  ``weights_manifest.json``; MERT-v2-FullSong's differently-shaped manifest is checked here.
* ``conversion`` — every YuE2 component is loaded natively *as published*: all weights are already
  safetensors in BF16/F32, which Candle reads directly, so the conversion is the identity and the
  native file is byte-identical to the original (``native.sha256 == source`` weights hash). Weight
  norm (``weight_g``/``weight_v`` in the VAEs) is left as published; folding it is load-time model
  code, not an asset conversion.
* ``tensors`` — one row per tensor, in file order: name, safetensors dtype, shape, and the SHA-256
  of the tensor's little-endian bytes **as PyTorch loads them** (``safetensors.safe_open(...,
  framework="pt")``). The Rust real-weight test loads every tensor through Candle's safetensors path
  and must reproduce each row exactly.
* ``tokenizer`` (qwen.tiktoken only) — the ordinary-token count upstream's
  ``YuE2TextTokenizer`` asserts (151643).

Run with the pinned reference environment (``setup_reference_env.sh``)::

    YUE2_HF_HUB=/path/to/huggingface/hub \\
        ~/.cache/sceneworks-yue2-ref/venv/bin/python scripts/reference/yue2/asset_manifest.py

Peak RSS is bounded by the largest single tensor (the 184704x2048 BF16 ``lm_head``, ~0.76 GB)
twice over plus the mmapped file pages the OS keeps resident; run it under an RSS guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "crates/audio/candle-audio-yue2/manifests"
YUE2_COMMIT = "92a73cc7652fcc1f937855e4b765e0a0edd7ff2e"

YUE2_3B = ("m-a-p/YuE2-3B", "1a96eca688d6ae5d7f0feb88573fec89920fcd19")
VAE = ("m-a-p/YuE2-Vae", "95535e72a97bc0f09b8ada125d26b4009428c0e8")
VAE_LEGACY = ("m-a-p/YuE2-Vae-legacy", "b54118f0fc462f08999d1ec07e88817f4ee3f770")
SHEETSAGE2 = ("m-a-p/SheetSage2", "eab522a8168e8b8b8c4856bf8609cd86198f01fe")
MERT = ("m-a-p/MERT-v2-FullSong", "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42")

YUE2_LICENSE_FILES = [
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "licenses/SnakeBeta-NVIDIA-MIT.txt",
    "licenses/stable-audio-tools-MIT.txt",
]

# component key -> (repo, revision, closure files, weights file or None, upstream manifest style)
COMPONENTS = {
    "yue2_3b": (
        *YUE2_3B,
        [
            "config.json",
            "generation_config.json",
            "yue2_generation_config.json",
            "weights_manifest.json",
            "model.safetensors",
            "README.md",
            *YUE2_LICENSE_FILES,
        ],
        "model.safetensors",
        "yue2",
    ),
    "yue2_qwen_tiktoken": (*YUE2_3B, ["qwen.tiktoken"], None, None),
    "yue2_vae": (
        *VAE,
        ["config.json", "weights_manifest.json", "model.safetensors", "README.md", *YUE2_LICENSE_FILES],
        "model.safetensors",
        "yue2",
    ),
    "yue2_vae_legacy": (
        *VAE_LEGACY,
        ["config.json", "weights_manifest.json", "model.safetensors", "README.md", *YUE2_LICENSE_FILES],
        "model.safetensors",
        "yue2",
    ),
    "yue2_sheetsage2": (
        *SHEETSAGE2,
        [
            "config.json",
            "processor_config.json",
            "model.safetensors",
            "README.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
        ],
        "model.safetensors",
        None,  # the repository publishes no weights manifest
    ),
    "yue2_mert_v2_fullsong": (
        *MERT,
        [
            "config.json",
            "preprocessor_config.json",
            "weights_manifest.json",
            "model.safetensors",
            "README.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
        ],
        "model.safetensors",
        "mert",
    ),
}

TORCH_TO_SAFETENSORS = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_dir(hub: Path, repo: str, revision: str) -> Path:
    path = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if not path.is_dir():
        sys.exit(f"offline cache miss: {repo}@{revision} is not in {hub} (expected {path})")
    return path


def check_upstream_manifest(style: str | None, snapshot: Path, files: dict) -> str:
    if style == "yue2":
        # Upstream's own integrity routine, from the pinned package.
        from yue2.storage import model_identity

        identity = model_identity(snapshot, verify=True)
        for name, entry in identity["files"].items():
            assert files[name] == entry, (name, files[name], entry)
        return "yue2.storage.model_identity(verify=True) against weights_manifest.json"
    if style == "mert":
        manifest = json.loads((snapshot / "weights_manifest.json").read_text(encoding="utf-8"))
        weights = files[manifest["filename"]]
        assert weights == {"bytes": manifest["bytes"], "sha256": manifest["sha256"]}, manifest
        return "weights_manifest.json filename/bytes/sha256"
    return "none published upstream (pinned revision + sha256 only)"


def tensor_rows(path: Path) -> list[dict]:
    rows = []
    with safe_open(str(path), framework="pt") as f:
        for name in f.keys():
            dtype = f.get_slice(name).get_dtype()
            tensor = f.get_tensor(name).contiguous()
            assert TORCH_TO_SAFETENSORS[tensor.dtype] == dtype, (name, tensor.dtype, dtype)
            data = tensor.view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
            rows.append(
                {
                    "name": name,
                    "dtype": dtype,
                    "shape": list(tensor.shape),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            del tensor, data
    return rows


def header_order(path: Path) -> list[str]:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return sorted(header, key=lambda k: header[k]["data_offsets"][0])


def render(manifest: dict) -> str:
    """Indented JSON with one compact line per tensor row, so a diff stays one line per tensor."""
    tensors = manifest.pop("tensors", None)
    text = json.dumps(manifest, indent=1)
    if tensors is not None:
        rows = ",\n".join("  " + json.dumps(row, separators=(", ", ": ")) for row in tensors)
        text = text[: text.rindex("}")].rstrip() + ',\n "tensors": [\n' + rows + "\n ]\n}"
    return text + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hub", type=Path, default=os.environ.get("YUE2_HF_HUB"))
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    if args.hub is None:
        sys.exit("set YUE2_HF_HUB or pass --hub")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import safetensors

    for key, (repo, revision, closure, weights, style) in COMPONENTS.items():
        if args.only and key not in args.only:
            continue
        snapshot = snapshot_dir(args.hub, repo, revision)
        files = {}
        for name in closure:
            p = snapshot / name
            if not p.is_file():
                sys.exit(f"{key}: missing {name} in {snapshot}")
            files[name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        manifest = {
            "schema": 1,
            "component": key,
            "source": {
                "repo": repo,
                "revision": revision,
                "upstream_integrity": check_upstream_manifest(style, snapshot, files)
                if weights
                else "none published upstream (pinned revision + sha256 only)",
                "files": files,
            },
            "reference": {
                "script": "scripts/reference/yue2/asset_manifest.py",
                "yue2_commit": YUE2_COMMIT,
                "torch": torch.__version__,
                "safetensors": safetensors.__version__,
            },
        }
        if weights:
            rows = tensor_rows(snapshot / weights)
            # Rows are emitted in data-offset order, the order the file stores them.
            order = {name: i for i, name in enumerate(header_order(snapshot / weights))}
            rows.sort(key=lambda r: order[r["name"]])
            manifest["conversion"] = {
                "kind": "identity",
                "note": "published safetensors (BF16/F32) are loaded natively as-is; the native "
                "file is the pinned original, byte for byte",
            }
            manifest["native"] = {"file": weights, **files[weights]}
            manifest["tensors"] = rows
        else:
            text = (snapshot / "qwen.tiktoken").read_bytes()
            ranks = [line for line in text.splitlines() if line]
            from yue2.tokenization_yue2 import YuE2TextTokenizer

            YuE2TextTokenizer(snapshot / "qwen.tiktoken")  # upstream's own 151643-rank assertion
            manifest["conversion"] = {
                "kind": "identity",
                "note": "tiktoken BPE ranks file read natively as-is",
            }
            manifest["native"] = {"file": "qwen.tiktoken", **files["qwen.tiktoken"]}
            manifest["tokenizer"] = {"format": "tiktoken-bpe-ranks", "ordinary_tokens": len(ranks)}
        out = OUT_DIR / f"{key}.json"
        count = len(manifest.get("tensors", []))
        out.write_text(render(manifest), encoding="utf-8")
        print(f"{key}: {count} tensors -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
