"""Stage the MiniMax-H3 inputs the sc-25777 Veda A/B reads as real files; dispatch only.

The runner's `CANDLE_MINIMAX_H3_*` variables name Hugging Face cache snapshots, whose files are
reparse-point links into `blobs/`, so `validate_h3_vram_staging.py` refuses them. This projects the
needed subset into a sibling `real-file-staging/` tree on the SAME volume as the cache: each file is
a hardlink to its blob (no bytes copied, nothing in the cache touched), falling back to a copy only
when the volume refuses a hardlink, after a free-space check. The tree is kept across runs, so a
later dispatch only re-checks it. The staged paths are exported through `GITHUB_ENV` so every later
step (the validator, the driver, the renders) reads the staging rather than the cache.

Base root: every top-level file plus `vae/`, `audio_vae/`, `FL2VA/`, `tokenizer/` (and the dense
`transformer/` + `text_encoder/` only when a cell asks for bf16). Tier (q4/q8): the whole tier
directory, i.e. `transformer/`, `transformer_ref/` (what r2va denoises on) and `text_encoder/`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

BASE_DIRS = ("vae", "audio_vae", "FL2VA", "tokenizer")
COPY_MARGIN = 20 * 1024**3


def staged_location(snapshot: Path) -> Path:
    """`<cache>/models--X/snapshots/<rev>[/sub]` -> `<cache>/../real-file-staging/models--X/<rev>[/sub]`."""
    parts = snapshot.parts
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] == "snapshots" and parts[i - 1].startswith("models--"):
            cache = Path(*parts[: i - 1])
            return cache.parent / "real-file-staging" / parts[i - 1] / Path(*parts[i + 1 :])
    raise ValueError(f"not a Hugging Face cache snapshot path: {snapshot}")


def is_reparse(path: Path) -> bool:
    info = os.lstat(path)
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def plan(src: Path, dst: Path, files: list[Path]) -> list[tuple[Path, Path]]:
    pairs = []
    for f in files:
        real = Path(os.path.realpath(f))
        if not real.is_file():
            raise RuntimeError(f"cache entry does not resolve to a file: {f}")
        target = dst / f.relative_to(src)
        if target.is_file() and not is_reparse(target) and (
            os.path.samefile(target, real) or target.stat().st_size == real.stat().st_size
        ):
            continue
        pairs.append((real, target))
    return pairs


def materialize(pairs: list[tuple[Path, Path]]) -> tuple[int, int]:
    linked = copied = 0
    for real, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(real, target)
            linked += 1
        except OSError:
            need = real.stat().st_size + COPY_MARGIN
            free = shutil.disk_usage(target.parent).free
            if free < need:
                raise RuntimeError(
                    f"hardlink refused and only {free / 1e9:.1f} GB free for a "
                    f"{real.stat().st_size / 1e9:.1f} GB copy of {real} (+20 GB margin)")
            tmp = target.with_name(target.name + ".partial")
            shutil.copyfile(real, tmp)
            os.replace(tmp, target)
            copied += 1
    return linked, copied


def stage_tree(src: Path, dst: Path, files: list[Path]) -> None:
    pairs = plan(src, dst, files)
    linked, copied = materialize(pairs)
    print(f"staged {src} -> {dst}: {len(files)} files, {linked} hardlinked, {copied} copied, "
          f"{len(files) - len(pairs)} already staged", flush=True)


def walk(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise RuntimeError(f"missing cache directory: {directory}")
    return sorted(p for p in directory.rglob("*") if p.is_file())


def main() -> int:
    tiers = {str(c.get("VEDA_AB_TIER", "q4")) for c in json.loads(os.environ["VEDA_AB_CELLS"])}
    env_out = []

    base_src = Path(os.environ["MINIMAX_H3_SNAPSHOT"])
    base_dst = staged_location(base_src)
    dirs = list(BASE_DIRS) + (["transformer", "text_encoder"] if "bf16" in tiers else [])
    files = [p for p in base_src.iterdir() if p.is_file()]
    for d in dirs:
        files += walk(base_src / d)
    stage_tree(base_src, base_dst, files)
    env_out.append(f"MINIMAX_H3_SNAPSHOT={base_dst}")

    for tier in sorted(tiers - {"bf16"}):
        dit = Path(os.environ[f"H3_{tier.upper()}_DIT_DIR"])
        te = Path(os.environ[f"H3_{tier.upper()}_TE_DIR"])
        tier_src = dit.parent
        if te.parent != tier_src:
            raise RuntimeError(f"{tier}: DiT and TE dirs are not siblings: {dit} {te}")
        # The tier row (`minimax-h3-mlx-<tier>`) carries `transformer_ref/`, which no earlier lane
        # fetched; heal it into the cache first (verify-only when it is already there).
        subprocess.run([sys.executable, "scripts/release/ensure_model_snapshot.py", "--model",
                        f"minimax-h3-mlx-{tier}", "--snapshot", str(tier_src.parent)], check=True)
        tier_dst = staged_location(tier_src)
        stage_tree(tier_src, tier_dst, walk(tier_src))
        env_out.append(f"H3_{tier.upper()}_DIT_DIR={tier_dst / dit.name}")
        env_out.append(f"H3_{tier.upper()}_TE_DIR={tier_dst / te.name}")

    # The unchanged `validate_h3_vram_staging.py` gate, against the staged paths, before any CUDA
    # context: only for the tiers the cells use.
    staged = dict(line.split("=", 1) for line in env_out)
    root = staged["MINIMAX_H3_SNAPSHOT"]
    for tier in sorted(tiers):
        if tier == "bf16":
            dit, te = str(Path(root) / "transformer"), str(Path(root) / "text_encoder")
        else:
            dit, te = staged[f"H3_{tier.upper()}_DIT_DIR"], staged[f"H3_{tier.upper()}_TE_DIR"]
        subprocess.run([sys.executable, "scripts/ci/validate_h3_vram_staging.py", "--root", root,
                        "--tier", tier, "--dit", dit, "--te", te], check=True)
        print(f"validated {tier} staging", flush=True)

    with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as stream:
        stream.write("".join(line + "\n" for line in env_out))
    print("\n".join(env_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
