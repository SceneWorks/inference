"""CPU fixture from the actual separately pinned Comfy tiled_decode methods."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import torch
from safetensors.torch import save_file

PIN = "7a5dad695fe1cae25efcb2550530fb20ef68da3d"
SOURCE_SHA = "8a889ea28fc24b13edfbdea9e85b8a0762b2e7e079bbe34db026c6e6d4f5a556"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.vae_source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise ValueError("unpinned VAE reference source")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3VideoVAE")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"split_tiles", "blend", "tiled_decode"}]
    reference = ast.ClassDef(name="PinnedDecode", bases=[], keywords=[], body=methods, decorator_list=[])
    scope = {"torch": torch, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[reference], type_ignores=[])), str(args.vae_source), "exec"), scope)
    values = {}
    for i in range(2):
        for j in range(3):
            values[f"tile.{i}.{j}"] = (torch.arange(16.).sin() + i * 7 + j * 3).reshape(1, 1, 1, 4, 4)
    class Fixture(scope["PinnedDecode"]):
        tile_size = 4
        tile_overlap_min = 2
        vae_ratio = 1
        def _decode_tile_row(self, z_row, x_idx, x_len):
            i = int(z_row.flatten()[0])
            for j in range(len(x_idx)):
                yield values[f"tile.{i}.{j}"]
    z = torch.arange(5.).reshape(1, 1, 1, 5, 1).expand(1, 1, 1, 5, 8)
    values["output"] = Fixture().tiled_decode(z)
    save_file(values, str(args.out), metadata={"comfy_commit": PIN, "source_sha256": SOURCE_SHA})
    args.out.with_suffix(".json").write_text(json.dumps({"comfy_commit": PIN, "source_sha256": SOURCE_SHA, "fixture_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(), "methods": [n.name for n in methods], "purpose": "crossed heavy overlaps distinguish original from already-blended neighbour strips"}, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
