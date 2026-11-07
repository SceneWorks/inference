"""Shape-faithful FLUX.1 fixtures. Materialization generates BF16 data on CPU."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct

HERE = Path(__file__).resolve().parent
COMPONENTS = ("transformer", "text_encoder", "text_encoder_2", "vae")
SEED = 104729


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry(variant):
    with (HERE / "geometry.tsv").open(encoding="utf-8") as source:
        rows = csv.DictReader(source, delimiter="\t")
        data = {component: {} for component in COMPONENTS}
        for row in rows:
            if variant in row["variants"].split(","):
                data[row["component"]][row["key"]] = list(map(int, row["shape"].split(",")))
    dit = data["transformer"]
    # Source header geometry after depth reduction. All widths remain production widths.
    if (dit["x_embedder.weight"] != [3072, 64]
            or data["text_encoder_2"]["shared.weight"] != [32128, 4096]
            or data["text_encoder"]["text_model.embeddings.token_embedding.weight"] != [49408, 768]
            or len(dit) != (66 if variant == "dev" else 62)
            or len(data["vae"]) != 244
            or any(k.split(".")[1] != "0" for k in dit
                   if k.startswith(("transformer_blocks.", "single_transformer_blocks.")))
            or any("guidance_embedder" in k for k in dit) != (variant == "dev")):
        raise ValueError("unexpected FLUX.1 fixture geometry")
    return data


def bf16_blocks(key, shape):
    # Stable nonzero signed weights, scaled by fan-in. Normalization scales are
    # centered on one. Generate bounded chunks without holding a second tensor.
    ident = hashlib.sha256(f"{SEED}/{key}".encode()).digest()
    rng = random.Random(int.from_bytes(ident, "little"))
    center = 1 if len(shape) == 1 and key.endswith("weight") and "norm" in key.lower() else 0
    scale = 1 / math.sqrt(math.prod(shape[1:])) if len(shape) > 1 else 0.01
    palette = [struct.pack("<f", center + scale * (2 * (n + 0.5) / 256 - 1))[2:] for n in range(256)]
    low, high = bytes(p[0] for p in palette), bytes(p[1] for p in palette)
    remaining = math.prod(shape)
    while remaining:
        count = min(remaining, 2**18)
        indices = rng.randbytes(count)
        out = bytearray(count * 2)
        out[::2], out[1::2] = indices.translate(low), indices.translate(high)
        yield out
        remaining -= count


def write_component(path, tensors):
    header = {}
    offset = 0
    for key, shape in tensors.items():
        end = offset + 2 * math.prod(shape)
        header[key] = dict(dtype="BF16", shape=shape, data_offsets=[offset, end])
        offset = end
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("xb") as output:
        output.write(struct.pack("<Q", len(raw)))
        output.write(raw)
        for key, shape in tensors.items():
            for block in bf16_blocks(key, shape):
                output.write(block)


def make_snapshot(root, variant):
    for component, tensors in geometry(variant).items():
        directory = root / component
        directory.mkdir(parents=True)
        write_component(directory / "model.safetensors", tensors)
        config = {"num_layers": 1, "num_single_layers": 1} if component == "transformer" else {}
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    directory = root / "tokenizer_2"
    directory.mkdir()
    tokenizer = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
                 "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
                 "post_processor": None, "decoder": None,
                 "model": {"type": "WordLevel", "unk_token": "<unk>",
                           "vocab": {"<pad>": 0, "</s>": 1, "<unk>": 2, "red": 3, "kite": 4}}}
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")


def snapshot_identity(root, variant, tier):
    """Hash the exact component files, tokenizer and configuration consumed by both arms."""
    files = []
    for component in COMPONENTS:
        weights = sorted((root / component).glob("*.safetensors"))
        if not weights:
            raise ValueError(f"missing weights for {component}")
        files.extend(weights)
        config = root / component / "config.json"
        if config.exists():
            files.append(config)
    files.append(root / "tokenizer_2" / "tokenizer.json")
    digests = {p.relative_to(root).as_posix(): sha(p) for p in files}
    dt_config = json.loads((root / "transformer/config.json").read_text(encoding="utf-8"))
    depths = [dt_config["num_layers"], dt_config["num_single_layers"]]
    if depths != ([1, 1] if tier == "tier1" else [19, 38]):
        raise ValueError("wrong DiT depth for proof tier")
    if tier == "tier2" and dt_config.get("quantization", {}).get("bits") != 4:
        raise ValueError("real snapshot must be the four-bit published tier")
    provenance = json.loads((HERE / "headers.json").read_text(encoding="utf-8"))[variant]
    if tier == "tier2":
        expected = {name: row["sha256"] for name, row in provenance["q4_files"].items()}
        if digests != expected:
            raise ValueError("installed Q4 snapshot differs from pinned files")
        provenance = {"repo": provenance["repo"], "snapshot": provenance["snapshot"], "tier": "q4"}
    else:
        provenance = {k: v for k, v in provenance.items() if k != "q4_files"}
    return {"tier": tier, "variant": variant, "source_snapshot": provenance,
            "depths": depths, "files": digests}


def verify_source_headers(hub):
    """Re-derive every retained row from upstream headers without reading a tensor payload."""
    sources = json.loads((HERE / "headers.json").read_text(encoding="utf-8"))
    for variant, spec in sources.items():
        root = hub / ("models--" + spec["repo"].replace("/", "--")) / "snapshots" / spec["snapshot"] / "bf16"
        observed = {c: {} for c in COMPONENTS}
        for relative, expected in spec["headers"].items():
            with (root / relative).open("rb") as f:
                length, = struct.unpack("<Q", f.read(8))
                if not 0 < length < 16 * 1024**2:
                    raise ValueError("invalid safetensors header length")
                header = f.read(length)
            if hashlib.sha256(header).hexdigest() != expected:
                raise ValueError("source header changed: " + relative)
            component = relative.split("/", 1)[0]
            for key, tensor in json.loads(header).items():
                if key == "__metadata__" or re.search(r"(?:transformer_blocks|encoder\.layers|encoder\.block)\.[1-9]\d*\.", key):
                    continue
                if tensor["dtype"] != "BF16" or key in observed[component]:
                    raise ValueError("unexpected dtype or duplicate tensor in source headers")
                observed[component][key] = tensor["shape"]
        if observed != geometry(variant):
            raise ValueError("source headers disagree with the frozen geometry: " + variant)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recheck the frozen model geometry using only local snapshot headers.")
    parser.add_argument("hub", type=Path)
    verify_source_headers(parser.parse_args().hub)
    print("All retained dev/schnell tensors match their pinned BF16 headers.")
