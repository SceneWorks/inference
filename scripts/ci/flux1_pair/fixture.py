"""Synthetic FLUX.1 fixtures with production widths and bounded materialization."""
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
ENG4B = "eng4b-tier1"
ENG4B_REAL = "eng4b-tier2"
ENG4B_TIERS = (ENG4B, ENG4B_REAL)
Q4 = {"bits": 4, "group_size": 64}


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


def weight_blocks(key, shape, dtype):
    # Stable nonzero signed weights, scaled by fan-in. Normalization scales are
    # centered on one. Generate bounded chunks without holding a second tensor.
    ident = hashlib.sha256(f"{SEED}/{key}".encode()).digest()
    rng = random.Random(int.from_bytes(ident, "little"))
    center = 1 if len(shape) == 1 and key.endswith("weight") and "norm" in key.lower() else 0
    scale = 1 / math.sqrt(math.prod(shape[1:])) if len(shape) > 1 else 0.01
    values = [center + scale * (2 * (n + 0.5) / 256 - 1) for n in range(256)]
    if dtype == "BF16":
        palette = [struct.pack("<f", value)[2:] for value in values]
    else:
        fmt = {"F16": "<e", "F32": "<f"}[dtype]
        palette = [struct.pack(fmt, value) for value in values]
    width = len(palette[0])
    planes = [bytes(p[i] for p in palette) for i in range(width)]
    remaining = math.prod(shape)
    while remaining:
        count = min(remaining, 2**18)
        indices = rng.randbytes(count)
        out = bytearray(count * width)
        for i, plane in enumerate(planes):
            out[i::width] = indices.translate(plane)
        yield out
        remaining -= count


def bf16_blocks(key, shape):
    return weight_blocks(key, shape, "BF16")


def tensor_layout(tensors, *, q4=False):
    """Yield the exact safetensors header and bounded payload recipe for each tensor."""
    for key, shape in tensors.items():
        if q4 and len(shape) == 2 and key.endswith(".weight") and shape[1] % Q4["group_size"] == 0:
            rows, columns = shape
            base = key.removesuffix(".weight")
            group_shape = [rows, columns // Q4["group_size"]]
            # Eight affine four-bit codes per U32. Preserve the published group size and store
            # a finite signed range centered at zero, with the same fan-in as the dense fixture.
            step = 2 / (15 * math.sqrt(columns))
            yield key, "U32", [rows, columns // 8], None
            yield base + ".scales", "F32", group_shape, step
            yield base + ".biases", "F32", group_shape, -7.5 * step
        else:
            yield key, "BF16", shape, None


def payload_blocks(key, dtype, shape, fill):
    if dtype == "BF16":
        yield from bf16_blocks(key, shape)
        return
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{SEED}/q4/{key}".encode()).digest(), "little"))
    fill_bytes = struct.pack("<f", fill) if dtype == "F32" else None
    remaining = math.prod(shape)
    while remaining:
        count = min(remaining, 2**18)
        # Uniform bytes contain two independent four-bit codes. Writing bytes directly also
        # preserves their order inside each little-endian U32 without allocating dense weights.
        yield fill_bytes * count if fill_bytes is not None else rng.randbytes(4 * count)
        remaining -= count


def tensor_header(entries):
    header = {}
    offset = 0
    for key, dtype, shape, _ in entries:
        end = offset + (2 if dtype == "BF16" else 4) * math.prod(shape)
        header[key] = dict(dtype=dtype, shape=shape, data_offsets=[offset, end])
        offset = end
    return header, offset


def write_component(path, tensors, *, q4=False):
    entries = list(tensor_layout(tensors, q4=q4))
    header, _ = tensor_header(entries)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("xb") as output:
        output.write(struct.pack("<Q", len(raw)))
        output.write(raw)
        for key, dtype, shape, fill in entries:
            for block in payload_blocks(key, dtype, shape, fill):
                output.write(block)


def make_snapshot(root, variant, *, q4=False):
    for component, tensors in geometry(variant).items():
        directory = root / component
        directory.mkdir(parents=True)
        packed = q4 and component != "vae"
        write_component(directory / "model.safetensors", tensors, q4=packed)
        config = ({"num_layers": 1, "num_single_layers": 1,
                   "guidance_embeds": variant == "dev"} if component == "transformer" else {})
        if packed:
            config["quantization"] = Q4
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    directory = root / "tokenizer_2"
    directory.mkdir()
    tokenizer = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [],
                 "normalizer": None, "pre_tokenizer": {"type": "Whitespace"},
                 "post_processor": None, "decoder": None,
                 "model": {"type": "WordLevel", "unk_token": "<unk>",
                           "vocab": {"<pad>": 0, "</s>": 1, "<unk>": 2, "red": 3, "kite": 4}}}
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")


def real_snapshot_identity(variant, tier):
    provenance = json.loads((HERE / "headers.json").read_text(encoding="utf-8"))[variant]
    return {
        "tier": tier, "variant": variant, "depths": [19, 38],
        "source_snapshot": {"repo": provenance["repo"], "snapshot": provenance["snapshot"], "tier": "q4"},
        "files": {name: row["sha256"] for name, row in provenance["q4_files"].items()},
    }


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
    if tier in ("tier2", ENG4B_REAL):
        identity = real_snapshot_identity(variant, tier)
        if {p.relative_to(root).as_posix() for p in files} != identity["files"].keys():
            raise ValueError("installed Q4 snapshot differs from pinned file names")
    digests = {p.relative_to(root).as_posix(): sha(p) for p in files}
    dt_config = json.loads((root / "transformer/config.json").read_text(encoding="utf-8"))
    depths = [dt_config["num_layers"], dt_config["num_single_layers"]]
    if depths != ([1, 1] if tier in ("tier1", ENG4B) else [19, 38]):
        raise ValueError("wrong DiT depth for proof tier")
    if tier in ("tier2", ENG4B_REAL):
        if dt_config.get("quantization", {}).get("bits") != 4 or digests != identity["files"]:
            raise ValueError("installed Q4 snapshot differs from pinned files")
        return identity
    if tier == ENG4B:
        validate_q4_snapshot(root, variant)
    provenance = json.loads((HERE / "headers.json").read_text(encoding="utf-8"))[variant]
    provenance = {k: v for k, v in provenance.items() if k != "q4_files"}
    identity = {"tier": tier, "variant": variant, "source_snapshot": provenance,
                "depths": depths, "files": digests}
    if tier == ENG4B:
        identity["fixture_storage"] = dict(Q4=Q4, dense_vae="BF16")
    return identity


def validate_q4_snapshot(root, variant):
    """Bind all component shapes, codes and affine sidecars without reading tensor payloads."""
    for component, tensors in geometry(variant).items():
        directory = root / component
        packed = component != "vae"
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("quantization") != (Q4 if packed else None):
            raise ValueError("wrong quantization metadata: " + component)
        files = sorted(directory.glob("*.safetensors"))
        if files != [directory / "model.safetensors"]:
            raise ValueError("wrong synthetic component files: " + component)
        with files[0].open("rb") as source:
            length, = struct.unpack("<Q", source.read(8))
            if not 0 < length < 16 * 1024**2:
                raise ValueError("invalid synthetic safetensors header length")
            recorded = json.loads(source.read(length))
        expected, payload_size = tensor_header(tensor_layout(tensors, q4=packed))
        if recorded != expected or files[0].stat().st_size != 8 + length + payload_size:
            raise ValueError("wrong synthetic tensor layout: " + component)


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
