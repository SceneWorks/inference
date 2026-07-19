#!/usr/bin/env python3
"""Convert the resemble-perth implicit watermarker checkpoint to safetensors.

sc-13240. `candle-audio-chatterbox`'s native PerTh watermarker (`crate::perth`) loads the encoder +
decoder convolution weights of Resemble AI's MIT-licensed `PerthImplicitWatermarker`
(`perth_net_250000.pth.tar`, the "implicit" run) as a flat `perth_implicit.safetensors` file. This
script performs that one-time conversion.

It is deliberately **torch-free**: it parses torch's zip serialization directly (the pickle
`archive/data.pkl` plus the raw little-endian float32 storages under `archive/data/<key>`), so it
runs on a bare Python 3 with no ML dependencies — the same "no deps" posture as the repository's
other gate scripts. Only the `encoder.*` and `decoder.*` conv tensors are emitted; the `ap.*` STFT
window buffers are recomputable constants and are dropped (the Rust port recomputes the Hann window).

Obtain the source checkpoint from the MIT resemble-perth package, e.g.:

    pip download resemble-perth --no-deps --no-binary :all: -d /tmp/perth-src
    # or: git clone https://github.com/resemble-ai/perth /tmp/perth
    python3 scripts/audio/convert_perth_watermarker.py \
        /tmp/perth/src/perth/perth_net/pretrained/implicit/perth_net_250000.pth.tar \
        "$PERTH_SNAPSHOT/perth_implicit.safetensors"

`$PERTH_SNAPSHOT` is the dir the real-weight conformance test (`--ignored`) resolves the weights from.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import struct
import zipfile
from pathlib import Path

# resemble-perth's checkpoint is float32 throughout (verified: file size == n_params * 4).
FLOAT32 = ("F32", 4)


class _Unpickler(pickle.Unpickler):
    """Recover each tensor's storage key + shape/stride without torch.

    torch's zip format pickles an ``OrderedDict`` of ``_rebuild_tensor_v2(storage, offset, size,
    stride, ...)`` calls; ``storage`` is the tuple returned by ``persistent_load``. Every other
    global is stubbed to an inert callable.
    """

    def find_class(self, module: str, name: str):
        if name == "OrderedDict":
            import collections

            return collections.OrderedDict
        if name == "_rebuild_tensor_v2":

            def rebuild(storage, storage_offset, size, stride, *_rest):
                _tag, _dtype, key, _location, numel = storage
                return {
                    "__tensor__": True,
                    "key": key,
                    "size": tuple(size),
                    "stride": tuple(stride),
                    "offset": storage_offset,
                    "numel": numel,
                }

            return rebuild
        return lambda *a, **k: None

    def persistent_load(self, pid):
        return pid  # ('storage', storage_type, key, location, numel)


def _contiguous_stride(size: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    acc = 1
    for s in reversed(size):
        stride.insert(0, acc)
        acc *= s
    return tuple(stride)


def convert(src: Path, dst: Path) -> None:
    zf = zipfile.ZipFile(src)
    pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]  # e.g. "archive/"
    obj = _Unpickler(io.BytesIO(zf.read(pkl_name))).load()
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    tensors = [
        (name, t)
        for name, t in state.items()
        if isinstance(t, dict)
        and t.get("__tensor__")
        and (name.startswith("encoder.") or name.startswith("decoder."))
    ]
    tensors.sort(key=lambda kv: kv[0])
    if not tensors:
        raise SystemExit(f"{src}: no encoder/decoder conv tensors found — is this the implicit run?")

    dtype, width = FLOAT32
    header: dict[str, object] = {}
    buffers: list[bytes] = []
    offset = 0
    for name, t in tensors:
        size = list(t["size"])
        numel = 1
        for s in size:
            numel *= s
        if t["stride"] != _contiguous_stride(tuple(size)):
            raise SystemExit(f"{name}: non-contiguous storage {t['stride']}; cannot copy verbatim")
        if t["offset"] != 0:
            raise SystemExit(f"{name}: nonzero storage offset {t['offset']}; unsupported")
        raw = zf.read(f"{prefix}data/{t['key']}")
        if len(raw) != numel * width:
            raise SystemExit(f"{name}: storage {len(raw)}B != {numel * width}B (dtype mismatch)")
        header[name] = {"dtype": dtype, "shape": size, "data_offsets": [offset, offset + len(raw)]}
        buffers.append(raw)
        offset += len(raw)

    header["__metadata__"] = {
        "source": "resemble-perth perth_net_250000.pth.tar (implicit)",
        "license": "MIT",
        "producer": "scripts/audio/convert_perth_watermarker.py (sc-13240)",
        "note": "encoder+decoder conv weights only; ap.* STFT windows are recomputed constants",
    }
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((8 - (len(hjson) % 8)) % 8)  # safetensors 8-byte data alignment

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in buffers:
            f.write(b)
    print(f"wrote {dst}: {len(tensors)} tensors, {offset} data bytes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="perth_net_250000.pth.tar (the implicit run)")
    ap.add_argument("dst", type=Path, help="output perth_implicit.safetensors path")
    args = ap.parse_args()
    convert(args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
