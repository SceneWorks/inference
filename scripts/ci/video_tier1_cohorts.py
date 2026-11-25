"""Fixed source and fixture bindings for the dispatched shared-driver video migrations."""
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath

WAN = "wan-curated-tier1"
MOCHI_SVD = "mochi-svd-tier1"
SCAIL = "scail2-tier1"
VIDEO_BASE = "c7911db0e6cdcc41762326e2f8aad7e60634c2a1"


@dataclass(frozen=True)
class Case:
    name: str
    model: str
    variant: str
    depth: int = 1


CASES = {
    WAN: {
        "mlx": (Case("mlx-a14b-t2v-resident-heun", "wan", "mlx-a14b-t2v"),
                Case("mlx-a14b-i2v-resident-heun", "wan", "mlx-a14b-i2v"),
                Case("mlx-a14b-i2v-swapped-heun", "wan", "mlx-a14b-i2v"),
                Case("mlx-ti2v5b-t2v-heun", "wan", "mlx-ti2v5b")),
        "candle-cuda": (Case("cuda-ti2v5b-t2v-heun", "wan", "cuda-ti2v5b"),)},
    MOCHI_SVD: {
        "mlx": (Case("mlx-mochi-masked-euler", "mochi", "mochi", 2),
                Case("mlx-svd-native-euler", "svd", "mlx-svd"), Case("mlx-svd-heun", "svd", "mlx-svd")),
        "candle-cuda": (Case("cuda-mochi-masked-euler", "mochi", "mochi", 2),
                        Case("cuda-svd-euler-cfg", "svd", "cuda-svd"), Case("cuda-svd-euler-single", "svd", "cuda-svd"),
                        Case("cuda-svd-heun-cfg", "svd", "cuda-svd"))},
    SCAIL: {
        "mlx": (Case("mlx-scail2-animation-unipc", "scail2", "scail2-bf16"),
                Case("mlx-scail2-replacement-dpmpp", "scail2", "scail2-bf16")),
        "candle-cuda": (Case("cuda-scail2-animation-unipc", "scail2", "scail2-bf16"),
                        Case("cuda-scail2-replacement-dpmpp", "scail2", "scail2-bf16"))},
}
BY_NAME = {case.name: case for lanes in CASES.values() for cases in lanes.values() for case in cases}
FQN = {"wan": "wan_pair_fixture::curated_tier1", "mochi": "video_pair_fixture::curated_tier1",
       "svd": "video_pair_fixture::curated_tier1", "scail2": "generate::scail_pair_fixture::curated_tier1"}
WEIGHT_FILE = {"mochi": "transformer/model.safetensors", "mlx-svd": "unet/diffusion_pytorch_model.fp16.safetensors",
               "cuda-svd": "unet/diffusion_pytorch_model.safetensors", "scail2-bf16": "dit.safetensors"}
DIRECTORY = {WAN: "wan_pair", MOCHI_SVD: "ms_pair", SCAIL: "scail_pair"}


def fixture(model):
    if model == "wan":
        from scripts.ci.wan_pair import fixture
    elif model in ("mochi", "svd"):
        from scripts.ci.ms_pair import fixture
    elif model == "scail2":
        from scripts.ci.scail_pair import fixture
    else:
        raise ValueError("unregistered proof model")
    return fixture


def cases(selection, backend):
    return tuple(c.name for c in CASES[selection][backend])


def models(names):
    return tuple(dict.fromkeys(BY_NAME[name].model for name in names))


def package(backend, model):
    return ("mlx-gen-" if backend == "mlx" else "candle-gen-") + model


def overlay_files(head, backend, arm, model, tracked, wan_source, *, engine_head=None):
    """The shared checkout writes only these fixed targets. All inputs come from Git blobs."""
    src = PurePosixPath("crates", "media", "mlx-gen" if backend == "mlx" else "candle-gen",
                        package(backend, model), "src")
    revision = (engine_head or head) if arm == "head" else VIDEO_BASE
    if model == "scail2":
        from scripts.ci.scail_pair import overlay
        module = src / "scail_pair_fixture.rs"
        module_bytes, changes = overlay.render(lambda path: tracked(revision, path).decode(),
                                               lambda path: tracked(head, path).decode(), backend, arm)
        edits = {PurePosixPath(name): data for name, data in changes.items()}
        expected = {src / "generate.rs", src / "model.rs"}
        scheduler = PurePosixPath("crates/media/candle-gen/candle-gen-wan/src/scheduler.rs")
        if backend == "candle-cuda":
            expected.update((src / "pipeline.rs", scheduler))
        if set(edits) != expected:
            raise ValueError("SCAIL proof overlay differs from the approved source files")
        for path, data in edits.items():
            original = tracked(revision, path.as_posix())
            if path == scheduler:
                # The dependency compiles without cfg(test). Permit only the proof-feature
                # getter returning FlowScheduler.sampler, with its exact body and signature.
                suffix = data.removeprefix(original)
                if (not data.startswith(original) or hashlib.sha256(suffix).hexdigest()
                        != "db2d59f01c010adf0d087639db47f7787a308e1946f65eb470669aa16228eb88"):
                    raise ValueError("unexpected dependency scheduler readback")
            elif not data.startswith(original + b"\n#[cfg(test)]"):
                raise ValueError("SCAIL probe must append test-only source to the unmodified production file")
        return module, module_bytes, edits
    if model == "wan":
        blob = wan_source(head, backend, arm)
        module = src / "wan_pair_fixture.rs"
    else:
        blob = fixture(model).overlay_source(head, backend, model, arm, tracked)
        module = src / "video_pair_fixture.rs"
        entry = src / "lib.rs"
        original = tracked(revision, entry.as_posix())
        declaration = b"#[cfg(test)]\nmod video_pair_fixture;"
        if declaration in original:
            raise ValueError("proof module already declared")
        return module, blob, {entry: original + b"\n" + declaration + b"\n"}
    return module, blob, {}
