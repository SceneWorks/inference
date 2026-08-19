import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "ci" / "decode_quality_implementation_fingerprint.py"
SPEC = importlib.util.spec_from_file_location("decode_quality_fingerprint", SCRIPT)
assert SPEC and SPEC.loader
fingerprint_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fingerprint_module)


class DecodeQualityImplementationFingerprintTests(unittest.TestCase):
    # `test_embedded_decode_quality_source_closure_is_current` used to live here. It asserted that a
    # constant compiled into gen-core equalled the derived closure, so every dependency bump, pin
    # bump, main sync, and unrelated `mlx-gen/src` edit turned this file red until someone pasted a
    # new digest in by hand. sc-19728 deleted the constant: the stamp is now derived at measurement
    # time and recorded rather than matched, so there is no second copy that can go stale and
    # nothing here to keep current.
    #
    # `test_module_exposes_no_embedded_constant_reader` below is the negative control for that
    # removal — it fails if `embedded()` comes back.

    def test_module_exposes_no_embedded_constant_reader(self) -> None:
        self.assertFalse(
            hasattr(fingerprint_module, "embedded"),
            "reintroducing embedded() means a hand-maintained copy of this digest exists again; "
            "sc-19728 removed it deliberately",
        )
        self.assertEqual(len(fingerprint_module.fingerprint()), 64)

    def test_source_closure_is_sorted_unique_and_excludes_build_artifacts(self) -> None:
        self.assertEqual(fingerprint_module.FILES, tuple(sorted(fingerprint_module.FILES)))
        self.assertEqual(len(fingerprint_module.FILES), len(set(fingerprint_module.FILES)))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            included = root / "crate/src/lib.rs"
            included.parent.mkdir(parents=True)
            included.write_text("source", encoding="utf-8")
            manifest = root / "crate/Cargo.toml"
            manifest.write_text("[package]", encoding="utf-8")
            target = root / "crate/target/debug/output"
            target.parent.mkdir(parents=True)
            target.write_text("build artifact", encoding="utf-8")
            pycache = root / "crate/src/__pycache__/cached.pyc"
            pycache.parent.mkdir(parents=True)
            pycache.write_bytes(b"cache")

            self.assertEqual(
                fingerprint_module.semantic_source_closure(root, ("crate",)),
                ("crate/Cargo.toml", "crate/src/lib.rs"),
            )

    def test_required_production_paths_are_bound_and_mutation_sensitive(self) -> None:
        required = (
            "crates/media/mlx-gen/src/array.rs",
            "crates/media/mlx-gen/src/attention.rs",
            "crates/media/mlx-gen/src/decoder.rs",
            "crates/media/mlx-gen/src/nn.rs",
            "crates/media/mlx-gen/src/quant.rs",
            "crates/media/mlx-gen/src/sampler.rs",
            "crates/media/mlx-gen/src/scheduler.rs",
            "crates/media/mlx-gen/src/tokenizer.rs",
            "crates/media/mlx-gen/src/weights.rs",
            "crates/media/mlx-gen/mlx-gen-kolors/src/chatglm3.rs",
            "crates/media/mlx-gen/mlx-gen-kolors/src/model.rs",
            "crates/media/mlx-gen/mlx-gen-kolors/src/sampler.rs",
            "crates/media/mlx-gen/mlx-gen-kolors/src/tokenizer.rs",
            "crates/media/mlx-gen/mlx-gen-sdxl/src/config.rs",
            "crates/media/mlx-gen/mlx-gen-sdxl/src/lib.rs",
            "crates/media/mlx-gen/mlx-gen-sdxl/src/loader.rs",
            "crates/media/mlx-gen/mlx-gen-sdxl/src/plan.rs",
            "crates/media/mlx-gen/mlx-gen-sdxl/src/sampler.rs",
            "crates/media/mlx-gen/mlx-gen-sdxl/src/text_encoder.rs",
            "crates/media/mlx-gen/mlx-gen-sdxl/src/tokenizer.rs",
            "crates/media/mlx-gen/mlx-gen-sdxl/src/unet/mod.rs",
            "crates/media/mlx-gen/mlx-gen-chroma/assets/t5_tokenizer.json",
            "crates/media/mlx-gen/mlx-gen-chroma/src/beta.rs",
            "crates/media/mlx-gen/mlx-gen-chroma/src/config.rs",
            "crates/media/mlx-gen/mlx-gen-chroma/src/loader.rs",
            "crates/media/mlx-gen/mlx-gen-chroma/src/text.rs",
            "crates/media/mlx-gen/mlx-gen-chroma/src/transformer.rs",
            "crates/media/mlx-gen/mlx-gen-flux/assets/clip_tokenizer.json",
            "crates/media/mlx-gen/mlx-gen-flux/src/loader.rs",
            "crates/media/mlx-gen/mlx-gen-flux/src/pipeline.rs",
            "crates/media/mlx-gen/mlx-gen-flux/src/text_encoder.rs",
        )
        self.assertEqual(set(required).difference(fingerprint_module.FILES), set())
        original_read_bytes = Path.read_bytes
        baseline = fingerprint_module.fingerprint()
        for relative in required:
            with self.subTest(relative=relative):
                runtime_source = fingerprint_module.ROOT / relative

                def read_bytes_with_runtime_mutation(path: Path) -> bytes:
                    payload = original_read_bytes(path)
                    if path == runtime_source:
                        return payload + b"\n// decode-quality fingerprint mutation\n"
                    return payload

                with mock.patch.object(Path, "read_bytes", read_bytes_with_runtime_mutation):
                    self.assertNotEqual(baseline, fingerprint_module.fingerprint())


if __name__ == "__main__":
    unittest.main()
