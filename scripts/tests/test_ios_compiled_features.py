"""Compile real identity code against Cargo units; no native MLX link or device work."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "crates/media/mlx-gen/mlx-gen-ios-catalog/src/source_identity.rs"
# Tiny Cargo fixture units compile the real, pure identity modules, with their actual local edges.
UNITS = {
    "core-llm": ("crates/contracts/core-llm", []),
    "sceneworks-gen-core": ("crates/contracts/gen-core", ["core-llm"]),
    "mlx-llm": ("crates/llm/mlx-llm", ["core-llm"]),
    "mlx-gen": ("crates/media/mlx-gen", ["sceneworks-gen-core", "mlx-llm"]),
    "mlx-gen-pid": ("crates/media/mlx-gen/mlx-gen-pid", ["mlx-gen"]),
    "mlx-gen-wan": ("crates/media/mlx-gen/mlx-gen-wan", ["mlx-gen"]),
    "mlx-gen-qwen-image": ("crates/media/mlx-gen/mlx-gen-qwen-image", ["mlx-gen", "mlx-gen-pid", "mlx-gen-wan"]),
    "mlx-gen-z-image": ("crates/media/mlx-gen/mlx-gen-z-image", ["mlx-gen", "mlx-gen-pid"]),
    "mlx-gen-sana": ("crates/media/mlx-gen/mlx-gen-sana", ["mlx-gen", "mlx-gen-pid"]),
    "mlx-gen-sdxl": ("crates/media/mlx-gen/mlx-gen-sdxl", ["mlx-gen", "mlx-gen-pid"]),
    "mlx-gen-iris": ("crates/media/mlx-gen/mlx-gen-iris", ["mlx-gen", "mlx-llm"]),
    "mlx-gen-anima": ("crates/media/mlx-gen/mlx-gen-anima", ["mlx-gen", "mlx-gen-z-image", "mlx-gen-qwen-image"]),
}


class CompiledFeatureIdentityTests(unittest.TestCase):
    def test_compiled_graph_changes_and_data_checkout_stability(self):
        with tempfile.TemporaryDirectory(prefix="ios-compiled-features-") as temporary:
            root = Path(temporary)
            directory = root / "checkout-a"
            (directory / "src").mkdir(parents=True)
            (directory / "evidence.json").write_text('{"capture": "original"}')
            dependencies = "\n".join(f'{name} = {{ path = "{name}" }}' for name in UNITS)
            (directory / "Cargo.toml").write_text(
                '[package]\nname = "mlx-gen-ios-catalog"\nversion = "0.0.0"\nedition = "2021"\n'
                '[workspace]\n[dependencies]\n' + dependencies + '\n'
                'serde = { version = "1", features = ["derive"] }\nserde_json = "1"\nsha2 = "0.10"\n'
                '[features]\ndevice-capture = []\n'
            )
            for name, (source, local_dependencies) in UNITS.items():
                leaf = directory / name
                (leaf / "src").mkdir(parents=True)
                manifest = f'[package]\nname = "{name}"\nversion = "0.0.0"\nedition = "2021"\n'
                if name == "sceneworks-gen-core":
                    manifest += '[lib]\nname = "gen_core"\n'
                manifest += '[features]\ntest-support = []\ntraining = []\ntest-clock = []\n'
                # Root unifies serde_json's flag without setting any provider's own feature.
                manifest += '[dependencies]\nserde_json = "1"\n'
                manifest += "\n".join(f'{dependency} = {{ path = "../{dependency}" }}' for dependency in local_dependencies)
                (leaf / "Cargo.toml").write_text(manifest)
                (leaf / "src/lib.rs").write_text(
                    f'#[path = {json.dumps(str(ROOT / source / "src/ios_build_identity.rs"))}]\n'
                    'mod ios_build_identity;\npub use ios_build_identity::*;\n'
                )
            constants = "\n".join(
                f"pub const {key}: &str = {json.dumps(value)};"
                for key, value in {
                    "SOURCE_CLOSURE_SHA256": "1" * 64,
                    "BUILD_TARGET": "aarch64-apple-ios",
                    "BUILD_RUSTC": "compiled feature regression",
                    "BUILD_CONFIG_SHA256": "2" * 64,
                    "NATIVE_MLX_FINGERPRINT": "3" * 64,
                }.items()
            )
            (directory / "build.rs").write_text(
                'fn main() { let features = if std::env::var_os("CARGO_FEATURE_DEVICE_CAPTURE").is_some() '
                '{ "device-capture" } else { "" }; '
                f'let mut source = {json.dumps(constants)}.to_owned();'
                'source.push_str(&format!("\\npub const BUILD_FEATURES: &str = {features:?};"));'
                'std::fs::write(std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap())'
                '.join("ios_source_identity.rs"), source).unwrap(); }\n'
            )
            (directory / "src/main.rs").write_text(
                f"#[path = {json.dumps(str(IDENTITY))}] mod identity;\n"
                'const _: &str = include_str!("../evidence.json");\n'
                'fn main() { let current = identity::BuildIdentity::current();'
                ' let reference = std::env::args().nth(1).map(|raw| serde_json::from_str(&raw).unwrap())'
                ' .unwrap_or_else(|| current.clone());'
                ' println!("{}", serde_json::json!({"identity": current,'
                ' "compatible": current.same_production_execution(&reference)})); }\n'
            )
            environment = os.environ.copy()
            environment["CARGO_TARGET_DIR"] = str(root / "target")
            environment["CARGO_INCREMENTAL"] = "0"

            def capture(features="", reference=None, checkout=directory):
                command = ["cargo", "run", "--quiet", "--offline", "--manifest-path", str(checkout / "Cargo.toml")]
                if features:
                    command.extend(["--features", features])
                if reference is not None:
                    command.extend(["--", json.dumps(reference)])
                result = subprocess.run(command, env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)

            product = capture()
            self.assertTrue(product["compatible"])
            fixture = capture("mlx-gen-z-image/test-support", product["identity"])
            self.assertEqual(fixture["identity"]["features"], product["identity"]["features"])
            self.assertFalse(fixture["compatible"])
            self.assertFalse(capture("mlx-gen-z-image/test-support")["compatible"])
            self.assertFalse(capture("mlx-gen-z-image/training")["compatible"])
            self.assertFalse(capture("core-llm/test-clock")["compatible"])
            registry = capture("serde_json/arbitrary_precision", product["identity"])
            self.assertEqual(registry["identity"]["features"], product["identity"]["features"])
            self.assertNotEqual(registry["identity"]["build_config_sha256"], product["identity"]["build_config_sha256"])
            self.assertFalse(registry["compatible"])
            self.assertTrue(capture("serde_json/arbitrary_precision")["compatible"])
            self.assertEqual(capture()["identity"], product["identity"], "same-feature rebuild restores identity")
            self.assertTrue(capture("device-capture", product["identity"])["compatible"])
            (directory / "evidence.json").write_text('{"capture": "reviewed after original device run"}')
            (directory / "README.md").write_text("Docs-only change after capture.\n")
            self.assertEqual(capture()["identity"], product["identity"], "policy/evidence/docs do not change dependency units")
            other = root / "checkout-b"
            shutil.copytree(directory, other)
            # Compile identical production module bytes at different physical source paths too.
            for name, (source, _) in UNITS.items():
                original = ROOT / source / "src/ios_build_identity.rs"
                moved = other / name / "src/ios_build_identity.rs"
                shutil.copyfile(original, moved)
                lib = other / name / "src/lib.rs"
                lib.write_text(lib.read_text().replace(json.dumps(str(original)), json.dumps(str(moved))))
            self.assertEqual(capture(checkout=other)["identity"], product["identity"], "checkout paths do not change dependency units")


if __name__ == "__main__":
    unittest.main()
