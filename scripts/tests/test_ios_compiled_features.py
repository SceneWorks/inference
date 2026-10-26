"""Compile real identity modules across the supported SDK/harness workspace boundary."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "crates/media/mlx-gen/mlx-gen-ios-catalog/src/source_identity.rs"
# Pure fixture units compile the production marker/feature guards with their actual local edges.
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
    def test_supported_root_sdk_harness_and_unsupported_external_topologies(self):
        with tempfile.TemporaryDirectory(prefix="ios-compiled-features-") as temporary:
            root = Path(temporary)
            checkout = root / "checkout-a"
            sdk = checkout / "sdk"
            harness = checkout / "ios-host/smoke"
            (sdk / "src").mkdir(parents=True)
            (harness / "src").mkdir(parents=True)
            (sdk / "evidence.json").write_text('{"capture": "original"}')
            (checkout / "Cargo.toml").write_text(
                '[workspace]\nresolver="2"\nmembers=["sdk","providers/*","ios-host/smoke"]\n'
                'exclude=["unsupported-host"]\n'
            )
            dependencies = "\n".join(f'{name} = {{ path = "../providers/{name}" }}' for name in UNITS)
            (sdk / "Cargo.toml").write_text(
                '[package]\nname="mlx-gen-ios-catalog"\nversion="0.0.0"\nedition="2021"\n'
                '[dependencies]\n' + dependencies + '\n'
                'serde={version="1",features=["derive"]}\nserde_json="1"\nsha2="0.10"\n'
                '[features]\ndevice-capture=[]\n'
            )
            for name, (source, local_dependencies) in UNITS.items():
                leaf = checkout / "providers" / name
                (leaf / "src").mkdir(parents=True)
                manifest = f'[package]\nname="{name}"\nversion="0.0.0"\nedition="2021"\n'
                if name == "sceneworks-gen-core":
                    manifest += '[lib]\nname="gen_core"\n'
                manifest += '[features]\ntest-support=[]\ntraining=[]\ntest-clock=[]\n'
                # A downstream registry feature unifies into these unchanged provider features.
                manifest += '[dependencies]\nserde_json="1"\n'
                manifest += "\n".join(f'{dependency}={{path="../{dependency}"}}' for dependency in local_dependencies)
                (leaf / "Cargo.toml").write_text(manifest)
                (leaf / "src/lib.rs").write_text(
                    f'#[path={json.dumps(str(ROOT / source / "src/ios_build_identity.rs"))}]\n'
                    'mod ios_build_identity;\npub use ios_build_identity::*;\n'
                )
            cargo_version = subprocess.check_output(["cargo", "-vV"])
            constants = "\n".join(
                f"pub const {key}: &str = {json.dumps(value)};"
                for key, value in {
                    "SOURCE_CLOSURE_SHA256": "1" * 64,
                    "BUILD_TARGET": "aarch64-apple-ios",
                    "BUILD_RUSTC": "compiled feature regression",
                    "BUILD_CONFIG_SHA256": hashlib.sha256(cargo_version).hexdigest(),
                    "NATIVE_MLX_FINGERPRINT": "3" * 64,
                }.items()
            )
            (sdk / "build.rs").write_text(
                'fn main(){let features=if std::env::var_os("CARGO_FEATURE_DEVICE_CAPTURE").is_some()'
                '{"device-capture"}else{""};'
                f'let mut source={json.dumps(constants)}.to_owned();'
                'source.push_str(&format!("\\npub const BUILD_FEATURES: &str = {features:?};"));'
                'std::fs::write(std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap())'
                '.join("ios_source_identity.rs"),source).unwrap();}\n'
            )
            (sdk / "src/lib.rs").write_text(
                f'#[path={json.dumps(str(IDENTITY))}] pub mod source_identity;\n'
                'const _: &str=include_str!("../evidence.json");\n'
            )
            main = (
                'fn main(){let current=mlx_gen_ios_catalog::source_identity::BuildIdentity::current();'
                'let reference=std::env::args().nth(1).map(|raw|serde_json::from_str(&raw).unwrap())'
                '.unwrap_or_else(||current.clone());'
                'println!("{}",serde_json::json!({"identity":current,'
                '"compatible":current.same_production_execution(&reference)}));}\n'
            )
            (sdk / "src/main.rs").write_text(main)
            harness_manifest = (
                '[package]\nname="ios-smoke"\nversion="0.0.0"\nedition="2021"\n'
                '[dependencies]\nmlx-gen-ios-catalog={path="../../sdk",features=["device-capture"]}\n'
                'serde_json="1"\n'
            )
            (harness / "Cargo.toml").write_text(harness_manifest)
            (harness / "src/main.rs").write_text(main)
            environment = os.environ.copy()
            environment["CARGO_TARGET_DIR"] = str(root / "target")
            environment["CARGO_INCREMENTAL"] = "0"
            environment["CARGO_BUILD_JOBS"] = "4"
            environment["CARGO_PROFILE_DEV_DEBUG"] = "0"

            def capture(features="", reference=None, manifest=sdk / "Cargo.toml", offline=True):
                command = ["cargo", "run", "--quiet", "--manifest-path", str(manifest)]
                if offline:
                    command.append("--offline")
                if features:
                    command.extend(["--features", features])
                if reference is not None:
                    command.extend(["--", json.dumps(reference)])
                result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=45)
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
            self.assertTrue(capture(manifest=harness / "Cargo.toml", reference=product["identity"])["compatible"], "real root-member SDK/harness composition must match")
            (sdk / "evidence.json").write_text('{"capture":"reviewed after original device run"}')
            (checkout / "README.md").write_text("Docs-only change after capture.\n")
            self.assertEqual(capture()["identity"], product["identity"], "policy/evidence/docs do not change dependency units")
            other = root / "checkout-b"
            shutil.copytree(checkout, other)
            for name, (source, _) in UNITS.items():
                original = ROOT / source / "src/ios_build_identity.rs"
                moved = other / "providers" / name / "src/ios_build_identity.rs"
                shutil.copyfile(original, moved)
                lib = other / "providers" / name / "src/lib.rs"
                lib.write_text(lib.read_text().replace(json.dumps(str(original)), json.dumps(str(moved))))
            relocated = capture(manifest=other / "sdk/Cargo.toml")
            self.assertEqual(relocated["identity"], product["identity"], "relocated root workspace must match")
            self.assertTrue(capture(manifest=other / "ios-host/smoke/Cargo.toml", reference=product["identity"])["compatible"])
            # A separate nested workspace is deliberately unsupported: it must not reuse root captures.
            for tree in [checkout, other]:
                external = tree / "unsupported-host"
                (external / "src").mkdir(parents=True)
                (external / "src/main.rs").write_text(main)
                (external / "Cargo.toml").write_text(harness_manifest.replace('"../../sdk"', '"../sdk"')+'[workspace]\n')
                external_capture = capture(manifest=external / "Cargo.toml", reference=product["identity"])
                self.assertEqual(external_capture["identity"]["source_closure_sha256"], product["identity"]["source_closure_sha256"])
                self.assertEqual(len(external_capture["identity"]["build_config_sha256"]), 64)
                self.assertFalse(external_capture["compatible"], "external source identity must not reuse a root-workspace capture")
            # Same source through Git transport is outside the supported root-workspace composition.
            for arguments in [["init", "-q"], ["add", "."], ["-c", "user.name=codex", "-c", "user.email=codex@example.invalid", "commit", "-qm", "fixture"]]:
                subprocess.run(["git", "-C", str(checkout), *arguments], check=True, capture_output=True)
            revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
            git_consumer = root / "git-consumer"
            (git_consumer / "src").mkdir(parents=True)
            (git_consumer / "src/main.rs").write_text(main)
            (git_consumer / "Cargo.toml").write_text(
                '[package]\nname="git-consumer"\nversion="0.0.0"\nedition="2021"\n[workspace]\n'
                f'[dependencies]\nmlx-gen-ios-catalog={{git="{checkout.as_uri()}",rev="{revision}"}}\nserde_json="1"\n'
            )
            git_capture = capture(manifest=git_consumer / "Cargo.toml", reference=product["identity"], offline=False)
            self.assertEqual(git_capture["identity"]["source_closure_sha256"], product["identity"]["source_closure_sha256"])
            self.assertEqual(git_capture["identity"]["features"], product["identity"]["features"])
            self.assertFalse(git_capture["compatible"], "Git consumption is outside the supported source-identity topology")


if __name__ == "__main__":
    unittest.main()
