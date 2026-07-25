import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.reference.sa3_reference import (
    COMMON_FILES,
    EXPECTED_RUNTIME,
    SNAPSHOTS,
    T5_FILES,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    InvalidReference,
    build_snapshot_lock,
    expected_artifact_filename,
    expected_artifact_metadata,
    expected_tensor_keys,
    reference_inputs,
    resolve_snapshots,
    validate_runtime_versions,
    validate_upstream_checkout,
    verify_artifacts,
)


class StableAudio3ReferenceTests(unittest.TestCase):
    def make_snapshots(
        self, root: Path
    ) -> tuple[dict[str, str], dict[str, object]]:
        environ = {}
        paths = {}
        for spec in SNAPSHOTS:
            snapshot = root / spec.revision
            snapshot.mkdir()
            for name in COMMON_FILES:
                path = snapshot / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{spec.key}:{name}".encode())
            if spec.kind == "dit":
                for name in T5_FILES:
                    path = snapshot / "t5gemma-b-b-ul2" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{spec.key}:t5:{name}".encode())
            environ[spec.env] = str(snapshot)
            paths[spec.key] = snapshot
        return environ, build_snapshot_lock(paths)

    @staticmethod
    def write_safetensors(
        path: Path, component: str, metadata: dict[str, str] | None = None
    ) -> dict[str, object]:
        keys = sorted(expected_tensor_keys(component))
        header: dict[str, object] = {
            "__metadata__": metadata or expected_artifact_metadata(component)
        }
        tensors = {}
        for offset, name in enumerate(keys):
            header[name] = {
                "dtype": "BOOL",
                "shape": [1],
                "data_offsets": [offset, offset + 1],
            }
            tensors[name] = {
                "dtype": "bool",
                "shape": [1],
                "sha256": hashlib.sha256(b"\0").hexdigest(),
            }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * ((8 - len(encoded) % 8) % 8)
        payload = len(encoded).to_bytes(8, "little") + encoded + b"\0" * len(keys)
        path.write_bytes(payload)
        return {
            "file": path.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "tensors": tensors,
        }

    def make_artifacts(self, output: Path) -> dict[str, object]:
        artifacts = {}
        for spec in SNAPSHOTS:
            filename = expected_artifact_filename(spec.key)
            artifacts[spec.key] = self.write_safetensors(
                output / filename, spec.key
            )
        manifest = {
            "schemaVersion": 1,
            "upstream": {
                "repository": UPSTREAM_REPOSITORY,
                "commit": UPSTREAM_COMMIT,
            },
            "inputs": reference_inputs(),
            "referenceEnvironment": {
                **EXPECTED_RUNTIME,
                "device": "cpu",
                "platform": "test-platform",
            },
            "artifacts": artifacts,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    def test_requires_all_explicit_snapshot_environment_variables(self) -> None:
        with self.assertRaisesRegex(InvalidReference, SNAPSHOTS[0].env):
            resolve_snapshots({})

    def test_accepts_all_pinned_complete_snapshot_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environ, lock = self.make_snapshots(Path(temporary))
            resolved = resolve_snapshots(environ, lock)
            self.assertEqual(set(resolved), {spec.key for spec in SNAPSHOTS})

    def test_rejects_snapshot_payload_size_and_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environ, lock = self.make_snapshots(root)
            config = Path(environ[SNAPSHOTS[0].env]) / "model_config.json"
            original = config.read_bytes()
            config.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.assertRaisesRegex(InvalidReference, "SHA-256 mismatch"):
                resolve_snapshots(environ, lock)

            config.write_bytes(original + b"x")
            with self.assertRaisesRegex(InvalidReference, "size mismatch"):
                resolve_snapshots(environ, lock)

    def test_rejects_revision_drift_and_missing_t5_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environ, lock = self.make_snapshots(root)
            first = SNAPSHOTS[0]
            wrong = root / ("a" * 40)
            Path(environ[first.env]).rename(wrong)
            environ[first.env] = str(wrong)
            with self.assertRaisesRegex(InvalidReference, "revision mismatch"):
                resolve_snapshots(environ, lock)

            wrong.rename(root / first.revision)
            environ[first.env] = str(root / first.revision)
            (root / first.revision / "t5gemma-b-b-ul2/tokenizer.json").unlink()
            with self.assertRaisesRegex(InvalidReference, "tokenizer.json"):
                resolve_snapshots(environ, lock)

    def test_runtime_version_validation_fails_each_drift_seam(self) -> None:
        validate_runtime_versions(EXPECTED_RUNTIME.copy())
        for package in EXPECTED_RUNTIME:
            with self.subTest(package=package):
                drifted = EXPECTED_RUNTIME.copy()
                drifted[package] = "0.0.0"
                with self.assertRaisesRegex(InvalidReference, package):
                    validate_runtime_versions(drifted)

    def test_upstream_checkout_requires_exact_clean_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "codex@example.test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Codex"],
                check=True,
            )
            source = root / "source.py"
            source.write_text("PINNED = True\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "fixture"],
                check=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.strip()
            validate_upstream_checkout(root, revision)
            with self.assertRaisesRegex(InvalidReference, "checkout mismatch"):
                validate_upstream_checkout(root, "0" * 40)
            untracked = root / "untracked"
            untracked.write_text("rejected\n", encoding="utf-8")
            with self.assertRaisesRegex(InvalidReference, "not clean"):
                validate_upstream_checkout(root, revision)
            untracked.unlink()
            source.write_text("PINNED = False\n", encoding="utf-8")
            with self.assertRaisesRegex(InvalidReference, "not clean"):
                validate_upstream_checkout(root, revision)

    def test_upstream_checkout_rejects_dependency_shadowing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "codex@example.test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Codex"],
                check=True,
            )
            source = root / "source.py"
            source.write_text("PINNED = True\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "fixture"],
                check=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.strip()
            for filename in ("torch.py", "transformers.py"):
                with self.subTest(filename=filename):
                    shadow = root / filename
                    shadow.write_text("raise RuntimeError('shadowed')\n", encoding="utf-8")
                    with self.assertRaisesRegex(InvalidReference, filename):
                        validate_upstream_checkout(root, revision)
                    shadow.unlink()

    def test_artifact_verifier_requires_all_components_and_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = self.make_artifacts(output)
            verify_artifacts(output)

            manifest["artifacts"].pop("same-l")
            (output / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(InvalidReference, "all eight components"):
                verify_artifacts(output)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = self.make_artifacts(output)
            manifest["artifacts"]["small-music"]["tensors"].pop("dit_prediction")
            (output / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(InvalidReference, "tensor inventory"):
                verify_artifacts(output)

    def test_artifact_verifier_checks_header_metadata_and_payload_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = self.make_artifacts(output)
            manifest["artifacts"]["small-music"]["tensors"]["dit_prediction"][
                "sha256"
            ] = "0" * 64
            (output / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(InvalidReference, "payload hash mismatch"):
                verify_artifacts(output)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = self.make_artifacts(output)
            path = output / expected_artifact_filename("same-s")
            manifest["artifacts"]["same-s"] = self.write_safetensors(
                path, "same-s", {"component": "wrong"}
            )
            (output / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(InvalidReference, "metadata mismatch"):
                verify_artifacts(output)


if __name__ == "__main__":
    unittest.main()
