from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "scripts" / "reference"


class Sa3ReferenceHardeningTests(unittest.TestCase):
    def test_shared_helpers_have_one_implementation_and_explicit_policy_calls(self) -> None:
        targets = [
            "sa3_reference.py",
            "sa3_same_reference.py",
            "sa3_same_l_reference.py",
            "sa3_chunked_autoencoder_reference.py",
            "sa3_small_music_provider_reference.py",
            "sa3_text_reference.py",
        ]
        definitions: dict[str, list[str]] = {
            "sha256_file": [],
            "tensor_records": [],
        }
        for name in targets:
            tree = ast.parse((REFERENCE / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in definitions:
                    definitions[node.name].append(name)
        self.assertEqual(definitions["sha256_file"], ["sa3_reference.py"])
        self.assertEqual(definitions["tensor_records"], ["sa3_reference.py"])

        for name in targets[1:]:
            source = (REFERENCE / name).read_text(encoding="utf-8")
            if name != "sa3_text_reference.py":
                self.assertIn("validate_upstream_checkout(", source)
            self.assertIn("allow_venv=", source)

    def test_generate_cli_has_no_misleading_component_subset(self) -> None:
        result = subprocess.run(
            [sys.executable, REFERENCE / "sa3_reference.py", "generate", "--help"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertNotIn("--components", result.stdout)
        source = (REFERENCE / "sa3_reference.py").read_text(encoding="utf-8")
        self.assertIn("selected_keys = tuple(SPEC_BY_KEY)", source)

    def test_default_outputs_are_repo_rooted_from_an_unrelated_cwd(self) -> None:
        scripts = (
            REFERENCE / "sa3_chunked_autoencoder_reference.py",
            REFERENCE / "sa3_small_music_provider_reference.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            for script in scripts:
                with self.subTest(script=script.name):
                    result = subprocess.run(
                        [sys.executable, script, "--verify"],
                        cwd=temporary,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_save_helper_imports_serializer_once_outside_tensor_loop(self) -> None:
        source = (REFERENCE / "sa3_reference.py").read_text(encoding="utf-8")
        self.assertNotIn('__import__("safetensors.torch"', source)
        tree = ast.parse(source)
        helper = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_save_tensors"
        )
        imports = [node for node in helper.body if isinstance(node, ast.ImportFrom)]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "safetensors.torch")

    def test_generators_emit_their_declared_inputs_contract_constant(self) -> None:
        for name in ("sa3_dit_reference.py", "sa3_sampler_guidance_reference.py"):
            with self.subTest(script=name):
                source = (REFERENCE / name).read_text(encoding="utf-8")
                self.assertIn('"inputs": EXPECTED_INPUTS', source)

    def test_verifier_failures_are_concise_stderr_without_tracebacks(self) -> None:
        commands = (
            [sys.executable, REFERENCE / "sa3_sampler_reference.py", "--output"],
            [sys.executable, REFERENCE / "sa3_sampler_guidance_reference.py", "--output"],
            [sys.executable, REFERENCE / "sa3_text_reference.py", "verify", "--output-dir"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            missing = str(Path(temporary) / "missing")
            for prefix in commands:
                with self.subTest(script=Path(prefix[1]).name):
                    result = subprocess.run(
                        [*prefix, missing],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue(result.stderr.strip())
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_non_object_manifests_are_domain_errors_not_tracebacks(self) -> None:
        commands = (
            (
                "sa3_sampler_reference.py",
                "manifest.json",
                lambda output: ["--output", str(output)],
            ),
            (
                "sa3_sampler_guidance_reference.py",
                "guidance-manifest.json",
                lambda output: ["--output", str(output)],
            ),
            (
                "sa3_text_reference.py",
                "manifest.json",
                lambda output: ["verify", "--output-dir", str(output)],
            ),
        )
        for document in ([], None):
            for script_name, manifest_name, arguments in commands:
                with self.subTest(script=script_name, document=document):
                    with tempfile.TemporaryDirectory() as temporary:
                        output = Path(temporary)
                        (output / manifest_name).write_text(
                            json.dumps(document), encoding="utf-8"
                        )
                        result = subprocess.run(
                            [
                                sys.executable,
                                REFERENCE / script_name,
                                *arguments(output),
                            ],
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                        )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("must be a JSON object", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
