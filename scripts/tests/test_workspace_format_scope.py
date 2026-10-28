"""CI_IMPACT: metadata/rustfmt-only local fixture; no compilation, network or model execution."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]


class WorkspaceFormatScopeTests(unittest.TestCase):
    def test_format_job_checks_the_virtual_workspace_without_dependency_recursion(self):
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text())
        self.assertIn("workspace", manifest)
        self.assertNotIn("package", manifest)
        workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
        steps = workflow["jobs"]["format"]["steps"]
        self.assertEqual(
            [step["run"] for step in steps if step.get("name") == "Check whole-workspace formatting"],
            ["cargo fmt --check"],
        )

    def test_all_member_targets_and_nested_drift_are_checked_without_touching_vendor(self):
        cargo = shutil.which("cargo")
        if cargo is None:
            self.skipTest("Cargo/rustfmt is unavailable on this tooling-only host")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()

            def write(relative, text):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                return path

            write("Cargo.toml", '[workspace]\nresolver="2"\nmembers=["first","second"]\n'
                  'default-members=["first"]\nexclude=["vendor"]\n')
            for name in ("first", "second", "vendor"):
                dependency = '\n[dependencies]\nvendor={path="../vendor"}\n' if name == "second" else ""
                write(f"{name}/Cargo.toml", f'[package]\nname="{name}"\nversion="0.0.0"\nedition="2021"\n' + dependency)
            write("first/src/lib.rs", "pub fn first() {}\n")
            write("second/src/lib.rs", "mod nested;\n")
            nested = write("second/src/nested.rs", "pub fn nested( )->u8{1}\n")
            write("second/src/main.rs", "fn main() {}\n")
            write("second/build.rs", "fn main() {}\n")
            write("second/examples/demo.rs", "fn main() {}\n")
            write("second/tests/contract.rs", "#[test]\nfn contract() {}\n")
            write("vendor/src/lib.rs", "pub fn vendor( )->u8{1}\n")
            vendor_before = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (root / "vendor").rglob("*") if p.is_file()}
            toolchain = tomllib.loads((ROOT / "rust-toolchain.toml").read_text())["toolchain"]["channel"]
            env = dict(os.environ, CARGO_NET_OFFLINE="true", RUSTUP_AUTO_INSTALL="0",
                       RUSTUP_TOOLCHAIN=toolchain)

            def run(*arguments):
                return subprocess.run([cargo, *arguments],
                                      cwd=root, env=env, capture_output=True, text=True, timeout=30)

            red = run("fmt", "--check")
            self.assertNotEqual(red.returncode, 0)
            self.assertIn("second/src/nested.rs", (red.stdout + red.stderr).replace("\\", "/"))
            nested.write_text("pub fn nested() -> u8 {\n    1\n}\n")
            green = run("fmt", "--check", "--verbose")
            self.assertEqual(green.returncode, 0, green.stdout + green.stderr)
            metadata = run("metadata", "--offline", "--no-deps", "--format-version", "1")
            self.assertEqual(metadata.returncode, 0, metadata.stderr)
            metadata = json.loads(metadata.stdout)
            members = set(metadata["workspace_members"])
            expected = {target["src_path"] for package in metadata["packages"] if package["id"] in members
                        for target in package["targets"]}
            formatted = {json.loads(line.split("] ", 1)[1]) for line in green.stdout.splitlines()
                         if line.startswith("[") and "] " in line}
            self.assertEqual(formatted, expected)
            self.assertEqual(len(members), 2)
            recursive = run("fmt", "--all", "--check")
            self.assertNotEqual(recursive.returncode, 0)
            self.assertIn("vendor/src/lib.rs", recursive.stdout.replace("\\", "/"))
            vendor_after = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (root / "vendor").rglob("*") if p.is_file()}
            self.assertEqual(vendor_after, vendor_before)


if __name__ == "__main__":
    unittest.main()
