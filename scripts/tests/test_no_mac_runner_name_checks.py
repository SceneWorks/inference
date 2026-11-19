"""No CI job may require, reject, skip or select itself by macOS runner NAME (SW-7218).

The two macOS self-hosted runners (`nax-macos`, `nax-macos-2`) are hardware-identical and
interchangeable. Which Mac runs a job is decided only by the `runs-on` labels, an operator setting.
A job may RECORD `RUNNER_NAME` as provenance; it must never compare it to a machine name.

This gate scans every workflow under `.github/workflows/`, every script under `scripts/ci/` and
`.github/kv-poc/`, and fails on a Mac runner name used as a value: a comparison, an allow-list,
a route table or a selector. Comments, prose and recorded JSON evidence are not code and are not
scanned. Windows CUDA listener names are out of scope.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
MAC_NAME = re.compile(r"(?<![\w.-])nax-macos(?:-2)?(?![\w-])")
EXACT_MAC_NAME = re.compile(r"nax-macos(?:-2)?")
SHELL_SUFFIXES = {".sh", ".cmd", ".ps1"}

# Literals that name a Mac but decide nothing about where a job runs, each with its reason.
# (path, exact literal) -> how many times it may appear.
ALLOWED_PYTHON_LITERALS = {
    # The request spelling accepted before inventory routes were keyed by label: it maps to the
    # `nax` label route, so a dispatch that still sends it selects labels, not a machine.
    ("scripts/ci/flux1_conditional_asset_inventory.py", "nax-macos-2"): 1,
    # Identity of one finished historical job (run 37127726908) that the GPU0 owner guard reads
    # back; the field is that job's recorded runner, not a placement rule for any new job.
    ("scripts/ci/yue2_gpu0_owner_guard.py", "nax-macos-2"): 1,
    # Recorded provenance of a committed reference transfer.
    ("scripts/ci/yue2_precision_reference_transfer.py", "nax-macos"): 1,
}


def strip_comment(line: str, rem: bool = False) -> str:
    stripped = line.lstrip()
    if stripped.startswith("#") or (rem and re.match(r"(?i)(rem\b|::)", stripped)):
        return ""
    index = line.find(" #")
    return line if index < 0 else line[:index]


def text_violations(label: str, text: str, rem: bool = False) -> list[str]:
    return [f"{label}:{number}: {line.strip()}"
            for number, line in enumerate(text.splitlines(), 1)
            if MAC_NAME.search(strip_comment(line, rem))]


def python_violations(label: str, text: str) -> list[str]:
    found: dict[str, list[int]] = {}
    for node in ast.walk(ast.parse(text)):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and EXACT_MAC_NAME.fullmatch(node.value.strip())):
            found.setdefault(node.value.strip(), []).append(node.lineno)
    errors = []
    for literal, lines in found.items():
        if len(lines) > ALLOWED_PYTHON_LITERALS.get((label, literal), 0):
            errors.append(f"{label}:{lines}: Mac runner name {literal!r} used as a value")
    return errors


def workflow_values(node, key=None):
    """Every executable/selector string in a workflow: never names or input descriptions."""
    if isinstance(node, dict):
        for child_key, value in node.items():
            if child_key in ("name", "description") or child_key is True or child_key == "on":
                continue
            yield from workflow_values(value, child_key)
    elif isinstance(node, list):
        for value in node:
            yield from workflow_values(value, key)
    elif isinstance(node, str):
        yield node


def workflow_violations(label: str, text: str) -> list[str]:
    errors = []
    for value in workflow_values(yaml.safe_load(text)):
        errors.extend(text_violations(label, value))
    return errors


def file_violations(path: Path) -> list[str]:
    label = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".py":
        return python_violations(label, text)
    if path.suffix in (".yml", ".yaml") and label.startswith(".github/workflows/"):
        return workflow_violations(label, text)
    if path.suffix in SHELL_SUFFIXES:
        return text_violations(label, text, rem=path.suffix == ".cmd")
    return []


def scanned_files() -> list[Path]:
    listed = subprocess.run(["git", "ls-files", ".github/workflows", ".github/kv-poc", "scripts/ci"],
                            cwd=ROOT, capture_output=True, text=True, check=True,
                            encoding="utf-8").stdout.split()
    return [ROOT / name for name in listed
            if Path(name).suffix in {".py", ".yml", ".yaml", *SHELL_SUFFIXES}]


class NoMacRunnerNameChecks(unittest.TestCase):
    def test_no_workflow_or_ci_script_names_a_mac_runner_as_a_value(self):
        files = scanned_files()
        self.assertTrue(any(f.name == "real-weights.yml" for f in files))
        self.assertTrue(any(f.parent.name == "kv-poc" for f in files))
        errors = [error for path in files for error in file_violations(path)]
        self.assertEqual(errors, [], "\n".join(errors))

    def test_allow_listed_literals_still_exist(self):
        """A stale exception would silently widen what the gate tolerates."""
        for (label, literal), count in ALLOWED_PYTHON_LITERALS.items():
            with self.subTest(label=label):
                tree = ast.parse((ROOT / label).read_text(encoding="utf-8"))
                seen = sum(isinstance(node, ast.Constant) and node.value == literal
                           for node in ast.walk(tree))
                self.assertEqual(seen, count)

    def test_gate_catches_each_way_of_checking_a_mac_name(self):
        workflow_mutants = {
            "bash equality": '[[ "$RUNNER_NAME" == nax-macos-2 ]]',
            "bash quoted": '[[ "$RUNNER_NAME" == "nax-macos" ]]',
            "test builtin": 'test "$RUNNER_NAME" = nax-macos-2',
            "either Mac": '[[ "$RUNNER_NAME" == nax-macos || "$RUNNER_NAME" == nax-macos-2 ]]',
        }
        for case, line in workflow_mutants.items():
            with self.subTest(case=case):
                workflow = yaml.safe_dump({"jobs": {"j": {"runs-on": ["self-hosted", "macOS"],
                                                          "steps": [{"run": f"set -e\n{line}\n"}]}}})
                self.assertTrue(workflow_violations("w.yml", workflow))
        step_if = yaml.safe_dump({"jobs": {"j": {"steps": [{"if": "runner.name == 'nax-macos'",
                                                            "run": "true"}]}}})
        self.assertTrue(workflow_violations("w.yml", step_if))
        for case, source in {
            "allow-list": 'if env.get("RUNNER_NAME") not in ("nax-macos", "nax-macos-2"):\n    pass\n',
            "route table": 'ROUTES = {"nax": {"name": "nax-macos-2"}}\n',
            "assert": 'assert os.environ.get("RUNNER_NAME") == "nax-macos-2"\n',
            "unlisted alias": 'ALIASES = {"nax-macos": "nax"}\n',
        }.items():
            with self.subTest(case=case):
                self.assertTrue(python_violations("scripts/ci/x.py", source))
        for case, source in {
            "shell skip": 'if [[ "$RUNNER_NAME" != "nax-macos-2" ]]; then exit 0; fi\n',
            "pinned name": "  rw-krea) host_labels=rw-krea; runner_name=nax-macos-2 ;;\n",
        }.items():
            with self.subTest(case=case):
                self.assertTrue(text_violations("x.sh", source))

    def test_provenance_and_prose_are_not_checks(self):
        workflow = yaml.safe_dump({
            "on": {"workflow_dispatch": {"inputs": {"x": {"description": "runs on nax-macos-2"}}}},
            "jobs": {"j": {"name": "Measured on nax-macos",
                           "concurrency": {"group": "yue2-app-precision-nax-macos-2"},
                           "runs-on": ["self-hosted", "macOS", "ARM64", "rw-llm"],
                           "steps": [{"run": "# measured on nax-macos\necho \"runner=$RUNNER_NAME\"\n"},
                                     {"if": "startsWith(inputs.profile, 'publish-mlx-nax-macos')",
                                      "run": "true"}]}}})
        self.assertEqual(workflow_violations("w.yml", workflow), [])
        self.assertEqual(text_violations("x.sh", '# on nax-macos-2\necho "runner=$RUNNER_NAME"  # nax-macos\n'), [])
        self.assertEqual(python_violations("scripts/ci/x.py",
                                           '"""Blocked forever on nax-macos-2."""\n'
                                           'record = {"runner": os.environ.get("RUNNER_NAME")}\n'), [])


if __name__ == "__main__":
    unittest.main()
