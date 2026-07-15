"""Guard the repository gates against locale-dependent text decoding.

The tools under scripts/ shell out to cargo and git and read repository files. Both emit
UTF-8 on every platform, but Python decodes with the locale encoding unless told otherwise
-- cp1252 on a default Windows install. `cargo metadata` output alone carries non-ASCII in
crate descriptions and dependency author names, so an implicit decode makes the documented
gates unrunnable on the Windows CUDA dev box while passing on UTF-8 Linux CI.

These are static checks rather than behavioural ones: they fail on the Linux runner that
cannot itself reproduce the Windows-only crash.
"""

import ast
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]

# `.open()` on these resolves to a module function taking a path first, not Path.open(mode).
MODULE_OPENERS = {"bz2", "codecs", "gzip", "io", "lzma", "shutil", "tarfile", "zipfile"}


def script_files() -> list[Path]:
    return sorted(SCRIPTS.rglob("*.py"))


def keyword(node: ast.Call, name: str) -> ast.keyword | None:
    return next((word for word in node.keywords if word.arg == name), None)


def literal_mode(node: ast.Call, index: int) -> str | None:
    word = keyword(node, "mode")
    if word is not None:
        candidate = word.value
    elif len(node.args) > index:
        candidate = node.args[index]
    else:
        return None
    return candidate.value if isinstance(candidate, ast.Constant) else None


def violations(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        has_encoding = keyword(node, "encoding") is not None

        if (
            isinstance(function, ast.Attribute)
            and function.attr == "run"
            and isinstance(function.value, ast.Name)
            and function.value.id == "subprocess"
        ):
            # Without text=True/universal_newlines the call returns bytes and never decodes.
            text_mode = keyword(node, "text") or keyword(node, "universal_newlines")
            if text_mode is not None and not has_encoding:
                found.append((node.lineno, "subprocess.run(text=...) without encoding="))
            continue

        if isinstance(function, ast.Attribute) and function.attr in {"read_text", "write_text"}:
            if not has_encoding:
                found.append((node.lineno, f"{function.attr}() without encoding="))
            continue

        if isinstance(function, ast.Name) and function.id == "open":
            mode = literal_mode(node, 1)
            if "b" not in (mode or "") and not has_encoding:
                found.append((node.lineno, "open() in text mode without encoding="))
            continue

        if isinstance(function, ast.Attribute) and function.attr == "open":
            if isinstance(function.value, ast.Name) and function.value.id in MODULE_OPENERS:
                continue
            mode = literal_mode(node, 0)
            if "b" not in (mode or "") and not has_encoding:
                found.append((node.lineno, "Path.open() in text mode without encoding="))
    return found


class ScriptEncodingTests(unittest.TestCase):
    def test_repository_tooling_never_relies_on_the_locale_encoding(self) -> None:
        offenders = []
        for path in script_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            offenders.extend(
                f"{path.relative_to(SCRIPTS.parent)}:{line}: {reason}"
                for line, reason in violations(tree)
            )
        self.assertEqual(
            offenders,
            [],
            "repository tooling must decode cargo/git output and repository files as UTF-8 "
            "explicitly; the locale default breaks these gates on Windows:\n"
            + "\n".join(offenders),
        )

    def test_detects_the_cargo_metadata_regression(self) -> None:
        # The exact shape that broke check-workspace.py on Windows/cp1252.
        tree = ast.parse(
            "import subprocess\n"
            "subprocess.run(cmd, capture_output=True, text=True)\n"
        )
        self.assertEqual(
            [reason for _, reason in violations(tree)],
            ["subprocess.run(text=...) without encoding="],
        )

    def test_allows_binary_and_explicitly_encoded_calls(self) -> None:
        tree = ast.parse(
            "import subprocess, tarfile\n"
            "subprocess.run(cmd, capture_output=True)\n"
            "subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')\n"
            "path.read_text(encoding='utf-8')\n"
            "path.open('rb')\n"
            "open(name, 'w', encoding='utf-8')\n"
            "tarfile.open(bundle / name, 'r:gz')\n"
        )
        self.assertEqual(violations(tree), [])
