import importlib.util
import unittest
from pathlib import Path

from scripts.bump_pins import (
    PIN_GROUPS,
    SHA_RE,
    BumpError,
    _dep_rev_pattern,
    _gate_rev_pattern,
    parse_ls_remote,
    read_group_rev,
    rewrite,
)


ROOT = Path(__file__).resolve().parents[2]

A = "a" * 40  # stand-in "current MLX" SHA
B = "b" * 40  # stand-in "current Candle" SHA
C = "c" * 40  # stand-in "new" SHA

MANIFEST = f"""[workspace.dependencies]
mlx-rs = {{ package = "pmetal-mlx-rs", git = "https://github.com/michaeltrefry/mlx-rs", rev = "{A}" }}
mlx-sys = {{ package = "pmetal-mlx-sys", git = "https://github.com/michaeltrefry/mlx-rs", rev = "{A}" }}

candle-core = {{ git = "https://github.com/huggingface/candle", rev = "{B}" }}
candle-nn = {{ git = "https://github.com/huggingface/candle", rev = "{B}" }}
candle-transformers = {{ git = "https://github.com/huggingface/candle", rev = "{B}" }}
candle-flash-attn = {{ git = "https://github.com/huggingface/candle", rev = "{B}" }}
"""

GATE = f"""PINNED_WORKSPACE_DEPENDENCIES = {{
    "mlx-rs": ("pmetal-mlx-rs", "{A}"),
    "mlx-sys": ("pmetal-mlx-sys", "{A}"),
    "candle-core": ("candle-core", "{B}"),
    "candle-nn": ("candle-nn", "{B}"),
    "candle-transformers": ("candle-transformers", "{B}"),
    "candle-flash-attn": ("candle-flash-attn", "{B}"),
}}
"""


def load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "check_workspace", ROOT / "scripts" / "check-workspace.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManifestRewriteTests(unittest.TestCase):
    def test_reads_the_single_group_rev(self) -> None:
        self.assertEqual(read_group_rev(MANIFEST, PIN_GROUPS["mlx"], patterns=_dep_rev_pattern), A)
        self.assertEqual(read_group_rev(MANIFEST, PIN_GROUPS["candle"], patterns=_dep_rev_pattern), B)

    def test_bumping_one_group_leaves_the_other_untouched(self) -> None:
        updated = rewrite(MANIFEST, PIN_GROUPS["candle"], B, C, patterns=_dep_rev_pattern)
        self.assertEqual(read_group_rev(updated, PIN_GROUPS["candle"], patterns=_dep_rev_pattern), C)
        self.assertEqual(read_group_rev(updated, PIN_GROUPS["mlx"], patterns=_dep_rev_pattern), A)
        # candle-flash-attn (CUDA-only, absent from the default graph) still bumps in the manifest.
        self.assertEqual(updated.count(C), 4)

    def test_mlx_bump_moves_both_alias_lines(self) -> None:
        updated = rewrite(MANIFEST, PIN_GROUPS["mlx"], A, C, patterns=_dep_rev_pattern)
        self.assertEqual(updated.count(C), 2)
        self.assertEqual(read_group_rev(updated, PIN_GROUPS["mlx"], patterns=_dep_rev_pattern), C)

    def test_inconsistent_group_pins_are_rejected(self) -> None:
        drifted = MANIFEST.replace(
            f'candle-nn = {{ git = "https://github.com/huggingface/candle", rev = "{B}" }}',
            f'candle-nn = {{ git = "https://github.com/huggingface/candle", rev = "{C}" }}',
        )
        with self.assertRaises(BumpError):
            read_group_rev(drifted, PIN_GROUPS["candle"], patterns=_dep_rev_pattern)

    def test_rewrite_rejects_an_unexpected_current_rev(self) -> None:
        with self.assertRaises(BumpError):
            rewrite(MANIFEST, PIN_GROUPS["candle"], A, C, patterns=_dep_rev_pattern)


class GateRewriteTests(unittest.TestCase):
    def test_reads_and_bumps_only_the_target_group(self) -> None:
        self.assertEqual(read_group_rev(GATE, PIN_GROUPS["candle"], patterns=_gate_rev_pattern), B)
        updated = rewrite(GATE, PIN_GROUPS["candle"], B, C, patterns=_gate_rev_pattern)
        self.assertEqual(read_group_rev(updated, PIN_GROUPS["candle"], patterns=_gate_rev_pattern), C)
        self.assertEqual(read_group_rev(updated, PIN_GROUPS["mlx"], patterns=_gate_rev_pattern), A)


class LsRemoteParsingTests(unittest.TestCase):
    def test_extracts_the_head_sha(self) -> None:
        self.assertEqual(parse_ls_remote(f"{C}\tHEAD\n"), C)

    def test_takes_the_first_ref_when_several_are_returned(self) -> None:
        self.assertEqual(parse_ls_remote(f"{C}\trefs/heads/main\n{A}\tHEAD\n"), C)

    def test_rejects_empty_and_non_sha_output(self) -> None:
        with self.assertRaises(BumpError):
            parse_ls_remote("")
        with self.assertRaises(BumpError):
            parse_ls_remote("not-a-sha\tHEAD\n")

    def test_only_full_lowercase_40_hex_is_a_sha(self) -> None:
        self.assertTrue(SHA_RE.match(C))
        self.assertIsNone(SHA_RE.match("abc123"))  # too short
        self.assertIsNone(SHA_RE.match(C.upper()))  # uppercase
        self.assertIsNone(SHA_RE.match("g" * 40))  # non-hex


class DriftGuardTests(unittest.TestCase):
    """The bump tool and the gate encode the same pin set in two files; keep them coupled."""

    def test_pin_groups_cover_exactly_what_the_gate_enforces(self) -> None:
        gate = load_gate_module()
        tool_deps = {dep for group in PIN_GROUPS.values() for dep in group.manifest_deps}
        self.assertEqual(tool_deps, set(gate.PINNED_WORKSPACE_DEPENDENCIES))

    def test_pin_groups_agree_with_the_gate_on_package_aliases(self) -> None:
        gate = load_gate_module()
        for group in PIN_GROUPS.values():
            for dep, package in zip(group.manifest_deps, group.lock_packages):
                gate_package, _revision = gate.PINNED_WORKSPACE_DEPENDENCIES[dep]
                self.assertEqual(gate_package, package, dep)


class LiveRepositoryTests(unittest.TestCase):
    """Exercise the regexes against the real files so a manifest reformat can't silently break them."""

    def test_real_manifest_and_gate_agree_on_every_pin(self) -> None:
        cargo_text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
        gate_text = (ROOT / "scripts" / "check-workspace.py").read_text(encoding="utf-8")
        for group in PIN_GROUPS.values():
            manifest_rev = read_group_rev(cargo_text, group, patterns=_dep_rev_pattern)
            gate_rev = read_group_rev(gate_text, group, patterns=_gate_rev_pattern)
            self.assertTrue(SHA_RE.match(manifest_rev), group.key)
            self.assertEqual(manifest_rev, gate_rev, group.key)


if __name__ == "__main__":
    unittest.main()
