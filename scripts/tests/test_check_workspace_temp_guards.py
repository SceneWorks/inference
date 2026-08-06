"""Guardrail tests for the test-fixture temp-root gate (sc-17791).

The lint keeps the leak class sc-17704 / sc-17755 / sc-17768 fixed by hand from coming back: a
test that builds its fixture root from ``env::temp_dir()`` leaks that tree whenever it panics, and
collides with a sibling test that derived the same ``{prefix}{pid}`` path.

Both directions are covered, because a gate that only ever passes proves nothing. The scoping
decisions get their own rows too — they are what keep the real tree green, and each one is a
deliberate judgement rather than an accident of the regex:

* production ``src/`` is out of scope (a process-lifetime materialization is a different question);
* the ``env::var(..)``-then-fall-back shape stays legal (the artifact is the point of the test);
* ``tests/`` / ``benches/`` / ``examples/`` are test code in full, with no ``#[cfg(test)]`` marker;
* ``#[cfg(test)] mod name;`` makes a whole *other* file test code.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "check_workspace", ROOT / "scripts" / "check-workspace.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARDED = """
#[cfg(test)]
mod tests {
    #[test]
    fn writes_a_fixture() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.json"), b"{}").unwrap();
    }
}
"""

UNGUARDED = """
#[cfg(test)]
mod tests {
    #[test]
    fn writes_a_fixture() {
        let dir = std::env::temp_dir().join(format!("thing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
"""


class TempDirGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = load_gate_module()

    def run_gate(self, files: dict) -> None:
        """Run the lint over a synthetic crate tree; raises AssertionError on a violation."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for relative, body in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")
            self.gate.check_test_temp_dir_guards(root)

    def assert_flags(self, files: dict, *, needle: str = "env::temp_dir()") -> None:
        with self.assertRaises(AssertionError) as caught:
            self.run_gate(files)
        self.assertIn(needle, str(caught.exception))

    # --- the defect itself -----------------------------------------------------------------

    def test_a_guarded_fixture_passes(self) -> None:
        self.run_gate({"crates/x/src/lib.rs": GUARDED})

    def test_an_unguarded_cfg_test_fixture_is_flagged(self) -> None:
        self.assert_flags({"crates/x/src/lib.rs": UNGUARDED})

    def test_an_unguarded_integration_test_is_flagged(self) -> None:
        # `tests/` needs no `#[cfg(test)]` marker — the whole target is test code.
        self.assert_flags(
            {
                "crates/x/tests/roundtrip.rs": (
                    "fn snapshot() -> std::path::PathBuf {\n"
                    "    let dir = std::env::temp_dir().join(format!(\"snap-{}\", std::process::id()));\n"
                    "    std::fs::create_dir_all(&dir).unwrap();\n"
                    "    dir\n"
                    "}\n"
                )
            }
        )

    def test_a_cfg_test_file_module_is_test_code_too(self) -> None:
        # The `#[cfg(test)]` marker lives in lib.rs; the violation is in the file it names.
        self.assert_flags(
            {
                "crates/x/src/lib.rs": "#[cfg(test)]\nmod fixtures;\n",
                "crates/x/src/fixtures.rs": (
                    "fn root() -> std::path::PathBuf {\n"
                    "    let d = std::env::temp_dir().join(format!(\"f-{}\", std::process::id()));\n"
                    "    std::fs::create_dir_all(&d).unwrap();\n"
                    "    d\n"
                    "}\n"
                ),
            }
        )

    # --- the scoping decisions -------------------------------------------------------------

    def test_production_temp_use_is_out_of_scope(self) -> None:
        # A materialization that must outlive the process is a reviewed design choice, not a
        # fixture that escaped its test. `mlx-gen-seedvr2`'s bundled negative embedding is the
        # real instance.
        self.run_gate(
            {
                "crates/x/src/lib.rs": (
                    "fn materialize() -> std::path::PathBuf {\n"
                    "    let dir = std::env::temp_dir();\n"
                    "    let path = dir.join(\"bundled.safetensors\");\n"
                    "    std::fs::write(&path, b\"x\").unwrap();\n"
                    "    path\n"
                    "}\n"
                )
            }
        )

    def test_an_env_override_fallback_stays_legal(self) -> None:
        # The artifact is what the test exists to produce — a rendered WAV the author listens to.
        # Guarding it would delete the output.
        self.run_gate(
            {
                "crates/x/tests/render.rs": (
                    "#[test]\n"
                    "fn renders() {\n"
                    "    let out = std::env::var(\"X_WAV_OUT\")\n"
                    "        .map(std::path::PathBuf::from)\n"
                    "        .unwrap_or_else(|_| std::env::temp_dir().join(\"x.wav\"));\n"
                    "    std::fs::write(&out, b\"RIFF\").unwrap();\n"
                    "}\n"
                )
            }
        )

    def test_a_fallback_arm_stays_legal_when_the_override_read_is_elsewhere(self) -> None:
        # How the `preview_real_weights.rs` suites are written: a repo-local `env_path` helper
        # reads the override, so no `env::var` appears in this function — only the fallback arm.
        # An earlier revision of this lint flagged these and the sweep deleted the PNGs the suites
        # exist to produce, which is why the fallback position is matched on its own.
        self.run_gate(
            {
                "crates/x/tests/preview.rs": (
                    "fn artifact_dir() -> std::path::PathBuf {\n"
                    "    env_path(\"X_PREVIEW_ARTIFACT_DIR\")\n"
                    "        .unwrap_or_else(|| std::env::temp_dir().join(\"x_preview\"))\n"
                    "}\n"
                    "\n"
                    "#[test]\n"
                    "fn renders() {\n"
                    "    let dir = artifact_dir();\n"
                    "    std::fs::create_dir_all(&dir).unwrap();\n"
                    "}\n"
                )
            }
        )

    def test_a_stable_name_is_out_of_scope_and_this_is_deliberate(self) -> None:
        # Pins the gap the docstring names rather than leaving it to be rediscovered. A stable
        # name is bounded at one entry and is usually a deliberate artifact or cross-run cache
        # (`krea_turbo_smoke`, `mlx_gen_flux2_dev_prequant_q4`); nothing syntactic tells it apart
        # from a fixture root someone forgot to clean, so the lint stays on the unbounded class.
        # If this row ever starts failing, the scope was widened — update the docstring with it.
        self.run_gate(
            {
                "crates/x/tests/artifact.rs": (
                    "#[test]\n"
                    "fn writes_a_render() {\n"
                    "    let dir = std::env::temp_dir().join(\"x_turbo_smoke\");\n"
                    "    std::fs::create_dir_all(&dir).unwrap();\n"
                    "}\n"
                )
            }
        )

    def test_the_fallback_exemption_does_not_cover_a_plain_binding(self) -> None:
        # The exemption is positional, not file-wide: an `unwrap_or_else` elsewhere in the same
        # function must not launder an ordinary `let dir = env::temp_dir().join(..)` fixture root.
        self.assert_flags(
            {
                "crates/x/tests/mixed.rs": (
                    "#[test]\n"
                    "fn t() {\n"
                    "    let label = std::option_env!(\"L\").map(str::to_owned)\n"
                    "        .unwrap_or_else(|| \"default\".to_owned());\n"
                    "    let dir = std::env::temp_dir().join(format!(\"{label}-{}\", std::process::id()));\n"
                    "    std::fs::create_dir_all(&dir).unwrap();\n"
                    "}\n"
                )
            }
        )

    def test_a_sibling_function_guard_does_not_excuse_the_violation(self) -> None:
        # Function-scoped on purpose: a guard three functions away protects nothing here.
        self.assert_flags(
            {
                "crates/x/tests/two.rs": (
                    "#[test]\n"
                    "fn guarded() {\n"
                    "    let tmp = tempfile::tempdir().unwrap();\n"
                    "    std::fs::write(tmp.path().join(\"a\"), b\"a\").unwrap();\n"
                    "}\n"
                    "\n"
                    "#[test]\n"
                    "fn unguarded() {\n"
                    "    let dir = std::env::temp_dir().join(format!(\"b-{}\", std::process::id()));\n"
                    "    std::fs::create_dir_all(&dir).unwrap();\n"
                    "}\n"
                )
            }
        )

    def test_ignored_trees_are_skipped(self) -> None:
        # Agent worktrees under `.claude/` carry their own copy of the workspace.
        self.run_gate(
            {
                ".claude/worktrees/w/crates/x/tests/t.rs": (
                    "#[test]\n"
                    "fn t() {\n"
                    "    let d = std::env::temp_dir().join(\"z\");\n"
                    "    std::fs::create_dir_all(&d).unwrap();\n"
                    "}\n"
                )
            }
        )

    def test_the_real_workspace_is_clean(self) -> None:
        self.gate.check_test_temp_dir_guards(ROOT)


if __name__ == "__main__":
    unittest.main()
