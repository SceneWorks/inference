"""Guardrail tests for the epic-13657 self-fetch boundary (sc-13667).

These cover the three assertions `check-workspace.py` adds on top of the graph invariants:
a network/HTTP client ban, a whole-tree HF-cache source lint, and production-scoped pins on the
deleted env side channels. Each is exercised for BOTH directions -- green on a clean fixture, red on
each violation class -- and for the scoping decisions that keep the real tree green (test-side env
reuse, the `.cache/huggingface` specificity, `huggingface.co` attribution, doc-prose that merely
names a removed var).
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


def metadata_with(members):
    """Build a minimal cargo-metadata dict. `members` maps a member name -> list of dep names;
    every named dep is also emitted as a resolved package so the whole-graph check can see it."""
    packages = []
    workspace_members = []
    seen = set()

    def package(name, deps):
        pid = f"{name} 0.0.0 (path+file:///{name})"
        packages.append(
            {
                "id": pid,
                "name": name,
                "dependencies": [{"name": dep} for dep in deps],
            }
        )
        return pid

    for name, deps in members.items():
        workspace_members.append(package(name, deps))
        seen.add(name)
        for dep in deps:
            if dep not in seen:
                package(dep, [])
                seen.add(dep)
    return {"packages": packages, "workspace_members": workspace_members}


class NetworkClientBanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = load_gate_module()

    def test_denylist_covers_the_required_clients(self) -> None:
        required = {
            "hf-hub", "reqwest", "ureq", "curl", "curl-sys", "git2", "isahc", "attohttpc", "hyper",
        }
        self.assertTrue(required <= self.gate.FORBIDDEN_NETWORK_CLIENT_PACKAGES)

    def test_clean_graph_passes(self) -> None:
        meta = metadata_with({"runtime-cpu": ["candle-core", "core-llm"], "core-llm": []})
        self.gate.check_network_clients(meta)  # no raise

    def test_direct_member_dependency_fails(self) -> None:
        meta = metadata_with({"candle-gen-flux": ["hf-hub", "candle-core"]})
        with self.assertRaises(AssertionError) as ctx:
            self.gate.check_network_clients(meta)
        message = str(ctx.exception)
        self.assertIn("hf-hub", message)
        self.assertIn("candle-gen-flux", message)  # attributed to the reintroduction site

    def test_transitive_only_client_still_fails(self) -> None:
        # `reqwest` resolves in the graph but is not a direct member dep -- the whole-graph ban
        # (like the `inventory` precedent) still trips, and reports it as transitive.
        meta = metadata_with({"runtime-cpu": ["some-lib"], "some-lib": []})
        meta["packages"].append(
            {"id": "reqwest 0.12.0 (registry)", "name": "reqwest", "dependencies": []}
        )
        with self.assertRaises(AssertionError) as ctx:
            self.gate.check_network_clients(meta)
        self.assertIn("reqwest", str(ctx.exception))
        self.assertIn("transitive", str(ctx.exception))


class RustSourceLintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = load_gate_module()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, relative: str, body: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def check(self) -> None:
        self.gate.check_rust_sources(self.root)

    # --- clean fixture: every legitimate shape that must stay green -------------------------------

    def test_clean_fixture_passes(self) -> None:
        self.write(
            "crates/audio/candle-audio-x/src/model.rs",
            'pub const URL: &str = "https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.5";\n'
            "/// sc-13664: no `$LTX_UNCENSORED_GEMMA_DIR` / HF-cache scan any more; caller stages it.\n"
            'pub fn tokenizer() -> String { std::env::var("MLX_LLM_TEST_MODEL").unwrap() }\n'
            "#[cfg(test)]\n"
            "mod tests {\n"
            "    fn gemma() -> String {\n"
            '        std::env::var("LTX_GEMMA_DIR").unwrap_or_else(|_| panic!("set it"))\n'
            "    }\n"
            '    #[test]\n'
            '    fn deleted_var_is_not_resurrected() {\n'
            '        let err = String::new();\n'
            '        assert!(!err.contains("PULID_FLUX_WEIGHTS"));\n'
            "    }\n"
            "}\n",
        )
        self.write(
            "crates/media/mlx-gen/mlx-gen-seedvr2/tests/e2e_parity.rs",
            'let golden = home.join(".cache/mlx-gen-seedvr2-golden");\n'
            'let p = std::env::var("CHATTERBOX_PERTH_SNAPSHOT").unwrap();\n'
            'let q = std::env::var("MOSS_XY_TOKENIZER_SNAPSHOT").unwrap();\n',
        )
        self.check()  # no raise

    # --- whole-tree HF-cache bans (src, tests, examples alike) ------------------------------------

    def test_hf_home_read_in_src_fails(self) -> None:
        self.write("crates/llm/x/src/lib.rs", 'let _ = std::env::var("HF_HOME");\n')
        with self.assertRaisesRegex(AssertionError, "HF_HOME"):
            self.check()

    def test_cache_huggingface_literal_in_tests_fails(self) -> None:
        self.write("crates/llm/x/tests/conf.rs", 'let d = home.join(".cache/huggingface/hub");\n')
        with self.assertRaisesRegex(AssertionError, r"\.cache/huggingface"):
            self.check()

    def test_hf_hub_path_in_src_fails(self) -> None:
        self.write("crates/llm/x/src/lib.rs", "use hf_hub::api::sync::Api;\n")
        with self.assertRaisesRegex(AssertionError, "hf_hub"):
            self.check()

    def test_api_new_in_src_fails(self) -> None:
        self.write("crates/llm/x/src/lib.rs", "let api = Api::new().unwrap();\n")
        with self.assertRaisesRegex(AssertionError, "Api::new"):
            self.check()

    def test_hf_hub_cache_env_name_fails(self) -> None:
        self.write("crates/llm/x/src/lib.rs", 'let _ = std::env::var("HF_HUB_CACHE");\n')
        with self.assertRaisesRegex(AssertionError, "HF_HUB_CACHE"):
            self.check()

    # --- deleted env side channels: production reads red, test-side / prose green -----------------

    def test_production_perth_read_fails(self) -> None:
        self.write(
            "crates/audio/candle-audio-chatterbox/src/perth.rs",
            'pub fn perth() -> String { std::env::var("PERTH_SNAPSHOT").unwrap() }\n',
        )
        with self.assertRaisesRegex(AssertionError, "PERTH_SNAPSHOT"):
            self.check()

    def test_cfg_test_env_read_is_allowed(self) -> None:
        self.write(
            "crates/media/mlx-gen/mlx-gen-ltx/src/training.rs",
            "#[cfg(test)]\n"
            "mod first_step_repro {\n"
            '    fn gemma() -> String { std::env::var("LTX_GEMMA_DIR").unwrap() }\n'
            "}\n",
        )
        self.check()  # no raise -- the read lives inside a #[cfg(test)] module

    def test_prefixed_test_var_is_allowed(self) -> None:
        # CHATTERBOX_PERTH_SNAPSHOT contains PERTH_SNAPSHOT as a substring but is a distinct,
        # explicit passed-in test path -- the pin matches only the exact quoted env name.
        self.write(
            "crates/audio/candle-audio-chatterbox/src/model.rs",
            'pub fn p() -> String { std::env::var("CHATTERBOX_PERTH_SNAPSHOT").unwrap() }\n',
        )
        self.check()  # no raise

    def test_doc_prose_naming_removed_var_is_allowed(self) -> None:
        # mlx-gen-ltx/src/model.rs documents the removal of $LTX_UNCENSORED_GEMMA_DIR in prose;
        # only actual env::var reads are pinned, so prose stays green.
        self.write(
            "crates/media/mlx-gen/mlx-gen-ltx/src/model.rs",
            "/// generate-time error (no `$LTX_UNCENSORED_GEMMA_DIR` / HF-cache scan any more).\n"
            "pub fn text_encoder() {}\n",
        )
        self.check()  # no raise

    def test_contains_assertion_naming_removed_var_is_allowed(self) -> None:
        # A production `#[cfg(test)]` assertion may reference the deleted name via `.contains(...)`
        # to prove the error message does not resurrect it -- that is not an env::var read.
        self.write(
            "crates/media/mlx-gen/mlx-gen-sensenova/src/model.rs",
            "#[cfg(test)]\n"
            "mod tests {\n"
            '    #[test]\n'
            '    fn no_side_channel() {\n'
            '        let err = String::from("distill_lora");\n'
            '        assert!(!err.contains("SENSENOVA_DISTILL_LORA"));\n'
            "    }\n"
            "}\n",
        )
        self.check()  # no raise


class RepositoryTreeIsCleanTests(unittest.TestCase):
    """The real inference tree must satisfy the guardrail (green on the committed clean tree)."""

    def test_workspace_rust_has_no_hf_cache_or_side_channel(self) -> None:
        gate = load_gate_module()
        gate.check_rust_sources(ROOT)  # no raise on the clean tree


if __name__ == "__main__":
    unittest.main()
