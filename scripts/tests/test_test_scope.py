import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

from scripts.ci.test_scope import (
    ROOT,
    affected_packages,
    file_reads,
    is_model,
    load_workspace,
    lock_seeds,
    scope,
    select,
)


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def _manifest(name: str, deps: str = "") -> str:
    return f'[package]\nname = "{name}"\nversion = "0.1.0"\n{deps}'


ROOT_MANIFEST = """\
[workspace]
members = ["crates/core", "crates/gen", "crates/gen/mlx-gen-*", "crates/other"]
exclude = ["crates/gen/mlx-gen-vendored"]
[workspace.dependencies]
core = { path = "crates/core" }
serde = "1"
[patch."https://example.com/engine"]
engine = { path = "crates/vendor/engine" }
"""


def _lock(*packages: tuple[str, str, str, list[str]]) -> str:
    blocks = []
    for name, version, source, deps in packages:
        block = f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        if source:
            block += f'source = "{source}"\n'
        if deps:
            block += "dependencies = [\n" + "".join(f' "{d}",\n' for d in deps) + "]\n"
        blocks.append(block)
    return "version = 4\n\n" + "\n".join(blocks)


REG = "registry+https://github.com/rust-lang/crates.io-index"
GIT = "git+https://example.com/engine"
BASE_LOCK = _lock(
    ("core", "0.1.0", "", []),
    ("engine", "0.1.0", "", []),
    ("engine-nn", "0.1.0", GIT, ["engine"]),
    ("half", "2.0.0", REG, []),
    ("image", "0.25.0", REG, ["half"]),
    ("mlx-gen", "0.1.0", "", ["core", "engine", "zstd"]),
    ("mlx-gen-a", "0.1.0", "", ["mlx-gen", "image", "engine-nn"]),
    ("mlx-gen-b", "0.1.0", "", ["mlx-gen-a"]),
    ("mlx-gen-c", "0.1.0", "", ["mlx-gen"]),
    ("mlx-gen-catalog", "0.1.0", "", ["mlx-gen-a", "other"]),
    ("other", "0.1.0", "", []),
    ("zstd", "0.13.0", REG, []),
)


class FixtureWorkspace(unittest.TestCase):
    """A small workspace exercising every edge kind the scope follows.

    core <- mlx-gen <- mlx-gen-a <- mlx-gen-b (dev)      other <- mlx-gen-catalog
                    <- mlx-gen-c (target dep)          mlx-gen-a <- mlx-gen-catalog
    """

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        _write(self.root, "Cargo.toml", ROOT_MANIFEST)
        _write(self.root, "Cargo.lock", BASE_LOCK)
        _write(self.root, "crates/core/Cargo.toml", _manifest("core"))
        _write(self.root, "crates/core/src/lib.rs", "")
        _write(
            self.root,
            "crates/gen/Cargo.toml",
            _manifest("mlx-gen", "[dependencies]\ncore = { workspace = true }\nserde = { workspace = true }\n"),
        )
        _write(self.root, "crates/gen/src/lib.rs", "")
        _write(self.root, "crates/gen/assets/a.bin", "")
        _write(self.root, "crates/gen/mlx-gen-a/Cargo.toml", _manifest("mlx-gen-a", '[dependencies]\nmlx-gen = { path = ".." }\n'))
        _write(
            self.root,
            "crates/gen/mlx-gen-a/src/lib.rs",
            """\
            #[cfg(test)]
            mod tests {
                fn fixture() -> &'static str { "../../other/fixtures/x.json" }
                fn escape() -> &'static str { "../../../../etc/passwd" }
                fn rooted() -> &'static str { "docs/ref/table.json" }
            }
            """,
        )
        _write(self.root, "crates/gen/mlx-gen-b/Cargo.toml", _manifest("mlx-gen-b", '[dev-dependencies]\nmlx-gen-a = { path = "../mlx-gen-a" }\n'))
        _write(self.root, "crates/gen/mlx-gen-b/src/lib.rs", "")
        _write(self.root, "crates/gen/mlx-gen-c/Cargo.toml", _manifest("mlx-gen-c", '[target.\'cfg(unix)\'.dependencies]\nmlx-gen = { path = ".." }\n'))
        _write(self.root, "crates/gen/mlx-gen-c/src/lib.rs", "")
        _write(self.root, "crates/gen/mlx-gen-catalog/Cargo.toml", _manifest("mlx-gen-catalog", '[dependencies]\nmlx-gen-a = { path = "../mlx-gen-a" }\nother = { path = "../../other" }\n'))
        _write(self.root, "crates/gen/mlx-gen-catalog/src/lib.rs", "")
        _write(self.root, "crates/gen/mlx-gen-vendored/Cargo.toml", _manifest("mlx-gen-vendored"))
        _write(self.root, "crates/other/Cargo.toml", _manifest("other"))
        _write(self.root, "crates/other/src/lib.rs", 'pub const B: &[u8] = include_bytes!("../../gen/assets/a.bin");\n')
        _write(self.root, "crates/other/fixtures/x.json", "{}")
        _write(self.root, "crates/vendor/engine/src/lib.rs", "")
        _write(self.root, "crates/stray/notes.rs", "")
        _write(self.root, "docs/ref/table.json", "{}")
        self.ws = load_workspace(self.root)
        self.test_reads, self.lib_reads = file_reads(self.root, self.ws.members)
        self.base: dict[str, str] = {"Cargo.toml": ROOT_MANIFEST, "Cargo.lock": BASE_LOCK}

    def tearDown(self) -> None:
        self._dir.cleanup()

    def affected(self, *paths: str) -> set[str] | None:
        return affected_packages(
            paths,
            self.ws,
            self.test_reads,
            self.lib_reads,
            base_text=self.base.get,
            head_text=lambda path: (self.root / path).read_text(encoding="utf-8"),
        )

    def test_members_expand_globs_and_honor_exclude(self) -> None:
        self.assertEqual(
            set(self.ws.members.values()),
            {"core", "mlx-gen", "mlx-gen-a", "mlx-gen-b", "mlx-gen-c", "mlx-gen-catalog", "other"},
        )
        self.assertEqual(self.ws.patched, {"crates/vendor/engine": "engine"})

    def test_every_dependency_kind_is_an_edge(self) -> None:
        self.assertEqual(self.ws.deps["mlx-gen"], {"core"})  # workspace = true
        self.assertEqual(self.ws.deps["mlx-gen-a"], {"mlx-gen"})  # path
        self.assertEqual(self.ws.deps["mlx-gen-b"], {"mlx-gen-a"})  # dev-dependency
        self.assertEqual(self.ws.deps["mlx-gen-c"], {"mlx-gen"})  # target-specific

    def test_model_crates_exclude_composition_roots(self) -> None:
        self.assertTrue(is_model("mlx-gen-a") and is_model("candle-audio-yue"))
        self.assertFalse(is_model("mlx-gen") or is_model("candle-audio") or is_model("mlx-gen-catalog"))

    def test_a_shared_crate_edit_runs_shared_crates_but_no_model_behind_them(self) -> None:
        self.assertEqual(self.affected("crates/core/src/lib.rs"), {"core", "mlx-gen"})
        self.assertEqual(self.affected("crates/gen/src/lib.rs"), {"mlx-gen"})

    def test_a_model_edit_runs_dependent_models_and_composition_roots(self) -> None:
        self.assertEqual(
            self.affected("crates/gen/mlx-gen-a/src/lib.rs"), {"mlx-gen-a", "mlx-gen-b", "mlx-gen-catalog"}
        )
        self.assertEqual(self.affected("crates/gen/mlx-gen-b/src/lib.rs"), {"mlx-gen-b"})

    def test_nested_members_are_owned_by_the_innermost_crate(self) -> None:
        self.assertEqual(self.affected("crates/gen/mlx-gen-c/Cargo.toml"), {"mlx-gen-c"})

    def test_a_test_read_selects_only_its_reader(self) -> None:
        self.assertEqual(self.lib_reads["mlx-gen-a"], set())
        self.assertEqual(
            self.test_reads["mlx-gen-a"], {"crates/other/fixtures/x.json", "docs/ref/table.json"}
        )
        # mlx-gen-b depends on mlx-gen-a, but the fixture is an input to mlx-gen-a's tests only.
        self.assertEqual(
            self.affected("crates/other/fixtures/x.json"), {"other", "mlx-gen-catalog", "mlx-gen-a"}
        )

    def test_a_read_outside_every_crate_selects_its_reader(self) -> None:
        self.assertEqual(self.affected("docs/ref/table.json"), {"mlx-gen-a"})
        self.assertEqual(self.affected("docs/unread.md", "crates/stray/notes.rs"), set())

    def test_a_library_read_propagates_to_dependents(self) -> None:
        self.assertEqual(self.lib_reads["other"], {"crates/gen/assets/a.bin"})
        self.assertEqual(
            self.affected("crates/gen/assets/a.bin"), {"mlx-gen", "other", "mlx-gen-catalog"}
        )

    def test_a_traversal_string_that_names_no_file_is_not_a_read(self) -> None:
        self.assertNotIn("etc/passwd", " ".join(self.test_reads["mlx-gen-a"]))

    def test_a_lockfile_bump_selects_crates_that_resolve_it(self) -> None:
        head = BASE_LOCK.replace('name = "half"\nversion = "2.0.0"', 'name = "half"\nversion = "2.1.0"')
        _write(self.root, "Cargo.lock", head)
        # half <- image <- mlx-gen-a: a model whose own resolution changed runs, then its dependents.
        self.assertEqual(self.affected("Cargo.lock"), {"mlx-gen-a", "mlx-gen-b", "mlx-gen-catalog"})

    def test_a_lockfile_bump_under_a_shared_crate_keeps_the_model_boundary(self) -> None:
        head = BASE_LOCK.replace('name = "zstd"\nversion = "0.13.0"', 'name = "zstd"\nversion = "0.13.1"')
        _write(self.root, "Cargo.lock", head)
        # zstd <- mlx-gen only: the lock walk stops there, so no model behind mlx-gen is selected.
        self.assertEqual(self.affected("Cargo.lock"), {"mlx-gen"})

    def test_an_edit_to_a_patched_engine_crate_selects_no_model(self) -> None:
        # engine <- mlx-gen, and engine <- engine-nn (git) <- mlx-gen-a: an engine edit all the way.
        self.assertEqual(self.affected("crates/vendor/engine/src/lib.rs"), {"mlx-gen"})
        self.assertEqual(lock_seeds(BASE_LOCK, BASE_LOCK, ["engine"], {"mlx-gen", "mlx-gen-a"}), {"mlx-gen"})

    def test_root_manifest_edits(self) -> None:
        _write(self.root, "Cargo.toml", ROOT_MANIFEST.replace('"crates/other"]', '"crates/other", "crates/new"]'))
        self.ws = load_workspace(self.root)
        self.assertEqual(self.affected("Cargo.toml"), set())  # a member-list edit alone
        _write(self.root, "Cargo.toml", ROOT_MANIFEST.replace('serde = "1"', 'serde = "1.0.200"'))
        self.ws = load_workspace(self.root)
        self.assertEqual(self.affected("Cargo.toml"), {"mlx-gen"})  # inherits `serde`
        for edit in (
            ROOT_MANIFEST + "[profile.dev]\nopt-level = 1\n",
            ROOT_MANIFEST.replace("[workspace]\n", '[workspace]\nresolver = "3"\n'),
        ):
            _write(self.root, "Cargo.toml", edit)
            self.ws = load_workspace(self.root)
            self.assertIsNone(self.affected("Cargo.toml"), edit)

    def test_build_and_ci_wiring_paths_run_everything(self) -> None:
        for path in ("rust-toolchain.toml", ".cargo/config.toml", ".github/workflows/ci.yml", "scripts/ci/test_scope.py"):
            self.assertIsNone(self.affected("crates/core/src/lib.rs", path), path)
        self.assertIsNone(self.affected())

    def test_only_pull_requests_and_merge_groups_are_narrowed(self) -> None:
        path = ["crates/gen/mlx-gen-b/src/lib.rs"]
        self.assertEqual(scope("pull_request", path, self.root), "mlx-gen-b")
        self.assertEqual(scope("merge_group", path, self.root), "mlx-gen-b")
        for event in ("push", "workflow_dispatch", "schedule"):
            self.assertEqual(scope(event, path, self.root), "all", event)
        self.assertEqual(scope("pull_request", [], self.root), "all")
        self.assertEqual(scope("pull_request", ["docs/unread.md"], self.root), "none")


class SelectTests(unittest.TestCase):
    def test_all_keeps_the_steps_own_selectors(self) -> None:
        self.assertEqual(select("all", ["a", "gen-*"]), ["-p", "a", "-p", "gen-*"])
        self.assertEqual(select("", ["gen-*"]), ["-p", "gen-*"])

    def test_a_scope_is_cut_to_the_steps_package_set(self) -> None:
        self.assertEqual(select("gen-b other gen-a", ["gen-*"]), ["-p", "gen-a", "-p", "gen-b"])
        self.assertEqual(select("other", ["gen-*"]), [])
        self.assertEqual(select("none", ["gen-*"]), [])


class RealWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ws = load_workspace()
        cls.test_reads, cls.lib_reads = file_reads(ROOT, cls.ws.members)

    def affected(self, *paths: str) -> set[str] | None:
        return affected_packages(paths, self.ws, self.test_reads, self.lib_reads)

    def test_the_parser_agrees_with_cargo_metadata(self) -> None:
        metadata = json.loads(
            subprocess.run(
                ["cargo", "metadata", "--no-deps", "--format-version", "1"],
                cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
            ).stdout
        )
        names = {package["name"] for package in metadata["packages"]}
        self.assertEqual(set(self.ws.members.values()), names)
        for package in metadata["packages"]:
            expected = {dep["name"] for dep in package["dependencies"] if "path" in dep}
            self.assertEqual(self.ws.deps[package["name"]], expected, package["name"])

    def test_a_one_model_change_does_not_select_the_other_models(self) -> None:
        affected = self.affected("crates/media/candle-gen/candle-gen-krea/src/lib.rs")
        self.assertIn("candle-gen-krea", affected)
        self.assertIn("candle-gen-catalog", affected)
        self.assertIn("runtime-cpu", affected)
        self.assertNotIn("candle-gen-flux2", affected)
        self.assertFalse({name for name in affected if name.startswith("mlx-gen-")})

    def test_a_contract_change_runs_shared_crates_bundles_and_exact_source_readers(self) -> None:
        changed = "crates/contracts/gen-core/src/lib.rs"
        affected = self.affected(changed)
        shared = ("sceneworks-gen-core", "mlx-gen", "candle-gen", "mlx-gen-catalog", "runtime-macos", "runtime-cpu")
        for name in shared:
            self.assertIn(name, affected)
        # FLUX3's nn.rs no-DMMV source guard recursively reads every Rust file under crates/.
        # This is a genuine test-input edge, not propagation through the shared dependency graph.
        reader = "candle-gen-flux3"
        self.assertIn("crates", self.test_reads[reader])
        self.assertEqual({name for name in affected if is_model(name)}, {reader})

        # Remove ONLY that read edge on a copy: no model should remain selected, while the
        # contract/shared/catalog/bundle dependency route stays intact. Never mutate the cache
        # used by the other real-workspace tests.
        without_guard = {name: set(targets) for name, targets in self.test_reads.items()}
        without_guard[reader].remove("crates")
        without_read = affected_packages([changed], self.ws, without_guard, self.lib_reads)
        self.assertFalse({name for name in without_read if is_model(name)})
        for name in shared:
            self.assertIn(name, without_read)
        self.assertIn("crates", self.test_reads[reader])

    def test_a_cross_backend_file_read_selects_the_reader(self) -> None:
        # candle-gen-minimax-h3 checks its port against mlx-gen-minimax-h3's reference source.
        affected = self.affected("crates/media/mlx-gen/mlx-gen-minimax-h3/src/reference.rs")
        self.assertIn("candle-gen-minimax-h3", affected)

    def test_committed_reference_artifacts_select_their_reader(self) -> None:
        affected = self.affected("docs/migration/sa3-reference/same-s-same.safetensors")
        self.assertIn("candle-audio-stable-audio-3", affected)


class WorkflowWiringTests(unittest.TestCase):
    STEPS = {
        "candle-cpu-test": ("Test Candle packages", "Test Candle gen packages and integration targets"),
        "macos-metal": ("Test MLX packages",),
    }

    def setUp(self) -> None:
        self.workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))

    def test_the_changes_job_publishes_the_scope(self) -> None:
        changes = self.workflow["jobs"]["changes"]
        self.assertEqual(changes["outputs"]["test_scope"], "${{ steps.scope.outputs.test_scope }}")
        step = next(step for step in changes["steps"] if step.get("id") == "scope")
        self.assertIn("scripts/ci/test_scope.py scope", step["run"])
        self.assertEqual(step["env"]["EVENT_NAME"], "${{ github.event_name }}")

    def test_each_package_test_step_runs_only_its_scoped_packages(self) -> None:
        for job, names in self.STEPS.items():
            steps = {step.get("name"): step for step in self.workflow["jobs"][job]["steps"]}
            for name in names:
                with self.subTest(step=name):
                    step = steps[name]
                    self.assertEqual(step["env"]["TEST_SCOPE"], "${{ needs.changes.outputs.test_scope }}")
                    run = step["run"]
                    self.assertIn('scripts/ci/test_scope.py select --scope "$TEST_SCOPE"', run)
                    cargo = [line for line in run.splitlines() if "cargo test" in line]
                    self.assertEqual(len(cargo), 1, run)
                    self.assertTrue(cargo[0].rstrip().endswith('"${packages[@]}"'), cargo[0])
                    self.assertNotIn(" -p ", cargo[0])


if __name__ == "__main__":
    unittest.main()
