"""One integration-test binary per crate (sc-21383) stays one binary per crate.

The hosted macOS lane linked 707 test binaries per run (666 per-file integration targets), each
statically linking the 187 MB libmlx.a; the link step dominated the lane and `target/` reached
35-38 GB. `scripts/ci/consolidate_integration_tests.py` collapses each crate's `tests/*.rs` into
a single `integration` target via a generated `tests/main.rs`. Two ways that silently regresses:

  * a new `tests/<name>.rs` is added without regenerating `tests/main.rs` -- with
    `autotests = false` cargo never compiles it, so the file is dead evidence. `check` fails on
    exactly that.
  * a new MLX-lane crate grows a `tests/` directory without being converted -- cargo goes back
    to one binary per file for it. This pins the MLX package set the macOS lane builds.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "consolidate_integration_tests.py"


def load_script():
    spec = importlib.util.spec_from_file_location("consolidate_integration_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def mlx_lane_crates() -> list[Path]:
    """Crates the `macos-metal` test step selects: mlx-llm*, mlx-gen, mlx-gen-*, runtime-macos."""
    out = []
    for manifest in ROOT.glob("crates/**/Cargo.toml"):
        if "target" in manifest.parts:
            continue
        name = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {}).get("name")
        if name and (name.startswith("mlx-") or name == "runtime-macos"):
            out.append(manifest.parent)
    return sorted(out)


class ConsolidatedTestTargetsTests(unittest.TestCase):
    def test_every_converted_crate_main_rs_is_current(self) -> None:
        errors = load_script().check_errors()
        self.assertEqual(errors, [], "\n".join(errors))

    def test_mlx_lane_crates_with_multiple_test_files_are_converted(self) -> None:
        unconverted = []
        for crate in mlx_lane_crates():
            files = [p for p in (crate / "tests").glob("*.rs") if p.name != "main.rs"]
            if len(files) < 2:
                continue
            manifest = crate / "Cargo.toml"
            text = manifest.read_text(encoding="utf-8")
            if "autotests = false" not in text or not (crate / "tests" / "main.rs").exists():
                unconverted.append(str(crate.relative_to(ROOT)))
        self.assertEqual(
            unconverted,
            [],
            "MLX-lane crates with several tests/*.rs files but no single integration target; run "
            "`python3 scripts/ci/consolidate_integration_tests.py convert <crate_dir>`: "
            + ", ".join(unconverted),
        )

    def test_converted_manifests_declare_exactly_the_integration_target(self) -> None:
        module = load_script()
        for crate in module.converted_crates():
            data = tomllib.loads((crate / "Cargo.toml").read_text(encoding="utf-8"))
            self.assertFalse(data["package"].get("autotests", True), crate)
            tests = data.get("test", [])
            self.assertEqual(
                [(t.get("name"), t.get("path")) for t in tests],
                [("integration", "tests/main.rs")],
                f"{crate.relative_to(ROOT)}: extra or missing [[test]] tables",
            )

    def test_generated_main_rs_declares_each_file_once_with_path(self) -> None:
        module = load_script()
        for crate in module.converted_crates():
            text = (crate / "tests" / "main.rs").read_text(encoding="utf-8")
            decls = re.findall(r'#\[path = "([a-z0-9_]+)\.rs"\]\nmod ([a-z0-9_]+);', text)
            self.assertEqual(len(decls), len(set(decls)), crate)
            for path, name in decls:
                self.assertEqual(path, name, f"{crate.relative_to(ROOT)}: {path} vs {name}")
                self.assertTrue((crate / "tests" / f"{path}.rs").exists(), f"{crate}: {path}.rs")

    def test_shared_common_is_declared_once_in_main_rs(self) -> None:
        # A `mod common;` inside a file that is now a `#[path]`-loaded submodule would resolve
        # to `tests/<file>/common.rs`, and a per-file `#[path = "common/mod.rs"]` copy trips
        # `clippy::duplicate_mod`. The generated main.rs declares `common` once; the files
        # import it.
        module = load_script()
        offenders = []
        for crate in module.converted_crates():
            main = (crate / "tests" / "main.rs").read_text(encoding="utf-8")
            has_common_dir = (crate / "tests" / "common" / "mod.rs").exists()
            self.assertEqual("\nmod common;\n" in main, has_common_dir, crate)
            for path in (crate / "tests").glob("*.rs"):
                if path.name == "main.rs":
                    continue
                for i, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
                    if re.match(r"^\s*(pub(\([a-z]+\))?\s+)?mod common;\s*$", line):
                        offenders.append(f"{path.relative_to(ROOT)}:{i + 1}: {line.strip()}")
        self.assertEqual(offenders, [], "use `use crate::common;` instead:\n" + "\n".join(offenders))

    def test_regen_preserves_the_hand_maintained_shared_block(self) -> None:
        module = load_script()
        flux2 = ROOT / "crates" / "media" / "mlx-gen" / "mlx-gen-flux2"
        main = (flux2 / "tests" / "main.rs").read_text(encoding="utf-8")
        self.assertIn('#[path = "../../tests/support/atomic_cache.rs"]', main)
        self.assertIn('#[path = "../../tests/support/atomic_cache.rs"]', module.render_main(flux2))


if __name__ == "__main__":
    unittest.main()
