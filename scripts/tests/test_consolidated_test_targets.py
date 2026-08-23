"""One integration-test binary per crate (sc-21383) stays one binary per crate.

The hosted macOS lane linked 707 test binaries per run (666 per-file integration targets), each
statically linking the 187 MB libmlx.a; the link step dominated the lane and `target/` reached
35-38 GB. `scripts/ci/consolidate_integration_tests.py` collapses each crate's `tests/*.rs` into
a single `integration` target via a generated `tests/main.rs`. Two ways that silently regresses:

  * a new `tests/<name>.rs` is added without regenerating `tests/main.rs` -- with
    `autotests = false` cargo never compiles it, so the file is dead evidence. `check` fails on
    exactly that.
  * a new MLX-lane or candle-gen* crate grows a `tests/` directory without being converted --
    cargo goes back to one binary per file for it. This pins both package sets.
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


def lane_crates() -> list[Path]:
    """Crates whose integration targets the consolidated lanes build: the `macos-metal` set
    (mlx-llm*, mlx-gen, mlx-gen-*, runtime-macos) and the `candle-cpu-test` / CUDA set
    (candle-gen, candle-gen-*)."""
    out = []
    for manifest in ROOT.glob("crates/**/Cargo.toml"):
        if "target" in manifest.parts:
            continue
        name = tomllib.loads(manifest.read_text(encoding="utf-8")).get("package", {}).get("name")
        if name and (name.startswith(("mlx-", "candle-gen")) or name == "runtime-macos"):
            out.append(manifest.parent)
    return sorted(out)


class ConsolidatedTestTargetsTests(unittest.TestCase):
    def test_every_converted_crate_main_rs_is_current(self) -> None:
        errors = load_script().check_errors()
        self.assertEqual(errors, [], "\n".join(errors))

    def test_lane_crates_with_multiple_test_files_are_converted(self) -> None:
        unconverted = []
        for crate in lane_crates():
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
            "MLX/candle-gen lane crates with several tests/*.rs files but no single integration target; run "
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

    def test_shared_dir_modules_are_declared_once_in_main_rs(self) -> None:
        # A `mod common;` inside a file that is now a `#[path]`-loaded submodule would resolve
        # to `tests/<file>/common.rs`, and a per-file `#[path = "common/mod.rs"]` copy trips
        # `clippy::duplicate_mod`. The generated main.rs declares `common` once; the files
        # import it.
        module = load_script()
        offenders = []
        for crate in module.converted_crates():
            main = (crate / "tests" / "main.rs").read_text(encoding="utf-8")
            shared = module.shared_dir_modules(crate)
            for name in shared:
                self.assertIn(f"\nmod {name};\n", main, f"{crate.relative_to(ROOT)}: {name}")
            for path in (crate / "tests").glob("*.rs"):
                if path.name == "main.rs":
                    continue
                for i, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
                    m = re.match(r"^\s*(pub(\([a-z]+\))?\s+)?mod ([a-z0-9_]+);\s*$", line)
                    if m and m.group(3) in shared:
                        offenders.append(f"{path.relative_to(ROOT)}:{i + 1}: {line.strip()}")
        self.assertEqual(offenders, [], "use `use crate::<dir>;` instead:\n" + "\n".join(offenders))

    def test_shared_dir_module_carries_the_cfg_all_its_users_carry(self) -> None:
        # When a shared module was declared by per-file `mod x;` inside `#![cfg(feature = "cuda")]`
        # files, the parents supplied the gate. Declared once at the root it compiles on every
        # lane unless it carries the gate itself (candle-gen's `rung4_support` broke the CPU
        # lanes exactly this way during sc-21386).
        module = load_script()
        file_cfg = re.compile(r"^#!\[cfg\((.*)\)\]\s*$", re.M)
        for crate in module.converted_crates():
            for shared in module.shared_dir_modules(crate):
                users = [
                    path
                    for path in (crate / "tests").glob("*.rs")
                    if path.name != "main.rs"
                    and re.search(rf"^\s*use crate::{shared}\b", path.read_text(encoding="utf-8"), re.M)
                ]
                if not users:
                    continue
                cfgs = {
                    (m.group(1) if (m := file_cfg.search(path.read_text(encoding="utf-8"))) else None)
                    for path in users
                }
                if len(cfgs) == 1 and None not in cfgs:
                    (cfg,) = cfgs
                    text = (crate / "tests" / shared / "mod.rs").read_text(encoding="utf-8")
                    self.assertIn(
                        f"#![cfg({cfg})]",
                        text,
                        f"{crate.relative_to(ROOT)}/tests/{shared}/mod.rs: every user is "
                        f"`#![cfg({cfg})]` but the shared module is not -- it compiles on lanes "
                        "where its users do not",
                    )

    def test_regen_preserves_the_hand_maintained_shared_block(self) -> None:
        module = load_script()
        flux2 = ROOT / "crates" / "media" / "mlx-gen" / "mlx-gen-flux2"
        main = (flux2 / "tests" / "main.rs").read_text(encoding="utf-8")
        self.assertIn('#[path = "../../tests/support/atomic_cache.rs"]', main)
        self.assertIn('#[path = "../../tests/support/atomic_cache.rs"]', module.render_main(flux2))


if __name__ == "__main__":
    unittest.main()
