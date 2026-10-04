"""Execute the phase selector without loading a model or touching a GPU."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "scripts/ci/real-weights/mlx-qwen-image-2-1/run-the-qwen-image-2-1-lora-real-weight-gates.sh"
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")


def shell_path(path):
    value = Path(path).as_posix()
    return re.sub(r"^([A-Za-z]):/", lambda match: "/" + match[1].lower() + "/", value)


@unittest.skipUnless(BASH and Path(BASH).is_file(), "requires bash")
class PhaseTests(unittest.TestCase):
    def run_phase(self, phase, fail=""):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cargo = root / "cargo"
            cargo.write_text('#!/bin/bash\n'
                'printf "%s|probe=%s|t2i=%s|edit=%s\\n" "$*" "$QWEN_IMAGE_2_1_PROBE_ONLY" '
                '"$QWEN_IMAGE_2_1_LORA_T2I_STEPS" "$QWEN_IMAGE_2_1_LORA_EDIT_STEPS" >> "$CARGO_CALLS"\n'
                'if [[ -n "$FAIL_TEST" && "$*" == *"$FAIL_TEST"* ]]; then exit 1; fi\n'
                'echo "test result: ok. 1 passed"\n', encoding="utf-8")
            cargo.chmod(0o755)
            calls = root / "calls"
            result = subprocess.run([BASH, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c",
                'export PATH="$FAKE_BIN:$PATH"; source ' + SCRIPT], cwd=ROOT,
                env={**os.environ, "FAKE_BIN": shell_path(root), "CARGO_CALLS": shell_path(calls),
                     "QWEN_IMAGE_2_1_RENDER_OUT": shell_path(root), "QWEN_IMAGE_2_1_LORA_PHASE": phase,
                     "QWEN_IMAGE_2_1_THIRD_PARTY_LORA": "fake.safetensors", "FAIL_TEST": fail},
                capture_output=True, text=True, encoding="utf-8", check=False)
            text = calls.read_text(encoding="utf-8") if calls.exists() else ""
            names = re.findall(r"lora_real_weights::(\w+)", text)
            return result, names, text

    def test_each_phase_has_exact_cells_and_probe_is_bounded(self):
        t2i = "t2i_lora_trains_reloads_and_moves_every_tier"
        edit = "edit_lokr_trains_on_two_references_and_moves_every_tier"
        stack = "stacked_adapters_apply_with_independent_weights"
        imports = "imported_adapters_move_t2i_and_two_reference_edit_every_tier"
        public = "third_party_lora_applies_strictly_and_moves_every_tier"
        expected = {"probe": [t2i, edit], "edit": [edit, stack, imports, public],
                    "imports": [imports, public], "full": [t2i, edit, stack, imports, public]}
        for phase, cells in expected.items():
            with self.subTest(phase=phase):
                result, names, text = self.run_phase(phase)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(names, cells)
                if phase == "probe":
                    self.assertEqual(text.count("probe=1|t2i=2|edit=2"), 2)

    def test_unknown_phase_runs_nothing_and_failed_cell_is_not_green(self):
        result, names, _ = self.run_phase("unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(names, [])
        result, names, _ = self.run_phase("edit", "stacked_adapters")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(names), 4, "retain evidence from independent cells after failure")
