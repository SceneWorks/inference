"""Fail-closed tests for the sc-24129 decode-perf wrapper (`scripts/release/decode_bench.py`)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "release" / "decode_bench.py"
SPEC = importlib.util.spec_from_file_location("decode_bench", SCRIPT)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

BENCH_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "llm"
    / "candle-llm"
    / "tests"
    / "decode_bench.rs"
)


def suite_document(new_tokens: int = 4, with_step: bool = True) -> dict:
    rows = [
        {
            "path": "reference",
            "mtp_drafts": None,
            "generated_tokens": new_tokens,
            "decode_tokens_per_second": 10.0,
            "acceptance_rate": None,
            "target_forwards_per_generated_token": 1.0,
            "host_syncs_per_token": None,
            "device_used_bytes_after": 2**30,
            "tokens_match_reference": None,
            "first_divergence": None,
            "tokens": list(range(new_tokens)),
        },
        {
            "path": "mtp",
            "mtp_drafts": 3,
            "generated_tokens": new_tokens,
            "decode_tokens_per_second": 15.5,
            "acceptance_rate": 0.5,
            "target_forwards_per_generated_token": 0.75,
            "host_syncs_per_token": 4.0,
            "device_used_bytes_after": 3 * 2**29,
            "tokens_match_reference": False,
            "first_divergence": 2,
            "tokens": [0, 1, 9, 3][:new_tokens],
        },
    ]
    if with_step:
        rows.insert(
            1,
            {
                "path": "step_model",
                "mtp_drafts": None,
                "generated_tokens": new_tokens,
                "decode_tokens_per_second": 10.5,
                "acceptance_rate": None,
                "target_forwards_per_generated_token": 1.0,
                "host_syncs_per_token": 1.0,
                "device_used_bytes_after": 2**30,
                "tokens_match_reference": True,
                "first_divergence": None,
                "tokens": list(range(new_tokens)),
            },
        )
    return {
        "schema_version": 1,
        "suite": "decode_bench",
        "label": bench.DEFAULT_LABEL,
        "prompt_tokens": 40,
        "new_tokens": new_tokens,
        "rows": rows,
    }


def fake_binary(directory: Path, document: dict | None, exit_code: int = 0) -> Path:
    """An executable that writes `document` to `$DECODE_BENCH_OUTPUT` (or nothing) and exits."""
    payload = directory / "payload.json"
    if document is not None:
        payload.write_text(json.dumps(document), encoding="utf-8")
    script = directory / "fake_bench.py"
    script.write_text(
        "import json, os, pathlib, sys\n"
        f"payload = pathlib.Path({str(payload)!r})\n"
        "if payload.exists():\n"
        "    pathlib.Path(os.environ['DECODE_BENCH_OUTPUT']).write_text(payload.read_text())\n"
        "print('rows', os.environ['DECODE_BENCH_ROWS'], 'drafts', os.environ['DECODE_BENCH_DRAFTS'])\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    if os.name == "nt":
        binary = directory / "fake_bench.cmd"
        binary.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        binary = directory / "fake_bench"
        binary.write_text(f"#!/bin/sh\nexec '{sys.executable}' '{script}' \"$@\"\n", encoding="utf-8")
        binary.chmod(0o755)
    return binary


class DecodeBenchWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.checkout, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=self.checkout, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=self.checkout, check=True)
        (self.checkout / "a.txt").write_text("a", encoding="utf-8")
        subprocess.run(["git", "add", "a.txt"], cwd=self.checkout, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=self.checkout, check=True)
        self.sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.checkout, text=True
        ).strip()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_args(self, binary: Path, output: Path, **overrides: str) -> list[str]:
        args = [
            "run",
            "--binary", str(binary),
            "--run-name", "head-test",
            "--runtime-sha", self.sha,
            "--checkout", str(self.checkout),
            "--snapshot", str(self.snapshot),
            "--output", str(output),
            "--sample-interval", "0.01",
        ]
        for key, value in overrides.items():
            args += [f"--{key.replace('_', '-')}", value]
        return args

    def test_run_seals_suite_document_and_renders_table(self) -> None:
        binary = fake_binary(self.root, suite_document())
        output = self.root / "evidence"
        self.assertEqual(bench.main(self.run_args(binary, output)), 0)
        for name in ("decode_bench.json", "run.json", "decode_bench.md", "SEAL.json", "stdout.log", "stderr.log"):
            self.assertTrue((output / name).is_file(), name)
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["source"]["head_sha"], self.sha)
        self.assertTrue(record["source"]["clean_tree"])
        self.assertEqual(record["label"], bench.DEFAULT_LABEL)
        self.assertEqual(record["binary"]["sha256"], bench.sha256_file(binary))
        seal = json.loads((output / "SEAL.json").read_text(encoding="utf-8"))
        self.assertEqual(seal["decode_bench.json"], record["suite_document_sha256"])
        table = (output / "decode_bench.md").read_text(encoding="utf-8")
        self.assertIn("| head-test | MTP off (reference) | 4 | (ref) | 10.00 | n/a | 1.000 | n/a | 1.00 GiB |", table)
        self.assertIn("| head-test | MTP off (StepModel) | 4 | yes | 10.50 | n/a | 1.000 | 1.00 | 1.00 GiB |", table)
        self.assertIn("| head-test | MTP K=3 | 4 | no @2 | 15.50 | 0.500 | 0.750 | 4.00 | 1.50 GiB |", table)
        self.assertIn("**RTX Pro 6000 / sm_120**", table)
        stdout = (output / "stdout.log").read_text(encoding="utf-8")
        self.assertIn("rows reference,step_model,mtp drafts 1,2,3,4,5", stdout)
        # `table` re-verifies the seal and combines runs.
        combined = self.root / "combined.md"
        self.assertEqual(bench.main(["table", str(output), str(output), "--output", str(combined)]), 0)
        self.assertEqual(combined.read_text(encoding="utf-8").count("| head-test |"), 6)

    def test_run_refuses_existing_output_dirty_checkout_and_failed_binary(self) -> None:
        binary = fake_binary(self.root, suite_document())
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            bench.main(self.run_args(binary, output))

        (self.checkout / "untracked.rs").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dirty"):
            bench.main(self.run_args(binary, self.root / "dirty"))
        self.assertEqual(bench.main(self.run_args(binary, self.root / "dirty-ok", allow_dirty="")[:-1]), 0)
        record = json.loads((self.root / "dirty-ok" / "run.json").read_text(encoding="utf-8"))
        self.assertFalse(record["source"]["clean_tree"])
        self.assertEqual(record["source"]["dirty_paths"], ["?? untracked.rs"])
        (self.checkout / "untracked.rs").unlink()

        with self.assertRaisesRegex(ValueError, "does not equal checkout HEAD"):
            bench.main(self.run_args(binary, self.root / "wrong-sha", runtime_sha="a" * 40))

        failing = fake_binary(self.root / "fail", None, exit_code=3) if (self.root / "fail").mkdir() is None else None
        with self.assertRaisesRegex(RuntimeError, "exited with 3"):
            bench.main(self.run_args(failing, self.root / "failed"))

        silent_dir = self.root / "silent"
        silent_dir.mkdir()
        silent = fake_binary(silent_dir, None)
        with self.assertRaisesRegex(RuntimeError, "did not write"):
            bench.main(self.run_args(silent, self.root / "silent-out"))

    def test_run_validates_the_suite_document(self) -> None:
        short = suite_document()
        short["rows"][0]["generated_tokens"] = 3
        short_dir = self.root / "short"
        short_dir.mkdir()
        with self.assertRaisesRegex(ValueError, "generated 3 tokens"):
            bench.main(self.run_args(fake_binary(short_dir, short), self.root / "short-out"))
        wrong_label = suite_document()
        wrong_label["label"] = "somewhere else"
        label_dir = self.root / "label"
        label_dir.mkdir()
        with self.assertRaisesRegex(ValueError, "different hardware label"):
            bench.main(self.run_args(fake_binary(label_dir, wrong_label), self.root / "label-out"))
        with self.assertRaisesRegex(ValueError, "schema-1"):
            bench.validate_suite_document({"suite": "other"})
        with self.assertRaisesRegex(ValueError, "different hardware labels"):
            bench.render_table([
                {"run_name": "a", "label": "x", "suite": suite_document()},
                {"run_name": "b", "label": "y", "suite": suite_document()},
            ])

    def test_table_rejects_a_tampered_run(self) -> None:
        binary = fake_binary(self.root, suite_document())
        output = self.root / "sealed"
        self.assertEqual(bench.main(self.run_args(binary, output)), 0)
        doc_path = output / "decode_bench.json"
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        doc["rows"][0]["decode_tokens_per_second"] = 999.0
        doc_path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match its seal"):
            bench.main(["table", str(output)])

    def test_baseline_source_replaces_only_the_head_only_block(self) -> None:
        source = BENCH_SOURCE.read_text(encoding="utf-8")
        rewritten = bench.baseline_source_text(source)
        self.assertNotIn("generate_step_timed", rewritten.split("BASELINE_STUB")[0])
        self.assertNotIn("host_sync_count", rewritten.split("BASELINE_STUB")[0])
        self.assertNotIn(bench.HEAD_ONLY_BEGIN, rewritten)
        self.assertIn("fn host_syncs_now() -> Option<u64> {\n    None\n}", rewritten)
        self.assertIn('unreachable!("the step_model row is not available on the pre-epic baseline")', rewritten)
        # Everything outside the block is untouched, so the two binaries measure the same rows.
        head_tail = source[source.index(bench.HEAD_ONLY_END) + len(bench.HEAD_ONLY_END):]
        self.assertTrue(rewritten.endswith(head_tail))
        self.assertTrue(rewritten.startswith(source[: source.index(bench.HEAD_ONLY_BEGIN)]))
        out = self.root / "baseline" / "decode_bench.rs"
        self.assertEqual(
            bench.main(["baseline-source", "--input", str(BENCH_SOURCE), "--output", str(out)]), 0
        )
        self.assertEqual(out.read_text(encoding="utf-8"), rewritten)
        with self.assertRaises(ValueError):
            bench.baseline_source_text(source.replace(bench.HEAD_ONLY_END, bench.HEAD_ONLY_END + bench.HEAD_ONLY_BEGIN + "x\n" + bench.HEAD_ONLY_END))


if __name__ == "__main__":
    unittest.main()
