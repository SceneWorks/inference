"""Fail-closed tests for the sc-24129 decode-perf wrapper (`scripts/release/decode_bench.py`)."""

from __future__ import annotations

import copy
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
MODEL_KEY = "decode-bench-test-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "fedcba9876543210fedcba9876543210fedcba98"


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
            "device_used_bytes_at_last_token": 2**30,
            "cache_live_bytes": None,
            "cache_checkpoint_bytes": None,
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
            "host_syncs_per_verify_step": 1.0,
            "proposer": "mtp",
            "device_used_bytes_at_last_token": 3 * 2**29,
            "cache_live_bytes": None,
            "cache_checkpoint_bytes": None,
            "nvfp4_projections": {
                "switch": "on",
                "gemv": 30,
                "cublaslt": 2,
                "cublaslt_reason": "rows",
                "path": "mixed",
            },
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
                "device_used_bytes_at_last_token": 2**30,
                "cache_live_bytes": 3 * 2**20,
                "cache_checkpoint_bytes": 2**21,
                "fused_primitives": {
                    "switch": "on",
                    "fused": 12,
                    "reference": 0,
                    "reference_reason": None,
                    "path": "fused",
                },
                "nvfp4_projections": {
                    "switch": "on",
                    "gemv": 0,
                    "cublaslt": 0,
                    "cublaslt_reason": None,
                    "path": "none",
                },
                "cuda_graphs": {
                    "switch": "on",
                    "replayed": 0,
                    "eager": 4,
                    "captured": 0,
                    "fallback_reason": "deltanet_state_unstable",
                    "path": "eager",
                },
                "tokens_match_reference": True,
                "first_divergence": None,
                "tokens": list(range(new_tokens)),
            },
        )
    return {
        "schema_version": bench.SCHEMA_VERSION,
        "suite": "decode_bench",
        "label": bench.DEFAULT_LABEL,
        "compute_dtype": "BF16",
        "prompt_tokens": 40,
        "new_tokens": new_tokens,
        "rows": rows,
    }


def fake_binary(directory: Path, document: dict | None, exit_code: int = 0) -> Path:
    """An executable that writes `document` to `$DECODE_BENCH_OUTPUT` (or nothing) and exits."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / "payload.json"
    if document is not None:
        payload.write_text(json.dumps(document), encoding="utf-8")
    script = directory / "fake_bench.py"
    script.write_text(
        "import json, os, pathlib, sys\n"
        f"payload = pathlib.Path({str(payload)!r})\n"
        "if payload.exists():\n"
        "    pathlib.Path(os.environ['DECODE_BENCH_OUTPUT']).write_text(payload.read_text())\n"
        "print('rows', os.environ['DECODE_BENCH_ROWS'], 'drafts', os.environ['DECODE_BENCH_DRAFTS'], 'format', os.environ['DECODE_BENCH_FORMAT'])\n"
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


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def comparable_run(name: str, **suite: object) -> dict:
    doc = suite_document()
    doc.update(suite)
    return {
        "run_name": name,
        "label": bench.DEFAULT_LABEL,
        "model": {
            "key": MODEL_KEY,
            "repository": "Test/Decode-Model",
            "revision": REVISION,
            "config_sha256": "c" * 64,
        },
        "suite": doc,
    }


class DecodeBenchWrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix=f"decode-bench-{os.getpid()}-")
        self.root = Path(self.temp.name)
        # A pinned manifest entry and a standard HF-cache snapshot for it.
        self.manifest = self.root / "models.toml"
        self.manifest.write_text(
            "[[models]]\n"
            f'key = "{MODEL_KEY}"\n'
            'repository = "Test/Decode-Model"\n'
            f'revision = "{REVISION}"\n'
            'expected_files = ["config.json"]\n',
            encoding="utf-8",
        )
        self.snapshot = self.snapshot_dir(REVISION)
        self.saved_pins = dict(bench.PINNED_CONFIG_SHA256)
        bench.PINNED_CONFIG_SHA256[MODEL_KEY] = bench.sha256_file(self.snapshot / "config.json")
        self.checkout = self.root / "checkout"
        self.checkout.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.checkout, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=self.checkout, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=self.checkout, check=True)
        subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=self.checkout, check=True)
        (self.checkout / "a.txt").write_text("a", encoding="utf-8")
        subprocess.run(["git", "add", "a.txt"], cwd=self.checkout, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=self.checkout, check=True)
        self.sha = git(self.checkout, "rev-parse", "HEAD")

    def tearDown(self) -> None:
        bench.PINNED_CONFIG_SHA256.clear()
        bench.PINNED_CONFIG_SHA256.update(self.saved_pins)
        self.temp.cleanup()

    def snapshot_dir(self, revision: str, config: str = '{"model_type": "qwen3_5"}') -> Path:
        snapshot = self.root / "hub" / "models--Test--Decode-Model" / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text(config, encoding="utf-8")
        return snapshot

    def run_args(self, binary: Path, output: Path, **overrides: str) -> list[str]:
        args = [
            "run",
            "--binary", str(binary),
            "--run-name", "head-test",
            "--runtime-sha", self.sha,
            "--checkout", str(self.checkout),
            "--snapshot", str(self.snapshot),
            "--manifest", str(self.manifest),
            "--model-key", MODEL_KEY,
            "--output", str(output),
            "--sample-interval", "0.01",
        ]
        for key, value in overrides.items():
            args += [f"--{key.replace('_', '-')}", value]
        return args

    def test_default_pin_is_the_manifest_qwen38_entry(self) -> None:
        self.assertEqual(bench.DEFAULT_MODEL_KEY, "bonsai-qwen38-parent")
        model = bench.load_model(bench.DEFAULT_MANIFEST, bench.DEFAULT_MODEL_KEY)
        self.assertEqual(model["repository"], "Qwen/Qwen3.8-27B")
        self.assertIn(bench.DEFAULT_MODEL_KEY, self.saved_pins)

    def test_run_seals_suite_document_and_renders_table(self) -> None:
        binary = fake_binary(self.root / "bin", suite_document())
        output = self.root / "evidence"
        self.assertEqual(bench.main(self.run_args(binary, output)), 0)
        for name in ("decode_bench.json", "run.json", "decode_bench.md", "SEAL.json", "stdout.log", "stderr.log"):
            self.assertTrue((output / name).is_file(), name)
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["source"]["head_sha"], self.sha)
        self.assertTrue(record["source"]["clean_tree"])
        self.assertEqual(record["run_kind"], "head")
        self.assertIsNone(record["baseline_bench_source"])
        self.assertEqual(record["label"], bench.DEFAULT_LABEL)
        self.assertEqual(record["binary"]["sha256"], bench.sha256_file(binary))
        self.assertEqual(record["model"]["key"], MODEL_KEY)
        self.assertEqual(record["model"]["revision"], REVISION)
        self.assertEqual(record["model"]["config_sha256"], bench.PINNED_CONFIG_SHA256[MODEL_KEY])
        self.assertEqual(record["model"]["manifest_sha256"], bench.sha256_file(self.manifest))
        seal = json.loads((output / "SEAL.json").read_text(encoding="utf-8"))
        self.assertEqual(seal["decode_bench.json"], record["suite_document_sha256"])
        table = (output / "decode_bench.md").read_text(encoding="utf-8")
        self.assertIn(
            "| head-test | MTP off (reference) | 4 | (ref) | yes | 10.00 | n/a | 1.000 | n/a | n/a | 1.00 GiB | n/a | n/a | n/a | n/a |",
            table,
        )
        self.assertIn(
            "| head-test | MTP off (StepModel) | 4 | yes | yes | 10.50 | n/a | 1.000 | 1.00 | n/a | 1.00 GiB | 3.0 MiB | 2.0 MiB | on: 12 fused / 0 ref | none | on: 0 replayed / 4 eager, 0 captured (deltanet_state_unstable) |",
            table,
        )
        self.assertIn(
            "| head-test | MTP K=3 | 4 | no @2 | no @2 | 15.50 | 0.500 | 0.750 | 4.00 | 1.00 | 1.50 GiB | n/a | n/a | n/a | on: 30 gemv / 2 cuBLASLt (rows) | n/a |",
            table,
        )
        # The heading names the recorded model, not a literal.
        self.assertIn(f"**RTX Pro 6000 / sm_120** — Test/Decode-Model @ {REVISION[:12]}", table)
        self.assertIn(f"`{MODEL_KEY}`", table)
        self.assertIn("BF16 greedy, 40 prompt tokens, 4 new tokens per row", table)
        self.assertNotIn("Qwen3.8-27B", table)
        stdout = (output / "stdout.log").read_text(encoding="utf-8")
        self.assertIn("rows reference,step_model,mtp drafts 1,2,3,4,5 format bf16", stdout)
        # `table` re-verifies the seal and combines runs.
        combined = self.root / "combined.md"
        self.assertEqual(bench.main(["table", str(output), str(output), "--output", str(combined)]), 0)
        self.assertEqual(combined.read_text(encoding="utf-8").count("| head-test |"), 6)

    def test_run_refuses_a_snapshot_that_is_not_the_pinned_one_and_leaves_no_evidence(self) -> None:
        binary = fake_binary(self.root / "bin", suite_document())
        # Wrong revision directory.
        wrong_revision = self.snapshot_dir(OTHER_REVISION)
        output = self.root / "wrong-revision"
        with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
            bench.main(self.run_args(binary, output, snapshot=str(wrong_revision)))
        self.assertFalse(output.exists(), "a refused run must not leave an evidence directory")
        # Right revision directory, different config.json.
        (self.snapshot / "config.json").write_text('{"model_type": "other"}', encoding="utf-8")
        output = self.root / "wrong-config"
        with self.assertRaisesRegex(ValueError, "does not match the pinned"):
            bench.main(self.run_args(binary, output))
        self.assertFalse(output.exists())
        # A model key the bench has no config pin for.
        bench.PINNED_CONFIG_SHA256.pop(MODEL_KEY)
        with self.assertRaisesRegex(ValueError, "no pinned config sha256"):
            bench.main(self.run_args(binary, self.root / "unpinned"))
        self.assertFalse((self.root / "unpinned").exists())

    def test_run_refuses_existing_output_dirty_checkout_and_failed_binary(self) -> None:
        binary = fake_binary(self.root / "bin", suite_document())
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            bench.main(self.run_args(binary, output))

        (self.checkout / "untracked.rs").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dirty"):
            bench.main(self.run_args(binary, self.root / "dirty"))
        self.assertFalse((self.root / "dirty").exists())
        (self.checkout / "untracked.rs").unlink()

        with self.assertRaisesRegex(ValueError, "does not equal checkout HEAD"):
            bench.main(self.run_args(binary, self.root / "wrong-sha", runtime_sha="a" * 40))
        self.assertFalse((self.root / "wrong-sha").exists())

        failing = fake_binary(self.root / "fail", None, exit_code=3)
        with self.assertRaisesRegex(RuntimeError, "exited with 3"):
            bench.main(self.run_args(failing, self.root / "failed"))

        silent = fake_binary(self.root / "silent", None)
        with self.assertRaisesRegex(RuntimeError, "did not write"):
            bench.main(self.run_args(silent, self.root / "silent-out"))

    def test_baseline_run_accepts_only_the_exact_rewritten_bench_file(self) -> None:
        binary = fake_binary(self.root / "bin", suite_document(with_step=False))
        bench_file = self.checkout / bench.BENCH_SOURCE_PATH
        bench_file.parent.mkdir(parents=True)
        rewritten = bench.baseline_source_text(BENCH_SOURCE.read_text(encoding="utf-8"))
        bench_file.write_bytes(rewritten.encode("utf-8"))
        output = self.root / "baseline"
        self.assertEqual(
            bench.main(self.run_args(binary, output, run_name="baseline-test", baseline_of=str(BENCH_SOURCE))),
            0,
        )
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["run_kind"], "baseline")
        self.assertEqual(record["source"]["dirty_paths"], [f"?? {bench.BENCH_SOURCE_PATH}"])
        self.assertEqual(record["baseline_bench_source"]["sha256"], bench.sha256_file(bench_file))
        self.assertEqual(
            record["baseline_bench_source"]["head_source_sha256"], bench.sha256_file(BENCH_SOURCE)
        )

        # One byte off the rewrite is refused.
        bench_file.write_bytes(rewritten.encode("utf-8") + b"// edited\n")
        with self.assertRaisesRegex(ValueError, "is not the baseline-source rewrite"):
            bench.main(self.run_args(binary, self.root / "edited", baseline_of=str(BENCH_SOURCE)))
        self.assertFalse((self.root / "edited").exists())
        bench_file.write_bytes(rewritten.encode("utf-8"))

        # Any other dirty path is refused.
        (self.checkout / "a.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must differ from its commit only by"):
            bench.main(self.run_args(binary, self.root / "extra", baseline_of=str(BENCH_SOURCE)))
        self.assertFalse((self.root / "extra").exists())

    def test_run_validates_the_suite_document(self) -> None:
        short = suite_document()
        short["rows"][0]["generated_tokens"] = 3
        with self.assertRaisesRegex(ValueError, "generated 3 tokens"):
            bench.main(self.run_args(fake_binary(self.root / "short", short), self.root / "short-out"))
        wrong_label = suite_document()
        wrong_label["label"] = "somewhere else"
        with self.assertRaisesRegex(ValueError, "different hardware label"):
            bench.main(self.run_args(fake_binary(self.root / "label", wrong_label), self.root / "label-out"))
        with self.assertRaisesRegex(ValueError, "schema-2"):
            bench.validate_suite_document({"suite": "other"})
        no_checkpoints = suite_document()
        no_checkpoints["rows"][1]["cache_checkpoint_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "no rollback-checkpoint bytes"):
            bench.validate_suite_document(no_checkpoints)
        one_token = suite_document(new_tokens=1)
        one_token["rows"][1]["cache_checkpoint_bytes"] = 0
        bench.validate_suite_document(one_token)

    def test_table_refuses_runs_that_are_not_comparable(self) -> None:
        base = comparable_run("base")
        for field, value in (("prompt_tokens", 41), ("new_tokens", 5)):
            with self.assertRaisesRegex(ValueError, f"suite.{field}"):
                bench.render_table([base, comparable_run("other", **{field: value})])
        other_config = comparable_run("other")
        other_config["model"] = {**other_config["model"], "config_sha256": "d" * 64}
        with self.assertRaisesRegex(ValueError, "model.config_sha256"):
            bench.render_table([base, other_config])
        other_label = comparable_run("other")
        other_label["label"] = "elsewhere"
        with self.assertRaisesRegex(ValueError, "different hardware labels"):
            bench.render_table([base, other_label])
        no_reference = comparable_run("noref")
        no_reference["suite"]["rows"] = no_reference["suite"]["rows"][1:]
        with self.assertRaisesRegex(ValueError, "no reference row"):
            bench.render_table([no_reference, base])

    def test_table_refuses_merging_a_bf16_run_with_an_nvfp4_run(self) -> None:
        # `base` has no `weight_format` key at all (a pre-sc-24136 document, implicitly bf16).
        base = comparable_run("base")
        nvfp4 = comparable_run("nvfp4-run", weight_format="nvfp4")
        with self.assertRaisesRegex(ValueError, "suite.weight_format"):
            bench.render_table([base, nvfp4])
        # An explicit "bf16" document still merges with an older, key-less bf16 document.
        explicit_bf16 = comparable_run("explicit-bf16", weight_format="bf16")
        bench.render_table([base, explicit_bf16])

    def test_table_compares_every_row_with_the_first_runs_reference(self) -> None:
        base = comparable_run("baseline")
        head = copy.deepcopy(base)
        head["run_name"] = "head"
        # Head's own rows agree with each other, but its reference drifted from the baseline's.
        for row in head["suite"]["rows"]:
            if row["path"] != "mtp":
                row["tokens"] = [0, 1, 2, 7]
        text = bench.render_table([base, head])
        self.assertIn("| baseline | MTP off (reference) | 4 | (ref) | yes |", text)
        self.assertIn("| head | MTP off (reference) | 4 | (ref) | no @3 |", text)
        self.assertIn("| head | MTP off (StepModel) | 4 | yes | no @3 |", text)
        self.assertIn("| head | MTP K=3 | 4 | no @2 | no @2 |", text)

    def test_nvfp4_rows_are_labelled_and_the_heading_names_the_format(self) -> None:
        doc = suite_document()
        doc["weight_format"] = "nvfp4"
        doc["rows"].append(
            {
                **doc["rows"][0],
                "path": "reference_cublaslt",
                "tokens_match_reference": False,
                "first_divergence": 1,
                "nvfp4_projections": {
                    "switch": "off",
                    "gemv": 0,
                    "cublaslt": 64,
                    "cublaslt_reason": "disabled",
                    "path": "cublaslt",
                },
            }
        )
        run = {"run_name": "nv", "label": bench.DEFAULT_LABEL, "model": {
            "repository": "Test/Decode-Model", "revision": REVISION, "key": MODEL_KEY,
            "config_sha256": "0" * 64}, "suite": doc}
        table = bench.render_table([run])
        self.assertIn("BF16 with NVFP4 projections greedy", table)
        self.assertIn("| nv | MTP off (reference, NVFP4 GEMV off) |", table)
        self.assertIn("| off: 0 gemv / 64 cuBLASLt (disabled) |", table)

    def test_table_rejects_a_tampered_run(self) -> None:
        binary = fake_binary(self.root / "bin", suite_document())
        output = self.root / "sealed"
        self.assertEqual(bench.main(self.run_args(binary, output)), 0)
        doc_path = output / "decode_bench.json"
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        doc["rows"][0]["decode_tokens_per_second"] = 999.0
        doc_path.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match its seal"):
            bench.main(["table", str(output)])

    def test_row_label_names_the_kv_cache_and_attention_formulation_when_reported(self) -> None:
        # A pre-epic baseline binary reports neither: the labels are unchanged.
        self.assertEqual(bench.row_label({"path": "step_model"}), "MTP off (StepModel)")
        self.assertEqual(bench.row_label({"path": "reference"}), "MTP off (reference)")
        self.assertEqual(bench.row_label({"path": "mtp", "mtp_drafts": 2}), "MTP K=2")
        self.assertEqual(
            bench.row_label({"path": "step_model", "kv_cache": "static"}),
            "MTP off (StepModel, static kv)",
        )
        self.assertEqual(
            bench.row_label({"path": "step_model", "kv_cache": "static", "attn_formulation": "gqa"}),
            "MTP off (StepModel, static kv, gqa attn)",
        )
        self.assertEqual(
            bench.row_label({"path": "step_model", "kv_cache": "growing", "attn_formulation": "expanded"}),
            "MTP off (StepModel, growing kv, expanded attn)",
        )
        self.assertEqual(
            bench.row_label({"path": "reference", "kv_cache": "growing"}),
            "MTP off (reference, growing kv)",
        )
        self.assertEqual(
            bench.row_label({"path": "reference", "kv_cache": "growing", "attn_formulation": "expanded"}),
            "MTP off (reference, growing kv, expanded attn)",
        )
        self.assertEqual(
            bench.row_label({"path": "reference", "attn_formulation": "gqa"}),
            "MTP off (reference, gqa attn)",
        )
        self.assertEqual(
            bench.row_label({"path": "ngram", "drafts": 3, "kv_cache": "static", "attn_formulation": "gqa"}),
            "n-gram K=3 (static kv, gqa attn)",
        )
        self.assertEqual(bench.row_label({"path": "ngram", "drafts": 2}), "n-gram K=2")
        self.assertEqual(
            bench.row_label({"path": "mtp", "mtp_drafts": 3, "kv_cache": "growing", "attn_formulation": "gqa"}),
            "MTP K=3 (growing kv, gqa attn)",
        )

    def test_baseline_source_replaces_only_the_head_only_block(self) -> None:
        source = BENCH_SOURCE.read_text(encoding="utf-8")
        rewritten = bench.baseline_source_text(source)
        before_stub = rewritten.split(bench.STUB_BEGIN)[0]
        self.assertNotIn("generate_step_timed", before_stub)
        self.assertNotIn("host_sync_count", before_stub)
        self.assertNotIn("CountingDecode", before_stub)
        self.assertNotIn(bench.HEAD_ONLY_BEGIN, rewritten)
        self.assertIn("fn host_syncs_now() -> Option<u64> {\n    None\n}", rewritten)
        self.assertIn("(out, prefill, decode, None)", before_stub)
        self.assertIn('unreachable!("the step_model row is not available on the pre-epic baseline")', rewritten)
        self.assertIn("fn select_step_kv_cache(_model: &mut Qwen35Model) {}", rewritten)
        self.assertNotIn("set_step_kv_cache", before_stub)
        self.assertIn("fn select_attn_formulation(_model: &mut Qwen35Model) {}", rewritten)
        self.assertIn(
            "fn growing_row_kinds(_model: &Qwen35Model) -> Option<(&'static str, &'static str)> {\n    None\n}",
            rewritten,
        )
        # The speculative rows: the baseline keeps the pre-epic MTP loop, has no n-gram row and
        # reports no per-verify-step syncs; the engine never appears before the stub.
        self.assertIn("generate_qwen35_mtp_timed", rewritten)
        self.assertNotIn("generate_speculative_with", before_stub)
        self.assertNotIn("MtpProposer", before_stub)
        self.assertIn('unreachable!("the ngram row is not available on the pre-epic baseline")', rewritten)
        self.assertIn('(out, stats, prefill_secs, decode_secs, None, None, "mtp")', rewritten)
        self.assertIn("fn replay_forwards(_stats: &SpeculativeStats) -> Option<u64> {", rewritten)
        self.assertNotIn("Some(stats.replays as u64)", rewritten)
        self.assertNotIn("set_attn_formulation", before_stub)
        self.assertNotIn("attn_formulation()", before_stub)
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
