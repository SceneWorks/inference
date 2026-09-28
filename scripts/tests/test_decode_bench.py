"""Fail-closed tests for the sc-24129 decode-perf wrapper (`scripts/release/decode_bench.py`)."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import shutil
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
# The device an RTX Pro 6000 / sm_120 run probes (sc-24140 feature-end review).
PROBED_DEVICE = {
    "device": "cuda",
    "device_name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "compute_capability": "12.0",
}
# The checkout a fake binary reports as its build provenance, when the document names none.
CHECKOUT_ENV = "DECODE_BENCH_TEST_CHECKOUT"
# What a fake binary adds to a document that lacks it: the build provenance of the checkout in
# `CHECKOUT_ENV` (as `CANDLE_LLM_BUILD_PROVENANCE=1` embeds it) and the probed device.
FILL_IDENTITY = r"""
def fill_identity(doc):
    import subprocess
    checkout = os.environ.get('DECODE_BENCH_TEST_CHECKOUT')
    if 'build' not in doc and checkout:
        sha = subprocess.check_output(['git', '-C', checkout, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', checkout, 'status', '--porcelain'], text=True).strip())
        doc['build'] = {'git_sha': sha, 'git_dirty': dirty}
    doc.setdefault('device', 'cuda')
    doc.setdefault('device_name', 'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition')
    doc.setdefault('compute_capability', '12.0')
    doc.setdefault('prompt_token_ids', list(range(doc.get('prompt_tokens', 0))))
    return doc
"""


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
            "verify_steps": 3,
            "direct_rollbacks": 2,
            "replay_forwards": 0,
            "target_forwards_per_verify_step": 1.0,
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
        **PROBED_DEVICE,
        "compute_dtype": "BF16",
        "prompt_tokens": 40,
        "prompt_token_ids": list(range(40)),
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
        + FILL_IDENTITY
        + f"payload = pathlib.Path({str(payload)!r})\n"
        "if payload.exists():\n"
        "    doc = fill_identity(json.loads(payload.read_text()))\n"
        "    pathlib.Path(os.environ['DECODE_BENCH_OUTPUT']).write_text(json.dumps(doc))\n"
        "print('rows', os.environ['DECODE_BENCH_ROWS'], 'drafts', os.environ['DECODE_BENCH_DRAFTS'], 'format', os.environ['DECODE_BENCH_FORMAT'])\n"
        "print('graphs', os.environ.get('CANDLE_LLM_CUDA_GRAPHS'), 'ngram', os.environ['DECODE_BENCH_NGRAM_DRAFTS'], 'sampling', os.environ['DECODE_BENCH_SAMPLING'])\n"
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


MATRIX_BENCH = r"""
import json, os, pathlib, sys
snapshot = pathlib.Path(os.environ['DECODE_BENCH_SNAPSHOT'])
config = json.loads((snapshot / 'config.json').read_text())
llama = 'qwen3_5' not in config.get('model_type', '')
rows = [r for r in os.environ['DECODE_BENCH_ROWS'].split(',') if r]
new_tokens = int(os.environ['DECODE_BENCH_NEW_TOKENS'])
graphs = 'on' if os.environ.get('CANDLE_LLM_CUDA_GRAPHS') == '1' else 'off'
temperature, top_p, seed = os.environ['DECODE_BENCH_SAMPLING'].split(',')
def row(path, **extra):
    base = {
        'path': path, 'generated_tokens': new_tokens, 'decode_tokens_per_second': 10.0,
        'tokens': list(range(new_tokens)), 'tokens_match_reference': None if path in ('reference', 'sampled') else True,
        'first_divergence': None, 'cache_checkpoint_bytes': 0 if llama else 1024,
        'sampler': {'path': 'device', 'device_draws': new_tokens, 'host_draws': 0,
                    'logits_to_host': 0, 'logits_to_host_per_token': 0.0},
    }
    if path.startswith('sampled'):
        base['sampling'] = {'temperature': float(temperature), 'top_p': float(top_p), 'seed': int(seed)}
    base.update(extra)
    return base
out = []
for r in rows:
    if r == 'mtp':
        if llama:
            sys.exit('the llama family has no MTP head')
        out += [row('mtp', mtp_drafts=int(k), drafts=int(k)) for k in os.environ['DECODE_BENCH_DRAFTS'].split(',')]
    elif r == 'ngram':
        out += [row('ngram', drafts=int(k)) for k in os.environ['DECODE_BENCH_NGRAM_DRAFTS'].split(',')]
    else:
        out.append(row(r))
doc = {
    'schema_version': 2, 'suite': 'decode_bench', 'label': os.environ['DECODE_BENCH_LABEL'],
    'compute_dtype': 'BF16', 'prompt_tokens': 40, 'new_tokens': new_tokens, 'rows': out,
    'weight_format': os.environ['DECODE_BENCH_FORMAT'], 'cuda_graphs': graphs,
}
if llama:
    doc['model_family'] = 'llama'
doc['sampling'] = {'temperature': float(temperature), 'top_p': float(top_p), 'top_k': 0, 'seed': int(seed)}
pathlib.Path(os.environ['DECODE_BENCH_OUTPUT']).write_text(json.dumps(fill_identity(doc)))
print('rows', os.environ['DECODE_BENCH_ROWS'], 'format', os.environ['DECODE_BENCH_FORMAT'], 'graphs', graphs)
"""


def matrix_binary(directory: Path) -> Path:
    """A fake bench that emits the rows it is asked for, in the format and graph switch it runs
    under, and refuses `mtp` on a llama-family snapshot as the real one does."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "matrix_bench.py"
    script.write_text("import os\n" + FILL_IDENTITY + MATRIX_BENCH, encoding="utf-8")
    if os.name == "nt":
        binary = directory / "matrix_bench.cmd"
        binary.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        binary = directory / "matrix_bench"
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
        # The fake binaries report this checkout as the tree they were built from.
        self.saved_checkout_env = os.environ.get(CHECKOUT_ENV)
        os.environ[CHECKOUT_ENV] = str(self.checkout)

    def tearDown(self) -> None:
        bench.PINNED_CONFIG_SHA256.clear()
        bench.PINNED_CONFIG_SHA256.update(self.saved_pins)
        if self.saved_checkout_env is None:
            os.environ.pop(CHECKOUT_ENV, None)
        else:
            os.environ[CHECKOUT_ENV] = self.saved_checkout_env
        self.temp.cleanup()

    def reseal(self, source: Path, destination: Path, record: dict | None = None, doc: dict | None = None) -> Path:
        """Copy a sealed run to `destination` with its `run.json` / `decode_bench.json` fields
        updated, and seal the copy again (a hand-made evidence directory)."""
        shutil.copytree(source, destination)
        for name, updates in (("run.json", record), ("decode_bench.json", doc)):
            if updates:
                value = json.loads((destination / name).read_text(encoding="utf-8"))
                value.update(updates)
                (destination / name).write_text(json.dumps(value), encoding="utf-8")
        run_record = json.loads((destination / "run.json").read_text(encoding="utf-8"))
        run_record["suite_document_sha256"] = bench.sha256_file(destination / "decode_bench.json")
        (destination / "run.json").write_text(json.dumps(run_record), encoding="utf-8")
        (destination / "SEAL.json").write_text(
            json.dumps(
                {
                    n: bench.sha256_file(destination / n)
                    for n in ("decode_bench.json", "run.json", "decode_bench.md", "stdout.log", "stderr.log")
                }
            ),
            encoding="utf-8",
        )
        return destination

    def as_baseline(self, head: Path, destination: Path) -> Path:
        """A sealed S1 baseline made from a sealed head run: a `baseline` run of another commit
        whose binary embeds no provenance, as the pre-epic build does."""
        record = json.loads((head / "run.json").read_text(encoding="utf-8"))
        return self.reseal(
            head,
            destination,
            record={
                "run_kind": "baseline",
                "run_name": f"baseline-{record['run_name']}",
                "source": {**record["source"], "head_sha": "b" * 40},
            },
            doc={"build": {"git_sha": None, "git_dirty": None}},
        )

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
        # A head run's own table has no S1 baseline: its rows say so (sc-24140 feature-end review).
        self.assertIn(
            "| head-test | bf16 | n/a | MTP off (reference) | 4 | (ref) | yes (vs head-test ref, no S1 baseline) | 10.00 | n/a | 1.000 | n/a | n/a | n/a | n/a | 1.00 GiB | n/a | n/a | n/a | n/a | n/a | n/a |",
            table,
        )
        self.assertIn(
            "| head-test | bf16 | n/a | MTP off (StepModel) | 4 | yes | yes (vs head-test ref, no S1 baseline) | 10.50 | n/a | 1.000 | 1.00 | n/a | n/a | n/a | 1.00 GiB | 3.0 MiB | 2.0 MiB | on: 12 fused / 0 ref | none | on: 0 replayed / 4 eager, 0 captured (deltanet_state_unstable) | n/a |",
            table,
        )
        self.assertIn(
            "| head-test | bf16 | n/a | MTP K=3 | 4 | no @2 | no @2 (vs head-test ref, no S1 baseline) | 15.50 | 0.500 | 0.750 | 4.00 | 1.00 | 1.00 | 0 | 1.50 GiB | n/a | n/a | n/a | on: 30 gemv / 2 cuBLASLt (rows) | n/a | n/a |",
            table,
        )
        # sc-24140 feature-end review: the record carries the binary's provenance, the probed
        # device and the prompt's hash.
        self.assertEqual(record["binary"]["build"], {"git_sha": self.sha, "git_dirty": False})
        self.assertEqual(record["probed_device"]["compute_capability"], "12.0")
        self.assertEqual(record["prompt"]["token_ids_sha256"], bench.prompt_sha256(suite_document()))
        # The heading names the recorded model, not a literal.
        self.assertIn(f"**RTX Pro 6000 / sm_120** — Test/Decode-Model @ {REVISION[:12]}", table)
        self.assertIn(f"`{MODEL_KEY}`", table)
        self.assertIn("BF16 greedy, 40 prompt tokens, 4 new tokens per row", table)
        self.assertNotIn("Qwen3.8-27B", table)
        stdout = (output / "stdout.log").read_text(encoding="utf-8")
        self.assertIn("rows reference,step_model,mtp drafts 1,2,3,4,5 format bf16", stdout)
        # No `--cuda-graphs`: the switch is inherited, not forced; the default knobs reach the binary.
        self.assertIn("ngram 3 sampling 0.7,0.9,0", stdout)
        self.assertEqual(record["requested"]["weight_format"], "bf16")
        self.assertIsNone(record["requested"]["cuda_graphs"])
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
        # sc-24138: a llama-family document's step cache keeps no checkpoints by design.
        llama = suite_document()
        llama["model_family"] = "llama"
        llama["rows"][1]["cache_checkpoint_bytes"] = 0
        bench.validate_suite_document(llama)
        # sc-24140: the hybrid's sampled step-seam row holds a checkpoint too.
        sampled_step = suite_document()
        sampled_step["rows"][1]["path"] = "sampled_step_model"
        sampled_step["rows"][1]["cache_checkpoint_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "sampled_step_model row reports no rollback-checkpoint"):
            bench.validate_suite_document(sampled_step)

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

    def test_weight_format_is_a_row_dimension_compared_per_format(self) -> None:
        # sc-24140: runs of different formats merge; each row compares with its own format's
        # baseline, and a non-bf16 row without one compares with its head's bf16 reference row.
        baseline = comparable_run("baseline")  # no `weight_format`: a pre-sc-24136 bf16 document
        baseline["run_kind"] = "baseline"
        baseline["source"] = {"head_sha": "b" * 40}
        head_bf16 = comparable_run("head-bf16", weight_format="bf16", cuda_graphs="off")
        head_bf16["source"] = {"head_sha": "a" * 40}
        for row in head_bf16["suite"]["rows"]:
            row["tokens"] = [0, 1, 2, 7] if row["path"] != "mtp" else row["tokens"]
        head_q8 = comparable_run("head-q8", weight_format="q8", cuda_graphs="on")
        head_q8["source"] = {"head_sha": "a" * 40}
        for row in head_q8["suite"]["rows"]:
            row["tokens"] = [0, 1, 5, 7] if row["path"] != "mtp" else row["tokens"]
        # An nvfp4 run from another commit has no bf16 reference of its own in the table.
        other_nvfp4 = comparable_run("other-nvfp4", weight_format="nvfp4")
        other_nvfp4["source"] = {"head_sha": "c" * 40}
        text = bench.render_table([baseline, head_bf16, head_q8, other_nvfp4])
        self.assertIn("with BF16 / Q8 / NVFP4 projections (the format column)", text)
        # bf16 rows: against the bf16 baseline, unlabelled.
        self.assertIn("| head-bf16 | bf16 | off | MTP off (reference) | 4 | (ref) | no @3 |", text)
        # q8 rows: no q8 baseline, so the head's own bf16 reference row, labelled.
        self.assertIn(
            "| head-q8 | q8 | on | MTP off (reference) | 4 | (ref) | no @2 (vs head-bf16 bf16 ref) |",
            text,
        )
        self.assertIn(
            "| head-q8 | q8 | on | MTP off (StepModel) | 4 | yes | no @2 (vs head-bf16 bf16 ref) |",
            text,
        )
        self.assertIn(
            "| other-nvfp4 | nvfp4 | n/a | MTP off (reference) | 4 | (ref) | n/a (no bf16 ref) |",
            text,
        )
        # A q8 baseline, when present, is the q8 rows' basis instead.
        q8_baseline = comparable_run("q8-baseline", weight_format="q8")
        q8_baseline["run_kind"] = "baseline"
        text = bench.render_table([baseline, q8_baseline, head_bf16, head_q8])
        self.assertIn("| head-q8 | q8 | on | MTP off (reference) | 4 | (ref) | no @2 |", text)
        self.assertNotIn("(vs head-bf16 bf16 ref)", text)
        # The basis selection itself.
        self.assertEqual(bench.comparison_basis([baseline, head_q8], head_q8, "greedy"), None)
        label, tokens = bench.comparison_basis([baseline, head_bf16, head_q8], head_q8, "greedy")
        self.assertEqual((label, tokens), ("vs head-bf16 bf16 ref", [0, 1, 2, 7]))
        # Differing formats no longer refuse, but everything else still does.
        other_model = comparable_run("other", weight_format="q8")
        other_model["model"] = {**other_model["model"], "revision": OTHER_REVISION}
        with self.assertRaisesRegex(ValueError, "model.revision"):
            bench.render_table([baseline, other_model])

    def test_sampled_rows_compare_with_the_sampled_reference_and_show_the_sampler(self) -> None:
        sampling = {"temperature": 0.7, "top_p": 0.9, "top_k": 0, "seed": 0}
        sampler = {
            "path": "device",
            "device_draws": 4,
            "host_draws": 0,
            "logits_to_host": 0,
            "logits_to_host_per_token": 0.0,
        }
        base = comparable_run("baseline")
        base["run_kind"] = "baseline"
        base["suite"]["rows"].append(
            {**base["suite"]["rows"][0], "path": "sampled", "tokens": [4, 4, 4, 4], "sampling": sampling}
        )
        head = comparable_run("head")
        head["suite"]["rows"] += [
            {
                **head["suite"]["rows"][0],
                "path": "sampled",
                "tokens": [4, 4, 9, 4],
                "sampling": sampling,
                "sampler": {**sampler, "path": "host:penalty", "device_draws": 0, "host_draws": 4,
                            "logits_to_host": 4, "logits_to_host_per_token": 1.0},
                "kv_cache": "growing",
            },
            {
                **head["suite"]["rows"][1],
                "path": "sampled_step_model",
                "tokens": [4, 4, 9, 4],
                "tokens_match_reference": True,
                "sampling": sampling,
                "sampler": sampler,
                "kv_cache": "static",
            },
        ]
        text = bench.render_table([base, head])
        self.assertIn("Sampled rows: temperature 0.7, top-p 0.9, seed 0", text)
        # The sampled rows compare with the baseline's *sampled* row, not its greedy reference.
        self.assertIn(
            "| head | bf16 | n/a | sampled (reference, growing kv) | 4 | (ref) | no @2 |", text
        )
        self.assertIn(
            "| head | bf16 | n/a | sampled (StepModel, static kv) | 4 | yes | no @2 |", text
        )
        self.assertIn("| host:penalty: 0 device / 4 host, 1.00 logits rows->host/tok |", text)
        self.assertIn("| device: 4 device / 0 host, 0.00 logits rows->host/tok |", text)
        self.assertEqual(bench.row_kind({"path": "sampled_step_model"}), "sampled")
        self.assertEqual(bench.row_kind({"path": "step_model"}), "greedy")
        self.assertEqual(bench.row_label({"path": "sampled"}), "sampled (reference)")

    def test_run_accepts_every_weight_format_and_refuses_a_mismatch(self) -> None:
        for fmt in ("q8", "q4", "nvfp4"):
            doc = suite_document()
            doc["weight_format"] = fmt
            output = self.root / f"out-{fmt}"
            binary = fake_binary(self.root / f"bin-{fmt}", doc)
            self.assertEqual(bench.main(self.run_args(binary, output, format=fmt)), 0)
            self.assertIn(f"format {fmt}", (output / "stdout.log").read_text(encoding="utf-8"))
            record = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(record["requested"]["weight_format"], fmt)
        with self.assertRaises(SystemExit):
            bench.parser().parse_args(self.run_args(self.root / "x", self.root / "y", format="fp8"))
        # The binary must record the format it was asked for.
        doc = suite_document()
        doc["weight_format"] = "bf16"
        with self.assertRaisesRegex(ValueError, "weight_format 'bf16', requested 'q8'"):
            bench.main(self.run_args(fake_binary(self.root / "wrong", doc), self.root / "wrong-out", format="q8"))
        # A document without the key is bf16: fine for bf16, refused for q8.
        bench.check_requested_dimensions({}, "bf16", None)
        with self.assertRaisesRegex(ValueError, "requested 'q8'"):
            bench.check_requested_dimensions({}, "q8", None)

    def test_run_forces_the_cuda_graph_switch_and_requires_it_recorded(self) -> None:
        doc = suite_document()
        doc["cuda_graphs"] = "on"
        output = self.root / "graphs-on"
        self.assertEqual(
            bench.main(self.run_args(fake_binary(self.root / "g-on", doc), output, cuda_graphs="on")), 0
        )
        self.assertIn("graphs 1 ", (output / "stdout.log").read_text(encoding="utf-8"))
        doc["cuda_graphs"] = "off"
        output = self.root / "graphs-off"
        self.assertEqual(
            bench.main(self.run_args(fake_binary(self.root / "g-off", doc), output, cuda_graphs="off")), 0
        )
        self.assertIn("graphs 0 ", (output / "stdout.log").read_text(encoding="utf-8"))
        # Asked for on, ran off: refused.
        with self.assertRaisesRegex(ValueError, "cuda_graphs 'off', requested 'on'"):
            bench.main(self.run_args(fake_binary(self.root / "g-bad", doc), self.root / "g-bad-out", cuda_graphs="on"))
        # A pre-epic binary records no switch: a forced switch is refused rather than assumed.
        doc.pop("cuda_graphs")
        with self.assertRaisesRegex(ValueError, "cuda_graphs None"):
            bench.main(self.run_args(fake_binary(self.root / "g-none", doc), self.root / "g-none-out", cuda_graphs="off"))

    def test_run_refuses_a_probed_device_that_does_not_meet_the_label(self) -> None:
        # sc-24140 feature-end review: the label is a claim about the hardware, checked against
        # what the binary probed — a CUDA device, that compute capability, that GPU.
        for case, (probe, refusal) in enumerate((
            ({"device": "cpu", "device_name": None, "compute_capability": None}, "claims a CUDA device"),
            ({"compute_capability": "8.9"}, "claims compute capability 12.0; the probed device has '8.9'"),
            ({"device_name": "NVIDIA GeForce RTX 4090"}, "names 'RTX Pro 6000'"),
        )):
            doc = {**suite_document(), **probe}
            name = f"probe-{case}"
            with self.assertRaisesRegex(ValueError, refusal):
                bench.main(self.run_args(fake_binary(self.root / name, doc), self.root / f"{name}-out"))
        # A label that claims no checkable hardware is refused too.
        doc = {**suite_document(), "label": "fast box"}
        with self.assertRaisesRegex(ValueError, "is not `<GPU> / sm_<NN>`"):
            bench.main(self.run_args(fake_binary(self.root / "free-label", doc), self.root / "free-label-out",
                                     label="fast box"))
        # Another sm_ label is checked the same way: sm_89 is 8.9.
        self.assertIsNone(bench.check_probed_hardware(
            {"device": "cuda", "device_name": "NVIDIA GeForce RTX 4090", "compute_capability": "8.9"},
            "RTX 4090 / sm_89",
        ))
        # The document must say which device it ran on, and which prompt ids.
        for field, value, refusal in (
            ("device", "gpu", "expected `cuda` or `cpu`"),
            ("prompt_token_ids", None, "no prompt_token_ids"),
            ("prompt_token_ids", [1, 2, 3], "3 prompt_token_ids for 40 prompt tokens"),
        ):
            with self.assertRaisesRegex(ValueError, refusal):
                bench.validate_suite_document({**suite_document(), field: value})

    def test_run_ties_the_binary_to_the_runtime_sha_and_a_clean_build(self) -> None:
        # sc-24140 feature-end review: a head run's binary must embed the runtime SHA and a clean
        # tree (CANDLE_LLM_BUILD_PROVENANCE=1); the checkout being clean is not enough.
        for build, refusal in (
            ({"git_sha": None, "git_dirty": None}, "embeds no build provenance; build it with CANDLE_LLM_BUILD_PROVENANCE=1"),
            ({"git_sha": "a" * 40, "git_dirty": False}, f"built from {'a' * 40}, not the runtime SHA"),
            ({"git_sha": self.sha, "git_dirty": True}, "built from a dirty tree"),
        ):
            doc = {**suite_document(), "build": build}
            name = f"build-{len(refusal)}"
            with self.assertRaisesRegex(ValueError, refusal):
                bench.main(self.run_args(fake_binary(self.root / name, doc), self.root / f"{name}-out"))
        # A baseline binary embeds nothing (its commit predates the build script); one that
        # embeds another commit is refused.
        bench.check_build_provenance({"build": {"git_sha": None, "git_dirty": None}}, self.sha, "baseline")
        bench.check_build_provenance({}, self.sha, "baseline")
        with self.assertRaisesRegex(ValueError, "baseline binary was built from"):
            bench.check_build_provenance({"build": {"git_sha": "c" * 40, "git_dirty": True}}, self.sha, "baseline")

    def test_campaign_rechecks_the_device_and_provenance_of_collected_runs(self) -> None:
        manifest, llama, _ = self.campaign_manifest()
        head = self.root / "sealed" / "llama-bf16"
        args = self.run_args(matrix_binary(self.root / "matrix"), head, run_name="head-llama-bf16",
                             format="bf16", cuda_graphs="off", rows="reference,step_model,ngram")
        args[args.index("--snapshot") + 1] = str(llama)
        args[args.index("--manifest") + 1] = str(manifest)
        args[args.index("--model-key") + 1] = "llama-test"
        self.assertEqual(bench.main(args), 0)
        s1 = self.as_baseline(head, self.root / "baselines" / "llama-s1")
        common = ["campaign", "--formats", "bf16", "--graphs", "off", "--no-sampled", "--baseline", str(s1)]
        for name, doc, refusal in (
            ("sm89", {"compute_capability": "8.9"}, "claims compute capability 12.0"),
            ("unbuilt", {"build": {"git_sha": None, "git_dirty": None}}, "embeds no build provenance"),
        ):
            bad = self.reseal(head, self.root / "sealed" / name, doc=doc)
            with self.assertRaisesRegex(ValueError, refusal):
                bench.main([*common, "--output", str(self.root / f"{name}-campaign"), "--collect", str(bad)])
            self.assertFalse((self.root / f"{name}-campaign").exists())
        # A baseline on another device is refused the same way.
        bad_s1 = self.reseal(s1, self.root / "baselines" / "cpu-s1", doc={"device": "cpu"})
        with self.assertRaisesRegex(ValueError, "claims a CUDA device"):
            bench.main(["campaign", "--formats", "bf16", "--graphs", "off", "--no-sampled",
                        "--baseline", str(bad_s1), "--output", str(self.root / "cpu-campaign"),
                        "--collect", str(head)])

    def test_campaign_run_needs_an_s1_baseline_per_model_before_anything_runs(self) -> None:
        # sc-24140 feature-end review: the S1 comparison row is part of the matrix.
        manifest, llama, hybrid = self.campaign_manifest()
        output = self.root / "no-baseline"
        with self.assertRaisesRegex(ValueError, "missing: llama-test / S1 baseline \\(bf16\\); hybrid-test / S1 baseline"):
            bench.main(self.campaign_args(output, manifest, "--model", f"llama-test={llama}",
                                          "--model", f"hybrid-test={hybrid}"))
        self.assertFalse(output.exists(), "refused before any run")
        # One model's baseline does not stand in for another's.
        with self.assertRaisesRegex(ValueError, "missing: hybrid-test / S1 baseline \\(bf16\\)$"):
            bench.main(self.campaign_args(output, manifest, "--model", f"llama-test={llama}",
                                          "--model", f"hybrid-test={hybrid}",
                                          "--baseline", str(self.s1_baseline(manifest, "llama-test", llama))))
        self.assertFalse(output.exists())
        self.assertEqual(
            bench.baseline_missing([{"key": "m"}], [{"model": {"key": "m"}, "suite": {"weight_format": "q8"}}]),
            ["m / S1 baseline (bf16)"],
        )

    def test_a_first_run_that_is_not_a_baseline_is_labelled_no_s1_baseline(self) -> None:
        # sc-24140 feature-end review: without a baseline the table's first run is still the basis,
        # but its rows can never read as the S1 comparison.
        first = comparable_run("head-a")
        second = comparable_run("head-b")
        text = bench.render_table([first, second])
        self.assertIn("| head-b | bf16 | n/a | MTP off (reference) | 4 | (ref) | yes (vs head-a ref, no S1 baseline) |", text)
        self.assertNotIn("| (ref) | yes |", text)
        self.assertEqual(bench.comparison_basis([first, second], second, "greedy")[0],
                         "vs head-a ref, no S1 baseline")
        first["run_kind"] = "baseline"
        self.assertEqual(bench.comparison_basis([first, second], second, "greedy")[0], "")
        # A bf16 row with no basis at all says why.
        q8_first = comparable_run("q8", weight_format="q8")
        self.assertEqual(bench.baseline_match_text([q8_first, second], second, second["suite"]["rows"][0]),
                         "n/a (no S1 baseline)")

    def test_table_refuses_runs_over_another_prompt_or_seed(self) -> None:
        # sc-24140 feature-end review: the same prompt length is not the same prompt, and sampled
        # rows are only comparable under one seed.
        base = comparable_run("base", sampling={"temperature": 0.7, "top_p": 0.9, "top_k": 0, "seed": 0})
        other_prompt = comparable_run("other", sampling=base["suite"]["sampling"],
                                      prompt_token_ids=list(range(1, 41)))
        with self.assertRaisesRegex(ValueError, "runs differ in suite.prompt_sha256"):
            bench.render_table([base, other_prompt])
        other_seed = comparable_run("other", sampling={**base["suite"]["sampling"], "seed": 7})
        with self.assertRaisesRegex(ValueError, "runs differ in suite.sampling.seed"):
            bench.render_table([base, other_seed])
        self.assertNotEqual(bench.prompt_sha256(base["suite"]), bench.prompt_sha256(other_prompt["suite"]))
        self.assertIsNone(bench.prompt_sha256({}))
        bench.render_table([base, comparable_run("same", sampling=dict(base["suite"]["sampling"]))])

    def test_table_compares_every_row_with_the_first_runs_reference(self) -> None:
        base = comparable_run("baseline")
        base["run_kind"] = "baseline"
        head = copy.deepcopy(base)
        head["run_name"] = "head"
        head["run_kind"] = "head"
        # Head's own rows agree with each other, but its reference drifted from the baseline's.
        for row in head["suite"]["rows"]:
            if row["path"] != "mtp":
                row["tokens"] = [0, 1, 2, 7]
        text = bench.render_table([base, head])
        self.assertIn("| baseline | bf16 | n/a | MTP off (reference) | 4 | (ref) | yes |", text)
        self.assertIn("| head | bf16 | n/a | MTP off (reference) | 4 | (ref) | no @3 |", text)
        self.assertIn("| head | bf16 | n/a | MTP off (StepModel) | 4 | yes | no @3 |", text)
        self.assertIn("| head | bf16 | n/a | MTP K=3 | 4 | no @2 | no @2 |", text)

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
        self.assertIn("| nv | nvfp4 | n/a | MTP off (reference, NVFP4 GEMV off) |", table)
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

    # ---- campaign (sc-24140) --------------------------------------------------------------

    def campaign_manifest(self) -> tuple[Path, Path, Path]:
        """A manifest with a llama-family and a hybrid model, both snapshots pinned."""
        manifest = self.root / "campaign-models.toml"
        manifest.write_text(
            "[[models]]\n"
            'key = "llama-test"\n'
            'repository = "Test/Llama"\n'
            f'revision = "{REVISION}"\n'
            'expected_files = ["config.json"]\n'
            "\n[[models]]\n"
            'key = "hybrid-test"\n'
            'repository = "Test/Hybrid"\n'
            f'revision = "{OTHER_REVISION}"\n'
            'expected_files = ["config.json"]\n',
            encoding="utf-8",
        )
        llama = self.root / "hub" / "models--Test--Llama" / "snapshots" / REVISION
        llama.mkdir(parents=True)
        (llama / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
        hybrid = self.root / "hub" / "models--Test--Hybrid" / "snapshots" / OTHER_REVISION
        hybrid.mkdir(parents=True)
        (hybrid / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
        bench.PINNED_CONFIG_SHA256["llama-test"] = bench.sha256_file(llama / "config.json")
        bench.PINNED_CONFIG_SHA256["hybrid-test"] = bench.sha256_file(hybrid / "config.json")
        return manifest, llama, hybrid

    def s1_baseline(self, manifest: Path, key: str, snapshot: Path) -> Path:
        """A sealed bf16 S1 baseline for `key`: a reference-row run of the matrix fake, re-sealed as
        a pre-epic `baseline` run (another commit, no build provenance)."""
        head = self.root / "baseline-heads" / key
        args = self.run_args(matrix_binary(self.root / "matrix"), head, run_name=f"s1-{key}",
                             format="bf16", cuda_graphs="off", rows="reference", new_tokens="4")
        args[args.index("--snapshot") + 1] = str(snapshot)
        args[args.index("--manifest") + 1] = str(manifest)
        args[args.index("--model-key") + 1] = key
        self.assertEqual(bench.main(args), 0)
        return self.as_baseline(head, self.root / "baselines" / key)

    def campaign_args(self, output: Path, manifest: Path, *extra: str) -> list[str]:
        return [
            "campaign",
            "--output", str(output),
            "--binary", str(matrix_binary(self.root / "matrix")),
            "--runtime-sha", self.sha,
            "--checkout", str(self.checkout),
            "--manifest", str(manifest),
            "--new-tokens", "4",
            "--sample-interval", "0.01",
            *extra,
        ]

    def test_campaign_runs_the_matrix_and_seals_an_index(self) -> None:
        manifest, llama, hybrid = self.campaign_manifest()
        output = self.root / "campaign"
        self.assertEqual(
            bench.main(
                self.campaign_args(
                    output, manifest,
                    "--model", f"llama-test={llama}",
                    "--model", f"hybrid-test={hybrid}",
                    "--formats", "bf16,q8",
                    "--drafts", "1,2",
                    "--baseline", str(self.s1_baseline(manifest, "llama-test", llama)),
                    "--baseline", str(self.s1_baseline(manifest, "hybrid-test", hybrid)),
                )
            ),
            0,
        )
        runs = sorted(p.name for p in (output / "runs").iterdir())
        self.assertEqual(
            runs,
            sorted(
                f"{m}-{f}-graphs-{g}"
                for m in ("llama-test", "hybrid-test")
                for f in ("bf16", "q8")
                for g in ("off", "on")
            ),
        )
        # Each run was asked for its cell's format, graph switch and family rows.
        record = json.loads((output / "runs" / "llama-test-q8-graphs-on" / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["requested"]["weight_format"], "q8")
        self.assertEqual(record["requested"]["cuda_graphs"], "on")
        self.assertEqual(record["requested"]["rows"], "reference,step_model,ngram,sampled,sampled_step_model")
        record = json.loads((output / "runs" / "hybrid-test-bf16-graphs-off" / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["requested"]["rows"], "reference,step_model,mtp,ngram,sampled,sampled_step_model")
        index = (output / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn(
            "| llama-test | q8 | on | ok | n/a (no MTP head) | ok (K=3) | ok | head-llama-test-q8-graphs-on |",
            index,
        )
        self.assertIn(
            "| hybrid-test | bf16 | off | ok | ok (K=1,2) | ok (K=3) | ok | head-hybrid-test-bf16-graphs-off |",
            index,
        )
        # One table per model, each holding that model's runs.
        self.assertIn("## llama-test", index)
        self.assertIn("## hybrid-test", index)
        self.assertIn("| head-llama-test-q8-graphs-on | q8 | on | MTP off (reference) |", index)
        self.assertIn("(vs head-llama-test-bf16-graphs-off bf16 ref)", index)
        # The S1 baseline of each model heads its table: its rows are the unlabelled basis.
        self.assertIn("| llama-test | baseline-s1-llama-test |", index)
        self.assertIn("| head-llama-test-bf16-graphs-off | bf16 | off | MTP off (reference) | 4 | (ref) | yes |", index)
        meta = json.loads((output / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["runtime_sha"], self.sha)
        self.assertEqual(meta["missing"], [])
        self.assertIs(meta["allow_partial"], False)
        self.assertEqual(len(meta["cells"]), 8)
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            self.assertEqual(bench.main(["campaign-verify", str(output)]), 0)
        self.assertIn("8 cells, seals verified, complete", report.getvalue())
        # Self-contained: runs are named relative to the campaign, which verifies after a move.
        self.assertEqual(
            {c["run_directory"] for c in meta["cells"]},
            {f"runs/{name}" for name in runs},
        )
        moved = self.root / "moved-campaign"
        shutil.move(str(output), str(moved))
        bench.verify_campaign(moved)
        output = moved
        # An index naming a run outside the campaign is refused, even when re-sealed.
        tampered = self.root / "tampered-campaign"
        shutil.copytree(moved, tampered)
        index = json.loads((tampered / "index.json").read_text(encoding="utf-8"))
        index["cells"][0]["run_directory"] = str(moved / index["cells"][0]["run_directory"])
        (tampered / "index.json").write_text(json.dumps(index), encoding="utf-8")
        (tampered / "SEAL.json").write_text(
            json.dumps({n: bench.sha256_file(tampered / n) for n in ("INDEX.md", "index.json")}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "outside the campaign directory"):
            bench.verify_campaign(tampered)
        # Tampering with the index or any sealed run is caught.
        with (output / "INDEX.md").open("a", encoding="utf-8") as handle:
            handle.write("edited\n")
        with self.assertRaisesRegex(ValueError, "does not match the campaign seal"):
            bench.verify_campaign(output)

    def test_campaign_verify_catches_a_tampered_run(self) -> None:
        manifest, llama, _ = self.campaign_manifest()
        output = self.root / "campaign"
        bench.main(
            self.campaign_args(
                output, manifest, "--model", f"llama-test={llama}", "--formats", "bf16",
                "--graphs", "off", "--speculative", "off", "--no-sampled",
                "--baseline", str(self.s1_baseline(manifest, "llama-test", llama)),
            )
        )
        self.assertIn("| llama-test | bf16 | off | ok | head-llama-test-bf16-graphs-off |",
                      (output / "INDEX.md").read_text(encoding="utf-8"))
        doc = output / "runs" / "llama-test-bf16-graphs-off" / "decode_bench.json"
        doc.write_text(doc.read_text(encoding="utf-8").replace("10.0", "99.0"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match its seal"):
            bench.verify_campaign(output)

    def test_campaign_refuses_a_dirty_tree_and_leaves_nothing(self) -> None:
        manifest, llama, _ = self.campaign_manifest()
        (self.checkout / "untracked.rs").write_text("x", encoding="utf-8")
        output = self.root / "dirty-campaign"
        with self.assertRaisesRegex(ValueError, "dirty"):
            bench.main(self.campaign_args(output, manifest, "--model", f"llama-test={llama}"))
        self.assertFalse(output.exists(), "a refused campaign must not leave a directory")
        (self.checkout / "untracked.rs").unlink()
        # Output inside the measured checkout would dirty it for the next run: refused up front.
        inside = self.checkout / "evidence" / "campaign"
        with self.assertRaisesRegex(ValueError, "inside the measured checkout"):
            bench.main(self.campaign_args(inside, manifest, "--model", f"llama-test={llama}"))
        self.assertFalse(inside.exists())
        # An unpinned snapshot is refused before anything runs.
        bench.PINNED_CONFIG_SHA256.pop("llama-test")
        with self.assertRaisesRegex(ValueError, "no pinned config sha256"):
            bench.main(self.campaign_args(output, manifest, "--model", f"llama-test={llama}"))
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "unknown values"):
            bench.main(self.campaign_args(output, manifest, "--model", f"x={llama}", "--formats", "bf16,fp8"))
        with self.assertRaisesRegex(ValueError, "KEY=SNAPSHOT"):
            bench.main(self.campaign_args(output, manifest, "--model", str(llama)))

    def test_campaign_collects_sealed_runs_and_refuses_dirty_or_foreign_runs(self) -> None:
        manifest, llama, hybrid = self.campaign_manifest()
        runs = self.root / "sealed"
        binary = matrix_binary(self.root / "matrix")
        for fmt in ("bf16", "nvfp4"):
            args = self.run_args(binary, runs / f"llama-{fmt}", run_name=f"head-llama-{fmt}",
                                 format=fmt, cuda_graphs="off",
                                 rows="reference,step_model,ngram")
            args[args.index("--snapshot") + 1] = str(llama)
            args[args.index("--manifest") + 1] = str(manifest)
            args[args.index("--model-key") + 1] = "llama-test"
            self.assertEqual(bench.main(args), 0)
        s1 = self.as_baseline(runs / "llama-bf16", self.root / "baselines" / "llama-s1")
        collect = ["--collect", str(runs / "llama-bf16"), "--collect", str(runs / "llama-nvfp4"),
                   "--baseline", str(s1)]
        common = ["campaign", "--formats", "bf16,nvfp4", "--graphs", "off", "--no-sampled"]
        output = self.root / "collected"
        self.assertEqual(bench.main([*common, "--output", str(output), *collect]), 0)
        index = (output / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("| llama-test | nvfp4 | off | ok | n/a (no MTP head) | ok (K=3) | head-llama-nvfp4 |", index)
        meta = json.loads((output / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["mode"], "collect")
        # The collected runs were copied in: the campaign verifies without its sources.
        self.assertEqual(
            sorted(c["run_directory"] for c in meta["cells"]), ["runs/llama-bf16", "runs/llama-nvfp4"]
        )
        self.assertEqual(meta["baselines"][0]["run_directory"], "baselines/llama-s1")
        self.assertTrue((output / "runs" / "llama-bf16" / "SEAL.json").is_file())
        bench.verify_campaign(output)
        # A requested cell with no run is refused unless the index is explicitly partial.
        partial = self.root / "partial"
        with self.assertRaisesRegex(ValueError, "missing: llama-test / q8 / graphs off / off"):
            bench.main(["campaign", "--formats", "bf16,q8", "--graphs", "off", "--no-sampled",
                        "--output", str(partial), "--collect", str(runs / "llama-bf16")])
        self.assertFalse(partial.exists())
        self.assertEqual(
            bench.main(["campaign", "--formats", "bf16,q8", "--graphs", "off", "--no-sampled",
                        "--allow-partial", "--output", str(partial), "--collect", str(runs / "llama-bf16")]),
            0,
        )
        partial_index = (partial / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("| llama-test | q8 | off | missing | n/a (no MTP head) | missing | missing |",
                      partial_index)
        # sc-24140 feature-end review: no S1 baseline is a missing cell too, and the partial flag is
        # recorded; campaign-verify reports every gap and passes only because it was allowed.
        self.assertIn("| llama-test | missing |", partial_index)
        self.assertIn("**Partial campaign** (sealed with `--allow-partial`): 3 missing", partial_index)
        meta = json.loads((partial / "index.json").read_text(encoding="utf-8"))
        self.assertIn("llama-test / S1 baseline (bf16)", meta["missing"])
        self.assertIs(meta["allow_partial"], True)
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            self.assertEqual(bench.main(["campaign-verify", str(partial)]), 0)
        self.assertIn("missing: llama-test / S1 baseline (bf16)", report.getvalue())
        self.assertIn("missing: llama-test / q8 / graphs off / off", report.getvalue())
        self.assertIn("partial campaign, explicitly allowed", report.getvalue())

        def resealed_index(name: str, **updates: object) -> Path:
            copy_dir = self.root / name
            shutil.copytree(partial, copy_dir)
            index = json.loads((copy_dir / "index.json").read_text(encoding="utf-8"))
            index.update(updates)
            (copy_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
            (copy_dir / "SEAL.json").write_text(
                json.dumps({n: bench.sha256_file(copy_dir / n) for n in ("INDEX.md", "index.json")}),
                encoding="utf-8",
            )
            return copy_dir

        # The same gaps without the recorded flag fail verification (so does an older index,
        # which has no flag at all).
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bench.main(["campaign-verify", str(resealed_index("unflagged", allow_partial=False))]), 1)
        # An index cannot hide its gaps behind a shorter missing list.
        with self.assertRaisesRegex(ValueError, "disagrees with its cells and baselines"):
            bench.verify_campaign(resealed_index("hidden", missing=[]))
        # A run outside the requested matrix is refused.
        with self.assertRaisesRegex(ValueError, "outside the requested matrix"):
            bench.main([*common[:1], "--formats", "bf16", "--graphs", "off", "--no-sampled",
                        "--output", str(self.root / "stray"), *collect])
        # A sealed run built from a dirty tree is refused.
        dirty = runs / "llama-dirty"
        dirty.mkdir()
        record = json.loads((runs / "llama-bf16" / "run.json").read_text(encoding="utf-8"))
        record["source"] = {**record["source"], "clean_tree": False, "dirty_paths": ["?? scratch.rs"]}
        record["run_name"] = "head-llama-dirty"
        (dirty / "run.json").write_text(json.dumps(record), encoding="utf-8")
        for name in ("decode_bench.json", "decode_bench.md", "stdout.log", "stderr.log"):
            (dirty / name).write_bytes((runs / "llama-bf16" / name).read_bytes())
        (dirty / "SEAL.json").write_text(
            json.dumps({n: bench.sha256_file(dirty / n) for n in ("decode_bench.json", "run.json", "decode_bench.md", "stdout.log", "stderr.log")}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "built from a dirty tree"):
            bench.main([*common, "--output", str(self.root / "dirty-out"), "--collect", str(dirty)])
        self.assertFalse((self.root / "dirty-out").exists())
        # Head runs from two commits do not make one campaign.
        record = json.loads((runs / "llama-nvfp4" / "run.json").read_text(encoding="utf-8"))
        other = self.reseal(
            runs / "llama-nvfp4",
            runs / "llama-other",
            record={"source": {**record["source"], "head_sha": "e" * 40}},
            doc={"build": {"git_sha": "e" * 40, "git_dirty": False}},
        )
        with self.assertRaisesRegex(ValueError, "different commits"):
            bench.main([*common, "--output", str(self.root / "mixed"),
                        "--collect", str(runs / "llama-bf16"), "--collect", str(other)])
        # A baseline is not a head run, and a head run is not a baseline.
        with self.assertRaisesRegex(ValueError, "is not a baseline run"):
            bench.main([*common, "--output", str(self.root / "b"), *collect, "--baseline", str(runs / "llama-bf16")])

    def test_campaign_rows_and_cell_status_per_family(self) -> None:
        self.assertEqual(bench.campaign_rows("llama", ["off", "mtp", "ngram"], False), ["reference", "step_model", "ngram"])
        self.assertEqual(
            bench.campaign_rows("qwen35", ["off", "mtp", "ngram"], True),
            ["reference", "step_model", "mtp", "ngram", "sampled", "sampled_step_model"],
        )
        self.assertEqual(bench.cell_status(None, "llama", "mtp", [1]), "n/a (no MTP head)")
        self.assertEqual(bench.cell_status(None, "qwen35", "mtp", [1]), "missing")
        run = {"suite": {"rows": [{"path": "mtp", "mtp_drafts": 1}]}}
        self.assertEqual(bench.cell_status(run, "qwen35", "mtp", [1]), "ok (K=1)")
        self.assertEqual(bench.cell_status(run, "qwen35", "mtp", [1, 2]), "missing")
        self.assertEqual(bench.snapshot_family(self.snapshot), "qwen35")
        cfg = self.root / "vl"
        cfg.mkdir()
        (cfg / "config.json").write_text('{"model_type": "qwen3_vl"}', encoding="utf-8")
        self.assertEqual(bench.snapshot_family(cfg), "llama")

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
        self.assertIn('(out, stats, prefill_secs, decode_secs, None, None, "mtp", None)', rewritten)
        self.assertIn("fn replay_forwards(_stats: &SpeculativeStats) -> Option<u64> {", rewritten)
        self.assertNotIn("Some(stats.replays as u64)", rewritten)
        self.assertNotIn("set_attn_formulation", before_stub)
        self.assertNotIn("attn_formulation()", before_stub)
        # sc-24140: nothing the baseline compiles — the whole file outside the `BASELINE_STUB`
        # literal — may call a head-only seam (the shared body once did: `mtp.set_attn_formulation`).
        literal_start = rewritten.index(bench.STUB_BEGIN)
        literal_end = rewritten.index(bench.STUB_END, literal_start) + len(bench.STUB_END)
        compiled = rewritten[:literal_start] + rewritten[literal_end:]
        code = "\n".join(
            line for line in compiled.splitlines() if not line.lstrip().startswith("//")
        )
        for head_only in (
            ".set_attn_formulation(",
            ".attn_formulation()",
            ".set_step_kv_cache(",
            "from_weights_format(",
            ".weight_census()",
            "RequestSpan::",
            "generate_speculative_with(",
            "generate_step_timed(",
            "CountingDecode::",
            "ProjectionFormat::",
        ):
            self.assertNotIn(head_only, code, head_only)
        # The llama family's pre-epic reference: the stub loads `CausalLm` with the pre-epic
        # loader and runs the shared `decode_logits` + `generate_from_prefill` reference.
        self.assertIn("CausalLm::from_weights_with(&weights, \"\", cfg, baseline_quant(format))", compiled)
        self.assertIn("run_causal_reference(model, model, prompt, config, device, on_event)", compiled)
        self.assertNotIn("fn is_causal_snapshot(_snapshot: &Path) -> bool {\n    false", compiled)
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
