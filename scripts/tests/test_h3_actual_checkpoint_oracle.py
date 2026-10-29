"""Pure primitive tests for the actual-checkpoint oracle adapter bridge.

No GPU, no model, no network, no real weights. These tests exercise the bridge's bounded spec /
mini-safetensors parsing, the exact provenance binding, the BF16-input contract and the byte-to-
pinned-function wiring. Numeric agreement is NOT re-derived here: the same frozen producer
functions are called on both sides, which is the point of an adapter test.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

# The W4A8/ConvRot replay authority is the frozen producer, which imports torch at module load;
# it cannot be edited (hash-pinned), so tests that drive it need torch. Input-refusal tests do not:
# the bridge binds every selection before importing a producer.
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

BRIDGE = Path(__file__).resolve().parents[1] / "h3_actual_checkpoint_oracle.py"
SPEC = importlib.util.spec_from_file_location("h3_actual_checkpoint_oracle", BRIDGE)
BRIDGE_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BRIDGE_MODULE)


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    header = {}
    offset = 0
    for name, dtype, shape, payload in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    while len(encoded) % 8:
        encoded += b" "
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(t[-1] for t in tensors))


def f32_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


def tiny_w4_tensors() -> dict:
    k = 256
    packed = bytes((index % 256) for index in range(k // 2))
    srel = bytes([0x38] * (k // 16))
    codebook = f32_bytes([float(index) - 8.0 for index in range(16)])
    s_channel = f32_bytes([1.0])
    comfy = json.dumps({"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}).encode()
    return {
        "p.weight": {"dtype": "I8", "shape": [1, k // 2], "bytes": packed},
        "p.weight_s_rel": {"dtype": "F8_E4M3", "shape": [1, k // 16], "bytes": srel},
        "p.weight_codebook": {"dtype": "F32", "shape": [16], "bytes": codebook},
        "p.weight_s_channel": {"dtype": "F32", "shape": [1], "bytes": s_channel},
        "p.comfy_quant": {"dtype": "U8", "shape": [len(comfy)], "bytes": comfy},
    }


def tiny_w4_case() -> dict:
    return {
        "id": "w4.tiny",
        "kind": "w4a8_projection",
        "select": {"base": "p", "bias": False},
        "input": {"dtype": "BF16", "shape": [1, 256], "values": [0.125] * 256},
        "tolerance": {"max_abs": 0.0},
    }

ARTIFACT_SHA = "ab" * 32


def make_manifest(root: Path, cases: list[dict], artifact_tensors=None) -> tuple[Path, Path]:
    """Write a shared probe manifest + artifact + SHA'd staging projection/parent."""
    artifact = root / "artifact.safetensors"
    if artifact_tensors is None:
        artifact_tensors = tiny_w4_tensor_list()
    write_safetensors(artifact, artifact_tensors)
    size = artifact.stat().st_size
    parent = root / "parent-stage.json"
    parent.write_bytes(b'{"schema":"h3-finetune-stage-receipt","schema_version":1}')
    staging = {
        "schema": "sceneworks.h3.staging_receipt",
        "schema_version": 1,
        "producer": "test staging projection",
        "full_file_digest_verified": True,
        "artifact_path": str(artifact),
        "artifact_sha256": ARTIFACT_SHA,
        "artifact_bytes": size,
        "parent_receipt_path": str(parent),
        "parent_receipt_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
    }
    staging_path = root / "staging-projection.json"
    staging_path.write_text(json.dumps(staging), encoding="utf-8")
    manifest = {
        "schema": BRIDGE_MODULE.MANIFEST_SCHEMA,
        "schema_version": 1,
        "backend": "both",
        "artifact": {"path": str(artifact), "sha256": ARTIFACT_SHA, "bytes": size},
        "staging": {
            "receipt_path": str(staging_path),
            "receipt_sha256": hashlib.sha256(staging_path.read_bytes()).hexdigest(),
            "verified_artifact_sha256": ARTIFACT_SHA,
        },
        "cases": cases,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, artifact


def tiny_w4_tensor_list() -> list[tuple[str, str, list[int], bytes]]:
    return [(name, tensor["dtype"], tensor["shape"], tensor["bytes"]) for name, tensor in tiny_w4_tensors().items()]


class ActualCheckpointBridgeTests(unittest.TestCase):
    def test_reader_validates_layout_and_hashes_raw_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.safetensors"
            write_safetensors(path, [("a", "F32", [2], f32_bytes([1.0, 2.0]))])
            tensors = BRIDGE_MODULE.read_selected_safetensors(path)
            self.assertEqual(tensors["a"]["dtype"], "F32")
            self.assertEqual(tensors["a"]["shape"], [2])
            self.assertEqual(len(tensors["a"]["bytes"]), 8)

    def test_reader_refuses_non_contiguous_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            header = json.dumps({"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}).encode()
            while len(header) % 8:
                header += b" "
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"ab")
            self.assertEqual(len(BRIDGE_MODULE.read_selected_safetensors(path)["a"]["bytes"]), 2)

    def test_manifest_refuses_unknown_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = make_manifest(
                root,
                [{"id": "x", "kind": "dense_projection", "select": {}, "tolerance": {"max_abs": 0.0}, "input": {"dtype": "F32", "shape": [1], "values": [0.0]}}],
            )
            with self.assertRaises(BRIDGE_MODULE.Refusal) as caught:
                BRIDGE_MODULE.parse_spec(manifest)
            self.assertEqual(caught.exception.code, "spec_malformed")

    def test_packed_cases_require_bf16_input(self) -> None:
        case = tiny_w4_case()
        case["input"]["dtype"] = "F32"
        with self.assertRaises(BRIDGE_MODULE.Refusal) as caught:
            BRIDGE_MODULE.validate_case(case, 0)
        self.assertEqual(caught.exception.code, "input_dtype")

    @unittest.skipUnless(TORCH_AVAILABLE, "the frozen W4A8 replay producer requires torch")
    def test_w4a8_wiring_calls_the_frozen_replay_functions(self) -> None:
        w4 = BRIDGE_MODULE.load_frozen(BRIDGE_MODULE.FROZEN_W4A8)
        tensors = tiny_w4_tensors()
        case = tiny_w4_case()
        rows = BRIDGE_MODULE.w4a8_rows(case, tensors, w4)
        weight = BRIDGE_MODULE.rows_i8(tensors["p.weight"], "p.weight")
        codebook = BRIDGE_MODULE.vector_f32(tensors["p.weight_codebook"], "codebook")
        s_channel = BRIDGE_MODULE.vector_f32(tensors["p.weight_s_channel"], "s_channel")
        decoded = w4.decode_weight_replay(
            [[byte & 0xFF for byte in row] for row in weight],
            BRIDGE_MODULE.rows_u8(tensors["p.weight_s_rel"], "s_rel"),
            codebook,
            256,
            group=16,
        )
        x = w4.to_bf16_rows(BRIDGE_MODULE.input_rows(case, 256))
        _, xq, xs, _ = w4.quantize_int8_rowwise_convrot_replay(x)
        acc, _, _ = w4.int_accumulate_replay(xq, decoded, k_tile=256)
        expected = [[w4.epilogue_replay(acc[0][0], xs[0], s_channel[0], None)]]
        self.assertEqual(rows, expected)
        self.assertTrue(all(value == value for row in rows for value in row))

    @unittest.skipUnless(TORCH_AVAILABLE, "the frozen W4A8 replay producer requires torch")
    def test_convrot_refuses_non_256_group(self) -> None:
        w4 = BRIDGE_MODULE.load_frozen(BRIDGE_MODULE.FROZEN_W4A8)
        tensors = {
            "p.weight": {"dtype": "I8", "shape": [1, 256], "bytes": bytes(256)},
            "p.weight_scale": {"dtype": "F32", "shape": [1], "bytes": f32_bytes([1.0])},
            "p.comfy_quant": {
                "dtype": "U8",
                "shape": [1],
                "bytes": json.dumps({"format": "convrot_int8", "convrot_groupsize": 64}).encode(),
            },
        }
        case = {
            "id": "convrot.tiny",
            "kind": "convrot_projection",
            "select": {"base": "p", "bias": False},
            "input": {"dtype": "BF16", "shape": [1, 256], "values": [0.0] * 256},
            "tolerance": {"max_abs": 0.0},
        }
        with self.assertRaises(BRIDGE_MODULE.Refusal) as caught:
            BRIDGE_MODULE.convrot_rows(case, tensors, w4)
        self.assertEqual(caught.exception.code, "convrot_group_unsupported")

    def test_run_writes_refusal_receipt_and_no_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {
                "id": "absent.tiny",
                "kind": "convrot_projection",
                "select": {"base": "p", "bias": False},
                "input": {"dtype": "BF16", "shape": [1, 2], "values": [0.0, 0.0]},
                "tolerance": {"max_abs": 0.0},
            }
            manifest, _ = make_manifest(root, [case], artifact_tensors=[("p.weight", "I8", [1, 2], b"\x00\x01")])
            selected = root / "selected.safetensors"
            write_safetensors(selected, [("p.weight", "I8", [1, 2], b"\x00\x01")])
            output = root / "oracle.json"
            receipt = root / "receipt.json"
            code = BRIDGE_MODULE.main(
                [
                    "--spec",
                    str(manifest),
                    "--selected-tensors",
                    str(selected),
                    "--output",
                    str(output),
                    "--receipt",
                    str(receipt),
                ]
            )
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            written = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(written["status"], "refused")
            self.assertEqual(written["code"], "selection_missing")
            self.assertFalse(written["network_used"])
            self.assertFalse(written["gpu_used"])

    def test_manifest_refuses_inconsistent_staging_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = make_manifest(root, [tiny_w4_case()])
            staging_path = root / "staging-projection.json"
            staging = json.loads(staging_path.read_text(encoding="utf-8"))
            staging["artifact_sha256"] = "cd" * 32
            staging_path.write_text(json.dumps(staging), encoding="utf-8")
            manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_json["staging"]["receipt_sha256"] = hashlib.sha256(staging_path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(manifest_json), encoding="utf-8")
            selected = root / "selected.safetensors"
            write_safetensors(selected, tiny_w4_tensor_list())
            receipt = root / "receipt.json"
            code = BRIDGE_MODULE.main(
                ["--spec", str(manifest), "--selected-tensors", str(selected),
                 "--output", str(root / "oracle.json"), "--receipt", str(receipt)]
            )
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["code"], "staging_binding_mismatch")

    def test_selected_tensor_must_match_actual_artifact_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = make_manifest(root, [tiny_w4_case()], artifact_tensors=tiny_w4_tensor_list())
            selected = root / "selected.safetensors"
            tensors = tiny_w4_tensors()
            tensors["p.weight"] = {"dtype": "F32", "shape": [1, 128], "bytes": f32_bytes([0.0] * 128)}
            write_safetensors(selected, [(n, t["dtype"], t["shape"], t["bytes"]) for n, t in tensors.items()])
            receipt = root / "receipt.json"
            code = BRIDGE_MODULE.main(
                ["--spec", str(manifest), "--selected-tensors", str(selected),
                 "--output", str(root / "oracle.json"), "--receipt", str(receipt)]
            )
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["code"], "selection_binding_mismatch")

    def test_artifact_size_mismatch_refuses_before_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = make_manifest(root, [tiny_w4_case()])
            manifest_json = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_json["artifact"]["bytes"] += 1
            manifest.write_text(json.dumps(manifest_json), encoding="utf-8")
            selected = root / "selected.safetensors"
            write_safetensors(selected, tiny_w4_tensor_list())
            receipt = root / "receipt.json"
            code = BRIDGE_MODULE.main(
                ["--spec", str(manifest), "--selected-tensors", str(selected),
                 "--output", str(root / "oracle.json"), "--receipt", str(receipt)]
            )
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["code"], "artifact_size_mismatch")

    def test_manifest_rejects_non_hex_sha_and_invalid_axes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = make_manifest(root, [tiny_w4_case()])
            broken = json.loads(manifest.read_text(encoding="utf-8"))
            broken["artifact"]["sha256"] = "z" * 64
            manifest.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(BRIDGE_MODULE.Refusal):
                BRIDGE_MODULE.parse_spec(manifest)

            manifest, _ = make_manifest(root, [tiny_w4_case()])
            broken = json.loads(manifest.read_text(encoding="utf-8"))
            broken["cases"][0]["input"]["shape"] = [1, 0]
            manifest.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(BRIDGE_MODULE.Refusal) as caught:
                BRIDGE_MODULE.parse_spec(manifest)
            self.assertEqual(caught.exception.code, "spec_malformed")

            manifest, _ = make_manifest(root, [tiny_w4_case()])
            broken = json.loads(manifest.read_text(encoding="utf-8"))
            broken["cases"][0]["input"]["shape"] = [1, True]
            manifest.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(BRIDGE_MODULE.Refusal):
                BRIDGE_MODULE.parse_spec(manifest)

            manifest, _ = make_manifest(root, [tiny_w4_case()])
            broken = json.loads(manifest.read_text(encoding="utf-8"))
            broken["cases"][0]["input"]["values"][0] = float("nan")
            manifest.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(BRIDGE_MODULE.Refusal):
                BRIDGE_MODULE.parse_spec(manifest)

    def test_bounded_read_rejects_oversize_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "big.json"
            path.write_bytes(b"0123456789")
            with self.assertRaises(BRIDGE_MODULE.Refusal) as caught:
                BRIDGE_MODULE.read_bounded(path, 4, "fixture")
            self.assertEqual(caught.exception.code, "metadata_too_large")

    def test_pruned_projection_uses_actual_rank_on_a_meta_parent(self) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest("torch is not installed")

        class Adaln:
            def __init__(self, linear):
                self.linear = linear

        seen = {}

        class FakeDiTBlock:
            def __init__(
                self,
                hidden_size,
                num_attention_heads,
                attention_head_dim,
                ffn_dim,
                time_embed_dim,
                norm_eps,
                qk_norm_eps,
                *,
                apply_silu,
                adaln_dtype,
                ffn_chunk_size,
                dtype,
                device,
            ):
                # A real body would be huge; on meta nothing is allocated.
                self.body = torch.nn.Parameter(torch.empty(hidden_size, ffn_dim, device=device))
                self.adaln_proj = Adaln(torch.nn.Linear(time_embed_dim, 18 * hidden_size, bias=True, device=device))
                seen["device"] = device
                seen["rank"] = time_embed_dim

        ns = {"DiTBlock": FakeDiTBlock}
        config = {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "ffn_dim": 16,
            "time_embed_dim": 16,  # original unpruned width: must NOT be used for the AdaLN input
            "text_dim": 8,
            "in_channels": 4,
            "audio_in_channels": 4,
            "patch_size": [1, 1, 1],
            "norm_eps": 1e-5,
            "qk_norm_eps": 1e-5,
            "final_norm_eps": 1e-5,
        }
        rank = 8
        weight = BRIDGE_MODULE._torch().zeros(18 * 8, rank)
        bias = BRIDGE_MODULE._torch().zeros(18 * 8)
        linear = BRIDGE_MODULE.projection_linear(
            ns, config, "transformer_blocks.0.adaln_proj.linear", weight, bias, rank
        )
        self.assertEqual(tuple(linear.weight.shape), (18 * 8, 8))
        self.assertEqual(linear.weight.device.type, "cpu")
        self.assertEqual(seen["device"], "meta", "the parent body must be constructed on meta")
        self.assertEqual(seen["rank"], 8, "the actual reduced rank must drive the AdaLN input width")
        output = linear(BRIDGE_MODULE._torch().zeros(2, rank))
        self.assertEqual(tuple(output.shape), (2, 18 * 8))

    def test_pruned_reduced_projection_binds_operands_with_their_header_shape(self) -> None:
        # Regression: f32_tensor once returned a flat vector, so the [out, rank] weight failed the
        # dim() check and every actual pruned-reduced-projection case refused.
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        from unittest.mock import patch

        class Adaln:
            def __init__(self, linear):
                self.linear = linear

        class FakeDiTBlock:
            def __init__(self, hidden_size, num_attention_heads, attention_head_dim, ffn_dim, time_embed_dim,
                         norm_eps, qk_norm_eps, *, apply_silu, adaln_dtype, ffn_chunk_size, dtype, device):
                self.adaln_proj = Adaln(torch.nn.Linear(time_embed_dim, 18 * hidden_size, bias=True, device=device))

        config = {"hidden_size": 2, "num_attention_heads": 1, "attention_head_dim": 2, "ffn_dim": 4,
                  "norm_eps": 1e-5, "qk_norm_eps": 1e-5}
        out, rank = 18 * 2, 3
        weight_values = [float(index) / 16.0 for index in range(out * rank)]
        bias_values = [float(index) for index in range(out)]
        tensors = {
            "p.weight": {"dtype": "F32", "shape": [out, rank], "bytes": f32_bytes(weight_values)},
            "p.bias": {"dtype": "F32", "shape": [out], "bytes": f32_bytes(bias_values)},
        }
        case = {"id": "pruned-reduced-projection", "kind": BRIDGE_MODULE.KIND_REDUCED,
                "select": {"prefix": "p", "weight": "p.weight", "bias_tensor": "p.bias"},
                "input": {"dtype": "F32", "shape": [1, rank], "values": [1.0, 0.0, 0.0]}}
        ns = {"DiTBlock": FakeDiTBlock}
        output = BRIDGE_MODULE.pruned_output(case, tensors, ns, config)
        self.assertEqual(tuple(output.shape), (1, out))
        # Input e0 selects the weight's first column, so row-major layout is preserved.
        expected = [weight_values[row * rank] + bias_values[row] for row in range(out)]
        self.assertEqual(output.flatten().tolist(), expected)

        def flat(tensor, name):
            return torch.tensor(BRIDGE_MODULE.f32_values(tensor, name), dtype=torch.float32)

        with patch.object(BRIDGE_MODULE, "f32_tensor", side_effect=flat):
            with self.assertRaises(BRIDGE_MODULE.Refusal) as caught:
                BRIDGE_MODULE.pruned_output(case, tensors, ns, config)
        self.assertEqual(caught.exception.code, "selected_dtype")  # the old flat behavior is RED

    def test_f32_tensor_refuses_values_that_do_not_fill_the_header_shape(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("torch is not installed")
        tensor = {"dtype": "F32", "shape": [2, 3], "bytes": f32_bytes([0.0] * 5)}
        with self.assertRaises(BRIDGE_MODULE.Refusal) as caught:
            BRIDGE_MODULE.f32_tensor(tensor, "w")
        self.assertEqual(caught.exception.code, "selected_dtype")
        shaped = BRIDGE_MODULE.f32_tensor({"dtype": "F32", "shape": [2, 3], "bytes": f32_bytes([0.0] * 6)}, "w")
        self.assertEqual(tuple(shaped.shape), (2, 3))

    def test_mixed_concat_preserves_distinct_projection_output_widths(self) -> None:
        from unittest.mock import patch
        case = {"id": "mixed", "select": {"base": "w", "convrot_base": "c"}}
        with patch.object(BRIDGE_MODULE, "w4a8_rows", return_value=[[1.0, 2.0]]) as w4, \
             patch.object(BRIDGE_MODULE, "convrot_rows", return_value=[[3.0, 4.0, 5.0]]) as conv:
            result = BRIDGE_MODULE.mixed_rows(case, {}, object())
        self.assertEqual(result, [[1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertEqual(w4.call_count, 1)
        self.assertEqual(conv.call_count, 1)

    def test_frozen_provenance_records_exact_upstream_revisions(self) -> None:
        records = BRIDGE_MODULE.frozen_provenance()
        self.assertEqual(len(records), 3)
        for record in records:
            self.assertEqual(record["source_revision_sha256"], record["sha256"])
            self.assertTrue(record["upstream_revisions"])
            data = (BRIDGE_MODULE.REPO_ROOT / record["script"]).read_bytes()
            self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())
            for revision in record["upstream_revisions"].values():
                self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_candle_interpolation_binds_full_raw_basis_before_table_only_selection(self) -> None:
        from unittest.mock import patch
        canonical = {"adaln_t_table": [2, 1]}
        for layer in range(2):
            canonical[f"transformer_blocks.{layer}.adaln_proj.linear.weight"] = [18, 1]
            canonical[f"transformer_blocks.{layer}.adaln_proj.linear.bias"] = [18]
        canonical.update({"norm_out.linear.weight": [2, 1], "norm_out.linear.bias": [2]})
        aliases = {}
        stored = []
        for index, (name, shape) in enumerate(canonical.items()):
            raw = "model.diffusion_model." + name.replace("transformer_blocks.", "blocks.").replace("norm_out.linear.", "final_layer.adaln_proj.linear.")
            aliases[raw] = name
            count = 1
            for axis in shape:
                count *= axis
            stored.append((raw, "F32", shape, f32_bytes([float(index)] * count)))
        case = {"id": "interpolation", "kind": BRIDGE_MODULE.KIND_INTERPOLATION,
                "select": {"topology": True, "aliases": aliases}}
        config = {"num_layers": 2}
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "tiny-pruned.safetensors"
            write_safetensors(file, stored)
            tensors = BRIDGE_MODULE.read_selected_safetensors(file)
            def verify():
                names = BRIDGE_MODULE.selected_raw_names(case, "candle", config)
                self.assertEqual(set(names), set(tensors), "Candle interpolation must bind its whole native basis, not table only")
                entries = BRIDGE_MODULE.selected_entries(names, tensors)
                self.assertEqual(len(entries), 7)
                for entry in entries:
                    self.assertEqual(entry["sha256"], hashlib.sha256(tensors[entry["name"]]["bytes"]).hexdigest())
            verify()
            original = BRIDGE_MODULE.selected_raw_names
            def old_table_only(value, backend, geometry):
                if value["kind"] == BRIDGE_MODULE.KIND_INTERPOLATION:
                    return [BRIDGE_MODULE.raw_for(value["select"], "adaln_t_table")]
                return original(value, backend, geometry)
            with patch.object(BRIDGE_MODULE, "selected_raw_names", side_effect=old_table_only):
                with self.assertRaises(AssertionError):
                    verify()  # Executed old behavior goes RED; no suppressive subTest.
            verify()  # Restored selection is GREEN over the same actual header and byte hashes.
            mlx = {**case, "select": {"aliases": aliases}}
            self.assertEqual(BRIDGE_MODULE.selected_raw_names(mlx, "mlx", config),
                             [BRIDGE_MODULE.raw_for(mlx["select"], "adaln_t_table")])
            reduced = {"id": "reduced", "kind": BRIDGE_MODULE.KIND_REDUCED,
                       "select": {"prefix": "transformer_blocks.0.adaln_proj.linear", "aliases": aliases}}
            self.assertEqual(len(BRIDGE_MODULE.selected_raw_names(reduced, "mlx", config)), 2)
            self.assertEqual(set(BRIDGE_MODULE.selected_raw_names(reduced, "candle", config)), set(tensors))

    def test_convrot_column_scale_is_preserved_and_flat_scale_is_refused(self) -> None:
        n, k = 16, 256
        scales = [(i + 1) / 16 for i in range(n)]
        descriptor = json.dumps({"convrot_groupsize": 256}).encode()
        stored = [
            ("p.weight", "I8", [n, k], bytes(n * k)),
            ("p.weight_scale", "F32", [n, 1], f32_bytes(scales)),
            ("p.comfy_quant", "U8", [len(descriptor)], descriptor),
        ]
        class Replay:
            def to_bf16_rows(self, rows):
                return rows
            def quantize_int8_rowwise_convrot_replay(self, rows):
                return None, [[0] * k], [1.0], None
            def int_accumulate_replay(self, rows, weight, k_tile):
                return [[0] * n], None, None
            def epilogue_replay(self, acc, activation_scale, row_scale, bias):
                return row_scale  # Wiring witness only, not a replacement arithmetic oracle.
        case = {"id": "convrot", "select": {"base": "p"},
                "input": {"dtype": "BF16", "shape": [1, k], "values": [0.0] * k}}
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "tiny-convrot.safetensors"
            write_safetensors(file, stored)
            tensors = BRIDGE_MODULE.read_selected_safetensors(file)
            self.assertEqual(BRIDGE_MODULE.convrot_rows(case, tensors, Replay()), [scales])
            self.assertEqual(tensors["p.weight_scale"]["shape"], [n, 1])
            self.assertEqual(tensors["p.weight_scale"]["bytes"], f32_bytes(scales))
            with self.assertRaises(BRIDGE_MODULE.Refusal):
                BRIDGE_MODULE.vector_f32(tensors["p.weight_scale"], "old ConvRot scale path")
            tensors["p.weight_scale"]["shape"] = [n]
            with self.assertRaises(BRIDGE_MODULE.Refusal) as caught:
                BRIDGE_MODULE.convrot_rows(case, tensors, Replay())
            self.assertEqual(caught.exception.code, "selected_dtype")

    def test_bridge_has_no_network_or_gpu_imports(self) -> None:
        source = BRIDGE.read_text(encoding="utf-8")
        for forbidden in (
            "import urllib",
            "import requests",
            "import http.client",
            "import socket",
            "import mlx",
            "import mlx_rs",
            "torch.cuda",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
