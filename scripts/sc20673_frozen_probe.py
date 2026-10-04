#!/usr/bin/env python3
"""Fresh-process SC-20673 probe; execute only on the owned MLX/Metal device."""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

FUNCTIONS = (
    ("group_affine_decode", "veloxquant_mlx.metal._scalar_attend", "scalar_fused_decode_attend"),
    ("rabitq_decode", "veloxquant_mlx.metal._rabitq_attend", "rabitq_fused_attend"),
    ("rabitq_prefill", "veloxquant_mlx.metal._rabitq_prefill", "rabitq_prefill_attend"),
    ("rvq_quant_pack", "veloxquant_mlx.metal._rvq_quant_pack", "rvq_quant_pack"),
)


def inputs_for(name, mx):
    b, h, d, skv, sq, group = 1, 8, 128, 2048, 1, 32
    q = mx.ones((b, h, sq if name != "rabitq_prefill" else 256, d), dtype=mx.float16)
    if name == "group_affine_decode":
        gk, gv = (skv + group - 1) // group, (d + group - 1) // group
        args = (q, mx.zeros((b,h,skv,d), dtype=mx.uint8), mx.ones((b,h,gk,d)), mx.zeros((b,h,gk,d)), mx.zeros((b,h,skv,d), dtype=mx.uint8), mx.ones((b,h,skv,gv)), mx.zeros((b,h,skv,gv)), group, d ** -0.5)
        return {"B":b,"H":h,"Sq":sq,"Skv":skv,"D":d,"group_size":group}, args
    if name == "rvq_quant_pack":
        bits, levels = 2, 4
        args = (mx.zeros((skv,d), dtype=mx.float16), mx.zeros((levels,)), mx.zeros((levels-1,)), mx.zeros((levels-1,)), bits)
        return {"N":skv,"D":d,"bits":bits}, args
    q_scale = mx.ones((b,h,q.shape[2])); k_bits = mx.zeros((b,h,skv,d//8), dtype=mx.uint8); k_mag = mx.ones((b,h,skv)); k_const = mx.zeros((b,h,skv)); v_idx = mx.zeros((b,h,skv,d//2), dtype=mx.uint8); v_cents = mx.zeros((16,))
    if name == "rabitq_decode":
        return {"B":b,"H":h,"Sq":sq,"Skv":skv,"D":d,"packed_values":True}, (q,q_scale,k_bits,k_mag,k_const,v_idx,v_cents)
    return {"B":b,"H":h,"Sq":q.shape[2],"Skv":skv,"D":d,"packed_values":True}, (q,mx.array([d ** -0.5]),k_bits,k_mag,k_const,v_idx,v_cents)


def output_bytes(output) -> int:
    if isinstance(output, (tuple, list)):
        return sum(output_bytes(value) for value in output)
    return int(output.nbytes)


def physical_bytes(name, geometry, output):
    """Exact allocated bytes for the representative geometry."""
    d = geometry["D"]
    out = output_bytes(output)
    if name == "rvq_quant_pack":
        n, bits = geometry["N"], geometry["bits"]
        levels = 1 << bits
        words = -(-d // (32 // bits))
        stream = n * words * 4
        return {
            "dense_input_bytes": n * d * 2,
            "uint8_index_intermediates_avoided_bytes": 2 * n * d,
            "packed_stream_1_bytes": stream,
            "packed_stream_2_bytes": stream,
            "metadata_bytes": (levels + 2 * (levels - 1)) * 4,
            "compressed_output_bytes": 2 * stream,
            "output_bytes": out,
        }
    b, h, skv = (geometry[key] for key in ("B", "H", "Skv"))
    dense_k = b * h * skv * d * 2
    dense_v = dense_k
    if name == "group_affine_decode":
        group = geometry["group_size"]
        key_groups = -(-skv // group)
        value_groups = -(-d // group)
        k_codes = b * h * skv * d
        k_meta = 2 * b * h * key_groups * d * 4
        v_codes = b * h * skv * d
        v_meta = 2 * b * h * skv * value_groups * 4
        return {
            "dense_key_bytes": dense_k,
            "dense_value_bytes": dense_v,
            "key_codes_bytes": k_codes,
            "key_scale_zero_bytes": k_meta,
            "value_codes_bytes": v_codes,
            "value_scale_zero_bytes": v_meta,
            "compressed_persistent_bytes": k_codes + k_meta + v_codes + v_meta,
            "dense_reference_transient_bytes": dense_k + dense_v,
            "output_bytes": out,
        }
    k_bits = b * h * skv * (d // 8)
    k_meta = 2 * b * h * skv * 4
    packed_values = b * h * skv * (d // 2)
    centroids = 16 * 4
    return {
        "dense_key_bytes": dense_k,
        "dense_value_bytes": dense_v,
        "key_bits_bytes": k_bits,
        "key_magnitude_constant_bytes": k_meta,
        "packed_values_bytes": packed_values,
        "value_centroids_bytes": centroids,
        "compressed_persistent_bytes": k_bits + k_meta + packed_values + centroids,
        "dense_reference_transient_bytes": dense_k + dense_v,
        "output_bytes": out,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Deliberately imported only in the fresh child, never by CPU-only validators.
    sys.path.insert(0, str(args.source))
    import mlx.core as mx  # type: ignore[import-not-found]

    records = []
    for name, module_name, function_name in FUNCTIONS:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        # The device owner supplies representative production geometry/input factories with the
        # frozen checkout. Do not substitute a process wall-time or RSS proxy for these fields.
        geometry, inputs = inputs_for(name, mx)
        mx.eval(inputs)  # input evaluation is outside the measured allocator delta
        mx.synchronize()
        cache = getattr(module, "_cache", None)
        if isinstance(cache, dict):
            cache.clear()
        mx.clear_cache()
        active_before = int(mx.get_active_memory())
        mx.reset_peak_memory()
        enqueued = time.perf_counter(); output = function(*inputs); host_graph_s = time.perf_counter() - enqueued
        first_start = time.perf_counter(); mx.eval(output); mx.synchronize(); first_s = time.perf_counter() - first_start
        first_peak = int(mx.get_peak_memory())
        first_active = int(mx.get_active_memory())
        pending = function(*inputs)
        async_start = time.perf_counter(); mx.async_eval(pending); async_submit_s = time.perf_counter() - async_start
        sync_start = time.perf_counter(); mx.synchronize(); sync_s = time.perf_counter() - sync_start
        warm = []
        for _ in range(7):
            start = time.perf_counter(); mx.eval(function(*inputs)); mx.synchronize(); warm.append(time.perf_counter() - start)
        peak = int(mx.get_peak_memory())
        median = statistics.median(warm)
        records.append({"name": name, "function": f"{module_name}.{function_name}", "geometry": geometry,
                        "metrics": {
                            "host_graph_build_s": host_graph_s,
                            "first_eval_compile_and_dispatch_s": first_s,
                            "compile_warmup_overhead_estimate_s": max(first_s - median, 0.0),
                            "async_submit_s": async_submit_s,
                            "explicit_synchronize_completion_s": sync_s,
                            "steady_dispatch_sync_median_s": median,
                            "mlx_active_before_bytes": active_before,
                            "mlx_active_after_first_bytes": first_active,
                            "mlx_first_peak_bytes": first_peak,
                            "mlx_campaign_peak_bytes": peak,
                            "mlx_peak_delta_bytes": max(peak - active_before, 0),
                        },
                        "physical_bytes": physical_bytes(name, geometry, output)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schemaVersion": 2, "deviceInfo": mx.metal.device_info(), "measurementLabels": {
        "host_graph_build_s": "Python wrapper and lazy graph construction only",
        "first_eval_compile_and_dispatch_s": "first synchronized evaluation; includes kernel compile and first dispatch",
        "compile_warmup_overhead_estimate_s": "first synchronized evaluation minus warm median; includes all first-run effects",
        "async_submit_s": "mx.async_eval submission only",
        "explicit_synchronize_completion_s": "mx.synchronize completion after async submission",
        "steady_dispatch_sync_median_s": "median of seven synchronized warm dispatches",
        "mlx_peak_delta_bytes": "MLX allocator peak minus active input allocation after reset_peak_memory",
    }, "probes": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
