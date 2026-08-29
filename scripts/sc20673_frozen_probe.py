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


def physical_bytes(geometry, output):
    b, h, d, skv = (geometry.get(k, 1) for k in ("B", "H", "D", "Skv"))
    dense_kv = b*h*skv*d*4  # fp16 K plus fp16 V
    return {"dense_kv_bytes": dense_kv, "key_codes_or_bits_bytes": b*h*skv*(d//8), "scales_zeros_or_magnitudes_constants_bytes": b*h*skv*8, "packed_values_bytes": b*h*skv*(d//2), "output_bytes": output.nbytes, "dense_reference_transient_bytes": dense_kv}


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
        function = getattr(importlib.import_module(module_name), function_name)
        # The device owner supplies representative production geometry/input factories with the
        # frozen checkout. Do not substitute a process wall-time or RSS proxy for these fields.
        geometry, inputs = inputs_for(name, mx)
        mx.eval(*inputs)  # input evaluation is outside the measured allocator delta
        baseline = mx.get_peak_memory(); mx.reset_peak_memory()
        enqueued = time.perf_counter(); output = function(*inputs); enqueue_s = time.perf_counter() - enqueued
        first_start = time.perf_counter(); mx.eval(output); mx.synchronize(); first_s = time.perf_counter() - first_start
        sync_start = time.perf_counter(); mx.synchronize(); sync_s = time.perf_counter() - sync_start
        warm = []
        for _ in range(7):
            start = time.perf_counter(); mx.eval(function(*inputs)); mx.synchronize(); warm.append(time.perf_counter() - start)
        peak = mx.get_peak_memory()
        records.append({"name": name, "function": f"{module_name}.{function_name}", "geometry": geometry,
                        "metrics": {"host_enqueue_s": enqueue_s, "first_eval_compile_dispatch_s": first_s,
                        "explicit_synchronize_s": sync_s, "warm_synchronized_median_s": statistics.median(warm),
                        "mlx_peak_bytes": peak, "mlx_peak_delta_bytes": peak - baseline},
                        "physical_bytes": physical_bytes(geometry, output)})
    args.output.write_text(json.dumps({"schemaVersion": 1, "measurementLabels": {"host_enqueue_s": "host enqueue only", "first_eval_compile_dispatch_s": "first synchronized evaluation; includes compile and first dispatch", "explicit_synchronize_s": "explicit mx.synchronize completion after dispatch", "warm_synchronized_median_s": "median of synchronized warm iterations"}, "probes": records}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
