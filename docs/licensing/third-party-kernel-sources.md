# Third-party kernel sources (feature SW-F287, requirement E10)

Every runtime-compiled (nvrtc) or Metal kernel source in this workspace that ports code or a
design from another project is listed here with its upstream revision and license, next to the
in-tree attribution it carries. Add a row when a kernel lands; the source file's header and its
notice file must agree with the row.

| In-tree source | Upstream | Revision | License | Attribution in tree |
|---|---|---|---|---|
| `crates/kernels/candle-quant-kernels/src/sage_attention.cu` (SW-6131, INT8 QK attention) | [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) `backends/cuda/sage_attention` (`quant_qk_int8.cu`, `quant_v_int8.cu`, `sage_attn_launcher.cu`, `qk_int_sv_i8_cuda.cuh`, `attn_utils.cuh`, `mma.cuh`) | `6022b48b7e39fdd31f0124de268a8267af46d4e0` | Apache-2.0 (NVIDIA) | `.cu` header; `sage_attention-NOTICE.md` |
| `crates/kernels/candle-quant-kernels/src/sage_attention.cu` (SW-6131, INT8 QK attention) | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) (algorithm) | `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` | Apache-2.0 | `.cu` header; `sage_attention-NOTICE.md` |
| `crates/kernels/candle-quant-kernels/src/int8_attention_common.cuh` (SW-6137, the INT8-attention device primitives shared by the two rows above and below) | the SW-6131 `sage_attention.cu` primitives (comfy-kitchen `mma.cuh` / `attn_utils.cuh` design, above) | `6022b48b7e39fdd31f0124de268a8267af46d4e0` | Apache-2.0 | `.cuh` header; `sage_attention-NOTICE.md` |
| `crates/kernels/candle-quant-kernels/src/sol_attention.cu` (SW-6137, Sol-Attn sparse block-routed attention; ported source) and its eager reference in `sol_attention.rs` | [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) `backends/cuda/sage_attention` (`sol_layout.cuh`, `sol_attn.cu`, `sol_attn_preprocess.cu`, `sol_attn_vtranspose.cu`, `sol_attn_route.cu`, `sol_attn_exact.cu`), `backends/cuda/__init__.py` (`_topk_from_pooled`), `backends/eager/sol_attn.py`; algorithm: Sol-Attn (arXiv 2607.24027) | `6022b48b7e39fdd31f0124de268a8267af46d4e0` | Apache-2.0 (Comfy Org) | `.cu` header; `sol_attention-NOTICE.md` |
| `crates/kernels/candle-quant-kernels/src/veda_int8.cu` (Veda sparse INT8 attention) | Veda-on-ComfyUI `sparse_int8.py` @ `60bfae6`; SageAttention arithmetic | — | MIT (Veda); BSD-3-Clause (SageAttention, at the revision Veda used) | `.cu` header; `veda-LICENSE.upstream`, `veda-LICENSE.sage` |
| `crates/kernels/candle-quant-kernels/src/kitchen_int8.cu` (ConvRot INT8 row quantizer) | [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) | `f8b7e97a1418d2e529bd5405fbf86312be198ae6` | Apache-2.0 | `.cu` header; [docs/README.md](../README.md#native-h3-convrot-int8-checkpoints-sc-26683) |
