# `sage_attention.cu` — third-party design notice (SW-6131)

`sage_attention.cu` is a single-source nvrtc port, written for this workspace, of the INT8
`Q·Kᵀ` attention design in the two projects below. No upstream file is vendored verbatim; copied from
comfy-kitchen are the four sign-diagonal words of the Hadamard rotation, the `S_U8_OFFSET`
constant and the 16-key `inv_perm16` V permutation of its INT8 `P·V` mode.

| Upstream | Revision | Files the design follows | License |
|---|---|---|---|
| [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) | `6022b48b7e39fdd31f0124de268a8267af46d4e0` | `comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu`, `quant_v_int8.cu`, `sage_attn_launcher.cu`, `qk_int_sv_i8_cuda.cuh`, `attn_utils.cuh`, `mma.cuh`; `comfy_kitchen/sage_attention.py` | Apache-2.0 — Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES |
| [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) | `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` | the INT8-QK / 16-bit-PV algorithm (`sageattn_qk_int8_pv_fp16_cuda`) | Apache-2.0 — Copyright (c) 2024 Jintao Zhang, Haofeng Huang et al. |

Both are licensed under the Apache License, Version 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>), the license of this crate. What was taken and
what deliberately differs is listed in the header of `sage_attention.cu`.
