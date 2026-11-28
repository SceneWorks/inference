# `sol_attention.cu` — third-party source notice (SW-6137)

`sol_attention.cu` is a single-source nvrtc port of comfy-kitchen's Sol-Attn CUDA kernels. Its
kernels are **adapted from the upstream source** (renamed into one file, `cuda_bf16.h` types replaced
by the raw 16-bit helpers of `int8_attention_common.cuh`, byte rows packed in registers, the token
augmentation / chunked producer / key bias / padded-tile / coarse-gate / key-prefix paths removed, a
query-block window added to route and exact) — so the upstream copyright and license apply to it.

| Upstream | Revision | Files ported | License |
|---|---|---|---|
| [Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) | `6022b48b7e39fdd31f0124de268a8267af46d4e0` | `comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh`, `sol_attn.cu`, `sol_attn_preprocess.cu`, `sol_attn_vtranspose.cu`, `sol_attn_route.cu`, `sol_attn_exact.cu`; the top-k threshold of `comfy_kitchen/backends/cuda/__init__.py` (`_topk_from_pooled`); the eager reference `comfy_kitchen/backends/eager/sol_attn.py` (as `sol_attention_reference` in `sol_attention.rs`) | Apache-2.0 — Copyright (c) 2025 Comfy Org. All rights reserved. |

The algorithm is Sol-Attn, training-free sparse attention for video diffusion (arXiv 2607.24027),
as comfy-kitchen implements it (its top-k mode is the SLA-style per-query-block selection). The
shared device primitives prepended to the source (`int8_attention_common.cuh`) come from SW-6131's
`sage_attention.cu`; see `sage_attention-NOTICE.md`.

comfy-kitchen is licensed under the Apache License, Version 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>), the license of this crate. The modifications are
listed in the header of `sol_attention.cu`. Unless required by applicable law or agreed to in
writing, the upstream software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND.
