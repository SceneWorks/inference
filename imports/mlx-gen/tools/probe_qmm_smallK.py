"""Decisive base-Q8 probe (sc-2349 → sc-2532): is the Q8 residual the quantized_matmul KERNEL, and is
it K-dependent? sc-2532 proved quantize + qmm are byte-identical / 1e-6 for a K=3840 weight; the
bisection found the K=64 x-embedder diverges ~0.3%. This dumps, for the **real** K=64 x-embedder
weight + the **real** activation that feeds it (the golden's patchified init), the fork's
`mx.quantize` outputs and `mx.quantized_matmul` result. The Rust test `qmm_smallK_matches_fork`:
  (1) re-quantizes the same weight → must match wq byte-for-byte + scales/biases exactly (quantize),
  (2) runs quantized_matmul on the FORK's exact (wq, scales, biases, x) → if that diverges with
      byte-identical inputs, it's definitively the kernel/build (not our code), and K-dependent.

Run: cd ~/Repos/mflux-sc2257 && uv run python ~/Repos/mlx-gen/tools/probe_qmm_smallK.py
"""

import glob
import os

import mlx.core as mx

from _paths import hf_hub_cache

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

# Real K=64 x-embedder weight [3840, 64] (bf16 on disk) from the Z-Image-Turbo transformer.
tdir = glob.glob(str(hf_hub_cache() / "models--Tongyi-MAI--Z-Image-Turbo" / "snapshots/*/transformer"))[0]
w = None
for f in glob.glob(f"{tdir}/*.safetensors"):
    t = mx.load(f)
    if "all_x_embedder.2-1.weight" in t:
        w = t["all_x_embedder.2-1.weight"].astype(mx.bfloat16)
        break
assert w is not None and tuple(w.shape) == (3840, 64), f"got {None if w is None else w.shape}"

# Real activation: the golden's seeded init, patchified exactly like ZImageTransformer._patchify
# (pf=1, ph=pw=2) → x [4096, 64], f32 (the dtype the divergent Q8 control path actually runs).
g = mx.load(f"{D}/z_image_control_q8_golden.safetensors")
init = g["init"].astype(mx.float32)  # [16,1,128,128]
C, F, H, W = init.shape
pf, ph, pw = 1, 2, 2
Ft, Ht, Wt = F // pf, H // ph, W // pw
x = (
    init.reshape(C, Ft, pf, Ht, ph, Wt, pw)
    .transpose(1, 3, 5, 2, 4, 6, 0)
    .reshape(Ft * Ht * Wt, pf * ph * pw * C)
)
assert tuple(x.shape) == (4096, 64), x.shape

wq, scales, biases = mx.quantize(w, group_size=64, bits=8)
qmm = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=8)
mx.eval(wq, scales, biases, qmm)

out = {
    "w": w.astype(mx.float32),          # bf16-exact round-trip in Rust
    "x": x.astype(mx.float32),          # used as f32 directly
    "wq": wq,                            # uint32, exact
    "scales": scales.astype(mx.float32),
    "biases": biases.astype(mx.float32),
    "qmm": qmm.astype(mx.float32),
}
path = f"{D}/qmm_smallK_probe.safetensors"
mx.save_safetensors(path, out, {"K": "64", "M": str(x.shape[0]), "group_size": "64", "bits": "8"})
print(f"wrote {path}")
print(f"  w{tuple(w.shape)} x{tuple(x.shape)} wq{tuple(wq.shape)}({wq.dtype}) qmm{tuple(qmm.shape)}")
