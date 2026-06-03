//! Diagnostic for a pmetal bf16 **fused-SDPA** bug (sibling of the bf16-GEMM bug, sc-2714).
//!
//! On the **local NAX build**, `mlx::fast::scaled_dot_product_attention` with **bf16** q/k/v and
//! `mask=None` at the connector's shape `(1, 32, 128, 128)` (head_dim 128) returns GARBAGE
//! (≈1.0 mean-relative vs the f32 result) — while bf16 matmul (sc-2714) and bf16 *masked* SDPA
//! (the Gemma path) are correct. So `mlx-gen-ltx::connector` runs its SDPA in f32 (upcast q/k/v,
//! cast back). The f32 workaround is correct on **every** build, so it stays regardless.
//!
//! **`#[ignore]` — local-only.** The bug is hardware/build-specific: CI's Metal GPU runner has a
//! CORRECT bf16 maskless SDPA (≈3e-3), so an "assert the bug is present" check would (and did)
//! fail CI. Run this manually on the NAX machine to confirm the bug still warrants the workaround;
//! when it drops to the bf16-rounding floor there too, the f32-SDPA upcast in `connector::attn`
//! can be removed.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{abs, subtract, sum};
use mlx_rs::{random, Array, Dtype};

fn mean_rel(got: &Array, want: &Array) -> f32 {
    let g = got.as_dtype(Dtype::Float32).unwrap();
    let w = want.as_dtype(Dtype::Float32).unwrap();
    let num: f32 = sum(abs(subtract(&g, &w).unwrap()).unwrap(), None)
        .unwrap()
        .item();
    let den: f32 = sum(abs(&w).unwrap(), None).unwrap().item();
    num / den.max(1e-12)
}

#[test]
#[ignore = "local NAX-build diagnostic — CI's Metal runner has a correct bf16 maskless SDPA"]
fn bf16_sdpa_maskless_is_still_broken() {
    let shape = [1, 32, 128, 128];
    let q = random::normal::<f32>(&shape, None, None, None).unwrap();
    let k = random::normal::<f32>(&shape, None, None, None).unwrap();
    let v = random::normal::<f32>(&shape, None, None, None).unwrap();
    let scale = 1.0 / (128f32).sqrt();
    let f32_out = scaled_dot_product_attention(&q, &k, &v, scale, None, None).unwrap();
    let bf16_out = scaled_dot_product_attention(
        q.as_dtype(Dtype::Bfloat16).unwrap(),
        k.as_dtype(Dtype::Bfloat16).unwrap(),
        v.as_dtype(Dtype::Bfloat16).unwrap(),
        scale,
        None,
        None,
    )
    .unwrap();
    let mr = mean_rel(&bf16_out, &f32_out);
    eprintln!("bf16 maskless SDPA (1,32,128,128) mean_rel vs f32 = {mr:.3e}");
    // If this drops to the bf16-rounding floor (~1e-2), the kernel is fixed → remove the f32-SDPA
    // upcast in connector::attn and relax this guard.
    assert!(
        mr > 1e-1,
        "bf16 maskless SDPA looks FIXED (mean_rel {mr:.3e}) — drop the connector f32-SDPA upcast"
    );
}
