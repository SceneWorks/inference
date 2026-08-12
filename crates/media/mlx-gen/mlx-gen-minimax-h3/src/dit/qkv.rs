//! **Rule 2 made executable: the DiT's fused-QKV transform is NOT the video VAE's.**
//!
//! [`crate::layout`] states the rule; this module is the code that makes it testable, and
//! `tests/dit_parity.rs` runs both transforms against both source layouts to prove they are
//! genuinely different partitions rather than two names for one gather.
//!
//! # What the published DiT actually ships
//!
//! **Nothing.** `to_qkv` / `qkv_proj` appears nowhere in
//! `transformer/diffusion_pytorch_model.safetensors.index.json` — the conversion already split the
//! fused projection, so [`crate::dit::DitAttention`] loads `to_q` / `to_k` / `to_v` verbatim and
//! applies **no** transform at all. That is the whole reason this module exists as a *contract*
//! rather than as loader code: the transform is invisible in the port, so only an explicit
//! assertion can pin which one produced the bytes.
//!
//! # The two transforms, and why confusing them is silent
//!
//! `scripts/convert_minimax_h3_to_diffusers.py` composes two steps for the DiT:
//!
//! ```text
//! raw shard rows          reorder_interleaved_qkv        reference state_dict        split_fused_qkv
//! [h0: q k v,       ──────────────────────────────►  [q_all; k_all; v_all]  ──────────────────────►  to_q / to_k / to_v
//!  h1: q k v, …]      (`_reorder_grouped_qkv_to_qkv`)                            (contiguous thirds)
//! ```
//!
//! * A fixture dumped from `MiniMaxH3Transformer3DModel.state_dict()` sits at the **middle** node,
//!   so it needs [`split_thirds`] — contiguous thirds.
//! * A fixture built from the **raw** MiniMax shards sits at the left node, so it needs the
//!   per-head interleaved gather, which is [`crate::vae::split_fused_qkv`] — the video VAE's
//!   transform, because the VAE reference applies no load-time reorder and therefore holds the
//!   interleaved rows in memory.
//!
//! Both produce a `[heads·head_dim, …]` tensor from a `[3·heads·head_dim, …]` one, so **the wrong
//! choice is shape-identical and silent** — the sc-18740 failure mode. Applying `split_thirds` to
//! raw rows yields "the first third of the heads' q, k and v" rather than "every head's q", which
//! is a plausible model that is wrong on real weights.

use mlx_rs::Array;

use mlx_gen::{Error, Result};

/// Split a `[q_all; k_all; v_all]` fused projection into `to_q` / `to_k` / `to_v` — **contiguous
/// thirds along the row axis**.
///
/// This is `split_fused_qkv` in the official conversion script, and it is the transform the **DiT**
/// needs, because the reference's `_reorder_grouped_qkv_to_qkv` already de-interleaved the raw
/// rows before `state_dict()` ever sees them.
///
/// It is deliberately **not** [`crate::vae::split_fused_qkv`], which gathers per-head-interleaved
/// rows and is the video VAE's transform. See the module docs.
pub fn split_thirds(fused: &Array, heads: i32, head_dim: i32) -> Result<[Array; 3]> {
    let s = fused.shape();
    let inner = heads * head_dim;
    if s.is_empty() || s[0] != 3 * inner {
        return Err(Error::Msg(format!(
            "minimax-h3 dit qkv: expected leading dim {}, got {s:?}",
            3 * inner
        )));
    }
    let parts = fused.split_axis(&[inner, 2 * inner], 0)?;
    let mut it = parts.into_iter();
    Ok([
        it.next().expect("3 splits"),
        it.next().expect("3 splits"),
        it.next().expect("3 splits"),
    ])
}

/// Reorder raw per-head-interleaved fused QKV rows into the reference's in-memory
/// `[q_all; k_all; v_all]` layout — `reorder_interleaved_qkv` in the conversion script, which is
/// `_reorder_grouped_qkv_to_qkv` in the reference loader.
///
/// Provided so the *whole* published pipeline is expressible here: `split_thirds(reorder(raw))` is
/// bit-identical to `crate::vae::split_fused_qkv(raw)`, and `tests/dit_parity.rs` asserts exactly
/// that against real reference bytes rather than trusting the script's comment.
pub fn reorder_interleaved(fused: &Array, heads: i32, head_dim: i32) -> Result<Array> {
    let s = fused.shape();
    if s.is_empty() || s[0] != 3 * heads * head_dim {
        return Err(Error::Msg(format!(
            "minimax-h3 dit qkv reorder: expected leading dim {}, got {s:?}",
            3 * heads * head_dim
        )));
    }
    let mut grouped = vec![heads, 3, head_dim];
    grouped.extend_from_slice(&s[1..]);
    let view = fused.reshape(&grouped)?;
    let mut flat = vec![heads * head_dim];
    flat.extend_from_slice(&s[1..]);

    let mut parts = Vec::with_capacity(3);
    for j in 0..3 {
        let idx = Array::from_slice(&[j], &[1]);
        parts.push(view.take_axis(&idx, 1)?.reshape(&flat)?);
    }
    Ok(mlx_rs::ops::concatenate_axis(&parts, 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::vae::split_fused_qkv as split_interleaved;

    /// A raw fused tensor whose every row is uniquely identifiable: row
    /// `h·(3·D) + j·D + d` carries `100·h + 10·j + d`.
    fn raw(heads: i32, head_dim: i32) -> Array {
        let mut v = Vec::new();
        for h in 0..heads {
            for j in 0..3 {
                for d in 0..head_dim {
                    v.push((100 * h + 10 * j + d) as f32);
                }
            }
        }
        Array::from_slice(&v, &[3 * heads * head_dim, 1])
    }

    /// The DiT's transform on the DiT's source layout: contiguous thirds of
    /// `[q_all; k_all; v_all]`.
    #[test]
    fn split_thirds_takes_contiguous_thirds() {
        let heads = 3;
        let head_dim = 2;
        let reordered = reorder_interleaved(&raw(heads, head_dim), heads, head_dim).unwrap();
        let [q, k, v] = split_thirds(&reordered, heads, head_dim).unwrap();
        // q_all is every head's `j = 0` rows, in head order.
        assert_eq!(q.as_slice::<f32>(), &[0.0, 1.0, 100.0, 101.0, 200.0, 201.0]);
        assert_eq!(
            k.as_slice::<f32>(),
            &[10.0, 11.0, 110.0, 111.0, 210.0, 211.0]
        );
        assert_eq!(
            v.as_slice::<f32>(),
            &[20.0, 21.0, 120.0, 121.0, 220.0, 221.0]
        );
    }

    /// **The reason this module exists.** The composed pipeline agrees with the video VAE's
    /// single-step gather, but only when each transform is applied to ITS OWN source layout.
    /// Crossing them is shape-identical and wrong.
    #[test]
    fn the_two_transforms_are_different_partitions() {
        let heads = 4;
        let head_dim = 3;
        let raw = raw(heads, head_dim);
        let reordered = reorder_interleaved(&raw, heads, head_dim).unwrap();

        // Correct compositions agree.
        let via_thirds = split_thirds(&reordered, heads, head_dim).unwrap();
        let via_gather = split_interleaved(&raw, heads, head_dim).unwrap();
        for i in 0..3 {
            assert_eq!(
                via_thirds[i].as_slice::<f32>(),
                via_gather[i].as_slice::<f32>(),
                "split_thirds∘reorder must equal the VAE's interleaved gather on raw rows"
            );
        }

        // Crossing them: same shapes, different bytes.
        let wrong_thirds = split_thirds(&raw, heads, head_dim).unwrap();
        let wrong_gather = split_interleaved(&reordered, heads, head_dim).unwrap();
        for i in 0..3 {
            assert_eq!(
                wrong_thirds[i].shape(),
                via_thirds[i].shape(),
                "shape-identical"
            );
            assert_eq!(
                wrong_gather[i].shape(),
                via_thirds[i].shape(),
                "shape-identical"
            );
            assert_ne!(
                wrong_thirds[i].as_slice::<f32>(),
                via_thirds[i].as_slice::<f32>(),
                "contiguous thirds of RAW rows must not accidentally equal the correct split"
            );
            assert_ne!(
                wrong_gather[i].as_slice::<f32>(),
                via_thirds[i].as_slice::<f32>(),
                "an interleaved gather of ALREADY-REORDERED rows must not equal the correct split"
            );
        }
    }

    #[test]
    fn a_wrong_leading_dim_is_rejected() {
        let t = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5, 1]);
        assert!(split_thirds(&t, 2, 1).is_err());
        assert!(reorder_interleaved(&t, 2, 1).is_err());
    }
}
