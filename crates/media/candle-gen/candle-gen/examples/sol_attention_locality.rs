//! SW-6137 locality probe: how the token order of a video self-attention maps onto Sol-Attn's
//! 64-token routing blocks. Loads one captured self-attention call (`q`, `k`, `v` `[1, H, S, 128]`
//! and `scale`, as the Wan driver's `--attn-capture` writes it), reorders the tokens of the
//! `[T, H, W]` latent grid into tile-major orders, runs the Sol-Attn kernel at several `topk_ratio`
//! budgets on each order, restores the raster order and compares with FlashAttention on the
//! original operands (cosine, rel-L2), plus the routed density.
//!
//! ```text
//! (vcvars) cargo run --release --example sol_attention_locality -p candle-gen --features flash-attn \
//!   -- <capture.safetensors> <T> <H> <W>
//! ```
//!
//! Tile `(tt, th, tw)` order: tiles in raster order over the grid, tokens raster inside each tile;
//! edge tiles are partial, so a 64-token block can straddle two of them.

#[cfg(not(feature = "flash-attn"))]
fn main() {
    eprintln!("build with --features flash-attn (the reference is FlashAttention)");
}

/// The tile-major permutation of a `[t, h, w]` raster grid: `perm[i]` is the raster index of the
/// i-th token in tile order.
#[cfg(feature = "flash-attn")]
fn tile_order(grid: (usize, usize, usize), tile: (usize, usize, usize)) -> Vec<u32> {
    let (t, h, w) = grid;
    let (tt, th, tw) = tile;
    let mut perm = Vec::with_capacity(t * h * w);
    for t0 in (0..t).step_by(tt) {
        for h0 in (0..h).step_by(th) {
            for w0 in (0..w).step_by(tw) {
                for f in t0..(t0 + tt).min(t) {
                    for y in h0..(h0 + th).min(h) {
                        for x in w0..(w0 + tw).min(w) {
                            perm.push(((f * h + y) * w + x) as u32);
                        }
                    }
                }
            }
        }
    }
    perm
}

#[cfg(feature = "flash-attn")]
fn main() -> candle_gen::Result<()> {
    use candle_gen::candle_core::{DType, Device, Tensor};
    use candle_gen::candle_nn::ops::softmax_last_dim;
    use candle_gen::fused_attention::SolConfig;
    use candle_quant_kernels::sol_attention::{similarity, sol_attention_with_stats};

    let args: Vec<String> = std::env::args().collect();
    let [_, path, t, h, w] = &args[..] else {
        eprintln!("usage: sol_attention_locality <capture.safetensors> <T> <H> <W>");
        std::process::exit(2);
    };
    let grid: (usize, usize, usize) = (t.parse().unwrap(), h.parse().unwrap(), w.parse().unwrap());
    let dev = Device::new_cuda(0)?;
    let tensors = candle_gen::candle_core::safetensors::load(path, &dev)?;
    let (q, k, v) = (&tensors["q"], &tensors["k"], &tensors["v"]);
    let scale = tensors["scale"].to_vec1::<f32>()?[0] as f64;
    let s = q.dim(2)?;
    assert_eq!(
        s,
        grid.0 * grid.1 * grid.2,
        "the grid must cover the sequence"
    );
    let plan = candle_gen::gen_core::attention_budget::AttentionPlan::budgeted(
        candle_gen::attention::attention_budget_from_usize(candle_gen::ATTN_SCORES_BUDGET),
    );
    let dense =
        candle_gen::attention::sdpa_planned_bhsd(q, k, v, scale, None, softmax_last_dim, plan)?;
    println!(
        "capture {path}: {:?} {:?}, scale {scale}, grid {grid:?}",
        q.dims(),
        q.dtype()
    );
    println!("order | tile | topk_ratio | density | cosine vs flash | rel-L2 vs flash");
    let mut orders: Vec<(&str, (usize, usize, usize))> = vec![("raster", (1, 1, 1))];
    orders.extend([
        ("tile", (1, 8, 8)),
        ("tile", (1, 4, 16)),
        ("tile", (4, 4, 4)),
        ("tile", (2, 4, 8)),
    ]);
    for (name, tile) in orders {
        let (pq, pk, pv, inverse) = if name == "raster" {
            (q.clone(), k.clone(), v.clone(), None)
        } else {
            let perm = tile_order(grid, tile);
            let mut inv = vec![0u32; perm.len()];
            for (i, &p) in perm.iter().enumerate() {
                inv[p as usize] = i as u32;
            }
            let perm = Tensor::from_vec(perm, s, &dev)?;
            let inv = Tensor::from_vec(inv, s, &dev)?;
            let g = |x: &Tensor| -> candle_gen::Result<Tensor> {
                Ok(x.index_select(&perm, 2)?.contiguous()?)
            };
            (g(q)?, g(k)?, g(v)?, Some(inv))
        };
        for ratio in [0.1f32, 0.25, 0.5] {
            let (out, stats) =
                sol_attention_with_stats(&pq, &pk, &pv, scale, &SolConfig::top_k(ratio))?;
            let out = match &inverse {
                Some(inv) => out.index_select(inv, 2)?,
                None => out,
            };
            let (cos, rel) = similarity(&out.to_dtype(DType::F32)?, &dense.to_dtype(DType::F32)?)?;
            println!(
                "{name} | {tile:?} | {ratio} | {:.4} | {cos:.6} | {rel:.4e}",
                stats.density()
            );
        }
    }
    Ok(())
}
