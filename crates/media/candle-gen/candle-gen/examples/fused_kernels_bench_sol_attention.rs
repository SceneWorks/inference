//! SW-6137 (SW-F287 S12) attention rows for `fused_kernels_bench`: FlashAttention-2 (the dense path
//! the opt-in kernel replaces) and the Sol-Attn sparse kernel at several `topk_ratio` budgets, each
//! through the shared `sdpa_planned_bhsd` helper, at Wan2.2 A14B self-attention shapes (40 heads x
//! 128, bf16, `[B, S, H, D]` projections viewed as `[B, H, S, D]`).
//!
//! Self-contained so the shared bench (`examples/fused_kernels_bench.rs`, SW-6126) can absorb
//! [`sol_rows`] verbatim; until then it prints its own table:
//!
//! ```text
//! (vcvars) cargo run --release --example fused_kernels_bench_sol_attention -p candle-gen \
//!   --features flash-attn
//! ```
//!
//! Columns: kernel x shape, ms/iter (median of synchronized single calls; every kernel of a shape
//! alternates call by call because the dev box is power-capped), peak MiB (the CUDA pool's
//! `USED_MEM_HIGH` above the live operands), the routed-pair density, cosine / rel-L2 against the
//! flash output of the same inputs, and the device label. The operands are random (N(0, 0.25) Q/K):
//! their attention is diffuse, so the vs-flash columns are a worst case for block routing, not the
//! video quality (that is the Wan clip). Dev-box numbers are **sm_120** and are not extrapolated to
//! sm_89 (E11). `SOL_BENCH_ITERS` (default 5) and `SOL_BENCH_ROUNDS` (default 3) override the
//! sampling.

#[cfg(not(feature = "flash-attn"))]
fn main() {
    eprintln!("build with --features flash-attn (the flash column needs the kernel)");
}

/// One bench row.
#[cfg(feature = "flash-attn")]
#[derive(Debug, Clone)]
pub struct SolRow {
    /// `budgeted`, `flash` or `sol_topk_<ratio>`.
    pub kernel: String,
    /// The shape label.
    pub shape: String,
    /// Median ms per call.
    pub ms: f64,
    /// Peak MiB above the operands.
    pub peak_mib: f64,
    /// Routed `(query block, key block)` fraction (sol rows).
    pub density: Option<f64>,
    /// `(cosine, rel-L2)` against the flash output of the same inputs (non-flash rows).
    pub vs_flash: Option<(f32, f32)>,
    /// The device label (`sm_120` on the dev box).
    pub device: String,
}

#[cfg(feature = "flash-attn")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The SW-6137 attention rows on CUDA device `ordinal`.
#[cfg(feature = "flash-attn")]
pub fn sol_rows(ordinal: usize) -> candle_gen::Result<Vec<SolRow>> {
    use candle_gen::attention::{
        attention_budget_from_usize, sdpa_planned_bhsd, sdpa_planned_bhsd_materialized,
    };
    use candle_gen::candle_core::{DType, Device, Tensor};
    use candle_gen::candle_nn::ops::softmax_last_dim;
    use candle_gen::cuda_mempool::MemPool;
    use candle_gen::fused_attention::{attention_path_counts, with_sol_attention, SolConfig};
    use candle_gen::gen_core::attention_budget::AttentionPlan;
    use candle_quant_kernels::sol_attention::{similarity, sol_attention_with_stats};

    let dev = Device::new_cuda(ordinal)?;
    let device = candle_gen::cuda_arch::device_compute_cap(&dev)
        .map(|cap| format!("sm_{cap}"))
        .unwrap_or_else(|| "sm_unknown".into());
    let pool = MemPool::device_current(ordinal as i32).expect("the CUDA device's current pool");
    let (iters, rounds) = (
        env_usize("SOL_BENCH_ITERS", 5),
        env_usize("SOL_BENCH_ROUNDS", 3),
    );
    let plan = AttentionPlan::budgeted(attention_budget_from_usize(candle_gen::ATTN_SCORES_BUDGET));
    const RATIOS: [f32; 4] = [0.1, 0.25, 0.5, 1.0];
    let mut rows = Vec::new();
    // Wan2.2 A14B self-attention: 480x832 at 33 and 81 frames (9 / 21 latent frames of 30 x 52
    // tokens) and the 1280x720 x 81 ceiling (21 x 45 x 80).
    for (label, s, budgeted) in [
        ("wan_480p_33f", 9 * 30 * 52, true),
        ("wan_480p_81f", 21 * 30 * 52, false),
        ("wan_720p_81f", 21 * 45 * 80, false),
    ] {
        let (h, d) = (40usize, 128usize);
        let shape = format!("{label} [1,{h},{s},{d}] bf16");
        let operand = |sc: f64| -> candle_gen::Result<Tensor> {
            Ok((Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)? * sc)?
                .to_dtype(DType::BF16)?
                .transpose(1, 2)?)
        };
        let (q, k, v) = (operand(0.5)?, operand(0.5)?, operand(1.0)?);
        let scale = 1.0 / (d as f64).sqrt();
        let mut kernels: Vec<String> = vec!["flash".into()];
        kernels.extend(RATIOS.iter().map(|r| format!("sol_topk_{r}")));
        let ratio_of = |name: &str| {
            name.strip_prefix("sol_topk_")
                .map(|r| r.parse::<f32>().unwrap())
        };
        let run = |name: &str| -> candle_gen::Result<Tensor> {
            match ratio_of(name) {
                Some(r) => with_sol_attention(SolConfig::top_k(r), || {
                    sdpa_planned_bhsd(&q, &k, &v, scale, None, softmax_last_dim, plan)
                }),
                None if name == "budgeted" => {
                    sdpa_planned_bhsd_materialized(&q, &k, &v, scale, None, softmax_last_dim, plan)
                }
                None => sdpa_planned_bhsd(&q, &k, &v, scale, None, softmax_last_dim, plan),
            }
        };
        for kernel in &kernels {
            drop(run(kernel)?); // compile / warm
        }
        let mut samples = vec![Vec::new(); kernels.len()];
        let mut peaks = vec![0u64; kernels.len()];
        let mut timed = |i: usize, name: &str| -> candle_gen::Result<f64> {
            dev.synchronize()?;
            let base = pool.used().unwrap_or(0);
            pool.reset_used_high_water();
            let before = attention_path_counts();
            let t = std::time::Instant::now();
            drop(run(name)?);
            dev.synchronize()?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            if i < peaks.len() {
                peaks[i] = peaks[i].max(pool.used_high().unwrap_or(0).saturating_sub(base));
            }
            let delta = attention_path_counts().since(&before);
            let routed = match (name, ratio_of(name)) {
                (_, Some(_)) => delta.sol,
                ("budgeted", _) => delta.materialized,
                _ => delta.flash,
            };
            assert_eq!(routed, 1, "{name} did not take its path: {delta:?}");
            Ok(ms)
        };
        for _ in 0..rounds {
            for _ in 0..iters {
                for (i, kernel) in kernels.iter().enumerate() {
                    let ms = timed(i, kernel)?;
                    samples[i].push(ms);
                }
            }
        }
        let flash = run("flash")?;
        let budgeted_row = if budgeted {
            // The materialized reference is far slower: one synchronized call.
            let ms = timed(usize::MAX, "budgeted")?;
            dev.synchronize()?;
            let base = pool.used().unwrap_or(0);
            pool.reset_used_high_water();
            let out = run("budgeted")?;
            dev.synchronize()?;
            let peak = pool.used_high().unwrap_or(0).saturating_sub(base);
            Some(SolRow {
                kernel: "budgeted".into(),
                shape: shape.clone(),
                ms,
                peak_mib: peak as f64 / 1048576.0,
                density: None,
                vs_flash: Some(similarity(&out, &flash)?),
                device: device.clone(),
            })
        } else {
            None
        };
        let median = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(f64::total_cmp);
            s.get(s.len() / 2).copied().unwrap_or(f64::NAN)
        };
        for (i, kernel) in kernels.iter().enumerate() {
            let (density, vs_flash) = match ratio_of(kernel) {
                Some(r) => {
                    let (out, stats) =
                        sol_attention_with_stats(&q, &k, &v, scale, &SolConfig::top_k(r))?;
                    (Some(stats.density()), Some(similarity(&out, &flash)?))
                }
                None => (None, None),
            };
            rows.push(SolRow {
                kernel: kernel.clone(),
                shape: shape.clone(),
                ms: median(&samples[i]),
                peak_mib: peaks[i] as f64 / 1048576.0,
                density,
                vs_flash,
                device: device.clone(),
            });
        }
        rows.extend(budgeted_row);
    }
    Ok(rows)
}

#[cfg(feature = "flash-attn")]
fn main() -> candle_gen::Result<()> {
    let rows = sol_rows(0)?;
    println!("kernel | shape | ms/iter | peak MiB | density | vs flash | device");
    for r in &rows {
        let density = r.density.map(|d| format!("{d:.4}")).unwrap_or_default();
        let note = r
            .vs_flash
            .map(|(c, rel)| format!("cosine {c:.6}, rel-L2 {rel:.2e}"))
            .unwrap_or_default();
        println!(
            "{} | {} | {:.3} | {:.1} | {density} | {note} | {}",
            r.kernel, r.shape, r.ms, r.peak_mib, r.device
        );
    }
    Ok(())
}
