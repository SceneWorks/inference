//! SW-6131 (SW-F287 S6) attention rows for `fused_kernels_bench`: the materialized (budgeted)
//! reference, FlashAttention-2 and the opt-in INT8 QK kernel, each through the shared
//! `sdpa_planned_bhsd` helper, at Krea 2 joint-sequence shapes (48 heads x 128, bf16).
//!
//! Self-contained so the shared bench (`examples/fused_kernels_bench.rs`, SW-6126) can absorb
//! [`attention_rows`] verbatim; until then it prints its own table:
//!
//! ```text
//! (vcvars) cargo run --release --example fused_kernels_bench_int8_attention -p candle-gen \
//!   --features flash-attn
//! ```
//!
//! Columns: kernel x shape, ms/iter (median of synchronized single calls; flash and INT8 alternate
//! call by call because the dev box is power-capped), peak MiB (the CUDA
//! pool's `USED_MEM_HIGH` above the live operands), and the device label. Dev-box numbers are
//! **sm_120** and are not extrapolated to sm_89 (E11). `INT8_BENCH_ITERS` (default 10) and
//! `INT8_BENCH_ROUNDS` (default 5) override the sampling.

#[cfg(not(feature = "flash-attn"))]
fn main() {
    eprintln!("build with --features flash-attn (the flash column needs the kernel)");
}

/// One bench row.
#[cfg(feature = "flash-attn")]
#[derive(Debug, Clone)]
pub struct AttentionRow {
    /// `budgeted`, `flash`, `int8` (16-bit P.V) or `int8_pv8` (INT8 P.V).
    pub kernel: &'static str,
    /// The shape label.
    pub shape: String,
    /// Median ms per call.
    pub ms: f64,
    /// Peak MiB above the operands.
    pub peak_mib: f64,
    /// Median ms per call over a burst of 28 back-to-back calls with one trailing sync (how a DiT
    /// forward issues its block attentions; host-side overhead shows here, not in `ms`).
    pub burst_ms: f64,
    /// For `int8`: `(cosine, rel-L2)` against the flash output of the same inputs.
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

/// The SW-6131 attention rows on CUDA device `ordinal`.
#[cfg(feature = "flash-attn")]
pub fn attention_rows(ordinal: usize) -> candle_gen::Result<Vec<AttentionRow>> {
    use candle_gen::attention::{
        attention_budget_from_usize, sdpa_planned_bhsd, sdpa_planned_bhsd_materialized,
    };
    use candle_gen::candle_core::{DType, Device, Tensor};
    use candle_gen::candle_nn::ops::softmax_last_dim;
    use candle_gen::cuda_mempool::MemPool;
    use candle_gen::fused_attention::{
        attention_path_counts, with_int8_attention_config, Int8AttentionConfig, PvPrecision,
    };
    use candle_gen::gen_core::attention_budget::AttentionPlan;

    let dev = Device::new_cuda(ordinal)?;
    let device = candle_gen::cuda_arch::device_compute_cap(&dev)
        .map(|cap| format!("sm_{cap}"))
        .unwrap_or_else(|| "sm_unknown".into());
    let pool = MemPool::device_current(ordinal as i32).expect("the CUDA device's current pool");
    let (iters, rounds) = (
        env_usize("INT8_BENCH_ITERS", 10),
        env_usize("INT8_BENCH_ROUNDS", 5),
    );
    let plan = AttentionPlan::budgeted(attention_budget_from_usize(candle_gen::ATTN_SCORES_BUDGET));
    let mut rows = Vec::new();
    // Krea 2 joint sequences: 1024² = 4096 image tokens, 2048² = 16384, plus 256 text tokens.
    for (label, s) in [("krea_1024", 4096 + 256), ("krea_2048", 16384 + 256)] {
        let (h, d) = (48usize, 128usize);
        let shape = format!("{label} [1,{h},{s},{d}] bf16");
        // Krea's production layout: `[B, S, H, D]` projections viewed as `[B, H, S, D]` (no copy).
        let operand = || -> candle_gen::Result<Tensor> {
            Ok(Tensor::randn(0f32, 1.0, (1, s, h, d), &dev)?
                .to_dtype(DType::BF16)?
                .transpose(1, 2)?)
        };
        let (q, k, v) = (operand()?, operand()?, operand()?);
        let scale = 1.0 / (d as f64).sqrt();
        let int8 = |pv| {
            with_int8_attention_config(
                Int8AttentionConfig {
                    pv,
                    ..Default::default()
                },
                || sdpa_planned_bhsd(&q, &k, &v, scale, None, softmax_last_dim, plan),
            )
        };
        let run = |kernel: &str| -> candle_gen::Result<Tensor> {
            match kernel {
                "budgeted" => {
                    sdpa_planned_bhsd_materialized(&q, &k, &v, scale, None, softmax_last_dim, plan)
                }
                "flash" => sdpa_planned_bhsd(&q, &k, &v, scale, None, softmax_last_dim, plan),
                "int8" => int8(PvPrecision::ValueDtype),
                _ => int8(PvPrecision::Int8),
            }
        };
        // `int8`: INT8 Q.K^T, 16-bit P.V; `int8_pv8`: comfy-kitchen's INT8 P.V as well.
        const KERNELS: [&str; 4] = ["budgeted", "flash", "int8", "int8_pv8"];
        let mut samples = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut peaks = [0u64; 4];
        for kernel in KERNELS {
            drop(run(kernel)?); // compile / warm
        }
        // One synchronized call, timed; its peak above the live operands; its route checked.
        let mut timed = |i: usize| -> candle_gen::Result<()> {
            let kernel = KERNELS[i];
            dev.synchronize()?;
            let base = pool.used().unwrap_or(0);
            pool.reset_used_high_water();
            let before = attention_path_counts();
            let t = std::time::Instant::now();
            drop(run(kernel)?);
            dev.synchronize()?;
            samples[i].push(t.elapsed().as_secs_f64() * 1e3);
            peaks[i] = peaks[i].max(pool.used_high().unwrap_or(0).saturating_sub(base));
            let delta = attention_path_counts().since(&before);
            let routed = match kernel {
                "flash" => delta.flash,
                "budgeted" => delta.materialized,
                _ => delta.int8,
            };
            assert_eq!(routed, 1, "{kernel} did not take its path: {delta:?}");
            Ok(())
        };
        // The materialized reference is ~10x slower: one call per round.
        const BURST: usize = 28;
        let mut bursts = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..rounds {
            timed(0)?;
            // The fused kernels alternate call by call, so clock and power drift on this
            // power-capped board lands on each alike.
            for _ in 0..iters {
                for i in 1..KERNELS.len() {
                    timed(i)?;
                }
            }
            for i in 1..KERNELS.len() {
                dev.synchronize()?;
                let t = std::time::Instant::now();
                let mut outs = Vec::with_capacity(BURST);
                for _ in 0..BURST {
                    outs.push(run(KERNELS[i])?);
                }
                dev.synchronize()?;
                bursts[i].push(t.elapsed().as_secs_f64() * 1e3 / BURST as f64);
            }
        }
        let flash = run("flash")?.to_dtype(DType::F32)?.flatten_all()?;
        let vs_flash = |kernel: &str| -> candle_gen::Result<(f32, f32)> {
            let a = run(kernel)?.to_dtype(DType::F32)?.flatten_all()?;
            let b = &flash;
            let dot = (&a * b)?.sum_all()?.to_scalar::<f32>()?;
            let na = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            let nb = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            let rel = (&a - b)?.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? / nb;
            Ok((dot / (na * nb), rel))
        };
        let median = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(f64::total_cmp);
            s.get(s.len() / 2).copied().unwrap_or(f64::NAN)
        };
        for (i, kernel) in KERNELS.iter().enumerate() {
            rows.push(AttentionRow {
                kernel,
                shape: shape.clone(),
                ms: median(&samples[i]),
                peak_mib: peaks[i] as f64 / 1048576.0,
                burst_ms: median(&bursts[i]),
                vs_flash: if kernel.starts_with("int8") {
                    Some(vs_flash(kernel)?)
                } else {
                    None
                },
                device: device.clone(),
            });
        }
    }
    Ok(rows)
}

#[cfg(feature = "flash-attn")]
fn main() -> candle_gen::Result<()> {
    let rows = attention_rows(0)?;
    println!("kernel | shape | ms/iter | burst ms/iter | peak MiB | device | note");
    for r in &rows {
        let note = r
            .vs_flash
            .map(|(c, rel)| format!("vs flash: cosine {c:.6}, rel-L2 {rel:.2e}"))
            .unwrap_or_default();
        println!(
            "{} | {} | {:.3} | {:.3} | {:.1} | {} | {note}",
            r.kernel, r.shape, r.ms, r.burst_ms, r.peak_mib, r.device
        );
    }
    Ok(())
}
