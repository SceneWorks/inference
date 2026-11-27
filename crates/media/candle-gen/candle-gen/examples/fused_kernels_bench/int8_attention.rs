//! SW-6131 (SW-F287 S6) attention rows: the materialized (budgeted) reference, FlashAttention-2
//! and the opt-in INT8 QK kernel in both `P·V` modes, each through the shared `sdpa_planned_bhsd`
//! helper, at Krea 2 joint-sequence shapes (48 heads x 128, bf16, the production `[B, S, H, D]`
//! projections viewed as `[B, H, S, D]`).
//!
//! Rows (the `ref` column names its path in the kernel label):
//!
//! * `flash_attention (ref: materialized)`: budgeted SDPA vs FlashAttention-2;
//! * `int8_attention pv16 (ref: flash)`: FlashAttention-2 vs INT8 `Q·Kᵀ` with 16-bit `P·V`;
//! * `int8_attention pv8 (ref: flash)`: FlashAttention-2 vs comfy-kitchen's INT8 `P·V` as well.
//!
//! Attention kernels count in `candle_gen::fused_attention::AttentionPathCounts`, not the media
//! fused-path counters, so every window is verified there ([`Bench::compare_with`]): the reference
//! window ran only its path, the fused window only its kernel (no INT8 refusal). Each INT8 row's
//! cosine / rel-L2 against the flash output of the same inputs goes to stderr. Without
//! `candle-gen/flash-attn` the flash row is skipped and the INT8 rows compare against the
//! materialized path.

use candle_gen::attention::{
    attention_budget_from_usize, sdpa_planned_bhsd, sdpa_planned_bhsd_materialized,
};
use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::fused_attention::{
    attention_path_counts, with_int8_attention_config, AttentionPathCounts, Int8AttentionConfig,
    PvPrecision, FLASH_ATTN_COMPILED,
};
use candle_gen::gen_core::attention_budget::AttentionPlan;

use super::harness::{Bench, Column, Row};

/// The path one column must take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionPath {
    Materialized,
    Flash,
    Int8,
}

/// Calls of `path` in a window, and every other call (refusals of the opt-in kernel included).
pub fn path_calls(path: AttentionPath, d: &AttentionPathCounts) -> (u64, u64) {
    let fused_other = |own: u64| d.flash + d.int8 - own;
    match path {
        AttentionPath::Materialized => (d.materialized, d.total() - d.materialized),
        AttentionPath::Flash => (d.flash, d.total() - d.flash),
        AttentionPath::Int8 => (
            d.int8,
            fused_other(d.int8) + d.budgeted() + d.int8_refusals.total(),
        ),
    }
}

/// The [`Bench::compare_with`] verifier: each column ran only its path.
pub fn verify_paths(
    reference: AttentionPath,
    fused: AttentionPath,
) -> impl Fn(Column, &AttentionPathCounts, &AttentionPathCounts) -> std::result::Result<(), String>
{
    move |column, before, after| {
        let path = match column {
            Column::Reference => reference,
            Column::Fused => fused,
        };
        let delta = after.since(before);
        let (own, other) = path_calls(path, &delta);
        if own == 0 || other != 0 {
            return Err(format!(
                "did not run only {path:?} ({own} {path:?}, {other} other calls): {delta:?}"
            ));
        }
        Ok(())
    }
}

/// A `candle_gen` helper result as the harness's `candle_core` result.
pub fn core<T>(r: candle_gen::Result<T>) -> Result<T> {
    r.map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))
}

/// Cosine and relative L2 of `a` against `b`.
pub fn cosine_rel(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?;
    let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
    let na = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let nb = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let rel = (&a - &b)?.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? / nb;
    Ok((dot / (na * nb), rel))
}

/// Krea 2 joint sequences: 1024² = 4096 image tokens, 2048² = 16384, plus 256 text tokens.
const SHAPES: [(&str, usize); 2] = [("krea_1024", 4096 + 256), ("krea_2048", 16384 + 256)];

pub fn rows(bench: &Bench) -> Result<Vec<Row>> {
    let plan = AttentionPlan::budgeted(attention_budget_from_usize(candle_gen::ATTN_SCORES_BUDGET));
    let (h, d) = (48usize, 128usize);
    let mut rows = Vec::new();
    for (label, s) in SHAPES {
        let shape = format!("{label} [1,{h},{s},{d}] bf16");
        let operand = || -> Result<Tensor> {
            Tensor::randn(0f32, 1.0, (1, s, h, d), &bench.dev)?
                .to_dtype(DType::BF16)?
                .transpose(1, 2)
        };
        let (q, k, v) = (operand()?, operand()?, operand()?);
        let scale = 1.0 / (d as f64).sqrt();
        let materialized = || {
            core(sdpa_planned_bhsd_materialized(
                &q,
                &k,
                &v,
                scale,
                None,
                softmax_last_dim,
                plan,
            ))
        };
        let flash = || {
            core(sdpa_planned_bhsd(
                &q,
                &k,
                &v,
                scale,
                None,
                softmax_last_dim,
                plan,
            ))
        };
        let int8 = |pv| {
            core(with_int8_attention_config(
                Int8AttentionConfig {
                    pv,
                    ..Default::default()
                },
                || sdpa_planned_bhsd(&q, &k, &v, scale, None, softmax_last_dim, plan),
            ))
        };
        let (baseline, baseline_path, baseline_label) = if FLASH_ATTN_COMPILED {
            rows.push(bench.compare_with(
                "flash_attention (ref: materialized)",
                &shape,
                attention_path_counts,
                verify_paths(AttentionPath::Materialized, AttentionPath::Flash),
                || materialized().map(drop),
                || flash().map(drop),
            )?);
            (flash()?, AttentionPath::Flash, "flash")
        } else {
            eprintln!("flash-attn is not compiled in: INT8 rows compare against materialized");
            (materialized()?, AttentionPath::Materialized, "materialized")
        };
        let baseline_run = || match baseline_path {
            AttentionPath::Flash => flash(),
            _ => materialized(),
        };
        for (tag, pv) in [
            ("pv16", PvPrecision::ValueDtype),
            ("pv8", PvPrecision::Int8),
        ] {
            rows.push(bench.compare_with(
                &format!("int8_attention {tag} (ref: {baseline_label})"),
                &shape,
                attention_path_counts,
                verify_paths(baseline_path, AttentionPath::Int8),
                || baseline_run().map(drop),
                || int8(pv).map(drop),
            )?);
            let (cos, rel) = cosine_rel(&int8(pv)?, &baseline)?;
            eprintln!(
                "int8_attention {tag} {shape}: vs {baseline_label} cosine {cos:.6}, rel-L2 {rel:.2e}"
            );
        }
    }
    Ok(rows)
}
