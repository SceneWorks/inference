//! SW-6137 (SW-F287 S12) attention rows: FlashAttention-2 (the dense path the opt-in kernel
//! replaces) vs the Sol-Attn sparse block-routed kernel at several `topk_ratio` budgets, each
//! through the shared `sdpa_planned_bhsd` helper, at Wan2.2 A14B self-attention shapes (40 heads x
//! 128, bf16, `[B, S, H, D]` projections viewed as `[B, H, S, D]`).
//!
//! Rows: `sol_attention topk_<ratio> (ref: flash)`. Every window is verified on
//! `candle_gen::fused_attention::AttentionPathCounts` ([`Bench::compare_with`]): the reference ran
//! only FlashAttention-2, the fused column only Sol-Attn (no Sol refusal). Each row's routed-pair
//! density and cosine / rel-L2 against the flash output go to stderr. The operands are random
//! (N(0, 0.25) Q/K): their attention is diffuse, so the vs-flash numbers are a worst case for block
//! routing, not the video quality (that is the Wan clip). Needs `candle-gen/flash-attn`.

use candle_gen::attention::{attention_budget_from_usize, sdpa_planned_bhsd};
use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::fused_attention::{
    attention_path_counts, with_sol_attention, SolConfig, FLASH_ATTN_COMPILED,
};
use candle_gen::gen_core::attention_budget::AttentionPlan;
use candle_quant_kernels::sol_attention::{similarity, sol_attention_with_stats};

use super::harness::{Bench, Row};
use super::int8_attention::{core, verify_paths, AttentionPath};

const RATIOS: [f32; 4] = [0.1, 0.25, 0.5, 1.0];

/// Wan2.2 A14B self-attention: 480x832 at 33 and 81 frames (9 / 21 latent frames of 30 x 52
/// tokens) and the 1280x720 x 81 ceiling (21 x 45 x 80).
const SHAPES: [(&str, usize); 3] = [
    ("wan_480p_33f", 9 * 30 * 52),
    ("wan_480p_81f", 21 * 30 * 52),
    ("wan_720p_81f", 21 * 45 * 80),
];

pub fn rows(bench: &Bench) -> Result<Vec<Row>> {
    if !FLASH_ATTN_COMPILED {
        candle_gen::candle_core::bail!("sol_attention rows need --features flash-attn");
    }
    let plan = AttentionPlan::budgeted(attention_budget_from_usize(candle_gen::ATTN_SCORES_BUDGET));
    let (h, d) = (40usize, 128usize);
    let mut rows = Vec::new();
    for (label, s) in SHAPES {
        let shape = format!("{label} [1,{h},{s},{d}] bf16");
        let operand = |sc: f64| -> Result<Tensor> {
            (Tensor::randn(0f32, 1.0, (1, s, h, d), &bench.dev)? * sc)?
                .to_dtype(DType::BF16)?
                .transpose(1, 2)
        };
        let (q, k, v) = (operand(0.5)?, operand(0.5)?, operand(1.0)?);
        let scale = 1.0 / (d as f64).sqrt();
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
        let reference = flash()?;
        for ratio in RATIOS {
            let config = SolConfig::top_k(ratio);
            let sol = || {
                core(with_sol_attention(config, || {
                    sdpa_planned_bhsd(&q, &k, &v, scale, None, softmax_last_dim, plan)
                }))
            };
            rows.push(bench.compare_with(
                &format!("sol_attention topk_{ratio} (ref: flash)"),
                &shape,
                attention_path_counts,
                verify_paths(AttentionPath::Flash, AttentionPath::Sol),
                || flash().map(drop),
                || sol().map(drop),
            )?);
            let (out, stats) = sol_attention_with_stats(&q, &k, &v, scale, &config)?;
            let (cos, rel) = similarity(&out, &reference)?;
            eprintln!(
                "sol_attention topk_{ratio} {shape}: density {:.4}, vs flash cosine {cos:.6}, \
                 rel-L2 {rel:.2e}",
                stats.density()
            );
        }
    }
    Ok(rows)
}
