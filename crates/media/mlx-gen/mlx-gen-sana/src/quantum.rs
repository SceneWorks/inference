//! SANA-specific iOS quantum accounting (trunk block, trunk entry/output, DC-AE stages, the tiled
//! decode/encode folds), built on the generic counted API in [`mlx_gen::ios_quantum`].
//!
//! Every figure is COUNTED from the ops of one quantum — how many tensors of each shape are live at
//! that quantum's peak — and each is proven by a CPU test (`quantum_proofs` below) that builds the
//! real module at a small shape, measures MLX's peak with
//! [`mlx_gen::ios_quantum::measured_peak_bytes`] and asserts the formula is at least the
//! measurement and at most [`MAX_RATIO`] times it. These are floors for one materialized quantum,
//! never a process bound and never production admission.
//!
//! **Convolution workspace.** MLX's CPU backend runs every ungrouped convolution as an explicit
//! GEMM: a zero-padded f32 copy of the input plus the unfolded `[outH·outW, k·k·C_in]` patch matrix
//! (and an f32 copy of a narrower weight). Metal picks implicit GEMM (no workspace), Winograd
//! (`~64/36` of the padded input and of the output) or the same explicit unfold, so the CPU unfold
//! bounds every backend path; [`conv_workspace_bytes`] prices it. Grouped/depthwise convolutions
//! take MLX's direct (unfold-free) path on both backends.

use mlx_gen::ios_quantum::{
    attention_block_bytes, conv_stage_bytes, scores_bytes, tensor_bytes, AttentionBlockGeometry,
    ConvStageGeometry, Scores,
};

/// Maximum formula/measurement ratio every SANA/DC-AE proof accepts (never above 2.0).
pub const MAX_RATIO: f64 = 1.5;

fn sum(values: &[u64]) -> u64 {
    values.iter().fold(0u64, |a, b| a.saturating_add(*b))
}

/// The workspace of one ungrouped `k×k` convolution (see the module doc), the larger of MLX's two
/// allocating paths: the CPU explicit GEMM's f32 zero-padded input plus its f32 unfolded patch
/// matrix at the output resolution, and Metal's Winograd F(6x6, 3x3) — taken only for a 3x3
/// stride-1 conv whose input and output channels are both multiples of 32 — the padded input plus
/// its `64/36` transformed copy, and the `64/36` transformed output.
pub fn conv_workspace_bytes(
    batch: u64,
    in_hw: (u64, u64),
    out_hw: (u64, u64),
    in_channels: u64,
    out_channels: u64,
    kernel: u64,
    padding: u64,
) -> u64 {
    let padded = tensor_bytes(
        4,
        &[
            batch,
            in_hw.0.saturating_add(2 * padding),
            in_hw.1.saturating_add(2 * padding),
            in_channels,
        ],
    );
    let unfolded = tensor_bytes(4, &[batch, out_hw.0, out_hw.1, kernel, kernel, in_channels]);
    let cpu = padded.saturating_add(unfolded);
    let winograd =
        if kernel == 3 && in_channels.is_multiple_of(32) && out_channels.is_multiple_of(32) {
            let out = tensor_bytes(4, &[batch, out_hw.0, out_hw.1, out_channels]);
            sum(&[
                padded,
                padded.saturating_mul(16).div_ceil(9),
                out.saturating_mul(16).div_ceil(9),
            ])
        } else {
            0
        };
    cpu.max(winograd)
}

// =================================================================================================
// Linear-DiT trunk.
// =================================================================================================

/// One SANA transformer block quantum, from the loaded block and the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SanaBlockQuantum {
    pub batch: u64,
    /// Latent tokens `H·W`.
    pub tokens: u64,
    /// Block width.
    pub dim: u64,
    /// Linear self-attention heads and head width (its `[hd, hd]` gram per head).
    pub self_heads: u64,
    pub self_head_dim: u64,
    /// Caption tokens attended by the cross-attention.
    pub caption_tokens: u64,
    pub cross_heads: u64,
    /// Cross-attention query rows scored at once (`tokens` unchunked; the rung-3 chunk otherwise).
    pub query_rows: u64,
    /// GLUMBConv expanded channels (`2 · hidden`).
    pub ff_channels: u64,
    /// Sprint's RMSNorm across heads on q/k of both attentions.
    pub qk_norm: bool,
    /// Activation width (4: the trunk runs at the f32 latent's dtype).
    pub element_bytes: u64,
    /// Every weight copy a forward creates, summed (a stored-narrower dense weight promoted to the
    /// activation width by its op; all are live together under MLX's encode-ahead — see
    /// `SanaBlock::quantum`); 0 when every weight is used in place.
    pub weight_bytes: u64,
}

/// Linear self-attention phase, in `[B, N, dim]` buffers: the residual, the LayerNorm's three
/// buffers (MLX's unfused CPU fallback holds the centred input, its scaled product and the
/// output), the modulated input, the q/k/v projections (3), the ReLU'd q and k (2), the contiguous
/// copies the two GEMMs make of their head-major operands (3), the numerator and the quotient — 15
/// (measured 14.7). Sprint's qk RMSNorms (`x²`, normalized product, weighted output and the f32
/// statistics) add 4 (measured +4.0 for the attention alone). Its per-head `[hd, hd]` key-value
/// gram and `[1, N]` denominators are priced exactly below.
const SELF_ATTN_HIDDEN_LIVE: u64 = 15;
const SELF_ATTN_QK_NORM_LIVE: u64 = 4;
/// Cross-attention, `[B, N, dim]`: the residual, the query projection and its bias sum, the
/// head-major query chunk's contiguous copy, the per-chunk contexts and their concatenation (2),
/// the token-major copy and the output projection — 8 (measured 6.25 beside the logits for the
/// attention alone). Sprint's query RMSNorm (3 buffers) runs before the scores beside only the
/// residual and the projection, inside the same count.
const CROSS_ATTN_HIDDEN_LIVE: u64 = 8;
/// `[B, M, dim]`: the key and value projections (2); Sprint's key RMSNorm holds three more.
const CROSS_CONTEXT_LIVE: u64 = 2;
/// `[B, heads, rows, M]` f32 logits. Each scale/mask/softmax step reuses its predecessor's buffer
/// (same shape and dtype, sole reference), so one chunk's logits are ONE buffer (measured 1.0 for
/// the attention alone); a second is live when the previous chunk's buffer — or the unchunked
/// buffer's predecessor — is still held by a command MLX has dispatched but not yet completed
/// (measured 2.0 at two 512-row chunks).
const CROSS_SCORES_LIVE: u64 = 2;
/// Mix-FFN phase, `[B, N, dim]`: the residual, the LayerNorm's three CPU buffers and the modulated
/// input (5).
const FFN_HIDDEN_LIVE: u64 = 5;
/// `[B, N, 2·hidden]`: the inverted conv output and its bias sum, the SiLU, the depthwise output and
/// its bias sum (MLX does not reuse a convolution's output for its bias add), and the gated
/// half-width product with the SiLU'd half (two halves = 1) — 6. Measured 5.2 for the GLUMBConv
/// alone (including the 1x1 point conv's workspace, priced separately below) and up to 6.9 for the
/// whole phase, whose CPU release order varies run to run.
const FFN_LIVE: u64 = 6;

/// `weight_bytes + max(self-attention, cross-attention, logits + Mix-FFN)` — the three phases run
/// one after another inside the block's single quantum, but MLX dispatches asynchronously and
/// releases a buffer only when the command holding it completes, so the cross-attention's logits
/// can still be resident while the Mix-FFN allocates (measured: a block whose cross-attention alone
/// peaks at 6.6 MB and whose Mix-FFN alone peaks at ~9 MB peaks at 14.5 MB).
pub fn block_bytes(q: SanaBlockQuantum) -> u64 {
    let norm = u64::from(q.qk_norm);
    let base = AttentionBlockGeometry {
        batch: q.batch,
        tokens: q.tokens,
        dim: q.dim,
        heads: q.cross_heads,
        context_tokens: 0,
        ff_hidden: 0,
        query_rows: q.query_rows,
        element_bytes: q.element_bytes,
        weight_bytes: 0,
        attn_hidden_live: 0,
        context_live: 0,
        scores: Scores::Fused,
        mlp_hidden_live: 0,
        ff_live: 0,
    };
    let self_attention = attention_block_bytes(AttentionBlockGeometry {
        attn_hidden_live: SELF_ATTN_HIDDEN_LIVE + norm * SELF_ATTN_QK_NORM_LIVE,
        ..base
    })
    .saturating_add(tensor_bytes(
        4,
        &[2, q.batch, q.self_heads, q.self_head_dim, q.self_head_dim],
    ))
    .saturating_add(tensor_bytes(4, &[2, q.batch, q.self_heads, q.tokens]));
    // Cross-attention: the caption-side k/v, the query-side buffers and the logits.
    let logits = |live: u64| {
        scores_bytes(
            Scores::Materialized {
                live,
                element_bytes: 4,
            },
            q.batch,
            q.cross_heads,
            q.query_rows,
            q.caption_tokens,
        )
    };
    let cross = sum(&[
        tensor_bytes(
            q.element_bytes,
            &[
                CROSS_CONTEXT_LIVE + 3 * norm,
                q.batch,
                q.caption_tokens,
                q.dim,
            ],
        ),
        tensor_bytes(
            q.element_bytes,
            &[CROSS_ATTN_HIDDEN_LIVE, q.batch, q.tokens, q.dim],
        ),
        logits(CROSS_SCORES_LIVE),
    ]);
    let half = q.ff_channels / 2;
    let ffn = attention_block_bytes(AttentionBlockGeometry {
        ff_hidden: q.ff_channels,
        mlp_hidden_live: FFN_HIDDEN_LIVE,
        ff_live: FFN_LIVE,
        ..base
    })
    .saturating_add(
        conv_workspace_bytes(
            q.batch,
            (1, q.tokens),
            (1, q.tokens),
            q.dim,
            q.ff_channels,
            1,
            0,
        )
        .max(conv_workspace_bytes(
            q.batch,
            (1, q.tokens),
            (1, q.tokens),
            half,
            q.dim,
            1,
            0,
        )),
    );
    q.weight_bytes
        .saturating_add(self_attention.max(cross).max(logits(1).saturating_add(ffn)))
}

/// The trunk's entry quantum (patchify, timestep/guidance embedding, caption projection).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SanaEntryQuantum {
    pub batch: u64,
    pub latent_channels: u64,
    /// Patch tokens `(H/p)·(W/p)`.
    pub tokens: u64,
    pub patch: u64,
    pub dim: u64,
    pub caption_tokens: u64,
    /// Width of the incoming caption embedding (Gemma's hidden).
    pub caption_channels: u64,
    /// Element width of the incoming caption (Gemma's dtype).
    pub caption_element_bytes: u64,
    pub element_bytes: u64,
    /// The promoted weight copies of the entry's dense projections and patch conv, summed.
    pub weight_bytes: u64,
}

/// `[B, M, dim]` f32 buffers live in the caption projection: the first projection (held by
/// gelu-tanh to its final product), gelu's chain (two, each step reusing its predecessor) and its
/// `0.5·x` factor, then the output — five; the second projection and its RMSNorm (`x²`, the
/// normalized product, the weighted output) never exceed that once gelu's buffers are released.
const ENTRY_CAPTION_LIVE: u64 = 5;
/// `[B, dim]`-sized timestep/guidance buffers: sincos (256 ≤ dim), two MLP stages and SiLU per
/// embedder (Sprint: two embedders and their sum), the combined embedding and the `6·dim` modulation.
const ENTRY_TIMESTEP_DIMS: u64 = 16;

/// `weight_bytes + patchify + caption projection + timestep buffers`. The four outputs are one
/// graph whose patchify and caption chains are independent, so MLX may interleave them: their
/// transients are SUMMED, not maxed. Patchify: the 1x1 conv's output and its bias sum (2 hidden)
/// plus its input workspace ([`conv_workspace_bytes`]): the CPU reads the transposed (NHWC view of
/// the NCHW) latent straight into its zero-padded copy and unfolds that; Metal makes the view
/// row-contiguous and unfolds (or runs implicit GEMM on) that copy. At padding 0 both are two
/// latent-sized f32 buffers, so the workspace bounds both without a separate transposed copy. Caption: `ENTRY_CAPTION_LIVE` `[B, M, dim]` buffers or, for a
/// caption narrower than f32, the f32 copy its projection's mixed-dtype matmul widens it to plus
/// that projection's output, whichever is larger (the copy is released before the chain).
pub fn entry_bytes(q: SanaEntryQuantum) -> u64 {
    let side = q.tokens; // workspace priced over N positions as a 1 x N plane
    let latent_positions = side.saturating_mul(q.patch).saturating_mul(q.patch);
    let patch = sum(&[
        tensor_bytes(q.element_bytes, &[2, q.batch, q.tokens, q.dim]),
        conv_workspace_bytes(
            q.batch,
            (1, latent_positions),
            (1, side),
            q.latent_channels,
            q.dim,
            q.patch,
            0,
        ),
    ]);
    // A caption narrower than f32 is widened to f32 for the first projection's mixed-dtype matmul;
    // that copy is read only by that matmul and released when it runs, before gelu's chain is
    // built (measured: a bf16 and an f32 caption peak identically at the chain-dominated proof
    // shape). It is live with the projection's output only, so the caption term is the larger of
    // the chain and (copy + projection output) — never their sum.
    let chain = tensor_bytes(4, &[ENTRY_CAPTION_LIVE, q.batch, q.caption_tokens, q.dim]);
    let caption = if q.caption_element_bytes < 4 {
        chain.max(sum(&[
            tensor_bytes(4, &[q.batch, q.caption_tokens, q.caption_channels]),
            tensor_bytes(4, &[q.batch, q.caption_tokens, q.dim]),
        ]))
    } else {
        chain
    };
    sum(&[
        q.weight_bytes,
        patch,
        caption,
        tensor_bytes(4, &[ENTRY_TIMESTEP_DIMS, q.batch, q.dim]),
    ])
}

/// The trunk's output quantum (modulated norm → `proj_out` → unpatchify).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SanaOutputQuantum {
    pub batch: u64,
    pub tokens: u64,
    pub dim: u64,
    /// `p·p·out_channels`.
    pub out_features: u64,
    pub element_bytes: u64,
    pub weight_bytes: u64,
}

/// `[B, N, dim]`: the incoming hidden, the LayerNorm's centred input and normalized output (its
/// CPU fallback), and the modulated copy — 3 at once (each modulation step reuses its predecessor).
const OUTPUT_HIDDEN_LIVE: u64 = 3;

/// `weight_bytes + 3 hidden-sized + 2 output-sized` (the projection and its unpatchified copy).
pub fn output_bytes(q: SanaOutputQuantum) -> u64 {
    sum(&[
        q.weight_bytes,
        tensor_bytes(
            q.element_bytes,
            &[OUTPUT_HIDDEN_LIVE, q.batch, q.tokens, q.dim],
        ),
        tensor_bytes(q.element_bytes, &[2, q.batch, q.tokens, q.out_features]),
        tensor_bytes(4, &[4, q.batch, q.dim]),
    ])
}

// =================================================================================================
// DC-AE (all f32, NHWC). Each formula is one `ConvStageGeometry` at the stage's extent with its
// counted live input-/output-width buffers, plus the convolution workspace of its widest conv.
// =================================================================================================

fn stage(b: u64, h: u64, w: u64, cin: u64, cout: u64, in_live: u64, out_live: u64) -> u64 {
    conv_stage_bytes(ConvStageGeometry {
        batch: b,
        height: h,
        width: w,
        in_channels: cin,
        out_channels: cout,
        element_bytes: 4,
        in_live,
        out_live,
        weight_bytes: 0,
    })
}

/// `ResBlock` at `[B, H, W, C]`: the residual input, `conv1`'s output and bias sum, the SiLU, `conv2`'s
/// output, and the RMSNorm/residual chain's buffer (6), plus BOTH 3x3 convolutions' workspaces:
/// MLX dispatches asynchronously and frees a workspace only when its command completes, which on
/// the CPU executor is after the second convolution has allocated its own (measured 25.4 C-wide
/// buffers at 64x64x32 against 26.1 priced).
pub fn dcae_res_block_bytes(b: u64, h: u64, w: u64, c: u64) -> u64 {
    stage(b, h, w, c, c, RES_LIVE, 0)
        .saturating_add(conv_workspace_bytes(b, (h, w), (h, w), c, c, 3, 1).saturating_mul(2))
}
const RES_LIVE: u64 = 6;

/// `EfficientViT` block at `[B, H, W, C]` with `scales` multiscale branches: its two halves run back
/// to back inside one quantum and, dispatched asynchronously, overlap (measured: attention alone
/// 31.3 C, GLUMBConv alone 41.0 C, the block 71.3 C at 64x64x32, one branch), so they are SUMMED:
///
/// * the multiscale linear attention, every intermediate it creates, in C-wide buffers: the
///   residual (1), q/k/v (3) and their concatenation (3), each branch's depthwise and grouped
///   projections (6 per branch), the `3C·(1+scales)` hidden and its head-major copy, the ReLU'd q
///   and k (`2·(1+scales)`), and the numerator, quotient and token-major copy (`3·(1+scales)`) —
///   `7 + 6·scales + 11·(1+scales)` (35 at one branch);
/// * the GLUMBConv (expand 4: `8C` expanded): the residual and five `8C` buffers (the inverted 1x1
///   output and its bias sum, the SiLU, the depthwise output and its bias sum; the gated `4C`
///   product fits in released halves), plus the 1x1 point conv's explicit-GEMM workspace over its
///   `4C` input.
pub fn dcae_evit_block_bytes(b: u64, h: u64, w: u64, c: u64, scales: u64) -> u64 {
    let attention = stage(b, h, w, c, c, 7 + 6 * scales + 11 * (1 + scales), 0);
    let glu = stage(b, h, w, c, 8 * c, 1, GLU_EXPANDED_LIVE).saturating_add(conv_workspace_bytes(
        b,
        (h, w),
        (h, w),
        4 * c,
        c,
        1,
        0,
    ));
    attention.saturating_add(glu)
}
const GLU_EXPANDED_LIVE: u64 = 5;

/// `DCUpBlock2d` producing `[B, 2H, 2W, Cout]` from `[B, H, W, Cin]`, at the OUTPUT extent: the
/// nearest-upsampled input (Cin) and the 3x3 conv's workspace over it; the conv output, the
/// `repeat_interleave` copy and its pixel-shuffled copy (each `Cout` wide at the output extent) and
/// their sum — 4 output-width buffers.
pub fn dcae_upsample_bytes(b: u64, h_out: u64, w_out: u64, cin: u64, cout: u64) -> u64 {
    stage(b, h_out, w_out, cin, cout, 1, 4).saturating_add(conv_workspace_bytes(
        b,
        (h_out, w_out),
        (h_out, w_out),
        cin,
        cout,
        3,
        1,
    ))
}

/// `DCDownBlock2d` (stride-2 conv) from `[B, H, W, Cin]` to `[B, H/2, W/2, Cout]`, at the INPUT
/// extent: the input, the stride-2 conv's workspace (padded input + `H/2·W/2·9·Cin` unfold), the
/// pixel-unshuffled copy (`4Cin` at half extent = one input-width buffer at full extent) and, at
/// the half extent, the conv output, the grouped mean and their sum (3·Cout/4 at full extent).
pub fn dcae_downsample_bytes(b: u64, h: u64, w: u64, cin: u64, cout: u64) -> u64 {
    let (ho, wo) = (h.div_ceil(2), w.div_ceil(2));
    sum(&[
        stage(b, h, w, cin, 0, 2, 0),
        stage(b, ho, wo, 0, cout, 0, 3),
        conv_workspace_bytes(b, (h, w), (ho, wo), cin, cout, 3, 1),
    ])
}

/// Decoder head input at the latent extent: the NHWC f32 latent copy, `conv_in`'s workspace over
/// it, `conv_in`'s `C` output, the `repeat_interleave` shortcut (`C`) and their sum.
pub fn dcae_decoder_input_bytes(b: u64, h: u64, w: u64, latent: u64, c: u64) -> u64 {
    stage(b, h, w, latent, c, 1, 3).saturating_add(conv_workspace_bytes(
        b,
        (h, w),
        (h, w),
        latent,
        c,
        3,
        1,
    ))
}

/// Decoder output at the image extent: the incoming `C` features, the RMSNorm's `x²`/quotient/
/// affine chain and the ReLU (2 more `C`-wide at once), `conv_out`'s workspace over `C` and its RGB
/// output.
pub fn dcae_decoder_output_bytes(b: u64, h: u64, w: u64, c: u64, out: u64) -> u64 {
    stage(b, h, w, c, out, 3, 1).saturating_add(conv_workspace_bytes(
        b,
        (h, w),
        (h, w),
        c,
        out,
        3,
        1,
    ))
}

/// Encoder input at the image extent: the NHWC f32 image copy, `conv_in`'s workspace over it and
/// its `C` output.
pub fn dcae_encoder_input_bytes(b: u64, h: u64, w: u64, in_c: u64, c: u64) -> u64 {
    stage(b, h, w, in_c, c, 1, 1).saturating_add(conv_workspace_bytes(
        b,
        (h, w),
        (h, w),
        in_c,
        c,
        3,
        1,
    ))
}

/// Encoder output at the latent extent: the incoming `C` features, `conv_out`'s workspace over `C`,
/// its latent output, the grouped-mean shortcut, their sum and its NCHW copy (4 latent-wide).
pub fn dcae_encoder_output_bytes(b: u64, h: u64, w: u64, c: u64, latent: u64) -> u64 {
    stage(b, h, w, c, latent, 1, 4).saturating_add(conv_workspace_bytes(
        b,
        (h, w),
        (h, w),
        c,
        latent,
        3,
        1,
    ))
}

/// The bounded encode's stitch quantum: the row concatenations (together one full `[B, H, W, C]`
/// prefix map) and the final concatenation (one more). The tiles' prefix outputs it reads were
/// materialized by their own quanta and are resident, not scratch.
pub fn dcae_encode_stitch_bytes(b: u64, h: u64, w: u64, c: u64) -> u64 {
    tensor_bytes(4, &[2, b, h, w, c])
}

// =================================================================================================
// Latent-sized sampler work.
// =================================================================================================

/// f32 latent-sized buffers one SCM step ([`crate::pipeline::scm_step`]) holds at its peak: its two
/// outputs (`pred_x0`, held, and the renoised latents), the renoise draw, and one buffer per
/// elementwise chain still in flight — the trigflow recombination, the x0 prediction and the
/// renoise (each chain's later steps reuse their predecessor's buffer) — 6 (measured 5.0).
const SCM_STEP_LIVE: u64 = 6;

/// One SCM step's latent-sized scratch (also the bound for materializing any SANA latent whose
/// pending graph is at most one sampler update).
pub fn scm_step_bytes(latent_elements: u64) -> u64 {
    tensor_bytes(4, &[SCM_STEP_LIVE, latent_elements])
}

// =================================================================================================
// Tiled DC-AE decode assembly (`mlx_gen::vae_tiling::tiled_decode_with_finalizer`, spatial only).
// =================================================================================================

/// One tile fold, priced for the worst tile (one that closes its row): the blend-weighted tile and
/// the row strip it is padded into plus the new strip accumulator (2 strips `[tile_h, out_w]`),
/// then the closed strip padded to the full output and the new output accumulator (2 full
/// `[out_h, out_w]` maps), all f32 at `channels`. Accumulators that existed before the fold are
/// already resident and are not scratch.
pub fn tile_fold_bytes(tile_h: u64, tile_w: u64, out_h: u64, out_w: u64, channels: u64) -> u64 {
    sum(&[
        tensor_bytes(4, &[tile_h, tile_w, channels]),
        tensor_bytes(4, &[2, tile_h, out_w, channels]),
        tensor_bytes(4, &[2, out_h, out_w, channels]),
    ])
}

/// The final normalization after the last fold: the separable normalizer's broadcast product (one
/// `[out_h, out_w]` plane; its clamp reuses it) plus the per-axis host factor vectors, their partial
/// product and the scalar operand (`out_h + out_w + 1` f32 each, 4 of them). The quotient reuses the
/// folded map — `tiled_decode_with_finalizer` hands it over as the division's only reference — and
/// the contiguous pass of a contiguous quotient is a no-op.
pub fn tile_finalize_bytes(out_h: u64, out_w: u64) -> u64 {
    sum(&[
        tensor_bytes(4, &[out_h, out_w]),
        tensor_bytes(4, &[4, out_h.saturating_add(out_w).saturating_add(1)]),
    ])
}

/// **Reachable-sizes prediction** (host-derived; a prediction, never evidence): for `sana_1600m`
/// and `sana_sprint_1600m` at q4/q8/bf16 × 512/768/1024, text-to-image and img2img, under the iOS
/// ladder (Sequential residency — text, then for img2img the reference encode with ONLY the DC-AE
/// encoder resident, then the trunk + decoder, then the decoder alone — bounded DC-AE decode AND
/// encode at the 192-pixel tile, unbounded attention), print each phase's resident bytes (from the snapshot's safetensors HEADERS), its
/// largest counted quantum scratch, and whether `resident + scratch + 512 MiB reserve` fits the
/// 6142 MiB entitled launch headroom. Header-only reads of the on-disk snapshot (`SANA_SNAPSHOT`
/// / `SANA_SPRINT_SNAPSHOT`, caller-provided snapshot roots) plus the Gemma tokenizer for
/// the CHI-prompt token count.
///
/// `cargo test -p mlx-gen-sana --lib reachable_sizes -- --ignored --nocapture`
#[cfg(test)]
mod reachable_sizes {
    use super::*;
    use crate::config::{BlockType, DcAeConfig, SanaTransformerConfig};
    use mlx_gen::attention::AttentionBudget;
    use mlx_gen::ios_quantum::{sampler_step_bytes, PROCESS_RESERVE_BYTES};
    use mlx_gen::tiling::{TilingConfig, VaeTiling};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    const MIB: u64 = 1024 * 1024;
    const ENTITLED_MIB: u64 = 6142;

    struct Header {
        bytes: u64,
        elements: u64,
    }

    fn headers(path: &Path) -> BTreeMap<String, Header> {
        mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .into_iter()
            .map(|h| {
                let elements = h.shape.iter().map(|d| *d as u64).product();
                (
                    h.name,
                    Header {
                        bytes: h.data_bytes,
                        elements,
                    },
                )
            })
            .collect()
    }

    fn sum_where(h: &BTreeMap<String, Header>, pred: impl Fn(&str) -> bool) -> u64 {
        h.iter()
            .filter(|(k, _)| pred(k))
            .map(|(_, v)| v.bytes)
            .sum()
    }

    /// f32 copy bytes of a stored-narrower DENSE weight (packed when `{base}.scales` exists).
    fn promoted(h: &BTreeMap<String, Header>, base: &str) -> u64 {
        if h.contains_key(&format!("{base}.scales")) {
            return 0;
        }
        h.get(&format!("{base}.weight"))
            .filter(|t| t.bytes < 4 * t.elements)
            .map_or(0, |t| 4 * t.elements)
    }

    /// The caller-provisioned snapshot (inference never derives a cache path).
    fn snapshot(var: &str, _repo: &str) -> Option<PathBuf> {
        std::env::var_os(var).map(PathBuf::from)
    }

    /// Largest counted DC-AE decode quantum at `edge`, tail tiled at `tile` output pixels.
    fn decode_scratch(cfg: &DcAeConfig, edge: u64, tile: i32, overlap: i32) -> (u64, &'static str) {
        let ch = |i: usize| cfg.block_out_channels[i] as u64;
        let n = cfg.num_stages();
        let tail = cfg
            .block_types
            .iter()
            .take_while(|t| **t == BlockType::Res)
            .count();
        let latent = edge / 32;
        let mut best = (
            dcae_decoder_input_bytes(1, latent, latent, 32, ch(n - 1)),
            "head input",
        );
        let mut take = |v: u64, label: &'static str| {
            if v > best.0 {
                best = (v, label);
            }
        };
        // Head: deep EfficientViT stages on the whole latent.
        let mut side = latent;
        for i in (tail..n).rev() {
            if i + 1 < n {
                take(
                    dcae_upsample_bytes(1, 2 * side, 2 * side, ch(i + 1), ch(i)),
                    "head upsample",
                );
                side *= 2;
            }
            for _ in 0..cfg.layers_per_block[i] {
                take(
                    dcae_evit_block_bytes(1, side, side, ch(i), 1),
                    "head EfficientViT block",
                );
            }
        }
        // Tail: per tile of the head map (tile plan keyed on the tail's x8).
        let vae = VaeTiling {
            spatial_scale: 1 << tail,
            temporal_scale: 1,
            causal_temporal: false,
            full_res_channels: 3,
        };
        let plan = TilingConfig::spatial_only(tile, overlap).plan(vae, 1, side as i32, side as i32);
        let t = plan
            .h
            .iter()
            .map(|a| (a.end - a.start) as u64)
            .max()
            .unwrap();
        let mut s = t;
        for i in (0..tail).rev() {
            take(
                dcae_upsample_bytes(1, 2 * s, 2 * s, ch(i + 1), ch(i)),
                "tail upsample",
            );
            s *= 2;
            for _ in 0..cfg.layers_per_block[i] {
                take(dcae_res_block_bytes(1, s, s, ch(i)), "tail ResBlock");
            }
        }
        take(dcae_decoder_output_bytes(1, s, s, ch(0), 3), "tail output");
        take(tile_fold_bytes(s, s, edge, edge, 3), "tile fold");
        take(tile_finalize_bytes(edge, edge), "finalize");
        best
    }

    /// Largest counted DC-AE bounded-encode quantum at `edge` with `core`-pixel tiles.
    fn encode_scratch(cfg: &DcAeConfig, edge: u64, core: u64) -> (u64, &'static str) {
        let ch = |i: usize| cfg.block_out_channels[i] as u64;
        let n = cfg.num_stages();
        let prefix = cfg
            .block_types
            .iter()
            .take_while(|t| **t == BlockType::Res)
            .count();
        // Receptive halo rounded to a prefix cell (see `DcAeEncoder::prefix_halo`).
        let (mut radius, mut scale) = (1u64, 1u64);
        for i in 0..prefix {
            radius += 2 * cfg.encoder_layers_per_block[i] as u64 * scale;
            radius += scale;
            scale *= 2;
        }
        let halo = radius.div_ceil(scale) * scale;
        let tile = (core + 2 * halo).min(edge);
        let mut best = (
            dcae_encoder_input_bytes(1, tile, tile, 3, ch(0)),
            "encoder input",
        );
        let mut take = |v: u64, label: &'static str| {
            if v > best.0 {
                best = (v, label);
            }
        };
        let mut s = tile;
        for i in 0..n {
            if i == prefix {
                take(
                    dcae_encode_stitch_bytes(1, edge >> prefix, edge >> prefix, ch(i)),
                    "encode stitch",
                );
                s = edge >> prefix;
            }
            for _ in 0..cfg.encoder_layers_per_block[i] {
                let v = match cfg.block_types[i] {
                    BlockType::Res => dcae_res_block_bytes(1, s, s, ch(i)),
                    BlockType::EfficientVit => dcae_evit_block_bytes(1, s, s, ch(i), 1),
                };
                take(
                    v,
                    if i < prefix {
                        "encode prefix ResBlock"
                    } else {
                        "encode EfficientViT block"
                    },
                );
            }
            if i + 1 < n {
                take(
                    dcae_downsample_bytes(1, s, s, ch(i), ch(i + 1)),
                    "encode downsample",
                );
                s /= 2;
            }
        }
        take(
            dcae_encoder_output_bytes(1, s, s, ch(n - 1), 32),
            "encoder output",
        );
        best
    }

    #[test]
    #[ignore = "prediction table from on-disk snapshot headers; run with --ignored --nocapture"]
    fn reachable_sizes_prediction() {
        let gemma_cfg = crate::text_encoder::Gemma2Config::gemma_2_2b();
        let dcae = DcAeConfig::sana_f32c32();
        for (model, var, repo, cfg) in [
            (
                "sana_1600m",
                "SANA_SNAPSHOT",
                "SceneWorks--Sana_1600M_1024px_mlx",
                SanaTransformerConfig::sana_1600m(),
            ),
            (
                "sana_sprint_1600m",
                "SANA_SPRINT_SNAPSHOT",
                "SceneWorks--Sana_Sprint_1.6B_1024px_mlx",
                SanaTransformerConfig::sana_sprint_1600m(),
            ),
        ] {
            let Some(root) = snapshot(var, repo) else {
                println!("{model}: no snapshot on disk; skipped");
                continue;
            };
            // Every advertised tier (`supported_quants: [Q4, Q8]` plus the dense bf16 source).
            for tier in ["q4", "q8", "bf16"] {
                let dir = root.join(tier);
                let te = headers(&dir.join("text_encoder/gemma-2-2b-it.safetensors"));
                let dit = headers(&dir.join("transformer/diffusion_pytorch_model.safetensors"));
                let vae = headers(&dir.join("vae/diffusion_pytorch_model.safetensors"));
                // Gemma token length: CHI prompt (with bos) + 300 − 2.
                let tok = mlx_gen::tokenizer::TextTokenizer::from_file(
                    dir.join("text_encoder/tokenizer.json"),
                    mlx_gen::tokenizer::TokenizerConfig {
                        max_length: 4096,
                        pad_token_id: 0,
                        chat_template: mlx_gen::tokenizer::ChatTemplate::None,
                        pad_to_max_length: false,
                    },
                )
                .unwrap();
                let chi = tok
                    .encode_ids(crate::text_encoder::SANA_CHI_PROMPT, true)
                    .unwrap()
                    .len() as u64;
                let gemma_tokens = chi + 300 - 2;
                let te_bytes = sum_where(&te, |_| true);
                // Gemma activations are the embedding's stored dtype (bf16 in both tiers).
                let text_scratch = [
                    mlx_gen_pid::gemma2::gemma_layer_bytes(&gemma_cfg, 1, gemma_tokens, 2, 0),
                    mlx_gen_pid::gemma2::gemma_embedding_bytes(&gemma_cfg, 1, gemma_tokens, 2),
                    mlx_gen_pid::gemma2::gemma_final_norm_bytes(&gemma_cfg, 1, gemma_tokens, 2),
                ]
                .into_iter()
                .max()
                .unwrap();
                let dit_bytes = sum_where(&dit, |_| true);
                let dec_bytes = sum_where(&vae, |k| k.starts_with("decoder."));
                let enc_bytes = sum_where(&vae, |k| k.starts_with("encoder."));
                // The Sequential img2img heavy bundle never builds the encoder (its reference was
                // encoded in its own phase), so the heavy load prices transformer + decoder only.
                // Every promoted weight copy of a block (and of the entry) is summed: the whole
                // quantum's ops fit MLX's in-flight command-buffer window (`SanaBlock::quantum`).
                let block_weight = ["ff.conv_inverted", "ff.conv_depth", "ff.conv_point"]
                    .iter()
                    .map(|c| {
                        let t = &dit[&format!("transformer_blocks.0.{c}.weight")];
                        if t.bytes < 4 * t.elements {
                            4 * t.elements
                        } else {
                            0
                        }
                    })
                    .chain(
                        [
                            "attn1.to_q",
                            "attn1.to_k",
                            "attn1.to_v",
                            "attn1.to_out.0",
                            "attn2.to_q",
                            "attn2.to_k",
                            "attn2.to_v",
                            "attn2.to_out.0",
                        ]
                        .iter()
                        .map(|l| promoted(&dit, &format!("transformer_blocks.0.{l}"))),
                    )
                    .sum::<u64>();
                let ts = if cfg.guidance_embeds {
                    "time_embed.timestep_embedder"
                } else {
                    "time_embed.emb.timestep_embedder"
                };
                let entry_weight = [
                    format!("{ts}.linear_1"),
                    format!("{ts}.linear_2"),
                    "time_embed.linear".into(),
                    "caption_projection.linear_1".into(),
                    "caption_projection.linear_2".into(),
                    "time_embed.guidance_embedder.linear_1".into(),
                    "time_embed.guidance_embedder.linear_2".into(),
                    "patch_embed.proj".into(),
                ]
                .iter()
                .map(|b| promoted(&dit, b))
                .sum::<u64>();
                println!(
                    "\n{model} {tier}: text_encoder {} MiB, transformer {} MiB, vae decoder {} MiB, vae encoder {} MiB; Gemma tokens {gemma_tokens}",
                    te_bytes / MIB,
                    dit_bytes / MIB,
                    dec_bytes / MIB,
                    enc_bytes / MIB
                );
                let row = |phase: &str, resident: u64, scratch: u64, what: &str| {
                    let total = resident + scratch + PROCESS_RESERVE_BYTES;
                    println!(
                        "  {phase:<26} resident {:>5} MiB + scratch {:>5} MiB ({what}) + reserve 512 = {:>5} MiB -> {}",
                        resident / MIB,
                        scratch.div_ceil(MIB),
                        total.div_ceil(MIB),
                        if total <= ENTITLED_MIB * MIB { "FITS 6142" } else { "EXCEEDS 6142" }
                    );
                };
                row(
                    "text load",
                    0,
                    te_bytes,
                    "header total (the measured-proven `before_load` floor)",
                );
                row(
                    "text (Gemma)",
                    te_bytes,
                    text_scratch,
                    "Gemma layer at L tokens",
                );
                row(
                    "reference encoder load",
                    0,
                    enc_bytes,
                    "vae encoder.* stored bytes",
                );
                row(
                    "heavy load",
                    0,
                    dit_bytes + dec_bytes,
                    "transformer + vae decoder.* stored bytes",
                );
                for edge in [512u64, 768, 1024] {
                    let latent = edge / 32;
                    let tokens = latent * latent;
                    let caption = 300u64;
                    let q = crate::transformer::configured_block_quantum(
                        &cfg,
                        1,
                        latent as i32,
                        latent as i32,
                        caption as i32,
                        AttentionBudget::UNBOUNDED,
                    );
                    let block = block_bytes(SanaBlockQuantum {
                        weight_bytes: block_weight,
                        ..q
                    });
                    let latent_elems = 32 * tokens;
                    let entry = entry_bytes(SanaEntryQuantum {
                        batch: 1,
                        latent_channels: 32,
                        tokens,
                        patch: 1,
                        dim: cfg.inner_dim() as u64,
                        caption_tokens: caption,
                        caption_channels: 2304,
                        caption_element_bytes: 2,
                        element_bytes: 4,
                        weight_bytes: entry_weight,
                    }) + sampler_step_bytes(latent_elems);
                    let output = output_bytes(SanaOutputQuantum {
                        batch: 1,
                        tokens,
                        dim: cfg.inner_dim() as u64,
                        out_features: 32,
                        element_bytes: 4,
                        weight_bytes: promoted(&dit, "proj_out"),
                    });
                    let denoise = [
                        (block, "SANA block"),
                        (entry, "trunk entry + sampler"),
                        (output, "trunk output"),
                        (scm_step_bytes(latent_elems), "SCM step"),
                    ]
                    .into_iter()
                    .max_by_key(|(v, _)| *v)
                    .unwrap();
                    let (decode, decode_what) = decode_scratch(&dcae, edge, 192, 48);
                    let (encode, encode_what) = encode_scratch(&dcae, edge, 192);
                    // Conditioning carried into the heavy phase: cond (+ uncond for base CFG).
                    let cond = (if cfg.guidance_embeds { 1 } else { 2 }) * 300 * 2304 * 2;
                    println!("  -- {edge}x{edge} (latent {latent}x{latent}, {tokens} tokens)");
                    row(
                        "denoise txt2img",
                        dit_bytes + dec_bytes + cond,
                        denoise.0,
                        denoise.1,
                    );
                    // Sequential img2img: the reference encodes with only the DC-AE encoder (and
                    // the conditioning) resident; the trunk phase then carries the f32 reference
                    // latent instead of the encoder.
                    row(
                        "img2img encode (tile 192)",
                        enc_bytes + cond,
                        encode,
                        encode_what,
                    );
                    row(
                        "denoise img2img",
                        dit_bytes + dec_bytes + cond + 4 * latent_elems,
                        denoise.0,
                        denoise.1,
                    );
                    row("decode (tile 192)", dec_bytes, decode, decode_what);
                }
            }
        }
    }
}
