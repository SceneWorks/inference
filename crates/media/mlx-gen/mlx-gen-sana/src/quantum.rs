//! SANA-specific iOS quantum accounting (trunk block, trunk entry/output, DC-AE stages, the tiled
//! decode/encode folds), built on the generic counted API in [`mlx_gen::ios_quantum`].
//!
//! Every figure is COUNTED from the ops of one quantum — how many tensors of each shape are live at
//! that quantum's peak — and each is proven by a CPU test (`quantum_proofs` below) that builds the
//! real module at a small shape, measures MLX's peak with
//! [`mlx_gen::ios_quantum::measured_peak_bytes`] and asserts the formula is at least the
//! measurement (hard), reporting a ratio above [`MAX_RATIO`] (report-only: low-end accuracy is
//! calibrated on the device). These are floors for one materialized quantum,
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

/// Every iOS quantum label this crate's guards pass, one string per quantum. A device capture
/// attributes its kernel peak to a phase by label ([`crate::ios_pricing::ios_quantum_phase`]), so no
/// label may be shared between two phases (the DC-AE encoder's blocks are labelled apart from the
/// decoder's for exactly that reason).
pub mod labels {
    /// Gemma-2 caption encoder load floor.
    pub const TEXT_LOAD: &str = "SANA text load";
    /// Sequential img2img: the DC-AE encoder alone.
    pub const REFERENCE_ENCODER_LOAD: &str = "SANA reference encoder load";
    /// The trunk + DC-AE decoder (+ encoder where the bundle builds it).
    pub const HEAVY_LOAD: &str = "SANA heavy load";
    /// Trunk entry (patchify, timestep/guidance, caption projection) + the pending sampler update.
    pub const ENTRY: &str = "SANA patch/caption/timestep";
    /// One Linear-DiT block.
    pub const BLOCK: &str = "SANA block";
    /// Trunk output (modulated norm, `proj_out`, unpatchify).
    pub const OUTPUT: &str = "SANA output projection";
    /// Sprint: one SCM update materialized.
    pub const SCM_STEP: &str = "SANA SCM step";
    /// Sprint: the final `denoised / sigma_data`.
    pub const SCM_FINAL: &str = "SANA SCM final";
    /// The standalone pipeline's final latent.
    pub const LATENT: &str = "SANA latent";
    /// The generator's batch of final latents.
    pub const LATENT_BATCH: &str = "SANA latent batch";
    /// img2img: the scaled reference latent.
    pub const REFERENCE_LATENT: &str = "SANA reference latent";
    /// DC-AE decoder `conv_in` + channel shortcut on the latent.
    pub const DECODER_INPUT: &str = "DC-AE head input";
    /// DC-AE decoder upsample block.
    pub const DECODER_UPSAMPLE: &str = "DC-AE upsample";
    /// DC-AE decoder `ResBlock` / `EfficientViT` block.
    pub const DECODER_BLOCK: &str = "DC-AE block";
    /// DC-AE decoder `norm_out → ReLU → conv_out`.
    pub const DECODER_OUTPUT: &str = "DC-AE output";
    /// Tiled decode: one tile folded into the accumulators.
    pub const TILE_FOLD: &str = "DC-AE tile fold";
    /// Tiled decode: the final normalization.
    pub const TILE_FINALIZE: &str = "DC-AE final normalize";
    /// The actual decoded RGB tensor's F32 denormalize/clamp/round/readback.
    pub const DECODED_PIXELS: &str = "SANA decoded pixels";
    /// DC-AE encoder `conv_in`.
    pub const ENCODER_INPUT: &str = "DC-AE encoder input";
    /// DC-AE encoder `ResBlock` / `EfficientViT` block.
    pub const ENCODER_BLOCK: &str = "DC-AE encoder block";
    /// DC-AE encoder stride-2 downsample.
    pub const ENCODER_DOWNSAMPLE: &str = "DC-AE downsample";
    /// Bounded encode: the tiles' prefix features stitched.
    pub const ENCODER_STITCH: &str = "DC-AE encode tile stitch";
    /// DC-AE encoder `conv_out` + out-shortcut.
    pub const ENCODER_OUTPUT: &str = "DC-AE encoder output";

    /// Every label above.
    pub const ALL: &[&str] = &[
        TEXT_LOAD,
        REFERENCE_ENCODER_LOAD,
        HEAVY_LOAD,
        ENTRY,
        BLOCK,
        OUTPUT,
        SCM_STEP,
        SCM_FINAL,
        LATENT,
        LATENT_BATCH,
        REFERENCE_LATENT,
        DECODER_INPUT,
        DECODER_UPSAMPLE,
        DECODER_BLOCK,
        DECODER_OUTPUT,
        TILE_FOLD,
        TILE_FINALIZE,
        DECODED_PIXELS,
        ENCODER_INPUT,
        ENCODER_BLOCK,
        ENCODER_DOWNSAMPLE,
        ENCODER_STITCH,
        ENCODER_OUTPUT,
    ];
}

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

/// **Reachable-sizes prediction** (host-derived from the snapshot's safetensors HEADERS; a
/// prediction, never evidence): for `sana_1600m` and `sana_sprint_1600m` at q4/q8/bf16 × text /
/// reference (img2img) × 512/768/1024 on the iOS load (`Sequential`, as
/// `mlx_gen_ios_catalog::sana_load_spec` builds it), print every phase's resident bytes, its largest
/// counted quantum and `resident + scratch + 512 MiB reserve` against the 6142 MiB entitled launch
/// headroom, at (A) the FULLY OPTIMIZED selection — the deepest rung the contract `Implemented`s on
/// that load, at the last (cheapest) selection of its `selection_space` — and (B) rung 0 Resident.
/// Every number comes from [`crate::ios_pricing::ios_pricing`] (the library the device prices
/// with), and every (A) cell must fit. Snapshot roots are caller-provided (`SANA_SNAPSHOT`,
/// `SANA_SPRINT_SNAPSHOT`).
///
/// `cargo test -p mlx-gen-sana --lib reachable_sizes -- --ignored --nocapture`
#[cfg(test)]
mod reachable_sizes {
    use mlx_gen::gen_core::{
        LoadSpec, MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryStrategy,
        MemoryStrategySupport, OffloadPolicy, WeightsSource,
    };
    use mlx_gen::ios_pricing::IosPricingRequest;
    use mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES;

    const MIB: u64 = 1024 * 1024;
    const ENTITLED_MIB: u64 = 6142;

    #[test]
    #[ignore = "prediction table from on-disk snapshot headers; run with --ignored --nocapture"]
    fn reachable_sizes_prediction() {
        let mut misfits = Vec::new();
        let mut summary = Vec::new();
        for (provider, var) in [
            (crate::MODEL_ID, "SANA_SNAPSHOT"),
            (crate::SPRINT_MODEL_ID, "SANA_SPRINT_SNAPSHOT"),
        ] {
            let Some(root) = std::env::var_os(var).map(std::path::PathBuf::from) else {
                println!("{provider}: no snapshot on disk ({var} unset); skipped");
                continue;
            };
            println!(
                "\n| provider | cfg | tier | mode | edge | phase | resident MiB | scratch MiB | binding quantum | peak MiB | fits 6142 |"
            );
            println!("|---|---|---|---|---:|---|---:|---:|---|---:|---|");
            for tier in ["q4", "q8", "bf16"] {
                // The iOS load: `mlx_gen_ios_catalog::sana_load_spec` (a Sequential load of the
                // tier directory); the catalog depends on this crate, so it is mirrored, not called.
                let spec = LoadSpec::new(WeightsSource::Dir(root.join(tier)))
                    .with_offload_policy(OffloadPolicy::Sequential);
                let contract =
                    crate::memory_strategy::memory_strategy_contract(provider, &spec).unwrap();
                let numeric = MemoryNumericTier {
                    precision: spec.precision,
                    quant: spec.quantize,
                    component_precision_floors: &[],
                };
                let deepest = MemoryStrategy::ALL
                    .into_iter()
                    .rev()
                    .find(|s| {
                        contract
                            .capability(*s)
                            .is_some_and(|c| c.support == MemoryStrategySupport::Implemented)
                    })
                    .unwrap();
                for (config, strategy) in [("A", deepest), ("B", MemoryStrategy::Resident)] {
                    let selection = *contract
                        .selection_space(strategy, numeric, false)
                        .unwrap()
                        .last()
                        .unwrap();
                    let execution = contract.generation_memory(&selection).unwrap_or_default();
                    println!(
                        "| {provider} | {config} = {strategy:?} {:?} | {tier} | | | | | | | | |",
                        selection.parameters
                    );
                    for (mode_name, mode, refs) in [
                        ("text", MemoryMode::TextToImage, 0),
                        ("reference", MemoryMode::ImageToImage, 1),
                    ] {
                        for edge in [512u32, 768, 1024] {
                            // Explicit native-default sampling request for this header diagnostic;
                            // mode/reference geometry is carried below, without fake media payloads.
                            let generation = mlx_gen::gen_core::GenerationRequest {
                                width: edge,
                                height: edge,
                                count: 1,
                                memory: Some(execution),
                                ..Default::default()
                            };
                            let req = IosPricingRequest {
                                provider,
                                spec: &spec,
                                mode: &mode,
                                geometry: MemoryGeometry {
                                    width: edge,
                                    height: edge,
                                    batch: 1,
                                    frames: 1,
                                    reference_count: refs,
                                },
                                selection: &selection,
                                execution: &execution,
                                token_window: 0,
                                generation: Some(&generation),
                                transform: None,
                                depth: None,
                            };
                            let pricing = crate::ios_pricing::ios_pricing(&req).unwrap();
                            for p in &pricing.phases {
                                let peak = p.peak_bytes() + PROCESS_RESERVE_BYTES;
                                let fits = peak <= ENTITLED_MIB * MIB;
                                println!(
                                    "| {provider} | {config} | {tier} | {mode_name} | {edge} | {} | {} | {} | {} | {} | {} |",
                                    p.phase.as_key(),
                                    p.resident_bytes.div_ceil(MIB),
                                    p.scratch_bytes.div_ceil(MIB),
                                    p.binding,
                                    peak.div_ceil(MIB),
                                    if fits { "yes" } else { "NO" }
                                );
                                if !fits && config == "A" {
                                    misfits.push(format!(
                                        "{provider} {tier} {mode_name} {edge} {}",
                                        p.phase.as_key()
                                    ));
                                }
                            }
                            let worst = pricing
                                .phases
                                .iter()
                                .max_by_key(|p| p.peak_bytes())
                                .unwrap();
                            summary.push(format!(
                                "| {provider} | {mode_name} | {tier} | {edge} | {config} | {} | {} / {} |",
                                (worst.peak_bytes() + PROCESS_RESERVE_BYTES).div_ceil(MIB),
                                worst.phase.as_key(),
                                worst.binding
                            ));
                        }
                    }
                }
            }
        }
        println!("\nWhole-run peak (worst phase, + 512 MiB reserve) per cell:");
        println!("| provider | mode | tier | edge | cfg | peak MiB | binding phase / quantum |");
        println!("|---|---|---|---:|---|---:|---|");
        for row in summary {
            println!("{row}");
        }
        assert!(
            misfits.is_empty(),
            "fully optimized cells over budget: {misfits:?}"
        );
    }
}
