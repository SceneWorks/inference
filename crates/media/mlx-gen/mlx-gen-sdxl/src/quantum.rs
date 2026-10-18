//! SDXL iOS execution quanta (sc-26270): the labels and the COUNTED scratch geometry every guarded
//! SDXL quantum is priced with.
//!
//! Every price is `live activations + kernel transients`, counted from the ops of that quantum:
//!
//! * **Activations** are counted from the code at each phase of the quantum (which tensors of which
//!   shape are alive together), read off the LOADED tensors (conv weight shapes, projection base
//!   shapes) and the request's actual activation shapes and dtype, never off production constants.
//! * **Kernel transients** are what an MLX kernel allocates internally for one call, modelled per
//!   backend from the pinned MLX source ([`Backend`]): a 3x3 convolution on Metal runs Winograd
//!   (padded input + transformed input/output/filter) when MLX's dispatch selects it, and copies a
//!   non-row-contiguous weight; on the CPU it is an explicit GEMM over an f32 im2col. Fused SDPA
//!   materializes no score tensor on Metal for head dims 64/80/128 (the SDXL U-Net and CLIP); the
//!   CPU always runs the unfused fallback (scaled q, scores, softmax).
//!
//! The device prices with [`Backend::Metal`]; each counted site is proven by a CPU test that
//! measures MLX's peak for a small shape and checks the SAME activation counts with the CPU's kernel
//! transients (`quantum_tests::proofs`). The guards themselves are `mlx_gen::ios_quantum::{run,
//! before, finish}`, which are pass-throughs whenever [`mlx_gen::ios_quantum::enabled`] is false
//! (macOS, Linux, the simulator), so nothing here can change host numerics or ordering.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::ios_quantum::{
    attention_block_bytes, scores_bytes, tensor_bytes, AttentionBlockGeometry, Scores,
};
// The AutoencoderKL convolution / GroupNorm / tiled-stage kernel accounting is shared with every
// diffusers VAE on iOS (sc-26433): it lives in `mlx_gen::ios_quantum` and is re-exported here so
// this crate's call sites and proofs keep their paths.
pub(crate) use mlx_gen::ios_quantum::{
    bytes, conv_channels, decoded_to_image_bytes, downsample_stage_bytes, downsample_tile_bytes,
    dtype_bytes, element_bytes, elements, group_norm_buffers, norm_conv_bytes,
    norm_promotion_units, page_slack, resnet_bytes, single_conv_bytes, tile_bytes,
    tiled_stage_bytes, tiled_stage_f32_accum, vae_attention_bytes, Backend, ConvCall, TileSource,
    VAE_ATTENTION_SCORE_ELEMENTS,
};
#[cfg(test)]
pub(crate) use mlx_gen::ios_quantum::{GROUP_NORM_BUFFERS, PAGE_BYTES};
use mlx_rs::{Array, Dtype};

fn u(v: i32) -> u64 {
    v.max(0) as u64
}

thread_local! {
    static PRICING_BACKEND: std::cell::Cell<Option<Backend>> = const { std::cell::Cell::new(None) };
}

/// The backend every counted SDXL quantum is priced for: the running device's
/// ([`Backend::current`]), except inside [`with_backend`], where header-only pricing
/// ([`crate::ios_pricing`]) prices the device's Metal kernels from any host.
pub(crate) fn backend() -> Backend {
    PRICING_BACKEND
        .with(std::cell::Cell::get)
        .unwrap_or_else(Backend::current)
}

/// Run `body` with every [`backend`] call answering `backend` (restored on return or unwind).
pub(crate) fn with_backend<T>(backend: Backend, body: impl FnOnce() -> T) -> T {
    struct Restore(Option<Backend>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PRICING_BACKEND.with(|b| b.set(self.0));
        }
    }
    let _restore = Restore(PRICING_BACKEND.with(|b| b.replace(Some(backend))));
    body()
}

/// The counted quanta one header-only pricing walk checks, in execution order: `(label, scratch)`.
pub(crate) type Quanta = Vec<(&'static str, u64)>;

/// `a` promoted with every array's dtype, as MLX promotes an elementwise chain over them.
pub(crate) fn promote_all(a: Dtype, arrays: &[&Array]) -> Dtype {
    arrays
        .iter()
        .fold(a, |d, x| Dtype::from_promoting_types(d, x.dtype()))
}

/// A shape-and-dtype stand-in for an activation in header-only pricing: a lazy MLX array that is
/// never evaluated, so it carries the `shape()` / `dtype()` every scratch formula reads and no
/// payload.
pub(crate) fn shaped(shape: &[i32], dtype: Dtype) -> mlx_gen::Result<Array> {
    Ok(mlx_rs::ops::zeros_dtype(shape, dtype)?)
}

/// U-Net stem: Kolors context projection, timestep/`text_time` embedding and `conv_in`.
pub(crate) const UNET_STEM: &str = "SDXL UNet stem";
/// One U-Net / ControlNet `ResnetBlock2D` (including its up-path skip concatenation).
pub(crate) const UNET_RESNET: &str = "SDXL UNet resnet";
/// A U-Net / ControlNet stride-2 downsample convolution.
pub(crate) const UNET_DOWNSAMPLE: &str = "SDXL UNet downsample";
/// A U-Net nearest-2x upsample plus its convolution.
pub(crate) const UNET_UPSAMPLE: &str = "SDXL UNet upsample";
/// A `Transformer2D`'s GroupNorm + `proj_in`.
pub(crate) const TRANSFORMER_PROJ_IN: &str = "SDXL transformer proj_in";
/// One spatial-transformer block (self-attn, cross-attn [+ IP branch], GEGLU FFN).
pub(crate) const TRANSFORMER_BLOCK: &str = "SDXL transformer block";
/// A `Transformer2D`'s `proj_out` + residual.
pub(crate) const TRANSFORMER_PROJ_OUT: &str = "SDXL transformer proj_out";
/// The ControlNet down residuals added into the U-Net skip stack.
pub(crate) const CONTROL_DOWN_INJECTION: &str = "SDXL ControlNet down injection";
/// The ControlNet mid residual added into the U-Net mid output.
pub(crate) const CONTROL_MID_INJECTION: &str = "SDXL ControlNet mid injection";
/// U-Net output GroupNorm + SiLU + `conv_out`.
pub(crate) const UNET_CONV_OUT: &str = "SDXL UNet conv_out";
/// ControlNet stem: context projection, embeddings, `conv_in` + conditioning embedding.
pub(crate) const CONTROL_STEM: &str = "SDXL ControlNet stem";
/// One ControlNet 1x1 zero-convolution head (down or mid), scaled.
pub(crate) const CONTROL_ZERO_CONV: &str = "SDXL ControlNet zero-conv";
/// One convolution of the ControlNet conditioning embedding.
pub(crate) const CONTROL_COND_CONV: &str = "SDXL ControlNet cond embedding";
/// CLIP text token + position embedding.
pub(crate) const CLIP_EMBEDDINGS: &str = "SDXL CLIP embeddings";
/// One CLIP text encoder layer.
pub(crate) const CLIP_LAYER: &str = "SDXL CLIP layer";
/// CLIP text final LayerNorm + pooled (projected) EOS token.
pub(crate) const CLIP_POOLED: &str = "SDXL CLIP final norm";
/// CLIP vision patch/class/position embedding + `pre_layrnorm`.
pub(crate) const VISION_EMBEDDINGS: &str = "SDXL CLIP vision embeddings";
/// One CLIP vision encoder layer.
pub(crate) const VISION_LAYER: &str = "SDXL CLIP vision layer";
/// IP-Adapter Resampler `proj_in`.
pub(crate) const RESAMPLER_PROJ_IN: &str = "SDXL IP resampler proj_in";
/// One IP-Adapter Resampler (perceiver attention + FFN) layer.
pub(crate) const RESAMPLER_LAYER: &str = "SDXL IP resampler layer";
/// IP-Adapter Resampler `proj_out` + `norm_out`.
pub(crate) const RESAMPLER_PROJ_OUT: &str = "SDXL IP resampler proj_out";
/// VAE decode de-normalization, `post_quant_conv` and decoder `conv_in`.
pub(crate) const VAE_DECODE_IN: &str = "SDXL VAE decode conv_in";
/// VAE encoder `conv_in`.
pub(crate) const VAE_ENCODE_IN: &str = "SDXL VAE encode conv_in";
/// One VAE (temb-free) resnet.
pub(crate) const VAE_RESNET: &str = "SDXL VAE resnet";
/// The VAE mid-block single-head attention.
pub(crate) const VAE_ATTENTION: &str = "SDXL VAE attention";
/// A VAE nearest-2x upsample plus its convolution.
pub(crate) const VAE_UPSAMPLE: &str = "SDXL VAE upsample";
/// A VAE asymmetric-pad stride-2 downsample convolution.
pub(crate) const VAE_DOWNSAMPLE: &str = "SDXL VAE downsample";
/// VAE decoder output GroupNorm + SiLU + `conv_out` (+ output cast).
pub(crate) const VAE_DECODE_OUT: &str = "SDXL VAE decode conv_out";
/// VAE encoder output GroupNorm + SiLU + `conv_out` + `quant_conv` + mean.
pub(crate) const VAE_ENCODE_OUT: &str = "SDXL VAE encode conv_out";
/// The decoded image's clamp / scale / round to 8-bit pixel values.
pub(crate) const DECODED_TO_IMAGE: &str = "SDXL decoded to image";
/// One denoise step's CFG combine + sampler update (+ inpaint blend), materialized.
pub(crate) const SAMPLER_STEP: &str = "SDXL sampler step";
/// One halo-expanded tile of a layer-wise tiled VAE convolution, checked before its graph exists.
pub(crate) const VAE_TILE: &str = "SDXL VAE tile";
/// The assembled output of one tiled VAE layer (resnet / upsample / output convolution).
pub(crate) const VAE_TILED_STAGE: &str = "SDXL VAE tiled stage";
/// One VAE ENCODER (temb-free) resnet — the encode twin of [`VAE_RESNET`], labelled apart so a
/// device capture attributes it to the encode phase ([`crate::ios_pricing::ios_quantum_phase`]).
pub(crate) const VAE_ENCODE_RESNET: &str = "SDXL VAE encode resnet";
/// The VAE ENCODER mid-block attention (the encode twin of [`VAE_ATTENTION`]).
pub(crate) const VAE_ENCODE_ATTENTION: &str = "SDXL VAE encode attention";
/// One halo-expanded tile of a bounded-encode convolution (the encode twin of [`VAE_TILE`]).
pub(crate) const VAE_ENCODE_TILE: &str = "SDXL VAE encode tile";
/// One assembled bounded-encode layer (the encode twin of [`VAE_TILED_STAGE`]).
pub(crate) const VAE_ENCODE_TILED_STAGE: &str = "SDXL VAE encode tiled stage";

/// The labels one VAE pass (encode or decode) checks its shared quanta under. The encoder and the
/// decoder run the same resnet / attention / tiled-convolution code, so the pass names its labels:
/// a label must say which phase it runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VaeLabels {
    pub resnet: &'static str,
    pub attention: &'static str,
    pub tiled_stage: &'static str,
    pub tile: &'static str,
}

/// The decode pass's labels.
pub(crate) const VAE_DECODE_LABELS: VaeLabels = VaeLabels {
    resnet: VAE_RESNET,
    attention: VAE_ATTENTION,
    tiled_stage: VAE_TILED_STAGE,
    tile: VAE_TILE,
};

/// The encode pass's labels.
pub(crate) const VAE_ENCODE_LABELS: VaeLabels = VaeLabels {
    resnet: VAE_ENCODE_RESNET,
    attention: VAE_ENCODE_ATTENTION,
    tiled_stage: VAE_ENCODE_TILED_STAGE,
    tile: VAE_ENCODE_TILE,
};

/// Component load floors (`ios_quantum::before` over a header-only floor before construction).
pub(crate) const TEXT_ENCODER_1_LOAD: &str = "SDXL text_encoder load";
pub(crate) const TEXT_ENCODER_2_LOAD: &str = "SDXL text_encoder_2 load";
pub(crate) const UNET_LOAD: &str = "SDXL UNet load";
pub(crate) const VAE_LOAD: &str = "SDXL VAE load";
pub(crate) const VAE_FP16_FIX_LOAD: &str = "SDXL fp16-fix VAE load";
pub(crate) const CONTROLNET_LOAD: &str = "SDXL ControlNet load";
pub(crate) const IP_IMAGE_ENCODER_LOAD: &str = "SDXL IP image encoder load";
pub(crate) const IP_ADAPTER_LOAD: &str = "SDXL IP adapter load";
/// The fused single-file (LDM) checkpoint's phase loads.
pub(crate) const LDM_TEXT_LOAD: &str = "SDXL LDM text load";
pub(crate) const LDM_HEAVY_LOAD: &str = "SDXL LDM UNet+VAE load";

/// One adapter (LoRA / LoKr / LoHa) file's read, checked against its header floor before the read.
pub(crate) const ADAPTER_LOAD: &str = "SDXL adapter load";
/// One adapter target's delta reconstruction + merge into the base (or, on a packed base, the
/// install of its forward-time residual).
pub(crate) const ADAPTER_MERGE: &str = "SDXL adapter merge";

/// The dtype `x`-typed activations leave `linear` at: MLX promotes a dense matmul to
/// `result_type(x, weight)` and a quantized one to `result_type(x, scales)`, so a pre-quantized
/// SceneWorks snapshot's BF16 `.scales` (every packed U-Net / CLIP projection) turn f16 activations
/// into f32 (measured: `quantum_tests::proofs::f16_transformer_quanta_track_their_measured_peaks`),
/// and everything downstream of that projection — residual sums, norms, convolutions — runs at f32.
pub(crate) fn linear_output_dtype(linear: &AdaptableLinear, x: Dtype) -> Dtype {
    let weight = match linear.quantized_params() {
        Some((_, scales, ..)) => scales.dtype(),
        None => linear.weight_dtype().unwrap_or(x),
    };
    Dtype::from_promoting_types(x, weight)
}

/// Output-row count of one projection (the `out` of its logical `[out, in]` weight).
pub(crate) fn linear_out(linear: &AdaptableLinear) -> u64 {
    linear.base_shape().first().map_or(0, |d| u(*d))
}

/// How a call site's attention scores exist on `backend`: MLX's Metal SDPA is fused for a query
/// length above 8 at head dims 64/80/128 (`sdpa_full`), or at most 8 queries at 64/96/128/256
/// (`sdpa_vector`); every other shape — and every CPU call — runs the unfused fallback, which
/// materializes the scores and (for a multi-head call, measured) a second score-sized buffer for
/// the mask add / softmax. The single-head VAE attention prices its own (donated) softmax.
pub(crate) fn scores(backend: Backend, head_dim: i32, query_rows: i32, act: u64) -> Scores {
    let fused = match backend {
        Backend::Cpu => false,
        Backend::Metal if query_rows > 8 => matches!(head_dim, 64 | 80 | 128),
        Backend::Metal => matches!(head_dim, 64 | 96 | 128 | 256),
    };
    if fused {
        Scores::Fused
    } else {
        Scores::Materialized {
            live: 2,
            element_bytes: act,
        }
    }
}

/// Buffers of one `mlx_rs::fast::layer_norm` at its input's size: Metal runs the fused kernel (its
/// output only); MLX's CPU backend runs the composed fallback — the centred input, its square and
/// the output (measured 3.03 input-sized buffers at f32). Below f32 the CPU fallback also holds f32
/// working copies ([`norm_promotion_units`]); a U-Net transformer block's measured f16 peak is
/// already covered without them (1.09-1.14x: its scores dominate), a CLIP layer's is not
/// ([`clip_layer_bytes`] adds them).
pub(crate) fn layer_norm_buffers(backend: Backend) -> u64 {
    match backend {
        Backend::Metal => 1,
        Backend::Cpu => 3,
    }
}
/// Query-sized buffers the unfused SDPA fallback adds per self-attention when `scores`
/// materializes: the scaled query (row-contiguous), and the row-contiguous copies of the head-split
/// k/v views its batched matmuls take.
fn fallback_hidden(scores: Scores) -> u64 {
    3 * u64::from(matches!(scores, Scores::Materialized { .. }))
}

/// One CLIP text / vision encoder layer over `[B, N, D]` (pre-norm attention + pre-norm MLP, both
/// residual), FFN width `ff`; `mask` when the text encoder's causal additive mask `[1, 1, N, N]` is
/// applied. On iOS the attention half is materialized before the MLP is built, so the price is
/// [`attention_block_bytes`]'s wider phase, each counted as new buffers under [`conv_sum`]'s
/// retention model (a residual sum donates its projection operand; each LayerNorm is
/// [`layer_norm_buffers`]):
/// * attention — LayerNorm, q, k, v, the head-merged SDPA output and its reshape copy, the output
///   projection [+ the fallback's], and the scores (+ the mask);
/// * MLP — the materialized attention result, LayerNorm, `fc2`; FFN-sized: `fc1` and its GELU.
#[allow(clippy::too_many_arguments)]
pub(crate) fn clip_layer_bytes(
    b: i32,
    n: i32,
    dim: u64,
    ff: u64,
    heads: i32,
    head_dim: i32,
    mask: bool,
    act: u64,
    backend: Backend,
) -> u64 {
    let s = scores(backend, head_dim, n, act);
    // A 77-token layer's small score tensors leave the CPU fallback LayerNorm's f32 working copies
    // at the layer's peak below f32 (measured at f16: 0.86x without them).
    let ln = layer_norm_buffers(backend) + norm_promotion_units(backend, act);
    let attn_hidden_live = 6 + ln + fallback_hidden(s);
    let mlp_hidden_live = 2 + ln;
    let mask_bytes = if mask {
        tensor_bytes(act, &[u(n), u(n)])
    } else {
        0
    };
    attention_block_bytes(AttentionBlockGeometry {
        batch: u(b),
        tokens: u(n),
        dim,
        heads: u(heads),
        context_tokens: 0,
        ff_hidden: ff,
        query_rows: u(n),
        element_bytes: act,
        weight_bytes: 0,
        attn_hidden_live,
        context_live: 0,
        scores: s,
        mlp_hidden_live,
        ff_live: 2,
    })
    .saturating_add(mask_bytes)
    .saturating_add(page_slack(attn_hidden_live.max(mlp_hidden_live + 2) + 3))
}

/// One U-Net spatial-transformer block over `[B, N, D]`, cross-attending to `context_tokens` keys
/// (text memory plus any IP tokens) projected at `context_batch` rows (a sequential CFG half
/// projects the pair's conditioning, then keeps its row), FFN (GEGLU half) width `ff`, `query_rows`
/// queries per SDPA call, `ip` when the IP branch runs.
///
/// On iOS the block materializes each sub-layer before building the next (`forward_ip`'s barriers),
/// so the price is its widest sub-layer, each counted as new buffers under [`conv_sum`]'s retention
/// model (a residual sum donates its projection operand; each LayerNorm is
/// [`layer_norm_buffers`]):
/// * self-attention — hidden: LayerNorm, q, k, v, the SDPA output and its head-merge copy, the
///   output projection [+ the fallback's per attention; a bounded plan's chunk outputs and their
///   concatenation]; its scores;
/// * cross-attention — hidden: the materialized self-attention result, LayerNorm, q, the SDPA
///   output and its copy, the projection [+ the IP branch's output and its scaled copy]; context:
///   k and v at `context_batch` rows, their selected rows under sequential CFG [+ the fallback's
///   k/v copies]; its scores;
/// * FFN — hidden: the materialized attention result, LayerNorm, `linear3`; FFN-sized: `linear1`,
///   `linear2`, the GELU (the GEGLU product and the final sum donate).
#[allow(clippy::too_many_arguments)]
pub(crate) fn transformer_block_bytes(
    batch: i32,
    context_batch: i32,
    tokens: i32,
    dim: u64,
    heads: i32,
    head_dim: i32,
    context_tokens: i32,
    ff: u64,
    query_rows: i32,
    ip: bool,
    act: u64,
    backend: Backend,
) -> u64 {
    let s = scores(backend, head_dim, query_rows, act);
    let fallback = u64::from(matches!(s, Scores::Materialized { .. }));
    let chunked = 2 * u64::from(query_rows < tokens);
    let (b, cb) = (u(batch).max(1), u(context_batch).max(u(batch)));
    let selected = u64::from(cb > b);
    let ln = layer_norm_buffers(backend);
    let geometry = AttentionBlockGeometry {
        batch: b,
        tokens: u(tokens),
        dim,
        heads: u(heads),
        context_tokens: 0,
        ff_hidden: ff,
        query_rows: u(query_rows),
        element_bytes: act,
        weight_bytes: 0,
        attn_hidden_live: 6 + ln + 3 * fallback + chunked,
        context_live: 0,
        scores: s,
        mlp_hidden_live: 2 + ln,
        ff_live: 3,
    };
    // Self-attention and FFN phases.
    let self_and_ffn = attention_block_bytes(geometry);
    // Cross-attention phase: k and v at the context batch (+ their selected rows, + the fallback's
    // k/v copies), in `[batch, S, D]` units. Its scores span the `S` context keys only: priced
    // here (`attention_block_bytes` would score `N + S` keys), the API pricing the buffers.
    let context_units = (2 * cb + 2 * selected * b + 2 * fallback * b).div_ceil(b);
    let cross = attention_block_bytes(AttentionBlockGeometry {
        context_tokens: u(context_tokens),
        attn_hidden_live: 5 + ln + fallback + chunked + 2 * u64::from(ip),
        context_live: context_units,
        scores: Scores::Fused,
        mlp_hidden_live: 0,
        ff_live: 0,
        ..geometry
    })
    .saturating_add(scores_bytes(
        s,
        b,
        u(heads),
        u(query_rows),
        u(context_tokens),
    ));
    let buffers = geometry.attn_hidden_live.max(5 + ln + 2 + context_units) + 2;
    self_and_ffn.max(cross).saturating_add(page_slack(buffers))
}

/// Check one halo-expanded tile of a layer-wise tiled VAE convolution BEFORE its norm/activation/
/// conv graph is constructed, priced at the tile's extent, its `source` (the activation's width,
/// whether the crop is normalized) and `weight`'s channel counts. Called from
/// the per-tile preprocess hook of `mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc*`, which evaluates
/// each output core before the next tile is built.
pub(crate) fn before_tile(
    label: &str,
    tile: &Array,
    source: TileSource,
    weight: &Array,
    f32_accum: bool,
) -> mlx_gen::Result<()> {
    mlx_gen::ios_quantum::before(
        label,
        tile_bytes(
            tile.shape(),
            source,
            element_bytes(tile),
            weight,
            f32_accum,
            backend(),
        ),
    )
}

/// One ancestral denoise step's latent-sized work, materialized by its per-step eval
/// (`pipeline::denoise_core`), in f32 latent buffers (the widest the step runs at), counted as new
/// buffers (every other op of the chain donates its single-consumer operand): the CFG combine's
/// difference; the update's scaled sample, the f32 noise draw and its compute-dtype cast; and, with
/// an inpaint blend, the re-noised init, the kept and the repainted halves.
pub(crate) fn sampler_step_bytes(latents: &Array, inpaint: bool) -> u64 {
    mlx_gen::ios_quantum::elementwise_bytes(latents.size() as u64, 4 + 3 * u64::from(inpaint))
}

/// `copies` buffers the size of `arrays` at their own widths: a small elementwise pass (residual
/// add, scale) over already-materialized tensors.
pub(crate) fn elementwise_bytes(arrays: &[&Array], copies: u64) -> u64 {
    arrays
        .iter()
        .fold(0u64, |n, a| n.saturating_add(bytes(a)))
        .saturating_mul(copies)
}

/// One adapter target's install (`adapters.rs`), counted under MLX's retention (every buffer the
/// install creates stays live until its command buffer completes): `widths` lists the bytes per
/// element of each target-sized (`elements`) buffer the delta reconstruction and the merge create
/// (an elementwise op whose single-consumer operand has its width donates and is not listed),
/// `small` the factor-sized buffers (f32 casts, low-rank products, a scaled factor), and
/// `unread_base` the base weight's stored bytes when it has not been read yet — a lazily loaded
/// base is read inside the merge, and the merged weight replaces it only once the merge completes.
pub(crate) fn adapter_install_bytes(
    elements: u64,
    widths: &[u64],
    small: u64,
    unread_base: u64,
) -> u64 {
    widths
        .iter()
        .fold(0u64, |n, w| n.saturating_add(w.saturating_mul(elements)))
        .saturating_add(small)
        .saturating_add(unread_base)
        .saturating_add(page_slack(widths.len() as u64 + 2))
}

/// iOS pre-load floor of a component that [`crate::loader::cast_for_load`]-style casts after
/// reading, as `(resident bytes after the load, floor)`. Header-only.
///
/// Every KEPT tensor (`keep(name)`: a rung-4 streamed U-Net leaves its `transformer_blocks`
/// bodies to the block stream) ends resident at its stored bytes, or — when `casts(name, stored)`
/// (the loader's rule) — at its `target_bytes`-wide copy, each one page-rounded allocation
/// ([`mlx_gen::ios_quantum::allocated_bytes`]). The loader materializes the kept
/// tensors through `Weights::materialize_*`, which evaluates them in sorted-key batches of
/// [`mlx_gen::weights::Weights::MATERIALIZE_BATCH_BYTES`] resident bytes, each batch's safetensors
/// reads first: within a batch every read source and every cast copy are live together (the
/// device holds a command buffer's operands until it completes), on top of the batches already
/// resident. The floor is the worst batch, and at least the whole resident set.
pub(crate) fn cast_load_parts(
    files: &[std::path::PathBuf],
    target_bytes: u64,
    casts: impl Fn(&str, mlx_gen::gen_core::weightsmeta::Dtype) -> bool,
    keep: impl Fn(&str) -> bool,
) -> mlx_gen::Result<(u64, u64)> {
    let mut tensors: Vec<(String, u64, u64)> = Vec::new();
    for file in files {
        for header in mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(file)
            .map_err(|e| mlx_gen::Error::Msg(e.to_string()))?
        {
            if !keep(&header.name) {
                continue;
            }
            let stored = mlx_gen::ios_quantum::allocated_bytes(header.data_bytes);
            let (resident, live) = if casts(&header.name, header.dtype) {
                let cast = mlx_gen::ios_quantum::allocated_bytes(
                    header
                        .shape
                        .iter()
                        .fold(target_bytes, |a, &d| a.saturating_mul(d as u64)),
                );
                // The loader holds every source until the batch completes
                // (`loader::hold_cast_sources`), so the copy is never donated the source's buffer:
                // it ends resident at its own width, the source and the copy live together.
                (cast, cast.saturating_add(stored))
            } else {
                (stored, stored)
            };
            tensors.push((header.name, resident, live));
        }
    }
    tensors.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let batch = mlx_gen::weights::Weights::MATERIALIZE_BATCH_BYTES as u64;
    let (mut done, mut floor) = (0u64, 0u64);
    let (mut batch_resident, mut batch_live) = (0u64, 0u64);
    for (_, resident, live) in tensors {
        batch_resident = batch_resident.saturating_add(resident);
        batch_live = batch_live.saturating_add(live);
        if batch_resident >= batch {
            floor = floor.max(done.saturating_add(batch_live));
            done = done.saturating_add(batch_resident);
            (batch_resident, batch_live) = (0, 0);
        }
    }
    floor = floor.max(done.saturating_add(batch_live));
    done = done.saturating_add(batch_resident);
    Ok((done, floor.max(done)))
}
