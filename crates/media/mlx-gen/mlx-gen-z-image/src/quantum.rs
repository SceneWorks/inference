//! Z-Image iOS execution quanta (epic 16774, sc-26433): the labels and the COUNTED scratch every
//! guarded Z-Image quantum is priced with.
//!
//! Every price is `new live buffers + kernel transients + weight copies`, counted from the ops of
//! that quantum, read off the LOADED tensors and the request's actual activation shapes and dtype,
//! never off production constants. The device prices with [`Backend::Metal`]; each counted site is
//! proven by a CPU test (`quantum_tests`) that measures MLX's peak for a small shape and checks the
//! SAME counts with the CPU's kernel transients (composed RMSNorm, unfused SDPA, explicit-GEMM
//! convolution). The guards are `mlx_gen::ios_quantum::{run, before, finish}`, pass-throughs
//! whenever [`mlx_gen::ios_quantum::enabled`] is false (macOS, Linux, the simulator), so nothing
//! here changes host numerics or ordering (`quantum_tests::host_paths_match_the_base_revision_anchor`).
//!
//! Prices count MLX allocations — what a quantum's graph allocates on the device's allocator. The
//! host-side transients a few quanta build on the process heap (the text mask's fill, the init
//! image's resize, the decoded image's `u8` pixels) are a few MiB at the shipped geometry and are
//! covered by the [`mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES`] engineering reserve every check
//! adds, not priced per quantum.
//!
//! The AutoencoderKL convolution / GroupNorm / tiled-stage kernels and the SDPA score treatment are
//! the shared `mlx_gen::ios_quantum` accounting (also SDXL's and Anima's); the Qwen3 layer's
//! geometry is [`crate::text_encoder::EncoderLayer::quantum_geometry`] (also Anima's).

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::ios_quantum::{tensor_bytes, Backend, Scores};
use mlx_rs::Array;

/// Text encoder: token embedding gather, RoPE tables and the causal/padding mask.
pub(crate) const TE_EMBED: &str = "Z-Image Qwen3 embed";
/// One Qwen3 decoder layer (resident, or materialized from the snapshot under rung 4).
pub(crate) const TE_LAYER: &str = "Z-Image Qwen3 layer";
/// The text encoder's pre-load floor.
pub(crate) const TE_LOAD: &str = "Z-Image text encoder load";
/// The DiT's pre-load floor (its non-block tensors only under rung 4).
pub(crate) const DIT_LOAD: &str = "Z-Image DiT load";
/// The VAE's pre-load floor (decoder + encoder).
pub(crate) const VAE_LOAD: &str = "Z-Image VAE load";
/// The Fun-Controlnet-Union checkpoint's pre-load floor (its non-block tensors under rung 4).
pub(crate) const CONTROL_LOAD: &str = "Z-Image ControlNet load";
/// One resident dense ControlNet block's load-time Q4/Q8 packing.
pub(crate) const CONTROL_QUANTIZE: &str = "Z-Image ControlNet quantize";
/// The step-invariant DiT inputs: caption padding, position ids, RoPE gathers.
pub(crate) const DIT_PREPARE: &str = "Z-Image DiT prepare";
/// Timestep embedding + image patchify / `x_embedder` / pad tokens.
pub(crate) const DIT_EMBED: &str = "Z-Image DiT image embed";
/// Caption RMSNorm / `cap_embedder` / pad tokens.
pub(crate) const DIT_CAPTION: &str = "Z-Image DiT caption embed";
/// One adaLN DiT block (noise refiner or unified stack; a control block's base forward).
pub(crate) const DIT_BLOCK: &str = "Z-Image DiT block";
/// One context-refiner block (no adaLN).
pub(crate) const DIT_CONTEXT_BLOCK: &str = "Z-Image DiT context block";
/// The image + caption concatenation into the unified stream.
pub(crate) const DIT_UNIFY: &str = "Z-Image DiT unify";
/// Final layer, unpatchify and the velocity negation.
pub(crate) const DIT_OUTPUT: &str = "Z-Image DiT output";
/// The ControlNet context patchify / `control_all_x_embedder` / pad tokens.
pub(crate) const CONTROL_EMBED: &str = "Z-Image ControlNet embed";
/// One ControlNet block: (block 0: `before_proj` seed) + base forward + `after_proj` hint.
pub(crate) const CONTROL_BLOCK: &str = "Z-Image ControlNet block";
/// The control context's mask/inpaint channel concatenation.
pub(crate) const CONTROL_CONTEXT: &str = "Z-Image control context";
/// VAE decode de-normalization + decoder `conv_in`.
pub(crate) const VAE_DECODE_IN: &str = "Z-Image VAE decode conv_in";
/// VAE encoder `conv_in`.
pub(crate) const VAE_ENCODE_IN: &str = "Z-Image VAE encode conv_in";
/// One VAE resnet.
pub(crate) const VAE_RESNET: &str = "Z-Image VAE resnet";
/// The VAE mid-block single-head attention.
pub(crate) const VAE_ATTENTION: &str = "Z-Image VAE attention";
/// A VAE nearest-2x upsample plus its convolution.
pub(crate) const VAE_UPSAMPLE: &str = "Z-Image VAE upsample";
/// A VAE bottom/right-pad stride-2 downsample convolution.
pub(crate) const VAE_DOWNSAMPLE: &str = "Z-Image VAE downsample";
/// VAE decoder output GroupNorm + SiLU + `conv_out`.
pub(crate) const VAE_DECODE_OUT: &str = "Z-Image VAE decode conv_out";
/// VAE encoder output GroupNorm + SiLU + `conv_out` + the latent mean's affine.
pub(crate) const VAE_ENCODE_OUT: &str = "Z-Image VAE encode conv_out";
/// One halo-expanded tile of a layer-wise tiled VAE convolution, checked before its graph exists.
pub(crate) const VAE_TILE: &str = "Z-Image VAE tile";
/// The assembled output of one tiled VAE layer (resnet / upsample / output convolution).
pub(crate) const VAE_TILED_STAGE: &str = "Z-Image VAE tiled stage";
/// The decoded image's f32 cast and pixel conversion.
pub(crate) const DECODED_TO_IMAGE: &str = "Z-Image decoded to image";
/// The caption features: the valid-token gather of the encoder output and its compute-dtype cast.
pub(crate) const CAPTION_FEATURES: &str = "Z-Image caption features";
/// The latents quantum's price over `elements` latent elements: the seed's PRNG key and its split
/// (`[4, 2]` u32), the f32 normal draw and its bf16 cast, and (img2img) the blend's two scalars and
/// two f32 products (the sum donates).
pub(crate) fn latents_bytes(elements: u64, img2img: bool) -> u64 {
    ab(4, &[4, 2])
        + ab(4, &[elements])
        + ab(2, &[elements])
        + if img2img {
            ab(4, &[2]) + 2 * ab(4, &[elements])
        } else {
            0
        }
}

/// The pixel-conversion quantum's price over a decode of `elements` at `element_bytes`: the f32
/// cast of a narrower decode, and the denormalize/clip/round chain's f32 images (measured four
/// live: the scale, the clip pair and the row-contiguous readback copy of the transposed result)
/// and its scalars.
pub(crate) fn decoded_to_image_bytes(elements: u64, element_bytes: u64) -> u64 {
    // + the chain's four f32 scalars (0.5, 0, 1, 255).
    (4 + u64::from(element_bytes < 4)) * ab(4, &[elements]) + ab(4, &[4])
}

/// Seeded noise, its bf16 cast and the img2img init blend.
pub(crate) const LATENTS: &str = "Z-Image latents";
/// The img2img / control init image upload.
pub(crate) const INIT_IMAGE: &str = "Z-Image init image";

/// Bytes of a `[dims]` tensor at `element_bytes`.
pub(crate) fn tb(element_bytes: u64, dims: &[u64]) -> u64 {
    tensor_bytes(element_bytes, dims)
}

/// [`tb`] as the allocator holds it (page-rounded above one page).
pub(crate) fn ab(element_bytes: u64, dims: &[u64]) -> u64 {
    mlx_gen::ios_quantum::allocated_tensor_bytes(element_bytes, dims)
}

/// The dtype `x`-typed activations leave `lin` at: a dense matmul promotes to
/// `result_type(x, weight)`, a quantized one to `result_type(x, scales)`.
pub(crate) fn linear_out_dtype(lin: &AdaptableLinear, x: mlx_rs::Dtype) -> mlx_rs::Dtype {
    let weight = match lin.quantized_params() {
        Some((_, scales, ..)) => scales.dtype(),
        None => lin.weight_dtype().unwrap_or(x),
    };
    mlx_rs::Dtype::from_promoting_types(x, weight)
}

/// The width of the pad-token merge over an embedding of dtype `emb` (`apply_pad` keys the keep
/// mask off the pad token, so the products run at `result_type(emb, pad)`).
pub(crate) fn pad_width(emb: mlx_rs::Dtype, pad: &Array) -> u64 {
    mlx_gen::ios_quantum::dtype_bytes(mlx_rs::Dtype::from_promoting_types(emb, pad.dtype()))
}

pub(crate) fn u(v: i32) -> u64 {
    v.max(0) as u64
}

/// Buffers one `mlx_rs::fast::rms_norm` allocates over its input on `backend`: Metal's fused kernel
/// writes only its output; MLX's CPU backend runs the composed fallback (measured: 2.01 input-sized
/// buffers at the peak — the square and the normalized product; the weighted product donates).
pub(crate) fn rms_units(backend: Backend) -> u64 {
    match backend {
        Backend::Metal => 1,
        Backend::Cpu => 2,
    }
}

/// Logical elements of a DENSE projection's weight (and bias); `0` for a packed one.
pub(crate) fn dense_elements(lin: &AdaptableLinear) -> u64 {
    if lin.is_quantized() {
        return 0;
    }
    let product = |shape: &[i32]| {
        shape
            .iter()
            .fold(1u64, |a, &d| a.saturating_mul(d.max(0) as u64))
    };
    product(&lin.base_shape()).saturating_add(lin.bias().map_or(0, |b| product(b.shape())))
}

/// The stream-width copy `matmul` makes of a dense projection narrower than the stream
/// (`element_bytes`); a packed projection is consumed in place by `quantized_matmul`.
fn promoted_copy(lin: &AdaptableLinear, element_bytes: u64) -> u64 {
    if lin
        .weight_dtype()
        .is_some_and(|d| mlx_gen::ios_quantum::dtype_bytes(d) < element_bytes)
    {
        dense_elements(lin).saturating_mul(element_bytes)
    } else {
        0
    }
}

/// The largest stream-width weight copy among `lins` ([`promoted_copy`]): each copy is consumed by
/// its one matmul and released before the next projection's is built (measured: a bf16-stored
/// block over an f32 stream peaks one copy above its f32-stored twin).
pub(crate) fn promoted_weight_bytes(lins: &[&AdaptableLinear], element_bytes: u64) -> u64 {
    lins.iter()
        .map(|lin| promoted_copy(lin, element_bytes))
        .max()
        .unwrap_or(0)
}

/// The largest forward-time adapter residual chain among `lins` over `rows` activation rows
/// (`Adapter::residual`; the host's accumulation donates), with activations at `act_bytes`:
///
/// - LoRA: the rank-wide product and the output-wide residual (its scaling donates in place);
/// - dense LoKr: the delta's cast to the activation dtype (when it is stored at another width) and
///   the output-wide residual (its scaling donates in place);
/// - structured LoKr: the factors' casts, the `[rows, a, d]` intermediate and the output-wide
///   residual (the scale is baked into the factors).
///
/// Residuals are counted at f32, the widest they run. One projection's chain is consumed by its
/// accumulation before the next projection runs (measured).
pub(crate) fn adapter_bytes(lins: &[&AdaptableLinear], rows: u64, act_bytes: u64) -> u64 {
    use mlx_gen::adapters::Adapter;
    let cast = |w: &Array| {
        if w.item_size() as u64 == act_bytes {
            0
        } else {
            ab(
                act_bytes,
                &w.shape().iter().map(|&d| u(d)).collect::<Vec<_>>(),
            )
        }
    };
    lins.iter()
        .flat_map(|lin| {
            let out = lin.base_shape().first().map_or(0, |&d| u(d));
            lin.adapters()
                .iter()
                .filter(|a| !a.is_disabled())
                .map(move |a| {
                    // The output-wide residual and the scale scalar it is multiplied by.
                    let scaled = ab(4, &[rows, out]).saturating_add(ab(4, &[1]));
                    match a {
                        Adapter::Lora { a, .. } => {
                            let rank = a.shape().last().map_or(0, |&d| u(d));
                            ab(4, &[rows, rank]).saturating_add(scaled)
                        }
                        Adapter::Lokr { delta, .. } => cast(delta).saturating_add(scaled),
                        Adapter::LokrStructured { factors: f } => cast(&f.w1)
                            .saturating_add(cast(&f.w2))
                            .saturating_add(ab(4, &[rows, u(f.a), u(f.d)]))
                            .saturating_add(ab(4, &[rows, out])),
                    }
                })
        })
        .max()
        .unwrap_or(0)
}

/// The packed Q4/Q8 bytes `AdaptableLinear::quantize` produces from a dense `[out, in]` projection
/// (group 64): the packed weight, its bf16 scales and biases (each page-rounded as allocated), plus the bf16 cast copy of a wider
/// dense weight that the quantizer packs from (PARITY-BF16). `0` for an already packed projection.
pub(crate) fn quantize_bytes(lin: &AdaptableLinear, bits: i32) -> u64 {
    let Some((weight, bias)) = lin.dense_weight() else {
        return 0;
    };
    let shape = lin.base_shape();
    let (out, inp) = (
        shape.first().map_or(0, |&d| d.max(0) as u64),
        shape.get(1).map_or(0, |&d| d.max(0) as u64),
    );
    let groups = inp.div_ceil(crate::quant::GROUP_SIZE as u64);
    let packed = mlx_gen::ios_quantum::allocated_bytes(
        out.saturating_mul(inp)
            .saturating_mul(bits.max(0) as u64)
            .div_ceil(8),
    )
    .saturating_add(2 * ab(2, &[out, groups]));
    let cast = |a: &Array| {
        if a.item_size() > 2 {
            mlx_gen::ios_quantum::bytes(a) / a.item_size() as u64 * 2
        } else {
            0
        }
    };
    packed
        .saturating_add(cast(weight))
        .saturating_add(bias.map_or(0, cast))
}

/// One DiT attention block's shape, as its quantum is priced.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DitBlockShape {
    pub(crate) batch: u64,
    pub(crate) tokens: u64,
    pub(crate) dim: u64,
    pub(crate) heads: u64,
    pub(crate) head_dim: u64,
    pub(crate) ff: u64,
    /// Query rows per SDPA call (`tokens` unchunked).
    pub(crate) query_rows: u64,
    /// The stream width: the f32 RoPE table promotes q/k and everything after them.
    pub(crate) element_bytes: u64,
    /// `true` for an adaLN block (`ZImageTransformerBlock`), `false` for a context block.
    pub(crate) adaln: bool,
}

/// One DiT block forward (`ZImageTransformerBlock::forward_budgeted` / the context block's). On iOS
/// the block is two phases: its attention half is materialized (a sub-quantum barrier) before the
/// MLP half is built, and a chunked attention evaluates each query chunk. Each phase is counted as
/// its new buffers, in `[B, T, D]` units (an RMSNorm is [`rms_units`]):
///
/// * attention, before the scores (`pre`): attention_norm1; adaLN only, the modulated input; q, k,
///   v; the q/k per-head norms; the complex RoPE of q and of k (its half-split copies, rotated
///   halves and their stack: 3 each);
/// * the SDPA call (`sdpa`): its output; the unfused CPU fallback's scaled query and the
///   attention-weighted product before its transpose (none on Metal's fused kernel);
/// * after it (`post`): the head-merge copy, `to_out`, attention_norm2 and the residual
///   (`x + gate·normed`: one fused kernel under compiled glue, two eager ops otherwise; a context
///   block's plain `add`: one);
/// * a chunked attention's per-chunk outputs and their concatenation (2), its scores one chunk at a
///   time, and — the chunk evaluations being barriers — the pre-attention temporaries released
///   before the rest is built: `max(pre + sdpa + chunk scores, q/k/v + sdpa + 2 + post)`;
/// * the MLP half: the materialized attention result, ffn_norm1, (adaLN) the modulated input,
///   `w2`, ffn_norm2 and the residual; in `[B, T, FF]` units `w1`, `w3` and the SwiGLU gate (one
///   fused kernel compiled; `sigmoid` and the product eager).
///
/// Plus the scores ([`dit_scores`]: fused on Metal at the shipped head dim 128; one score tensor on
/// the CPU fallback — the single-head-softmax runs in place, measured) and the adaLN modulation's
/// small `[B, 4D]` vectors (projection, the two `+ 1`, the two `tanh`).
///
/// `attention_extra` / `mlp_extra` are what each half's projections add while it runs (the largest
/// stream-width weight copy, the largest adapter residual chain); the adaLN modulation runs with the
/// attention half.
pub(crate) fn dit_block_bytes(
    s: DitBlockShape,
    backend: Backend,
    compiled: bool,
    attention_extra: u64,
    mlp_extra: u64,
) -> u64 {
    let r = rms_units(backend);
    let scores = dit_scores(backend, s.head_dim, s.element_bytes, s.tokens, s.query_rows);
    let materialized = matches!(scores, Scores::Materialized { .. });
    let modulated = u64::from(s.adaln);
    let residual = if s.adaln && !compiled { 2 } else { 1 };
    let pre = r + modulated + 3 + 2 * r + 6;
    let sdpa = 1 + if materialized { 2 } else { 0 };
    let post = 1 + 1 + r + residual;
    let alloc = mlx_gen::ios_quantum::allocated_bytes;
    let hidden = alloc(tb(s.element_bytes, &[s.batch, s.tokens, s.dim]));
    let score_bytes = |rows: u64| {
        alloc(mlx_gen::ios_quantum::scores_bytes(
            scores, s.batch, s.heads, rows, s.tokens,
        ))
    };
    let attention = if s.query_rows < s.tokens {
        (hidden * (pre + sdpa) + score_bytes(s.query_rows)).max(hidden * (3 + sdpa + 2 + post))
    } else {
        hidden * (pre + sdpa + post) + score_bytes(s.tokens)
    };
    let ff = alloc(tb(s.element_bytes, &[s.batch, s.tokens, s.ff]));
    let mlp =
        hidden * (1 + r + modulated + 1 + r + residual) + ff * (2 + if compiled { 1 } else { 2 });
    let modulation = if s.adaln {
        5 * alloc(tb(4, &[s.batch, 4 * s.dim]))
    } else {
        0
    };
    (attention + attention_extra)
        .max(mlp + mlp_extra)
        .saturating_add(modulation)
}

/// The score treatment of one DiT attention over `tokens` keys, `query_rows` per call: fused on
/// Metal where its kernel serves the shape ([`mlx_gen::ios_quantum::sdpa_scores`]); otherwise ONE
/// materialized score tensor per call (measured: the unmasked softmax runs in place).
pub(crate) fn dit_scores(
    backend: Backend,
    head_dim: u64,
    element_bytes: u64,
    tokens: u64,
    query_rows: u64,
) -> Scores {
    match mlx_gen::ios_quantum::sdpa_scores(
        backend,
        head_dim,
        element_bytes,
        &[(tokens, query_rows, tokens)],
    ) {
        Scores::Fused => Scores::Fused,
        Scores::Materialized { element_bytes, .. } => Scores::Materialized {
            live: 1,
            element_bytes,
        },
    }
}

/// `[B, C, H, W]` (NCHW) → the NHWC shape `[B, H, W, C]` the convolution accounting reads.
pub(crate) fn nhwc(x_nchw: &Array) -> [i32; 4] {
    let s = x_nchw.shape();
    [s[0], s[2], s[3], s[1]]
}

/// Check one halo-expanded tile of a layer-wise tiled VAE convolution BEFORE its norm/activation/
/// conv graph is constructed, priced at the tile's extent and `weight`'s channel counts
/// ([`mlx_gen::ios_quantum::tile_bytes`]). Called from the per-tile preprocess hook of
/// `mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc`, which evaluates each output core before the next
/// tile is built.
pub(crate) fn before_tile(tile: &Array, weight: &Array) -> mlx_gen::Result<()> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(());
    }
    mlx_gen::ios_quantum::before(
        VAE_TILE,
        mlx_gen::ios_quantum::tile_bytes(
            tile.shape(),
            tile.item_size() as u64,
            weight,
            false,
            Backend::current(),
        ),
    )
}

/// One layer-wise tiled VAE stage's check over `x_nhwc` producing `cout` channels
/// ([`mlx_gen::ios_quantum::tiled_stage_bytes`]); a no-op off device.
pub(crate) fn before_tiled_stage(
    x_nhwc: &[i32],
    act: u64,
    weight: &Array,
    resnet: bool,
) -> mlx_gen::Result<()> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(());
    }
    let (cin, cout) = mlx_gen::ios_quantum::conv_channels(weight);
    mlx_gen::ios_quantum::before(
        VAE_TILED_STAGE,
        mlx_gen::ios_quantum::tiled_stage_bytes(x_nhwc, cin, cout, act, resnet),
    )
}

/// A convolutional quantum whose price is only computed on device (pricing reads shapes; off
/// device the guard is a pass-through and the price is never needed).
pub(crate) fn conv_quantum(
    label: &str,
    price: impl FnOnce(Backend) -> u64,
    build: impl FnOnce() -> mlx_gen::Result<Array>,
) -> mlx_gen::Result<Array> {
    let scratch = if mlx_gen::ios_quantum::enabled() {
        price(Backend::current())
    } else {
        0
    };
    mlx_gen::ios_quantum::run(label, scratch, build)
}

/// The `.safetensors` files a component source loads (a directory's non-hidden shards, or the one
/// file), for the header-only pre-load floors. Off device nothing is listed.
pub(crate) fn source_files(
    source: &mlx_gen::WeightsSource,
) -> mlx_gen::Result<Vec<std::path::PathBuf>> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(Vec::new());
    }
    Ok(match source {
        mlx_gen::WeightsSource::File(path) => vec![path.clone()],
        mlx_gen::WeightsSource::Dir(dir) => {
            let mut files = Vec::new();
            for entry in std::fs::read_dir(dir)? {
                let path = entry?.path();
                if path.extension().and_then(|e| e.to_str()) == Some("safetensors")
                    && !mlx_gen::gen_core::weightsmeta::is_hidden_file(&path)
                {
                    files.push(path);
                }
            }
            files.sort();
            files
        }
    })
}

/// The header-only pre-load floor of `source` over the tensors `resident(name)` keeps
/// ([`mlx_gen::ios_quantum::before_load_where`]); a no-op off device.
pub(crate) fn before_load(
    label: &str,
    source: &mlx_gen::WeightsSource,
    resident: impl Fn(&str) -> bool,
) -> mlx_gen::Result<()> {
    mlx_gen::ios_quantum::before_load_where(label, &source_files(source)?, resident)
}

/// An iOS sub-quantum barrier: materialize `arrays` so the buffers of the ops that produced them are
/// released before the rest of the quantum's graph is built (MLX keeps an encoded op's inputs alive
/// until its command buffer completes, so without it a whole block's intermediates are live
/// together). The quantum's own `finish` still synchronizes and drains. Values are unchanged; a
/// no-op off device.
pub(crate) fn barrier(arrays: &[&Array]) -> mlx_gen::Result<()> {
    if mlx_gen::ios_quantum::enabled() {
        mlx_rs::transforms::eval(arrays.iter().copied())?;
        #[cfg(test)]
        BARRIER_HOOK.with(|hook| {
            if let Some(hook) = hook.borrow_mut().as_mut() {
                hook();
            }
        });
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Run after every sub-quantum barrier's evaluation — the quantum proofs' seam to measure each
    /// phase from its own start (`crate::quantum_proofs`). Test builds only.
    pub(crate) static BARRIER_HOOK: std::cell::RefCell<Option<Box<dyn FnMut()>>> =
        const { std::cell::RefCell::new(None) };
}
