//! Per-phase host and GPU **estimates** for one MiniMax-H3-Image request on CUDA (SW-7095).
//!
//! Nothing here is measured (E13: the anchors are measured once, in the feature's terminal item).
//! Every figure is structural arithmetic over the resolved sources' safetensors headers and the
//! request geometry:
//!
//! * the text encoder's layer-50 tap and its streamed layer stack
//!   (`crate::memory_strategy::text_encoder_stream`), plus its f32 working set;
//! * the Qwen3-VL vision tower at the f32 width it is loaded at, for an edit;
//! * the image DiT through H3's own CUDA window allocation
//!   ([`gen_core::h3_memory::window::WindowAllocation`]): root modules (projections, refiner, the
//!   six `pdd_proj_out` heads), the block stack, the AdaLN projections that are precomputed and
//!   evicted, the retained modulation table, and the CUDA working set — which prices the
//!   FlashAttention route **and** the query-chunked materializing fallback and takes the larger,
//!   so the estimate holds for whichever of the two actually executes;
//! * the LoRA stack: every installed factor in full, plus its forward working memory inside one
//!   block (on the streamed route each window carries its block's factors);
//! * the image VAE encode (edit references) and the tiled 256 / 64 decode.
//!
//! Successive lifetimes take a **maximum**, never a sum: the generator stages vision tower → text
//! encoder → VAE encoder → DiT → VAE decoder, releasing and synchronizing each before the next
//! (rung 1). Host and GPU bytes are separate columns of every phase. [`decimal_gb`] and
//! [`binary_gib`] are the only unit conversions, and every rendering names which one it uses.

use std::fmt;
use std::path::Path;

use candle_gen::gen_core::h3_image as contract;
use candle_gen::gen_core::h3_memory::window::WindowAllocation;
use candle_gen::gen_core::h3_memory::{NativeRows, NativeTaskAllocation, NativeTaskSelection};
use candle_gen::gen_core::{
    self, weightsmeta, GenerationRequest, LoadSpec, MemoryBudget, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemorySelection, MemoryStrategy, MemoryStrategyParameters,
    StreamedWeightFacts, TransformerComponent, WeightsSource,
};

use crate::image_model::{
    image_dit_config_at, resolve_dit_dir, resolve_text_encoder_dir, IMAGE_MODEL_ID,
};
use crate::image_vae::{IMAGE_DECODE_OVERLAP, IMAGE_DECODE_TILE_EDGE, IMAGE_VAE_COMPONENT};

/// Bytes in one decimal gigabyte (GB, 10^9).
pub const BYTES_PER_GB: u64 = 1_000_000_000;
/// Bytes in one binary gibibyte (GiB, 2^30).
pub const BYTES_PER_GIB: u64 = 1 << 30;

/// `bytes` in decimal gigabytes.
pub fn decimal_gb(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GB as f64
}

/// `bytes` in binary gibibytes.
pub fn binary_gib(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GIB as f64
}

/// Pixels per DiT token side: 16 px per VAE latent cell × the 2 × 2 patch.
pub const PIXELS_PER_TOKEN_SIDE: u32 = contract::VAE_SPATIAL_RATIO * 2;

/// Vision-tower patch edge in pixels (`minimax_h3_vision_config().patch_size`).
const VISION_PATCH: u32 = 16;

/// Presentation tokens around each reference's merged vision block: the `Picture i: ` label and
/// the vision start/end markers. A bound, not a tokenizer count.
const LABEL_TOKENS_BOUND: u64 = 16;

/// The image VAE's tile with its overlap on both sides, in pixels (256 + 2 · 64).
const TILE_WITH_OVERLAP: u64 = 384;

/// Widest encoder feature map at full tile resolution (`ENCODER_BLOCK_OUT_CHANNELS[1..]` at
/// level 0 after the first residual widen is 256).
const ENCODE_TILE_CHANNELS: u64 = 256;

/// Live encoder feature-map buffers at once inside one tile (input, two residual intermediates,
/// the skip, the norm output and its cast). Structural bound, not measured.
const ENCODE_TILE_BUFFERS: u64 = 6;

/// The decoder ViT width (`DecoderLayout::Image2d`, 1024) and its live per-token buffers.
const DECODE_HIDDEN: u64 = 1024;
const DECODE_TOKEN_BUFFERS: u64 = 24;

/// One phase of the staged render, in execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImagePhase {
    /// Edit only: the f32 Qwen3-VL vision tower over every reference, then released.
    VisionTower,
    /// The layer-50 text-encoder pass, then released.
    Conditioning,
    /// Edit only: the image VAE encode of every reference, then released.
    ReferenceEncode,
    /// The DiT load, AdaLN precompute-and-evict and the six-step loop, then released.
    Denoise,
    /// The tiled 2-D VAE decode.
    Decode,
}

impl ImagePhase {
    /// The shared lifecycle phase this image phase belongs to: the vision tower and the reference
    /// encode are conditioning work, released before the DiT is mapped.
    pub fn memory_phase(self) -> MemoryPhase {
        match self {
            Self::VisionTower | Self::Conditioning | Self::ReferenceEncode => {
                MemoryPhase::Conditioning
            }
            Self::Denoise => MemoryPhase::Denoise,
            Self::Decode => MemoryPhase::Decode,
        }
    }
}

/// One phase's bytes, in the two pools (the MLX twin's `ImagePhaseBytes`). On CUDA the pools
/// are physically separate, so the two columns are compared to their own free figures and never
/// added.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImagePhaseBytes {
    pub phase: ImagePhase,
    /// CUDA device allocations: weights, activations, the stitched canvas.
    pub device_bytes: u64,
    /// Host RAM: prepared host-packed stacks, source maps, resized references, the read-back
    /// canvas and the 8-bit output.
    pub host_bytes: u64,
}

/// The per-phase estimate of one request at one selection (the MLX twin's
/// `ImageMemoryEstimate`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryEstimate {
    pub selection: MemorySelection,
    /// Whether the text encoder and the DiT blocks run through the one-block window (rung 4).
    pub streamed: bool,
    /// In execution order; an edit adds the vision tower and the reference encode.
    pub phases: Vec<ImagePhaseBytes>,
}

impl ImageMemoryEstimate {
    /// The largest device column — phases are successive lifetimes.
    pub fn peak_device_bytes(&self) -> u64 {
        self.phases
            .iter()
            .map(|p| p.device_bytes)
            .max()
            .unwrap_or(0)
    }

    /// The largest host column.
    pub fn peak_host_bytes(&self) -> u64 {
        self.phases.iter().map(|p| p.host_bytes).max().unwrap_or(0)
    }

    /// The phase that sets [`Self::peak_device_bytes`].
    pub fn binding_phase(&self) -> Option<ImagePhase> {
        self.phases
            .iter()
            .max_by_key(|p| p.device_bytes)
            .map(|p| p.phase)
    }

    /// Whether the device peak fits `free_device_bytes` (the device's actually-free memory), and
    /// the host peak fits `free_host_bytes` when one is given.
    pub fn fits(&self, free_device_bytes: u64, free_host_bytes: Option<u64>) -> bool {
        self.peak_device_bytes() <= free_device_bytes
            && free_host_bytes.is_none_or(|host| self.peak_host_bytes() <= host)
    }
}

impl fmt::Display for ImageMemoryEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device = self.peak_device_bytes();
        let host = self.peak_host_bytes();
        write!(
            f,
            "{:?}{} estimate: GPU peak {:.2} GB ({:.2} GiB) in {:?}, host peak {:.2} GB ({:.2} GiB)",
            self.selection.strategy,
            if self.streamed { " (streamed)" } else { "" },
            decimal_gb(device),
            binary_gib(device),
            self.binding_phase(),
            decimal_gb(host),
            binary_gib(host),
        )?;
        for p in &self.phases {
            write!(
                f,
                "; {:?} GPU {:.2} GB / host {:.2} GB",
                p.phase,
                decimal_gb(p.device_bytes),
                decimal_gb(p.host_bytes)
            )?;
        }
        Ok(())
    }
}

/// What one request asks for, as admission sees it (the MLX twin's `ImageRequestShape`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageRequestShape {
    pub width: u32,
    pub height: u32,
    /// Images rendered sequentially over one encoded prompt and one loaded DiT; the peak is one
    /// image's.
    pub count: u32,
    /// Prompt tokens before truncation; `None` prices the 4096-token maximum.
    pub prompt_tokens: Option<u32>,
    /// Each edit reference's original `(width, height)`; empty for `text2image`.
    pub references: Vec<(u32, u32)>,
}

impl ImageRequestShape {
    /// A `text2image` request at the maximum prompt length.
    pub fn text_to_image(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            count: 1,
            prompt_tokens: None,
            references: Vec::new(),
        }
    }

    /// The admission geometry alone (`width x height` output, `reference_count` references of
    /// unknown size): each reference is priced at the largest square VAE copy the contract keeps
    /// (2048², 4096 rows), so the bound holds for any reference set of that count.
    pub fn with_reference_count(width: u32, height: u32, reference_count: u32) -> Self {
        let edge = contract::REFERENCE_MAX_SHORT_EDGE;
        Self {
            references: vec![(edge, edge); reference_count as usize],
            ..Self::text_to_image(width, height)
        }
    }

    /// The exact geometry of `req`: its output size (`0 x 0` is the 2048² default) and each
    /// reference's own size. The prompt is priced at its 4096-token cap.
    pub fn from_request(req: &GenerationRequest) -> gen_core::Result<Self> {
        let (width, height) = if req.width == 0 && req.height == 0 {
            (
                crate::image_model::DEFAULT_SIZE,
                crate::image_model::DEFAULT_SIZE,
            )
        } else {
            (req.width, req.height)
        };
        Ok(Self {
            width,
            height,
            count: req.count.max(1),
            prompt_tokens: None,
            references: contract::edit_reference_images(&req.conditioning)?
                .iter()
                .map(|r| (r.width, r.height))
                .collect(),
        })
    }
}

/// The token and pixel counts one request puts through each phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageRequestRows {
    /// The 32-multiple generation canvas.
    pub canvas: (u32, u32),
    /// The requested output size the canvas is resized back to.
    pub output: (u32, u32),
    /// Generated-image token rows: `(canvas_w / 32) · (canvas_h / 32)`.
    pub image_rows: u64,
    /// Text rows: the prompt plus every reference's merged vision block and label.
    pub text_rows: u64,
    /// DiT token rows of each reference's VAE copy.
    pub reference_rows: Vec<u64>,
    /// Pixels of each reference's VAE copy.
    pub reference_pixels: Vec<u64>,
    /// Pre-merge vision-tower patches of each reference's vision copy.
    pub vision_patches: Vec<u64>,
}

impl ImageRequestRows {
    /// The rows `shape` puts through each phase.
    pub fn of(shape: &ImageRequestShape) -> Self {
        let output = (shape.width, shape.height);
        let canvas =
            contract::generation_canvas(output.0, output.1, contract::MIN_GENERATION_AREA_DEFAULT);
        let tokens = |w: u32, h: u32| {
            u64::from(w / PIXELS_PER_TOKEN_SIDE) * u64::from(h / PIXELS_PER_TOKEN_SIDE)
        };
        let mut reference_rows = Vec::with_capacity(shape.references.len());
        let mut reference_pixels = Vec::with_capacity(shape.references.len());
        let mut vision_patches = Vec::with_capacity(shape.references.len());
        let mut text_rows = shape
            .prompt_tokens
            .map_or(contract::MAX_PROMPT_TOKENS as u64, |n| {
                u64::from(n).min(contract::MAX_PROMPT_TOKENS as u64)
            });
        for &(w, h) in &shape.references {
            let (vw, vh) = contract::reference_vae_size(w, h);
            let (lh, lw) = contract::reference_latent_grid(vw, vh, (2, 2));
            reference_rows.push(u64::from(lh / 2) * u64::from(lw / 2));
            reference_pixels.push(u64::from(vw) * u64::from(vh));
            let (sw, sh) = contract::reference_vision_size(w, h);
            let patches = u64::from(sw / VISION_PATCH) * u64::from(sh / VISION_PATCH);
            vision_patches.push(patches);
            text_rows += patches / contract::VISION_MERGE_TOKENS as u64 + LABEL_TOKENS_BOUND;
        }
        Self {
            canvas,
            output,
            image_rows: tokens(canvas.0, canvas.1),
            text_rows,
            reference_rows,
            reference_pixels,
            vision_patches,
        }
    }

    /// Rows of the packed DiT sequence: text, references, then the generated image.
    pub fn sequence_rows(&self) -> u64 {
        self.text_rows + self.reference_rows.iter().sum::<u64>() + self.image_rows
    }

    fn has_references(&self) -> bool {
        !self.reference_rows.is_empty()
    }
}

/// The text encoder's shape, read from its own `config.json`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TeDims {
    hidden: u64,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    intermediate: u64,
}

/// Header-only facts about one load, built once ([`ImageMemoryFacts::resolve`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryFacts {
    /// The layer-50 tap's root (embedding, rotary tables) and its 50-layer stack, without the
    /// vision tower, at the runtime format the encoder holds them in.
    te: StreamedWeightFacts,
    te_dims: TeDims,
    /// The largest one-projection f32 decode during a TE layer.
    te_decode: u64,
    /// Host bytes to prepare the host-packed TE (source map + runtime packs + one repack scratch).
    te_prepare_host: u64,
    /// The f32 vision tower.
    vision_bytes: u64,
    dit_hidden: u64,
    dit_text_dim: u64,
    dit_patch_dim: u64,
    task: NativeTaskAllocation,
    window: WindowAllocation,
    /// Host bytes to prepare the DiT (source map + runtime packs + one repack scratch).
    dit_prepare_host: u64,
    /// Host bytes of the runtime-format DiT stack a streamed request keeps on the host.
    dit_retained_host: u64,
    /// Every installed adapter factor plus its prepared cache.
    adapter_resident: u64,
    /// The adapter install high-water (not additional to `adapter_resident`).
    adapter_install: u64,
    decoder_bytes: u64,
    encoder_bytes: u64,
    /// The streamable DiT as the shared phase facts spell it ([`transformer_stream`]).
    stream: StreamedWeightFacts,
    /// Whether the source supports the streamed route (rung 4).
    pub streamable: bool,
}

impl ImageMemoryFacts {
    /// The resident trunk (I/O projections, the six per-step heads, the refiner) plus one stack
    /// of 50 per-block byte counts (body + `adaln_proj`) — the contract's
    /// `phase_facts.transformer_stream`, spelled as the MLX twin spells it.
    pub fn transformer_stream(&self) -> &StreamedWeightFacts {
        &self.stream
    }
}

/// [`ImageMemoryFacts::transformer_stream`] from the DiT allocation's per-tensor runtime bytes:
/// every `transformer_blocks.{i}.*` tensor counts toward block `i`, everything else toward the
/// trunk, so the unwindowed sum is the loaded DiT.
pub(crate) fn transformer_stream(
    resident_by_name: &[(String, u64)],
    num_layers: usize,
) -> gen_core::Result<StreamedWeightFacts> {
    let mut trunk = 0u64;
    let mut blocks = vec![0u64; num_layers];
    for (name, bytes) in resident_by_name {
        let block = name
            .strip_prefix("transformer_blocks.")
            .and_then(|tail| tail.split_once('.'))
            .and_then(|(index, _)| index.parse::<usize>().ok());
        match block.and_then(|i| blocks.get_mut(i)) {
            Some(slot) => *slot = add_all([*slot, *bytes])?,
            None => trunk = add_all([trunk, *bytes])?,
        }
    }
    Ok(StreamedWeightFacts {
        resident_bytes: trunk,
        stacks: vec![blocks],
    })
}

fn add_all(values: impl IntoIterator<Item = u64>) -> gen_core::Result<u64> {
    gen_core::checked_byte_total(IMAGE_MODEL_ID, "phase estimate", values)
}

fn mul(a: u64, b: u64) -> gen_core::Result<u64> {
    a.checked_mul(b).ok_or_else(|| {
        gen_core::Error::Unsupported(format!("{IMAGE_MODEL_ID}: phase estimate overflow"))
    })
}

/// Bytes of the tensors under `path` that `keep` selects, floats at `width` bytes.
fn header_bytes(path: &Path, width: u64, keep: impl Fn(&str) -> bool) -> gen_core::Result<u64> {
    weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .filter(|h| keep(&h.name))
        .try_fold(0u64, |sum, h| {
            let bytes = if h.is_float() {
                h.materialized_bytes(width)?
            } else {
                h.data_bytes
            };
            add_all([sum, bytes])
        })
}

impl ImageMemoryFacts {
    /// Read the load's facts from the resolved sources' headers. No tensor is read.
    pub fn resolve(spec: &LoadSpec) -> gen_core::Result<ImageMemoryFacts> {
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(gen_core::Error::Unsupported(format!(
                "{IMAGE_MODEL_ID}: memory estimates require a snapshot directory"
            )));
        };
        let unsupported = |e: candle_gen::CandleError| gen_core::Error::Unsupported(e.to_string());
        let te_dir = resolve_text_encoder_dir(root, spec);
        let dit_dir = resolve_dit_dir(root, spec);

        // --- text encoder -------------------------------------------------------------------------
        let te_cfg = crate::text_encoder::MiniMaxH3TeConfig::from_component_dir(&te_dir)
            .map_err(unsupported)?;
        let mut te = crate::memory_strategy::text_encoder_stream(&te_dir)?.ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{IMAGE_MODEL_ID}: the text encoder has no shards"
            ))
        })?;
        let (payload, te_prepare_host, te_decode) =
            crate::memory_strategy::runtime_payloads(&te_dir, true)?;
        // `text_encoder_stream` folds the vision tower into the root for H3 video's grounded pass;
        // here the tower is its own earlier phase, so it is taken back out of the encoder's root.
        let vision_runtime = payload
            .iter()
            .filter(|(name, _)| name.starts_with(crate::text_encoder::VISION_PREFIX))
            .try_fold(0u64, |n, (_, &b)| add_all([n, b]))?;
        te.resident_bytes = te.resident_bytes.saturating_sub(vision_runtime);
        let vision_bytes = header_bytes(&te_dir, 4, |name| {
            name.starts_with(crate::text_encoder::VISION_PREFIX)
        })?;

        // --- DiT ------------------------------------------------------------------------------------
        let dit_cfg = image_dit_config_at(&dit_dir).map_err(unsupported)?;
        let allocation = crate::memory_strategy::dit_allocation(&dit_dir, false)?;
        let window = crate::memory_strategy::dit_window_for(
            &dit_dir,
            &dit_cfg,
            &allocation,
            allocation.resident_bytes,
            false,
        )?;
        let streamable = window.is_some()
            && crate::image_model::image_streaming_supported(spec).map_err(unsupported)?;
        let window = window.ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{IMAGE_MODEL_ID}: the DiT source has no CUDA block-window inventory to price"
            ))
        })?;
        let adapter = if spec.adapters.is_empty() {
            None
        } else {
            Some(crate::memory_strategy::adapter_profile_for(
                &dit_dir,
                &crate::adapters::image_adapter_target_paths(&dit_cfg),
                &spec.adapters,
                &allocation.projections,
            )?)
        };
        let (adapter_resident, adapter_install) = adapter
            .as_ref()
            .map_or((0, 0), |p| (p.resident_bytes, p.install_peak_bytes));
        let stream = transformer_stream(&allocation.resident_by_name, dit_cfg.num_layers)?;
        let (dit_payload, dit_prepare_host, _) =
            crate::memory_strategy::runtime_payloads(&dit_dir, false)?;
        let dit_retained_host = add_all(dit_payload.values().copied())?;
        let task = NativeTaskAllocation {
            adapter,
            resident_bytes: allocation.resident_bytes,
            constructor_bytes: allocation.constructor_bytes,
            patch_size: dit_cfg.patch_size.map(|n| n as u64),
            reference_task: false,
            window: Some(window.clone()),
            evicted_bytes: allocation.adaln_bytes,
            table_bytes_per_timestep: allocation.table_bytes_per_timestep,
            projections: allocation.projections,
        };

        // --- VAE --------------------------------------------------------------------------------------
        let vae = root.join(IMAGE_VAE_COMPONENT);
        let decoder_bytes = header_bytes(&vae, 4, |name| {
            name.starts_with("decoder.") || name.starts_with("post_quant_conv.")
        })?;
        let encoder_bytes = header_bytes(&vae, 4, |name| {
            name.starts_with("encoder.") || name.starts_with("quant_conv.")
        })?;

        Ok(ImageMemoryFacts {
            te,
            te_dims: TeDims {
                hidden: te_cfg.hidden_size as u64,
                heads: te_cfg.num_heads as u64,
                kv_heads: te_cfg.num_kv_heads as u64,
                head_dim: te_cfg.head_dim as u64,
                intermediate: te_cfg.intermediate_size as u64,
            },
            te_decode,
            te_prepare_host,
            vision_bytes,
            dit_hidden: dit_cfg.hidden_size as u64,
            dit_text_dim: dit_cfg.text_dim as u64,
            dit_patch_dim: dit_cfg.video_patch_dim() as u64,
            task,
            window,
            dit_prepare_host,
            dit_retained_host,
            adapter_resident,
            adapter_install,
            decoder_bytes,
            encoder_bytes,
            stream,
            streamable,
        })
    }
}

/// One pre-norm transformer block's live working set over `rows` tokens: the residual and
/// normalized hidden, the attention inputs (K/V repeated to every query head), the
/// materializing fallback's query chunk of scores at `score_budget` elements (16 bytes per score
/// covers scores, scaling and softmax), and the FFN's budget-chunked gate/up/product with the
/// concatenated chunk outputs. Attention and FFN do not overlap.
#[allow(clippy::too_many_arguments)]
fn block_working_bytes(
    rows: u64,
    hidden: u64,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    intermediate: u64,
    element_bytes: u64,
    score_budget: u64,
) -> gen_core::Result<u64> {
    let hidden_buf = mul(mul(rows, hidden)?, element_bytes)?;
    let inner = mul(mul(mul(rows, heads)?, head_dim)?, element_bytes)?;
    let kv = mul(mul(mul(rows, kv_heads)?, head_dim)?, element_bytes)?;
    let score_row = mul(heads, rows)?.max(1);
    let query_rows = (score_budget / score_row).clamp(1, rows.max(1));
    let scores = mul(query_rows, score_row)?;
    // q, k, v, the repeated k/v at every head, their contiguous copies and the output, plus the
    // materializing fallback's float32 `kᵀ` (`crate::nn::sdpa_materialized` computes half-dtype
    // scores in f32): twice a half-dtype key copy, one of which the cast replaces.
    let f32_keys_extra = if element_bytes < 4 { inner } else { 0 };
    let attention = add_all([
        mul(6, inner)?,
        f32_keys_extra,
        mul(2, kv)?,
        mul(16, scores)?,
    ])?;
    let ffn_elements = mul(mul(rows, 2)?, intermediate)?.min(score_budget.max(1));
    let ffn = add_all([mul(3 * element_bytes, ffn_elements)?, mul(2, hidden_buf)?])?;
    add_all([mul(4, hidden_buf)?, attention.max(ffn)])
}

/// The materializing attention budget a selection executes with: its bounded-attention chunk
/// when `engaged`, else the CUDA default. A larger chunk never widens it.
pub fn score_budget(engaged: bool, attention_chunk_size: Option<u32>) -> u64 {
    let default = gen_core::veda::workspace::CUDA_DENSE_SCORE_ELEMENTS;
    attention_chunk_size
        .filter(|_| engaged)
        .map_or(default, |n| u64::from(n).min(default))
        .max(1)
}

/// The per-phase estimate of `shape` at `selection` on `contract` — see the module docs for what
/// each column holds. An estimate, not a measurement.
pub fn estimate(
    contract: &MemoryProviderContract,
    facts: &ImageMemoryFacts,
    shape: &ImageRequestShape,
    selection: &MemorySelection,
) -> gen_core::Result<ImageMemoryEstimate> {
    let rows = &ImageRequestRows::of(shape);
    let streamed =
        contract.engages_selection(selection, MemoryStrategy::BoundedTransformerResidency);
    if streamed && !facts.streamable {
        return Err(gen_core::Error::Unsupported(format!(
            "{IMAGE_MODEL_ID}: this source does not support the streamed route"
        )));
    }
    let budget = score_budget(
        contract.engages_selection(selection, MemoryStrategy::BoundedAttention),
        selection.parameters.attention_chunk_size,
    );
    let mut phases = Vec::with_capacity(5);
    let f32_bytes = 4u64;
    let merged_vision: u64 = rows
        .vision_patches
        .iter()
        .map(|p| p / contract::VISION_MERGE_TOKENS as u64)
        .sum();
    // The tower's merged embeds plus its three deepstack features, f32 at the conditioner width.
    let grounded = mul(mul(merged_vision, facts.dit_text_dim)?, f32_bytes * 4)?;
    let reference_pixels_host: u64 = rows.reference_pixels.iter().sum::<u64>() * 3 * (1 + 4);

    // --- 1. vision tower (edit) ------------------------------------------------------------------
    if rows.has_references() {
        let largest = rows.vision_patches.iter().copied().max().unwrap_or(0);
        let tower = minimax_vision_dims();
        let work = block_working_bytes(
            largest, tower.0, tower.1, tower.1, tower.2, tower.3, f32_bytes, budget,
        )?;
        let pixels = mul(largest, u64::from(VISION_PATCH * VISION_PATCH) * 3 * 4 * 2)?;
        phases.push(ImagePhaseBytes {
            phase: ImagePhase::VisionTower,
            device_bytes: add_all([facts.vision_bytes, work, pixels, grounded])?,
            host_bytes: reference_pixels_host,
        });
    }

    // --- 2. conditioning --------------------------------------------------------------------------
    let te_weights = facts.te.peak_bytes(streamed.then_some(1));
    let d = facts.te_dims;
    let te_work = block_working_bytes(
        rows.text_rows,
        d.hidden,
        d.heads,
        d.kv_heads,
        d.head_dim,
        d.intermediate,
        f32_bytes,
        budget,
    )?;
    phases.push(ImagePhaseBytes {
        phase: ImagePhase::Conditioning,
        device_bytes: add_all([te_weights, te_work, facts.te_decode, grounded])?,
        // Both routes prepare the host-packed encoder; the streamed one keeps it there.
        host_bytes: add_all([facts.te_prepare_host, reference_pixels_host])?,
    });

    // The layer-50 context the later phases carry, f32.
    let context = mul(mul(rows.text_rows, facts.dit_text_dim)?, f32_bytes)?;

    // --- 3. reference encode (edit) ----------------------------------------------------------------
    let reference_latents = mul(
        rows.reference_rows.iter().sum::<u64>(),
        facts.dit_patch_dim * f32_bytes * 2,
    )?;
    if rows.has_references() {
        let largest = rows.reference_pixels.iter().copied().max().unwrap_or(0);
        let tile = mul(
            TILE_WITH_OVERLAP * TILE_WITH_OVERLAP * ENCODE_TILE_CHANNELS * f32_bytes,
            ENCODE_TILE_BUFFERS,
        )?;
        // The f32 pixels in, and the posterior (mean + log-variance) out at full resolution.
        let image = mul(largest, 3 * f32_bytes)?;
        let posterior = mul(largest / 256, 48 * f32_bytes * 2)?;
        phases.push(ImagePhaseBytes {
            phase: ImagePhase::ReferenceEncode,
            device_bytes: add_all([
                facts.encoder_bytes,
                tile,
                image,
                posterior,
                context,
                reference_latents,
            ])?,
            host_bytes: reference_pixels_host,
        });
    }

    // --- 4. denoise ----------------------------------------------------------------------------------
    let backend = crate::memory_strategy::native_backend();
    let image_and_refs = rows.image_rows + rows.reference_rows.iter().sum::<u64>();
    let native_rows = NativeRows {
        task: NativeTaskSelection::Base,
        packed: rows.sequence_rows(),
        conditioning: rows.text_rows,
        // Six fixed timesteps, plus the references' constant 0.999 row class on an edit.
        schedule: u64::from(contract::FIXED_STEPS) + u64::from(rows.has_references()),
        visual_elements: mul(image_and_refs, 4)?,
        global: rows.text_rows,
    };
    let dit = if streamed {
        facts
            .window
            .precompute_peak_bytes(&facts.task, backend, native_rows, 1)?
            .max(facts.window.bounded_cuda_denoise_peak_bytes(
                &facts.task,
                backend,
                native_rows,
                1,
            )?)
    } else {
        facts
            .window
            .precompute_peak_bytes_full(&facts.task, backend, native_rows)?
            .max(
                facts
                    .window
                    .denoise_peak_bytes_full(&facts.task, backend, native_rows)?,
            )
            // The resident load's source map and its derived device weights coexist once.
            .max(facts.task.constructor_bytes)
    };
    // Sampler state: the packed image/reference rows, their velocity, the noise and the update
    // (f32 at the patch width), plus the carried context.
    let sampler = add_all([
        mul(image_and_refs, facts.dit_patch_dim * f32_bytes * 4)?,
        context,
        mul(mul(rows.sequence_rows(), facts.dit_hidden)?, 2)?,
    ])?;
    phases.push(ImagePhaseBytes {
        phase: ImagePhase::Denoise,
        device_bytes: add_all([dit, facts.adapter_resident, sampler])?,
        host_bytes: add_all([
            if streamed {
                facts.dit_prepare_host.max(facts.dit_retained_host)
            } else {
                facts.dit_prepare_host
            },
            facts.adapter_install.max(facts.adapter_resident),
        ])?,
    });

    // --- 5. decode -------------------------------------------------------------------------------------
    let canvas_pixels = u64::from(rows.canvas.0) * u64::from(rows.canvas.1);
    let output_pixels = u64::from(rows.output.0) * u64::from(rows.output.1);
    // The blended f32 RGB canvas plus its blend weights, the f32 latent, and one tile's decoder
    // tokens and pixels.
    let tile_tokens = (TILE_WITH_OVERLAP / 16) * (TILE_WITH_OVERLAP / 16) + 5;
    let decode_tile = add_all([
        mul(
            tile_tokens,
            DECODE_HIDDEN * f32_bytes * DECODE_TOKEN_BUFFERS,
        )?,
        mul(TILE_WITH_OVERLAP * TILE_WITH_OVERLAP, 3 * f32_bytes * 4)?,
    ])?;
    phases.push(ImagePhaseBytes {
        phase: ImagePhase::Decode,
        device_bytes: add_all([
            facts.decoder_bytes,
            mul(canvas_pixels, 3 * f32_bytes * 2)?,
            mul(rows.image_rows, facts.dit_patch_dim * f32_bytes * 2)?,
            decode_tile,
        ])?,
        // The f32 canvas read back, and the resized 8-bit output.
        host_bytes: add_all([mul(canvas_pixels, 3 * f32_bytes)?, mul(output_pixels, 3)?])?,
    });

    Ok(ImageMemoryEstimate {
        selection: *selection,
        streamed,
        phases,
    })
}

/// `(hidden, heads, head_dim, mlp)` of the Qwen3-VL tower H3 ships
/// ([`crate::text_encoder::minimax_h3_vision_config`]).
fn minimax_vision_dims() -> (u64, u64, u64, u64) {
    let c = crate::text_encoder::minimax_h3_vision_config();
    let hidden = c.hidden_size as u64;
    let heads = c.num_heads as u64;
    // Qwen3-VL's ViT MLP widens 1152 to 4304.
    (hidden, heads, hidden / heads.max(1), 4304)
}

/// The ladder candidates this contract declares, cheapest-to-run first: rung 0, rung 1, rung 2 at
/// the shipped tile, rung 3 at its smallest chunk, and rung 4 (window 1 over both transformers)
/// when the load can stream — the MLX twin's `ladder`, plus the rung only CUDA has.
pub fn ladder(contract: &MemoryProviderContract, tier: MemoryNumericTier) -> Vec<MemorySelection> {
    let decode = MemoryStrategyParameters {
        decode_tile_edge: Some(IMAGE_DECODE_TILE_EDGE),
        decode_overlap: Some(IMAGE_DECODE_OVERLAP),
        ..Default::default()
    };
    let attention = MemoryStrategyParameters {
        attention_chunk_size: Some(super::IMAGE_ATTENTION_CHUNK_SIZES[0]),
        ..decode
    };
    [
        (
            MemoryStrategy::Resident,
            MemoryStrategyParameters::default(),
        ),
        (
            MemoryStrategy::StagedResidency,
            MemoryStrategyParameters::default(),
        ),
        (MemoryStrategy::BoundedDecode, decode),
        (MemoryStrategy::BoundedAttention, attention),
        (
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyParameters {
                transformer_window_size: Some(super::IMAGE_TRANSFORMER_WINDOW_SIZE),
                transformer_window_component: Some(TransformerComponent::Both),
                ..attention
            },
        ),
    ]
    .into_iter()
    .map(|(strategy, parameters)| MemorySelection {
        strategy,
        parameters,
        tier,
    })
    .filter(|selection| contract.validate_selection(selection).is_ok())
    .collect()
}

/// The ladder walk's result: the selected rung, its estimate, and every rung rejected first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageAdmission {
    pub estimate: ImageMemoryEstimate,
    pub rejected: Vec<ImageMemoryEstimate>,
}

/// Walk [`ladder`] and return the first selection whose device peak fits `budget` — the device's
/// **actually-free** memory as the caller measured it — with the estimates of the rungs rejected
/// before it. `Err` names every rung's estimate when nothing fits.
pub fn admit(
    contract: &MemoryProviderContract,
    facts: &ImageMemoryFacts,
    shape: &ImageRequestShape,
    tier: MemoryNumericTier,
    budget: MemoryBudget,
) -> gen_core::Result<ImageAdmission> {
    let mut rejected = Vec::new();
    for selection in ladder(contract, tier) {
        let estimate = estimate(contract, facts, shape, &selection)?;
        if budget.fits(estimate.peak_device_bytes()) {
            return Ok(ImageAdmission { estimate, rejected });
        }
        rejected.push(estimate);
    }
    let effective = budget.effective_bytes();
    Err(gen_core::Error::Unsupported(format!(
        "{IMAGE_MODEL_ID}: {}x{} does not fit at any memory rung against {:.2} GB ({:.2} GiB) of \
         free GPU memory: {}",
        shape.width,
        shape.height,
        decimal_gb(effective),
        binary_gib(effective),
        rejected
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" | ")
    )))
}

/// A shipped-scale synthetic inventory (no files): the published bf16 component sizes split the
/// way the real headers split them — the 50-layer TE tap, the 66.28 GB DiT as root + 50 bodies +
/// 50 `adaln_proj` stacks (26.02 GB), the f32 vision tower and VAE halves. Arithmetic fixtures for
/// the estimate and admission tests; the real headers are read by [`ImageMemoryFacts::resolve`].
#[cfg(test)]
pub(crate) fn shipped_scale_facts(adapter_resident: u64) -> ImageMemoryFacts {
    use candle_gen::gen_core::h3_memory::window::{CudaAdalnWorkingSet, CudaWorkingSet};
    let cfg = crate::dit::MiniMaxH3DitConfig::image();
    let layers = cfg.num_layers;
    let adaln_per_block = crate::memory_strategy::ADALN_EVICTED_BYTES / layers as u64;
    let root = 1_500_000_000u64;
    let body = (crate::memory_strategy::DIT_BF16_BYTES
        - crate::memory_strategy::ADALN_EVICTED_BYTES
        - root)
        / layers as u64;
    let mut resident = vec![("proj_in.weight".to_owned(), root)];
    for i in 0..layers {
        resident.push((format!("transformer_blocks.{i}.attn.to_q.weight"), body));
        resident.push((
            format!("transformer_blocks.{i}.adaln_proj.linear.weight"),
            adaln_per_block,
        ));
    }
    let total: u64 = resident.iter().map(|(_, b)| b).sum();
    let window = WindowAllocation::from_resident_tensors(&resident, &[], 0, false)
        .unwrap()
        .with_cuda_working_set(CudaWorkingSet {
            hidden: cfg.hidden_size as u64,
            heads: cfg.num_attention_heads as u64,
            head_dim: cfg.attention_head_dim as u64,
            rotary_dim: cfg.rotary_dim() as u64,
            ffn_inner: cfg.ffn_dim as u64,
            element_bytes: 2,
            projection_decode_bytes: 0,
        })
        .unwrap()
        .with_cuda_adaln_working_sets(vec![
            CudaAdalnWorkingSet {
                input: cfg.time_embed_dim as u64,
                output: cfg.adaln_out_features() as u64,
                affine: false,
                bias_rows: 1,
            };
            layers
        ])
        .unwrap();
    let task = NativeTaskAllocation {
        adapter: None,
        resident_bytes: total,
        constructor_bytes: total,
        patch_size: [1, 2, 2],
        reference_task: false,
        window: Some(window.clone()),
        evicted_bytes: crate::memory_strategy::ADALN_EVICTED_BYTES,
        table_bytes_per_timestep: layers as u64 * cfg.adaln_out_features() as u64 * 2,
        projections: Vec::new(),
    };
    // 151936 x 5120 bf16 embedding plus f32 rotary tables; 50 layers of the 58 GB tap.
    let te_root = 151_936 * 5120 * 2 + 2 * 16_384 * 128 * 4;
    ImageMemoryFacts {
        te: StreamedWeightFacts {
            resident_bytes: te_root,
            stacks: vec![vec![(58_000_000_000 - te_root) / 50; 50]],
        },
        te_dims: TeDims {
            hidden: 5120,
            heads: 64,
            kv_heads: 8,
            head_dim: 128,
            intermediate: 25_600,
        },
        te_decode: 25_600 * 5120 * 4,
        te_prepare_host: 58_000_000_000,
        vision_bytes: 2_300_000_000,
        dit_hidden: cfg.hidden_size as u64,
        dit_text_dim: cfg.text_dim as u64,
        dit_patch_dim: cfg.video_patch_dim() as u64,
        task,
        window,
        dit_prepare_host: total,
        dit_retained_host: total,
        adapter_resident,
        adapter_install: adapter_resident,
        decoder_bytes: 1_400_000_000,
        encoder_bytes: 700_000_000,
        stream: transformer_stream(&resident, layers).unwrap(),
        streamable: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{Conditioning, LoadShape};

    const GB: u64 = BYTES_PER_GB;

    fn contract(shape: LoadShape) -> MemoryProviderContract {
        super::super::weights_free_contract(
            &LoadSpec::new(WeightsSource::Dir("/minimax-h3-image".into())).with_load_shape(shape),
        )
        .unwrap()
    }

    fn tier() -> MemoryNumericTier {
        MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        }
    }

    fn selection(c: &MemoryProviderContract, strategy: MemoryStrategy) -> MemorySelection {
        ladder(c, tier())
            .into_iter()
            .find(|s| s.strategy == strategy)
            .unwrap_or_else(|| panic!("{strategy:?} is not on the ladder"))
    }

    fn budget(bytes: u64) -> MemoryBudget {
        MemoryBudget {
            total_bytes: bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        }
    }

    fn t2i_16mp() -> ImageRequestShape {
        ImageRequestShape::text_to_image(4096, 4096)
    }

    /// E7: 16 MP is 16,384 image tokens at 32 px/token, plus the 4096-token prompt bound; 16
    /// references extend the sequence by their own rows.
    #[test]
    fn rows_follow_the_32_px_token_and_the_reference_copies() {
        let rows = ImageRequestRows::of(&t2i_16mp());
        assert_eq!(rows.canvas, (4096, 4096));
        assert_eq!(rows.image_rows, 16_384);
        assert_eq!(rows.text_rows, 4096);
        assert_eq!(rows.sequence_rows(), 20_480);
        // A small request generates on the 4 MP floor canvas and is resized back.
        assert_eq!(
            ImageRequestRows::of(&ImageRequestShape::text_to_image(1000, 1001)).image_rows,
            4096
        );
        // A short prompt prices its own length.
        let mut short = t2i_16mp();
        short.prompt_tokens = Some(77);
        assert_eq!(ImageRequestRows::of(&short).text_rows, 77);
        // Sixteen 2048² references: 4096 rows each, plus 1024 merged vision tokens and a label.
        let edit = ImageRequestRows::of(&ImageRequestShape::with_reference_count(4096, 4096, 16));
        assert_eq!(edit.reference_rows, vec![4096; 16]);
        assert_eq!(edit.text_rows, 4096 + 16 * (1024 + LABEL_TOKENS_BOUND));
        assert_eq!(edit.sequence_rows(), edit.text_rows + 16 * 4096 + 16_384);
        // An exact request prices its own reference sizes.
        let mut req = GenerationRequest {
            width: 2048,
            height: 2048,
            ..Default::default()
        };
        req.conditioning.push(Conditioning::Reference {
            image: gen_core::Image {
                width: 1024,
                height: 1024,
                pixels: vec![0; 1024 * 1024 * 3],
            },
            strength: None,
        });
        let exact = ImageRequestRows::of(&ImageRequestShape::from_request(&req).unwrap());
        assert_eq!(exact.reference_rows, vec![1024]);
        assert_eq!(exact.vision_patches, vec![4096]);
    }

    /// AC2's arithmetic: at 16 MP the staged bf16 render holds the whole 66 GB DiT on the device
    /// while the streamed one holds a root, one block and the working set — so with the 40 GB a
    /// shared 96 GB card has free, `admit` lands on rung 4; with an empty card it stops at rung 0.
    #[test]
    fn the_16_mp_streamed_route_fits_where_the_resident_one_does_not() {
        let c = contract(LoadShape::DeferredMaterialization);
        let facts = shipped_scale_facts(0);
        let shape = t2i_16mp();
        let staged = estimate(
            &c,
            &facts,
            &shape,
            &selection(&c, MemoryStrategy::StagedResidency),
        )
        .unwrap();
        let streamed = estimate(
            &c,
            &facts,
            &shape,
            &selection(&c, MemoryStrategy::BoundedTransformerResidency),
        )
        .unwrap();
        assert!(staged.peak_device_bytes() > 66 * GB, "{staged}");
        assert_eq!(staged.binding_phase(), Some(ImagePhase::Denoise));
        assert!(streamed.streamed && !staged.streamed);
        assert!(streamed.peak_device_bytes() < 20 * GB, "{streamed}");
        // Streaming moves the stacks to the host; it does not make them disappear.
        assert!(streamed.peak_host_bytes() >= facts.dit_retained_host);

        let admitted = admit(&c, &facts, &shape, tier(), budget(40 * GB)).unwrap();
        assert_eq!(
            admitted.estimate.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency
        );
        assert_eq!(
            admitted
                .rejected
                .iter()
                .map(|e| e.selection.strategy)
                .collect::<Vec<_>>(),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
            ]
        );
        let roomy = admit(&c, &facts, &shape, tier(), budget(200 * GB)).unwrap();
        assert_eq!(roomy.estimate.selection.strategy, MemoryStrategy::Resident);
        // Nothing fits a tiny card, and the refusal names every rung's estimate in both units.
        let e = admit(&c, &facts, &shape, tier(), budget(GB))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("BoundedTransformerResidency") && e.contains("GiB"),
            "{e}"
        );
        // An eager load does not offer rung 4, so the 16 MP request has nowhere to go.
        let eager = contract(LoadShape::EagerMaterialization);
        assert!(admit(&eager, &facts, &shape, tier(), budget(40 * GB)).is_err());
        // A source that cannot stream refuses the streamed estimate itself.
        let mut fixed = facts.clone();
        fixed.streamable = false;
        assert!(estimate(
            &c,
            &fixed,
            &shape,
            &selection(&c, MemoryStrategy::BoundedTransformerResidency)
        )
        .is_err());
    }

    /// The contract's stream facts: the trunk plus one entry per block, summing to the DiT.
    #[test]
    fn the_transformer_stream_is_trunk_plus_fifty_blocks() {
        let facts = shipped_scale_facts(0);
        let stream = facts.transformer_stream();
        assert_eq!(stream.stacks.len(), 1);
        assert_eq!(stream.stacks[0].len(), 50);
        assert_eq!(stream.peak_bytes(None), facts.task.resident_bytes);
        assert!(stream.peak_bytes(Some(IMAGE_TRANSFORMER_WINDOW)) < 4 * GB);
    }

    const IMAGE_TRANSFORMER_WINDOW: u32 = super::super::IMAGE_TRANSFORMER_WINDOW_SIZE;

    /// Successive lifetimes take a maximum: the peak is one phase's bytes, not their sum, and host
    /// and GPU are separate columns.
    #[test]
    fn phases_are_successive_lifetimes_priced_on_two_columns() {
        let c = contract(LoadShape::DeferredMaterialization);
        let facts = shipped_scale_facts(0);
        let plan = estimate(
            &c,
            &facts,
            &ImageRequestShape::with_reference_count(4096, 4096, 3),
            &selection(&c, MemoryStrategy::BoundedTransformerResidency),
        )
        .unwrap();
        let kinds: Vec<_> = plan.phases.iter().map(|p| p.phase).collect();
        assert_eq!(
            kinds,
            vec![
                ImagePhase::VisionTower,
                ImagePhase::Conditioning,
                ImagePhase::ReferenceEncode,
                ImagePhase::Denoise,
                ImagePhase::Decode,
            ]
        );
        assert_eq!(
            ImagePhase::VisionTower.memory_phase(),
            MemoryPhase::Conditioning
        );
        let sum: u64 = plan.phases.iter().map(|p| p.device_bytes).sum();
        assert_eq!(
            plan.peak_device_bytes(),
            plan.phases.iter().map(|p| p.device_bytes).max().unwrap()
        );
        assert!(plan.peak_device_bytes() < sum);
        assert!(plan.phases[0].device_bytes > facts.vision_bytes);
        // The streamed conditioner keeps its prepared stack on the host.
        assert!(plan.phases[1].host_bytes >= facts.te_prepare_host);
        // A text-to-image request has no vision or reference-encode phase.
        let t2i = estimate(
            &c,
            &facts,
            &t2i_16mp(),
            &selection(&c, MemoryStrategy::StagedResidency),
        )
        .unwrap();
        assert_eq!(t2i.phases.len(), 3);
    }

    /// SW-7096's charge at every rung: the installed factors in full on the device during
    /// denoise (on the streamed route each window carries its block's factors on top of them),
    /// and their install high-water on the host.
    #[test]
    fn lora_factors_are_charged_in_full_on_both_routes() {
        let c = contract(LoadShape::DeferredMaterialization);
        let lora = 3 * GB;
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let at = |facts: &ImageMemoryFacts| {
                estimate(&c, facts, &t2i_16mp(), &selection(&c, strategy))
                    .unwrap()
                    .phases
                    .into_iter()
                    .find(|e| e.phase == ImagePhase::Denoise)
                    .unwrap()
            };
            let bare = at(&shipped_scale_facts(0));
            let with = at(&shipped_scale_facts(lora));
            assert_eq!(with.device_bytes - bare.device_bytes, lora, "{strategy:?}");
            assert!(with.host_bytes >= bare.host_bytes + lora);
        }
    }

    /// Rung 3's chunk bounds the materializing fallback the TE and the vision tower run; a chunk
    /// above the CUDA default never widens it, and an unengaged rung uses the default.
    #[test]
    fn the_selected_chunk_bounds_the_materializing_fallback() {
        let default = gen_core::veda::workspace::CUDA_DENSE_SCORE_ELEMENTS;
        assert_eq!(score_budget(true, Some(8 << 20)), 8 << 20);
        assert_eq!(score_budget(true, Some(u32::MAX)), default);
        assert_eq!(score_budget(false, Some(8 << 20)), default);
        let c = contract(LoadShape::DeferredMaterialization);
        let facts = shipped_scale_facts(0);
        let shape = ImageRequestShape::with_reference_count(4096, 4096, 16);
        let wide = estimate(
            &c,
            &facts,
            &shape,
            &selection(&c, MemoryStrategy::BoundedDecode),
        )
        .unwrap();
        let narrow = estimate(
            &c,
            &facts,
            &shape,
            &selection(&c, MemoryStrategy::BoundedAttention),
        )
        .unwrap();
        assert!(narrow.phases[1].device_bytes < wide.phases[1].device_bytes);
    }

    /// Decimal GB and binary GiB are distinct, named conversions.
    #[test]
    fn unit_conversions_are_explicit() {
        assert_eq!(decimal_gb(BYTES_PER_GB), 1.0);
        assert_eq!(binary_gib(BYTES_PER_GIB), 1.0);
        assert!((decimal_gb(BYTES_PER_GIB) - 1.073_741_824).abs() < 1e-12);
        let c = contract(LoadShape::DeferredMaterialization);
        let text = estimate(
            &c,
            &shipped_scale_facts(0),
            &t2i_16mp(),
            &selection(&c, MemoryStrategy::StagedResidency),
        )
        .unwrap()
        .to_string();
        assert!(text.contains(" GB (") && text.contains(" GiB)"), "{text}");
    }
}
