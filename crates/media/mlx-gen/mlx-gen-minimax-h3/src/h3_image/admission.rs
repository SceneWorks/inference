//! MiniMax-H3-Image request admission on MLX: per-phase host and GPU bytes for one request at one
//! memory selection, and the ladder walk that picks the first selection that fits (SW-7094).
//!
//! # Every number here is DERIVED, none is measured
//!
//! The weight terms are exact tensor bytes read from the resolved files' safetensors headers
//! ([`ImageMemoryFacts::resolve`]) — or, weights-free, from the shipped architecture
//! ([`ImageMemoryFacts::derived`]). The activation terms are element counts of the tensors each
//! forward holds at once, multiplied by the compute width. They are upper bounds from the source,
//! not allocator high-water marks: the family's two memory anchors per tier are measured once, at
//! the end of the feature (E13), and nothing in this module claims to be one.
//!
//! # Host and GPU are separate columns, and on MLX they share one pool
//!
//! [`ImagePhaseBytes::device_bytes`] is what MLX allocates (weights, activations, the stitched
//! canvas). [`ImagePhaseBytes::host_bytes`] is what the generator allocates on the CPU heap: the
//! resized reference copies, the 8-bit decoded canvas, the Lanczos resize buffers and the
//! returned images. Apple silicon has unified memory, so admission charges their **sum** per phase
//! ([`ImagePhaseBytes::total_bytes`]) and the request peak is the largest phase, never the sum of
//! phases: the generator releases each phase's component before the next maps (rung 1).
//!
//! # The ladder
//!
//! [`admit`] walks rungs 0 to 4 in order and returns the first selection whose peak fits the
//! budget (rung 4 only on a load that can stream). Rung 3 lowers the denoise phase only: it
//! chunks each block's attention on whole heads ([`crate::h3_image::memory`]), so the head-major
//! attention operands are held for one chunk's heads instead of all of them. Rung 0's estimate
//! charges every component co-resident — the "resident peak" a
//! request is rejected at when nothing streams — although the generator always stages; that is
//! the over-provisioning direction.

use mlx_gen::gen_core::h3_adapter_memory::H3AdapterMemoryProfile;
use mlx_gen::gen_core::h3_image::{
    generation_canvas, latent_grid, reference_vae_size, reference_vision_size, FIXED_STEPS,
    MAX_PROMPT_TOKENS, MIN_GENERATION_AREA_DEFAULT, VAE_SPATIAL_RATIO, VISION_MERGE_TOKENS,
};
use mlx_gen::gen_core::h3_memory::NativeBackend;
use mlx_gen::gen_core::weightsmeta::{self, SafetensorsTensorHeader};
use mlx_gen::gen_core::{
    Error, LoadSpec, MemoryBudget, MemoryNumericTier, MemoryPhase, MemoryProviderContract,
    MemorySelection, MemoryStrategy, MemoryStrategyParameters, Result, TransformerComponent,
    WeightsSource,
};

use crate::dit::config::{MiniMaxH3ImageDitConfig, MODALITY_NUM, MODULATION_PARAMS};
use crate::h3_image::dit::image_adapter_target_paths;
use crate::image_vae::{IMAGE_DECODE_OVERLAP, IMAGE_DECODE_TILE_EDGE, IMAGE_VAE_COMPONENT};
use crate::model::TEXT_ENCODER_COMPONENT;
use crate::text_encoder::{MiniMaxH3TeConfig, LM_PREFIX, TE_SHARDS, VISION_PREFIX, VISION_SHARD};

/// Tokens a `Picture i: ` label and its vision-block delimiters add to an edit presentation, per
/// reference. An upper bound on `MiniMaxH3Tokenizer::image_edit_ids`' framing (`Picture`, the
/// index, `:`, the space, `<|vision_start|>`, `<|vision_end|>` and a separator).
const EDIT_LABEL_TOKENS_PER_REFERENCE: u64 = 8;

/// The weight bytes and architecture one image request is priced from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryFacts {
    /// Text encoder: what stays resident under a layer window (`embed_tokens`).
    pub te_resident_bytes: u64,
    /// Text encoder: the tapped decoder layers (`select_hidden` of them), in order.
    pub te_layer_bytes: Vec<u64>,
    /// The Qwen3-VL vision tower — mapped only by `edit`, alongside the text encoder.
    pub vision_bytes: u64,
    /// DiT: the I/O projections, the six per-step heads and the token refiner — resident at every
    /// rung.
    pub dit_trunk_bytes: u64,
    /// DiT: each block's body (attention, FFN, norms), in order.
    pub dit_body_bytes: Vec<u64>,
    /// DiT: each block's `adaln_proj` — read once per request by the precompute, then evicted
    /// (resident) or never held (windowed).
    pub dit_adaln_bytes: Vec<u64>,
    /// VAE decode half (`decoder.*`, `post_quant_conv.*`, latent statistics).
    pub decoder_bytes: u64,
    /// VAE encode half (`encoder.*`, `quant_conv.*`) — loaded by `edit` only.
    pub encoder_bytes: u64,
    /// The configured LoRA / LoKr stack priced by H3's own installer accounting
    /// (`gen_core_h3::h3_adapter_memory`): resident at every tier and rung, in full.
    pub adapters: H3AdapterMemoryProfile,
    /// Per DiT block, the LoRA residual working bytes per packed row and the fixed casts — one
    /// block's worth is live at a time.
    pub adapter_block_rows: Vec<(u64, u64)>,
    /// The DiT's compute width (2 for bf16, 4 for an Fp32 load).
    pub activation_width: u64,
}

/// What one request asks for, as admission sees it — the axes the per-phase bytes depend on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageRequestShape {
    pub width: u32,
    pub height: u32,
    /// Images rendered sequentially over one encoded prompt and one loaded DiT.
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
}

/// One phase's bytes, in the two pools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImagePhaseBytes {
    pub phase: MemoryPhase,
    /// MLX (Metal) allocations: weights, activations, the stitched canvas.
    pub device_bytes: u64,
    /// CPU-heap allocations: resized references, decoded 8-bit canvases, resize buffers, outputs.
    pub host_bytes: u64,
}

impl ImagePhaseBytes {
    /// The phase's demand on unified memory.
    pub fn total_bytes(&self) -> u64 {
        self.device_bytes.saturating_add(self.host_bytes)
    }
}

/// The per-phase estimate of one request at one selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryEstimate {
    pub selection: MemorySelection,
    /// Conditioning, denoise, decode — in execution order.
    pub phases: [ImagePhaseBytes; 3],
}

impl ImageMemoryEstimate {
    /// The request peak: the largest phase (phases never overlap — rung 1 is structural).
    pub fn peak_bytes(&self) -> u64 {
        self.phases
            .iter()
            .map(ImagePhaseBytes::total_bytes)
            .max()
            .unwrap_or(0)
    }

    /// The largest device column.
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

    /// The phase that sets [`Self::peak_bytes`].
    pub fn binding_phase(&self) -> MemoryPhase {
        self.phases
            .iter()
            .max_by_key(|p| p.total_bytes())
            .map_or(MemoryPhase::Denoise, |p| p.phase)
    }
}

/// The ladder walk's result: the selected rung, its estimate, and every rung it rejected first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageAdmission {
    pub estimate: ImageMemoryEstimate,
    pub rejected: Vec<ImageMemoryEstimate>,
}

fn sum(values: impl IntoIterator<Item = u64>) -> u64 {
    values.into_iter().fold(0u64, u64::saturating_add)
}

fn headers(path: &std::path::Path) -> Result<Vec<SafetensorsTensorHeader>> {
    if weightsmeta::safetensors_path_bytes(path) == 0 {
        return Ok(Vec::new());
    }
    weightsmeta::safetensors_path_tensor_headers(path)
}

/// Bytes a header occupies once the loader has built it: stored width, except a dense float
/// that follows an Fp32 load's cast. A cast tensor is priced at the wider of its stored and its
/// cast width: MLX casts lazily over the evaluated source, so the stored-width source is never
/// less than what the load holds.
fn loaded_bytes(header: &SafetensorsTensorHeader, float_width: Option<u64>) -> Result<u64> {
    match float_width {
        Some(width) if header.is_float() => {
            Ok(header.materialized_bytes(width)?.max(header.data_bytes))
        }
        _ => Ok(header.data_bytes),
    }
}

/// `transformer_blocks.{i}.` → `i`.
fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("transformer_blocks.")?
        .split_once('.')?
        .0
        .parse()
        .ok()
}

impl ImageMemoryFacts {
    /// Read every weight term off the resolved files' headers (no tensor payload is read). Absent
    /// components contribute zero, as the contract's asset facts do.
    pub fn resolve(spec: &LoadSpec) -> Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root.clone(),
            WeightsSource::File(path) => path.parent().unwrap_or(path).to_path_buf(),
        };
        let fp32 = spec.precision == mlx_gen::Precision::Fp32;
        let float_width = fp32.then_some(4);
        let cfg = MiniMaxH3ImageDitConfig::default();

        // --- DiT ---------------------------------------------------------------------------
        let dit_dir = crate::h3_image::tier::resolve_dit_dir(&root, spec);
        let dit_headers = headers(&dit_dir)?;
        let blocks = cfg.base.num_layers.max(0) as usize;
        let mut body = vec![0u64; blocks];
        let mut adaln = vec![0u64; blocks];
        let mut trunk = 0u64;
        for h in &dit_headers {
            match block_index(&h.name) {
                Some(i) if i < blocks => {
                    // The block stack and the refiner follow `spec.precision`; the I/O heads keep
                    // their published widths (four ship float32).
                    let bytes = loaded_bytes(h, float_width)?;
                    if h.name.contains(".adaln_proj.") {
                        adaln[i] = adaln[i].saturating_add(bytes);
                    } else {
                        body[i] = body[i].saturating_add(bytes);
                    }
                }
                _ if h.name.starts_with("token_refiner.") => {
                    trunk = trunk.saturating_add(loaded_bytes(h, float_width)?)
                }
                _ => trunk = trunk.saturating_add(h.data_bytes),
            }
        }

        // --- text encoder ------------------------------------------------------------------
        let te_dir = match spec.components.get(TEXT_ENCODER_COMPONENT) {
            Some(WeightsSource::Dir(dir)) => dir.clone(),
            _ => root.join(TEXT_ENCODER_COMPONENT),
        };
        let te_cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        let tapped = te_cfg
            .out_layer()
            .map_err(|e| Error::Unsupported(e.to_string()))?
            + 1;
        let mut te_layers = vec![0u64; tapped];
        let mut te_resident = 0u64;
        let mut vision = 0u64;
        let layers = format!("{LM_PREFIX}.layers.");
        let embed = format!("{LM_PREFIX}.embed_tokens.");
        let visual = format!("{VISION_PREFIX}.");
        let shards = TE_SHARDS.chain(std::iter::once(VISION_SHARD));
        for shard in shards {
            let path = te_dir.join(format!("model-{shard:05}-of-00014.safetensors"));
            for h in headers(&path)? {
                if let Some(rest) = h.name.strip_prefix(&layers) {
                    let index = rest
                        .split_once('.')
                        .and_then(|(i, _)| i.parse::<usize>().ok());
                    if let Some(slot) = index.and_then(|i| te_layers.get_mut(i)) {
                        *slot = slot.saturating_add(h.data_bytes);
                    }
                } else if h.name.starts_with(&embed) {
                    te_resident = te_resident.saturating_add(h.data_bytes);
                } else if h.name.starts_with(&visual) {
                    vision = vision.saturating_add(h.data_bytes);
                }
            }
        }

        // --- VAE ---------------------------------------------------------------------------
        let mut decoder = 0u64;
        let mut encoder = 0u64;
        for h in headers(&root.join(IMAGE_VAE_COMPONENT))? {
            if h.name.starts_with("encoder.") || h.name.starts_with("quant_conv.") {
                encoder = encoder.saturating_add(h.data_bytes);
            } else {
                // `MiniMaxH3ImageVae::load` casts the decode half to `spec.precision`'s dtype; the
                // encode half (`MiniMaxH3ImageVaeEncoder::load`) keeps its stored widths.
                decoder = decoder.saturating_add(loaded_bytes(&h, float_width)?);
            }
        }

        // --- adapters ----------------------------------------------------------------------
        let activation_width: u64 = if fp32 { 4 } else { 2 };
        let (adapters, adapter_block_rows) = if spec.adapters.is_empty() || dit_headers.is_empty() {
            (H3AdapterMemoryProfile::default(), Vec::new())
        } else {
            let paths = image_adapter_target_paths(&cfg);
            let targets = gen_core_h3::h3_adapter_memory::targets_from_headers(
                &dit_headers,
                &[],
                &paths,
                activation_width,
            )?;
            let profile =
                gen_core_h3::h3_adapter_memory::profile_stack(backend(), &spec.adapters, &targets)?;
            let rows = adapter_rows_per_block(&profile, blocks);
            (profile, rows)
        };

        Ok(Self {
            te_resident_bytes: te_resident,
            te_layer_bytes: te_layers,
            vision_bytes: vision,
            dit_trunk_bytes: trunk,
            dit_body_bytes: body,
            dit_adaln_bytes: adaln,
            decoder_bytes: decoder,
            encoder_bytes: encoder,
            adapters,
            adapter_block_rows,
            activation_width,
        })
    }

    /// The weights-free twin of [`Self::resolve`] at one artifact tier (`None` = bf16, `Some(8)`,
    /// `Some(4)`): every linear priced from the shipped architecture at that tier's packed width
    /// (`codes + bf16 scales and biases per 64-wide group`), heads and norms dense. DERIVED, not
    /// read and not measured. The text encoder is H3's tier artifact at the same width.
    pub fn derived(bits: Option<u32>) -> Self {
        let cfg = MiniMaxH3ImageDitConfig::default();
        let b = &cfg.base;
        let (hidden, inner, ffn) = (b.hidden_size as u64, b.inner_dim() as u64, b.ffn_dim as u64);
        let linear = |out: u64, inp: u64| -> u64 {
            match bits {
                None => out * inp * 2,
                Some(bits) => out * inp * u64::from(bits) / 8 + out * (inp / 64) * 2 * 2,
            }
        };
        let attn_ffn = 3 * linear(inner, hidden)
            + linear(hidden, inner)
            + linear(2 * ffn, hidden)
            + linear(hidden, ffn);
        let norms = 2 * hidden * 2 + 2 * b.attention_head_dim as u64 * 2;
        let body = attn_ffn + norms;
        let adaln_out = b.adaln_out_features() as u64;
        let adaln = linear(adaln_out, b.time_embed_dim as u64) + adaln_out * 2;
        let patch = b.video_patch_dim() as u64;
        let heads = cfg.num_pdd_heads as u64 * (hidden * patch + patch) * 4;
        let trunk = (patch * hidden + hidden) * 4
            + (b.text_dim as u64 * hidden + hidden) * 2
            + (b.freq_dim as u64 * b.time_embed_hidden_dim as u64
                + b.time_embed_hidden_dim as u64 * b.time_embed_dim as u64)
                * 4
            + (b.time_embed_dim as u64 * 2 * hidden) * 2
            + heads
            + b.num_refiner_layers as u64 * body;

        let te = MiniMaxH3TeConfig::qwen3_vl_32b();
        let (h, q, kv, i) = (
            te.hidden_size as u64,
            (te.num_heads * te.head_dim) as u64,
            (te.num_kv_heads * te.head_dim) as u64,
            te.intermediate_size as u64,
        );
        let te_layer =
            linear(q, h) + 2 * linear(kv, h) + linear(h, q) + 3 * linear(i, h) + 2 * h * 2;
        let tapped = te.out_layer().unwrap_or(49) + 1;
        let embed = match bits {
            None => te.vocab_size as u64 * h * 2,
            Some(_) => linear(te.vocab_size as u64, h),
        };
        let vision = vision_tower_bytes();
        Self {
            te_resident_bytes: embed,
            te_layer_bytes: vec![te_layer; tapped],
            vision_bytes: vision,
            dit_trunk_bytes: trunk,
            dit_body_bytes: vec![body; b.num_layers as usize],
            dit_adaln_bytes: vec![adaln; b.num_layers as usize],
            decoder_bytes: decoder_weight_bytes(),
            encoder_bytes: encoder_weight_bytes(),
            adapters: H3AdapterMemoryProfile::default(),
            adapter_block_rows: Vec::new(),
            activation_width: 2,
        }
    }

    /// No weights at all — the registry's weights-free declaration surface.
    pub fn weights_free() -> Self {
        Self {
            te_resident_bytes: 0,
            te_layer_bytes: Vec::new(),
            vision_bytes: 0,
            dit_trunk_bytes: 0,
            dit_body_bytes: Vec::new(),
            dit_adaln_bytes: Vec::new(),
            decoder_bytes: 0,
            encoder_bytes: 0,
            adapters: H3AdapterMemoryProfile::default(),
            adapter_block_rows: Vec::new(),
            activation_width: 2,
        }
    }

    /// The whole DiT as loaded resident.
    pub fn dit_bytes(&self) -> u64 {
        sum([
            self.dit_trunk_bytes,
            sum(self.dit_body_bytes.iter().copied()),
            sum(self.dit_adaln_bytes.iter().copied()),
        ])
    }

    /// The tapped text encoder as loaded resident.
    pub fn te_bytes(&self) -> u64 {
        sum([
            self.te_resident_bytes,
            sum(self.te_layer_bytes.iter().copied()),
        ])
    }
}

fn backend() -> NativeBackend {
    NativeBackend::Mlx {
        tile_columns: mlx_gen::w4a8_int8::TILE_COLS as u64,
    }
}

/// Group the adapter profile's projections by DiT block: `(Σ per-row bytes, Σ fixed bytes)`.
/// Non-block targets (refiner, heads, embedders) are folded into every block's figure, so the
/// per-block maximum stays an upper bound on any one forward's live residuals.
fn adapter_rows_per_block(profile: &H3AdapterMemoryProfile, blocks: usize) -> Vec<(u64, u64)> {
    let mut rows = vec![(0u64, 0u64); blocks];
    let mut shared = (0u64, 0u64);
    for p in &profile.projections {
        let slot = match block_index(&p.target.path) {
            Some(i) if i < blocks => &mut rows[i],
            _ => &mut shared,
        };
        slot.0 = slot.0.saturating_add(p.forward_bytes_per_row);
        slot.1 = slot.1.saturating_add(p.forward_fixed_bytes);
    }
    rows.iter_mut().for_each(|r| {
        r.0 = r.0.saturating_add(shared.0);
        r.1 = r.1.saturating_add(shared.1);
    });
    rows
}

/// The Qwen3-VL vision tower's dense bf16 bytes from its shipped configuration (27 ViT blocks of
/// width 1152, the patch embed and the merger + three deepstack mergers into 5120).
fn vision_tower_bytes() -> u64 {
    let v = crate::text_encoder::minimax_h3_vision_config();
    let (d, m) = (v.hidden_size as u64, v.intermediate_size as u64);
    let block = 4 * d * d + 2 * d * m + 8 * d;
    let patch = 3 * v.temporal_patch_size as u64 * (v.patch_size as u64).pow(2) * d;
    let merge_in = d * (v.spatial_merge_size as u64).pow(2);
    let merger = merge_in * merge_in + merge_in * v.out_hidden_size as u64;
    let mergers = (1 + v.deepstack_visual_indexes.len() as u64) * merger;
    (v.depth as u64 * block + patch + v.num_position_embeddings as u64 * d + mergers) * 2
}

/// The 2-D ViT decoder's float32 bytes from its shipped configuration (24 layers of width 1024).
fn decoder_weight_bytes() -> u64 {
    let c = crate::config::MiniMaxH3VaeConfig::image();
    let d = c.dim() as u64;
    let f = d * c.ffn_mult as u64;
    (c.num_layers as u64 * (4 * d * d + 3 * d * f) + d * c.patch_dim() as u64) * 4
}

/// The 3-D CNN encoder's float32 bytes, bounded by its widest block (`layers_per_block` 3x3x3
/// convolutions per level at that width).
fn encoder_weight_bytes() -> u64 {
    let c = crate::config::MiniMaxH3VaeConfig::image();
    let widest = c.block_out_channels.iter().copied().max().unwrap_or(0) as u64;
    let levels = c.block_out_channels.len() as u64;
    levels * (c.layers_per_block as u64 + 1) * 2 * widest * widest * 27 * 4
}

/// The token counts of one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rows {
    /// Text-encoder sequence (prompt + an edit's vision tokens and labels).
    text: u64,
    /// Edit reference rows in the DiT sequence.
    reference: u64,
    /// Generated image rows.
    image: u64,
    /// Largest single reference's vision patches (pre-merge), for the tower's working set.
    vision_patches_max: u64,
    /// Every reference's merged vision tokens.
    vision_tokens: u64,
    canvas: (u32, u32),
    latent: (u64, u64),
}

impl Rows {
    fn of(shape: &ImageRequestShape) -> Result<Self> {
        let prompt = u64::from(
            shape
                .prompt_tokens
                .unwrap_or(MAX_PROMPT_TOKENS as u32)
                .min(MAX_PROMPT_TOKENS as u32),
        );
        let canvas = generation_canvas(shape.width, shape.height, MIN_GENERATION_AREA_DEFAULT);
        let (lh, lw) = latent_grid(canvas.0, canvas.1)?;
        let patch = 2u64;
        let image = (u64::from(lh) / patch) * (u64::from(lw) / patch);
        let mut reference = 0u64;
        let mut vision_tokens = 0u64;
        let mut vision_patches_max = 0u64;
        for &(w, h) in &shape.references {
            let (rw, rh) = reference_vae_size(w, h);
            let r = u64::from(VAE_SPATIAL_RATIO) * patch;
            reference += (u64::from(rh) / r) * (u64::from(rw) / r);
            let (vw, vh) = reference_vision_size(w, h);
            let patches = (u64::from(vw) / 16) * (u64::from(vh) / 16);
            vision_patches_max = vision_patches_max.max(patches);
            vision_tokens += patches / VISION_MERGE_TOKENS as u64;
        }
        let labels = EDIT_LABEL_TOKENS_PER_REFERENCE * shape.references.len() as u64;
        Ok(Self {
            text: prompt + vision_tokens + labels,
            reference,
            image,
            vision_patches_max,
            vision_tokens,
            canvas,
            latent: (u64::from(lh), u64::from(lw)),
        })
    }

    fn sequence(&self) -> u64 {
        self.text + self.reference + self.image
    }
}

/// `(text, reference, image)` rows of `shape`'s packed DiT sequence, counted exactly as [`estimate`]
/// counts them — what the trainer (SW-7097) prices its train phase over.
pub fn sequence_rows(shape: &ImageRequestShape) -> Result<(u64, u64, u64)> {
    let rows = Rows::of(shape)?;
    Ok((rows.text, rows.reference, rows.image))
}

/// The modulation tables one request's precompute holds: every block, every distinct timestep (the
/// six steps, plus the reference class on an edit), every modality, at width `a`.
pub fn modulation_table_bytes(edit: bool, a: u64) -> u64 {
    let b = MiniMaxH3ImageDitConfig::default().base;
    let distinct_timesteps = FIXED_STEPS as u64 + u64::from(edit);
    MODULATION_PARAMS as u64
        * distinct_timesteps
        * MODALITY_NUM as u64
        * b.hidden_size as u64
        * b.num_layers as u64
        * a
}

/// What a selection engages, read through the contract so an undeclared rung never prices.
#[derive(Clone, Copy, Debug)]
struct Engaged {
    staged: bool,
    stream_te: bool,
    stream_dit: bool,
    window: u64,
    /// Rung 3's declared score budget, when the selection engages it.
    attention_chunk: Option<u32>,
}

fn engaged(contract: &MemoryProviderContract, selection: &MemorySelection) -> Engaged {
    // The generator stages every request (`unconditionally_engages_staged_residency`), so every
    // rung above 0 runs its phases apart whether or not the shared graph composes rung 1 into it.
    // Rung 0 alone is priced co-resident: that is the resident peak, the conservative direction.
    let staged = selection.strategy != MemoryStrategy::Resident
        || contract.engages_selection(selection, MemoryStrategy::StagedResidency);
    let rung4 = contract.engages_selection(selection, MemoryStrategy::BoundedTransformerResidency);
    let component = selection.parameters.window_component();
    let attention_chunk = contract
        .engages_selection(selection, MemoryStrategy::BoundedAttention)
        .then(|| {
            selection
                .parameters
                .attention_chunk_size
                .unwrap_or(crate::h3_image::memory::IMAGE_ATTENTION_CHUNK_SIZE)
        });
    Engaged {
        attention_chunk,
        staged,
        stream_te: rung4 && component.includes_text_encoder(),
        stream_dit: rung4 && component.includes_dit(),
        window: u64::from(
            selection
                .parameters
                .transformer_window_size
                .unwrap_or(1)
                .max(1),
        ),
    }
}

/// The largest sum of `window` consecutive entries.
fn window_peak(bytes: &[u64], window: u64) -> u64 {
    bytes
        .chunks(window.max(1) as usize)
        .map(|c| sum(c.iter().copied()))
        .max()
        .unwrap_or(0)
}

/// The head-major attention operands one block's attention holds at once over `s` rows: two
/// `[S, heads · head_dim]` copies at width `a`, for every head unbounded and for one chunk's whole
/// heads under rung 3 ([`crate::h3_image::memory::image_attention_budget`]). Two copies, not
/// three: run 43042251357 measured rung 3 saving 137 MB of the q4 denoise peak at 2048², and
/// `2 · s · (inner − 3 · head_dim) · a` is 125 MB of it, so the term never prices the rung below
/// what it was measured to hold.
fn attention_operand_bytes(chunk: Option<u32>, s: u64, a: u64) -> u64 {
    let b = MiniMaxH3ImageDitConfig::default().base;
    let heads = b.num_attention_heads as u64;
    let held = chunk.map_or(heads, |chunk| {
        let budget = crate::h3_image::memory::image_attention_budget(chunk, s);
        crate::h3_image::memory::image_attention_heads_per_chunk(budget, heads, s)
    });
    2 * s * held * b.attention_head_dim as u64 * a
}

/// The per-phase estimate of `shape` at `selection` on `contract` — see the module docs for what
/// each column holds. DERIVED.
pub fn estimate(
    contract: &MemoryProviderContract,
    facts: &ImageMemoryFacts,
    shape: &ImageRequestShape,
    selection: &MemorySelection,
) -> Result<ImageMemoryEstimate> {
    let rows = Rows::of(shape)?;
    let e = engaged(contract, selection);
    let a = facts.activation_width;
    let count = u64::from(shape.count.max(1));
    let edit = !shape.references.is_empty();
    let dit = MiniMaxH3ImageDitConfig::default();
    let b = &dit.base;
    let (hidden, inner, ffn) = (b.hidden_size as u64, b.inner_dim() as u64, b.ffn_dim as u64);
    let te = MiniMaxH3TeConfig::qwen3_vl_32b();
    let vae = crate::config::MiniMaxH3VaeConfig::image();

    // ---- weights each phase holds ------------------------------------------------------------
    let te_weights = if e.stream_te {
        facts
            .te_resident_bytes
            .saturating_add(window_peak(&facts.te_layer_bytes, e.window))
    } else {
        facts.te_bytes()
    };
    let vision_weights = if edit { facts.vision_bytes } else { 0 };
    let table = modulation_table_bytes(edit, a);
    let (precompute_weights, step_weights) = if e.stream_dit {
        (
            sum([
                facts.dit_trunk_bytes,
                window_peak(&facts.dit_adaln_bytes, e.window),
                table,
            ]),
            sum([
                facts.dit_trunk_bytes,
                window_peak(&facts.dit_body_bytes, e.window),
                table,
            ]),
        )
    } else {
        (
            sum([facts.dit_bytes(), table]),
            sum([
                facts.dit_trunk_bytes,
                sum(facts.dit_body_bytes.iter().copied()),
                table,
            ]),
        )
    };
    // Every component that is not this phase's own, when nothing stages (rung 0).
    let co_resident = |own: u64| {
        if e.staged {
            0
        } else {
            sum([
                facts.te_bytes(),
                vision_weights,
                facts.dit_bytes(),
                facts.decoder_bytes,
            ])
            .saturating_sub(own)
        }
    };

    // ---- conditioning ------------------------------------------------------------------------
    let te_dims = 2 * te.hidden_size as u64
        + 2 * (te.num_heads * te.head_dim) as u64
        + 2 * (te.num_kv_heads * te.head_dim) as u64
        + 3 * te.intermediate_size as u64;
    let context = rows.text * te.hidden_size as u64 * 4;
    let te_work = rows.text * te_dims * 2 + context;
    let vision = crate::text_encoder::minimax_h3_vision_config();
    let vision_work = if edit {
        rows.vision_patches_max
            * (6 * vision.hidden_size as u64 + 2 * vision.intermediate_size as u64)
            * 4
            + rows.vision_tokens
                * vision.out_hidden_size as u64
                * 2
                * (1 + vision.deepstack_visual_indexes.len() as u64)
    } else {
        0
    };
    // The reference encode runs after the text encoder is released: its weights, one tile's
    // activations at the widest level, and every reference latent so far.
    let tile = u64::from(IMAGE_DECODE_TILE_EDGE + 2 * IMAGE_DECODE_OVERLAP).pow(2);
    let widest = vae.block_out_channels.iter().copied().max().unwrap_or(0) as u64;
    let latent_c = vae.latent_channels as u64;
    let ref_latents = rows.reference * 4 * latent_c * 4;
    let encode = if edit {
        sum([
            facts.encoder_bytes,
            tile * widest * 4 * 3,
            ref_latents,
            context,
        ])
    } else {
        0
    };
    let te_step = sum([te_weights, vision_weights, te_work, vision_work]);
    let conditioning_device = te_step
        .max(encode)
        .saturating_add(co_resident(te_weights.saturating_add(vision_weights)));
    let conditioning_host = sum(shape.references.iter().map(|&(w, h)| {
        let (vw, vh) = reference_vision_size(w, h);
        let (rw, rh) = reference_vae_size(w, h);
        u64::from(vw) * u64::from(vh) * 3 + 2 * u64::from(rw) * u64::from(rh) * 3
    }));

    // ---- denoise -----------------------------------------------------------------------------
    let s = rows.sequence();
    let block_elements = 6 * hidden + 5 * inner + 2 * ffn + ffn;
    let step_work = s * block_elements * a
        + (rows.reference + rows.image) * b.video_patch_dim() as u64 * 4 * 4
        + s * b.attention_head_dim as u64 * 4 * 2
        + attention_operand_bytes(e.attention_chunk, s, a);
    // The LoRA overlay is resident in full at every rung — rung 4's streamed blocks keep their
    // factors as an overlay for the whole render (`DitBlockStream::adapters`) — plus one block's
    // residual working set (every target's per-row bytes over all `s` rows, an upper bound for the
    // refiner's text-only rows).
    let lora_resident = facts.adapters.resident_bytes;
    let lora_work = facts
        .adapter_block_rows
        .iter()
        .map(|&(per_row, fixed)| per_row.saturating_mul(s).saturating_add(fixed))
        .max()
        .unwrap_or(0);
    let retained = sum([
        context,
        ref_latents,
        count * latent_c * rows.latent.0 * rows.latent.1 * 4,
    ]);
    let denoise_device = sum([
        precompute_weights.max(
            step_weights
                .saturating_add(step_work)
                .saturating_add(lora_work),
        ),
        lora_resident,
        retained,
        co_resident(facts.dit_bytes()),
    ]);
    // The adapter install (`apply_minimax_h3_adapters`) runs right after the DiT load and before
    // the precompute evicts any AdaLN projection: its ordered construction peak (one file's source
    // map, its converted factors and every earlier file's installed factors —
    // `H3AdapterMemoryProfile::install_peak_bytes`) coexists with the DiT as loaded (the whole
    // DiT resident, the trunk alone under rung 4's deferred load) and the carried conditioning.
    let install_device = if facts.adapters.install_peak_bytes == 0 {
        0
    } else {
        sum([
            if e.stream_dit {
                facts.dit_trunk_bytes
            } else {
                facts.dit_bytes()
            },
            facts.adapters.install_peak_bytes,
            context,
            ref_latents,
            co_resident(facts.dit_bytes()),
        ])
    };
    let denoise_device = denoise_device.max(install_device);

    // ---- decode ------------------------------------------------------------------------------
    let (cw, ch) = (u64::from(rows.canvas.0), u64::from(rows.canvas.1));
    let tile_tokens = (u64::from(IMAGE_DECODE_TILE_EDGE) / u64::from(VAE_SPATIAL_RATIO)).pow(2) + 5;
    let d = vae.dim() as u64;
    let decode_tile = tile_tokens * (8 * d + vae.ffn_mult as u64 * d) * 4
        + u64::from(IMAGE_DECODE_TILE_EDGE + 2 * IMAGE_DECODE_OVERLAP).pow(2) * 3 * 4 * 2;
    // The stitched float32 canvas and the conversions `decoded_to_image` keeps live.
    let canvas = cw * ch * 3 * 4 * 5;
    let decode_device = sum([
        facts.decoder_bytes,
        decode_tile,
        canvas,
        count * latent_c * rows.latent.0 * rows.latent.1 * 4,
        co_resident(facts.decoder_bytes),
    ]);
    let resized = (rows.canvas != (shape.width, shape.height)).then(|| {
        mlx_gen::gen_core::imageops::resize_lanczos_u8_host_bytes(
            rows.canvas.1,
            rows.canvas.0,
            shape.height,
            shape.width,
        )
    });
    let resize = resized.transpose()?.unwrap_or(0);
    let outputs = count * u64::from(shape.width) * u64::from(shape.height) * 3;
    let decode_host = sum([cw * ch * 3, resize, outputs]);

    Ok(ImageMemoryEstimate {
        selection: *selection,
        phases: [
            ImagePhaseBytes {
                phase: MemoryPhase::Conditioning,
                device_bytes: conditioning_device,
                host_bytes: conditioning_host,
            },
            ImagePhaseBytes {
                phase: MemoryPhase::Denoise,
                device_bytes: denoise_device,
                host_bytes: 0,
            },
            ImagePhaseBytes {
                phase: MemoryPhase::Decode,
                device_bytes: decode_device,
                host_bytes: decode_host,
            },
        ],
    })
}

/// The ladder candidates this contract declares, cheapest-to-run first: rung 0, rung 1, rung 2 at
/// the shipped tile, rung 3 at the declared attention budget, and rung 4 (window 1 over both
/// transformers, attention chunked by cost order) when the load can stream.
pub fn ladder(contract: &MemoryProviderContract, tier: MemoryNumericTier) -> Vec<MemorySelection> {
    let decode = MemoryStrategyParameters {
        decode_tile_edge: Some(IMAGE_DECODE_TILE_EDGE),
        decode_overlap: Some(IMAGE_DECODE_OVERLAP),
        ..Default::default()
    };
    let attention = MemoryStrategyParameters {
        attention_chunk_size: Some(crate::h3_image::memory::IMAGE_ATTENTION_CHUNK_SIZE),
        ..decode
    };
    let candidates = [
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
                transformer_window_size: Some(
                    crate::h3_image::memory::IMAGE_TRANSFORMER_WINDOW_SIZE,
                ),
                transformer_window_component: Some(TransformerComponent::Both),
                ..attention
            },
        ),
    ];
    candidates
        .into_iter()
        .map(|(strategy, parameters)| MemorySelection {
            strategy,
            parameters,
            tier,
        })
        .filter(|selection| contract.validate_selection(selection).is_ok())
        .collect()
}

/// Walk [`ladder`] and return the first selection whose [`ImageMemoryEstimate::peak_bytes`] fits
/// `budget`, with the estimates of the rungs rejected before it. `Err` names the binding phase and
/// pool of the last rung when nothing fits.
pub fn admit(
    contract: &MemoryProviderContract,
    facts: &ImageMemoryFacts,
    shape: &ImageRequestShape,
    tier: MemoryNumericTier,
    budget: MemoryBudget,
) -> Result<ImageAdmission> {
    let mut rejected = Vec::new();
    for selection in ladder(contract, tier) {
        let estimate = estimate(contract, facts, shape, &selection)?;
        if budget.fits(estimate.peak_bytes()) {
            return Ok(ImageAdmission { estimate, rejected });
        }
        rejected.push(estimate);
    }
    let last = rejected.last().ok_or_else(|| {
        Error::Unsupported(format!(
            "{}: the contract declares no admissible memory selection",
            contract.provider_id
        ))
    })?;
    let phase = last.binding_phase();
    let binding = last
        .phases
        .iter()
        .find(|p| p.phase == phase)
        .copied()
        .unwrap_or(last.phases[1]);
    Err(Error::Unsupported(format!(
        "{}: {}x{} does not fit at any memory rung — the smallest, {:?}, needs {} B in its {:?} \
         phase ({} B GPU + {} B host) against an effective budget of {} B",
        contract.provider_id,
        shape.width,
        shape.height,
        last.selection.strategy,
        last.peak_bytes(),
        phase,
        binding.device_bytes,
        binding.host_bytes,
        budget.effective_bytes()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        standard_memory_behavior_context, MemoryBehaviorRoute, MemoryMode, MemorySafetyDecision,
    };

    const GB: u64 = 1_000_000_000;

    fn tier(quant: Option<mlx_gen::gen_core::Quant>) -> MemoryNumericTier {
        MemoryNumericTier {
            precision: mlx_gen::Precision::Bf16,
            quant,
            component_precision_floors: &[],
        }
    }

    fn contract(shape: LoadShape) -> MemoryProviderContract {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        spec.load_shape = shape;
        crate::h3_image::memory::weights_free_contract(&spec).unwrap()
    }

    use mlx_gen::gen_core::LoadShape;

    fn budget(total_bytes: u64) -> MemoryBudget {
        MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        }
    }

    fn at(c: &MemoryProviderContract, strategy: MemoryStrategy) -> MemorySelection {
        ladder(c, tier(Some(mlx_gen::gen_core::Quant::Q4)))
            .into_iter()
            .find(|s| s.strategy == strategy)
            .unwrap_or_else(|| panic!("{strategy:?} is not on the ladder"))
    }

    /// The derived weights land where the architecture puts them: a 33 B-parameter DiT is ~66 GB
    /// at bf16 and its AdaLN stack is H3's 26.02 GB; the tapped Qwen3-VL-32B text encoder is
    /// ~50 GB; the packed tiers shrink both.
    #[test]
    fn derived_weights_match_the_architecture() {
        let bf16 = ImageMemoryFacts::derived(None);
        assert!(
            (60 * GB..70 * GB).contains(&bf16.dit_bytes()),
            "{}",
            bf16.dit_bytes()
        );
        assert_eq!(
            sum(bf16.dit_adaln_bytes.iter().copied()),
            crate::memory_strategy::ADALN_EVICTED_BYTES
        );
        assert!(
            (45 * GB..55 * GB).contains(&bf16.te_bytes()),
            "{}",
            bf16.te_bytes()
        );
        assert_eq!(bf16.te_layer_bytes.len(), 50);
        assert_eq!(bf16.dit_body_bytes.len(), 50);
        let q4 = ImageMemoryFacts::derived(Some(4));
        assert!(q4.dit_bytes() * 3 < bf16.dit_bytes());
        assert!(q4.te_bytes() * 3 < bf16.te_bytes());
        // The six per-step heads are float32 and part of the resident trunk at every tier.
        assert!(q4.dit_trunk_bytes > 6 * 5376 * 96 * 4);
        assert!(
            (500_000_000..2 * GB).contains(&bf16.vision_bytes),
            "{}",
            bf16.vision_bytes
        );
    }

    /// The 16 MP request's sequence: 16,384 image rows at 32 px/token, plus the prompt.
    #[test]
    fn a_16_mp_request_packs_16384_image_rows() {
        let rows = Rows::of(&ImageRequestShape::text_to_image(4096, 4096)).unwrap();
        assert_eq!(rows.image, 16_384);
        assert_eq!(rows.text, 4096);
        assert_eq!(rows.sequence(), 20_480);
        // A small request is generated at the 4 MP minimum canvas.
        let small = Rows::of(&ImageRequestShape::text_to_image(512, 512)).unwrap();
        assert_eq!(small.canvas, (2048, 2048));
        assert_eq!(small.image, 4096);
    }

    /// Per-phase arithmetic at 16 MP q4: rung 0 charges everything co-resident; rung 1 takes the
    /// max over phases; rung 4 bounds both transformers to one window and keeps every
    /// non-transformer term. Host bytes are the decoded canvas and the output in decode only.
    #[test]
    fn per_phase_estimates_follow_the_ladder() {
        let c = contract(LoadShape::DeferredMaterialization);
        let facts = ImageMemoryFacts::derived(Some(4));
        let shape = ImageRequestShape::text_to_image(4096, 4096);
        let resident = estimate(&c, &facts, &shape, &at(&c, MemoryStrategy::Resident)).unwrap();
        let staged =
            estimate(&c, &facts, &shape, &at(&c, MemoryStrategy::StagedResidency)).unwrap();
        let streamed = estimate(
            &c,
            &facts,
            &shape,
            &at(&c, MemoryStrategy::BoundedTransformerResidency),
        )
        .unwrap();
        assert!(resident.peak_bytes() > staged.peak_bytes());
        assert!(
            staged.peak_bytes() > 2 * streamed.peak_bytes(),
            "staged {} vs streamed {}",
            staged.peak_bytes(),
            streamed.peak_bytes()
        );
        // Staged: the denoise phase holds the whole DiT at its precompute.
        let den = staged.phases[1];
        assert_eq!(den.phase, MemoryPhase::Denoise);
        assert!(den.device_bytes >= facts.dit_bytes());
        // Streamed conditioning holds `embed_tokens` and one layer, not the 50.
        let cond = streamed.phases[0];
        assert!(cond.device_bytes < facts.te_resident_bytes + 2 * facts.te_layer_bytes[0] + 2 * GB);
        assert!(cond.device_bytes >= facts.te_resident_bytes + facts.te_layer_bytes[0]);
        // Streamed denoise: the trunk, one block and the table, plus the 20,480-row activations.
        let den = streamed.phases[1];
        assert!(den.device_bytes >= facts.dit_trunk_bytes + facts.dit_body_bytes[0]);
        assert!(den.device_bytes < facts.dit_bytes() / 2);
        // Host: no reference copies; the 8-bit canvas and the output (same size: no resize).
        assert_eq!(cond.host_bytes, 0);
        assert_eq!(streamed.phases[2].host_bytes, 2 * 4096 * 4096 * 3);
        // The decode phase is identical across rungs 1 and 4 (the decoder is never streamed).
        assert_eq!(streamed.phases[2], staged.phases[2]);
    }

    /// AC2 (estimate side): a 16 MP request that does not fit resident — nor staged — is admitted
    /// on rung 4, the shared safety check accepts it at its streamed estimate, and refuses the
    /// same request priced at its resident peak.
    #[test]
    fn a_16_mp_request_that_does_not_fit_resident_is_admitted_streamed() {
        let c = contract(LoadShape::DeferredMaterialization);
        let facts = ImageMemoryFacts::derived(Some(4));
        let shape = ImageRequestShape::text_to_image(4096, 4096);
        let streamed = estimate(
            &c,
            &facts,
            &shape,
            &at(&c, MemoryStrategy::BoundedTransformerResidency),
        )
        .unwrap();
        let staged =
            estimate(&c, &facts, &shape, &at(&c, MemoryStrategy::StagedResidency)).unwrap();
        let cap = streamed.peak_bytes() + (staged.peak_bytes() - streamed.peak_bytes()) / 2;
        let admission = admit(
            &c,
            &facts,
            &shape,
            tier(Some(mlx_gen::gen_core::Quant::Q4)),
            budget(cap),
        )
        .unwrap();
        assert_eq!(
            admission.estimate.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency
        );
        let rejected: Vec<MemoryStrategy> = admission
            .rejected
            .iter()
            .map(|e| e.selection.strategy)
            .collect();
        assert_eq!(
            rejected,
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention
            ]
        );
        let generation = c.generation_memory(&admission.estimate.selection).unwrap();
        assert!(generation.stream_transformer_blocks && generation.tile_vae_decode);
        assert!(generation.chunk_attention);
        assert_eq!(generation.transformer_window_size, Some(1));

        // The loaded generator's safety check at that budget: the streamed estimate is accepted,
        // the resident peak is refused.
        let mut context = standard_memory_behavior_context(
            &c,
            MemoryStrategy::BoundedTransformerResidency,
            tier(Some(mlx_gen::gen_core::Quant::Q4)),
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .unwrap();
        context.selection = admission.estimate.selection;
        context.geometry.width = 4096;
        context.geometry.height = 4096;
        context.budget = budget(cap);
        context.predicted_peak_bytes = admission.estimate.peak_bytes();
        let check = |ctx: &mlx_gen::gen_core::MemoryRunContext| {
            crate::h3_image::memory::safety_check_at_tier(
                &c,
                tier(Some(mlx_gen::gen_core::Quant::Q4)),
                ctx,
            )
        };
        assert_eq!(check(&context), MemorySafetyDecision::Accept);
        context.predicted_peak_bytes = admission.rejected[0].peak_bytes();
        assert!(matches!(
            check(&context),
            MemorySafetyDecision::Reject { .. }
        ));

        // A resident load has no rung 4: the same budget is refused, naming the binding phase.
        let resident = contract(LoadShape::EagerMaterialization);
        let error = admit(
            &resident,
            &facts,
            &shape,
            tier(Some(mlx_gen::gen_core::Quant::Q4)),
            budget(cap),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("Denoise") || error.contains("Conditioning"),
            "{error}"
        );
        assert!(error.contains("GPU") && error.contains("host"), "{error}");
    }

    /// Rung 3 (SW-7094): on the ladder of every load, between rung 2 and rung 4. It lowers the
    /// per-step working set by exactly the attention operands of the heads outside one chunk —
    /// 1 head per chunk at the 8,192- and 20,480-row sequences, 3 at a 512-token 4 MP request's
    /// 4,608 — and changes neither conditioning nor decode.
    ///
    /// On an eager load the AdaLN precompute (the whole DiT plus the table) still binds the
    /// denoise phase of these text-to-image requests, so the derived facts show rung 3 at most
    /// equal to rung 2 there. Zeroing the AdaLN stack isolates the per-step term rung 3 changes:
    /// then the difference is exact, and a budget that fits rung 3 but not rung 2 admits rung 3.
    #[test]
    fn rung_3_lowers_the_denoise_step_by_the_unchunked_heads() {
        let q4 = tier(Some(mlx_gen::gen_core::Quant::Q4));
        let resident = contract(LoadShape::EagerMaterialization);
        let strategies: Vec<MemoryStrategy> =
            ladder(&resident, q4).iter().map(|s| s.strategy).collect();
        assert_eq!(
            strategies,
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention
            ]
        );
        let deferred = contract(LoadShape::DeferredMaterialization);
        assert_eq!(
            ladder(&deferred, q4).last().unwrap().strategy,
            MemoryStrategy::BoundedTransformerResidency
        );

        let derived = ImageMemoryFacts::derived(Some(4));
        let mut step_bound = derived.clone();
        step_bound.dit_adaln_bytes = vec![0; step_bound.dit_adaln_bytes.len()];
        let (heads, head_dim) = (56u64, 128u64);
        let short = |edge| ImageRequestShape {
            prompt_tokens: Some(512),
            ..ImageRequestShape::text_to_image(edge, edge)
        };
        let pair = |facts: &ImageMemoryFacts, shape: &ImageRequestShape| {
            let at_rung = |strategy| estimate(&resident, facts, shape, &at(&resident, strategy));
            (
                at_rung(MemoryStrategy::BoundedDecode).unwrap(),
                at_rung(MemoryStrategy::BoundedAttention).unwrap(),
            )
        };
        for (shape, s, per_chunk) in [
            (ImageRequestShape::text_to_image(2048, 2048), 8192u64, 1u64),
            (ImageRequestShape::text_to_image(4096, 4096), 20_480, 1),
            (short(2048), 4608, 3),
        ] {
            assert_eq!(Rows::of(&shape).unwrap().sequence(), s);
            let (rung2, rung3) = pair(&step_bound, &shape);
            assert_eq!(
                rung2.phases[1].device_bytes - rung3.phases[1].device_bytes,
                2 * s * (heads - per_chunk) * head_dim * 2,
                "{s} rows"
            );
            assert_eq!(rung2.phases[0], rung3.phases[0]);
            assert_eq!(rung2.phases[2], rung3.phases[2]);
            let (rung2, rung3) = pair(&derived, &shape);
            assert!(rung3.phases[1].device_bytes <= rung2.phases[1].device_bytes);
        }

        let shape = ImageRequestShape::text_to_image(4096, 4096);
        let (rung2, rung3) = pair(&step_bound, &shape);
        assert!(rung3.peak_bytes() < rung2.peak_bytes());
        let admission = admit(
            &resident,
            &step_bound,
            &shape,
            q4,
            budget(rung3.peak_bytes()),
        )
        .unwrap();
        assert_eq!(
            admission.estimate.selection.strategy,
            MemoryStrategy::BoundedAttention
        );
        assert_eq!(admission.rejected.len(), 3);
        let generation = resident
            .generation_memory(&admission.estimate.selection)
            .unwrap();
        assert!(generation.chunk_attention && generation.tile_vae_decode);
        assert_eq!(
            generation.attention_chunk_size,
            Some(crate::h3_image::memory::IMAGE_ATTENTION_CHUNK_SIZE)
        );
    }

    /// LoRA install: the ordered construction peak is its own denoise-phase envelope — the DiT as
    /// loaded (whole when resident, the trunk alone when streamed) plus the peak and the carried
    /// conditioning — so a stack whose install outgrows the forward's residency binds admission.
    #[test]
    fn the_adapter_install_peak_binds_the_denoise_phase() {
        let c = contract(LoadShape::DeferredMaterialization);
        let plain = ImageMemoryFacts::derived(Some(4));
        let shape = ImageRequestShape::text_to_image(4096, 4096);
        let peak = 200 * GB;
        let mut lora = plain.clone();
        lora.adapters = H3AdapterMemoryProfile {
            install_peak_bytes: peak,
            ..H3AdapterMemoryProfile::default()
        };
        let context = Rows::of(&shape).unwrap().text
            * MiniMaxH3TeConfig::qwen3_vl_32b().hidden_size as u64
            * 4;
        for (strategy, loaded) in [
            (MemoryStrategy::StagedResidency, plain.dit_bytes()),
            (
                MemoryStrategy::BoundedTransformerResidency,
                plain.dit_trunk_bytes,
            ),
        ] {
            let selection = at(&c, strategy);
            let without = estimate(&c, &plain, &shape, &selection).unwrap();
            assert!(without.phases[1].device_bytes < peak, "{strategy:?}");
            let with = estimate(&c, &lora, &shape, &selection).unwrap();
            assert_eq!(
                with.phases[1].device_bytes,
                loaded + peak + context,
                "{strategy:?}"
            );
            assert_eq!(with.phases[0], without.phases[0], "{strategy:?}");
            assert_eq!(with.phases[2], without.phases[2], "{strategy:?}");
        }
    }

    /// LoRA: the overlay is charged in full at every rung — never window-bounded — plus one
    /// block's residual working set over the whole packed sequence.
    #[test]
    fn adapters_are_charged_in_full_at_every_rung() {
        use mlx_gen::gen_core::h3_adapter_memory::{H3AdapterProjectionMemory, H3AdapterTarget};
        let c = contract(LoadShape::DeferredMaterialization);
        let plain = ImageMemoryFacts::derived(Some(4));
        let mut lora = plain.clone();
        // Two rank-16 bf16 LoRAs on every block's to_q: rank·(in + out)·2 bytes per target.
        let per_target = 16 * (5376 + 7168) * 2;
        lora.adapters = H3AdapterMemoryProfile {
            resident_bytes: 2 * 50 * per_target,
            install_peak_bytes: 2 * 50 * per_target,
            projections: (0..50)
                .map(|i| H3AdapterProjectionMemory {
                    target: H3AdapterTarget {
                        path: format!("transformer_blocks.{i}.attn.to_q"),
                        input: 5376,
                        output: 7168,
                        packed: true,
                        activation_width: 2,
                        activation_dtype: weightsmeta::Dtype::BF16,
                        rows: mlx_gen::gen_core::h3_memory::RowDomain::Packed,
                    },
                    installed_bytes: 2 * per_target,
                    resident_bytes: 2 * per_target,
                    forward_fixed_bytes: 0,
                    forward_bytes_per_row: 2 * (16 + 3 * 7168) * 2,
                })
                .collect(),
        };
        lora.adapter_block_rows = adapter_rows_per_block(&lora.adapters, 50);
        let shape = ImageRequestShape::text_to_image(4096, 4096);
        let s = 20_480u64;
        let working = 2 * (16 + 3 * 7168) * 2 * s;
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let selection = at(&c, strategy);
            let without = estimate(&c, &plain, &shape, &selection).unwrap();
            let with = estimate(&c, &lora, &shape, &selection).unwrap();
            let delta = with.phases[1].device_bytes - without.phases[1].device_bytes;
            let overlay = 2 * 50 * per_target;
            // Staged: the precompute (whole DiT) may still bind; the overlay is always added.
            assert!(delta >= overlay, "{strategy:?}: {delta} < {overlay}");
            assert!(delta <= overlay + working, "{strategy:?}: {delta}");
            if strategy == MemoryStrategy::BoundedTransformerResidency {
                assert_eq!(
                    delta,
                    overlay + working,
                    "streamed: full overlay + one block"
                );
            }
            // Conditioning and decode never carry the DiT's adapters.
            assert_eq!(with.phases[0], without.phases[0]);
            assert_eq!(with.phases[2], without.phases[2]);
        }
    }

    /// Edit: the vision tower is charged in conditioning alongside the text encoder, the
    /// references' copies on the host, their latents' rows in the DiT sequence — up to 16.
    #[test]
    fn edit_charges_the_vision_tower_and_sixteen_references() {
        let c = contract(LoadShape::DeferredMaterialization);
        let facts = ImageMemoryFacts::derived(Some(4));
        let t2i = ImageRequestShape::text_to_image(4096, 4096);
        let edit = ImageRequestShape {
            references: vec![(1024, 1024); 16],
            prompt_tokens: Some(512),
            ..t2i.clone()
        };
        let rows = Rows::of(&edit).unwrap();
        // Each 1024² reference: 32x32 = 1024 VAE rows and 64x64/4 = 1024 vision tokens.
        assert_eq!(rows.reference, 16 * 1024);
        assert_eq!(rows.vision_tokens, 16 * 1024);
        assert_eq!(rows.text, 512 + 16 * 1024 + 16 * 8);
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let selection = at(&c, strategy);
            let a = estimate(&c, &facts, &t2i, &selection).unwrap();
            let b = estimate(&c, &facts, &edit, &selection).unwrap();
            assert!(
                b.phases[0].device_bytes >= a.phases[0].device_bytes.min(b.phases[0].device_bytes)
            );
            assert!(
                b.phases[0].device_bytes
                    >= facts.vision_bytes + facts.te_resident_bytes + facts.te_layer_bytes[0],
                "{strategy:?}: the vision tower is co-resident with the encoder"
            );
            assert_eq!(b.phases[0].host_bytes, 16 * (1024 * 1024 * 3 * 3));
            assert!(b.phases[1].device_bytes > a.phases[1].device_bytes);
        }
    }
}
