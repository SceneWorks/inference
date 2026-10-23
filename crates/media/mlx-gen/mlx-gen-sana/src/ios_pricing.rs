//! Header-only, per-phase COUNTED pricing of one iOS SANA / SANA-Sprint request (epic 16774): the
//! architecture facts the anchor law scales a measured device anchor by. See
//! [`mlx_gen::ios_pricing`] for the contract.
//!
//! Every scratch figure is the SAME counted formula the production guard of that quantum checks
//! ([`crate::quantum`], `mlx_gen_pid::gemma2`'s Gemma quanta, [`mlx_gen::ios_quantum`]'s load
//! floors and sampler step), evaluated at the shapes the request runs under its regime (staging, the
//! DC-AE decode tile and overlap, the bounded encode tile, the cross-attention score budget). Weight
//! promotions are read from the snapshot's safetensors HEADERS (dtype + shape), the way a loaded
//! module reads them from its arrays; nothing is built and no tensor payload is read. The one
//! non-header read is the Gemma tokenizer (the CHI prompt's token count fixes the encoder's padded
//! window), memoized per snapshot.
//!
//! [`price`] is the pure core over [`SanaPricingFacts`] (configs + header inventories) and a
//! [`SanaPricingPlan`]; [`ios_pricing`] reads the facts of the request's snapshot, resolves the plan
//! from the request's execution levers exactly as the generator does on device, and prices it. The
//! CPU proof (`tests::observed_peak_quantum_per_phase_is_the_priced_one`) runs the production
//! modules on a tiny snapshot and asserts the largest checked quantum of every phase IS the price.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use mlx_gen::attention::AttentionBudget;
#[cfg(test)]
use mlx_gen::gen_core::WeightsSource;
use mlx_gen::gen_core::{MemoryMode, OffloadPolicy};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosPricingRequest, IosWindowPrice, IosWorkBasis};
use mlx_gen::ios_quantum::{elementwise_bytes, load_floor_bytes, sampler_step_bytes, tensor_bytes};
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::{Error, Result};
use mlx_gen_pid::gemma2::{gemma_embedding_bytes, gemma_final_norm_bytes, gemma_layer_bytes};
use mlx_rs::Dtype;

use crate::config::{BlockType, DcAeConfig, SanaTransformerConfig};
use crate::quantum::{self, labels};
use crate::text_encoder::{Gemma2Config, MAX_SEQUENCE_LENGTH, SANA_CHI_PROMPT};

/// The Gemma-2 caption encoder's quantum labels (`mlx_gen_pid::gemma2::Gemma2::forward`).
pub const GEMMA_LABELS: [&str; 3] = ["Gemma embedding/mask", "Gemma layer", "Gemma final norm"];

/// The phase every quantum label a SANA request emits runs in, `None` for a foreign label.
pub fn ios_quantum_phase(label: &str) -> Option<IosPhase> {
    use labels::*;
    match label {
        TEXT_LOAD | REFERENCE_ENCODER_LOAD | HEAVY_LOAD => Some(IosPhase::Load),
        l if GEMMA_LABELS.contains(&l) => Some(IosPhase::Conditioning),
        ENCODER_INPUT | ENCODER_BLOCK | ENCODER_DOWNSAMPLE | ENCODER_STITCH | ENCODER_OUTPUT
        | REFERENCE_LATENT => Some(IosPhase::Encode),
        ENTRY | BLOCK | OUTPUT | SCM_STEP | SCM_FINAL | LATENT | LATENT_BATCH => {
            Some(IosPhase::Denoise)
        }
        mlx_gen::sampler::SAMPLER_STEP_QUANTUM => Some(IosPhase::Denoise),
        DECODER_INPUT | DECODER_UPSAMPLE | DECODER_BLOCK | DECODER_OUTPUT | TILE_FOLD
        | TILE_FINALIZE | DECODED_PIXELS => Some(IosPhase::Decode),
        _ => None,
    }
}

/// Legacy diagnostic inventory, NOT an accepted additivity domain or capture-family policy.
/// The old `derive` fixtures do not certify correlated native/process transfer. Runtime and
/// capture grouping use the catalog's source-role policy; unproved reference modes retain their
/// own two-endpoint families. An empty legacy inventory grants no text-anchor inheritance.
pub const IOS_NON_ADDITIVE_MODES: &[&str] = &[];

/// One tensor of a safetensors header inventory.
#[derive(Clone, Debug)]
struct Tensor {
    dtype: Dtype,
    shape: Vec<u64>,
    elements: u64,
    bytes: u64,
}

/// A component directory's tensor headers, by name.
#[derive(Clone, Debug, Default)]
pub struct Inventory(BTreeMap<String, Tensor>);

fn mlx_dtype(dtype: mlx_gen::gen_core::weightsmeta::Dtype) -> Dtype {
    use mlx_gen::gen_core::weightsmeta::Dtype as H;
    match dtype {
        H::F16 => Dtype::Float16,
        H::BF16 => Dtype::Bfloat16,
        H::F32 => Dtype::Float32,
        H::F64 => Dtype::Float64,
        H::U8 | H::F8_E5M2 | H::F8_E4M3 | H::F8_E8M0 => Dtype::Uint8,
        H::I8 => Dtype::Int8,
        H::BOOL => Dtype::Bool,
        H::I16 => Dtype::Int16,
        H::U16 => Dtype::Uint16,
        H::I32 => Dtype::Int32,
        H::U32 => Dtype::Uint32,
        H::I64 => Dtype::Int64,
        H::U64 => Dtype::Uint64,
    }
}

impl Inventory {
    /// Every safetensors header under `dir` (header bytes only).
    pub fn read(dir: &Path) -> Result<Self> {
        let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(dir)
            .map_err(|e| Error::Msg(format!("{}: {e}", dir.display())))?;
        Ok(Self(
            headers
                .into_iter()
                .map(|h| {
                    let tensor = Tensor {
                        dtype: mlx_dtype(h.dtype),
                        shape: h.shape.iter().map(|d| *d as u64).collect(),
                        elements: h.shape.iter().map(|d| *d as u64).product(),
                        bytes: h.data_bytes,
                    };
                    (h.name, tensor)
                })
                .collect(),
        ))
    }

    fn stored_where(&self, pred: impl Fn(&str) -> bool) -> u64 {
        self.0
            .iter()
            .filter(|(k, _)| pred(k))
            .fold(0u64, |a, (_, t)| a.saturating_add(t.bytes))
    }

    /// The f32-resident bytes of the kept tensors: the DC-AE loaders cast every tensor they read to
    /// f32 (`crate::memory_strategy::resident_decoder_bytes`'s projection).
    fn f32_where(&self, pred: impl Fn(&str) -> bool) -> u64 {
        self.0
            .iter()
            .filter(|(k, _)| pred(k))
            .fold(0u64, |a, (_, t)| a.saturating_add(4 * t.elements))
    }

    fn require(&self, name: &str) -> Result<&Tensor> {
        self.0
            .get(name)
            .ok_or_else(|| Error::Msg(format!("SANA pricing: tensor {name} not in the snapshot")))
    }

    /// `crate::quant::lin`'s packed detection: a `{base}.scales` companion.
    fn is_packed(&self, base: &str) -> bool {
        self.0.contains_key(&format!("{base}.scales"))
    }
}

/// `crate::transformer::dtype_width`: the activation width a dtype occupies.
fn width(d: Dtype) -> u64 {
    match d {
        Dtype::Float16 | Dtype::Bfloat16 => 2,
        _ => 4,
    }
}

/// `crate::transformer::promoted_elements_bytes` at the trunk's f32 activation.
fn promoted_to_f32(t: &Tensor) -> u64 {
    if t.dtype != Dtype::Float32 && width(t.dtype) < 4 {
        t.elements * 4
    } else {
        0
    }
}

/// The f32 copy a dense (unpacked) linear `{base}.weight` makes; 0 when packed.
fn promoted_linear(inv: &Inventory, base: &str) -> Result<u64> {
    if inv.is_packed(base) {
        return Ok(0);
    }
    Ok(promoted_to_f32(inv.require(&format!("{base}.weight"))?))
}

/// Everything a SANA price depends on that is not the request: the routed configs, the Gemma
/// padded window, and the three components' header inventories.
#[derive(Clone, Debug)]
pub struct SanaPricingFacts {
    /// The snapshot root (`text_encoder/`, `transformer/`, `vae/`); the load floors are read here by
    /// the guards' own [`load_floor_bytes`].
    pub root: PathBuf,
    pub transformer: SanaTransformerConfig,
    pub dc_ae: DcAeConfig,
    pub gemma: Gemma2Config,
    /// Gemma's padded input window (`num_chi_tokens + 300 − 2`).
    pub gemma_tokens: u64,
    /// The caption tokens the trunk attends (`MAX_SEQUENCE_LENGTH` = 300).
    pub caption_tokens: u64,
    pub text_encoder: Inventory,
    pub trunk: Inventory,
    pub vae: Inventory,
}

static GEMMA_WINDOWS: OnceLock<Mutex<HashMap<PathBuf, u64>>> = OnceLock::new();

/// Gemma's padded window for the tokenizer at `tokenizer_json` (memoized per path).
fn gemma_window(tokenizer_json: &Path) -> Result<u64> {
    let cache = GEMMA_WINDOWS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = cache
        .lock()
        .ok()
        .and_then(|c| c.get(tokenizer_json).copied())
    {
        return Ok(v);
    }
    let tok = mlx_gen::tokenizer::TextTokenizer::from_file(
        tokenizer_json,
        mlx_gen::tokenizer::TokenizerConfig {
            max_length: 4096,
            pad_token_id: 0,
            chat_template: mlx_gen::tokenizer::ChatTemplate::None,
            pad_to_max_length: false,
        },
    )?;
    let chi = tok.encode_ids(SANA_CHI_PROMPT, true)?.len() as u64;
    let window = chi + MAX_SEQUENCE_LENGTH as u64 - 2;
    if let Ok(mut c) = cache.lock() {
        c.insert(tokenizer_json.to_path_buf(), window);
    }
    Ok(window)
}

impl SanaPricingFacts {
    /// The facts of the snapshot at `root` for the routed variant (`sprint` selects Sprint's trunk
    /// config, exactly as the loader does).
    pub fn read(root: &Path, sprint: bool) -> Result<Self> {
        let te = root.join("text_encoder");
        Ok(Self {
            root: root.to_path_buf(),
            transformer: if sprint {
                SanaTransformerConfig::sana_sprint_1600m()
            } else {
                SanaTransformerConfig::sana_1600m()
            },
            dc_ae: DcAeConfig::sana_f32c32(),
            gemma: Gemma2Config::gemma_2_2b(),
            gemma_tokens: gemma_window(&te.join("tokenizer.json"))?,
            caption_tokens: MAX_SEQUENCE_LENGTH as u64,
            text_encoder: Inventory::read(&te)?,
            trunk: Inventory::read(&root.join("transformer"))?,
            vae: Inventory::read(&root.join("vae"))?,
        })
    }
}

/// The memory-relevant execution of one request, resolved the way the generator resolves it.
#[derive(Clone, Copy, Debug)]
pub struct SanaPricingPlan {
    /// Sprint's CFG-free SCM loop (else base SANA's true-CFG flow-match).
    pub sprint: bool,
    /// The `Sequential` load (phases shed; the img2img reference encodes in its own phase).
    pub sequential: bool,
    /// An img2img request (a reference encode runs).
    pub img2img: bool,
    pub width: u32,
    pub height: u32,
    /// Images rendered (their latents are held until decode).
    pub count: u32,
    /// Bounded DC-AE decode `(tile edge, overlap)` in output pixels; `None` = whole-image decode.
    pub decode_tiling: Option<(i32, i32)>,
    /// Bounded img2img encode core in image pixels; `None` = whole-image encode.
    pub encode_tile: Option<i32>,
    /// Rung 3's cross-attention score budget.
    pub attention: AttentionBudget,
}

/// The DC-AE x32 spatial scale and latent width the noise is drawn at (`pipeline::create_noise`).
const LATENT_SCALE: u32 = 32;

/// Largest counted DC-AE decode quantum of one latent `[1, C, lh, lw]` (`DcAeDecoder::decode_head`,
/// then the tail tiled by `pipeline::decode_tiled` or whole).
fn decode_quanta(cfg: &DcAeConfig, lh: u64, lw: u64, tiling: Option<(i32, i32)>, out: &mut Quanta) {
    let start = out.0.len();
    decode_quanta_body(cfg, lh, lw, tiling, out);
    let (height, width) = (lh * u64::from(LATENT_SCALE), lw * u64::from(LATENT_SCALE));
    let elements = 3 * height * width;
    out.take_at(
        mlx_gen::ios_quantum::decoded_to_image_bytes_from_metadata(elements),
        labels::DECODED_PIXELS,
        height,
        width,
    );
    // NHWC decoder result and its NCHW transpose are two handles to ONE
    // source backing. The shared raw guard counts the new conversion buffer;
    // the still-owned decoded input is a separate, non-subtractive held term.
    out.0.last_mut().unwrap().3 = tensor_bytes(4, &[elements]);
    let unscaled = tensor_bytes(4, &[1, cfg.latent_channels as u64, lh, lw]);
    for q in &mut out.0[start..] {
        if q.1 != labels::DECODER_INPUT {
            // decode_to_image_with_tiling owns `unscaled` until it returns.
            // The head-input guard already includes that input; later quanta
            // operate on features while this distinct source target stays held.
            q.3 = q.3.saturating_add(unscaled);
        }
    }
}

fn decode_quanta_body(
    cfg: &DcAeConfig,
    lh: u64,
    lw: u64,
    tiling: Option<(i32, i32)>,
    out: &mut Quanta,
) {
    let ch = |i: usize| cfg.block_out_channels[i] as u64;
    let n = cfg.num_stages();
    let tail = cfg
        .block_types
        .iter()
        .take_while(|t| **t == BlockType::Res)
        .count();
    let block = |i: usize, h: u64, w: u64| match cfg.block_types[i] {
        BlockType::Res => quantum::dcae_res_block_bytes(1, h, w, ch(i)),
        BlockType::EfficientVit => {
            quantum::dcae_evit_block_bytes(1, h, w, ch(i), cfg.qkv_multiscales.len() as u64)
        }
    };
    let latent_channels = cfg.latent_channels as u64;
    out.take_at(
        quantum::dcae_decoder_input_bytes(1, lh, lw, latent_channels, ch(n - 1)),
        labels::DECODER_INPUT,
        lh,
        lw,
    );
    // `run_decoder` per stage, deepest first: its upsample (every stage but the deepest), then its
    // blocks. `stages` is one closure over (range, extent) so head and tail price identically.
    let stages = |range: std::ops::Range<usize>, mut h: u64, mut w: u64, out: &mut Quanta| {
        for i in range.rev() {
            if i + 1 < n {
                h *= 2;
                w *= 2;
                out.take_at(
                    quantum::dcae_upsample_bytes(1, h, w, ch(i + 1), ch(i)),
                    labels::DECODER_UPSAMPLE,
                    h,
                    w,
                );
            }
            for _ in 0..cfg.layers_per_block[i] {
                out.take_at(block(i, h, w), labels::DECODER_BLOCK, h, w);
            }
        }
        (h, w)
    };
    if tail == 0 {
        let (h, w) = stages(0..n, lh, lw, out);
        out.take_at(
            quantum::dcae_decoder_output_bytes(1, h, w, ch(0), cfg.in_channels as u64),
            labels::DECODER_OUTPUT,
            h,
            w,
        );
        return;
    }
    let (hh, hw) = stages(tail..n, lh, lw, out);
    let ups = (0..tail).filter(|i| i + 1 < n).count() as u32;
    let tail_scale = 1u64 << ups;
    let vae = VaeTiling {
        spatial_scale: tail_scale as i32,
        temporal_scale: 1,
        causal_temporal: false,
        full_res_channels: 3,
    };
    let tiles = tiling
        .map(|(edge, overlap)| TilingConfig::spatial_only(edge, overlap))
        .filter(|t| t.needs_tiling(vae, 1, hh as i32, hw as i32));
    let Some(tiles) = tiles else {
        let (h, w) = stages(0..tail, hh, hw, out);
        out.take_at(
            quantum::dcae_decoder_output_bytes(1, h, w, ch(0), cfg.in_channels as u64),
            labels::DECODER_OUTPUT,
            h,
            w,
        );
        return;
    };
    let plan = tiles.plan(vae, 1, hh as i32, hw as i32);
    let extents = |axis: &[mlx_gen::tiling::AxisTile]| {
        axis.iter()
            .map(|a| (a.end - a.start) as u64)
            .collect::<std::collections::BTreeSet<_>>()
    };
    let (out_h, out_w) = (hh * tail_scale, hw * tail_scale);
    let head = tensor_bytes(4, &[1, hh, hw, ch(tail)]);
    for th in extents(&plan.h) {
        for tw in extents(&plan.w) {
            let start = out.0.len();
            let (h, w) = stages(0..tail, th, tw, out);
            out.take_at(
                quantum::dcae_decoder_output_bytes(1, h, w, ch(0), cfg.in_channels as u64),
                labels::DECODER_OUTPUT,
                h,
                w,
            );
            // Per-axis fold owners exist beside a later tile's decode. They are not two
            // independently maximized full-image slabs: w_acc spans THIS tile row, h_acc the image.
            let accumulators = if plan.w.len() > 1 {
                tensor_bytes(4, &[h, out_w, cfg.in_channels as u64])
            } else {
                0
            } + if plan.h.len() > 1 {
                tensor_bytes(4, &[out_h, out_w, cfg.in_channels as u64])
            } else {
                0
            };
            for q in &mut out.0[start..] {
                q.3 = q.3.saturating_add(head).saturating_add(accumulators);
            }
            out.take_at(
                quantum::tile_fold_bytes(h, w, out_h, out_w, cfg.in_channels as u64),
                labels::TILE_FOLD,
                h,
                w,
            );
            out.0.last_mut().unwrap().3 = head; // fold guard already prices its accumulators
        }
    }
    out.take_at(
        quantum::tile_finalize_bytes(out_h, out_w),
        labels::TILE_FINALIZE,
        out_h,
        out_w,
    );
    out.0.last_mut().unwrap().3 = head + tensor_bytes(4, &[out_h, out_w, cfg.in_channels as u64]);
}

/// Largest counted DC-AE encode quantum of an `height x width` image (`DcAeEncoder::encode`, or
/// `encode_tiled` at `tile` image pixels).
fn encode_quanta(cfg: &DcAeConfig, height: u64, width: u64, tile: Option<i32>, out: &mut Quanta) {
    let ch = |i: usize| cfg.block_out_channels[i] as u64;
    let n = cfg.num_stages();
    let prefix = cfg
        .block_types
        .iter()
        .take_while(|t| **t == BlockType::Res)
        .count();
    let block = |i: usize, h: u64, w: u64| match cfg.block_types[i] {
        BlockType::Res => quantum::dcae_res_block_bytes(1, h, w, ch(i)),
        BlockType::EfficientVit => {
            quantum::dcae_evit_block_bytes(1, h, w, ch(i), cfg.qkv_multiscales.len() as u64)
        }
    };
    // `EncStage::run_encoder` over stages `range` from extent (h, w): blocks, then the downsample.
    let stages = |range: std::ops::Range<usize>, mut h: u64, mut w: u64, out: &mut Quanta| {
        for i in range {
            for _ in 0..cfg.encoder_layers_per_block[i] {
                out.take_at(block(i, h, w), labels::ENCODER_BLOCK, h, w);
            }
            if i + 1 < n {
                out.take_at(
                    quantum::dcae_downsample_bytes(1, h, w, ch(i), ch(i + 1)),
                    labels::ENCODER_DOWNSAMPLE,
                    h,
                    w,
                );
                h = h.div_ceil(2);
                w = w.div_ceil(2);
            }
        }
        (h, w)
    };
    let input = |h: u64, w: u64, out: &mut Quanta| {
        out.take_at(
            quantum::dcae_encoder_input_bytes(1, h, w, cfg.in_channels as u64, ch(0)),
            labels::ENCODER_INPUT,
            h,
            w,
        );
    };
    let downsamples = (0..prefix).filter(|i| i + 1 < n).count() as u32;
    let cell = 1u64 << downsamples;
    // `DcAeEncoder::prefix_halo`.
    let (mut radius, mut scale) = (1u64, 1u64);
    for i in 0..prefix {
        radius += 2 * cfg.encoder_layers_per_block[i] as u64 * scale;
        if i + 1 < n {
            radius += scale;
            scale *= 2;
        }
    }
    let halo = radius.div_ceil(cell) * cell;
    let tiled = tile.map(|core| (core.max(1) as u64).div_ceil(cell) * cell);
    let (h, w) = match tiled {
        Some(core) if prefix > 0 && (height > core || width > core) => {
            // Preserve every border/interior extent family, not only the largest tile. Cropped
            // prefix features remain in rows until stitch and can retain the full halo parent.
            let extents = |len: u64| {
                (0..len)
                    .step_by(core as usize)
                    .map(|s0| ((s0 + core).min(len) + halo).min(len) - s0.saturating_sub(halo))
                    .collect::<Vec<_>>()
            };
            let hs = extents(height);
            let ws = extents(width);
            let channels = ch(prefix.min(n - 1));
            let parent_elements = hs
                .iter()
                .map(|h| h / cell)
                .sum::<u64>()
                .saturating_mul(ws.iter().map(|w| w / cell).sum::<u64>())
                .saturating_mul(channels);
            let parents = parent_elements.saturating_mul(4);
            for th in hs
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
            {
                for tw in ws
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                {
                    let start = out.0.len();
                    input(th, tw, out);
                    stages(0..prefix, th, tw, out);
                    let current_parent = tensor_bytes(4, &[th / cell, tw / cell, channels]);
                    for q in &mut out.0[start..] {
                        q.3 = q.3.saturating_add(parents.saturating_sub(current_parent));
                    }
                }
            }
            let (sh, sw) = (height / cell, width / cell);
            out.take_at(
                quantum::dcae_encode_stitch_bytes(1, sh, sw, channels),
                labels::ENCODER_STITCH,
                sh,
                sw,
            );
            // The stitch guard counts the core crops; only their additional halo parent extent
            // belongs beside it. Logical view/backing accounting is NOT a physical certificate.
            out.0.last_mut().unwrap().3 =
                parents.saturating_sub(tensor_bytes(4, &[sh, sw, channels]));
            stages(prefix..n, sh, sw, out)
        }
        _ => {
            input(height, width, out);
            stages(0..n, height, width, out)
        }
    };
    out.take_at(
        quantum::dcae_encoder_output_bytes(1, h, w, ch(n - 1), cfg.latent_channels as u64),
        labels::ENCODER_OUTPUT,
        h,
        w,
    );
}

/// One quantum's scratch bytes, label, optional spatial extent, and extra held bytes.
type PhaseQuantum = (u64, &'static str, Option<(u64, u64)>, u64);

/// The counted quanta of one phase, `(scratch, label)` in run order.
#[derive(Clone, Default)]
struct Quanta(Vec<PhaseQuantum>);
impl Quanta {
    fn take(&mut self, scratch: u64, label: &'static str) {
        self.0.push((scratch, label, None, 0));
    }
    fn take_at(&mut self, scratch: u64, label: &'static str, h: u64, w: u64) {
        self.0.push((scratch, label, Some((h, w)), 0));
    }
}

/// One counted quantum of a priced request: its phase, the component bytes resident around it, its
/// guard's scratch and its label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PricedQuantum {
    pub phase: IosPhase,
    pub resident_bytes: u64,
    pub scratch_bytes: u64,
    pub label: &'static str,
    pub window: IosWindowPrice,
}

/// Gemma-2's compute width and the largest promoted dense-weight copy of any layer
/// (`Gemma2::compute_dtype` / `Layer::promoted_weight_bytes`, from headers).
fn gemma_facts(inv: &Inventory, cfg: &Gemma2Config) -> Result<(u64, Vec<u64>)> {
    let embed = "model.embed_tokens";
    let compute = if inv.is_packed(embed) {
        inv.require(&format!("{embed}.scales"))?.dtype
    } else {
        inv.require(&format!("{embed}.weight"))?.dtype
    };
    let cw = mlx_gen::ios_quantum::dtype_bytes(compute);
    let mut layers = Vec::new();
    for i in 0..cfg.num_layers {
        let mut total = 0u64;
        for l in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            let base = format!("model.layers.{i}.{l}");
            if inv.is_packed(&base) {
                continue;
            }
            let t = inv.require(&format!("{base}.weight"))?;
            if t.dtype != compute {
                total += t.elements * mlx_gen::ios_quantum::dtype_bytes(t.dtype).max(cw);
            }
        }
        layers.push(total);
    }
    Ok((cw, layers))
}

/// The promoted f32 weight copies of trunk block `i` (`SanaBlock::quantum`'s `weight_bytes`).
fn block_weight_bytes(inv: &Inventory, i: i32) -> Result<u64> {
    let p = format!("transformer_blocks.{i}");
    let mut total = 0u64;
    for conv in ["ff.conv_inverted", "ff.conv_depth", "ff.conv_point"] {
        total += promoted_to_f32(inv.require(&format!("{p}.{conv}.weight"))?);
    }
    for lin in [
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out.0",
        "attn1.to_out.0",
    ] {
        total += promoted_linear(inv, &format!("{p}.{lin}"))?;
    }
    // attn1's q/k/v (packed or not, the same elements) at `to_out`'s storage dtype; none when the
    // projections or `to_out` are quantized.
    let out = format!("{p}.attn1.to_out.0");
    if !inv.is_packed(&format!("{p}.attn1.to_q")) && !inv.is_packed(&out) {
        let stored = inv.require(&format!("{out}.weight"))?.dtype;
        let elements: u64 = ["to_q", "to_k", "to_v"]
            .iter()
            .map(|q| {
                inv.require(&format!("{p}.attn1.{q}.weight"))
                    .map(|t| t.elements)
            })
            .sum::<Result<u64>>()?;
        if stored != Dtype::Float32 && width(stored) < 4 {
            total += elements * 4;
        }
    }
    Ok(total)
}

/// The trunk entry's promoted copies (`SanaTransformer::entry_quantum`'s `weight_bytes`).
fn entry_weight_bytes(inv: &Inventory, cfg: &SanaTransformerConfig) -> Result<u64> {
    let ts = if cfg.guidance_embeds {
        "time_embed.timestep_embedder"
    } else {
        "time_embed.emb.timestep_embedder"
    };
    let mut linears = vec![
        format!("{ts}.linear_1"),
        format!("{ts}.linear_2"),
        "time_embed.linear".into(),
        "caption_projection.linear_1".into(),
        "caption_projection.linear_2".into(),
    ];
    if cfg.guidance_embeds {
        linears.push("time_embed.guidance_embedder.linear_1".into());
        linears.push("time_embed.guidance_embedder.linear_2".into());
    }
    let mut total = promoted_to_f32(inv.require("patch_embed.proj.weight")?);
    for l in linears {
        total += promoted_linear(inv, &l)?;
    }
    Ok(total)
}

/// Price one request (see the module doc): every phase's worst quantum. Pure over `facts` and
/// `plan`.
pub fn price(facts: &SanaPricingFacts, plan: &SanaPricingPlan) -> Result<IosPricing> {
    let mut pricing = IosPricing::default();
    for q in priced_quanta(facts, plan)? {
        pricing.take_window(q.window);
    }
    if !plan.img2img {
        pricing.phase_not_run(IosPhase::Encode, "text mode has no reference encode");
    }
    Ok(pricing)
}

/// Every counted quantum `plan` runs on `facts`, in run order (the guards' own formulas at the
/// request's shapes). [`price`] folds them into one price per phase.
pub fn priced_quanta(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
) -> Result<Vec<PricedQuantum>> {
    priced_quanta_with_cfg(facts, plan, !plan.sprint)
}

fn priced_quanta_with_cfg(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
    cfg_on: bool,
) -> Result<Vec<PricedQuantum>> {
    let cfg = &facts.transformer;
    let dcae = &facts.dc_ae;
    let mut out = Vec::new();
    let count = u64::from(plan.count.max(1));

    // ---- component bytes (resident widths) ----
    let te_bytes = facts.text_encoder.stored_where(|_| true);
    let dit_bytes = facts.trunk.stored_where(|_| true);
    let dec_bytes = facts.vae.f32_where(|k| k.starts_with("decoder."));
    let enc_bytes = facts.vae.f32_where(|k| k.starts_with("encoder."));

    // ---- carried tensors ----
    let (gw, layer_weights) = gemma_facts(&facts.text_encoder, &facts.gemma)?;
    let hidden = facts.gemma.hidden_size as u64;
    let n_cond = if cfg_on { 2 } else { 1 }; // actual encode_conditioning guidance > 1 gate
    let cond_bytes = tensor_bytes(gw, &[n_cond, facts.caption_tokens, hidden])
        .saturating_add(tensor_bytes(4, &[n_cond, facts.caption_tokens]));
    let (lh, lw) = (
        u64::from(plan.height / LATENT_SCALE),
        u64::from(plan.width / LATENT_SCALE),
    );
    let latent_elems = cfg.in_channels as u64 * lh * lw;
    let latent_bytes = tensor_bytes(4, &[latent_elems]);
    let reference = if plan.img2img { latent_bytes } else { 0 };

    // ---- regime: which components each phase holds ----
    let staged = plan.sequential;
    // A staged img2img encodes separately. The resident warm pair eagerly loads the reusable
    // heavy superset (`Residency::ensure_warm_locked` passes true to `load_heavy`), including the
    // DC-AE encoder even on text requests. Encoder and decoder use one shared weight inventory.
    let heavy_has_encoder = !staged;
    let all = te_bytes + dit_bytes + dec_bytes + if heavy_has_encoder { enc_bytes } else { 0 };
    let resident = |sequential: u64| if staged { sequential } else { all };
    let mut points = BTreeMap::<&str, usize>::new();
    let mut phase = |phase: IosPhase, resident_bytes: u64, quanta: Quanta| {
        for (scratch_bytes, label, extent, extra_held) in quanta.0 {
            let structural = match phase {
                IosPhase::Load => {
                    if staged || label == labels::TEXT_LOAD {
                        0
                    } else {
                        te_bytes
                    }
                }
                IosPhase::Conditioning => resident(te_bytes),
                IosPhase::Encode => resident(enc_bytes),
                IosPhase::Denoise => resident(dit_bytes + dec_bytes),
                IosPhase::Decode => resident(dec_bytes),
            };
            let mut held = resident_bytes
                .saturating_sub(structural)
                .saturating_add(extra_held);
            if phase == IosPhase::Decode {
                held = held.saturating_add(
                    (count - 1) * 3 * u64::from(plan.width) * u64::from(plan.height),
                );
            }
            if matches!(label, labels::BLOCK | labels::OUTPUT) {
                // entry() returns these owners and forward_with_memory retains them across the
                // stack. The block guard counts projected K/V, not this projected caption input.
                held = held.saturating_add(tensor_bytes(
                    4,
                    &[1, facts.caption_tokens, cfg.inner_dim() as u64],
                ));
                let time_columns = if label == labels::BLOCK { 7 } else { 6 };
                // OUTPUT guard already counts emb among its modulation vectors; temb still lives.
                held = held
                    .saturating_add(tensor_bytes(4, &[1, time_columns, cfg.inner_dim() as u64]));
            }
            let point = points.entry(label).or_default();
            let (op_h, op_w) = extent.unwrap_or((u64::from(plan.height), u64::from(plan.width)));
            let patch = cfg.patch_size.max(1) as u64;
            let basis = IosWorkBasis {
                forward_rows: if phase == IosPhase::Load { 0 } else { 1 },
                token_count: if phase == IosPhase::Conditioning {
                    facts.gemma_tokens
                } else if phase == IosPhase::Denoise {
                    (lh / patch) * (lw / patch)
                } else {
                    0
                },
                model_width: op_w as u32,
                model_height: op_h as u32,
                processed_plane_bytes: if phase == IosPhase::Encode {
                    4 * dcae.in_channels as u64 * u64::from(plan.height) * u64::from(plan.width)
                } else {
                    0
                },
                output_plane_bytes: 3 * u64::from(plan.width) * u64::from(plan.height),
                patch_width: if phase == IosPhase::Denoise {
                    patch as u32
                } else {
                    0
                },
                patch_height: if phase == IosPhase::Denoise {
                    patch as u32
                } else {
                    0
                },
                ..Default::default()
            };
            let window = IosWindowPrice {
                phase,
                class: Some(format!(
                    "sana/{}/{}/{label}/source-op-{}",
                    if plan.sprint { "scm" } else { "flow" },
                    phase.as_key(),
                    *point
                )),
                label: label.into(),
                resident_bytes: structural,
                held_bytes: held,
                scratch_bytes,
                basis: Some(basis),
                owner_domain: "sana/source-logical-owners-v1".into(),
            };
            *point += 1;
            out.push(PricedQuantum {
                phase,
                resident_bytes,
                scratch_bytes,
                label,
                window,
            });
        }
    };

    // ---- Load ----
    let te_dir = facts.root.join("text_encoder");
    let heavy_dirs = [facts.root.join("transformer"), facts.root.join("vae")];
    let mut load = Quanta::default();
    load.take(
        load_floor_bytes(std::slice::from_ref(&te_dir), |_| true)?,
        labels::TEXT_LOAD,
    );
    phase(IosPhase::Load, 0, load);
    if plan.img2img && staged {
        let mut load = Quanta::default();
        load.take(
            load_floor_bytes(std::slice::from_ref(&heavy_dirs[1]), |n| {
                n.starts_with("encoder.")
            })?,
            labels::REFERENCE_ENCODER_LOAD,
        );
        // The preprocessed f32 reference image (`encode_init_latents_with_tiling`'s input) is
        // allocated after this load and before the encode's first guard, i.e. inside the load's
        // interval of the Load phase: it is resident by the phase's end. (Unstaged, the image is
        // allocated after the last Gemma quantum synchronizes, so it belongs to Encode alone.)
        let image = tensor_bytes(
            4,
            &[
                dcae.in_channels as u64,
                u64::from(plan.height),
                u64::from(plan.width),
            ],
        );
        phase(IosPhase::Load, cond_bytes + image, load);
    }
    let mut load = Quanta::default();
    load.take(
        load_floor_bytes(&heavy_dirs, |n| {
            heavy_has_encoder || !n.starts_with("encoder.")
        })?,
        labels::HEAVY_LOAD,
    );
    phase(
        IosPhase::Load,
        if staged {
            cond_bytes + reference
        } else {
            te_bytes
        },
        load,
    );

    // ---- Conditioning (Gemma-2 at its padded window, batch 1 per prompt) ----
    let l = facts.gemma_tokens;
    let g = &facts.gemma;
    let mut text = Quanta::default();
    text.take(gemma_embedding_bytes(g, 1, l, gw), GEMMA_LABELS[0]);
    for w in &layer_weights {
        text.take(gemma_layer_bytes(g, 1, l, gw, *w), GEMMA_LABELS[1]);
    }
    text.take(gemma_final_norm_bytes(g, 1, l, gw), GEMMA_LABELS[2]);
    phase(IosPhase::Conditioning, resident(te_bytes), text.clone());
    if cfg_on {
        phase(
            IosPhase::Conditioning,
            resident(te_bytes) + cond_bytes / n_cond,
            text,
        );
    }

    // ---- Encode (img2img: the reference, from the preprocessed f32 NCHW image) ----
    if plan.img2img {
        let (h, w) = (u64::from(plan.height), u64::from(plan.width));
        let mut enc = Quanta::default();
        encode_quanta(dcae, h, w, plan.encode_tile, &mut enc);
        enc.take(
            elementwise_bytes(latent_elems + 1, 1),
            labels::REFERENCE_LATENT,
        );
        let image = tensor_bytes(4, &[dcae.in_channels as u64, h, w]);
        phase(
            IosPhase::Encode,
            resident(enc_bytes + cond_bytes) + image,
            enc,
        );
    }

    // ---- Denoise (batch-1 trunk forwards; CFG runs cond and uncond one after the other) ----
    let p = cfg.patch_size.max(1) as u64;
    let tokens = (lh / p) * (lw / p);
    let dim = cfg.inner_dim() as u64;
    let mut den = Quanta::default();
    den.take(
        quantum::entry_bytes(quantum::SanaEntryQuantum {
            batch: 1,
            latent_channels: cfg.in_channels as u64,
            tokens,
            patch: p,
            dim,
            caption_tokens: facts.caption_tokens,
            caption_channels: hidden,
            caption_element_bytes: if gw >= 4 { 4 } else { 2 },
            element_bytes: 4,
            weight_bytes: entry_weight_bytes(&facts.trunk, cfg)?,
        })
        .saturating_add(sampler_step_bytes(latent_elems)),
        labels::ENTRY,
    );
    for i in 0..cfg.num_layers {
        let q = crate::transformer::configured_block_quantum(
            cfg,
            1,
            (lh / p) as i32,
            (lw / p) as i32,
            facts.caption_tokens as i32,
            plan.attention,
        );
        den.take(
            quantum::block_bytes(quantum::SanaBlockQuantum {
                weight_bytes: block_weight_bytes(&facts.trunk, i)?,
                ..q
            }),
            labels::BLOCK,
        );
    }
    den.take(
        quantum::output_bytes(quantum::SanaOutputQuantum {
            batch: 1,
            tokens,
            dim,
            out_features: p * p * cfg.out_channels as u64,
            element_bytes: 4,
            weight_bytes: promoted_linear(&facts.trunk, "proj_out")?,
        }),
        labels::OUTPUT,
    );
    if plan.sprint {
        den.take(quantum::scm_step_bytes(latent_elems), labels::SCM_STEP);
        den.take(
            quantum::scm_step_bytes(latent_elems)
                .saturating_add(elementwise_bytes(latent_elems, 1)),
            labels::SCM_FINAL,
        );
    } else {
        den.take(
            sampler_step_bytes(latent_elems),
            mlx_gen::sampler::SAMPLER_STEP_QUANTUM,
        );
    }
    den.take(
        quantum::scm_step_bytes(count * latent_elems),
        labels::LATENT_BATCH,
    );
    // The sampler state carried beside the latents through every trunk forward: the drawn noise
    // outlives the loop whenever it is not itself the starting latents (`denoise_cfg` /
    // `denoise_sprint` keep `noise` alive after blending the img2img start, and Sprint's prior is the
    // separate `noise · σ_data`), and Sprint's SCM loop feeds the trunk a separate scaled model input
    // (`lat_in`, live beside the latents for the whole forward).
    let noise = if plan.img2img || plan.sprint {
        latent_bytes
    } else {
        0
    };
    let model_input = if plan.sprint { latent_bytes } else { 0 };
    phase(
        IosPhase::Denoise,
        resident(dit_bytes + dec_bytes)
            + cond_bytes
            + reference
            + noise
            + model_input
            + (count - 1) * latent_bytes,
        den,
    );

    // ---- Decode (one latent at a time; every latent of the batch is held) ----
    let mut dec = Quanta::default();
    decode_quanta(dcae, lh, lw, plan.decode_tiling, &mut dec);
    phase(
        IosPhase::Decode,
        resident(dec_bytes) + count * latent_bytes,
        dec,
    );
    Ok(out)
}

/// Price an iOS SANA (`sana_1600m`) or SANA-Sprint (`sana_sprint_1600m`) request: text-to-image or
/// img2img (one reference), under the request's load policy and execution levers. A request the
/// generator would refuse (an out-of-domain lever, the withheld rung 4) is refused here too.
enum PricingSampling {
    Flow(mlx_gen::sampler::flow_sampling::ResolvedFlowControls),
    Scm(crate::sampling::ResolvedSprintControls),
}

pub fn ios_pricing(req: &IosPricingRequest<'_>) -> Result<IosPricing> {
    req.validate_work_context()?;
    let generation = mlx_gen::sampler::flow_sampling::require_generation(req.generation)?;
    let has_init = matches!(req.mode, MemoryMode::ImageToImage);
    let sampling = match req.provider {
        crate::MODEL_ID => PricingSampling::Flow(crate::sampling::resolve_base_with_init(
            generation, has_init,
        )?),
        crate::SPRINT_MODEL_ID => PricingSampling::Scm(crate::sampling::resolve_sprint_with_init(
            generation, has_init,
        )?),
        other => {
            return Err(Error::Unsupported(format!(
                "SANA pricing: unknown provider {other}"
            )))
        }
    };
    price_request_controls(req, generation.count, sampling)
}

/// Explicit payload-free captured-controls replay. No GenerationRequest or media carrier is
/// constructed; typed mode supplies init presence and the anchor supplies its serial count.
pub fn ios_pricing_with_token(
    req: &IosPricingRequest<'_>,
    token: &str,
    count: u32,
) -> Result<IosPricing> {
    req.validate_work_context()?;
    let has_init = matches!(req.mode, MemoryMode::ImageToImage);
    let sampling = match req.provider {
        crate::MODEL_ID => {
            let recipe = mlx_gen::sampler::flow_sampling::FlowSamplingRecipe::from_token(token)?;
            PricingSampling::Flow(
                mlx_gen::sampler::flow_sampling::ResolvedFlowControls::from_recipe(
                    &recipe,
                    crate::sampling::base_defaults(),
                    mlx_gen::TimestepConvention::Sigma,
                )?,
            )
        }
        crate::SPRINT_MODEL_ID => {
            let recipe = crate::sampling::ScmSamplingRecipe::from_token(token)?;
            PricingSampling::Scm(crate::sampling::ResolvedSprintControls::from_recipe(
                &recipe, has_init,
            )?)
        }
        other => {
            return Err(Error::Unsupported(format!(
                "SANA pricing: unknown provider {other}"
            )))
        }
    };
    price_request_controls(req, count, sampling)
}

fn resolve_source_plan(
    req: &IosPricingRequest<'_>,
    count: u32,
) -> Result<(SanaPricingFacts, SanaPricingPlan)> {
    if count == 0 {
        return Err(Error::Unsupported(
            "pricing requires a positive serial output count".into(),
        ));
    }
    let sprint = match req.provider {
        crate::MODEL_ID => false,
        crate::SPRINT_MODEL_ID => true,
        other => {
            return Err(Error::Unsupported(format!(
                "SANA pricing: unknown provider {other}"
            )))
        }
    };
    let img2img = match req.mode {
        MemoryMode::TextToImage if req.geometry.reference_count == 0 => false,
        MemoryMode::ImageToImage if req.geometry.reference_count == 1 => true,
        other => {
            return Err(Error::Unsupported(format!(
                "{}: no iOS SANA route for mode {} with {} references",
                req.provider,
                other.as_key(),
                req.geometry.reference_count
            )))
        }
    };
    let root = crate::model::load_components(req.spec, req.provider)?;
    crate::memory_strategy::validate_request_memory(req.provider, req.spec, req.execution)
        .map_err(|e| Error::Unsupported(e.to_string()))?;
    let facts = SanaPricingFacts::read(root, sprint)?;
    let sequential = req.spec.offload_policy == OffloadPolicy::Sequential;
    // The actual legacy None path has existing Sequential/env decode defaults;
    // Some(GenerationMemory::default()) is NOT equivalent to it. Payload-free
    // catalog replay carries its already resolved effective execution instead.
    let memory = req
        .generation
        .map_or(Some(*req.execution), |generation| generation.memory);
    let plan = SanaPricingPlan {
        sprint,
        sequential,
        img2img,
        width: req.geometry.width,
        height: req.geometry.height,
        count,
        decode_tiling: crate::pipeline::resolved_decode_plan(memory, sequential)
            .map(|p| (p.edge, p.overlap)),
        // On device the iOS quantum scope is always enabled.
        encode_tile: crate::pipeline::resolved_encode_tile_on(memory, true),
        attention: crate::pipeline::resolved_rung_plan(
            memory,
            facts.transformer.num_layers as usize,
        )?
        .attention,
    };
    Ok((facts, plan))
}

fn price_request_controls(
    req: &IosPricingRequest<'_>,
    count: u32,
    sampling: PricingSampling,
) -> Result<IosPricing> {
    let (facts, plan) = resolve_source_plan(req, count)?;
    let (sprint, img2img) = (plan.sprint, plan.img2img);
    let cfg_on = matches!(&sampling, PricingSampling::Flow(flow) if flow.effective_guidance().is_some_and(|g| g > 1.0));
    let elements = u64::from(plan.width / crate::pipeline::SPATIAL_SCALE)
        * u64::from(plan.height / crate::pipeline::SPATIAL_SCALE)
        * facts.transformer.in_channels as u64;
    // A finite source family is a POSSIBLE callback/update window, never an observed witness.
    // Tuple: family, enclosing callback owner bytes, prior-output owner bytes at the step guard.
    let (families, host_schedule, zero_scm) = if let PricingSampling::Scm(sampling) = &sampling {
        let transitions = sampling.effective_transitions();
        let latent = elements.saturating_mul(4);
        let mut families = Vec::new();
        if transitions == 1 {
            families.push(("scm/warmup-terminal".to_owned(), latent, 0));
        } else if transitions > 1 {
            families.push(("scm/warmup-renoise".to_owned(), latent, 0));
            if transitions > 2 {
                families.push(("scm/history-renoise".to_owned(), 2 * latent, latent));
            }
            families.push(("scm/history-terminal".to_owned(), 2 * latent, latent));
        }
        (families, sampling.host_schedule_bytes()?, transitions == 0)
    } else {
        let PricingSampling::Flow(sampling) = &sampling else {
            unreachable!("typed FLOW branch")
        };
        let callbacks = mlx_gen::sampler::solver_memory::solver_callback_windows(
            sampling.solver(),
            sampling.convention(),
            elements,
            4,
            4,
        )?;
        let cfg = sampling.effective_guidance().is_some_and(|g| g > 1.0);
        let mut families = Vec::new();
        for callback in callbacks {
            for uncond in [false, true].into_iter().filter(|u| !*u || cfg) {
                let prior_prediction = if uncond {
                    elements.saturating_mul(4)
                } else {
                    0
                };
                families.push((
                    format!(
                        "{}/{}/{}",
                        sampling.allocation_class(),
                        callback.family(),
                        if uncond {
                            "uncond-held-conditional"
                        } else {
                            "conditional"
                        }
                    ),
                    callback
                        .callback_outer_bytes
                        .saturating_add(prior_prediction),
                    callback.callback_outer_bytes,
                ));
            }
        }
        (families, sampling.host_schedule_bytes()?, false)
    };
    let mut pricing = IosPricing::default();
    for quantum in priced_quanta_with_cfg(&facts, &plan, cfg_on)? {
        let forward = matches!(
            quantum.label,
            labels::ENTRY | labels::BLOCK | labels::OUTPUT
        );
        let step = quantum.label == mlx_gen::sampler::SAMPLER_STEP_QUANTUM
            || quantum.label == labels::SCM_STEP;
        if forward || step {
            for (family, callback_held, step_held) in &families {
                let mut window = quantum.window.clone();
                window.class = Some(format!("{}/{family}", window.class.as_deref().unwrap()));
                let held = if step { *step_held } else { *callback_held };
                // ENTRY guard includes its input. At block/output points the forward input remains
                // in the enclosing callback beside the block hidden state priced by that guard.
                let forward_input = if forward && quantum.label != labels::ENTRY {
                    elements * 4
                } else {
                    0
                };
                window.held_bytes = window
                    .held_bytes
                    .saturating_add(held)
                    .saturating_add(forward_input)
                    .saturating_add(host_schedule);
                pricing.take_window(window);
            }
            if !sprint && step {
                let mut final_window = quantum.window.clone();
                final_window.class = Some(format!(
                    "{}/solve-complete",
                    final_window.class.as_deref().unwrap()
                ));
                final_window.held_bytes = final_window.held_bytes.saturating_add(host_schedule);
                pricing.take_window(final_window);
            }
        } else {
            let mut window = quantum.window;
            if quantum.label == labels::SCM_FINAL {
                window.class = Some(format!(
                    "{}/{}",
                    window.class.as_deref().unwrap(),
                    if zero_scm {
                        "zero-transition-final-divide"
                    } else {
                        "terminal-final-divide"
                    }
                ));
                window.held_bytes = window.held_bytes.saturating_add(host_schedule);
            }
            pricing.take_window(window);
        }
    }
    if !img2img {
        pricing.phase_not_run(IosPhase::Encode, "text mode has no reference encode");
    }
    Ok(pricing)
}

mod native_alloc;
mod native_certificate;
mod native_graph;
pub use native_certificate::ios_certified_bounder;

#[cfg(all(test, target_os = "macos"))]
mod additivity;
#[cfg(test)]
mod tests;
