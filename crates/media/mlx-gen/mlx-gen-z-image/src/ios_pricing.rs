//! Header-only, per-phase COUNTED pricing of one Z-Image Turbo iOS request (epic 16774): the
//! `mlx_gen::ios_pricing` interface for `z_image_turbo` (text, reference) and
//! `z_image_turbo_control` (control, control + reference).
//!
//! Every price here is the price the production guard of that quantum checks, computed by the SAME
//! functions the guard calls (`crate::quantum`, the modules' `*_bytes` / `quantum_scratch`, the
//! shared `mlx_gen::ios_quantum` kernels) over the request's shapes, and labelled with the exact
//! label the guard passes. The modules those functions read (weight shapes and dtypes: packed or
//! dense, bf16 or f32) are built from the snapshot's safetensors HEADERS as never-evaluated,
//! payload-free shape carriers (`zeros` of each header's shape and dtype): no tensor payload is read
//! and no forward graph is built. Device admission can run it.
//!
//! What runs on iOS is what is priced: the bounded VAE encode at
//! [`crate::vae::decoder::IOS_ENCODE_TILE_PX`] and the query-chunked VAE attention
//! (`VaeAttention::bounded_plan`) engage only under the iOS quantum scope, and a
//! deferred-materialization load streams every text-encoder layer, DiT block and ControlNet main
//! block through its own quantum (each block read inside it) whatever the window scope (an
//! all-covering window still materializes one block per quantum). The request's execution decides
//! staging (`stage_residency`), the decode tile edge and the DiT attention budget, read through the
//! pipeline's own resolvers.
//!
//! Each quantum is priced with what is resident while it runs: the components loaded under the
//! request's regime (staged: the text encoder only through conditioning, the DiT and ControlNet
//! dropped before the decode) plus the carried tensors live across it (the caption features, the
//! latents, the control context, the hints, and the quantum's own live input). A phase's price is its
//! worst quantum (`IosPricing::take`). The caption is priced at the full padded token window (the
//! prompt-dependent valid-token count is bounded by it).
//!
//! Not priced (a typed `Unsupported`): a load-time requantization (a snapshot stored at another tier
//! than requested), LoRA/LoKr adapters (their residuals depend on adapter payload structure), a PiD
//! decode overlay, and modes this provider does not serve.

use std::path::PathBuf;

use mlx_gen::attention::AttentionPlan;
use mlx_gen::gen_core::weightsmeta::{Dtype as HeaderDtype, SafetensorsTensorHeader};
use mlx_gen::gen_core::{GenerationMemory, LoadShape, LoadSpec, MemoryMode, WeightsSource};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosPricingRequest};
use mlx_gen::ios_quantum::{Backend, ConvCall};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, GenerationRequest, Result};
use mlx_rs::{Array, Dtype};

use crate::control_transformer::ZImageControlTransformer;
use crate::control_transformer_block::ZImageControlBlock;
use crate::quantum as q;
use crate::text_encoder::{EncoderLayer, TextEncoder, ZTextEncoderConfig};
use crate::transformer::{ZImageTransformer, ZImageTransformerConfig};
use crate::transformer_block::ZImageTransformerBlock;
use crate::vae::Vae;

/// The phase every Z-Image quantum label (and the shared labels a Z-Image request emits: the
/// curated sampler's step and an adapter load) runs in. `None` for a label Z-Image never emits.
pub fn ios_quantum_phase(label: &str) -> Option<IosPhase> {
    use IosPhase::*;
    Some(match label {
        q::TE_LOAD
        | q::DIT_LOAD
        | q::VAE_LOAD
        | q::CONTROL_LOAD
        | q::CONTROL_QUANTIZE
        | mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM => Load,
        q::TE_EMBED | q::TE_LAYER | q::CAPTION_FEATURES => Conditioning,
        q::INIT_IMAGE
        | q::VAE_ENCODE_IN
        | q::VAE_ENCODE_RESNET
        | q::VAE_ENCODE_ATTENTION
        | q::VAE_DOWNSAMPLE
        | q::VAE_ENCODE_OUT
        | q::VAE_ENCODE_TILE
        | q::VAE_ENCODE_TILED_STAGE
        | q::CONTROL_CONTEXT => Encode,
        q::LATENTS
        | q::DIT_PREPARE
        | q::DIT_EMBED
        | q::DIT_CAPTION
        | q::DIT_BLOCK
        | q::DIT_CONTEXT_BLOCK
        | q::DIT_UNIFY
        | q::DIT_OUTPUT
        | q::CONTROL_EMBED
        | q::CONTROL_BLOCK
        | mlx_gen::sampler::SAMPLER_STEP_QUANTUM => Denoise,
        q::VAE_DECODE_IN
        | q::VAE_DECODE_RESNET
        | q::VAE_DECODE_ATTENTION
        | q::VAE_UPSAMPLE
        | q::VAE_DECODE_OUT
        | q::VAE_DECODE_TILE
        | q::VAE_DECODE_TILED_STAGE
        | q::DECODED_TO_IMAGE => Decode,
        _ => return None,
    })
}

/// Price one iOS request header-only on the device backend (Metal).
pub fn ios_pricing(req: &IosPricingRequest<'_>) -> Result<IosPricing> {
    let facts = Facts::from_spec(req.provider, req.spec)?;
    let route = Route::from_request(req)?;
    Ok(fold(&Model::new(facts)?.quanta(&route, Backend::Metal)?))
}

/// One priced quantum: the guard's label, its counted scratch and what is resident while it runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Priced {
    pub(crate) label: &'static str,
    pub(crate) resident: u64,
    pub(crate) scratch: u64,
}

/// The per-phase pricing of `quanta`: each phase its worst quantum.
pub(crate) fn fold(quanta: &[Priced]) -> IosPricing {
    let mut pricing = IosPricing::default();
    for p in quanta {
        let phase = ios_quantum_phase(p.label).expect("every priced label maps to a phase");
        pricing.take(phase, p.resident, p.scratch, p.label);
    }
    pricing
}

/// The request modes this provider serves on iOS (`mlx-gen-ios-catalog::ios_admission::work_mode`:
/// a control request is `Other("control")`, with or without a reference init image).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Text,
    Reference,
    Control,
    ControlReference,
}

impl Mode {
    pub(crate) fn control(self) -> bool {
        matches!(self, Mode::Control | Mode::ControlReference)
    }
    pub(crate) fn img2img(self) -> bool {
        matches!(self, Mode::Reference | Mode::ControlReference)
    }
}

/// One request's route: mode, geometry, and the execution levers written onto it.
#[derive(Clone, Debug)]
pub(crate) struct Route {
    pub(crate) mode: Mode,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) count: u32,
    pub(crate) execution: GenerationMemory,
}

impl Route {
    fn from_request(req: &IosPricingRequest<'_>) -> Result<Self> {
        let control_route = req.provider == crate::model_control::MODEL_ID;
        let mode = match (req.mode, control_route) {
            (MemoryMode::TextToImage, false) => Mode::Text,
            (MemoryMode::ImageToImage, false) => Mode::Reference,
            (MemoryMode::Other(m), true) if m == "control" => {
                // The control image is one reference; an img2img init the second.
                if req.geometry.reference_count >= 2 {
                    Mode::ControlReference
                } else {
                    Mode::Control
                }
            }
            (mode, _) => {
                return Err(Error::Unsupported(format!(
                    "{}: iOS pricing does not serve mode {mode:?}",
                    req.provider
                )))
            }
        };
        Ok(Self {
            mode,
            width: req.geometry.width,
            height: req.geometry.height,
            count: req.geometry.batch.max(1),
            execution: *req.execution,
        })
    }

    /// The request the pipeline's resolvers read (geometry, count, execution).
    pub(crate) fn request(&self) -> GenerationRequest {
        GenerationRequest {
            width: self.width,
            height: self.height,
            count: self.count,
            memory: Some(self.execution),
            ..Default::default()
        }
    }
}

/// One component's safetensors files and their headers.
#[derive(Clone, Debug)]
pub(crate) struct Component {
    pub(crate) files: Vec<PathBuf>,
    pub(crate) headers: Vec<SafetensorsTensorHeader>,
}

impl Component {
    pub(crate) fn read(source: &WeightsSource) -> Result<Self> {
        let files = q::component_files(source)?;
        if files.is_empty() {
            return Err(Error::Msg(format!(
                "iOS pricing: no .safetensors under {source:?}"
            )));
        }
        let mut headers = Vec::new();
        for file in &files {
            headers.extend(
                mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(file)
                    .map_err(|e| Error::Msg(e.to_string()))?,
            );
        }
        Ok(Self { files, headers })
    }

    /// The header-only load floor over the tensors `keep` keeps — the guard's own
    /// [`mlx_gen::ios_quantum::load_floor_bytes`].
    pub(crate) fn floor(&self, keep: impl Fn(&str) -> bool) -> Result<u64> {
        mlx_gen::ios_quantum::load_floor_bytes(&self.files, keep)
    }

    /// The bytes the tensors `keep` keeps occupy once loaded: each tensor's stored bytes as the
    /// allocator holds them (page-rounded above one page, [`mlx_gen::ios_quantum::allocated_bytes`]).
    pub(crate) fn resident(&self, keep: impl Fn(&str) -> bool) -> u64 {
        self.headers
            .iter()
            .filter(|h| keep(&h.name))
            .map(|h| mlx_gen::ios_quantum::allocated_bytes(h.data_bytes))
            .sum()
    }

    /// [`Self::resident`] over the tensors under `prefix`.
    fn bytes_under(&self, prefix: &str) -> u64 {
        self.resident(|name| name.starts_with(prefix))
    }

    /// A lazy view of every tensor as a payload-free shape carrier (never evaluated: only shapes and
    /// dtypes are read by the prices).
    fn view(&self) -> Result<Weights> {
        let mut map = std::collections::HashMap::new();
        for h in &self.headers {
            let shape: Vec<i32> = h.shape.iter().map(|&d| d as i32).collect();
            map.insert(h.name.clone(), carrier(&shape, header_dtype(h.dtype)?)?);
        }
        Ok(Weights::from_map(map))
    }
}

fn header_dtype(d: HeaderDtype) -> Result<Dtype> {
    Ok(match d {
        HeaderDtype::BOOL => Dtype::Bool,
        HeaderDtype::U8 => Dtype::Uint8,
        HeaderDtype::I8 => Dtype::Int8,
        HeaderDtype::I16 => Dtype::Int16,
        HeaderDtype::U16 => Dtype::Uint16,
        HeaderDtype::F16 => Dtype::Float16,
        HeaderDtype::BF16 => Dtype::Bfloat16,
        HeaderDtype::I32 => Dtype::Int32,
        HeaderDtype::U32 => Dtype::Uint32,
        HeaderDtype::F32 => Dtype::Float32,
        HeaderDtype::I64 => Dtype::Int64,
        HeaderDtype::U64 => Dtype::Uint64,
        other => {
            return Err(Error::Unsupported(format!(
                "iOS pricing: no MLX dtype for a {other:?} tensor"
            )))
        }
    })
}

/// A never-evaluated array of `shape` and `dtype`: a shape carrier for the price functions.
fn carrier(shape: &[i32], dtype: Dtype) -> Result<Array> {
    Ok(mlx_rs::ops::zeros_dtype(shape, dtype)?)
}

/// The architecture and header facts one request's pricing reads.
#[derive(Clone, Debug)]
pub(crate) struct Facts {
    pub(crate) te_cfg: ZTextEncoderConfig,
    pub(crate) dit_cfg: ZImageTransformerConfig,
    pub(crate) te: Component,
    pub(crate) dit: Component,
    pub(crate) vae: Component,
    /// The Fun-Controlnet-Union checkpoint (the control route only).
    pub(crate) control: Option<Component>,
    /// The tokenizer's padded window (the text encoder's sequence length).
    pub(crate) window_tokens: u32,
    /// The caption's valid tokens (at most the window).
    pub(crate) caption_tokens: u32,
    /// The load's replayed quantization (`LoadSpec::quantize`): packs the dense ControlNet blocks.
    pub(crate) quant_bits: Option<i32>,
    /// A deferred-materialization snapshot load: every stack streams through per-block quanta.
    pub(crate) streamable: bool,
}

impl Facts {
    /// The facts of `spec`'s snapshot at the production geometry.
    pub(crate) fn from_spec(provider: &str, spec: &LoadSpec) -> Result<Self> {
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(Error::Unsupported(format!(
                "{provider}: iOS pricing reads a snapshot directory"
            )));
        };
        if !spec.adapters.is_empty() {
            return Err(Error::Unsupported(format!(
                "{provider}: iOS pricing does not price LoRA/LoKr adapter residuals"
            )));
        }
        if spec.pid.is_some() {
            return Err(Error::Unsupported(format!(
                "{provider}: a PiD decode overlay is not an iOS load"
            )));
        }
        let stored = mlx_gen::quant::packed_quant_bits(root, "transformer")?;
        let requested = spec.quantize.map(|q| q.bits());
        if stored != requested {
            return Err(Error::Unsupported(format!(
                "{provider}: the snapshot is stored at {stored:?} bits but the load requests \
                 {requested:?}; a load-time requantization is not priced"
            )));
        }
        let control = match (provider == crate::model_control::MODEL_ID, &spec.control) {
            (true, Some(control)) => Some(Component::read(control)?),
            (true, None) => {
                return Err(Error::Unsupported(format!(
                    "{provider}: needs the Fun-Controlnet-Union checkpoint"
                )))
            }
            (false, _) => None,
        };
        let te_source = spec
            .text_encoder
            .clone()
            .unwrap_or_else(|| WeightsSource::Dir(root.join("text_encoder")));
        Ok(Self {
            te_cfg: ZTextEncoderConfig::z_image(),
            dit_cfg: ZImageTransformerConfig::turbo(),
            te: Component::read(&te_source)?,
            dit: Component::read(&WeightsSource::Dir(root.join("transformer")))?,
            vae: Component::read(&WeightsSource::Dir(root.join("vae")))?,
            control,
            window_tokens: crate::loader::MAX_LENGTH as u32,
            caption_tokens: crate::loader::MAX_LENGTH as u32,
            quant_bits: requested,
            streamable: spec.load_shape == LoadShape::DeferredMaterialization,
        })
    }
}

/// The transformer the route runs: the base DiT, or the ControlNet composition over it.
enum Dit {
    Base(Box<ZImageTransformer>),
    Control(Box<ZImageControlTransformer>),
}

impl Dit {
    fn base(&self) -> &ZImageTransformer {
        match self {
            Dit::Base(t) => t,
            Dit::Control(c) => &c.base,
        }
    }
}

/// The ControlNet facts: the composition, its streamed main blocks (with their per-block read) and
/// its resident bytes before and after the load's quantization.
struct ControlFacts {
    /// Main control block `i` and what its quantum reads when streamed.
    main: Vec<(ZImageControlBlock, u64)>,
    /// The load floor (what the load reads up front), and what it holds once read.
    floor: u64,
    loaded: u64,
    /// Each block's (control_layers 0..15, then the refiner) stored bytes.
    block_stored: Vec<u64>,
    /// Each block's resident bytes once packed at the load's bits (= stored when not packed).
    block_packed: Vec<u64>,
    /// Each block's load-time quantization price (`ZImageControlBlock::quantize_bytes`).
    block_quantize: Vec<u64>,
    /// Whether each block is resident after the load reads (the refiner, or an unstreamed main block).
    block_loaded: Vec<bool>,
    /// The non-block tensors (the patch embedder).
    embedder: u64,
}

/// The modules and byte totals one load's pricing reads, built once from its [`Facts`].
pub(crate) struct Model {
    facts: Facts,
    te: TextEncoder,
    /// Each encoder layer and what its quantum reads when streamed.
    te_layers: Vec<(EncoderLayer, u64)>,
    dit: Dit,
    /// Each unified block and what its quantum reads when streamed.
    dit_blocks: Vec<(ZImageTransformerBlock, u64)>,
    vae: Vae,
    control: Option<ControlFacts>,
    te_floor: u64,
    dit_floor: u64,
    vae_floor: u64,
    /// What each loaded component holds resident (its kept tensors as allocated; the DiT also its
    /// RoPE tables).
    te_res: u64,
    dit_res: u64,
    vae_res: u64,
}

impl Model {
    pub(crate) fn new(facts: Facts) -> Result<Self> {
        let streamable = facts.streamable;
        let te_cfg = facts.te_cfg;
        let te = TextEncoder::from_weights(&facts.te.view()?, "model", &te_cfg)?;
        let mut te_view = facts.te.view()?;
        let te_layers = (0..te_cfg.n_layers)
            .map(|i| {
                crate::text_encoder::stream::layer_counted(
                    &mut te_view,
                    &format!("model.layers.{i}"),
                    &te_cfg,
                    None,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let mut dit_view = facts.dit.view()?;
        crate::loader::remap_transformer_keys(&mut dit_view);
        let base = ZImageTransformer::from_weights(&dit_view, "", facts.dit_cfg.clone())?;
        let mut block_view = facts.dit.view()?;
        let dit_blocks = (0..facts.dit_cfg.n_layers)
            .map(|i| {
                crate::block_stream::base_block_counted(
                    &mut block_view,
                    &format!("layers.{i}"),
                    facts.dit_cfg.block_cfg(),
                    facts.quant_bits,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let mut vae_view = facts.vae.view()?;
        let vae = crate::loader::build_vae(&mut vae_view)?;

        let (dit, control) = match &facts.control {
            None => (Dit::Base(Box::new(base)), None),
            Some(component) => {
                let composed =
                    ZImageControlTransformer::from_weights(base, &component.view()?, "")?;
                let mut view = component.view()?;
                let main = (0..crate::control_transformer::CONTROL_LAYERS_PLACES.len())
                    .map(|i| {
                        crate::block_stream::control_block_counted(
                            &mut view,
                            &format!("control_layers.{i}"),
                            facts.dit_cfg.block_cfg(),
                            i == 0,
                            facts.quant_bits,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let blocks: Vec<(&ZImageControlBlock, String, bool)> = composed
                    .control_layers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (b, format!("control_layers.{i}."), !streamable))
                    .chain(
                        composed
                            .control_noise_refiner()
                            .iter()
                            .enumerate()
                            .map(|(i, b)| (b, format!("control_noise_refiner.{i}."), true)),
                    )
                    .collect();
                let block_stored: Vec<u64> = blocks
                    .iter()
                    .map(|(_, p, _)| component.bytes_under(p))
                    .collect();
                let block_packed = blocks
                    .iter()
                    .zip(&block_stored)
                    .map(|((b, _, _), &stored)| packed_resident(b, stored, facts.quant_bits))
                    .collect();
                let block_quantize = blocks
                    .iter()
                    .map(|(b, _, _)| facts.quant_bits.map_or(0, |bits| b.quantize_bytes(bits)))
                    .collect();
                let block_loaded = blocks.iter().map(|(_, _, loaded)| *loaded).collect();
                let keep = |n: &str| crate::loader::control_load_keeps(streamable, n);
                let control = ControlFacts {
                    main,
                    floor: component.floor(keep)?,
                    loaded: component.resident(keep),
                    embedder: component.bytes_under("control_all_x_embedder."),
                    block_stored,
                    block_packed,
                    block_quantize,
                    block_loaded,
                };
                (Dit::Control(Box::new(composed)), Some(control))
            }
        };
        let dit_res = facts
            .dit
            .resident(|n| crate::loader::dit_load_keeps(streamable, n))
            + dit.base().rope.resident_bytes();
        Ok(Self {
            te_res: facts
                .te
                .resident(|n| crate::loader::te_load_keeps(streamable, n)),
            dit_res,
            vae_res: facts.vae.resident(|_| true),
            te_floor: facts
                .te
                .floor(|n| crate::loader::te_load_keeps(streamable, n))?,
            dit_floor: facts
                .dit
                .floor(|n| crate::loader::dit_load_keeps(streamable, n))?,
            vae_floor: facts.vae.floor(|_| true)?,
            facts,
            te,
            te_layers,
            dit,
            dit_blocks,
            vae,
            control,
        })
    }

    /// What the DiT's load leaves resident.
    #[cfg(test)]
    pub(crate) fn dit_resident(&self) -> u64 {
        self.dit_res
    }

    /// The ControlNet resident once loaded (and packed at the load's bits): the patch embedder plus
    /// every block the load leaves resident (the refiner; the main blocks unless a deferred load
    /// streams them — `ZImageControlTransformer::quantize` packs only resident blocks).
    pub(crate) fn control_resident(&self) -> u64 {
        self.control.as_ref().map_or(0, |c| {
            c.embedder
                + (0..c.block_stored.len())
                    .filter(|&i| c.block_loaded[i])
                    .map(|i| c.block_packed[i])
                    .sum::<u64>()
        })
    }

    /// Every quantum `route` runs, in order, priced on `backend`.
    pub(crate) fn quanta(&self, route: &Route, backend: Backend) -> Result<Vec<Priced>> {
        let f = &self.facts;
        if route.mode.control() != self.control.is_some() {
            return Err(Error::Unsupported(format!(
                "iOS pricing: mode {:?} does not match the loaded route",
                route.mode
            )));
        }
        let req = route.request();
        let staged = route.execution.stage_residency;
        let streamable = f.streamable;
        // The pipeline's own validation of the rung-4 window against the load shape.
        crate::pipeline::resolve_block_window(&req, streamable, "z_image_turbo")?;

        let (te_res, dit_res) = (self.te_res, self.dit_res);
        let ctrl_res = self.control_resident();
        let heavy = dit_res + self.vae_res + ctrl_res;
        let te_cfg = &f.te_cfg;
        let s = f.window_tokens as i32;
        let n_valid = f.caption_tokens.min(f.window_tokens) as i32;
        let hidden = te_cfg.hidden_size;
        let mode = route.mode;
        let cap_dtype = if mode.img2img() {
            Dtype::Float32
        } else {
            Dtype::Bfloat16
        };
        let cap_bytes = q::ab(
            mlx_gen::ios_quantum::dtype_bytes(cap_dtype),
            &[n_valid.max(0) as u64, hidden as u64],
        );
        let f32b = |dims: &[i32]| q::ab(4, &dims.iter().map(|&d| q::u(d)).collect::<Vec<_>>());

        let mut out = Vec::new();
        let mut push = |label: &'static str, resident: u64, scratch: u64| {
            out.push(Priced {
                label,
                resident,
                scratch,
            })
        };

        // ---- Load: each component floor, with whatever is resident when it loads. ----
        push(q::TE_LOAD, 0, self.te_floor);
        let at_heavy = if staged { cap_bytes } else { te_res };
        push(q::DIT_LOAD, at_heavy, self.dit_floor);
        let mut loaded = at_heavy + dit_res;
        if let Some(c) = &self.control {
            push(q::CONTROL_LOAD, loaded, c.floor);
            loaded += c.loaded;
        }
        push(q::VAE_LOAD, loaded, self.vae_floor);
        loaded += self.vae_res;
        if let (Some(c), Some(_)) = (&self.control, f.quant_bits) {
            // `quantize_quantum` over each resident block (the main blocks unless streamed, then
            // the refiner): dense while it packs, packed after.
            for i in (0..c.block_stored.len()).filter(|&i| c.block_loaded[i]) {
                push(q::CONTROL_QUANTIZE, loaded, c.block_quantize[i]);
                loaded = loaded - c.block_stored[i] + c.block_packed[i];
            }
        }

        // ---- Conditioning ----
        let cond_res = te_res + if staged { 0 } else { heavy };
        push(q::TE_EMBED, cond_res, self.te.embed_bytes(1, s));
        // The layer stack's carry: prev + second-to-last hidden states, cos / sin, the mask.
        let head_dim = te_cfg.head_dim;
        let te_carry =
            2 * f32b(&[1, s, hidden]) + 2 * f32b(&[1, s, head_dim]) + f32b(&[1, 1, s, s]);
        for (layer, read) in &self.te_layers {
            let scratch = layer
                .quantum_scratch(1, s, 4, backend)
                .saturating_add(if streamable { *read } else { 0 });
            push(q::TE_LAYER, cond_res + te_carry, scratch);
        }
        push(
            q::CAPTION_FEATURES,
            cond_res + f32b(&[1, s, hidden]),
            crate::pipeline::caption_features_bytes(n_valid, hidden),
        );
        if !mode.img2img() {
            push(
                q::CAPTION_FEATURES,
                cond_res + f32b(&[n_valid, hidden]),
                crate::pipeline::caption_cast_bytes((n_valid as u64) * hidden as u64),
            );
        }

        // ---- Encode (inside the denoise phase's components) ----
        let (w, h) = (route.width as i32, route.height as i32);
        let spatial = crate::pipeline::SPATIAL_SCALE as i32;
        let (lh, lw) = (h / spatial, w / spatial);
        let lc = crate::pipeline::LATENT_CHANNELS;
        let latent = f32b(&[lc, 1, lh, lw]);
        let context = f32b(&[2 * lc + 1, 1, lh, lw]);
        let render_res = heavy + if staged { 0 } else { te_res } + cap_bytes;
        // The control image encodes first; a reference init second (the context then resident).
        let mut encodes = Vec::new();
        if mode.control() {
            encodes.push((true, 0));
        }
        if mode.img2img() {
            encodes.push((false, if mode.control() { context } else { 0 }));
        }
        for (is_control, carried) in encodes {
            let base = render_res + carried;
            push(
                q::INIT_IMAGE,
                base,
                crate::pipeline::init_image_bytes(route.width, route.height),
            );
            for (label, live, scratch) in self.encode_quanta(w, h, backend)? {
                push(label, base + f32b(&[1, 3, h, w]) + live, scratch);
            }
            if is_control {
                push(
                    q::CONTROL_CONTEXT,
                    base + latent,
                    crate::pipeline::control_context_bytes(lc, 1, lh, lw),
                );
            }
        }

        // ---- Denoise ----
        let _glue = crate::CompileGlueGuard::enable();
        let dit = self.dit.base();
        let cfg = &f.dit_cfg;
        let dim = cfg.dim;
        let budget = crate::pipeline::attention_budget(&req);
        let plan = AttentionPlan::budgeted(budget);
        let img_tokens = (1 / cfg.f_patch_size) * (lh / cfg.patch_size) * (lw / cfg.patch_size);
        let (_, _, cap_pad) = crate::transformer::caption_pos_keep(n_valid);
        let c = n_valid + cap_pad;
        let (_, _, img_pad) = crate::transformer::image_pos_keep(
            1 / cfg.f_patch_size,
            lh / cfg.patch_size,
            lw / cfg.patch_size,
            c,
        );
        let n = img_tokens + img_pad;
        let l = n + c;
        let rope_dim: i32 = cfg.axes_dims.iter().sum();
        let cap = carrier(&[n_valid, cfg.cap_feat_dim], cap_dtype)?;
        let cap_tokens_dtype = if cap_pad > 0 {
            Dtype::Float32
        } else {
            cap_dtype
        };
        let cap_tokens = carrier(&[c, cfg.cap_feat_dim], cap_tokens_dtype)?;
        let x_freqs = carrier(&[n, rope_dim], Dtype::Float32)?;
        let cap_freqs = carrier(&[c, rope_dim], Dtype::Float32)?;
        let unified_freqs = carrier(&[l, rope_dim], Dtype::Float32)?;
        let stream = |rows: i32| carrier(&[1, rows, dim], Dtype::Float32);
        let (x_stream, cap_stream, unified) = (stream(n)?, stream(c)?, stream(l)?);
        let row = |rows: i32| f32b(&[1, rows, dim]);
        let prep = mlx_gen::ios_quantum::bytes(&cap_tokens)
            + f32b(&[n, rope_dim])
            + f32b(&[c, rope_dim])
            + f32b(&[l, rope_dim]);
        let mut denoise_res = render_res
            + prep
            + latent
            + latent * u64::from(route.count - 1)
            + if mode.img2img() { latent } else { 0 };
        if mode.control() {
            // The control context and its cached embedding.
            denoise_res += context + row(n);
        }
        let elements = (lc as u64) * (lh as u64) * (lw as u64);
        push(
            q::LATENTS,
            denoise_res,
            q::latents_bytes(elements, mode.img2img()),
        );
        push(
            q::DIT_PREPARE,
            denoise_res,
            dit.prepare_bytes((lc, 1, lh, lw), &cap),
        );
        if let Dit::Control(control) = &self.dit {
            let cc = carrier(&[2 * lc + 1, 1, lh, lw], Dtype::Float32)?;
            push(
                q::CONTROL_EMBED,
                denoise_res,
                control.control_embed_bytes(&cc, img_pad),
            );
        }
        // Step 0 of a text-to-image denoise runs on the bf16 noise; every later step (and an
        // img2img blend) on f32 latents.
        let mut latent_dtypes = vec![Dtype::Float32];
        if !mode.img2img() {
            latent_dtypes.push(Dtype::Bfloat16);
        }
        for dtype in latent_dtypes {
            let x = carrier(&[lc, 1, lh, lw], dtype)?;
            push(
                q::DIT_EMBED,
                denoise_res,
                dit.embed_bytes(&x, img_pad, backend),
            );
        }
        let x_emb_live = row(n);
        // The control refiner: its 2 hints and threaded state live through the base refiner.
        let mut refiner_live = 0;
        if let Dit::Control(control) = &self.dit {
            for (k, block) in control.control_noise_refiner().iter().enumerate() {
                let scratch = block.quantum_scratch(&x_stream, &x_stream, &x_freqs, plan, backend);
                push(
                    q::CONTROL_BLOCK,
                    denoise_res + x_emb_live + row(n) + (k as u64) * row(n),
                    scratch,
                );
            }
            refiner_live = (control.control_noise_refiner().len() as u64 + 1) * row(n);
        }
        for (i, block) in dit.noise_refiner.iter().enumerate() {
            let hinted = self.control.is_some()
                && crate::control_transformer::hint_index(
                    &crate::control_transformer::CONTROL_REFINER_PLACES,
                    i,
                )
                .is_some();
            let scratch = block
                .quantum_scratch(&x_stream, &x_freqs, plan, backend)
                .saturating_add(if hinted {
                    crate::control_transformer::add_hint_bytes(&x_stream)
                } else {
                    0
                });
            push(
                q::DIT_BLOCK,
                denoise_res + x_emb_live + refiner_live,
                scratch,
            );
        }
        let streams_res = denoise_res + x_emb_live + refiner_live;
        push(
            q::DIT_CAPTION,
            streams_res,
            dit.caption_bytes(&cap_tokens, backend),
        );
        for block in &dit.context_refiner {
            push(
                q::DIT_CONTEXT_BLOCK,
                streams_res + row(c),
                block.quantum_scratch(&cap_stream, &cap_freqs, plan, backend),
            );
        }
        let unify = crate::transformer::unify_bytes(&x_stream, &cap_stream);
        push(q::DIT_UNIFY, streams_res + row(c), unify);
        let mut stack_res = streams_res + row(c) + row(l);
        if let Some(cf) = &self.control {
            // The control stream's own unify, then the 15 main blocks (their hints accumulate).
            push(q::DIT_UNIFY, stack_res, unify);
            let resident_blocks = match &self.dit {
                Dit::Control(control) => &control.control_layers,
                Dit::Base(_) => unreachable!(),
            };
            for (k, (block, read)) in cf.main.iter().enumerate() {
                let (block, read) = if streamable {
                    (block, *read)
                } else {
                    (&resident_blocks[k], 0)
                };
                let scratch = block
                    .quantum_scratch(&unified, &unified, &unified_freqs, plan, backend)
                    .saturating_add(read);
                push(
                    q::CONTROL_BLOCK,
                    stack_res + row(l) + (k as u64) * row(l),
                    scratch,
                );
            }
            stack_res += (cf.main.len() as u64) * row(l);
        }
        for (i, (streamed, read)) in self.dit_blocks.iter().enumerate() {
            let (block, read) = if streamable {
                (streamed, *read)
            } else {
                (&dit.layers[i], 0)
            };
            let hinted = self.control.is_some()
                && crate::control_transformer::hint_index(
                    &crate::control_transformer::CONTROL_LAYERS_PLACES,
                    i,
                )
                .is_some();
            let scratch = block
                .quantum_scratch(&unified, &unified_freqs, plan, backend)
                .saturating_add(read)
                .saturating_add(if hinted {
                    crate::control_transformer::add_hint_bytes(&unified)
                } else {
                    0
                });
            push(q::DIT_BLOCK, stack_res, scratch);
        }
        push(
            q::DIT_OUTPUT,
            stack_res,
            dit.output_bytes(&unified, backend),
        );
        push(
            mlx_gen::sampler::SAMPLER_STEP_QUANTUM,
            denoise_res + latent,
            mlx_gen::ios_quantum::sampler_step_bytes(elements),
        );
        drop(_glue);

        // ---- Decode: staged, only the VAE stays (the DiT and ControlNet shed). ----
        let decode_res = self.vae_res
            + if staged {
                0
            } else {
                te_res + dit_res + ctrl_res
            }
            + latent * u64::from(route.count);
        let tiling = crate::pipeline::decode_tiling(&req, staged);
        for (label, live, scratch) in self.decode_quanta(w, h, tiling.as_ref(), backend)? {
            push(label, decode_res + live, scratch);
        }
        let pixels = 3 * (h as u64) * (w as u64);
        push(
            q::DECODED_TO_IMAGE,
            decode_res + q::ab(4, &[pixels]),
            q::zimage_pixel_conversion_bytes(pixels, 4),
        );
        Ok(out)
    }

    /// The VAE encode of one `w x h` image as the device runs it ([`Vae::encode`]: the bounded,
    /// layer-wise tiled encode above [`crate::vae::decoder::IOS_ENCODE_TILE_PX`], else whole), as
    /// `(label, live input beside the image, scratch)`.
    fn encode_quanta(&self, w: i32, h: i32, b: Backend) -> Result<Vec<(&'static str, u64, u64)>> {
        let encoder = self
            .vae
            .encoder()
            .ok_or_else(|| Error::Msg("iOS pricing: the VAE has no encoder".into()))?;
        let (conv_in, downs, mid, conv_out) = encoder.parts();
        let labels = &q::ENCODE_LABELS;
        let tile_px = crate::vae::decoder::IOS_ENCODE_TILE_PX;
        let tiled = h > tile_px || w > tile_px;
        let image = carrier(&[1, 3, h, w], Dtype::Float32)?;
        let mut out = Vec::new();
        let (mut ch, mut cw) = (h, w);
        let mut c = mlx_gen::ios_quantum::conv_channels(conv_in.weight()).1;
        if tiled {
            let x = [1, h, w, 3];
            stage_and_tiles(
                &mut out,
                labels,
                0,
                &x,
                conv_in.weight(),
                tile_px.max(3),
                false,
                b,
            );
            let mut scale = 1;
            for down in downs {
                let edge = ((tile_px + scale - 1) / scale).max(3);
                let (resnets, downsampler) = down.parts();
                let next = if downsampler.is_some() {
                    scale * 2
                } else {
                    scale
                };
                for r in resnets {
                    let live = q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                    let (c1, c2) = r.conv_weights();
                    let x = [1, ch, cw, c];
                    stage_and_tiles(&mut out, labels, live, &x, c1, edge, true, b);
                    let cout = mlx_gen::ios_quantum::conv_channels(c1).1;
                    let stage = mlx_gen::ios_quantum::tiled_stage_bytes(&x, c, cout, 4, true);
                    let t = tile_extent(ch, cw, edge);
                    out.push((
                        labels.tile,
                        live + stage,
                        mlx_gen::ios_quantum::tile_bytes(&[1, t.0, t.1, cout], 4, c2, false, b),
                    ));
                    c = mlx_gen::ios_quantum::conv_channels(c2).1;
                }
                if let Some(ds) = downsampler {
                    let core = ((tile_px + next - 1) / next).max(1);
                    let live = q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                    let (oh, ow) = ((ch - 2) / 2 + 1, (cw - 2) / 2 + 1);
                    let cout = ds.weight().shape()[0];
                    let stage = mlx_gen::ios_quantum::downsample_stage_bytes(&[1, oh, ow, cout], 4);
                    out.push((labels.stage, live, stage));
                    let crop = [1, 2 * core.min(oh) + 1, 2 * core.min(ow) + 1, c];
                    out.push((
                        labels.tile,
                        live + stage,
                        mlx_gen::ios_quantum::downsample_tile_bytes(&crop, 4, ds.weight(), b),
                    ));
                    (ch, cw, c) = (oh, ow, cout);
                }
                scale = next;
            }
        } else {
            let call = ConvCall::new(&[1, h, w, 3], 4, conv_in.weight(), 1, 1);
            out.push((
                q::VAE_ENCODE_IN,
                0,
                mlx_gen::ios_quantum::single_conv_bytes(&call, 0, 0, b),
            ));
            for down in downs {
                let (resnets, downsampler) = down.parts();
                for r in resnets {
                    let x = carrier(&[1, c, ch, cw], Dtype::Float32)?;
                    out.push((
                        labels.resnet,
                        mlx_gen::ios_quantum::bytes(&x),
                        r.quantum_scratch(&x, b),
                    ));
                    c = mlx_gen::ios_quantum::conv_channels(r.conv_weights().1).1;
                }
                if let Some(ds) = downsampler {
                    let call = ConvCall::new(&[1, ch + 1, cw + 1, c], 4, ds.weight(), 2, 0);
                    out.push((
                        q::VAE_DOWNSAMPLE,
                        q::ab(4, &[1, c as u64, ch as u64, cw as u64]),
                        mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, b),
                    ));
                    (ch, cw) = ((ch - 2) / 2 + 1, (cw - 2) / 2 + 1);
                    c = ds.weight().shape()[0];
                }
            }
        }
        mid_quanta(&mut out, labels, mid, c, ch, cw, b)?;
        let call = ConvCall::new(&[1, ch, cw, c], 4, conv_out.weight(), 1, 1);
        out.push((
            q::VAE_ENCODE_OUT,
            q::ab(4, &[1, c as u64, ch as u64, cw as u64]),
            mlx_gen::ios_quantum::norm_conv_bytes(&call, b)
                .saturating_add(Vae::latent_mean_bytes(&image)),
        ));
        Ok(out)
    }

    /// The decode of one `w x h` image (`decode_batch`: [`Vae::decode_tiled`] under `tiling`, else
    /// [`Vae::decode`]), as `(label, live input, scratch)`.
    fn decode_quanta(
        &self,
        w: i32,
        h: i32,
        tiling: Option<&mlx_gen::tiling::TilingConfig>,
        b: Backend,
    ) -> Result<Vec<(&'static str, u64, u64)>> {
        let decoder = self.vae.decoder();
        let (conv_in, mid, ups, conv_out) = decoder.parts();
        let labels = &q::DECODE_LABELS;
        let spatial = crate::pipeline::SPATIAL_SCALE as i32;
        let (lh, lw) = (h / spatial, w / spatial);
        let lc = crate::pipeline::LATENT_CHANNELS;
        let mut out = Vec::new();
        let call = ConvCall::new(&[1, lh, lw, lc], 4, conv_in.weight(), 1, 1);
        out.push((
            q::VAE_DECODE_IN,
            0,
            mlx_gen::ios_quantum::single_conv_bytes(&call, 0, 0, b)
                .saturating_add(Vae::denorm_bytes((lc * lh * lw) as u64)),
        ));
        let mut c = mlx_gen::ios_quantum::conv_channels(conv_in.weight()).1;
        mid_quanta(&mut out, labels, mid, c, lh, lw, b)?;
        // The head (the mid block's output) stays live through the tail.
        let head = q::ab(4, &[1, c as u64, lh as u64, lw as u64]);
        let tile =
            tiling.filter(|t| t.needs_tiling(mlx_gen::tiling::VaeTiling::QWEN_IMAGE, 1, lh, lw));
        let (mut ch, mut cw) = (lh, lw);
        match tile.and_then(|t| t.spatial.as_ref()).map(|s| s.tile_px) {
            Some(tile_px) => {
                let edge_at = |remaining: i32| ((tile_px + remaining - 1) / remaining).max(3);
                let mut remaining = ups
                    .iter()
                    .filter(|u| u.upsamples())
                    .fold(1_i32, |s, _| s.saturating_mul(2));
                for up in ups {
                    let edge = edge_at(remaining);
                    let next = if up.upsamples() {
                        (remaining / 2).max(1)
                    } else {
                        remaining
                    };
                    let (resnets, upsampler) = up.parts();
                    for r in resnets {
                        let live = head + q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                        let (c1, c2) = r.conv_weights();
                        let x = [1, ch, cw, c];
                        stage_and_tiles(&mut out, labels, live, &x, c1, edge, true, b);
                        let cout = mlx_gen::ios_quantum::conv_channels(c1).1;
                        let stage = mlx_gen::ios_quantum::tiled_stage_bytes(&x, c, cout, 4, true);
                        let t = tile_extent(ch, cw, edge);
                        out.push((
                            labels.tile,
                            live + stage,
                            mlx_gen::ios_quantum::tile_bytes(&[1, t.0, t.1, cout], 4, c2, false, b),
                        ));
                        c = mlx_gen::ios_quantum::conv_channels(c2).1;
                    }
                    if let Some(u) = upsampler {
                        let live = head + q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                        let x = [1, 2 * ch, 2 * cw, c];
                        stage_and_tiles(
                            &mut out,
                            labels,
                            live,
                            &x,
                            u.weight(),
                            edge_at(next),
                            false,
                            b,
                        );
                        (ch, cw) = (2 * ch, 2 * cw);
                        c = mlx_gen::ios_quantum::conv_channels(u.weight()).1;
                    }
                    remaining = next;
                }
                let live = head + q::ab(4, &[1, c as u64, ch as u64, cw as u64]);
                stage_and_tiles(
                    &mut out,
                    labels,
                    live,
                    &[1, ch, cw, c],
                    conv_out.weight(),
                    tile_px.max(3),
                    false,
                    b,
                );
            }
            None => {
                for up in ups {
                    let (resnets, upsampler) = up.parts();
                    for r in resnets {
                        let x = carrier(&[1, c, ch, cw], Dtype::Float32)?;
                        out.push((
                            labels.resnet,
                            head + mlx_gen::ios_quantum::bytes(&x),
                            r.quantum_scratch(&x, b),
                        ));
                        c = mlx_gen::ios_quantum::conv_channels(r.conv_weights().1).1;
                    }
                    if let Some(u) = upsampler {
                        let call = ConvCall::new(&[1, 2 * ch, 2 * cw, c], 4, u.weight(), 1, 1);
                        out.push((
                            q::VAE_UPSAMPLE,
                            head + q::ab(4, &[1, c as u64, ch as u64, cw as u64]),
                            mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, b),
                        ));
                        (ch, cw) = (2 * ch, 2 * cw);
                        c = mlx_gen::ios_quantum::conv_channels(u.weight()).1;
                    }
                }
                let call = ConvCall::new(&[1, ch, cw, c], 4, conv_out.weight(), 1, 1);
                out.push((
                    q::VAE_DECODE_OUT,
                    head + q::ab(4, &[1, c as u64, ch as u64, cw as u64]),
                    mlx_gen::ios_quantum::norm_conv_bytes(&call, b),
                ));
            }
        }
        Ok(out)
    }
}

/// The resident bytes of a control block once `bits` packs it (`stored` when not packed): each dense
/// projection's weight replaced by its packed weight, scales and biases ([`q::packed_bytes`]).
fn packed_resident(block: &ZImageControlBlock, stored: u64, bits: Option<i32>) -> u64 {
    let Some(bits) = bits else {
        return stored;
    };
    block
        .projections()
        .into_iter()
        .fold(stored, |total, lin| match lin.dense_weight() {
            Some((weight, _)) => total
                .saturating_sub(mlx_gen::ios_quantum::allocated_bytes(
                    mlx_gen::ios_quantum::bytes(weight),
                ))
                .saturating_add(q::packed_bytes(lin, bits)),
            None => total,
        })
}

/// The mid block's quanta (resnet, attention, resnet) over a `[1, c, h, w]` f32 activation.
fn mid_quanta(
    out: &mut Vec<(&'static str, u64, u64)>,
    labels: &'static q::VaeLabels,
    mid: &crate::vae::mid_block::UNetMidBlock,
    c: i32,
    h: i32,
    w: i32,
    b: Backend,
) -> Result<()> {
    let (r0, _attention, r1) = mid.parts();
    let x = carrier(&[1, c, h, w], Dtype::Float32)?;
    let live = mlx_gen::ios_quantum::bytes(&x);
    out.push((labels.resnet, live, r0.quantum_scratch(&x, b)));
    let n = h * w;
    let rows = crate::vae::VaeAttention::bounded_plan()
        .budget
        .query_block(1, 1, n, n);
    out.push((
        labels.attention,
        live,
        mlx_gen::ios_quantum::vae_attention_bytes(1, n, c, rows, 4, b),
    ));
    out.push((labels.resnet, live, r1.quantum_scratch(&x, b)));
    Ok(())
}

/// One layer-wise tiled stage over `x` (NHWC, f32) run by `weight`'s 3x3 convolution — its stage
/// check ([`q::before_tiled_stage`]) and its largest tile check ([`q::before_tile`]) at `edge`.
#[allow(clippy::too_many_arguments)]
fn stage_and_tiles(
    out: &mut Vec<(&'static str, u64, u64)>,
    labels: &'static q::VaeLabels,
    live: u64,
    x: &[i32; 4],
    weight: &Array,
    edge: i32,
    resnet: bool,
    b: Backend,
) {
    let (cin, cout) = mlx_gen::ios_quantum::conv_channels(weight);
    let stage = mlx_gen::ios_quantum::tiled_stage_bytes(x, cin, cout, 4, resnet);
    out.push((labels.stage, live, stage));
    let t = tile_extent(x[1], x[2], edge);
    out.push((
        labels.tile,
        live + stage,
        mlx_gen::ios_quantum::tile_bytes(&[1, t.0, t.1, x[3]], 4, weight, false, b),
    ));
}

/// The largest halo-expanded tile `mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc` cuts from an
/// `h x w` activation at `max_tile_edge`: the whole activation when it fits one tile; otherwise
/// cores of `max_tile_edge - 2` with a one-pixel halo on every interior side, so the largest extent
/// along each axis (every row/column pair is a tile).
pub(crate) fn tile_extent(h: i32, w: i32, max_tile_edge: i32) -> (i32, i32) {
    if h <= max_tile_edge && w <= max_tile_edge {
        return (h, w);
    }
    let axis = |n: i32| {
        let core = max_tile_edge - 2;
        let (mut start, mut largest) = (0, 0);
        while start < n {
            let end = (start + core).min(n);
            largest = largest.max((end + 1).min(n) - (start - 1).max(0));
            start = end;
        }
        largest
    };
    (axis(h), axis(w))
}
