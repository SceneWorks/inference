//! Header-only iOS counted pricing of the three Iris routes (epic 16774): `iris_3b` (generation,
//! including prompt batches), `iris_3b_restore` and `iris_3b_depth`.
//!
//! [`ios_pricing`] prices one request per phase — the component bytes resident under the request's
//! regime (eager or streamed load, staged or warm, the rung-4 window and scope, the rung-3 pixel-row
//! bound) plus the LARGEST counted quantum the phase checks — through the very formulas the runtime
//! guards check ([`crate::dit`], [`crate::text_encoder`], [`crate::stream`], [`crate::solver`],
//! [`crate::model`], [`crate::depth`], [`crate::restoration`]), each fed the widths the loaded
//! module would have, read here from the safetensors headers and the configs. Nothing is read
//! beyond headers, configs and (when the request names no token window) the tokenizer.
//! [`ios_quantum_phase`] attributes every quantum label the crate checks to its phase.
//!
//! A phase's resident bytes are one figure for the whole phase (its largest live set besides the
//! quantum), so its price binds at its largest quantum: `scratch_bytes` is exactly the largest
//! scratch any guard of the phase checks. `stream_tests::the_pricing_states_the_largest_check_of_
//! every_phase` proves it on the CPU fixtures against the checks real runs observe — every quantum
//! label priced at exactly its largest observed check, no label priced that the run never checks —
//! for generation (eager rung 0 / rung 3, deferred rung 4 under each scope, text and prompt batch,
//! dense, quantized at load and packed), restoration and depth (eager, deferred, packed).
//!
//! A generation request's `geometry.batch` is priced as ONE solve of that many rows (a prompt
//! batch, doubled under classifier-free guidance) — exact for a prompt batch, an upper bound for
//! `count` repetitions — and `token_window` (when non-zero) as the longest encoded window.

use std::path::Path;

use mlx_gen::gen_core::iris::depth::DEPTH_EXTRA_INPUT_CHANNELS;
use mlx_gen::gen_core::iris::downstream::TaskExport;
use mlx_gen::gen_core::iris::restoration::{plan, Dims, RestorationOptions, RestorationResources};
use mlx_gen::gen_core::iris::{
    GenerationResources, IrisConfig, IrisTask, ModelConfig as IrisModel, PROMPT_PREFIX,
};
use mlx_gen::gen_core::weightsmeta::{Dtype as Stored, SafetensorsTensorHeader as Header};
use mlx_gen::gen_core::{
    GenerationMemory, InputSizing, LoadShape, MemoryMode, OffloadPolicy, Precision,
};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosPricingRequest};
use mlx_gen::ios_quantum::{sampler_step_bytes, tensor_bytes};
use mlx_gen::{Error, Quant, Result};

use crate::dit::{
    quanta as dit_q, EmbedGeometry, IrisDiT, PixelBlockGeometry, Stack, StackInventory,
    TextBlockGeometry, TrunkGeometry,
};
use crate::nn::{BackboneQuant, ProjectionLayout, QUANT_GROUP_SIZE};
use crate::retention::allocated;
use crate::stream::{largest_window, BlockRead, CastPolicy};
use crate::text_encoder::{quanta as te_q, LANGUAGE_TOWER_PREFIX};

/// The phase every quantum label of this crate runs in.
pub fn ios_quantum_phase(label: &str) -> Option<IosPhase> {
    use crate::depth::quanta as depth_q;
    use crate::restoration::quanta as restore_q;
    let phase = match label {
        te_q::LOAD
        | crate::model::BACKBONE_LOAD_QUANTUM
        | crate::adapters::ADAPTER_LOAD_QUANTUM
        | depth_q::LOAD
        | restore_q::LOAD => IosPhase::Load,
        te_q::TOWER
        | te_q::EMBEDDING
        | te_q::LAYER
        | te_q::DECODER_LAYER
        | te_q::CONDITIONING
        | te_q::LAYER_READ
        | te_q::STREAMED_LAYER => IosPhase::Conditioning,
        depth_q::INPUT | restore_q::HOST_PLANES => IosPhase::Encode,
        dit_q::EMBED
        | dit_q::TEXT_LAYER_BLOCK
        | dit_q::TEXT_POOL
        | dit_q::TEXT_REFINER_BLOCK
        | dit_q::TEXT_OUTPUT
        | dit_q::TRUNK_BLOCK
        | dit_q::PIXEL_EMBED
        | dit_q::PIXEL_BLOCK
        | dit_q::OUTPUT
        | dit_q::BLOCK_READ
        | dit_q::OUTPUT_ROWS
        | crate::solver::STEP_QUANTUM
        | restore_q::TILE => IosPhase::Denoise,
        crate::model::IMAGE_QUANTUM | depth_q::REDUCER | restore_q::POST_TILE => IosPhase::Decode,
        _ => return None,
    };
    Some(phase)
}

/// Every quantum label the crate checks (the coverage set of [`ios_quantum_phase`]).
pub fn quantum_labels() -> Vec<&'static str> {
    let mut labels = vec![
        crate::model::BACKBONE_LOAD_QUANTUM,
        crate::model::IMAGE_QUANTUM,
        crate::adapters::ADAPTER_LOAD_QUANTUM,
        crate::solver::STEP_QUANTUM,
    ];
    labels.extend(te_q::ALL);
    labels.extend(dit_q::ALL);
    labels.extend(crate::depth::quanta::ALL);
    labels.extend(crate::restoration::quanta::ALL);
    labels.sort_unstable();
    labels.dedup();
    labels
}

/// Price one Iris request: see the module docs.
pub fn ios_pricing(req: &IosPricingRequest<'_>) -> Result<IosPricing> {
    Ok(price(req, Pricing::default())?.pricing)
}

/// How [`price`] treats the artifacts and the levers (the prediction's knobs; the public pricing
/// is [`Pricing::default`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pricing {
    /// Price a dense artifact asked for at `q8`/`q4` as the packed export of that tier
    /// (`crate::packed`) — what an iOS load reads — instead of the dense file.
    pub assume_packed: bool,
    /// Rung 3 runs the PiT stage per batch row. `false` prices the batched stage under rung 3 —
    /// the route before that lever existed (the prediction's "before" column).
    pub pixel_rows: bool,
}

impl Default for Pricing {
    fn default() -> Self {
        Self {
            assume_packed: false,
            pixel_rows: true,
        }
    }
}

pub(crate) fn price(req: &IosPricingRequest<'_>, opts: Pricing) -> Result<Priced> {
    let spec = req.spec;
    match req.provider {
        crate::model::MODEL_ID => {
            expect_mode(req.mode, &MemoryMode::TextToImage)?;
            generation(req, opts)
        }
        crate::restoration::MODEL_ID => {
            expect_mode(
                req.mode,
                &crate::memory_strategy::memory_mode(IrisTask::Restoration),
            )?;
            let resources = RestorationResources::from_spec(spec, crate::restoration::MODEL_ID)
                .map_err(Error::from)?;
            restoration(req, opts, &resources)
        }
        crate::depth::DEPTH_MODEL_ID => {
            expect_mode(
                req.mode,
                &crate::memory_strategy::memory_mode(IrisTask::Depth),
            )?;
            let export = TaskExport::from_spec(spec, IrisTask::Depth, crate::depth::DEPTH_MODEL_ID)
                .map_err(Error::from)?;
            depth(req, opts, &export)
        }
        other => Err(Error::Unsupported(format!(
            "{other}: not an Iris-3B MLX memory provider"
        ))),
    }
}

fn expect_mode(got: &MemoryMode, want: &MemoryMode) -> Result<()> {
    if got != want {
        return Err(Error::Unsupported(format!(
            "iris pricing: the route's request surface is {:?}, got {:?}",
            want.as_key(),
            got.as_key()
        )));
    }
    Ok(())
}

fn compute_bytes(precision: Precision) -> u64 {
    match precision {
        Precision::Bf16 => 2,
        Precision::Fp32 => 4,
    }
}

fn compute_dtype(precision: Precision) -> mlx_rs::Dtype {
    match precision {
        Precision::Bf16 => mlx_rs::Dtype::Bfloat16,
        Precision::Fp32 => mlx_rs::Dtype::Float32,
    }
}

fn elements(h: &Header) -> u64 {
    h.shape.iter().map(|&d| d as u64).product()
}

fn header(name: String, dtype: Stored, shape: Vec<usize>, width: u64) -> Header {
    let data_bytes = shape.iter().map(|&d| d as u64).product::<u64>() * width;
    Header {
        name,
        dtype,
        shape,
        data_bytes,
    }
}

/// The headers of a backbone checkpoint as the packed export of `quant` stores them
/// (`crate::packed`): every projection [`ProjectionLayout::of`] packs as its u32 codes over the
/// padded input, bf16 scales and biases, and an `in_features` marker when padded; every other
/// matrix at bf16; vectors and the text position table as released.
fn emulate_packed_backbone(headers: &[Header], prefix: &str, quant: Quant) -> Result<Vec<Header>> {
    let bq = BackboneQuant::for_load(Some(quant))?;
    let mut out = Vec::with_capacity(headers.len());
    for h in headers {
        let name = h.name.strip_prefix(prefix).unwrap_or(&h.name);
        match h.shape.as_slice() {
            [rows, input] if name.ends_with(".weight") => match ProjectionLayout::of(*input, bq) {
                ProjectionLayout::Quantized { bits, padded_in } => {
                    let base = &h.name[..h.name.len() - ".weight".len()];
                    let words = padded_in * bits as usize / 32;
                    out.push(header(h.name.clone(), Stored::U32, vec![*rows, words], 4));
                    for part in ["scales", "biases"] {
                        out.push(header(
                            format!("{base}.{part}"),
                            Stored::BF16,
                            vec![*rows, padded_in / QUANT_GROUP_SIZE],
                            2,
                        ));
                    }
                    if padded_in != *input {
                        out.push(header(
                            format!("{base}{}", crate::packed::IN_FEATURES_SUFFIX),
                            Stored::I32,
                            Vec::new(),
                            4,
                        ));
                    }
                }
                ProjectionLayout::Dense => {
                    out.push(header(h.name.clone(), Stored::BF16, h.shape.clone(), 2))
                }
            },
            shape if shape.len() >= 2 && !name.ends_with("y_pos_embedding") => {
                out.push(header(h.name.clone(), Stored::BF16, h.shape.clone(), 2))
            }
            _ => out.push(h.clone()),
        }
    }
    Ok(out)
}

/// The headers of the Qwen3-VL tower as its packed export stores them: every attention / MLP
/// projection packed at the tower's group, the embedding and every other float tensor at bf16.
fn emulate_packed_tower(
    headers: &[Header],
    cfg: &mlx_llm::ModelConfig,
    quant: Quant,
) -> Result<Vec<Header>> {
    let spec = crate::text_encoder::text_encoder_quant_spec(cfg, Some(quant))?
        .ok_or_else(|| Error::Msg("iris pricing: no tower quantization".into()))?;
    let embedding = format!("{LANGUAGE_TOWER_PREFIX}embed_tokens.weight");
    let mut out = Vec::with_capacity(headers.len());
    for h in headers {
        match h.shape.as_slice() {
            [rows, input] if h.name.ends_with(".weight") && h.name != embedding => {
                let base = &h.name[..h.name.len() - ".weight".len()];
                let group = spec.group_size as usize;
                out.push(header(
                    h.name.clone(),
                    Stored::U32,
                    vec![*rows, input * spec.bits as usize / 32],
                    4,
                ));
                for part in ["scales", "biases"] {
                    out.push(header(
                        format!("{base}.{part}"),
                        Stored::BF16,
                        vec![*rows, input / group],
                        2,
                    ));
                }
            }
            _ if matches!(h.dtype, Stored::F32 | Stored::F16) => {
                out.push(header(h.name.clone(), Stored::BF16, h.shape.clone(), 2))
            }
            _ => out.push(h.clone()),
        }
    }
    Ok(out)
}

fn stored(headers: &[Header], keep: impl Fn(&str) -> bool) -> u64 {
    headers
        .iter()
        .filter(|h| keep(&h.name))
        .fold(0u64, |a, h| a.saturating_add(h.data_bytes))
}

/// One backbone checkpoint's facts, from its headers (emulated as packed where the artifact will
/// be read packed): the module widths every runtime scratch formula reads, and the per-stack
/// window reads.
struct Backbone {
    cfg: IrisModel,
    layers: u64,
    cw: u64,
    input_channels: u64,
    prefix: &'static str,
    /// The headers the loaded module is built from (packed where the load is packed).
    headers: std::collections::HashMap<String, Header>,
    inventory: StackInventory,
}

impl Backbone {
    fn new(
        config: &IrisConfig,
        headers: &[Header],
        prefix: &'static str,
        precision: Precision,
        input_channels: u64,
    ) -> Result<Self> {
        let inventory = StackInventory::from_headers(
            headers,
            prefix,
            &config.model,
            CastPolicy::Backbone {
                compute: compute_dtype(precision),
            },
        )?;
        Ok(Self {
            cfg: config.model.clone(),
            layers: config.text_encoder.hidden_layers.len() as u64,
            cw: compute_bytes(precision),
            input_channels,
            prefix,
            headers: headers
                .iter()
                .map(|h| {
                    let name = h.name.strip_prefix(prefix).unwrap_or(&h.name).to_owned();
                    (name, h.clone())
                })
                .collect(),
            inventory,
        })
    }

    fn get(&self, name: &str) -> Result<&Header> {
        self.headers
            .get(name)
            .ok_or_else(|| Error::MissingTensor(format!("{}{name}", self.prefix)))
    }

    /// Output rows of projection `p`.
    fn rows(&self, p: &str) -> Result<u64> {
        Ok(self.get(&format!("{p}.weight"))?.shape[0] as u64)
    }

    /// Logical input width of projection `p` (a packed one's padded input; no Iris projection
    /// but the SwiGLU down projections is padded, and those are priced through [`Self::pad`]).
    fn input(&self, p: &str) -> Result<u64> {
        match self.headers.get(&format!("{p}.scales")) {
            Some(scales) => Ok(scales.shape[1] as u64 * QUANT_GROUP_SIZE as u64),
            None => Ok(self.get(&format!("{p}.weight"))?.shape[1] as u64),
        }
    }

    /// Zero columns a packed projection `p` with logical input `input` is padded by.
    fn pad(&self, p: &str, input: u64) -> Result<u64> {
        Ok(match self.headers.get(&format!("{p}.scales")) {
            Some(scales) => {
                (scales.shape[1] as u64 * QUANT_GROUP_SIZE as u64).saturating_sub(input)
            }
            None => 0,
        })
    }

    fn norm(&self, n: &str) -> Result<u64> {
        Ok(self.get(&format!("{n}.weight"))?.shape[0] as u64)
    }

    fn trunk(&self, i: usize, g: TrunkGeometry) -> Result<u64> {
        let p = |name: &str| format!("blocks.{i}.{name}");
        let text_out = self.cfg.final_block_text == "keep" || i + 1 < self.cfg.depth;
        let unfused = crate::retention::unfused();
        if i < self.cfg.dual_depth {
            let ff_x = self.rows(&p("mlp_x.w1"))?;
            let pad_x = self.pad(&p("mlp_x.w2"), ff_x)?;
            let (ff, pad) = if text_out {
                let ff_y = self.rows(&p("mlp_y.w1"))?;
                (ff_x.max(ff_y), pad_x.max(self.pad(&p("mlp_y.w2"), ff_y)?))
            } else {
                (ff_x, pad_x)
            };
            let g = TrunkGeometry {
                dim: self.norm(&p("norm_x1"))?,
                ff,
                text: text_out,
                down_pad: pad,
                ..g
            };
            Ok(IrisDiT::dual_block_bytes(&g, unfused))
        } else {
            let ff = self.rows(&p("mlp.w1"))?;
            let g = TrunkGeometry {
                dim: self.norm(&p("norm1"))?,
                ff,
                text: text_out,
                down_pad: self.pad(&p("mlp.w2"), ff)?,
                ..g
            };
            Ok(IrisDiT::single_block_bytes(&g, unfused))
        }
    }

    /// Every quantum one forward over `[b, C, h, w]` checks under `plan`, by label; and its
    /// largest live carry between quanta.
    fn forward(
        &self,
        b: u64,
        h: u64,
        w: u64,
        plan: mlx_gen::AttentionPlan<'_>,
        by_row: bool,
        out: &mut Vec<(&'static str, u64)>,
    ) -> Result<u64> {
        let m = &self.cfg;
        let cw = self.cw;
        let p = m.patch_size as u64;
        let n = (h / p) * (w / p);
        let t = m.text_len as u64;
        let hidden = m.hidden_size as u64;
        let heads = m.num_heads as u64;
        let kv_heads = m.num_kv_heads.unwrap_or(m.num_heads) as u64;
        let unfused = crate::retention::unfused();
        let rows = |plan, batch: u64, heads: u64, tokens: u64| {
            IrisDiT::query_rows(plan, batch as i32, heads as i32, tokens as i32)
        };
        let cores = self.rows("modulation_cores.adaln_img")?
            + if m.dual_depth > 0 {
                self.rows("modulation_cores.adaln_txt")?
            } else {
                0
            };
        out.push((
            dit_q::EMBED,
            IrisDiT::embed_bytes(EmbedGeometry {
                compute_bytes: cw,
                batch: b,
                patches: n,
                patch_in: self.input("s_embedder.proj")?,
                hidden: self.rows("s_embedder.proj")?,
                sinusoid: self.input("t_embedder.mlp.0")?,
                t_hidden: self.rows("t_embedder.mlp.0")?,
                t_out: self.rows("t_embedder.mlp.2")?,
                cores,
            })
            .saturating_add(sampler_step_bytes(b * self.input_channels * h * w)),
        ));
        for i in 0..Stack::Layer.depth(m) {
            let s = |name: &str| format!("y_embedder.layer_blocks.{i}.{name}");
            let heads = m.text_lap_num_heads as u64;
            let ff = self.rows(&s("mlp.0"))?;
            out.push((
                dit_q::TEXT_LAYER_BLOCK,
                IrisDiT::text_block_bytes(
                    TextBlockGeometry {
                        compute_bytes: cw,
                        batch: b * t,
                        tokens: self.layers,
                        dim: self.norm(&s("norm1"))?,
                        heads,
                        ff,
                        query_rows: rows(plan, b * t, heads, self.layers),
                        eval_per_chunk: plan.budget.eval_per_chunk(),
                        refiner: false,
                        down_pad: self.pad(&s("mlp.2"), ff)?,
                        stream_bytes: 4,
                    },
                    unfused,
                ),
            ));
        }
        out.push((
            dit_q::TEXT_POOL,
            IrisDiT::text_pool_bytes(
                cw,
                b,
                t,
                self.layers,
                m.text_dim as u64,
                self.rows("y_embedder.refiner.proj")?,
            ),
        ));
        for i in 0..Stack::Refiner.depth(m) {
            let s = |name: &str| format!("y_embedder.refiner.blocks.{i}.{name}");
            let ff = self.rows(&s("mlp.w1"))?;
            out.push((
                dit_q::TEXT_REFINER_BLOCK,
                IrisDiT::text_block_bytes(
                    TextBlockGeometry {
                        compute_bytes: cw,
                        batch: b,
                        tokens: t,
                        dim: self.norm(&s("norm1"))?,
                        heads,
                        ff,
                        query_rows: rows(plan, b, heads, t),
                        eval_per_chunk: plan.budget.eval_per_chunk(),
                        refiner: true,
                        down_pad: self.pad(&s("mlp.w2"), ff)?,
                        stream_bytes: cw,
                    },
                    unfused,
                ),
            ));
        }
        out.push((
            dit_q::TEXT_OUTPUT,
            IrisDiT::text_output_bytes(cw, b, t, self.norm("y_embedder.refiner.norm")?, unfused),
        ));
        let trunk = TrunkGeometry {
            compute_bytes: cw,
            batch: b,
            img_tokens: n,
            txt_tokens: t,
            dim: hidden,
            heads,
            kv_heads,
            ff: 0,
            query_rows: rows(plan, b, heads, n + t),
            eval_per_chunk: plan.budget.eval_per_chunk(),
            split_qkv: kv_heads != heads,
            text: true,
            down_pad: 0,
            img_stream_bytes: 4,
        };
        for i in 0..m.depth {
            let g = TrunkGeometry {
                img_stream_bytes: if i == 0 { cw } else { 4 },
                ..trunk
            };
            let entry = if i == 0 {
                IrisDiT::trunk_entry_bytes(b, n, t, hidden / heads.max(1), hidden)
            } else {
                0
            };
            out.push((dit_q::TRUNK_BLOCK, self.trunk(i, g)?.saturating_add(entry)));
        }
        out.push((
            dit_q::PIXEL_EMBED,
            IrisDiT::pixel_embed_bytes(
                cw,
                b,
                h,
                w,
                p,
                self.input("modulation_cores.adaln_img")?,
                self.rows("pixel_embedder.proj")?,
                self.input("pixel_embedder.proj")?,
            ),
        ));
        let row_b = if by_row { 1 } else { b };
        for i in 0..m.pixel.depth {
            let s = |name: &str| format!("pixel_blocks.{i}.{name}");
            let heads = m.pixel.num_heads as u64;
            out.push((
                dit_q::PIXEL_BLOCK,
                IrisDiT::pixel_block_bytes(
                    PixelBlockGeometry {
                        compute_bytes: cw,
                        batch: row_b,
                        patches: n,
                        patch_pixels: p * p,
                        pixel_width: self.norm(&s("norm1"))?,
                        compact_width: self.rows(&s("compress"))?,
                        modulation_width: self.rows(&s("adaln"))?,
                        condition_width: self.input(&s("adaln"))?,
                        fc1_width: self.rows(&s("mlp.fc1"))?,
                        heads,
                        query_rows: rows(plan, row_b, heads, n),
                        eval_per_chunk: plan.budget.eval_per_chunk(),
                    },
                    unfused,
                ),
            ));
        }
        out.push((
            dit_q::OUTPUT,
            IrisDiT::output_bytes(
                cw,
                row_b,
                n,
                p * p,
                self.input("final_layer.linear")?,
                self.rows("final_layer.linear")?,
                unfused,
            ),
        ));
        if by_row {
            out.push((
                dit_q::OUTPUT_ROWS,
                IrisDiT::output_rows_bytes(b as i32, m.in_channels as i32, h as i32, w as i32),
            ));
        }
        // The largest carry between quanta: the trunk's image + text streams, or the PiT stage's
        // pixel stream + per-patch conditioning (+ the finished rows when run per row).
        let trunk_carry = allocated(4 * b * n * hidden) + allocated(4 * b * t * hidden);
        let pixel_width = m.pixel.hidden_size as u64;
        let pixel_carry = allocated(cw * b * n * p * p * pixel_width)
            + allocated(4 * b * n * hidden)
            + allocated(4 * b * m.in_channels as u64 * h * w);
        Ok(trunk_carry.max(pixel_carry))
    }

    /// The largest window read and held over the four stacks, at `window` blocks per window.
    fn stream(&self, window: Option<usize>) -> (u64, u64) {
        Stack::ALL
            .into_iter()
            .map(|stack| {
                let reads: &[BlockRead] = self.inventory.reads(stack);
                largest_window(reads, window.unwrap_or(reads.len()))
            })
            .fold((0, 0), |(r, h), (rr, hh)| (r.max(rr), h.max(hh)))
    }
}

/// A backbone checkpoint priced for a load: its facts, its resident bytes and its load floor.
struct PricedBackbone {
    facts: Backbone,
    resident: u64,
    floor: u64,
}

/// Price the backbone `path` (export key prefix `prefix`) for a load at `precision` / `requested`,
/// eager or `streamed`.
#[allow(clippy::too_many_arguments)]
fn priced_backbone(
    config: &IrisConfig,
    path: &Path,
    prefix: &'static str,
    precision: Precision,
    requested: Option<Quant>,
    streamed: bool,
    input_channels: u64,
    opts: Pricing,
) -> Result<PricedBackbone> {
    let real = crate::stream::headers(path)?;
    let artifact = crate::packed::artifact_tier(path)?;
    crate::packed::check_requested("the backbone", artifact, requested)?;
    let requantized = artifact.is_none() && requested.is_some();
    if requantized && streamed && !opts.assume_packed {
        crate::stream::refuse_stream_requantization(
            crate::model::MODEL_ID,
            "the backbone",
            artifact,
            requested,
        )?;
    }
    let built = match (requantized, requested) {
        (true, Some(quant)) => emulate_packed_backbone(&real, prefix, quant)?,
        _ => real.clone(),
    };
    let skeleton = |name: &str| {
        name.strip_prefix(prefix)
            .is_none_or(|n| Stack::of(n).is_none())
    };
    let keep = |name: &str| !streamed || skeleton(name);
    // The load floor is the file the load reads: the packed export when one is assumed.
    let floor_headers = if opts.assume_packed { &built } else { &real };
    let floor = stored(floor_headers, keep);
    let quant = if requantized {
        BackboneQuant::for_load(requested)?
    } else {
        None
    };
    let facts = Backbone::new(config, &built, prefix, precision, input_channels)?;
    let resident = crate::memory_strategy::backbone_bytes_where(path, precision, quant, keep)
        .map_err(Error::from)?;
    Ok(PricedBackbone {
        facts,
        resident,
        floor,
    })
}

/// The Qwen3-VL tower priced for a load.
struct PricedTower {
    cfg: mlx_llm::ModelConfig,
    resident: u64,
    floor: u64,
    reads: Vec<BlockRead>,
}

fn priced_tower(
    dir: &Path,
    requested: Option<Quant>,
    streamed: bool,
    opts: Pricing,
) -> Result<PricedTower> {
    let cfg = mlx_llm::ModelConfig::from_dir(dir)
        .map_err(|e| Error::Msg(format!("iris pricing: the text encoder config: {e}")))?;
    let untied = !cfg.tie_word_embeddings;
    let files = crate::text_encoder::weight_shards(dir, untied)?;
    let mut real = Vec::new();
    for file in &files {
        real.extend(
            crate::stream::headers(file)?
                .into_iter()
                .filter(|h| crate::text_encoder::is_text_tower_tensor(&h.name, untied)),
        );
    }
    let packed = cfg.quantization.is_some();
    let artifact = if packed {
        crate::packed::text_encoder_tier(dir)?
    } else {
        None
    };
    crate::packed::check_requested("the text encoder", artifact, requested)?;
    let requantized = !packed && requested.is_some();
    if requantized && streamed && !opts.assume_packed {
        crate::stream::refuse_stream_requantization(
            crate::model::MODEL_ID,
            "the text encoder",
            None,
            requested,
        )?;
    }
    let built = match (requantized, requested) {
        (true, Some(quant)) => emulate_packed_tower(&real, &cfg, quant)?,
        _ => real.clone(),
    };
    let embed = format!("{LANGUAGE_TOWER_PREFIX}embed_tokens.weight");
    let norm = format!("{LANGUAGE_TOWER_PREFIX}norm.weight");
    let skeleton = |name: &str| name == embed || name == norm;
    let floor_headers = if opts.assume_packed { &built } else { &real };
    let resident = if streamed {
        built
            .iter()
            .filter(|h| skeleton(&h.name))
            .map(|h| allocated(elements(h) * 2))
            .sum()
    } else {
        crate::memory_strategy::text_encoder_bytes(dir, requested).map_err(Error::from)?
    };
    let floor = if streamed {
        stored(floor_headers, skeleton)
    } else {
        stored(floor_headers, |_| true)
    };
    let mut grouped: Vec<Vec<&Header>> = vec![Vec::new(); cfg.num_layers];
    for h in &built {
        if let Some(i) = crate::text_encoder::tower_layer_of(&h.name) {
            if let Some(slot) = grouped.get_mut(i) {
                slot.push(h);
            }
        }
    }
    let reads = grouped
        .iter()
        .map(|layer| crate::stream::block_read(layer, CastPolicy::Tower))
        .collect();
    Ok(PricedTower {
        cfg,
        resident,
        floor,
        reads,
    })
}

/// A request's pricing and every quantum it was folded from (`(phase, label, scratch)`).
#[derive(Default)]
pub(crate) struct Priced {
    pub pricing: IosPricing,
    pub quanta: Vec<(IosPhase, &'static str, u64)>,
}

/// Fold one phase's quanta into `pricing` at the phase's resident bytes.
fn take(pricing: &mut Priced, phase: IosPhase, resident: u64, quanta: &[(&'static str, u64)]) {
    for &(label, scratch) in quanta {
        debug_assert_eq!(ios_quantum_phase(label), Some(phase), "{label}");
        pricing.pricing.take(phase, resident, scratch, label);
        pricing.quanta.push((phase, label, scratch));
    }
}

fn plan_of(execution: &GenerationMemory) -> mlx_gen::AttentionPlan<'static> {
    if execution.chunk_attention {
        mlx_gen::AttentionPlan::budgeted(mlx_gen::AttentionBudget::from_score_elements(
            u64::from(
                execution
                    .attention_chunk_size
                    .unwrap_or(crate::memory_strategy::ATTENTION_CHUNK_SIZE),
            ),
            true,
        ))
    } else {
        mlx_gen::AttentionPlan::UNBOUNDED
    }
}

/// The encoded window priced: the request's own when it names one, else the full window (the
/// chat-template prefix and a caption at the budget, its suffix inside the budget).
fn token_window(req: &IosPricingRequest<'_>, te_dir: &Path, max_length: u64) -> Result<u64> {
    if req.token_window > 0 {
        return Ok(u64::from(req.token_window));
    }
    let tokenizer = mlx_gen::gen_core::tokenizer::TextTokenizer::from_file(
        te_dir.join("tokenizer.json"),
        mlx_gen::gen_core::tokenizer::TokenizerConfig {
            max_length: usize::MAX,
            pad_token_id: 0,
            chat_template: mlx_gen::gen_core::tokenizer::ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(Error::from)?;
    let prefix = tokenizer
        .encode_ids(PROMPT_PREFIX, false)
        .map_err(Error::from)?
        .len() as u64;
    Ok(prefix + max_length)
}

fn generation(req: &IosPricingRequest<'_>, opts: Pricing) -> Result<Priced> {
    let spec = req.spec;
    crate::memory_strategy::refuse_streamed_adapters(crate::model::MODEL_ID, spec)
        .map_err(Error::from)?;
    crate::memory_strategy::refuse_unloadable_quant(crate::model::MODEL_ID, spec)
        .map_err(Error::from)?;
    let resources =
        GenerationResources::from_spec(spec, crate::model::MODEL_ID).map_err(Error::from)?;
    let config = IrisConfig::from_dir(&resources.backbone_dir).map_err(Error::from)?;
    let streamed = spec.load_shape == LoadShape::DeferredMaterialization;
    let staged = req.execution.stage_residency || spec.offload_policy == OffloadPolicy::Sequential;
    let mut probe = mlx_gen::GenerationRequest {
        memory: Some(*req.execution),
        ..Default::default()
    };
    if *req.execution == GenerationMemory::default() {
        probe.memory = None;
    }
    let rungs = crate::model::request_rungs(&probe, streamed)?;
    let plan = plan_of(req.execution);
    let bb = priced_backbone(
        &config,
        &resources
            .backbone_dir
            .join(mlx_gen::gen_core::iris::BACKBONE_WEIGHTS_FILE),
        "",
        spec.precision,
        spec.quantize,
        streamed,
        config.model.in_channels as u64,
        opts,
    )?;
    let te = priced_tower(&resources.text_encoder_dir, spec.quantize, streamed, opts)?;
    let te_cfg = &te.cfg;
    let max_length = config.text_encoder.max_length as u64;
    let layers = config.text_encoder.hidden_layers.len() as u64;
    let dim = config.text_encoder.dim as u64;
    let deepest = *config.text_encoder.hidden_layers.last().unwrap_or(&0);
    let n = token_window(req, &resources.text_encoder_dir, max_length)?;
    let prompts = u64::from(req.geometry.batch.max(1));
    let cond = allocated(tensor_bytes(4, &[max_length, layers, dim]));
    let (h, w) = (
        u64::from(req.geometry.height),
        u64::from(req.geometry.width),
    );
    let channels = config.model.in_channels as u64;
    let b = 2 * prompts; // classifier-free guidance doubles the solve
    let bb_resident = if streamed {
        bb.resident
            .saturating_add(bb.facts.stream(rungs.dit_window).1)
    } else {
        bb.resident
    };
    let warm_te = if staged { 0 } else { te.resident };
    // Outside the denoise a streamed backbone holds no window: only its skeleton.
    let warm_bb = if staged { 0 } else { bb.resident };
    let mut pricing = Priced::default();

    // Load: the tower, then the backbone (staged: once the conditioning exists; warm: beside the
    // tower), and every adapter file.
    let mut loads = vec![
        (te_q::LOAD, te.floor),
        (crate::model::BACKBONE_LOAD_QUANTUM, bb.floor),
    ];
    for adapter in &spec.adapters {
        loads.push((
            crate::adapters::ADAPTER_LOAD_QUANTUM,
            mlx_gen::ios_quantum::load_floor_bytes(std::slice::from_ref(&adapter.path), |_| true)?,
        ));
    }
    let load_resident = if staged {
        (prompts + 1) * cond
    } else {
        te.resident
    };
    take(&mut pricing, IosPhase::Load, load_resident, &loads);

    // Conditioning: every prompt and the CFG null in turn; the conditionings already returned
    // stay resident.
    let held_conds = prompts * cond;
    let n_i = n as i32;
    let real_max = (0..=max_length as i32)
        .map(|real| crate::text_encoder::conditioning_bytes(layers, max_length, dim, real))
        .max()
        .unwrap_or(0);
    let mut text = vec![(te_q::CONDITIONING, real_max)];
    let mut walk = 0;
    if streamed {
        let run = deepest.min(te_cfg.num_layers);
        let window = rungs.text_window.unwrap_or(run.max(1));
        text.push((
            te_q::EMBEDDING,
            crate::text_encoder::streamed_embedding_bytes(te_cfg, n),
        ));
        text.push((te_q::LAYER_READ, largest_window(&te.reads[..run], window).0));
        text.push((
            te_q::STREAMED_LAYER,
            crate::text_encoder::streamed_layer_bytes(te_cfg, n),
        ));
        // The walk's carry and picked states, and the RoPE tables.
        walk = (layers + 1) * allocated(2 * n * te_cfg.hidden_size.max(0) as u64)
            + 2 * allocated(2 * n * te_cfg.head_dim.max(0) as u64)
            + largest_window(&te.reads[..run], window).1;
    } else {
        let scratch = |k: usize| {
            let (layer, held) = crate::text_encoder::decoder_layer_parts(te_cfg, n_i, k);
            layer.saturating_add(held)
        };
        text.push((te_q::TOWER, scratch(deepest)));
        text.push((te_q::EMBEDDING, scratch(0)));
        text.push((te_q::LAYER, (1..=deepest).map(scratch).max().unwrap_or(0)));
        text.push((
            te_q::DECODER_LAYER,
            (1..=te_cfg.num_layers).map(scratch).max().unwrap_or(0),
        ));
    }
    take(
        &mut pricing,
        IosPhase::Conditioning,
        te.resident + warm_bb + held_conds + walk,
        &text,
    );

    // Denoise: the backbone, every conditioning and their CFG batch, the solver state.
    let mut denoise = Vec::new();
    let by_row = opts.pixel_rows && rungs.pixel_rows && b > 1;
    let carry = bb.facts.forward(b, h, w, plan, by_row, &mut denoise)?;
    denoise.push((
        crate::solver::STEP_QUANTUM,
        crate::solver::worst_step_scratch(prompts * channels * h * w),
    ));
    if streamed {
        denoise.push((dit_q::BLOCK_READ, bb.facts.stream(rungs.dit_window).0));
    }
    let state = prompts * channels * h * w;
    let solver = 3 * allocated(4 * state) + allocated(4 * b * channels * h * w);
    take(
        &mut pricing,
        IosPhase::Denoise,
        bb_resident + warm_te + (prompts + 1) * cond + b * cond + solver + carry,
        &denoise,
    );

    // Decode: the RGB8 conversion of the solved batch.
    take(
        &mut pricing,
        IosPhase::Decode,
        bb.resident + warm_te + (prompts + 1) * cond + allocated(4 * state),
        &[(
            crate::model::IMAGE_QUANTUM,
            crate::model::image_scratch(prompts * 3 * h * w),
        )],
    );
    Ok(pricing)
}

/// The encoder-free routes' backbone forward and stream (their deferred load streams at the
/// published window; no request selects anything else).
fn export_window(streamed: bool) -> Option<usize> {
    streamed.then_some(crate::stream::TRANSFORMER_WINDOW_SIZE as usize)
}

fn restoration(
    req: &IosPricingRequest<'_>,
    opts: Pricing,
    resources: &RestorationResources,
) -> Result<Priced> {
    let spec = req.spec;
    crate::memory_strategy::refuse_unloadable_quant(crate::restoration::MODEL_ID, spec)
        .map_err(Error::from)?;
    if !opts.assume_packed {
        crate::packed::refuse_dense_requested_as_packed(
            crate::restoration::MODEL_ID,
            &resources.export.weights_path(),
            spec.quantize,
        )?;
    }
    let streamed = spec.load_shape == LoadShape::DeferredMaterialization;
    let config = &resources.export.config;
    let bb = priced_backbone(
        config,
        &resources.export.weights_path(),
        "",
        spec.precision,
        spec.quantize,
        streamed,
        config.model.in_channels as u64,
        opts,
    )?;
    let states = stored(
        &crate::stream::headers(&resources.export.empty_prompt_path())?,
        |_| true,
    );
    // The geometry is the processing size (`restoration_memory_geometry`): priced as an output of
    // that size with the colour fix on — the largest host planes a plan of it can allocate.
    let geometry = resources.settings.geometry(config);
    let p = plan(
        Dims::new(req.geometry.width, req.geometry.height),
        &RestorationOptions {
            scale: 1.0,
            input_sizing: InputSizing::Original,
            color_fix: true,
        },
        geometry,
        crate::restoration::MODEL_ID,
    )
    .map_err(Error::from)?;
    let (th, tw) = if p.single_pass() {
        (p.padded.height as u64, p.padded.width as u64)
    } else {
        (p.tile as u64, p.tile as u64)
    };
    let host = crate::restoration::host_plane_bytes(&p);
    let window = export_window(streamed);
    // A streamed backbone holds a window only inside its forward (the tile phase).
    let rb = bb.resident + states;
    let held_window = if streamed {
        bb.facts.stream(window).1
    } else {
        0
    };
    let mut pricing = Priced::default();
    take(
        &mut pricing,
        IosPhase::Load,
        0,
        &[
            (crate::restoration::quanta::LOAD, bb.floor + states),
            (crate::model::BACKBONE_LOAD_QUANTUM, bb.floor),
        ],
    );
    take(
        &mut pricing,
        IosPhase::Encode,
        rb,
        &[(crate::restoration::quanta::HOST_PLANES, host.peak())],
    );
    let tile = crate::restoration::tile_scratch(3 * th * tw);
    let mut denoise = vec![(crate::restoration::quanta::TILE, tile)];
    let carry = bb.facts.forward(
        1,
        th,
        tw,
        mlx_gen::AttentionPlan::UNBOUNDED,
        false,
        &mut denoise,
    )?;
    if streamed {
        denoise.push((dit_q::BLOCK_READ, bb.facts.stream(window).0));
    }
    take(
        &mut pricing,
        IosPhase::Denoise,
        rb + held_window + host.tile_phase + tile + carry,
        &denoise,
    );
    take(
        &mut pricing,
        IosPhase::Decode,
        rb + host.live_at_last_tile,
        &[(crate::restoration::quanta::POST_TILE, host.post_tile_new())],
    );
    Ok(pricing)
}

fn depth(req: &IosPricingRequest<'_>, opts: Pricing, export: &TaskExport) -> Result<Priced> {
    let spec = req.spec;
    crate::memory_strategy::refuse_unloadable_quant(crate::depth::DEPTH_MODEL_ID, spec)
        .map_err(Error::from)?;
    if !opts.assume_packed {
        crate::packed::refuse_dense_requested_as_packed(
            crate::depth::DEPTH_MODEL_ID,
            &export.weights_path(),
            spec.quantize,
        )?;
    }
    let streamed = spec.load_shape == LoadShape::DeferredMaterialization;
    let config = &export.config;
    let input_channels = (config.model.in_channels + DEPTH_EXTRA_INPUT_CHANNELS) as u64;
    let bb = priced_backbone(
        config,
        &export.weights_path(),
        "pixel.",
        spec.precision,
        spec.quantize,
        streamed,
        input_channels,
        opts,
    )?;
    let states = stored(
        &crate::stream::headers(&export.empty_prompt_path())?,
        |_| true,
    );
    let window = export_window(streamed);
    // A streamed backbone holds a window only inside its forward.
    let db = bb.resident + states;
    let held_window = if streamed {
        bb.facts.stream(window).1
    } else {
        0
    };
    let (h, w) = (
        u64::from(req.geometry.height),
        u64::from(req.geometry.width),
    );
    let mut pricing = Priced::default();
    take(
        &mut pricing,
        IosPhase::Load,
        0,
        &[(crate::depth::quanta::LOAD, bb.floor + states)],
    );
    let input = crate::depth::depth_input_scratch(1, h as i32, w as i32);
    take(
        &mut pricing,
        IosPhase::Encode,
        db,
        &[(crate::depth::quanta::INPUT, input)],
    );
    let mut denoise = Vec::new();
    let carry = bb.facts.forward(
        1,
        h,
        w,
        mlx_gen::AttentionPlan::UNBOUNDED,
        false,
        &mut denoise,
    )?;
    if streamed {
        denoise.push((dit_q::BLOCK_READ, bb.facts.stream(window).0));
    }
    take(
        &mut pricing,
        IosPhase::Denoise,
        db + held_window + input + carry,
        &denoise,
    );
    let channels = config.model.in_channels as u64;
    take(
        &mut pricing,
        IosPhase::Decode,
        db + allocated(4 * channels * h * w),
        &[(
            crate::depth::quanta::REDUCER,
            crate::depth::depth_reducer_bytes(
                channels * h * w,
                channels,
                compute_bytes(spec.precision),
            ),
        )],
    );
    Ok(pricing)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every quantum label the crate checks is attributed to a phase.
    ///
    /// *Mutation that reds this:* dropping any arm of `ios_quantum_phase`.
    #[test]
    fn every_quantum_label_has_a_phase() {
        let labels = quantum_labels();
        assert!(labels.len() >= 30, "{labels:?}");
        for label in labels {
            assert!(ios_quantum_phase(label).is_some(), "{label} has no phase");
        }
        assert_eq!(ios_quantum_phase("not an Iris quantum"), None);
    }
}
