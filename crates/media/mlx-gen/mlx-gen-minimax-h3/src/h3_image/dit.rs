//! `MiniMaxH3ImageTransformer2DModel` on MLX: H3's DiT layers in image mode.
//!
//! ```text
//! proj_in(image rows) ─────────────┐
//! context_embedder ─► token_refiner ┴─► [text | image] ─► 50 × DitBlock ─► norm_out ─► pdd_proj_out[i] ─► velocity
//! ```
//!
//! Everything the graph runs is H3's: [`DitBlock`] (attention, SwiGLU FFN, AdaLN modulation),
//! [`TokenRefiner`], [`MmRope`], and the [`LinearBias`](crate::dit::LinearBias) /
//! [`TimestepEmbedder`](crate::dit::TimestepEmbedder) / [`AdaLayerNormOut`] input and output
//! layers. Three things differ from the video DiT, each a property of the checkpoint:
//!
//! * **No audio stream.** The checkpoint carries no `audio_proj_in` / `audio_proj_out`, so the
//!   input/output set is [`ImageDitProjections`] (644 published tensors, not 638). H3's
//!   [`crate::dit::DitProjections`], which requires the audio pair at load, is unchanged.
//! * **One output head per step.** Step `i` projects with `pdd_proj_out.{i}`
//!   ([`crate::dit::heads::PddHeads::select`]); the step index is an argument of [`ImageDenoiser::velocity`]
//!   rather than a property of the model, so it reaches the head application on every call.
//! * **At most two row classes.** Every text and generated row sits at the step's timestep
//!   (upstream `row_timestep_plan`); an edit's reference rows sit at `0.999` for every step. The
//!   gather index is `global_timestep · 3 + tag`. The six timesteps are enumerated up front,
//!   which lets the whole run's modulation be projected once and `adaln_proj` evicted — H3's own
//!   [`AdaLnCache::precompute_and_evict`].
//!
//! The sampler is the H3-Image checkpoint's: the [`SigmaSchedule::h3_image_distilled`] grid, and
//! the `velocity` Euler form its scheduler config selects ([`ImageDenoiser::step`]).
//!
//! **Rung 4 (SW-7094).** [`MiniMaxH3ImageDit::load_dir_deferred`] keeps only the non-block tensors
//! (the I/O projections, the six heads, the refiner) and hands the 50 blocks to H3's
//! [`DitBlockStream`]; [`ImageDenoiser::new_windowed`] projects the modulation through H3's
//! windowed precompute and rebuilds the block bodies a window at a time on every step.

use std::path::Path;

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::attention::{AttentionBudget, AttentionChunkAxis, AttentionPlan, BoundedAttention};
use mlx_gen::block_residency::{run_windowed, BlockPlan};
use mlx_gen::gen_core::h3_image::REFERENCE_NOISE_AUG;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};

use crate::block_stream::{precompute_adaln_windowed, DitBlockStream};

use crate::denoise::schedule::SigmaSchedule;
use crate::dit::adaln::{AdaLnCache, AdaLnResidency, TimestepSchedule};
use crate::dit::block::DitBlock;
use crate::dit::config::MiniMaxH3ImageDitConfig;
use crate::dit::heads::{AdaLayerNormOut, ImageDitProjections, NormOutModulation};
use crate::dit::model::UnverifiedSources;
use crate::dit::refiner::TokenRefiner;
use crate::dit::rope::{MmRope, MmRopeTables};
use crate::h3_image::layout::ImageLayout;
use crate::tensor::slice_axis;

/// The published component directory the image DiT loads from. `transformer/` is the native
/// (non-diffusers) layout and is never read.
pub const IMAGE_DIT_COMPONENT: &str = "transformer_diffusers";

/// Tensors the published `transformer_diffusers/` index carries: `50 · 12 + 21 + 11 + 2 · 6`.
pub const PUBLISHED_IMAGE_DIT_TENSORS: usize = 644;

/// The whole image DiT.
#[derive(Debug, Clone)]
pub struct MiniMaxH3ImageDit {
    cfg: MiniMaxH3ImageDitConfig,
    projections: ImageDitProjections,
    refiner: TokenRefiner,
    /// Empty under a deferred load (SW-7094, rung 4) — the blocks live in `stream` instead.
    blocks: Vec<DitBlock>,
    /// `Some` under [`mlx_gen::gen_core::LoadShape::DeferredMaterialization`]: H3's
    /// [`DitBlockStream`] over this DiT's 50 blocks (bodies per step, `adaln_proj` once per request).
    stream: Option<DitBlockStream>,
    rope: MmRope,
    dtype: Dtype,
    /// Source tensors not yet GPU-verified, grouped by first consumer (H3's sc-23402 scheme).
    unverified: UnverifiedSources,
}

impl MiniMaxH3ImageDit {
    /// Build from an already-populated [`Weights`] map. `dtype` is the block stack's precision;
    /// the input/output tensors keep their published dtypes (four of them ship float32).
    pub fn from_weights(
        w: &mut Weights,
        cfg: &MiniMaxH3ImageDitConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        cfg.validate()?;
        let projections = ImageDitProjections::from_weights(w, cfg)?;
        let refiner = TokenRefiner::from_weights(w, "token_refiner", &cfg.base, dtype)?;
        let blocks = (0..cfg.base.num_layers)
            .map(|i| {
                DitBlock::from_weights(w, &format!("transformer_blocks.{i}"), &cfg.base, dtype)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg: cfg.clone(),
            projections,
            refiner,
            blocks,
            stream: None,
            rope: MmRope::new(cfg.base.rope_freq_dim, cfg.base.rope_theta)?,
            dtype,
            unverified: UnverifiedSources::partition(w.accessed_entries()),
        })
    }

    /// Load **only** the non-block tensors — the I/O projections, the six per-step output heads
    /// (`pdd_proj_out.{0..6}`) and the token refiner — and describe the 600 block tensors as an H3
    /// [`DitBlockStream`]: rung 4's [`mlx_gen::gen_core::LoadShape::DeferredMaterialization`] loader
    /// (SW-7094).
    ///
    /// The heads stay resident because each one is used once per request and together they are
    /// `6 · 5376 · 96` float32 weights (12.4 MB): windowing them would re-read a head per step to
    /// bound bytes one block window already exceeds 60-fold. Everything under `transformer_blocks.`
    /// is left unread; the map is dropped before returning so no retained handle keeps a block
    /// tensor reachable across windows (the hazard `run_windowed` documents).
    pub fn load_dir_deferred(dir: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = Self::read_config(dir)?;
        cfg.validate()?;
        let stream = DitBlockStream::new(dir, dtype, cfg.base.clone())?;
        let (projections, refiner, unverified) = {
            let mut w = Weights::from_dir(dir)?;
            let projections = ImageDitProjections::from_weights(&mut w, &cfg)?;
            let refiner = TokenRefiner::from_weights(&mut w, "token_refiner", &cfg.base, dtype)?;
            // Only the head tensors were read, so only the `head` group is populated.
            (
                projections,
                refiner,
                UnverifiedSources::partition(w.accessed_entries()),
            )
        };
        Ok(Self {
            cfg: cfg.clone(),
            projections,
            refiner,
            blocks: Vec::new(),
            stream: Some(stream),
            rope: MmRope::new(cfg.base.rope_freq_dim, cfg.base.rope_theta)?,
            dtype,
            unverified,
        })
    }

    /// The block stream backing a deferred load, `None` for a resident one.
    pub fn stream(&self) -> Option<&DitBlockStream> {
        self.stream.as_ref()
    }

    /// Whether this load defers block materialization (rung 4).
    pub fn is_deferred(&self) -> bool {
        self.stream.is_some()
    }

    /// Evaluate the adapter residuals a deferred load holds for its streamed blocks, once, so no
    /// window pulls a lazy adapter-file read into its per-block eval. A no-op on a resident load.
    pub fn materialize_streamed_adapters(&self) -> Result<()> {
        match &self.stream {
            Some(stream) => stream.adapters().materialize(),
            None => Ok(()),
        }
    }

    fn read_config(dir: &Path) -> Result<MiniMaxH3ImageDitConfig> {
        let config_path = dir.join("config.json");
        let text = mlx_gen::gen_core::__model_input_choose!(
            std::fs::read_to_string(&config_path),
            mlx_gen::gen_core::model_input_proof::read_configuration_to_string(&config_path)
        )
        .map_err(|e| {
            Error::Msg(format!(
                "minimax-h3-image dit: reading {}: {e}",
                config_path.display()
            ))
        })?;
        MiniMaxH3ImageDitConfig::from_diffusers_json(&text)
    }

    /// Load `transformer_diffusers/` from a snapshot root.
    pub fn load(root: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        Self::load_dir(root.as_ref().join(IMAGE_DIT_COMPONENT), dtype)
    }

    /// Load from a resolved component directory: its `config.json` and its shards.
    pub fn load_dir(dir: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = Self::read_config(dir)?;
        let mut w = Weights::from_dir(dir)?;
        Self::from_weights(&mut w, &cfg, dtype)
    }

    /// Every tensor name the image DiT consumes — the published 644 at the shipped geometry.
    pub fn names(cfg: &MiniMaxH3ImageDitConfig) -> Vec<String> {
        let mut v = ImageDitProjections::names(cfg);
        v.extend(TokenRefiner::names("token_refiner", &cfg.base));
        for i in 0..cfg.base.num_layers {
            v.extend(DitBlock::names(&format!("transformer_blocks.{i}")));
        }
        v
    }

    /// The resident blocks, in order — empty under a deferred load.
    pub fn blocks(&self) -> &[DitBlock] {
        &self.blocks
    }

    /// The geometry in force.
    pub fn config(&self) -> &MiniMaxH3ImageDitConfig {
        &self.cfg
    }

    /// The block stack's precision.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The input/output projections.
    pub fn projections(&self) -> &ImageDitProjections {
        &self.projections
    }

    /// Blocks in the stack — the streamed depth under a deferred load.
    pub fn num_layers(&self) -> usize {
        match &self.stream {
            Some(stream) => stream.n_blocks(),
            None => self.blocks.len(),
        }
    }

    /// `context_embedder` then the token refiner: the text rows of the packed sequence.
    /// `context` is `[1, num_text_tokens, text_dim]`, the text encoder's layer-50 hidden state.
    pub fn embed_context(&self, context: &Array) -> Result<Array> {
        let s = context.shape();
        if s.len() != 3 || s[0] != 1 || s[2] != self.cfg.base.text_dim {
            return Err(Error::Msg(format!(
                "minimax-h3-image dit: the text context must be [1, num_text_tokens, {}], got {s:?}",
                self.cfg.base.text_dim
            )));
        }
        let embedded = self.projections.context_embedder.forward(context)?;
        self.refiner.forward(&embedded.as_dtype(self.dtype)?)
    }
}

/// `sigma_i - sigma_{i+1}` of the released sampler, each sigma recovered from its float32 timestep
/// (`sigma = 1 - t`, the terminal one 0), as `MiniMaxH3ImageSetTimestepsStep` builds the scheduler
/// grid. Below 0.5 that round trip moves a sigma by an ulp, so it is recomputed from the timesteps
/// rather than read off the unrounded grid.
pub fn velocity_euler_delta(timesteps: &[f32], step: usize) -> Result<f32> {
    if step >= timesteps.len() {
        return Err(Error::Msg(format!(
            "minimax-h3-image sampler: step {step} is outside the {}-step schedule",
            timesteps.len()
        )));
    }
    let sigma = |i: usize| timesteps.get(i).map_or(0.0, |&t| 1.0 - t);
    Ok(sigma(step) - sigma(step + 1))
}

/// Per sequence row, its AdaLN row class: `1` for an edit's reference rows (held at
/// [`REFERENCE_NOISE_AUG`]), `0` for every text and generated row (the step's timestep).
pub fn image_row_classes(layout: &ImageLayout) -> Vec<i32> {
    let mut classes = vec![0i32; layout.seq_len() as usize];
    let start = layout.num_text_tokens() as usize;
    let end = start + layout.num_reference_rows() as usize;
    classes[start..end].fill(1);
    classes
}

/// The image DiT's LoRA target surface (SW-7096): **every linear layer it has**, under the
/// `MiniMaxH3ImageTransformer2DModel` (diffusers) module names a PEFT/diffusers LoRA uses:
///
/// | path | module |
/// | --- | --- |
/// | `transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0}`, `….ff.net.{0.proj,2}` | block leaves |
/// | `transformer_blocks.{i}.adaln_proj.linear` | the block's AdaLN projection |
/// | `token_refiner.refiner_blocks.{i}.…` | refiner leaves (its only linears) |
/// | `proj_in`, `context_embedder` | input projections |
/// | `time_embedder.linear_1`, `time_embedder.linear_2` | the timestep MLP |
/// | `norm_out.linear` | the final AdaLN projection |
/// | `pdd_proj_out.{i}` | per-step output head `i` |
///
/// `pdd_proj_out` is an `nn.ModuleList` in the reference, so PEFT addresses each head on its own
/// and a LoRA on head `i` changes only step `i`'s output; there is no `proj_out` module, so a
/// `proj_out` target names no module and is refused, as PEFT would fail to inject it.
///
/// **The AdaLN cache stays exact.** The image denoiser enumerates its six timesteps up front and
/// precomputes every block's modulation (and `norm_out`'s) from `time_embedder` once, then evicts
/// `adaln_proj`. Adapters are installed when the DiT is mapped for a render, before
/// [`ImageDenoiser::new`] runs that precompute, so the tables are computed **with** every
/// `time_embedder` / `adaln_proj` / `norm_out` residual applied; a new adapter set means a new
/// load and therefore a new precompute.
impl AdaptableHost for MiniMaxH3ImageDit {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            // SW-7094: a deferred load holds no blocks; every block residual (the AdaLN projection
            // included — this surface admits it) lands in the stream's overlay, which re-applies
            // it to each block a window rebuilds.
            ["transformer_blocks", ..] if self.stream.is_some() => {
                self.stream.as_mut()?.adaptable_mut(path)
            }
            ["transformer_blocks", i, "adaln_proj", "linear"] => {
                let idx: usize = i.parse().ok()?;
                self.blocks.get_mut(idx)?.adaln_linear_mut()
            }
            ["transformer_blocks", i, rest @ ..] => {
                let idx: usize = i.parse().ok()?;
                self.blocks.get_mut(idx)?.adaptable_mut(rest)
            }
            ["token_refiner", rest @ ..] => self.refiner.adaptable_mut(rest),
            ["proj_in"] => self.projections.proj_in.adaptable_mut(),
            ["context_embedder"] => self.projections.context_embedder.adaptable_mut(),
            ["time_embedder", linear] => self
                .projections
                .time_embedder
                .linear_mut(linear)?
                .adaptable_mut(),
            ["norm_out", "linear"] => self.projections.norm_out.linear_mut()?.adaptable_mut(),
            ["pdd_proj_out", i] => {
                let idx: usize = i.parse().ok()?;
                self.projections.pdd_heads.head_mut(idx)?.adaptable_mut()
            }
            _ => None,
        }
    }
}

/// Every linear of the image DiT is addressable, so an unmatched target is reported as naming no
/// module (see [`crate::adapters::AdaLnSurface::ImageAllLinears`]).
impl crate::adapters::MiniMaxH3AdapterHost for MiniMaxH3ImageDit {
    fn adapter_surface(&self) -> crate::adapters::AdaLnSurface {
        crate::adapters::AdaLnSurface::ImageAllLinears
    }
}

/// Every module path an adapter can address on the image DiT at `cfg` (SW-7096).
pub fn image_adapter_target_paths(cfg: &MiniMaxH3ImageDitConfig) -> Vec<String> {
    let mut v = crate::adapters::adapter_target_paths(&cfg.base);
    v.extend((0..cfg.base.num_layers).map(|i| format!("transformer_blocks.{i}.adaln_proj.linear")));
    v.extend(
        [
            "proj_in",
            "context_embedder",
            "time_embedder.linear_1",
            "time_embedder.linear_2",
            "norm_out.linear",
        ]
        .map(String::from),
    );
    v.extend((0..cfg.num_pdd_heads).map(|i| format!("pdd_proj_out.{i}")));
    v
}

/// The image DiT as the fixed 6-step sampler's velocity model: everything constant across a
/// request (refined text rows, rotary tables, the whole run's modulation and the per-step gather
/// indices) built once, so a step is "project, concat, 50 blocks, norm, head `i`".
pub struct ImageDenoiser {
    dit: MiniMaxH3ImageDit,
    layout: ImageLayout,
    tables: MmRopeTables,
    text_rows: Array,
    cache: AdaLnCache,
    norm_out: NormOutModulation,
    schedule: SigmaSchedule,
    /// Per step: `[seq_len]` `global_timestep · 3 + tag`, what the blocks gather with.
    adaln_indices: Vec<Array>,
    /// Per step: `[seq_len]` bare timestep index, what `norm_out` gathers with.
    timestep_indices: Vec<Array>,
    released_bytes: usize,
    /// Rung 4 (SW-7094): `Some` over a deferred DiT — the 50 blocks are rebuilt from the staged
    /// tier one window at a time on every step. Set by [`Self::new_windowed`] only.
    block_window: Option<BlockPlan>,
    window_cancel: CancelFlag,
    /// Rung 3's measurement seam (SW-7094): `UNBOUNDED` on every shipped render, which is
    /// byte-identical to the pre-rung forward. The image-geometry verdict is taken through it
    /// (`tests/h3_image_bounded_attention_real.rs`).
    attention_budget: AttentionBudget,
    attention_axis: AttentionChunkAxis,
}

/// What [`ImageDenoiser`] shares between its resident and windowed constructors: the checks, the
/// text rows, the rotary tables and the per-step gather indices.
struct DenoiserParts {
    text_rows: Array,
    tables: MmRopeTables,
    adaln_schedule: TimestepSchedule,
    adaln_indices: Vec<Array>,
    timestep_indices: Vec<Array>,
}

fn denoiser_parts(
    dit: &mut MiniMaxH3ImageDit,
    layout: &ImageLayout,
    context: &Array,
    schedule: &SigmaSchedule,
) -> Result<DenoiserParts> {
    let steps = schedule.num_evals();
    if steps != dit.projections.pdd_heads.len() {
        return Err(Error::Msg(format!(
            "minimax-h3-image dit: a {steps}-step schedule against a checkpoint distilled to {} \
             steps (one output head per step)",
            dit.projections.pdd_heads.len()
        )));
    }
    if layout.patch() != dit.cfg.base.patch_size {
        return Err(Error::Msg(format!(
            "minimax-h3-image dit: the layout was packed with patch {:?}, the DiT expects {:?}",
            layout.patch(),
            dit.cfg.base.patch_size
        )));
    }
    UnverifiedSources::verify(&mut dit.unverified.head)?;
    let text_rows = dit.embed_context(context)?;
    if text_rows.shape()[1] != layout.num_text_tokens() {
        return Err(Error::Msg(format!(
            "minimax-h3-image dit: the context carries {} rows but the layout declares {} text \
             rows",
            text_rows.shape()[1],
            layout.num_text_tokens()
        )));
    }
    let tables = dit.rope.tables(layout.position_ids())?;

    // Row class 0 is the step's timestep, which every text and generated row sits at; an
    // edit's reference rows are class 1, held at `REFERENCE_NOISE_AUG` for every step
    // (`MiniMaxH3ImageSetTimestepsStep.row_timestep_plan`).
    let has_references = layout.num_reference_rows() > 0;
    let adaln_schedule = TimestepSchedule::new(
        schedule
            .timesteps()
            .iter()
            .map(|&t| {
                if has_references {
                    vec![t, REFERENCE_NOISE_AUG]
                } else {
                    vec![t]
                }
            })
            .collect(),
    )?;
    let row_classes = Array::from_slice(&image_row_classes(layout), &[layout.seq_len()]);
    let mut adaln_indices = Vec::with_capacity(steps);
    let mut timestep_indices = Vec::with_capacity(steps);
    for step in 0..steps {
        let idx = adaln_schedule.adaln_indices(step, &row_classes, layout.token_tags())?;
        timestep_indices.push(AdaLayerNormOut::timestep_indices_from_adaln(&idx)?);
        adaln_indices.push(idx);
    }
    Ok(DenoiserParts {
        text_rows,
        tables,
        adaln_schedule,
        adaln_indices,
        timestep_indices,
    })
}

impl ImageDenoiser {
    /// Build the velocity model for one request. Consumes the DiT because the AdaLN eviction
    /// mutates it irreversibly.
    ///
    /// `schedule` is the sampler's grid — [`SigmaSchedule::h3_image_distilled`] for the shipped
    /// checkpoint. Its step count must equal the number of PDD heads: a head per step is the
    /// checkpoint's contract, and a mismatch would either run a step with no head or leave heads
    /// the distillation trained unused.
    pub fn new(
        mut dit: MiniMaxH3ImageDit,
        layout: ImageLayout,
        context: &Array,
        schedule: SigmaSchedule,
    ) -> Result<Self> {
        if dit.is_deferred() {
            return Err(Error::Msg(
                "minimax-h3-image dit: ImageDenoiser::new needs a resident load; a deferred \
                 (rung 4) DiT is driven through ImageDenoiser::new_windowed"
                    .into(),
            ));
        }
        let parts = denoiser_parts(&mut dit, &layout, context, &schedule)?;

        // Project the whole run's modulation once, then release `adaln_proj`. `norm_out` reads the
        // SAME `temb` the blocks were projected from.
        UnverifiedSources::verify(&mut dit.unverified.adaln)?;
        let embedder = dit.projections.time_embedder.clone();
        let mut captured: Option<Array> = None;
        let (cache, released_bytes) = AdaLnCache::precompute_and_evict(
            &mut dit.blocks,
            parts.adaln_schedule,
            AdaLnResidency::PrecomputeAndEvict,
            |ts| {
                let temb = embedder.forward(ts)?;
                captured = Some(temb.clone());
                Ok(temb)
            },
        )?;
        let temb = captured.ok_or_else(|| {
            Error::Msg(
                "minimax-h3-image dit: the AdaLN precompute did not embed the timesteps".into(),
            )
        })?;
        let norm_out = dit.projections.norm_out.modulation(&temb)?;
        mlx_rs::transforms::eval([&norm_out.shift, &norm_out.scale])?;
        UnverifiedSources::verify(&mut dit.unverified.body)?;

        Ok(Self {
            dit,
            layout,
            tables: parts.tables,
            text_rows: parts.text_rows,
            cache,
            norm_out,
            schedule,
            adaln_indices: parts.adaln_indices,
            timestep_indices: parts.timestep_indices,
            released_bytes,
            block_window: None,
            window_cancel: CancelFlag::default(),
            attention_budget: AttentionBudget::UNBOUNDED,
            attention_axis: AttentionChunkAxis::Heads,
        })
    }

    /// The velocity model over a **deferred** DiT with a bounded block window — rung 4 (SW-7094).
    ///
    /// The two differences from [`Self::new`] are forced by the residency:
    ///
    /// * the AdaLN modulation is projected through H3's
    ///   [`crate::block_stream::precompute_adaln_windowed`], holding `window` projections at a time,
    ///   instead of `AdaLnCache::precompute_and_evict` — a deferred load never held the projections,
    ///   so nothing is evicted and [`Self::released_adaln_bytes`] is `0`;
    /// * every step rebuilds the blocks from the staged tier, `window` at a time, through
    ///   [`DitBlockStream::materialize`], which re-applies the configured LoRA residuals to each
    ///   rebuilt block (SW-7096).
    ///
    /// The per-step heads, `norm_out`, `proj_in`, `context_embedder`, `time_embedder` and the
    /// refiner stay resident (see [`MiniMaxH3ImageDit::load_dir_deferred`]).
    pub fn new_windowed(
        mut dit: MiniMaxH3ImageDit,
        layout: ImageLayout,
        context: &Array,
        schedule: SigmaSchedule,
        window: usize,
        cancel: CancelFlag,
    ) -> Result<Self> {
        let stream = dit
            .stream()
            .ok_or_else(|| {
                Error::Msg(
                    "minimax-h3-image dit: ImageDenoiser::new_windowed needs a deferred load; use \
                     MiniMaxH3ImageDit::load_dir_deferred"
                        .into(),
                )
            })?
            .clone();
        let plan = stream.plan(window)?;
        let parts = denoiser_parts(&mut dit, &layout, context, &schedule)?;
        let temb = dit
            .projections
            .time_embedder
            .forward(parts.adaln_schedule.distinct_timesteps())?;
        mlx_rs::transforms::eval([&temb])?;
        let cache =
            precompute_adaln_windowed(&stream, &plan, &cancel, parts.adaln_schedule, &temb)?;
        let norm_out = dit.projections.norm_out.modulation(&temb)?;
        mlx_rs::transforms::eval([&norm_out.shift, &norm_out.scale])?;
        Ok(Self {
            dit,
            layout,
            tables: parts.tables,
            text_rows: parts.text_rows,
            cache,
            norm_out,
            schedule,
            adaln_indices: parts.adaln_indices,
            timestep_indices: parts.timestep_indices,
            // Nothing was released, because nothing was ever resident.
            released_bytes: 0,
            block_window: Some(plan),
            window_cancel: cancel,
            attention_budget: AttentionBudget::UNBOUNDED,
            attention_axis: AttentionChunkAxis::Heads,
        })
    }

    /// Rung 3's measurement seam: the attention plan every block runs under. Shipped renders never
    /// call this — the family declares rung 3 not applicable on MLX — and the default `UNBOUNDED`
    /// plan is the pre-rung forward byte for byte.
    pub fn set_bounded_attention(&mut self, budget: AttentionBudget, axis: AttentionChunkAxis) {
        self.attention_budget = budget;
        self.attention_axis = axis;
    }

    /// The block window in force: `Some` under rung 4.
    pub fn block_window(&self) -> Option<&BlockPlan> {
        self.block_window.as_ref()
    }

    fn bounded_attention(&self) -> BoundedAttention<'_> {
        // The shipped path is the literal pre-rung forward, not a plan that happens to be unbounded.
        if self.attention_budget == AttentionBudget::UNBOUNDED {
            return BoundedAttention::UNBOUNDED;
        }
        BoundedAttention::new(
            AttentionPlan::budgeted(self.attention_budget),
            self.attention_axis,
        )
    }

    /// The packed layout this denoiser was built for.
    pub fn layout(&self) -> &ImageLayout {
        &self.layout
    }

    /// The sampler grid.
    pub fn schedule(&self) -> &SigmaSchedule {
        &self.schedule
    }

    /// Steps the sampler runs.
    pub fn num_steps(&self) -> usize {
        self.schedule.num_evals()
    }

    /// Bytes of `adaln_proj` released by the eviction.
    pub fn released_adaln_bytes(&self) -> usize {
        self.released_bytes
    }

    /// The DiT velocity of every image row at sampling step `step`, through head `step`.
    ///
    /// `image_rows` is `[1, num_packed_image_rows, in_channels · prod(patch)]` — an edit's
    /// reference rows first, then the generated rows; the result has the same shape, float32 (the
    /// heads are float32), exactly what the reference transformer returns.
    pub fn velocity(&self, step: usize, image_rows: &Array) -> Result<Array> {
        let head = self.dit.projections.pdd_heads.select(step)?;
        let adaln = self.adaln_indices.get(step).ok_or_else(|| {
            Error::Msg(format!(
                "minimax-h3-image dit: step {step} is outside the {}-step schedule",
                self.adaln_indices.len()
            ))
        })?;
        let rows = self.layout.num_packed_image_rows();
        let want = [1, rows, self.dit.cfg.base.video_patch_dim()];
        if image_rows.shape() != want {
            return Err(Error::Msg(format!(
                "minimax-h3-image dit: expected image rows {want:?}, got {:?}",
                image_rows.shape()
            )));
        }
        // The text stream sets the packed sequence's dtype, as in the reference.
        let stream = self.text_rows.dtype();
        let image = self
            .dit
            .projections
            .proj_in
            .forward(image_rows)?
            .as_dtype(stream)?;
        let x = concatenate_axis(&[self.text_rows.clone(), image], 1)?;
        let x =
            match (&self.block_window, self.dit.stream()) {
                (None, None) => {
                    let mut x = x;
                    for (layer, block) in self.dit.blocks.iter().enumerate() {
                        x = block.forward_bounded(
                            &x,
                            self.cache.modulation(layer)?,
                            adaln,
                            &self.dit.rope,
                            &self.tables,
                            self.bounded_attention(),
                        )?;
                    }
                    x
                }
                (Some(plan), Some(stream)) => run_windowed(
                    plan,
                    &self.window_cancel,
                    x,
                    || stream.open(),
                    |mut hidden: Array, view: &mut Weights, range: std::ops::Range<usize>| {
                        for layer in range {
                            let block = stream.materialize(view, layer)?;
                            hidden = block.forward_bounded(
                                &hidden,
                                self.cache.modulation(layer)?,
                                adaln,
                                &self.dit.rope,
                                &self.tables,
                                self.bounded_attention(),
                            )?;
                            // The block drops per iteration, not per window.
                        }
                        Ok(hidden)
                    },
                    // LOAD-BEARING: the carried activation references this window's weights until it
                    // is forced (H3's `run_windowed_stack` measured 8.0 MiB with against 238.4 MiB
                    // without).
                    |hidden: &Array| mlx_rs::transforms::eval([hidden]).map_err(Into::into),
                )?,
                _ => return Err(Error::Msg(
                    "minimax-h3-image dit: the block window and the DiT's residency disagree — a \
                     window over a resident load, or a deferred load with no window"
                        .into(),
                )),
            };
        let x = self.dit.projections.norm_out.apply(
            &x,
            &self.norm_out,
            &self.timestep_indices[step],
        )?;
        let text = self.layout.num_text_tokens();
        let image = slice_axis(&x, 1, text, text + rows)?;
        head.forward(&image)
    }

    /// One sampler step: the velocity at step `step` through head `step`, then the checkpoint's
    /// `euler_update_form = "velocity"` update `x_next = x_t + (sigma - sigma_next) · v`, in float32
    /// (`scheduler/scheduler_config.json`; `MiniMaxH3Scheduler.step`).
    ///
    /// Only the generated rows are written: an edit's reference rows ride through every step
    /// unchanged (`MiniMaxH3ImageLoopSchedulerStep`).
    pub fn step(&self, step: usize, image_rows: &Array) -> Result<Array> {
        let v = self.velocity(step, image_rows)?;
        let delta = velocity_euler_delta(self.schedule.timesteps(), step)?;
        let x = image_rows.as_dtype(Dtype::Float32)?;
        let refs = self.layout.num_reference_rows();
        let all = self.layout.num_packed_image_rows();
        let generated = mlx_rs::ops::add(
            &slice_axis(&x, 1, refs, all)?,
            &mlx_rs::ops::multiply(
                &slice_axis(&v.as_dtype(Dtype::Float32)?, 1, refs, all)?,
                Array::from_f32(delta),
            )?,
        )?;
        if refs == 0 {
            return Ok(generated);
        }
        Ok(concatenate_axis(
            &[slice_axis(&x, 1, 0, refs)?, generated],
            1,
        )?)
    }

    /// The generated rows of a packed `[1, num_packed_image_rows, ·]` image stream.
    pub fn generated_rows(&self, packed: &Array) -> Result<Array> {
        slice_axis(
            packed,
            1,
            self.layout.num_reference_rows(),
            self.layout.num_packed_image_rows(),
        )
    }

    /// The whole fixed-step loop over the packed image rows (an edit's reference rows first),
    /// forcing each step and checking `cancel` between steps. Returns the packed rows.
    pub fn denoise(
        &self,
        noise_rows: &Array,
        cancel: &mlx_gen::CancelFlag,
        on_step: &mut dyn FnMut(usize),
    ) -> Result<Array> {
        let mut x = noise_rows.as_dtype(Dtype::Float32)?;
        for step in 0..self.num_steps() {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            x = self.step(step, &x)?;
            mlx_rs::transforms::eval([&x])?;
            on_step(step + 1);
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The declared name set partitions the published `transformer_diffusers` index exactly: 644
    /// names, no audio tensor, no H3 `proj_out`, and `pdd_proj_out.{0..6}`.
    #[test]
    fn the_image_dit_names_are_the_published_644() {
        let cfg = MiniMaxH3ImageDitConfig::default();
        let names = MiniMaxH3ImageDit::names(&cfg);
        assert_eq!(names.len(), PUBLISHED_IMAGE_DIT_TENSORS);
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate names");
        assert!(!names.iter().any(|n| n.contains("audio")));
        assert!(names.iter().any(|n| n == "pdd_proj_out.5.weight"));
        assert!(names
            .iter()
            .any(|n| n == "transformer_blocks.49.adaln_proj.linear.weight"));
    }

    /// The six deltas sum to the whole unit interval, and step 0 starts at sigma 1.
    #[test]
    fn velocity_euler_deltas_walk_the_published_grid() {
        let sch = SigmaSchedule::h3_image_distilled().unwrap();
        let t = sch.timesteps();
        let deltas: Vec<f32> = (0..6)
            .map(|i| velocity_euler_delta(t, i).unwrap())
            .collect();
        assert!(deltas.iter().all(|d| *d > 0.0));
        assert!((deltas.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((deltas[0] - (1.0 - 0.967_523_7)).abs() < 1e-6);
        assert!((deltas[5] - 0.232_008_88).abs() < 1e-6);
        assert!(velocity_euler_delta(t, 6).is_err());
    }

    /// H3 video's published set is untouched: still 638, still including both audio heads.
    #[test]
    fn the_video_dit_name_set_is_unchanged() {
        let names = crate::dit::MiniMaxH3Dit::names(&crate::dit::MiniMaxH3DitConfig::default());
        assert_eq!(names.len(), crate::dit::PUBLISHED_DIT_TENSORS);
        assert!(names.iter().any(|n| n == "audio_proj_out.weight"));
    }

    /// An edit's reference rows are the only class-1 rows.
    #[test]
    fn reference_rows_are_the_held_row_class() {
        let layout = ImageLayout::edit(&[1, 0, 1], &[(2, 4)], 2, 2, [1, 2, 2]).unwrap();
        assert_eq!(image_row_classes(&layout), vec![0, 0, 0, 1, 1, 0]);
        let t2i = ImageLayout::t2i(2, 2, 2, [1, 2, 2]).unwrap();
        assert_eq!(image_row_classes(&t2i), vec![0, 0, 0]);
    }
}
