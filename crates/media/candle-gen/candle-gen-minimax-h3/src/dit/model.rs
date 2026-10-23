//! The **whole** `MiniMaxH3Transformer3DModel`: the 17 input/output projections of
//! [`crate::dit::heads`] wrapped around the block stack, the token refiner and the MM-RoPE, plus the
//! [`crate::denoise::JointVelocity`] implementation the joint loop drives.
//!
//! ```text
//! proj_in(video rows) ─┐
//! audio_proj_in(audio) ─┼─► scatter ─► 50 × DitBlock ─► norm_out ─┬─► proj_out       ─► video velocity
//! context_embedder ─►   │                                          └─► audio_proj_out ─► audio velocity
//!   token_refiner ─────┘
//! ```
//!
//! # The text stream is projected once per request, not once per step
//!
//! The reference runs `context_embedder` and the token refiner inside every `forward`, but both are
//! pure functions of the text context, which is constant across the whole denoise. [`JointDit`]
//! therefore evaluates them at construction and reuses the rows — identical arithmetic, 2 refiner
//! blocks × `num_evals` fewer. It is recorded here rather than left implicit because the reference's
//! call site makes it *look* per-step.
//!
//! # Two index tensors, not one
//!
//! The block stack gathers its modulation with `timestep_index · MODALITY_NUM + token_tag`;
//! `norm_out` gathers its shift/scale with the **bare timestep index**. Both are derived here from
//! the single `adaln_indices` the loop supplies (see
//! [`crate::dit::heads::AdaLayerNormOut::timestep_indices_from_adaln`]), so no caller can hand the
//! two halves of the model inconsistent row addressing.
//!
//! # Gather-then-project, and why that is the same model
//!
//! The reference runs both output heads over **every** row of the packed sequence and then
//! `index_select`s each modality's rows out. Both heads are row-wise affine maps, so selecting first
//! and projecting second is bit-identical arithmetic on the rows that survive and simply does not
//! compute the text rows' outputs, which the reference discards. At the shipped geometry that is a
//! few hundred rows of 5376 → 96 saved, not a numerical change.
//!
//! # Memory discipline on the load path
//!
//! [`MiniMaxH3Dit::load`] reads `transformer/` at the **stored** dtype and casts only the block
//! stack, because twelve of the 17 top-level tensors ship float32 and a uniform cast would round the
//! timestep MLP every AdaLN projection in the model reads. The consequence for a caller is that the
//! `Weights` map and the built model are alive at the same time, and candle tensors share storage
//! through an `Arc` — so the map must be **dropped before** the AdaLN eviction, or the 26.02 GB does
//! not go anywhere. [`crate::dit::adaln`] documents that trap; `load` closes it by construction, by
//! owning the map and dropping it before returning.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::{CandleError, Result, Weights};

use crate::denoise::{JointStep, JointVelocity, PackedLayout};
use crate::dit::adaln::{AdaLnCache, AdaLnResidency, TimestepSchedule};
use crate::dit::block::DitBlock;
use crate::dit::config::{MiniMaxH3DitConfig, MODALITY_NUM};
use crate::dit::heads::{AdaLayerNormOut, DitProjections, NormOutModulation};
use crate::dit::layers::LinearNoBias;
use crate::dit::refiner::TokenRefiner;
use crate::dit::rope::{MmRope, MmRopeTables};

/// Tensors the published `transformer/` partition carries: `50 · 12 + 21 + 17`.
pub const PUBLISHED_DIT_TENSORS: usize = 638;

/// The full DiT — every tensor of `transformer/`.
#[derive(Debug, Clone)]
pub struct MiniMaxH3Dit {
    cfg: MiniMaxH3DitConfig,
    projections: DitProjections,
    refiner: TokenRefiner,
    blocks: Vec<DitBlock>,
    rope: MmRope,
    dtype: DType,
    device: Device,
}

impl MiniMaxH3Dit {
    /// Build from an already-populated [`Weights`] map.
    ///
    /// `dtype` is the **block stack's** precision. The 17 input/output tensors are loaded at their
    /// own stored dtypes instead — twelve of them ship float32 — see [`crate::dit::heads`].
    ///
    /// **The caller keeps the map alive.** If the eviction is going to run, drop it first; see the
    /// module docs.
    pub fn from_weights(
        w: &Weights,
        cfg: &MiniMaxH3DitConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        cfg.validate()?;
        let projections = DitProjections::from_weights(w, cfg)?;
        let refiner = TokenRefiner::from_weights(w, "token_refiner", cfg, dtype)?;
        let blocks = (0..cfg.num_layers)
            .map(|i| DitBlock::from_weights(w, &format!("transformer_blocks.{i}"), cfg, dtype))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg: cfg.clone(),
            projections,
            refiner,
            blocks,
            rope: MmRope::new(cfg.rope_freq_dim, cfg.rope_theta)?,
            dtype,
            device: device.clone(),
        })
    }

    /// Load `transformer/` (or `transformer_ref/`) from a snapshot root: its `config.json` and its
    /// shards.
    ///
    /// The shards are read at their **stored** dtype (`DType::F32` is not forced on the mixed
    /// float32/bfloat16 top-level tensors), and the loader map is dropped before returning so that
    /// no storage-sharing clone outlives this call — the precondition
    /// [`AdaLnCache::precompute_and_evict`] needs.
    /// **Flat-layout convenience.** A staged tier's partition is not under the snapshot root
    /// (sc-20267 — `crate::tier` resolves it from the caller's component override), so the render path
    /// goes through [`Self::load_from_dir`] with the directory it resolved. This wrapper is the
    /// `root` + `partition` join for callers that genuinely hold a flat snapshot.
    pub fn load(
        root: impl AsRef<Path>,
        partition: &str,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::load_from_dir(root.as_ref().join(partition), device, dtype)
    }

    /// Load a **resolved** DiT partition directory: its `config.json` and its shards.
    ///
    /// The tier-aware entry point. `dir` is whatever `crate::tier::MiniMaxH3TierPaths` resolved for
    /// this partition, which on a split install is outside the snapshot root entirely. Whether the
    /// weights inside are packed or dense is *not* a parameter — `crate::quant::lin` auto-detects it
    /// per tensor from the `{base}.scales` sibling, so one loader serves every tier.
    pub fn load_from_dir(dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = dir.as_ref();
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|e| {
            CandleError::Msg(format!(
                "minimax-h3 dit: reading {}: {e}",
                config_path.display()
            ))
        })?;
        let cfg = MiniMaxH3DitConfig::from_diffusers_json(&text)?;
        let shards = candle_gen::loader::sorted_safetensors(dir, "minimax-h3 dit")?;
        let w = Weights::from_files_native_int8(&shards, device, dtype)?;
        let model = Self::from_weights(&w, &cfg, device, dtype)?;
        // Drop the map here rather than at the call site: every tensor in it shares storage with
        // the model's, and a surviving clone is exactly what makes an eviction free nothing.
        drop(w);
        Ok(model)
    }

    /// Every tensor name the DiT consumes — the published 638 at the shipped geometry.
    ///
    /// The exhaustive mapping: any `transformer/` tensor outside this set would be silently ignored,
    /// which is the failure mode this list exists to make testable.
    pub fn names(cfg: &MiniMaxH3DitConfig) -> Vec<String> {
        let mut v = DitProjections::names();
        v.extend(TokenRefiner::names("token_refiner", cfg));
        for i in 0..cfg.num_layers {
            v.extend(DitBlock::names(&format!("transformer_blocks.{i}")));
        }
        v
    }

    /// The geometry in force.
    pub fn config(&self) -> &MiniMaxH3DitConfig {
        &self.cfg
    }

    /// The block stack's precision.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device this model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The 17 input/output projections.
    pub fn projections(&self) -> &DitProjections {
        &self.projections
    }

    /// The rotary this model was built with.
    pub fn rope(&self) -> &MmRope {
        &self.rope
    }

    /// Blocks loaded.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Whether every block still holds its AdaLN projection.
    pub fn holds_adaln(&self) -> bool {
        self.blocks.iter().all(DitBlock::holds_adaln)
    }

    /// **The adapter addressing seam** (sc-18728) — resolve a dotted module path, pre-split into
    /// segments, to the projection a LoRA residual attaches to.
    ///
    /// The candle twin of `mlx_gen::adapters::AdaptableHost::adaptable_mut`. It takes a segment
    /// slice rather than a dotted string because `to_out.0` and `ff.net.0.proj` carry
    /// `nn.Sequential` indices that are genuine path segments, so a naive rsplit on `.` would
    /// mis-parse them.
    ///
    /// Reachable: `transformer_blocks.{i}.{attn.to_q|attn.to_k|attn.to_v|attn.to_out.0|
    /// ff.net.0.proj|ff.net.2}` and the same six under `token_refiner.refiner_blocks.{i}`. Nothing
    /// else — in particular not `adaln_proj`, not the 17 input/output projections, and not any norm.
    /// See [`crate::adapters::adapter_target_paths`], which enumerates exactly this set from the
    /// config and is asserted to resolve here.
    pub fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut LinearNoBias> {
        match path {
            ["transformer_blocks", idx, rest @ ..] => {
                let i: usize = idx.parse().ok()?;
                self.blocks.get_mut(i)?.adaptable_mut(rest)
            }
            ["token_refiner", rest @ ..] => self.refiner.adaptable_mut(rest),
            _ => None,
        }
    }

    /// Drop every LoRA residual installed anywhere in the model, reverting it to the bare
    /// checkpoint.
    pub fn clear_adapters(&mut self) {
        for b in &mut self.blocks {
            b.clear_adapters();
        }
        self.refiner.clear_adapters();
    }

    /// How many projections currently carry at least one residual — the count a caller checks
    /// against [`crate::adapters::MiniMaxH3LoraReport::applied`].
    pub fn adapted_module_count(&mut self) -> usize {
        let cfg = self.cfg.clone();
        crate::adapters::adapter_target_paths(&cfg)
            .iter()
            .filter(|p| {
                let segs: Vec<&str> = p.split('.').collect();
                self.adaptable_mut(&segs).is_some_and(|l| l.is_adapted())
            })
            .count()
    }

    /// Bytes of `adaln_proj` weight the blocks currently hold — the eviction's headline number.
    pub fn adaln_nbytes(&self) -> usize {
        self.blocks
            .iter()
            .filter_map(DitBlock::adaln_proj)
            .map(|p| p.nbytes())
            .sum()
    }

    /// `context_embedder` then the token refiner — the text rows of the packed sequence.
    ///
    /// `context` is `[1, num_text_tokens, text_dim]`, the text encoder's select-layer hidden state.
    pub fn embed_context(&self, context: &Tensor) -> Result<Tensor> {
        let s = context.dims();
        if s.len() != 3 || s[0] != 1 || s[2] != self.cfg.text_dim {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit: the text context must be [1, num_text_tokens, {}], got {s:?}",
                self.cfg.text_dim
            )));
        }
        let embedded = self.projections.context_embedder.forward(context)?;
        self.refiner.forward(&embedded.to_dtype(self.dtype)?)
    }

    /// The timestep MLP, as the closure [`AdaLnCache::precompute`] consumes.
    pub fn embed_timesteps(&self, timesteps: &[f32]) -> Result<Tensor> {
        self.projections
            .time_embedder
            .forward(timesteps, &self.device)
    }

    /// **One whole model evaluation**, in the reference's own argument shape.
    ///
    /// This is `MiniMaxH3Transformer3DModel.forward` minus the parts a caller already owns: the text
    /// rows arrive already embedded and refined ([`Self::embed_context`], constant across a run) and
    /// the modulation arrives already projected. Everything else — the two patch projections, the
    /// scatter, the 50 blocks, `norm_out` and the two output heads — happens here, so
    /// `tests/dit_io.rs` can compare it against the reference's own whole-model golden and
    /// [`JointDit`] is a thin adapter over it rather than a second implementation.
    ///
    /// Taking the three index tensors as slices rather than a [`PackedLayout`] is what makes that
    /// comparison possible at all: the reference's fixture is dumped at a tiny geometry that
    /// [`crate::denoise::JointGeometry`] correctly refuses to build (3 latent frames is not `5n + 2`).
    pub fn forward_packed(
        &self,
        packed: &PackedForward<'_>,
        blocks: BlockModulation<'_>,
        norm_out: &NormOutModulation,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_packed_veda(packed, blocks, norm_out, None)
    }

    fn forward_packed_veda(
        &self,
        packed: &PackedForward<'_>,
        blocks: BlockModulation<'_>,
        norm_out: &NormOutModulation,
        veda: Option<(&crate::veda::VedaRuntime, usize)>,
    ) -> Result<(Tensor, Tensor)> {
        // 1. Per-modality input projections. The two patch projections are float32 in the published
        //    checkpoint; the text stream sets the packed sequence's dtype, as in the reference.
        let stream = packed.text_rows.dtype();
        let video = self
            .projections
            .proj_in
            .forward(packed.video_rows)?
            .to_dtype(stream)?;
        let audio = self
            .projections
            .audio_proj_in
            .forward(packed.audio_rows)?
            .to_dtype(stream)?;
        let mut x = scatter_rows(
            packed.text_rows,
            &video,
            &audio,
            packed.text_indices,
            packed.video_indices,
            packed.audio_indices,
        )?;

        // 2. The stack. The AdaLN index bounds-check runs ONCE here rather than per block: it is a
        //    blocking D2H readback, and all 50 blocks gather with the same index tensor against
        //    same-shaped tables, so the per-block placement repeated the stall 50× per step for
        //    one answer.
        let modulation_rows = match blocks {
            BlockModulation::Cached(cache) => cache.modulation(0)?.scale_msa.dims()[0],
            // One table row block per timestep row of `temb` — `AdaLnProjection::forward` rejects
            // a non-rank-2 `temb` later, so a missing first dim here just defers to that error.
            BlockModulation::Temb(temb) => temb.dims().first().copied().unwrap_or(0) * MODALITY_NUM,
        };
        crate::dit::block::check_adaln_indices(packed.adaln_indices, modulation_rows)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            x = match blocks {
                BlockModulation::Cached(cache) => block.forward_veda(
                    &x,
                    cache.modulation(layer)?,
                    packed.adaln_indices,
                    &self.rope,
                    packed.tables,
                    veda.map(|(runtime, step)| (runtime, layer, step)),
                )?,
                BlockModulation::Temb(temb) => block.forward_veda(
                    &x,
                    &block.modulation(temb)?,
                    packed.adaln_indices,
                    &self.rope,
                    packed.tables,
                    veda.map(|(runtime, step)| (runtime, layer, step)),
                )?,
            };
        }

        // 3. The shared output norm, then the two heads over their own rows. Gathering before
        //    projecting is bit-identical to the reference's project-then-gather (both heads are
        //    row-wise affine maps) and skips the text rows' outputs, which it discards.
        let x = self
            .projections
            .norm_out
            .apply(&x, norm_out, packed.timestep_indices)?;
        let seq_len = x.dims()[1];
        let video_out =
            self.projections
                .proj_out
                .forward(&crate::denoise::packing::gather_rows(
                    &x,
                    packed.video_indices,
                    seq_len,
                )?)?;
        let audio_out =
            self.projections
                .audio_proj_out
                .forward(&crate::denoise::packing::gather_rows(
                    &x,
                    packed.audio_indices,
                    seq_len,
                )?)?;
        Ok((video_out, audio_out))
    }
}

/// Everything one [`MiniMaxH3Dit::forward_packed`] needs besides the weights and the modulation.
#[derive(Debug)]
pub struct PackedForward<'a> {
    /// `[1, video_indices.len(), in_channels · prod(patch)]`, conditioning rows first.
    pub video_rows: &'a Tensor,
    /// `[1, audio_indices.len(), audio_in_channels]`, channel-major.
    pub audio_rows: &'a Tensor,
    /// `[1, text_indices.len(), hidden_size]` — already `context_embedder` + `token_refiner`.
    pub text_rows: &'a Tensor,
    /// `[seq_len]` — `timestep_index · MODALITY_NUM + token_tag`, what the blocks gather with.
    pub adaln_indices: &'a Tensor,
    /// `[seq_len]` — the **bare** timestep index, what `norm_out` gathers with.
    pub timestep_indices: &'a Tensor,
    /// MM-RoPE tables over the packed sequence's position grid.
    pub tables: &'a MmRopeTables,
    /// Rows the text condition occupies.
    pub text_indices: &'a [u32],
    /// Rows the video stream occupies — conditioning rows first, and **discontiguous**.
    pub video_indices: &'a [u32],
    /// Rows the audio stream occupies.
    pub audio_indices: &'a [u32],
}

/// Where a [`MiniMaxH3Dit::forward_packed`] takes its per-block modulation from.
#[derive(Debug, Clone, Copy)]
pub enum BlockModulation<'a> {
    /// The precomputed table the eviction leaves behind.
    Cached(&'a AdaLnCache),
    /// A live timestep embedding, projected per block — the resident path.
    Temb(&'a Tensor),
}

/// Scatter `concat(text, video, audio)` into one `[1, seq_len, F]` packed sequence.
///
/// The same single-gather-through-an-inverse-permutation [`PackedLayout::scatter`] performs,
/// expressed over the three index **slices** so a fixture geometry that
/// [`crate::denoise::JointGeometry`] refuses to build can still be driven. The duplicate/coverage
/// checks are kept: a row claimed twice, or not at all, is an error rather than a sequence with a
/// stale value in it.
fn scatter_rows(
    text: &Tensor,
    video: &Tensor,
    audio: &Tensor,
    text_indices: &[u32],
    video_indices: &[u32],
    audio_indices: &[u32],
) -> Result<Tensor> {
    let seq_len = text_indices.len() + video_indices.len() + audio_indices.len();
    let mut inverse = vec![u32::MAX; seq_len];
    for (source, &dest) in text_indices
        .iter()
        .chain(video_indices)
        .chain(audio_indices)
        .enumerate()
    {
        if dest as usize >= seq_len {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit: row {dest} is outside [0, {seq_len})"
            )));
        }
        if inverse[dest as usize] != u32::MAX {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit: row {dest} is claimed by two index tensors"
            )));
        }
        inverse[dest as usize] = source as u32;
    }
    if inverse.contains(&u32::MAX) {
        return Err(CandleError::Msg(
            "minimax-h3 dit: the three index tensors do not cover every row".into(),
        ));
    }
    for (block, rows, what) in [
        (text, text_indices.len(), "text"),
        (video, video_indices.len(), "video"),
        (audio, audio_indices.len(), "audio"),
    ] {
        let s = block.dims();
        if s.len() != 3 || s[0] != 1 || s[1] != rows {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit: the {what} block must be [1, {rows}, hidden], got {s:?}"
            )));
        }
    }
    let sources = Tensor::cat(&[text, video, audio], 1)?.contiguous()?;
    let perm = Tensor::from_vec(inverse, (seq_len,), text.device())?;
    Ok(sources.index_select(&perm, 1)?)
}

/// How a [`JointDit`] sources its per-block modulation.
#[derive(Debug, Clone)]
enum Modulation {
    /// The whole schedule projected up front and `adaln_proj` released — the 26.02 GB lever.
    Cached(Box<AdaLnCache>),
    /// The projections kept loaded and re-run per step from the four row-class timesteps.
    Resident,
}

/// The DiT as the joint denoise loop's velocity model.
///
/// Holds everything that is constant across a request — the rotary tables, the refined text rows,
/// the modulation source and the two per-row index tensors — so a step is exactly "project, scatter,
/// 50 blocks, norm, two heads".
pub struct JointDit {
    dit: MiniMaxH3Dit,
    layout: PackedLayout,
    tables: MmRopeTables,
    text_rows: Tensor,
    modulation: Modulation,
    /// `Some` under [`Modulation::Cached`] — the whole schedule's `norm_out` shift/scale, projected
    /// from the same `temb` the block stack's table was. `None` under [`Modulation::Resident`],
    /// where the table moves every step and is rebuilt inside `forward`.
    norm_out_modulation: Option<NormOutModulation>,
    /// Non-`None` only in [`Modulation::Resident`]: the per-request local AdaLN index, which is
    /// `row_class · MODALITY_NUM + token_tag` rather than the global schedule's row.
    resident_adaln_indices: Option<Tensor>,
    /// Non-`None` only in [`Modulation::Resident`]: the per-row class index `norm_out` gathers with.
    resident_timestep_indices: Option<Tensor>,
    /// Bytes of `adaln_proj` weight released by the eviction, 0 when resident.
    released_bytes: usize,
    forwards: usize,
    veda: Option<crate::veda::VedaRuntime>,
}

impl JointDit {
    /// Build the velocity model for one request.
    ///
    /// * `context` — `[1, num_text_tokens, text_dim]` from the text encoder;
    /// * `schedule` — the run's [`TimestepSchedule`], from [`crate::denoise::adaln_schedule`];
    /// * `residency` — [`AdaLnResidency::PrecomputeAndEvict`] projects the whole schedule up front
    ///   and releases 26.02 GB of `adaln_proj`; [`AdaLnResidency::Resident`] keeps them and projects
    ///   the four row-class timesteps per step.
    ///
    /// Consumes the model because the eviction mutates it irreversibly.
    pub fn new(
        mut dit: MiniMaxH3Dit,
        layout: PackedLayout,
        context: &Tensor,
        schedule: TimestepSchedule,
        residency: AdaLnResidency,
    ) -> Result<Self> {
        if schedule.num_row_classes() != crate::denoise::NUM_ROW_CLASSES {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit: the schedule declares {} row classes, expected {}",
                schedule.num_row_classes(),
                crate::denoise::NUM_ROW_CLASSES
            )));
        }
        let text_rows = dit.embed_context(context)?;
        if text_rows.dims()[1] != layout.num_text_tokens() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 dit: the context carries {} rows but the layout declares {} text rows",
                text_rows.dims()[1],
                layout.num_text_tokens()
            )));
        }
        let tables = dit.rope.tables(layout.position_ids())?;
        let device = dit.device.clone();

        let (modulation, norm_out_modulation, resident_adaln, resident_ts, released_bytes) =
            match residency {
                AdaLnResidency::PrecomputeAndEvict => {
                    // The timestep embedding is captured out of the closure so `norm_out`'s own
                    // table is projected from the SAME `temb` the block stack's was — a second
                    // `embed_timesteps` call would be a second chance to bind the wrong timesteps.
                    let mut captured: Option<Tensor> = None;
                    let embedder = dit.projections.time_embedder.clone();
                    let embed_device = device.clone();
                    let (cache, released) = AdaLnCache::precompute_and_evict(
                        &mut dit.blocks,
                        schedule,
                        residency,
                        &device,
                        |timesteps| {
                            let temb = embedder.forward(timesteps, &embed_device)?;
                            captured = Some(temb.clone());
                            Ok(temb)
                        },
                    )?;
                    let temb = captured.ok_or_else(|| {
                        CandleError::Msg(
                            "minimax-h3 dit: the AdaLN precompute did not evaluate the timestep \
                             embedding"
                                .into(),
                        )
                    })?;
                    let norm_out = dit.projections.norm_out.modulation(&temb)?;
                    (
                        Modulation::Cached(Box::new(cache)),
                        Some(norm_out),
                        None,
                        None,
                        released,
                    )
                }
                AdaLnResidency::Resident => {
                    // One temb row per ROW CLASS, rebuilt per step; the index tensors are therefore
                    // the local `row_class · MODALITY_NUM + tag`, which is constant across steps
                    // because neither the class nor the tag of a row ever moves.
                    let local: Vec<u32> = layout
                        .row_classes()
                        .iter()
                        .zip(layout.token_tags())
                        .map(|(&c, &t)| c * MODALITY_NUM as u32 + t)
                        .collect();
                    let seq = local.len();
                    let local = Tensor::from_vec(local, (seq,), &device)?;
                    let classes = Tensor::from_vec(layout.row_classes().to_vec(), (seq,), &device)?;
                    (Modulation::Resident, None, Some(local), Some(classes), 0)
                }
            };

        Ok(Self {
            dit,
            layout,
            tables,
            text_rows,
            modulation,
            norm_out_modulation,
            resident_adaln_indices: resident_adaln,
            resident_timestep_indices: resident_ts,
            released_bytes,
            forwards: 0,
            veda: None,
        })
    }

    pub fn prepare_veda(
        &mut self,
        request: &candle_gen::gen_core::VedaRequest,
        predictor: Option<&std::path::Path>,
        report: crate::veda::ReportSink,
    ) {
        let cfg = self.dit.config();
        self.veda = crate::veda::VedaRuntime::prepare(
            request,
            predictor,
            &self.layout,
            cfg.num_layers,
            cfg.num_attention_heads,
            cfg.attention_head_dim,
            &self.dit.device,
            report,
        );
    }

    /// Bytes of `adaln_proj` weight the eviction released — 0 under [`AdaLnResidency::Resident`].
    pub fn released_bytes(&self) -> usize {
        self.released_bytes
    }

    /// Model evaluations run so far. The loop must call [`JointVelocity::forward`] exactly once per
    /// step — the checkpoint is guidance-distilled and has no unconditional pass.
    pub fn forwards(&self) -> usize {
        self.forwards
    }

    /// The layout this model was built for.
    pub fn layout(&self) -> &PackedLayout {
        &self.layout
    }

    /// The DiT itself.
    pub fn dit(&self) -> &MiniMaxH3Dit {
        &self.dit
    }
}

impl JointVelocity for JointDit {
    fn forward(&mut self, step: &JointStep<'_>) -> Result<(Tensor, Tensor)> {
        self.forwards += 1;

        // The two row-addressing tensors. `timestep_indices` is DERIVED from `adaln_indices` rather
        // than rebuilt, so the block stack and `norm_out` cannot be handed inconsistent rows.
        let (adaln_indices, timestep_indices, resident_temb) = match &self.modulation {
            Modulation::Cached(_) => (
                step.adaln_indices.clone(),
                AdaLayerNormOut::timestep_indices_from_adaln(step.adaln_indices)?,
                None,
            ),
            Modulation::Resident => {
                let local = self.resident_adaln_indices.as_ref().ok_or_else(|| {
                    CandleError::Msg(
                        "minimax-h3 dit: resident residency without its local AdaLN index".into(),
                    )
                })?;
                let classes = self.resident_timestep_indices.as_ref().ok_or_else(|| {
                    CandleError::Msg(
                        "minimax-h3 dit: resident residency without its row-class index".into(),
                    )
                })?;
                (
                    local.clone(),
                    classes.clone(),
                    Some(self.dit.embed_timesteps(&step.row_timesteps)?),
                )
            }
        };
        // `norm_out`'s table is per-timestep, so under the resident residency it moves every step
        // and is rebuilt from that step's own four row-class timesteps.
        let norm_out = match (&resident_temb, &self.norm_out_modulation) {
            (Some(temb), _) => self.dit.projections.norm_out.modulation(temb)?,
            (None, Some(cached)) => cached.clone(),
            (None, None) => {
                return Err(CandleError::Msg(
                    "minimax-h3 dit: the cached residency has no norm_out modulation".into(),
                ))
            }
        };

        let packed = PackedForward {
            video_rows: step.video_rows,
            audio_rows: step.audio_rows,
            text_rows: &self.text_rows,
            adaln_indices: &adaln_indices,
            timestep_indices: &timestep_indices,
            tables: &self.tables,
            // The layout's OWN index tensors drive the scatter, so a wrong one produces a wrong
            // sequence rather than a coincidentally-right concatenation.
            text_indices: self.layout.text_indices(),
            video_indices: self.layout.video_indices(),
            audio_indices: self.layout.audio_indices(),
        };
        let blocks = match (&self.modulation, &resident_temb) {
            (Modulation::Cached(cache), _) => BlockModulation::Cached(cache),
            (Modulation::Resident, Some(temb)) => BlockModulation::Temb(temb),
            (Modulation::Resident, None) => {
                return Err(CandleError::Msg(
                    "minimax-h3 dit: resident residency without a timestep embedding".into(),
                ))
            }
        };
        self.dit.forward_packed_veda(
            &packed,
            blocks,
            &norm_out,
            self.veda.as_ref().map(|v| (v, step.index)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The declared name set is the published 638, partitioned into the three groups with nothing
    /// double-counted.
    #[test]
    fn the_declared_names_are_the_published_six_hundred_and_thirty_eight() {
        let cfg = MiniMaxH3DitConfig::default();
        let names = MiniMaxH3Dit::names(&cfg);
        assert_eq!(names.len(), PUBLISHED_DIT_TENSORS);
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "no name is declared twice");

        let top = names
            .iter()
            .filter(|n| !n.starts_with("transformer_blocks.") && !n.starts_with("token_refiner."))
            .count();
        let refiner = names
            .iter()
            .filter(|n| n.starts_with("token_refiner."))
            .count();
        let blocks = names
            .iter()
            .filter(|n| n.starts_with("transformer_blocks."))
            .count();
        assert_eq!((top, refiner, blocks), (17, 21, 600));
    }
}

#[cfg(test)]
mod veda_controls_tests {
    use super::*;
    use crate::denoise::{
        adaln_schedule, denoise_av, JointGeometry, JointSchedule, NUM_ROW_CLASSES, TEXT_TAG,
    };
    use crate::dit::positions::{KeyframeAnchor, ReferenceLatentGeometry};
    use crate::reference::ReferenceKind;
    use candle_gen::gen_core::{CancelFlag, VedaBudget, VedaRequest};

    fn config() -> MiniMaxH3DitConfig {
        MiniMaxH3DitConfig {
            num_attention_heads: 4,
            attention_head_dim: 24,
            hidden_size: 64,
            num_layers: 2,
            num_refiner_layers: 2,
            ffn_dim: 32,
            text_dim: 40,
            freq_dim: 16,
            time_embed_hidden_dim: 64,
            time_embed_dim: 48,
            rope_freq_dim: 2,
            ..Default::default()
        }
    }
    fn weights(bits: Option<usize>) -> Weights {
        let device = Device::Cpu;
        let mut map = candle_gen::candle_core::safetensors::load(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/dit_block.safetensors"
            ),
            &device,
        )
        .unwrap();
        if let Some(bits) = bits {
            // Native q4/q8 attention projection loaders, not a fake enum standing in for a tier.
            let lanes = 32 / bits;
            let mask = (1u32 << bits) - 1;
            let codes: Vec<u32> = (0..96 * 64 / lanes)
                .map(|i| (0..lanes).fold(0, |w, j| w | (((i + j) as u32 & mask) << (j * bits))))
                .collect();
            map.insert(
                "transformer_blocks.0.attn.to_q.weight".into(),
                Tensor::from_vec(codes, (96, 64 / lanes), &device).unwrap(),
            );
            map.insert(
                "transformer_blocks.0.attn.to_q.scales".into(),
                Tensor::full(0.002f32, (96, 1), &device).unwrap(),
            );
            map.insert(
                "transformer_blocks.0.attn.to_q.biases".into(),
                Tensor::full(-0.01f32, (96, 1), &device).unwrap(),
            );
        }
        Weights::from_map(map)
    }
    fn layout(mode: usize, channels: usize) -> PackedLayout {
        let geometry = JointGeometry::new(5, 2, 2).unwrap();
        let dev = Device::Cpu;
        let anchors = match mode {
            1 => vec![KeyframeAnchor::First],
            2 => vec![KeyframeAnchor::Last],
            3 => vec![KeyframeAnchor::First, KeyframeAnchor::Last],
            _ => vec![],
        };
        if mode < 4 {
            return PackedLayout::build(
                geometry,
                [1, 2, 2],
                &[TEXT_TAG; 5],
                channels,
                &anchors,
                &dev,
            )
            .unwrap();
        }
        // Image + clip with soundtrack + standalone audio: the normal R2VA packing mechanisms.
        let refs = [
            ReferenceLatentGeometry {
                kind: ReferenceKind::Image,
                num_latent_frames: 1,
                latent_height: 2,
                latent_width: 2,
                num_audio_latents: 0,
            },
            ReferenceLatentGeometry {
                kind: ReferenceKind::Video,
                num_latent_frames: 7,
                latent_height: 2,
                latent_width: 2,
                num_audio_latents: 37,
            },
            ReferenceLatentGeometry {
                kind: ReferenceKind::Audio,
                num_latent_frames: 0,
                latent_height: 0,
                latent_width: 0,
                num_audio_latents: 3,
            },
        ];
        PackedLayout::build_ref2va(geometry, [1, 2, 2], &[TEXT_TAG; 5], channels, &refs, &dev)
            .unwrap()
    }
    fn run(
        mode: usize,
        channels: usize,
        bits: Option<usize>,
        evals: usize,
        shift: f32,
        enabled: bool,
    ) -> (
        Vec<u32>,
        Vec<crate::veda::AttentionCall>,
        candle_gen::gen_core::VedaExecutionReport,
    ) {
        let device = Device::Cpu;
        let w = weights(bits);
        // CPU Candle does not implement BF16 matmul; dense fixture arithmetic is F32.
        // BF16 request/tier admission is tested at the native loader; device numerics stay gated.
        let mut dit = MiniMaxH3Dit::from_weights(&w, &config(), &device, DType::F32).unwrap();
        // Turbo then style: two additive factors on the actual native adaptable projection.
        let projection = dit
            .adaptable_mut(&["transformer_blocks", "0", "attn", "to_q"])
            .unwrap();
        assert_eq!(projection.is_packed(), bits.is_some());
        for strength in [0.0625, 0.5] {
            projection
                .push_lora(
                    Tensor::full(0.01f32, (64, 2), &device).unwrap(),
                    Tensor::full(0.02f32, (2, 96), &device).unwrap(),
                    strength,
                )
                .unwrap();
        }
        assert_eq!(projection.adapters().len(), 2);
        let layout = layout(mode, channels);
        let schedule = JointSchedule::with_shifts(evals + 1, shift, 3.0).unwrap();
        let adaln = adaln_schedule(&schedule).unwrap();
        assert_eq!(adaln.num_row_classes(), NUM_ROW_CLASSES);
        let context = w.require("in.refiner.context").unwrap();
        let mut model = JointDit::new(
            dit,
            layout.clone(),
            &context,
            adaln.clone(),
            AdaLnResidency::PrecomputeAndEvict,
        )
        .unwrap();
        let controls = VedaRequest {
            enabled,
            generated: VedaBudget::KeyTiles(2),
            reference: VedaBudget::SparsityPercent(0.0),
            full_attention_layers: vec![1],
            full_attention_steps: vec![0, evals - 1],
        };
        controls.validate(2, evals).unwrap();
        let report = crate::veda::ReportSink::default();
        model.prepare_veda(
            &controls,
            Some(std::path::Path::new("/managed/predictor.safetensors")),
            report.clone(),
        );
        let video = Tensor::full(0.02f32, (1, layout.video_indices().len(), 96), &device).unwrap();
        let audio = Tensor::full(0.03f32, (1, layout.audio_indices().len(), 32), &device).unwrap();
        let (outv, outa) = denoise_av(
            &mut model,
            &layout,
            &schedule,
            &adaln,
            &video,
            &audio,
            &device,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(outv.dims(), video.dims());
        assert_eq!(outa.dims(), audio.dims());
        for (input, output, rows) in [
            (&video, &outv, layout.num_condition_video_rows()),
            (&audio, &outa, layout.num_condition_audio_rows()),
        ] {
            if rows > 0 {
                assert_eq!(
                    input
                        .narrow(1, 0, rows)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                    output
                        .narrow(1, 0, rows)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                );
            }
        }
        let bits = Tensor::cat(
            &[outv.flatten_all().unwrap(), outa.flatten_all().unwrap()],
            0,
        )
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect();
        let calls = model
            .veda
            .as_ref()
            .map(|v| v.calls.borrow().clone())
            .unwrap_or_default();
        let observed = report.borrow().clone();
        if enabled {
            assert_eq!(calls.len(), evals * 2);
            for (index, call) in calls.iter().enumerate() {
                assert_eq!((call.layer, call.step), (index % 2, index / 2));
                assert_eq!(call.request, controls);
                assert_eq!(
                    call.full_attention,
                    call.layer == 1 || call.step == 0 || call.step == evals - 1
                );
                assert_eq!(call.shape, [1, 4, layout.seq_len(), 24]);
            }
            assert_eq!(observed.sparse_calls, 0);
            assert_eq!(observed.dense_calls, (evals * 2) as u64);
            assert_eq!(observed.selected_token_pairs, observed.dense_token_pairs);
            assert!(observed
                .fallback_reason
                .as_deref()
                .unwrap()
                .contains("unavailable"));
        } else {
            assert!(model.veda.is_none());
            assert_eq!(observed, Default::default());
        }
        (bits, calls, observed)
    }
    #[test]
    fn native_controls_capture_every_mode_packed_tier_and_adapter_stack() {
        for mode in 0..5 {
            for tier in [None, Some(4), Some(8)] {
                for channels in [1, 2] {
                    let off = run(mode, channels, tier, 4, 12.0, false);
                    let on = run(mode, channels, tier, 4, 12.0, true);
                    assert_eq!(on.0, off.0, "unavailable sparse must preserve dense output: mode={mode} tier={tier:?} channels={channels}");
                }
            }
        }
    }
    #[test]
    fn native_base_and_all_published_turbo_steps_reach_attention() {
        for (name, steps, shift, mode) in [
            ("base", 50, 12.0, 0),
            ("4step_768p", 4, 6.0, 1),
            ("8step", 8, 12.0, 2),
            ("4step_v01", 4, 12.0, 3),
            ("ref2v_4step", 4, 12.0, 4),
        ] {
            let (_, calls, _) = run(mode, 2, None, steps, shift, true);
            assert_eq!(calls.last().unwrap().step, steps - 1, "{name}");
        }
    }
}
