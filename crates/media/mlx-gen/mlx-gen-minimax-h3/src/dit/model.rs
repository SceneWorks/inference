//! The **whole** `MiniMaxH3Transformer3DModel` (sc-17147): the 17 input/output projections of
//! [`crate::dit::heads`] wrapped around the block stack, the token refiner and the MM-RoPE, plus the
//! [`crate::denoise::JointVelocity`] implementation the joint loop drives.
//!
//! ```text
//! proj_in(video rows) ─┐
//! audio_proj_in(audio) ─┼─► PackedLayout::scatter ─► 50 × DitBlock ─► norm_out ─┬─► proj_out       ─► video velocity
//! context_embedder ─►   │                                                        └─► audio_proj_out ─► audio velocity
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

use std::path::Path;

use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::attention::{AttentionBudget, AttentionChunkAxis, AttentionPlan, BoundedAttention};
use mlx_gen::block_residency::{run_windowed, BlockPlan};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};

use crate::block_stream::DitBlockStream;

use crate::denoise::{JointStep, JointVelocity, PackedLayout};
use crate::dit::adaln::{AdaLnCache, AdaLnResidency, TimestepSchedule};
use crate::dit::block::DitBlock;
use crate::dit::config::{MiniMaxH3DitConfig, MODALITY_NUM};
use crate::dit::heads::{AdaLayerNormOut, DitProjections, NormOutModulation};
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
    /// Empty under [`LoadShape::DeferredMaterialization`] — the blocks live in `stream` instead.
    blocks: Vec<DitBlock>,
    /// `Some` under a deferred load (sc-18662, rung 4). Mutually exclusive with a populated
    /// `blocks`, which [`Self::assert_one_residency_mode`] holds.
    stream: Option<DitBlockStream>,
    rope: MmRope,
    dtype: Dtype,
}

impl MiniMaxH3Dit {
    /// Build from an already-populated [`Weights`] map.
    ///
    /// `dtype` is the **block stack's** precision. The 17 input/output tensors are loaded at their
    /// own published dtypes instead — twelve of them ship float32 — see [`crate::dit::heads`].
    pub fn from_weights(w: &mut Weights, cfg: &MiniMaxH3DitConfig, dtype: Dtype) -> Result<Self> {
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
            stream: None,
            rope: MmRope::new(cfg.rope_freq_dim, cfg.rope_theta)?,
            dtype,
        })
    }

    /// Load **only** the 38 non-block tensors and describe the 600 block ones — rung 4's
    /// [`LoadShape::DeferredMaterialization`](mlx_gen::gen_core::LoadShape::DeferredMaterialization)
    /// loader (sc-18662).
    ///
    /// The 17 input/output projections and the 21-tensor token refiner stay resident for the whole
    /// request; see [`crate::block_stream`] for why windowing them would cost more than it bounds.
    /// Everything under `transformer_blocks.` is left unread, so a `bf16` install that would map
    /// 66.28 GB resident maps the projections and the refiner instead.
    ///
    /// The [`Weights`] map is dropped before returning. It is only lazily-mapped handles, but a
    /// retained one would keep every block tensor reachable and make the first window's release
    /// free nothing — the exact failure `run_windowed`'s contract names.
    pub fn load_dir_deferred(dir: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let dir = dir.as_ref();
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|e| {
            Error::Msg(format!(
                "minimax-h3 dit: reading {}: {e}",
                config_path.display()
            ))
        })?;
        let cfg = MiniMaxH3DitConfig::from_diffusers_json(&text)?;
        cfg.validate()?;
        let stream = DitBlockStream::new(dir, dtype, cfg.clone())?;

        let (projections, refiner) = {
            let mut w = Weights::from_dir(dir)?;
            let projections = DitProjections::from_weights(&mut w, &cfg)?;
            let refiner = TokenRefiner::from_weights(&mut w, "token_refiner", &cfg, dtype)?;
            (projections, refiner)
        };

        Ok(Self {
            cfg: cfg.clone(),
            projections,
            refiner,
            blocks: Vec::new(),
            stream: Some(stream),
            rope: MmRope::new(cfg.rope_freq_dim, cfg.rope_theta)?,
            dtype,
        })
    }

    /// The block stream backing a deferred load, or `None` for a resident one.
    pub fn stream(&self) -> Option<&DitBlockStream> {
        self.stream.as_ref()
    }

    /// Whether this load defers block materialization.
    pub fn is_deferred(&self) -> bool {
        self.stream.is_some()
    }

    /// The two residency modes are mutually exclusive, and a load that satisfied both would bound
    /// nothing while reporting that it did.
    fn assert_one_residency_mode(&self) -> Result<()> {
        match (&self.stream, self.blocks.is_empty()) {
            (Some(_), true) | (None, false) => Ok(()),
            (Some(_), false) => Err(Error::Msg(
                "minimax-h3 dit: a deferred load is holding resident blocks; rung 4 would bound                  nothing"
                    .into(),
            )),
            (None, true) => Err(Error::Msg(
                "minimax-h3 dit: a resident load has no blocks and no stream".into(),
            )),
        }
    }

    /// Load `transformer/` (or `transformer_ref/`) from a snapshot root: its `config.json` and its
    /// shards.
    pub fn load(root: impl AsRef<Path>, partition: &str, dtype: Dtype) -> Result<Self> {
        Self::load_dir(root.as_ref().join(partition), dtype)
    }

    /// Load from an already-resolved component directory.
    ///
    /// The seam a **tiered** install needs (sc-17150): a `q4` / `q8` DiT is staged from the
    /// `SceneWorks/minimax-h3-mlx` mirror and does not sit under the snapshot root at all, so there
    /// is no `(root, partition)` pair that names it. Packed tensors are detected per-Linear on
    /// `{base}.scales` ([`crate::quant`]), so this one call loads every tier with no branch.
    pub fn load_dir(dir: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let dir = dir.as_ref();
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|e| {
            Error::Msg(format!(
                "minimax-h3 dit: reading {}: {e}",
                config_path.display()
            ))
        })?;
        let cfg = MiniMaxH3DitConfig::from_diffusers_json(&text)?;
        let mut w = Weights::from_dir(dir)?;
        Self::from_weights(&mut w, &cfg, dtype)
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
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The 17 input/output projections.
    pub fn projections(&self) -> &DitProjections {
        &self.projections
    }

    /// Blocks the forward will run — the resident stack's length, or the streamed stack's declared
    /// depth. A deferred load runs all 50 without ever holding them.
    pub fn num_layers(&self) -> usize {
        match &self.stream {
            Some(stream) => stream.n_blocks(),
            None => self.blocks.len(),
        }
    }

    /// Blocks currently materialized. `0` under a deferred load, which is the residency claim.
    pub fn resident_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// The materialized block stack — empty under a deferred load.
    ///
    /// Exposed for the rung-4 residency harness, which needs to walk the resident arm through the
    /// **same** per-block call the windowed arm walks so the two differ in one variable only.
    pub fn blocks(&self) -> &[DitBlock] {
        &self.blocks
    }

    /// Whether every block still holds its AdaLN projection.
    ///
    /// **`false` under a deferred load**, and that is the honest answer rather than a vacuous
    /// `all()` over an empty stack: a streamed block is materialized body-only, so no block holds a
    /// projection at any instant. `[].iter().all(..)` returns `true`, which would report a deferred
    /// load as pre-eviction.
    pub fn holds_adaln(&self) -> bool {
        !self.blocks.is_empty() && self.blocks.iter().all(DitBlock::holds_adaln)
    }

    /// `context_embedder` then the token refiner — the text rows of the packed sequence.
    ///
    /// `context` is `[1, num_text_tokens, text_dim]`, the text encoder's select-layer hidden state.
    pub fn embed_context(&self, context: &Array) -> Result<Array> {
        let s = context.shape();
        if s.len() != 3 || s[0] != 1 || s[2] != self.cfg.text_dim {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: the text context must be [1, num_text_tokens, {}], got {s:?}",
                self.cfg.text_dim
            )));
        }
        let embedded = self.projections.context_embedder.forward(context)?;
        self.refiner.forward(&embedded.as_dtype(self.dtype)?)
    }

    /// The timestep MLP, as the closure [`AdaLnCache::precompute`] consumes.
    pub fn embed_timesteps(&self, timesteps: &[f32]) -> Result<Array> {
        self.projections.time_embedder.forward(timesteps)
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
    ) -> Result<(Array, Array)> {
        self.forward_packed_bounded(packed, blocks, norm_out, BoundedAttention::UNBOUNDED)
    }

    /// [`Self::forward_packed`] under an explicit [`BoundedAttention`] — the rung-3 seam (sc-18661).
    ///
    /// The plan reaches all 50 blocks and nothing else. The **token refiner is deliberately not on
    /// this path**, and that is arithmetic rather than an omission: it runs once per request over the
    /// text rows alone, so its score domain is `B·H·Sq·Sk = 1·56·N²` for a prompt of `N` tokens, which
    /// stays under [`mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET`]'s 64 Mi for every prompt
    /// below 1 090 tokens and is 0.4 % of one block's domain at the shipped 19 574-row sequence. A
    /// bounded plan there would never engage; threading it would publish a lever with an empty domain.
    pub fn forward_packed_bounded(
        &self,
        packed: &PackedForward<'_>,
        blocks: BlockModulation<'_>,
        norm_out: &NormOutModulation,
        bounded: BoundedAttention<'_>,
    ) -> Result<(Array, Array)> {
        self.forward_packed_windowed(packed, blocks, norm_out, bounded, None)
    }

    /// [`Self::forward_packed_bounded`] over a **windowed** block schedule — rung 4 (sc-18662).
    ///
    /// `window` is `Some((plan, cancel))` under a deferred load and `None` under a resident one.
    /// Identical arithmetic either way: the same `DitBlock::forward_bounded` on the same modulation
    /// in the same order, with the only difference being when each block's weights exist. That is
    /// what lets `tests/block_window.rs` compare the two arms against each other rather than against
    /// a golden.
    ///
    /// Passing a plan to a resident load, or omitting one on a deferred load, is a typed error
    /// rather than a silent fallback: both mean the caller believes a residency mode the model is
    /// not in, and the wrong one bounds nothing while reporting that it does.
    pub fn forward_packed_windowed(
        &self,
        packed: &PackedForward<'_>,
        blocks: BlockModulation<'_>,
        norm_out: &NormOutModulation,
        bounded: BoundedAttention<'_>,
        window: Option<(&BlockPlan, &CancelFlag)>,
    ) -> Result<(Array, Array)> {
        // 1. Per-modality input projections. The two patch projections are float32 in the published
        //    checkpoint; the text stream sets the packed sequence's dtype, as in the reference.
        let stream = packed.text_rows.dtype();
        let video = self
            .projections
            .proj_in
            .forward(packed.video_rows)?
            .as_dtype(stream)?;
        let audio = self
            .projections
            .audio_proj_in
            .forward(packed.audio_rows)?
            .as_dtype(stream)?;
        let mut x = scatter_rows(
            packed.text_rows,
            &video,
            &audio,
            packed.text_indices,
            packed.video_indices,
            packed.audio_indices,
        )?;

        // 2. The stack.
        x = match window {
            None => {
                self.assert_one_residency_mode()?;
                let mut x = x;
                for (layer, block) in self.blocks.iter().enumerate() {
                    x = self.run_block(block, layer, x, packed, blocks, bounded)?;
                }
                x
            }
            Some((plan, cancel)) => {
                self.run_windowed_stack(x, packed, blocks, bounded, plan, cancel)?
            }
        };

        // 3. The shared output norm, then the two heads over their own rows. Gathering before
        //    projecting is bit-identical to the reference's project-then-gather (both heads are
        //    row-wise affine maps) and skips the text rows' outputs, which it discards.
        let x = self
            .projections
            .norm_out
            .apply(&x, norm_out, packed.timestep_indices)?;
        let seq_len = x.shape()[1];
        let video_out =
            self.projections
                .proj_out
                .forward(&gather_rows(&x, packed.video_indices, seq_len)?)?;
        let audio_out = self.projections.audio_proj_out.forward(&gather_rows(
            &x,
            packed.audio_indices,
            seq_len,
        )?)?;
        Ok((video_out, audio_out))
    }
}

impl MiniMaxH3Dit {
    /// One block of the stack, under either residency mode.
    fn run_block(
        &self,
        block: &DitBlock,
        layer: usize,
        x: Array,
        packed: &PackedForward<'_>,
        blocks: BlockModulation<'_>,
        bounded: BoundedAttention<'_>,
    ) -> Result<Array> {
        match blocks {
            BlockModulation::Cached(cache) => block.forward_bounded(
                &x,
                cache.modulation(layer)?,
                packed.adaln_indices,
                &self.rope,
                packed.tables,
                bounded,
            ),
            BlockModulation::Temb(temb) => block.forward_with_temb_bounded(
                &x,
                temb,
                packed.adaln_indices,
                &self.rope,
                packed.tables,
                bounded,
            ),
        }
    }

    /// The 50 blocks over a bounded window, through the shared driver.
    ///
    /// [`BlockModulation::Temb`] is **refused** here, and that is the rung's shape rather than an
    /// omission: the `Temb` arm calls `DitBlock::modulation`, which needs `adaln_proj`, and a window
    /// that materialized `adaln_proj` would re-read 39.3 % of the DiT every step for a table the
    /// precompute already holds (see [`crate::block_stream`]). A deferred load projects its
    /// modulation once through `block_stream::precompute_adaln_windowed` and denoises from the
    /// cache.
    fn run_windowed_stack(
        &self,
        x: Array,
        packed: &PackedForward<'_>,
        blocks: BlockModulation<'_>,
        bounded: BoundedAttention<'_>,
        plan: &BlockPlan,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        let stream = self.stream.as_ref().ok_or_else(|| {
            Error::Msg(
                "minimax-h3 dit: a block window was supplied to a RESIDENT load. Load with                  `load_dir_deferred` for rung 4, or drop the window."
                    .into(),
            )
        })?;
        self.assert_one_residency_mode()?;
        let BlockModulation::Cached(cache) = blocks else {
            return Err(Error::Msg(
                "minimax-h3 dit: a windowed stack needs the precomputed AdaLN cache; the resident                  `Temb` arm projects per block and would pull adaln_proj into every window"
                    .into(),
            ));
        };
        if plan.n_blocks() != stream.n_blocks() {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: the block plan covers {} blocks, the streamed stack has {}",
                plan.n_blocks(),
                stream.n_blocks()
            )));
        }
        if cache.num_layers() != stream.n_blocks() {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: the AdaLN cache carries {} layers, the streamed stack has {}",
                cache.num_layers(),
                stream.n_blocks()
            )));
        }

        run_windowed(
            plan,
            cancel,
            x,
            || stream.open(),
            |mut hidden: Array, view: &mut Weights, range: std::ops::Range<usize>| {
                for layer in range {
                    let block = stream.materialize(view, layer)?;
                    hidden = self.run_block(
                        &block,
                        layer,
                        hidden,
                        packed,
                        BlockModulation::Cached(cache),
                        bounded,
                    )?;
                    // The block drops per iteration rather than per window: at `window > 1` the
                    // alternative holds every block of the window plus the activation.
                }
                Ok(hidden)
            },
            // LOAD-BEARING: MLX is lazy, so the carried activation still references this window's
            // weights until it is forced. Measured elsewhere at 8.0 MiB with the guard against
            // 238.4 MiB without, with correct output either way — silent, not loud.
            |hidden: &Array| mlx_rs::transforms::eval([hidden]).map_err(Into::into),
        )
    }
}

/// The DiT's LoRA target surface (sc-18724): `transformer_blocks.{i}.…` and `token_refiner.…`,
/// spelled exactly as the published diffusers checkpoint — and therefore exactly as the lightx2v
/// turbo LoRAs key their 624 factors.
///
/// **The 17 input/output projections of [`DitProjections`] are not on this surface.** They are
/// [`crate::dit::heads::LinearBias`], a raw weight/bias pair rather than an
/// [`AdaptableLinear`], because they are the mixed-precision half of the checkpoint (twelve ship
/// float32) and [`crate::dit::heads`] reads them with a bare `Weights::require` precisely so a
/// packed tensor cannot be loaded as floats (sc-14980). No published MiniMax-H3 adapter targets
/// them; one that did would surface in `unmatched_paths` and fail the strict install rather than
/// fold onto nothing. See [`crate::adapters`].
impl AdaptableHost for MiniMaxH3Dit {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["transformer_blocks", i, rest @ ..] => {
                let idx: usize = i.parse().ok()?;
                self.blocks.get_mut(idx)?.adaptable_mut(rest)
            }
            ["token_refiner", rest @ ..] => self.refiner.adaptable_mut(rest),
            _ => None,
        }
    }
}

/// Everything one [`MiniMaxH3Dit::forward_packed`] needs besides the weights and the modulation.
#[derive(Debug)]
pub struct PackedForward<'a> {
    /// `[1, video_indices.len(), in_channels · prod(patch)]`, conditioning rows first.
    pub video_rows: &'a Array,
    /// `[1, audio_indices.len(), audio_in_channels]`, channel-major.
    pub audio_rows: &'a Array,
    /// `[1, text_indices.len(), hidden_size]` — already `context_embedder` + `token_refiner`.
    pub text_rows: &'a Array,
    /// `[seq_len]` — `timestep_index · MODALITY_NUM + token_tag`, what the blocks gather with.
    pub adaln_indices: &'a Array,
    /// `[seq_len]` — the **bare** timestep index, what `norm_out` gathers with.
    pub timestep_indices: &'a Array,
    /// MM-RoPE tables over the packed sequence's position grid.
    pub tables: &'a MmRopeTables,
    /// Rows the text condition occupies.
    pub text_indices: &'a [i32],
    /// Rows the video stream occupies — conditioning rows first, and **discontiguous**.
    pub video_indices: &'a [i32],
    /// Rows the audio stream occupies.
    pub audio_indices: &'a [i32],
}

/// Where a [`MiniMaxH3Dit::forward_packed`] takes its per-block modulation from.
#[derive(Debug, Clone, Copy)]
pub enum BlockModulation<'a> {
    /// The precomputed table sc-17145's eviction leaves behind.
    Cached(&'a AdaLnCache),
    /// A live timestep embedding, projected per block — the resident path.
    Temb(&'a Array),
}

/// Scatter `concat(text, video, audio)` into one `[1, seq_len, F]` packed sequence.
///
/// The same single-gather-through-an-inverse-permutation
/// [`PackedLayout::scatter`] performs, expressed over the three index **slices** so a fixture
/// geometry that [`crate::denoise::JointGeometry`] refuses to build can still be driven. The
/// duplicate/coverage checks are kept: a row claimed twice, or not at all, is an error rather than
/// a sequence with a stale value in it.
fn scatter_rows(
    text: &Array,
    video: &Array,
    audio: &Array,
    text_indices: &[i32],
    video_indices: &[i32],
    audio_indices: &[i32],
) -> Result<Array> {
    let seq_len = (text_indices.len() + video_indices.len() + audio_indices.len()) as i32;
    let mut inverse = vec![-1i32; seq_len as usize];
    for (source, &dest) in text_indices
        .iter()
        .chain(video_indices)
        .chain(audio_indices)
        .enumerate()
    {
        if dest < 0 || dest >= seq_len {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: row {dest} is outside [0, {seq_len}); MLX scatters out of bounds \
                 silently"
            )));
        }
        if inverse[dest as usize] != -1 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: row {dest} is claimed by two index tensors"
            )));
        }
        inverse[dest as usize] = source as i32;
    }
    for (block, rows, what) in [
        (text, text_indices.len(), "text"),
        (video, video_indices.len(), "video"),
        (audio, audio_indices.len(), "audio"),
    ] {
        let s = block.shape();
        if s.len() != 3 || s[0] != 1 || s[1] != rows as i32 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: the {what} block must be [1, {rows}, hidden], got {s:?}"
            )));
        }
    }
    let sources = mlx_rs::ops::concatenate_axis(&[text.clone(), video.clone(), audio.clone()], 1)?;
    let perm = Array::from_slice(&inverse, &[seq_len]);
    Ok(sources.take_axis(&perm, 1)?)
}

/// Gather one modality's rows out of the packed sequence, bounds-checked on the host because MLX
/// gathers out of bounds silently.
fn gather_rows(packed: &Array, rows: &[i32], seq_len: i32) -> Result<Array> {
    if rows.is_empty() {
        return Err(Error::Msg("minimax-h3 dit: cannot gather zero rows".into()));
    }
    if let Some(bad) = rows.iter().find(|&&r| r < 0 || r >= seq_len) {
        return Err(Error::Msg(format!(
            "minimax-h3 dit: row {bad} is outside [0, {seq_len}); MLX gathers out of bounds \
             silently rather than failing"
        )));
    }
    let idx = Array::from_slice(rows, &[rows.len() as i32]);
    Ok(packed.take_axis(&idx, 1)?)
}

/// How a [`JointDit`] sources its per-block modulation.
#[derive(Debug, Clone)]
enum Modulation {
    /// The whole schedule projected up front and `adaln_proj` released — sc-17145's 26.02 GB lever.
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
    text_rows: Array,
    modulation: Modulation,
    /// `Some` under [`Modulation::Cached`] — the whole schedule's `norm_out` shift/scale, projected
    /// from the same `temb` the block stack's table was. `None` under [`Modulation::Resident`],
    /// where the table moves every step and is rebuilt inside `forward`.
    norm_out_modulation: Option<NormOutModulation>,
    /// Non-`None` only in [`Modulation::Resident`]: the per-request local AdaLN index, which is
    /// `row_class · MODALITY_NUM + token_tag` rather than the global schedule's row.
    resident_adaln_indices: Option<Array>,
    /// Bytes of `adaln_proj` weight released by the eviction, 0 when resident.
    released_bytes: usize,
    forwards: usize,
    /// The rung-3 plan every step's block stack runs under (sc-18661).
    ///
    /// Stored decomposed rather than as a [`BoundedAttention`] because that type borrows the cancel
    /// flag, and a self-referential borrow is not expressible on this struct. Reassembled per step.
    attention_budget: AttentionBudget,
    attention_axis: AttentionChunkAxis,
    attention_cancel: Option<CancelFlag>,
    /// `Some` under rung 4 (sc-18662): the per-step block window the stack runs over. `None` is the
    /// resident stack. Set by [`JointDit::new_windowed`] and never afterwards — a residency mode
    /// that could change mid-request would leave half a render bounded.
    block_window: Option<BlockPlan>,
    /// The cancel flag `run_windowed` checks at every window boundary. Rung 4 is the *other* place
    /// (with rung 3's chunk boundaries) a cancel can land inside a single DiT forward.
    window_cancel: CancelFlag,
}

impl JointDit {
    /// Build the velocity model for one request.
    ///
    /// * `context` — `[1, num_text_tokens, text_dim]` from the text encoder;
    /// * `schedule` — the run's [`TimestepSchedule`], from
    ///   [`crate::denoise::adaln_schedule`];
    /// * `residency` — [`AdaLnResidency::PrecomputeAndEvict`] projects the whole schedule up front
    ///   and releases 26.02 GB of `adaln_proj`; [`AdaLnResidency::Resident`] keeps them and projects
    ///   the four row-class timesteps per step.
    ///
    /// Consumes the model because the eviction mutates it irreversibly.
    pub fn new(
        mut dit: MiniMaxH3Dit,
        layout: PackedLayout,
        context: &Array,
        schedule: TimestepSchedule,
        residency: AdaLnResidency,
    ) -> Result<Self> {
        if schedule.num_row_classes() != crate::denoise::NUM_ROW_CLASSES {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: the schedule declares {} row classes, expected {}",
                schedule.num_row_classes(),
                crate::denoise::NUM_ROW_CLASSES
            )));
        }
        let text_rows = dit.embed_context(context)?;
        if text_rows.shape()[1] != layout.num_text_tokens() {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: the context carries {} rows but the layout declares {} text rows",
                text_rows.shape()[1],
                layout.num_text_tokens()
            )));
        }
        let tables = dit.rope.tables(layout.position_ids())?;

        let (modulation, norm_out_modulation, resident_adaln_indices, released_bytes) =
            match residency {
                AdaLnResidency::PrecomputeAndEvict => {
                    // The timestep embedding is captured out of the closure so `norm_out`'s own
                    // table is projected from the SAME `temb` the block stack's was — a second
                    // `embed_timesteps` call would be a second chance to bind the wrong timesteps.
                    let mut captured: Option<Array> = None;
                    let embedder = dit.projections.time_embedder.clone();
                    let (cache, released) = AdaLnCache::precompute_and_evict(
                        &mut dit.blocks,
                        schedule,
                        residency,
                        |timesteps| {
                            let temb = embedder.forward(timesteps)?;
                            captured = Some(temb.clone());
                            Ok(temb)
                        },
                    )?;
                    let temb = captured.ok_or_else(|| {
                        Error::Msg(
                            "minimax-h3 dit: the AdaLN precompute did not evaluate the timestep \
                             embedding"
                                .into(),
                        )
                    })?;
                    let norm_out = dit.projections.norm_out.modulation(&temb)?;
                    // Force it, for the same reason the AdaLN precompute forces its own tables: an
                    // unevaluated node would defer this work into the first step's timing and hold
                    // `temb` alive for the whole run.
                    mlx_rs::transforms::eval([&norm_out.shift, &norm_out.scale])?;
                    (
                        Modulation::Cached(Box::new(cache)),
                        Some(norm_out),
                        None,
                        released,
                    )
                }
                AdaLnResidency::Resident => {
                    // One temb row per ROW CLASS, rebuilt per step; the index tensors are therefore
                    // the local `row_class · MODALITY_NUM + tag`, which is constant across steps
                    // because neither the class nor the tag of a row ever moves.
                    let classes = layout.row_classes().as_dtype(Dtype::Int32)?;
                    let tags = layout.token_tags().as_dtype(Dtype::Int32)?;
                    let local = mlx_rs::ops::add(
                        &mlx_rs::ops::multiply(&classes, Array::from_int(MODALITY_NUM))?,
                        &tags,
                    )?;
                    (Modulation::Resident, None, Some(local), 0)
                }
            };

        Ok(Self {
            dit,
            layout,
            tables,
            text_rows,
            modulation,
            norm_out_modulation,
            resident_adaln_indices,
            released_bytes,
            forwards: 0,
            // Unbounded until a measured rung selects otherwise — the ladder's rule everywhere.
            attention_budget: AttentionBudget::UNBOUNDED,
            attention_axis: AttentionChunkAxis::Heads,
            attention_cancel: None,
            block_window: None,
            window_cancel: CancelFlag::default(),
        })
    }

    /// Build the velocity model over a **deferred** DiT with a bounded block window — rung 4
    /// (sc-18662).
    ///
    /// The two differences from [`Self::new`] are both forced by the residency, not chosen:
    ///
    /// * the AdaLN modulation is projected through
    ///   [`crate::block_stream::precompute_adaln_windowed`] rather than
    ///   `AdaLnCache::precompute_and_evict`, because there is no resident stack to evict *from* —
    ///   the deferred load never held the projections. `released_bytes` is therefore `0`, and that
    ///   is the honest figure: reporting a release that did not happen is the over-declared-saving
    ///   direction the contract refuses;
    /// * [`AdaLnResidency::Resident`] is refused. Its whole shape is keeping `adaln_proj` loaded so
    ///   a run-state-dependent sampler can project per step, which is precisely the residency rung 4
    ///   removes.
    pub fn new_windowed(
        dit: MiniMaxH3Dit,
        layout: PackedLayout,
        context: &Array,
        schedule: TimestepSchedule,
        residency: AdaLnResidency,
        window: usize,
        cancel: CancelFlag,
    ) -> Result<Self> {
        if residency != AdaLnResidency::PrecomputeAndEvict {
            return Err(Error::Msg(
                "minimax-h3 dit: a windowed load cannot use AdaLnResidency::Resident — keeping                  adaln_proj loaded is exactly the residency rung 4 removes"
                    .into(),
            ));
        }
        let stream = dit
            .stream()
            .ok_or_else(|| {
                Error::Msg(
                    "minimax-h3 dit: JointDit::new_windowed needs a deferred load; use                      MiniMaxH3Dit::load_dir_deferred"
                        .into(),
                )
            })?
            .clone();
        let plan = stream.plan(window)?;

        let text_rows = dit.embed_context(context)?;
        if text_rows.shape()[1] != layout.num_text_tokens() {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: the context carries {} rows but the layout declares {} text rows",
                text_rows.shape()[1],
                layout.num_text_tokens()
            )));
        }
        let tables = dit.rope.tables(layout.position_ids())?;

        // One `temb` for both tables, captured rather than re-embedded — a second
        // `embed_timesteps` call would be a second chance to bind the wrong timesteps.
        let temb = dit.embed_timesteps(schedule.distinct_timesteps())?;
        let cache = crate::block_stream::precompute_adaln_windowed(
            &stream, &plan, &cancel, schedule, &temb,
        )?;
        let norm_out = dit.projections.norm_out.modulation(&temb)?;
        mlx_rs::transforms::eval([&norm_out.shift, &norm_out.scale])?;

        Ok(Self {
            dit,
            layout,
            tables,
            text_rows,
            modulation: Modulation::Cached(Box::new(cache)),
            norm_out_modulation: Some(norm_out),
            resident_adaln_indices: None,
            // Nothing was released, because nothing was ever resident.
            released_bytes: 0,
            forwards: 0,
            attention_budget: AttentionBudget::UNBOUNDED,
            attention_axis: AttentionChunkAxis::Heads,
            attention_cancel: None,
            block_window: Some(plan),
            window_cancel: cancel,
        })
    }

    /// The block window in force, or `None` on a resident stack.
    pub fn block_window(&self) -> Option<&BlockPlan> {
        self.block_window.as_ref()
    }

    /// Select the rung-3 bounded-attention plan for every subsequent step (sc-18661).
    ///
    /// [`AttentionBudget::UNBOUNDED`] restores the byte-identical un-chunked forward, so this is
    /// reversible on a live model — which is what lets one loaded 50-block DiT serve both arms of a
    /// peak comparison.
    pub fn set_bounded_attention(&mut self, budget: AttentionBudget, axis: AttentionChunkAxis) {
        self.attention_budget = budget;
        self.attention_axis = axis;
    }

    /// Attach the request's cancel flag, checked **between attention chunks** — the only cancellation
    /// point that exists inside a single DiT forward (`gen_core::attention_budget::AttentionPlan`).
    ///
    /// Inert on the unbounded path, which has no chunk boundary to check at.
    pub fn set_attention_cancel(&mut self, cancel: CancelFlag) {
        self.attention_cancel = Some(cancel);
    }

    /// The plan in force, as the block stack will see it.
    pub fn bounded_attention(&self) -> BoundedAttention<'_> {
        let plan = match &self.attention_cancel {
            Some(cancel) => AttentionPlan::budgeted(self.attention_budget).with_cancel(cancel),
            None => AttentionPlan::budgeted(self.attention_budget),
        };
        BoundedAttention::new(plan, self.attention_axis)
    }

    /// Bytes of `adaln_proj` weight the eviction released — 0 under
    /// [`AdaLnResidency::Resident`].
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
    fn forward(&mut self, step: &JointStep<'_>) -> Result<(Array, Array)> {
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
                    Error::Msg(
                        "minimax-h3 dit: resident residency without its local AdaLN index".into(),
                    )
                })?;
                (
                    local.clone(),
                    self.layout.row_classes().as_dtype(Dtype::Int32)?,
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
                return Err(Error::Msg(
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
                return Err(Error::Msg(
                    "minimax-h3 dit: resident residency without a timestep embedding".into(),
                ))
            }
        };
        self.dit.forward_packed_windowed(
            &packed,
            blocks,
            &norm_out,
            self.bounded_attention(),
            self.block_window
                .as_ref()
                .map(|plan| (plan, &self.window_cancel)),
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
