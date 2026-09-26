//! Acoustic flow matching (the NAR stage) with bounded attention, native on Candle (sc-22992).
//!
//! Ported from the pinned upstream `src/yue2/nar.py` (`song_chunks`, `attention`, `CachedNAR`,
//! `_offload_ar`, `synthesize`) and the NAR auxiliary modules of `src/yue2/modeling_yue2.py`
//! (`vae2llm`, `llm2vae`, `TimestepEmbedder`, `AudioPositionEmbedding`, `_shift_t_value`), Apache-2.0,
//! commit [`YUE2_SOURCE_COMMIT`](crate::inventory::YUE2_SOURCE_COMMIT). It turns the semantic stage's
//! codec tokens into the `[frames, 64]` FP32 [`AcousticLatents`] the VAE decodes.
//!
//! # Composition (upstream `song_chunks`)
//!
//! * The song's noise is **one** `[frames, 64]` tensor ([`SongNoise`]) and every original chunk
//!   solves the rows of its own frame range, so changing how a song is chunked never changes which
//!   noise a frame starts from.
//! * The chunks are the protocol's original context chunks,
//!   [`protocol::chunk_ranges`]`(frames, prefix, context)`. Chunk `[a, b)`'s AR sequence is the
//!   plan's prefix, the chunk's codec ids (`+ CODEC_OFFSET`) and `MUSIC_END`; its NAR sequence is
//!   `LATENT_START`-slot, the `b − a` latent frames, `LATENT_END`-slot (`b − a + 2` positions).
//!   Chunks are solved serially and concatenated; nothing is windowed, shortened or dropped.
//!
//! # One chunk (upstream `CachedNAR`)
//!
//! 1. **AR prefill, once per chunk.** The AR path runs causally over the chunk's AR sequence into a
//!    [`StaticKvCache`] of `ar + nar` slots ([`Yue2Lm::prefill`]). Those keys/values are invariant
//!    for the whole ODE and are reused by all `2 · steps` velocity evaluations.
//! 2. **Velocity.** The NAR input is `vae2llm(pad(x_t))` + the timestep embedding + the sinusoidal
//!    latent-position embedding (local positions `0 … nar − 1`, clamped to `max_latent_frames − 1`).
//!    RoPE positions continue after the AR sequence (`ar … ar + nar − 1`). Every layer runs the
//!    `nar_` twins; attention is **bidirectional** over the visible AR keys followed by the NAR
//!    keys — the cache is truncated back to the visible AR prefix and this evaluation's NAR keys are
//!    written after it, so no evaluation sees another's keys. `llm2vae(norm(x))` at the latent
//!    positions is the velocity.
//! 3. **Visible AR keys.** Every AR key (prompt, codec ids and `MUSIC_END`) unless the chunk sets a
//!    `nar_cond_end` (upstream's text-only codec dropout), in which case only the first
//!    `min(nar_cond_end, ar)` are visible.
//! 4. **Solver.** The released explicit midpoint rule from `t = 1` (noise) to `t = 0` with
//!    `dt = 1 / steps`: the model sees `logit(t)` clamped to `[−20, 20]` (computed in F64, handed to
//!    the model as F32), shifted by `timestep_shift` through a sigmoid; `x ← x − v(x − v(x, t)·dt/2,
//!    t − dt/2)·dt`. The state is kept in the model dtype and returned as host F32; a non-finite
//!    result is an error.
//!
//! # Bounded memory
//!
//! * **Query tiling** ([`QueryTile`]): attention is computed for bounded blocks of NAR query rows,
//!   and every block attends the **whole** visible key set — tiling changes temporary storage, never
//!   the keys a query sees. The attention kernel tiles further by its own index bound.
//! * **AR offload** ([`NarOptions::offload_ar`]): after a chunk's prefill the AR-only weights move to
//!   host memory for its solve ([`Yue2Lm::offload_ar`]); the chunk's cache is released **before**
//!   they are restored, on success, error and cancellation alike, so the restoration peak stays
//!   bounded. On a model that already lives in host memory nothing moves (upstream moves only
//!   modules whose device is not the CPU).
//!
//! Neither knob enters the [stage identity](SynthesisRequest::stage_identity): they change memory,
//! not the composition.
//!
//! # Cancellation and progress
//!
//! The cancellation hook is polled before every chunk's prefill, between its prefill forwards and
//! before each of the two velocity evaluations of every step; a cancelled synthesis returns
//! [`gen_core::Error::Canceled`], releases the chunk's cache and restores offloaded weights.
//! Progress counts midpoint steps across all chunks (`chunk · steps + completed` of
//! `steps · chunks`), like upstream's `on_progress`.
//!
//! # Noise (epic E9)
//!
//! Upstream draws the song's noise from PyTorch's CPU generator, which is not reproduced natively
//! (a seed is no cross-platform bit-exact guarantee). [`SongNoise::seeded`] draws a standard normal
//! per element from the request seed's own stream (Box–Muller over SplitMix64, [`stage_rng`]), and
//! [`SongNoise::injected`] takes an exact tensor — how parity against the reference is isolated.

use std::time::Instant;

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::gen_core;
use candle_llm::primitives::{sdpa_gqa, AttnMask, KvCache, StaticKvCache};
#[cfg(test)]
use candle_nn::VarBuilder;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::generate::stage_rng;
use crate::latent::{AcousticLatents, LatentSource, LATENT_CHANNELS};
use crate::model::{backend, rms_norm, MotPaths, Yue2Config, Yue2Lm};
use crate::protocol::{self, CODEC_OFFSET, CODEC_SIZE, MUSIC_END, ODE_METHOD, PROTOCOL_VERSION};
use crate::precision::{Residency, Tier};
use crate::snapshot::SnapshotDirs;
use crate::weights::{Loader, Proj};

/// Upstream's query block on CPU / MPS (`attention(query_chunk_size=None)` off CUDA): 256 rows.
pub const UPSTREAM_QUERY_TILE: usize = 256;
/// `logit(t)` is clamped to `[−LOGIT_CLAMP, LOGIT_CLAMP]` before the model sees it.
pub const LOGIT_CLAMP: f64 = 20.0;
/// Width of the sinusoidal timestep features (`TimestepEmbedder.frequency_embedding_size`).
pub const TIME_FREQUENCIES: usize = 256;
/// The identity schema of a synthesis stage ([`SynthesisRequest::stage_identity`]).
pub const STAGE_IDENTITY_SCHEMA: &str = "yue2-acoustic-synthesis-v1";

/// The acoustic fields of `config.json` (upstream `YuE2Config`'s `latent_dim`, `max_latent_frames`,
/// `timestep_shift`).
#[derive(Clone, Debug, PartialEq)]
pub struct NarConfig {
    /// Latent channels; the released protocol (and [`AcousticLatents`]) is 64.
    pub latent_dim: usize,
    /// Rows of the latent-position table; local positions past it reuse its last row.
    pub max_latent_frames: usize,
    /// The sigmoid-time shift (`1.0` released).
    pub timestep_shift: f64,
}

impl NarConfig {
    /// Parse the acoustic fields of a YuE2 `config.json`. A latent width other than 64, an empty
    /// position table or a non-finite / non-positive shift is [`gen_core::Error::Unsupported`].
    pub fn from_json(text: &str) -> gen_core::Result<Self> {
        let v: Value = serde_json::from_str(text)
            .map_err(|e| gen_core::Error::Msg(format!("YuE2 config.json: {e}")))?;
        let uint = |key: &str| {
            v.get(key)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| gen_core::Error::Msg(format!("YuE2 config.json: missing `{key}`")))
        };
        let config =
            Self {
                latent_dim: uint("latent_dim")?,
                max_latent_frames: uint("max_latent_frames")?,
                timestep_shift: v.get("timestep_shift").and_then(Value::as_f64).ok_or_else(
                    || gen_core::Error::Msg("YuE2 config.json: missing `timestep_shift`".into()),
                )?,
            };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> gen_core::Result<()> {
        if self.latent_dim != LATENT_CHANNELS
            || self.max_latent_frames == 0
            || !(self.timestep_shift.is_finite() && self.timestep_shift > 0.0)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "YuE2 acoustic config must have latent_dim {LATENT_CHANNELS}, a non-empty \
                 position table and a finite positive timestep_shift: {self:?}"
            )));
        }
        Ok(())
    }
}

/// The NAR auxiliary heads: `vae2llm`, `llm2vae`, `time_embedder` and the checkpoint's
/// `latent_pos_embed.pe` table (loaded, not recomputed: it is a stored BF16 buffer).
#[derive(Debug)]
pub struct NarHeads {
    config: NarConfig,
    vae2llm: (Proj, Tensor),
    llm2vae: (Proj, Tensor),
    time_in: (Proj, Tensor),
    time_out: (Proj, Tensor),
    /// `[max_latent_frames, hidden]` in the model dtype.
    pe: Tensor,
    /// `exp(−ln(10⁴) · i / 128)`, `i < 128`, F32 — the timestep frequencies.
    freqs: Tensor,
}

impl NarHeads {
    pub(crate) fn from_loader(
        config: NarConfig,
        hidden: usize,
        w: &Loader,
    ) -> gen_core::Result<Self> {
        config.validate()?;
        let err = backend("load NAR heads");
        let d = config.latent_dim;
        let linear_pair = |name: &str, out: usize, inp: usize| -> gen_core::Result<_> {
            Ok((
                w.matrix(&format!("{name}.weight"), (out, inp))
                    .map_err(&err)?,
                w.tensor(&format!("{name}.bias"), out).map_err(&err)?,
            ))
        };
        let half = TIME_FREQUENCIES / 2;
        let freqs = Tensor::arange(0u32, half as u32, w.device())
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t * -(10_000f64.ln()))
            .and_then(|t| t / half as f64)
            .and_then(|t| t.exp())
            .map_err(&err)?;
        Ok(Self {
            vae2llm: linear_pair("vae2llm", hidden, d)?,
            llm2vae: linear_pair("llm2vae", d, hidden)?,
            time_in: linear_pair("time_embedder.mlp.0", hidden, TIME_FREQUENCIES)?,
            time_out: linear_pair("time_embedder.mlp.2", hidden, hidden)?,
            pe: w
                .tensor("latent_pos_embed.pe", (config.max_latent_frames, hidden))
                .map_err(&err)?,
            freqs,
            config,
        })
    }

    /// The parsed acoustic configuration.
    pub fn config(&self) -> &NarConfig {
        &self.config
    }

    /// Upstream `_shift_t_value(raw_t)` then `time_embedder(t)`: `[1, 1, hidden]` in the model
    /// dtype. The sigmoid and the shift run in the model dtype, the sinusoid in F32.
    fn time_embedding(&self, raw_t: f32) -> candle_llm::Result<Tensor> {
        let dtype = self.pe.dtype();
        let raw = Tensor::new(&[raw_t], self.pe.device())?.to_dtype(dtype)?;
        let sig = candle_nn::ops::sigmoid(&raw)?;
        let shift = self.config.timestep_shift;
        let shifted = ((&sig * shift)? / ((&sig * (shift - 1.0))? + 1.0)?)?;
        let args = shifted
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .broadcast_mul(&self.freqs.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], 1)?.to_dtype(dtype)?;
        let h = self.time_in.0.forward(&emb, Some(&self.time_in.1))?.silu()?;
        Ok(self
            .time_out
            .0
            .forward(&h, Some(&self.time_out.1))?
            .unsqueeze(0)?)
    }

    /// Resident bytes of the heads' weights, biases and position table.
    fn resident_bytes(&self) -> u64 {
        let dense = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        [&self.vae2llm, &self.llm2vae, &self.time_in, &self.time_out]
            .iter()
            .map(|(w, b)| w.resident_bytes() + dense(b))
            .sum::<u64>()
            + dense(&self.pe)
    }

    /// Local latent positions `0 … len − 1`, clamped to the table: `[1, len, hidden]`.
    fn position_embedding(&self, len: usize) -> candle_audio::candle_core::Result<Tensor> {
        let last = self.config.max_latent_frames - 1;
        let ids: Vec<u32> = (0..len).map(|i| i.min(last) as u32).collect();
        let ids = Tensor::from_vec(ids, len, self.pe.device())?;
        self.pe.index_select(&ids, 0)?.unsqueeze(0)
    }
}

/// The YuE2 MoT with both paths and the NAR heads — everything the ABC, semantic and acoustic
/// stages run, loaded once.
#[derive(Debug)]
pub struct Yue2Nar {
    lm: Yue2Lm,
    heads: NarHeads,
    weights_sha256: String,
}

impl Yue2Nar {
    /// Resolve and verify the pinned YuE2-3B snapshot **immediately before loading** (the crate's
    /// load-boundary rule), then load the MoT with both paths ([`MotPaths::ArAndNar`]) and the NAR
    /// heads from exactly the verified `config.json` and weights file. `dtype` as for
    /// [`Yue2Lm::load`] (F32 on a CPU device).
    pub fn load(dirs: &SnapshotDirs, dtype: DType, device: &Device) -> gen_core::Result<Self> {
        Self::load_tier(dirs, None, dtype, device)
    }

    /// [`Yue2Nar::load`], asserting the staged tier when `tier` is set (a different staged tier is
    /// refused, never loaded). The `m-a-p/YuE2-3B` directory holds either the pinned original
    /// (`bf16`) or a derived tier snapshot ([`crate::tier`]).
    pub fn load_tier(
        dirs: &SnapshotDirs,
        tier: Option<Tier>,
        dtype: DType,
        device: &Device,
    ) -> gen_core::Result<Self> {
        let opened = crate::model::open_verified(dirs, dtype, device, tier)?;
        Self::from_opened(opened)
    }

    pub(crate) fn from_opened(opened: crate::model::VerifiedLmFiles) -> gen_core::Result<Self> {
        Self::from_loader(
            Yue2Config::from_json(&opened.config_json)?,
            NarConfig::from_json(&opened.config_json)?,
            &opened.loader,
            opened.weights_sha256,
            opened.tier,
        )
    }

    /// Build the released-precision model from an arbitrary [`VarBuilder`]. Crate-private:
    /// production loads go through [`Yue2Nar::load`], which verifies the bytes first.
    #[cfg(test)]
    pub(crate) fn from_var_builder(
        config: Yue2Config,
        nar: NarConfig,
        vb: VarBuilder,
        weights_sha256: String,
    ) -> gen_core::Result<Self> {
        Self::from_loader(
            config,
            nar,
            &Loader::new(vb, None),
            weights_sha256,
            Tier::Bf16,
        )
    }

    /// Build from a tier-aware [`Loader`].
    pub(crate) fn from_loader(
        config: Yue2Config,
        nar: NarConfig,
        w: &Loader,
        weights_sha256: String,
        tier: Tier,
    ) -> gen_core::Result<Self> {
        let hidden = config.hidden_size;
        let heads = NarHeads::from_loader(nar, hidden, w)?;
        let lm = Yue2Lm::from_loader(config, w, MotPaths::ArAndNar, tier)?;
        Ok(Self {
            lm,
            heads,
            weights_sha256,
        })
    }

    /// The MoT backbone, mutably (the FP8 AR mode, [`crate::fp8`]).
    pub fn lm_mut(&mut self) -> &mut Yue2Lm {
        &mut self.lm
    }

    /// Measured resident weight bytes of everything loaded: the MoT
    /// ([`Yue2Lm::weight_residency`]) and the NAR heads.
    pub fn weight_residency(&self) -> Residency {
        let mut r = self.lm.weight_residency();
        r.device_bytes += self.heads.resident_bytes();
        r
    }

    /// The MoT backbone (for the ABC and semantic stages).
    pub fn lm(&self) -> &Yue2Lm {
        &self.lm
    }

    /// The NAR heads.
    pub fn heads(&self) -> &NarHeads {
        &self.heads
    }

    /// SHA-256 of the weights file the model was loaded from (part of the stage identity).
    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    /// Retry restoring offloaded AR weights after a restore failed (a synthesis restores them
    /// itself on success, error, cancellation and panic; see [`Yue2Lm::restore_ar`]). A no-op
    /// when nothing is offloaded.
    pub fn restore_ar(&mut self) -> gen_core::Result<()> {
        self.lm.restore_ar()
    }
}

/// How the song's `[frames, 64]` starting noise was obtained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoiseSource {
    /// Drawn from the request seed ([`SongNoise::seeded`]).
    Seeded {
        /// The request seed.
        seed: u64,
    },
    /// Supplied exactly by the caller ([`SongNoise::injected`]).
    Injected,
}

/// The whole song's starting noise, `[frames, 64]` row-major F32, drawn or supplied **once**; every
/// chunk solves its own rows of it.
#[derive(Clone, Debug, PartialEq)]
pub struct SongNoise {
    values: Vec<f32>,
    frames: usize,
    source: NoiseSource,
}

impl SongNoise {
    /// A standard normal per element from the seed's stream: SplitMix64 ([`stage_rng`]), two
    /// 53-bit uniforms per Box–Muller pair, element order row-major. Frame `f`'s noise depends only
    /// on `(seed, f)`, so a longer song extends a shorter one's noise.
    pub fn seeded(seed: u64, frames: usize) -> Self {
        let mut rng = stage_rng(seed);
        let mut unit = || (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        let n = frames * LATENT_CHANNELS;
        let mut values = Vec::with_capacity(n);
        while values.len() < n {
            // `1 − u ∈ (0, 1]`: the logarithm is finite.
            let radius = (-2.0 * (1.0 - unit()).ln()).sqrt();
            let theta = std::f64::consts::TAU * unit();
            values.push((radius * theta.cos()) as f32);
            values.push((radius * theta.sin()) as f32);
        }
        values.truncate(n);
        Self {
            values,
            frames,
            source: NoiseSource::Seeded { seed },
        }
    }

    /// Exact caller-supplied noise (the reference's, for parity). Refuses a ragged, empty or
    /// non-finite tensor.
    pub fn injected(values: Vec<f32>, frames: usize) -> gen_core::Result<Self> {
        if frames == 0 || values.len() != frames * LATENT_CHANNELS {
            return Err(gen_core::Error::Msg(format!(
                "YuE2 acoustic noise: {} values do not form [{frames}, {LATENT_CHANNELS}] with \
                 at least one frame",
                values.len()
            )));
        }
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            return Err(gen_core::Error::Msg(format!(
                "YuE2 acoustic noise contains a non-finite value at frame {}",
                i / LATENT_CHANNELS
            )));
        }
        Ok(Self {
            values,
            frames,
            source: NoiseSource::Injected,
        })
    }

    /// Frames.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Row-major `[frames, 64]` values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Where the noise came from.
    pub fn source(&self) -> &NoiseSource {
        &self.source
    }

    /// SHA-256 of the values as little-endian F32 bytes.
    pub fn sha256(&self) -> String {
        crate::latent::sha256_f32(&self.values)
    }

    /// Rows `start..end` as a `[end − start, 64]` host F32 tensor.
    fn rows(&self, start: usize, end: usize) -> gen_core::Result<Tensor> {
        let slice = &self.values[start * LATENT_CHANNELS..end * LATENT_CHANNELS];
        Tensor::from_slice(slice, (end - start, LATENT_CHANNELS), &Device::Cpu)
            .map_err(backend("noise"))
    }
}

/// How many NAR query rows one attention call computes (upstream `query_chunk_size`). Every
/// setting attends the full visible key set and computes the same function; only the size of the
/// temporary score tile changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryTile {
    /// Upstream's CPU / MPS default, [`UPSTREAM_QUERY_TILE`] rows.
    Upstream,
    /// Every query row at once (upstream's CUDA default).
    Whole,
    /// At most this many rows. `Rows(0)` is refused (upstream refuses `query_chunk_size < 1`).
    Rows(usize),
    /// A memory budget in bytes for the attention scores of one call: as many rows as keep
    /// [`SCORE_TILES_LIVE`] score-sized tiles (`heads · rows · keys` elements of the model dtype
    /// each) within the budget. It bounds the score temporaries only — not the weights, the KV
    /// cache or the activations. A budget too small for one row is refused, never rounded up.
    ScoreBytes(usize),
}

/// Score-sized tiles one attention call can hold at once in `sdpa_gqa`: the raw `QKᵀ` product
/// while it is scaled, the scaled scores, and the softmax weights while the scores are still alive.
/// [`QueryTile::ScoreBytes`] divides its budget by this, so the budget covers the peak, not one tile.
pub const SCORE_TILES_LIVE: usize = 3;

impl QueryTile {
    /// Query rows per attention call over `queries` rows and `keys` keys (at most `queries`).
    pub fn rows(
        self,
        queries: usize,
        heads: usize,
        keys: usize,
        dtype: DType,
    ) -> gen_core::Result<usize> {
        let rows = match self {
            QueryTile::Upstream => UPSTREAM_QUERY_TILE,
            QueryTile::Whole => queries,
            QueryTile::Rows(0) => {
                return Err(gen_core::Error::Msg(
                    "YuE2 acoustic: a query tile must hold at least one row".into(),
                ))
            }
            QueryTile::Rows(n) => n,
            QueryTile::ScoreBytes(bytes) => {
                let per_row = SCORE_TILES_LIVE * heads * keys * dtype.size_in_bytes();
                if bytes < per_row {
                    return Err(gen_core::Error::Msg(format!(
                        "YuE2 acoustic: a {bytes}-byte score budget cannot hold one query row \
                         ({per_row} bytes for {SCORE_TILES_LIVE} score tiles of {heads} heads × \
                         {keys} keys)"
                    )));
                }
                bytes / per_row
            }
        };
        Ok(rows.min(queries))
    }
}

/// Memory controls of the acoustic stage. Neither changes the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NarOptions {
    /// Attention query tiling.
    pub query_tile: QueryTile,
    /// Move the AR-only weights to host memory while each chunk is solved (upstream `offload_ar`).
    pub offload_ar: bool,
}

impl Default for NarOptions {
    /// Upstream's defaults off CUDA: 256-row query tiles, no offload.
    fn default() -> Self {
        Self {
            query_tile: QueryTile::Upstream,
            offload_ar: false,
        }
    }
}

/// Which evaluation of a midpoint step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// `v(x, t)`.
    First,
    /// `v(x − v(x, t)·dt/2, t − dt/2)`.
    Midpoint,
}

/// One velocity evaluation, as the observer sees it.
#[derive(Debug)]
pub struct Evaluation<'a> {
    /// Chunk index.
    pub chunk: usize,
    /// Midpoint step (0-based).
    pub step: usize,
    /// Which evaluation of the step.
    pub stage: Stage,
    /// The clamped `logit(t)` the model received (before the sigmoid shift).
    pub raw_t: f32,
    /// The state evaluated, `[frames, 64]` in the model dtype.
    pub input: &'a Tensor,
    /// The velocity, same shape.
    pub velocity: &'a Tensor,
}

/// Observes a synthesis. Every method has a no-op default; nothing is copied off the device unless
/// an implementation does it.
pub trait SynthesisObserver {
    /// After each midpoint step: `completed` of `total` steps over all chunks.
    fn on_progress(&mut self, completed: usize, total: usize) {
        let _ = (completed, total);
    }

    /// After each velocity evaluation.
    fn on_evaluation(&mut self, evaluation: &Evaluation<'_>) {
        let _ = evaluation;
    }

    /// The state after step `step` of chunk `chunk`, `[frames, 64]` in the model dtype.
    fn on_step(&mut self, chunk: usize, step: usize, state: &Tensor) {
        let _ = (chunk, step, state);
    }
}

/// The no-op observer.
impl SynthesisObserver for () {}

/// Cancellation and observation for a synthesis.
pub struct SynthesisHooks<'a> {
    /// Polled at every bounded boundary (see the module docs); `true` cancels.
    pub cancelled: &'a dyn Fn() -> bool,
    /// Receives progress, evaluations and step states.
    pub observer: &'a mut dyn SynthesisObserver,
}

/// One song's acoustic synthesis — upstream `synthesize(model, prefix, codec, seed, steps,
/// context)`'s arguments (the memory knobs are [`NarOptions`]).
#[derive(Clone, Copy, Debug)]
pub struct SynthesisRequest<'a> {
    /// The plan's semantic prefix (`EOD … MUSIC_START`), vocabulary ids.
    pub prefix: &'a [u32],
    /// The semantic stage's codec indices, `0 … CODEC_SIZE − 1`.
    pub codes: &'a [u32],
    /// The song's starting noise, one row per codec frame.
    pub noise: &'a SongNoise,
    /// Midpoint steps (`GenerationConfig::ode_steps`, 32 released).
    pub steps: usize,
    /// The acoustic context the chunks are cut for, `1 … CONTEXT` ([`protocol::CONTEXT`]
    /// released).
    pub context: usize,
}

impl SynthesisRequest<'_> {
    /// SHA-256 over everything the latents are a function of: the schema, the protocol version,
    /// the model weights and the dtype they compute in, the prefix, the codec ids, the noise
    /// values, the step count, the context and the ODE method. The memory controls
    /// ([`NarOptions`]) are deliberately absent.
    pub fn stage_identity(&self, weights_sha256: &str, dtype: DType) -> String {
        // A JSON array, not an object: its byte form does not depend on serde_json's map-order
        // feature, which workspace feature unification could otherwise flip between builds.
        let record = json!([
            STAGE_IDENTITY_SCHEMA,
            PROTOCOL_VERSION,
            weights_sha256,
            dtype.as_str(),
            self.prefix,
            self.codes,
            self.noise.sha256(),
            self.steps,
            self.context,
            ODE_METHOD,
        ]);
        let digest = Sha256::digest(record.to_string().as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// What a synthesis produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Synthesis {
    /// `[frames, 64]` F32 latents, [`LatentSource::Synthesis`] with the stage identity.
    pub latents: AcousticLatents,
    /// The original chunks solved, `[a, b)` frame ranges.
    pub chunks: Vec<(usize, usize)>,
    /// Bytes moved off the device per offload (`0` without offload or on a host model).
    pub offloaded_bytes: usize,
    /// Wall time, seconds.
    pub seconds: f64,
}

fn check_cancel(cancelled: &dyn Fn() -> bool) -> gen_core::Result<()> {
    if cancelled() {
        Err(gen_core::Error::Canceled)
    } else {
        Ok(())
    }
}

/// `logit(t)` in F64, clamped — upstream `torch.logit(float64).clamp(-20, 20).item()`.
fn clamped_logit(t: f64) -> f64 {
    (t / (1.0 - t)).ln().clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
}

/// Bidirectional grouped-query attention of `q` `[1, heads, n, d]` over **all** of `keys` /
/// `values` `[1, kv_heads, k, d]`, `rows` query rows per call.
fn attend(
    q: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    scale: f32,
    rows: usize,
) -> candle_llm::Result<Tensor> {
    let n = q.dim(2)?;
    if rows >= n {
        return sdpa_gqa(q, keys, values, scale, AttnMask::None);
    }
    let mut tiles = Vec::with_capacity(n.div_ceil(rows));
    for start in (0..n).step_by(rows) {
        let tile = q.narrow(2, start, rows.min(n - start))?;
        tiles.push(sdpa_gqa(&tile, keys, values, scale, AttnMask::None)?);
    }
    Ok(Tensor::cat(&tiles, 2)?)
}

/// One original chunk to solve — upstream `nar.Chunk` (see [`solve_chunk`]).
#[derive(Clone, Copy, Debug)]
pub struct ChunkInput<'a> {
    /// The chunk's AR sequence (`prefix`, codec ids `+ CODEC_OFFSET`, `MUSIC_END`).
    pub ar_tokens: &'a [u32],
    /// The chunk's starting noise `[frames, 64]` (host F32).
    pub noise: &'a Tensor,
    /// `0`: every AR key is visible (the released synthesis). `> 0`: only the first
    /// `min(nar_cond_end, ar)` AR keys are (upstream's text-only mode).
    pub nar_cond_end: usize,
}

/// The per-chunk state of upstream `CachedNAR`: the prefilled cache (visible AR keys + room for
/// one evaluation's NAR keys), the NAR RoPE tables and position embedding.
struct ChunkSolver {
    cache: StaticKvCache,
    visible: usize,
    frames: usize,
    cos: Tensor,
    sin: Tensor,
    positions: Tensor,
}

impl ChunkSolver {
    fn prefill(
        nar: &Yue2Nar,
        input: &ChunkInput<'_>,
        tile: QueryTile,
        cancelled: &dyn Fn() -> bool,
    ) -> gen_core::Result<Self> {
        let lm = &nar.lm;
        let refuse = |what: String| Err(gen_core::Error::Msg(format!("YuE2 acoustic: {what}")));
        let (frames, width) = input.noise.dims2().map_err(backend("noise"))?;
        if frames == 0 || width != LATENT_CHANNELS {
            return refuse(format!(
                "expected non-empty noise [frames, {LATENT_CHANNELS}], got {:?}",
                input.noise.dims()
            ));
        }
        let ar = input.ar_tokens.len();
        let nar_len = frames + 2;
        if ar == 0 {
            return refuse("the AR prefix is empty".into());
        }
        if ar + nar_len > lm.config().max_position_embeddings {
            return refuse(format!(
                "chunk of {ar} AR + {nar_len} NAR positions exceeds the model's {} positions",
                lm.config().max_position_embeddings
            ));
        }
        let visible = if input.nar_cond_end > 0 {
            input.nar_cond_end.min(ar)
        } else {
            ar
        };
        // Refuse an unusable tile before any work (the key count is fixed for the chunk).
        tile.rows(
            nar_len,
            lm.config().num_attention_heads,
            visible + nar_len,
            lm.dtype(),
        )?;
        let mut cache = lm.new_cache(ar.max(visible + nar_len))?;
        lm.prefill(input.ar_tokens, &mut cache, || check_cancel(cancelled))?;
        let (cos, sin) = lm.rope_tables(ar, nar_len)?;
        let positions = nar
            .heads
            .position_embedding(nar_len)
            .map_err(backend("latent positions"))?;
        Ok(Self {
            cache,
            visible,
            frames,
            cos,
            sin,
            positions,
        })
    }

    /// Upstream `CachedNAR.velocity(state, raw_t)`.
    fn velocity(
        &mut self,
        nar: &Yue2Nar,
        state: &Tensor,
        raw_t: f32,
        tile: QueryTile,
    ) -> gen_core::Result<Tensor> {
        let lm = &nar.lm;
        let heads = &nar.heads;
        let cfg = lm.config();
        let err = backend("acoustic velocity");
        if state.dims() != [self.frames, LATENT_CHANNELS] {
            return Err(gen_core::Error::Msg(format!(
                "YuE2 acoustic: ODE state shape changed to {:?}",
                state.dims()
            )));
        }
        let pad =
            Tensor::zeros((1, LATENT_CHANNELS), state.dtype(), state.device()).map_err(&err)?;
        let x_nar = Tensor::cat(&[&pad, state, &pad], 0)
            .and_then(|x| x.unsqueeze(0))
            .map_err(&err)?;
        let time = heads
            .time_embedding(raw_t)
            .map_err(backend("time embedding"))?;
        let mut x = heads.vae2llm.0.forward(&x_nar, Some(&heads.vae2llm.1))
            .map_err(backend("vae2llm"))?
            .broadcast_add(&time)
            .and_then(|x| x.broadcast_add(&self.positions))
            .map_err(&err)?;
        // Drop the previous evaluation's NAR keys: the next writes land right after the visible
        // AR prefix, which is never overwritten.
        self.cache
            .truncate(self.visible as i32)
            .map_err(backend("kv cache"))?;
        let keys_total = self.visible + self.frames + 2;
        let rows = tile.rows(
            self.frames + 2,
            cfg.num_attention_heads,
            keys_total,
            lm.dtype(),
        )?;
        for (i, layer) in lm.layers().iter().enumerate() {
            let p = layer.nar.as_ref().ok_or_else(|| {
                gen_core::Error::Msg(
                    "YuE2 acoustic: the model was loaded without the NAR path".into(),
                )
            })?;
            let normed = rms_norm(&x, &p.attn_norm, cfg.rms_norm_eps).map_err(&err)?;
            let (q, k, v) = p.attn.project_qkv(&normed, &self.cos, &self.sin)?;
            let (keys, values) = self.cache.update(i, &k, &v).map_err(backend("kv cache"))?;
            debug_assert_eq!(keys.dim(2).ok(), Some(keys_total));
            let h =
                attend(&q, &keys, &values, p.attn.scale(), rows).map_err(backend("attention"))?;
            x = (x + p.attn.project_out(&h)?).map_err(&err)?;
            let normed = rms_norm(&x, &p.mlp_norm, cfg.rms_norm_eps).map_err(&err)?;
            x = (&x + p.mlp.forward(&normed)?).map_err(&err)?;
        }
        let x = rms_norm(&x, lm.final_norm(), cfg.rms_norm_eps).map_err(&err)?;
        heads.llm2vae.0.forward(&x, Some(&heads.llm2vae.1))
            .map_err(backend("llm2vae"))?
            .squeeze(0)
            .and_then(|v| v.narrow(0, 1, self.frames))
            .map_err(err)
    }

    /// Upstream `CachedNAR.solve`: the explicit midpoint rule from `t = 1` to `t = 0`.
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        nar: &Yue2Nar,
        noise: &Tensor,
        steps: usize,
        tile: QueryTile,
        chunk: usize,
        chunks: usize,
        hooks: &mut SynthesisHooks<'_>,
    ) -> gen_core::Result<Tensor> {
        let err = backend("acoustic solve");
        let lm = &nar.lm;
        let mut state = noise
            .to_device(lm.device())
            .and_then(|t| t.to_dtype(lm.dtype()))
            .map_err(&err)?;
        let dt = 1.0 / steps as f64;
        for step in 0..steps {
            check_cancel(hooks.cancelled)?;
            let t = 1.0 - step as f64 * dt;
            let raw = clamped_logit(t) as f32;
            let first = self.velocity(nar, &state, raw, tile)?;
            hooks.observer.on_evaluation(&Evaluation {
                chunk,
                step,
                stage: Stage::First,
                raw_t: raw,
                input: &state,
                velocity: &first,
            });
            let mid = (&state - (first * (dt / 2.0)).map_err(&err)?).map_err(&err)?;
            check_cancel(hooks.cancelled)?;
            let raw_mid = clamped_logit(t - dt / 2.0) as f32;
            let second = self.velocity(nar, &mid, raw_mid, tile)?;
            hooks.observer.on_evaluation(&Evaluation {
                chunk,
                step,
                stage: Stage::Midpoint,
                raw_t: raw_mid,
                input: &mid,
                velocity: &second,
            });
            state = (&state - (second * dt).map_err(&err)?).map_err(&err)?;
            hooks.observer.on_step(chunk, step, &state);
            hooks
                .observer
                .on_progress(chunk * steps + step + 1, chunks * steps);
        }
        let result = state
            .to_dtype(DType::F32)
            .and_then(|t| t.to_device(&Device::Cpu))
            .map_err(&err)?;
        let host: Vec<f32> = result
            .flatten_all()
            .and_then(|t| t.to_vec1())
            .map_err(&err)?;
        if host.iter().any(|v| !v.is_finite()) {
            return Err(gen_core::Error::Msg(
                "YuE2 acoustic flow matching produced non-finite latents".into(),
            ));
        }
        Ok(result)
    }
}

/// Solve one original chunk end to end — upstream `CachedNAR(model, chunk).solve(steps)` inside
/// `_offload_ar(model, offload_ar)`: prefill, offload (if asked), solve, release the cache, restore.
/// `chunk` / `chunks` only label progress. Returns host F32 `[frames, 64]` and the bytes offloaded.
///
/// The AR weights come back on success, error, cancellation and a panic inside the solve (caught,
/// restored, then resumed). If the restore itself fails, its error is appended to the original one
/// and the model stays marked offloaded, so the AR path keeps refusing until
/// [`Yue2Nar::restore_ar`] succeeds.
#[allow(clippy::too_many_arguments)]
pub fn solve_chunk(
    nar: &mut Yue2Nar,
    input: &ChunkInput<'_>,
    steps: usize,
    options: &NarOptions,
    chunk: usize,
    chunks: usize,
    hooks: &mut SynthesisHooks<'_>,
) -> gen_core::Result<(Tensor, usize)> {
    if steps == 0 {
        return Err(gen_core::Error::Msg(
            "YuE2 acoustic: steps must be a positive integer".into(),
        ));
    }
    if nar.lm.ar_offloaded() {
        return Err(gen_core::Error::Msg(
            "YuE2 acoustic: the AR path is still offloaded from an interrupted synthesis; \
             restore it first (`Yue2Nar::restore_ar`)"
                .into(),
        ));
    }
    check_cancel(hooks.cancelled)?;
    let mut solver = ChunkSolver::prefill(nar, input, options.query_tile, hooks.cancelled)?;
    let offloaded = if options.offload_ar {
        match nar.lm.offload_ar() {
            Ok(bytes) => bytes,
            Err(e) => {
                drop(solver);
                return Err(with_restore(e, nar.lm.restore_ar()));
            }
        }
    } else {
        0
    };
    // A panic inside the solve (a caller's observer, say) must not leave the AR weights on the
    // host: catch it, release the cache and restore, then let it continue unwinding.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solver.solve(
            nar,
            input.noise,
            steps,
            options.query_tile,
            chunk,
            chunks,
            hooks,
        )
    }));
    // Release the prefix cache before the AR weights come back (upstream's ordering), on every
    // path.
    drop(solver);
    let restored = if options.offload_ar {
        nar.lm.restore_ar()
    } else {
        Ok(())
    };
    let result = match result {
        Ok(result) => result,
        // A restore failure here leaves the model marked offloaded, so the AR path keeps refusing.
        Err(panic) => std::panic::resume_unwind(panic),
    };
    match (result, restored) {
        (Ok(latents), Ok(())) => Ok((latents, offloaded)),
        (Ok(_), Err(restore)) => Err(restore),
        (Err(e), restored) => Err(with_restore(e, restored)),
    }
}

/// `original`, unless restoring the AR path afterwards also failed — then both, the original first
/// (a restore failure never replaces the error that caused it, nor is it dropped).
fn with_restore(original: gen_core::Error, restored: gen_core::Result<()>) -> gen_core::Error {
    match restored {
        Ok(()) => original,
        Err(restore) => gen_core::Error::Msg(format!(
            "{original}; restoring the offloaded AR path afterwards also failed: {restore}"
        )),
    }
}

/// Chunk `codes`' AR sequence — upstream `song_chunks`: `prefix + [c + CODEC_OFFSET …] +
/// [MUSIC_END]`.
fn chunk_ar_tokens(prefix: &[u32], codes: &[u32]) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(prefix.len() + codes.len() + 1);
    tokens.extend_from_slice(prefix);
    tokens.extend(codes.iter().map(|&c| c + CODEC_OFFSET));
    tokens.push(MUSIC_END);
    tokens
}

/// Synthesize one song's acoustic latents — upstream `synthesize` (see the module docs).
pub fn synthesize(
    nar: &mut Yue2Nar,
    request: &SynthesisRequest<'_>,
    options: &NarOptions,
    mut hooks: SynthesisHooks<'_>,
) -> gen_core::Result<Synthesis> {
    let start = Instant::now();
    let refuse = |what: String| Err(gen_core::Error::Msg(format!("YuE2 acoustic: {what}")));
    if request.prefix.is_empty() || request.codes.is_empty() {
        return refuse("the prefix and the codec tokens must both be non-empty".into());
    }
    if let Some(&bad) = request.codes.iter().find(|&&c| c >= CODEC_SIZE) {
        return refuse(format!(
            "codec index {bad} is outside the {CODEC_SIZE}-entry codebook"
        ));
    }
    let vocab = nar.lm.config().vocab_size;
    if let Some(&bad) = request.prefix.iter().find(|&&t| t as usize >= vocab) {
        return refuse(format!(
            "prefix id {bad} is outside the {vocab}-token vocabulary"
        ));
    }
    if request.noise.frames() != request.codes.len() {
        return refuse(format!(
            "{} noise frames for {} codec frames",
            request.noise.frames(),
            request.codes.len()
        ));
    }
    if request.steps == 0 {
        return refuse("steps must be a positive integer".into());
    }
    let chunks = protocol::chunk_ranges(request.codes.len(), request.prefix.len(), request.context)
        .map_err(|e| gen_core::Error::Msg(format!("YuE2 acoustic: {e}")))?;
    let mut out = Vec::with_capacity(chunks.len());
    let mut offloaded_bytes = 0;
    for (index, &(a, b)) in chunks.iter().enumerate() {
        check_cancel(hooks.cancelled)?;
        let ar_tokens = chunk_ar_tokens(request.prefix, &request.codes[a..b]);
        let noise = request.noise.rows(a, b)?;
        let input = ChunkInput {
            ar_tokens: &ar_tokens,
            noise: &noise,
            nar_cond_end: 0,
        };
        let (latents, moved) = solve_chunk(
            nar,
            &input,
            request.steps,
            options,
            index,
            chunks.len(),
            &mut hooks,
        )?;
        offloaded_bytes = offloaded_bytes.max(moved);
        out.push(latents);
    }
    let all = Tensor::cat(&out, 0).map_err(backend("acoustic concat"))?;
    let source = LatentSource::Synthesis {
        stage_identity: request.stage_identity(&nar.weights_sha256, nar.lm.dtype()),
    };
    let latents = AcousticLatents::from_tensor(&all, source)
        .map_err(|e| gen_core::Error::Msg(format!("YuE2 acoustic latents: {e}")))?;
    Ok(Synthesis {
        latents,
        chunks,
        offloaded_bytes,
        seconds: start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
pub(crate) mod synthetic {
    //! The synthetic MoT of [`crate::model::synthetic`] plus NAR heads whose weights are the same
    //! integer hash of (tensor name, element index) — bit-identical to
    //! `scripts/reference/yue2/nar_fixtures.py`'s `synthetic_heads`, so the committed upstream
    //! reference latents apply with nothing but the fixture committed.
    use super::*;
    use crate::model::synthetic as mot;

    /// The synthetic model's latent-position table (smaller than a short chunk, so the clamp runs).
    pub(crate) const MAX_LATENT_FRAMES: usize = 64;

    pub(crate) fn nar_config(timestep_shift: f64) -> NarConfig {
        NarConfig {
            latent_dim: LATENT_CHANNELS,
            max_latent_frames: MAX_LATENT_FRAMES,
            timestep_shift,
        }
    }

    /// `(name, shape, scale, offset)` of every NAR head tensor.
    pub(crate) fn heads_state_dict(hidden: usize) -> Vec<(String, Vec<usize>, f32, f32)> {
        let proj = |fan_in: usize| 1.7 / (fan_in as f32).sqrt();
        let d = LATENT_CHANNELS;
        vec![
            ("vae2llm.weight".into(), vec![hidden, d], proj(d), 0.0),
            ("vae2llm.bias".into(), vec![hidden], 0.1, 0.0),
            ("llm2vae.weight".into(), vec![d, hidden], proj(hidden), 0.0),
            ("llm2vae.bias".into(), vec![d], 0.1, 0.0),
            (
                "time_embedder.mlp.0.weight".into(),
                vec![hidden, TIME_FREQUENCIES],
                proj(TIME_FREQUENCIES),
                0.0,
            ),
            ("time_embedder.mlp.0.bias".into(), vec![hidden], 0.1, 0.0),
            (
                "time_embedder.mlp.2.weight".into(),
                vec![hidden, hidden],
                proj(hidden),
                0.0,
            ),
            ("time_embedder.mlp.2.bias".into(), vec![hidden], 0.1, 0.0),
            (
                "latent_pos_embed.pe".into(),
                vec![MAX_LATENT_FRAMES, hidden],
                1.0,
                0.0,
            ),
        ]
    }

    /// The synthetic MoT with NAR heads on the CPU in F32.
    pub(crate) fn model(timestep_shift: f64) -> Yue2Nar {
        model_on(timestep_shift, &Device::Cpu)
    }

    /// The synthetic MoT with NAR heads on `device` in F32.
    pub(crate) fn model_on(timestep_shift: f64, device: &Device) -> Yue2Nar {
        model_on_dtype(timestep_shift, device, DType::F32)
    }

    /// Every synthetic tensor (MoT and NAR heads) by name, F32 on the CPU.
    pub(crate) fn all_tensors() -> std::collections::HashMap<String, Tensor> {
        let cfg = mot::config();
        let mut tensors = mot::tensors(&cfg);
        for (name, shape, scale, offset) in heads_state_dict(cfg.hidden_size) {
            let n = shape.iter().product();
            let data = mot::values(&name, n, scale, offset);
            let t = Tensor::from_vec(data, shape, &Device::Cpu).expect("synthetic tensor");
            tensors.insert(name, t);
        }
        tensors
    }

    /// The synthetic MoT with NAR heads on `device` computing in `dtype`.
    pub(crate) fn model_on_dtype(timestep_shift: f64, device: &Device, dtype: DType) -> Yue2Nar {
        let vb = VarBuilder::from_tensors(all_tensors(), dtype, device);
        Yue2Nar::from_var_builder(
            mot::config(),
            nar_config(timestep_shift),
            vb,
            "synthetic".into(),
        )
        .expect("synthetic YuE2 with NAR heads loads")
    }
}

#[cfg(test)]
mod tests;
