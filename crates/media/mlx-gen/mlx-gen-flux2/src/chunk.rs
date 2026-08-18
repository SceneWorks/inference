//! Activation-peak control for the FLUX.2 MMDiT denoise forward (sc-6266).
//!
//! A FLUX.2-dev **multi-reference** edit concatenates each reference image's ~4096 latent tokens onto
//! the joint `[txt, target, ref0, ref1, …]` DiT sequence (`model.rs`), so a 2-reference 1024² edit
//! runs the 8 double + 48 single blocks over ~12.8K tokens and is **activation-bound during denoise**
//! — it peaks ~104 GB (sc-5923 / sc-6124), over the model's `minMemoryGb = 96`. The shipped paths
//! (T2I, single-reference edit, strict-pose, LoRA) run a shorter sequence and fit, so these knobs are
//! **gated on sequence length** (`model.rs`) and default **OFF** — the shipped forward stays
//! byte-identical.
//!
//! The levers mirror the SCAIL-2 sc-5681 `mlx-gen-wan` `DitMemoryConfig` (kept flux2-local so the
//! shipped Wan video path is untouched; a future DRY-to-`mlx-gen`-core is tracked separately):
//! 1. **Lazy-graph depth** ([`MemoryConfig::eval_cadence`]). The whole 8+48-block forward is one
//!    lazy graph evaluated once per denoise step, so without intervention the peak holds many blocks'
//!    transients at once; force-evaluating every *n*-th block's output before the next caps it at
//!    ~that many blocks'. **Bit-exact** at every cadence — it only forces materialization, so the
//!    multi-reference edit's *pixels* are unchanged, only its memory schedule. This is the dominant
//!    lever and the production default for the long-sequence path ([`MemoryConfig::LONG_SEQ`]).
//! 2. **The FFN intermediate** ([`MemoryConfig::ffn_seq_chunk`]). The double block's image FFN
//!    materializes a `[L, 2·mlp_ratio·inner]` SwiGLU intermediate — the largest single transient;
//!    running it over sequence row-blocks bounds it. **Numerically equivalent, not bit-identical**
//!    (the FFN is per-token so the math is unchanged, but MLX's Metal GEMM is tile-specialized by the
//!    row dimension → cosine ≈ 1, max|Δ| ~1e-3, the model's own torch-parity class). Off by default;
//!    available as headroom for extreme configs (3+ references / higher resolution) and tunable from
//!    the environment without a recompile.
//!
//! # sc-18317: both levers are now request-selectable typed domains
//!
//! Until sc-18317 these two were reachable only from the sequence-length gate in `model.rs` and from
//! the environment, so epic 18304's execution planner could neither discover nor select them — and a
//! caller who did set them had no way to be told a provider had ignored them. They are now the
//! shared typed [`GraphEvalCadence`] / [`FfnChunk`] domains on
//! [`gen_core::GenerationMemory`](mlx_gen::gen_core::GenerationMemory), declared on this family's
//! `Capabilities::execution` surface and admitted (or refused by name) at the shared request floor.
//!
//! The bool became a cadence in the process: `eval_per_block: true` is exactly
//! [`GraphEvalCadence::EVERY_BLOCK`], `false` is `None`, and every intermediate cadence is new
//! reachable range rather than changed behaviour. [`MemoryConfig::with_request`] is the one place a
//! request overlays the gate's base config, and it is applied **after**
//! [`MemoryConfig::from_env`] so an explicit planner selection wins over a deployment-wide default
//! (the env knobs remain the way to steer a route the planner does not select).
//!
//! Attention needs no query-chunking lever here: FLUX.2 attention is flash
//! `scaled_dot_product_attention` (`transformer.rs`), which never materializes the `[heads, L, L]`
//! score matrix, so `eval_per_block` already bounds the per-block attention transient (the SCAIL-2
//! `attn_query_chunk` lever is for the materialized-SDPA fallback and is OFF there too).

use std::num::NonZeroU32;

use mlx_gen::gen_core::{
    CfgBatchingDomain, ExecutionSurface, ExecutionValueDomain, FfnChunk, GenerationMemory,
    GraphEvalCadence,
};
use mlx_gen::Result;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

/// The execution-domain surface every FLUX.2 descriptor advertises (sc-18317).
///
/// Both levers accept any positive value by construction rather than a measured candidate set:
/// [`map_seq_chunks`] degrades a chunk at least as large as the sequence to the single whole-sequence
/// call, and a cadence wider than a block stack simply forces no evaluation inside it (the denoise
/// step's own end-of-step evaluation is unchanged) — so the mechanism is exact over the whole range
/// and this declaration claims no per-value measurement (see [`ExecutionValueDomain::AtLeast`]). CFG batching stays `Unsupported`: FLUX.2 guidance is
/// distilled/embedded, not a two-branch classifier-free batch, so there is no batching axis to
/// select and a request naming one must be refused rather than quietly ignored.
pub const EXECUTION_SURFACE: ExecutionSurface = ExecutionSurface {
    graph_eval_cadence_blocks: ExecutionValueDomain::ANY_POSITIVE,
    ffn_chunk_rows: ExecutionValueDomain::ANY_POSITIVE,
    cfg_batching: CfgBatchingDomain::Unsupported,
};

/// Knobs that bound the per-step activation high-water of the FLUX.2 MMDiT denoise so a long-sequence
/// multi-reference edit fits under `minMemoryGb` (sc-6266). All configs produce the same image up to
/// the kernel-rounding class noted per field; [`OFF`](Self::OFF) is the historical whole-sequence,
/// single-eval-per-step behaviour and is byte-identical to today's shipped forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryConfig {
    /// Run each double block's **image** FFN (`[L, 2·mlp_ratio·inner]` SwiGLU intermediate) over
    /// sequence row-blocks of at most this many tokens. `None` ⇒ the whole sequence at once (the
    /// no-op fast path). Numerically equivalent, not bit-identical (see the module doc).
    pub ffn_seq_chunk: Option<FfnChunk>,
    /// Force-evaluate (and free) every *n*-th transformer block's output before starting the next, so
    /// the step's peak is ~that many blocks' activations instead of the whole-depth lazy graph.
    /// `None` ⇒ no forced evaluation inside the forward (one eval at the end of the step).
    /// **Bit-exact at every cadence.**
    ///
    /// Generalized from the pre-sc-18317 `eval_per_block: bool`:
    /// [`GraphEvalCadence::EVERY_BLOCK`] is that bool's `true`, `None` is its `false`.
    pub eval_cadence: Option<GraphEvalCadence>,
}

impl MemoryConfig {
    /// No activation control — whole-sequence FFN with one eval at the end of the step (today's
    /// shipped behaviour). Byte-identical to the pre-sc-6266 forward.
    pub const OFF: Self = Self {
        ffn_seq_chunk: None,
        eval_cadence: None,
    };

    /// Production default for the gated long-sequence (multi-reference edit) path: per-block
    /// evaluation only. Bit-exact (identical pixels), and on its own brings the 2-reference 1024²
    /// edit well under `minMemoryGb = 96`. FFN chunking stays available as request/env-tunable
    /// headroom.
    pub const LONG_SEQ: Self = Self {
        ffn_seq_chunk: None,
        eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
    };

    /// `true` if no lever is active (the [`OFF`](Self::OFF) fast path — skip the chunk plumbing).
    pub fn is_off(&self) -> bool {
        self.ffn_seq_chunk.is_none() && self.eval_cadence.is_none()
    }

    /// The FFN chunk in the `usize` sequence rows [`map_seq_chunks`] takes.
    pub fn ffn_chunk_rows(&self) -> Option<usize> {
        self.ffn_seq_chunk.map(FfnChunk::rows_usize)
    }

    /// Whether the block at zero-based `index` **within one stack** is an evaluation boundary.
    ///
    /// The single decision point behind both block loops (`transformer.rs`), replacing the two
    /// `if mem.eval_per_block` branches so a cadence cannot be honoured in one stack and dropped in
    /// the other. The per-stack index restart is [`GraphEvalCadence::evaluates_after_block`]'s
    /// documented contract: the double→single boundary is already an evaluation point.
    pub fn evaluates_after_block(&self, index: usize) -> bool {
        self.eval_cadence
            .is_some_and(|cadence| cadence.evaluates_after_block(index))
    }

    /// Overlay the environment onto `base` so a deployment can tune the memory/throughput tradeoff
    /// without a recompile:
    ///   * `MLX_GEN_FLUX2_FFN_SEQ_CHUNK` — FFN sequence chunk (`0` disables; unset keeps `base`).
    ///   * `MLX_GEN_FLUX2_EVAL_PER_BLOCK` — the evaluation cadence. `0`/`false`/`off` disables;
    ///     `1`/`true`/`on` is every block; a positive integer *n* is every *n*-th block (the
    ///     sc-18317 generalization — the boolean spellings keep their exact previous meaning, since
    ///     `1` *is* every block).
    pub fn from_env(base: Self) -> Self {
        Self {
            ffn_seq_chunk: env_chunk("MLX_GEN_FLUX2_FFN_SEQ_CHUNK", base.ffn_seq_chunk),
            eval_cadence: env_cadence("MLX_GEN_FLUX2_EVAL_PER_BLOCK", base.eval_cadence),
        }
    }

    /// Overlay one request's typed execution selections onto `base` (sc-18317).
    ///
    /// The **only** seam by which a request reaches these levers, so the planner's selection and the
    /// sequence-length gate cannot disagree about precedence: a set field replaces the base value, an
    /// unset field leaves the base — which is what makes a request that selects neither
    /// byte-for-byte the pre-sc-18317 render on every route, gated or not.
    ///
    /// Domain admission is **not** repeated here: `Capabilities::validate_request` has already
    /// refused any value outside [`EXECUTION_SURFACE`] before `generate` reaches a forward, and
    /// re-deriving the domain at the consumer is how the two drift apart.
    pub fn with_request(base: Self, memory: Option<&GenerationMemory>) -> Self {
        let Some(memory) = memory else {
            return base;
        };
        Self {
            ffn_seq_chunk: memory.ffn_chunk.or(base.ffn_seq_chunk),
            eval_cadence: memory.graph_eval_cadence.or(base.eval_cadence),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self::OFF
    }
}

/// An [`FfnChunk`] knob from `var`: a positive integer enables, `0` disables (`None`), anything else
/// (unset / unparseable) keeps `base`.
fn env_chunk(var: &str, base: Option<FfnChunk>) -> Option<FfnChunk> {
    match std::env::var(var) {
        Ok(s) => match s.trim().parse::<u32>() {
            Ok(0) => None,
            Ok(n) => NonZeroU32::new(n).map(FfnChunk::from_nonzero),
            Err(_) => base,
        },
        Err(_) => base,
    }
}

/// A [`GraphEvalCadence`] knob from `var`. Accepts the historical boolean spellings
/// (`1`/`true`/`on`/`yes` ⇒ every block, `0`/`false`/`off`/`no` ⇒ disabled) **and** a positive
/// integer cadence; unset / unrecognized keeps `base`.
fn env_cadence(var: &str, base: Option<GraphEvalCadence>) -> Option<GraphEvalCadence> {
    match std::env::var(var) {
        Ok(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "on" | "yes" => Some(GraphEvalCadence::EVERY_BLOCK),
            "false" | "off" | "no" => None,
            other => match other.parse::<u32>() {
                Ok(0) => None,
                Ok(n) => NonZeroU32::new(n).map(GraphEvalCadence::from_nonzero),
                Err(_) => base,
            },
        },
        Err(_) => base,
    }
}

/// Map a per-token function `f` over sequence row-blocks of `x` `[B, L, *]` and concatenate the
/// results back along the sequence axis. `chunk` `None` / `0` / `≥ L` runs a single `f(&x)` — the
/// no-op fast path, byte-identical to calling `f(&x)` directly (no op here reduces across the
/// sequence axis and `concatenate(split(x)) == x`).
pub fn map_seq_chunks<F>(x: &Array, chunk: Option<usize>, mut f: F) -> Result<Array>
where
    F: FnMut(&Array) -> Result<Array>,
{
    let l = x.shape()[1] as usize;
    let c = match chunk {
        Some(c) if c > 0 && c < l => c,
        _ => return f(x),
    };
    let mut outs: Vec<Array> = Vec::with_capacity(l.div_ceil(c));
    let mut start = 0usize;
    while start < l {
        let len = c.min(l - start);
        let part = slice_seq(x, start as i32, len as i32)?;
        outs.push(f(&part)?);
        start += len;
    }
    let refs: Vec<&Array> = outs.iter().collect();
    Ok(concatenate_axis(&refs, 1)?)
}

/// Contiguous `[:, start:start+len, …]` slice along the sequence axis (axis 1). Use boundary splits
/// so the opt-in FFN chunking path does not build and gather through a host index vector per chunk.
fn slice_seq(x: &Array, start: i32, len: i32) -> Result<Array> {
    Ok(x.split_axis(&[start, start + len], 1)?.swap_remove(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    fn flat(a: &Array) -> Vec<f32> {
        a.reshape(&[-1])
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    #[test]
    fn off_and_long_seq_presets() {
        assert!(MemoryConfig::OFF.is_off());
        assert!(MemoryConfig::default().is_off());
        assert!(!MemoryConfig::LONG_SEQ.is_off());
        // LONG_SEQ is eval-only (bit-exact); FFN chunking stays opt-in headroom.
        assert_eq!(
            MemoryConfig::LONG_SEQ,
            MemoryConfig {
                ffn_seq_chunk: None,
                eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
            }
        );
    }

    /// **sc-18317 default preservation.** The two levers' `None` states must keep meaning exactly
    /// what the pre-story `eval_per_block: false` / `ffn_seq_chunk: None` meant: no forced evaluation
    /// anywhere in the forward, and the whole-sequence FFN.
    #[test]
    fn off_forces_no_evaluation_and_no_chunking() {
        let off = MemoryConfig::OFF;
        assert_eq!(off.ffn_chunk_rows(), None);
        for index in 0..64 {
            assert!(
                !off.evaluates_after_block(index),
                "OFF must not force an evaluation at block {index}"
            );
        }
    }

    /// `EVERY_BLOCK` reproduces the retired `if mem.eval_per_block` branch at every index, and a
    /// wider cadence lands only on its multiples. This is the decision the two `transformer.rs` block
    /// loops read, so an off-by-one here is a silently changed memory schedule.
    #[test]
    fn cadence_boundaries_generalize_the_retired_bool() {
        let every = MemoryConfig::LONG_SEQ;
        for index in 0..64 {
            assert!(every.evaluates_after_block(index), "block {index}");
        }
        let quarter = MemoryConfig {
            eval_cadence: Some(GraphEvalCadence::new(4).unwrap()),
            ..MemoryConfig::OFF
        };
        let boundaries: Vec<usize> = (0..12)
            .filter(|index| quarter.evaluates_after_block(*index))
            .collect();
        assert_eq!(boundaries, vec![3, 7, 11]);
    }

    /// **The request→config seam.** An unset selection must not perturb the gate's base config (the
    /// default-preservation half), and a set one must replace it (the reach half).
    #[test]
    fn with_request_overlays_only_what_the_request_selects() {
        let unset = GenerationMemory::default();
        assert_eq!(
            MemoryConfig::with_request(MemoryConfig::OFF, Some(&unset)),
            MemoryConfig::OFF,
            "an unset selection must leave the shipped OFF route byte-identical"
        );
        assert_eq!(
            MemoryConfig::with_request(MemoryConfig::LONG_SEQ, Some(&unset)),
            MemoryConfig::LONG_SEQ,
            "an unset selection must leave the gated long-sequence default alone"
        );
        assert_eq!(
            MemoryConfig::with_request(MemoryConfig::LONG_SEQ, None),
            MemoryConfig::LONG_SEQ,
            "a request with no memory block at all must leave the base alone"
        );

        let selected = GenerationMemory {
            graph_eval_cadence: Some(GraphEvalCadence::new(8).unwrap()),
            ffn_chunk: Some(FfnChunk::new(2048).unwrap()),
            ..Default::default()
        };
        let overlaid = MemoryConfig::with_request(MemoryConfig::OFF, Some(&selected));
        assert_eq!(
            overlaid.eval_cadence,
            Some(GraphEvalCadence::new(8).unwrap())
        );
        assert_eq!(overlaid.ffn_chunk_rows(), Some(2048));

        // A partial selection overlays only its own field.
        let cadence_only = GenerationMemory {
            graph_eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
            ..Default::default()
        };
        let overlaid = MemoryConfig::with_request(
            MemoryConfig {
                ffn_seq_chunk: Some(FfnChunk::new(64).unwrap()),
                eval_cadence: None,
            },
            Some(&cadence_only),
        );
        assert_eq!(overlaid.ffn_chunk_rows(), Some(64), "base chunk preserved");
        assert!(overlaid.evaluates_after_block(0));
    }

    /// **The CONFLICT case: a request selection outranks a base that already has a value.**
    ///
    /// Distinct from the overlay test above, which only ever resolves a request value against a `None`
    /// base. Precedence is the whole point of the seam — epic 18304's planner is the authority for
    /// these two knobs, above both the sequence-length gate (`model.rs`, which hands in
    /// [`MemoryConfig::LONG_SEQ`] on the long-sequence edit) and the `MLX_GEN_FLUX2_*` deployment
    /// defaults ([`MemoryConfig::from_env`], applied *before* this overlay). Without a `Some`-vs-`Some`
    /// assertion the `.or()` operands can be swapped and every other test still passes, while the
    /// planner's selection silently loses to whatever the route already chose.
    #[test]
    fn a_request_selection_outranks_a_non_off_route_default() {
        // A base with BOTH knobs already set — the gated long-sequence default plus an env-tuned chunk.
        let route_default = MemoryConfig {
            ffn_seq_chunk: Some(FfnChunk::new(4096).unwrap()),
            eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
        };
        let selected = GenerationMemory {
            graph_eval_cadence: Some(GraphEvalCadence::new(8).unwrap()),
            ffn_chunk: Some(FfnChunk::new(2048).unwrap()),
            ..Default::default()
        };
        let overlaid = MemoryConfig::with_request(route_default, Some(&selected));

        assert_eq!(
            overlaid.eval_cadence,
            Some(GraphEvalCadence::new(8).unwrap()),
            "the request's cadence must outrank the route default's cadence"
        );
        assert_eq!(
            overlaid.ffn_chunk_rows(),
            Some(2048),
            "the request's FFN chunk must outrank the route default's chunk"
        );
        assert_ne!(
            overlaid, route_default,
            "a conflicting selection must actually change the config"
        );
        // Read through the consumer's own decision point too, not just the field: at cadence 8 the
        // first block is no longer an evaluation boundary, which it would still be if the base's
        // EVERY_BLOCK had won.
        assert!(
            !overlaid.evaluates_after_block(0),
            "the base's EVERY_BLOCK must not survive a conflicting request cadence"
        );
        assert!(overlaid.evaluates_after_block(7));

        // The same precedence on the real gated base the long-sequence edit route hands in.
        let overlaid = MemoryConfig::with_request(MemoryConfig::LONG_SEQ, Some(&selected));
        assert_eq!(
            overlaid.eval_cadence,
            Some(GraphEvalCadence::new(8).unwrap()),
            "the request must outrank LONG_SEQ's per-block cadence"
        );
        assert_eq!(overlaid.ffn_chunk_rows(), Some(2048));
    }

    /// The declared surface must admit every value this file's own mechanism accepts, and must not
    /// advertise a CFG batching axis FLUX.2 does not have.
    #[test]
    fn declared_execution_surface_is_coherent_and_cfg_free() {
        assert!(EXECUTION_SURFACE.declaration_errors().is_empty());
        assert!(EXECUTION_SURFACE.graph_eval_cadence_blocks.accepts(1));
        assert!(EXECUTION_SURFACE.graph_eval_cadence_blocks.accepts(56));
        assert!(EXECUTION_SURFACE.ffn_chunk_rows.accepts(1));
        assert!(EXECUTION_SURFACE.ffn_chunk_rows.accepts(12_800));
        assert!(!EXECUTION_SURFACE.cfg_batching.is_supported());
    }

    #[test]
    fn map_seq_chunks_is_bit_identical_for_per_token_ops() {
        // [B=2, L=37, D=5] so the last block is a ragged remainder for several chunk sizes.
        let l = 37;
        let d = 5;
        let n = 2 * l * d;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013 - 1.7).collect();
        let x = Array::from_slice(&data, &[2, l, d]);

        // A pure per-token op (elementwise scale) ⇒ chunk-invariant and bit-identical.
        let scale = Array::from_slice(&[2.5f32], &[1]);
        let apply = |chunk: Option<usize>| -> Array {
            map_seq_chunks(&x, chunk, |part| Ok(mlx_rs::ops::multiply(part, &scale)?)).unwrap()
        };
        let full = apply(None);
        for chunk in [Some(1), Some(7), Some(16), Some(37), Some(100)] {
            let chunked = apply(chunk);
            assert_eq!(chunked.shape(), full.shape(), "chunk {chunk:?} shape");
            let (fa, fb) = (flat(&full), flat(&chunked));
            let max_abs = fa
                .iter()
                .zip(&fb)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(max_abs, 0.0, "chunk {chunk:?} not bit-identical");
        }
    }

    /// An unset variable keeps `base` for both knobs — the property the whole gated-route default
    /// rests on. (The set cases are deliberately not exercised here: `RUST_TEST_THREADS=1` makes
    /// process env mutation ordered, but a `set_var` in a library test still leaks into every later
    /// test in the binary, and the parse arms are covered by the cadence/chunk unit tests above.)
    #[test]
    fn env_helpers_keep_the_base_when_unset() {
        let base_chunk = Some(FfnChunk::new(99).unwrap());
        assert_eq!(
            env_chunk("flux2_definitely_unset_xyz", base_chunk),
            base_chunk
        );
        let base_cadence = Some(GraphEvalCadence::new(5).unwrap());
        assert_eq!(
            env_cadence("flux2_definitely_unset_xyz", base_cadence),
            base_cadence
        );
        assert_eq!(env_cadence("flux2_definitely_unset_xyz", None), None);
    }
}
