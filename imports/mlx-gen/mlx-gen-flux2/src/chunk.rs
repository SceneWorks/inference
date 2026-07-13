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
//! 1. **Lazy-graph depth** ([`MemoryConfig::eval_per_block`]). The whole 8+48-block forward is one
//!    lazy graph evaluated once per denoise step, so without intervention the peak holds many blocks'
//!    transients at once; force-evaluating each block's output before the next caps it at ~one
//!    block's. **Bit-exact** — it only forces materialization, so the multi-reference edit's *pixels*
//!    are unchanged, only its memory schedule. This is the dominant lever and the production default
//!    for the long-sequence path ([`MemoryConfig::LONG_SEQ`]).
//! 2. **The FFN intermediate** ([`MemoryConfig::ffn_seq_chunk`]). The double block's image FFN
//!    materializes a `[L, 2·mlp_ratio·inner]` SwiGLU intermediate — the largest single transient;
//!    running it over sequence row-blocks bounds it. **Numerically equivalent, not bit-identical**
//!    (the FFN is per-token so the math is unchanged, but MLX's Metal GEMM is tile-specialized by the
//!    row dimension → cosine ≈ 1, max|Δ| ~1e-3, the model's own torch-parity class). Off by default;
//!    available as headroom for extreme configs (3+ references / higher resolution) and tunable from
//!    the environment without a recompile.
//!
//! Attention needs no query-chunking lever here: FLUX.2 attention is flash
//! `scaled_dot_product_attention` (`transformer.rs`), which never materializes the `[heads, L, L]`
//! score matrix, so `eval_per_block` already bounds the per-block attention transient (the SCAIL-2
//! `attn_query_chunk` lever is for the materialized-SDPA fallback and is OFF there too).

use mlx_gen::Result;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

/// Knobs that bound the per-step activation high-water of the FLUX.2 MMDiT denoise so a long-sequence
/// multi-reference edit fits under `minMemoryGb` (sc-6266). All configs produce the same image up to
/// the kernel-rounding class noted per field; [`OFF`](Self::OFF) is the historical whole-sequence,
/// single-eval-per-step behaviour and is byte-identical to today's shipped forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryConfig {
    /// Run each double block's **image** FFN (`[L, 2·mlp_ratio·inner]` SwiGLU intermediate) over
    /// sequence row-blocks of at most this many tokens. `None` ⇒ the whole sequence at once (the
    /// no-op fast path). Numerically equivalent, not bit-identical (see the module doc).
    pub ffn_seq_chunk: Option<usize>,
    /// Force-evaluate (and free) each transformer block's output before starting the next, so the
    /// step's peak is ~one block's activations instead of the whole-depth lazy graph. **Bit-exact.**
    pub eval_per_block: bool,
}

impl MemoryConfig {
    /// No activation control — whole-sequence FFN with one eval at the end of the step (today's
    /// shipped behaviour). Byte-identical to the pre-sc-6266 forward.
    pub const OFF: Self = Self {
        ffn_seq_chunk: None,
        eval_per_block: false,
    };

    /// Production default for the gated long-sequence (multi-reference edit) path: `eval_per_block`
    /// only. Bit-exact (identical pixels), and on its own brings the 2-reference 1024² edit well
    /// under `minMemoryGb = 96`. FFN chunking stays available as env-tunable headroom.
    pub const LONG_SEQ: Self = Self {
        ffn_seq_chunk: None,
        eval_per_block: true,
    };

    /// `true` if no lever is active (the [`OFF`](Self::OFF) fast path — skip the chunk plumbing).
    pub fn is_off(&self) -> bool {
        self.ffn_seq_chunk.is_none() && !self.eval_per_block
    }

    /// Overlay the environment onto `base` so a deployment can tune the memory/throughput tradeoff
    /// without a recompile:
    ///   * `MLX_GEN_FLUX2_FFN_SEQ_CHUNK` — FFN sequence chunk (`0` disables; unset keeps `base`).
    ///   * `MLX_GEN_FLUX2_EVAL_PER_BLOCK` — `1`/`true`/`on` or `0`/`false`/`off` (unset keeps `base`).
    pub fn from_env(base: Self) -> Self {
        Self {
            ffn_seq_chunk: env_chunk("MLX_GEN_FLUX2_FFN_SEQ_CHUNK", base.ffn_seq_chunk),
            eval_per_block: env_bool("MLX_GEN_FLUX2_EVAL_PER_BLOCK", base.eval_per_block),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self::OFF
    }
}

/// A `usize` chunk knob from `var`: a positive integer enables, `0` disables (`None`), anything else
/// (unset / unparseable) keeps `base`.
fn env_chunk(var: &str, base: Option<usize>) -> Option<usize> {
    match std::env::var(var) {
        Ok(s) => match s.trim().parse::<usize>() {
            Ok(0) => None,
            Ok(n) => Some(n),
            Err(_) => base,
        },
        Err(_) => base,
    }
}

/// A boolean knob from `var` (`1`/`true`/`on` vs `0`/`false`/`off`, case-insensitive); unset /
/// unrecognized keeps `base`.
fn env_bool(var: &str, base: bool) -> bool {
    match std::env::var(var) {
        Ok(s) => match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => true,
            "0" | "false" | "off" | "no" => false,
            _ => base,
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

/// Contiguous `[:, start:start+len, …]` slice along the sequence axis (axis 1), via the arange-index
/// `take_axis` idiom used throughout the crate.
fn slice_seq(x: &Array, start: i32, len: i32) -> Result<Array> {
    let idx = Array::from_slice(&(start..start + len).collect::<Vec<i32>>(), &[len]);
    Ok(x.take_axis(&idx, 1)?)
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
                eval_per_block: true,
            }
        );
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

    #[test]
    fn env_helpers_parse() {
        assert_eq!(env_chunk("flux2_definitely_unset_xyz", Some(99)), Some(99));
        assert!(env_bool("flux2_definitely_unset_xyz", true));
        assert!(!env_bool("flux2_definitely_unset_xyz", false));
    }
}
