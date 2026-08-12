//! **The MiniMax-H3 published-checkpoint layout contract**, candle side.
//!
//! This is the candle twin of [`mlx_gen_minimax_h3::layout`]; the rules are properties of the
//! *checkpoint*, not of a backend, so they are restated here rather than re-derived. Read it before
//! porting any further MiniMax-H3 component to candle (sc-17155 DiT).
//!
//! MiniMax publishes `MiniMaxAI/MiniMax-H3` in the **diffusers** layout, produced from the original
//! release by `scripts/convert_minimax_h3_to_diffusers.py` on `huggingface/diffusers`. That script
//! is not a pure rename: it applies **tensor transforms** whose results are *shape-identical* to
//! their inputs. Nothing structural — not a shape check, not an exhaustive key-mapping proof, not a
//! checksum — can tell the two apart. Only semantics can, and only if something asserts them.
//!
//! sc-18740 is what happens when nothing does: the MLX video VAE decoder shipped reading the wrong
//! half of its gated FFN projection for 36 blocks, at relative max-abs-diff 0.86–0.99 per block,
//! while every gate in that crate stayed green.
//!
//! # Rule 1 — a gated FFN's fused projection is `[value | gate]`, not `[gate | value]`
//!
//! The original MiniMax modules store the fused SwiGLU projection **gate-first** and compute
//! `w2( silu(gate) · value )` (`FL2VA/video_vae/base_module.py`: `gate, hidden = h.chunk(2, -1)`).
//! diffusers' `SwiGLU` (`diffusers/models/activations.py`) reads the same tensor **value-first** —
//! `hidden, gate = proj(x).chunk(2, -1); return hidden * silu(gate)` — so the conversion physically
//! swaps the two row halves:
//!
//! ```text
//! # convert_minimax_h3_to_diffusers.py, video VAE — `ff.w1` -> `ff.net.0.proj`
//! gate, up = tensor.chunk(2, dim=0)
//! return [(target_key, torch.cat([up, gate], dim=0).contiguous())]
//!
//! # ...and identically for the DiT — `mlp.fc1` -> `ff.net.0.proj`
//! gate, value = tensor.chunk(2, dim=0)
//! return [(target_key, torch.cat([value, gate], dim=0).contiguous())]
//! ```
//!
//! **Therefore: a port that loads the published checkpoint must read the FIRST half as the VALUE
//! and the SECOND half as the GATE.** That is [`PUBLISHED_GATED_FFN_LAYOUT`], and
//! [`split_gate_value`] is the single implementation every component in this crate calls.
//!
//! This rule applies to the video VAE decoder (`decoder.transformer_blocks.N.ff.net.0.proj`) and
//! will apply to the **DiT** (`transformer_blocks.N.ff.net.0.proj`) when sc-17155 lands it. It does
//! **not** apply to the text encoder, whose Qwen3 MLP ships `gate_proj` and `up_proj` as two
//! separate tensors. It does **not** apply to the audio VAE, which the conversion carries over
//! unchanged (see [`AUDIO_VAE_IS_UNCONVERTED`]).
//!
//! Independent third-party corroboration: lightx2v's `Minimax-h3-Turbo` ComfyUI LoRA variants carry
//! `"swi_glu_mapping": "Diffusers [value;gate] -> ComfyUI [gate;value]"` in their metadata.
//!
//! ## This layout is not permanent — pin it, do not infer it
//!
//! diffusers issue **#14410** ("MiniMax H3 unnecessarily swapped w1 in VAE") is **open**. If
//! upstream reverts the swap, the published tensors change while every shape stays identical and
//! every loader keeps loading cleanly. A port must therefore *assert* which layout it expects
//! against real bytes. `tests/video_vae_parity.rs`'s `published_ffn_projection_is_value_then_gate`
//! is that assertion here, and `reading_the_ffn_halves_gate_first_breaks_parity` proves the
//! assertion is not vacuous by performing the mutation and measuring how far it moves the decode.
//!
//! # Rule 2 — fused QKV: two different transforms that must not be confused
//!
//! Both the video VAE decoder and the DiT publish `to_q` / `to_k` / `to_v` split out of a fused
//! source projection, but the *source* layouts differ, so the split rules differ:
//!
//! | component | raw checkpoint rows | reference in-memory | published `to_q` is |
//! |---|---|---|---|
//! | video VAE decoder | per-head interleaved `[h0: q k v, h1: q k v, …]` | same (no load-time reorder; `attention.py` reads `qkv.view(B, S, -1, 3·dim_head)`) | the per-head `q` slabs, concatenated |
//! | DiT | per-head interleaved | `[q_all; k_all; v_all]` (the reference reorders at load, `_reorder_grouped_qkv_to_qkv`) | the same per-head `q` slabs |
//!
//! **This crate wants the VAE form**, which is what [`crate::vae::split_fused_qkv`] implements and
//! what `fused_qkv_split_reproduces_the_published_split` pins against the fixture's own
//! pre-conversion `src.` tensors. A naive `chunk(3, dim=0)` of the raw fused tensor is a
//! **different partition** of the same rows and is silently wrong — the two are shape-identical.
//!
//! # Rule 3 — a fixture generated from reference modules cannot validate a converted-checkpoint
//! loader
//!
//! `tools/dump_minimax_h3_video_vae.py` originally ran the **original MiniMax modules** and emitted
//! their parameters under the *published* key names via a pure rename table. The fixture therefore
//! carried the **source** layout under **published** names; the loader read it gate-first, the
//! fixture supplied gate-first, parity passed at 1e-3 — and production, which loads the genuinely
//! converted `vae/`, was wrong by a half-swap.
//!
//! > **A golden and a loader that share a layout prove only that they agree with each other.**
//!
//! The generator now builds the golden from `AutoencoderKLMiniMaxH3` — the converted layout
//! production actually reads — and records the path that produced it in the fixture's safetensors
//! `__metadata__` (`provenance`, `reference`, `reference_version`).
//! `fixture_provenance_records_the_converted_path` asserts that metadata says "converted", so a
//! regeneration that silently reverts to the reference-module path fails rather than passing.
//! Note that `candle_core::safetensors` **drops** `__metadata__`, which is why this crate's tests
//! parse the container header by hand (`tests/common/mod.rs`'s `Golden`) rather than loading the
//! fixture through candle.
//!
//! # Rule 4 — the conditioner applies no chat template
//!
//! Not a tensor-layout rule and not reachable from this crate yet (the text encoder is sc-17155's
//! neighbourhood), but the same class of silent divergence: every MiniMax-H3 presentation
//! (`t2va`, `fl2va`, `ref2va`) tokenizes with `tokenizer(text, add_special_tokens=False)` and **no
//! chat template** (sc-18741).
//!
//! [`mlx_gen_minimax_h3::layout`]: https://github.com/SceneWorks/inference/blob/main/crates/media/mlx-gen/mlx-gen-minimax-h3/src/layout.rs

use candle_gen::candle_core::Tensor;
use candle_gen::{CandleError, Result};

/// Which half of a fused SwiGLU projection carries the gate.
///
/// The two variants are shape-identical and therefore silently interchangeable at load, which is
/// exactly why this is a named type rather than a comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedFfnLayout {
    /// `[gate | value]` — the **original MiniMax** modules' layout. `w2( silu(gate) · value )`
    /// reading the first half as the gate.
    GateFirst,
    /// `[value | gate]` — the **published / diffusers** layout, what
    /// `convert_minimax_h3_to_diffusers.py` writes and what this crate loads.
    ValueFirst,
}

/// The layout of every fused SwiGLU projection in the **published** `MiniMaxAI/MiniMax-H3`
/// checkpoint: `ff.net.0.proj` for the video VAE decoder and for the DiT alike.
///
/// Pinned as a constant so that flipping it is a deliberate, reviewable, one-line change with a
/// failing test attached — not an invisible property of a slice index. See the module docs for why
/// diffusers issue #14410 makes that a live possibility.
pub const PUBLISHED_GATED_FFN_LAYOUT: GatedFfnLayout = GatedFfnLayout::ValueFirst;

/// The audio VAE's tensors are carried through the conversion **unchanged** — no renames, no
/// swaps, no reorders (`convert_audio_vae` reports "`N` keys carried over unchanged"), verified
/// over all 914 decode tensors.
///
/// Recorded here so the audit result is a checked constant rather than a claim in a comment.
/// `tests/audio_vae_parity.rs` asserts the audio decoder has no fused gated projection to mis-read.
pub const AUDIO_VAE_IS_UNCONVERTED: bool = true;

/// Split a fused SwiGLU projection's output into `(gate, value)` under
/// [`PUBLISHED_GATED_FFN_LAYOUT`].
///
/// `h` is the output of `ff.net.0.proj`, `[.., 2·inner]`; the split is over its LAST axis. Returns
/// `(gate, value)` such that the block computes `w2( silu(gate) · value )`.
///
/// **The published layout is `[value | gate]`**, so the gate is the SECOND half. Reading the first
/// half as the gate — the original modules' convention — computes `w2( silu(value) · gate )` and is
/// the sc-18740 defect: measured relative max-abs-diff 0.86–0.99 per block on real weights, while
/// output norms barely move (89 vs 85), which is why no magnitude or checksum gate can see it.
pub fn split_gate_value(h: &Tensor) -> Result<(Tensor, Tensor)> {
    let rank = h.dims().len();
    if rank == 0 {
        return Err(CandleError::Msg(
            "minimax-h3 ffn: a gated projection cannot be rank 0".into(),
        ));
    }
    let axis = rank - 1;
    let inner = h.dims()[axis];
    if !inner.is_multiple_of(2) {
        return Err(CandleError::Msg(format!(
            "minimax-h3 ffn: a gated projection must emit an even width, got {inner}"
        )));
    }
    let half = inner / 2;
    let first = h.narrow(axis, 0, half)?;
    let second = h.narrow(axis, half, half)?;
    Ok(match PUBLISHED_GATED_FFN_LAYOUT {
        GatedFfnLayout::ValueFirst => (second, first),
        GatedFfnLayout::GateFirst => (first, second),
    })
}

/// Swap the two row halves of a `[2·inner, ..]` fused gated projection — the inverse of the
/// official conversion, i.e. what turns a published `[value | gate]` tensor back into the original
/// modules' `[gate | value]`.
///
/// Shipped (not test-only) because it is the mutation `tests/video_vae_parity.rs` performs to prove
/// the layout assertion is not vacuous, and because a future re-hosting slice (sc-17150) has to
/// name this transform to pin which layout it publishes.
pub fn swap_gated_halves(t: &Tensor) -> Result<Tensor> {
    let rows = t.dims()[0];
    if !rows.is_multiple_of(2) {
        return Err(CandleError::Msg(format!(
            "minimax-h3 ffn: a gated projection must have an even leading dim, got {rows}"
        )));
    }
    let half = rows / 2;
    let lo = t.narrow(0, 0, half)?;
    let hi = t.narrow(0, half, half)?;
    Ok(Tensor::cat(&[&hi, &lo], 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// The contract itself. If this ever needs changing, diffusers #14410 landed and every
    /// MiniMax-H3 fixture in the repo must be regenerated in the same commit.
    #[test]
    fn published_checkpoint_is_value_first() {
        assert_eq!(PUBLISHED_GATED_FFN_LAYOUT, GatedFfnLayout::ValueFirst);
    }

    /// `split_gate_value` must take the SECOND half as the gate. Written against a tensor whose two
    /// halves are trivially distinguishable, so a reversed implementation cannot pass.
    #[test]
    fn gate_is_the_second_half_of_the_published_projection() {
        // `[value=1, value=2 | gate=10, gate=20]` — the published `[value | gate]` order.
        let h = Tensor::from_vec(vec![1.0f32, 2.0, 10.0, 20.0], (1, 4), &Device::Cpu).unwrap();
        let (gate, value) = split_gate_value(&h).unwrap();
        assert_eq!(
            gate.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![10.0, 20.0],
            "gate is the 2nd half"
        );
        assert_eq!(
            value.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0],
            "value is the 1st half"
        );
    }

    /// An odd width is a typed error, not a silently truncated split.
    #[test]
    fn odd_width_is_rejected() {
        let h = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &Device::Cpu).unwrap();
        let e = split_gate_value(&h).unwrap_err().to_string();
        assert!(e.contains("even width"), "unexpected error: {e}");
    }

    /// The half-swap is an involution and really moves the rows.
    #[test]
    fn swap_gated_halves_is_an_involution() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4, 1), &Device::Cpu).unwrap();
        let swapped = swap_gated_halves(&t).unwrap();
        assert_eq!(
            swapped.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![3.0, 4.0, 1.0, 2.0]
        );
        let back = swap_gated_halves(&swapped).unwrap();
        assert_eq!(
            back.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        let odd = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3, 1), &Device::Cpu).unwrap();
        assert!(swap_gated_halves(&odd).is_err());
    }

    /// The audio VAE audit result, pinned so it is not re-litigated per backend.
    #[test]
    fn audio_vae_needs_no_conversion_transform() {
        const { assert!(AUDIO_VAE_IS_UNCONVERTED) };
    }
}
