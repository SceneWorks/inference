//! **The MiniMax-H3 published-checkpoint layout contract.** Read this before porting any further
//! MiniMax-H3 component (sc-17144 DiT, sc-17154 / sc-17155 candle).
//!
//! MiniMax publishes `MiniMaxAI/MiniMax-H3` in the **diffusers** layout, produced from the original
//! release by `scripts/convert_minimax_h3_to_diffusers.py` on `huggingface/diffusers`. That script
//! is not a pure rename: it applies **tensor transforms** whose results are *shape-identical* to
//! their inputs. Nothing structural — not a shape check, not an exhaustive key-mapping proof, not a
//! checksum — can tell the two apart. Only semantics can, and only if something asserts them.
//!
//! sc-18740 is what happens when nothing does: the video VAE decoder shipped reading the wrong half
//! of its gated FFN projection for 36 blocks, at relative max-abs-diff 0.86–0.99 per block, while
//! every gate in the crate stayed green.
//!
//! # Rule 1 — a gated FFN's fused projection is `[value | gate]`, not `[gate | value]`
//!
//! The original MiniMax modules store the fused SwiGLU projection **gate-first** and compute
//! `w2( silu(gate) · value )` (`FL2VA/video_vae/base_module.py`: `gate, hidden = h.chunk(2, -1)`).
//! diffusers' `SwiGLU` (`diffusers/models/activations.py`) reads the same tensor **value-first** — `hidden, gate = proj(x).chunk(2,
//! -1); return hidden * silu(gate)` — so the conversion physically swaps the two row halves:
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
//! This rule applies to the video VAE decoder (`decoder.transformer_blocks.N.ff.net.0.proj`) and to
//! the **DiT** (`transformer_blocks.N.ff.net.0.proj`) alike. It does **not** apply to the text
//! encoder, whose Qwen3 MLP ships `gate_proj` and `up_proj` as two separate tensors, so there is no
//! fused half to mis-read. It does **not** apply to the audio VAE, which the conversion carries
//! over unchanged (see [`AUDIO_VAE_IS_UNCONVERTED`]).
//!
//! Independent third-party corroboration: lightx2v's `Minimax-h3-Turbo` ComfyUI LoRA variants carry
//! `"swi_glu_mapping": "Diffusers [value;gate] -> ComfyUI [gate;value]"` in their metadata.
//!
//! ## This layout is not permanent — pin it, do not infer it
//!
//! diffusers issue **#14410** ("MiniMax H3 unnecessarily swapped w1 in VAE") is **open**. If
//! upstream reverts the swap, the published tensors change while every shape stays identical and
//! every loader keeps loading cleanly. A port must therefore *assert* which layout it expects
//! against real bytes, not rely on the layout being obvious. `tests/video_vae_parity.rs`'s
//! `published_ffn_projection_is_value_then_gate` is that assertion for the video VAE, and
//! `tests/dit_parity.rs` carries the DiT's own (sc-17144) — over the 50-layer stack **and** the
//! token refiner, both of which the conversion swaps.
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
//! The conversion expresses both as `reorder_interleaved_qkv` followed by a contiguous-thirds
//! `split_fused_qkv`. Composed, that is exactly "gather row `h·(3·D) + j·D + d` into projection
//! `j`" — which is what [`crate::vae::split_fused_qkv`] implements and what
//! `fused_qkv_split_reproduces_the_published_split` pins. A naive `chunk(3, dim=0)` of the raw
//! fused tensor is a **different partition** and is silently wrong.
//!
//! For the DiT specifically: because its reference already holds `[q_all; k_all; v_all]` in memory,
//! a fixture dumped from the reference's `state_dict()` needs contiguous thirds, whereas a fixture
//! dumped from the raw shards needs the interleaved gather. Getting the wrong one produces a
//! plausible model that is wrong on real weights — the sc-18740 failure mode again.
//!
//! [`crate::dit::qkv`] makes both transforms executable, and
//! `tests/dit_parity.rs::published_qkv_is_contiguous_thirds_of_the_reordered_fused_projection`
//! asserts against real reference bytes that **crossing them is shape-identical and wrong** — which
//! is what turns "we picked a transform" into "we picked the right one". Note the published DiT
//! ships `to_q`/`to_k`/`to_v` already split (`to_qkv` appears nowhere in its index), so the loader
//! applies no transform at all; that is precisely why only an explicit assertion can pin it.
//!
//! # Rule 3 — a fixture generated from reference modules cannot validate a converted-checkpoint
//! loader
//!
//! This is the methodology half of sc-18740, and it is the reason the defect shipped green.
//!
//! `tools/dump_minimax_h3_video_vae.py` originally ran the **original MiniMax modules** and emitted
//! their parameters under the *published* key names via a pure rename table (`"ff.w1":
//! "ff.net.0.proj"`). The fixture therefore carried the **source** layout under **published**
//! names. The loader read it gate-first, the fixture supplied gate-first, parity passed at 1e-3 —
//! and production, which loads the genuinely converted `vae/`, was wrong by a half-swap.
//!
//! > **A golden and a loader that share a layout prove only that they agree with each other.**
//!
//! The generators in `tools/` now build the golden from `AutoencoderKLMiniMaxH3` — the converted
//! layout production actually reads — and every regenerated fixture records the path that produced
//! it in its safetensors metadata (`provenance`, `reference`, `reference_version`). A test asserts
//! that metadata is present and says "converted", so a regeneration that silently reverts to the
//! reference-module path fails rather than passing.
//!
//! # Rule 4 — the conditioner applies no chat template
//!
//! Not a tensor-layout rule, but the same class of silent divergence, and sc-18741 is its instance.
//! Every MiniMax-H3 presentation (`t2va`, `fl2va`, `ref2va`) tokenizes with
//! `tokenizer(text, add_special_tokens=False)` and **no chat template**. See
//! [`crate::text_encoder::tokenizer`].

use mlx_rs::Array;

use mlx_gen::Result;

use crate::tensor::slice_axis;

/// Which half of a fused SwiGLU projection carries the gate.
///
/// The two variants are shape-identical and therefore silently interchangeable at load, which is
/// exactly why this is a named type rather than a comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatedFfnLayout {
    /// `[gate | value]` — the **original MiniMax** modules' layout. `w2( silu(gate) · value )`
    /// reading the first half as the gate.
    GateFirst,
    /// `[value | gate]` — the **published / diffusers** layout, what `convert_minimax_h3_to_diffusers.py`
    /// writes and what this crate loads.
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
/// swaps, no reorders (`convert_audio_vae` reports "`N` keys carried over unchanged").
///
/// Recorded here so the audit result is a checked constant rather than a claim in a comment, and so
/// the candle port (sc-17155) does not re-derive it. `tests/audio_vae_parity.rs` asserts the audio
/// decoder has no fused gated projection to mis-read.
pub const AUDIO_VAE_IS_UNCONVERTED: bool = true;

/// Split a fused SwiGLU projection's output into `(gate, value)` under
/// [`PUBLISHED_GATED_FFN_LAYOUT`].
///
/// `h` is the output of `ff.net.0.proj`, `[.., 2·inner]`; `axis` is its last axis. Returns
/// `(gate, value)` such that the block computes `w2( silu(gate) · value )`.
///
/// **The published layout is `[value | gate]`**, so the gate is the SECOND half. Reading the first
/// half as the gate — the original modules' convention — computes `w2( silu(value) · gate )` and is
/// the sc-18740 defect: measured relative max-abs-diff 0.86–0.99 per block on real weights, while
/// output norms barely move (89 vs 85), which is why no magnitude or checksum gate can see it.
pub fn split_gate_value(h: &Array, axis: i32) -> Result<(Array, Array)> {
    let rank = h.shape().len();
    let inner = h.shape()[rank - 1];
    if inner % 2 != 0 {
        return Err(mlx_gen::Error::Msg(format!(
            "minimax-h3 ffn: a gated projection must emit an even width, got {inner}"
        )));
    }
    let half = inner / 2;
    let first = slice_axis(h, axis, 0, half)?;
    let second = slice_axis(h, axis, half, inner)?;
    Ok(match PUBLISHED_GATED_FFN_LAYOUT {
        GatedFfnLayout::ValueFirst => (second, first),
        GatedFfnLayout::GateFirst => (first, second),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let h = Array::from_slice(&[1.0f32, 2.0, 10.0, 20.0], &[1, 4]);
        let (gate, value) = split_gate_value(&h, 1).unwrap();
        assert_eq!(
            gate.as_slice::<f32>(),
            &[10.0, 20.0],
            "gate is the 2nd half"
        );
        assert_eq!(
            value.as_slice::<f32>(),
            &[1.0, 2.0],
            "value is the 1st half"
        );
    }

    /// An odd width is a typed error, not a silently truncated split.
    #[test]
    fn odd_width_is_rejected() {
        let h = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]);
        let e = split_gate_value(&h, 1).unwrap_err().to_string();
        assert!(e.contains("even width"), "unexpected error: {e}");
    }

    /// The audio VAE audit result, pinned so it is not re-litigated per backend.
    #[test]
    fn audio_vae_needs_no_conversion_transform() {
        const { assert!(AUDIO_VAE_IS_UNCONVERTED) };
    }
}
