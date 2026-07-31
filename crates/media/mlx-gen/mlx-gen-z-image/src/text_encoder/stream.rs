//! Rung 4's **text-encoder** stream (SC-15794): materialize `model.layers.<i>` one window at a time
//! instead of holding all 36 resident.
//!
//! This is the same rung and the same driver the DiT stream uses
//! ([`mlx_gen::block_residency::run_windowed`]) applied to a different transformer — which is the
//! whole point of the component scope. Nothing here re-implements the window lifecycle.
//!
//! ## Why the encoder is worth streaming
//!
//! Rung 4 cut the denoise far enough that **conditioning became the binding phase** on the larger
//! tiers. Measured on z_image_turbo at 1024², Apple/Metal, real weights (SC-15794), against a decode
//! floor of ~4.365 GiB:
//!
//! | tier | encoder weights | conditioning, resident | one full window | window = 1 |
//! |---|---:|---:|---:|---:|
//! | bf16 | 7.440 GiB | 8.489 GiB (**bound the request**) | 2.718 GiB | 1.455 GiB |
//! | q4 | 2.132 GiB | 3.669 GiB | 2.398 GiB | 0.413 GiB |
//!
//! Weights, not activations, dominate the conditioning peak — the activation share is 13.7% on bf16
//! and 42% on q4 — so a window has real work to do.
//!
//! **Two separate effects, and it matters which is which.** Most of the bf16 drop is *streaming at
//! all*: even one full-width window drains the view per layer (see `materialize`) instead of holding
//! all 36 layers' tensors alive the way `TextEncoder::from_weights` does, which is 8.489 → 2.718 GiB
//! on its own. Tightening the window buys the rest. Before this change bf16 conditioning bound the
//! request above an 8 GB budget; it is now decode-bound.
//!
//! `text_encoder_window_real_weights` owns the ladder above and
//! `component_scope_real_weights` the end-to-end request-peak effect.
//!
//! ## It is a cheaper trade than the DiT stream
//!
//! The DiT re-materializes every block **once per denoise step**; the encoder runs **once per
//! generation**. Same mechanism, and the re-materialization is paid a single time — which is why the
//! encoder tolerates a tighter window than the DiT does.
//!
//! ## Z-Image is not where this matters most
//!
//! Rung 1 (staged residency) bounds the staged peak to `max(text_encoder, everything_else)`. For a
//! model whose encoder **is** the largest component, staging buys nothing — the encoder is already the
//! maximum — so an encoder window is the only lever that touches the binding component at all.
//!
//! Measured TE share of total on-disk weights across the locally cached catalog (SC-15794):
//!
//! | family | total | text encoder | share |
//! |---|---:|---:|---:|
//! | Mage-Flow components | 17.7 GiB | 16.7 GiB | **94.5%** |
//! | lens | 17.7 GiB | 13.4 GiB | **75.6%** |
//! | Kolors | 16.6 GiB | 11.6 GiB | **70.1%** |
//! | flux2-klein-9b | 47.6 GiB | 30.5 GiB | **64.1%** |
//! | Sana 1600M / Sana Sprint | 9.0 GiB | 4.9 GiB | **54.1%** |
//! | Mage-Flow (model repos) | 16.3 GiB | 8.3 GiB | **50.8%** |
//!
//! Z-Image is *not* in that list — its encoder is a minority of the model, and the win here came from
//! conditioning happening to bind the request after rung 4 cut the denoise. The families above are
//! where this rung should pay most, and each needs its own measurement before any of it is assumed:
//! share-of-disk predicts nothing about the *resident* peak (lens-turbo's mxfp4-on-disk gpt-oss
//! encoder expands to bf16 on load, which is where 17.24 GiB of its calibrated headroom goes).
//!
//! Caveat on the table: a multi-tier snapshot directory sums every tier, so these are whole-repo
//! ratios rather than per-tier ones. They rank the candidates; they are not the per-tier figure a
//! calibration record needs.

use std::ops::Range;

use mlx_rs::Array;

use mlx_gen::block_residency::BlockPlan;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result, WeightsSource};

use super::{EncoderLayer, ZTextEncoderConfig};

/// A re-openable source for the encoder's `model.layers.<i>` stack.
///
/// `source` must be re-openable — a snapshot directory or a single `.safetensors` file. An in-memory
/// / ComfyUI-normalized `Weights` has none, so those loads simply do not construct a stream and the
/// contract declares the text-encoder scope unavailable for them, exactly as the DiT stream does.
pub(crate) struct TextEncoderBlockStream {
    source: WeightsSource,
    /// Key prefix for the stack, so block `n` is `{base}.{n}` (`"model.layers"` on a real snapshot).
    base: String,
    cfg: ZTextEncoderConfig,
    /// The load-time quantization the resident encoder received, replayed per materialized block so a
    /// streamed layer is byte-identical to its resident twin. `None` on a dense/bf16 tier; on a
    /// pre-quantized (packed) snapshot this is recorded but is the same documented no-op the resident
    /// path performed.
    quant_bits: Option<i32>,
}

impl TextEncoderBlockStream {
    pub(crate) fn new(source: WeightsSource, prefix: &str, cfg: ZTextEncoderConfig) -> Self {
        Self {
            source,
            base: super::join(prefix, "layers"),
            cfg,
            quant_bits: None,
        }
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.cfg.n_layers
    }

    pub(crate) fn set_quant_bits(&mut self, bits: i32) {
        self.quant_bits = Some(bits);
    }

    /// Open a fresh lazy view of the encoder's weights. Called once per window by `run_windowed`.
    pub(crate) fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    /// Materialize encoder layer `index` out of `view`, quantized exactly like its resident twin, then
    /// drain the view of precisely the tensors that layer read.
    pub(crate) fn materialize(&self, view: &mut Weights, index: usize) -> Result<EncoderLayer> {
        if index >= self.cfg.n_layers {
            return Err(Error::Msg(format!(
                "z-image text-encoder stream: layer {index} is out of range for a {}-layer encoder",
                self.cfg.n_layers
            )));
        }
        let prefix = format!("{}.{index}", self.base);
        let mut layer = EncoderLayer::from_weights(
            view,
            &prefix,
            self.cfg.n_heads,
            self.cfg.n_kv_heads,
            self.cfg.head_dim,
            self.cfg.rms_norm_eps,
        )?;
        // LOAD-BEARING (SC-15750, re-established for this stack by SC-15794): the view keeps its own
        // refcounted handle to every tensor the constructor cloned. Draining exactly the accessed keys
        // is what makes the window's drop a real release rather than a no-op that still produces a
        // correct conditioning tensor. The DiT twin measured 8.0 MiB vs 238.4 MiB on this distinction.
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            layer.quantize(bits)?;
        }
        Ok(layer)
    }
}

/// The encoder's carried state across a window boundary.
///
/// The resident forward returns `all_hidden_states[-2]` — the **second-to-last** layer output — so it
/// tracks two arrays, not one. Both must cross the window boundary, and both must be materialized
/// before the window's weights are released: `second_to_last` is an unevaluated graph referencing an
/// earlier window's weights, so dropping the window would otherwise free nothing (the exact trap the
/// materialize hook exists for).
pub(crate) struct EncoderCarry {
    pub(crate) prev: Array,
    pub(crate) second_to_last: Array,
}

/// Run the encoder's layer stack in windows, materializing each window's weights and releasing them
/// before advancing. Returns `all_hidden_states[-2]`, identical to the resident loop.
pub(crate) fn run_windowed_layers(
    stream: &TextEncoderBlockStream,
    plan: &BlockPlan,
    cancel: &CancelFlag,
    embed: Array,
    cos: &Array,
    sin: &Array,
    mask: &Array,
) -> Result<Array> {
    if plan.n_blocks() != stream.n_blocks() {
        return Err(Error::Msg(format!(
            "z-image text-encoder stream: block plan covers {} layers but the encoder has {}",
            plan.n_blocks(),
            stream.n_blocks()
        )));
    }
    let init = EncoderCarry {
        second_to_last: embed.clone(),
        prev: embed,
    };
    let out = mlx_gen::block_residency::run_windowed(
        plan,
        cancel,
        init,
        || stream.open(),
        |carry: EncoderCarry, view: &mut Weights, range: Range<usize>| {
            let EncoderCarry {
                mut prev,
                mut second_to_last,
            } = carry;
            for i in range {
                let layer = stream.materialize(view, i)?;
                let cur = layer.forward(&prev, cos, sin, mask)?;
                // Identical bookkeeping to the resident loop: after this step the hidden-state list
                // would end [..., prev, cur], so the second-to-last is the old `prev`.
                second_to_last = std::mem::replace(&mut prev, cur);
            }
            Ok(EncoderCarry {
                prev,
                second_to_last,
            })
        },
        |carry: &EncoderCarry| {
            Ok(mlx_rs::transforms::eval([
                &carry.prev,
                &carry.second_to_last,
            ])?)
        },
    )?;
    Ok(out.second_to_last)
}
