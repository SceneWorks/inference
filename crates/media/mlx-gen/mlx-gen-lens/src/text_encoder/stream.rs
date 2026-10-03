//! Lens rung-4 text-encoder streaming (SC-15800): the gpt-oss decoder stack is materialized from a
//! re-openable snapshot in bounded windows through [`mlx_gen::block_residency::run_windowed`]. The
//! shared driver owns window ordering, synchronization-before-release, cancellation, and cache
//! eviction; this module only describes how to rebuild and apply one Lens decoder layer.

use std::ops::Range;

use mlx_rs::{Array, Dtype};

use mlx_gen::block_residency::BlockPlan;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Quant, Result, WeightsSource};

use crate::config::GptOssConfig;
use crate::text_encoder::gpt_oss::GptOssDecoderLayer;

/// Re-openable source plus every constructor input needed to replay a resident layer exactly.
pub(super) struct TextEncoderBlockStream {
    source: WeightsSource,
    cfg: GptOssConfig,
    dtype: Dtype,
    selected_layers: Vec<usize>,
    quant: Option<Quant>,
}

impl TextEncoderBlockStream {
    pub(super) fn new(
        source: WeightsSource,
        cfg: GptOssConfig,
        dtype: Dtype,
        selected_layers: Vec<usize>,
        quant: Option<Quant>,
    ) -> Self {
        Self {
            source,
            cfg,
            dtype,
            selected_layers,
            quant,
        }
    }

    pub(super) fn n_blocks(&self) -> usize {
        self.selected_layers.iter().copied().max().unwrap_or(0) + 1
    }

    fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    fn materialize(&self, view: &mut Weights, index: usize) -> Result<GptOssDecoderLayer> {
        if index >= self.n_blocks() {
            return Err(Error::Msg(format!(
                "lens text-encoder stream: layer {index} is out of range for a {}-layer prefix",
                self.n_blocks()
            )));
        }
        let layer = GptOssDecoderLayer::from_weights(
            view,
            &format!("model.layers.{index}"),
            &self.cfg,
            self.dtype,
            self.quant,
        )?;
        // LOAD-BEARING: constructors clone refcounted Array handles. Draining precisely the keys read
        // by this layer removes the view's second handle; without it a dropped window remains resident
        // while producing the same output, so correctness tests alone cannot detect the regression.
        view.remove_accessed();
        Ok(layer)
    }
}

struct EncoderCarry {
    hidden: Array,
    captured: Vec<Option<Array>>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_windowed_layers(
    stream: &TextEncoderBlockStream,
    plan: &BlockPlan,
    cancel: &CancelFlag,
    hidden: Array,
    inv_freq: &Array,
    attn_scaling: f32,
    full_mask: &Array,
    sliding_mask: &Array,
) -> Result<Vec<Array>> {
    if plan.n_blocks() != stream.n_blocks() {
        return Err(Error::Msg(format!(
            "lens text-encoder stream: block plan covers {} layers but the selected prefix has {}",
            plan.n_blocks(),
            stream.n_blocks()
        )));
    }
    let out = mlx_gen::block_residency::run_windowed(
        plan,
        cancel,
        EncoderCarry {
            hidden,
            captured: vec![None; stream.selected_layers.len()],
        },
        || stream.open(),
        |mut carry: EncoderCarry, view: &mut Weights, range: Range<usize>| {
            for index in range {
                // Preserve Lens's original per-layer cancellation granularity even for windows > 1.
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let layer = stream.materialize(view, index)?;
                let mask = if stream.cfg.is_sliding(index) {
                    sliding_mask
                } else {
                    full_mask
                };
                carry.hidden = layer.forward(&carry.hidden, inv_freq, attn_scaling, mask)?;
                if let Some(slot) = stream.selected_layers.iter().position(|&s| s == index) {
                    carry.captured[slot] = Some(carry.hidden.clone());
                }
            }
            Ok(carry)
        },
        |carry: &EncoderCarry| {
            // Every captured state is part of the output, not only the current hidden state. Force all
            // filled slots before releasing the active window or their lazy graphs retain its weights.
            let mut arrays = Vec::with_capacity(1 + carry.captured.len());
            arrays.push(&carry.hidden);
            arrays.extend(carry.captured.iter().filter_map(Option::as_ref));
            Ok(mlx_rs::transforms::eval(arrays)?)
        },
    )?;
    out.captured
        .into_iter()
        .enumerate()
        .map(|(slot, value)| {
            value.ok_or_else(|| {
                Error::Msg(format!(
                    "lens text-encoder stream: selected capture slot {slot} was not produced"
                ))
            })
        })
        .collect()
}
