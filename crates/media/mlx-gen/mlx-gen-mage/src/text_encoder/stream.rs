//! Rung-4 residency for Mage-Flow's Qwen3-VL language stack (SC-15800).
//!
//! The schedule is the shared [`mlx_gen::block_residency::run_windowed`] driver. This module owns
//! only Mage's re-openable layer loader and carried hidden state. Each materialized layer drains
//! the exact source tensors it touched, and the carried activation is evaluated before the window
//! drops; both operations are required for the residency bound to be real under MLX's lazy graph.

use std::ops::Range;

use mlx_rs::Array;

use mlx_gen::block_residency::BlockPlan;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result, WeightsSource};

use crate::config::QwenVlTextConfig;

use super::{join, Qwen3VlDecoderLayer};

/// Re-openable source for `model.language_model.layers.<i>`.
pub(crate) struct TextEncoderBlockStream {
    source: WeightsSource,
    base: String,
    cfg: QwenVlTextConfig,
    eps: f32,
    quant_bits: Option<i32>,
    materialize_carry: bool,
}

impl TextEncoderBlockStream {
    pub(crate) fn new(
        source: WeightsSource,
        prefix: &str,
        cfg: QwenVlTextConfig,
        eps: f32,
    ) -> Self {
        Self {
            source,
            base: join(prefix, "layers"),
            cfg,
            eps,
            quant_bits: None,
            materialize_carry: true,
        }
    }

    pub(crate) fn without_carry_materialization(mut self) -> Self {
        self.materialize_carry = false;
        self
    }

    pub(crate) fn materializes_carry(&self) -> bool {
        self.materialize_carry
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.cfg.num_layers
    }

    pub(crate) fn set_quant_bits(&mut self, bits: i32) {
        self.quant_bits = Some(bits);
    }

    pub(crate) fn has_quant_bits(&self) -> bool {
        self.quant_bits.is_some()
    }

    fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    fn materialize(&self, view: &mut Weights, index: usize) -> Result<Qwen3VlDecoderLayer> {
        if index >= self.cfg.num_layers {
            return Err(Error::Msg(format!(
                "mage_flow text-encoder stream: layer {index} is out of range for a {}-layer encoder",
                self.cfg.num_layers
            )));
        }
        let mut layer = Qwen3VlDecoderLayer::from_weights(
            view,
            &format!("{}.{index}", self.base),
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
            self.eps,
        )?;
        // Load-bearing: constructors clone refcounted handles from the view. Removing only the keys
        // read by this layer makes dropping the window release them instead of retaining the whole
        // source map while producing perfectly correct output.
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            let layer_bits = crate::quant::floor_bits(crate::quant::LM_LAYER_PREFIX, bits);
            layer.quantize(layer_bits)?;
        }
        Ok(layer)
    }
}

pub(crate) fn run_windowed_layers(
    stream: &TextEncoderBlockStream,
    plan: &BlockPlan,
    cancel: &CancelFlag,
    mut hidden: Array,
    cos: &Array,
    sin: &Array,
    mask: &Array,
) -> Result<Array> {
    if plan.n_blocks() != stream.n_blocks() {
        return Err(Error::Msg(format!(
            "mage_flow text-encoder stream: block plan covers {} layers but the encoder has {}",
            plan.n_blocks(),
            stream.n_blocks()
        )));
    }
    hidden = mlx_gen::block_residency::run_windowed(
        plan,
        cancel,
        hidden,
        || stream.open(),
        |mut carry: Array, view: &mut Weights, range: Range<usize>| {
            for index in range {
                carry = stream
                    .materialize(view, index)?
                    .forward(&carry, cos, sin, mask)?;
            }
            Ok(carry)
        },
        |carry: &Array| {
            if stream.materializes_carry() {
                mlx_rs::transforms::eval([carry])?;
            }
            Ok(())
        },
    )?;
    Ok(hidden)
}
