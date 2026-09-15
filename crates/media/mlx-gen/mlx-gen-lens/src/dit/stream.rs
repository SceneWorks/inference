//! Request-scoped Lens DiT block materialization.

use std::ops::Range;

use mlx_gen::attention::AttentionPlan;
use mlx_gen::block_residency::BlockPlan;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Quant, Result, WeightsSource};
use mlx_rs::{Array, Dtype};

use super::{LensDitConfig, LensTransformerBlock};

pub(super) struct DitBlockStream {
    source: WeightsSource,
    cfg: LensDitConfig,
    dtype: Dtype,
    quant: Option<Quant>,
}

impl DitBlockStream {
    pub(super) fn new(
        source: WeightsSource,
        cfg: LensDitConfig,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Self {
        Self {
            source,
            cfg,
            dtype,
            quant,
        }
    }

    fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    fn materialize(&self, view: &mut Weights, index: usize) -> Result<LensTransformerBlock> {
        if index >= self.cfg.num_layers {
            return Err(Error::Msg(format!(
                "lens DiT stream: block {index} is outside the {}-block stack",
                self.cfg.num_layers
            )));
        }
        let mut block = LensTransformerBlock::from_weights(
            view,
            &format!("transformer_blocks.{index}"),
            self.cfg.num_heads,
            self.cfg.head_dim,
            self.dtype,
        )?;
        if let Some(quant) = self.quant {
            block.quantize(quant.bits())?;
        }
        // LOAD-BEARING: constructors clone Array handles. Drain exactly what this block consumed so
        // dropping the window returns the active trunk weights instead of retaining a second handle.
        view.remove_accessed();
        Ok(block)
    }
}

struct DitCarry {
    enc: Array,
    hidden: Array,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_windowed_blocks(
    stream: &DitBlockStream,
    window: usize,
    cancel: &CancelFlag,
    enc: Array,
    hidden: Array,
    temb: &Array,
    img_cos: &Array,
    img_sin: &Array,
    txt_cos: &Array,
    txt_sin: &Array,
    mask: Option<&Array>,
    attention: AttentionPlan<'_>,
) -> Result<(Array, Array)> {
    let plan = BlockPlan::new(stream.cfg.num_layers, window)?;
    let out = mlx_gen::block_residency::run_windowed(
        &plan,
        cancel,
        DitCarry { enc, hidden },
        || stream.open(),
        |mut carry: DitCarry, view: &mut Weights, range: Range<usize>| {
            for index in range {
                if cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let block = stream.materialize(view, index)?;
                let (enc, hidden) = block.forward_with_attention(
                    &carry.hidden,
                    &carry.enc,
                    temb,
                    img_cos,
                    img_sin,
                    txt_cos,
                    txt_sin,
                    mask,
                    attention,
                )?;
                carry.enc = enc;
                carry.hidden = hidden;
            }
            Ok(carry)
        },
        |carry: &DitCarry| Ok(mlx_rs::transforms::eval([&carry.enc, &carry.hidden])?),
    )?;
    Ok((out.enc, out.hidden))
}
