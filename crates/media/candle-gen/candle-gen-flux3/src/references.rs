//! Frozen `Pipeline.encode_references`: independent AE image_cond rows, never noisy-state init.
use crate::autoencoder::Flux3AutoEncoder;
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{
    flux3::references::{conditioning_ids, PreparedReferences},
    CancelFlag,
};
use candle_gen::{CandleError as Error, Result};

pub struct ReferenceConditioning {
    pub tokens: Tensor,
    pub ids: Vec<[f32; 4]>,
}

/// All references were host-validated/prepared before heavy components are entered.
pub fn encode(
    ae: &Flux3AutoEncoder,
    prepared: &PreparedReferences,
    device: &Device,
    cancel: &CancelFlag,
) -> Result<ReferenceConditioning> {
    let mut tokens = Vec::with_capacity(prepared.images.len());
    let mut ids = Vec::new();
    for (index, image) in prepared.images.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // upstream to_tensor: RGB HWC u8 -> CHW float /127.5-1, AE remains f32.
        let values: Vec<f32> = image
            .pixels
            .iter()
            .map(|&v| v as f32 / 127.5 - 1.0)
            .collect();
        let input = Tensor::from_vec(
            values,
            (image.height as usize, image.width as usize, 3),
            &Device::Cpu,
        )?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .contiguous()?
        .to_device(device)?;
        let (t, (h, w)) = ae.encode(&input)?;
        tokens.push(t.to_dtype(DType::BF16)?);
        ids.extend(conditioning_ids(h, w, index)?);
    }
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    if tokens.is_empty() {
        return Err(Error::Unsupported(
            "flux3: reference encoder requires 1–10 prepared references".into(),
        ));
    }
    Ok(ReferenceConditioning {
        tokens: Tensor::cat(&tokens, 0)?,
        ids,
    })
}
