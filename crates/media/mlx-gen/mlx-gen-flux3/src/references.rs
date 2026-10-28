//! Frozen `Pipeline.encode_references`: independent AE image_cond rows, never noisy-state init.
use crate::autoencoder::Flux3AutoEncoder;
use mlx_gen::gen_core::flux3::references::{conditioning_ids, PreparedReferences};
use mlx_gen::{CancelFlag, Error, Result};
use mlx_rs::{Array, Dtype};

pub struct ReferenceConditioning {
    pub tokens: Array,
    pub ids: Vec<[f32; 4]>,
}

/// All references were host-validated/prepared before heavy components are entered.
pub fn encode(
    ae: &Flux3AutoEncoder,
    prepared: &PreparedReferences,
    cancel: &CancelFlag,
) -> Result<ReferenceConditioning> {
    let mut tokens = Vec::with_capacity(prepared.images.len());
    let mut ids = Vec::new();
    for (index, image) in prepared.images.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let values: Vec<f32> = image
            .pixels
            .iter()
            .map(|&v| v as f32 / 127.5 - 1.0)
            .collect();
        let input = Array::from_slice(&values, &[1, image.height as i32, image.width as i32, 3]);
        let (t, (h, w)) = ae.encode(&input)?;
        let t = t.as_dtype(Dtype::Bfloat16)?;
        // Bound lazy AE graphs per reference; do not retain ten encoders' activation graphs.
        mlx_rs::transforms::eval([&t])?;
        tokens.push(t);
        ids.extend(conditioning_ids(h as usize, w as usize, index)?);
    }
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    if tokens.is_empty() {
        return Err(Error::Unsupported(
            "flux3: reference encoder requires 1–10 prepared references".into(),
        ));
    }
    let refs: Vec<&Array> = tokens.iter().collect();
    Ok(ReferenceConditioning {
        tokens: mlx_rs::ops::concatenate_axis(&refs, 0)?,
        ids,
    })
}
