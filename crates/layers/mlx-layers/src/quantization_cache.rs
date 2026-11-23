//! Persist affine weight quantization through the consumer's shared artifact store.
use crate::{Error, Result};
use mlx_rs::{
    module::Param,
    nn::{Linear, QuantizedLinear},
    Array, Dtype,
};

pub fn quantize_cached(weight: &Array, group: i32, bits: i32) -> Result<(Array, Array, Array)> {
    let context = gen_core::QuantizationCacheContext::capture();
    let Some(cache) = context.cache() else {
        return Ok(mlx_rs::ops::quantize(weight, group, bits)?);
    };
    let input = crate::source::contiguous_weight(weight)?;
    input.eval()?;
    let shape: Vec<usize> = weight.shape().iter().map(|n| *n as usize).collect();
    let bytes: &[u8] = match input.dtype() {
        Dtype::Bfloat16 => bytemuck::cast_slice(
            input
                .try_as_slice::<half::bf16>()
                .map_err(|e| Error::Msg(e.to_string()))?,
        ),
        Dtype::Float16 => bytemuck::cast_slice(
            input
                .try_as_slice::<half::f16>()
                .map_err(|e| Error::Msg(e.to_string()))?,
        ),
        Dtype::Float32 => bytemuck::cast_slice(
            input
                .try_as_slice::<f32>()
                .map_err(|e| Error::Msg(e.to_string()))?,
        ),
        dtype => {
            return Err(Error::Msg(format!(
                "cannot identify {dtype:?} weight quantization"
            )))
        }
    };
    let key = gen_core::QuantizationArtifactKey::new(
        &format!("mlx-affine-{:?}-g{group}-b{bits}-v1", weight.dtype()),
        &shape,
        bytes,
        "safetensors",
    );
    let mut produced = None;
    let artifact = cache
        .materialize(&key, &mut |path| {
            let parts =
                mlx_rs::ops::quantize(weight, group, bits).map_err(gen_core::Error::backend)?;
            Array::save_safetensors(
                [
                    ("weight", &parts.0),
                    ("scales", &parts.1),
                    ("biases", &parts.2),
                ],
                None,
                path,
            )
            .map_err(gen_core::Error::backend)?;
            produced = Some(parts);
            Ok(())
        })
        .map_err(|error| Error::Msg(format!("saved weight quantization: {error}")))?;
    if let Some(parts) = produced {
        return Ok(parts);
    }
    let mut tensors: std::collections::HashMap<String, Array> =
        Array::load_safetensors(artifact.path())?
            .into_iter()
            .collect();
    let mut take = |name: &str| {
        tensors
            .remove(name)
            .ok_or_else(|| Error::Msg(format!("saved quantization lacks {name}")))
    };
    let parts = (take("weight")?, take("scales")?, take("biases")?);
    // Consume the file-backed graph while its artifact lease still protects the file.
    mlx_rs::transforms::eval([&parts.0, &parts.1, &parts.2])?;
    Ok(parts)
}

pub fn quantize_linear_cached(linear: Linear, group: i32, bits: i32) -> Result<QuantizedLinear> {
    if gen_core::QuantizationCacheContext::capture()
        .cache()
        .is_none()
    {
        return Ok(QuantizedLinear::try_from_linear(linear, group, bits)?);
    }
    let (weight, scales, biases) = quantize_cached(&linear.weight.value, group, bits)?;
    Ok(QuantizedLinear {
        group_size: group,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner: Linear {
            weight: Param::new(weight),
            bias: linear.bias,
        },
    })
}
