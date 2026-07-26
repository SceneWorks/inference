//! Weight-normalized Conv1d checkpoint compatibility.

use candle_audio::candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

/// Materialize a `weight_norm(dim=0)` Conv1d from any layout emitted by PyTorch.
///
/// Supported layouts are a folded `weight`, legacy `weight_g`/`weight_v`, and modern
/// `parametrizations.weight.original0`/`original1`. The returned module is an ordinary Conv1d;
/// inference never needs the training-time parametrization hook.
pub fn wn_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    bias: bool,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let shape = (out_channels, in_channels / config.groups, kernel_size);
    let weight = if vb.contains_tensor("weight") {
        vb.get(shape, "weight")?
    } else {
        let (g_name, v_name) = if vb.contains_tensor("weight_g") {
            ("weight_g", "weight_v")
        } else {
            (
                "parametrizations.weight.original0",
                "parametrizations.weight.original1",
            )
        };
        let g = vb.get((out_channels, 1, 1), g_name)?;
        let v = vb.get(shape, v_name)?;
        fold_weight_norm(&v, &g)?
    };
    let bias = if bias {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, config))
}

/// Fold `w = g * v / ||v||`, where the norm spans every non-output dimension.
pub fn fold_weight_norm(v: &Tensor, g: &Tensor) -> Result<Tensor> {
    let dims: Vec<usize> = (1..v.rank()).collect();
    let norm = v.sqr()?.sum_keepdim(dims)?.sqrt()?;
    v.broadcast_mul(g)?.broadcast_div(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::{DType, Device};
    use candle_nn::{Module, VarBuilder};
    use std::collections::HashMap;

    fn tensors(layout: &str) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let v = Tensor::from_vec(vec![3f32, 4., 0., 5.], (2, 2, 1), &dev).unwrap();
        let g = Tensor::from_vec(vec![10f32, 15.], (2, 1, 1), &dev).unwrap();
        // Independently calculated: norms are both 5, hence rows [6, 8] and [0, 15].
        let w = Tensor::from_vec(vec![6f32, 8., 0., 15.], (2, 2, 1), &dev).unwrap();
        let mut map = HashMap::new();
        match layout {
            "folded" => {
                map.insert("weight".into(), w);
            }
            "legacy" => {
                map.insert("weight_g".into(), g);
                map.insert("weight_v".into(), v);
            }
            _ => {
                map.insert("parametrizations.weight.original0".into(), g);
                map.insert("parametrizations.weight.original1".into(), v);
            }
        }
        map
    }

    #[test]
    fn all_three_layouts_materialize_identically() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &dev).unwrap();
        let mut outputs = Vec::new();
        for layout in ["folded", "legacy", "modern"] {
            let vb = VarBuilder::from_tensors(tensors(layout), DType::F32, &dev);
            outputs.push(
                wn_conv1d(2, 2, 1, false, Conv1dConfig::default(), vb)
                    .unwrap()
                    .forward(&x)
                    .unwrap()
                    .to_vec3::<f32>()
                    .unwrap(),
            );
        }
        let expected = vec![vec![vec![30.0, 44.0], vec![45.0, 60.0]]];
        assert_eq!(outputs[0], expected);
        assert_eq!(outputs[1], expected);
        assert_eq!(outputs[2], expected);
    }
}
