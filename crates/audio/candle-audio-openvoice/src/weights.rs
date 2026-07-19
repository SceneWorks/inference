//! Checkpoint loading for the OpenVoice V2 converter (sc-13223).
//!
//! `converter/checkpoint.pth` is a torch zip whose top-level `model` key holds the
//! `SynthesizerTrn` state dict (the reference `load_ckpt` reads `checkpoint_dict['model']`). Keys
//! carry no DataParallel `module.` prefix and use OLD-style weight-norm pairs (`weight_g` /
//! `weight_v`) on every Conv1d/Conv2d/ConvTranspose1d inside `dec`, `enc_q.enc`, `flow`, and
//! `ref_enc.convs`. This module loads that one section, resolves every weight-norm pair into a plain
//! `weight` (`w = g · v / ‖v‖`, norm over all dims except 0 — the torch `weight_norm(dim=0)` default
//! used throughout VITS/HiFi-GAN, which for a ConvTranspose1d's `[in, out, k]` weight norms over
//! `[out, k]` exactly as the shipped `weight_g` shape `[in, 1, 1]` implies), and hands back a single
//! [`VarBuilder`] rooted at the state-dict top.

use std::collections::HashMap;
use std::path::Path;

use candle_audio::candle_core::pickle::PthTensors;
use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::{AudioError, Result};
use candle_nn::VarBuilder;

/// The top-level checkpoint key holding the `SynthesizerTrn` state dict.
pub const CHECKPOINT_SECTION: &str = "model";

/// Load the `model` state dict into a name→tensor map with weight-norm pairs resolved to plain
/// `weight` tensors (all tensors materialized to f32 on `device`).
pub fn load_state_dict(pth: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let tensors = PthTensors::new(pth, Some(CHECKPOINT_SECTION)).map_err(|e| {
        AudioError::Msg(format!(
            "open {} [{CHECKPOINT_SECTION}]: {e}",
            pth.display()
        ))
    })?;
    let mut raw: HashMap<String, Tensor> = HashMap::new();
    for name in tensors.tensor_infos().keys() {
        let t = tensors
            .get(name)
            .map_err(|e| AudioError::Msg(format!("read {CHECKPOINT_SECTION}.{name}: {e}")))?
            .ok_or_else(|| AudioError::Msg(format!("tensor {CHECKPOINT_SECTION}.{name} vanished")))?
            .to_dtype(DType::F32)?
            .to_device(device)?;
        raw.insert(name.clone(), t);
    }
    resolve_weight_norm(raw)
}

/// Fold every `X.weight_g` / `X.weight_v` pair into `X.weight = g · v / ‖v‖` (norm per out-channel —
/// all dims except 0, the torch `weight_norm(dim=0)` default). Non-paired tensors pass through
/// unchanged.
fn resolve_weight_norm(raw: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(raw.len());
    let g_suffix = ".weight_g";
    for (name, tensor) in &raw {
        if let Some(base) = name.strip_suffix(g_suffix) {
            let v = raw
                .get(&format!("{base}.weight_v"))
                .ok_or_else(|| AudioError::Msg(format!("{base}: weight_g without weight_v")))?;
            // ‖v‖ over all dims except 0, kept broadcastable against v.
            let mut sq = v.sqr()?;
            for d in 1..v.rank() {
                sq = sq.sum_keepdim(d)?;
            }
            let norm = sq.sqrt()?;
            let w = v.broadcast_mul(&tensor.broadcast_div(&norm)?)?;
            out.insert(format!("{base}.weight"), w);
        } else if !name.ends_with(".weight_v") {
            out.insert(name.clone(), tensor.clone());
        }
    }
    Ok(out)
}

/// A [`VarBuilder`] over the resolved state dict, rooted at the top (`dec.*`, `enc_q.*`, `flow.*`,
/// `ref_enc.*`), on `device` (f32 — the checkpoint's dtype).
pub fn state_var_builder(pth: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let tensors = load_state_dict(pth, device)?;
    Ok(VarBuilder::from_tensors(tensors, DType::F32, device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_norm_resolution_matches_the_definition() {
        let dev = Device::Cpu;
        // v: [2, 1, 3]; g: [2, 1, 1] → w[o] = g[o] · v[o] / ‖v[o]‖ (norm over dims 1..).
        let v = Tensor::from_slice(&[3.0f32, 4.0, 0.0, 0.0, 5.0, 12.0], (2, 1, 3), &dev).unwrap();
        let g = Tensor::from_slice(&[10.0f32, 26.0], (2, 1, 1), &dev).unwrap();
        let mut raw = HashMap::new();
        raw.insert("conv.weight_v".to_string(), v);
        raw.insert("conv.weight_g".to_string(), g);
        raw.insert(
            "conv.bias".to_string(),
            Tensor::zeros(2, DType::F32, &dev).unwrap(),
        );
        let out = resolve_weight_norm(raw).unwrap();
        let w: Vec<f32> = out["conv.weight"].flatten_all().unwrap().to_vec1().unwrap();
        // ‖v0‖ = 5, ‖v1‖ = 13 → w0 = 2·[3,4,0]; w1 = 2·[0,5,12].
        assert_eq!(w, [6.0, 8.0, 0.0, 0.0, 10.0, 24.0]);
        assert!(out.contains_key("conv.bias"));
        assert!(!out.contains_key("conv.weight_g"));
        assert!(!out.contains_key("conv.weight_v"));
    }

    #[test]
    fn weight_norm_without_its_pair_is_an_error() {
        let dev = Device::Cpu;
        let mut raw = HashMap::new();
        raw.insert(
            "conv.weight_g".to_string(),
            Tensor::zeros((2, 1, 1), DType::F32, &dev).unwrap(),
        );
        assert!(resolve_weight_norm(raw).is_err());
    }
}
