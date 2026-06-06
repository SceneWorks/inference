//! Native (Rust/MLX) Wan2.2 weight converter (sc-3224). Replaces the Python `mlx_video.convert_wan`.
//!
//! Wan native checkpoints ship the transformer as safetensors but the T5 encoder and VAE as torch
//! `.pth` (zip-of-pickle) — read via [`crate::pth`]. This module ports the reference sanitizers that
//! map the native key layout onto the MLX model layout the Wan loaders consume.
//!
//! **sc-3237 (this slice): the Wan2.2 VAE path.** [`convert_vae22`] reads `Wan2.2_VAE.pth`, applies
//! [`sanitize_wan22_vae`] (the reference `sanitize_wan22_vae_weights`), and writes
//! `vae.safetensors` in f32 (official Wan runs VAE decode in float32). The transformer + T5 +
//! orchestration (single-model TI2V-5B, dual-expert I2V-14B) are sc-3238 / sc-3239.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::{Error, Result};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

/// Channels-last transpose of a PyTorch conv weight: Conv3d `[O,I,D,H,W]→[O,D,H,W,I]`, Conv2d
/// `[O,I,H,W]→[O,H,W,I]`. Other ranks pass through.
fn conv_channels_last(v: &Array) -> Result<Array> {
    match v.ndim() {
        5 => Ok(v.transpose_axes(&[0, 2, 3, 4, 1])?),
        4 => Ok(v.transpose_axes(&[0, 2, 3, 1])?),
        _ => Ok(v.clone()),
    }
}

/// Drop every size-1 axis (`np.squeeze`) — for the RMS_norm `gamma` tensors `(dim,1,1,1)`/`(dim,1,1)`
/// → `(dim,)`.
fn squeeze_all(v: &Array) -> Result<Array> {
    let new_shape: Vec<i32> = v.shape().iter().copied().filter(|&d| d != 1).collect();
    Ok(v.reshape(&new_shape)?)
}

/// Port of `sanitize_wan22_vae_weights` (mlx_video/models/wan/vae22.py): map the native Wan2.2 VAE
/// key layout (PyTorch `nn.Sequential` indices, channels-first convs, 4-D RMS gammas) onto the MLX
/// `WanVae22` layout. With `include_encoder=false` the encoder + `conv1.*` are dropped (decode-only);
/// TI2V/I2V keep them. Conv weights → channels-last; `gamma` → squeezed.
pub fn sanitize_wan22_vae(
    raw: &HashMap<String, Array>,
    include_encoder: bool,
) -> Result<HashMap<String, Array>> {
    let mut out = HashMap::new();
    for (k, src) in raw {
        if !include_encoder && (k.starts_with("encoder.") || k.starts_with("conv1.")) {
            continue;
        }

        // Sequential index → named layer: residual.{0,2,3,6} and head.{0,2}.
        let mut new = k.clone();
        for idx in ["0", "2", "3", "6"] {
            new = new.replace(
                &format!(".residual.{idx}."),
                &format!(".residual.layer_{idx}."),
            );
        }
        for idx in ["0", "2"] {
            new = new.replace(&format!(".head.{idx}."), &format!(".head.layer_{idx}."));
        }
        // Resample Conv2d + AttentionBlock Conv2d renames (first match wins, mirroring the if/elif).
        if new.contains(".resample.1.weight") {
            new = new.replace(".resample.1.weight", ".resample_weight");
        } else if new.contains(".resample.1.bias") {
            new = new.replace(".resample.1.bias", ".resample_bias");
        }
        if new.contains(".to_qkv.weight") {
            new = new.replace(".to_qkv.weight", ".to_qkv_weight");
        } else if new.contains(".to_qkv.bias") {
            new = new.replace(".to_qkv.bias", ".to_qkv_bias");
        } else if new.contains(".proj.weight") && !new.contains("time_projection") {
            new = new.replace(".proj.weight", ".proj_weight");
        } else if new.contains(".proj.bias") && !new.contains("time_projection") {
            new = new.replace(".proj.bias", ".proj_bias");
        }

        // Conv-weight channels-last (keys ending `.weight` OR the renamed `_weight`).
        let mut value = if new.ends_with(".weight") || new.ends_with("_weight") {
            conv_channels_last(src)?
        } else {
            src.clone()
        };
        // RMS_norm gamma: squeeze trailing singleton dims.
        if new.contains("gamma") {
            value = squeeze_all(&value)?;
        }
        out.insert(new, value);
    }
    Ok(out)
}

/// Convert a Wan2.2 `Wan2.2_VAE.pth` into `out_file` (`vae.safetensors`), f32. `include_encoder` is
/// `true` for TI2V/I2V (encode path needed), `false` for decode-only T2V.
pub fn convert_vae22(
    vae_pth: impl AsRef<Path>,
    out_file: impl AsRef<Path>,
    include_encoder: bool,
) -> Result<()> {
    let vae_pth = vae_pth.as_ref();
    if !vae_pth.is_file() {
        return Err(Error::Msg(format!(
            "Wan VAE .pth not found: {}",
            vae_pth.display()
        )));
    }
    // Load the native .pth as f32 (mirrors torch.load(...).float()), then sanitize.
    let raw = crate::pth::load_pth_f32(vae_pth)?;
    let sanitized = sanitize_wan22_vae(&raw, include_encoder)?;

    let arrays: Vec<&Array> = sanitized.values().collect();
    eval(arrays)?;
    if let Some(parent) = out_file.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    Array::save_safetensors(
        sanitized.iter().map(|(k, v)| (k.as_str(), v)),
        None::<&HashMap<String, String>>,
        out_file.as_ref(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::all_close;

    fn exact_eq(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape() && all_close(a, b, 0.0, 0.0, false).unwrap().item::<bool>()
    }

    fn m(entries: &[(&str, Array)]) -> HashMap<String, Array> {
        entries
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    /// Key renames: Sequential index → layer_N, resample/to_qkv/proj conv renames.
    #[test]
    fn vae_key_renames() {
        let ones5 = Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(); // conv3d weight
        let s = sanitize_wan22_vae(
            &m(&[
                ("decoder.middle.0.residual.0.weight", ones5.clone()),
                (
                    "decoder.middle.0.residual.6.bias",
                    Array::ones::<f32>(&[2]).unwrap(),
                ),
                (
                    "decoder.head.0.gamma",
                    Array::ones::<f32>(&[4, 1, 1, 1]).unwrap(),
                ),
                ("decoder.head.2.weight", ones5.clone()),
                (
                    "decoder.upsamples.0.upsamples.0.resample.1.weight",
                    Array::ones::<f32>(&[2, 2, 3, 3]).unwrap(),
                ),
                (
                    "decoder.middle.0.to_qkv.weight",
                    Array::ones::<f32>(&[6, 2, 1, 1]).unwrap(),
                ),
                (
                    "decoder.middle.0.proj.bias",
                    Array::ones::<f32>(&[2]).unwrap(),
                ),
            ]),
            true,
        )
        .unwrap();
        let mut keys: Vec<&str> = s.keys().map(String::as_str).collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "decoder.head.layer_0.gamma",
                "decoder.head.layer_2.weight",
                "decoder.middle.0.proj_bias",
                "decoder.middle.0.residual.layer_0.weight",
                "decoder.middle.0.residual.layer_6.bias",
                "decoder.middle.0.to_qkv_weight",
                "decoder.upsamples.0.upsamples.0.resample_weight",
            ]
        );
    }

    /// `include_encoder=false` drops `encoder.*` and `conv1.*`; `true` keeps them.
    #[test]
    fn vae_encoder_gating() {
        let entries = [
            (
                "encoder.conv1.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
            (
                "conv1.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
            (
                "conv2.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
            (
                "decoder.conv1.weight",
                Array::ones::<f32>(&[2, 2, 1, 1, 1]).unwrap(),
            ),
        ];
        let dec_only = sanitize_wan22_vae(&m(&entries), false).unwrap();
        assert!(!dec_only
            .keys()
            .any(|k| k.starts_with("encoder.") || k.starts_with("conv1.")));
        assert!(dec_only.contains_key("conv2.weight"));
        assert!(dec_only.contains_key("decoder.conv1.weight")); // not a top-level conv1
        let with_enc = sanitize_wan22_vae(&m(&entries), true).unwrap();
        assert!(with_enc.contains_key("conv1.weight"));
        assert!(with_enc.contains_key("encoder.conv1.weight"));
    }

    /// Conv3d weight → channels-last; gamma squeezed; bias untouched.
    #[test]
    fn vae_transpose_and_squeeze() {
        // Conv3d [O=1,I=2,D=1,H=1,W=2] row-major 0..3 → [O,D,H,W,I]=[1,1,1,2,2] values [0,2,1,3].
        let v = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 2, 1, 1, 2]);
        let s = sanitize_wan22_vae(
            &m(&[
                ("conv2.weight", v),
                (
                    "decoder.middle.0.norm.gamma",
                    Array::ones::<f32>(&[3, 1, 1, 1]).unwrap(),
                ),
            ]),
            true,
        )
        .unwrap();
        assert!(exact_eq(
            &s["conv2.weight"],
            &Array::from_slice(&[0.0f32, 2.0, 1.0, 3.0], &[1, 1, 1, 2, 2])
        ));
        assert_eq!(s["decoder.middle.0.norm.gamma"].shape(), &[3]);
    }
}
