//! Mochi 1 static configuration — the T5-XXL conditioning geometry and the AsymmVAE decoder
//! structure. Both mirror `genmo/mochi-1-preview` (diffusers `MochiPipeline` /
//! `AutoencoderKLMochi`); the VAE structure is read from the snapshot's `vae/config.json` so the
//! per-channel latent statistics ride on the checkpoint rather than being hardcoded twice.

use std::path::Path;

use mlx_gen::{Error, Result};

/// T5-XXL (google/t5-v1.1-xxl) conditioning geometry — identical to the encoder FLUX/Chroma reuse.
/// The structural constants (24 layers, 64 heads × 64 head-dim → 4096 model dim, gated-GELU FFN) are
/// baked into [`mlx_gen_flux::T5TextEncoder`], so this type carries them for documentation + the
/// `text_len` padding policy that Mochi's `_get_t5_prompt_embeds` applies (`max_sequence_length=256`).
#[derive(Debug, Clone, Copy)]
pub struct MochiConfig {
    /// T5 model dimension (`d_model`).
    pub t5_dim: usize,
    /// Number of T5 encoder blocks.
    pub t5_layers: usize,
    /// Number of self-attention heads.
    pub t5_heads: usize,
    /// Per-head dimension (`inner_dim = heads * head_dim = 4096`).
    pub t5_head_dim: usize,
    /// Padded conditioning length (`max_sequence_length`, Mochi default 256).
    pub text_len: usize,
}

impl Default for MochiConfig {
    fn default() -> Self {
        Self {
            t5_dim: 4096,
            t5_layers: 24,
            t5_heads: 64,
            t5_head_dim: 64,
            text_len: 256,
        }
    }
}

impl MochiConfig {
    /// The Mochi 1 defaults. `root` is accepted for symmetry with [`MochiVaeConfig::from_model_dir`]
    /// and future config-driven fields; the T5-XXL geometry is fixed by the reused encoder so this is
    /// currently infallible and equal to [`Default`].
    pub fn from_model_dir(_root: &Path) -> Result<Self> {
        Ok(Self::default())
    }
}

/// AsymmVAE decoder structure + per-channel latent statistics (diffusers `AutoencoderKLMochi`).
///
/// The decoder is asymmetric to the encoder (distinct `decoder_block_out_channels`); only the decode
/// path is ported (A2). Compression is `6×` temporal (∏ `temporal_expansions`) and `8×` spatial
/// (∏ `spatial_expansions`). `latents_mean`/`latents_std` are the per-channel de-normalization stats
/// read from `vae/config.json`.
#[derive(Debug, Clone)]
pub struct MochiVaeConfig {
    /// Latent channels fed to the decoder `conv_in` (12).
    pub latent_channels: usize,
    /// Output (pixel) channels (3).
    pub out_channels: usize,
    /// Per-stage decoder channel widths, low→high (`[128, 256, 512, 768]`).
    pub decoder_block_out_channels: Vec<usize>,
    /// Resnet-block counts per stage (`[3, 3, 4, 6, 3]`).
    pub layers_per_block: Vec<usize>,
    /// Temporal expansion per up-stage (`[1, 2, 3]`, ∏ = 6× temporal ratio).
    pub temporal_expansions: Vec<usize>,
    /// Spatial expansion per up-stage (`[2, 2, 2]`, ∏ = 8× spatial ratio).
    pub spatial_expansions: Vec<usize>,
    /// Per-channel latent mean (len == `latent_channels`).
    pub latents_mean: Vec<f32>,
    /// Per-channel latent std (len == `latent_channels`).
    pub latents_std: Vec<f32>,
    /// Latent scaling factor (Mochi = 1.0).
    pub scaling_factor: f32,
}

impl Default for MochiVaeConfig {
    /// The `genmo/mochi-1-preview` `vae/config.json` values, hardcoded so the crate compiles and the
    /// synthetic CI test has a real reference; the real path prefers [`from_model_dir`](Self::from_model_dir).
    fn default() -> Self {
        Self {
            latent_channels: 12,
            out_channels: 3,
            decoder_block_out_channels: vec![128, 256, 512, 768],
            layers_per_block: vec![3, 3, 4, 6, 3],
            temporal_expansions: vec![1, 2, 3],
            spatial_expansions: vec![2, 2, 2],
            latents_mean: vec![
                -0.067_308_96,
                -0.038_011_383,
                -0.074_778_21,
                -0.055_652_645,
                0.012_767_231,
                -0.047_035_426,
                0.043_896_97,
                -0.093_463_06,
                -0.099_183_15,
                -0.008_729_793,
                -0.011_931_556,
                -0.032_199_34,
            ],
            latents_std: vec![
                0.926_379_5,
                0.924_889_45,
                0.939_306,
                0.959_253_7,
                0.824_456,
                0.917_26,
                0.929_415_46,
                1.372_094_2,
                0.881_393_7,
                0.916_831_6,
                0.918_524_9,
                0.927_475_75,
            ],
            scaling_factor: 1.0,
        }
    }
}

impl MochiVaeConfig {
    /// Read `<root>/vae/config.json` (diffusers `AutoencoderKLMochi`) into a decoder config. Falls
    /// back to a field's [`Default`] value when the key is absent so a partial config still loads.
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let path = root.join("vae").join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Msg(format!("mochi vae config {}: {e}", path.display())))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("mochi vae config {}: {e}", path.display())))?;
        let d = Self::default();

        let usize_at = |key: &str, dflt: usize| -> usize {
            json.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(dflt)
        };
        let usize_vec = |key: &str, dflt: &[usize]| -> Vec<usize> {
            json.get(key)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64().map(|v| v as usize))
                        .collect()
                })
                .unwrap_or_else(|| dflt.to_vec())
        };
        let f32_vec = |key: &str, dflt: &[f32]| -> Vec<f32> {
            json.get(key)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64().map(|v| v as f32))
                        .collect()
                })
                .unwrap_or_else(|| dflt.to_vec())
        };

        Ok(Self {
            latent_channels: usize_at("latent_channels", d.latent_channels),
            out_channels: usize_at("out_channels", d.out_channels),
            decoder_block_out_channels: usize_vec(
                "decoder_block_out_channels",
                &d.decoder_block_out_channels,
            ),
            layers_per_block: usize_vec("layers_per_block", &d.layers_per_block),
            temporal_expansions: usize_vec("temporal_expansions", &d.temporal_expansions),
            spatial_expansions: usize_vec("spatial_expansions", &d.spatial_expansions),
            latents_mean: f32_vec("latents_mean", &d.latents_mean),
            latents_std: f32_vec("latents_std", &d.latents_std),
            scaling_factor: json
                .get("scaling_factor")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(d.scaling_factor),
        })
    }

    /// `∏ temporal_expansions` — the temporal compression ratio (6). Also the count of leading
    /// decoded frames dropped (`ratio − 1`) to realign to the causal output length.
    pub fn temporal_compression_ratio(&self) -> usize {
        self.temporal_expansions.iter().product()
    }

    /// `∏ spatial_expansions` — the spatial compression ratio (8).
    pub fn spatial_compression_ratio(&self) -> usize {
        self.spatial_expansions.iter().product()
    }
}

/// The tier `split_model.json` manifest's quantization fields — the Mochi analog of LTX's
/// `SplitModel` (sc-2686). A pre-quantized tier dir (`q4/`, `q8/`) carries `quantized:true` plus
/// `quantization_bits` (4 → Q4, 8 → Q8) and `quantization_group_size` (64); the `bf16/` tier is dense
/// (`quantized:false`). The quant geometry is **read from the manifest, never hardcoded** — a tier dir
/// *is* the quant selection — and the per-Linear `.scales` probe (in [`crate::transformer`]) picks
/// which Linears consume packed weights. `convert.rs` (story A6) writes this file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MochiSplitModel {
    /// `quantized` — whether the transformer ships selectively-quantized Linears.
    pub quantized: bool,
    /// `quantization_bits` (default 4): 4 → Q4, 8 → Q8.
    pub bits: i32,
    /// `quantization_group_size` (default 64): the affine-quant group width.
    pub group: i32,
}

impl MochiSplitModel {
    /// A non-quantized manifest (file absent or `quantized:false`) — the raw bf16 snapshot / `bf16/`
    /// tier. Bits/group hold the reference defaults so a downstream quant geometry is well-defined.
    pub fn dense() -> Self {
        Self {
            quantized: false,
            bits: 4,
            group: 64,
        }
    }

    /// Parse `<root>/split_model.json`. Absent → [`dense`](Self::dense) (the raw snapshot has no
    /// manifest), mirroring the LTX loader (only quantize when the manifest sets `quantized:true`).
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let path = root.join("split_model.json");
        if !path.exists() {
            return Ok(Self::dense());
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Msg(format!("mochi split_model {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("mochi split_model {}: {e}", path.display())))?;
        Ok(Self::from_value(&v))
    }

    /// Parse the manifest fields from a parsed `split_model.json` value.
    pub fn from_value(v: &serde_json::Value) -> Self {
        let i32_at = |key: &str, dflt: i32| -> i32 {
            v.get(key)
                .and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|f| f as i64)))
                .map(|x| x as i32)
                .unwrap_or(dflt)
        };
        Self {
            quantized: v
                .get("quantized")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            bits: i32_at("quantization_bits", 4),
            group: i32_at("quantization_group_size", 64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_model_defaults_and_parse() {
        // Absent manifest → dense.
        let d = MochiSplitModel::dense();
        assert!(!d.quantized);
        assert_eq!((d.bits, d.group), (4, 64));

        // Quantized manifest parses bits/group.
        let v = serde_json::json!({
            "quantized": true,
            "quantization_bits": 4,
            "quantization_group_size": 64,
        });
        let q = MochiSplitModel::from_value(&v);
        assert!(q.quantized);
        assert_eq!((q.bits, q.group), (4, 64));

        // A dense manifest (bf16 tier) stays dense; missing bits fall back to defaults.
        let vb = serde_json::json!({ "quantized": false });
        let b = MochiSplitModel::from_value(&vb);
        assert!(!b.quantized);
        assert_eq!((b.bits, b.group), (4, 64));
    }

    #[test]
    fn defaults_match_mochi_geometry() {
        let c = MochiConfig::default();
        assert_eq!(c.t5_dim, 4096);
        assert_eq!(c.t5_heads * c.t5_head_dim, c.t5_dim);
        assert_eq!(c.text_len, 256);

        let v = MochiVaeConfig::default();
        assert_eq!(v.latent_channels, 12);
        assert_eq!(v.temporal_compression_ratio(), 6);
        assert_eq!(v.spatial_compression_ratio(), 8);
        assert_eq!(v.latents_mean.len(), v.latent_channels);
        assert_eq!(v.latents_std.len(), v.latent_channels);
        // decoder up-stage count == block_out_channels - 1.
        assert_eq!(v.decoder_block_out_channels.len(), 4);
        assert_eq!(
            v.temporal_expansions.len(),
            v.decoder_block_out_channels.len() - 1
        );
    }
}
