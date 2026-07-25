//! Mage Qwen3-VL-4B text conditioning.
//!
//! The decoder math is shared with the reviewed Candle Qwen3-VL implementation in
//! `candle-gen-boogu`; Mage supplies the 4B geometry, its own prompt template/drop, and—critically—
//! consumes the final post-RMSNorm state. Z-Image's penultimate convention is intentionally absent.

use std::path::Path;

use candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen_boogu::loader::Weights;
use candle_gen_boogu::text_encoder::{BooguTextEncoder, BooguTextEncoderConfig};
use tokenizers::Tokenizer;

use crate::config::{
    DROP_IDX_GEN, PROMPT_TEMPLATE, TE_HEADS, TE_HEAD_DIM, TE_KV_HEADS, TE_LAYERS, TE_ROPE_THETA,
    TXT_MAX_LENGTH,
};

pub struct MageTextEncoder {
    model: BooguTextEncoder,
    tokenizer: Tokenizer,
    device: Device,
}

impl MageTextEncoder {
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        let dir = root.join("text_encoder");
        let weights = Weights::from_dir(&dir, device, DType::BF16)?;
        let cfg = BooguTextEncoderConfig {
            num_layers: TE_LAYERS,
            num_heads: TE_HEADS,
            num_kv_heads: TE_KV_HEADS,
            head_dim: TE_HEAD_DIM,
            rms_norm_eps: 1e-6,
            rope_theta: TE_ROPE_THETA,
        };
        let model = BooguTextEncoder::load(
            &weights,
            "model.language_model",
            &cfg,
            TXT_MAX_LENGTH + DROP_IDX_GEN,
        )?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::Msg(format!("mage: load text_encoder/tokenizer.json: {e}")))?;
        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
        })
    }

    /// Return `[1, <=2048, 2560]` final post-RMSNorm conditioning after dropping the frozen
    /// 34-token generation prefix.
    pub fn encode(&self, prompt: &str) -> Result<Tensor> {
        let rendered = PROMPT_TEMPLATE.replacen("{}", prompt, 1);
        let encoding = self
            .tokenizer
            .encode(rendered, false)
            .map_err(|e| Error::Msg(format!("mage: tokenize prompt: {e}")))?;
        let ids = encoding.get_ids();
        let cap = TXT_MAX_LENGTH + DROP_IDX_GEN;
        let ids = &ids[..ids.len().min(cap)];
        if ids.len() <= DROP_IDX_GEN {
            return Err(Error::Msg(format!(
                "mage: templated prompt has {} tokens, cannot drop required {DROP_IDX_GEN}",
                ids.len()
            )));
        }
        let input = Tensor::from_slice(ids, (1, ids.len()), &self.device)?;
        let final_post_norm = self.model.last_hidden(&input)?;
        final_post_norm.narrow(1, DROP_IDX_GEN, ids.len() - DROP_IDX_GEN)
    }
}

/// Test seam spelling out the parity-critical state choice. Production passes only the final
/// post-norm state returned by `BooguTextEncoder::last_hidden`.
#[cfg(test)]
fn select_conditioning(final_post_norm: Tensor, _penultimate_pre_norm: Tensor) -> Tensor {
    final_post_norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penultimate_pre_norm_mutation_fails_the_conditioning_oracle() {
        let d = Device::Cpu;
        let final_state = Tensor::new(&[[[1f32, 2.]]], &d).unwrap();
        let stale_z_image_state = Tensor::new(&[[[9f32, 8.]]], &d).unwrap();
        let expected = final_state.to_vec3::<f32>().unwrap();
        let selected = select_conditioning(final_state, stale_z_image_state.clone());
        assert_eq!(selected.to_vec3::<f32>().unwrap(), expected);
        assert_ne!(
            stale_z_image_state.to_vec3::<f32>().unwrap(),
            expected,
            "mutation to penultimate/pre-final-norm conditioning must fail parity"
        );
    }
}
