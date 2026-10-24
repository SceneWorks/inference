//! Headless Qwen3.5: fresh unpadded per-prompt hidden-state forwards.
use crate::loader::{read_weights, Sources};
use mlx_gen::gen_core::moss_image::{
    MossImageConditioningInputReceipt, MossImageConditioningSource, MossImageEmbeddings,
    MOSS_TEXT_ENCODER_PREFIX,
};
use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::{CancelFlag, Error, Result};
use mlx_llm::models::qwen35::{Qwen35Config, Qwen35Model};
use mlx_rs::Array;
use sha2::{Digest, Sha256};
fn tokenizer(s: &Sources) -> Result<TextTokenizer> {
    let json = String::from_utf8(s.find("tokenizer/tokenizer.json")?.bytes()?)
        .map_err(|e| Error::Msg(e.to_string()))?;
    let tokenizer = TextTokenizer::from_json_str(
        &json,
        TokenizerConfig {
            chat_template: ChatTemplate::MossImageQwen35,
            max_length: 4096,
            pad_token_id: 0,
            pad_to_max_length: false,
        },
    )?;
    Ok(tokenizer)
}
pub(crate) fn token_bound(sources: &Sources, req: &mlx_gen::TrainingRequest) -> Result<u64> {
    let tokenizer = tokenizer(sources)?;
    let mut bound = 0usize;
    for prompt in mlx_gen::gen_core::moss_image_training::recovery::conditioning_prompts(req) {
        crate::transformer::check(&req.cancel)?;
        bound = bound.max(tokenizer.encode_chat_ids(&prompt, false)?.len().min(4096));
    }
    if bound == 0 {
        return Err(Error::Msg("MOSS captions produce no tokens".into()));
    }
    Ok(bound as u64)
}
pub(crate) struct Encoder {
    model: Qwen35Model,
    tokenizer: TextTokenizer,
    owned_weight_bytes: u64,
}
impl Encoder {
    #[cfg(feature = "test-support")]
    pub(crate) fn frozen_training_arrays(&self) -> Vec<Array> {
        self.model.param_groups().into_iter().flatten().collect()
    }
    pub(crate) fn load(s: &Sources, cancel: &CancelFlag) -> Result<Self> {
        let config = Qwen35Config::from_json(&s.find("text_encoder/config.json")?.json()?)
            .map_err(|e| Error::Msg(e.to_string()))?;
        let w = read_weights(
            s.ordinary.iter().filter(|a| {
                a.relative.starts_with("text_encoder/") && a.relative.ends_with(".safetensors")
            }),
            |k| k.starts_with("language_model."),
            cancel,
        )?;
        let weights = mlx_llm::primitives::Weights::from_map(w.into_tensors());
        let model = Qwen35Model::from_weights_headless(&weights, MOSS_TEXT_ENCODER_PREFIX, config)
            .map_err(|e| Error::Msg(e.to_string()))?;
        if !weights.unused_keys().is_empty() {
            return Err(Error::Unsupported(format!(
                "unconsumed MOSS Qwen3.5 language tensors {:?}",
                weights.unused_keys()
            )));
        }
        let tokenizer = tokenizer(s)?;
        let mut owned_weight_bytes = 0u64;
        for array in model.param_groups().into_iter().flatten() {
            crate::transformer::check(cancel)?;
            mlx_rs::transforms::eval([&array])?;
            owned_weight_bytes = owned_weight_bytes.saturating_add(array.nbytes() as u64);
        }
        Ok(Self {
            model,
            tokenizer,
            owned_weight_bytes,
        })
    }
    pub(crate) fn resident_bytes(&self) -> u64 {
        self.owned_weight_bytes
    }
    pub(crate) fn encode(
        &self,
        prompt: &str,
        limit: u32,
        cancel: &CancelFlag,
    ) -> Result<(Array, MossImageConditioningInputReceipt)> {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let mut ids = self.tokenizer.encode_chat_ids(prompt, false)?;
        ids.truncate(limit as usize);
        if ids.is_empty() {
            return Err(Error::Unsupported(
                "MOSS formatted prompt produced no tokens".into(),
            ));
        }
        let input = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let mut cache = self.model.new_cache();
        let hidden = self
            .model
            .forward_hidden_with_block_hook(&input, &mut cache, 0, &mut |_, hidden| {
                if cancel.is_cancelled() {
                    return Err(mlx_llm::Error::Canceled);
                }
                mlx_rs::transforms::eval([hidden])?;
                if cancel.is_cancelled() {
                    return Err(mlx_llm::Error::Canceled);
                }
                Ok(())
            })
            .map_err(|e| match e {
                mlx_llm::Error::Canceled => Error::Canceled,
                other => Error::Msg(other.to_string()),
            })?;
        mlx_rs::transforms::eval([&hidden])?;
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        Ok((
            hidden,
            MossImageConditioningInputReceipt {
                source: MossImageConditioningSource::Text,
                sha256: format!("{:x}", Sha256::digest(prompt.as_bytes())),
                sequence_length: ids.len() as u32,
                valid_tokens: ids.len() as u32,
            },
        ))
    }
}
pub(crate) fn supplied(
    e: MossImageEmbeddings<'_>,
    row: usize,
) -> Result<(Array, MossImageConditioningInputReceipt)> {
    let values = &e.values[row * e.sequence * 4096..(row + 1) * e.sequence * 4096];
    let mask = e.mask.map(|m| &m[row * e.sequence..(row + 1) * e.sequence]);
    let mut compact = Vec::new();
    for i in 0..e.sequence {
        if mask.is_none_or(|m| m[i] == 1) {
            compact.extend_from_slice(&values[i * 4096..(i + 1) * 4096]);
        }
    }
    let mut digest = Sha256::new();
    for v in values {
        digest.update(v.to_le_bytes());
    }
    if let Some(m) = mask {
        digest.update(m)
    } else {
        digest.update(vec![1u8; e.sequence]);
    }
    let valid = compact.len() / 4096;
    Ok((
        Array::from_slice(&compact, &[1, valid as i32, 4096]),
        MossImageConditioningInputReceipt {
            source: MossImageConditioningSource::Embeddings,
            sha256: format!("{:x}", digest.finalize()),
            sequence_length: e.sequence as u32,
            valid_tokens: valid as u32,
        },
    ))
}
