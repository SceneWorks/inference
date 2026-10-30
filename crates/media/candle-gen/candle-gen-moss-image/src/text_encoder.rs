//! Individual unpadded headless Qwen3.5 forwards, preserving GDN state isolation.
use crate::{
    loader::{tensor_f32, Sources},
    transformer::check_cancel,
};
use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::CancelFlag;
use candle_gen::Result;
use candle_llm::models::qwen35::Qwen35Model;
use gen_core_moss_image::moss_image::*;
use gen_core_tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use sha2::{Digest, Sha256};

pub(crate) struct Conditioning {
    pub rows: Vec<Tensor>,
    pub receipts: Vec<MossImageConditioningInputReceipt>,
}
pub(crate) fn digest_f32(values: &[f32], mask: Option<&[u8]>) -> String {
    let mut d = Sha256::new();
    for value in values {
        d.update(value.to_le_bytes());
    }
    if let Some(mask) = mask {
        d.update(mask);
    }
    format!("{:x}", d.finalize())
}
fn tokenizer(sources: &Sources, limit: u32) -> Result<TextTokenizer> {
    let bytes = sources.tokenizer.authenticated_bytes()?;
    let json =
        std::str::from_utf8(&bytes).map_err(|e| candle_gen::CandleError::Msg(e.to_string()))?;
    let tokenizer = TextTokenizer::from_json_str(
        json,
        TokenizerConfig {
            max_length: limit as usize,
            pad_token_id: 151643,
            chat_template: ChatTemplate::MossImageQwen35,
            pad_to_max_length: false,
        },
    )?;
    #[cfg(feature = "model-inference-proof")]
    candle_gen::gen_core::model_input_proof::record_reader_paths(
        "candle-gen-moss-image::text_encoder::tokenizer",
        candle_gen::gen_core::model_input_proof::InputRole::Tokenizer,
        &[sources.tokenizer.pin.canonical_target_path()],
    );
    Ok(tokenizer)
}
pub(crate) fn token_bound(
    sources: &Sources,
    req: &candle_gen::gen_core::TrainingRequest,
    host_budget: u64,
) -> Result<u64> {
    gen_core_moss_image::moss_image_training::recovery::admit_conditioning_preprocess(
        req,
        host_budget,
        0,
    )?;
    let tokenizer = tokenizer(sources, 4096)?;
    let mut bound = 0usize;
    for prompt in gen_core_moss_image::moss_image_training::recovery::conditioning_prompts_iter(req)
    {
        check_cancel(&req.cancel)?;
        bound = bound.max(tokenizer.encode_chat_ids(&prompt, true)?.len().min(4096));
    }
    if bound == 0 {
        return Err(candle_gen::CandleError::Msg(
            "MOSS captions produce no tokens".into(),
        ));
    }
    Ok(bound as u64)
}
pub(crate) fn strings(
    sources: &Sources,
    encoder: &Qwen35Model,
    prompts: &[String],
    limit: u32,
    device: &Device,
    cancel: &CancelFlag,
) -> Result<Conditioning> {
    let tokenizer = tokenizer(sources, limit)?;
    let mut rows = Vec::new();
    let mut receipts = Vec::new();
    for text in prompts {
        check_cancel(cancel)?;
        let mut ids = tokenizer.encode_chat_ids(text, true)?;
        ids.truncate(limit as usize);
        let ids: Vec<u32> = ids.into_iter().map(|i| i as u32).collect();
        let len = ids.len();
        let mut cache = encoder.new_cache();
        let row = encoder
            .forward_hidden_checked(
                &Tensor::from_vec(ids, (1, len), device)?,
                &mut cache,
                0,
                &mut |hidden| {
                    hidden.device().synchronize()?;
                    if cancel.is_cancelled() {
                        Err(candle_llm::Error::Canceled)
                    } else {
                        Ok(())
                    }
                },
            )
            .map_err(|e| match e {
                candle_llm::Error::Canceled => candle_gen::CandleError::Canceled,
                other => candle_gen::CandleError::Msg(other.to_string()),
            })?;
        check_cancel(cancel)?;
        device.synchronize()?;
        // Reject failures before a later backend read can disguise them as output success.
        if tensor_f32(&row)?.iter().any(|v| !v.is_finite()) {
            return Err(candle_gen::CandleError::Msg(
                "MOSS nonfinite text states".into(),
            ));
        }
        rows.push(row);
        receipts.push(MossImageConditioningInputReceipt {
            source: MossImageConditioningSource::Text,
            sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
            sequence_length: len as u32,
            valid_tokens: len as u32,
        });
    }
    Ok(Conditioning { rows, receipts })
}
pub(crate) fn supplied(e: MossImageEmbeddings<'_>, device: &Device) -> Result<Conditioning> {
    let mut rows = Vec::new();
    let mut receipts = Vec::new();
    for b in 0..e.batch {
        let values = &e.values[b * e.sequence * 4096..(b + 1) * e.sequence * 4096];
        let mask = e.mask.map(|m| &m[b * e.sequence..(b + 1) * e.sequence]);
        let selected: Vec<f32> = values
            .chunks_exact(4096)
            .enumerate()
            .filter(|(i, _)| mask.is_none_or(|m| m[*i] == 1))
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        let len = selected.len() / 4096;
        // sc-26324 (F-276): author prepare preserves supplied FP32 conditioning on every
        // device. The selected transformer's projection owns any weight-dtype conversion.
        rows.push(Tensor::from_vec(selected, (1, len, 4096), device)?);
        receipts.push(MossImageConditioningInputReceipt {
            source: MossImageConditioningSource::Embeddings,
            sha256: digest_f32(values, mask),
            sequence_length: e.sequence as u32,
            valid_tokens: len as u32,
        });
    }
    Ok(Conditioning { rows, receipts })
}
