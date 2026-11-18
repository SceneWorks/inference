//! Teacher-forced real-weight comparison using the loaded provider's own
//! tokenizer, image encoder, splice, positions and decoder. Manual proof only.
use super::*;

/// Build the same U32 embedding indices as generation. An out-of-vocabulary
/// oracle token must fail before it reaches a device gather.
fn proof_token_indices(ids: &[i32], vocab_size: i32, device: &Device) -> crate::Result<Tensor> {
    if let Some(id) = ids.iter().copied().find(|id| !(0..vocab_size).contains(id)) {
        return Err(crate::Error::Msg(format!(
            "SenseNova proof token {id} is outside vocabulary 0..{vocab_size}"
        )));
    }
    input_ids(ids, device)
}

impl LlamaProvider {
    /// Actual expanded prompt and last-position logits at prefill and each
    /// forced continuation. Each row is paired with a fresh all-token prefill.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn sensenova_proof_logits(
        &self,
        req: &TextLlmRequest,
        forced: &[i32],
    ) -> Result<(Vec<i32>, Vec<(Vec<f32>, Vec<f32>)>), Box<dyn std::error::Error>> {
        let Decoder::Qwen35(model) = &self.model else {
            return Err("SenseNova decoder required".into());
        };
        let plan = model
            .config()
            .sensenova_loop
            .ok_or("SenseNova Loop required")?;
        let device = model.device();
        if !device.is_cuda() {
            return Err("SenseNova release proof requires CUDA".into());
        }
        let image = !collect_images(&req.messages).is_empty();
        let messages = if image {
            substitute_vision_placeholders(&req.messages, 2)?
        } else {
            req.messages.clone()
        };
        let (disabled, ids) = self.render_request(req, &messages)?;
        assert!(disabled.ends_with("<think>\n\n</think>\n\n"));
        for mode in [
            core_llm::ThinkingMode::Auto,
            core_llm::ThinkingMode::Enabled,
        ] {
            let mut thinking = req.clone();
            thinking.thinking = mode;
            let (prompt, tokens) = self.render_request(&thinking, &messages)?;
            assert!(prompt.ends_with("<think>\n"));
            assert_ne!(tokens, ids);
        }
        let prepared = if image {
            Some(self.prepare_multimodal(&ids, &req.messages)?)
        } else {
            None
        };
        let prompt_ids = prepared.as_ref().map_or(ids, |p| p.expanded_ids.clone());
        let n = prompt_ids.len();
        let input = |ids: &[i32]| proof_token_indices(ids, model.config().vocab_size, device);
        let mut cache = model.new_static_cache(n + forced.len() + 1, 0)?;
        let mut embeds = if let Some(p) = &prepared {
            p.embeds.clone()
        } else {
            model.embed_input_ids(&input(&prompt_ids)?)?
        };
        let mut positions = prepared.as_ref().map_or_else(
            || {
                let text: Vec<_> = (0..n as i32).collect();
                (text.clone(), text.clone(), text, 0)
            },
            |p| p.positions.clone(),
        );
        cache.set_rope_delta(positions.3);
        let mut logits = model.decode_logits_from_embeds(
            &embeds,
            [&positions.0, &positions.1, &positions.2],
            &mut cache,
        )?;
        let host = |x: &Tensor| x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
        // Every recurrent visit must own both its conv state and its SSM, every
        // full visit its K/V allocation. Verify live storage on the proof device.
        let kv = cache.static_kv_addresses()?;
        let recurrent = cache.recurrent_states();
        let mut addresses = std::collections::HashSet::new();
        let full = plan
            .visits()
            .filter(|i| !model.config().is_linear(*i))
            .count();
        assert_eq!(kv.len(), full);
        assert_eq!(
            recurrent.len(),
            core_llm::sensenova_loop::SenseNovaLoop::VISITS - full
        );
        for (k, v) in kv {
            assert!(addresses.insert(k));
            assert!(addresses.insert(v));
        }
        for (conv, state) in recurrent {
            for t in [conv, state] {
                assert!(addresses.insert(crate::primitives::storage_address(
                    t.ok_or("missing recurrent state")?
                )?));
            }
        }
        let first = host(&logits)?;
        let mut rows = vec![(first.clone(), first)];
        for &token in forced {
            let offset = cache.offset() + cache.rope_delta();
            let token = input(&[token])?;
            logits = model.decode_logits(&token, &mut cache, offset)?;
            embeds = Tensor::cat(&[&embeds, &model.embed_input_ids(&token)?], 1)?;
            positions.0.push(offset);
            positions.1.push(offset);
            positions.2.push(offset);
            let cold = model.decode_logits_from_embeds(
                &embeds,
                [&positions.0, &positions.1, &positions.2],
                &mut model.new_cache(),
            )?;
            rows.push((host(&logits)?, host(&cold)?));
        }
        Ok((prompt_ids, rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensenova_proof_uses_cuda_compatible_indices_for_prompt_and_forced_token() {
        // Only build indices. Their dtype is independent of which device owns
        // the storage and CUDA's index_select explicitly rejects I32 storage.
        let vocab_size = 248_320;
        for ids in [&[248_045, 17, 248_046][..], &[248_319][..]] {
            let indices = proof_token_indices(ids, vocab_size, &Device::Cpu).unwrap();
            assert_eq!(indices.dtype(), DType::U32);
            assert_eq!(indices.dims(), &[1, ids.len()]);
            assert_eq!(
                indices.flatten_all().unwrap().to_vec1::<u32>().unwrap(),
                ids.iter().map(|&id| id as u32).collect::<Vec<_>>()
            );
        }
        for id in [-1, vocab_size] {
            assert!(proof_token_indices(&[id], vocab_size, &Device::Cpu).is_err());
        }
    }
}
