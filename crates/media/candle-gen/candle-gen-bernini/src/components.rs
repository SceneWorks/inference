//! Shared resident renderer components for the full Bernini and renderer-only providers.

use std::collections::HashMap;
use std::path::Path;

use crate::ProviderVae;
use candle_gen::candle_core::{DType, Device};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{AdapterSpec, MoeExpert};
use candle_gen::{CandleError, Result as CResult};
use candle_gen_wan::config::{TextEncoderConfig, TransformerConfig, Vae16Config};
use candle_gen_wan::text_encoder::Umt5Encoder;
use candle_gen_wan::transformer::WanTransformer;

const COMPONENT_PLAN: [(&str, DType); 4] = [
    ("text_encoder", DType::F32),
    ("transformer", DType::BF16),
    ("transformer_2", DType::BF16),
    ("vae", DType::F32),
];

/// The heavy resident renderer state. Each provider owns its own lazy `Arc` cache of this state.
pub(crate) struct RendererComponents {
    pub(crate) te: Umt5Encoder,
    pub(crate) high: WanTransformer,
    pub(crate) low: WanTransformer,
    pub(crate) vae: ProviderVae,
    pub(crate) tok: TextTokenizer,
}

impl RendererComponents {
    /// Load one staging `VarBuilder` at a time, preserving the A14B resident-load order.
    pub(crate) fn load(
        root: &Path,
        device: &Device,
        provider_id: &str,
        adapters: &[AdapterSpec],
    ) -> CResult<Self> {
        let component = |index: usize| {
            let (subdir, dtype) = COMPONENT_PLAN[index];
            candle_gen::component_vb(root, subdir, dtype, device, provider_id)
        };
        let te = Umt5Encoder::new(&TextEncoderConfig::umt5_xxl(), component(0)?)?;
        let dit_cfg = TransformerConfig::t2v_14b();
        let (high, high_applied) = load_expert(
            root,
            device,
            provider_id,
            "transformer",
            &dit_cfg,
            adapters,
            MoeExpert::High,
        )?;
        let (low, low_applied) = load_expert(
            root,
            device,
            provider_id,
            "transformer_2",
            &dit_cfg,
            adapters,
            MoeExpert::Low,
        )?;
        if !adapters.is_empty()
            && high_applied.is_some()
            && low_applied.is_some()
            && high_applied.unwrap_or_default() + low_applied.unwrap_or_default() == 0
        {
            return Err(CandleError::Msg(format!(
                "{provider_id}: selected adapter stack matched no projection on either Bernini expert"
            )));
        }
        let vae = ProviderVae::new_with_encoder(&Vae16Config::wan21(), component(3)?)?;
        let tok =
            TextTokenizer::from_file(root.join("tokenizer/tokenizer.json"), tokenizer_config())
                .map_err(|e| CandleError::Msg(tokenizer_error(provider_id, &e.to_string())))?;
        Ok(Self {
            te,
            high,
            low,
            vae,
            tok,
        })
    }
}

fn load_expert(
    root: &Path,
    device: &Device,
    provider_id: &str,
    subdir: &str,
    config: &TransformerConfig,
    adapters: &[AdapterSpec],
    expert: MoeExpert,
) -> CResult<(WanTransformer, Option<usize>)> {
    let specs: Vec<AdapterSpec> = adapters
        .iter()
        .filter(|spec| spec.moe_expert.is_none_or(|target| target == expert))
        .cloned()
        .collect();
    let vb = candle_gen::component_vb(root, subdir, DType::BF16, device, provider_id)?;
    if vb.contains_tensor("proj_out.scales") {
        let mut dit = WanTransformer::new(config, vb)?;
        let applied = if specs.is_empty() {
            0
        } else {
            candle_gen_wan::adapters::install_additive(&mut dit, &specs, expert)?.applied
        };
        return Ok((dit, Some(applied)));
    }
    if specs.is_empty() {
        return Ok((WanTransformer::new(config, vb)?, None));
    }
    drop(vb);
    let mut map = HashMap::new();
    for file in candle_gen::sorted_safetensors(&root.join(subdir), provider_id)? {
        map.extend(candle_gen::candle_core::safetensors::load(
            &file,
            &Device::Cpu,
        )?);
    }
    candle_gen_wan::adapters::merge_adapters(&mut map, &specs)?;
    let vb = VarBuilder::from_tensors(map, DType::BF16, device);
    Ok((WanTransformer::new(config, vb)?, None))
}

fn tokenizer_config() -> TokenizerConfig {
    let te_cfg = TextEncoderConfig::umt5_xxl();
    TokenizerConfig {
        max_length: te_cfg.max_length,
        pad_token_id: te_cfg.pad_token_id,
        chat_template: ChatTemplate::None,
        pad_to_max_length: false,
    }
}

fn tokenizer_error(provider_id: &str, detail: &str) -> String {
    format!("{provider_id}: load tokenizer: {detail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_component_surface_is_exact() {
        assert_eq!(
            COMPONENT_PLAN,
            [
                ("text_encoder", DType::F32),
                ("transformer", DType::BF16),
                ("transformer_2", DType::BF16),
                ("vae", DType::F32),
            ]
        );
        let expected = TextEncoderConfig::umt5_xxl();
        let tokenizer = tokenizer_config();
        assert_eq!(tokenizer.max_length, expected.max_length);
        assert_eq!(tokenizer.pad_token_id, expected.pad_token_id);
        assert!(matches!(tokenizer.chat_template, ChatTemplate::None));
        assert!(!tokenizer.pad_to_max_length);
        assert_eq!(
            tokenizer_error("bernini", "fixture"),
            "bernini: load tokenizer: fixture"
        );
        assert_eq!(
            tokenizer_error("bernini_renderer", "fixture"),
            "bernini_renderer: load tokenizer: fixture"
        );
    }
}
