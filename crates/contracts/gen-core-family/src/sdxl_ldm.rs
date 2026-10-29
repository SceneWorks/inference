//! Pure SDXL LDM/A1111 single-file key conversion shared by MLX and candle.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdxlComponent {
    Unet,
    ClipL,
    ClipBigG,
    Vae,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorRemap {
    Rename(SdxlComponent, String),
    SplitQkv(SdxlComponent, [String; 3]),
    Transpose(SdxlComponent, String),
    Squeeze(SdxlComponent, String),
}

fn index_after(key: &str, prefix: &str) -> Option<usize> {
    key.strip_prefix(prefix)?.split('.').next()?.parse().ok()
}

fn resnet(mut key: String) -> String {
    for (from, to) in [
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"),
        ("skip_connection", "conv_shortcut"),
    ] {
        key = key.replace(from, to);
    }
    key
}

fn unet(key: &str) -> Option<String> {
    for (from, to) in [
        ("time_embed.0.", "time_embedding.linear_1."),
        ("time_embed.2.", "time_embedding.linear_2."),
        ("input_blocks.0.0.", "conv_in."),
        ("out.0.", "conv_norm_out."),
        ("out.2.", "conv_out."),
        ("label_emb.0.0.", "add_embedding.linear_1."),
        ("label_emb.0.2.", "add_embedding.linear_2."),
    ] {
        if let Some(rest) = key.strip_prefix(from) {
            return Some(format!("{to}{rest}"));
        }
    }
    if let Some(i) = index_after(key, "input_blocks.") {
        if i == 0 {
            return None;
        }
        let block = (i - 1) / 3;
        let layer = (i - 1) % 3;
        let prefix = format!("input_blocks.{i}.");
        if let Some(rest) = key.strip_prefix(&(prefix.clone() + "0.op.")) {
            return Some(format!("down_blocks.{block}.downsamplers.0.conv.{rest}"));
        }
        if let Some(rest) = key.strip_prefix(&(prefix.clone() + "0.")) {
            return Some(resnet(format!(
                "down_blocks.{block}.resnets.{layer}.{rest}"
            )));
        }
        if let Some(rest) = key.strip_prefix(&(prefix + "1.")) {
            return Some(format!("down_blocks.{block}.attentions.{layer}.{rest}"));
        }
    }
    if let Some(i) = index_after(key, "middle_block.") {
        let rest = key.strip_prefix(&format!("middle_block.{i}."))?;
        return Some(if i % 2 == 0 {
            resnet(format!("mid_block.resnets.{}.{rest}", i.saturating_sub(1)))
        } else {
            format!("mid_block.attentions.{}.{rest}", i - 1)
        });
    }
    if let Some(i) = index_after(key, "output_blocks.") {
        let block = i / 3;
        let layer = i % 3;
        let prefix = format!("output_blocks.{i}.");
        if let Some(rest) = key.strip_prefix(&(prefix.clone() + "0.")) {
            return Some(resnet(format!("up_blocks.{block}.resnets.{layer}.{rest}")));
        }
        for slot in [1, 2] {
            if let Some(rest) = key.strip_prefix(&format!("{prefix}{slot}.conv.")) {
                return Some(format!("up_blocks.{block}.upsamplers.0.conv.{rest}"));
            }
        }
        if let Some(rest) = key.strip_prefix(&(prefix + "1.")) {
            return Some(format!("up_blocks.{block}.attentions.{layer}.{rest}"));
        }
    }
    None
}

fn vae_resnet(key: String) -> String {
    key.replace("nin_shortcut", "conv_shortcut")
}

fn vae_attn(mut key: String) -> String {
    for (from, to) in [
        (".norm.", ".group_norm."),
        (".q.", ".to_q."),
        (".k.", ".to_k."),
        (".v.", ".to_v."),
        (".proj_out.", ".to_out.0."),
    ] {
        key = key.replace(from, to);
    }
    key
}

fn vae(key: &str) -> Option<String> {
    for (from, to) in [
        ("encoder.conv_in.", "encoder.conv_in."),
        ("encoder.conv_out.", "encoder.conv_out."),
        ("encoder.norm_out.", "encoder.conv_norm_out."),
        ("decoder.conv_in.", "decoder.conv_in."),
        ("decoder.conv_out.", "decoder.conv_out."),
        ("decoder.norm_out.", "decoder.conv_norm_out."),
        ("quant_conv.", "quant_conv."),
        ("post_quant_conv.", "post_quant_conv."),
    ] {
        if let Some(rest) = key.strip_prefix(from) {
            return Some(format!("{to}{rest}"));
        }
    }
    for side in ["encoder", "decoder"] {
        for block in [1, 2] {
            if let Some(rest) = key.strip_prefix(&format!("{side}.mid.block_{block}.")) {
                return Some(vae_resnet(format!(
                    "{side}.mid_block.resnets.{}.{rest}",
                    block - 1
                )));
            }
        }
        if let Some(rest) = key.strip_prefix(&format!("{side}.mid.attn_1.")) {
            return Some(vae_attn(format!("{side}.mid_block.attentions.0.{rest}")));
        }
    }
    if let Some(i) = index_after(key, "encoder.down.") {
        let prefix = format!("encoder.down.{i}.");
        if let Some(rest) = key.strip_prefix(&(prefix.clone() + "downsample.conv.")) {
            return Some(format!(
                "encoder.down_blocks.{i}.downsamplers.0.conv.{rest}"
            ));
        }
        if let Some(rest) = key.strip_prefix(&(prefix + "block.")) {
            return Some(vae_resnet(format!(
                "encoder.down_blocks.{i}.resnets.{rest}"
            )));
        }
    }
    if let Some(source) = index_after(key, "decoder.up.") {
        let target = 3usize.checked_sub(source)?;
        let prefix = format!("decoder.up.{source}.");
        if let Some(rest) = key.strip_prefix(&(prefix.clone() + "upsample.conv.")) {
            return Some(format!(
                "decoder.up_blocks.{target}.upsamplers.0.conv.{rest}"
            ));
        }
        if let Some(rest) = key.strip_prefix(&(prefix + "block.")) {
            return Some(vae_resnet(format!(
                "decoder.up_blocks.{target}.resnets.{rest}"
            )));
        }
    }
    None
}

fn openclip(key: &str) -> Option<String> {
    for (from, to) in [
        (
            "positional_embedding",
            "text_model.embeddings.position_embedding.weight",
        ),
        (
            "token_embedding.weight",
            "text_model.embeddings.token_embedding.weight",
        ),
        ("ln_final.weight", "text_model.final_layer_norm.weight"),
        ("ln_final.bias", "text_model.final_layer_norm.bias"),
    ] {
        if key == from {
            return Some(to.into());
        }
    }
    let rest = key.strip_prefix("transformer.")?;
    Some(
        rest.replace("resblocks.", "text_model.encoder.layers.")
            .replace("ln_1", "layer_norm1")
            .replace("ln_2", "layer_norm2")
            .replace(".mlp.c_fc.", ".mlp.fc1.")
            .replace(".mlp.c_proj.", ".mlp.fc2.")
            .replace(".attn", ".self_attn"),
    )
}

/// Convert one fused checkpoint key. Metadata, EMA, and unsupported keys return `None`.
pub fn remap_sdxl_ldm_key(key: &str) -> Option<TensorRemap> {
    if let Some(inner) = key.strip_prefix("model.diffusion_model.") {
        return unet(inner).map(|k| TensorRemap::Rename(SdxlComponent::Unet, k));
    }
    if let Some(inner) = key.strip_prefix("conditioner.embedders.0.transformer.") {
        return Some(TensorRemap::Rename(SdxlComponent::ClipL, inner.into()));
    }
    if let Some(inner) = key.strip_prefix("conditioner.embedders.1.model.") {
        if inner == "text_projection" {
            return Some(TensorRemap::Transpose(
                SdxlComponent::ClipBigG,
                "text_projection.weight".into(),
            ));
        }
        for suffix in [".in_proj_weight", ".in_proj_bias"] {
            if let Some(base) = inner.strip_suffix(suffix) {
                let base = openclip(base)?;
                let leaf = if suffix.ends_with("weight") {
                    "weight"
                } else {
                    "bias"
                };
                return Some(TensorRemap::SplitQkv(
                    SdxlComponent::ClipBigG,
                    [
                        format!("{base}.q_proj.{leaf}"),
                        format!("{base}.k_proj.{leaf}"),
                        format!("{base}.v_proj.{leaf}"),
                    ],
                ));
            }
        }
        return openclip(inner).map(|k| TensorRemap::Rename(SdxlComponent::ClipBigG, k));
    }
    if let Some(inner) = key.strip_prefix("first_stage_model.") {
        return vae(inner).map(|k| {
            if k.contains(".attentions.") && k.ends_with(".weight") {
                TensorRemap::Squeeze(SdxlComponent::Vae, k)
            } else {
                TensorRemap::Rename(SdxlComponent::Vae, k)
            }
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_non_default_unet_and_vae_blocks() {
        assert_eq!(
            remap_sdxl_ldm_key("model.diffusion_model.input_blocks.7.0.out_layers.3.weight"),
            Some(TensorRemap::Rename(
                SdxlComponent::Unet,
                "down_blocks.2.resnets.0.conv2.weight".into()
            ))
        );
        assert_eq!(
            remap_sdxl_ldm_key("first_stage_model.decoder.up.1.block.2.nin_shortcut.weight"),
            Some(TensorRemap::Rename(
                SdxlComponent::Vae,
                "decoder.up_blocks.2.resnets.2.conv_shortcut.weight".into()
            ))
        );
    }

    #[test]
    fn maps_openclip_qkv_split() {
        assert_eq!(
            remap_sdxl_ldm_key(
                "conditioner.embedders.1.model.transformer.resblocks.9.attn.in_proj_weight"
            ),
            Some(TensorRemap::SplitQkv(
                SdxlComponent::ClipBigG,
                [
                    "text_model.encoder.layers.9.self_attn.q_proj.weight".into(),
                    "text_model.encoder.layers.9.self_attn.k_proj.weight".into(),
                    "text_model.encoder.layers.9.self_attn.v_proj.weight".into(),
                ]
            ))
        );
    }
}
