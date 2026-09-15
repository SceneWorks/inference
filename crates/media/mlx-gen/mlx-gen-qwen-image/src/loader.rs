//! Real-checkpoint loading for Qwen-Image: assemble the tokenizer, Qwen2.5-VL text encoder,
//! 60-layer MMDiT transformer, and causal-Conv3d VAE from a `Qwen/Qwen-Image` snapshot directory,
//! applying the diffusers-checkpoint → internal-name remaps (the fork's `qwen_weight_mapping.py`).
//!
//! Snapshot layout (standard diffusers multi-component tree):
//! ```text
//!   <root>/tokenizer/{tokenizer.json | vocab.json + merges.txt}
//!   <root>/text_encoder/*.safetensors
//!   <root>/transformer/*.safetensors
//!   <root>/vae/*.safetensors
//! ```
//! The transformer/VAE checkpoints are keyed by the diffusers tree; we remap to the *internal*
//! names the slice-1/2/3 modules expect (the same `to_pattern` the parity goldens were dumped
//! under). The text-encoder layout (`model.*`) maps directly onto the encoder under the `"model"`
//! prefix, so it needs no remap.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Result, WeightsSource};
use mlx_rs::Array;

use crate::control_transformer::{QwenFunControlBranch, QwenFunControlConfig};
use crate::text_encoder::vision::{VisionConfig, VisionTransformer};
use crate::text_encoder::{QwenTextEncoder, QwenTextEncoderConfig, QwenVisionLanguageEncoder};
use crate::transformer::{QwenTransformer, QwenTransformerConfig};
use crate::vae::QwenVae;

/// Qwen2 pad token id (`<|endoftext|>`).
const PAD_TOKEN_ID: i32 = 151643;
/// The fork's `LanguageTokenizer` max_length for the `qwen` tokenizer.
const MAX_LENGTH: usize = 1058;

/// Load the Qwen2 tokenizer with the Qwen-Image T2I template + padding policy (`padding="longest"`
/// → no max-length padding for a single prompt). The snapshot must contain `tokenizer/tokenizer.json`
/// (the HF *fast* serialization); the upstream repo ships only `vocab.json` + `merges.txt`, so run
/// `tools/build_qwen_tokenizer.py` once to materialize it (the same fast tokenizer the fork builds
/// at runtime).
pub fn load_tokenizer(root: &Path) -> Result<TextTokenizer> {
    let source = crate::ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    load_validated_tokenizer(&source)
}

pub(crate) fn load_validated_tokenizer(
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
) -> Result<TextTokenizer> {
    source.read_tokenizer_unchanged(|path| {
        TextTokenizer::from_file(
            path,
            TokenizerConfig {
                max_length: MAX_LENGTH,
                pad_token_id: PAD_TOKEN_ID,
                chat_template: ChatTemplate::QwenImage,
                pad_to_max_length: false,
            },
        )
        .map_err(Into::into)
    })
}

/// Load the Qwen2.5-VL text encoder (text path). The on-disk `model.*` keys map directly onto the
/// encoder tree under the `"model"` prefix (validated in slice 2) — no remap needed.
pub fn load_text_encoder(root: &Path) -> Result<QwenTextEncoder> {
    let selected = crate::ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    selected.read_unchanged(load_text_encoder_from_source)
}

pub(crate) fn load_text_encoder_from_source(source: &WeightsSource) -> Result<QwenTextEncoder> {
    let w = weights_from_source(source)?;
    let encoder = QwenTextEncoder::from_weights(&w, "model", &QwenTextEncoderConfig::qwen_image())?;
    w.materialize_accessed()?;
    Ok(encoder)
}

/// Load the Qwen2.5-VL **vision transformer** (Qwen-Image-Edit) from a snapshot's `text_encoder/`
/// shards. The vision weights live under `visual.*` alongside the LM; we apply the fork's vision
/// rules ([`remap_vision_keys`]) then read under the `"visual"` prefix. Edit-only — the T2I snapshot
/// has no `visual.*` weights.
pub fn load_vision_encoder(root: &Path) -> Result<VisionTransformer> {
    let selected = crate::ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    selected.validate_vision(&crate::VISION_ENCODER_CONTRACT, &crate::ENCODER_CONTRACT)?;
    selected.read_unchanged(load_vision_encoder_from_source)
}

pub(crate) fn load_vision_encoder_from_source(source: &WeightsSource) -> Result<VisionTransformer> {
    let mut w = weights_from_source(source)?;
    remap_vision_keys(&mut w)?;
    let encoder = VisionTransformer::from_weights(&w, "visual", &VisionConfig::qwen_image_edit())?;
    w.materialize_accessed()?;
    Ok(encoder)
}

/// Load the Qwen-Image-**Edit** vision-language conditioning encoder: the Qwen2.5-VL LM (`model.*`,
/// same layout as T2I) + the vision transformer (`visual.*`), composed into a
/// [`QwenVisionLanguageEncoder`]. Edit-only.
///
/// The LM (`model.*`) and the vision transformer (`visual.*`) both live in the same `text_encoder/`
/// shard set, so the ~16 GB is parsed ONCE and reused for both trees (F-080) — previously
/// `load_text_encoder` + `load_vision_encoder` each ran their own `Weights::from_dir`, reading every
/// shard twice. `remap_vision_keys` only touches `visual.*`, so it is safe to apply before the LM read.
pub fn load_vision_language_encoder(root: &Path) -> Result<QwenVisionLanguageEncoder> {
    let selected =
        crate::ENCODER_CONTRACT.validate_source(&WeightsSource::Dir(root.join("text_encoder")))?;
    selected.validate_vision(&crate::VISION_ENCODER_CONTRACT, &crate::ENCODER_CONTRACT)?;
    selected.read_unchanged(load_vision_language_encoder_from_source)
}

pub(crate) fn load_vision_language_encoder_from_source(
    source: &WeightsSource,
) -> Result<QwenVisionLanguageEncoder> {
    load_vision_language_encoder_from_sources(source, source)
}

/// Compose the edit encoder from a selectable language tower and the checkpoint-coupled vision
/// tower. When both halves share one source, preserve the established one-parse fast path.
pub(crate) fn load_vision_language_encoder_from_sources(
    language_source: &WeightsSource,
    vision_source: &WeightsSource,
) -> Result<QwenVisionLanguageEncoder> {
    if same_source(language_source, vision_source) {
        let mut w = weights_from_source(language_source)?;
        remap_vision_keys(&mut w)?;
        let lm = QwenTextEncoder::from_weights(&w, "model", &QwenTextEncoderConfig::qwen_image())?;
        let visual =
            VisionTransformer::from_weights(&w, "visual", &VisionConfig::qwen_image_edit())?;
        w.materialize_accessed()?;
        return Ok(QwenVisionLanguageEncoder::new(lm, visual));
    }

    let language_weights = weights_from_source(language_source)?;
    let lm = QwenTextEncoder::from_weights(
        &language_weights,
        "model",
        &QwenTextEncoderConfig::qwen_image(),
    )?;
    language_weights.materialize_accessed()?;
    let mut vision_weights = weights_from_source(vision_source)?;
    remap_vision_keys(&mut vision_weights)?;
    let visual = VisionTransformer::from_weights(
        &vision_weights,
        "visual",
        &VisionConfig::qwen_image_edit(),
    )?;
    vision_weights.materialize_accessed()?;
    Ok(QwenVisionLanguageEncoder::new(lm, visual))
}

fn same_source(left: &WeightsSource, right: &WeightsSource) -> bool {
    match (left, right) {
        (WeightsSource::Dir(left), WeightsSource::Dir(right))
        | (WeightsSource::File(left), WeightsSource::File(right)) => left == right,
        _ => false,
    }
}

fn weights_from_source(source: &WeightsSource) -> Result<Weights> {
    match source {
        WeightsSource::Dir(path) => Weights::from_dir(path),
        WeightsSource::File(path) => Weights::from_file(path),
    }
}

/// The fork's vision weight transforms (`qwen_weight_mapping.py`), applied in place: transpose the
/// patch-embed conv (PyTorch `[O,I,kD,kH,kW]` → MLX `[O,kD,kH,kW,I]`) and rename the merger
/// `Sequential` `mlp.{0,2}` → `mlp_{0,1}`. Everything else under `visual.*` matches 1:1.
pub fn remap_vision_keys(w: &mut Weights) -> Result<()> {
    const PATCH_EMBED: &str = "visual.patch_embed.proj.weight";
    if let Some(t) = w.get(PATCH_EMBED).cloned() {
        if t.shape().len() == 5 {
            w.insert(PATCH_EMBED, t.transpose_axes(&[0, 2, 3, 4, 1])?);
        }
    }
    for (from, to) in [
        ("visual.merger.mlp.0.weight", "visual.merger.mlp_0.weight"),
        ("visual.merger.mlp.0.bias", "visual.merger.mlp_0.bias"),
        ("visual.merger.mlp.2.weight", "visual.merger.mlp_1.weight"),
        ("visual.merger.mlp.2.bias", "visual.merger.mlp_1.bias"),
    ] {
        w.alias(from, to);
    }
    Ok(())
}

/// Load the 60-layer MMDiT transformer, applying the diffusers→internal key renames.
pub fn load_transformer(root: &Path) -> Result<QwenTransformer> {
    let mut w = Weights::from_dir(root.join("transformer"))?;
    remap_transformer_keys(&mut w);
    QwenTransformer::from_weights(&w, "", &QwenTransformerConfig::qwen_image())
}

/// Load the transformer for Qwen-Image-**Edit-2511** — identical to [`load_transformer`] but with
/// `zero_cond_t` on (the conditioning-image latent tokens are modulated as clean / timestep 0).
pub fn load_transformer_edit(root: &Path) -> Result<QwenTransformer> {
    let mut w = Weights::from_dir(root.join("transformer"))?;
    remap_transformer_keys(&mut w);
    QwenTransformer::from_weights(&w, "", &QwenTransformerConfig::qwen_image_edit())
}

/// Load the alibaba-pai `Qwen-Image-2512-Fun-Controlnet-Union` VACE control branch (sc-8267 — this
/// **replaces** the retired InstantX `Qwen-Image-ControlNet-Union`). The checkpoint is a single
/// `Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors` (`File`) or a dir of shards (`Dir`). Its
/// control-block keys (`control_blocks.{i}.attn.to_out.0`, `…img_mod.1`, `…img_mlp.net.0.proj`, …)
/// are the same diffusers block names as the base, so we apply the same [`remap_transformer_keys`];
/// the control top-level modules (`control_img_in`, `control_blocks.{i}.{before,after}_proj`) match
/// 1:1 and pass through unchanged.
pub fn load_controlnet(control: &WeightsSource) -> Result<QwenFunControlBranch> {
    let mut w = match control {
        WeightsSource::File(p) => Weights::from_file(p)?,
        WeightsSource::Dir(p) => Weights::from_dir(p)?,
    };
    remap_transformer_keys(&mut w);
    QwenFunControlBranch::from_weights(&w, "", &QwenFunControlConfig::qwen_image_2512_fun())
}

/// Load the causal-Conv3d VAE, applying the diffusers→internal key remap (structural renames +
/// conv-weight transposes + RMSNorm `gamma`→1-D).
pub fn load_vae(root: &Path) -> Result<QwenVae> {
    let mut w = Weights::from_dir(root.join("vae"))?;
    remap_vae_keys(&mut w)?;
    QwenVae::from_weights(&w)
}

/// diffusers transformer checkpoint → internal names (port of `QwenWeightMapping`'s transformer
/// rules). All weights are plain Linears (no transpose); only a handful of modules are renamed —
/// `to_out.0`→`attn_to_out.0`, `{img,txt}_mod.1`→`{img,txt}_mod_linear`, the
/// `{img,txt}_mlp.net.{0.proj,2}` feed-forwards → `{img,txt}_ff.mlp_{in,out}`. Everything else
/// matches 1:1 and is left in place. Applied across all 60 blocks by substring match.
const TRANSFORMER_RENAMES: &[(&str, &str)] = &[
    (".attn.to_out.0.", ".attn.attn_to_out.0."),
    (".img_mod.1.", ".img_mod_linear."),
    (".txt_mod.1.", ".txt_mod_linear."),
    (".img_mlp.net.0.proj.", ".img_ff.mlp_in."),
    (".img_mlp.net.2.", ".img_ff.mlp_out."),
    (".txt_mlp.net.0.proj.", ".txt_ff.mlp_in."),
    (".txt_mlp.net.2.", ".txt_ff.mlp_out."),
];

pub fn remap_transformer_keys(w: &mut Weights) {
    let keys: Vec<String> = w.keys().map(String::from).collect();
    for k in keys {
        for (from, to) in TRANSFORMER_RENAMES {
            if k.contains(from) {
                w.alias(&k, &k.replace(from, to));
                break;
            }
        }
    }
}

/// Remove the on-disk side of aliases for one already-built block. `remove_accessed` drains the
/// internal names read by the constructor, but `Weights::alias` deliberately retains the original
/// diffusers key. A streamed view must drop those twin handles too or every window retains seven
/// Linears' weights after the block has finished. An omitted constructor read remains observable:
/// its unaccessed internal alias is deliberately not removed here.
pub(crate) fn remove_transformer_source_aliases(w: &mut Weights, block_prefix: &str) {
    let keys: Vec<String> = w
        .keys()
        .filter(|key| {
            key.starts_with(block_prefix)
                && TRANSFORMER_RENAMES
                    .iter()
                    .any(|(source, _)| key.contains(source))
        })
        .map(String::from)
        .collect();
    for key in keys {
        w.remove(&key);
    }
}

/// diffusers VAE checkpoint → internal names (port of `QwenWeightMapping`'s VAE rules). Renames the
/// structure (decoder `up_blocks.{b}`→`up_block{b}`; the encoder's *flat* `down_blocks.{0..10}`→the
/// grouped `down_blocks.{g}.resnets.{r}` / `downsamplers.0` tree), inserts `.conv3d` for the
/// `CausalConv3d` modules, renames `conv_shortcut`→`skip_conv` and `resample.1`→`resample_conv`,
/// and applies the conv-weight transposes (`[O,I,D,H,W]`→`[O,D,H,W,I]`, `[O,I,H,W]`→`[O,H,W,I]`)
/// + RMSNorm `gamma`→1-D. The unused temporal `time_conv` is skipped (the fork never calls it).
pub fn remap_vae_keys(w: &mut Weights) -> Result<()> {
    let keys: Vec<String> = w.keys().map(String::from).collect();
    for k in &keys {
        let Some(target) = vae_internal_key(k) else {
            continue; // skipped (time_conv)
        };
        let t = w.require(k)?;
        let t = transform_vae_tensor(k, t)?;
        w.insert(target, t);
    }
    Ok(())
}

/// Map one on-disk VAE key to its internal name, or `None` to skip it.
fn vae_internal_key(k: &str) -> Option<String> {
    if k.contains(".time_conv.") {
        return None; // unused temporal conv (T2I up/down-sampling is purely spatial)
    }
    // Encoder: flat `down_blocks.{flat}` → grouped `down_blocks.{g}.resnets.{r}` / `downsamplers.0`.
    if let Some(rest) = k.strip_prefix("encoder.down_blocks.") {
        let (flat_str, tail) = rest.split_once('.')?;
        let flat: usize = flat_str.parse().ok()?;
        let (group, slot) = (flat / 3, flat % 3);
        if slot == 2 {
            let leaf = tail.strip_prefix("resample.1.")?;
            return Some(format!(
                "encoder.down_blocks.{group}.downsamplers.0.resample_conv.{leaf}"
            ));
        }
        return Some(format!(
            "encoder.down_blocks.{group}.resnets.{slot}.{}",
            remap_resnet_tail(tail)
        ));
    }
    // Decoder: `up_blocks.{b}` → `up_block{b}`.
    if let Some(rest) = k.strip_prefix("decoder.up_blocks.") {
        let (b, tail) = rest.split_once('.')?;
        if let Some(up) = tail.strip_prefix("upsamplers.0.") {
            let leaf = up.strip_prefix("resample.1.")?;
            return Some(format!(
                "decoder.up_block{b}.upsamplers.0.resample_conv.{leaf}"
            ));
        }
        let after = tail.strip_prefix("resnets.")?;
        let (r, rtail) = after.split_once('.')?;
        return Some(format!(
            "decoder.up_block{b}.resnets.{r}.{}",
            remap_resnet_tail(rtail)
        ));
    }
    Some(remap_generic_vae(k))
}

/// Leaf rename for a resnet sub-tree tail (`conv1.weight`, `norm1.gamma`, `conv_shortcut.bias`, …).
fn remap_resnet_tail(tail: &str) -> String {
    if let Some(p) = tail.strip_suffix(".gamma") {
        return format!("{p}.weight");
    }
    let Some((parent, leaf)) = tail.rsplit_once('.') else {
        return tail.to_string();
    };
    match parent {
        "conv1" | "conv2" => format!("{parent}.conv3d.{leaf}"),
        "conv_shortcut" => format!("skip_conv.conv3d.{leaf}"),
        _ => tail.to_string(),
    }
}

/// Leaf rename for the regular (non-down/up-block) VAE keys: `gamma`→`weight`, and `.conv3d` insert
/// for the `CausalConv3d` modules (attention `to_qkv`/`proj` stay flat — they're 2-D convs).
fn remap_generic_vae(k: &str) -> String {
    if let Some(p) = k.strip_suffix(".gamma") {
        return format!("{p}.weight");
    }
    let Some((parent, leaf)) = k.rsplit_once('.') else {
        return k.to_string();
    };
    let parent_name = parent.rsplit('.').next().unwrap_or(parent);
    if matches!(
        parent_name,
        "conv_in" | "conv_out" | "conv1" | "conv2" | "quant_conv" | "post_quant_conv"
    ) {
        return format!("{parent}.conv3d.{leaf}");
    }
    k.to_string()
}

/// Apply the fork's weight transform for a VAE tensor, keyed off the leaf + rank (mirrors
/// `WeightTransforms`): `gamma`→1-D, rank-5 conv weight `[O,I,D,H,W]`→`[O,D,H,W,I]`, rank-4 conv
/// weight `[O,I,H,W]`→`[O,H,W,I]`, biases unchanged.
fn transform_vae_tensor(src_key: &str, t: &Array) -> Result<Array> {
    if src_key.ends_with(".gamma") {
        return Ok(t.reshape(&[t.shape()[0]])?);
    }
    if src_key.ends_with(".weight") {
        return Ok(match t.shape().len() {
            5 => t.transpose_axes(&[0, 2, 3, 4, 1])?,
            4 => t.transpose_axes(&[0, 2, 3, 1])?,
            _ => t.clone(),
        });
    }
    Ok(t.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounded_multimodal_fixture() -> (
        tempfile::TempDir,
        std::path::PathBuf,
        mlx_gen::gen_core::EncoderContract,
    ) {
        let fixture = tempfile::tempdir().unwrap();
        let component = fixture.path().join("text_encoder");
        let language = crate::bounded_encoder_contract();
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &component,
            language,
            crate::bounded_vision_encoder_contract(),
        )
        .unwrap();
        (fixture, component, language)
    }

    #[test]
    fn qwen_edit_vision_contract_rejects_missing_visual_headers() {
        let fixture = tempfile::tempdir().unwrap();
        let component = fixture.path().join("text_encoder");
        let language = crate::bounded_encoder_contract();
        gen_core_testkit::write_encoder_contract_fixture(&component, language).unwrap();
        let selected = language
            .validate_source(&WeightsSource::Dir(component))
            .unwrap();
        let error = selected
            .validate_vision(&crate::bounded_vision_encoder_contract(), &language)
            .unwrap_err()
            .to_string();
        assert!(error.contains("vision_config"), "{error}");
    }

    #[test]
    fn qwen2_5_vision_behavior_defaults_are_accepted_but_explicit_conflicts_fail() {
        let contract = crate::VISION_ENCODER_CONTRACT;
        contract
            .validate_definition(&crate::ENCODER_CONTRACT)
            .unwrap();
        assert_eq!(
            contract.architecture,
            mlx_gen::gen_core::VisionEncoderArchitecture::Qwen2_5Vl
        );
        assert_eq!(contract.rope_theta.get(), 10_000.0);
        assert_eq!(contract.normalization_eps.get(), 1e-6);
        assert_eq!(contract.hidden_size, 1280);
        assert_eq!(contract.intermediate_size, 3420);
        assert_eq!(contract.num_hidden_layers, 32);
        assert_eq!(contract.num_attention_heads, 16);
        assert_eq!(contract.output_width, 3584);
        assert_eq!(contract.patch_size, 14);
        assert_eq!(contract.temporal_patch_size, 2);
        assert_eq!(contract.spatial_merge_size, 2);
        assert_eq!(contract.in_channels, 3);
        assert_eq!(contract.window_size, Some(112));
        assert_eq!(contract.full_attention_block_indexes, &[7, 15, 23, 31]);

        let (_fixture, component, language) = bounded_multimodal_fixture();
        let config_path = component.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["vision_config"]
            .as_object_mut()
            .unwrap()
            .remove("rope_theta");
        config["vision_config"]
            .as_object_mut()
            .unwrap()
            .remove("rms_norm_eps");
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        language
            .validate_source(&WeightsSource::Dir(component))
            .unwrap()
            .validate_vision(&crate::bounded_vision_encoder_contract(), &language)
            .expect("omission must resolve to the exact Qwen2.5-VL runtime defaults");

        for (field, value) in [("rope_theta", 9_999.0), ("rms_norm_eps", 1e-5)] {
            let (_fixture, component, language) = bounded_multimodal_fixture();
            let config_path = component.join("config.json");
            let mut config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
            config["vision_config"][field] = serde_json::json!(value);
            std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
            let error = language
                .validate_source(&WeightsSource::Dir(component))
                .unwrap()
                .validate_vision(&crate::bounded_vision_encoder_contract(), &language)
                .unwrap_err()
                .to_string();
            assert!(error.contains(field), "{error}");
        }
    }

    /// F-120: exercise the PRODUCTION `remap_transformer_keys` over an in-memory `Weights` fixture
    /// (not a duplicated copy of the rename table), so a regression in the real table fails CI. One
    /// representative key from each rename family; the aliased name must be present after the remap.
    #[test]
    fn transformer_renames() {
        let cases = [
            (
                "transformer_blocks.7.attn.to_out.0.weight",
                "transformer_blocks.7.attn.attn_to_out.0.weight",
            ),
            (
                "transformer_blocks.12.img_mod.1.bias",
                "transformer_blocks.12.img_mod_linear.bias",
            ),
            (
                "transformer_blocks.5.txt_mod.1.weight",
                "transformer_blocks.5.txt_mod_linear.weight",
            ),
            (
                "transformer_blocks.3.img_mlp.net.0.proj.weight",
                "transformer_blocks.3.img_ff.mlp_in.weight",
            ),
            (
                "transformer_blocks.3.img_mlp.net.2.weight",
                "transformer_blocks.3.img_ff.mlp_out.weight",
            ),
            (
                "transformer_blocks.3.txt_mlp.net.0.proj.weight",
                "transformer_blocks.3.txt_ff.mlp_in.weight",
            ),
            (
                "transformer_blocks.3.txt_mlp.net.2.weight",
                "transformer_blocks.3.txt_ff.mlp_out.weight",
            ),
        ];

        let mut w = Weights::empty();
        for (from, _) in cases {
            w.insert(from, mlx_rs::Array::from_slice(&[0f32], &[1]));
        }
        remap_transformer_keys(&mut w);

        let keys: std::collections::HashSet<&str> = w.keys().collect();
        for (from, want) in cases {
            assert!(keys.contains(want), "remap must alias {from} → {want}");
        }
        // A key that matches no rename family is left untouched.
        let mut plain = Weights::empty();
        plain.insert(
            "transformer_blocks.0.attn.norm_q.weight",
            mlx_rs::Array::from_slice(&[0f32], &[1]),
        );
        remap_transformer_keys(&mut plain);
        assert_eq!(plain.keys().count(), 1, "unmatched key must not be aliased");
    }

    /// sc-8267: the 2512-Fun control checkpoint's `control_blocks.{i}` carry the SAME diffusers block
    /// keys as the base, so `remap_transformer_keys` must alias them (the control block reuses the
    /// base `QwenTransformerBlock::from_weights`), while the VACE top-level keys (`control_img_in`,
    /// `control_blocks.{i}.{before,after}_proj`) match the internal names 1:1 and pass through.
    #[test]
    fn fun_control_keys_remap() {
        let renamed = [
            (
                "control_blocks.3.attn.to_out.0.weight",
                "control_blocks.3.attn.attn_to_out.0.weight",
            ),
            (
                "control_blocks.0.img_mod.1.bias",
                "control_blocks.0.img_mod_linear.bias",
            ),
            (
                "control_blocks.2.img_mlp.net.0.proj.weight",
                "control_blocks.2.img_ff.mlp_in.weight",
            ),
        ];
        // These VACE control modules carry no diffusers→internal rename — they must survive verbatim.
        let passthrough = [
            "control_img_in.weight",
            "control_img_in.bias",
            "control_blocks.0.before_proj.weight",
            "control_blocks.4.after_proj.weight",
        ];

        let mut w = Weights::empty();
        for (from, _) in renamed {
            w.insert(from, mlx_rs::Array::from_slice(&[0f32], &[1]));
        }
        for k in passthrough {
            w.insert(k, mlx_rs::Array::from_slice(&[0f32], &[1]));
        }
        remap_transformer_keys(&mut w);

        let keys: std::collections::HashSet<&str> = w.keys().collect();
        for (from, want) in renamed {
            assert!(
                keys.contains(want),
                "control remap must alias {from} → {want}"
            );
        }
        for k in passthrough {
            assert!(
                keys.contains(k),
                "VACE control key {k} must pass through unchanged"
            );
        }
    }

    #[test]
    fn vae_encoder_flat_to_grouped() {
        assert_eq!(
            vae_internal_key("encoder.down_blocks.0.conv1.weight").unwrap(),
            "encoder.down_blocks.0.resnets.0.conv1.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("encoder.down_blocks.1.norm2.gamma").unwrap(),
            "encoder.down_blocks.0.resnets.1.norm2.weight"
        );
        assert_eq!(
            vae_internal_key("encoder.down_blocks.2.resample.1.bias").unwrap(),
            "encoder.down_blocks.0.downsamplers.0.resample_conv.bias"
        );
        assert_eq!(
            vae_internal_key("encoder.down_blocks.3.conv_shortcut.weight").unwrap(),
            "encoder.down_blocks.1.resnets.0.skip_conv.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("encoder.down_blocks.9.conv1.weight").unwrap(),
            "encoder.down_blocks.3.resnets.0.conv1.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("encoder.down_blocks.10.conv2.bias").unwrap(),
            "encoder.down_blocks.3.resnets.1.conv2.conv3d.bias"
        );
        assert!(vae_internal_key("encoder.down_blocks.8.time_conv.weight").is_none());
    }

    #[test]
    fn vae_decoder_and_generic() {
        assert_eq!(
            vae_internal_key("decoder.up_blocks.0.resnets.2.conv1.weight").unwrap(),
            "decoder.up_block0.resnets.2.conv1.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.up_blocks.1.resnets.0.conv_shortcut.weight").unwrap(),
            "decoder.up_block1.resnets.0.skip_conv.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.up_blocks.0.upsamplers.0.resample.1.weight").unwrap(),
            "decoder.up_block0.upsamplers.0.resample_conv.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.conv_in.weight").unwrap(),
            "decoder.conv_in.conv3d.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.norm_out.gamma").unwrap(),
            "decoder.norm_out.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.mid_block.attentions.0.to_qkv.weight").unwrap(),
            "decoder.mid_block.attentions.0.to_qkv.weight"
        );
        assert_eq!(
            vae_internal_key("decoder.mid_block.attentions.0.norm.gamma").unwrap(),
            "decoder.mid_block.attentions.0.norm.weight"
        );
        assert_eq!(
            vae_internal_key("post_quant_conv.weight").unwrap(),
            "post_quant_conv.conv3d.weight"
        );
    }
}
