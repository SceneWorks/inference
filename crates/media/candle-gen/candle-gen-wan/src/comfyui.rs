//! ComfyUI single-file Wan2.2 expert → candle in-memory remap+dequant seam (epic 10451 Phase 2c,
//! sc-10671).
//!
//! SceneWorks lets a user point at an existing ComfyUI `models/` tree and generate from the weights in
//! place — no copy, no re-download (Phase 1 did LoRAs; sc-10668 bf16 Z-Image; sc-10670 plain-fp8
//! Qwen-Image). This is the Wan2.2 A14B slice. Wan2.2 ships as a **dual-expert MoE**: two separate DiT
//! files, the **high-noise** and **low-noise** experts (`unet/wan2.2_{t2v,i2v}_{high,low}_noise_14B_*`).
//!
//! Two transforms make an in-place expert loadable via [`candle_nn::VarBuilder::from_tensors`]:
//!
//! 1. **Scaled-fp8 dequant.** The ComfyUI file is the *companion* scaled-fp8 convention: each quantized
//!    Linear weight is `F8_E4M3` with a per-tensor scalar `.scale_weight` sibling (and, on the stock
//!    ComfyUI export, a `.scale_input` sibling + a top-level `scaled_fp8` marker tensor). The real
//!    weight is `w = w_fp8·scale_weight`; `scale_input` is the *activation* quant scale, only used by
//!    ComfyUI's fp8×fp8 matmul — irrelevant when we dequant to bf16 and run a normal bf16 matmul, so it
//!    is dropped (same posture as candle-gen-ideogram's fp8 convert). The Kijai variant
//!    (`Wan2_2-*_fp8_e4m3fn_scaled_KJ`) carries `.scale_weight` only (no `.scale_input`); the same code
//!    handles it. Non-quantized tensors (norms, biases, `modulation`, `patch_embedding`) stay dense and
//!    are cast to the compute dtype.
//! 2. **Native-Wan → diffusers key remap.** The ComfyUI file uses the **native Wan** tensor names
//!    (`blocks.N.self_attn.q`, `cross_attn`, `ffn.0/2`, `modulation`, `norm3`, `head.head`,
//!    `text_embedding.0/2`, `time_projection.1`); candle's [`crate::transformer::WanTransformer`] reads
//!    the **diffusers** schema (`blocks.N.attn1/attn2`, `ffn.net.0.proj/net.2`, `scale_shift_table`,
//!    `norm2`, `proj_out`, `condition_embedder.*`). This is the same native→diffusers mapping diffusers'
//!    own `convert_wan_to_diffusers.py` applies. Wan2.2 A14B is channel-concat (no I2V image
//!    cross-attention), so T2V and I2V experts share the key schema — only `patch_embedding` in-channels
//!    (16 vs 36) differ, which the config carries, not the remap.
//!
//! The UMT5 text encoder and the Wan VAE are ALSO read from the ComfyUI tree when present (epic 10451
//! Phase 2c follow-up, sc-10909), leaving only the tiny tokenizer snapshot-sourced:
//!
//! * **UMT5-XXL TE** (`text_encoders/umt5_xxl_fp8_e4m3fn_scaled`) is itself the *same* companion
//!   scaled-fp8 convention as the DiT experts, so it reuses this module's dequant
//!   ([`scaled_builder`]) — but with **no** key remap: the ComfyUI file already carries the HF
//!   `UMT5EncoderModel` keys candle's [`crate::text_encoder::Umt5Encoder`] reads.
//! * **Wan VAE** (`vae/wan_2.1_vae.safetensors`) is the *same physical* Wan2.1 16-channel VAE the
//!   Qwen-Image lane reads (sc-10830); its native-key layout is remapped to the diffusers schema
//!   [`crate::vae16::WanVae16`] reads by the shared core-crate [`candle_gen::comfyui_vae::wan_vae_var_builder`].
//!
//! When a TE or VAE file is absent the component falls back to a resident `SceneWorks/wan2.2-*-candle`
//! snapshot tier (the sc-10671 posture); with both present a Wan generation is sourced entirely from
//! the user's ComfyUI tree plus our tiny tokenizer. The two big DiT experts are always read in place.
//!
//! Header-only classification (which file is a Wan DiT, which quant) is done upstream by SceneWorks
//! (`sceneworks-core::base_weights`, sc-10662); this module is handed files already identified as Wan
//! scaled-fp8 experts.

#[cfg(test)]
use std::collections::HashMap;
use std::path::PathBuf;

use candle_gen::candle_core::DType;
use candle_gen::{CandleError, Result};

/// The in-place ComfyUI files for a Wan2.2 A14B load. The two DiT experts are always read in place
/// (sc-10671); the UMT5 text encoder and the Wan VAE are read in place too when present (sc-10909),
/// else they fall back to the resident snapshot tier. Read in place, never copied; each is remapped +
/// dequant'd/upcast in memory at component build ([`scaled_builder`],
/// [`scaled_builder`], [`candle_gen::comfyui_vae::wan_vae_var_builder`]).
#[derive(Clone, Debug)]
pub(crate) struct ComfyuiExperts {
    /// The **high-noise** expert (ComfyUI `*_high_noise_*` file) → candle `transformer/`.
    pub high_file: PathBuf,
    /// The **low-noise** expert (ComfyUI `*_low_noise_*` file) → candle `transformer_2/`.
    pub low_file: PathBuf,
    /// The UMT5-XXL text encoder (`text_encoders/umt5_xxl_fp8_e4m3fn_scaled`, companion scaled-fp8).
    /// `Some` ⇒ read in place ([`scaled_builder`]); `None` ⇒ the snapshot `text_encoder/` dir.
    pub te_file: Option<PathBuf>,
    /// The Wan VAE (`vae/wan_2.1_vae.safetensors`, native WAN-VAE keys, bf16). `Some` ⇒ read in place
    /// ([`candle_gen::comfyui_vae::wan_vae_var_builder`]); `None` ⇒ the snapshot `vae/` dir.
    pub vae_file: Option<PathBuf>,
}

/// Construct name views over the import while keeping scale planes private to the source codec.
pub(crate) fn scaled_builder(
    file: &std::path::Path,
    dtype: DType,
    dev: &candle_gen::candle_core::Device,
    rename: impl Fn(&str) -> String,
) -> Result<candle_gen::candle_nn::VarBuilder<'static>> {
    let (builder, encoded) =
        candle_gen::loader::remap_file_var_builder_with_source(file, dtype, dev, |key, _| {
            if key == "scaled_fp8"
                || key == "freqs"
                || key.ends_with(".freqs")
                || candle_gen::gen_core::checkpoint_codec::projection_companion_base(key).is_some()
            {
                return Ok(Vec::new());
            }
            Ok(vec![candle_gen::loader::SourceView::Whole(rename(key))])
        })?;
    if !encoded {
        return Err(CandleError::Msg(
            "Wan ComfyUI encoded checkpoint has no supported projection operands".into(),
        ));
    }
    Ok(builder)
}

pub(crate) fn remap_wan_key(key: &str) -> String {
    // --- top-level head ---
    if let Some(rest) = key.strip_prefix("head.head.") {
        return format!("proj_out.{rest}"); // native head Linear → diffusers proj_out
    }
    if key == "head.modulation" {
        return "scale_shift_table".to_owned(); // top-level [1,2,dim] head modulation
    }
    // --- top-level condition embedders ---
    const TOP: &[(&str, &str)] = &[
        (
            "text_embedding.0.",
            "condition_embedder.text_embedder.linear_1.",
        ),
        (
            "text_embedding.2.",
            "condition_embedder.text_embedder.linear_2.",
        ),
        (
            "time_embedding.0.",
            "condition_embedder.time_embedder.linear_1.",
        ),
        (
            "time_embedding.2.",
            "condition_embedder.time_embedder.linear_2.",
        ),
        ("time_projection.1.", "condition_embedder.time_proj."),
    ];
    for (from, to) in TOP {
        if let Some(rest) = key.strip_prefix(from) {
            return format!("{to}{rest}");
        }
    }
    // --- per-block ---
    if key.starts_with("blocks.") {
        // Block modulation `[1,6,dim]` → per-block scale_shift_table (exact suffix).
        if let Some(prefix) = key.strip_suffix(".modulation") {
            return format!("{prefix}.scale_shift_table");
        }
        // Segment renames inside a block. Each block key matches at most one (all mutually exclusive),
        // so a single pass suffices; norm_q/norm_k precede q/k so the shorter pattern can't shadow them.
        const BLOCK: &[(&str, &str)] = &[
            (".self_attn.norm_q.", ".attn1.norm_q."),
            (".self_attn.norm_k.", ".attn1.norm_k."),
            (".self_attn.q.", ".attn1.to_q."),
            (".self_attn.k.", ".attn1.to_k."),
            (".self_attn.v.", ".attn1.to_v."),
            (".self_attn.o.", ".attn1.to_out.0."),
            (".cross_attn.norm_q.", ".attn2.norm_q."),
            (".cross_attn.norm_k.", ".attn2.norm_k."),
            (".cross_attn.q.", ".attn2.to_q."),
            (".cross_attn.k.", ".attn2.to_k."),
            (".cross_attn.v.", ".attn2.to_v."),
            (".cross_attn.o.", ".attn2.to_out.0."),
            (".ffn.0.", ".ffn.net.0.proj."),
            (".ffn.2.", ".ffn.net.2."),
            (".norm3.", ".norm2."),
        ];
        for (from, to) in BLOCK {
            if let Some(idx) = key.find(from) {
                return format!("{}{to}{}", &key[..idx], &key[idx + from.len()..]);
            }
        }
    }
    // patch_embedding.* and anything else already match candle's schema.
    key.to_owned()
}

#[cfg(test)]
mod tests {
    const SCALED_FP8_MARKER: &str = "scaled_fp8";
    fn fixture(
        src: HashMap<String, Tensor>,
        dtype: DType,
        rename: fn(&str) -> String,
    ) -> Result<HashMap<String, Tensor>> {
        let dir = tempfile::tempdir().map_err(|e| CandleError::Msg(e.to_string()))?;
        let path = dir.path().join("model.safetensors");
        candle_gen::candle_core::safetensors::save(&src, &path)?;
        let names: Vec<_> = src.keys().map(|s| rename(s)).collect();
        let vb = super::scaled_builder(&path, dtype, &Device::Cpu, rename)?;
        names
            .into_iter()
            .filter(|k| vb.contains_tensor(k))
            .map(|k| Ok((k.clone(), vb.get_unchecked(&k)?)))
            .collect()
    }
    fn remap_and_dequant_comfyui_expert(
        src: HashMap<String, Tensor>,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        fixture(src, dtype, remap_wan_key)
    }
    fn dequant_comfyui_umt5(
        src: HashMap<String, Tensor>,
        dtype: DType,
    ) -> Result<HashMap<String, Tensor>> {
        fixture(src, dtype, str::to_owned)
    }

    use super::*;
    use candle_gen::candle_core::{Device, Tensor};

    fn scalar(v: f32) -> Tensor {
        Tensor::new(v, &Device::Cpu).unwrap()
    }
    fn w(dims: &[usize], dtype: DType) -> Tensor {
        Tensor::ones(dims, DType::F32, &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    // --- key remap ---

    #[test]
    fn remaps_self_and_cross_attn() {
        assert_eq!(
            remap_wan_key("blocks.3.self_attn.q.weight"),
            "blocks.3.attn1.to_q.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.3.self_attn.o.bias"),
            "blocks.3.attn1.to_out.0.bias"
        );
        assert_eq!(
            remap_wan_key("blocks.3.self_attn.norm_q.weight"),
            "blocks.3.attn1.norm_q.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.3.cross_attn.k.weight"),
            "blocks.3.attn2.to_k.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.3.cross_attn.norm_k.weight"),
            "blocks.3.attn2.norm_k.weight"
        );
    }

    #[test]
    fn remaps_ffn_norm_and_modulation() {
        assert_eq!(
            remap_wan_key("blocks.0.ffn.0.weight"),
            "blocks.0.ffn.net.0.proj.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.0.ffn.2.bias"),
            "blocks.0.ffn.net.2.bias"
        );
        assert_eq!(
            remap_wan_key("blocks.0.norm3.weight"),
            "blocks.0.norm2.weight"
        );
        assert_eq!(
            remap_wan_key("blocks.0.modulation"),
            "blocks.0.scale_shift_table"
        );
    }

    #[test]
    fn remaps_top_level_embedders_and_head() {
        assert_eq!(
            remap_wan_key("text_embedding.0.weight"),
            "condition_embedder.text_embedder.linear_1.weight"
        );
        assert_eq!(
            remap_wan_key("time_embedding.2.bias"),
            "condition_embedder.time_embedder.linear_2.bias"
        );
        assert_eq!(
            remap_wan_key("time_projection.1.weight"),
            "condition_embedder.time_proj.weight"
        );
        assert_eq!(remap_wan_key("head.head.weight"), "proj_out.weight");
        assert_eq!(remap_wan_key("head.modulation"), "scale_shift_table");
    }

    #[test]
    fn passes_patch_embedding_unchanged() {
        assert_eq!(
            remap_wan_key("patch_embedding.weight"),
            "patch_embedding.weight"
        );
    }

    // --- dequant ---

    #[test]
    fn dequants_fp8_weight_by_scale_and_remaps_key() {
        let mut src = HashMap::new();
        // A fp8 weight of all-ones with scale 3.0 → dequant to 3.0, remapped to attn1.to_q.
        src.insert(
            "blocks.0.self_attn.q.weight".to_string(),
            w(&[4, 4], DType::F8E4M3),
        );
        src.insert("blocks.0.self_attn.q.scale_weight".to_string(), scalar(3.0));
        src.insert("blocks.0.self_attn.q.scale_input".to_string(), scalar(0.5));
        let out = remap_and_dequant_comfyui_expert(src, DType::BF16).unwrap();
        let t = out
            .get("blocks.0.attn1.to_q.weight")
            .expect("remapped weight");
        assert_eq!(t.dtype(), DType::BF16);
        let v = t
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            (v[0] - 3.0).abs() < 0.05,
            "1.0 * scale 3.0 = 3.0, got {}",
            v[0]
        );
        // Companions + were consumed, not emitted.
        assert!(!out.contains_key("blocks.0.attn1.to_q.scale_weight"));
        assert!(!out.contains_key("blocks.0.self_attn.q.scale_input"));
    }

    #[test]
    fn drops_marker_and_casts_dense_tensors() {
        let mut src = HashMap::new();
        src.insert(
            "blocks.0.self_attn.q.weight".to_string(),
            w(&[2, 2], DType::F8E4M3),
        );
        src.insert("blocks.0.self_attn.q.scale_weight".to_string(), scalar(1.0));
        src.insert(SCALED_FP8_MARKER.to_string(), w(&[0], DType::F8E4M3));
        src.insert("blocks.0.norm3.weight".to_string(), w(&[2], DType::F16)); // dense, no scale
        let out = remap_and_dequant_comfyui_expert(src, DType::BF16).unwrap();
        assert!(!out.contains_key(SCALED_FP8_MARKER));
        // Dense F16 norm cast to bf16 + remapped norm3 → norm2.
        assert_eq!(
            out.get("blocks.0.norm2.weight").unwrap().dtype(),
            DType::BF16
        );
    }

    #[test]
    fn drain_preserves_all_entries_across_many_weights() {
        // Guards the sc-11222 F-122 drain: consuming `src` by value (so each fp8 source drops as its
        // bf16 twin is inserted) must still emit every dequanted weight with its own scale — no entry
        // lost, each key dequanted against its *own* companion.
        let mut src = HashMap::new();
        for (i, s) in [(0usize, 2.0f32), (1, 3.0), (2, 4.0)] {
            src.insert(
                format!("blocks.{i}.self_attn.q.weight"),
                w(&[2, 2], DType::F8E4M3),
            );
            src.insert(format!("blocks.{i}.self_attn.q.scale_weight"), scalar(s));
        }
        let out = remap_and_dequant_comfyui_expert(src, DType::BF16).unwrap();
        for (i, s) in [(0usize, 2.0f32), (1, 3.0), (2, 4.0)] {
            let t = out
                .get(&format!("blocks.{i}.attn1.to_q.weight"))
                .unwrap_or_else(|| panic!("weight {i} survived the drain"));
            let v = t
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(
                (v[0] - s).abs() < 0.05,
                "block {i}: 1.0 * scale {s} = {s}, got {}",
                v[0]
            );
        }
    }

    #[test]
    fn kijai_variant_scale_weight_only() {
        // Kijai carries scale_weight but NO scale_input — must still dequant.
        let mut src = HashMap::new();
        src.insert("head.head.weight".to_string(), w(&[4, 4], DType::F8E4M3));
        src.insert("head.head.scale_weight".to_string(), scalar(2.0));
        let out = remap_and_dequant_comfyui_expert(src, DType::BF16).unwrap();
        assert!(out.contains_key("proj_out.weight"));
    }

    #[test]
    fn rejects_a_map_with_no_scale_companions() {
        let mut src = HashMap::new();
        src.insert(
            "blocks.0.self_attn.q.weight".to_string(),
            w(&[2, 2], DType::BF16),
        );
        assert!(remap_and_dequant_comfyui_expert(src, DType::BF16).is_err());
    }

    #[test]
    fn bare_fp8_expert_uses_shared_source_without_a_scale_companion() {
        let key = "blocks.0.self_attn.q.weight";
        let source = HashMap::from([(key.to_owned(), w(&[2, 2], DType::F8E4M3))]);
        let remapped = remap_and_dequant_comfyui_expert(source, DType::F32).unwrap();
        let weight = &remapped["blocks.0.attn1.to_q.weight"];
        assert_eq!(weight.dims(), &[2, 2]);
        assert_eq!(weight.dtype(), DType::F32);
    }

    // --- UMT5 dequant (sc-10909): same scaled-fp8, but identity key remap ---

    #[test]
    fn umt5_dequants_fp8_linear_without_remapping_keys() {
        let mut src = HashMap::new();
        // A UMT5 SelfAttention Linear: fp8 ones × scale 2.0 → 2.0, key unchanged (no DiT remap).
        src.insert(
            "encoder.block.0.layer.0.SelfAttention.q.weight".to_string(),
            w(&[4, 4], DType::F8E4M3),
        );
        src.insert(
            "encoder.block.0.layer.0.SelfAttention.q.scale_weight".to_string(),
            scalar(2.0),
        );
        src.insert(
            "encoder.block.0.layer.0.SelfAttention.q.scale_input".to_string(),
            scalar(0.5),
        );
        // Dense tensors (embedding, norm, relative bias) cast to the compute dtype, keys unchanged.
        src.insert("shared.weight".to_string(), w(&[8, 4], DType::F16));
        src.insert(
            "encoder.block.0.layer.0.layer_norm.weight".to_string(),
            w(&[4], DType::F16),
        );
        src.insert(SCALED_FP8_MARKER.to_string(), w(&[0], DType::F8E4M3));
        let out = dequant_comfyui_umt5(src, DType::F32).unwrap();

        // The T5 key passes through verbatim (a DiT-style `self_attn.q`→`attn1.to_q` remap would break
        // Umt5Encoder's VarBuilder).
        let t = out
            .get("encoder.block.0.layer.0.SelfAttention.q.weight")
            .expect("umt5 Linear key unchanged");
        assert_eq!(t.dtype(), DType::F32);
        let v = t
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            (v[0] - 2.0).abs() < 0.05,
            "1.0 * scale 2.0 = 2.0, got {}",
            v[0]
        );
        // Dense tensors kept + cast; marker + scale companions dropped.
        assert_eq!(out.get("shared.weight").unwrap().dtype(), DType::F32);
        assert!(out.contains_key("encoder.block.0.layer.0.layer_norm.weight"));
        assert!(!out.contains_key(SCALED_FP8_MARKER));
        assert!(!out.contains_key("encoder.block.0.layer.0.SelfAttention.q.scale_weight"));
        assert!(!out.contains_key("encoder.block.0.layer.0.SelfAttention.q.scale_input"));
    }

    #[test]
    fn umt5_rejects_a_map_with_no_scale_companions() {
        let mut src = HashMap::new();
        src.insert("shared.weight".to_string(), w(&[8, 4], DType::BF16));
        assert!(dequant_comfyui_umt5(src, DType::F32).is_err());
    }
}
