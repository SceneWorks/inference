//! Fail-closed resolver for the hosted VACE-Fun MLX-affine q4/q8 tiers.
//!
//! A tier directory carries only the two packed VACE experts.  Shared dense components stay at
//! the parent snapshot root, so accepting one expert, a loose sidecar, or a mixed q4/dense pair
//! would otherwise silently select the wrong model.  This resolver validates that contract from
//! safetensors headers only; the actual load remains mmap-backed through `VarBuilder`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::Quant;
use candle_gen::quant::{PackedConfig, MLX_GROUP_SIZE};
use candle_gen::{CandleError, Result};

const MARKER: &str = "split_model.json";
const HIGH: &str = "high_noise_model.safetensors";
const LOW: &str = "low_noise_model.safetensors";

#[derive(Clone, Debug)]
pub(crate) struct VaceFunTierPaths {
    root: PathBuf,
    shared_root: PathBuf,
    bits: Quant,
}

impl VaceFunTierPaths {
    /// Detect a hosted split tier.  A raw diffusers snapshot deliberately returns `None`.
    pub(crate) fn detect(root: &Path) -> Result<Option<Self>> {
        if !root.join(MARKER).is_file() {
            return Ok(None);
        }
        let high = root.join(HIGH);
        let low = root.join(LOW);
        if !high.is_file() || !low.is_file() {
            return Err(CandleError::Msg(format!(
                "wan VACE-Fun tier {} has {MARKER} but must contain both {HIGH} and {LOW}; \\
                 partial/orphaned MoE payloads are rejected",
                root.display()
            )));
        }
        // A split tier stages the dense VAE/TE/tokenizer once beside q4/q8.  A self-contained
        // tier is also valid, but never borrow an arbitrary ancestor without this explicit check.
        let shared_root = root
            .parent()
            .filter(|p| has_shared_components(p))
            .map(Path::to_path_buf)
            .unwrap_or_else(|| root.to_path_buf());
        if !has_shared_components(&shared_root) {
            return Err(CandleError::Msg(format!(
                "wan VACE-Fun tier {} has no complete shared text_encoder/, vae/, and tokenizer/ layout \\
                 beside (or inside) the split q4/q8 experts",
                root.display()
            )));
        }
        let cfg = read_config(root)?;
        let bits = match cfg.bits {
            4 => Quant::Q4,
            8 => Quant::Q8,
            other => return Err(CandleError::Msg(format!(
                "wan VACE-Fun tier {} declares unsupported packed bit width {other}; expected q4 or q8",
                root.display()
            ))),
        };
        if cfg.group_size as usize != MLX_GROUP_SIZE {
            return Err(CandleError::Msg(format!(
                "wan VACE-Fun tier {} declares group_size {}, but the MLX-affine Candle seam requires {}; \\
                 refusing to misalign packed weights",
                root.display(), cfg.group_size, MLX_GROUP_SIZE
            )));
        }
        validate_expert(&high, "high", cfg.bits)?;
        validate_expert(&low, "low", cfg.bits)?;
        Ok(Some(Self {
            root: root.to_path_buf(),
            shared_root,
            bits,
        }))
    }

    pub(crate) fn validate_requested_quant(&self, requested: Option<Quant>) -> Result<()> {
        if let Some(requested) = requested {
            if requested != self.bits {
                return Err(CandleError::Msg(format!(
                    "wan VACE-Fun tier {} is {}, but LoadSpec requested {}; refusing a mixed tier request",
                    self.root.display(), quant_name(self.bits), quant_name(requested)
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn component_vb(
        &self,
        sub: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'static>> {
        let file = match sub {
            "transformer" => self.root.join(HIGH),
            "transformer_2" => self.root.join(LOW),
            _ => {
                return Err(CandleError::Msg(format!(
                    "{sub} is not a VACE-Fun tier expert"
                )))
            }
        };
        candle_gen::mmap_var_builder(&[file], dtype, device)
    }

    pub(crate) fn shared_root(&self) -> &Path {
        &self.shared_root
    }
}

fn has_shared_components(root: &Path) -> bool {
    ["text_encoder", "vae", "tokenizer"]
        .iter()
        .all(|component| root.join(component).is_dir())
}

fn read_config(root: &Path) -> Result<PackedConfig> {
    let path = root.join("quantize_config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        CandleError::Msg(format!("wan VACE-Fun tier: read {}: {e}", path.display()))
    })?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        CandleError::Msg(format!("wan VACE-Fun tier: parse {}: {e}", path.display()))
    })?;
    // Hosted WAN tiers retain sc-10026's top-level `bits` plus nested group metadata.
    let bits = value.get("bits").and_then(|v| v.as_i64()).ok_or_else(|| {
        CandleError::Msg(format!(
            "wan VACE-Fun tier: {} has no top-level `bits`",
            path.display()
        ))
    })? as i32;
    let group_size = value
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(MLX_GROUP_SIZE as i64) as i32;
    Ok(PackedConfig { bits, group_size })
}

/// Header-only validation: every packed payload is a complete MLX affine triple, both experts have
/// a packed projection, and the two split files spell the diffusers keys Candle reads.  No tensor is
/// copied or dequantized here.
fn validate_expert(path: &Path, label: &str, bits: i32) -> Result<()> {
    let mapped = unsafe { MmapedSafetensors::new(path) }?;
    let names: BTreeSet<String> = mapped.tensors().into_iter().map(|(name, _)| name).collect();
    let packed: Vec<String> = names
        .iter()
        .filter_map(|n| n.strip_suffix(".scales").map(str::to_owned))
        .collect();
    if packed.is_empty() {
        return Err(CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} has no MLX-affine `.scales` payloads; q{bits} must not mix with dense experts",
            path.display()
        )));
    }
    if names
        .iter()
        .any(|name| name.starts_with("blocks.") && name.contains("self_attn."))
    {
        return Err(CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} uses native-WAN keys; this tier needs a producer remap to the diffusers keys Candle loads",
            path.display()
        )));
    }
    for base in &packed {
        for suffix in ["weight", "biases"] {
            let key = format!("{base}.{suffix}");
            if !names.contains(&key) {
                return Err(CandleError::Msg(format!(
                    "wan VACE-Fun {label} expert {} has incomplete packed triple: missing {key}",
                    path.display()
                )));
            }
        }
    }
    for key in names.iter().filter(|n| n.ends_with(".biases")) {
        let base = key.strip_suffix(".biases").unwrap();
        if !names.contains(&format!("{base}.scales")) {
            return Err(CandleError::Msg(format!(
                "wan VACE-Fun {label} expert {} has orphaned packed sidecar {key}",
                path.display()
            )));
        }
    }
    for base in required_packed_bases() {
        if !names.contains(&format!("{base}.scales")) {
            return Err(CandleError::Msg(format!(
                "wan VACE-Fun {label} expert {} is missing {base}.scales; refusing a partial packed projection layout",
                path.display()
            )));
        }
    }
    Ok(())
}

/// The VACE-Fun hosted-tier contract packs every rank-two projection Candle builds.  Keeping this
/// explicit protects the less-obvious VACE control projections (`vace_blocks.*.proj_{in,out}`) as
/// well as the base Wan attention/FFN surface; a single dense straggler would otherwise force a
/// hidden mixed-precision expert.
fn required_packed_bases() -> Vec<String> {
    let mut bases = vec![
        "condition_embedder.text_embedder.linear_1".to_owned(),
        "condition_embedder.text_embedder.linear_2".to_owned(),
        "condition_embedder.time_embedder.linear_1".to_owned(),
        "condition_embedder.time_embedder.linear_2".to_owned(),
        "condition_embedder.time_proj".to_owned(),
        "proj_out".to_owned(),
    ];
    for block in 0..40 {
        for attn in ["attn1", "attn2"] {
            for projection in ["to_q", "to_k", "to_v", "to_out.0"] {
                bases.push(format!("blocks.{block}.{attn}.{projection}"));
            }
        }
        for projection in ["net.0.proj", "net.2"] {
            bases.push(format!("blocks.{block}.ffn.{projection}"));
        }
    }
    for block in 0..8 {
        if block == 0 {
            bases.push("vace_blocks.0.proj_in".to_owned());
        }
        bases.push(format!("vace_blocks.{block}.proj_out"));
    }
    bases
}

fn quant_name(q: Quant) -> &'static str {
    match q {
        Quant::Q4 => "q4",
        Quant::Q8 => "q8",
        Quant::Nvfp4 => "nvfp4",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Tensor;
    use std::collections::HashMap;

    fn write_expert(path: &Path, complete: bool) {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for base in required_packed_bases() {
            map.insert(
                format!("{base}.weight"),
                Tensor::zeros((8, 8), DType::U32, &dev).unwrap(),
            );
            map.insert(
                format!("{base}.scales"),
                Tensor::ones((8, 1), DType::F32, &dev).unwrap(),
            );
            map.insert(
                format!("{base}.biases"),
                Tensor::zeros((8, 1), DType::F32, &dev).unwrap(),
            );
        }
        if !complete {
            map.remove("vace_blocks.7.proj_out.biases");
        }
        candle_gen::candle_core::safetensors::save(&map, path).unwrap();
    }

    fn tier(bits: i32) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        for component in ["text_encoder", "vae", "tokenizer"] {
            std::fs::create_dir(tmp.path().join(component)).unwrap();
        }
        std::fs::write(tmp.path().join(MARKER), "{}").unwrap();
        std::fs::write(
            tmp.path().join("quantize_config.json"),
            format!(r#"{{"bits":{bits},"quantization":{{"bits":{bits},"group_size":64}}}}"#),
        )
        .unwrap();
        write_expert(&tmp.path().join(HIGH), true);
        write_expert(&tmp.path().join(LOW), true);
        tmp
    }

    #[test]
    fn q4_and_q8_split_tiers_load_without_dense_staging() {
        for bits in [4, 8] {
            let tmp = tier(bits);
            let paths = VaceFunTierPaths::detect(tmp.path()).unwrap().unwrap();
            assert_eq!(paths.bits, if bits == 4 { Quant::Q4 } else { Quant::Q8 });
            let vb = paths
                .component_vb("transformer", DType::F32, &Device::Cpu)
                .unwrap();
            assert!(vb.contains_tensor("blocks.0.attn1.to_q.scales"));
        }
    }

    #[test]
    fn incomplete_or_mixed_expert_layout_fails_closed() {
        let tmp = tier(4);
        write_expert(&tmp.path().join(LOW), false);
        assert!(VaceFunTierPaths::detect(tmp.path()).is_err());
        let tmp = tier(8);
        let paths = VaceFunTierPaths::detect(tmp.path()).unwrap().unwrap();
        assert!(paths.validate_requested_quant(Some(Quant::Q4)).is_err());
    }
}
