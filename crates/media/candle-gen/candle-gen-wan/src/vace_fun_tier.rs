//! Fail-closed resolver for the hosted VACE-Fun MLX-affine q4/q8 tiers.
//!
//! A tier directory carries only the two packed VACE experts.  Shared dense components stay at
//! the parent snapshot root, so accepting one expert, a loose sidecar, or a mixed q4/dense pair
//! would otherwise silently select the wrong model.  This resolver validates that contract from
//! safetensors headers only; the actual load remains mmap-backed through `VarBuilder`.

use std::collections::{BTreeMap, BTreeSet};
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

/// The safetensors information needed to prove a packed projection is really an MLX-affine q4/q8
/// payload before a `QLinear` can fall back to its dense arm. Header inspection intentionally keeps
/// the experts mmap-backed; no tensor data is staged or dequantized during tier admission.
#[derive(Clone, Debug)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
}

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

    #[cfg(test)]
    pub(crate) fn test_paths(root: PathBuf, shared_root: PathBuf, bits: Quant) -> Self {
        Self {
            root,
            shared_root,
            bits,
        }
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

/// Header-only validation: every required VACE-Fun projection is a complete MLX-affine triple with
/// its dense output bias, its q4/q8 geometry agrees with the tier marker, and both experts spell the
/// diffusers keys Candle reads. No tensor is copied or dequantized here.
fn validate_expert(path: &Path, label: &str, bits: i32) -> Result<()> {
    let mapped = unsafe { MmapedSafetensors::new(path) }?;
    let headers: BTreeMap<String, TensorHeader> = mapped
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            (
                name,
                TensorHeader {
                    dtype: format!("{:?}", view.dtype()),
                    shape: view.shape().to_vec(),
                },
            )
        })
        .collect();
    let names: BTreeSet<&str> = headers.keys().map(String::as_str).collect();
    let packed: Vec<&str> = names
        .iter()
        .filter_map(|name| name.strip_suffix(".scales"))
        .collect();
    if packed.is_empty() {
        return Err(CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} has no MLX-affine `.scales` payloads; q{bits} must not mix with dense experts",
            path.display()
        )));
    }
    if names.iter().any(|name| name.contains(".self_attn.")) {
        return Err(CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} uses native-WAN keys; this tier needs a producer remap to the diffusers keys Candle loads",
            path.display()
        )));
    }

    // A U32 `.weight` is an MLX packed code stream, never a dense Wan weight. Requiring its
    // sidecars prevents a damaged packed tensor from taking a future dense fallback by accident.
    for name in names.iter().filter(|name| name.ends_with(".weight")) {
        if headers[*name].dtype == "U32" {
            let base = name.strip_suffix(".weight").unwrap();
            if !names.contains(&format!("{base}.scales").as_str()) {
                return Err(CandleError::Msg(format!(
                    "wan VACE-Fun {label} expert {} has orphaned packed payload {name}",
                    path.display()
                )));
            }
        }
    }
    for key in names.iter().filter(|name| name.ends_with(".biases")) {
        let base = key.strip_suffix(".biases").unwrap();
        if !names.contains(&format!("{base}.scales").as_str()) {
            return Err(CandleError::Msg(format!(
                "wan VACE-Fun {label} expert {} has orphaned packed sidecar {key}",
                path.display()
            )));
        }
    }
    for base in packed {
        validate_packed_projection(&headers, base, label, path, bits, false)?;
    }
    for base in required_packed_bases() {
        validate_packed_projection(&headers, &base, label, path, bits, true)?;
    }
    Ok(())
}

/// Validate an MLX affine triple's key schema, scalar dtypes, group-64 shape relation, and exact
/// payload bit-width. `require_dense_bias` applies to the VACE-Fun projection surface consumed by
/// `QLinear::linear_detect(..., bias = true)`; without the sidecar, loading must fail here rather
/// than admitting a tier that cannot construct the declared transformer.
fn validate_packed_projection(
    headers: &BTreeMap<String, TensorHeader>,
    base: &str,
    label: &str,
    path: &Path,
    expected_bits: i32,
    require_dense_bias: bool,
) -> Result<()> {
    let key = |suffix: &str| format!("{base}.{suffix}");
    let missing = |suffix: &str| {
        CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} has incomplete packed projection {base}: missing {}",
            path.display(),
            key(suffix)
        ))
    };
    let weight = headers
        .get(&key("weight"))
        .ok_or_else(|| missing("weight"))?;
    let scales = headers
        .get(&key("scales"))
        .ok_or_else(|| missing("scales"))?;
    let biases = headers
        .get(&key("biases"))
        .ok_or_else(|| missing("biases"))?;
    if weight.dtype != "U32" {
        return Err(CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} has {} dtype {}; packed codes must be U32",
            path.display(),
            key("weight"),
            weight.dtype
        )));
    }
    if !is_floating_dtype(&scales.dtype) || !is_floating_dtype(&biases.dtype) {
        return Err(CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} has non-floating affine sidecars for {base}: scales={} biases={}",
            path.display(),
            scales.dtype,
            biases.dtype
        )));
    }
    let ([out, weight_cols], [scale_out, scale_cols], _bias_shape) = match (
        weight.shape.as_slice(),
        scales.shape.as_slice(),
        biases.shape.as_slice(),
    ) {
        ([out, weight_cols], [scale_out, scale_cols], bias_shape) => {
            ([*out, *weight_cols], [*scale_out, *scale_cols], bias_shape)
        }
        _ => {
            return Err(CandleError::Msg(format!(
                "wan VACE-Fun {label} expert {} has non-matrix packed triple for {base}: \
                 weight={:?} scales={:?} biases={:?}",
                path.display(),
                weight.shape,
                scales.shape,
                biases.shape
            )))
        }
    };
    if out == 0 || scale_cols == 0 || out != scale_out || scales.shape != biases.shape {
        return Err(CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} has incompatible affine sidecar shapes for {base}: \
             weight={:?} scales={:?} biases={:?}",
            path.display(),
            weight.shape,
            scales.shape,
            biases.shape
        )));
    }
    let logical_in = scale_cols.checked_mul(MLX_GROUP_SIZE).ok_or_else(|| {
        CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} overflows group-{} shape for {base}",
            path.display(),
            MLX_GROUP_SIZE
        ))
    })?;
    let packed_bits_numerator = weight_cols.checked_mul(32).ok_or_else(|| {
        CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} overflows packed-code shape for {base}",
            path.display()
        ))
    })?;
    if packed_bits_numerator % logical_in != 0 {
        return Err(CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} has invalid group-{} packed geometry for {base}: \
             weight={:?} scales={:?}",
            path.display(),
            MLX_GROUP_SIZE,
            weight.shape,
            scales.shape
        )));
    }
    let actual_bits = packed_bits_numerator / logical_in;
    if actual_bits != expected_bits as usize {
        return Err(CandleError::Msg(format!(
            "wan VACE-Fun {label} expert {} marker says q{expected_bits}, but {base} carries q{actual_bits} \
             group-{} payload geometry; refusing a mixed tier",
            path.display(),
            MLX_GROUP_SIZE
        )));
    }
    if require_dense_bias {
        let bias = headers.get(&key("bias")).ok_or_else(|| missing("bias"))?;
        if !is_floating_dtype(&bias.dtype) || bias.shape.as_slice() != [out] {
            return Err(CandleError::Msg(format!(
                "wan VACE-Fun {label} expert {} has invalid dense bias {}: dtype={} shape={:?}, \
                 expected floating [{out}]",
                path.display(),
                key("bias"),
                bias.dtype,
                bias.shape
            )));
        }
    }
    Ok(())
}

fn is_floating_dtype(dtype: &str) -> bool {
    matches!(dtype, "F16" | "BF16" | "F32" | "F64")
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
        append_block_projection_bases(&mut bases, &format!("blocks.{block}"));
    }
    for block in 0..8 {
        append_block_projection_bases(&mut bases, &format!("vace_blocks.{block}"));
        if block == 0 {
            // Diffusers injects the main-token stream only once; `VaceBlock::new` loads proj_in
            // exclusively for block 0, while every VACE block owns a proj_out hint projection.
            bases.push("vace_blocks.0.proj_in".to_owned());
        }
        bases.push(format!("vace_blocks.{block}.proj_out"));
    }
    bases
}

fn append_block_projection_bases(bases: &mut Vec<String>, prefix: &str) {
    for attn in ["attn1", "attn2"] {
        for projection in ["to_q", "to_k", "to_v", "to_out.0"] {
            bases.push(format!("{prefix}.{attn}.{projection}"));
        }
    }
    for projection in ["net.0.proj", "net.2"] {
        bases.push(format!("{prefix}.ffn.{projection}"));
    }
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

    const OUT: usize = 8;
    const IN: usize = MLX_GROUP_SIZE;

    fn expert_tensors(bits: i32) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        let packed_cols = match bits {
            4 => IN / 8,
            8 => IN / 4,
            _ => panic!("test supports q4/q8 only"),
        };
        for base in required_packed_bases() {
            map.insert(
                format!("{base}.weight"),
                Tensor::zeros((OUT, packed_cols), DType::U32, &dev).unwrap(),
            );
            map.insert(
                format!("{base}.scales"),
                Tensor::ones((OUT, IN / MLX_GROUP_SIZE), DType::F32, &dev).unwrap(),
            );
            map.insert(
                format!("{base}.biases"),
                Tensor::zeros((OUT, IN / MLX_GROUP_SIZE), DType::F32, &dev).unwrap(),
            );
            map.insert(
                format!("{base}.bias"),
                Tensor::zeros(OUT, DType::F32, &dev).unwrap(),
            );
        }
        map
    }

    fn write_expert(path: &Path, bits: i32, amend: impl FnOnce(&mut HashMap<String, Tensor>)) {
        let mut map = expert_tensors(bits);
        amend(&mut map);
        candle_gen::candle_core::safetensors::save(&map, path).unwrap();
    }

    fn add_shared_components(root: &Path) {
        for component in ["text_encoder", "vae", "tokenizer"] {
            std::fs::create_dir(root.join(component)).unwrap();
        }
    }

    fn write_tier(root: &Path, bits: i32) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(root.join(MARKER), "{}").unwrap();
        std::fs::write(
            root.join("quantize_config.json"),
            format!(r#"{{"bits":{bits},"quantization":{{"bits":{bits},"group_size":64}}}}"#),
        )
        .unwrap();
        write_expert(&root.join(HIGH), bits, |_| {});
        write_expert(&root.join(LOW), bits, |_| {});
    }

    fn tier(bits: i32) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        add_shared_components(tmp.path());
        write_tier(tmp.path(), bits);
        tmp
    }

    #[test]
    fn q4_and_q8_split_tiers_have_real_group_64_payload_geometry() {
        for bits in [4, 8] {
            let tmp = tier(bits);
            let paths = VaceFunTierPaths::detect(tmp.path()).unwrap().unwrap();
            assert_eq!(paths.bits, if bits == 4 { Quant::Q4 } else { Quant::Q8 });
            let vb = paths
                .component_vb("transformer", DType::F32, &Device::Cpu)
                .unwrap();
            assert!(vb.contains_tensor("blocks.0.attn1.to_q.scales"));
            let mapped = unsafe { MmapedSafetensors::new(tmp.path().join(HIGH)) }.unwrap();
            let weight = mapped.get("blocks.0.attn1.to_q.weight").unwrap();
            assert_eq!(weight.dtype().to_string(), "U32");
            assert_eq!(
                weight.shape(),
                [OUT, if bits == 4 { IN / 8 } else { IN / 4 }]
            );
            let scales = mapped.get("blocks.0.attn1.to_q.scales").unwrap();
            assert_eq!(scales.shape(), [OUT, IN / MLX_GROUP_SIZE]);
        }
    }

    #[test]
    fn shared_parent_layout_is_selected_over_the_tier_root() {
        let snapshot = tempfile::tempdir().unwrap();
        let tier_root = snapshot.path().join("q8");
        add_shared_components(snapshot.path());
        write_tier(&tier_root, 8);
        let paths = VaceFunTierPaths::detect(&tier_root).unwrap().unwrap();
        assert_eq!(paths.shared_root(), snapshot.path());
        assert!(!tier_root.join("tokenizer").exists());
    }

    #[test]
    fn incomplete_native_or_mixed_expert_layout_fails_closed() {
        let tmp = tier(4);
        write_expert(&tmp.path().join(LOW), 4, |map| {
            map.remove("vace_blocks.7.proj_out.biases");
        });
        assert!(VaceFunTierPaths::detect(tmp.path()).is_err());

        let tmp = tier(4);
        write_expert(&tmp.path().join(LOW), 4, |map| {
            map.remove("vace_blocks.7.attn2.to_v.bias");
        });
        assert!(VaceFunTierPaths::detect(tmp.path()).is_err());

        let tmp = tier(4);
        write_expert(&tmp.path().join(HIGH), 4, |map| {
            map.insert(
                "vace_blocks.0.proj_in.weight".to_owned(),
                Tensor::zeros((OUT, IN), DType::F32, &Device::Cpu).unwrap(),
            );
        });
        assert!(VaceFunTierPaths::detect(tmp.path()).is_err());

        let tmp = tier(4);
        write_expert(&tmp.path().join(HIGH), 4, |map| {
            map.insert(
                "blocks.0.self_attn.to_q.weight".to_owned(),
                Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap(),
            );
        });
        assert!(VaceFunTierPaths::detect(tmp.path()).is_err());

        let tmp = tier(4);
        write_expert(&tmp.path().join(HIGH), 4, |map| {
            map.insert(
                "vace_blocks.0.proj_in.scales".to_owned(),
                Tensor::zeros((OUT, IN / MLX_GROUP_SIZE), DType::U32, &Device::Cpu).unwrap(),
            );
        });
        assert!(VaceFunTierPaths::detect(tmp.path()).is_err());

        let tmp = tier(8);
        // A q8 marker with a q4 U32 payload is a mixed tier, not a q8 expert that can safely fall
        // through `QLinear`'s dense arm.
        write_expert(&tmp.path().join(HIGH), 4, |_| {});
        assert!(VaceFunTierPaths::detect(tmp.path()).is_err());

        let tmp = tier(8);
        let paths = VaceFunTierPaths::detect(tmp.path()).unwrap().unwrap();
        assert!(paths.validate_requested_quant(Some(Quant::Q4)).is_err());
    }
}
