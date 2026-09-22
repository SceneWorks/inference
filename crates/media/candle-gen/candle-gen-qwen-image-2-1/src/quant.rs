//! Qwen-Image 2.1 packed-tier seam for candle (sc-24112) — the consumer half of
//! `mlx_gen_qwen_image_2_1::convert`.
//!
//! Candle has **no affine-quantize-at-load path** for this family and this story does not add one.
//! What it adds is the other half of the same sentence: candle now *installs* the pre-quantized
//! tiers the MLX converter produces, on exactly the artefacts MLX loads, so `supported_quants` can
//! stop reading empty without becoming a lie.
//!
//! # What is installable on this backend after this story
//!
//! | tier | `transformer/` | `text_encoder/` language tower | `vae/` |
//! |---|---|---|---|
//! | bf16 | dense | dense | dense |
//! | q8 | packed Q8 g64 | packed Q8 g64 | dense |
//! | q4 | packed Q4 g64 | packed **Q8** g64 ([`TEXT_ENCODER_Q4_FLOOR`]) | dense |
//!
//! Both packed components bind through [`QLinear::linear_detect_gs`], which reads the packed
//! triple `{base}.weight` (u32 codes) + `.scales` + `.biases` straight into the quantized weight on
//! the target device — **no dense bf16 weight is ever materialized**, so a Q4 tier lands at its
//! packed size rather than staging the dense component first. The packed forward dequantizes into a
//! dense matmul rather than taking candle's int8 `QMatMul` fast path, so a Q4 denoise stays
//! coherent; that behaviour lives in the shared [`candle_gen::quant`] projection.
//!
//! This is the deliberate divergence from `candle_gen_qwen_image` (2512), whose text encoder is
//! dense at every tier and which therefore ships a [`guard_dense`] over the whole encoder. Here only
//! the parts that genuinely stay dense are guarded: the token embedding and the norms. The reasons
//! are in `mlx_gen_qwen_image_2_1::quant` — in one line, 2.1's tower is as large as its DiT, so a
//! dense tower would pin the staged floor at ~15 GB and make the Q4 tier pointless.
//!
//! # `spec.quantize` is a tier selector here, not a transform request
//!
//! `crate::validate_load_spec` accepts `Q4`/`Q8` **only** when the snapshot on disk already is
//! that tier, and refuses a dense snapshot with the same typed `Unsupported` as before ("no
//! on-the-fly quantization"). A request that disagrees with the installed tier is also refused
//! rather than silently served at the installed one. See [`resolve_requested_tier`].

use std::path::Path;

use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    ComponentPrecisionFloor, Error, PrecisionFloorComponent, Quant, Result,
};

pub use candle_gen::quant::AdaptLinear as QLinear;

/// Group size every packed Qwen-Image 2.1 component is written and read at.
pub const GROUP_SIZE: usize = 64;

/// The tier the Qwen3 language tower is resident at when **Q4** is selected — the same declared
/// floor the MLX converter writes into the tier's `text_encoder/config.json`.
pub const TEXT_ENCODER_Q4_FLOOR: Quant = Quant::Q8;

/// The `component_precision_floors` this provider advertises. Published at every tier; it *applies*
/// only at Q4, which `ComponentPrecisionFloor::applies_to` enforces.
pub const COMPONENT_PRECISION_FLOORS: &[ComponentPrecisionFloor] = &[ComponentPrecisionFloor {
    component: PrecisionFloorComponent::TextEncoder,
    selected_tier: Quant::Q4,
    resident_tier: TEXT_ENCODER_Q4_FLOOR,
}];

/// One installable numeric tier, mirroring `mlx_gen_qwen_image_2_1::quant::Tier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    Bf16,
    Q8,
    Q4,
}

impl Tier {
    pub const ALL: [Self; 3] = [Self::Bf16, Self::Q8, Self::Q4];

    pub const fn dir_name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    pub const fn selected_quant(self) -> Option<Quant> {
        match self {
            Self::Bf16 => None,
            Self::Q8 => Some(Quant::Q8),
            Self::Q4 => Some(Quant::Q4),
        }
    }

    pub const fn from_selected(quant: Option<Quant>) -> Option<Self> {
        match quant {
            None => Some(Self::Bf16),
            Some(Quant::Q8) => Some(Self::Q8),
            Some(Quant::Q4) => Some(Self::Q4),
            Some(_) => None,
        }
    }

    /// Bits the `transformer/` is packed at (`None` = dense).
    pub const fn transformer_bits(self) -> Option<i64> {
        match self {
            Self::Bf16 => None,
            Self::Q8 => Some(8),
            Self::Q4 => Some(4),
        }
    }

    /// Bits the `text_encoder/` decoder layers are packed at (`None` = dense).
    pub const fn text_encoder_bits(self) -> Option<i64> {
        match self {
            Self::Bf16 => None,
            Self::Q8 | Self::Q4 => Some(8),
        }
    }
}

/// The `quantization.bits` a component's `config.json` declares, or `None` when it is dense.
///
/// A present-but-damaged marker is an error, never "dense": mistaking a packed component for a
/// dense one is exactly the mislabelling `mlx_gen::quant::write_quantized_config` documents.
pub fn packed_bits(component_dir: &Path) -> Result<Option<i64>> {
    let config = component_dir.join("config.json");
    let bytes = match std::fs::read(&config) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: read {}: {err}",
                config.display()
            )))
        }
    };
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|err| Error::Msg(format!("qwen_image_2_1: parse {}: {err}", config.display())))?;
    let Some(marker) = value.get("quantization") else {
        return Ok(None);
    };
    let bits = marker.get("bits").and_then(serde_json::Value::as_i64);
    let group = marker.get("group_size").and_then(serde_json::Value::as_i64);
    match (bits, group) {
        (Some(bits), Some(group)) if matches!(bits, 4 | 8) && group == GROUP_SIZE as i64 => {
            Ok(Some(bits))
        }
        _ => Err(Error::Msg(format!(
            "qwen_image_2_1: {} declares quantization {{bits: {bits:?}, group_size: {group:?}}}; \
             every installable tier of this model is Q4 or Q8 at group {GROUP_SIZE}",
            config.display()
        ))),
    }
}

/// The tier a snapshot on disk **is**, read from its components' `quantization` markers.
pub fn installed_tier(root: &Path) -> Result<Tier> {
    let dit = packed_bits(&root.join("transformer"))?;
    let te = packed_bits(&root.join("text_encoder"))?;
    Tier::ALL
        .into_iter()
        .find(|tier| tier.transformer_bits() == dit && tier.text_encoder_bits() == te)
        .ok_or_else(|| {
            Error::Msg(format!(
                "qwen_image_2_1: no installable tier has transformer bits {dit:?} with text-encoder \
                 bits {te:?}; the installable tiers are bf16 (dense/dense), q8 (8/8) and q4 (4/8)"
            ))
        })
}

/// Resolve a caller's `LoadSpec::quantize` against the tier on disk.
///
/// * a request that matches the installed tier → that tier, loaded packed;
/// * a Q4/Q8 request against a **dense** snapshot → the historical typed `Unsupported`, because
///   candle cannot produce that tier itself;
/// * a request that disagrees with a packed snapshot, or no request against one → refused with the
///   snapshot to point at instead, because the packed-detect loaders would otherwise serve the
///   installed tier under the requested tier's label.
pub fn resolve_requested_tier(root: &Path, requested: Option<Quant>) -> Result<Tier> {
    let installed = installed_tier(root)?;
    match (installed, requested) {
        (Tier::Bf16, None) => Ok(Tier::Bf16),
        (Tier::Bf16, Some(_)) => Err(Error::Unsupported(
            "qwen_image_2_1: candle has no on-the-fly Q4/Q8 quantization; provision an \
             already-packed snapshot instead"
                .into(),
        )),
        (installed, requested) if installed.selected_quant() == requested => Ok(installed),
        (installed, None) => Err(Error::Msg(format!(
            "qwen_image_2_1: {} is a pre-quantized {} tier but no quantization was requested; \
             point at the bf16 snapshot, or pass the matching quantize tier",
            root.display(),
            installed.dir_name()
        ))),
        (installed, Some(requested)) => Err(Error::Msg(format!(
            "qwen_image_2_1: {} is a pre-quantized {} tier but {requested:?} was requested; the \
             packed-detect loaders would serve {} under the wrong label. Point at the matching \
             snapshot (or a dense one).",
            root.display(),
            installed.dir_name(),
            installed.dir_name()
        ))),
    }
}

/// Error loudly if `{base}.scales` exists under `vb` — a packed weight on a path that reads floats.
///
/// Applied to the leaves that stay dense in **every** tier: the language tower's token embedding and
/// its RMSNorm scales, and the whole all-conv VAE. Reading a u32 code stream as bf16 there would be
/// silent garbage rather than a load failure, so this turns a future tier that packed them into a
/// hard error naming the key.
pub fn guard_dense(vb: &VarBuilder, base: &str) -> Result<()> {
    if vb.contains_tensor(&format!("{base}.scales")) {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: `{base}.scales` is present — this weight is MLX-packed, but the loader \
             here is a dense float path. Every installable tier keeps the token embedding, the \
             norms and the all-conv VAE dense (see `crate::quant`); a tier that packs one of them \
             must add a real packed path rather than reinterpret its codes."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_marker(dir: &Path, bits: i64, group: i64) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "quantization": { "bits": bits, "group_size": group }
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn dense_root(tmp: &Path) -> &Path {
        for component in ["transformer", "text_encoder"] {
            std::fs::create_dir_all(tmp.join(component)).unwrap();
            std::fs::write(tmp.join(component).join("config.json"), "{}").unwrap();
        }
        tmp
    }

    #[test]
    fn the_tier_table_mirrors_the_mlx_converter() {
        assert_eq!(Tier::Q8.transformer_bits(), Some(8));
        assert_eq!(Tier::Q4.transformer_bits(), Some(4));
        assert_eq!(Tier::Q8.text_encoder_bits(), Some(8));
        assert_eq!(
            Tier::Q4.text_encoder_bits(),
            Some(8),
            "the Q4 tier holds the tower at the declared Q8 floor"
        );
        assert_eq!(COMPONENT_PRECISION_FLOORS.len(), 1);
        assert_eq!(
            COMPONENT_PRECISION_FLOORS[0].resident_tier,
            TEXT_ENCODER_Q4_FLOOR
        );
        assert!(COMPONENT_PRECISION_FLOORS[0].applies_to(Quant::Q4));
        assert!(!COMPONENT_PRECISION_FLOORS[0].applies_to(Quant::Q8));
    }

    /// A dense snapshot keeps the historical contract exactly: dense loads, and a Q4/Q8 request is
    /// the same typed `Unsupported` it was before this story.
    #[test]
    fn a_dense_snapshot_still_refuses_on_the_fly_quantization() {
        let tmp = tempfile::tempdir().unwrap();
        let root = dense_root(tmp.path());
        assert_eq!(installed_tier(root).unwrap(), Tier::Bf16);
        assert_eq!(resolve_requested_tier(root, None).unwrap(), Tier::Bf16);
        for quant in [Quant::Q4, Quant::Q8] {
            let err = resolve_requested_tier(root, Some(quant)).unwrap_err();
            assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
            assert!(err.to_string().contains("on-the-fly"), "{err}");
        }
    }

    #[test]
    fn a_packed_tier_is_installable_only_under_its_own_label() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write_marker(&root.join("transformer"), 4, GROUP_SIZE as i64);
        write_marker(&root.join("text_encoder"), 8, GROUP_SIZE as i64);

        assert_eq!(installed_tier(root).unwrap(), Tier::Q4);
        assert_eq!(
            resolve_requested_tier(root, Some(Quant::Q4)).unwrap(),
            Tier::Q4
        );
        let err = resolve_requested_tier(root, Some(Quant::Q8))
            .unwrap_err()
            .to_string();
        assert!(err.contains("wrong label"), "{err}");
        let err = resolve_requested_tier(root, None).unwrap_err().to_string();
        assert!(err.contains("no quantization was requested"), "{err}");
    }

    #[test]
    fn a_foreign_group_size_or_an_unproducible_tier_is_a_hard_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_marker(&tmp.path().join("transformer"), 8, 32);
        let err = packed_bits(&tmp.path().join("transformer"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("group 64"), "{err}");

        let tmp = tempfile::tempdir().unwrap();
        // Q4 DiT with a Q4 tower is below the declared floor: no converter produces it.
        write_marker(&tmp.path().join("transformer"), 4, GROUP_SIZE as i64);
        write_marker(&tmp.path().join("text_encoder"), 4, GROUP_SIZE as i64);
        let err = installed_tier(tmp.path()).unwrap_err().to_string();
        assert!(err.contains("no installable tier"), "{err}");
    }

    #[test]
    fn guard_dense_fires_on_an_unexpected_packed_sibling() {
        use candle_core::{DType, Device, Tensor};
        use candle_gen::candle_core::safetensors::MmapedSafetensors;
        use std::collections::HashMap;

        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "embed_tokens.weight".into(),
            Tensor::zeros((8, 4), DType::U32, &dev).unwrap(),
        );
        map.insert(
            "embed_tokens.scales".into(),
            Tensor::zeros((8, 1), DType::F32, &dev).unwrap(),
        );
        map.insert(
            "norm.weight".into(),
            Tensor::zeros((8,), DType::F32, &dev).unwrap(),
        );
        let path = tmp.path().join("guard.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &path).unwrap();
        // SAFETY: freshly written by this test, single reader.
        let st = unsafe { MmapedSafetensors::new(&path).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev);

        let err = guard_dense(&vb, "embed_tokens").unwrap_err().to_string();
        assert!(err.contains("embed_tokens.scales"), "{err}");
        assert!(guard_dense(&vb, "norm").is_ok());
    }
}
