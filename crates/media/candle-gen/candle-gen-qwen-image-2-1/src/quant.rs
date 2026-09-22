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
//! **A tier is a whole-pipeline contract**: selecting q4 runs q4 through every packable component.
//! There is no per-component promotion and no `component_precision_floors` declaration —
//! [`Tier::text_encoder_bits`] is literally [`Tier::transformer_bits`].
//!
//! | tier | `transformer/` | `text_encoder/` language tower | `vae/` |
//! |---|---|---|---|
//! | bf16 | dense | dense | dense |
//! | q8 | packed Q8 g64 | packed Q8 g64 | dense |
//! | q4 | packed Q4 g64 | packed Q4 g64 | dense |
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
use candle_gen::gen_core::{self, Error, Quant, Result};

pub use candle_gen::quant::AdaptLinear as QLinear;

/// Group size every packed Qwen-Image 2.1 component is written and read at.
pub const GROUP_SIZE: usize = 64;

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
    ///
    /// **Structurally** [`Self::transformer_bits`]: a tier is a whole-pipeline contract, so this
    /// cannot drift from the DiT's width without editing one expression that names the other.
    pub const fn text_encoder_bits(self) -> Option<i64> {
        self.transformer_bits()
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

/// The bit-width a packed component's **weights actually are**, derived from one packed leaf's
/// `{base}.weight` (u32 codes, `[out, in·bits/32]`) and `{base}.scales` (`[out, in/group]`) shapes.
///
/// `None` when the component holds no packed leaf. This is what makes [`installed_tier`] a reading
/// of the artefact rather than of its label: a `config.json` is a text file anyone can edit, and a
/// q4-labelled Q8 snapshot renders perfectly well at Q8 — a parity bar against the real Q8 tier
/// passes and a size-monotonicity check passes on equality, so nothing downstream notices.
fn derived_component_bits(component_dir: &Path) -> Result<Option<i64>> {
    // Header-only: `safetensors_path_tensor_headers` walks the component's shards and parses each
    // file's JSON header without touching the data region.
    let headers = match gen_core::weightsmeta::safetensors_path_tensor_headers(component_dir) {
        Ok(headers) => headers,
        // A component with no readable safetensors is the loaders' problem, not this check's.
        Err(_) => return Ok(None),
    };
    let Some(base) = headers
        .iter()
        .filter_map(|header| header.name.strip_suffix(".scales"))
        .min()
        .map(str::to_owned)
    else {
        return Ok(None);
    };
    let shape = |suffix: &str| {
        headers
            .iter()
            .find(|header| header.name == format!("{base}.{suffix}"))
            .map(|header| header.shape.clone())
    };
    let (Some(wq), Some(scales)) = (shape("weight"), shape("scales")) else {
        return Ok(None);
    };
    // `scales` is `[out, in/group]` ⇒ in = cols·group; the u32-packed `weight` is
    // `[out, in·bits/32]` ⇒ bits = cols·32/in. Exact for any group-aligned Q4/Q8 pack.
    let (&[wq_rows, wq_cols], &[sc_rows, sc_cols]) = (&wq[..], &scales[..]) else {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: {} holds a packed leaf whose code/scale tensors are not rank-2 \
             ({wq:?} / {scales:?})",
            component_dir.display()
        )));
    };
    if wq_rows != sc_rows || sc_cols == 0 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: {} holds a packed leaf whose codes {wq:?} and scales {scales:?} \
             disagree on the output width",
            component_dir.display()
        )));
    }
    let in_features = sc_cols * GROUP_SIZE;
    let bits = wq_cols * 32;
    if in_features == 0 || bits % in_features != 0 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: {} holds a packed leaf whose codes {wq:?} do not divide exactly into \
             {in_features} inputs at group {GROUP_SIZE}",
            component_dir.display()
        )));
    }
    Ok(Some((bits / in_features) as i64))
}

/// The tier a snapshot on disk **is**, read from its components' `quantization` markers **and
/// cross-checked against the packed shapes**.
pub fn installed_tier(root: &Path) -> Result<Tier> {
    let dit = packed_bits(&root.join("transformer"))?;
    let te = packed_bits(&root.join("text_encoder"))?;
    for (label, declared) in [("transformer", dit), ("text_encoder", te)] {
        let Some(declared) = declared else { continue };
        if let Some(derived) = derived_component_bits(&root.join(label))? {
            if derived != declared {
                return Err(Error::Msg(format!(
                    "qwen_image_2_1: {label}/config.json declares Q{declared} but its packed \
                     weights are Q{derived} (derived from the code/scale shapes at group \
                     {GROUP_SIZE}). The label is wrong, not the weights: a mislabelled tier renders \
                     correctly at the width it really is, so nothing downstream would notice."
                )));
            }
        }
    }
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
/// Applied to the leaves that stay dense in **every** tier: the language tower's token embedding,
/// its final `norm` and the per-head q/k RMSNorms ([`crate::text_encoder`]), and the all-conv VAE's
/// outermost convolutions ([`crate::loader::load_vae`]). Reading a u32 code stream as a float there
/// would be silent garbage rather than a load failure, so this turns a future tier that packed one
/// of them into a hard error naming the key.
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

    /// **A tier is a whole-pipeline contract**, and this backend declares no precision floor.
    ///
    /// *Mutation that reds this:* `text_encoder_bits()` returning anything but
    /// `transformer_bits()`, or re-adding a `component_precision_floors` declaration.
    #[test]
    fn a_tier_is_one_width_and_declares_no_precision_floor() {
        assert_eq!(Tier::Q8.transformer_bits(), Some(8));
        assert_eq!(Tier::Q4.transformer_bits(), Some(4));
        for tier in Tier::ALL {
            assert_eq!(
                tier.text_encoder_bits(),
                tier.transformer_bits(),
                "{tier:?}: the tower must run the tier the caller selected"
            );
        }
        assert!(crate::descriptor()
            .capabilities
            .component_precision_floors
            .is_empty());
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
        write_marker(&root.join("text_encoder"), 4, GROUP_SIZE as i64);

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
        // A mixed-width pair: no installable tier names it, because a tier is one width.
        write_marker(&tmp.path().join("transformer"), 4, GROUP_SIZE as i64);
        write_marker(&tmp.path().join("text_encoder"), 8, GROUP_SIZE as i64);
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
