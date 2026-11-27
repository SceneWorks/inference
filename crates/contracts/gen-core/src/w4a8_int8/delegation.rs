//! Delegating a native checkpoint's **W4A8 projections out of the codec plan** (SW-6135).
//!
//! The checkpoint-codec plan (`crate::checkpoint_codec`) has no W4A8 codec row, and it does not
//! need one: a W4A8 projection is served whole by the backend's packed-forward layer
//! (`W4A8Linear`), which reads its own descriptor and four operands and never needs the plan's
//! dense decode, packed residency or companion pricing. SW-6134 recorded the decision: W4A8 is
//! *delegated* out of the plan, the way [`crate::checkpoint_codec::ProjectionSourceHeaders::select`]
//! already strips it from a projected-source inventory.
//!
//! A family loader that plans a **whole** native single file (every dense tensor included) needs
//! the same split before planning: [`delegate_projections`] finds every `.comfy_quant` descriptor
//! naming [`FORMAT`], validates the operand set against the shared [`W4A8Int8`] contract from the
//! headers alone (no tensor data is read), and returns the physical keys the delegated projections
//! own. The caller plans the remaining tensors and serves the delegated bases through `W4A8Linear`.
//! A projection's `.bias` is deliberately **not** delegated: it is an ordinary dense tensor the plan
//! maps and prices like any other.

use std::collections::{BTreeMap, BTreeSet};

use super::{W4A8Int8, FORMAT, MAX_DESCRIPTOR_BYTES};
use crate::weightsmeta::{Dtype, SafetensorsTensorHeader};

/// The tensor suffixes one delegated W4A8 projection owns, beside `{base}`. The bias is not one of
/// them (see the module docs).
pub const DELEGATED_SUFFIXES: [&str; 5] = [
    "comfy_quant",
    "weight",
    "weight_codebook",
    "weight_s_channel",
    "weight_s_rel",
];

/// Sibling suffixes a W4A8 projection must not carry: each belongs to a different dialect (MLX
/// affine, plain/ConvRot INT8, a correction tensor this dialect does not define), and accepting one
/// silently would leave a tensor nobody reads.
const FORBIDDEN_SUFFIXES: [&str; 4] = ["weight_scale", "scales", "biases", "weight_correction"];

/// The W4A8 projections of one checkpoint, split out of its codec plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct W4A8Delegation {
    /// Validated geometry per physical projection base (for example `blocks.4.attn.wq`).
    pub projections: BTreeMap<String, W4A8Int8>,
    /// Every physical key the delegated projections own: the descriptor and the four operands.
    pub keys: BTreeSet<String>,
}

impl W4A8Delegation {
    /// No W4A8 projection: the checkpoint plans exactly as it would without delegation.
    pub fn is_empty(&self) -> bool {
        self.projections.is_empty()
    }

    /// Whether `physical_key` belongs to a delegated projection.
    pub fn owns(&self, physical_key: &str) -> bool {
        self.keys.contains(physical_key)
    }

    /// The validated geometry of a delegated physical base.
    pub fn geometry(&self, base: &str) -> Option<W4A8Int8> {
        self.projections.get(base).copied()
    }

    /// Resident bytes of every delegated projection (packed codes plus companions; no dense
    /// shadow), the figure a memory contract prices the delegated half at.
    pub fn resident_bytes(&self) -> u64 {
        self.projections
            .values()
            .map(|d| d.resident_bytes() as u64)
            .sum()
    }
}

/// Split the W4A8 projections out of a checkpoint's header inventory.
///
/// `descriptors` holds the raw `.comfy_quant` payloads keyed by full physical name (the same map
/// the plan compiler reads). A descriptor naming another format is left to the plan untouched. For
/// each W4A8 descriptor every operand must be present at its exact dtype (`I8` weight, `F8_E4M3`
/// `weight_s_rel`, `F32` `weight_s_channel` / `weight_codebook`), its shapes must satisfy
/// [`W4A8Int8::parse`], and no foreign-dialect sibling may sit beside it. A W4A8 operand suffix
/// whose base has no W4A8 descriptor is refused too: it would otherwise reach the plan as an
/// unexplained tensor, or worse, be read as something else.
pub fn delegate_projections(
    headers: &[SafetensorsTensorHeader],
    descriptors: &BTreeMap<String, Vec<u8>>,
) -> Result<W4A8Delegation, String> {
    let by_name: BTreeMap<&str, &SafetensorsTensorHeader> =
        headers.iter().map(|h| (h.name.as_str(), h)).collect();
    let mut delegation = W4A8Delegation::default();
    for (name, payload) in descriptors {
        let Some(base) = name.strip_suffix(".comfy_quant") else {
            continue;
        };
        if payload.len() > MAX_DESCRIPTOR_BYTES {
            return Err(format!(
                "{name}: descriptor exceeds {MAX_DESCRIPTOR_BYTES} bytes"
            ));
        }
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(payload) else {
            // Not ours to judge: the plan compiler owns every non-W4A8 descriptor and its errors.
            continue;
        };
        if json.get("format").and_then(serde_json::Value::as_str) != Some(FORMAT) {
            continue;
        }
        let operand = |suffix: &str, dtype: Dtype| -> Result<&SafetensorsTensorHeader, String> {
            let key = format!("{base}.{suffix}");
            let header = by_name
                .get(key.as_str())
                .ok_or_else(|| format!("{base}: W4A8 descriptor without its `{key}` operand"))?;
            if header.dtype != dtype {
                return Err(format!(
                    "{key}: W4A8 requires {dtype:?} storage, found {:?}",
                    header.dtype
                ));
            }
            Ok(header)
        };
        let descriptor = by_name
            .get(name.as_str())
            .ok_or_else(|| format!("{name}: descriptor payload has no header"))?;
        if descriptor.dtype != Dtype::U8 || descriptor.shape.len() != 1 {
            return Err(format!("{name}: descriptor must be a U8 vector"));
        }
        let weight = operand("weight", Dtype::I8)?;
        let s_rel = operand("weight_s_rel", Dtype::F8_E4M3)?;
        let s_channel = operand("weight_s_channel", Dtype::F32)?;
        let codebook = operand("weight_codebook", Dtype::F32)?;
        for suffix in FORBIDDEN_SUFFIXES {
            if by_name.contains_key(format!("{base}.{suffix}").as_str()) {
                return Err(format!("{base}: W4A8 forbids a `.{suffix}` sibling"));
            }
        }
        let geometry = W4A8Int8::parse(
            payload,
            &weight.shape,
            &s_rel.shape,
            &s_channel.shape,
            &codebook.shape,
        )
        .map_err(|e| format!("{base}: {e}"))?;
        delegation.projections.insert(base.to_owned(), geometry);
        delegation.keys.extend(
            DELEGATED_SUFFIXES
                .iter()
                .map(|suffix| format!("{base}.{suffix}")),
        );
    }
    for header in headers {
        for suffix in ["weight_codebook", "weight_s_channel", "weight_s_rel"] {
            if let Some(base) = header.name.strip_suffix(&format!(".{suffix}")) {
                if !delegation.projections.contains_key(base) {
                    return Err(format!(
                        "{}: W4A8 operand without an authoritative `{base}.comfy_quant` descriptor",
                        header.name
                    ));
                }
            }
        }
    }
    Ok(delegation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(name: &str, dtype: Dtype, shape: &[usize]) -> SafetensorsTensorHeader {
        let elems: usize = shape.iter().product();
        SafetensorsTensorHeader {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: (elems * dtype.size()) as u64,
        }
    }

    fn projection(base: &str, out: usize, inp: usize) -> Vec<SafetensorsTensorHeader> {
        vec![
            h(
                &format!("{base}.comfy_quant"),
                Dtype::U8,
                &[super::super::pack::DESCRIPTOR_PAYLOAD.len()],
            ),
            h(&format!("{base}.weight"), Dtype::I8, &[out, inp / 2]),
            h(&format!("{base}.weight_codebook"), Dtype::F32, &[16]),
            h(&format!("{base}.weight_s_channel"), Dtype::F32, &[out]),
            h(
                &format!("{base}.weight_s_rel"),
                Dtype::F8_E4M3,
                &[out, inp / 16],
            ),
        ]
    }

    fn descriptors(bases: &[&str]) -> BTreeMap<String, Vec<u8>> {
        bases
            .iter()
            .map(|b| {
                (
                    format!("{b}.comfy_quant"),
                    super::super::pack::DESCRIPTOR_PAYLOAD.to_vec(),
                )
            })
            .collect()
    }

    #[test]
    fn delegates_exactly_the_w4a8_operands_and_leaves_the_rest_to_the_plan() {
        let mut headers = projection("blocks.4.attn.wq", 32, 256);
        headers.push(h("blocks.4.attn.wq.bias", Dtype::BF16, &[32]));
        headers.push(h("blocks.0.attn.wq.weight", Dtype::BF16, &[32, 256]));
        let mut descs = descriptors(&["blocks.4.attn.wq"]);
        // A non-W4A8 descriptor is not ours: it stays with the plan compiler.
        descs.insert(
            "blocks.1.attn.wq.comfy_quant".into(),
            br#"{"format":"int8_tensorwise"}"#.to_vec(),
        );
        let d = delegate_projections(&headers, &descs).unwrap();
        assert_eq!(d.projections.len(), 1);
        let g = d.geometry("blocks.4.attn.wq").unwrap();
        assert_eq!((g.out_features, g.in_features), (32, 256));
        assert_eq!(d.keys.len(), 5);
        assert!(d.owns("blocks.4.attn.wq.weight_s_rel"));
        assert!(
            !d.owns("blocks.4.attn.wq.bias"),
            "the bias stays a planned dense tensor"
        );
        assert!(!d.owns("blocks.0.attn.wq.weight"));
        assert_eq!(d.resident_bytes(), g.resident_bytes() as u64);
    }

    #[test]
    fn no_w4a8_descriptor_is_an_empty_delegation() {
        let headers = vec![h("blocks.0.attn.wq.weight", Dtype::BF16, &[32, 256])];
        assert!(delegate_projections(&headers, &BTreeMap::new())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn refuses_a_missing_operand_a_wrong_dtype_a_foreign_sibling_and_an_orphan() {
        let base = "blocks.4.attn.wq";
        let descs = descriptors(&[base]);

        let missing: Vec<_> = projection(base, 32, 256)
            .into_iter()
            .filter(|h| !h.name.ends_with(".weight_s_rel"))
            .collect();
        let err = delegate_projections(&missing, &descs).unwrap_err();
        assert!(err.contains("weight_s_rel"), "{err}");

        let mut wrong = projection(base, 32, 256);
        wrong[2].dtype = Dtype::BF16; // the codebook
        let err = delegate_projections(&wrong, &descs).unwrap_err();
        assert!(err.contains("weight_codebook"), "{err}");

        let mut foreign = projection(base, 32, 256);
        foreign.push(h(&format!("{base}.weight_scale"), Dtype::F32, &[32]));
        let err = delegate_projections(&foreign, &descs).unwrap_err();
        assert!(err.contains("weight_scale"), "{err}");

        let mut bad_geometry = projection(base, 32, 256);
        bad_geometry[4].shape = vec![32, 8]; // s_rel must be [out, in/16]
        assert!(delegate_projections(&bad_geometry, &descs).is_err());

        let orphan = vec![h(
            "blocks.9.attn.wq.weight_s_rel",
            Dtype::F8_E4M3,
            &[32, 16],
        )];
        let err = delegate_projections(&orphan, &BTreeMap::new()).unwrap_err();
        assert!(err.contains("comfy_quant"), "{err}");
    }
}
