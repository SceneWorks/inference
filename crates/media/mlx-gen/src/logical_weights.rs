//! MLX **mapped logical-weight reader** and the engine's **baseline codec table** (epic 20398,
//! sc-20634).
//!
//! The shared seam every MLX family provider reads imported checkpoints through:
//!
//! 1. the provider supplies its adapter-owned [`LogicalKeyMapping`] (the same `mapping_id` its
//!    registry adapter row declares);
//! 2. [`plan_logical_weights`] reads the safetensors **header only** and compiles a
//!    [`LogicalWeightPlan`] against [`baseline_codec_registry`] — an unmapped key, a key
//!    collision, or a stored encoding without a registered codec is a typed refusal before any
//!    MLX array exists;
//! 3. [`read_logical_weights`] loads the planned tensors, renames each to its canonical logical
//!    key, runs the selected codec, and returns a [`LogicalWeightReceipt`] whose resident bytes
//!    are measured from the decoded arrays.
//!
//! Codecs are registered **once** per platform catalog via [`register_checkpoint_codecs`]; the
//! registry rows and [`CODEC_IMPLEMENTATION_IDS`] are kept in lockstep by the catalog conformance
//! test, so a declared codec without an implementation (or vice versa) cannot ship.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

use gen_core::checkpoint_codec::{
    compile_logical_weight_plan, CheckpointCodecRegistration, CheckpointCodecRegistry,
    CodecResidencyReport, LogicalKeyMapping, LogicalReadMaterialization, LogicalTensorPlan,
    LogicalWeightPlan, LogicalWeightReceipt, WeightEncoding, DENSE_BF16_CODEC, DENSE_CODECS,
    DENSE_F16_CODEC, DENSE_F32_CODEC,
};
use gen_core::ProviderRegistryBuilder;
use mlx_rs::{Array, Dtype};

use crate::weights::Weights;
use crate::{Error, Result};

/// The portable codec rows this engine implements. Register through
/// [`register_checkpoint_codecs`]; never per family crate.
pub const BASELINE_CODECS: &[CheckpointCodecRegistration] = DENSE_CODECS;

/// The codec ids this engine has a decode implementation for. Must equal the ids of
/// [`BASELINE_CODECS`] (the catalog test proves it).
pub const CODEC_IMPLEMENTATION_IDS: &[&str] = &[
    DENSE_BF16_CODEC.codec_id,
    DENSE_F16_CODEC.codec_id,
    DENSE_F32_CODEC.codec_id,
];

/// Register the engine's baseline codec table exactly once into a platform catalog.
pub fn register_checkpoint_codecs(mut builder: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    for codec in BASELINE_CODECS {
        builder = builder.register_checkpoint_codec(*codec);
    }
    builder
}

/// The validated baseline registry the loaders plan against.
pub fn baseline_codec_registry() -> &'static CheckpointCodecRegistry {
    static REGISTRY: OnceLock<CheckpointCodecRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        CheckpointCodecRegistry::new(BASELINE_CODECS.iter().copied())
            .expect("the engine baseline codec table is a valid registry")
    })
}

/// Header-only plan for one safetensors file under the given mapping. Refuses before any array is
/// created; the error names the exact on-disk tensor.
pub fn plan_logical_weights(
    path: &Path,
    mapping: &dyn LogicalKeyMapping,
) -> Result<LogicalWeightPlan> {
    let headers = gen_core::safetensors_path_tensor_headers(path)?;
    compile_logical_weight_plan(&headers, mapping, baseline_codec_registry()).map_err(|error| {
        Error::Msg(format!(
            "logical weight plan for {} ({}): {error}",
            path.display(),
            mapping.mapping_id()
        ))
    })
}

/// How a read leaves its payloads.
pub enum LogicalReadMode<'a> {
    /// Evaluate every planned tensor through the supplied materializer and measure what each
    /// codec left resident. Providers apply their adapter-owned shape normalization inside the
    /// closure (a lossless reshape is part of the canonical logical form) and then run their
    /// pin-guarded `Weights::materialize`, so residency is measured on the final resident arrays.
    Eager(&'a mut dyn FnMut(&mut Weights) -> Result<()>),
    /// Leave payloads lazy; block-window and bounded loaders consume them under their own pin
    /// guards. The receipt reports `Deferred` and no residency rows.
    Deferred,
}

/// The reader's output: logical-keyed weights plus the receipt.
pub struct LogicalWeights {
    pub weights: Weights,
    pub receipt: LogicalWeightReceipt,
}

/// Load exactly the planned tensors from `path`, rename them to their logical keys, decode each
/// through its planned codec, and (eagerly) measure residency.
///
/// The file's tensor set must equal the plan's physical key set: the plan was compiled from this
/// file's header, so any difference means the source changed between planning and reading and the
/// read refuses instead of loading a different checkpoint.
pub fn read_logical_weights(
    path: &Path,
    plan: &LogicalWeightPlan,
    mode: LogicalReadMode<'_>,
) -> Result<LogicalWeights> {
    let mut physical = Weights::from_file(path)?;
    let mut on_disk: Vec<&str> = physical.keys().collect();
    on_disk.sort_unstable();
    let mut planned: Vec<&str> = plan.physical_keys().collect();
    planned.sort_unstable();
    if on_disk != planned {
        let missing: Vec<&str> = planned
            .iter()
            .copied()
            .filter(|key| on_disk.binary_search(key).is_err())
            .collect();
        let unplanned: Vec<&str> = on_disk
            .iter()
            .copied()
            .filter(|key| !planned.contains(key))
            .collect();
        return Err(Error::Msg(format!(
            "logical weight read of {}: tensor set changed since planning ({} planned tensor(s) \
             missing, {} unplanned tensor(s) present); refusing to load a different checkpoint",
            path.display(),
            missing.len(),
            unplanned.len()
        )));
    }

    let mut logical = Weights::empty();
    for tensor in &plan.tensors {
        let array = physical
            .remove(&tensor.physical_key)
            .ok_or_else(|| Error::MissingTensor(tensor.physical_key.clone()))?;
        let decoded = decode(tensor, array)?;
        logical.insert(tensor.logical_key.clone(), decoded);
    }

    let materialization = match mode {
        LogicalReadMode::Eager(materialize) => {
            materialize(&mut logical)?;
            LogicalReadMaterialization::Materialized
        }
        LogicalReadMode::Deferred => LogicalReadMaterialization::Deferred,
    };
    let residency = match materialization {
        LogicalReadMaterialization::Materialized => measure_residency(plan, &logical)?,
        LogicalReadMaterialization::Deferred => Vec::new(),
    };
    Ok(LogicalWeights {
        weights: logical,
        receipt: LogicalWeightReceipt {
            mapping_id: plan.mapping_id,
            tensor_count: plan.tensors.len(),
            source_bytes: plan.source_bytes,
            materialization,
            residency,
        },
    })
}

/// Run the planned codec on one array. The dense codecs are byte-preserving: each asserts the
/// backend decoded exactly the stored dtype it planned for and returns the array untouched, so no
/// cast or substitution can hide inside it.
fn decode(tensor: &LogicalTensorPlan, array: Array) -> Result<Array> {
    let expected = match tensor.codec_id {
        id if id == DENSE_BF16_CODEC.codec_id => Dtype::Bfloat16,
        id if id == DENSE_F16_CODEC.codec_id => Dtype::Float16,
        id if id == DENSE_F32_CODEC.codec_id => Dtype::Float32,
        other => {
            return Err(Error::Msg(format!(
                "codec {other:?} is registered but this engine has no implementation for it \
                 (tensor {:?})",
                tensor.physical_key
            )))
        }
    };
    if array.dtype() != expected {
        return Err(Error::Msg(format!(
            "codec {}: tensor {:?} was planned as {} but the backend loaded {:?}",
            tensor.codec_id,
            tensor.physical_key,
            tensor.encoding.label(),
            array.dtype()
        )));
    }
    Ok(array)
}

/// Resident bytes per codec, measured from the decoded arrays (dtype × shape after decode), not
/// from the header.
fn measure_residency(
    plan: &LogicalWeightPlan,
    logical: &Weights,
) -> Result<Vec<CodecResidencyReport>> {
    let mut by_codec: BTreeMap<&'static str, CodecResidencyReport> = BTreeMap::new();
    for tensor in &plan.tensors {
        let array = logical.require(&tensor.logical_key)?;
        let resident_bytes = u64::try_from(array.nbytes()).map_err(|_| {
            Error::Msg(format!(
                "tensor {:?} resident size overflows u64",
                tensor.logical_key
            ))
        })?;
        let report = by_codec
            .entry(tensor.codec_id)
            .or_insert(CodecResidencyReport {
                codec_id: tensor.codec_id,
                tensor_count: 0,
                source_bytes: 0,
                resident_bytes: 0,
            });
        report.tensor_count += 1;
        report.source_bytes = report.source_bytes.saturating_add(tensor.source_bytes);
        report.resident_bytes = report.resident_bytes.saturating_add(resident_bytes);
    }
    Ok(by_codec.into_values().collect())
}

/// Whether a stored encoding has a registered baseline codec on this engine.
pub fn encoding_is_supported(encoding: WeightEncoding) -> bool {
    baseline_codec_registry().for_encoding(encoding).is_some()
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::path::Path;

    /// Write a minimal safetensors file from `(name, dtype, shape, little-endian payload)` rows.
    pub(crate) fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], Vec<u8>)]) {
        let mut header_entries = Vec::new();
        let mut body = Vec::new();
        for (name, dtype, shape, payload) in tensors {
            let start = body.len();
            body.extend_from_slice(payload);
            let end = body.len();
            header_entries.push(format!(
                "{:?}:{{\"dtype\":{:?},\"shape\":{:?},\"data_offsets\":[{start},{end}]}}",
                name, dtype, shape
            ));
        }
        let mut header = format!("{{{}}}", header_entries.join(",")).into_bytes();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header.len() + body.len());
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&body);
        std::fs::write(path, file).expect("write safetensors fixture");
    }

    pub(crate) fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{bf16_bytes, write_safetensors};
    use super::*;

    struct StripModel;

    impl LogicalKeyMapping for StripModel {
        fn mapping_id(&self) -> &'static str {
            "strip-model-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            physical_key.strip_prefix("model.").map(str::to_owned)
        }
    }

    fn fixture_dir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(&format!("logical-weights-{}-", std::process::id()))
            .tempdir()
            .expect("fixture dir")
    }

    #[test]
    fn baseline_registry_matches_the_implementation_table() {
        let registry = baseline_codec_registry();
        let mut registered: Vec<&str> = registry.codecs().map(|codec| codec.codec_id).collect();
        registered.sort_unstable();
        let mut implemented = CODEC_IMPLEMENTATION_IDS.to_vec();
        implemented.sort_unstable();
        assert_eq!(registered, implemented);
        assert!(encoding_is_supported(WeightEncoding::DenseBf16));
        assert!(!encoding_is_supported(WeightEncoding::Fp8E4M3));
        let catalog = register_checkpoint_codecs(ProviderRegistryBuilder::new())
            .build()
            .expect("baseline codecs register into an empty catalog");
        assert_eq!(
            catalog
                .checkpoint_codecs()
                .codecs()
                .copied()
                .collect::<Vec<_>>(),
            BASELINE_CODECS
        );
    }

    #[test]
    fn eager_read_renames_decodes_and_measures_resident_bytes_from_the_arrays() {
        let dir = fixture_dir();
        let path = dir.path().join("dense.safetensors");
        write_safetensors(
            &path,
            &[
                (
                    "model.b.weight",
                    "BF16",
                    &[2, 2],
                    bf16_bytes(&[1.0, 2.0, 3.0, 4.0]),
                ),
                (
                    "model.a.weight",
                    "BF16",
                    &[3],
                    bf16_bytes(&[0.5, -1.5, 8.0]),
                ),
            ],
        );
        let plan = plan_logical_weights(&path, &StripModel).expect("plan");
        assert_eq!(
            plan.logical_keys().collect::<Vec<_>>(),
            ["a.weight", "b.weight"]
        );
        assert_eq!(plan.source_bytes, 6 + 8);

        let mut materialized = false;
        let mut materialize = |weights: &mut Weights| {
            materialized = true;
            weights.materialize()
        };
        let LogicalWeights { weights, receipt } =
            read_logical_weights(&path, &plan, LogicalReadMode::Eager(&mut materialize))
                .expect("read");
        assert!(
            materialized,
            "eager mode must run the provider's materializer"
        );
        let mut keys: Vec<&str> = weights.keys().collect();
        keys.sort_unstable();
        assert_eq!(keys, ["a.weight", "b.weight"]);
        assert!(weights.get("model.a.weight").is_none());

        // Source consumption: the decoded values are the bytes written to the file.
        let a = weights.require("a.weight").unwrap();
        assert_eq!(a.dtype(), Dtype::Bfloat16);
        let a_values: Vec<f32> = crate::weights::to_f32(a)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(a_values, [0.5, -1.5, 8.0]);
        let b = weights.require("b.weight").unwrap();
        assert_eq!(b.shape(), [2, 2]);

        assert_eq!(receipt.mapping_id, "strip-model-test");
        assert_eq!(receipt.tensor_count, 2);
        assert_eq!(receipt.source_bytes, 14);
        assert_eq!(
            receipt.materialization,
            LogicalReadMaterialization::Materialized
        );
        assert_eq!(receipt.residency.len(), 1);
        let report = &receipt.residency[0];
        assert_eq!(report.codec_id, "dense-bf16-v1");
        assert_eq!(report.tensor_count, 2);
        assert_eq!(report.source_bytes, 14);
        // Dense bf16 leaves exactly the stored bytes resident: measured from the arrays.
        let measured: usize = weights
            .keys()
            .map(|key| weights.get(key).unwrap().nbytes())
            .sum();
        assert_eq!(report.resident_bytes, measured as u64);
        assert_eq!(report.resident_bytes, 14);
        assert_eq!(receipt.resident_bytes(), 14);
    }

    #[test]
    fn mixed_precision_dense_file_reports_one_residency_row_per_codec() {
        // A "dense bf16" community DiT keeps a few biases in f32; each encoding decodes through its
        // own pass-through row and reports its own bytes — nothing is cast or folded.
        let dir = fixture_dir();
        let path = dir.path().join("mixed.safetensors");
        write_safetensors(
            &path,
            &[
                ("model.w", "BF16", &[4], bf16_bytes(&[1.0, 2.0, 3.0, 4.0])),
                (
                    "model.b",
                    "F32",
                    &[2],
                    [0.5_f32, 0.25]
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect(),
                ),
            ],
        );
        let plan = plan_logical_weights(&path, &StripModel).unwrap();
        assert_eq!(plan.codec_ids(), ["dense-bf16-v1", "dense-f32-v1"]);
        let mut materialize = |weights: &mut Weights| weights.materialize();
        let LogicalWeights { weights, receipt } =
            read_logical_weights(&path, &plan, LogicalReadMode::Eager(&mut materialize)).unwrap();
        assert_eq!(weights.require("b").unwrap().dtype(), Dtype::Float32);
        assert_eq!(weights.require("w").unwrap().dtype(), Dtype::Bfloat16);
        assert_eq!(receipt.residency.len(), 2);
        let bf16 = receipt
            .residency
            .iter()
            .find(|report| report.codec_id == "dense-bf16-v1")
            .unwrap();
        let f32 = receipt
            .residency
            .iter()
            .find(|report| report.codec_id == "dense-f32-v1")
            .unwrap();
        assert_eq!(
            (bf16.tensor_count, bf16.source_bytes, bf16.resident_bytes),
            (1, 8, 8)
        );
        assert_eq!(
            (f32.tensor_count, f32.source_bytes, f32.resident_bytes),
            (1, 8, 8)
        );
        assert_eq!(receipt.resident_bytes(), 16);
    }

    #[test]
    fn deferred_read_leaves_payloads_lazy_and_reports_no_residency() {
        let dir = fixture_dir();
        let path = dir.path().join("deferred.safetensors");
        write_safetensors(&path, &[("model.w", "BF16", &[2], bf16_bytes(&[1.0, 2.0]))]);
        let plan = plan_logical_weights(&path, &StripModel).unwrap();
        let LogicalWeights { weights, receipt } =
            read_logical_weights(&path, &plan, LogicalReadMode::Deferred).unwrap();
        assert!(weights.get("w").is_some());
        assert_eq!(
            receipt.materialization,
            LogicalReadMaterialization::Deferred
        );
        assert!(receipt.residency.is_empty());
        assert_eq!(receipt.resident_bytes(), 0);
        assert_eq!(receipt.source_bytes, 4);
    }

    #[test]
    fn planning_refuses_foreign_keys_and_unregistered_encodings_before_any_array_exists() {
        let dir = fixture_dir();
        let foreign = dir.path().join("foreign.safetensors");
        write_safetensors(&foreign, &[("other.w", "BF16", &[1], bf16_bytes(&[1.0]))]);
        let error = plan_logical_weights(&foreign, &StripModel).unwrap_err();
        assert!(
            error.to_string().contains("\"other.w\"")
                && error.to_string().contains("no canonical logical key"),
            "{error}"
        );

        let fp8 = dir.path().join("fp8.safetensors");
        write_safetensors(
            &fp8,
            &[
                ("model.ok", "BF16", &[1], bf16_bytes(&[1.0])),
                ("model.packed", "F8_E4M3", &[2], vec![0x38, 0x40]),
            ],
        );
        let error = plan_logical_weights(&fp8, &StripModel).unwrap_err();
        assert!(
            error.to_string().contains("\"model.packed\"")
                && error.to_string().contains("fp8-e4m3")
                && error
                    .to_string()
                    .contains("no checkpoint codec is registered"),
            "{error}"
        );
    }

    struct Reverse;

    impl LogicalKeyMapping for Reverse {
        fn mapping_id(&self) -> &'static str {
            "reverse-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            match physical_key {
                "model.a" => Some("zeta".to_owned()),
                "model.b" => Some("alpha".to_owned()),
                _ => None,
            }
        }
    }

    #[test]
    fn read_accepts_plans_whose_logical_order_differs_from_physical_order() {
        // The plan is sorted by logical key; the on-disk check must compare physical key SETS, not
        // positions, or a perfectly valid remap refuses as "changed since planning".
        let dir = fixture_dir();
        let path = dir.path().join("reorder.safetensors");
        write_safetensors(
            &path,
            &[
                ("model.a", "BF16", &[1], bf16_bytes(&[1.0])),
                ("model.b", "BF16", &[1], bf16_bytes(&[2.0])),
            ],
        );
        let plan = plan_logical_weights(&path, &Reverse).unwrap();
        assert_eq!(plan.logical_keys().collect::<Vec<_>>(), ["alpha", "zeta"]);
        assert_eq!(
            plan.physical_keys().collect::<Vec<_>>(),
            ["model.b", "model.a"]
        );
        let LogicalWeights { weights, .. } =
            read_logical_weights(&path, &plan, LogicalReadMode::Deferred)
                .expect("a reordering remap is not source drift");
        let alpha = crate::weights::to_f32(weights.require("alpha").unwrap()).unwrap();
        assert_eq!(alpha.as_slice::<f32>(), [2.0]);
    }

    #[test]
    fn read_refuses_a_tensor_set_that_changed_since_planning() {
        let dir = fixture_dir();
        let path = dir.path().join("drift.safetensors");
        write_safetensors(&path, &[("model.w", "BF16", &[2], bf16_bytes(&[1.0, 2.0]))]);
        let plan = plan_logical_weights(&path, &StripModel).unwrap();
        write_safetensors(
            &path,
            &[
                ("model.w", "BF16", &[2], bf16_bytes(&[1.0, 2.0])),
                ("model.extra", "BF16", &[1], bf16_bytes(&[3.0])),
            ],
        );
        let error = read_logical_weights(&path, &plan, LogicalReadMode::Deferred)
            .err()
            .expect("a changed tensor set must refuse");
        assert!(
            error
                .to_string()
                .contains("tensor set changed since planning"),
            "{error}"
        );
    }

    #[test]
    fn dense_codec_refuses_a_substituted_dtype() {
        // A plan that claims dense bf16 for a tensor the backend actually loads as f32 must not
        // pass through the codec — the codec guards the representation it reports.
        let dir = fixture_dir();
        let path = dir.path().join("subst.safetensors");
        write_safetensors(
            &path,
            &[("model.w", "F32", &[1], 1.0_f32.to_le_bytes().to_vec())],
        );
        let plan = LogicalWeightPlan {
            mapping_id: "strip-model-test",
            tensors: vec![LogicalTensorPlan {
                logical_key: "w".to_owned(),
                physical_key: "model.w".to_owned(),
                encoding: WeightEncoding::DenseBf16,
                shape: vec![1],
                source_bytes: 4,
                codec_id: DENSE_BF16_CODEC.codec_id,
            }],
            source_bytes: 4,
        };
        let error = read_logical_weights(&path, &plan, LogicalReadMode::Deferred)
            .err()
            .expect("dtype substitution must refuse");
        assert!(
            error.to_string().contains("planned as dense-bf16"),
            "{error}"
        );
    }
}
