//! Multi-phase Krea denoise primitive (epic 13879): one trajectory over one global sigma schedule.
//!
//! The host math is backend-neutral and lives in [`gen_core_family::krea_multiphase`] (sc-27329), shared
//! with the Candle twin; this module re-exports it so `multiphase::*` paths stay stable.

pub use gen_core_family::krea_multiphase::*;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use mlx_gen::adapters::loader::{
    classify_unfused_adapter_format, kohya_factor_key, peft_lora_factor, KohyaRole, LoraFactorRole,
    UnfusedAdapterFormat,
};
use mlx_gen::gen_core::weightsmeta::{
    safetensors_file_metadata, safetensors_file_tensor_locations, split_factor_key,
    SafetensorsTensorHeader, LOHA_TP_SUFFIXES, LOKR_TP_SUFFIXES,
};
use mlx_gen::gen_core::{AdapterKind, Error, GenerationRequest, LoadSpec, Result};

#[derive(Clone, Copy, Default)]
struct AdapterPhaseFootprint {
    source: u64,
    derived: u64,
    install_scratch: u64,
}

fn price_overflow() -> Error {
    Error::Unsupported("Krea phase adapter memory size overflow".into())
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0u64, |total, bytes| {
        total.checked_add(bytes).ok_or_else(price_overflow)
    })
}

fn checked_product(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(1u64, |total, dim| {
        total.checked_mul(dim).ok_or_else(price_overflow)
    })
}

fn broadcast_dimensions<'a>(
    left: &'a [usize],
    right: &'a [usize],
) -> impl Iterator<Item = usize> + 'a {
    let width = left.len().max(right.len());
    (0..width).rev().map(move |index| {
        let left = left.iter().rev().nth(index).copied().unwrap_or(1);
        let right = right.iter().rev().nth(index).copied().unwrap_or(1);
        left.max(right)
    })
}

fn matrix_product_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let [left_batch @ .., rows, _] = left else {
        return Err(Error::Unsupported(
            "Krea adapter matrix factor needs two axes".into(),
        ));
    };
    let [right_batch @ .., _, cols] = right else {
        return Err(Error::Unsupported(
            "Krea adapter matrix factor needs two axes".into(),
        ));
    };
    Ok(broadcast_dimensions(left_batch, right_batch)
        .chain([*rows, *cols])
        .collect())
}

fn element_count(shape: &[usize]) -> Result<u64> {
    checked_product(shape.iter().map(|&dimension| dimension as u64))
}

fn tucker_sizes(tensor: &[usize], a: &[usize], b: &[usize]) -> Result<([usize; 4], u64)> {
    let [rank_a, rank_b, height, width] = tensor else {
        return Err(Error::Unsupported(
            "Krea LyCORIS Tucker factor needs four axes".into(),
        ));
    };
    let (Some(&output), Some(&input)) = (a.get(1), b.get(1)) else {
        return Err(Error::Unsupported(
            "Krea LyCORIS Tucker matrix needs two axes".into(),
        ));
    };
    let shape = [output, input, *height, *width];
    // A contraction cannot retain more elements than the full product over its six labels.
    let temporary = checked_product([element_count(&shape)?, *rank_a as u64, *rank_b as u64])?;
    Ok((shape, temporary))
}

fn lycoris_footprint(
    groups: BTreeMap<&str, BTreeMap<&str, &SafetensorsTensorHeader>>,
    loha: bool,
) -> Result<(u64, u64)> {
    let mut derived = 0_u64;
    let mut install_scratch = 0_u64;
    for factors in groups.into_values() {
        let shape = |key| {
            factors
                .get(key)
                .map(|factor| factor.shape.as_slice())
                .ok_or_else(|| {
                    Error::Unsupported(format!("Krea LyCORIS adapter source is missing {key}"))
                })
        };
        // MLX LyCORIS reconstruction keeps file precision until the final cast. Use at least
        // F32 for promoted operands; the retained Krea residual uses the default BF16 dtype.
        let work_width = factors.values().try_fold(4u64, |width, factor| {
            let elements = factor.element_count()?.max(1);
            Ok::<_, Error>(width.max(factor.data_bytes.div_ceil(elements)))
        })?;
        let (retained_elements, scratch_elements) = if loha {
            let a = shape("hada_w1_a")?;
            let b = shape("hada_w1_b")?;
            let c = shape("hada_w2_a")?;
            let d = shape("hada_w2_b")?;
            let (first, second, contraction) =
                if let (Some(t1), Some(t2)) = (factors.get("hada_t1"), factors.get("hada_t2")) {
                    let (first, first_work) = tucker_sizes(&t1.shape, a, b)?;
                    let (second, second_work) = tucker_sizes(&t2.shape, c, d)?;
                    (first.to_vec(), second.to_vec(), first_work.max(second_work))
                } else {
                    (matrix_product_shape(a, b)?, matrix_product_shape(c, d)?, 0)
                };
            let output = checked_product(broadcast_dimensions(&first, &second).map(|d| d as u64))?;
            let first = element_count(&first)?;
            let second = element_count(&second)?;
            // Both products, the Hadamard product and the scaled result can coexist.
            (
                output,
                checked_sum([first, second, output, output, contraction])?,
            )
        } else {
            let first = match factors.get("lokr_w1") {
                Some(factor) => factor.element_count()?,
                None => element_count(&matrix_product_shape(
                    shape("lokr_w1_a")?,
                    shape("lokr_w1_b")?,
                )?)?,
            };
            let (second, contraction) = match (factors.get("lokr_w2"), factors.get("lokr_t2")) {
                (Some(factor), _) => (factor.element_count()?, 0),
                (None, Some(t2)) => {
                    let a = shape("lokr_w2_a")?;
                    let b = shape("lokr_w2_b")?;
                    let (shape, work) = tucker_sizes(&t2.shape, a, b)?;
                    (element_count(&shape)?, work)
                }
                (None, None) => (
                    element_count(&matrix_product_shape(
                        shape("lokr_w2_a")?,
                        shape("lokr_w2_b")?,
                    )?)?,
                    0,
                ),
            };
            let dense = checked_product([first, second])?;
            let structured = checked_sum([first, second])?;
            (
                dense.max(structured),
                checked_sum([structured, dense, dense, contraction])?,
            )
        };
        derived = checked_sum([derived, checked_product([retained_elements, 2])?])?;
        // Adapter payloads materialize one target at a time. Other outputs stay live, but
        // temporary reconstruction does not accumulate across targets.
        install_scratch = install_scratch.max(checked_product([scratch_elements, work_width])?);
    }
    Ok((derived, install_scratch))
}

fn file_footprint(path: &Path, kind: AdapterKind) -> Result<AdapterPhaseFootprint> {
    let layout = safetensors_file_tensor_locations(path)?;
    let metadata = safetensors_file_metadata(path)?;
    let headers: Vec<_> = layout.tensors.iter().map(|tensor| &tensor.header).collect();
    let format = classify_unfused_adapter_format(
        headers.iter().map(|header| header.name.as_str()),
        metadata.get("networkType").map(String::as_str),
        kind,
    );
    let (derived, install_scratch) = match format {
        UnfusedAdapterFormat::KohyaLora | UnfusedAdapterFormat::PeftLora => {
            // The transposed A/B views normally share the source. A per-module alpha or PEFT
            // alpha pattern can allocate one normalized B at promoted precision.
            let mut bytes = 0;
            for header in headers.iter().filter(|header| {
                if format == UnfusedAdapterFormat::KohyaLora {
                    matches!(kohya_factor_key(&header.name), Some((_, KohyaRole::Up)))
                } else {
                    matches!(
                        peft_lora_factor(&header.name, ""),
                        Some((_, LoraFactorRole::Up))
                    )
                }
            }) {
                bytes = checked_sum([bytes, header.materialized_bytes(4)?.max(header.data_bytes)])?;
            }
            (bytes, 0)
        }
        UnfusedAdapterFormat::PeftLokr
        | UnfusedAdapterFormat::ThirdpartyLokr
        | UnfusedAdapterFormat::ThirdpartyLoha => {
            let loha = format == UnfusedAdapterFormat::ThirdpartyLoha;
            let suffixes: &[&str] = if loha {
                &LOHA_TP_SUFFIXES
            } else {
                &LOKR_TP_SUFFIXES
            };
            let mut groups = BTreeMap::<_, BTreeMap<_, _>>::new();
            for header in headers {
                if let Some((module, factor)) = split_factor_key(&header.name, suffixes) {
                    groups.entry(module).or_default().insert(factor, header);
                }
            }
            lycoris_footprint(groups, loha)?
        }
        UnfusedAdapterFormat::LoraKindMismatch => {
            return Err(Error::Unsupported(
                "Krea adapter kind does not match its header".into(),
            ));
        }
    };
    Ok(AdapterPhaseFootprint {
        source: layout.file_len,
        derived,
        install_scratch,
    })
}

/// Additional unified-memory reservation above the existing default adapter source-byte charge.
/// The provider retains immutable raw factors for selected files across phases. A phase holds only
/// its distinct-strength derived variants and releases them before the next phase. The ordinary
/// resident DiT's installed default stack remains live in either snapshot or native-file residency.
pub fn adapter_phase_additional_bytes(spec: &LoadSpec, req: &GenerationRequest) -> Result<u64> {
    let Some(phases) = req.phases.as_ref() else {
        return Ok(0);
    };
    let footprints = spec
        .adapters
        .iter()
        .map(|adapter| {
            spec.read_file_unchanged_if_prepared(&adapter.path, |path| {
                file_footprint(path, adapter.kind)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let default = checked_sum(footprints.iter().map(|slot| slot.derived))?;
    let default_scratch = footprints
        .iter()
        .map(|slot| slot.install_scratch)
        .max()
        .unwrap_or(0);
    let mut retained_sources = BTreeMap::new();
    let mut active_peak = 0;
    for phase in phases {
        let mut keys = BTreeSet::new();
        let mut active_derived = 0;
        let mut active_scratch = 0;
        for selected in &phase.adapters {
            let adapter = spec.adapters.get(selected.adapter).ok_or_else(|| {
                Error::Unsupported("Krea adapter phase slot is outside the loaded stack".into())
            })?;
            let slot = footprints[selected.adapter];
            retained_sources.insert(&adapter.path, slot.source);
            let scale_bits = selected.weight.unwrap_or(adapter.scale).to_bits();
            if keys.insert((selected.adapter, scale_bits)) {
                active_derived = checked_sum([active_derived, slot.derived])?;
                active_scratch = active_scratch.max(slot.install_scratch);
            }
        }
        active_peak = active_peak.max(checked_sum([active_derived, active_scratch])?);
    }
    checked_sum([
        checked_sum(retained_sources.into_values())?,
        default,
        default_scratch.max(active_peak),
    ])
}

#[cfg(test)]
mod pricing_tests {
    use super::*;
    use mlx_gen::gen_core::{AdapterSpec, GenerationPhase, PhaseAdapter, WeightsSource};

    fn write_header(path: &Path, rows: &[(&str, &[usize])]) -> u64 {
        let mut tensors = serde_json::Map::new();
        let mut payload = 0;
        for (name, shape) in rows {
            let end = payload + 4 * shape.iter().product::<usize>();
            tensors.insert(
                (*name).to_owned(),
                serde_json::json!({"dtype":"F32","shape":shape,"data_offsets":[payload,end]}),
            );
            payload = end;
        }
        let mut header = serde_json::to_vec(&tensors).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + payload, 0);
        std::fs::write(path, &bytes).unwrap();
        bytes.len() as u64
    }

    #[test]
    fn charges_raw_sources_default_deltas_and_only_the_peak_distinct_phase() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("loha.safetensors");
        let file_bytes = write_header(
            &file,
            &[
                ("projector.hada_w1_a", &[3, 2]),
                ("projector.hada_w1_b", &[2, 5]),
                ("projector.hada_w2_a", &[3, 2]),
                ("projector.hada_w2_b", &[2, 5]),
            ],
        );
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().into()))
            .with_adapters(vec![AdapterSpec::new(file, 1.0, AdapterKind::Lora)]);
        let strength = |value| PhaseAdapter {
            adapter: 0,
            weight: Some(value),
        };
        let request = GenerationRequest {
            phases: Some(vec![
                GenerationPhase {
                    steps: 1,
                    adapters: vec![strength(0.5), strength(0.5), strength(0.7)],
                    ..Default::default()
                },
                GenerationPhase {
                    steps: 1,
                    adapters: vec![strength(0.8)],
                    ..Default::default()
                },
            ]),
            ..Default::default()
        };
        // One raw source + one ordinary BF16 3x5 delta + two distinct phase deltas.
        // The current LoHa reconstruction can hold four F32 3x5 matrices.
        assert_eq!(
            adapter_phase_additional_bytes(&spec, &request).unwrap(),
            file_bytes + 30 + 60 + 240,
        );
        assert_eq!(
            adapter_phase_additional_bytes(&spec, &GenerationRequest::default()).unwrap(),
            0,
        );
    }

    #[test]
    fn loha_broadcast_expansion_is_priced_from_matrix_dimensions() {
        let rows = [
            ("hada_w1_a", vec![3, 2]),
            ("hada_w1_b", vec![2, 1]),
            ("hada_w2_a", vec![1, 2]),
            ("hada_w2_b", vec![2, 5]),
        ];
        let headers: Vec<_> = rows
            .into_iter()
            .map(|(name, shape)| SafetensorsTensorHeader {
                name: name.into(),
                dtype: mlx_gen::gen_core::weightsmeta::Dtype::F32,
                data_bytes: 4 * shape.iter().product::<usize>() as u64,
                shape,
            })
            .collect();
        let factors = headers
            .iter()
            .map(|header| (header.name.as_str(), header))
            .collect();
        let (retained, scratch) =
            lycoris_footprint(BTreeMap::from([("module", factors)]), true).unwrap();
        assert_eq!(retained, 2 * 3 * 5);
        assert_eq!(scratch, 4 * (3 + 5 + 2 * 3 * 5));
    }
}
