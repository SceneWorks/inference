//! Header-only configured-adapter semantic facts (no tensor, model or media is built).
//!
//! Split out of `loader.rs` (sc-27427); every item is re-exported there, so
//! `mlx_gen::adapters::loader::…` paths are unchanged.

use super::*;

/// Header-only semantic graph of the SAME unfused adapter loader used by Anima/Z-Image.
/// The provider supplies its actual pure target namespace/packing facts. No tensor/model/media is
/// built. The only payload access is a bounded alpha CONTROL scalar when a packed LoKr probe's
/// actual scale/disable branch needs it; values/paths never enter the returned semantic identity.
/// This describes configured mechanisms, not physical liveness, a certificate or proof by digest.
pub fn configured_adapter_semantic_facts(
    specs: &[AdapterSpec],
    kohya_paths: &[String],
    lycoris_dtype: Dtype,
    projection: impl FnMut(&str) -> Result<Option<(String, super::super::LinearFacts)>>,
) -> Result<serde_json::Value> {
    Ok(configured_adapter_facts(specs, kohya_paths, lycoris_dtype, projection)?.0)
}

/// The same configured normalization plus typed retained residual facts for source pricing.
/// The typed payloads describe logical owners only; no view/backing or allocator proof is implied.
pub fn configured_adapter_facts(
    specs: &[AdapterSpec],
    kohya_paths: &[String],
    lycoris_dtype: Dtype,
    mut projection: impl FnMut(&str) -> Result<Option<(String, super::super::LinearFacts)>>,
) -> Result<(
    serde_json::Value,
    Vec<(String, super::super::facts::AdapterFacts)>,
)> {
    use super::super::facts::{dtype_name, native_header_dtype, AdapterFacts, TensorFacts};
    let mut retained = Vec::new();
    let mut batch_applied = 0usize;
    use serde_json::json;
    let table = wmeta::kohya_table(kohya_paths);
    let mut files = Vec::new();
    for spec in specs {
        // Same neutral-header gate as the always-enabled real-iOS before_load hook. In particular,
        // C64 is not privately aliased into another dtype merely because native MLX can load it.
        let layout = wmeta::safetensors_file_tensor_locations(&spec.path).map_err(Error::from)?;
        let headers: BTreeMap<String, gen_core::weightsmeta::SafetensorsTensorHeader> = layout
            .tensors
            .iter()
            .map(|location| (location.header.name.clone(), location.header.clone()))
            .collect();
        let locations = layout
            .tensors
            .iter()
            .map(|location| (location.header.name.as_str(), location))
            .collect::<BTreeMap<_, _>>();
        let mut control_file = None;
        let mut alpha_control = |key: &str| -> Result<f32> {
            if control_file.is_none() {
                control_file = Some(std::fs::File::open(&spec.path)?);
            }
            let location = locations
                .get(key)
                .ok_or_else(|| Error::Msg("missing alpha control location".into()))?;
            read_control_scalar_f32_at(
                control_file.as_mut().expect("opened alpha control file"),
                location,
                layout.file_len,
            )
        };
        let metadata = wmeta::safetensors_file_metadata(&spec.path).map_err(Error::from)?;
        let mut sources = Vec::new();
        for header in headers.values() {
            sources.push(json!({"shape": semantic_header_shape(header)?,
                "dtype": dtype_name(native_header_dtype(header.dtype)?)}));
        }
        // Original key spellings and file paths are not graph identities; preserve only the
        // multiset of actually loaded source shapes/native dtypes, including inactive inputs.
        sources.sort_by_key(serde_json::Value::to_string);
        let format = classify_unfused_adapter_format(
            headers.keys().map(String::as_str),
            metadata.get("networkType").map(String::as_str),
            spec.kind,
        );
        let format_name = match format {
            UnfusedAdapterFormat::PeftLora => "peft_lora",
            UnfusedAdapterFormat::KohyaLora => "kohya_lora",
            UnfusedAdapterFormat::PeftLokr => "peft_lokr",
            UnfusedAdapterFormat::ThirdpartyLokr => "thirdparty_lokr",
            UnfusedAdapterFormat::ThirdpartyLoha => "thirdparty_loha",
            UnfusedAdapterFormat::LoraKindMismatch => {
                return Err(Error::Msg(
                    "adapter declared LoRA but native file dispatch selects stamped LoKr".into(),
                ))
            }
        };
        let mut installed = Vec::new();
        if matches!(
            format,
            UnfusedAdapterFormat::PeftLora | UnfusedAdapterFormat::KohyaLora
        ) {
            let prefix =
                wmeta::detect_lora_prefix(headers.keys().map(String::as_str)).unwrap_or("");
            let meta = if format == UnfusedAdapterFormat::PeftLora {
                wmeta::LoraAdapterMeta::from_metadata(
                    metadata
                        .get(wmeta::LORA_ADAPTER_METADATA_KEY)
                        .map(String::as_str),
                )
            } else {
                None
            };
            let mut groups: BTreeMap<String, (Option<String>, Option<String>, bool)> =
                BTreeMap::new();
            let mut claimed = BTreeMap::new();
            for key in headers.keys() {
                let parsed = if format == UnfusedAdapterFormat::KohyaLora {
                    if let Some((stem, role)) = kohya_factor_key(key) {
                        let path = table.get(stem).ok_or_else(|| {
                            Error::Msg(format!("unmatched kohya adapter target {stem}"))
                        })?;
                        Some((
                            path.as_str(),
                            match role {
                                KohyaRole::Down => Some(LoraFactorRole::Down),
                                KohyaRole::Up => Some(LoraFactorRole::Up),
                                KohyaRole::Alpha => None,
                            },
                        ))
                    } else {
                        None
                    }
                } else if let Some((path, role)) = peft_lora_factor(key, prefix) {
                    Some((path, Some(role)))
                } else {
                    key.strip_prefix(prefix)
                        .and_then(|key| key.strip_suffix(".alpha"))
                        .or_else(|| key.strip_suffix(".alpha"))
                        .map(|path| (path, None))
                };
                let Some((path, role)) = parsed else { continue };
                let group = groups.entry(path.to_owned()).or_default();
                if let Some(role) = role {
                    claim_lora_factor(&mut claimed, path, role, key)?;
                    *if role == LoraFactorRole::Down {
                        &mut group.0
                    } else {
                        &mut group.1
                    } = Some(key.clone());
                } else {
                    semantic_nonempty_alpha(&headers[key])?;
                    group.2 = true;
                }
            }
            for (path, (down, up, tensor_alpha)) in groups {
                let (Some(down), Some(up)) = (down, up) else {
                    continue;
                }; // same partial-group skip
                let (canonical, base) = projection(&path)?
                    .ok_or_else(|| Error::Msg(format!("unmatched adapter target {path}")))?;
                let a = semantic_header_shape(&headers[&down])?;
                let b = semantic_header_shape(&headers[&up])?;
                validate_lora_shapes(&path, &a, &b)?;
                let fold = tensor_alpha
                    || meta
                        .as_ref()
                        .is_some_and(|meta| meta.effective(&path).0.is_some());
                if fold {
                    validate_lora_alpha_rank(&path, a[0])?;
                }
                let a_dtype = native_header_dtype(headers[&down].dtype)?;
                let b_dtype = native_header_dtype(headers[&up].dtype)?;
                let b_dtype = if fold {
                    Dtype::from_promoting_types(b_dtype, Dtype::Float32)
                } else {
                    b_dtype
                };
                batch_applied += 1; // Native successful push count, not per-file strictness.
                retained.push((
                    canonical.clone(),
                    AdapterFacts::Lora {
                        a: TensorFacts::materialized_matrix(a[1], a[0], a_dtype),
                        b: TensorFacts::materialized_matrix(b[1], b[0], b_dtype),
                        disabled: spec.scale == 0.0,
                    },
                ));
                installed.push(
                    json!({"projection":canonical,"base_logical_shape":base.base_shape,
                    "kind":"lora","a_shape":[a[1],a[0]],"a_dtype":dtype_name(a_dtype),
                    "b_shape":[b[1],b[0]],"b_dtype":dtype_name(b_dtype),"alpha_fold":fold,
                    "disabled":spec.scale==0.0}),
                );
            }
        } else {
            let loha = format == UnfusedAdapterFormat::ThirdpartyLoha;
            let peft = format == UnfusedAdapterFormat::PeftLokr;
            let suffixes = if loha {
                &wmeta::LOHA_TP_SUFFIXES[..]
            } else if peft {
                &wmeta::LOKR_SUFFIXES[..]
            } else {
                &wmeta::LOKR_TP_SUFFIXES[..]
            };
            let mut groups: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
            for key in headers.keys() {
                if !peft {
                    if let Some(raw) = key.strip_suffix(".alpha") {
                        semantic_nonempty_alpha(&headers[key])?;
                        groups
                            .entry(raw.into())
                            .or_default()
                            .insert("alpha".into(), key.clone());
                        continue;
                    }
                }
                if let Some((raw, role)) = wmeta::split_factor_key(key, suffixes) {
                    groups
                        .entry(raw.into())
                        .or_default()
                        .insert(role.into(), key.clone());
                }
            }
            let mut claimed = BTreeMap::new();
            for (raw, factors) in groups {
                let dotted = if peft {
                    raw.clone()
                } else {
                    wmeta::resolve_lokr_path(&raw, &table)
                        .unwrap_or_else(|| strip_common_lora_prefix(&raw))
                        .to_owned()
                };
                if claimed.insert(dotted.clone(), raw).is_some() {
                    return Err(Error::Msg(
                        "LyCORIS aliases resolve to the same target twice".into(),
                    ));
                }
                let (canonical, base) = projection(&dotted)?
                    .ok_or_else(|| Error::Msg(format!("unmatched adapter target {dotted}")))?;
                let shape = |role: &str| -> Result<Option<Vec<i32>>> {
                    factors
                        .get(role)
                        .map(|key| semantic_header_shape(&headers[key]))
                        .transpose()
                };
                let required = |role: &str| -> Result<Vec<i32>> {
                    shape(role)?.ok_or_else(|| Error::Msg(format!("missing adapter factor {role}")))
                };
                let mut factor_facts = BTreeMap::new();
                for (role, key) in &factors {
                    factor_facts.insert(
                        role.clone(),
                        json!({"shape":semantic_header_shape(&headers[key])?,
                        "dtype":dtype_name(native_header_dtype(headers[key].dtype)?)}),
                    );
                }
                let mut structured_shapes = None;
                let (kind, disabled, branches) = if loha {
                    let a1 = required("hada_w1_a")?;
                    let b1 = required("hada_w1_b")?;
                    let a2 = required("hada_w2_a")?;
                    let b2 = required("hada_w2_b")?;
                    thirdparty_loha_rank(Some(&b1))?;
                    let t1 = shape("hada_t1")?;
                    let t2 = shape("hada_t2")?;
                    let tucker = t1.is_some() && t2.is_some();
                    let (m1, m2) = if let (Some(t1), Some(t2)) = (&t1, &t2) {
                        (
                            semantic_tucker_shape(t1, &a1, &b1)?,
                            semantic_tucker_shape(t2, &a2, &b2)?,
                        )
                    } else {
                        (
                            semantic_matmul_shape(&a1, &b1)?,
                            semantic_matmul_shape(&a2, &b2)?,
                        )
                    };
                    let product = semantic_broadcast_shape(&m1, &m2)?;
                    semantic_reshape_matches(&product, &base.base_shape)?;
                    (
                        "materialized_delta",
                        spec.scale == 0.0,
                        json!({"tucker":tucker}),
                    )
                } else {
                    let w1a = shape("lokr_w1_a")?;
                    let w1b = shape("lokr_w1_b")?;
                    let w2a = shape("lokr_w2_a")?;
                    let w2b = shape("lokr_w2_b")?;
                    let t2 = shape("lokr_t2")?;
                    let rank = if peft {
                        None
                    } else {
                        thirdparty_lokr_rank(w1a.as_deref(), t2.as_deref(), w2a.as_deref())?
                    };
                    let w1 = shape("lokr_w1")?;
                    let f1 = if let Some(full) = &w1 {
                        full.clone()
                    } else {
                        semantic_matmul_shape(
                            w1a.as_deref()
                                .ok_or_else(|| Error::Msg("LoKr w1 missing".into()))?,
                            w1b.as_deref()
                                .ok_or_else(|| Error::Msg("LoKr w1_b missing".into()))?,
                        )?
                    };
                    let w2 = shape("lokr_w2")?;
                    let f2 = if let Some(full) = &w2 {
                        full.clone()
                    } else if let Some(t2) = &t2 {
                        semantic_tucker_shape(
                            t2,
                            w2a.as_deref()
                                .ok_or_else(|| Error::Msg("LoKr w2_a missing".into()))?,
                            w2b.as_deref()
                                .ok_or_else(|| Error::Msg("LoKr w2_b missing".into()))?,
                        )?
                    } else {
                        semantic_matmul_shape(
                            w2a.as_deref()
                                .ok_or_else(|| Error::Msg("LoKr w2 missing".into()))?,
                            w2b.as_deref()
                                .ok_or_else(|| Error::Msg("LoKr w2_b missing".into()))?,
                        )?
                    };
                    // Actual packed probe rejects ANY t2 before choosing full w2. Dense delta
                    // reconstruction instead chooses full w2 first; keep these branches distinct.
                    let structured = base.is_quantized
                        && super::super::lokr_structured_layout(
                            &base.base_shape,
                            &f1,
                            &f2,
                            t2.is_some(),
                        )
                        .is_some();
                    if structured {
                        structured_shapes = Some((f1.clone(), f2.clone()));
                    }
                    let (meta_rank, meta_alpha) = wmeta::parse_rank_alpha(
                        metadata.get("rank").map(String::as_str),
                        metadata.get("alpha").map(String::as_str),
                    );
                    let coefficient = if base.is_quantized {
                        let normalized = if peft {
                            meta_alpha / meta_rank
                        } else {
                            let alpha = if rank.is_some() {
                                factors
                                    .get("alpha")
                                    .map(|key| alpha_control(key))
                                    .transpose()?
                            } else {
                                None
                            };
                            thirdparty_lokr_scale(rank, alpha)
                        };
                        let coefficient = normalized * spec.scale;
                        if !coefficient.is_finite() {
                            return Err(Error::Msg("LoKr: non-finite packed scale".into()));
                        }
                        coefficient
                    } else {
                        spec.scale
                    };
                    if peft && !structured {
                        super::super::validate_lokr_metadata_scale(meta_alpha, meta_rank)?;
                    }
                    let count1 = semantic_elements(&f1)?;
                    let count2 = semantic_elements(&f2)?;
                    let count = count1
                        .checked_mul(count2)
                        .ok_or_else(|| Error::Msg("LoKr shape product overflow".into()))?;
                    if count != semantic_elements(&base.base_shape)? {
                        return Err(Error::Msg(
                            "LoKr reconstruction cannot reshape to its actual target".into(),
                        ));
                    }
                    (
                        if structured {
                            "structured_lokr"
                        } else {
                            "materialized_delta"
                        },
                        if structured {
                            coefficient == 0.0
                        } else {
                            spec.scale == 0.0
                        },
                        json!({"full_w1":w1.is_some(),"full_w2":w2.is_some(),
                            "dense_tucker_active":w2.is_none() && t2.is_some(),
                            "packed_t2_probe_rejected":base.is_quantized && t2.is_some()}),
                    )
                };
                let payload = if let Some((w1, w2)) = structured_shapes {
                    AdapterFacts::Structured {
                        w1: TensorFacts::materialized_matrix(w1[0], w1[1], lycoris_dtype),
                        w2: TensorFacts::materialized_matrix(w2[0], w2[1], lycoris_dtype),
                        disabled,
                    }
                } else {
                    AdapterFacts::Materialized {
                        delta: TensorFacts {
                            shape: base.base_shape.clone(),
                            dtype: lycoris_dtype,
                            strides: Vec::new(),
                        },
                        disabled,
                    }
                };
                batch_applied += 1;
                retained.push((canonical.clone(), payload));
                installed.push(
                    json!({"projection":canonical,"base_logical_shape":base.base_shape,
                    "kind":kind,"retained_dtype":dtype_name(lycoris_dtype),"disabled":disabled,
                    "factors":factor_facts,"branches":branches}),
                );
            }
        }
        // Strict NON-diff autoprefix checks aggregate successful pushes, not each file.
        // Keep orphan/raw sources; all existing unknown-target/kind/shape/scale errors above stay.
        files.push(json!({"format":format_name,"loaded_sources":sources,"installed":installed}));
    }
    if !specs.is_empty() && batch_applied == 0 {
        return Err(Error::Msg(
            "adapter batch matched no actual projection".into(),
        ));
    }
    Ok((
        json!({"schema":"unfused_adapter_semantic_graph_v1","files":files}),
        retained,
    ))
}

pub(super) fn semantic_header_shape(
    header: &gen_core::weightsmeta::SafetensorsTensorHeader,
) -> Result<Vec<i32>> {
    header
        .shape
        .iter()
        .map(|&dim| {
            i32::try_from(dim).map_err(|_| {
                Error::Unsupported("adapter dimension exceeds native i32 shape".into())
            })
        })
        .collect()
}

pub(super) fn semantic_nonempty_alpha(
    header: &gen_core::weightsmeta::SafetensorsTensorHeader,
) -> Result<()> {
    if header.element_count().map_err(Error::from)? == 0 {
        return Err(Error::Msg(
            "scalar_alpha: alpha tensor present but empty".into(),
        ));
    }
    Ok(())
}

pub(super) fn semantic_elements(shape: &[i32]) -> Result<u64> {
    shape.iter().try_fold(1u64, |count, &dim| {
        count
            .checked_mul(dim as u64)
            .ok_or_else(|| Error::Msg("adapter element count overflow".into()))
    })
}

pub(super) fn semantic_reshape_matches(from: &[i32], to: &[i32]) -> Result<()> {
    if semantic_elements(from)? != semantic_elements(to)? {
        return Err(Error::Msg(
            "adapter reconstruction cannot reshape to its actual target".into(),
        ));
    }
    Ok(())
}

pub(super) fn semantic_broadcast_shape(a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
    let mut out = Vec::new();
    for index in 0..a.len().max(b.len()) {
        let x = a.len().checked_sub(index + 1).map_or(1, |i| a[i]);
        let y = b.len().checked_sub(index + 1).map_or(1, |i| b[i]);
        if x != y && x != 1 && y != 1 {
            return Err(Error::Msg("adapter broadcast dimensions disagree".into()));
        }
        out.push(if x == 1 { y } else { x });
    }
    out.reverse();
    Ok(out)
}

pub(super) fn semantic_matmul_shape(a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
    if a.is_empty() || b.is_empty() {
        return Err(Error::Msg("adapter matmul cannot use a scalar".into()));
    }
    let (m, k) = if a.len() == 1 {
        (1, a[0])
    } else {
        (a[a.len() - 2], a[a.len() - 1])
    };
    let (kb, n) = if b.len() == 1 {
        (b[0], 1)
    } else {
        (b[b.len() - 2], b[b.len() - 1])
    };
    if k != kb {
        return Err(Error::Msg(
            "adapter matmul contraction dimensions disagree".into(),
        ));
    }
    let mut out = semantic_broadcast_shape(
        &a[..a.len().saturating_sub(2)],
        &b[..b.len().saturating_sub(2)],
    )?;
    if a.len() != 1 {
        out.push(m);
    }
    if b.len() != 1 {
        out.push(n);
    }
    Ok(out)
}

pub(super) fn semantic_tucker_shape(t: &[i32], a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
    if t.len() != 4 || a.len() != 2 || b.len() != 2 || t[0] != a[0] || t[1] != b[0] {
        return Err(Error::Msg(
            "adapter Tucker dimensions disagree with native contraction".into(),
        ));
    }
    Ok(vec![a[1], b[1], t[2], t[3]])
}
