//! Repository policy and on-disk inventory validation, plus the resident-byte projections.

use super::*;

pub(super) fn repository_policy(
    backend: WanI2vBackend,
    route: WanI2vRoute,
    tier: MemoryNumericTier,
) -> (&'static str, &'static str) {
    match (backend, route, tier.quant) {
        (WanI2vBackend::Mlx, WanI2vRoute::Ti2v5b, _) => (
            "SceneWorks/wan2.2-ti2v-5b-mlx",
            "bb1b055249614cf9d7cf4373fbdbc184b77dee88",
        ),
        (WanI2vBackend::Mlx, WanI2vRoute::I2v14b, _) => (
            "SceneWorks/wan2.2-i2v-a14b-mlx",
            "c6c786170031eccc3a1fac0f98f1ad4ff988271e",
        ),
        (WanI2vBackend::Mlx, WanI2vRoute::T2v14b, _) => (
            "SceneWorks/wan2.2-t2v-a14b-mlx",
            "991eb255c544bbb2e1f1e07da4355c2f0a5337b7",
        ),
        (WanI2vBackend::Candle, WanI2vRoute::Ti2v5b, Some(_)) => (
            "SceneWorks/wan2.2-ti2v-5b-candle",
            "9b173dc8660334a87a11e67de58939afe68f8cb2",
        ),
        (WanI2vBackend::Candle, WanI2vRoute::I2v14b, Some(_)) => (
            "SceneWorks/wan2.2-i2v-a14b-candle",
            "d01bf1ea995c01a5bc545cefb977a320c9cb9fd0",
        ),
        (WanI2vBackend::Candle, WanI2vRoute::T2v14b, Some(_)) => (
            "SceneWorks/wan2.2-t2v-a14b-candle",
            "da1909b66b360e1ea8cdeb3e39e40dca172cfa32",
        ),
        // The manifest pins no revision for any of the three upstream dense snapshots, so the
        // policy names the repository and lets the snapshot id itself be the content pin — see
        // [`UNPINNED_UPSTREAM_REVISION`].
        (WanI2vBackend::Candle, WanI2vRoute::Ti2v5b, None) => (
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            UNPINNED_UPSTREAM_REVISION,
        ),
        (WanI2vBackend::Candle, WanI2vRoute::I2v14b, None) => (
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            UNPINNED_UPSTREAM_REVISION,
        ),
        (WanI2vBackend::Candle, WanI2vRoute::T2v14b, None) => (
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            UNPINNED_UPSTREAM_REVISION,
        ),
        (WanI2vBackend::Mlx, WanI2vRoute::Vace, _) => (
            "Wan-AI/Wan2.1-VACE-1.3B-diffusers+shared-wan-components",
            "artifact-receipt",
        ),
        (WanI2vBackend::Candle, WanI2vRoute::Vace, None) => {
            ("Wan-AI/Wan2.1-VACE-14B-diffusers", "artifact-receipt")
        }
        (WanI2vBackend::Candle, WanI2vRoute::Vace, Some(_)) => (
            "Wan-AI/Wan2.1-VACE-14B-diffusers",
            "unsupported-packed-tier",
        ),
        (WanI2vBackend::Mlx, WanI2vRoute::VaceFun, _) => (
            "linoyts/Wan2.2-VACE-Fun-14B-diffusers+shared-wan-components",
            "artifact-receipt",
        ),
        (WanI2vBackend::Candle, WanI2vRoute::VaceFun, None) => {
            (VACE_FUN_REPOSITORY, "artifact-receipt")
        }
        (WanI2vBackend::Candle, WanI2vRoute::VaceFun, Some(_)) => {
            (VACE_FUN_REPOSITORY, "unsupported-packed-tier")
        }
    }
}

pub(super) fn repo_dir(repository: &str) -> String {
    format!("models--{}", repository.replace('/', "--"))
}

/// Where a converted snapshot for `route` on `backend` must live under an HF-cache-shaped `base`
/// for the dense (bf16) tier to pass [`PreparedWanI2vMemory::prepare`]'s repository/revision check:
/// `<base>/models--<owner>--<name>/snapshots/<pinned revision>[/bf16]`.
///
/// The VACE routes carry an artifact receipt instead of a pinned revision, so any directory is a
/// valid root for them and `base.join("vace")` is returned.
///
/// Test-support only: the backend crates' loaded-contract tests seal a receipt over a tiny synthetic
/// snapshot, and this is the one piece of the layout they cannot express through the public API.
#[doc(hidden)]
pub fn fixture_snapshot_root(base: &Path, backend: WanI2vBackend, route: WanI2vRoute) -> PathBuf {
    if matches!(route, WanI2vRoute::Vace | WanI2vRoute::VaceFun) {
        return base.join("vace");
    }
    let tier = MemoryNumericTier {
        precision: Precision::Bf16,
        quant: None,
        component_precision_floors: &[],
    };
    let (repository, revision) = repository_policy(backend, route, tier);
    let root = base
        .join(repo_dir(repository))
        .join("snapshots")
        .join(fixture_revision(revision));
    match backend {
        WanI2vBackend::Mlx => root.join("bf16"),
        WanI2vBackend::Candle => root,
    }
}

pub(super) fn repository_revision(
    root: &Path,
    repository: &str,
    expected_revision: &str,
    tier: MemoryNumericTier,
    backend: WanI2vBackend,
) -> gen_core::Result<String> {
    let canonical = std::fs::canonicalize(root)?;
    let parts = canonical
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let marker = repo_dir(repository);
    let Some(index) = parts.iter().position(|part| part == &marker) else {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: weights are not from canonical repository {repository}",
            root.display()
        )));
    };
    if parts.get(index + 1).map(String::as_str) != Some("snapshots") {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: canonical repository path is not a snapshot",
            root.display()
        )));
    }
    let revision = parts.get(index + 2).cloned().unwrap_or_default();
    // An unpinned upstream repository still has to be a well-formed HF snapshot; only the
    // *identity* of the snapshot is left to the artifact rather than to this table.
    let revision_matches =
        expected_revision == UNPINNED_UPSTREAM_REVISION || revision == expected_revision;
    if !revision_matches
        || revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: revision does not match immutable {repository}@{expected_revision}",
            root.display()
        )));
    }
    let expected_suffix = match tier.quant {
        Some(Quant::Q4) => Some("q4"),
        Some(Quant::Q8) => Some("q8"),
        None if backend == WanI2vBackend::Mlx => Some("bf16"),
        None => None,
        _ => unreachable!("tier validated"),
    };
    let suffix = parts.get(index + 3).map(String::as_str);
    if suffix != expected_suffix
        || index + 3 + usize::from(expected_suffix.is_some()) != parts.len()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: snapshot suffix does not match selected {:?} tier",
            root.display(),
            tier.quant
        )));
    }
    Ok(revision)
}

pub(super) fn walk_safetensors(root: &Path) -> gen_core::Result<Vec<PathBuf>> {
    fn walk(root: &Path, at: &Path, output: &mut Vec<PathBuf>) -> gen_core::Result<()> {
        for entry in std::fs::read_dir(at)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            let path = entry.path();
            if ty.is_symlink() && std::fs::metadata(&path)?.is_dir() {
                return Err(gen_core::Error::Unsupported(format!(
                    "directory symlink is not permitted in Wan inventory: {}",
                    path.display()
                )));
            }
            if std::fs::metadata(&path)?.is_dir() {
                walk(root, &path, output)?;
            } else if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
                output.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
            }
        }
        Ok(())
    }
    let mut output = Vec::new();
    walk(root, root, &mut output)?;
    output.sort();
    Ok(output)
}

pub(super) fn validate_mlx_inventory(
    root: &Path,
    route: WanI2vRoute,
) -> gen_core::Result<Vec<PathBuf>> {
    if matches!(route, WanI2vRoute::Vace | WanI2vRoute::VaceFun) {
        for required in [
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
        ] {
            if !root.join(required).is_file() {
                return Err(gen_core::Error::Unsupported(format!(
                    "wan_vace: missing required MLX component {required}"
                )));
            }
        }
        if !root.join("transformer/config.json").is_file() {
            return Err(gen_core::Error::Unsupported(
                "wan_vace: missing transformer/config.json".to_owned(),
            ));
        }
        if route == WanI2vRoute::VaceFun && !root.join("transformer_2/config.json").is_file() {
            return Err(gen_core::Error::Unsupported(
                "wan2_2_vace_fun_14b: missing transformer_2/config.json".to_owned(),
            ));
        }
        let actual = walk_safetensors(root)?;
        let has_transformer = actual.iter().any(|relative| {
            relative == Path::new("model.safetensors")
                || relative == Path::new("high_noise_model.safetensors")
                || relative.starts_with(Path::new("transformer"))
        });
        let has_low_expert = route != WanI2vRoute::VaceFun
            || actual.iter().any(|relative| {
                relative == Path::new("low_noise_model.safetensors")
                    || relative.starts_with(Path::new("transformer_2"))
            });
        if !has_transformer
            || !has_low_expert
            || !actual.contains(&PathBuf::from("t5_encoder.safetensors"))
            || !actual.contains(&PathBuf::from("vae.safetensors"))
        {
            return Err(gen_core::Error::Unsupported(
                "wan_vace: incomplete MLX transformer/T5/VAE tensor inventory".to_owned(),
            ));
        }
        return Ok(actual);
    }
    let expected: &[&str] = match route {
        WanI2vRoute::Ti2v5b => &[
            "model.safetensors",
            "t5_encoder.safetensors",
            "vae.safetensors",
        ],
        WanI2vRoute::T2v14b | WanI2vRoute::I2v14b => &[
            "high_noise_model.safetensors",
            "low_noise_model.safetensors",
            "t5_encoder.safetensors",
            "vae.safetensors",
        ],
        WanI2vRoute::Vace => unreachable!("handled above"),
        WanI2vRoute::VaceFun => unreachable!("handled above"),
    };
    for required in ["config.json", "tokenizer.json"] {
        if !root.join(required).is_file() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: missing required direct file {required}",
                route.provider_id()
            )));
        }
    }
    let actual = walk_safetensors(root)?;
    let expected = expected.iter().map(PathBuf::from).collect::<Vec<_>>();
    if actual != expected {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: incomplete or extra MLX tensor inventory (expected {expected:?}, got {actual:?})",
            route.provider_id()
        )));
    }
    Ok(actual)
}

pub(super) fn validate_component_inventory(
    root: &Path,
    component: &str,
) -> gen_core::Result<Vec<PathBuf>> {
    let dir = root.join(component);
    if !dir.join("config.json").is_file() {
        return Err(gen_core::Error::Unsupported(format!(
            "Wan Candle inventory is missing {component}/config.json"
        )));
    }
    let actual = walk_safetensors(&dir)?;
    if actual.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "Wan Candle inventory has no tensors under {component}/"
        )));
    }
    let indexes = std::fs::read_dir(&dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.ends_with(".safetensors.index.json"))
        })
        .collect::<Vec<_>>();
    if indexes.len() > 1 {
        return Err(gen_core::Error::Unsupported(format!(
            "Wan Candle {component}/ has multiple shard indexes"
        )));
    }
    if let Some(index) = indexes.first() {
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(index)?).map_err(|error| {
                gen_core::Error::Unsupported(format!(
                    "invalid shard index {}: {error}",
                    index.display()
                ))
            })?;
        let expected = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                gen_core::Error::Unsupported(format!("invalid shard index {}", index.display()))
            })?
            .values()
            .filter_map(serde_json::Value::as_str)
            .map(PathBuf::from)
            .collect::<BTreeSet<_>>();
        let actual_set = actual.iter().cloned().collect::<BTreeSet<_>>();
        if expected != actual_set {
            return Err(gen_core::Error::Unsupported(format!(
                "Wan Candle {component}/ shard index does not cover its exact tensor inventory"
            )));
        }
    } else if actual.len() != 1 {
        return Err(gen_core::Error::Unsupported(format!(
            "Wan Candle {component}/ has multiple tensors without one authoritative shard index"
        )));
    }
    Ok(actual
        .into_iter()
        .map(|relative| PathBuf::from(component).join(relative))
        .collect())
}

pub(super) fn validate_candle_inventory(
    root: &Path,
    route: WanI2vRoute,
) -> gen_core::Result<Vec<PathBuf>> {
    if !root.join("tokenizer/tokenizer.json").is_file()
        || (!matches!(route, WanI2vRoute::Vace | WanI2vRoute::VaceFun)
            && !root.join("model_index.json").is_file())
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: incomplete Candle model/tokenizer inventory",
            route.provider_id()
        )));
    }
    let components: &[&str] = match route {
        WanI2vRoute::Ti2v5b => &["text_encoder", "transformer", "vae"],
        WanI2vRoute::T2v14b | WanI2vRoute::I2v14b => {
            &["text_encoder", "transformer", "transformer_2", "vae"]
        }
        WanI2vRoute::Vace => &["text_encoder", "transformer", "vae"],
        WanI2vRoute::VaceFun => &["text_encoder", "transformer", "transformer_2", "vae"],
    };
    let mut inventory = Vec::new();
    for component in components {
        inventory.extend(validate_component_inventory(root, component)?);
    }
    let actual = walk_safetensors(root)?;
    inventory.sort();
    if actual != inventory {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: incomplete or extra Candle tensor inventory",
            route.provider_id()
        )));
    }
    Ok(inventory)
}

pub(super) fn structural_files(
    root: &Path,
    backend: WanI2vBackend,
    route: WanI2vRoute,
) -> gen_core::Result<Vec<PathBuf>> {
    if matches!(route, WanI2vRoute::Vace | WanI2vRoute::VaceFun) {
        let mut files = match backend {
            WanI2vBackend::Mlx => vec![
                PathBuf::from("transformer/config.json"),
                PathBuf::from("tokenizer.json"),
            ],
            WanI2vBackend::Candle => vec![
                PathBuf::from("transformer/config.json"),
                PathBuf::from("text_encoder/config.json"),
                PathBuf::from("vae/config.json"),
                PathBuf::from("tokenizer/tokenizer.json"),
            ],
        };
        if route == WanI2vRoute::VaceFun {
            files.push(PathBuf::from("transformer_2/config.json"));
            files.push(PathBuf::from(VACE_FUN_SOURCE_RECEIPT_FILE));
        } else {
            files.push(PathBuf::from(VACE_SOURCE_RECEIPT_FILE));
        }
        fn collect_indexes(
            root: &Path,
            at: &Path,
            files: &mut Vec<PathBuf>,
        ) -> gen_core::Result<()> {
            for entry in std::fs::read_dir(at)? {
                let path = entry?.path();
                if std::fs::metadata(&path)?.is_dir() {
                    collect_indexes(root, &path, files)?;
                } else if path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|name| name.ends_with(".safetensors.index.json"))
                {
                    files.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
                }
            }
            Ok(())
        }
        collect_indexes(root, root, &mut files)?;
        files.sort();
        files.dedup();
        if let Some(missing) = files.iter().find(|relative| !root.join(relative).is_file()) {
            return Err(gen_core::Error::Unsupported(format!(
                "missing Wan VACE structural file {}",
                missing.display()
            )));
        }
        return Ok(files);
    }
    let mut files = match backend {
        WanI2vBackend::Mlx => vec![
            PathBuf::from("config.json"),
            PathBuf::from("tokenizer.json"),
        ],
        WanI2vBackend::Candle => {
            let components: &[&str] = match route {
                WanI2vRoute::Ti2v5b => &["text_encoder", "transformer", "vae"],
                WanI2vRoute::T2v14b | WanI2vRoute::I2v14b => {
                    &["text_encoder", "transformer", "transformer_2", "vae"]
                }
                WanI2vRoute::Vace => unreachable!("handled above"),
                WanI2vRoute::VaceFun => unreachable!("handled above"),
            };
            let mut paths = vec![
                PathBuf::from("model_index.json"),
                PathBuf::from("tokenizer/tokenizer.json"),
            ];
            paths.extend(
                components
                    .iter()
                    .map(|component| PathBuf::from(component).join("config.json")),
            );
            paths
        }
    };
    if backend == WanI2vBackend::Candle {
        fn find_indexes(root: &Path, at: &Path, output: &mut Vec<PathBuf>) -> gen_core::Result<()> {
            for entry in std::fs::read_dir(at)? {
                let path = entry?.path();
                if std::fs::metadata(&path)?.is_dir() {
                    find_indexes(root, &path, output)?;
                } else if path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|name| name.ends_with(".safetensors.index.json"))
                {
                    output.push(path.strip_prefix(root).unwrap_or(&path).to_path_buf());
                }
            }
            Ok(())
        }
        find_indexes(root, root, &mut files)?;
        if let Some(quant) = match root.file_name().and_then(|name| name.to_str()) {
            Some("q4") => Some(Quant::Q4),
            Some("q8") => Some(Quant::Q8),
            _ => None,
        } {
            let components: &[&str] = match route {
                WanI2vRoute::Ti2v5b => &["transformer"],
                WanI2vRoute::T2v14b | WanI2vRoute::I2v14b => &["transformer", "transformer_2"],
                WanI2vRoute::Vace => unreachable!("handled above"),
                WanI2vRoute::VaceFun => unreachable!("handled above"),
            };
            for component in components {
                let marker = PathBuf::from(component).join("quantize_config.json");
                let value: serde_json::Value = serde_json::from_slice(&std::fs::read(
                    root.join(&marker),
                )?)
                .map_err(|error| {
                    gen_core::Error::Unsupported(format!(
                        "invalid Wan packed marker {}: {error}",
                        marker.display()
                    ))
                })?;
                let bits = value.get("bits").and_then(serde_json::Value::as_u64);
                let group = value
                    .get("quantization")
                    .and_then(|value| value.get("group_size"))
                    .and_then(serde_json::Value::as_u64);
                if bits != Some(quant.bits() as u64) || group != Some(64) {
                    return Err(gen_core::Error::Unsupported(format!(
                        "Wan packed marker {} does not match the selected Q{} group-64 tier",
                        marker.display(),
                        quant.bits()
                    )));
                }
                files.push(marker);
            }
        }
    }
    files.sort();
    files.dedup();
    if let Some(missing) = files.iter().find(|relative| !root.join(relative).is_file()) {
        return Err(gen_core::Error::Unsupported(format!(
            "missing Wan structural file {}",
            missing.display()
        )));
    }
    Ok(files)
}

pub(super) fn component_for(route: WanI2vRoute, relative: &str) -> MemoryPhase {
    if relative.contains("t5_encoder") || relative.starts_with("text_encoder/") {
        MemoryPhase::Conditioning
    } else if relative.contains("vae") {
        MemoryPhase::Decode
    } else {
        let _ = route;
        MemoryPhase::Denoise
    }
}

pub(super) fn tensor_elements(
    path: &Path,
    header: &gen_core::weightsmeta::SafetensorsTensorHeader,
) -> gen_core::Result<u64> {
    let elements = header.shape.iter().try_fold(1_u64, |total, &dimension| {
        total
            .checked_mul(u64::try_from(dimension).map_err(|_| {
                gen_core::Error::Msg(format!(
                    "{} tensor shape is not representable",
                    path.display()
                ))
            })?)
            .ok_or_else(|| {
                gen_core::Error::Msg(format!("{} tensor shape overflows", path.display()))
            })
    })?;
    let stored = elements
        .checked_mul(header.dtype.size() as u64)
        .ok_or_else(|| gen_core::Error::Msg(format!("{} tensor bytes overflow", path.display())))?;
    if stored != header.data_bytes || stored == 0 {
        return Err(gen_core::Error::Unsupported(format!(
            "{} tensor {} has crossed dtype/shape/data geometry",
            path.display(),
            header.name
        )));
    }
    Ok(elements)
}

pub(super) fn projected_dense_bytes(path: &Path, resident_width: u64) -> gen_core::Result<u64> {
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    if headers.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{} contains no tensors",
            path.display()
        )));
    }
    headers.into_iter().try_fold(0_u64, |sum, header| {
        if !header.is_float()
            || header.name.ends_with(".scales")
            || header.name.ends_with(".biases")
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{} dense tensor {} has unsupported packed/non-float storage",
                path.display(),
                header.name
            )));
        }
        let bytes = tensor_elements(path, &header)?
            .checked_mul(resident_width)
            .ok_or_else(|| {
                gen_core::Error::Msg("Wan projected tensor bytes overflow".to_owned())
            })?;
        sum.checked_add(bytes)
            .ok_or_else(|| gen_core::Error::Msg("Wan tensor byte total overflow".to_owned()))
    })
}

pub(super) fn stored_tensor_bytes(path: &Path) -> gen_core::Result<u64> {
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    if headers.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{} contains no tensors",
            path.display()
        )));
    }
    headers.into_iter().try_fold(0_u64, |total, header| {
        tensor_elements(path, &header).and_then(|_| {
            total
                .checked_add(header.data_bytes)
                .ok_or_else(|| gen_core::Error::Msg("Wan stored tensor bytes overflow".to_owned()))
        })
    })
}

pub(super) fn packed_transformer_bytes(
    path: &Path,
    quant: Quant,
    backend: WanI2vBackend,
) -> gen_core::Result<u64> {
    const GROUP: usize = 64;
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    let tensors = headers
        .into_iter()
        .map(|header| (header.name.clone(), header))
        .collect::<BTreeMap<_, _>>();
    let mut packed_bases = BTreeSet::new();
    let mut packed = 0_u64;
    for weight in tensors
        .values()
        .filter(|header| header.name.ends_with(".weight") && header.shape.len() == 2)
    {
        let base = weight.name.strip_suffix(".weight").expect("suffix checked");
        // A 2-D `.weight` that is stored float AND carries no quant metadata at all is dense by
        // pipeline convention, not a packed tensor that lost its leaves: the converter leaves the
        // output projection (`head.head`) dense BF16 on every packed tier of the shipped rehosts
        // (`SceneWorks/wan2.2-i2v-a14b-mlx`, `wan2.2-t2v-a14b-mlx`). Skip it here and let the dense
        // residual pass below price it at the resident float width, exactly as it already prices
        // every non-2-D dense tensor.
        //
        // The discriminator is deliberately conjunctive so genuine corruption still fails closed:
        // a `U32` weight is unambiguously packed storage and must carry both leaves, and a weight
        // with a lone `.scales` or `.biases` sibling is a packed tensor missing metadata. Either
        // case falls through to the errors below.
        let has_scales = tensors.contains_key(&format!("{base}.scales"));
        let has_biases = tensors.contains_key(&format!("{base}.biases"));
        if weight.is_float() && !has_scales && !has_biases {
            continue;
        }
        let scales = tensors.get(&format!("{base}.scales")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!("{} packed {base} lacks scales", path.display()))
        })?;
        let biases = tensors.get(&format!("{base}.biases")).ok_or_else(|| {
            gen_core::Error::Unsupported(format!("{} packed {base} lacks biases", path.display()))
        })?;
        if weight.dtype != gen_core::weightsmeta::Dtype::U32
            || !scales.is_float()
            || !biases.is_float()
            || scales.shape != biases.shape
            || scales.shape.len() != 2
            || scales.shape.first() != weight.shape.first()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{} packed {base} has invalid typed geometry",
                path.display()
            )));
        }
        let input = scales.shape[1]
            .checked_mul(GROUP)
            .ok_or_else(|| gen_core::Error::Msg("Wan packed input overflow".to_owned()))?;
        let packed_bits = weight.shape[1]
            .checked_mul(32)
            .and_then(|bits| bits.checked_div(input))
            .ok_or_else(|| {
                gen_core::Error::Unsupported("Wan packed width is inconsistent".to_owned())
            })?;
        if input == 0 || packed_bits != quant.bits() as usize {
            return Err(gen_core::Error::Unsupported(format!(
                "{} packed {base} does not encode Q{} group-64",
                path.display(),
                quant.bits()
            )));
        }
        let elements = u64::try_from(weight.shape[0])
            .ok()
            .and_then(|rows| {
                u64::try_from(input)
                    .ok()
                    .and_then(|cols| rows.checked_mul(cols))
            })
            .ok_or_else(|| gen_core::Error::Msg("Wan packed element count overflow".to_owned()))?;
        if !elements.is_multiple_of(32) {
            return Err(gen_core::Error::Unsupported(
                "Wan packed element count is not block aligned".to_owned(),
            ));
        }
        let bytes = match backend {
            WanI2vBackend::Mlx => weight
                .data_bytes
                .checked_add(scales.data_bytes)
                .and_then(|sum| sum.checked_add(biases.data_bytes)),
            WanI2vBackend::Candle => elements
                .checked_div(32)
                .and_then(|blocks| blocks.checked_mul(if quant == Quant::Q4 { 20 } else { 34 })),
        }
        .ok_or_else(|| gen_core::Error::Msg("Wan packed resident bytes overflow".to_owned()))?;
        packed = packed
            .checked_add(bytes)
            .ok_or_else(|| gen_core::Error::Msg("Wan packed resident total overflow".to_owned()))?;
        packed_bases.insert(base.to_owned());
    }
    if packed_bases.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{} has no packed transformer weights",
            path.display()
        )));
    }
    let dense = tensors.values().try_fold(0_u64, |total, header| {
        if header.name.ends_with(".scales") || header.name.ends_with(".biases") {
            let base = header
                .name
                .strip_suffix(".scales")
                .or_else(|| header.name.strip_suffix(".biases"))
                .expect("suffix checked");
            if !packed_bases.contains(base) {
                return Err(gen_core::Error::Unsupported(format!(
                    "{} has orphan packed leaf {}",
                    path.display(),
                    header.name
                )));
            }
            return Ok(total);
        }
        if header
            .name
            .strip_suffix(".weight")
            .is_some_and(|base| packed_bases.contains(base))
        {
            return Ok(total);
        }
        if !header.is_float() {
            return Err(gen_core::Error::Unsupported(format!(
                "{} has unaccounted non-float tensor {}",
                path.display(),
                header.name
            )));
        }
        let bytes = tensor_elements(path, header)?
            .checked_mul(2)
            .ok_or_else(|| gen_core::Error::Msg("Wan dense residual bytes overflow".to_owned()))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| gen_core::Error::Msg("Wan dense residual total overflow".to_owned()))
    })?;
    packed
        .checked_add(dense)
        .ok_or_else(|| gen_core::Error::Msg("Wan transformer resident bytes overflow".to_owned()))
}

pub(super) fn physical_resident_bytes(
    path: &Path,
    backend: WanI2vBackend,
    route: WanI2vRoute,
    tier: MemoryNumericTier,
    phase: MemoryPhase,
) -> gen_core::Result<u64> {
    if matches!(route, WanI2vRoute::Vace | WanI2vRoute::VaceFun) && phase == MemoryPhase::Denoise {
        // VACE's public q4/q8 tiers are selected at load but materialized from the sealed dense
        // diffusers transformer after adapter folding. Charge the complete dense construction
        // surface rather than pretending the on-disk artifact is prepacked.
        //
        // SC-22667: this arm is scoped to the **denoise** component, which is the only one the
        // sentence above is about. It used to return for every phase of a VACE route, which routed
        // the route's decoder past the `Decode` arm below and priced a VAE that `model_vace.rs` /
        // `model_vace_fun.rs` open at `VAE_DTYPE = DType::F32` as if it were materialized bf16 —
        // half the bytes the render actually holds.
        return projected_dense_bytes(path, 2);
    }
    match (phase, tier.quant) {
        (MemoryPhase::Denoise, Some(quant)) => packed_transformer_bytes(path, quant, backend),
        // The UMT5 encoder is packed at load on every MLX-affine quantized tier (sc-12831): the
        // seven attention/FFN projections per block go to the Q8 floor `model::effective_te_quant`
        // resolves, while the token embedding, the per-block norms and position-bias tables and the
        // final norm stay dense. Charging the dense bf16 projection stack here — which is what the
        // `_` arm below did — over-declared the largest staged component of the 5B and 14B routes
        // by the whole packing win (measured 11.83 -> 7.72 GiB on the 5B).
        (MemoryPhase::Conditioning, Some(quant))
            if backend == WanI2vBackend::Mlx && matches!(quant, Quant::Q4 | Quant::Q8) =>
        {
            mlx_text_encoder_bytes(path, Some(MLX_TE_QUANT_BITS))
        }
        // The dense MLX tier keeps the seven projections bf16, but `Umt5Block::from_weights` and
        // `assemble` upcast the same leaves on BOTH paths (`text_encoder.rs`: `norm1`, `norm2`, the
        // per-layer `pos_embedding` table and the final `norm` are `as_dtype(Float32)` regardless of
        // quant), so the dense arm prices those four leaves f32 too (SC-22667 review).
        (MemoryPhase::Conditioning, None) if backend == WanI2vBackend::Mlx => {
            mlx_text_encoder_bytes(path, None)
        }
        (MemoryPhase::Decode, _) => projected_dense_bytes(
            path,
            if backend == WanI2vBackend::Candle && candle_decodes_f32(route) {
                4
            } else {
                2
            },
        ),
        _ => projected_dense_bytes(path, 2),
    }
}

/// Whether the **candle** provider for `route` opens its video autoencoder f32.
///
/// Three of the four do — `lib.rs` (TI2V-5B), `model_vace.rs` and `model_vace_fun.rs` all pin
/// `VAE_DTYPE = DType::F32` — while `wan14b.rs` pins bf16 for the I2V-14B route (sc-12818). The
/// split is per crate module rather than per family, so it is enumerated rather than derived.
pub(super) fn candle_decodes_f32(route: WanI2vRoute) -> bool {
    match route {
        WanI2vRoute::Ti2v5b | WanI2vRoute::Vace | WanI2vRoute::VaceFun => true,
        // `wan14b.rs` is the module behind BOTH A14B routes, and it pins bf16 for both.
        WanI2vRoute::T2v14b | WanI2vRoute::I2v14b => false,
    }
}
