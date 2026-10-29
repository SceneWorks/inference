//! Bounded source discovery and resolution helpers: path confinement, shard and tokenizer resolution, config reads.

use super::*;

pub(super) fn parse_packed_quantization(
    config: &Value,
    path: &Path,
) -> Result<Option<PackedQuantization>> {
    let Some(marker) = config.get("quantization") else {
        return Ok(None);
    };
    let marker = marker.as_object().ok_or_else(|| {
        Error::Unsupported(format!(
            "text encoder quantization: {} `quantization` must be an object",
            path.display()
        ))
    })?;
    let bits = marker
        .get("bits")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "text encoder quantization: {} `quantization.bits` must be an integer",
                path.display()
            ))
        })?;
    if !matches!(bits, 4 | 8) {
        return Err(Error::Unsupported(format!(
            "text encoder quantization: {} declares unsupported `quantization.bits` {bits}; expected 4 or 8",
            path.display()
        )));
    }
    let group_size = marker
        .get("group_size")
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "text encoder quantization: {} `quantization.group_size` must be an integer when present",
                        path.display()
                    ))
                })
        })
        .transpose()?
        .unwrap_or(64);
    if group_size != 64 {
        return Err(Error::Unsupported(format!(
            "text encoder quantization: {} declares unsupported `quantization.group_size` {group_size}; every shipping selected-encoder loader requires group_size 64",
            path.display()
        )));
    }
    Ok(Some(PackedQuantization { bits, group_size }))
}

pub(super) fn source_path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path,
    }
}

pub(super) fn resolved_source_layout(
    requested: &WeightsSource,
    resolved: &WeightsSource,
) -> Result<TextEncoderSourceLayout> {
    match (requested, resolved) {
        (WeightsSource::File(requested), WeightsSource::File(resolved))
            if requested == resolved =>
        {
            Ok(TextEncoderSourceLayout::File)
        }
        (WeightsSource::Dir(requested), WeightsSource::Dir(resolved)) if requested == resolved => {
            Ok(TextEncoderSourceLayout::DirectDirectory)
        }
        (WeightsSource::Dir(requested), WeightsSource::Dir(resolved))
            if requested.join("text_encoder") == *resolved =>
        {
            Ok(TextEncoderSourceLayout::CompleteSnapshot)
        }
        _ => Err(Error::Unsupported(
            "text encoder resolved source shape is inconsistent with its requested source".into(),
        )),
    }
}

#[derive(Debug)]
pub(super) struct DiscoveryInspection {
    pub(super) weights: WeightsSource,
    pub(super) config_path: Option<PathBuf>,
    pub(super) config: Option<Value>,
    pub(super) headers: Vec<SafetensorsTensorHeader>,
    pub(super) source_bytes: u64,
}

pub(super) fn inspect_encoder_source_for_discovery(
    source: &WeightsSource,
    allowed_roots: &[PathBuf],
    require_config: bool,
    inspect_content: bool,
) -> Result<DiscoveryInspection> {
    // Confinement of the caller's original entry is deliberately first. In particular, do not ask
    // whether nested configs exist or enumerate a directory until the entry itself is authorized.
    ensure_discovery_paths_confined(&[source_path(source).to_path_buf()], allowed_roots)?;
    let (weights, config_path) = resolve_source_for_discovery(source, allowed_roots)?;
    let mut preflight = vec![source_path(&weights).to_path_buf()];
    preflight.extend(config_path.iter().cloned());
    ensure_discovery_paths_confined(&preflight, allowed_roots)?;

    if require_config && config_path.is_none() {
        return Err(missing_encoder_config_error(source));
    }

    let shard_paths = encoder_shard_paths_for_discovery(&weights, allowed_roots)?;
    ensure_discovery_paths_confined(&shard_paths, allowed_roots)?;
    let source_bytes = discovery_direct_shard_bytes(&shard_paths)?;
    let (config, headers) = if inspect_content {
        let config = config_path
            .as_deref()
            .map(read_selected_encoder_config)
            .transpose()?;
        let headers =
            collect_unique_encoder_headers(shard_paths.iter().map(safetensors_path_tensor_headers))
                .map_err(|error| {
                    Error::Msg(format!(
                        "text encoder contract: inspect {}: {error}",
                        source_path(&weights).display()
                    ))
                })?;
        (config, headers)
    } else {
        (None, Vec::new())
    };

    let (current_weights, current_config) = resolve_source_for_discovery(source, allowed_roots)?;
    if require_config && current_config.is_none() {
        return Err(missing_encoder_config_error(source));
    }
    let current_shards = encoder_shard_paths_for_discovery(&current_weights, allowed_roots)?;
    let current_bytes = discovery_direct_shard_bytes(&current_shards)?;
    if current_weights != weights
        || current_config != config_path
        || current_shards != shard_paths
        || current_bytes != source_bytes
    {
        return Err(Error::Unsupported(
            "text encoder discovery source shape changed during validation".into(),
        ));
    }
    let mut postflight = vec![source_path(&current_weights).to_path_buf()];
    postflight.extend(current_config.iter().cloned());
    postflight.extend(current_shards);
    ensure_discovery_paths_confined(&postflight, allowed_roots)?;

    Ok(DiscoveryInspection {
        weights,
        config_path,
        config,
        headers,
        source_bytes,
    })
}

pub(super) fn missing_encoder_config_error(source: &WeightsSource) -> Error {
    Error::Msg(format!(
        "text encoder substitution has no config.json (source {}); exact behavior, tokenizer, head-topology, and precision compatibility cannot be proven from tensor shapes alone",
        source_path(source).display()
    ))
}

pub(super) fn resolve_source_for_discovery(
    source: &WeightsSource,
    allowed_roots: &[PathBuf],
) -> Result<(WeightsSource, Option<PathBuf>)> {
    match source {
        WeightsSource::File(path) => {
            ensure_discovery_paths_confined(std::slice::from_ref(path), allowed_roots)?;
            let config = match path.parent().map(|parent| parent.join("config.json")) {
                Some(candidate) if discovery_regular_file_candidate(&candidate, allowed_roots)? => {
                    Some(candidate)
                }
                _ => None,
            };
            Ok((WeightsSource::File(path.clone()), config))
        }
        WeightsSource::Dir(path) => {
            ensure_discovery_paths_confined(std::slice::from_ref(path), allowed_roots)?;
            let nested = path.join("text_encoder");
            // Authorize the intermediate component directory itself before appending or inspecting
            // its config path. `symlink_metadata(nested/config.json)` would otherwise traverse an
            // untrusted `text_encoder` directory symlink before its target was root-confined.
            if discovery_directory_candidate(&nested, allowed_roots)? {
                let nested_config = nested.join("config.json");
                if discovery_regular_file_candidate(&nested_config, allowed_roots)? {
                    return Ok((WeightsSource::Dir(nested), Some(nested_config)));
                }
            }
            let direct = path.join("config.json");
            let config =
                discovery_regular_file_candidate(&direct, allowed_roots)?.then_some(direct);
            Ok((WeightsSource::Dir(path.clone()), config))
        }
    }
}

pub(super) fn discovery_directory_candidate(
    path: &Path,
    allowed_roots: &[PathBuf],
) -> Result<bool> {
    ensure_discovery_paths_lexically_confined(
        std::slice::from_ref(&path.to_path_buf()),
        allowed_roots,
    )?;
    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            ensure_discovery_paths_confined(
                std::slice::from_ref(&path.to_path_buf()),
                allowed_roots,
            )?;
            Ok(std::fs::metadata(path)?.is_dir())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn discovery_regular_file_candidate(
    path: &Path,
    allowed_roots: &[PathBuf],
) -> Result<bool> {
    ensure_discovery_paths_lexically_confined(
        std::slice::from_ref(&path.to_path_buf()),
        allowed_roots,
    )?;
    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            ensure_discovery_paths_confined(
                std::slice::from_ref(&path.to_path_buf()),
                allowed_roots,
            )?;
            Ok(std::fs::metadata(path)?.is_file())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn ensure_discovery_paths_lexically_confined(
    paths: &[PathBuf],
    allowed_roots: &[PathBuf],
) -> Result<()> {
    let roots = allowed_roots
        .iter()
        .map(std::path::absolute)
        .collect::<std::io::Result<Vec<_>>>()?;
    if roots.is_empty() {
        return Err(Error::Unsupported(
            "validated text encoder has no authorized model roots".into(),
        ));
    }
    for path in paths {
        let loader_path = std::path::absolute(path)?;
        if !roots.iter().any(|root| loader_path.starts_with(root)) {
            return Err(Error::Unsupported(format!(
                "validated text encoder loader entry escapes authorized model roots: {}",
                loader_path.display()
            )));
        }
    }
    Ok(())
}

/// The authorized discovery roots for an encoder source living at `path`.
///
/// Every root set carries the path itself, lexically absolute and canonical. When `path` lies
/// inside a Hugging Face cache snapshot — `<hub>/models--<org>--<repo>/snapshots/<revision>/…` —
/// the repository directory is authorized as well, because a snapshot's files are symlinks into the
/// sibling `blobs/` tree: the canonical target of `<snapshot>/text_encoder/model-00001.safetensors`
/// is `<hub>/models--<org>--<repo>/blobs/<sha256>`, which the snapshot directory alone can never
/// contain. Nothing above the repository directory is authorized, so a symlink escaping it (another
/// repository, another volume, an arbitrary file) still fails `ensure_discovery_paths_confined`.
/// The repository directory must lie strictly above the snapshot revision on `path`; a directory
/// merely *named* `snapshots` elsewhere on the path does not widen the roots.
///
/// The repository is derived from the **canonical** path, never the lexical one:
/// `std::path::absolute` keeps `..` components, so a lexical path such as
/// `<hub>/models--org--repo/snapshots/<rev>/../../../models--evil--x/text_encoder` names a
/// `models--org--repo` snapshot ancestor while its real location is an unrelated repository.
/// Deriving from the canonical path authorizes only the repository the source actually lives in.
pub fn hf_cache_discovery_roots(path: &Path) -> std::io::Result<Vec<PathBuf>> {
    let absolute = std::path::absolute(path)?;
    let canonical = std::fs::canonicalize(&absolute)?;
    let mut roots = vec![absolute.clone(), canonical.clone()];
    if let Some(repository) = hf_cache_repository_dir(&canonical) {
        roots.push(repository.to_path_buf());
        // The lexical repository is authorized only when it denotes the same directory as the
        // canonical one, so a `..`-bearing path cannot smuggle in a sibling repository.
        if let Some(lexical_repository) = hf_cache_repository_dir(&absolute) {
            if std::fs::canonicalize(lexical_repository)
                .is_ok_and(|resolved| resolved == repository)
            {
                roots.push(lexical_repository.to_path_buf());
            }
        }
    }
    roots.sort();
    roots.dedup();
    Ok(roots)
}

/// `<hub>/models--<org>--<repo>` for an absolute path strictly below that repository's
/// `snapshots/<revision>` directory, else `None`.
pub(super) fn hf_cache_repository_dir(absolute: &Path) -> Option<&Path> {
    absolute.ancestors().skip(1).find_map(|ancestor| {
        let snapshots = ancestor.parent()?;
        let repository = snapshots.parent()?;
        let is_snapshot_revision = snapshots
            .file_name()
            .is_some_and(|name| name == "snapshots")
            && repository
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .is_some_and(|name| name.starts_with("models--"))
            && ancestor.file_name().is_some();
        is_snapshot_revision.then_some(repository)
    })
}

pub(super) fn ensure_discovery_paths_confined(
    paths: &[PathBuf],
    allowed_roots: &[PathBuf],
) -> Result<()> {
    ensure_discovery_paths_lexically_confined(paths, allowed_roots)?;
    let roots = allowed_roots
        .iter()
        .map(std::path::absolute)
        .collect::<std::io::Result<Vec<_>>>()?;
    for path in paths {
        let canonical_target_path = std::fs::canonicalize(std::path::absolute(path)?)?;
        if !roots
            .iter()
            .any(|root| canonical_target_path.starts_with(root))
        {
            return Err(Error::Unsupported(format!(
                "validated text encoder canonical target escapes authorized model roots: {}",
                canonical_target_path.display()
            )));
        }
    }
    Ok(())
}

pub(super) fn read_selected_encoder_config(path: &Path) -> Result<Value> {
    let file = std::fs::File::open(path).map_err(|error| {
        Error::Msg(format!(
            "text encoder contract: read {}: {error}",
            path.display()
        ))
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_SELECTED_ENCODER_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            Error::Msg(format!(
                "text encoder contract: read {}: {error}",
                path.display()
            ))
        })?;
    if bytes.len() as u64 > MAX_SELECTED_ENCODER_CONFIG_BYTES {
        return Err(Error::Unsupported(format!(
            "text encoder config {} exceeds the {MAX_SELECTED_ENCODER_CONFIG_BYTES}-byte maximum",
            path.display()
        )));
    }
    serde_json::from_slice(&bytes).map_err(|error| {
        Error::Msg(format!(
            "text encoder contract: parse {}: {error}",
            path.display()
        ))
    })
}

pub(super) fn json_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter()
        .try_fold(value, |current, segment| current.get(*segment))
}

pub(super) fn encoder_shard_paths(source: &WeightsSource) -> Result<Vec<PathBuf>> {
    let mut paths = match source {
        WeightsSource::File(path) => vec![std::path::absolute(path)?],
        WeightsSource::Dir(path) => {
            let mut paths = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let candidate = entry?.path();
                if candidate
                    .extension()
                    .and_then(|extension| extension.to_str())
                    != Some("safetensors")
                    || crate::weightsmeta::is_hidden_file(&candidate)
                {
                    continue;
                }
                let metadata = std::fs::metadata(&candidate).map_err(|error| {
                    Error::Msg(format!(
                        "text encoder direct shard candidate {} cannot be inspected: {error}",
                        candidate.display()
                    ))
                })?;
                if !metadata.is_file() {
                    return Err(Error::Unsupported(format!(
                        "text encoder direct shard candidate is not a regular file: {}",
                        candidate.display()
                    )));
                }
                paths.push(candidate);
            }
            paths
                .into_iter()
                .map(std::path::absolute)
                .collect::<std::io::Result<Vec<_>>>()?
        }
    };
    paths.sort();
    if paths.is_empty() {
        return Err(Error::Msg(format!(
            "no direct .safetensors shards in text encoder source {}",
            source_path(source).display()
        )));
    }
    Ok(paths)
}

pub(super) fn encoder_shard_paths_for_discovery(
    source: &WeightsSource,
    allowed_roots: &[PathBuf],
) -> Result<Vec<PathBuf>> {
    ensure_discovery_paths_confined(&[source_path(source).to_path_buf()], allowed_roots)?;
    let mut paths = match source {
        WeightsSource::File(path) => {
            let absolute = std::path::absolute(path)?;
            ensure_discovery_paths_confined(std::slice::from_ref(&absolute), allowed_roots)?;
            if !std::fs::metadata(&absolute)?.is_file() {
                return Err(Error::Unsupported(format!(
                    "text encoder direct shard candidate is not a regular file: {}",
                    absolute.display()
                )));
            }
            vec![absolute]
        }
        WeightsSource::Dir(path) => {
            let mut paths = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let candidate = entry?.path();
                if candidate
                    .extension()
                    .and_then(|extension| extension.to_str())
                    != Some("safetensors")
                    || crate::weightsmeta::is_hidden_file(&candidate)
                {
                    continue;
                }
                let absolute = std::path::absolute(candidate)?;
                // Do not follow a shard symlink for metadata until both its lexical loader entry and
                // current canonical target have passed the caller's root authorization.
                ensure_discovery_paths_confined(std::slice::from_ref(&absolute), allowed_roots)?;
                let metadata = std::fs::metadata(&absolute).map_err(|error| {
                    Error::Msg(format!(
                        "text encoder direct shard candidate {} cannot be inspected: {error}",
                        absolute.display()
                    ))
                })?;
                if !metadata.is_file() {
                    return Err(Error::Unsupported(format!(
                        "text encoder direct shard candidate is not a regular file: {}",
                        absolute.display()
                    )));
                }
                paths.push(absolute);
            }
            paths
        }
    };
    paths.sort();
    if paths.is_empty() {
        return Err(Error::Msg(format!(
            "no direct .safetensors shards in text encoder source {}",
            source_path(source).display()
        )));
    }
    Ok(paths)
}

pub(super) fn discovery_direct_shard_bytes(paths: &[PathBuf]) -> Result<u64> {
    paths.iter().try_fold(0u64, |total, path| {
        let bytes = std::fs::metadata(path)
            .map_err(|error| {
                Error::Msg(format!(
                    "text encoder contract: stat direct shard {}: {error}",
                    path.display()
                ))
            })?
            .len();
        total.checked_add(bytes).ok_or_else(|| {
            Error::Unsupported("text encoder direct-shard byte total overflowed u64".into())
        })
    })
}

pub(super) fn validate_quantization_evidence(
    headers: &[SafetensorsTensorHeader],
    path: &Path,
    declared: Option<PackedQuantization>,
) -> Result<Option<PackedQuantization>> {
    let names = headers
        .iter()
        .map(|header| header.name.as_str())
        .collect::<BTreeSet<_>>();
    let has_packed_evidence = headers.iter().any(|header| {
        if !header.name.ends_with(".weight") {
            return false;
        }
        let base = header.name.strip_suffix(".weight").unwrap_or(&header.name);
        header.dtype == Dtype::U32
            || names.contains(format!("{base}.scales").as_str())
            || names.contains(format!("{base}.biases").as_str())
    });
    match (declared, has_packed_evidence) {
        (Some(quant), true) => Ok(Some(quant)),
        (Some(quant), false) => Err(Error::Unsupported(format!(
            "text encoder quantization mismatch at {}: config declares Q{} group_size {} but the exact direct-shard surface is dense",
            path.display(), quant.bits, quant.group_size
        ))),
        (None, true) => Err(Error::Unsupported(format!(
            "text encoder quantization mismatch at {}: packed tensor evidence requires authoritative config.json bits/group_size",
            path.display()
        ))),
        (None, false) => Ok(None),
    }
}

/// Keep tier evidence confined to tensors every supported language loader consumes: the token
/// embedding and decoder layer zero. A provider contract subsequently checks every matrix in its
/// complete loaded layer window, so this anchor cannot make a mixed dense/packed source valid. It
/// only prevents ignored vision tensors and intentionally unloaded decoder tails from choosing the
/// effective tier before contract validation.
pub(super) fn language_quantization_evidence_headers(
    headers: &[SafetensorsTensorHeader],
) -> Vec<SafetensorsTensorHeader> {
    const PREFIXES: [&str; 3] = ["language_model.model", "language_model", "model"];
    headers
        .iter()
        .filter(|header| {
            PREFIXES.iter().any(|prefix| {
                header.name.starts_with(&format!("{prefix}.embed_tokens."))
                    || header.name.starts_with(&format!("{prefix}.layers.0."))
            })
        })
        .cloned()
        .collect()
}

pub(super) fn architecture_layer_index(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(&format!("{prefix}.layers."))?
        .split('.')
        .next()?
        .parse()
        .ok()
}

pub(super) fn selected_complete_snapshot_root(source: &WeightsSource) -> Option<&Path> {
    match source {
        WeightsSource::Dir(path) if path.join("text_encoder/config.json").is_file() => Some(path),
        WeightsSource::Dir(_) | WeightsSource::File(_) => None,
    }
}

pub(super) fn tokenizer_candidate_paths(root: &Path, candidates: &[&str]) -> Result<Vec<PathBuf>> {
    candidates
        .iter()
        .map(|candidate| std::path::absolute(root.join(candidate)).map_err(Error::from))
        .collect()
}

pub(super) fn resolve_tokenizer_artifact(candidates: &[PathBuf], label: &str) -> Result<PathBuf> {
    for path in candidates {
        if path.is_file() {
            return Ok(path.clone());
        }
    }
    Err(Error::Unsupported(format!(
        "{label} has no retained tokenizer artifact for encoder compatibility: expected one of {candidates:?}"
    )))
}

pub(super) fn ensure_tokenizer_resolution_unchanged(
    candidates: &[PathBuf],
    expected: &Path,
    label: &str,
) -> Result<()> {
    let current = resolve_tokenizer_artifact(candidates, label)?;
    if current != expected {
        return Err(Error::Unsupported(format!(
            "{label} tokenizer selection changed after validation: expected {}, got {}",
            expected.display(),
            current.display()
        )));
    }
    Ok(())
}

pub(super) fn pin_tokenizer_artifact(
    path: &Path,
    required_tokens: &[EncoderRequiredToken],
) -> Result<PinnedTokenizerArtifact> {
    let pin = PinnedWeightsFile::pin(path)?;
    let value: Value = pin.read_unchanged(|path| {
        let bytes = std::fs::read(path).map_err(Error::from)?;
        serde_json::from_slice(&bytes).map_err(|error| {
            Error::Msg(format!(
                "text encoder tokenizer contract: parse {}: {error}",
                path.display()
            ))
        })
    })?;
    for required in required_tokens {
        match tokenizer_literal_id(&value, required.literal) {
            Some(actual) if actual == required.id => {}
            Some(actual) => {
                return Err(Error::Unsupported(format!(
                    "text encoder tokenizer contract: {} literal {:?} expected id {}, got {} in {}",
                    required.role,
                    required.literal,
                    required.id,
                    actual,
                    path.display()
                )));
            }
            None => {
                return Err(Error::Unsupported(format!(
                    "text encoder tokenizer contract: required {} literal {:?} is missing from {}",
                    required.role,
                    required.literal,
                    path.display()
                )));
            }
        }
    }
    let mut canonical = Vec::new();
    write_canonical_json(&value, &mut canonical)?;
    let semantic_sha256: [u8; 32] = Sha256::digest(&canonical).into();
    pin.ensure_unchanged()?;
    Ok(PinnedTokenizerArtifact {
        pin,
        semantic_sha256,
    })
}

pub(super) fn tokenizer_literal_id(tokenizer: &Value, literal: &str) -> Option<i64> {
    tokenizer
        .get("added_tokens")
        .and_then(Value::as_array)
        .and_then(|tokens| {
            tokens.iter().find_map(|token| {
                (token.get("content").and_then(Value::as_str) == Some(literal))
                    .then(|| token.get("id").and_then(Value::as_i64))
                    .flatten()
            })
        })
        .or_else(|| {
            let vocab = tokenizer.get("model")?.get("vocab")?;
            match vocab {
                Value::Object(vocab) => vocab.get(literal).and_then(Value::as_i64),
                // Unigram tokenizers store `[token, score]` rows; the row index is the token id.
                Value::Array(vocab) => vocab.iter().enumerate().find_map(|(id, row)| {
                    (row.as_array()
                        .and_then(|row| row.first())
                        .and_then(Value::as_str)
                        == Some(literal))
                    .then(|| i64::try_from(id).ok())
                    .flatten()
                }),
                _ => None,
            }
        })
}

pub(super) fn write_canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Object(object) => {
            output.push(b'{');
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_by_key(|(key, _)| *key);
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                output.extend(
                    serde_json::to_vec(key).map_err(|error| {
                        Error::Msg(format!("canonicalize tokenizer key: {error}"))
                    })?,
                );
                output.push(b':');
                write_canonical_json(value, output)?;
            }
            output.push(b'}');
        }
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        scalar => output.extend(
            serde_json::to_vec(scalar)
                .map_err(|error| Error::Msg(format!("canonicalize tokenizer value: {error}")))?,
        ),
    }
    Ok(())
}

pub(super) fn digest_hex(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn resolve_source(source: &WeightsSource) -> Result<(WeightsSource, Option<PathBuf>)> {
    match source {
        WeightsSource::File(path) => Ok((
            WeightsSource::File(path.clone()),
            path.parent().map(|parent| parent.join("config.json")),
        )),
        WeightsSource::Dir(path) => {
            let nested = path.join("text_encoder");
            let nested_config = nested.join("config.json");
            if nested_config.is_file() {
                return Ok((WeightsSource::Dir(nested), Some(nested_config)));
            }
            let direct = path.join("config.json");
            if direct.is_file() {
                return Ok((WeightsSource::Dir(path.clone()), Some(direct)));
            }
            Ok((WeightsSource::Dir(path.clone()), Some(direct)))
        }
    }
}

pub(super) fn architecture_matches(actual: &str, expected: &str) -> bool {
    fn normalize(value: &str) -> String {
        value
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    }
    let actual = normalize(actual);
    let expected = normalize(expected);
    if actual == expected {
        return true;
    }
    // Canonical Qwen-Image configs nest a text-only `model_type=qwen2_5_vl_text` beside the known
    // multimodal wrapper class. This one explicit wrapper is architecture-equivalent for the text
    // contract; unrelated `ForConditionalGeneration` wrappers remain rejected.
    if expected == "qwen25vltext" && actual == "qwen25vlforconditionalgeneration" {
        return true;
    }
    actual.strip_prefix(&expected).is_some_and(|suffix| {
        matches!(
            suffix,
            "model" | "forcausallm" | "forconditionalgeneration" | "textmodel"
        )
    })
}
