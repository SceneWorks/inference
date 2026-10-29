//! Header-only quantization and source-inventory queries over a text-encoder source.

use super::*;

/// Resolve the load-time action needed to keep a selected text encoder on the provider's numeric
/// policy. `expected_bits` is the effective base-model policy (`None` = dense, `Some(4|8)` = packed).
///
/// A matching pre-packed encoder needs no conversion, while a dense encoder inherits the effective
/// Q4/Q8 tier at load. Any packed-vs-expected mismatch is rejected because backend `quantize`
/// implementations intentionally no-op on packed weights and would otherwise silently serve the
/// wrong tier.
#[cfg(test)]
pub(super) fn text_encoder_load_time_quant_bits(
    source: &WeightsSource,
    expected_bits: Option<i32>,
    provider_id: &str,
) -> Result<Option<i32>> {
    let packed_bits = text_encoder_packed_quant_bits(source)?;
    resolve_encoder_load_time_quant_bits(packed_bits, expected_bits, provider_id)
}

/// Resolve the pure packed-source versus provider-tier policy used by
/// [`ValidatedEncoderSource::load_time_quant_bits`].
///
/// Keeping this independent of filesystem receipts lets memory projection and catalog tests cover
/// the complete policy matrix from already validated header facts. Production source admission
/// still owns artifact sealing and calls the same resolver only after validation has established
/// `packed_bits` from the selected source.
pub fn resolve_encoder_load_time_quant_bits(
    packed_bits: Option<i32>,
    expected_bits: Option<i32>,
    provider_id: &str,
) -> Result<Option<i32>> {
    if let Some(bits) = expected_bits {
        if !matches!(bits, 4 | 8) {
            return Err(Error::Unsupported(format!(
                "{provider_id}: unsupported text encoder quantization policy Q{bits}; expected Q4 or Q8"
            )));
        }
    }
    match (packed_bits, expected_bits) {
        (Some(packed), Some(expected)) if packed == expected => Ok(None),
        (Some(packed), Some(expected)) => Err(Error::Unsupported(format!(
            "{provider_id}: selected text encoder is pre-quantized Q{packed} but the model policy is Q{expected}; quantize is a no-op on packed weights so this would silently serve Q{packed}. Select a Q{expected} or dense compatible encoder."
        ))),
        (Some(packed), None) => Err(Error::Unsupported(format!(
            "{provider_id}: selected text encoder is pre-quantized Q{packed} but the model policy is dense; select a dense compatible encoder."
        ))),
        (None, Some(expected)) => Ok(Some(expected)),
        (None, None) => Ok(None),
    }
}

/// Read a selected encoder component's optional Q4/Q8 marker. Missing `quantization` means dense;
/// malformed or unsupported markers fail closed. Header validation remains separately authoritative
/// for architecture and shape.
pub fn text_encoder_packed_quant_bits(source: &WeightsSource) -> Result<Option<i32>> {
    let (resolved, config_path) = resolve_source(source)?;
    let pinned = PinnedEncoderSource::pin(&resolved, config_path.clone())?;
    let declared = match pinned.config_pin.as_ref() {
        Some(pin) => {
            let bytes = pin.read_unchanged(|path| {
                std::fs::read(path).map_err(|error| {
                    Error::Msg(format!(
                        "text encoder quantization: read {}: {error}",
                        path.display()
                    ))
                })
            })?;
            let config: Value = serde_json::from_slice(&bytes).map_err(|error| {
                Error::Msg(format!(
                    "text encoder quantization: parse {}: {error}",
                    config_path
                        .as_deref()
                        .unwrap_or_else(|| Path::new("<none>"))
                        .display()
                ))
            })?;
            parse_packed_quantization(
                &config,
                config_path
                    .as_deref()
                    .unwrap_or_else(|| Path::new("<none>")),
            )?
        }
        None => None,
    };
    let headers = pinned.headers()?;
    let language_headers = language_quantization_evidence_headers(&headers);
    let quant =
        validate_quantization_evidence(&language_headers, source_path(&resolved), declared)?;
    pinned.ensure_unchanged()?;
    Ok(quant.map(|quant| quant.bits as i32))
}

/// Exact safetensors bytes for the encoder component named by `source`. A complete snapshot Dir is
/// resolved to its `text_encoder/` child, and the direct shard/config inventory is pinned around the
/// read so memory facts cannot price a different tree than the selected loader later consumes.
pub fn read_text_encoder_source_unchanged<T>(
    source: &WeightsSource,
    read: impl FnOnce(&WeightsSource) -> Result<T>,
) -> Result<T> {
    let (resolved, config_path) = resolve_source(source)?;
    let pinned = PinnedEncoderSource::pin(&resolved, config_path)?;
    pinned.read_unchanged(read)
}

pub fn text_encoder_source_bytes(source: &WeightsSource) -> Result<u64> {
    let (resolved, config_path) = resolve_source(source)?;
    PinnedEncoderSource::pin(&resolved, config_path)?.direct_shard_bytes()
}

/// Exact direct-shard tensor inventory for an encoder source, without asserting an architecture.
/// This is the header analogue of [`text_encoder_source_bytes`] for weights-free catalog pricing:
/// it resolves complete snapshots to `text_encoder/`, excludes nested files exactly like the
/// concrete loaders, and pins the inventory across inspection.
pub fn text_encoder_source_tensor_headers(
    source: &WeightsSource,
) -> Result<Vec<SafetensorsTensorHeader>> {
    let (resolved, config_path) = resolve_source(source)?;
    let pinned = PinnedEncoderSource::pin(&resolved, config_path)?;
    pinned.ensure_unchanged()?;
    let headers = pinned.headers()?;
    pinned.ensure_unchanged()?;
    Ok(headers)
}

/// Inspect one encoder source's direct shards for catalog classification or weights-free pricing
/// without acquiring an executable artifact seal. The returned inventory is non-authorizing and
/// has not asserted a particular [`EncoderContract`]; callers that have behavior config must prefer
/// [`EncoderContract::validate_source_for_discovery`].
pub fn text_encoder_source_inventory_for_discovery(
    source: &WeightsSource,
    allowed_roots: &[PathBuf],
) -> Result<TextEncoderDiscoveryInventory> {
    let inspected = inspect_encoder_source_for_discovery(source, allowed_roots, false, true)?;
    Ok(TextEncoderDiscoveryInventory {
        tensor_headers: inspected.headers,
        source_bytes: inspected.source_bytes,
    })
}

/// Inspect path-free selected-encoder planning facts without acquiring an executable artifact
/// seal. `allowed_roots` confines every lexical entry and canonical target before traversal; the
/// returned value deliberately cannot authorize a later load.
pub fn text_encoder_planning_facts_for_discovery(
    source: &WeightsSource,
    allowed_roots: &[PathBuf],
) -> Result<TextEncoderPlanningFacts> {
    let inspected = inspect_encoder_source_for_discovery(source, allowed_roots, false, false)?;
    Ok(TextEncoderPlanningFacts {
        source_layout: resolved_source_layout(source, &inspected.weights)?,
        direct_shard_bytes: inspected.source_bytes,
    })
}
