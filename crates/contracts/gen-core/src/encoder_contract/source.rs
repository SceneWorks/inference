//! Validated and pinned encoder/tokenizer sources, load receipts and the dtype/config policies they enforce.

use super::*;

impl ValidatedEncoderSource {
    /// Prove that every lexical loader entry and exact canonical target retained by this receipt is
    /// under one of the caller's pre-authorized model roots.
    pub fn ensure_confined_to(&self, allowed_roots: &[PathBuf]) -> Result<()> {
        self.ensure_unchanged()?;
        let roots = allowed_roots
            .iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        if roots.is_empty() {
            return Err(Error::Unsupported(
                "validated text encoder has no authorized model roots".into(),
            ));
        }
        for pin in self.receipt_pins()? {
            for (kind, path) in [
                ("loader entry", pin.loader_path()),
                ("canonical target", pin.canonical_target_path()),
            ] {
                if !roots.iter().any(|root| path.starts_with(root)) {
                    return Err(Error::Unsupported(format!(
                        "validated text encoder {kind} escapes authorized model roots: {}",
                        path.display()
                    )));
                }
            }
        }
        self.ensure_unchanged()
    }

    /// Attach the exact validated source and its complete shard/config/tokenizer receipt to a load
    /// spec in one atomic operation.
    ///
    /// The requested source shape is preserved: a complete snapshot remains a complete-snapshot
    /// `Dir`, while validation continues to expose only its resolved `text_encoder/` component to
    /// providers. Every direct shard plus the behavior config and retained/selected tokenizer files
    /// enters the prepared identity set used by cache keys and provider load brackets. This is the
    /// only public receipt-export seam; callers never recreate the contract's shard inventory.
    pub fn prepare_load_spec(&self, spec: &mut crate::LoadSpec) -> Result<()> {
        self.ensure_unchanged()?;
        let mut candidate = spec.clone();
        if let Some(existing) = candidate.text_encoder.as_ref() {
            if existing != &self.requested_source {
                return Err(Error::Unsupported(format!(
                    "validated text encoder cannot replace an already selected source: existing {}, validated {}",
                    source_path(existing).display(),
                    source_path(&self.requested_source).display()
                )));
            }
        } else {
            candidate.text_encoder = Some(self.requested_source.clone());
        }

        let receipt_pins = self.receipt_pins()?;
        let receipt_by_path = receipt_pins
            .iter()
            .map(|pin| (pin.loader_path().to_path_buf(), pin))
            .collect::<BTreeMap<_, _>>();
        let mut prepared = Vec::new();
        for path in candidate.file_source_paths() {
            let absolute = std::path::absolute(path)?;
            if let Some(pin) = receipt_by_path.get(&absolute) {
                prepared.push((*pin).clone());
            } else if let Some(pin) = candidate.prepared_file_pins().get(&absolute) {
                pin.ensure_unchanged()?;
                prepared.push(pin.clone());
            } else {
                prepared.push(PinnedWeightsFile::pin(path)?);
            }
        }
        prepared.extend(receipt_pins.iter().cloned());
        let receipt_paths = receipt_pins
            .iter()
            .map(|pin| pin.loader_path().to_path_buf())
            .collect::<Vec<_>>();
        candidate.prepare_with_validated_receipt_pins(
            prepared,
            receipt_paths,
            PreparedEncoderLoadReceipt {
                requested_source: self.requested_source.clone(),
                pinned: self.pinned.clone(),
                tokenizer: self.tokenizer.clone(),
            },
        )?;
        self.ensure_unchanged()?;
        *spec = candidate;
        Ok(())
    }

    pub(super) fn receipt_pins(&self) -> Result<Vec<PinnedWeightsFile>> {
        self.ensure_unchanged()?;
        let mut pins = self.pinned.shard_pins.clone();
        pins.extend(self.pinned.config_pin.iter().cloned());
        if let Some(tokenizer) = &self.tokenizer {
            pins.push(tokenizer.base.pin.clone());
            pins.extend(
                tokenizer
                    .selected
                    .iter()
                    .map(|selected| selected.pin.clone()),
            );
        }
        pins.sort_by(|left, right| left.loader_path().cmp(right.loader_path()));
        pins.dedup_by(|left, right| left.loader_path() == right.loader_path());
        self.ensure_unchanged()?;
        Ok(pins)
    }

    /// Whether an authoritative behavior config accompanied the source. The path itself stays
    /// encapsulated so callers cannot reopen it outside the unchanged-read bracket.
    pub fn has_config(&self) -> bool {
        self.config_path.is_some()
    }

    /// The packed Q4/Q8 tier this exact validated source stores, or `None` for a dense source.
    pub fn packed_quant_bits(&self) -> Option<i32> {
        self.packed_quant.map(|quant| quant.bits as i32)
    }

    /// The affine group size of the packed tier this exact validated source stores, or `None` for
    /// a dense source. Paired with [`Self::packed_quant_bits`] so a backend can build the packed
    /// parts from the validated source instead of re-reading the on-disk `quantization` block.
    pub fn packed_quant_group_size(&self) -> Option<i32> {
        self.packed_quant.map(|quant| quant.group_size as i32)
    }

    /// Derive the load-time conversion from the quantization evidence retained by this exact
    /// validated source. This prevents a swap/restore between a second metadata read and payload
    /// load from selecting the conversion for different bytes.
    pub fn load_time_quant_bits(
        &self,
        expected_bits: Option<i32>,
        provider_id: &str,
    ) -> Result<Option<i32>> {
        self.ensure_unchanged()?;
        resolve_encoder_load_time_quant_bits(self.packed_quant_bits(), expected_bits, provider_id)
    }

    /// Exact bytes in the direct shard inventory accepted by both backends. Nested files are not
    /// loadable shards and therefore cannot inflate conditioning memory facts.
    pub fn source_bytes(&self) -> Result<u64> {
        self.ensure_unchanged()?;
        let bytes = self.pinned.direct_shard_bytes()?;
        self.ensure_unchanged()?;
        Ok(bytes)
    }

    /// Retained tokenizer receipt. `None` is limited to metadata-only validation helpers; every
    /// production source path binds against a base snapshot.
    pub fn tokenizer_source(&self) -> Option<&ValidatedTokenizerSource> {
        self.tokenizer.as_ref()
    }

    pub fn tokenizer_path(&self) -> Option<&Path> {
        self.tokenizer.as_ref().map(ValidatedTokenizerSource::path)
    }

    pub fn tokenizer_disposition(&self) -> Option<EncoderTokenizerDisposition> {
        self.tokenizer
            .as_ref()
            .map(ValidatedTokenizerSource::disposition)
    }

    /// Parse the retained runtime tokenizer inside the same exact pin bracket as encoder loading.
    pub fn read_tokenizer_unchanged<T, E>(
        &self,
        read: impl FnOnce(&Path) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<Error>,
    {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| {
                Error::Unsupported(
                    "validated encoder source has no retained tokenizer receipt".into(),
                )
            })?
            .read_unchanged(read)
    }

    /// Exact tensor headers from the direct shard inventory accepted by the selected-encoder
    /// loaders. Memory projections must use this receipt instead of recursively walking the source
    /// path, because nested safetensors are not loadable shards.
    pub fn tensor_headers(&self) -> Result<Vec<SafetensorsTensorHeader>> {
        self.pinned.ensure_unchanged()?;
        let headers = self.pinned.headers()?;
        self.pinned.ensure_unchanged()?;
        Ok(headers)
    }

    /// Validate the vision half against the same pinned config and shard inventory already accepted
    /// for the language half. The returned receipt remains this source; callers must still use
    /// [`Self::read_unchanged`] for materialization.
    pub fn validate_vision(
        &self,
        vision: &VisionEncoderContract,
        language: &EncoderContract,
    ) -> Result<()> {
        self.ensure_unchanged()?;
        let config_pin = self.pinned.config_pin.as_ref().ok_or_else(|| {
            Error::Unsupported(format!(
                "multimodal vision encoder requires authoritative config.json for {}",
                source_path(&self.weights).display()
            ))
        })?;
        let config: Value = config_pin.read_unchanged(|path| {
            let bytes = std::fs::read(path).map_err(|error| {
                Error::Msg(format!(
                    "vision encoder contract: read {}: {error}",
                    path.display()
                ))
            })?;
            serde_json::from_slice(&bytes).map_err(|error| {
                Error::Msg(format!(
                    "vision encoder contract: parse {}: {error}",
                    path.display()
                ))
            })
        })?;
        vision.validate_config(&config, config_pin.loader_path(), language)?;
        let headers = self.pinned.headers()?;
        vision.validate_tensor_headers(&headers, source_path(&self.weights))?;
        self.ensure_unchanged()
    }

    /// Exact language tensor surface consumed by this contract, excluding unused authored layers
    /// and unrelated `visual.*` tensors that share the same multimodal checkpoint.
    pub fn materialized_language_tensor_headers(
        &self,
        contract: &EncoderContract,
    ) -> Result<Vec<SafetensorsTensorHeader>> {
        self.ensure_unchanged()?;
        let headers = self.pinned.headers()?;
        let packing = if self.packed_quant.is_some() {
            Some(contract.packing.ok_or_else(|| {
                Error::Unsupported(format!(
                    "validated packed encoder has no packing contract for architecture {}",
                    contract.architecture
                ))
            })?)
        } else {
            None
        };
        let expected = contract.materialized_language_tensor_names(packing)?;
        let selected = headers
            .into_iter()
            .filter(|header| expected.contains(&header.name))
            .collect();
        self.ensure_unchanged()?;
        Ok(selected)
    }

    /// Exact vision tensor surface consumed by a previously validated multimodal source.
    pub fn materialized_vision_tensor_headers(
        &self,
        vision: &VisionEncoderContract,
        language: &EncoderContract,
    ) -> Result<Vec<SafetensorsTensorHeader>> {
        self.validate_vision(vision, language)?;
        let expected = vision
            .expected_headers()?
            .into_iter()
            .map(|(name, _)| name)
            .collect::<BTreeSet<_>>();
        let selected = self
            .pinned
            .headers()?
            .into_iter()
            .filter(|header| expected.contains(&header.name))
            .collect();
        self.ensure_unchanged()?;
        Ok(selected)
    }

    /// Execute one backend load against the exact config and direct-shard set that passed contract
    /// validation. This brackets the backend's own directory enumeration so a persistent shard
    /// addition, removal, retarget, or replacement cannot turn validated bytes into different bytes.
    pub fn read_unchanged<T, E>(
        &self,
        read: impl FnOnce(&WeightsSource) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<Error>,
    {
        self.ensure_unchanged().map_err(E::from)?;
        let result = read(&self.weights);
        self.ensure_unchanged().map_err(E::from)?;
        #[cfg(feature = "model-inference-proof")]
        if result.is_ok() {
            crate::runtime::record_sealed_input_read(
                "gen-core::ValidatedEncoderSource::read_unchanged",
                self.pinned
                    .shard_pins
                    .iter()
                    .chain(self.pinned.config_pin.iter()),
            );
        }
        result
    }

    pub(super) fn ensure_unchanged(&self) -> Result<()> {
        self.pinned.ensure_unchanged()?;
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer.ensure_unchanged()?;
        }
        Ok(())
    }
}

impl PreparedEncoderLoadReceipt {
    pub(crate) fn ensure_unchanged_for(
        &self,
        selected_source: Option<&WeightsSource>,
    ) -> Result<()> {
        if selected_source != Some(&self.requested_source) {
            return Err(Error::Unsupported(
                "prepared text-encoder receipt no longer matches the LoadSpec source".into(),
            ));
        }
        let (resolved, _) = resolve_source(&self.requested_source)?;
        if resolved != self.pinned.weights {
            return Err(Error::Unsupported(format!(
                "prepared text-encoder source shape changed: expected {}, got {}",
                source_path(&self.pinned.weights).display(),
                source_path(&resolved).display()
            )));
        }
        self.pinned.ensure_unchanged().map_err(|error| {
            Error::Unsupported(format!("prepared text-encoder receipt changed: {error}"))
        })?;
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer.ensure_unchanged().map_err(|error| {
                Error::Unsupported(format!("prepared text-encoder receipt changed: {error}"))
            })?;
        }
        Ok(())
    }

    pub(crate) fn planning_facts_for(
        &self,
        selected_source: Option<&WeightsSource>,
    ) -> Result<TextEncoderPlanningFacts> {
        self.ensure_unchanged_for(selected_source)?;
        let direct_shard_bytes = self
            .pinned
            .shard_pins
            .iter()
            .try_fold(0_u64, |total, pin| {
                total
                    .checked_add(pin.target_fingerprint().size)
                    .ok_or_else(|| {
                        Error::Unsupported(
                            "text encoder direct-shard byte total overflowed u64".into(),
                        )
                    })
            })?;
        let facts = TextEncoderPlanningFacts {
            source_layout: resolved_source_layout(&self.requested_source, &self.pinned.weights)?,
            direct_shard_bytes,
        };
        self.ensure_unchanged_for(selected_source)?;
        Ok(facts)
    }
}

impl ValidatedTokenizerSource {
    pub fn path(&self) -> &Path {
        self.base.pin.loader_path()
    }

    pub fn disposition(&self) -> EncoderTokenizerDisposition {
        self.disposition
    }

    pub fn semantic_sha256(&self) -> [u8; 32] {
        self.base.semantic_sha256
    }

    pub fn read_unchanged<T, E>(
        &self,
        read: impl FnOnce(&Path) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<Error>,
    {
        self.ensure_unchanged().map_err(E::from)?;
        let result = read(self.path());
        self.ensure_unchanged().map_err(E::from)?;
        #[cfg(feature = "model-inference-proof")]
        if result.is_ok() {
            crate::runtime::record_sealed_input_read_with_role(
                "gen-core::ValidatedTokenizerSource::read_unchanged",
                crate::model_input_proof::InputRole::Tokenizer,
                std::iter::once(&self.base.pin),
            );
        }
        result
    }

    pub(super) fn ensure_unchanged(&self) -> Result<()> {
        ensure_tokenizer_resolution_unchanged(
            &self.base_candidates,
            self.base.pin.loader_path(),
            "base snapshot",
        )?;
        self.base.pin.ensure_unchanged()?;
        if let Some(selected) = &self.selected {
            ensure_tokenizer_resolution_unchanged(
                self.selected_candidates.as_deref().ok_or_else(|| {
                    Error::Unsupported(
                        "selected tokenizer receipt has no candidate inventory".into(),
                    )
                })?,
                selected.pin.loader_path(),
                "selected complete snapshot",
            )?;
            selected.pin.ensure_unchanged()?;
        }
        Ok(())
    }
}

impl PinnedEncoderSource {
    pub(super) fn pin(weights: &WeightsSource, config_candidate: Option<PathBuf>) -> Result<Self> {
        let shard_paths = encoder_shard_paths(weights)?;
        let shard_pins = shard_paths
            .iter()
            .map(PinnedWeightsFile::pin)
            .collect::<Result<Vec<_>>>()?;
        let config_pin = config_candidate
            .as_ref()
            .filter(|path| path.is_file())
            .map(PinnedWeightsFile::pin)
            .transpose()?;
        let pinned = Self {
            weights: weights.clone(),
            shard_paths,
            shard_pins,
            config_candidate,
            config_pin,
        };
        pinned.ensure_unchanged()?;
        Ok(pinned)
    }

    pub(super) fn ensure_unchanged(&self) -> Result<()> {
        let current = encoder_shard_paths(&self.weights)?;
        if current != self.shard_paths {
            return Err(Error::Unsupported(format!(
                "text encoder shard inventory changed after validation: expected {:?}, got {:?}",
                self.shard_paths, current
            )));
        }
        let config_present = self
            .config_candidate
            .as_ref()
            .is_some_and(|path| path.is_file());
        if config_present != self.config_pin.is_some() {
            return Err(Error::Unsupported(format!(
                "text encoder config presence changed after validation: {}",
                self.config_candidate
                    .as_deref()
                    .unwrap_or_else(|| Path::new("<none>"))
                    .display()
            )));
        }
        if let Some(pin) = &self.config_pin {
            pin.ensure_unchanged()?;
        }
        for pin in &self.shard_pins {
            pin.ensure_unchanged()?;
        }
        Ok(())
    }

    pub(super) fn headers(&self) -> Result<Vec<SafetensorsTensorHeader>> {
        collect_unique_encoder_headers(
            self.shard_pins
                .iter()
                .map(|pin| pin.read_unchanged(|path| safetensors_path_tensor_headers(path))),
        )
    }

    pub(super) fn direct_shard_bytes(&self) -> Result<u64> {
        self.ensure_unchanged()?;
        let mut total = 0u64;
        for pin in &self.shard_pins {
            let bytes = pin.read_unchanged(|path| {
                std::fs::metadata(path)
                    .map(|metadata| metadata.len())
                    .map_err(|error| {
                        Error::Msg(format!(
                            "text encoder contract: stat direct shard {}: {error}",
                            path.display()
                        ))
                    })
            })?;
            total = total.checked_add(bytes).ok_or_else(|| {
                Error::Unsupported("text encoder direct-shard byte total overflowed u64".into())
            })?;
        }
        self.ensure_unchanged()?;
        Ok(total)
    }

    pub(super) fn read_unchanged<T>(
        &self,
        read: impl FnOnce(&WeightsSource) -> Result<T>,
    ) -> Result<T> {
        self.ensure_unchanged()?;
        let result = read(&self.weights)?;
        self.ensure_unchanged()?;
        Ok(result)
    }
}

pub(super) fn collect_unique_encoder_headers(
    inventories: impl IntoIterator<Item = Result<Vec<SafetensorsTensorHeader>>>,
) -> Result<Vec<SafetensorsTensorHeader>> {
    let mut headers = BTreeMap::new();
    for inventory in inventories {
        for header in inventory? {
            if headers
                .insert(header.name.clone(), header.clone())
                .is_some()
            {
                return Err(Error::Unsupported(format!(
                    "text encoder source contains duplicate tensor key {:?} across direct shards; selected encoders must be accepted identically by the strict MLX and Candle loaders",
                    header.name
                )));
            }
        }
    }
    Ok(headers.into_values().collect())
}

/// One affine packed-quantization marker as a selected encoder's `config.json` declares it
/// (`quantization: {bits, group_size}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PackedQuantization {
    pub(super) bits: usize,
    pub(super) group_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EncoderDtypePolicy {
    Native,
    ComfyUiFp8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EncoderConfigPolicy {
    /// Ordinary alternates must supply their own behavior-bearing config.
    Required,
    /// Legacy ComfyUI exports contain only tensors. Their consuming route freezes the provider's
    /// config/tokenizer policy, so a present sibling config is checked but its absence is allowed.
    ProviderOwnedComfyUi,
}

impl EncoderConfigPolicy {
    pub(super) fn requires_config(self) -> bool {
        matches!(self, Self::Required)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MatrixExpectation<'a> {
    pub(super) name: &'a str,
    pub(super) field: &'static str,
    pub(super) shape: [usize; 2],
    pub(super) must_be_packed: bool,
}

impl EncoderDtypePolicy {
    pub(super) fn accepts_dense(self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
            || (self == Self::ComfyUiFp8 && dtype == Dtype::F8_E4M3)
    }

    pub(super) fn expected_dense(self) -> &'static str {
        match self {
            Self::Native => "F16, BF16, or F32",
            Self::ComfyUiFp8 => "F8_E4M3, F16, BF16, or F32",
        }
    }
}
