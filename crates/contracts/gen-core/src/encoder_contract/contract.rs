//! `EncoderContract` validation of a candidate encoder against what a generator advertises.

use super::*;

impl EncoderContract {
    /// Project a previously contract-validated dense header inventory onto the exact language
    /// tensor names retained by this encoder's constructors. This is the embedded-checkpoint twin
    /// of [`ValidatedEncoderSource::materialized_language_tensor_headers`]: fused ComfyUI sources
    /// retain their own file pin, so they cannot produce a standalone encoder receipt, but memory
    /// planning still must exclude validated yet unmaterialized tails and unrelated extras.
    pub fn materialized_dense_language_tensor_headers(
        &self,
        validated_headers: &[SafetensorsTensorHeader],
    ) -> Result<Vec<SafetensorsTensorHeader>> {
        let expected = self.materialized_language_tensor_names(None)?;
        Ok(validated_headers
            .iter()
            .filter(|header| expected.contains(&header.name))
            .cloned()
            .collect())
    }

    /// Validate the provider-authored contract itself before it is advertised or used to inspect a
    /// caller's files. This keeps malformed descriptors from turning later shape arithmetic into a
    /// panic or from advertising an architecture for which no exhaustive header signature exists.
    pub fn validate_definition(&self) -> Result<()> {
        self.validate_tokenizer_definition()?;
        self.validate_prompt_execution_definition()?;
        if self.architecture.trim().is_empty()
            || self.hidden_activation.trim().is_empty()
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.vocab_size == 0
            || self.output_width == 0
            || self.loaded_hidden_layers == 0
            || self.loaded_hidden_layers > self.num_hidden_layers
            || self.max_position_embeddings == 0
            || !self.attention_dropout.get().is_finite()
            || self.attention_dropout.get() != 0.0
            || !self.rms_norm_eps.get().is_finite()
            || self.rms_norm_eps.get() <= 0.0
            || self
                .qk_norm_eps
                .is_some_and(|eps| !eps.get().is_finite() || eps.get() <= 0.0)
            || !self.rope_theta.get().is_finite()
            || self.rope_theta.get() <= 0.0
            || !self.output_width.is_multiple_of(self.hidden_size)
            || self.num_key_value_heads > self.num_attention_heads
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self
                .num_attention_heads
                .checked_mul(self.head_dim)
                .is_none()
            || self
                .num_key_value_heads
                .checked_mul(self.head_dim)
                .is_none()
            || self.requires_lm_head && !self.requires_final_norm
            || self.selected_hidden_layers.is_empty()
            || self
                .selected_hidden_layers
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .selected_hidden_layers
                .iter()
                .any(|&layer| layer == 0 || layer > self.loaded_hidden_layers)
            || self
                .mrope_section
                .iter()
                .try_fold(0usize, |sum, &part| sum.checked_add(part))
                .is_none()
            || (!self.mrope_section.is_empty()
                && (!self.head_dim.is_multiple_of(2)
                    || self.mrope_section.iter().sum::<usize>() != self.head_dim / 2))
            || self.mrope_interleaved.is_some() && self.mrope_section.is_empty()
            || matches!(self.architecture, "qwen3" | "qwen3_vl_text") != self.qk_norm_eps.is_some()
            || [
                self.bos_token_id,
                self.eos_token_id,
                self.image_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
            ]
            .into_iter()
            .flatten()
            .any(|token| match usize::try_from(token) {
                Ok(token) => token >= self.vocab_size,
                Err(_) => true,
            })
            || self.packing.is_some_and(|packing| {
                packing.group_size == 0
                    || !packing.group_size.is_power_of_two()
                    || !matches!(packing.group_size, 64)
            })
        {
            return Err(Error::Unsupported(format!(
                "invalid text encoder contract for {:?}: dimensions/config values and the exact loaded/selected-layer surface must be coherent, attention widths must not overflow, output_width ({}) must be a multiple of hidden_size ({}), attention heads ({}) must be divisible by key/value heads ({}), token ids and multimodal RoPE must fit their declared geometry, and packed loaders require group_size 64",
                self.architecture,
                self.output_width,
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads
            )));
        }
        self.expected_header_prefix()?;
        Ok(())
    }

    pub(super) fn validate_tokenizer_definition(&self) -> Result<()> {
        let tokenizer = self.tokenizer;
        if tokenizer.family.trim().is_empty() || tokenizer.artifact_candidates.is_empty() {
            return Err(Error::Unsupported(
                "text encoder tokenizer contract requires a family and at least one artifact candidate"
                    .into(),
            ));
        }
        let mut candidates = BTreeSet::new();
        for candidate in tokenizer.artifact_candidates {
            let path = Path::new(candidate);
            if candidate.trim().is_empty()
                || path.is_absolute()
                || !path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_)))
                || !candidates.insert(*candidate)
            {
                return Err(Error::Unsupported(format!(
                    "invalid tokenizer artifact candidate {candidate:?} for family {}: candidates must be unique, non-empty, relative normal paths",
                    tokenizer.family
                )));
            }
        }

        let mut roles = BTreeSet::new();
        let mut literals = BTreeSet::new();
        for required in tokenizer.required_tokens {
            if required.role.trim().is_empty()
                || required.literal.is_empty()
                || required.id < 0
                || usize::try_from(required.id).map_or(true, |id| id >= self.vocab_size)
                || !roles.insert(required.role)
                || !literals.insert(required.literal)
                || required
                    .config_field
                    .is_some_and(|field| field.trim().is_empty())
            {
                return Err(Error::Unsupported(format!(
                    "invalid required tokenizer literal {:?} ({}) for family {}: roles/literals must be unique and token ids must fit vocab_size {}",
                    required.literal, required.id, tokenizer.family, self.vocab_size
                )));
            }
        }
        Ok(())
    }

    pub(super) fn validate_prompt_execution_definition(&self) -> Result<()> {
        if self.prompt_executions.is_empty() {
            return Err(Error::Unsupported(
                "text encoder contract requires at least one prompt execution policy".into(),
            ));
        }
        let mut purposes = BTreeSet::new();
        for execution in self.prompt_executions {
            if execution.purpose.trim().is_empty() || !purposes.insert(execution.purpose) {
                return Err(Error::Unsupported(format!(
                    "invalid or duplicate text encoder prompt purpose {:?}",
                    execution.purpose
                )));
            }
            let finite_max = match execution.length {
                EncoderPromptLengthPolicy::Unbounded => None,
                EncoderPromptLengthPolicy::RightTruncate { max_tokens }
                | EncoderPromptLengthPolicy::RejectAbove { max_tokens } => Some(max_tokens),
            };
            if finite_max.is_some_and(|max| max == 0 || execution.prefix_trim >= max) {
                return Err(Error::Unsupported(format!(
                    "invalid text encoder prompt policy {:?}: finite max must be non-zero and exceed prefix_trim {}",
                    execution.purpose, execution.prefix_trim
                )));
            }
            if let EncoderPromptPadding::RightToMax { pad_token_id } = execution.padding {
                if !matches!(
                    execution.length,
                    EncoderPromptLengthPolicy::RightTruncate { .. }
                ) || !self
                    .tokenizer
                    .required_tokens
                    .iter()
                    .any(|required| required.id == pad_token_id)
                {
                    return Err(Error::Unsupported(format!(
                        "invalid text encoder prompt policy {:?}: right-padding requires a finite right-truncation max and a declared pad-token literal for id {pad_token_id}",
                        execution.purpose
                    )));
                }
            }
        }
        Ok(())
    }

    /// Pin and validate the tokenizer a provider retains from its base snapshot. Runtime tokenizer
    /// parsers must consume the returned receipt through [`ValidatedTokenizerSource::read_unchanged`].
    pub fn tokenizer_for_base(&self, base_root: &Path) -> Result<ValidatedTokenizerSource> {
        self.validate_definition()?;
        match self.tokenizer.binding {
            EncoderTokenizerBinding::RetainBase => {}
        }
        let base_candidates =
            tokenizer_candidate_paths(base_root, self.tokenizer.artifact_candidates)?;
        let base_path = resolve_tokenizer_artifact(&base_candidates, "base snapshot")?;
        let base = pin_tokenizer_artifact(&base_path, self.tokenizer.required_tokens)?;
        let tokenizer = ValidatedTokenizerSource {
            base,
            base_candidates,
            selected: None,
            selected_candidates: None,
            disposition: EncoderTokenizerDisposition::InheritedBase,
        };
        tokenizer.ensure_unchanged()?;
        Ok(tokenizer)
    }

    pub(super) fn bind_tokenizer(
        &self,
        base_root: &Path,
        selected_source: &WeightsSource,
    ) -> Result<ValidatedTokenizerSource> {
        let mut tokenizer = self.tokenizer_for_base(base_root)?;
        if let Some(selected_root) = selected_complete_snapshot_root(selected_source) {
            let selected_candidates =
                tokenizer_candidate_paths(selected_root, self.tokenizer.artifact_candidates)?;
            let selected_path =
                resolve_tokenizer_artifact(&selected_candidates, "selected complete snapshot")?;
            let selected = pin_tokenizer_artifact(&selected_path, self.tokenizer.required_tokens)?;
            if selected.semantic_sha256 != tokenizer.base.semantic_sha256 {
                return Err(Error::Unsupported(format!(
                    "selected complete snapshot tokenizer is incompatible with retained base tokenizer for family {}: {} has semantic sha256 {}, base {} has {}; pass the selected text_encoder component directly only when it is intentionally a weights-only fine-tune that inherits the base tokenizer",
                    self.tokenizer.family,
                    selected_path.display(),
                    digest_hex(selected.semantic_sha256),
                    tokenizer.path().display(),
                    digest_hex(tokenizer.base.semantic_sha256),
                )));
            }
            tokenizer.selected = Some(selected);
            tokenizer.selected_candidates = Some(selected_candidates);
            tokenizer.disposition = EncoderTokenizerDisposition::MatchedSelectedTokenizer;
        }
        tokenizer.ensure_unchanged()?;
        Ok(tokenizer)
    }

    /// Resolve and validate the effective text-encoder component for one generator load.  The
    /// bundled component remains the zero-change source default, but it is subject to the same
    /// config-and-header contract as an explicit override.
    pub fn source_for_load(
        &self,
        spec: &crate::LoadSpec,
        base_root: &Path,
    ) -> Result<ValidatedEncoderSource> {
        spec.validate_prepared_file_pins()?;
        let builtin = WeightsSource::Dir(base_root.join("text_encoder"));
        let source = spec.text_encoder.as_ref().unwrap_or(&builtin);
        let mut validated = self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
            None,
        )?;
        validated.tokenizer = Some(self.bind_tokenizer(base_root, source)?);
        validated.ensure_unchanged()?;
        spec.validate_prepared_file_pins()?;
        Ok(validated)
    }

    /// Validate one catalog/discovery candidate without acquiring a reusable source receipt.
    ///
    /// This path is intentionally limited to direct-file inventory, bounded behavior config, and
    /// safetensors headers. It never acquires an [`crate::ArtifactSeal`] or reads tensor payloads.
    /// Every lexical loader entry and its current canonical target must remain under one of
    /// `allowed_roots`; callers cannot use discovery as an authorization bypass. The result is only
    /// compatibility information for the current call. Executable load/worker preparation must use
    /// [`Self::validate_source_against_base`] (or [`Self::source_for_load`]) to acquire the complete
    /// retained seal.
    pub fn validate_source_for_discovery(
        &self,
        source: &WeightsSource,
        allowed_roots: &[PathBuf],
    ) -> Result<EncoderDiscoveryFacts> {
        self.validate_source_for_discovery_with_policy(
            source,
            allowed_roots,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
        )
    }

    /// Bounded discovery counterpart to [`Self::validate_comfyui_source`]. Only a direct File is
    /// accepted, and the same provider-owned-config and FP8 normalization policy is applied without
    /// acquiring an executable artifact receipt.
    pub fn validate_comfyui_source_for_discovery(
        &self,
        source: &WeightsSource,
        allowed_roots: &[PathBuf],
    ) -> Result<EncoderDiscoveryFacts> {
        if !matches!(source, WeightsSource::File(_)) {
            return Err(Error::Unsupported(
                "ComfyUI text encoder validation requires one File source".into(),
            ));
        }
        self.validate_source_for_discovery_with_policy(
            source,
            allowed_roots,
            EncoderDtypePolicy::ComfyUiFp8,
            EncoderConfigPolicy::ProviderOwnedComfyUi,
        )
    }

    pub(super) fn validate_source_for_discovery_with_policy(
        &self,
        source: &WeightsSource,
        allowed_roots: &[PathBuf],
        dtype_policy: EncoderDtypePolicy,
        config_policy: EncoderConfigPolicy,
    ) -> Result<EncoderDiscoveryFacts> {
        self.validate_definition()?;
        let inspected = inspect_encoder_source_for_discovery(
            source,
            allowed_roots,
            config_policy.requires_config(),
            true,
        )?;
        let packed_quant = self.validate_inspected_source(
            source,
            &inspected.weights,
            inspected
                .config_path
                .as_deref()
                .zip(inspected.config.as_ref()),
            &inspected.headers,
            dtype_policy,
            config_policy,
        )?;
        let packing = packed_quant
            .map(|_| {
                self.packing.ok_or_else(|| {
                    Error::Unsupported(format!(
                        "validated packed encoder has no packing contract for architecture {}",
                        self.architecture
                    ))
                })
            })
            .transpose()?;
        let expected = self.materialized_language_tensor_names(packing)?;
        let materialized_language_tensor_headers = inspected
            .headers
            .into_iter()
            .filter(|header| expected.contains(&header.name))
            .collect();
        Ok(EncoderDiscoveryFacts {
            materialized_language_tensor_headers,
            source_bytes: inspected.source_bytes,
        })
    }

    /// Resolve and validate the effective text-encoder component for metadata-only planning.
    /// Unlike [`Self::source_for_load`], this deliberately does not require or retain a tokenizer:
    /// memory admission prices tensor materialization, while the executable load remains the seam
    /// that proves tokenizer compatibility. Complete selected snapshots are accepted here so their
    /// tensor surface can be priced before that later load-time binding.
    pub fn source_for_planning(
        &self,
        spec: &crate::LoadSpec,
        base_root: &Path,
    ) -> Result<ValidatedEncoderSource> {
        let builtin = WeightsSource::Dir(base_root.join("text_encoder"));
        let source = spec.text_encoder.as_ref().unwrap_or(&builtin);
        self.validate_source_for_planning(source)
    }

    /// Validate a provider-selected encoder's behavior and tensor surface for metadata-only
    /// planning, without claiming that its tokenizer has passed the production load-time binding.
    pub fn validate_source_for_planning(
        &self,
        source: &WeightsSource,
    ) -> Result<ValidatedEncoderSource> {
        self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
            None,
        )
    }

    /// Validate a provider-selected source while binding it to the tokenizer retained from
    /// `base_root`. This is the production counterpart to metadata-only [`Self::validate_source`].
    pub fn validate_source_against_base(
        &self,
        source: &WeightsSource,
        base_root: &Path,
    ) -> Result<ValidatedEncoderSource> {
        let mut validated = self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
            None,
        )?;
        validated.tokenizer = Some(self.bind_tokenizer(base_root, source)?);
        validated.ensure_unchanged()?;
        Ok(validated)
    }

    /// Validate one substituted encoder without reading tensor payloads.
    ///
    /// A `Dir` may point directly at the encoder component or at a complete diffusers snapshot.  A
    /// `File` requires a sibling `config.json`: tensor geometry cannot prove behavior-bearing config
    /// or tokenizer identity. A caller cannot lend separately authored bytes the built-in config;
    /// each ordinary selected component must carry its own authoritative behavior evidence.
    pub fn validate_source(&self, source: &WeightsSource) -> Result<ValidatedEncoderSource> {
        if selected_complete_snapshot_root(source).is_some() {
            return Err(Error::Unsupported(
                "a complete text-encoder snapshot requires retained-tokenizer binding; use validate_source_against_base with the provider's base snapshot root"
                    .into(),
            ));
        }
        self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
            None,
        )
    }

    /// Validate a single-file ComfyUI encoder that the consuming route explicitly normalizes from
    /// plain/scaled FP8 before construction. Ordinary native routes remain stricter and cannot use
    /// this method to admit an FP8 source their backend loader does not normalize.
    pub fn validate_comfyui_source(
        &self,
        source: &WeightsSource,
    ) -> Result<ValidatedEncoderSource> {
        if !matches!(source, WeightsSource::File(_)) {
            return Err(Error::Unsupported(
                "ComfyUI text encoder validation requires one pinned File source".into(),
            ));
        }
        self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::ComfyUiFp8,
            EncoderConfigPolicy::ProviderOwnedComfyUi,
            None,
        )
    }

    pub fn validate_comfyui_source_against_base(
        &self,
        source: &WeightsSource,
        base_root: &Path,
    ) -> Result<ValidatedEncoderSource> {
        if !matches!(source, WeightsSource::File(_)) {
            return Err(Error::Unsupported(
                "ComfyUI text encoder validation requires one pinned File source".into(),
            ));
        }
        let mut validated = self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::ComfyUiFp8,
            EncoderConfigPolicy::ProviderOwnedComfyUi,
            None,
        )?;
        validated.tokenizer = Some(self.bind_tokenizer(base_root, source)?);
        validated.ensure_unchanged()?;
        Ok(validated)
    }

    pub(super) fn validate_source_with_policy(
        &self,
        source: &WeightsSource,
        dtype_policy: EncoderDtypePolicy,
        config_policy: EncoderConfigPolicy,
        tokenizer: Option<ValidatedTokenizerSource>,
    ) -> Result<ValidatedEncoderSource> {
        self.validate_definition()?;
        let (weights, own_config) = resolve_source(source)?;
        let own_config = own_config.filter(|path| path.is_file());
        let config_path = match &weights {
            WeightsSource::File(_) => own_config,
            WeightsSource::Dir(_) => own_config,
        };
        let config_candidate = config_path.clone().or_else(|| match &weights {
            WeightsSource::File(path) => path.parent().map(|parent| parent.join("config.json")),
            WeightsSource::Dir(path) => Some(path.join("config.json")),
        });
        let pinned = PinnedEncoderSource::pin(&weights, config_candidate)?;
        let config = if let Some(config_path) = &config_path {
            let config_pin = pinned.config_pin.as_ref().ok_or_else(|| {
                Error::Unsupported(format!(
                    "text encoder config disappeared during validation: {}",
                    config_path.display()
                ))
            })?;
            let config = config_pin.read_unchanged(read_selected_encoder_config)?;
            Some(config)
        } else {
            None
        };
        let headers = pinned.headers().map_err(|error| {
            Error::Msg(format!(
                "text encoder contract: inspect {}: {error}",
                source_path(&weights).display()
            ))
        })?;
        let packed_quant = self.validate_inspected_source(
            source,
            &weights,
            config_path.as_deref().zip(config.as_ref()),
            &headers,
            dtype_policy,
            config_policy,
        )?;
        pinned.ensure_unchanged()?;
        Ok(ValidatedEncoderSource {
            requested_source: source.clone(),
            weights,
            config_path,
            packed_quant,
            pinned,
            tokenizer,
        })
    }

    pub(super) fn validate_inspected_source(
        &self,
        requested_source: &WeightsSource,
        weights: &WeightsSource,
        config: Option<(&Path, &Value)>,
        headers: &[SafetensorsTensorHeader],
        dtype_policy: EncoderDtypePolicy,
        config_policy: EncoderConfigPolicy,
    ) -> Result<Option<PackedQuantization>> {
        let declared_quant = if let Some((config_path, config)) = config {
            self.validate_config(config, config_path)?;
            parse_packed_quantization(config, config_path)?
        } else if config_policy.requires_config() {
            return Err(Error::Msg(format!(
                "text encoder substitution has no config.json (source {}); exact behavior, tokenizer, head-topology, and precision compatibility cannot be proven from tensor shapes alone",
                source_path(requested_source).display()
            )));
        } else {
            None
        };
        let language_headers = language_quantization_evidence_headers(headers);
        let packed_quant = validate_quantization_evidence(
            &language_headers,
            source_path(weights),
            declared_quant,
        )?;
        self.validate_packing_config(
            source_path(weights),
            packed_quant,
            matches!(weights, WeightsSource::File(_)),
        )?;
        self.validate_headers(headers, source_path(weights), packed_quant, dtype_policy)?;
        Ok(packed_quant)
    }

    /// Validate an encoder embedded inside a fused safetensors checkpoint. `component_prefixes`
    /// are stripped before applying this contract's normal architecture and geometry checks. The
    /// caller must retain its own [`PinnedWeightsFile`] for the fused payload; this method pins the
    /// same file across header inspection so the validation itself cannot race a replacement.
    pub fn validate_embedded_file(&self, file: &Path, component_prefixes: &[&str]) -> Result<()> {
        self.validate_embedded_file_with_policy(
            file,
            component_prefixes,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
        )
    }

    /// Validate a fused ComfyUI checkpoint whose selected route explicitly normalizes FP8 tensors
    /// before constructing the native encoder.
    pub fn validate_embedded_comfyui_file(
        &self,
        file: &Path,
        component_prefixes: &[&str],
    ) -> Result<()> {
        self.validate_embedded_file_with_policy(
            file,
            component_prefixes,
            EncoderDtypePolicy::ComfyUiFp8,
            EncoderConfigPolicy::ProviderOwnedComfyUi,
        )
    }

    /// Validate a fused ComfyUI encoder and return the exact retained tokenizer receipt that must be
    /// used by the eventual runtime parser.
    pub fn validate_embedded_comfyui_file_against_base(
        &self,
        file: &Path,
        component_prefixes: &[&str],
        base_root: &Path,
    ) -> Result<ValidatedTokenizerSource> {
        let tokenizer = self.bind_tokenizer(base_root, &WeightsSource::File(file.to_path_buf()))?;
        self.validate_embedded_comfyui_file(file, component_prefixes)?;
        tokenizer.ensure_unchanged()?;
        Ok(tokenizer)
    }

    pub(super) fn validate_embedded_file_with_policy(
        &self,
        file: &Path,
        component_prefixes: &[&str],
        dtype_policy: EncoderDtypePolicy,
        config_policy: EncoderConfigPolicy,
    ) -> Result<()> {
        self.validate_definition()?;
        let pin = PinnedWeightsFile::pin(file)?;
        let config_path = file
            .parent()
            .map(|parent| parent.join("config.json"))
            .ok_or_else(|| {
                Error::Unsupported(format!(
                "embedded text encoder {} has no parent directory for an authoritative config.json",
                file.display()
            ))
            })?;
        if config_policy.requires_config() && !config_path.is_file() {
            return Err(Error::Unsupported(format!(
                "embedded text encoder {} requires sibling config.json; tensor headers cannot prove behavior, tokenizer, or head topology",
                file.display()
            )));
        }
        let config_pin = config_path
            .is_file()
            .then(|| PinnedWeightsFile::pin(&config_path))
            .transpose()?;
        if let Some(config_pin) = config_pin.as_ref() {
            let config: Value = config_pin.read_unchanged(|path| {
                let text = std::fs::read_to_string(path).map_err(|error| {
                    Error::Msg(format!(
                        "text encoder contract: read {}: {error}",
                        path.display()
                    ))
                })?;
                serde_json::from_str(&text).map_err(|error| {
                    Error::Msg(format!(
                        "text encoder contract: parse {}: {error}",
                        path.display()
                    ))
                })
            })?;
            self.validate_config(&config, &config_path)?;
            if parse_packed_quantization(&config, &config_path)?.is_some() {
                return Err(Error::Unsupported(format!(
                    "embedded text encoder {} cannot use a packed quantization marker; fused imports support dense or explicitly normalized ComfyUI FP8 tensors only",
                    file.display()
                )));
            }
        }
        let headers = pin.read_unchanged(|path| {
            safetensors_path_tensor_headers(path).map_err(|error| {
                Error::Msg(format!(
                    "text encoder contract: inspect embedded source {}: {error}",
                    path.display()
                ))
            })
        })?;
        let encoder_prefix = self.expected_header_prefix()?;
        let embedding_suffix = format!("{encoder_prefix}.embed_tokens.weight");
        let matching = component_prefixes
            .iter()
            .copied()
            .filter(|prefix| {
                let embedding = format!("{prefix}{embedding_suffix}");
                headers.iter().any(|header| header.name == embedding)
            })
            .collect::<Vec<_>>();
        let component_prefix = match matching.as_slice() {
            [prefix] => *prefix,
            [] => {
                return Err(self.mismatch(
                    file,
                    "embedded_architecture_header",
                    format!("one of {component_prefixes:?} followed by {embedding_suffix}"),
                    "missing",
                ))
            }
            _ => {
                return Err(self.mismatch(
                    file,
                    "embedded_architecture_header",
                    "one unambiguous text encoder component",
                    format!("multiple prefixes {matching:?}"),
                ))
            }
        };
        let embedded = headers
            .into_iter()
            .filter_map(|header| {
                header
                    .name
                    .strip_prefix(component_prefix)
                    .map(|name| SafetensorsTensorHeader {
                        name: name.to_owned(),
                        ..header
                    })
            })
            .collect::<Vec<_>>();
        self.validate_headers(&embedded, file, None, dtype_policy)?;
        if let Some(config_pin) = config_pin {
            config_pin.ensure_unchanged()?;
        } else if config_path.is_file() {
            return Err(Error::Unsupported(format!(
                "embedded text encoder config appeared during validation: {}",
                config_path.display()
            )));
        }
        pin.ensure_unchanged()?;
        Ok(())
    }

    pub(super) fn validate_config(&self, config: &Value, path: &Path) -> Result<()> {
        self.validate_definition()?;
        let text = match config.get("text_config") {
            Some(text) => text
                .as_object()
                .map(|_| text)
                .ok_or_else(|| self.mismatch(path, "text_config", "object", "non-object"))?,
            None => config,
        };
        self.validate_architecture_config(text, path)?;

        self.expect_json_usize(text, path, "hidden_size", self.hidden_size)?;
        self.expect_json_usize(text, path, "intermediate_size", self.intermediate_size)?;
        self.expect_json_usize(text, path, "num_hidden_layers", self.num_hidden_layers)?;
        self.expect_json_usize(text, path, "num_attention_heads", self.num_attention_heads)?;
        self.expect_json_usize(text, path, "num_key_value_heads", self.num_key_value_heads)?;
        let config_head_dim = match text.get("head_dim") {
            None | Some(Value::Null) => self
                .hidden_size
                .is_multiple_of(self.num_attention_heads)
                .then_some(self.hidden_size / self.num_attention_heads),
            Some(value) => value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| self.mismatch(path, "head_dim", self.head_dim, value))
                .map(Some)?,
        };
        match config_head_dim {
            Some(actual) if actual == self.head_dim => {}
            Some(actual) => {
                return Err(self.mismatch(path, "head_dim", self.head_dim, actual));
            }
            None => return Err(self.mismatch(path, "head_dim", self.head_dim, "missing")),
        }
        self.expect_json_usize(text, path, "vocab_size", self.vocab_size)?;
        self.expect_json_str(text, path, "hidden_act", self.hidden_activation)?;
        self.expect_json_f64(
            text,
            path,
            "attention_dropout",
            self.attention_dropout.get(),
        )?;
        self.expect_json_f64(text, path, "rms_norm_eps", self.rms_norm_eps.get())?;
        let rope_aliases = [
            &["rope_theta"][..],
            &["rope_parameters", "rope_theta"][..],
            &["rope_scaling", "rope_theta"][..],
        ];
        self.expect_json_f64_from_all_aliases(
            text,
            path,
            "rope_theta",
            self.rope_theta.get(),
            &rope_aliases,
        )?;
        if !std::ptr::eq(config, text) {
            self.validate_optional_json_f64_aliases(
                config,
                path,
                "rope_theta",
                self.rope_theta.get(),
                &rope_aliases,
            )?;
        }
        self.expect_json_usize(
            text,
            path,
            "max_position_embeddings",
            self.max_position_embeddings,
        )?;
        self.validate_json_bool_aliases(config, text, path, "attention_bias", self.attention_bias)?;
        self.validate_json_bool_aliases(
            config,
            text,
            path,
            "tie_word_embeddings",
            self.tie_word_embeddings,
        )?;
        for (field, expected) in [
            ("bos_token_id", self.bos_token_id),
            ("eos_token_id", self.eos_token_id),
            ("image_token_id", self.image_token_id),
            ("vision_start_token_id", self.vision_start_token_id),
            ("vision_end_token_id", self.vision_end_token_id),
        ] {
            self.validate_json_i64_aliases(config, text, path, field, expected)?;
        }
        for required in self.tokenizer.required_tokens {
            if let Some(field) = required.config_field {
                self.validate_json_i64_aliases(config, text, path, field, Some(required.id))?;
            }
        }
        self.validate_mrope_section_aliases(text, path, true)?;
        self.validate_mrope_interleaving_aliases(text, path, true)?;
        self.validate_rope_type_aliases(text, path)?;
        if !std::ptr::eq(config, text) {
            self.validate_mrope_section_aliases(config, path, false)?;
            self.validate_mrope_interleaving_aliases(config, path, false)?;
            self.validate_rope_type_aliases(config, path)?;
        }
        let allowed_rope_fields = BTreeSet::from([
            "mrope_interleaved",
            "mrope_section",
            "rope_theta",
            "rope_type",
            "type",
        ]);
        for (scope_name, scope) in [("text_config", text), ("root", config)] {
            if scope_name == "root" && std::ptr::eq(config, text) {
                continue;
            }
            for object_name in ["rope_parameters", "rope_scaling"] {
                let Some(value) = scope.get(object_name) else {
                    continue;
                };
                if value.is_null() {
                    continue;
                }
                let Some(object) = value.as_object() else {
                    return Err(self.mismatch(
                        path,
                        "rope_behavior_fields",
                        format!("{object_name} object or null"),
                        format!("{scope_name}={value}"),
                    ));
                };
                let extra = object
                    .keys()
                    .filter(|field| !allowed_rope_fields.contains(field.as_str()))
                    .cloned()
                    .collect::<Vec<_>>();
                if !extra.is_empty() {
                    return Err(self.mismatch(
                        path,
                        "rope_behavior_fields",
                        format!("only {allowed_rope_fields:?}"),
                        format!("unsupported {scope_name}.{object_name} keys {extra:?}"),
                    ));
                }
            }
        }
        if let Some(layer_types) = text.get("layer_types") {
            let layer_types = layer_types.as_array().ok_or_else(|| {
                self.mismatch(
                    path,
                    "layer_types",
                    "array of full_attention strings",
                    layer_types,
                )
            })?;
            if layer_types.len() < self.loaded_hidden_layers {
                return Err(self.mismatch(
                    path,
                    "layer_types",
                    format!("at least {} entries", self.loaded_hidden_layers),
                    layer_types.len(),
                ));
            }
            for (index, value) in layer_types
                .iter()
                .take(self.loaded_hidden_layers)
                .enumerate()
            {
                if value.as_str() != Some("full_attention") {
                    return Err(self.mismatch(
                        path,
                        "layer_types",
                        "full_attention for every loaded layer",
                        format!("index {index}={value}"),
                    ));
                }
            }
        }
        match text.get("sliding_window") {
            None | Some(Value::Null) => {}
            Some(value) => {
                return Err(self.mismatch(path, "sliding_window", "missing or null", value));
            }
        }
        match text.get("use_sliding_window") {
            None | Some(Value::Null) | Some(Value::Bool(false)) => {}
            Some(value) => {
                return Err(self.mismatch(path, "use_sliding_window", false, value));
            }
        }
        Ok(())
    }

    pub(super) fn expect_json_str(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: &str,
    ) -> Result<()> {
        match config.get(field).and_then(Value::as_str) {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    pub(super) fn validate_architecture_config(&self, text: &Value, path: &Path) -> Result<()> {
        let mut found = false;
        if let Some(value) = text.get("model_type") {
            let actual = value.as_str().ok_or_else(|| {
                self.mismatch(path, "architecture.model_type", self.architecture, value)
            })?;
            found = true;
            if !architecture_matches(actual, self.architecture) {
                return Err(self.mismatch(
                    path,
                    "architecture.model_type",
                    self.architecture,
                    actual,
                ));
            }
        }
        if let Some(value) = text.get("architectures") {
            let values = value.as_array().ok_or_else(|| {
                self.mismatch(
                    path,
                    "architecture.architectures",
                    "array of strings",
                    value,
                )
            })?;
            for (index, value) in values.iter().enumerate() {
                let actual = value.as_str().ok_or_else(|| {
                    self.mismatch(
                        path,
                        "architecture.architectures",
                        "array of strings",
                        format!("index {index}={value}"),
                    )
                })?;
                found = true;
                if !architecture_matches(actual, self.architecture) {
                    return Err(self.mismatch(
                        path,
                        "architecture.architectures",
                        self.architecture,
                        format!("index {index}={actual}"),
                    ));
                }
            }
        }
        if found {
            Ok(())
        } else {
            Err(self.mismatch(path, "architecture", self.architecture, "missing"))
        }
    }

    pub(super) fn expect_json_f64(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: f64,
    ) -> Result<()> {
        match config.get(field).and_then(Value::as_f64) {
            Some(actual) if actual.to_bits() == expected.to_bits() => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    pub(super) fn expect_json_f64_from_all_aliases(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: f64,
        candidates: &[&[&str]],
    ) -> Result<()> {
        let mut found = false;
        for candidate in candidates {
            let Some(value) = json_path(config, candidate) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value.as_f64().ok_or_else(|| {
                self.mismatch(
                    path,
                    field,
                    expected,
                    format!("{}={value}", candidate.join(".")),
                )
            })?;
            found = true;
            if actual.to_bits() != expected.to_bits() {
                return Err(self.mismatch(
                    path,
                    field,
                    expected,
                    format!("{}={actual}", candidate.join(".")),
                ));
            }
        }
        if found {
            Ok(())
        } else {
            Err(self.mismatch(path, field, expected, "missing"))
        }
    }

    pub(super) fn validate_optional_json_f64_aliases(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: f64,
        candidates: &[&[&str]],
    ) -> Result<()> {
        for candidate in candidates {
            let Some(value) = json_path(config, candidate) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value.as_f64().ok_or_else(|| {
                self.mismatch(
                    path,
                    field,
                    expected,
                    format!("{}={value}", candidate.join(".")),
                )
            })?;
            if actual.to_bits() != expected.to_bits() {
                return Err(self.mismatch(
                    path,
                    field,
                    expected,
                    format!("{}={actual}", candidate.join(".")),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn root_text_aliases<'a>(
        root: &'a Value,
        text: &'a Value,
        field: &str,
    ) -> Vec<(&'static str, &'a Value)> {
        let mut values = Vec::with_capacity(2);
        if let Some(value) = text.get(field) {
            values.push(("text_config", value));
        }
        if !std::ptr::eq(root, text) {
            if let Some(value) = root.get(field) {
                values.push(("root", value));
            }
        }
        values
    }

    pub(super) fn validate_json_bool_aliases(
        &self,
        root: &Value,
        text: &Value,
        path: &Path,
        field: &'static str,
        contract: EncoderConfigBool,
    ) -> Result<()> {
        let mut first = None;
        let expected = contract.effective();
        for (location, value) in Self::root_text_aliases(root, text, field) {
            if value.is_null() {
                continue;
            }
            let actual = value.as_bool().ok_or_else(|| {
                self.mismatch(path, field, "boolean", format!("{location}={value}"))
            })?;
            if actual != expected {
                return Err(self.mismatch(path, field, expected, format!("{location}={actual}")));
            }
            if first.is_some_and(|first| first != actual) {
                return Err(self.mismatch(
                    path,
                    field,
                    "one consistent root/text_config value",
                    format!("{location}={actual}"),
                ));
            }
            first = Some(actual);
        }
        if first.is_none() && contract.is_required() {
            Err(self.mismatch(path, field, expected, "missing"))
        } else {
            Ok(())
        }
    }

    pub(super) fn validate_json_i64_aliases(
        &self,
        root: &Value,
        text: &Value,
        path: &Path,
        field: &'static str,
        expected: Option<i64>,
    ) -> Result<()> {
        let mut first = None;
        for (location, value) in Self::root_text_aliases(root, text, field) {
            if value.is_null() {
                continue;
            }
            let actual = value.as_i64().ok_or_else(|| {
                self.mismatch(path, field, "integer", format!("{location}={value}"))
            })?;
            if let Some(expected) = expected {
                if actual != expected {
                    return Err(self.mismatch(
                        path,
                        field,
                        expected,
                        format!("{location}={actual}"),
                    ));
                }
            }
            if first.is_some_and(|first| first != actual) {
                return Err(self.mismatch(
                    path,
                    field,
                    "one consistent root/text_config value",
                    format!("{location}={actual}"),
                ));
            }
            first = Some(actual);
        }
        match (expected, first) {
            (Some(expected), None) => Err(self.mismatch(path, field, expected, "missing")),
            _ => Ok(()),
        }
    }

    pub(super) fn validate_mrope_section_aliases(
        &self,
        text: &Value,
        path: &Path,
        required: bool,
    ) -> Result<()> {
        let aliases = [
            &["rope_parameters", "mrope_section"][..],
            &["rope_scaling", "mrope_section"][..],
        ];
        let mut found = false;
        for alias in aliases {
            let Some(value) = json_path(text, alias) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value
                .as_array()
                .and_then(|values| {
                    values
                        .iter()
                        .map(|value| value.as_u64().and_then(|value| usize::try_from(value).ok()))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| {
                    self.mismatch(
                        path,
                        "mrope_section",
                        format!("{:?}", self.mrope_section),
                        format!("{}={value}", alias.join(".")),
                    )
                })?;
            found = true;
            if actual != self.mrope_section {
                return Err(self.mismatch(
                    path,
                    "mrope_section",
                    format!("{:?}", self.mrope_section),
                    format!("{}={actual:?}", alias.join(".")),
                ));
            }
        }
        if required && !self.mrope_section.is_empty() && !found {
            Err(self.mismatch(
                path,
                "mrope_section",
                format!("{:?}", self.mrope_section),
                "missing",
            ))
        } else {
            Ok(())
        }
    }

    pub(super) fn validate_mrope_interleaving_aliases(
        &self,
        text: &Value,
        path: &Path,
        required: bool,
    ) -> Result<()> {
        let aliases = [
            &["rope_parameters", "mrope_interleaved"][..],
            &["rope_scaling", "mrope_interleaved"][..],
        ];
        let expected = self.mrope_interleaved.unwrap_or(false);
        let mut found = false;
        for alias in aliases {
            let Some(value) = json_path(text, alias) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value.as_bool().ok_or_else(|| {
                self.mismatch(
                    path,
                    "mrope_interleaved",
                    expected,
                    format!("{}={value}", alias.join(".")),
                )
            })?;
            found = true;
            if actual != expected {
                return Err(self.mismatch(
                    path,
                    "mrope_interleaved",
                    expected,
                    format!("{}={actual}", alias.join(".")),
                ));
            }
        }
        if required && self.mrope_interleaved.is_some() && !found {
            Err(self.mismatch(path, "mrope_interleaved", expected, "missing"))
        } else {
            Ok(())
        }
    }

    pub(super) fn validate_rope_type_aliases(&self, text: &Value, path: &Path) -> Result<()> {
        let aliases = [
            &["rope_type"][..],
            &["rope_parameters", "rope_type"][..],
            &["rope_parameters", "type"][..],
            &["rope_scaling", "rope_type"][..],
            &["rope_scaling", "type"][..],
        ];
        for alias in aliases {
            let Some(value) = json_path(text, alias) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value.as_str().ok_or_else(|| {
                self.mismatch(
                    path,
                    "rope_type",
                    "default",
                    format!("{}={value}", alias.join(".")),
                )
            })?;
            if actual != "default" {
                return Err(self.mismatch(
                    path,
                    "rope_type",
                    "default",
                    format!("{}={actual}", alias.join(".")),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn expect_json_usize(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: usize,
    ) -> Result<()> {
        match config
            .get(field)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
        {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    pub(super) fn validate_headers(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        packed_quant: Option<PackedQuantization>,
        dtype_policy: EncoderDtypePolicy,
    ) -> Result<()> {
        let prefix = self.expected_header_prefix()?;
        self.validate_architecture_signature(headers, path, prefix)?;
        let lm_head_prefix = self
            .requires_lm_head
            .then(|| self.expected_lm_head_prefix())
            .transpose()?
            .map(|prefix| format!("{prefix}.lm_head."));
        let lm_head_name = lm_head_prefix
            .as_ref()
            .map(|prefix| format!("{prefix}weight"));
        let relevant = headers
            .iter()
            .filter(|header| {
                header.name.starts_with(&format!("{prefix}."))
                    || lm_head_prefix
                        .as_ref()
                        .is_some_and(|prefix| header.name.starts_with(prefix))
            })
            .cloned()
            .collect::<Vec<_>>();
        let headers = relevant.as_slice();
        let packing = packed_quant.and(self.packing);
        let mut allowed_packed_bases = BTreeSet::new();
        let embedding_name = format!("{prefix}.embed_tokens.weight");
        let embedding_base = embedding_name
            .strip_suffix(".weight")
            .expect("embedding name has weight suffix");
        let pack_embedding = packing.is_some_and(|packing| packing.pack_embedding);
        if pack_embedding {
            allowed_packed_bases.insert(embedding_base.to_owned());
        }
        self.validate_matrix(
            headers,
            path,
            MatrixExpectation {
                name: &embedding_name,
                field: "vocab_size",
                shape: [self.vocab_size, self.hidden_size],
                must_be_packed: pack_embedding,
            },
            packed_quant,
            dtype_policy,
        )?;
        let layer_marker = format!("{prefix}.layers.");
        let layers: BTreeSet<usize> = headers
            .iter()
            .filter_map(|header| {
                header
                    .name
                    .strip_prefix(&layer_marker)
                    .and_then(|tail| tail.split('.').next())
                    .and_then(|index| index.parse().ok())
            })
            .collect();
        let missing_loaded_layers = (0..self.loaded_hidden_layers)
            .filter(|layer| !layers.contains(layer))
            .collect::<Vec<_>>();
        let out_of_contract_layers = layers
            .iter()
            .copied()
            .filter(|&layer| layer >= self.num_hidden_layers)
            .collect::<Vec<_>>();
        if !missing_loaded_layers.is_empty() || !out_of_contract_layers.is_empty() {
            return Err(self.mismatch(
                path,
                "loaded_hidden_layers",
                format!(
                    "every layer 0..{} and no layer >= {}",
                    self.loaded_hidden_layers, self.num_hidden_layers
                ),
                format!(
                    "missing={missing_loaded_layers:?}, out_of_contract={out_of_contract_layers:?}"
                ),
            ));
        }

        let attention_width = self
            .num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| {
                self.mismatch(
                    path,
                    "attention_width",
                    "non-overflowing head count × head dimension",
                    format!("{} × {}", self.num_attention_heads, self.head_dim),
                )
            })?;
        let kv_width = self
            .num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| {
                self.mismatch(
                    path,
                    "key_value_width",
                    "non-overflowing head count × head dimension",
                    format!("{} × {}", self.num_key_value_heads, self.head_dim),
                )
            })?;
        for layer in 0..self.loaded_hidden_layers {
            let base = format!("{prefix}.layers.{layer}");
            for (suffix, field, shape) in [
                (
                    "self_attn.q_proj.weight",
                    "num_attention_heads/head_dim",
                    [attention_width, self.hidden_size],
                ),
                (
                    "self_attn.k_proj.weight",
                    "num_key_value_heads/head_dim",
                    [kv_width, self.hidden_size],
                ),
                (
                    "self_attn.v_proj.weight",
                    "num_key_value_heads/head_dim",
                    [kv_width, self.hidden_size],
                ),
                (
                    "self_attn.o_proj.weight",
                    "attention_output_width",
                    [self.hidden_size, attention_width],
                ),
                (
                    "mlp.gate_proj.weight",
                    "intermediate_size",
                    [self.intermediate_size, self.hidden_size],
                ),
                (
                    "mlp.up_proj.weight",
                    "intermediate_size",
                    [self.intermediate_size, self.hidden_size],
                ),
                (
                    "mlp.down_proj.weight",
                    "intermediate_size",
                    [self.hidden_size, self.intermediate_size],
                ),
            ] {
                let name = format!("{base}.{suffix}");
                if packing.is_some() {
                    allowed_packed_bases.insert(
                        name.strip_suffix(".weight")
                            .expect("projection name has weight suffix")
                            .to_owned(),
                    );
                }
                self.validate_matrix(
                    headers,
                    path,
                    MatrixExpectation {
                        name: &name,
                        field,
                        shape,
                        must_be_packed: packing.is_some(),
                    },
                    packed_quant,
                    dtype_policy,
                )?;
            }
            self.validate_vector(
                headers,
                path,
                &format!("{base}.input_layernorm.weight"),
                "hidden_size",
                self.hidden_size,
                dtype_policy,
            )?;
            self.validate_vector(
                headers,
                path,
                &format!("{base}.post_attention_layernorm.weight"),
                "hidden_size",
                self.hidden_size,
                dtype_policy,
            )?;
            match self.architecture {
                "qwen3" | "qwen3_vl_text" => {
                    self.validate_vector(
                        headers,
                        path,
                        &format!("{base}.self_attn.q_norm.weight"),
                        "head_dim",
                        self.head_dim,
                        dtype_policy,
                    )?;
                    self.validate_vector(
                        headers,
                        path,
                        &format!("{base}.self_attn.k_norm.weight"),
                        "head_dim",
                        self.head_dim,
                        dtype_policy,
                    )?;
                }
                "qwen2_5_vl_text" => {
                    for (projection, width) in [
                        ("q_proj", attention_width),
                        ("k_proj", kv_width),
                        ("v_proj", kv_width),
                    ] {
                        self.validate_vector(
                            headers,
                            path,
                            &format!("{base}.self_attn.{projection}.bias"),
                            "attention_bias_width",
                            width,
                            dtype_policy,
                        )?;
                    }
                }
                "mistral" => {}
                _ => unreachable!("expected_header_prefix rejects unsupported architectures"),
            }
        }
        self.validate_bias_surface(headers, path, prefix)?;
        if self.requires_final_norm {
            self.validate_vector(
                headers,
                path,
                &format!("{prefix}.norm.weight"),
                "hidden_size",
                self.hidden_size,
                dtype_policy,
            )?;
        }
        if self.requires_lm_head {
            let name = lm_head_name.expect("requires_lm_head constructed a validated name");
            let pack_lm_head = packing.is_some_and(|packing| packing.pack_lm_head);
            if pack_lm_head {
                allowed_packed_bases.insert(
                    name.strip_suffix(".weight")
                        .expect("LM-head name has weight suffix")
                        .to_owned(),
                );
            }
            self.validate_matrix(
                headers,
                path,
                MatrixExpectation {
                    name: &name,
                    field: "lm_head_output_width",
                    shape: [self.vocab_size, self.hidden_size],
                    must_be_packed: pack_lm_head,
                },
                packed_quant,
                dtype_policy,
            )?;
        }
        self.validate_all_packed_triples(headers, path, packed_quant, &allowed_packed_bases)?;
        self.validate_dense_storage_probe(headers, path)?;
        Ok(())
    }

    pub(super) fn validate_dense_storage_probe(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
    ) -> Result<()> {
        let Some(probe_name) = self.dense_storage_dtype_probe else {
            return Ok(());
        };
        let probe = headers
            .iter()
            .find(|header| header.name == probe_name)
            .ok_or_else(|| {
                self.mismatch(path, "dense_storage_dtype_probe", probe_name, "missing")
            })?;
        // Candle Krea first opens at BF16 and retains that store only when this native probe is
        // itself BF16. In that branch every consumed dense matrix must already be BF16 or the open
        // would silently narrow it. A F16/F32 probe makes the loader reopen the whole component at
        // F32, where every accepted native float dtype is widened safely and need not be uniform.
        if probe.dtype != Dtype::BF16 {
            return Ok(());
        }
        let prefix = self.expected_header_prefix()?;
        let lm_head_name = self
            .requires_lm_head
            .then(|| self.expected_lm_head_prefix())
            .transpose()?
            .map(|prefix| format!("{prefix}.lm_head.weight"));
        for header in headers.iter().filter(|header| {
            self.is_consumed_dense_store_tensor(&header.name, prefix, lm_head_name.as_deref())
                && matches!(header.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
        }) {
            if header.dtype != Dtype::BF16 {
                return Err(self.mismatch(
                    path,
                    "dense_storage_dtype",
                    format!("BF16 selected by {probe_name}"),
                    format!("{}={:?}", header.name, header.dtype),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn is_consumed_dense_store_tensor(
        &self,
        name: &str,
        prefix: &str,
        lm_head_name: Option<&str>,
    ) -> bool {
        if name == format!("{prefix}.embed_tokens.weight") || Some(name) == lm_head_name {
            return true;
        }
        let Some(tail) = name.strip_prefix(&format!("{prefix}.layers.")) else {
            return false;
        };
        let Some((layer, suffix)) = tail.split_once('.') else {
            return false;
        };
        let Ok(layer) = layer.parse::<usize>() else {
            return false;
        };
        if layer >= self.loaded_hidden_layers {
            return false;
        }
        matches!(
            suffix,
            "self_attn.q_proj.weight"
                | "self_attn.k_proj.weight"
                | "self_attn.v_proj.weight"
                | "self_attn.o_proj.weight"
                | "mlp.gate_proj.weight"
                | "mlp.up_proj.weight"
                | "mlp.down_proj.weight"
                | "self_attn.q_proj.bias"
                | "self_attn.k_proj.bias"
                | "self_attn.v_proj.bias"
        )
    }

    pub(super) fn validate_bias_surface(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        prefix: &str,
    ) -> Result<()> {
        let mut allowed = BTreeSet::new();
        if self.architecture == "qwen2_5_vl_text" {
            for layer in 0..self.loaded_hidden_layers {
                for projection in ["q_proj", "k_proj", "v_proj"] {
                    allowed.insert(format!(
                        "{prefix}.layers.{layer}.self_attn.{projection}.bias"
                    ));
                }
            }
        }
        let actual = headers
            .iter()
            .filter(|header| {
                let Some(layer) = architecture_layer_index(&header.name, prefix) else {
                    return false;
                };
                layer < self.loaded_hidden_layers && header.name.ends_with(".bias")
            })
            .map(|header| header.name.clone())
            .collect::<BTreeSet<_>>();
        if actual != allowed {
            return Err(self.mismatch(
                path,
                "projection_bias_surface",
                format!("{allowed:?}"),
                format!("{actual:?}"),
            ));
        }
        Ok(())
    }

    pub(super) fn validate_packing_config(
        &self,
        path: &Path,
        packed: Option<PackedQuantization>,
        is_file: bool,
    ) -> Result<()> {
        match (packed, self.packing) {
            (None, _) => Ok(()),
            (Some(_), Some(expected)) if is_file && !expected.supports_file => Err(self.mismatch(
                path,
                "packed_file_support",
                "directory-backed packed encoder",
                "single File source",
            )),
            (Some(actual), Some(expected)) if actual.group_size == expected.group_size => Ok(()),
            (Some(actual), Some(expected)) => Err(self.mismatch(
                path,
                "quantization.group_size",
                expected.group_size,
                actual.group_size,
            )),
            (Some(actual), None) => Err(self.mismatch(
                path,
                "quantization",
                "dense-only encoder",
                format!("Q{} group_size {}", actual.bits, actual.group_size),
            )),
        }
    }

    pub(super) fn validate_vector(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        name: &str,
        field: &'static str,
        expected: usize,
        dtype_policy: EncoderDtypePolicy,
    ) -> Result<()> {
        let Some(tensor) = headers.iter().find(|header| header.name == name) else {
            return Err(self.mismatch(
                path,
                field,
                format!("[{expected}]"),
                format!("missing {name}"),
            ));
        };
        if tensor.shape != [expected] {
            return Err(self.mismatch(
                path,
                field,
                format!("[{expected}]"),
                format!("{:?}", tensor.shape),
            ));
        }
        if !dtype_policy.accepts_dense(tensor.dtype) {
            return Err(self.mismatch(
                path,
                "tensor_dtype",
                dtype_policy.expected_dense(),
                format!("{}={:?}", tensor.name, tensor.dtype),
            ));
        }
        Ok(())
    }

    pub(super) fn validate_all_packed_triples(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        packed_quant: Option<PackedQuantization>,
        allowed_packed_bases: &BTreeSet<String>,
    ) -> Result<()> {
        let find = |name: &str| headers.iter().find(|header| header.name == name);
        let prefix = self.expected_header_prefix()?;
        let is_unloaded_architecture_layer = |name: &str| {
            architecture_layer_index(name, prefix).is_some_and(|layer| {
                layer >= self.loaded_hidden_layers && layer < self.num_hidden_layers
            })
        };
        for weight in headers.iter().filter(|header| {
            header.name.ends_with(".weight")
                && (header.dtype == Dtype::U32
                    || find(&format!(
                        "{}.scales",
                        header.name.strip_suffix(".weight").unwrap_or(&header.name)
                    ))
                    .is_some()
                    || find(&format!(
                        "{}.biases",
                        header.name.strip_suffix(".weight").unwrap_or(&header.name)
                    ))
                    .is_some())
        }) {
            let base = weight.name.strip_suffix(".weight").unwrap_or(&weight.name);
            if is_unloaded_architecture_layer(base) {
                continue;
            }
            if !allowed_packed_bases.contains(base) {
                return Err(self.mismatch(
                    path,
                    "packed_surface",
                    format!("exactly {allowed_packed_bases:?}"),
                    format!("unexpected packed tensor {base}"),
                ));
            }
            let scales = find(&format!("{base}.scales"));
            let biases = find(&format!("{base}.biases"));
            let (Some(scales), Some(biases)) = (scales, biases) else {
                return Err(self.mismatch(
                    path,
                    "packed_components",
                    format!("{base}.weight + .scales + .biases"),
                    "incomplete packed triple",
                ));
            };
            let quant = packed_quant.ok_or_else(|| {
                self.mismatch(
                    path,
                    "quantization",
                    "sibling config.json with bits/group_size",
                    format!("packed tensor {base} without authoritative metadata"),
                )
            })?;
            if weight.dtype != Dtype::U32
                || !matches!(scales.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                || !matches!(biases.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                || weight.shape.len() != 2
                || scales.shape.len() != 2
                || biases.shape != scales.shape
                || weight.shape[0] != scales.shape[0]
            {
                return Err(self.mismatch(
                    path,
                    "packed_triple",
                    "U32 rank-2 weight plus matching float rank-2 scales/biases",
                    format!(
                        "weight={:?}/{:?}, scales={:?}/{:?}, biases={:?}/{:?}",
                        weight.dtype,
                        weight.shape,
                        scales.dtype,
                        scales.shape,
                        biases.dtype,
                        biases.shape
                    ),
                ));
            }
            let logical_input = weight.shape[1]
                .checked_mul(32)
                .and_then(|value| value.checked_div(quant.bits))
                .ok_or_else(|| {
                    self.mismatch(path, "packed_geometry", "non-overflowing matrix", base)
                })?;
            if !logical_input.is_multiple_of(quant.group_size)
                || scales.shape[1] != logical_input / quant.group_size
            {
                return Err(self.mismatch(
                    path,
                    "packed_affine_shape",
                    format!(
                        "[{}, {}] from bits={} group_size={}",
                        weight.shape[0],
                        logical_input / quant.group_size,
                        quant.bits,
                        quant.group_size
                    ),
                    format!("scales={:?}, biases={:?}", scales.shape, biases.shape),
                ));
            }
        }
        for affine in headers
            .iter()
            .filter(|header| header.name.ends_with(".scales") || header.name.ends_with(".biases"))
        {
            let base = affine
                .name
                .strip_suffix(".scales")
                .or_else(|| affine.name.strip_suffix(".biases"))
                .expect("suffix filter guarantees one match");
            if is_unloaded_architecture_layer(base) {
                continue;
            }
            if find(&format!("{base}.weight")).is_none()
                || find(&format!("{base}.scales")).is_none()
                || find(&format!("{base}.biases")).is_none()
            {
                return Err(self.mismatch(
                    path,
                    "packed_components",
                    format!("{base}.weight + .scales + .biases"),
                    "orphaned packed component",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn expected_header_prefix(&self) -> Result<&'static str> {
        match self.architecture {
            "qwen3" | "qwen2_5_vl_text" => Ok("model"),
            "qwen3_vl_text" => Ok("language_model"),
            "mistral" => Ok("language_model.model"),
            architecture => Err(Error::Unsupported(format!(
                "text encoder contract has no header signature for architecture {architecture}"
            ))),
        }
    }

    pub(super) fn expected_lm_head_prefix(&self) -> Result<&'static str> {
        match self.architecture {
            "mistral" => Ok("language_model"),
            architecture => Err(Error::Unsupported(format!(
                "text encoder contract requests an LM head but has no LM-head signature for architecture {architecture}"
            ))),
        }
    }

    /// Exact tensor names retained by the concrete language constructors for this contract.
    ///
    /// Packed matrices are three independent safetensors payloads. Keep the affine tables in the
    /// selected surface alongside their code tensor, while ignoring arbitrary keys that merely share
    /// a loaded layer prefix.
    pub(super) fn materialized_language_tensor_names(
        &self,
        packing: Option<EncoderPackingContract>,
    ) -> Result<BTreeSet<String>> {
        fn insert_matrix(names: &mut BTreeSet<String>, base: String, packed: bool) {
            names.insert(format!("{base}.weight"));
            if packed {
                names.insert(format!("{base}.scales"));
                names.insert(format!("{base}.biases"));
            }
        }

        let prefix = self.expected_header_prefix()?;
        let mut names = BTreeSet::new();
        insert_matrix(
            &mut names,
            format!("{prefix}.embed_tokens"),
            packing.is_some_and(|packing| packing.pack_embedding),
        );
        for layer in 0..self.loaded_hidden_layers {
            let base = format!("{prefix}.layers.{layer}");
            for suffix in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                insert_matrix(&mut names, format!("{base}.{suffix}"), packing.is_some());
            }
            names.extend([
                format!("{base}.input_layernorm.weight"),
                format!("{base}.post_attention_layernorm.weight"),
            ]);
            match self.architecture {
                "qwen3" | "qwen3_vl_text" => names.extend([
                    format!("{base}.self_attn.q_norm.weight"),
                    format!("{base}.self_attn.k_norm.weight"),
                ]),
                "qwen2_5_vl_text" => names.extend([
                    format!("{base}.self_attn.q_proj.bias"),
                    format!("{base}.self_attn.k_proj.bias"),
                    format!("{base}.self_attn.v_proj.bias"),
                ]),
                "mistral" => {}
                _ => unreachable!("expected_header_prefix rejects unsupported architectures"),
            }
        }
        if self.requires_final_norm {
            names.insert(format!("{prefix}.norm.weight"));
        }
        if self.requires_lm_head {
            let prefix = self.expected_lm_head_prefix()?;
            insert_matrix(
                &mut names,
                format!("{prefix}.lm_head"),
                packing.is_some_and(|packing| packing.pack_lm_head),
            );
        }
        Ok(names)
    }

    pub(super) fn validate_architecture_signature(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        prefix: &str,
    ) -> Result<()> {
        let has = |suffix: &str| {
            let name = format!("{prefix}.layers.0.self_attn.{suffix}");
            headers.iter().any(|header| header.name == name)
        };
        let qk_norm = has("q_norm.weight") && has("k_norm.weight");
        let qk_bias = has("q_proj.bias") && has("k_proj.bias");
        let matches = match self.architecture {
            "qwen3" | "qwen3_vl_text" => qk_norm && !qk_bias,
            "qwen2_5_vl_text" => qk_bias && !qk_norm,
            "mistral" => !qk_norm && !qk_bias,
            _ => false,
        };
        if matches {
            Ok(())
        } else {
            Err(self.mismatch(
                path,
                "architecture_header",
                self.architecture,
                format!("prefix={prefix}, qk_norm={qk_norm}, qk_bias={qk_bias}"),
            ))
        }
    }

    pub(super) fn validate_matrix(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        expectation: MatrixExpectation<'_>,
        packed_quant: Option<PackedQuantization>,
        dtype_policy: EncoderDtypePolicy,
    ) -> Result<()> {
        let MatrixExpectation {
            name,
            field,
            shape: [expected_output, expected_input],
            must_be_packed,
        } = expectation;
        let Some(weight) = headers.iter().find(|header| header.name == name) else {
            return Err(self.mismatch(path, field, expected_output, format!("missing {name}")));
        };
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let scales = headers
            .iter()
            .find(|header| header.name == format!("{base}.scales"));
        let biases = headers
            .iter()
            .find(|header| header.name == format!("{base}.biases"));
        match (scales, biases) {
            (None, None) => {
                if must_be_packed {
                    return Err(self.mismatch(
                        path,
                        "packed_surface",
                        format!("packed triple for {base}"),
                        "dense weight",
                    ));
                }
                if !dtype_policy.accepts_dense(weight.dtype) {
                    return Err(self.mismatch(
                        path,
                        "tensor_dtype",
                        dtype_policy.expected_dense(),
                        format!("{}={:?}", weight.name, weight.dtype),
                    ));
                }
                if weight.shape.len() != 2 {
                    return Err(self.mismatch(
                        path,
                        field,
                        format!("[{expected_output}, {expected_input}] dense matrix"),
                        format!("{:?}", weight.shape),
                    ));
                }
                if weight.shape[0] != expected_output {
                    return Err(self.mismatch(path, field, expected_output, weight.shape[0]));
                }
                if weight.shape[1] != expected_input {
                    return Err(self.mismatch(
                        path,
                        "hidden_size",
                        expected_input,
                        weight.shape[1],
                    ));
                }
            }
            (Some(scales), Some(biases)) => {
                if !must_be_packed {
                    return Err(self.mismatch(
                        path,
                        "packed_surface",
                        format!("dense weight for {base}"),
                        "packed triple",
                    ));
                }
                let quant = packed_quant.ok_or_else(|| {
                    self.mismatch(
                        path,
                        "quantization",
                        "sibling config.json with bits/group_size",
                        format!("packed tensor {base} without authoritative metadata"),
                    )
                })?;
                if weight.dtype != Dtype::U32 {
                    return Err(self.mismatch(
                        path,
                        "packed_dtype",
                        "U32",
                        format!("{:?}", weight.dtype),
                    ));
                }
                if !matches!(scales.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                    || !matches!(biases.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                {
                    return Err(self.mismatch(
                        path,
                        "packed_scale_dtype",
                        "F16, BF16, or F32",
                        format!("scales={:?}, biases={:?}", scales.dtype, biases.dtype),
                    ));
                }
                let packed_input_bits =
                    expected_input.checked_mul(quant.bits).ok_or_else(|| {
                        self.mismatch(
                            path,
                            "packed_geometry",
                            "non-overflowing matrix",
                            format!("input={expected_input}, bits={}", quant.bits),
                        )
                    })?;
                if !expected_input.is_multiple_of(quant.group_size)
                    || !packed_input_bits.is_multiple_of(32)
                {
                    return Err(self.mismatch(
                        path,
                        "packed_geometry",
                        "group-aligned Q4/Q8 matrix",
                        format!(
                            "input={expected_input}, bits={}, group_size={}",
                            quant.bits, quant.group_size
                        ),
                    ));
                }
                let expected_weight = vec![expected_output, packed_input_bits / 32];
                let expected_affine = vec![expected_output, expected_input / quant.group_size];
                if weight.shape != expected_weight {
                    return Err(self.mismatch(
                        path,
                        "packed_weight_shape",
                        format!("{expected_weight:?}"),
                        format!("{:?}", weight.shape),
                    ));
                }
                if scales.shape != expected_affine || biases.shape != expected_affine {
                    return Err(self.mismatch(
                        path,
                        "packed_affine_shape",
                        format!("{expected_affine:?}"),
                        format!("scales={:?}, biases={:?}", scales.shape, biases.shape),
                    ));
                }
            }
            _ => {
                return Err(self.mismatch(
                    path,
                    "packed_components",
                    format!("{base}.weight + .scales + .biases"),
                    "incomplete packed triple",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn mismatch(
        &self,
        path: &Path,
        field: &'static str,
        expected: impl std::fmt::Display,
        actual: impl std::fmt::Display,
    ) -> Error {
        Error::Unsupported(format!(
            "text encoder contract mismatch at {}: field {field} expected {expected}, got {actual}",
            path.display()
        ))
    }
}
