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
}
