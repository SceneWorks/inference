//! `EncoderContract` header validation: dense/packed tensor headers, bias surfaces and matrix
//! shapes checked against the advertised architecture.

use super::*;

impl EncoderContract {
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
