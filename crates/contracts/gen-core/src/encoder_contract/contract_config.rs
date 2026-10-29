//! `EncoderContract` config validation: the Hugging Face `config.json` checked field by field,
//! across every alias a released checkpoint uses, against the advertised contract.

use super::*;

impl EncoderContract {
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
}
