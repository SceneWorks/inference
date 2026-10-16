//! Tensor-neutral Qwen3-VL RoPE metadata normalization shared by both native backends.
//!
//! Transformers' `convert_rope_params_to_dict` chooses a nonempty legacy `rope_scaling`
//! over `rope_parameters`, then takes theta/partial-factor from that selected block before
//! legacy root values. Normalize that spelling, not model precision or parameter values.

use serde_json::Value;

/// Normalize an already text-config-descended Qwen3-VL config to the legacy fields both
/// backend parsers consume. Callers deliberately leave other architectures (including
/// Gemma 4's per-layer RoPE maps) on their existing parsers.
pub fn normalize_qwen3_vl_rope(config: &Value) -> Result<Value, String> {
    let legacy = config.get("rope_scaling").filter(|value| {
        !value.is_null() && !value.as_object().is_some_and(|object| object.is_empty())
    });
    let (selected, label) = if let Some(value) = legacy {
        (Some(value), "rope_scaling")
    } else {
        (
            config
                .get("rope_parameters")
                .filter(|value| !value.is_null()),
            "rope_parameters",
        )
    };
    let mut normalized = config.clone();
    let had_parameters = selected.is_some();
    let mut parameters = match selected {
        Some(value) => value
            .as_object()
            .cloned()
            .ok_or_else(|| format!("Qwen3-VL {label} must be a flat RoPE parameter object"))?,
        None => serde_json::Map::new(),
    };
    let kind = parameters
        .get("rope_type")
        .or_else(|| parameters.get("type"));
    let kind = match kind {
        None => "default",
        Some(value) => value
            .as_str()
            .ok_or_else(|| format!("{label}.rope_type must be a string"))?,
    };
    if !matches!(kind, "default" | "llama3" | "yarn") {
        return Err(format!("unsupported Qwen3-VL {label}.rope_type {kind:?}"));
    }
    let kind = kind.to_owned();
    parameters.remove("type");
    parameters.insert("rope_type".to_owned(), Value::String(kind));
    for key in ["rope_theta", "partial_rotary_factor"] {
        if let Some(value) = parameters.get(key).or_else(|| config.get(key)) {
            let number = value
                .as_f64()
                .ok_or_else(|| format!("{label}.{key} must be a number"))?;
            if !number.is_finite()
                || number <= 0.0
                || !(number as f32).is_finite()
                || (key == "partial_rotary_factor" && number > 1.0)
            {
                return Err(format!(
                    "{label}.{key} must be finite and positive{}",
                    if key == "partial_rotary_factor" {
                        " and at most one"
                    } else {
                        ""
                    }
                ));
            }
            normalized[key] = value.clone();
        }
    }
    for key in [
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
    ] {
        if let Some(value) = parameters.get(key) {
            let number = value
                .as_f64()
                .ok_or_else(|| format!("{label}.{key} must be a number"))?;
            let zero_allowed = matches!(key, "mscale" | "mscale_all_dim");
            if !number.is_finite()
                || !(number as f32).is_finite()
                || number < 0.0
                || (!zero_allowed && number == 0.0)
            {
                return Err(format!(
                    "{label}.{key} must be finite and {}",
                    if zero_allowed {
                        "nonnegative"
                    } else {
                        "positive"
                    }
                ));
            }
        }
    }
    if let Some(value) = parameters.get("mrope_interleaved") {
        if !value.is_boolean() {
            return Err(format!("{label}.mrope_interleaved must be a boolean"));
        }
    }
    if let Some(value) = parameters.get("mrope_section") {
        let sections = value
            .as_array()
            .filter(|array| array.len() == 3)
            .ok_or_else(|| {
                format!("{label}.mrope_section must contain three nonnegative integers")
            })?;
        let mut sum = 0i64;
        for section in sections {
            let section = section
                .as_i64()
                .filter(|value| (0..=i32::MAX as i64).contains(value))
                .ok_or_else(|| {
                    format!("{label}.mrope_section must contain three nonnegative integers")
                })?;
            sum += section;
        }
        let head_dim = config.get("head_dim").and_then(Value::as_i64).or_else(|| {
            let hidden = config.get("hidden_size")?.as_i64()?;
            let heads = config.get("num_attention_heads")?.as_i64()?;
            (heads > 0).then(|| hidden / heads)
        });
        if let Some(head_dim) = head_dim.filter(|value| (2..=i32::MAX as i64).contains(value)) {
            let partial = normalized
                .get("partial_rotary_factor")
                .and_then(Value::as_f64)
                .unwrap_or(1.0);
            let rotary_dim = ((head_dim as f32 * partial as f32).round() as i32) & !1;
            if sum != i64::from(rotary_dim / 2) {
                return Err(format!(
                    "{label}.mrope_section sums to {sum}, expected half the rotary dimension {}",
                    rotary_dim / 2
                ));
            }
        }
    }
    if had_parameters {
        normalized["rope_scaling"] = Value::Object(parameters);
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn modern() -> Value {
        json!({"head_dim":16,"rope_parameters":{"rope_theta":5000000.0,"rope_type":"default","mrope_section":[4,2,2],"mrope_interleaved":true}})
    }

    #[test]
    fn modern_qwen_parameters_preserve_nondefault_theta_sections_and_input() {
        let original = modern();
        let normalized = normalize_qwen3_vl_rope(&original).unwrap();
        assert_eq!(normalized["rope_theta"], 5000000.0);
        assert_eq!(
            normalized["rope_scaling"]["mrope_section"],
            json!([4, 2, 2])
        );
        assert_eq!(normalized["rope_scaling"]["rope_type"], "default");
        assert_eq!(
            original,
            modern(),
            "normalization must not rewrite the supplied snapshot"
        );
    }

    #[test]
    fn hf_precedence_selected_block_then_root_and_nonempty_legacy_first() {
        let mut config = modern();
        config["rope_theta"] = json!(123456.0);
        assert_eq!(
            normalize_qwen3_vl_rope(&config).unwrap()["rope_theta"],
            5000000.0
        );
        config["rope_scaling"] =
            json!({"rope_theta":7654321.0,"rope_type":"default","mrope_section":[2,3,3]});
        let normalized = normalize_qwen3_vl_rope(&config).unwrap();
        assert_eq!(normalized["rope_theta"], 7654321.0);
        assert_eq!(
            normalized["rope_scaling"]["mrope_section"],
            json!([2, 3, 3])
        );
        config["rope_scaling"] = json!({});
        assert_eq!(
            normalize_qwen3_vl_rope(&config).unwrap()["rope_theta"],
            5000000.0
        );
        config.as_object_mut().unwrap().remove("rope_parameters");
        assert_eq!(
            normalize_qwen3_vl_rope(&config).unwrap()["rope_theta"],
            123456.0
        );
    }

    #[test]
    fn legacy_layout_and_scaling_types_are_preserved() {
        let legacy = json!({"head_dim":16,"rope_theta":5000000.0,"rope_scaling":{"type":"default","mrope_section":[4,2,2]}});
        let normalized = normalize_qwen3_vl_rope(&legacy).unwrap();
        assert_eq!(normalized["rope_theta"], 5000000.0);
        assert_eq!(normalized["rope_scaling"]["rope_type"], "default");
        for kind in ["llama3", "yarn"] {
            let mut config = modern();
            config["rope_parameters"]["rope_type"] = json!(kind);
            config["rope_parameters"]["factor"] = json!(2.0);
            let normalized = normalize_qwen3_vl_rope(&config).unwrap();
            assert_eq!(normalized["rope_scaling"]["rope_type"], kind);
            assert_eq!(normalized["rope_scaling"]["factor"], 2.0);
        }
    }

    #[test]
    fn explicit_partial_rotary_factor_preserves_the_resolved_section_width() {
        let mut config = modern();
        config["rope_parameters"]["partial_rotary_factor"] = json!(0.5);
        config["rope_parameters"]["mrope_section"] = json!([2, 1, 1]);
        let normalized = normalize_qwen3_vl_rope(&config).unwrap();
        assert_eq!(normalized["partial_rotary_factor"], 0.5);
        assert_eq!(
            normalized["rope_scaling"]["mrope_section"],
            json!([2, 1, 1])
        );
        assert!(normalize_qwen3_vl_rope(&json!({"rope_theta":"invalid"}))
            .unwrap_err()
            .contains("rope_theta"));
    }

    #[test]
    fn malformed_explicit_parameters_never_silently_fall_back() {
        for (key, value) in [
            ("rope_theta", json!("5000000")),
            ("rope_theta", json!(0)),
            ("rope_theta", Value::Null),
            ("rope_type", json!(7)),
            ("rope_type", json!("unsupported")),
            ("mrope_section", json!([4, 2])),
            ("mrope_section", json!([4, -1, 5])),
            ("mrope_section", json!([4, 2, 3])),
            ("mrope_interleaved", json!("true")),
            ("partial_rotary_factor", json!(2.0)),
        ] {
            let mut config = modern();
            config["rope_parameters"][key] = value;
            assert!(
                normalize_qwen3_vl_rope(&config).unwrap_err().contains(key),
                "{key}"
            );
        }
        let mut config = modern();
        config["rope_parameters"] = json!("invalid");
        assert!(normalize_qwen3_vl_rope(&config)
            .unwrap_err()
            .contains("rope_parameters"));
    }
}
