//! Execution metadata for the released SenseNova-Loop-9B checkpoint. Both
//! backends bind weights by physical layer and cache slots by visit order.
use serde_json::{json, Value};

pub const MODEL_REVISION: &str = "3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SenseNovaLoop(());

impl SenseNovaLoop {
    pub const START: usize = 4;
    pub const END: usize = 28;
    pub const STREAMS: usize = 2;
    pub const ITERATIONS: usize = 20;
    pub const VISITS: usize = 56;

    /// Recognize the wrapper before descending into its ordinary Qwen text config.
    pub fn from_config(config: &Value) -> Result<Option<Self>, String> {
        let model = config["model_type"].as_str();
        if model != Some("sensenova_loop")
            && config.get("loop_settings").is_none()
            && config.get("mhc").is_none()
            && !config["architectures"].as_array().is_some_and(|names| {
                names.iter().any(|name| {
                    name.as_str()
                        .is_some_and(|s| s.starts_with("SenseNovaLoop"))
                })
            })
        {
            return Ok(None);
        }
        validate_fields(config, &release_config(), "")?;
        for name in ["loop_settings", "mhc"] {
            let expected_keys = if name == "mhc" { 6 } else { 4 };
            if config[name].as_object().map(|o| o.len()) != Some(expected_keys) {
                return Err(format!("SenseNova-Loop-9B: unsupported {name} fields"));
            }
        }
        let rope = &config["text_config"]["rope_parameters"];
        if rope.as_object().map(|o| o.len()) != Some(7) {
            return Err("SenseNova-Loop-9B: unsupported YaRN options".into());
        }
        Ok(Some(Self(())))
    }

    pub fn visits(self) -> impl Iterator<Item = usize> {
        (0..Self::END).chain(Self::START..32)
    }

    pub fn mhc(self, physical: usize) -> bool {
        (Self::START..Self::END).contains(&physical)
    }

    /// Transformers YaRN uses its default 32/1 rotation thresholds and applies
    /// its attention factor to both FP32 trigonometric tables before the cast.
    pub fn yarn_frequencies(self) -> (Vec<f32>, f32) {
        let dim = 64.;
        let theta = 10_000_000_f64;
        let correction = |rotations: f64| {
            dim * (262_144. / (rotations * std::f64::consts::TAU)).ln() / (2. * theta.ln())
        };
        let low = correction(32.).floor();
        let high = correction(1.).ceil();
        let frequencies = (0..32)
            .map(|i| {
                let extrapolated = 1. / (theta as f32).powf(i as f32 / 32.);
                let ramp = ((i as f64 - low) / (high - low)).clamp(0., 1.) as f32;
                extrapolated * (1. - ramp) + extrapolated * 0.5 * ramp
            })
            .collect();
        (frequencies, 1. + 0.1 * 2_f32.ln())
    }
}

fn validate_fields(actual: &Value, expected: &Value, path: &str) -> Result<(), String> {
    if let Some(fields) = expected.as_object() {
        for (name, value) in fields {
            validate_fields(&actual[name], value, &format!("{path}/{name}"))?;
        }
    } else if actual != expected {
        return Err(format!(
            "SenseNova-Loop-9B requires {path}={expected}, found {actual}"
        ));
    }
    Ok(())
}

fn release_config() -> Value {
    let layer_types: Vec<_> = (1..=32)
        .map(|i| {
            if i % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    json!({
        "model_type": "sensenova_loop", "architectures": ["SenseNovaLoopForConditionalGeneration"],
        "loop_settings": {"counts":[2],"start":4,"end":28,"eval_count":2},
        "mhc":{"start":4,"end":28,"streams":2,"iterations":20,"fixed_identity":false,"pre_scale":1.0},
        "dtype":"bfloat16", "tie_word_embeddings":false,
        "eos_token_id":248046,"pad_token_id":248044,"image_token_id":248056,"video_token_id":248057,
        "vision_start_token_id":248053,"vision_end_token_id":248054,"patch_conv3d":true,
        "text_config":{
            "model_type":"qwen3_5_text","num_hidden_layers":32,"hidden_size":4096,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,"intermediate_size":12288,
            "dtype":"bfloat16","attention_bias":false,"attn_output_gate":true,"hidden_act":"silu","layer_types":layer_types,"full_attention_interval":4,
            "linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,"linear_num_value_heads":32,"linear_value_head_dim":128,"mamba_ssm_dtype":"float32",
            "max_position_embeddings":524288,"partial_rotary_factor":0.25,"rms_norm_eps":1e-6,"tie_word_embeddings":false,"vocab_size":248320,
            "rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000,"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144}
        },
        "vision_config":{
            "model_type":"qwen3_5_vision","depth":27,"hidden_act":"gelu_pytorch_tanh","hidden_size":1152,"in_channels":3,"intermediate_size":4304,"num_heads":16,"num_position_embeddings":2304,"out_hidden_size":4096,
            "patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2,"deepstack_visual_indexes":[]
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_visits_and_yarn() {
        let plan = SenseNovaLoop::from_config(&release_config())
            .unwrap()
            .unwrap();
        let visits: Vec<_> = plan.visits().collect();
        assert_eq!(visits.len(), 56);
        assert_eq!(&visits[4..28], &visits[28..52]);
        assert_eq!(visits.iter().filter(|&&i| i % 4 == 3).count(), 14);
        assert_eq!(visits.iter().filter(|&&i| plan.mhc(i)).count(), 48);
        let (freq, scale) = plan.yarn_frequencies();
        assert_eq!(freq.len(), 32);
        assert_eq!(freq[0], 1.);
        assert!((freq[31] - 0.5 * 10_000_000_f32.powf(-31. / 32.)).abs() < 1e-12);
        assert!((scale - 1.0693147).abs() < 1e-6);
    }

    #[test]
    fn refuses_partial_or_incompatible_release() {
        for path in [
            "/model_type",
            "/loop_settings/eval_count",
            "/mhc/iterations",
            "/text_config/rope_parameters/mrope_section",
            "/vision_config/patch_size",
            "/text_config/layer_types/2",
        ] {
            let mut cfg = release_config();
            *cfg.pointer_mut(path).unwrap() = Value::Null;
            assert!(SenseNovaLoop::from_config(&cfg)
                .unwrap_err()
                .contains(path.split('/').nth(1).unwrap()));
        }
        assert_eq!(
            SenseNovaLoop::from_config(&json!({"model_type":"qwen3_5"})),
            Ok(None)
        );
    }
}
