//! Metadata-only comparison of the native public API against the portable golden corpus.
#[path = "../../gen-core-supra2-policy/tests/support/mod.rs"]
mod support;
use gen_core::{Conditioning, Error, GenerationRequest, Image};
use gen_core_checkpoint::supra2::Supra2Resolved;
use gen_core_supra2_policy::Supra2PolicyRequest;
use serde_json::json;

fn native(request: &Supra2PolicyRequest) -> GenerationRequest {
    GenerationRequest {
        width: request.width,
        height: request.height,
        count: request.count,
        steps: request.steps,
        guidance: request.guidance,
        seed: request.seed,
        negative_prompt: request.negative_prompt.clone(),
        prompt_batch: if request.prompt_batch_present {
            vec!["prompt".into()]
        } else {
            vec![]
        },
        conditioning: if request.conditioning_present {
            vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }]
        } else {
            vec![]
        },
        ..Default::default()
    }
}
#[test]
fn native_adapter_preserves_exact_values_and_refusals() {
    assert_eq!(
        support::run(|request| match Supra2Resolved::resolve(&native(request)) {
            Ok(value) =>
                json!({"ok":{"steps":value.steps,"guidance":value.guidance,"use_cfg":value.use_cfg,"seed":value.seed,"count":value.count}}),
            Err(Error::Unsupported(detail)) => json!({"error":detail}),
            Err(error) => panic!("changed native error variant: {error}"),
        }),
        17
    );
}
#[test]
fn native_adapter_preserves_nonfinite_refusal() {
    for guidance in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let request = Supra2PolicyRequest {
            guidance: Some(guidance),
            ..Default::default()
        };
        assert!(
            matches!(Supra2Resolved::resolve(&native(&request)), Err(Error::Unsupported(detail)) if detail == "supra2: guidance must be finite")
        );
    }
}
