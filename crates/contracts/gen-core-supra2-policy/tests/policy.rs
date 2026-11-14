mod support;
use gen_core_supra2_policy::{Supra2PolicyRequest, Supra2Resolved};
use serde_json::json;

#[test]
fn published_policy_corpus() {
    assert_eq!(
        support::run(|request| match Supra2Resolved::resolve(request) {
            Ok(value) => json!({"ok":value}),
            Err(error) => json!({"error":error.to_string()}),
        }),
        17
    );
}
#[test]
fn nonfinite_guidance_is_refused() {
    for guidance in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let request = Supra2PolicyRequest {
            guidance: Some(guidance),
            ..Default::default()
        };
        assert_eq!(
            Supra2Resolved::resolve(&request).unwrap_err().to_string(),
            "supra2: guidance must be finite"
        );
    }
}
#[test]
fn serialized_projection_is_strict_and_roundtrips() {
    let request = Supra2PolicyRequest::default();
    assert_eq!(
        serde_json::from_value::<Supra2PolicyRequest>(serde_json::to_value(&request).unwrap())
            .unwrap(),
        request
    );
    for value in [
        json!({"steps":"50"}),
        json!({"unknown":true}),
        json!({"conditioning_present":1}),
    ] {
        assert!(serde_json::from_value::<Supra2PolicyRequest>(value).is_err());
    }
}
