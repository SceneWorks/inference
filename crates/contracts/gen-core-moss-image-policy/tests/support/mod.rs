use gen_core_moss_image_policy::{
    apply_edit_defaults, MossImageOptions, MossImageResolved, MossImageVariant, MossPolicyRequest,
    MossSkipLayerGuidance,
};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Case {
    name: String,
    operation: String,
    #[serde(default)]
    variant: Option<MossImageVariant>,
    #[serde(default)]
    input: Value,
    expected: Value,
}

fn input_request(value: Value) -> Result<MossPolicyRequest, String> {
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn scalar(value: &Value) -> f32 {
    match value.as_str() {
        Some("NaN") => f32::NAN,
        Some("Infinity") => f32::INFINITY,
        Some("-Infinity") => f32::NEG_INFINITY,
        _ => value.as_f64().unwrap() as f32,
    }
}

fn outcome(case: &Case) -> Value {
    let result: Result<Value, String> = match case.operation.as_str() {
        "resolve" => input_request(case.input.clone()).and_then(|request| {
            MossImageResolved::resolve(case.variant.expect("resolve variant"), &request)
                .map_err(|error| error.to_string())
                .and_then(|value| serde_json::to_value(value).map_err(|error| error.to_string()))
        }),
        "edit" => input_request(case.input.clone()).and_then(|request| {
            apply_edit_defaults(&request)
                .map_err(|error| error.to_string())
                .map(|value| {
                    json!({"width":value.width,"height":value.height,"count":value.count,
                        "steps":value.steps,"guidance":value.guidance,
                        "scheduler_shift":value.scheduler_shift,"moss_image":value.moss_image})
                })
        }),
        "deserialize_slg" => serde_json::from_value::<MossSkipLayerGuidance>(case.input.clone())
            .map_err(|error| error.to_string())
            .and_then(|value| serde_json::to_value(value).map_err(|error| error.to_string())),
        "deserialize_options" => serde_json::from_value::<MossImageOptions>(case.input.clone())
            .map_err(|error| error.to_string())
            .and_then(|value| serde_json::to_value(value).map_err(|error| error.to_string())),
        "serialize_slg" => {
            let input = &case.input;
            let value = MossSkipLayerGuidance {
                layers: input["layers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect(),
                scale: scalar(&input["scale"]),
                start: scalar(&input["start"]),
                stop: scalar(&input["stop"]),
            };
            serde_json::to_value(value).map_err(|error| error.to_string())
        }
        other => panic!("unknown corpus operation: {other}"),
    };
    match result {
        Ok(value) => json!({"ok":value}),
        Err(error) => json!({"error":error}),
    }
}

/// One checked JSON corpus runs as native tests and inside a zero-import wasm32 module.
pub fn run() -> u32 {
    let cases: Vec<Case> = serde_json::from_str(include_str!("../corpus.json")).unwrap();
    for case in &cases {
        let actual = outcome(case);
        if let Some(fragment) = case.expected.get("error_contains").and_then(Value::as_str) {
            assert!(
                actual["error"]
                    .as_str()
                    .is_some_and(|error| error.contains(fragment)),
                "{}: {actual}",
                case.name
            );
        } else {
            assert_eq!(actual, case.expected, "{}", case.name);
        }
    }
    cases.len().try_into().unwrap()
}
