use gen_core_pag_policy::{resolve_pag, PagCapability, PagConfig};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Case {
    name: String,
    capability: Option<PagCapability>,
    config: Option<PagConfig>,
    total_steps: u32,
    expected: Value,
}

fn evaluate(case: &Case) -> Value {
    match &case.capability {
        Some(cap) => match cap.resolve(case.config.as_ref()) {
            Ok(resolved) => {
                let step_scales: Vec<f32> = (0..case.total_steps as usize)
                    .map(|step| resolved.step_scale(step, case.total_steps as usize))
                    .collect();
                json!({
                    "ok": {
                        "active": resolved.is_active(),
                        "resolved": resolved,
                        "active_steps": resolved.active_steps(case.total_steps),
                        "step_scales": step_scales,
                    }
                })
            }
            Err(err) => json!({
                "error_kind": err.kind(),
                "error": err.to_string(),
            }),
        },
        None => match resolve_pag(None, "unsupported_model", case.config.as_ref()) {
            Ok(None) => json!({"ok_none": true}),
            Ok(Some(resolved)) => json!({"ok": resolved}),
            Err(err) => json!({
                "error_kind": err.kind(),
                "error": err.to_string(),
            }),
        },
    }
}

pub fn run() -> u32 {
    let cases: Vec<Case> = serde_json::from_str(include_str!("../corpus.json")).unwrap();
    for case in &cases {
        let actual = evaluate(case);
        if let Some(fragment) = case.expected.get("error_contains").and_then(Value::as_str) {
            assert_eq!(
                actual["error_kind"], case.expected["error_kind"],
                "{}: {actual}",
                case.name
            );
            assert!(
                actual["error"]
                    .as_str()
                    .is_some_and(|e| e.contains(fragment)),
                "{}: {actual}",
                case.name
            );
        } else {
            assert_eq!(actual, case.expected, "{}", case.name);
        }
    }
    cases.len().try_into().unwrap()
}
