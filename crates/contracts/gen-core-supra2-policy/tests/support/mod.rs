use gen_core_supra2_policy::Supra2PolicyRequest;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Case {
    name: String,
    request: Supra2PolicyRequest,
    expected: Value,
}

/// The same versioned JSON corpus runs in native contract tests and the standalone WASM module.
pub fn run(resolve: impl Fn(&Supra2PolicyRequest) -> Value) -> u32 {
    let cases: Vec<Case> = serde_json::from_str(include_str!("../corpus.json")).unwrap();
    for case in &cases {
        assert_eq!(resolve(&case.request), case.expected, "{}", case.name);
    }
    cases.len().try_into().unwrap()
}
