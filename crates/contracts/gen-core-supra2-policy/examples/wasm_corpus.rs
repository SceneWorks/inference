//! Build with `cargo build -p sceneworks-gen-core-supra2-policy --example wasm_corpus --target wasm32-unknown-unknown`.
//! Instantiate without imports and call the exported `run_corpus`; a failed assertion traps.
#[path = "../tests/support/mod.rs"]
mod support;
use gen_core_supra2_policy::Supra2Resolved;
use serde_json::json;

#[no_mangle]
pub extern "C" fn run_corpus() -> u32 {
    support::run(|request| match Supra2Resolved::resolve(request) {
        Ok(value) => json!({"ok":value}),
        Err(error) => json!({"error":error.to_string()}),
    })
}
