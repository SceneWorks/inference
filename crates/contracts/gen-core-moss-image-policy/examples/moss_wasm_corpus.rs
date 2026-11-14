//! Build this example for wasm32, instantiate with zero imports under Node, and call `run_corpus`.
#[path = "../tests/support/mod.rs"]
mod support;

#[no_mangle]
pub extern "C" fn run_corpus() -> u32 {
    support::run()
}
