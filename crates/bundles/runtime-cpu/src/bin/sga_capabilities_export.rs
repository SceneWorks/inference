//! Hardware-free descriptor-only `sga.capabilities/2` exporter for Candle shipping feature sets
//! (`candle-cpu` and `candle-cuda`) — builds and runs on hosted Linux/macOS/Windows with zero
//! GPU and zero CUDA toolkit (`SW-6980`, `REG-3`, `AC1`).

use runtime_cpu::gen_core::export::{FEATURE_SET_CANDLE_CPU, FEATURE_SET_CANDLE_CUDA};

fn main() {
    let targets = [
        runtime_catalog::ExportCliTarget {
            feature_set: FEATURE_SET_CANDLE_CPU,
            build: runtime_cpu::capabilities_export,
        },
        runtime_catalog::ExportCliTarget {
            feature_set: FEATURE_SET_CANDLE_CUDA,
            build: runtime_cpu::cuda_descriptor_capabilities_export,
        },
    ];
    if let Err(err) = runtime_catalog::run_export_cli("sga-capabilities-export-candle", &targets) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
