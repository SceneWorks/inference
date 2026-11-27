//! Hardware-free descriptor-only `sga.capabilities/2` exporter for the shipping macOS MLX
//! feature set (`macos-mlx`) — builds and runs on hosted macOS ARM64 with zero Metal GPU
//! execution (`SW-6980`, `REG-3`, `AC1`).

use runtime_macos::gen_core::export::FEATURE_SET_MACOS_MLX;

fn main() {
    let targets = [runtime_catalog::ExportCliTarget {
        feature_set: FEATURE_SET_MACOS_MLX,
        build: runtime_macos::capabilities_export,
    }];
    if let Err(err) = runtime_catalog::run_export_cli("sga-capabilities-export-macos", &targets) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
