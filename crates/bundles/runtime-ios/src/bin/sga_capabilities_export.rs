//! Hardware-free descriptor-only `sga.capabilities/2` exporter for the shipping iOS feature sets
//! (`ios-media` with default features, `ios-llm-only` with `--no-default-features`) — builds and
//! runs on hosted macOS ARM64 with zero Metal GPU execution (`SW-6980`, `REG-3`, `AC1`).

fn main() {
    let targets = [runtime_catalog::ExportCliTarget {
        feature_set: runtime_ios::EXPORT_FEATURE_SET,
        build: runtime_ios::capabilities_export,
    }];
    if let Err(err) = runtime_catalog::run_export_cli("sga-capabilities-export-ios", &targets) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
