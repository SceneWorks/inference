//! Bakes the quantized-kernel architecture ladder into the binary, for the runtime diagnostic in
//! `src/cuda_arch.rs` (sc-19545).
//!
//! # Why a build script and not a constant
//!
//! The ladder is decided in exactly one place — `vendor/candle-kernels/build.rs`, the file that
//! passes the `-gencode` flags to nvcc. A constant restated over here would keep claiming coverage
//! after those flags changed, which is the drift the diagnostic exists to report on. But the
//! shipped worker cannot read that file at run time (it is a source file, not an artifact), so the
//! ladder has to cross the compile boundary. That is what a build script is for.
//!
//! # This must never fail a build
//!
//! It runs on every build of this crate, including CPU and Metal ones that have nothing to do with
//! CUDA. Every failure path here degrades to an EMPTY ladder, and an empty ladder makes the runtime
//! check stay silent rather than warn. A diagnostic that could break compilation — or cry wolf
//! because it could not find a file — would be worse than the black render it is meant to explain.

use std::path::PathBuf;

/// `vendor/candle-kernels/build.rs`, relative to this crate. Same repository, so the relative path
/// holds for the git-pinned consumers (SceneWorks) as well as for a local checkout.
fn vendored_build_rs() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vendor/candle-kernels/build.rs")
}

/// `(native SASS rungs, PTX floors)` from the `.arg("-gencode=…")` calls the file actually makes.
///
/// Anchored on the `.arg("…")` call rather than the bare `-gencode` text: the same file quotes the
/// ladder in a prose comment, and a parser that matched prose would keep reporting full coverage
/// after the real flags were deleted.
fn parse_gencodes(source: &str) -> (Vec<u32>, Vec<u32>) {
    let (mut sass, mut ptx) = (Vec::new(), Vec::new());
    for emitted in source.split(".arg(\"").skip(1) {
        let Some(flag) = emitted.split('"').next() else {
            continue;
        };
        let Some(code) = flag
            .strip_prefix("-gencode=")
            .and_then(|rest| rest.split("code=").nth(1))
        else {
            continue;
        };
        if let Some(cap) = code.strip_prefix("sm_").and_then(|c| c.parse().ok()) {
            sass.push(cap);
        } else if let Some(cap) = code.strip_prefix("compute_").and_then(|c| c.parse().ok()) {
            ptx.push(cap);
        }
    }
    (sass, ptx)
}

fn csv(values: &[u32]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    let vendored = vendored_build_rs();
    println!("cargo::rerun-if-changed={}", vendored.display());
    // The cudaforge baseline rung comes from the environment at BUILD time, so a rebuild under a
    // different cap has to re-run this script or the baked ladder would describe the previous build.
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");

    let (sass, ptx) = match std::fs::read_to_string(&vendored) {
        Ok(source) => parse_gencodes(&source),
        // Vendor directory moved or absent. Emit nothing and let the runtime check stay quiet.
        Err(_) => (Vec::new(), Vec::new()),
    };

    // `CUDA_COMPUTE_CAP` contributes the ladder's bottom rung via cudaforge's single `-gencode`.
    // Absent on CPU/Metal builds, where none of this is consulted anyway.
    let baseline = std::env::var("CUDA_COMPUTE_CAP")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok());

    println!("cargo::rustc-env=CANDLE_GEN_FATBIN_SASS={}", csv(&sass));
    println!("cargo::rustc-env=CANDLE_GEN_FATBIN_PTX={}", csv(&ptx));
    println!(
        "cargo::rustc-env=CANDLE_GEN_FATBIN_BASELINE={}",
        baseline.map(|c| c.to_string()).unwrap_or_default()
    );
}
