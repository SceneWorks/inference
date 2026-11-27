//! Convert the published MLX-affine MiniMax-H3 tiers into the Candle-native GGML tiers
//! `SceneWorks/minimax-h3-candle` publishes (see `candle_gen_minimax_h3::ggml_tier`). NOT part of
//! CI: it reads ~150 GB of real weights.
//!
//! ```text
//! MINIMAX_H3_MLX_TIER_ROOT=<SceneWorks/minimax-h3-mlx snapshot root, holding q4/ and q8/> \
//! MINIMAX_H3_GGML_OUT=<output root> \
//! MINIMAX_H3_TIERS=q4,q8 MINIMAX_H3_COMPONENTS=transformer,transformer_ref,text_encoder \
//!   cargo run --release --example minimax_h3_ggml_tier -p candle-gen-minimax-h3
//! ```
//!
//! Writes `<out>/<tier>/<component>/`. An existing component directory is a hard error, so a
//! partially converted root resumes by naming only the components still missing.
use std::path::PathBuf;

use candle_gen_minimax_h3::ggml_tier::{convert_component, COMPONENTS};

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn list(key: &str, default: &[&str]) -> Vec<String> {
    env(key).map_or_else(
        || default.iter().map(|s| s.to_string()).collect(),
        |v| v.split(',').map(|s| s.trim().to_string()).collect(),
    )
}

fn main() {
    let src = PathBuf::from(env("MINIMAX_H3_MLX_TIER_ROOT").expect("MINIMAX_H3_MLX_TIER_ROOT"));
    let out = PathBuf::from(env("MINIMAX_H3_GGML_OUT").expect("MINIMAX_H3_GGML_OUT"));
    let tiers = list("MINIMAX_H3_TIERS", &["q4", "q8"]);
    let components = list("MINIMAX_H3_COMPONENTS", &COMPONENTS);
    for tier in &tiers {
        assert!(
            matches!(tier.as_str(), "q4" | "q8"),
            "MINIMAX_H3_TIERS entries must be q4 or q8, got {tier:?}"
        );
        for component in &components {
            assert!(
                COMPONENTS.contains(&component.as_str()),
                "unknown component {component:?}"
            );
            let from = src.join(tier).join(component);
            let to = out.join(tier).join(component);
            let started = std::time::Instant::now();
            println!(
                "[start] {tier}/{component}: {} -> {}",
                from.display(),
                to.display()
            );
            let report = convert_component(&from, &to, &mut |path, r| {
                println!(
                    "  [shard] {} ({} converted, {} passthrough, {} B, {:.1}s)",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    r.converted,
                    r.passthrough,
                    r.tensor_bytes,
                    started.elapsed().as_secs_f64()
                );
                Ok(())
            })
            .unwrap_or_else(|e| panic!("{tier}/{component}: {e}"));
            println!(
                "[done] {tier}/{component}: {} shards, {} converted, {} passthrough, {} B in {:.1}s",
                report.shards,
                report.converted,
                report.passthrough,
                report.tensor_bytes,
                started.elapsed().as_secs_f64()
            );
        }
    }
}
