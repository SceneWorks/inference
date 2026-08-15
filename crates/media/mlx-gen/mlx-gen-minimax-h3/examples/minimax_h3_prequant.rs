//! Produce the physical per-tier MiniMax-H3 artifacts `SceneWorks/minimax-h3-mlx` hosts — the DiT
//! (sc-17150) and the text encoder (sc-19120).
//!
//! Two components are tiered; both VAEs and the tokenizer are dense in every tier and shared as
//! co-requisites. This driver takes **one** component directory and writes one tier of it.
//!
//! ```sh
//! # DiT (`transformer/` or `transformer_ref/`)
//! MINIMAX_H3_SRC=<snapshot>/transformer \
//! MINIMAX_H3_OUT=<scratch>/minimax-h3-tiers/q4/transformer \
//! MINIMAX_H3_TIER=q4 \
//!   cargo run --release --example minimax_h3_prequant -p mlx-gen-minimax-h3
//!
//! # Text encoder — the stage that sets the process high-water
//! MINIMAX_H3_SRC=<snapshot>/text_encoder \
//! MINIMAX_H3_OUT=<scratch>/minimax-h3-tiers/q4/text_encoder \
//! MINIMAX_H3_TIER=q4 \
//! MINIMAX_H3_COMPONENT=text_encoder \
//!   cargo run --release --example minimax_h3_prequant -p mlx-gen-minimax-h3
//! ```
//!
//! `MINIMAX_H3_COMPONENT` selects which converter runs (`transformer`, the default, or
//! `text_encoder`). `MINIMAX_H3_ADALN_BITS` overrides the AdaLN projection width independently of
//! the tier — the [`AdaLnTierPolicy`](mlx_gen_minimax_h3::convert::AdaLnTierPolicy) knob, DiT only.
//! Default is the tier's own width; see that type for the measurement behind the default.
//!
//! Paths are passed in, never derived (the epic-13657 boundary `check-workspace.py` enforces). An
//! output directory that already exists is a hard error rather than a silent overwrite: a tier is a
//! published artifact and a half-merged one is worse than no tier at all.
//!
//! # Memory
//!
//! [`quantize_minimax_h3_dit`](mlx_gen_minimax_h3::convert::quantize_minimax_h3_dit) streams one
//! source shard at a time, so this holds ~7.5 GB rather than the ~101 GB a whole-map convert of a
//! 66.28 GB component would. The `on_shard` hook below reports each shard's finished size as it
//! lands, which is also the running total the manifest (sc-17158) needs.

use std::path::PathBuf;

use mlx_gen_minimax_h3::convert::{
    quantize_minimax_h3_dit, quantize_minimax_h3_text_encoder, tier_bits, AdaLnTierPolicy,
};
use mlx_gen_minimax_h3::model::{DIT_COMPONENT, TEXT_ENCODER_COMPONENT};

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn main() {
    let src = PathBuf::from(
        env("MINIMAX_H3_SRC").expect("MINIMAX_H3_SRC=<snapshot>/transformer (a DiT component dir)"),
    );
    let dst = PathBuf::from(
        env("MINIMAX_H3_OUT").expect("MINIMAX_H3_OUT=<destination dir for this tier's DiT>"),
    );
    let tier = env("MINIMAX_H3_TIER").expect("MINIMAX_H3_TIER=q4|q8");
    let component = env("MINIMAX_H3_COMPONENT").unwrap_or_else(|| DIT_COMPONENT.to_string());
    assert!(
        component == DIT_COMPONENT || component == TEXT_ENCODER_COMPONENT,
        "MINIMAX_H3_COMPONENT must be {DIT_COMPONENT:?} or {TEXT_ENCODER_COMPONENT:?}, got \
         {component:?}"
    );

    assert!(
        src.is_dir(),
        "MINIMAX_H3_SRC {} is not a directory",
        src.display()
    );
    assert!(
        !dst.exists(),
        "MINIMAX_H3_OUT {} already exists — refusing to overwrite a published tier",
        dst.display()
    );

    let bits = tier_bits(&tier)
        .expect("tier name")
        .unwrap_or_else(|| panic!("{tier} is the dense passthrough; nothing to quantize"));
    let adaln = match env("MINIMAX_H3_ADALN_BITS") {
        Some(v) => AdaLnTierPolicy {
            bits: v.parse().expect("MINIMAX_H3_ADALN_BITS must be 4 or 8"),
        },
        None => AdaLnTierPolicy::for_tier(bits),
    };

    println!(
        "[start] {component} tier {tier} (weights {bits}-bit, adaln {}-bit) {} -> {}",
        adaln.bits,
        src.display(),
        dst.display()
    );
    let started = std::time::Instant::now();
    let mut total: u64 = 0;
    let mut on_shard = |path: &std::path::Path| -> mlx_gen::Result<()> {
        let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        total += bytes;
        println!(
            "  [shard] {} {bytes} B (running total {total} B, {:.1}s)",
            path.file_name().unwrap_or_default().to_string_lossy(),
            started.elapsed().as_secs_f64()
        );
        Ok(())
    };

    let written = if component == TEXT_ENCODER_COMPONENT {
        quantize_minimax_h3_text_encoder(&src, &dst, bits, &mut on_shard)
            .expect("quantize_minimax_h3_text_encoder")
    } else {
        quantize_minimax_h3_dit(&src, &dst, bits, adaln, &mut on_shard)
            .expect("quantize_minimax_h3_dit")
    };

    // The hosted size the manifest records is the WHOLE directory — shards plus `config.json` plus
    // the index — not just the tensor bytes, because that is what a user downloads.
    let mut hosted: u64 = 0;
    for entry in std::fs::read_dir(&dst).expect("read tier dir") {
        let entry = entry.expect("dir entry");
        if entry.path().is_file() {
            hosted += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    println!(
        "[done] {component} tier {tier}: {} shards, {hosted} B hosted, {:.1}s",
        written.len(),
        started.elapsed().as_secs_f64()
    );
}
