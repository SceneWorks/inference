//! Build the Mochi 1 **per-tier artifacts** (`q4/`, `q8/`, `bf16/`) from the upstream snapshot
//! (story A6, sc-11990). `#[ignore]`d — it reads the ~tens-of-GB `$MOCHI_SNAPSHOT` (bf16 DiT shards +
//! fp32 T5-XXL + AsymmVAE + tokenizer) and writes real tier dirs to a scratch out root.
//!
//! Each tier dir carries `transformer/model.safetensors` + `split_model.json` (+ `quantize_config.json`
//! for `q4`/`q8`). The **shared** T5-XXL/VAE/tokenizer are staged **once** as siblings of the tier
//! dirs (not duplicated per tier), and [`mlx_gen_mochi::load`] resolves them from the parent.
//!
//! Run (writes to a scratch dir OUTSIDE the repo):
//!   MOCHI_SNAPSHOT=/path/to/models--genmo--mochi-1-preview/snapshots/<rev> \
//!   MOCHI_TIERS_OUT=~/mochi-tiers \
//!   cargo test -p mlx-gen-mochi --test build_tiers -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen_mochi::{convert_and_assemble, stage_shared_components, MochiConvertOpts};

fn snapshot_dir() -> PathBuf {
    std::env::var("MOCHI_SNAPSHOT")
        .expect("set MOCHI_SNAPSHOT to the mochi-1-preview snapshot dir")
        .into()
}

fn out_root() -> PathBuf {
    std::env::var("MOCHI_TIERS_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join("mochi-tiers"))
}

/// Sum of file sizes under `dir` (follows symlinks — so a shared dir reports its resolved bytes).
fn dir_size(dir: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            match std::fs::metadata(&p) {
                Ok(m) if m.is_dir() => total += dir_size(&p),
                Ok(m) => total += m.len(),
                Err(_) => {}
            }
        }
    }
    total
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[test]
#[ignore = "builds the q4/q8/bf16 Mochi tiers from ~tens-of-GB $MOCHI_SNAPSHOT"]
fn build_all_tiers() {
    let src = snapshot_dir();
    let root = out_root();
    assert!(src.is_dir(), "snapshot missing: {}", src.display());
    std::fs::create_dir_all(&root).unwrap();

    // Stage the shared T5-XXL / VAE / tokenizer once (idempotent; convert_and_assemble also does this).
    let shared = stage_shared_components(&src, &root)
        .unwrap_or_else(|e| panic!("stage_shared_components failed: {e}"));
    for p in &shared {
        eprintln!("SHARED {}  ({:.2} GiB)", p.display(), gib(dir_size(p)));
    }

    for opts in [
        MochiConvertOpts::quant(4),
        MochiConvertOpts::quant(8),
        MochiConvertOpts::default(), // bf16
    ] {
        let tier = opts.tier_name();
        let out = root.join(&tier);
        let produced = convert_and_assemble(&src, &out, &opts)
            .unwrap_or_else(|e| panic!("convert {tier} failed: {e}"));
        let manifest = std::fs::read_to_string(produced.join("split_model.json")).unwrap();
        eprintln!(
            "BUILT tier {} at {}  transformer={:.2} GiB\n{}",
            tier,
            produced.display(),
            gib(dir_size(&produced.join("transformer"))),
            manifest
        );
    }
}
