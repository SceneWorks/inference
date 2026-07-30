//! Produce the physical per-tier Mage-Flow artifacts the SceneWorks mirrors host (sc-14980).
//!
//! Two modes, because Mage's tier layout is split (see [`mlx_gen_mage::convert`]):
//!
//! ```sh
//! # A variant mirror's DiT-only tiers → <out>/{bf16,q8,q4}/transformer/…
//! MAGE_MODE=variant \
//! MAGE_SRC=/path/to/Mage-Flow-Base-snapshot \
//! MAGE_OUT=/tmp/mage-tiers/Mage-Flow-Base \
//!   cargo run --release --example mage_prequant -p mlx-gen-mage
//!
//! # The shared text-encoder + VAE tiers, written ONCE for the whole family
//! MAGE_MODE=components MAGE_SRC=<any variant snapshot> MAGE_OUT=/tmp/mage-tiers/components \
//!   cargo run --release --example mage_prequant -p mlx-gen-mage
//! ```
//!
//! `MAGE_TIERS` (default `bf16,q8,q4`) narrows the run; a tier whose output dir already exists is
//! skipped, so an interrupted run resumes rather than re-quantizing 8 GB. Paths are passed in, never
//! derived — this repository's models arrive as caller-provisioned local paths (the epic-13657
//! boundary enforced by `check-workspace.py`).

use std::path::PathBuf;

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn main() {
    let src = PathBuf::from(env("MAGE_SRC").expect("MAGE_SRC=<dense flat Mage-Flow snapshot dir>"));
    let out =
        PathBuf::from(env("MAGE_OUT").expect("MAGE_OUT=<destination root for the tier tree>"));
    let mode = env("MAGE_MODE").unwrap_or_else(|| "variant".to_owned());
    let components_repo =
        env("MAGE_COMPONENTS_REPO").unwrap_or_else(|| "SceneWorks/Mage-Flow-Components-mlx".into());
    let tiers: Vec<String> = env("MAGE_TIERS")
        .map(|v| v.split(',').map(|t| t.trim().to_owned()).collect())
        .unwrap_or_else(|| {
            mlx_gen_mage::convert::TIERS
                .iter()
                .map(|t| (*t).to_owned())
                .collect()
        });

    assert!(
        src.is_dir(),
        "MAGE_SRC {} is not a directory",
        src.display()
    );

    for tier in &tiers {
        let dst = out.join(tier);
        if dst.exists() {
            println!("[skip] {tier}: {} already exists", dst.display());
            continue;
        }
        // Write to a staging dir and rename on success, so an interrupted run never leaves a
        // half-written tier that the `dst.exists()` resume check would then skip.
        let staging = out.join(format!(".{tier}.partial"));
        if staging.exists() {
            std::fs::remove_dir_all(&staging).expect("clear stale staging dir");
        }
        let started = std::time::Instant::now();
        println!("[start] {mode} tier {tier} → {}", dst.display());
        match mode.as_str() {
            "variant" => {
                mlx_gen_mage::convert::prequantize_variant_tier(
                    &src,
                    &staging,
                    tier,
                    &components_repo,
                )
                .expect("prequantize_variant_tier");
            }
            "components" => {
                mlx_gen_mage::convert::prequantize_shared_components(&src, &staging, tier)
                    .expect("prequantize_shared_components");
            }
            other => panic!("MAGE_MODE must be `variant` or `components`, got {other:?}"),
        }
        std::fs::rename(&staging, &dst).expect("promote staging dir");
        println!(
            "[done]  {tier} in {:.1}s — {:.3} GB",
            started.elapsed().as_secs_f64(),
            dir_bytes(&dst) as f64 / 1e9
        );
    }
}

fn dir_bytes(dir: &std::path::Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            total += dir_bytes(&path);
        } else if let Ok(meta) = std::fs::metadata(&path) {
            total += meta.len();
        }
    }
    total
}
