//! Offline pre-quantization of a `Qwen/Qwen-Image-2.1` snapshot into one installable tier
//! (sc-24112).
//!
//! ```text
//! QWEN21_SRC=<dense snapshot> QWEN21_TIER=q4 QWEN21_DST=<out root> \
//!   cargo run --release --example qwen_image_2_1_prequant -p mlx-gen-qwen-image-2-1
//! ```
//!
//! `QWEN21_DST` is the **tiers root**; the tier lands in `<QWEN21_DST>/<tier>/` as a complete
//! standalone snapshot (see `mlx_gen_qwen_image_2_1::convert`). `QWEN21_TIER` is `q8` or `q4`; the
//! `bf16` tier is the dense source itself and is refused rather than copied.
//!
//! The converter itself writes the tier's `CHANGES.md` (the Qwen Research License change record)
//! and `SHA256SUMS` (the manifest a published tier is bound to) — this driver only selects the
//! tier and echoes the manifest it wrote (sc-24114). The conversion is byte-reproducible, so those
//! digests are stable across runs.
//!
//! **Resource note.** The released snapshot is ~31 GB and the converter holds one component's tensor
//! map at a time; run it under an external RSS guard with a stated cap rather than unattended.

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen_qwen_image_2_1::convert::{prequantize_turnkey, CHANGES_FILE, SHA256SUMS_FILE};
use mlx_gen_qwen_image_2_1::quant::Tier;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
}

fn tier_from_env() -> Tier {
    let name = env_or("QWEN21_TIER", "q8");
    Tier::ALL
        .into_iter()
        .find(|tier| tier.dir_name() == name)
        .unwrap_or_else(|| {
            panic!(
                "QWEN21_TIER={name} is not an installable tier; expected one of {:?}",
                Tier::ALL.map(Tier::dir_name)
            )
        })
}

fn main() {
    let src = PathBuf::from(env_or("QWEN21_SRC", ""));
    assert!(
        src.is_dir(),
        "set QWEN21_SRC to the dense Qwen/Qwen-Image-2.1 snapshot directory"
    );
    let tier = tier_from_env();
    let dst_root = PathBuf::from(env_or(
        "QWEN21_DST",
        &format!("{}/SceneWorks/qwen-image-2-1-tiers", env_or("HOME", "/tmp")),
    ));
    let dst = dst_root.join(tier.dir_name());

    eprintln!(
        "[prequant] {} -> {} tier at {}",
        src.display(),
        tier.dir_name(),
        dst.display()
    );
    let started = Instant::now();
    prequantize_turnkey(&src, &dst, tier).expect("prequantize_turnkey");
    eprintln!(
        "[prequant] wrote the {} tier in {:.0}s",
        tier.dir_name(),
        started.elapsed().as_secs_f32()
    );

    // Echo the manifest the converter wrote beside the tier, and the tier's size.
    let manifest_path = dst.join(SHA256SUMS_FILE);
    let manifest = std::fs::read_to_string(&manifest_path).expect("the converter wrote SHA256SUMS");
    let mut total = 0_u64;
    for line in manifest.lines() {
        println!("{line}");
        if let Some((_, rel)) = line.split_once("  ") {
            total += std::fs::metadata(dst.join(rel))
                .map(|m| m.len())
                .unwrap_or(0);
        }
    }
    eprintln!(
        "[prequant] change record at {}, manifest at {}",
        dst.join(CHANGES_FILE).display(),
        manifest_path.display()
    );
    eprintln!(
        "[prequant] {} tier total {:.2} GiB",
        tier.dir_name(),
        total as f64 / (1_u64 << 30) as f64
    );
}
