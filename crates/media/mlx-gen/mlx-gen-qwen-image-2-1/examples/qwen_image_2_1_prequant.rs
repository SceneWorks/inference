//! Offline pre-quantization of a `Qwen/Qwen-Image-2.1` snapshot into one installable tier, plus the
//! SHA-256 manifest that binds the result (sc-24112).
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
//! Every file's SHA-256 is printed on completion. The conversion is byte-reproducible, so those
//! digests are the manifest a published tier is bound to.
//!
//! **Resource note.** The released snapshot is ~31 GB and the converter holds one component's tensor
//! map at a time; run it under an external RSS guard with a stated cap rather than unattended.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_gen_qwen_image_2_1::convert::prequantize_turnkey;
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

/// Every regular file under `root`, relative path → SHA-256, sorted.
fn digest_tree(root: &Path) -> BTreeMap<String, String> {
    use sha2::{Digest, Sha256};
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out);
            } else if let Ok(bytes) = std::fs::read(&path) {
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                out.insert(rel, format!("{:x}", Sha256::digest(&bytes)));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
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

    let mut total = 0_u64;
    for (rel, digest) in digest_tree(&dst) {
        total += std::fs::metadata(dst.join(&rel))
            .map(|m| m.len())
            .unwrap_or(0);
        println!("{digest}  {rel}");
    }
    eprintln!(
        "[prequant] {} tier total {:.2} GiB",
        tier.dir_name(),
        total as f64 / (1_u64 << 30) as f64
    );
}
