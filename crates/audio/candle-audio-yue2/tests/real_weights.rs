//! Real-weight checks of the pinned YuE2 closure (sc-22989).
//!
//! Gated like the other real-weight tests in this workspace: `#[ignore]`d in ordinary runs (CI has
//! no weights); under `--ignored` a missing `YUE2_HF_HUB` panics rather than silently passing.
//! `YUE2_HF_HUB` is a Hugging Face hub directory (the one holding `models--m-a-p--*/`) that
//! already contains the five pinned revisions — see `scripts/reference/yue2/README.md`. Nothing is
//! downloaded.
//!
//! ```text
//! YUE2_HF_HUB=/path/to/huggingface/hub cargo test --release -p candle-audio-yue2 \
//!   --test real_weights -- --ignored --nocapture --test-threads 1
//! ```
//!
//! CPU only. Peak RSS is the mmapped weights file resident plus one tensor copy (YuE2-3B: ~7.3 GB
//! file, 0.76 GB largest tensor).

use std::path::PathBuf;
use std::time::Instant;

use candle_audio_yue2::inventory::{self, ComponentId, FileRole};
use candle_audio_yue2::snapshot::{resolve_closure, resolve_component, verify_native_tensors};
use candle_audio_yue2::{Closure, SnapshotDirs, VaeVariant};

/// The provisioned snapshot directories, the way the application hands them over: this test maps
/// each pinned repository to its pinned-revision snapshot inside the operator-supplied hub
/// directory. (Production code never derives a cache layout; this is test-side provisioning.)
fn hub() -> SnapshotDirs {
    let hub = PathBuf::from(std::env::var_os("YUE2_HF_HUB").unwrap_or_else(|| {
        panic!(
            "real-weight test run without YUE2_HF_HUB (a hub directory holding the pinned repos)"
        )
    }));
    inventory::REPOS
        .iter()
        .fold(SnapshotDirs::new(), |dirs, repo| {
            let dir = hub
                .join(format!("models--{}", repo.id.replace('/', "--")))
                .join("snapshots")
                .join(repo.revision);
            dirs.with(repo.id, dir)
        })
}

/// Every closure resolves offline and verifies through the production entry point, and the
/// upstream's own `weights_manifest.json` in each snapshot agrees with the pinned inventory.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB (see the module docs)"]
fn every_pinned_closure_verifies_from_the_local_hub_cache() {
    let root = hub();
    for closure in [
        Closure::Generation {
            vae: VaeVariant::Standard,
        },
        Closure::Generation {
            vae: VaeVariant::Legacy,
        },
        Closure::Cover,
    ] {
        let start = Instant::now();
        let verified = resolve_closure(closure, &root).unwrap_or_else(|e| panic!("{e}"));
        let bytes: u64 = verified
            .components()
            .iter()
            .flat_map(|c| c.component().files.iter().map(|f| f.bytes))
            .sum();
        println!(
            "{closure:?}: {} components, {} files, {:.2} GB hashed in {:.1?}",
            verified.components().len(),
            verified
                .components()
                .iter()
                .map(|c| c.files().len())
                .sum::<usize>(),
            bytes as f64 / 1e9,
            start.elapsed()
        );
        for c in verified.components() {
            let Some(manifest_path) = c.path("weights_manifest.json") else {
                println!(
                    "  {}: upstream publishes no weights_manifest.json",
                    c.component().key
                );
                continue;
            };
            let upstream: serde_json::Value =
                serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
            let weights = c.component().weights().unwrap();
            // Two upstream shapes: YuE2's `{"files": {name: {bytes, sha256}}}` and MERT2's flat
            // `{filename, bytes, sha256}`.
            let (bytes, sha) = match upstream.get("files") {
                Some(files) => (
                    files[weights.path]["bytes"].as_u64(),
                    files[weights.path]["sha256"].as_str(),
                ),
                None => {
                    assert_eq!(upstream["filename"].as_str(), Some(weights.path));
                    (upstream["bytes"].as_u64(), upstream["sha256"].as_str())
                }
            };
            assert_eq!(bytes, Some(weights.bytes), "{}", c.component().key);
            assert_eq!(sha, Some(weights.sha256), "{}", c.component().key);
            println!(
                "  {}: upstream weights_manifest.json agrees",
                c.component().key
            );
        }
    }
}

/// Every tensor of every pinned weights file, loaded through Candle's safetensors reader,
/// reproduces the name, dtype, shape and value digest PyTorch recorded from the pinned original.
#[test]
#[ignore = "real weights: set YUE2_HF_HUB (see the module docs)"]
fn native_tensors_reproduce_the_pinned_originals() {
    let root = hub();
    for id in ComponentId::ALL {
        let component = id.component();
        if !component.files.iter().any(|f| f.role == FileRole::Weights) {
            println!(
                "{}: no weights (tokenizer data; hash-verified)",
                component.key
            );
            continue;
        }
        let verified = resolve_component(id, &root).unwrap_or_else(|e| panic!("{e}"));
        let start = Instant::now();
        let n = verify_native_tensors(&verified).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(n, verified.manifest().tensors.len());
        println!(
            "{}: {n} tensors reproduce the pinned originals ({:.1?})",
            component.key,
            start.elapsed()
        );
    }
    // The tokenizer's pinned file is the one the manifest describes.
    let tiktoken = resolve_component(ComponentId::QwenTiktoken, &root).unwrap();
    let path = tiktoken.path("qwen.tiktoken").unwrap();
    let ranks = std::fs::read(path)
        .unwrap()
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .count() as u64;
    assert_eq!(Some(ranks), tiktoken.manifest().ordinary_tokens);
    assert_eq!(inventory::QWEN_TIKTOKEN.repo, inventory::YUE2_3B_REPO);
}
