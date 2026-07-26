//! sc-14980 — the physical per-tier artifacts load, and are numerically the SAME model that
//! load-time quantization produced.
//!
//! These run against real weights and an authorized Metal device, so they are `#[ignore]`d and
//! env-gated like every other real-weights test in this crate:
//!
//! ```sh
//! MAGE_SNAPSHOT=<dense flat microsoft/Mage-Flow-Base snapshot> \
//! MAGE_TIER_ROOT=<the generated variant tier tree, holding bf16/ q8/ q4/> \
//! MAGE_COMPONENTS_ROOT=<the generated shared components tree, holding bf16/ q8/ q4/> \
//!   cargo test --locked -p mlx-gen-mage --test prequantize_real_weights -- --ignored --nocapture
//! ```
//!
//! `MAGE_TIER_ROOT` / `MAGE_COMPONENTS_ROOT` are produced by `examples/mage_prequant.rs`.

use std::path::{Path, PathBuf};

use mlx_gen_mage::convert::{prequantize_shared_components, prequantize_variant_tier};
use mlx_gen_mage::{MageComponentDirs, MageFlowPipeline};

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
}

fn snapshot() -> PathBuf {
    env_dir("MAGE_SNAPSHOT").expect("MAGE_SNAPSHOT=<dense flat Mage-Flow snapshot dir>")
}

/// The split component dirs for one tier: DiT from the variant tree, TE + VAE from the shared tree.
fn tier_dirs(tier_root: &Path, components_root: &Path, tier: &str) -> MageComponentDirs {
    MageComponentDirs {
        transformer: tier_root.join(tier).join("transformer"),
        text_encoder: components_root.join(tier).join("text_encoder"),
        vae: components_root.join(tier).join("vae"),
    }
}

/// Generate the tier trees into a temp dir when the caller did not point at pre-built ones, so the
/// test is self-contained from a dense snapshot alone.
fn ensure_trees() -> (PathBuf, PathBuf) {
    if let (Some(t), Some(c)) = (env_dir("MAGE_TIER_ROOT"), env_dir("MAGE_COMPONENTS_ROOT")) {
        return (t, c);
    }
    let src = snapshot();
    let base = std::env::temp_dir().join("mage-prequant-test");
    let (tiers, components) = (base.join("variant"), base.join("components"));
    // Both packed tiers, so the tier-mismatch test has a q8 artifact to point at even when the
    // caller did not pre-build a full tree. `bf16` is skipped here: it is a byte-exact copy of the
    // dense snapshot, and the tests that want it tolerate its absence.
    for tier in ["q8", "q4"] {
        if !tiers.join(tier).exists() {
            prequantize_variant_tier(
                &src,
                &tiers.join(tier),
                tier,
                "SceneWorks/Mage-Flow-Components-mlx",
            )
            .expect("prequantize_variant_tier");
        }
        if !components.join(tier).exists() {
            prequantize_shared_components(&src, &components.join(tier), tier)
                .expect("prequantize_shared_components");
        }
    }
    (tiers, components)
}

/// The core sc-14980 guarantee: a **pre-quantized tier artifact is the same model** as load-time
/// quantization over the dense snapshot.
///
/// `quantize_map` packs with the identical `mlx_rs::ops::quantize(bf16, group 64)` call that
/// `AdaptableLinear::quantize` runs, so the two paths must agree to the bit, not merely
/// "approximately". Comparing a full generation rather than raw tensors makes the check
/// discriminating end-to-end: a mis-ordered key, a wrongly-excluded projection, or a `pos_embed`
/// that got packed all move this number far off zero while still loading cleanly.
#[test]
#[ignore = "needs full real weights and an authorized Metal device"]
fn a_packed_tier_renders_identically_to_load_time_quantization() {
    let src = snapshot();
    let (tiers, components) = ensure_trees();

    let packed = MageFlowPipeline::load_components(
        &tier_dirs(&tiers, &components, "q4"),
        Some(4),
        mlx_gen_mage::VaePart::Decode,
    )
    .expect("load packed q4 tier");
    let packed_image = render(&packed);
    drop(packed);

    let load_time = MageFlowPipeline::load_with_quant(&src, Some(4)).expect("load-time q4");
    let load_time_image = render(&load_time);
    drop(load_time);

    assert_eq!(
        packed_image.len(),
        load_time_image.len(),
        "both paths decode the same geometry"
    );

    // ---- the probe must DISCRIMINATE ------------------------------------------------------
    // `max_abs == 0.0` is also what two empty or two constant buffers produce, so establish that
    // this comparison is capable of failing before trusting that it passed.
    assert_eq!(
        packed_image.len(),
        3 * 512 * 512,
        "a 512x512 RGB decode, not an empty buffer"
    );
    let (lo, hi) = packed_image
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    assert!(
        hi - lo >= 64.0,
        "the render must carry real dynamic range (got {lo}..{hi}); a flat image would make the \
         identity comparison vacuous"
    );
    // A DIFFERENT tier through the same comparison must be far from zero. This is the mutation
    // check: it proves the metric responds to a real weight change, so the 0.0 below is a fact
    // about the artifacts and not about the harness.
    let q8 = MageFlowPipeline::load_with_quant(&src, Some(8)).expect("load-time q8");
    let q8_image = render(&q8);
    drop(q8);
    let cross_tier = packed_image
        .iter()
        .zip(&q8_image)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        cross_tier > 0.0,
        "a q8 render must differ from a q4 render — otherwise this comparison cannot detect any \
         weight difference and the identity assertion below proves nothing"
    );

    let max_abs = packed_image
        .iter()
        .zip(&load_time_image)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "packed-vs-load-time max_abs = {max_abs}; cross-tier (q4 vs q8) max_abs = {cross_tier}; \
         range = {lo}..{hi}"
    );
    assert!(
        max_abs == 0.0,
        "a pre-quantized q4 tier must be BIT-IDENTICAL to load-time q4 quantization \
         (quantize_map runs the same op); got max_abs {max_abs}"
    );
}

/// The packed tier must arrive already packed — the whole point is skipping the dense read. If
/// auto-detection silently failed, the load-time path would re-quantize and the previous test would
/// still pass, so assert the packed state directly.
#[test]
#[ignore = "needs full real weights and an authorized Metal device"]
fn a_packed_tier_is_packed_on_arrival_and_needs_no_load_time_quantization() {
    let (tiers, components) = ensure_trees();
    // `quant_bits: None` = do NOT run load-time quantize. If the artifact were dense, the counts
    // below would be 0.
    let dirs = tier_dirs(&tiers, &components, "q4");
    let pipeline = MageFlowPipeline::load_components(&dirs, None, mlx_gen_mage::VaePart::Decode);
    // A dense (bf16) request against a packed artifact is a hard error, not a silent Q4 render.
    let err = pipeline
        .err()
        .expect("a bf16 request against a packed q4 artifact must fail");
    let message = format!("{err}");
    assert!(
        message.contains("pre-quantized Q4") && message.contains("bf16"),
        "the mismatch must name both tiers; got {message}"
    );

    let pipeline = MageFlowPipeline::load_components(&dirs, Some(4), mlx_gen_mage::VaePart::Decode)
        .expect("load packed q4");
    assert_eq!(
        pipeline.transformer.quantized_linear_count(),
        174,
        "all 174 DiT Linears arrive packed from disk"
    );
    assert_eq!(
        pipeline.text_encoder.quantized_linear_count(),
        253,
        "all 253 text-encoder projections arrive packed from disk"
    );
}

/// Serving a Q4 request from a Q8 artifact (or vice versa) must fail loudly. `quantize` is a no-op
/// over packed weights, so without this guard the request would quietly render at the wrong tier —
/// exactly the class of bug per-tier artifacts are supposed to eliminate.
#[test]
#[ignore = "needs full real weights and an authorized Metal device"]
fn a_tier_mismatch_is_a_hard_error_not_a_silent_downgrade() {
    let (tiers, components) = ensure_trees();
    let Ok(q8) = std::fs::metadata(tiers.join("q8")) else {
        eprintln!("skipping: no q8 tier built (set MAGE_TIER_ROOT to a full tree)");
        return;
    };
    assert!(q8.is_dir());
    let err = MageFlowPipeline::load_components(
        &tier_dirs(&tiers, &components, "q8"),
        Some(4),
        mlx_gen_mage::VaePart::Decode,
    )
    .err()
    .expect("a Q4 request against a Q8 artifact must fail");
    let message = format!("{err}");
    assert!(
        message.contains("Q8") && message.contains("Q4"),
        "the mismatch must name both tiers; got {message}"
    );
}

/// The bf16 tier is a byte-exact passthrough of the dense DiT, so the checkpoint-identity
/// fingerprint — and therefore the whole variant-routing guarantee — is preserved through the
/// re-host. `add_k_proj.bias` is a bias, so it also survives the q4/q8 packs untouched.
#[test]
#[ignore = "needs full real weights and an authorized Metal device"]
fn the_identity_fingerprint_tensor_is_byte_identical_across_every_tier() {
    let src = snapshot();
    let (tiers, _) = ensure_trees();
    let dense = read_tensor_bytes(
        &src.join("transformer")
            .join("diffusion_pytorch_model.safetensors"),
        "transformer_blocks.0.attn.add_k_proj.bias",
    );
    for tier in ["bf16", "q8", "q4"] {
        let path = tiers
            .join(tier)
            .join("transformer")
            .join("diffusion_pytorch_model.safetensors");
        if !path.exists() {
            continue;
        }
        let got = read_tensor_bytes(&path, "transformer_blocks.0.attn.add_k_proj.bias");
        assert_eq!(
            got, dense,
            "{tier}: the identity fingerprint tensor must survive the re-host byte-for-byte, or \
             checkpoint-identity verification cannot gate a tier install"
        );
    }
}

fn render(pipeline: &MageFlowPipeline) -> Vec<f32> {
    let key = mlx_gen_mage::resolve_gs_key(None).expect("gs key");
    let image = pipeline
        .generate(
            "a red cube on a white table",
            " ",
            512,
            512,
            4,
            1.0,
            42,
            &key,
            false,
        )
        .expect("generate");
    let flat = image
        .flatten(None, None)
        .expect("flatten")
        .as_dtype(mlx_rs::Dtype::Float32)
        .expect("as f32");
    flat.as_slice::<f32>().to_vec()
}

/// Read one tensor's raw bytes out of a safetensors file without materializing the rest.
fn read_tensor_bytes(path: &Path, name: &str) -> Vec<u8> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut len = [0u8; 8];
    file.read_exact(&mut len).expect("header length");
    let header_len = u64::from_le_bytes(len);
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header).expect("header");
    let meta: serde_json::Value = serde_json::from_slice(&header).expect("header json");
    let offsets = meta[name]["data_offsets"]
        .as_array()
        .unwrap_or_else(|| panic!("{name} missing from {}", path.display()));
    let (start, end) = (
        offsets[0].as_u64().expect("start"),
        offsets[1].as_u64().expect("end"),
    );
    file.seek(SeekFrom::Start(8 + header_len + start))
        .expect("seek");
    let mut bytes = vec![0u8; (end - start) as usize];
    file.read_exact(&mut bytes).expect("tensor bytes");
    bytes
}
