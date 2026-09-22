//! Installable Q8/Q4 tiers on the committed miniature snapshot (sc-24112): the converter's output
//! layout, its byte-level reproducibility, its equivalence to the canonical `mlx_rs::ops::quantize`,
//! the packed reload path through the production catalog, and the descriptor's truthfulness about
//! which tier a snapshot actually is.
//!
//! Runs by default — the tiny snapshot is ~1 MB and a converted tier is smaller still.
//!
//! **Why the tiny snapshot only packs part of itself.** Its DiT is 32 wide and its Qwen3 tower is
//! 32/64 wide, and a shippable tier may declare exactly one `quantization.group_size` (64), so every
//! `Linear` narrower than that stays dense — see `mlx_gen_qwen_image_2_1::quant`. The released
//! geometry has no such width (`convert::released_geometries_are_fully_group_aligned` proves that
//! from the frozen config numbers), so in production the same converter packs everything. That makes
//! the parity bars here **conservative**: they bound the error of a partially-packed tier, and the
//! fully-packed released tier is the terminal story's to measure on real weights.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Quant, WeightsSource,
};
use mlx_gen_qwen_image_2_1::convert::{is_transformer_target, prequantize_turnkey};
use mlx_gen_qwen_image_2_1::quant::{installed_tier, Tier, GROUP_SIZE};
use sha2::{Digest, Sha256};

use crate::common::tiny_snapshot;

const ID: &str = "qwen_image_2_1";
/// Wide enough that a 32-px decode tile is a real 2×2 tiling rather than a single pass.
const EDGE: u32 = 64;
/// Bounded-decode geometry for the miniature VAE.
///
/// The decoder upsamples 16x through four stages, so its receptive field at *output* resolution
/// spans several latent cells; a tile only a couple of latent cells wide cannot reproduce the
/// untiled decode however it is blended, which is a property of any such autoencoder and not of
/// this tier work. A 128-px tile over a 256-px render is a real 2x2 tiling at 8 latent cells per
/// tile — the smallest geometry on this fixture that is representative of the shipped 512/64
/// production default.
const DECODE_EDGE: u32 = 256;
const DECODE_TILE: u32 = 128;
const DECODE_TILE_OVERLAP: u32 = 64;

// ── helpers ──────────────────────────────────────────────────────────────────────────────────────

/// The committed packed components of `tier` — `transformer/` and `text_encoder/` only; the rest of
/// a tier is the dense tiny snapshot, copied through unchanged.
fn committed_tier_components(tier: Tier) -> PathBuf {
    crate::common::fixtures()
        .join("tiers")
        .join(tier.dir_name())
}

fn convert_into(dir: &Path, tier: Tier) -> PathBuf {
    let out = dir.join(tier.dir_name());
    prequantize_turnkey(&tiny_snapshot(), &out, tier).expect("the tiny snapshot converts");
    out
}

/// Every regular file under `root`, keyed by its path relative to `root`, valued by SHA-256.
fn digest_tree(root: &Path) -> BTreeMap<String, String> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
        for entry in std::fs::read_dir(dir).expect("readable directory") {
            let path = entry.expect("readable entry").path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .expect("under root")
                    .to_string_lossy()
                    .replace('\\', "/");
                let bytes = std::fs::read(&path).expect("readable file");
                out.insert(rel, format!("{:x}", Sha256::digest(&bytes)));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

fn marker(dir: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(dir.join("config.json")).expect("annotated config.json");
    serde_json::from_str::<serde_json::Value>(&text).expect("valid json")["quantization"].clone()
}

fn request(width: u32, height: u32) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in the forest".to_owned(),
        width,
        height,
        steps: Some(3),
        seed: Some(42),
        ..Default::default()
    }
}

fn render(spec: &LoadSpec, req: &GenerationRequest) -> Image {
    let generator = mlx_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(ID, spec)
        .expect("the snapshot loads through the catalog path");
    match generator.generate(req, &mut |_| {}).expect("render") {
        GenerationOutput::Images(mut images) => images.remove(0),
        other => panic!("images expected, got {other:?}"),
    }
}

/// `(max |Δ|, mean |Δ|)` over two RGB8 buffers, in 0..=255 units.
fn pixel_errors(got: &Image, want: &Image) -> (f32, f32) {
    assert_eq!(got.pixels.len(), want.pixels.len(), "buffer length");
    let mut max = 0f32;
    let mut sum = 0f32;
    for (a, b) in got.pixels.iter().zip(&want.pixels) {
        let d = (*a as f32 - *b as f32).abs();
        max = max.max(d);
        sum += d;
    }
    (max, sum / got.pixels.len().max(1) as f32)
}

fn report(name: &str, got: &Image, want: &Image, max_bound: f32, mean_bound: f32) {
    let (max, mean) = pixel_errors(got, want);
    eprintln!(
        "{name}: max|Δ|={max:.2}/255 mean|Δ|={mean:.4}/255 (bounds {max_bound}/{mean_bound})"
    );
    assert!(max <= max_bound, "{name}: max|Δ|={max} exceeds {max_bound}");
    assert!(
        mean <= mean_bound,
        "{name}: mean|Δ|={mean} exceeds {mean_bound}"
    );
}

// ── layout ───────────────────────────────────────────────────────────────────────────────────────

/// The converted tier is a **complete standalone snapshot**: packed `transformer/` and
/// `text_encoder/` with their `quantization` markers, everything else copied through dense, and the
/// `model_index.json` + licence assets the Qwen Research License requires a derived bundle to carry.
#[test]
fn a_converted_tier_is_a_complete_standalone_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    for tier in [Tier::Q8, Tier::Q4] {
        let out = convert_into(tmp.path(), tier);

        for rel in [
            "model_index.json",
            "transformer/config.json",
            "transformer/model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "processor/tokenizer.json",
            "scheduler/scheduler_config.json",
        ] {
            assert!(out.join(rel).is_file(), "{} missing {rel}", tier.dir_name());
        }
        // The sharded source is replaced by one file per packed component; no stale index rides along.
        assert!(!out
            .join("transformer/diffusion_pytorch_model.safetensors.index.json")
            .exists());

        let dit_bits = tier.transformer_bits().unwrap();
        assert_eq!(
            marker(&out.join("transformer")),
            serde_json::json!({ "bits": dit_bits, "group_size": GROUP_SIZE })
        );
        // The tower reads 8 at BOTH packed tiers — the declared Q4 floor, so the marker labels what
        // is actually on disk rather than the tier's name.
        assert_eq!(
            marker(&out.join("text_encoder")),
            serde_json::json!({ "bits": 8, "group_size": GROUP_SIZE })
        );
        // The dense components are copied verbatim, byte for byte.
        for rel in [
            "vae/diffusion_pytorch_model.safetensors",
            "processor/tokenizer.json",
        ] {
            assert_eq!(
                std::fs::read(out.join(rel)).unwrap(),
                std::fs::read(tiny_snapshot().join(rel)).unwrap(),
                "{rel} must be copied unchanged"
            );
        }

        // And the snapshot self-reports the tier it is.
        assert_eq!(installed_tier(&out).unwrap(), tier);
    }
    // The two tiers are distinguishable on disk (the Q4 DiT is the smaller artefact).
    let q8 = std::fs::metadata(tmp.path().join("q8/transformer/model.safetensors"))
        .unwrap()
        .len();
    let q4 = std::fs::metadata(tmp.path().join("q4/transformer/model.safetensors"))
        .unwrap()
        .len();
    assert!(q4 < q8, "the Q4 DiT must be smaller than the Q8 DiT");
    // The towers are the same artefact at both tiers — the floor, visible in bytes.
    assert_eq!(
        std::fs::read(tmp.path().join("q8/text_encoder/model.safetensors")).unwrap(),
        std::fs::read(tmp.path().join("q4/text_encoder/model.safetensors")).unwrap(),
        "both packed tiers hold the tower at Q8, so its artefact is identical"
    );
}

// ── reproducibility ──────────────────────────────────────────────────────────────────────────────

/// **Hash pin.** Converting the same source twice produces a byte-identical tier — every file, not
/// just the weights — which is what lets a published tier be bound to a SHA-256 manifest.
///
/// The digests are printed rather than committed as hex constants: a committed digest of a
/// GPU-computed artefact would be a machine-dependent golden. The machine-independent property is
/// pinned instead, here (run-to-run identity) and in
/// [`packed_codes_equal_the_canonical_quantize`] (identity to the canonical op). Capture the printed
/// digests when publishing a real tier.
#[test]
fn conversion_is_byte_reproducible_across_runs() {
    let tmp = tempfile::tempdir().unwrap();
    for tier in [Tier::Q8, Tier::Q4] {
        let a = tmp.path().join(format!("a-{}", tier.dir_name()));
        let b = tmp.path().join(format!("b-{}", tier.dir_name()));
        prequantize_turnkey(&tiny_snapshot(), &a, tier).unwrap();
        prequantize_turnkey(&tiny_snapshot(), &b, tier).unwrap();

        let (da, db) = (digest_tree(&a), digest_tree(&b));
        assert_eq!(
            da.keys().collect::<Vec<_>>(),
            db.keys().collect::<Vec<_>>(),
            "{}: the two conversions must produce the same file set",
            tier.dir_name()
        );
        for (rel, digest) in &da {
            eprintln!("tier {} {rel} sha256={digest}", tier.dir_name());
            assert_eq!(
                db.get(rel),
                Some(digest),
                "{}: {rel} is not byte-reproducible",
                tier.dir_name()
            );
        }
    }
}

/// **The reproducibility pin.** The committed `tests/fixtures/tiers/{q8,q4}/` packed components —
/// the artefacts the *Candle* backend's tier tests read across the crate boundary — are
/// byte-identical to what the converter produces right now.
///
/// This is where a reproducibility break surfaces: if `mlx_rs::ops::quantize` or the safetensors
/// writer ever produced different bytes for the same input, this reds on the macOS/Metal lane
/// instead of a published tier silently ceasing to match its manifest. Regeneration instructions
/// are in `tests/fixtures/tiers/README.md`; read it before refreshing anything.
#[test]
fn the_committed_tier_fixtures_match_a_fresh_conversion() {
    let tmp = tempfile::tempdir().unwrap();
    for tier in [Tier::Q8, Tier::Q4] {
        let fresh = convert_into(tmp.path(), tier);
        let committed = committed_tier_components(tier);
        for component in ["transformer", "text_encoder"] {
            for file in ["config.json", "model.safetensors"] {
                let want = committed.join(component).join(file);
                let got = fresh.join(component).join(file);
                assert_eq!(
                    std::fs::read(&got).unwrap(),
                    std::fs::read(&want).unwrap(),
                    "{}/{component}/{file} is not byte-reproducible against the committed fixture \
                     (see tests/fixtures/tiers/README.md)",
                    tier.dir_name()
                );
            }
        }
    }
}

/// The packed triple on disk is byte-identical to `mlx_rs::ops::quantize` over the bf16 source —
/// the same op `QwenImage21Transformer::quantize` runs at load. This is the machine-independent
/// form of the hash pin: the artefact is not merely stable, it is *the* canonical output.
#[test]
fn packed_codes_equal_the_canonical_quantize() {
    use mlx_gen::weights::Weights;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::Dtype;

    let tmp = tempfile::tempdir().unwrap();
    for tier in [Tier::Q8, Tier::Q4] {
        let out = convert_into(tmp.path(), tier);
        let bits = tier.transformer_bits().unwrap();
        let dense = Weights::from_dir(tiny_snapshot().join("transformer")).unwrap();
        let packed = Weights::from_dir(out.join("transformer")).unwrap();

        let mut checked = 0usize;
        let mut left_dense = 0usize;
        for key in dense.keys().map(str::to_owned).collect::<Vec<_>>() {
            let Some(base) = key.strip_suffix(".weight") else {
                continue;
            };
            let source = dense.get(&key).unwrap();
            let eligible = is_transformer_target(base)
                && source.shape().len() == 2
                && source.shape()[1] % GROUP_SIZE == 0
                && source.shape()[1] >= GROUP_SIZE;
            if !eligible {
                assert!(
                    packed.get(&format!("{base}.scales")).is_none(),
                    "{base} must stay dense at group {GROUP_SIZE}"
                );
                left_dense += 1;
                continue;
            }
            let (wq, scales, biases) =
                quantize(source.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, bits).unwrap();
            for (suffix, want) in [("weight", wq), ("scales", scales), ("biases", biases)] {
                let got = packed
                    .get(&format!("{base}.{suffix}"))
                    .unwrap_or_else(|| panic!("{base}.{suffix} present in the packed tier"));
                assert_eq!(got.shape(), want.shape(), "{base}.{suffix} shape");
                assert_eq!(got.dtype(), want.dtype(), "{base}.{suffix} dtype");
                assert!(
                    eq(got, &want).unwrap().all(None).unwrap().item::<bool>(),
                    "{base}.{suffix} is not the canonical quantize output"
                );
            }
            checked += 1;
        }
        eprintln!(
            "tier {}: {checked} packed / {left_dense} left dense (miniature widths below group {GROUP_SIZE})",
            tier.dir_name()
        );
        assert!(
            checked > 0,
            "{}: the fixture must exercise at least one packed Linear",
            tier.dir_name()
        );
    }
}

// ── reload + parity ──────────────────────────────────────────────────────────────────────────────

/// A converted tier loads through the production catalog path with its matching `quantize` selector
/// and renders. The quantization error against the dense render is reported and bounded, and the Q4
/// tier is never *closer* to bf16 than the Q8 tier is.
#[test]
fn packed_tiers_reload_and_render_within_the_declared_bars() {
    let tmp = tempfile::tempdir().unwrap();
    let req = request(EDGE, EDGE);
    let dense = render(&LoadSpec::new(WeightsSource::Dir(tiny_snapshot())), &req);

    let mut mean_by_tier = Vec::new();
    for tier in [Tier::Q8, Tier::Q4] {
        let out = convert_into(tmp.path(), tier);
        let spec =
            LoadSpec::new(WeightsSource::Dir(out)).with_quant(tier.selected_quant().unwrap());
        let packed = render(&spec, &req);
        assert_eq!((packed.width, packed.height), (EDGE, EDGE));
        // Bars for a PARTIALLY packed miniature tier (see the module docs). Generous on max — a
        // single 8-bit pixel step is 1/255 and quantization noise concentrates on edges — and tight
        // on the mean, which is what a systematic break would move.
        report(
            &format!("{} vs bf16", tier.dir_name()),
            &packed,
            &dense,
            64.0,
            8.0,
        );
        mean_by_tier.push(pixel_errors(&packed, &dense).1);
    }
    assert!(
        mean_by_tier[1] >= mean_by_tier[0],
        "Q4 ({:.4}) must not be closer to bf16 than Q8 ({:.4})",
        mean_by_tier[1],
        mean_by_tier[0]
    );
}

/// The descriptor tells the truth about what is installable, and a request that disagrees with the
/// tier on disk is refused rather than silently served at the installed tier.
#[test]
fn the_installed_tier_is_honoured_exactly() {
    let caps = mlx_gen_qwen_image_2_1::descriptor().capabilities;
    assert_eq!(caps.supported_quants, &[Quant::Q4, Quant::Q8]);
    assert_eq!(
        caps.component_precision_floors,
        mlx_gen_qwen_image_2_1::COMPONENT_PRECISION_FLOORS,
        "the Q4 text-encoder floor is descriptor-visible"
    );

    let tmp = tempfile::tempdir().unwrap();
    let q8 = convert_into(tmp.path(), Tier::Q8);
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let at = |quant: Option<Quant>| {
        let mut spec = LoadSpec::new(WeightsSource::Dir(q8.clone()));
        if let Some(quant) = quant {
            spec = spec.with_quant(quant);
        }
        registry.load(ID, &spec)
    };

    assert!(at(Some(Quant::Q8)).is_ok(), "the matching tier loads");

    let err = match at(Some(Quant::Q4)) {
        Ok(_) => panic!("a Q4 request against a Q8 tier must be refused"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("silently serve q8"), "{err}");
    assert!(err.contains("Point at the q4 snapshot"), "{err}");

    let err = match at(None) {
        Ok(_) => panic!("a dense request against a packed tier must be refused"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("no quantization was requested"), "{err}");

    // A dense snapshot still quantizes at load — the historical behaviour is unchanged.
    assert!(registry
        .load(
            ID,
            &LoadSpec::new(WeightsSource::Dir(tiny_snapshot())).with_quant(Quant::Q8)
        )
        .is_ok());
}

// ── memory strategies ────────────────────────────────────────────────────────────────────────────

/// Staged residency is an execution decision, not a numeric one: dropping the tower between
/// conditioning and denoise must reproduce the resident render **exactly**, on a packed tier as
/// well as a dense one.
#[test]
fn staged_residency_preserves_the_resident_output_at_every_tier() {
    let tmp = tempfile::tempdir().unwrap();
    let req = request(EDGE, EDGE);
    for (label, root, quant) in [
        ("bf16", tiny_snapshot(), None),
        ("q8", convert_into(tmp.path(), Tier::Q8), Some(Quant::Q8)),
        ("q4", convert_into(tmp.path(), Tier::Q4), Some(Quant::Q4)),
    ] {
        let build = |policy: OffloadPolicy| {
            let mut spec =
                LoadSpec::new(WeightsSource::Dir(root.clone())).with_offload_policy(policy);
            if let Some(quant) = quant {
                spec = spec.with_quant(quant);
            }
            render(&spec, &req)
        };
        assert_eq!(
            build(OffloadPolicy::Sequential).pixels,
            build(OffloadPolicy::Resident).pixels,
            "{label}: staged residency must be byte-identical to the resident render"
        );
    }
}

/// Bounded (tiled) decode is a memory lever, not a quality one: the head-once/tail-tiled decode
/// must reproduce the untiled decode to within seam-blend noise, at every tier.
#[test]
fn bounded_decode_preserves_the_untiled_output_at_every_tier() {
    use mlx_gen::gen_core::GenerationMemory;

    let tmp = tempfile::tempdir().unwrap();
    for (label, root, quant) in [
        ("bf16", tiny_snapshot(), None),
        ("q8", convert_into(tmp.path(), Tier::Q8), Some(Quant::Q8)),
        ("q4", convert_into(tmp.path(), Tier::Q4), Some(Quant::Q4)),
    ] {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root));
        if let Some(quant) = quant {
            spec = spec.with_quant(quant);
        }
        let resident = render(&spec, &request(DECODE_EDGE, DECODE_EDGE));

        let mut tiled_req = request(DECODE_EDGE, DECODE_EDGE);
        tiled_req.memory = Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(DECODE_TILE),
            decode_overlap: Some(DECODE_TILE_OVERLAP),
            ..Default::default()
        });
        let tiled = render(&spec, &tiled_req);
        assert_eq!((tiled.width, tiled.height), (DECODE_EDGE, DECODE_EDGE));
        // Seam blending is a weighted average of two exact decodes, so the deviation is small but
        // not zero — unlike staged residency, which is exact.
        report(
            &format!("{label} tiled vs untiled"),
            &tiled,
            &resident,
            24.0,
            2.0,
        );
    }
}
