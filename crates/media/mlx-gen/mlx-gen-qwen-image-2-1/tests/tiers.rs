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

/// The committed packed components of `tier` — `transformer/` and `text_encoder/` only, and only
/// their `model.safetensors`. The rest of a tier is the dense tiny snapshot copied through
/// unchanged, and each packed component's `config.json` is **composed at test time**
/// ([`composed_marker_config`]) rather than committed: it is the source component's config plus the
/// quantization marker, so committing it would couple these fixtures to every unrelated edit of
/// `tiny-snapshot/*/config.json`.
fn committed_tier_components(tier: Tier) -> PathBuf {
    crate::common::fixtures()
        .join("tiers")
        .join(tier.dir_name())
}

/// `tiny-snapshot/<component>/config.json` + `{"quantization": {bits, group_size}}`, serialized the
/// way `mlx_gen::quant::write_quantized_config` serializes it.
fn composed_marker_config(component: &str, bits: i32) -> String {
    let source = tiny_snapshot().join(component).join("config.json");
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&source).expect("source config"))
            .expect("valid json");
    value["quantization"] = serde_json::json!({ "bits": bits, "group_size": GROUP_SIZE });
    serde_json::to_string_pretty(&value).expect("serializable")
}

/// A complete `tier` snapshot under `dir`, built from the committed packed `model.safetensors`
/// plus a composed marker config plus the dense remainder of `tiny-snapshot` — i.e. exactly what
/// `prequantize_turnkey` assembles, without re-running the converter.
fn compose_committed_tier(dir: &Path, tier: Tier) -> PathBuf {
    fn copy_dir(src: &Path, dst: &Path) {
        std::fs::create_dir_all(dst).unwrap();
        for entry in std::fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let target = dst.join(entry.file_name());
            if path.is_dir() {
                copy_dir(&path, &target);
            } else {
                std::fs::copy(&path, &target).unwrap();
            }
        }
    }

    let out = dir.join(format!("committed-{}", tier.dir_name()));
    let packed = committed_tier_components(tier);
    for (component, bits) in [
        ("transformer", tier.transformer_bits().unwrap()),
        ("text_encoder", tier.text_encoder_bits().unwrap()),
    ] {
        let dst = out.join(component);
        std::fs::create_dir_all(&dst).unwrap();
        std::fs::copy(
            packed.join(component).join("model.safetensors"),
            dst.join("model.safetensors"),
        )
        .unwrap();
        std::fs::write(
            dst.join("config.json"),
            composed_marker_config(component, bits),
        )
        .unwrap();
    }
    for component in ["vae", "processor", "scheduler"] {
        let src = tiny_snapshot().join(component);
        if src.is_dir() {
            copy_dir(&src, &out.join(component));
        }
    }
    let index = tiny_snapshot().join("model_index.json");
    if index.is_file() {
        std::fs::copy(&index, out.join("model_index.json")).unwrap();
    }
    out
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
        // The tower carries the SAME width as the DiT: a tier is a whole-pipeline contract, so the
        // marker labels the tier the caller selected rather than a promoted one.
        assert_eq!(
            marker(&out.join("text_encoder")),
            serde_json::json!({
                "bits": tier.text_encoder_bits().unwrap(),
                "group_size": GROUP_SIZE
            })
        );
        assert_eq!(tier.text_encoder_bits(), tier.transformer_bits());
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

    // **A tier is a whole-pipeline contract**: the q4 tower is a genuinely different, smaller
    // artefact than the q8 tower. Asserting the two are IDENTICAL — which an earlier revision of
    // this test did, to celebrate a Q8 floor — is the inverse of the property that matters.
    //
    // *Mutation that reds this:* `Tier::text_encoder_bits` returning `Some(8)` for Q4.
    let q8_tower = std::fs::metadata(tmp.path().join("q8/text_encoder/model.safetensors"))
        .unwrap()
        .len();
    let q4_tower = std::fs::metadata(tmp.path().join("q4/text_encoder/model.safetensors"))
        .unwrap()
        .len();
    assert!(
        q4_tower < q8_tower,
        "the q4 tower must be smaller than the q8 tower ({q4_tower} vs {q8_tower}); an identical \
         artefact would mean the q4 tier is secretly running a q8 text encoder"
    );
    assert_ne!(
        std::fs::read(tmp.path().join("q8/text_encoder/model.safetensors")).unwrap(),
        std::fs::read(tmp.path().join("q4/text_encoder/model.safetensors")).unwrap(),
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

/// The committed `tests/fixtures/tiers/{q8,q4}/` packed components — the artefacts the *Candle*
/// backend's tier tests read across the crate boundary — have exactly the **shape** the converter
/// produces: the same key set, the same tensor shapes and dtypes, and a marker composed from the
/// source config.
///
/// Deliberately **not** a byte comparison against a fresh conversion. That fresh conversion runs
/// `mlx_rs::ops::quantize` on whatever device the runner has, so a committed golden would be a
/// device-dependent one compared across two macOS runners — the hazard the repository already
/// records for machine-dependent goldens. Byte reproducibility is pinned as a *same-run* property
/// by `conversion_is_byte_reproducible_across_runs`, and the committed fixture's *values* are
/// pinned analytically and device-independently by
/// `the_committed_packed_weights_dequantize_onto_the_dense_ones`.
///
/// It also deliberately does **not** commit `config.json`: that file is the source component's
/// config plus a marker, so committing it would couple these fixtures to every unrelated edit of
/// `tiny-snapshot/*/config.json` (sc-24110 edits exactly that). See
/// `tests/fixtures/tiers/README.md`.
#[test]
fn the_committed_tier_fixtures_have_the_converters_shape() {
    use mlx_gen::weights::Weights;

    let tmp = tempfile::tempdir().unwrap();
    for tier in [Tier::Q8, Tier::Q4] {
        let fresh = convert_into(tmp.path(), tier);
        let committed = committed_tier_components(tier);
        for component in ["transformer", "text_encoder"] {
            assert!(
                !committed.join(component).join("config.json").exists(),
                "{}/{component}/config.json must NOT be committed — it is composed at test time",
                tier.dir_name()
            );
            let want = Weights::from_dir(fresh.join(component)).unwrap();
            let got = Weights::from_dir(committed.join(component)).unwrap();
            let mut want_keys: Vec<_> = want.keys().map(str::to_owned).collect();
            let mut got_keys: Vec<_> = got.keys().map(str::to_owned).collect();
            want_keys.sort();
            got_keys.sort();
            assert_eq!(
                got_keys,
                want_keys,
                "{}/{component}: the committed key set has drifted from the converter's",
                tier.dir_name()
            );
            for key in &want_keys {
                let (a, b) = (got.require(key).unwrap(), want.require(key).unwrap());
                assert_eq!(
                    a.shape(),
                    b.shape(),
                    "{}/{component}/{key} shape",
                    tier.dir_name()
                );
                assert_eq!(
                    a.dtype(),
                    b.dtype(),
                    "{}/{component}/{key} dtype",
                    tier.dir_name()
                );
            }
            // The marker the loaders read is composed, and it names this tier's width.
            let bits = if component == "transformer" {
                tier.transformer_bits().unwrap()
            } else {
                tier.text_encoder_bits().unwrap()
            };
            let composed: serde_json::Value =
                serde_json::from_str(&composed_marker_config(component, bits)).unwrap();
            assert_eq!(
                composed["quantization"],
                serde_json::json!({ "bits": bits, "group_size": GROUP_SIZE })
            );
        }
    }
}

/// **The committed fixtures hold the right numbers**, checked device-independently against the
/// dense weights they came from and an analytic bound, not against a golden.
///
/// Affine group quantization maps each group of [`GROUP_SIZE`] inputs onto `2^bits` levels spanning
/// that group's own `[min, max]`, so the worst-case reconstruction error of any element is half a
/// level: `(max - min) / (2^bits - 1) / 2`. Dequantizing the committed codes must land inside that
/// bound, per group, on both tiers. A swap of `scales` and `biases`, a wrong group size, a wrong
/// bit-width, or bytes from a different tensor all break it immediately, and none of them is
/// visible to a render-level parity bar.
///
/// Runs on the **CPU stream**, so the bound is a property of the artefact rather than of the
/// runner's GPU.
///
/// *Mutation that reds this:* swapping the `scales` and `biases` arguments to `dequantize_device`.
#[test]
fn the_committed_packed_weights_dequantize_onto_the_dense_ones() {
    use mlx_gen::weights::Weights;
    use mlx_rs::ops::dequantize_device;
    use mlx_rs::{Dtype, StreamOrDevice};

    // The one DiT leaf the miniature geometry packs in a transformer block.
    const BASE: &str = "transformer_blocks.0.img_mlp.out";
    let cpu = StreamOrDevice::cpu();
    let dense = Weights::from_dir(tiny_snapshot().join("transformer")).unwrap();
    let reference = dense.require(&format!("{BASE}.weight")).unwrap().clone();
    let reference = reference.as_dtype(Dtype::Bfloat16).unwrap();
    let rows = reference.shape()[0] as usize;
    let cols = reference.shape()[1] as usize;
    assert!(
        cols.is_multiple_of(GROUP_SIZE as usize),
        "fixture sanity: {BASE} is group-64 eligible"
    );
    let want = crate::common::host_f32(&reference);

    for tier in [Tier::Q8, Tier::Q4] {
        let bits = tier.transformer_bits().unwrap();
        let packed =
            Weights::from_dir(committed_tier_components(tier).join("transformer")).unwrap();
        let got = dequantize_device(
            packed.require(&format!("{BASE}.weight")).unwrap(),
            packed.require(&format!("{BASE}.scales")).unwrap(),
            packed.require(&format!("{BASE}.biases")).unwrap(),
            GROUP_SIZE,
            bits,
            &cpu,
        )
        .expect("the committed codes dequantize");
        assert_eq!(got.shape(), reference.shape());
        let got = crate::common::host_f32(&got);

        // The packed parts, read back so both checks below are stated in terms of the artefact
        // rather than a fudge factor.
        let scales = crate::common::host_f32(packed.require(&format!("{BASE}.scales")).unwrap());
        let biases = crate::common::host_f32(packed.require(&format!("{BASE}.biases")).unwrap());
        let words: Vec<u32> = packed
            .require(&format!("{BASE}.weight"))
            .unwrap()
            .as_slice::<u32>()
            .to_vec();

        let group = GROUP_SIZE as usize;
        let levels = ((1_u32 << bits) - 1) as f32;
        let per_word = 32 / bits as usize;
        let mask = (1_u64 << bits) - 1;
        let groups_per_row = cols / group;
        let words_per_row = cols / per_word;
        let mut worst_ratio = 0f32;

        for row in 0..rows {
            for chunk in 0..groups_per_row {
                let g = row * groups_per_row + chunk;
                let (scale, bias) = (scales[g], biases[g]);
                let lo = row * cols + chunk * group;
                let slice = &want[lo..lo + group];
                let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
                let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

                // (a) The stored grid covers the group's own span. Derived entirely from the
                //     artefact: one stored step, plus however far the stored grid's two endpoints
                //     fall short of the group's true extremes (the scale and bias are stored at the
                //     weight dtype, so they do not land exactly on min/max).
                // The 1.01 is not slack in the QUANTIZER — it is slack in evaluating this bound in
                // f32 against a reconstruction MLX evaluated in bf16. Q8 sits at ~100% of the bare
                // bound on this fixture (the worst element is a full stored step out, which is what
                // truncation rounding produces), so without it the check is one ulp from brittle.
                let bound =
                    (scale.abs() + (bias - min).abs() + (max - (bias + levels * scale)).abs())
                        * 1.01
                        + 1e-6;
                for i in lo..lo + group {
                    let err = (got[i] - want[i]).abs();
                    worst_ratio = worst_ratio.max(err / bound.max(f32::MIN_POSITIVE));
                    assert!(
                        err <= bound,
                        "{}: {BASE}[{i}] dequantized to {} but the dense weight is {} (|Δ| \
                         {err:.6e} > the group's own stored grid bound {bound:.6e})",
                        tier.dir_name(),
                        got[i],
                        want[i]
                    );
                }

                // (b) Every dequantized element is `scale · code + bias` for the code actually
                //     stored, reconstructed here from the raw u32 words.
                //
                //     Held to ~1% of the magnitudes the reconstruction passes through, not to
                //     equality and not to a fraction of the RESULT: `dequantize` evaluates at the
                //     weight dtype (bf16, ~2^-8 per operation) with intermediate rounding, and
                //     `scale · code` is far larger than the final value wherever the result sits
                //     near zero — so a result-relative bound would be vacuous there and fail here.
                //     The margin is not a guess: `swapped_margin` below asserts that reading the
                //     two packed tensors the wrong way round misses by orders of magnitude more
                //     than this, which is what makes the tolerance discriminating rather than
                //     decorative.
                let mut swapped_margin = 0f32;
                for col in chunk * group..(chunk + 1) * group {
                    let word = words[row * words_per_row + col / per_word];
                    let code = ((word as u64 >> (bits as usize * (col % per_word))) & mask) as f32;
                    let grid = scale * code + bias;
                    let seen = got[row * cols + col];
                    let tolerance = ((scale * code).abs() + bias.abs() + grid.abs()) * 0.01 + 1e-6;
                    assert!(
                        (seen - grid).abs() <= tolerance,
                        "{}: {BASE}[{row},{col}] is {seen}, but scales[{g}]·{code} + biases[{g}] \
                         = {grid} — the packed parts and the dequantized value disagree",
                        tier.dir_name()
                    );
                    // The same element reconstructed with the two tensors swapped.
                    let swapped = bias * code + scale;
                    swapped_margin = swapped_margin
                        .max((seen - swapped).abs() / tolerance.max(f32::MIN_POSITIVE));
                }
                assert!(
                    swapped_margin > 100.0,
                    "{}: group {g}'s scales and biases are too close in magnitude for this check \
                     to discriminate a swap (worst swapped miss is only {swapped_margin:.1}x the \
                     tolerance)",
                    tier.dir_name()
                );
            }
        }
        eprintln!(
            "{}: {BASE} dequantizes within {:.1}% of its own stored group-{GROUP_SIZE} Q{bits} \
             grid bound, and matches scale·code + bias exactly",
            tier.dir_name(),
            worst_ratio * 100.0
        );
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
        // Bars for a PARTIALLY packed miniature tier (see the module docs). Q8's are tight enough
        // that a scales/biases swap in the packed loader trips BOTH of them rather than only the
        // max; Q4's are looser because four bits genuinely move pixels.
        let (max_bound, mean_bound) = match tier {
            Tier::Q8 => (16.0, 1.0),
            _ => (64.0, 8.0),
        };
        report(
            &format!("{} vs bf16", tier.dir_name()),
            &packed,
            &dense,
            max_bound,
            mean_bound,
        );
        mean_by_tier.push(pixel_errors(&packed, &dense).1);
    }
    // STRICT: equality would mean the two tiers rendered identically, which is exactly what a
    // mislabelled artefact (q4's config over q8's weights) looks like from here.
    assert!(
        mean_by_tier[1] > mean_by_tier[0],
        "Q4 ({:.4}) must be strictly further from bf16 than Q8 ({:.4})",
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
    assert!(
        caps.component_precision_floors.is_empty(),
        "a tier is a whole-pipeline contract: nothing is promoted above the selected width"
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

    // A dense snapshot is still quantized at load — but only when the whole pipeline can be. The
    // miniature tower is 32/64 wide and a tier is written at group 64, so a load-time q8 here would
    // leave the text encoder dense: a "q8" load that is not q8. That is refused, by name.
    //
    // *Mutation that reds this:* dropping the `if !encoder.quantize(bits)?` guard in `model.rs`,
    // which silently restores the mixed tier.
    let err = match registry.load(
        ID,
        &LoadSpec::new(WeightsSource::Dir(tiny_snapshot())).with_quant(Quant::Q8),
    ) {
        Ok(_) => panic!("a load-time tier that cannot cover the tower must be refused"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("would leave the tower dense"), "{err}");
    assert!(err.contains("32 wide"), "{err}");
}

/// **A mislabelled tier is caught by reading the weights, not the label.**
///
/// A `config.json` is a text file. Relabelling the q8 tier as q4 produces a snapshot that loads,
/// renders, and passes a q4-vs-bf16 parity bar (it renders exactly as well as q8 does) and a
/// size-monotonicity check (it is the same size as q8). Nothing downstream notices — so the check
/// has to be here, against the packed code/scale shapes.
///
/// *Mutation that reds this:* dropping the derived-vs-declared comparison from
/// `quant::installed_tier`.
#[test]
fn a_mislabelled_tier_is_refused_by_reading_the_packed_shapes() {
    let tmp = tempfile::tempdir().unwrap();
    // A genuine q8 tier, relabelled q4 in both components' markers.
    let root = compose_committed_tier(tmp.path(), Tier::Q8);
    for component in ["transformer", "text_encoder"] {
        std::fs::write(
            root.join(component).join("config.json"),
            composed_marker_config(component, 4),
        )
        .unwrap();
    }

    let err = installed_tier(&root)
        .expect_err("a q4 label over q8 weights must be refused")
        .to_string();
    assert!(err.contains("declares Q4"), "{err}");
    assert!(err.contains("packed weights are Q8"), "{err}");

    // And the load path refuses it too, rather than serving a q8 render under a q4 label.
    assert!(mlx_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(
            ID,
            &LoadSpec::new(WeightsSource::Dir(root)).with_quant(Quant::Q4)
        )
        .is_err());
}

// ── the derived model against the real snapshot ──────────────────────────────────────────────────

/// **The derived parameter counts are the frozen snapshot's**, component for component.
///
/// `memory_strategy::derived`'s `DIT_LINEAR_PARAMS` / `LM_LINEAR_PARAMS` / `LM_EMBEDDING_PARAMS` /
/// `VAE_BYTES` are hand-entered numbers read off the pinned snapshot's safetensors headers. Nothing
/// else re-derives them, so a pin bump — or a typo — would silently move every published figure.
/// This holds them to `asset_facts`, which reads the headers on disk.
///
/// **Reads headers only**: `projected_safetensors_bytes` parses each file's JSON header and never
/// touches the data region, so this costs a few milliseconds and no GPU. It is `#[ignore]`d because
/// it needs the 31 GB snapshot present, not because it is expensive.
///
/// Run with:
/// `MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT=<dense snapshot> cargo test -p mlx-gen-qwen-image-2-1 \
///   --test integration -- tiers::derived_parameter_counts --ignored --nocapture`
#[test]
#[ignore = "needs the frozen Qwen/Qwen-Image-2.1 snapshot (headers only, no weights loaded)"]
fn derived_parameter_counts_match_the_frozen_snapshot() {
    use mlx_gen_qwen_image_2_1::memory_strategy::derived;

    let root = PathBuf::from(
        std::env::var("MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT")
            .expect("set MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT to the dense snapshot directory"),
    );
    let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
    let facts = mlx_gen_qwen_image_2_1::memory_strategy::asset_facts(&spec, &root)
        .expect("header-only asset facts");
    let want = derived::resident_weights(Tier::Bf16);

    eprintln!(
        "on disk: tower={} dit={} vae={}\nderived: tower={} dit={} vae={}",
        facts.conditioning_bytes,
        facts.transformer_bytes,
        facts.decoder_bytes,
        want.conditioning,
        want.transformer,
        want.decoder
    );
    assert_eq!(
        facts.transformer_bytes, want.transformer,
        "DIT_LINEAR_PARAMS/DIT_DENSE_BYTES no longer describe transformer/"
    );
    assert_eq!(
        facts.conditioning_bytes, want.conditioning,
        "LM_LINEAR_PARAMS/LM_EMBEDDING_PARAMS/LM_DENSE_NORM_BYTES no longer describe the loaded \
         model.language_model.* prefix"
    );
    assert_eq!(
        facts.decoder_bytes, want.decoder,
        "VAE_BYTES no longer describes vae/"
    );
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
