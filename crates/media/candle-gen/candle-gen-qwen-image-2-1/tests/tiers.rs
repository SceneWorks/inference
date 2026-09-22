//! Installable Q8/Q4 tiers on the Candle backend (sc-24112).
//!
//! The tiers this backend installs are **the artefacts the MLX converter writes** — the committed
//! `mlx-gen-qwen-image-2-1/tests/fixtures/tiers/{q8,q4}/` packed components, reached across the
//! backend boundary by the same relative-path convention the parity fixtures already use. Building a
//! second, hand-rolled packed fixture here would test a file no one publishes; these tests
//! deliberately read the one that is.
//!
//! A tier's other components (`vae/`, `processor/`, `scheduler/`, `model_index.json`) are copied
//! through **dense and unchanged** by the converter, so a full tier is composed at run time from the
//! committed packed dirs plus the dense remainder of `tiny-snapshot`. That composition is itself a
//! check: if the dense remainder ever stopped being byte-identical to the source snapshot, the
//! composed tier would not load.
//!
//! **Why the fixture tiers are partly dense**: the miniature geometry is 32/64 wide and a shippable
//! tier declares exactly one `quantization.group_size` (64), so narrower `Linear`s stay dense. The
//! released geometry has no such width, so in production the same converter packs everything —
//! which makes the parity bars here conservative.

use std::path::{Path, PathBuf};

use candle_gen::gen_core::{
    GenerationMemory, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    OffloadPolicy as Policy, Quant, WeightsSource,
};
use candle_gen_qwen_image_2_1::quant::{installed_tier, resolve_requested_tier, Tier, GROUP_SIZE};

use crate::common::{fixtures, tiny_snapshot};

const ID: &str = "qwen_image_2_1";
const EDGE: u32 = 64;
/// See `tiers::bounded_decode_*` in the MLX twin: the decoder upsamples 16x through four stages, so
/// a tile only a couple of latent cells wide cannot reproduce the untiled decode however it is
/// blended. 128 over 256 is the smallest representative geometry on this fixture.
const DECODE_EDGE: u32 = 256;
const DECODE_TILE: u32 = 128;
const DECODE_TILE_OVERLAP: u32 = 64;

// ── composing a tier ─────────────────────────────────────────────────────────────────────────────

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

/// `tiny-snapshot/<component>/config.json` + `{"quantization": {bits, group_size}}` — the same
/// composition `mlx_gen::quant::write_quantized_config` performs.
///
/// The fixture deliberately ships **no** `config.json`: a packed component's config is the source
/// config plus a marker, so committing it would couple these fixtures to every unrelated edit of
/// `tiny-snapshot/*/config.json` (sc-24110 edits exactly that).
fn composed_marker_config(component: &str, bits: i64) -> String {
    let source = tiny_snapshot().join(component).join("config.json");
    let mut value: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&source).expect("source config"))
            .expect("valid json");
    value["quantization"] = serde_json::json!({ "bits": bits, "group_size": GROUP_SIZE });
    serde_json::to_string_pretty(&value).expect("serializable")
}

/// A complete `tier` snapshot under `dir`: the committed packed `model.safetensors` of
/// `transformer/` + `text_encoder/` with a composed marker config, plus the dense `vae/`,
/// `processor/`, `scheduler/` and `model_index.json` the converter copies through unchanged.
fn compose_tier(dir: &Path, tier: Tier) -> PathBuf {
    let out = dir.join(tier.dir_name());
    std::fs::create_dir_all(&out).unwrap();
    let packed = fixtures().join("tiers").join(tier.dir_name());
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
    let generator = candle_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(ID, spec)
        .expect("the snapshot loads through the catalog path");
    match generator.generate(req, &mut |_| {}).expect("render") {
        GenerationOutput::Images(mut images) => images.remove(0),
        other => panic!("images expected, got {other:?}"),
    }
}

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

/// The three `(label, root, quant)` triples every parity test sweeps.
fn tier_cases(dir: &Path) -> Vec<(&'static str, PathBuf, Option<Quant>)> {
    vec![
        ("bf16", tiny_snapshot(), None),
        ("q8", compose_tier(dir, Tier::Q8), Some(Quant::Q8)),
        ("q4", compose_tier(dir, Tier::Q4), Some(Quant::Q4)),
    ]
}

fn spec_for(root: &Path, quant: Option<Quant>, policy: OffloadPolicy) -> LoadSpec {
    let mut spec =
        LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_offload_policy(policy);
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    spec
}

// ── the tiers install ────────────────────────────────────────────────────────────────────────────

/// **The fixture tiers genuinely pack BOTH components** — the guard against the rest of this file
/// becoming vacuous.
///
/// Everything below renders through the packed-detect seam, but a snapshot with no packed tensor at
/// all would render identically to the dense one and every parity bar would pass while proving
/// nothing. The miniature geometry is mostly narrower than group 64, so only a few leaves are
/// eligible; this pins that the eligible ones ARE packed, in the DiT **and** in the Qwen3 tower.
#[test]
fn the_fixture_tiers_pack_both_the_dit_and_the_tower() {
    for tier in [Tier::Q8, Tier::Q4] {
        let dir = fixtures().join("tiers").join(tier.dir_name());
        for (component, expected) in [
            (
                "transformer",
                vec![
                    "time_text_embed.timestep_embedder.linear_1",
                    "transformer_blocks.0.img_mlp.out",
                    "transformer_blocks.1.img_mlp.out",
                ],
            ),
            (
                // The tower's only group-64-eligible width on this geometry is the SwiGLU
                // `down_proj` (in = intermediate_size = 64). Two layers, so two packed triples —
                // which is what makes the candle text-encoder packed path non-vacuous here.
                "text_encoder",
                vec![
                    "model.language_model.layers.0.mlp.down_proj",
                    "model.language_model.layers.1.mlp.down_proj",
                ],
            ),
        ] {
            assert!(
                !dir.join(component).join("config.json").exists(),
                "{}/{component}/config.json must NOT be committed — it is composed at test time",
                tier.dir_name()
            );
            let bytes = std::fs::read(dir.join(component).join("model.safetensors")).unwrap();
            let parsed = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let mut packed: Vec<String> = parsed
                .names()
                .into_iter()
                .filter_map(|name| name.strip_suffix(".scales").map(str::to_owned))
                .collect();
            packed.sort();
            assert_eq!(
                packed,
                expected,
                "{}/{component}: the committed tier must pack exactly the group-64-eligible leaves",
                tier.dir_name()
            );
            for base in &packed {
                assert!(
                    parsed.tensor(&format!("{base}.biases")).is_ok(),
                    "{base} is missing its packed bias table"
                );
                assert_eq!(
                    parsed.tensor(&format!("{base}.weight")).unwrap().dtype(),
                    safetensors::Dtype::U32,
                    "{base} codes must be u32-packed"
                );
            }
        }
    }
}

/// **The q4 tier's tower is a q4 artefact**, not a q8 one carried under a q4 label.
///
/// An earlier revision of this story held the tower at Q8 on the q4 tier and asserted the two
/// artefacts were IDENTICAL. That is the inverse of the property that matters: a tier is a
/// whole-pipeline contract.
///
/// *Mutation that reds this:* `Tier::text_encoder_bits` returning `Some(8)` for Q4.
#[test]
fn the_q4_tower_is_a_q4_artefact_not_the_q8_one() {
    let q8 = fixtures().join("tiers/q8/text_encoder/model.safetensors");
    let q4 = fixtures().join("tiers/q4/text_encoder/model.safetensors");
    assert_ne!(
        std::fs::read(&q8).unwrap(),
        std::fs::read(&q4).unwrap(),
        "an identical tower artefact would mean the q4 tier runs a q8 text encoder"
    );
    assert!(
        std::fs::metadata(&q4).unwrap().len() < std::fs::metadata(&q8).unwrap().len(),
        "the q4 tower must be the smaller artefact"
    );
    // And the codes really are half as wide: `[out, in·bits/32]`.
    let base = "model.language_model.layers.0.mlp.down_proj.weight";
    let width = |path: &Path| {
        let bytes = std::fs::read(path).unwrap();
        safetensors::SafeTensors::deserialize(&bytes)
            .unwrap()
            .tensor(base)
            .unwrap()
            .shape()[1]
    };
    assert_eq!(
        width(&q4) * 2,
        width(&q8),
        "q4 codes are half the width of q8's"
    );

    // ...and the committed artefact matches what the TIER TABLE says its tower should be. Without
    // this the test would pin the fixture without pinning the contract, and a code change that
    // re-floored the tower would pass here until someone regenerated the fixture.
    for (tier, path) in [(Tier::Q8, &q8), (Tier::Q4, &q4)] {
        let bytes = std::fs::read(path).unwrap();
        let parsed = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let wq = parsed.tensor(base).unwrap().shape().to_vec();
        let scales = parsed
            .tensor("model.language_model.layers.0.mlp.down_proj.scales")
            .unwrap()
            .shape()
            .to_vec();
        // `[out, in/group]` scales ⇒ in = cols·group; `[out, in·bits/32]` codes ⇒ bits = cols·32/in.
        let derived = (wq[1] * 32 / (scales[1] * GROUP_SIZE)) as i64;
        assert_eq!(
            Some(derived),
            tier.text_encoder_bits(),
            "{}: the committed tower is Q{derived}, but the tier table says {:?}",
            tier.dir_name(),
            tier.text_encoder_bits()
        );
    }
}

/// The composed tier is what the converter says it is, and candle reads the marker the same way the
/// MLX loader does.
#[test]
fn a_composed_tier_self_reports_and_resolves_its_own_label() {
    let tmp = tempfile::tempdir().unwrap();
    for tier in [Tier::Q8, Tier::Q4] {
        let root = compose_tier(tmp.path(), tier);
        assert_eq!(installed_tier(&root).unwrap(), tier);
        assert_eq!(
            resolve_requested_tier(&root, tier.selected_quant()).unwrap(),
            tier
        );
        // The wrong label, and no label at all, are both refused rather than served.
        let other = if tier == Tier::Q8 {
            Quant::Q4
        } else {
            Quant::Q8
        };
        assert!(resolve_requested_tier(&root, Some(other)).is_err());
        assert!(resolve_requested_tier(&root, None).is_err());
    }
    // The dense snapshot is unchanged: bf16, and a quantize request is still the typed refusal.
    assert_eq!(installed_tier(&tiny_snapshot()).unwrap(), Tier::Bf16);
    let err = resolve_requested_tier(&tiny_snapshot(), Some(Quant::Q8))
        .unwrap_err()
        .to_string();
    assert!(err.contains("on-the-fly"), "{err}");
}

/// **The descriptor is truthful after this story.** Candle advertises both affine tiers because both
/// are installable here — and the Q4 text-encoder floor is descriptor-visible, so a caller's
/// effective-tier label carries the substitution.
#[test]
fn the_descriptor_advertises_what_is_actually_installable() {
    let caps = candle_gen_qwen_image_2_1::descriptor().capabilities;
    assert_eq!(caps.supported_quants, &[Quant::Q4, Quant::Q8]);
    // **A tier is a whole-pipeline contract**: nothing is resident above the selected width, so
    // there is no floor to declare.
    //
    // *Mutation that reds this:* reinstating a `component_precision_floors` table on the descriptor.
    assert!(caps.component_precision_floors.is_empty());
    for tier in [Tier::Q8, Tier::Q4] {
        assert_eq!(tier.text_encoder_bits(), tier.transformer_bits());
    }

    // And the load path honours exactly that: each tier loads under its own label only.
    let tmp = tempfile::tempdir().unwrap();
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    for tier in [Tier::Q8, Tier::Q4] {
        let root = compose_tier(tmp.path(), tier);
        assert!(
            registry
                .load(
                    ID,
                    &spec_for(&root, tier.selected_quant(), Policy::Resident)
                )
                .is_ok(),
            "{} must install",
            tier.dir_name()
        );
        assert!(
            registry
                .load(ID, &spec_for(&root, None, Policy::Resident))
                .is_err(),
            "{} must not be served unlabelled",
            tier.dir_name()
        );
    }
    // A dense snapshot with a quantize request keeps the historical typed refusal.
    let err = match registry.load(
        ID,
        &spec_for(&tiny_snapshot(), Some(Quant::Q8), Policy::Resident),
    ) {
        Ok(_) => panic!("candle must still refuse on-the-fly quantization"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("on-the-fly"), "{err}");
}

/// **A mislabelled tier is caught by reading the weights, not the label.**
///
/// Relabelling the q8 tier as q4 produces a snapshot that loads, renders, and passes a q4-vs-bf16
/// parity bar (it renders exactly as well as q8 does) and a size-monotonicity check (it is the same
/// size as q8). Nothing downstream notices — so the check has to be against the packed code/scale
/// shapes.
///
/// *Mutation that reds this:* dropping the derived-vs-declared comparison from
/// `quant::installed_tier`.
#[test]
fn a_mislabelled_tier_is_refused_by_reading_the_packed_shapes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = compose_tier(tmp.path(), Tier::Q8);
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

    assert!(candle_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load(ID, &spec_for(&root, Some(Quant::Q4), Policy::Resident))
        .is_err());
}

/// A packed tier loads through the production catalog path and renders. Both the DiT **and** the
/// Qwen3 tower are packed in an installed tier, so this exercises the packed-detect seam on both.
#[test]
fn packed_tiers_reload_and_render_within_the_declared_bars() {
    let tmp = tempfile::tempdir().unwrap();
    let req = request(EDGE, EDGE);
    let dense = render(&spec_for(&tiny_snapshot(), None, Policy::Resident), &req);

    let mut mean_by_tier = Vec::new();
    for (label, root, quant) in tier_cases(tmp.path()).into_iter().skip(1) {
        let packed = render(&spec_for(&root, quant, Policy::Resident), &req);
        assert_eq!((packed.width, packed.height), (EDGE, EDGE));
        // Q8's bars are tight enough that a scales/biases swap in the packed loader trips BOTH of
        // them rather than only the max; Q4's are looser because four bits genuinely move pixels.
        let (max_bound, mean_bound) = if label == "q8" {
            (16.0, 1.0)
        } else {
            (64.0, 8.0)
        };
        report(
            &format!("{label} vs bf16"),
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

// ── memory strategies ────────────────────────────────────────────────────────────────────────────

/// Staged residency is an execution decision, not a numeric one: dropping the tower between
/// conditioning and denoise reproduces the resident render **exactly**, at every tier.
#[test]
fn staged_residency_preserves_the_resident_output_at_every_tier() {
    let tmp = tempfile::tempdir().unwrap();
    let req = request(EDGE, EDGE);
    for (label, root, quant) in tier_cases(tmp.path()) {
        assert_eq!(
            render(&spec_for(&root, quant, Policy::Sequential), &req).pixels,
            render(&spec_for(&root, quant, Policy::Resident), &req).pixels,
            "{label}: staged residency must be byte-identical to the resident render"
        );
    }
}

/// Bounded (tiled) decode is a memory lever, not a quality one: the head-once/tail-tiled decode
/// reproduces the untiled decode to within seam-blend noise, at every tier.
#[test]
fn bounded_decode_preserves_the_untiled_output_at_every_tier() {
    let tmp = tempfile::tempdir().unwrap();
    for (label, root, quant) in tier_cases(tmp.path()) {
        let spec = spec_for(&root, quant, Policy::Resident);
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
        report(
            &format!("{label} tiled vs untiled"),
            &tiled,
            &resident,
            24.0,
            2.0,
        );
    }
}

/// The memory contract prices the tier actually on disk: a packed tier's components are smaller than
/// the dense snapshot's, the staged floor is `max(tower, DiT + VAE)` rather than the sum, and the Q4
/// tier's tower equals the Q8 tier's — the declared floor, visible in the priced bytes.
#[test]
fn the_contract_prices_each_installed_tier_from_its_own_inventory() {
    let tmp = tempfile::tempdir().unwrap();
    let facts = |root: &Path, quant: Option<Quant>| {
        candle_gen_qwen_image_2_1::memory_strategy::memory_strategy_contract(
            ID,
            &spec_for(root, quant, Policy::Resident),
        )
        .expect("contract")
        .asset_facts
    };

    let dense = facts(&tiny_snapshot(), None);
    let q8 = facts(&compose_tier(tmp.path(), Tier::Q8), Some(Quant::Q8));
    let q4 = facts(&compose_tier(tmp.path(), Tier::Q4), Some(Quant::Q4));

    for f in [&dense, &q8, &q4] {
        assert!(f.conditioning_bytes > 0 && f.transformer_bytes > 0 && f.decoder_bytes > 0);
        assert_eq!(
            f.base_bytes,
            f.conditioning_bytes + f.transformer_bytes + f.decoder_bytes
        );
        // Staging drops the tower before the heavy pair loads: the floor is the max, not the sum.
        let floor = f
            .conditioning_bytes
            .max(f.transformer_bytes + f.decoder_bytes);
        assert!(floor < f.base_bytes, "the staged floor must be a max");
    }

    assert!(
        q8.transformer_bytes < dense.transformer_bytes,
        "the packed Q8 DiT must be cheaper than the dense one"
    );
    assert!(
        q4.transformer_bytes < q8.transformer_bytes,
        "the packed Q4 DiT must be cheaper than the Q8 one"
    );
    assert!(
        q4.conditioning_bytes < q8.conditioning_bytes,
        "a q4 tier prices a q4 tower ({} must be below {})",
        q4.conditioning_bytes,
        q8.conditioning_bytes
    );
    assert_eq!(
        q4.decoder_bytes, dense.decoder_bytes,
        "the all-conv VAE never packs"
    );
}

/// The admission envelope a consumer gates on is reported, and it is the same on both backends —
/// it is a property of the model, not of the backend. Pinned here against the numbers the SceneWorks
/// half will read.
#[test]
fn the_admission_envelope_is_reported_for_the_consumer_contract() {
    let g = candle_gen_qwen_image_2_1::admission_geometry();
    let caps = candle_gen_qwen_image_2_1::descriptor().capabilities;
    assert_eq!(g.max_side, caps.max_size);
    assert_eq!(g.max_batch, caps.max_count);
    // Largest by AREA is the 4:3 preset, not the widest and not the square default.
    assert_eq!(g.max_preset_area, 2400 * 1792);
    assert_eq!(g.max_target_image_tokens, 150 * 112);
    assert_eq!(
        g.max_reference_images,
        candle_gen_qwen_image_2_1::MAX_REFERENCE_IMAGES as u32
    );
    // A reference is FITTED to `OUTPUT_RESOLUTION` (1024) before the vision tower and the VAE see
    // it, so it is 64x64 latent tokens whatever the target size — NOT the target's 150x112.
    //
    // *Mutation that reds this:* `tokens_per_max_reference: target_tokens`, which overstates the
    // worst-case joint sequence by 3.4x and would have a consumer refuse requests that do fit.
    assert_eq!(
        g.tokens_per_max_reference,
        candle_gen_qwen_image_2_1::memory_strategy::REFERENCE_FIT_TOKENS
    );
    assert_eq!(g.tokens_per_max_reference, 64 * 64);
    assert!(g.tokens_per_max_reference < g.max_target_image_tokens);
    assert_eq!(
        g.max_joint_tokens,
        candle_gen_qwen_image_2_1::memory_strategy::TABLE_CONDITIONING_TOKENS
            + g.max_target_image_tokens
            + 10 * g.tokens_per_max_reference,
        "the largest-area target plus ten FITTED references"
    );
    assert_eq!(g.max_joint_tokens, 58_016, "256 + 16_800 + 10 x 4_096");
}
