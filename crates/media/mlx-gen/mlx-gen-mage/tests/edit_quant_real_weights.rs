//! sc-15154 / sc-15111 — the **edit-path** tier gate.
//!
//! sc-15071 gave the generation path a structural per-tier gate, and explicitly left the Qwen3-VL
//! **vision tower** unvalidated: it is instantiated only by the three Edit variants, so a
//! text-to-image probe never loads it. That blind spot hid a defect of a different class from the
//! precision floors sc-15071 fixed — not a quality regression but a **geometry** one.
//!
//! `mlx_gen_boogu::VisionTower` is shared by Boogu, Krea and Mage, and until sc-15154 its packed
//! loader hard-coded Boogu's `quant::GROUP_SIZE` (**32**, forced by Boogu's `3360 = 32·105` DiT
//! hidden). Mage packs every component at **64**. Reading a group-64 pack at 32 halves the derived
//! input dim and doubles the derived bit-width:
//!
//! | tier | `model.visual.blocks.*.attn.qkv` | at 64 (as packed) | at 32 (as read) | symptom |
//! |---|---|---|---|---|
//! | q4 | `weight [3072,128]` u32, `scales [3072,16]` | in 1024, bits 4 | in 512, bits 8 | `[quantized_matmul] … (512, 3072) … group_size=32, bits=8` |
//! | q8 | `weight [3072,256]` u32, `scales [3072,16]` | in 1024, bits 8 | in 512, bits 16 | rejected at load: bit-width 16 ∉ {4,8} |
//! | bf16 | `weight [3072,1024]` bf16 | dense | dense | — (the only tier that could edit) |
//!
//! So **both** pre-quantized tiers — including the shipped default q4 — could not render an edit at
//! all, while the bf16 tier (a 17.46 GB install) could.
//!
//! ## What this file gates
//!
//! 1. **Every tier renders the reference edit**, scored against the bf16 edit at the same seed with
//!    the same `luma`/`block_means` correlation sc-15071 uses. This is the sc-15111 measurement: it
//!    is what says the vision tower is *correct* at q4/q8, not merely loadable. No speculative
//!    precision floor was added to `crate::quant::floor_bits` for `model.visual.*` because the
//!    measurement says none is needed — see the printed scores.
//! 2. **The mutation**: the same published tier with the tower read at Boogu's 32. It must fail,
//!    and it must fail with the geometry error, not with a plausible-looking image. That is the
//!    proof this gate discriminates — the floors in (1) are meaningless without it.
//!
//! ## Running it
//!
//! ```sh
//! MAGE_EDIT_SNAPSHOT=<microsoft/Mage-Flow-Edit dense snapshot> \
//! MAGE_EDIT_TIER_ROOT=<SceneWorks/Mage-Flow-Edit snapshot with q4/ q8/> \
//! MAGE_COMPONENTS_ROOT=<SceneWorks/Mage-Flow-Components-mlx snapshot with q4/ q8/> \
//! MAGE_EDIT_DUMP_DIR=/tmp/mage-edit \
//!   cargo test -p mlx-gen-mage --release --test edit_quant_real_weights -- --ignored --nocapture
//! ```
//!
//! `MAGE_EDIT_DUMP_DIR` is optional and writes each render as a PNG — the tiers must be **looked
//! at**, not only scored. A numbers-only acceptance is how sc-15071's tiled q4 shipped.

use image::RgbImage;
use mlx_gen::weights::Weights;
use mlx_gen_boogu::VisionTower;
use mlx_gen_mage::text_encoder::{
    load_lm_dir, load_tokenizer_dir, mage_vision_config, QUANT_GROUP_SIZE,
};
use mlx_gen_mage::{GsKey, MageComponentDirs, MageFlowPipeline, MageTextEncoder};

/// Boogu's group size — the constant the shared tower used to bake in. Named here so the mutation
/// below is unmistakably "what the loader did before sc-15154", not an arbitrary wrong number.
const BOOGU_GROUP_SIZE: i32 = 32;

const INSTRUCTION: &str = "Change the red cube to a blue sphere";
const NEGATIVE: &str = " ";
const EDIT_SEED: i64 = 42;
const EDIT_STEPS: usize = 30;
const EDIT_CFG: f32 = 5.0;
const EDIT_SIZE: u32 = 512;

/// Luma correlation each pre-quantized tier must reach against the **bf16 edit** at the same seed.
///
/// Fixed seed ⇒ the tiers integrate the same trajectory from the same noise ⇒ the renders are
/// spatially aligned, so a plain correlation measures "is this the same edited scene". Scale/offset
/// invariant, so a tier that is merely flatter than bf16 is not punished.
///
/// The floor sits in the empty gap between the tiers that render and the mutation that does not:
/// the mutation produces **no image at all** (a typed error), so there is no score to straddle.
/// See the module comment for the measured values printed by this test.
const EDIT_CORR_FLOOR: f64 = 0.70;
/// The same floor over 16×16 block means — scene layout rather than fine texture.
const EDIT_BLOCK_CORR_FLOOR: f64 = 0.75;

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("set {key} — see the module comment"))
}

/// The dense flat `microsoft/Mage-Flow-Edit` snapshot (the bf16 reference).
fn dense_snapshot() -> String {
    env("MAGE_EDIT_SNAPSHOT")
}

/// The SPLIT layout for `tier`: the per-tier DiT from the variant mirror, the (shared) text encoder
/// and VAE from the components mirror. Exactly the composition the app stages at install time.
fn tier_dirs(tier: &str) -> MageComponentDirs {
    let variant = std::path::PathBuf::from(env("MAGE_EDIT_TIER_ROOT")).join(tier);
    let components = std::path::PathBuf::from(env("MAGE_COMPONENTS_ROOT")).join(tier);
    MageComponentDirs {
        transformer: variant.join("transformer"),
        text_encoder: components.join("text_encoder"),
        vae: components.join("vae"),
    }
}

/// A deterministic 512² reference scene: a red cube on a white table under a warm wall, with a
/// window and a shadow. Synthesized rather than vendored so the gate carries no image bytes and no
/// licence question, and drawn with real edges/planes so the vision tower has structure to encode.
fn reference_image() -> RgbImage {
    let n = EDIT_SIZE;
    let mut img = RgbImage::new(n, n);
    let (nf, hf) = (n as f32, n as f32 / 2.0);
    for y in 0..n {
        for x in 0..n {
            let (xf, yf) = (x as f32, y as f32);
            // Wall (top ~55%) with a soft vertical gradient, then a light table surface.
            let mut px = if yf < nf * 0.55 {
                let t = yf / (nf * 0.55);
                [
                    (196.0 - 26.0 * t) as u8,
                    (188.0 - 24.0 * t) as u8,
                    (172.0 - 20.0 * t) as u8,
                ]
            } else {
                let t = (yf - nf * 0.55) / (nf * 0.45);
                [
                    (244.0 - 30.0 * t) as u8,
                    (242.0 - 30.0 * t) as u8,
                    (236.0 - 28.0 * t) as u8,
                ]
            };
            // A window on the wall — a bright rectangle with a mullion cross.
            if yf > nf * 0.08 && yf < nf * 0.36 && xf > nf * 0.62 && xf < nf * 0.92 {
                px = [232, 238, 246];
                let mid_x = (xf - nf * 0.77).abs() < 2.0;
                let mid_y = (yf - nf * 0.22).abs() < 2.0;
                if mid_x || mid_y {
                    px = [150, 152, 156];
                }
            }
            // Contact shadow on the table, left of and under the cube.
            let sh = ((xf - hf + 14.0) / 132.0).powi(2) + ((yf - nf * 0.70) / 34.0).powi(2);
            if sh < 1.0 && yf > nf * 0.55 {
                let k = 1.0 - sh;
                px = [
                    (px[0] as f32 * (1.0 - 0.30 * k)) as u8,
                    (px[1] as f32 * (1.0 - 0.30 * k)) as u8,
                    (px[2] as f32 * (1.0 - 0.28 * k)) as u8,
                ];
            }
            // The red cube: front face, a lighter top face, and a darker right face.
            let (cx, cy, s) = (hf - 12.0, nf * 0.60, 96.0);
            let front = xf >= cx - s / 2.0 && xf <= cx + s / 2.0 && yf >= cy && yf <= cy + s;
            let dx = xf - (cx - s / 2.0);
            let top =
                yf < cy && yf >= cy - 40.0 && dx >= (cy - yf) * 0.8 && dx <= s + (cy - yf) * 0.8;
            let dy = yf - cy;
            let right = xf > cx + s / 2.0
                && xf <= cx + s / 2.0 + 40.0
                && dy >= -(xf - (cx + s / 2.0)) * 0.8
                && dy <= s - (xf - (cx + s / 2.0)) * 0.8;
            if front {
                px = [198, 42, 38];
            } else if top {
                px = [232, 74, 62];
            } else if right {
                px = [142, 28, 26];
            }
            img.put_pixel(x, y, image::Rgb(px));
        }
    }
    img
}

fn render_edit(pipeline: &MageFlowPipeline, reference: &RgbImage) -> mlx_gen::Result<Vec<u8>> {
    let trace = pipeline.edit_trace(
        INSTRUCTION,
        NEGATIVE,
        std::slice::from_ref(reference),
        EDIT_SIZE,
        EDIT_SIZE,
        EDIT_STEPS,
        EDIT_CFG,
        EDIT_SEED,
        &GsKey::default(),
        false,
        &mut |_| {},
    )?;
    mlx_rs::transforms::eval([&trace.image_u8])?;
    Ok(trace.image_u8.as_slice::<u8>().to_vec())
}

fn luma(pixels: &[u8]) -> Vec<f64> {
    pixels
        .chunks_exact(3)
        .map(|c| 0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64)
        .collect()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mb) = (mean(a), mean(b));
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (u, v) = (x - ma, y - mb);
        num += u * v;
        da += u * u;
        db += v * v;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-12)
}

fn block_means(pixels: &[u8]) -> Vec<f64> {
    const BLOCK: usize = 16;
    let n = EDIT_SIZE as usize;
    let l = luma(pixels);
    let mut out = Vec::with_capacity((n / BLOCK) * (n / BLOCK));
    for by in 0..n / BLOCK {
        for bx in 0..n / BLOCK {
            let mut sum = 0.0;
            for y in 0..BLOCK {
                for x in 0..BLOCK {
                    sum += l[(by * BLOCK + y) * n + bx * BLOCK + x];
                }
            }
            out.push(sum / (BLOCK * BLOCK) as f64);
        }
    }
    out
}

fn stddev(pixels: &[u8]) -> f64 {
    let mean = pixels.iter().map(|&x| x as f64).sum::<f64>() / pixels.len() as f64;
    (pixels
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / pixels.len() as f64)
        .sqrt()
}

/// Write a render as a PNG when `MAGE_EDIT_DUMP_DIR` is set. The tiers have to be looked at.
fn dump(name: &str, pixels: &[u8]) {
    let Ok(dir) = std::env::var("MAGE_EDIT_DUMP_DIR") else {
        return;
    };
    std::fs::create_dir_all(&dir).expect("create dump dir");
    let path = std::path::Path::new(&dir).join(format!("{name}.png"));
    RgbImage::from_raw(EDIT_SIZE, EDIT_SIZE, pixels.to_vec())
        .expect("RGB8 render")
        .save(&path)
        .expect("write png");
    println!("wrote {}", path.display());
}

/// Rebuild `tier`'s text encoder with the vision tower read at Boogu's [`BOOGU_GROUP_SIZE`] — the
/// shared loader exactly as it behaved before sc-15154, over the real published packed artifact.
/// Returns the tower's construction error when it fails at load (q8), or the assembled encoder when
/// it does not (q4, which fails later in the first vision matmul).
fn text_encoder_with_vision_at_boogu_group_size(tier: &str) -> mlx_gen::Result<MageTextEncoder> {
    let dir = tier_dirs(tier).text_encoder;
    let weights = Weights::from_dir(&dir)?;
    let vision = VisionTower::from_weights(
        &weights,
        mage_vision_config(),
        "model.visual",
        BOOGU_GROUP_SIZE,
    )?;
    Ok(MageTextEncoder::new_multimodal(
        load_tokenizer_dir(&dir)?,
        load_lm_dir(&dir)?,
        vision,
    ))
}

#[test]
#[ignore = "needs the published Mage-Flow-Edit tiers, the components mirror and a Metal device"]
fn every_tier_renders_the_reference_edit_and_the_boogu_group_size_fails_the_gate() {
    let reference = reference_image();
    dump("source", reference.as_raw());

    // --- the bf16 reference edit (dense tower — the only tier that ever worked) -----------------
    let bf16 = {
        let pipeline = MageFlowPipeline::load_edit(dense_snapshot(), None).unwrap();
        let pixels = render_edit(&pipeline, &reference).expect("bf16 edit");
        drop(pipeline);
        pixels
    };
    dump("edit_bf16", &bf16);
    println!("bf16: stddev={:.3}", stddev(&bf16));
    mlx_rs::memory::clear_cache();

    // --- the pre-quantized tiers, through the real SPLIT layout ---------------------------------
    let mut scores = Vec::new();
    for (tier, bits) in [("q8", 8), ("q4", 4)] {
        let pipeline = MageFlowPipeline::load_components(
            &tier_dirs(tier),
            Some(bits),
            mlx_gen_mage::vae::VaePart::Both,
        )
        .unwrap_or_else(|e| panic!("{tier} edit pipeline failed to load: {e}"));
        let pixels = render_edit(&pipeline, &reference)
            .unwrap_or_else(|e| panic!("{tier} edit failed to render: {e}"));
        drop(pipeline);
        dump(&format!("edit_{tier}"), &pixels);
        let corr = correlation(&luma(&pixels), &luma(&bf16));
        let block = correlation(&block_means(&pixels), &block_means(&bf16));
        println!(
            "{tier}: luma_corr={corr:.4} block_corr={block:.4} stddev={:.3}",
            stddev(&pixels)
        );
        assert!(
            corr >= EDIT_CORR_FLOOR,
            "{tier} did not render the reference edit: luma correlation {corr:.4} < \
             {EDIT_CORR_FLOOR}. The tier is producing a different image, not a lower-fidelity \
             version of the same one."
        );
        assert!(
            block >= EDIT_BLOCK_CORR_FLOOR,
            "{tier} edit layout diverged: block correlation {block:.4} < {EDIT_BLOCK_CORR_FLOOR}"
        );
        scores.push((tier, corr, block));
        mlx_rs::memory::clear_cache();
    }
    assert!(
        scores[0].1 > EDIT_CORR_FLOOR && scores[1].1 > EDIT_CORR_FLOOR,
        "both pre-quantized tiers must clear the floor"
    );

    // --- the mutation: the shared tower read at Boogu's group size ------------------------------
    // This is the defect, reproduced over the real published bytes rather than simulated. It must
    // fail for BOTH tiers, and the failure must be the geometry — a mutation that merely produced a
    // slightly worse image would mean the group size is not actually load-bearing here and the
    // floors above are measuring nothing.
    for tier in ["q8", "q4"] {
        let outcome = text_encoder_with_vision_at_boogu_group_size(tier).and_then(|encoder| {
            let mut pipeline = MageFlowPipeline::load_components(
                &tier_dirs(tier),
                Some(if tier == "q8" { 8 } else { 4 }),
                mlx_gen_mage::vae::VaePart::Both,
            )?;
            pipeline.text_encoder = encoder;
            render_edit(&pipeline, &reference)
        });
        let error = match outcome {
            Ok(pixels) => {
                dump(&format!("edit_{tier}_mutation"), &pixels);
                panic!(
                    "MUTATION SURVIVED: {tier} rendered an edit with the vision tower read at \
                     group_size {BOOGU_GROUP_SIZE} while the artifact is packed at \
                     {QUANT_GROUP_SIZE}. That is the sc-15154 defect, so a gate it passes is not a \
                     gate — the tower's group size is evidently no longer load-bearing on this \
                     path and this test must be re-derived rather than deleted."
                );
            }
            Err(error) => error.to_string(),
        };
        println!("mutation [{tier} vision tower at group_size {BOOGU_GROUP_SIZE}]: {error}");
        assert!(
            error.contains("group_size")
                || error.contains("bit-width")
                || error.contains("quantized_matmul"),
            "the mutation must fail on the packed GEOMETRY (the observed q4/q8 symptoms), not on \
             something incidental: {error}"
        );
        mlx_rs::memory::clear_cache();
    }
}
