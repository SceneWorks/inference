//! Reference / edit parity vs the frozen `QwenImage21Pipeline` condition-image branch on the
//! committed miniature snapshot (`crates/media/mlx-gen/tools/dump_qwen21_edit.py`), plus the
//! ordering and refusal claims the route has to hold (sc-24110).
//!
//! Fixture and snapshot are the SAME ones the MLX twin reads.
//!
//! Every stage the port reproduces is gated separately, so a divergence names the stage rather
//! than showing up only in the final latents:
//!
//! 1. **preprocessing** — the `calculate_dimensions` fit, the single LANCZOS resize, the Qwen3-VL
//!    `pixel_values` / `grid_thw`, and the RGBA `[-1, 1]` VAE input;
//! 2. **text conditioning** — `encode_prompt(image=…)`'s embeddings and image-pad mask, i.e. the
//!    whole vision tower + DeepStack + interleaved-M-RoPE path;
//! 3. **reference latents** — the mode-sampled, normalised, packed condition latents, in order;
//! 4. **end to end** — the joint block-causal denoise from upstream's own initial noise, with and
//!    without true CFG, at one, two and the documented ten-reference boundary.
//!
//! Tolerances are set from **this lane's own measurement** (candle CPU f32 against torch CPU f32 —
//! the same arithmetic on the same hardware), not from the MLX twin's Metal numbers. Each bound
//! carries ~3-4x headroom over the worst measured drift; `assert_close` compares against
//! `tol * max(1, peak)` and prints its numbers, so no bound is ever tighter than its evidence.
//! Measured on this fixture:
//!
//! * **preprocessing** (`pixel_values`, `vae_input`) — every `vae_input` gate is **exactly 0**;
//!   the worst `pixel_values` gate is `max|Δ| = 2.384e-7` at `peak = 2.146` → `1.1e-7 x peak`,
//!   which is **one f32 ULP** at that magnitude. Bound **1e-6 x peak** (~4 ULP): this is host
//!   `u8 -> f32` arithmetic and is held near exact.
//! * **`prompt_embeds`** (ViT tower + DeepStack + interleaved M-RoPE + the decoder) — worst
//!   `ref10/prompt_embeds`, `max|Δ| = 1.132e-6` at `peak = 3.063` → `3.7e-7 x peak`.
//!   Bound **2e-6 x peak**.
//! * **reference latents** (VAE posterior mode, normalised, packed) — worst
//!   `aspect/ref_latents_0`, `max|Δ| = 2.682e-7` against a bound floor of 1.0. Bound **1e-6**.
//! * **end-to-end latents** (the whole joint block-causal denoise) — worst
//!   `ref10/latents_final`, `max|Δ| = 4.894e-5` at `peak = 5.007` → `9.8e-6 x peak`.
//!   Bound **3e-5 x peak**.
//!
//! The reference-ORDER claim is deliberately **not** made against any of these: `LATENT_TOL x
//! peak` is larger than upstream's own gap between the two orders, so an absolute bound there
//! would accept a port that ignored ordering entirely. See
//! [`swapping_two_references_changes_the_output`].
//!
//! For reference, the MLX twin measures 2.4e-7 / 4.9e-3 / 2.9e-4 / 2.1e-2 on Metal; the candle CPU
//! lane is two to three orders of magnitude tighter on the three model gates, which is why the
//! bounds here are not copied across.

use candle_core::{DType, Tensor};
use candle_gen::gen_core::{CancelFlag, Conditioning, Image, Progress};
use candle_gen_qwen_image_2_1::{
    denoise, encode_references, joint_layout, load_scheduler_config, load_text_encoder,
    load_tokenizer, load_transformer, load_vae, load_vision_config, prepare_references, scheduler,
    system_prompt_drop_count, text_rows, DenoiseInputs, PreparedReference, QwenImage21TextEncoder,
    ReferenceConditioning, TextConditioning,
};

use crate::common::{assert_close, device, host_f32, tiny_snapshot, Fixture};

const HOST_TOL: f32 = 1e-6;
const TEXT_TOL: f32 = 2e-6;
const REF_LATENT_TOL: f32 = 1e-6;
const LATENT_TOL: f32 = 3e-5;
/// The text-to-image conditioning against S1's frozen oracle gets its OWN bound, sized from its
/// own evidence (measured 1.192e-7 at peak 0.841). ~8x headroom.
const T2I_ORACLE_TOL: f32 = 1e-6;

/// The per-case metadata `dump_qwen21_edit.py` writes as JSON.
struct Case {
    name: &'static str,
    prompt: String,
    negative: Option<String>,
    true_cfg: f32,
    width: u32,
    height: u32,
    steps: usize,
    references: usize,
}

fn case(w: &Fixture, name: &'static str) -> Case {
    let v: serde_json::Value = serde_json::from_str(w.meta(name)).expect("case metadata is JSON");
    Case {
        name,
        prompt: v["prompt"].as_str().expect("prompt").to_string(),
        negative: v["negative_prompt"].as_str().map(str::to_string),
        true_cfg: v["true_cfg_scale"].as_f64().unwrap_or(1.0) as f32,
        width: v["width"].as_u64().expect("width") as u32,
        height: v["height"].as_u64().expect("height") as u32,
        steps: v["steps"].as_u64().expect("steps") as usize,
        references: v["references"].as_u64().expect("references") as usize,
    }
}

/// The case's source images, read back as the RGB8 `Image`s a request would carry.
fn sources(w: &Fixture, case: &Case) -> Vec<Image> {
    (0..case.references)
        .map(|i| {
            let tensor = w.tensor(&format!("{}/source_{i}", case.name));
            let dims = tensor.dims().to_vec();
            assert_eq!(dims.len(), 3, "source images are HWC RGB8");
            Image {
                height: dims[0] as u32,
                width: dims[1] as u32,
                pixels: host_f32(&tensor).into_iter().map(|v| v as u8).collect(),
            }
        })
        .collect()
}

struct Snapshot {
    tokenizer: candle_gen::gen_core::tokenizer::TextTokenizer,
    encoder: QwenImage21TextEncoder,
    transformer: candle_gen_qwen_image_2_1::QwenImage21Transformer,
    vae: candle_gen_qwen_image_2_1::QwenImage21Vae,
    scheduler: candle_gen_qwen_image_2_1::SchedulerConfig,
    drop: usize,
    vision: candle_gen_qwen_image_2_1::VisionConfig,
}

fn snapshot() -> Snapshot {
    let root = tiny_snapshot();
    let dev = device();
    let tokenizer = load_tokenizer(&root).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let vision = load_vision_config(&root)
        .unwrap()
        .expect("the tiny snapshot declares a Qwen3-VL vision tower");
    Snapshot {
        encoder: load_text_encoder(&root, &dev).unwrap(),
        transformer: load_transformer(&root, &dev).unwrap(),
        vae: load_vae(&root, &dev).unwrap(),
        scheduler: load_scheduler_config(&root).unwrap(),
        tokenizer,
        drop,
        vision,
    }
}

fn i32_host(t: &Tensor) -> Vec<i32> {
    t.to_dtype(DType::U32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap()
        .into_iter()
        .map(|v| v as i32)
        .collect()
}

fn prepared(snap: &Snapshot, w: &Fixture, c: &Case) -> Vec<PreparedReference> {
    prepare_references(&sources(w, c), &snap.vision, &device()).unwrap()
}

fn conditioning(
    snap: &Snapshot,
    prompt: &str,
    references: &[PreparedReference],
) -> TextConditioning {
    snap.encoder
        .encode_conditioning(&snap.tokenizer, prompt, snap.drop, references)
        .expect("image-conditioned encode")
}

/// The frozen `QwenImage21Pipeline.encode_prompt` text-to-image oracle S1 committed
/// (`crates/media/mlx-gen/tools/dump_qwen21_text_encoder.py`): `(tensor key, its prompt)`.
const FIXTURE_PROMPT_ORACLE: (&str, &str) = ("fox/prompt_embeds", "a red fox in the forest");

fn fixture_text_encoder() -> Fixture {
    Fixture::open("qwen21_text_encoder.safetensors")
}

fn te_tensor(w: &Fixture, key: &str) -> Tensor {
    w.tensor(key)
}

/// `[1, L, hidden]` -> `[L, hidden]`, the shape the S1 oracle stores (`embeds[0]`).
fn squeeze_batch(t: &Tensor) -> Tensor {
    t.squeeze(0).expect("a single-sample conditioning")
}

#[test]
fn the_tiny_snapshot_derives_upstreams_fit_from_its_own_pixel_budget() {
    let snap = snapshot();
    // The generator ran upstream at `output_resolution = 64`; the port derives the same value
    // from the snapshot's processor budget rather than taking it as a request field.
    assert_eq!(snap.vision.output_resolution(), 64);
    assert!(snap.encoder.has_vision());
}

#[test]
fn reference_preprocessing_matches_upstream() {
    let w = Fixture::open("qwen21_edit.safetensors");
    let snap = snapshot();
    for name in ["ref1", "ref2", "annotated", "mask_ref", "aspect"] {
        let c = case(&w, name);
        let refs = prepared(&snap, &w, &c);
        assert_eq!(refs.len(), c.references);
        for (i, reference) in refs.iter().enumerate() {
            let grid = i32_host(&w.tensor(&format!("{name}/grid_thw_{i}")));
            assert_eq!(
                reference.grid_thw.to_vec(),
                grid,
                "{name}/grid_thw_{i}: the vision grid must match upstream's"
            );
            assert_eq!(
                reference.latent_tokens(),
                reference.vision_slots() * 4,
                "{name}: every vision slot stands for a 2x2 latent group"
            );
            // Exact, and the only transposition-sensitive claim available: the VAE compresses 16x
            // and the processor's patch is 16 px, so the latent grid IS the vision grid, in the
            // same orientation. A transposed grid keeps the token count (and so slips through
            // every count check and, on the Metal lane, under the end-to-end bound) while binding
            // each non-square block to the wrong RoPE geometry.
            assert_eq!(
                reference.latent_grid(),
                (grid[1] as usize, grid[2] as usize),
                "{name}/latent_grid_{i}: the latent grid must match the vision grid's orientation"
            );
            assert_close(
                &format!("{name}/pixel_values_{i}"),
                &reference.pixel_values,
                &w.tensor(&format!("{name}/pixel_values_{i}")),
                HOST_TOL,
            );
            assert_close(
                &format!("{name}/vae_input_{i}"),
                &reference.vae_input,
                &w.tensor(&format!("{name}/vae_input_{i}")),
                HOST_TOL,
            );
        }
    }
}

#[test]
fn conditioning_matches_upstream_for_one_two_and_ten_references() {
    let w = Fixture::open("qwen21_edit.safetensors");
    let snap = snapshot();
    for name in ["ref1", "ref2", "ref10", "annotated", "mask_ref", "aspect"] {
        let c = case(&w, name);
        let refs = prepared(&snap, &w, &c);
        let cond = conditioning(&snap, &c.prompt, &refs);
        let want_mask = i32_host(&w.tensor(&format!("{name}/image_pad_mask")));
        assert_eq!(
            cond.image_pad_mask
                .iter()
                .map(|&m| i32::from(m))
                .collect::<Vec<_>>(),
            want_mask,
            "{name}: the image-pad mask places the condition blocks in the joint sequence"
        );
        assert_close(
            &format!("{name}/prompt_embeds"),
            &cond.hidden,
            &w.tensor(&format!("{name}/prompt_embeds")),
            TEXT_TOL,
        );
    }
}

#[test]
fn reference_latents_match_upstream_in_order() {
    let w = Fixture::open("qwen21_edit.safetensors");
    let snap = snapshot();
    for name in ["ref1", "ref2", "annotated", "mask_ref", "aspect"] {
        let c = case(&w, name);
        let refs = prepared(&snap, &w, &c);
        let latents = encode_references(&snap.vae, &refs).unwrap();
        assert_eq!(latents.len(), c.references);
        for (i, got) in latents.iter().enumerate() {
            assert_close(
                &format!("{name}/ref_latents_{i}"),
                got,
                &w.tensor(&format!("{name}/ref_latents_{i}")),
                REF_LATENT_TOL,
            );
        }
    }
}

/// Run the whole conditioned denoise for one case from upstream's own initial noise.
fn run_case(snap: &Snapshot, w: &Fixture, c: &Case) -> Tensor {
    run_case_feeding(snap, w, c, ReferenceFeed::InOrder)
}

/// Which order the condition latents reach the denoiser in. `Reversed` is a deliberately
/// wrong feed used by [`swapping_two_references_changes_the_output`]: the joint layout, the
/// text conditioning and the per-reference RoPE blocks all stay put, only the latents change
/// places, which is precisely the defect a magnitude-only ordering check cannot see.
#[derive(Clone, Copy, PartialEq)]
enum ReferenceFeed {
    InOrder,
    Reversed,
}

fn run_case_feeding(snap: &Snapshot, w: &Fixture, c: &Case, feed: ReferenceFeed) -> Tensor {
    let refs = prepared(snap, w, c);
    let pos = conditioning(snap, &c.prompt, &refs);
    let neg = c
        .negative
        .as_deref()
        .filter(|_| c.true_cfg > 1.0)
        .map(|n| conditioning(snap, n, &refs));

    let pos_layout = joint_layout(&pos.image_pad_mask, &refs, c.width, c.height).unwrap();
    let neg_layout = neg
        .as_ref()
        .map(|n| joint_layout(&n.image_pad_mask, &refs, c.width, c.height).unwrap());
    let pos_text = text_rows(&pos.hidden, &pos.image_pad_mask).unwrap();
    let neg_text = neg
        .as_ref()
        .map(|n| text_rows(&n.hidden, &n.image_pad_mask).unwrap());
    let mut reference_latents = encode_references(&snap.vae, &refs).unwrap();
    if feed == ReferenceFeed::Reversed {
        reference_latents.reverse();
    }

    let sigmas = scheduler::sigmas_for_image(&snap.scheduler, c.steps, c.width, c.height).unwrap();
    let init = w.tensor(&format!("{}/latents_init", c.name));
    let cancel = CancelFlag::new();
    let mut steps = Vec::new();
    let out = denoise(
        DenoiseInputs {
            transformer: &snap.transformer,
            sigmas: &sigmas,
            latents: init,
            prompt_embeds: &pos_text,
            negative_embeds: neg_text.as_ref(),
            true_cfg_scale: c.true_cfg,
            width: c.width,
            height: c.height,
            sampler: None,
            seed: 0,
            cancel: &cancel,
            references: Some(ReferenceConditioning {
                latents: &reference_latents,
                layout: &pos_layout,
                negative_layout: neg_layout.as_ref(),
            }),
        },
        &mut |p| {
            if let Progress::Step { current, total } = p {
                steps.push((current, total));
            }
        },
    )
    .unwrap();
    assert_eq!(
        steps,
        (1..=c.steps as u32)
            .map(|s| (s, c.steps as u32))
            .collect::<Vec<_>>(),
        "{}: progress",
        c.name
    );
    out
}

#[test]
fn conditioned_denoise_matches_upstream_end_to_end() {
    let w = Fixture::open("qwen21_edit.safetensors");
    let snap = snapshot();
    for name in [
        "ref1",
        "ref2",
        "ref10",
        "annotated",
        "mask_ref",
        "ref2_cfg",
        "aspect",
    ] {
        let c = case(&w, name);
        let got = run_case(&snap, &w, &c);
        assert_close(
            &format!("{name}/latents_final"),
            &got,
            &w.tensor(&format!("{name}/latents_final")),
            LATENT_TOL,
        );
    }
}

#[test]
fn swapping_two_references_changes_the_output() {
    let w = Fixture::open("qwen21_edit.safetensors");
    let snap = snapshot();
    let straight = run_case(&snap, &w, &case(&w, "ref2"));
    let swapped = run_case(&snap, &w, &case(&w, "ref2_swap"));

    // Both orders reproduce upstream…
    assert_close(
        "ref2_swap/latents_final",
        &swapped,
        &w.tensor("ref2_swap/latents_final"),
        LATENT_TOL,
    );
    // …and they are genuinely different renders. The claim is made against **upstream's own**
    // gap between the two orders rather than against an invented threshold: upstream separates
    // the two orders by `want_gap`, and the port has to separate them by the same amount. A port
    // that ignored ordering would score a gap of ~0 against a large `want_gap`.
    let gap = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let (got_a, got_b) = (host_f32(&straight), host_f32(&swapped));
    let want_a = host_f32(&w.tensor("ref2/latents_final"));
    let want_b = host_f32(&w.tensor("ref2_swap/latents_final"));
    let got_gap = gap(&got_a, &got_b);
    let want_gap = gap(&want_a, &want_b);
    let parity = gap(&got_a, &want_a).max(gap(&got_b, &want_b));
    let peak = want_a.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!(
        "reference order: got max|Δ|={got_gap:.3e} upstream max|Δ|={want_gap:.3e} \
         parity residual={parity:.3e} peak={peak:.3e}"
    );
    // FIRST, the direction. A magnitude comparison alone is order-INSENSITIVE:
    // Nor can a shape guard help — `calculate_dimensions` fits EVERY reference to the same target
    // area, so all of them carry the same token count (the `aspect` case's (2x8), (8x2) and (4x4)
    // blocks are all 16 tokens) and the joint layout's per-block counts still add up under any
    // permutation. Only the rendered values separate the orders.
    // A port that fed the condition latents to the denoiser in reverse would render `ref2` as
    // `ref2_swap` and vice versa, so `got_gap` would be just as large — and the end-to-end parity
    // bound is too loose to separate the two on its own. So pin the direction directly: feeding
    // the SAME latents in reverse must land measurably further from upstream's `ref2` than the
    // in-order feed does. Under a reversed port the two runs trade places and this inverts.
    let reversed_feed = run_case_feeding(&snap, &w, &case(&w, "ref2"), ReferenceFeed::Reversed);
    let in_order_err = gap(&got_a, &want_a);
    let reversed_err = gap(&host_f32(&reversed_feed), &want_a);
    eprintln!(
        "reference feed: in-order max|Δ|={in_order_err:.3e} reversed max|Δ|={reversed_err:.3e} \
         ratio={:.1}x",
        reversed_err / in_order_err.max(f32::MIN_POSITIVE)
    );
    assert!(
        reversed_err > 2.0 * in_order_err,
        "feeding the condition latents in REVERSE lands {reversed_err:.3e} from upstream \
         where the in-order feed lands {in_order_err:.3e}; the denoiser is not consuming the \
         condition latents in request order (measured 2.6x on the Metal lane, where the \
         numerical floor is what limits the separation, and ~1000x on the candle CPU lane)"
    );

    // THEN the magnitude, which catches the other failure: a port that dropped the ordering
    // signal entirely would score `got_gap` ~ 0 against a large `want_gap`. The signal has to
    // sit clear of this port's own numerical floor first, or the comparison measures noise.
    assert!(
        want_gap > 3.0 * parity,
        "upstream separates the two reference orders by only {want_gap:.3e}, within 3x this \
         port's parity residual {parity:.3e}; the ordering claim cannot be tested against it"
    );
    // RELATIVE to upstream's own gap, never to the parity tolerance: `LATENT_TOL * peak` is
    // larger than `want_gap` itself here, so an absolute bound would accept `got_gap = 0` — i.e.
    // a port that dropped the ordering signal entirely.
    assert!(
        (got_gap - want_gap).abs() <= 0.25 * want_gap,
        "the port separates the two reference orders by {got_gap:.3e} where upstream separates \
         them by {want_gap:.3e}; reference order is not reaching the denoiser the same way"
    );

    // The exact, noise-free half of the claim: the condition latents are carried in request
    // order, so swapping the request swaps the latent blocks position for position.
    let snapshot_refs = |name: &'static str| {
        let c = case(&w, name);
        encode_references(&snap.vae, &prepared(&snap, &w, &c)).unwrap()
    };
    let (a, b) = (snapshot_refs("ref2"), snapshot_refs("ref2_swap"));
    assert_eq!(host_f32(&a[0]), host_f32(&b[1]));
    assert_eq!(host_f32(&a[1]), host_f32(&b[0]));
    assert_ne!(host_f32(&a[0]), host_f32(&a[1]));
}

#[test]
fn the_boundary_holds_and_eleven_references_are_refused() {
    let w = Fixture::open("qwen21_edit.safetensors");
    let snap = snapshot();
    let dev = device();
    let limits: serde_json::Value = serde_json::from_str(w.meta("_limits")).unwrap();
    assert_eq!(limits["max_reference_images"].as_u64(), Some(10));

    let ten = sources(&w, &case(&w, "ref10"));
    assert_eq!(ten.len(), 10);
    assert!(
        prepare_references(&ten, &snap.vision, &dev).is_ok(),
        "ten references is the documented boundary and must be accepted"
    );

    let mut eleven = ten.clone();
    eleven.push(eleven[0].clone());
    let err = prepare_references(&eleven, &snap.vision, &dev)
        .unwrap_err()
        .to_string();
    assert!(err.contains("at most 10"), "{err}");

    // …and the same refusal through the request seam a caller actually uses.
    let mut req = candle_gen::gen_core::GenerationRequest {
        prompt: "a red fox".into(),
        width: 64,
        height: 64,
        ..Default::default()
    };
    req.conditioning = vec![Conditioning::MultiReference { images: eleven }];
    let err = candle_gen_qwen_image_2_1::collect_references(&req)
        .unwrap_err()
        .to_string();
    assert!(err.contains("at most 10"), "{err}");

    // Zero references on a request that carries conditioning is a refusal, not a silent T2I.
    req.conditioning = vec![Conditioning::MultiReference { images: vec![] }];
    let err = candle_gen_qwen_image_2_1::collect_references(&req)
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty MultiReference"), "{err}");
}

#[test]
fn a_mask_conditioning_is_refused_with_the_upstream_workaround() {
    let w = Fixture::open("qwen21_edit.safetensors");
    let limits: serde_json::Value = serde_json::from_str(w.meta("_limits")).unwrap();
    assert!(
        limits["mask_conditioning"]
            .as_str()
            .unwrap_or_default()
            .contains("refused"),
        "the fixture records that upstream exposes no mask tensor"
    );
    let c = case(&w, "mask_ref");
    let images = sources(&w, &c);
    assert_eq!(images.len(), 2, "the mask travels as an ordinary reference");

    let mut req = candle_gen::gen_core::GenerationRequest {
        prompt: "a red fox".into(),
        width: 64,
        height: 64,
        ..Default::default()
    };
    req.conditioning = vec![
        Conditioning::Reference {
            image: images[0].clone(),
            strength: None,
        },
        Conditioning::Mask {
            image: images[1].clone(),
        },
    ];
    let err = candle_gen_qwen_image_2_1::collect_references(&req)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no mask input"), "{err}");
    assert!(err.contains("extra reference"), "{err}");

    // The supported spelling — the mask as the second ordered reference — is accepted.
    req.conditioning = vec![Conditioning::MultiReference { images }];
    assert_eq!(
        candle_gen_qwen_image_2_1::collect_references(&req)
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn the_text_to_image_path_is_unchanged_by_the_reference_route() {
    // With no references the conditioned entry point must be the plain 1-D-RoPE text path: the
    // interleaved M-RoPE collapses when all three position rows are the token index, and nothing
    // else engages.
    //
    // The load-bearing claim is against the **committed upstream oracle**, not against this
    // crate's own `encode_prompt` — comparing the two entry points to each other only proves they
    // agree, and both share the early return, so a T2I regression introduced by this story would
    // move both together and still pass. `qwen21_text_encoder.safetensors` is the frozen
    // `QwenImage21Pipeline.encode_prompt` output for the same prompt, dumped before any of this.
    let snap = snapshot();
    let oracle = FIXTURE_PROMPT_ORACLE;
    let cond = conditioning(&snap, oracle.1, &[]);
    assert!(cond.image_pad_mask.iter().all(|m| !*m));
    let te = fixture_text_encoder();
    assert_close(
        "t2i/prompt_embeds (vs the frozen upstream oracle)",
        &squeeze_batch(&cond.hidden),
        &te_tensor(&te, oracle.0),
        T2I_ORACLE_TOL,
    );

    // …and, as the cheaper half, the two entry points still agree bit-for-bit, so the reference
    // route has not perturbed the text-to-image code path at all.
    let plain = snap
        .encoder
        .encode_prompt(&snap.tokenizer, oracle.1, snap.drop)
        .unwrap();
    let (a, b) = (host_f32(&plain), host_f32(&cond.hidden));
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().zip(&b).all(|(x, y)| x == y),
        "the reference route changed the text-to-image conditioning"
    );
}
