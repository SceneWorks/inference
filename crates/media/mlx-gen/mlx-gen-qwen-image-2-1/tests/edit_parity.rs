//! Reference / edit parity vs the frozen `QwenImage21Pipeline` condition-image branch on the
//! committed miniature snapshot (`tools/dump_qwen21_edit.py`), plus the ordering and refusal
//! claims the route has to hold (sc-24110).
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
//! Tolerances, from the measured drift with stated headroom (f32 Metal against f32 CPU torch;
//! every comparison prints its numbers, and the preprocessing gates are host arithmetic so they
//! are held near exact):
//!
//! * `pixel_values` / `vae_input`: bound **1e-5 × peak** (host `u8 → f32` arithmetic);
//! * `prompt_embeds`: bound **2e-2 × peak** — the same bar the text-encoder gate holds, over a
//!   longer graph (ViT tower + DeepStack + the decoder);
//! * reference latents and end-to-end latents: bound **2.5e-2 × peak**, the bar
//!   `pipeline_parity` already holds the text-to-image denoise to.

use mlx_gen::gen_core::{Conditioning, Image, Progress};
use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_qwen_image_2_1::{
    denoise, encode_references, joint_layout, load_scheduler_config, load_text_encoder,
    load_tokenizer, load_transformer, load_vae, load_vision_config, prepare_references, scheduler,
    system_prompt_drop_count, DenoiseInputs, PreparedReference, QwenImage21TextEncoder,
    ReferenceConditioning, TextConditioning,
};
use mlx_rs::{Array, Dtype};

use crate::common::{assert_close, fixture, host_f32, tiny_snapshot};

const HOST_TOL: f32 = 1e-5;
const TEXT_TOL: f32 = 2e-2;
const LATENT_TOL: f32 = 2.5e-2;

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

fn case(w: &Weights, name: &'static str) -> Case {
    let raw = w
        .metadata(name)
        .unwrap_or_else(|| panic!("fixture metadata for case {name}"));
    let v: serde_json::Value = serde_json::from_str(raw).expect("case metadata is JSON");
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
fn sources(w: &Weights, case: &Case) -> Vec<Image> {
    (0..case.references)
        .map(|i| {
            let tensor = w
                .require(&format!("{}/source_{i}", case.name))
                .expect("source image");
            let shape = tensor.shape().to_vec();
            assert_eq!(shape.len(), 3, "source images are HWC RGB8");
            Image {
                height: shape[0] as u32,
                width: shape[1] as u32,
                pixels: host_f32(tensor).into_iter().map(|v| v as u8).collect(),
            }
        })
        .collect()
}

struct Snapshot {
    tokenizer: mlx_gen::tokenizer::TextTokenizer,
    encoder: QwenImage21TextEncoder,
    transformer: mlx_gen_qwen_image_2_1::QwenImage21Transformer,
    vae: mlx_gen_qwen_image_2_1::QwenImage21Vae,
    scheduler: mlx_gen_qwen_image_2_1::SchedulerConfig,
    drop: usize,
    vision: mlx_gen_qwen_image_2_1::VisionConfig,
}

fn snapshot() -> Snapshot {
    let root = tiny_snapshot();
    let tokenizer = load_tokenizer(&root).unwrap();
    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    let vision = load_vision_config(&root)
        .unwrap()
        .expect("the tiny snapshot declares a Qwen3-VL vision tower");
    Snapshot {
        encoder: load_text_encoder(&root).unwrap(),
        transformer: load_transformer(&root).unwrap(),
        vae: load_vae(&root).unwrap(),
        scheduler: load_scheduler_config(&root).unwrap(),
        tokenizer,
        drop,
        vision,
    }
}

fn i32_host(a: &Array) -> Vec<i32> {
    let n: i32 = a.shape().iter().product();
    a.as_dtype(Dtype::Int32)
        .unwrap()
        .reshape(&[n])
        .unwrap()
        .as_slice::<i32>()
        .to_vec()
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
    let w = fixture("qwen21_edit.safetensors");
    let snap = snapshot();
    for name in ["ref1", "ref2", "annotated", "mask_ref"] {
        let c = case(&w, name);
        let prepared = prepare_references(&sources(&w, &c), &snap.vision).unwrap();
        assert_eq!(prepared.len(), c.references);
        for (i, reference) in prepared.iter().enumerate() {
            let grid = i32_host(w.require(&format!("{name}/grid_thw_{i}")).unwrap());
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
            assert_close(
                &format!("{name}/pixel_values_{i}"),
                &reference.pixel_values,
                w.require(&format!("{name}/pixel_values_{i}")).unwrap(),
                HOST_TOL,
            );
            assert_close(
                &format!("{name}/vae_input_{i}"),
                &reference.vae_input,
                w.require(&format!("{name}/vae_input_{i}")).unwrap(),
                HOST_TOL,
            );
        }
    }
}

#[test]
fn conditioning_matches_upstream_for_one_two_and_ten_references() {
    let w = fixture("qwen21_edit.safetensors");
    let snap = snapshot();
    for name in ["ref1", "ref2", "ref10", "annotated", "mask_ref"] {
        let c = case(&w, name);
        let prepared = prepare_references(&sources(&w, &c), &snap.vision).unwrap();
        let cond = conditioning(&snap, &c.prompt, &prepared);
        let want_mask = i32_host(w.require(&format!("{name}/image_pad_mask")).unwrap());
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
            w.require(&format!("{name}/prompt_embeds")).unwrap(),
            TEXT_TOL,
        );
    }
}

#[test]
fn reference_latents_match_upstream_in_order() {
    let w = fixture("qwen21_edit.safetensors");
    let snap = snapshot();
    for name in ["ref1", "ref2", "annotated", "mask_ref"] {
        let c = case(&w, name);
        let prepared = prepare_references(&sources(&w, &c), &snap.vision).unwrap();
        let latents = encode_references(&snap.vae, &prepared).unwrap();
        assert_eq!(latents.len(), c.references);
        for (i, got) in latents.iter().enumerate() {
            assert_close(
                &format!("{name}/ref_latents_{i}"),
                got,
                w.require(&format!("{name}/ref_latents_{i}")).unwrap(),
                LATENT_TOL,
            );
        }
    }
}

/// Run the whole conditioned denoise for one case from upstream's own initial noise.
fn run_case(snap: &Snapshot, w: &Weights, c: &Case) -> Array {
    let prepared = prepare_references(&sources(w, c), &snap.vision).unwrap();
    let pos = conditioning(snap, &c.prompt, &prepared);
    let neg = c
        .negative
        .as_deref()
        .filter(|_| c.true_cfg > 1.0)
        .map(|n| conditioning(snap, n, &prepared));

    let pos_layout = joint_layout(&pos.image_pad_mask, &prepared, c.width, c.height).unwrap();
    let neg_layout = neg
        .as_ref()
        .map(|n| joint_layout(&n.image_pad_mask, &prepared, c.width, c.height).unwrap());
    let pos_text = mlx_gen_qwen_image_2_1::text_rows(&pos.hidden, &pos.image_pad_mask).unwrap();
    let neg_text = neg
        .as_ref()
        .map(|n| mlx_gen_qwen_image_2_1::text_rows(&n.hidden, &n.image_pad_mask).unwrap());
    let reference_latents = encode_references(&snap.vae, &prepared).unwrap();

    let sigmas = scheduler::sigmas_for_image(&snap.scheduler, c.steps, c.width, c.height).unwrap();
    let init = w
        .require(&format!("{}/latents_init", c.name))
        .unwrap()
        .clone();
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
    let w = fixture("qwen21_edit.safetensors");
    let snap = snapshot();
    for name in ["ref1", "ref2", "ref10", "annotated", "mask_ref", "ref2_cfg"] {
        let c = case(&w, name);
        let got = run_case(&snap, &w, &c);
        assert_close(
            &format!("{name}/latents_final"),
            &got,
            w.require(&format!("{name}/latents_final")).unwrap(),
            LATENT_TOL,
        );
    }
}

#[test]
fn swapping_two_references_changes_the_output() {
    let w = fixture("qwen21_edit.safetensors");
    let snap = snapshot();
    let straight = run_case(&snap, &w, &case(&w, "ref2"));
    let swapped = run_case(&snap, &w, &case(&w, "ref2_swap"));

    // Both orders reproduce upstream…
    assert_close(
        "ref2_swap/latents_final",
        &swapped,
        w.require("ref2_swap/latents_final").unwrap(),
        LATENT_TOL,
    );
    // …and they are genuinely different renders. The claim is made against **upstream's own**
    // gap between the two orders rather than against an invented threshold: upstream separates
    // the two orders by `want_gap`, and the port has to separate them by the same amount. A port
    // that ignored ordering would score a gap of ~0 against a large `want_gap`.
    let gap = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let (got_a, got_b) = (host_f32(&straight), host_f32(&swapped));
    let want_a = host_f32(w.require("ref2/latents_final").unwrap());
    let want_b = host_f32(w.require("ref2_swap/latents_final").unwrap());
    let got_gap = gap(&got_a, &got_b);
    let want_gap = gap(&want_a, &want_b);
    let parity = gap(&got_a, &want_a).max(gap(&got_b, &want_b));
    let peak = want_a.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!(
        "reference order: got max|Δ|={got_gap:.3e} upstream max|Δ|={want_gap:.3e} \
         parity residual={parity:.3e} peak={peak:.3e}"
    );
    // The ordering signal has to sit well clear of this port's own numerical floor, otherwise the
    // comparison below would be measuring noise rather than the ordering.
    assert!(
        want_gap > 3.0 * parity,
        "upstream separates the two reference orders by only {want_gap:.3e}, within 3x this \
         port's parity residual {parity:.3e}; the ordering claim cannot be tested against it"
    );
    assert!(
        (got_gap - want_gap).abs() <= LATENT_TOL * peak.max(1.0),
        "the port separates the two reference orders by {got_gap:.3e} where upstream separates \
         them by {want_gap:.3e}; reference order is not reaching the denoiser the same way"
    );

    // The exact, noise-free half of the claim: the condition latents are carried in request
    // order, so swapping the request swaps the latent blocks position for position.
    let snapshot_refs = |name: &'static str| {
        let c = case(&w, name);
        encode_references(
            &snap.vae,
            &prepare_references(&sources(&w, &c), &snap.vision).unwrap(),
        )
        .unwrap()
    };
    let (a, b) = (snapshot_refs("ref2"), snapshot_refs("ref2_swap"));
    assert_eq!(host_f32(&a[0]), host_f32(&b[1]));
    assert_eq!(host_f32(&a[1]), host_f32(&b[0]));
    assert_ne!(host_f32(&a[0]), host_f32(&a[1]));
}

#[test]
fn the_boundary_holds_and_eleven_references_are_refused() {
    let w = fixture("qwen21_edit.safetensors");
    let snap = snapshot();
    let limits: serde_json::Value =
        serde_json::from_str(w.metadata("_limits").expect("_limits metadata")).unwrap();
    assert_eq!(limits["max_reference_images"].as_u64(), Some(10));

    let ten = sources(&w, &case(&w, "ref10"));
    assert_eq!(ten.len(), 10);
    assert!(
        prepare_references(&ten, &snap.vision).is_ok(),
        "ten references is the documented boundary and must be accepted"
    );

    let mut eleven = ten.clone();
    eleven.push(eleven[0].clone());
    let err = prepare_references(&eleven, &snap.vision)
        .unwrap_err()
        .to_string();
    assert!(err.contains("at most 10"), "{err}");

    // …and the same refusal through the request seam a caller actually uses.
    let mut req = mlx_gen::GenerationRequest {
        prompt: "a red fox".into(),
        width: 64,
        height: 64,
        ..Default::default()
    };
    req.conditioning = vec![Conditioning::MultiReference { images: eleven }];
    let err = mlx_gen_qwen_image_2_1::collect_references(&req)
        .unwrap_err()
        .to_string();
    assert!(err.contains("at most 10"), "{err}");

    // Zero references on a request that carries conditioning is a refusal, not a silent T2I.
    req.conditioning = vec![Conditioning::MultiReference { images: vec![] }];
    let err = mlx_gen_qwen_image_2_1::collect_references(&req)
        .unwrap_err()
        .to_string();
    assert!(err.contains("empty MultiReference"), "{err}");
}

#[test]
fn a_mask_conditioning_is_refused_with_the_upstream_workaround() {
    let w = fixture("qwen21_edit.safetensors");
    let limits: serde_json::Value =
        serde_json::from_str(w.metadata("_limits").expect("_limits metadata")).unwrap();
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

    let mut req = mlx_gen::GenerationRequest {
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
    let err = mlx_gen_qwen_image_2_1::collect_references(&req)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no mask input"), "{err}");
    assert!(err.contains("extra reference"), "{err}");

    // The supported spelling — the mask as the second ordered reference — is accepted.
    req.conditioning = vec![Conditioning::MultiReference { images }];
    assert_eq!(
        mlx_gen_qwen_image_2_1::collect_references(&req)
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn the_text_to_image_path_is_unchanged_by_the_reference_route() {
    // With no references the conditioned entry point must be the plain 1-D-RoPE text path,
    // bit-for-bit: the interleaved M-RoPE collapses when all three position rows are the token
    // index, and nothing else engages.
    let snap = snapshot();
    let plain = snap
        .encoder
        .encode_prompt(&snap.tokenizer, "a red fox in the forest", snap.drop)
        .unwrap();
    let cond = conditioning(&snap, "a red fox in the forest", &[]);
    assert!(cond.image_pad_mask.iter().all(|m| !*m));
    let (a, b) = (host_f32(&plain), host_f32(&cond.hidden));
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter().zip(&b).all(|(x, y)| x == y),
        "the reference route changed the text-to-image conditioning"
    );
}
