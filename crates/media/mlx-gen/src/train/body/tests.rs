//! Body-loss tests (sc-24832): reference-implementation parity on the tiny fixture, LoRA gradient
//! through the shared perceptual path, the no-person skip, and the real-weight parity harness.

use super::*;
use crate::train::perceptual::{AuxLossSchedule, PerceptualPath};
use crate::train::tae::{synthetic_tiny_decoder_weights, TinyDecoder, TinyDecoderConfig};
use mlx_rs::transforms::{eval, grad};
use mlx_rs::{random, Array};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../docs/migration/body-losses-reference/body_losses_tiny.safetensors"
);

fn fixture() -> Weights {
    Weights::from_file(FIXTURE).expect("the committed body-loss reference fixture")
}

fn t(w: &Weights, k: &str) -> Array {
    to_f32(w.require(k).unwrap()).unwrap()
}

/// NCHW fixture image → NHWC.
fn nhwc(w: &Weights, k: &str) -> Array {
    t(w, k).transpose_axes(&[0, 2, 3, 1]).unwrap()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let d = subtract(a, b).unwrap().abs().unwrap().max(None).unwrap();
    eval([&d]).unwrap();
    d.item::<f32>()
}

fn scalar(a: &Array) -> f32 {
    eval([a]).unwrap();
    a.item::<f32>()
}

fn close(name: &str, got: &Array, want: &Array, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    let d = max_abs_diff(got, want);
    assert!(d <= tol, "{name}: max |Δ| {d} > {tol}");
}

fn vitpose(w: &Weights) -> VitPose {
    VitPose::from_weights(w, "vitpose.w", VitPoseConfig::tiny()).unwrap()
}

fn hybrik(w: &Weights) -> HybrikEncoder {
    HybrikEncoder::from_weights(w, "hybrik.w", HybrikConfig::tiny()).unwrap()
}

fn sapiens(w: &Weights) -> SapiensNormal {
    SapiensNormal::from_weights(w, "sapiens.w", SapiensConfig::tiny()).unwrap()
}

/// The fixture's ViTPose with every keypoint peak shifted by `bias` (a large positive shift makes
/// every keypoint confident — a detected person; a large negative one makes none confident).
fn vitpose_with_peak(bias: f32) -> VitPose {
    let mut w = fixture();
    let b = t(&w, "vitpose.w.head.conv.bias");
    w.insert(
        "vitpose.w.head.conv.bias",
        add(&b, Array::from_f32(bias)).unwrap(),
    );
    vitpose(&w)
}

// ---------------------------------------------------------------------------------------------
// AC2: reference-implementation parity (tiny models, committed fixture).
// ---------------------------------------------------------------------------------------------

/// ViTPose: the HF `VitPoseForPoseEstimation` heatmaps of upstream's warped input, the integral
/// keypoints, the ratios with and without the reference substitution, and the proportion loss.
/// Mutations: drop `+ pos[:, :1]` in `VitPose::from_weights`; swap `(x, y)` in
/// `heatmaps_to_keypoints`; skip `substitute_low_confidence` in the loss — each ⇒ red.
#[test]
fn vitpose_matches_the_reference_implementation() {
    let w = fixture();
    let m = vitpose(&w);
    let a = nhwc(&w, "input.a");
    let (hm, _) = m.forward_pixels(&a).unwrap();
    let hm_nchw = hm.transpose_axes(&[0, 3, 1, 2]).unwrap();
    close("heatmaps", &hm_nchw, &t(&w, "vitpose.out.heatmaps_a"), 2e-4);
    let (coords, conf) = vitpose::heatmaps_to_keypoints(&hm).unwrap();
    close("coords", &coords, &t(&w, "vitpose.out.coords_a"), 1e-4);
    close("confidence", &conf, &t(&w, "vitpose.out.confidence_a"), 2e-4);
    let (ra, va) = body_ratios(&coords, &conf, true).unwrap();
    close("ratios_a", &ra, &t(&w, "vitpose.out.ratios_a"), 1e-4);
    close("ratio_vis_a", &va, &t(&w, "vitpose.out.ratio_vis_a"), 2e-4);
    let loss = BodyProportionLoss::new(Rc::new(m), true);
    let reference = ProportionReference {
        ratios: ra,
        ratio_vis: va,
    };
    let b = nhwc(&w, "input.b");
    let got = loss.loss(&b, &reference).unwrap();
    close("proportion loss", &got, &t(&w, "vitpose.out.loss").reshape(&[]).unwrap(), 1e-4);
    // The live ratios after substitution.
    let (hb, _) = loss.pose.forward_pixels(&b).unwrap();
    let (cb, fb) = vitpose::heatmaps_to_keypoints(&hb).unwrap();
    let (rb, vb) = body_ratios(&cb, &fb, true).unwrap();
    let (rb, vb) = substitute_low_confidence(&rb, &vb, &reference.ratios).unwrap();
    close("ratios_b", &rb, &t(&w, "vitpose.out.ratios_b"), 1e-4);
    close("ratio_vis_b", &vb, &t(&w, "vitpose.out.ratio_vis_b"), 2e-4);
}

/// HybrIK: upstream's square person crop + ResNet + beta head, and the shape comparison.
/// Mutations: use `align_corners=True` in `forward_crop`'s resize; drop `+ init_shape` ⇒ red.
#[test]
fn hybrik_matches_the_reference_implementation() {
    let w = fixture();
    let m = hybrik(&w);
    let bbox_v = t(&w, "input.person_bbox");
    eval([&bbox_v]).unwrap();
    let b = bbox_v.as_slice::<f32>();
    let a = nhwc(&w, "input.a");
    let crop = HybrikEncoder::crop_for([b[0], b[1], b[2], b[3]], 40, 30);
    let ba = m.forward_crop(&a, crop).unwrap();
    close("betas_a", &ba, &t(&w, "hybrik.out.betas_a"), 1e-4);
    let bb = m.forward_crop(&nhwc(&w, "input.b"), crop).unwrap();
    close("betas_b", &bb, &t(&w, "hybrik.out.betas_b"), 1e-4);
    let l1 = t(&w, "hybrik.out.l1").reshape(&[]).unwrap();
    close("shape loss", &shape_comparison(&ba, &bb, -1.0).unwrap(), &l1, 1e-5);
    // The cosine gate: just above the pair's cosine ⇒ the loss is zero.
    let cos = scalar(&t(&w, "hybrik.out.cos"));
    assert!(cos > 0.5 && cos < 0.99, "fixture cosine {cos}");
    assert_eq!(scalar(&shape_comparison(&ba, &bb, cos + 1e-3).unwrap()), 0.0);
    assert!(scalar(&shape_comparison(&ba, &bb, cos - 1e-3).unwrap()) > 0.0);
}

/// Sapiens: the letterbox, the ViT + deconv head, the resampled unit normals, the letterboxed
/// mask and both normal losses. Mutations: drop the pos-embedding resample (add the stored grid
/// unresampled ⇒ shape error / mismatch); drop an InstanceNorm ⇒ red.
#[test]
fn sapiens_matches_the_reference_implementation() {
    let w = fixture();
    let m = sapiens(&w);
    let na = m.forward_pixels(&nhwc(&w, "input.a")).unwrap();
    let nb = m.forward_pixels(&nhwc(&w, "input.b")).unwrap();
    let want_a = t(&w, "sapiens.out.normals_a").transpose_axes(&[0, 2, 3, 1]).unwrap();
    close("normals_a", &na, &want_a, 2e-4);
    let want_b = t(&w, "sapiens.out.normals_b").transpose_axes(&[0, 2, 3, 1]).unwrap();
    close("normals_b", &nb, &want_b, 2e-4);
    let mask = t(&w, "input.mask").reshape(&[40, 30]).unwrap();
    let grid = m.mask_to_normal_grid(&mask).unwrap();
    close("mask", &grid, &t(&w, "sapiens.out.mask").reshape(&[16, 16]).unwrap(), 1e-5);
    let l = sapiens::normal_comparison(&na, &nb, None).unwrap();
    close("normal loss", &l, &t(&w, "sapiens.out.loss").reshape(&[]).unwrap(), 1e-4);
    let lm = sapiens::normal_comparison(&na, &nb, Some(&grid)).unwrap();
    close("masked normal loss", &lm, &t(&w, "sapiens.out.loss_masked").reshape(&[]).unwrap(), 1e-4);
}

// ---------------------------------------------------------------------------------------------
// AC1: gradient into a LoRA through the shared path; zero on a no-person reference.
// ---------------------------------------------------------------------------------------------

/// A 4-channel TAESD-shaped decoder (upscale 8): latent `[1, 4, 4, 3]` → pixels `[1, 32, 24, 3]`.
fn tiny_decoder() -> TinyDecoder {
    let cfg = TinyDecoderConfig {
        latent_channels: 4,
        channels: 8,
        blocks: [1, 1, 1, 1],
    };
    TinyDecoder::from_weights(&synthetic_tiny_decoder_weights(&cfg, 7).unwrap(), cfg).unwrap()
}

fn normal(seed: u64, shape: &[i32], std: f32) -> Array {
    multiply(
        random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap())).unwrap(),
        Array::from_f32(std),
    )
    .unwrap()
}

/// The "trainer": a frozen 1×1 channel mix with a rank-2 LoRA (`W + B·A`) applied to a latent.
struct LoraModule {
    base: Array,
    a: Array,
}

impl LoraModule {
    fn new() -> Self {
        Self {
            base: add(
                normal(11, &[4, 4], 0.1),
                Array::from_slice(&[1.0f32, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.], &[4, 4]),
            )
            .unwrap(),
            a: normal(12, &[2, 4], 0.5),
        }
    }

    /// `x0 = (W + B·A) ⋅ latent` per pixel, NCHW.
    fn x0(&self, b: &Array, latent: &Array) -> Result<Array> {
        let w = add(&self.base, &matmul(b, &self.a)?)?;
        let flat = latent.reshape(&[4, -1])?;
        Ok(matmul(&w, &flat)?.reshape(latent.shape())?)
    }
}

fn path_with(loss: Box<dyn PerceptualLoss>) -> PerceptualPath {
    PerceptualPath::new(
        Some(Box::new(tiny_decoder())),
        vec![AuxLoss {
            schedule: AuxLossSchedule {
                weight: 0.5,
                t_min: 0.0,
                t_max: 1.0,
                every_n: 1,
            },
            loss,
        }],
    )
    .unwrap()
}

fn latent() -> Array {
    normal(21, &[1, 4, 4, 3], 1.0)
}

/// Run the step through the path and return `|∂ aux / ∂ B|₁` for the LoRA up-projection `B`
/// (`None` when the plan has no aux term).
fn lora_grad(path: &mut PerceptualPath, mask: Option<&Array>) -> Option<f32> {
    let lora = LoraModule::new();
    let clean = latent();
    let b0 = normal(13, &[4, 2], 0.3);
    let x_clean = lora.x0(&b0, &clean).unwrap();
    path.ensure_reference_with_mask(0, &x_clean, mask).unwrap();
    let plan = path.plan(0, 0, 0.5).unwrap();
    path.aux_loss(&plan, 0, &x_clean).unwrap()?;
    // The live x0: the same latent nudged (as a noisy step's prediction would be).
    let live = add(&clean, &normal(22, &[1, 4, 4, 3], 0.2)).unwrap();
    let f = |b: &Array| -> mlx_rs::error::Result<Array> {
        let x0 = lora
            .x0(b, &live)
            .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))?;
        Ok(path
            .aux_loss(&plan, 0, &x0)
            .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))?
            .expect("aux planned")
            .weighted)
    };
    let g = grad(f)(&b0).unwrap();
    Some(scalar(&g.abs().unwrap().sum(None).unwrap()))
}

fn person_pose() -> Rc<VitPose> {
    Rc::new(vitpose_with_peak(50.0))
}

fn nobody_pose() -> Rc<VitPose> {
    Rc::new(vitpose_with_peak(-50.0))
}

fn full_mask() -> Array {
    let mut v = vec![0.0f32; 32 * 24];
    for y in 4..28 {
        for x in 4..20 {
            v[y * 24 + x] = 1.0;
        }
    }
    Array::from_slice(&v, &[32, 24])
}

/// AC1, per loss: with a detected person each loss back-propagates a nonzero, finite gradient
/// through the decoder into the LoRA factor. Mutation: `mlx_rs::stop_gradient` the live
/// heatmaps / betas / normals inside the loss ⇒ zero gradient ⇒ red.
#[test]
fn each_body_loss_trains_the_lora_through_the_shared_path() {
    let w = fixture();
    let losses: Vec<(&str, Box<dyn PerceptualLoss>, Option<Array>)> = vec![
        (
            "proportion",
            Box::new(BodyProportionLoss::new(person_pose(), false)),
            None,
        ),
        (
            "shape",
            Box::new(BodyShapeLoss::new(person_pose(), hybrik(&w), -1.0)),
            None,
        ),
        (
            "normal",
            Box::new(NormalLoss::new(person_pose(), sapiens(&w), false)),
            None,
        ),
        (
            "normal (subject)",
            Box::new(NormalLoss::new(person_pose(), sapiens(&w), true)),
            Some(full_mask()),
        ),
    ];
    for (name, loss, mask) in losses {
        let mut path = path_with(loss);
        let g = lora_grad(&mut path, mask.as_ref()).unwrap_or_else(|| panic!("{name}: no aux term"));
        assert!(g > 0.0 && g.is_finite(), "{name}: LoRA gradient {g}");
    }
}

/// AC1, per loss: a reference with no detected person makes the loss unusable for that image —
/// `reference()` is `None`, the plan drops the aux term (the step trains diffusion instead) and
/// the aux contribution is absent, i.e. zero. Mutation: return a reference regardless of
/// `detect` in any loss ⇒ red.
#[test]
fn each_body_loss_is_zero_without_a_detected_person() {
    let w = fixture();
    let losses: Vec<(&str, Box<dyn PerceptualLoss>)> = vec![
        ("proportion", Box::new(BodyProportionLoss::new(nobody_pose(), true))),
        ("shape", Box::new(BodyShapeLoss::new(nobody_pose(), hybrik(&w), 0.2))),
        ("normal", Box::new(NormalLoss::new(nobody_pose(), sapiens(&w), true))),
    ];
    for (name, loss) in losses {
        let mut path = path_with(loss);
        assert_eq!(lora_grad(&mut path, Some(&full_mask())), None, "{name}");
        assert!(!path.is_usable(0, 0).unwrap(), "{name}");
        let plan = path.plan(0, 0, 0.5).unwrap();
        assert!(plan.diffusion && plan.aux.is_empty(), "{name}: {plan:?}");
    }
}

/// The restricted normal loss refuses a reference without a mask instead of silently averaging
/// over the whole frame, and the mask changes the loss. Mutation: ignore `r.mask` in
/// `NormalLoss::loss` ⇒ the two losses match ⇒ red.
#[test]
fn the_subject_restricted_normal_loss_needs_and_uses_the_mask() {
    let w = fixture();
    let restricted = NormalLoss::new(person_pose(), sapiens(&w), true);
    let clean = tiny_decoder().decode(&latent()).unwrap();
    assert!(restricted.reference(&clean).is_err());
    let live = tiny_decoder()
        .decode(&add(&latent(), &normal(5, &[1, 4, 4, 3], 0.5)).unwrap())
        .unwrap();
    let r_masked = restricted
        .reference_with_mask(&clean, Some(&full_mask()))
        .unwrap()
        .unwrap();
    let plain = NormalLoss::new(person_pose(), sapiens(&w), false);
    let r_plain = plain.reference(&clean).unwrap().unwrap();
    let a = scalar(&restricted.loss(&live, r_masked.as_ref()).unwrap());
    let b = scalar(&plain.loss(&live, r_plain.as_ref()).unwrap());
    assert!((a - b).abs() > 1e-5, "masked {a} vs full {b}");
}

/// E7: the MLX footprints mirror the backend-neutral figures. Mutation: drop ViTPose from
/// `gen_core::train::body::body_loss_footprints` for the normal-only case ⇒ red.
#[test]
fn footprints_count_the_detector_and_each_enabled_model() {
    let mut cfg = BodyLossesConfig::default();
    assert!(body_loss_footprints(&cfg).is_empty());
    assert!(body_aux_losses(&cfg).unwrap().is_empty(), "off ⇒ nothing loads");
    cfg.normal.weight = 0.1;
    let f = body_loss_footprints(&cfg);
    assert_eq!(f.len(), 2);
    assert_eq!(
        f[0].param_bytes,
        VitPoseConfig::plus_base().param_count() * 4
    );
}

// ---------------------------------------------------------------------------------------------
// AC2 at real scale: the real checkpoints vs the reference implementation's outputs.
// ---------------------------------------------------------------------------------------------

/// Real-weight parity — runs only where the three checkpoints and the reference outputs exist
/// (S9's real-weight phase): `SCENEWORKS_BODY_LOSS_REAL` names a directory holding `vitpose/`,
/// `hybrik/`, `sapiens/` (each a `model.safetensors` snapshot, as the catalog installs them) and
/// `reference.safetensors` from `scripts/reference/body_losses_reference.py --real`. Fails —
/// never passes vacuously — when run without them.
#[test]
#[ignore = "needs the real ViTPose+/HybrIK/Sapiens checkpoints + --real reference outputs \
            (SCENEWORKS_BODY_LOSS_REAL); never downloaded in ordinary test runs"]
fn real_checkpoints_match_the_reference_implementation() {
    let root = std::path::PathBuf::from(
        std::env::var("SCENEWORKS_BODY_LOSS_REAL")
            .expect("SCENEWORKS_BODY_LOSS_REAL must name the real-weight directory"),
    );
    let r = Weights::from_file(root.join("reference.safetensors")).unwrap();
    let img = nhwc(&r, "input.a");
    let pose = VitPose::from_dir(root.join("vitpose"), VitPoseConfig::plus_base()).unwrap();
    let (hm, _) = pose.forward_pixels(&img).unwrap();
    let (coords, conf) = vitpose::heatmaps_to_keypoints(&hm).unwrap();
    let (ratios, vis) = body_ratios(&coords, &conf, true).unwrap();
    close("real heatmaps", &hm.transpose_axes(&[0, 3, 1, 2]).unwrap(), &t(&r, "vitpose.out.heatmaps_a"), 2e-3);
    close("real ratios", &ratios, &t(&r, "vitpose.out.ratios_a"), 1e-3);
    close("real ratio vis", &vis, &t(&r, "vitpose.out.ratio_vis_a"), 2e-3);
    let bbox_v = t(&r, "input.person_bbox");
    eval([&bbox_v]).unwrap();
    let b = bbox_v.as_slice::<f32>();
    let sh = img.shape();
    let crop = HybrikEncoder::crop_for([b[0], b[1], b[2], b[3]], sh[1] as usize, sh[2] as usize);
    let hyb = HybrikEncoder::from_dir(root.join("hybrik"), HybrikConfig::resnet34()).unwrap();
    close("real betas", &hyb.forward_crop(&img, crop).unwrap(), &t(&r, "hybrik.out.betas_a"), 1e-3);
    let sap = SapiensNormal::from_dir(root.join("sapiens"), SapiensConfig::normal_0_3b()).unwrap();
    let want = t(&r, "sapiens.out.normals_a").transpose_axes(&[0, 2, 3, 1]).unwrap();
    close("real normals", &sap.forward_pixels(&img).unwrap(), &want, 5e-3);
}
