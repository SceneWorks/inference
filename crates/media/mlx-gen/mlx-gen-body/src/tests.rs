//! Body-loss tests (sc-24832): reference-implementation parity on the tiny fixture, LoRA gradient
//! through the shared perceptual path, the no-person skip, and the real-weight parity harness.

use super::*;
use mlx_gen::train::perceptual::X0Decoder;
use mlx_gen::train::perceptual::{AuxLoss, AuxLossSchedule, PerceptualPath};
use mlx_rs::ops::matmul;
use mlx_rs::transforms::{eval, grad};
use mlx_rs::{random, Array};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../docs/migration/body-losses-reference/body_losses_tiny.safetensors"
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

thread_local! {
    /// Tolerance multiplier of the parity checks: `1` on the CPU stream (exact f32 GEMMs), larger
    /// on the default GPU stream, whose f32 GEMMs run on the matrix unit at reduced (TF32-class)
    /// precision.
    static TOL_SCALE: std::cell::Cell<f32> = const { std::cell::Cell::new(1.0) };
}

fn close(name: &str, got: &Array, want: &Array, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    let tol = tol * TOL_SCALE.with(|c| c.get());
    let d = max_abs_diff(got, want);
    assert!(d <= tol, "{name}: max |Δ| {d} > {tol}");
}

/// Run a parity check on the CPU stream at the stated tolerances.
fn on_cpu(f: impl FnOnce()) {
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), f);
}

/// Run a parity check on the default (GPU) stream at `scale ×` the stated tolerances.
fn on_gpu(scale: f32, f: impl FnOnce()) {
    TOL_SCALE.with(|c| c.set(scale));
    f();
    TOL_SCALE.with(|c| c.set(1.0));
}

/// The GPU stream's tolerance multiplier (TF32-class f32 GEMMs, ~1e-3 relative).
const GPU_TOL_SCALE: f32 = 100.0;

fn vitpose(w: &Weights) -> VitPose {
    VitPose::from_weights(w, "vitpose.w", VitPoseConfig::tiny()).unwrap()
}

fn hybrik(w: &Weights) -> HybrikEncoder {
    HybrikEncoder::from_weights(w, "hybrik.w", HybrikConfig::tiny()).unwrap()
}

fn sapiens(w: &Weights) -> SapiensNormal {
    SapiensNormal::from_weights(w, "sapiens.w", SapiensConfig::tiny()).unwrap()
}

/// The fixture's ViTPose with its keypoint head rescaled: heatmaps `gain · conv + bias + shift`.
fn vitpose_with_head(gain: f32, shift: f32) -> VitPose {
    let mut w = fixture();
    let k = "vitpose.w.head.conv.weight";
    let cw = t(&w, k);
    w.insert(k, multiply(&cw, Array::from_f32(gain)).unwrap());
    let k = "vitpose.w.head.conv.bias";
    let b = t(&w, k);
    w.insert(k, add(&b, Array::from_f32(shift)).unwrap());
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
fn vitpose_matches_the_reference_implementation_on_cpu() {
    on_cpu(vitpose_parity);
}

#[test]
fn vitpose_matches_the_reference_implementation_on_gpu() {
    on_gpu(GPU_TOL_SCALE, vitpose_parity);
}

fn vitpose_parity() {
    let w = fixture();
    let m = vitpose(&w);
    let a = nhwc(&w, "input.a");
    let (hm, _) = m.forward_pixels(&a).unwrap();
    let hm_nchw = hm.transpose_axes(&[0, 3, 1, 2]).unwrap();
    close("heatmaps", &hm_nchw, &t(&w, "vitpose.out.heatmaps_a"), 2e-4);
    let (coords, conf) = vitpose::heatmaps_to_keypoints(&hm).unwrap();
    close("coords", &coords, &t(&w, "vitpose.out.coords_a"), 1e-4);
    close(
        "confidence",
        &conf,
        &t(&w, "vitpose.out.confidence_a"),
        2e-4,
    );
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
    close(
        "proportion loss",
        &got,
        &t(&w, "vitpose.out.loss").reshape(&[]).unwrap(),
        1e-4,
    );
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
fn hybrik_matches_the_reference_implementation_on_cpu() {
    on_cpu(hybrik_parity);
}

#[test]
fn hybrik_matches_the_reference_implementation_on_gpu() {
    on_gpu(GPU_TOL_SCALE, hybrik_parity);
}

fn hybrik_parity() {
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
    close(
        "shape loss",
        &shape_comparison(&ba, &bb, -1.0).unwrap(),
        &l1,
        1e-5,
    );
    // The cosine gate: just above the pair's cosine ⇒ the loss is zero.
    let cos = scalar(&t(&w, "hybrik.out.cos"));
    assert!(cos > 0.5 && cos < 0.99, "fixture cosine {cos}");
    assert_eq!(
        scalar(&shape_comparison(&ba, &bb, cos + 1e-3).unwrap()),
        0.0
    );
    assert!(scalar(&shape_comparison(&ba, &bb, cos - 1e-3).unwrap()) > 0.0);
    // The reference path (upstream `encode`): the `int()`-truncated crop, which differs from the
    // rounded live crop on this box. Mutation: round in `hybrik_encode_crop` ⇒ red.
    let enc = HybrikEncoder::encode_crop_for([b[0], b[1], b[2], b[3]], 40, 30);
    assert_ne!(enc, crop);
    let br = m.forward_crop(&a, enc).unwrap();
    close(
        "betas_a_encode",
        &br,
        &t(&w, "hybrik.out.betas_a_encode"),
        1e-4,
    );
    close(
        "encode shape loss",
        &shape_comparison(&br, &bb, -1.0).unwrap(),
        &t(&w, "hybrik.out.l1_encode").reshape(&[]).unwrap(),
        1e-5,
    );
}

/// Sapiens: the letterbox, the ViT + deconv head, the resampled unit normals, the letterboxed
/// mask and both normal losses. Mutations: drop the pos-embedding resample (add the stored grid
/// unresampled ⇒ shape error / mismatch); drop an InstanceNorm ⇒ red.
#[test]
fn sapiens_matches_the_reference_implementation_on_cpu() {
    on_cpu(sapiens_parity);
}

#[test]
fn sapiens_matches_the_reference_implementation_on_gpu() {
    on_gpu(GPU_TOL_SCALE, sapiens_parity);
}

fn sapiens_parity() {
    let w = fixture();
    let m = sapiens(&w);
    let na = m.forward_pixels(&nhwc(&w, "input.a")).unwrap();
    let nb = m.forward_pixels(&nhwc(&w, "input.b")).unwrap();
    let want_a = t(&w, "sapiens.out.normals_a")
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    close("normals_a", &na, &want_a, 2e-4);
    let want_b = t(&w, "sapiens.out.normals_b")
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    close("normals_b", &nb, &want_b, 2e-4);
    let mask = t(&w, "input.mask").reshape(&[40, 30]).unwrap();
    let grid = m.mask_to_normal_grid(&mask).unwrap();
    close(
        "mask",
        &grid,
        &t(&w, "sapiens.out.mask").reshape(&[16, 16]).unwrap(),
        1e-5,
    );
    let l = sapiens::normal_comparison(&na, &nb, None).unwrap();
    close(
        "normal loss",
        &l,
        &t(&w, "sapiens.out.loss").reshape(&[]).unwrap(),
        1e-4,
    );
    let lm = sapiens::normal_comparison(&na, &nb, Some(&grid)).unwrap();
    close(
        "masked normal loss",
        &lm,
        &t(&w, "sapiens.out.loss_masked").reshape(&[]).unwrap(),
        1e-4,
    );
    // The reference path (upstream `encode`): the native-size letterbox, resampled to the same
    // grid. Mutation: letterbox `encode_pixels` at the training size ⇒ red.
    let nr = m.encode_pixels(&nhwc(&w, "input.a")).unwrap();
    let want_r = t(&w, "sapiens.out.normals_a_encode")
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    close("normals_a_encode", &nr, &want_r, 2e-4);
    assert!(max_abs_diff(&nr, &na) > 1e-2, "the two letterboxes differ");
    close(
        "encode normal loss",
        &sapiens::normal_comparison(&nr, &nb, None).unwrap(),
        &t(&w, "sapiens.out.loss_encode").reshape(&[]).unwrap(),
        1e-4,
    );
    close(
        "encode masked normal loss",
        &sapiens::normal_comparison(&nr, &nb, Some(&grid)).unwrap(),
        &t(&w, "sapiens.out.loss_masked_encode")
            .reshape(&[])
            .unwrap(),
        1e-4,
    );
}

// ---------------------------------------------------------------------------------------------
// AC1: gradient into a LoRA through the shared path; zero on a no-person reference.
// ---------------------------------------------------------------------------------------------

/// A tiny differentiable x0 decoder: latent NCHW `[1, 4, h, w]` → a fixed 1×1 channel mix →
/// sigmoid → ×8 bilinear upsample → NHWC pixels `[1, 8h, 8w, 3]` in `(0, 1)`. (A random-init
/// TAESD saturates its `[0, 1]` clamp on ~99% of pixels, which zeroes the gradient the test is
/// about; the sigmoid never saturates.)
struct TinyTestDecoder {
    mix: Array,
}

impl TinyTestDecoder {
    fn new() -> Self {
        Self {
            mix: normal(31, &[3, 4], 0.8),
        }
    }
}

impl mlx_gen::train::perceptual::X0Decoder for TinyTestDecoder {
    fn decode(&self, latents: &Array) -> Result<Array> {
        let sh = latents.shape();
        let (h, w) = (sh[2] as usize, sh[3] as usize);
        let flat = latents.reshape(&[4, -1])?;
        let rgb = mlx_rs::ops::sigmoid(&matmul(&self.mix, &flat)?)?
            .reshape(&[1, 3, sh[2], sh[3]])?
            .transpose_axes(&[0, 2, 3, 1])?;
        resample_nhwc(
            &rgb,
            &AxisMatrix::resize(h, 8 * h, false),
            &AxisMatrix::resize(w, 8 * w, false),
        )
    }
}

fn tiny_decoder() -> TinyTestDecoder {
    TinyTestDecoder::new()
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
                Array::from_slice(
                    &[
                        1.0f32, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                    ],
                    &[4, 4],
                ),
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
    let live = add(&clean, normal(22, &[1, 4, 4, 3], 0.2)).unwrap();
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

/// A ViTPose that finds a person: a ×10 head gain makes peaked, confident heatmaps whose
/// keypoints move with the image (the random-init heatmaps alone are nearly flat, so every
/// keypoint lands on the centre and every bone length sits on the `1e-6` floor).
fn person_pose() -> Rc<VitPose> {
    Rc::new(vitpose_with_head(10.0, 5.0))
}

/// A ViTPose that finds nobody: every heatmap is far below zero (no confident keypoint).
fn nobody_pose() -> Rc<VitPose> {
    Rc::new(vitpose_with_head(1.0, -50.0))
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
        let g =
            lora_grad(&mut path, mask.as_ref()).unwrap_or_else(|| panic!("{name}: no aux term"));
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
        (
            "proportion",
            Box::new(BodyProportionLoss::new(nobody_pose(), true)),
        ),
        (
            "shape",
            Box::new(BodyShapeLoss::new(nobody_pose(), hybrik(&w), 0.2)),
        ),
        (
            "normal",
            Box::new(NormalLoss::new(nobody_pose(), sapiens(&w), true)),
        ),
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
        .decode(&add(latent(), normal(5, &[1, 4, 4, 3], 0.5)).unwrap())
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

/// E7: the MLX arm footprints are the backend-neutral split (ViTPose once). Mutation: drop the
/// ViTPose attribution in `body_arm_footprint` ⇒ red.
#[test]
fn arm_footprints_count_the_detector_once() {
    let mut cfg = BodyLossesConfig::default();
    cfg.normal.weight = 0.1;
    let f = arm_footprint(&cfg, BodyArm::Normal);
    assert_eq!(
        f.param_bytes,
        (VitPoseConfig::plus_base().param_count() + SapiensConfig::normal_0_3b().param_count()) * 4
    );
    assert_eq!(
        arm_footprint(&cfg, BodyArm::Proportion),
        AuxModelFootprint::default()
    );
}

/// The arm builders name the missing checkpoint instead of loading nothing.
#[test]
fn arm_builders_name_a_missing_checkpoint() {
    let mut cfg = BodyLossesConfig::default();
    cfg.shape.weight = 0.1;
    let e = shape_loss(&cfg).err().unwrap().to_string();
    assert!(e.contains("HybrIK") && e.contains("shape_model_dir"), "{e}");
    cfg.shape_model_dir = Some(std::path::PathBuf::from("/nonexistent/hybrik"));
    let e = shape_loss(&cfg).err().unwrap().to_string();
    assert!(e.contains("HybrIK"), "{e}");
    let e = proportion_loss(&cfg).err().unwrap().to_string();
    assert!(e.contains("pose_model_dir"), "{e}");
}

// ---------------------------------------------------------------------------------------------
// AC2 at real scale: the real checkpoints vs the reference implementation's outputs.
// ---------------------------------------------------------------------------------------------

/// The real-weight parity directory (`SCENEWORKS_BODY_LOSS_REAL`): `reference.safetensors` from
/// `scripts/reference/body_losses_reference.py --real` plus each model's re-hosted snapshot in
/// `vitpose/`, `hybrik/`, `sapiens/`. These tests run on the CPU stream (no long Metal job) and
/// fail — never pass vacuously — when the directory or the model's outputs are missing.
fn real_dir() -> (std::path::PathBuf, Weights) {
    // CPU end to end: the default device too, so the checkpoints' lazy loads never land on the GPU
    // (a 1.4 GB Sapiens graph on Metal is the long GPU job these tests must not be).
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let root = std::path::PathBuf::from(
        std::env::var("SCENEWORKS_BODY_LOSS_REAL")
            .expect("SCENEWORKS_BODY_LOSS_REAL must name the real-weight directory"),
    );
    let r = Weights::from_file(root.join("reference.safetensors")).unwrap();
    (root, r)
}

/// AC2 at real scale: ViTPose+ base vs HF transformers + upstream's encoder.
#[test]
#[ignore = "needs the real vitpose-plus-base snapshot + --real outputs (SCENEWORKS_BODY_LOSS_REAL)"]
fn real_vitpose_matches_the_reference_implementation() {
    on_cpu(|| {
        let (root, r) = real_dir();
        let img = nhwc(&r, "input.a");
        let pose = VitPose::from_dir(root.join("vitpose"), VitPoseConfig::plus_base()).unwrap();
        let (hm, _) = pose.forward_pixels(&img).unwrap();
        let (coords, conf) = vitpose::heatmaps_to_keypoints(&hm).unwrap();
        let (ratios, vis) = body_ratios(&coords, &conf, true).unwrap();
        let hm_nchw = hm.transpose_axes(&[0, 3, 1, 2]).unwrap();
        close(
            "real heatmaps",
            &hm_nchw,
            &t(&r, "vitpose.out.heatmaps_a"),
            2e-3,
        );
        close("real ratios", &ratios, &t(&r, "vitpose.out.ratios_a"), 1e-3);
        close(
            "real ratio vis",
            &vis,
            &t(&r, "vitpose.out.ratio_vis_a"),
            2e-3,
        );
    });
}

/// AC2 at real scale: the re-hosted HybrIK ResNet-34 vs upstream's encoder on the original `.pth`.
#[test]
#[ignore = "needs the real HybrIK snapshot + --real outputs (SCENEWORKS_BODY_LOSS_REAL)"]
fn real_hybrik_matches_the_reference_implementation() {
    on_cpu(|| {
        let (root, r) = real_dir();
        let img = nhwc(&r, "input.a");
        let bbox_v = t(&r, "input.person_bbox");
        eval([&bbox_v]).unwrap();
        let b = bbox_v.as_slice::<f32>();
        let sh = img.shape();
        let crop =
            HybrikEncoder::crop_for([b[0], b[1], b[2], b[3]], sh[1] as usize, sh[2] as usize);
        let hyb = HybrikEncoder::from_dir(root.join("hybrik"), HybrikConfig::resnet34()).unwrap();
        let got = hyb.forward_crop(&img, crop).unwrap();
        close("real betas", &got, &t(&r, "hybrik.out.betas_a"), 1e-3);
        let enc = HybrikEncoder::encode_crop_for(
            [b[0], b[1], b[2], b[3]],
            sh[1] as usize,
            sh[2] as usize,
        );
        let got = hyb.forward_crop(&img, enc).unwrap();
        close(
            "real encode betas",
            &got,
            &t(&r, "hybrik.out.betas_a_encode"),
            1e-3,
        );
    });
}

/// AC2 at real scale: the re-hosted Sapiens normal 0.3B vs upstream's estimator on the `.pth`.
#[test]
#[ignore = "needs the real Sapiens snapshot + --real outputs (SCENEWORKS_BODY_LOSS_REAL)"]
fn real_sapiens_matches_the_reference_implementation() {
    on_cpu(|| {
        let (root, r) = real_dir();
        let img = nhwc(&r, "input.a");
        let sap =
            SapiensNormal::from_dir(root.join("sapiens"), SapiensConfig::normal_0_3b()).unwrap();
        let want = t(&r, "sapiens.out.normals_a")
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap();
        close(
            "real normals",
            &sap.forward_pixels(&img).unwrap(),
            &want,
            5e-3,
        );
        let want = t(&r, "sapiens.out.normals_a_encode")
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap();
        close(
            "real encode normals",
            &sap.encode_pixels(&img).unwrap(),
            &want,
            5e-3,
        );
    });
}

/// sc-24832: the subject-restricted normal comparison ignores the normal error outside the mask —
/// zero gradient there — and keeps it inside. Mutation: ignore `mask` in `normal_comparison` ⇒ the
/// outside gradient is nonzero ⇒ red.
#[test]
fn restricted_normals_ignore_error_outside_the_mask() {
    let s = 4;
    let unit = |seed: u64| {
        let v = normal(seed, &[1, s, s, 3], 1.0);
        let n = v
            .square()
            .unwrap()
            .sum_axes(&[3], true)
            .unwrap()
            .sqrt()
            .unwrap();
        mlx_rs::ops::divide(&v, &n).unwrap()
    };
    let reference = unit(41);
    let live = unit(42);
    let mut m = vec![0.0f32; (s * s) as usize];
    for y in 0..2 {
        for x in 0..s as usize {
            m[y * s as usize + x] = 1.0; // top half is the subject
        }
    }
    let mask = Array::from_slice(&m, &[s, s]);
    let f = |l: &Array| -> mlx_rs::error::Result<Array> {
        sapiens::normal_comparison(&reference, l, Some(&mask))
            .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
    };
    let g = grad(f)(&live).unwrap();
    let inside = scalar(&g.index((.., ..2, .., ..)).abs().unwrap().sum(None).unwrap());
    let outside = scalar(&g.index((.., 2.., .., ..)).abs().unwrap().sum(None).unwrap());
    assert!(inside > 0.0, "inside {inside}");
    assert_eq!(outside, 0.0, "outside gradient must be zero");
}

/// sc-24832: every body loss carries upstream's `t_ratio` weight (`t` clamped to [0, 1]) and the
/// shared path scales its term by it: weighted = schedule weight × t × raw. Mutation: drop the
/// `timestep_weight` override of any body loss ⇒ the weight is 1 ⇒ red.
#[test]
fn each_body_loss_is_weighted_by_the_noise_level() {
    let w = fixture();
    let losses: Vec<(&str, Box<dyn PerceptualLoss>)> = vec![
        (
            "proportion",
            Box::new(BodyProportionLoss::new(person_pose(), false)),
        ),
        (
            "shape",
            Box::new(BodyShapeLoss::new(person_pose(), hybrik(&w), -1.0)),
        ),
        (
            "normal",
            Box::new(NormalLoss::new(person_pose(), sapiens(&w), false)),
        ),
    ];
    for (name, loss) in losses {
        assert_eq!(loss.timestep_weight(0.3), 0.3, "{name}");
        assert_eq!(loss.timestep_weight(1.7), 1.0, "{name}");
        assert_eq!(loss.timestep_weight(-0.2), 0.0, "{name}");
        let mut path = path_with(loss);
        let lora = LoraModule::new();
        let b0 = normal(13, &[4, 2], 0.3);
        path.ensure_reference(0, &lora.x0(&b0, &latent()).unwrap())
            .unwrap();
        let t = 0.4f32;
        let plan = path.plan(0, 0, t).unwrap();
        assert!((plan.noise_level - t).abs() < 1e-6, "{name}");
        let live = lora
            .x0(&b0, &add(latent(), normal(22, &[1, 4, 4, 3], 0.2)).unwrap())
            .unwrap();
        let terms = path
            .aux_loss(&plan, 0, &live)
            .unwrap()
            .expect("aux planned");
        let raw = scalar(&terms.per_loss[0].1);
        let weighted = scalar(&terms.weighted);
        assert!(raw > 0.0, "{name}: raw {raw}");
        assert!(
            (weighted - 0.5 * t * raw).abs() <= 1e-6 * raw.max(1.0),
            "{name}: weighted {weighted} vs 0.5·t·raw {}",
            0.5 * t * raw
        );
    }
}

/// Keypoint `i` of a synthetic pose at DSNT `[-1, 1]` coordinates — distinct, non-degenerate bones.
fn synthetic_keypoint(i: usize) -> (f32, f32) {
    (-0.8 + 0.07 * i as f32, -0.9 + 0.1 * i as f32)
}

fn detection(
    frames: &[Vec<(f32, f32)>],
    conf: &[f32],
    include_head: bool,
) -> Option<PersonDetection> {
    let b = frames.len() as i32;
    let c: Vec<f32> = frames.iter().flatten().flat_map(|&(x, y)| [x, y]).collect();
    let k: Vec<f32> = frames.iter().flat_map(|_| conf.iter().copied()).collect();
    let warp = VitPoseWarp::full_frame(64, 48, (256, 192));
    detection_from_keypoints(
        &Array::from_slice(&c, &[b, 17, 2]),
        &Array::from_slice(&k, &[b, 17]),
        &warp,
        include_head,
        64,
        48,
    )
    .unwrap()
}

/// The person box spans the confident keypoints of EVERY frame (the Candle twin's rule): frame 1
/// moves one keypoint further right, and the box follows it. Mutation: build the points from frame
/// 0 only (`0..17`) ⇒ the box stops at frame 0's extent ⇒ red.
#[test]
fn detect_box_spans_every_frame() {
    let f0: Vec<(f32, f32)> = (0..17).map(synthetic_keypoint).collect();
    let mut f1 = f0.clone();
    f1[1].0 = 0.95;
    let conf = [0.9f32; 17];
    let warp = VitPoseWarp::full_frame(64, 48, (256, 192));
    let pts = |frames: &[&Vec<(f32, f32)>]| -> Vec<(f32, f32)> {
        frames
            .iter()
            .flat_map(|f| f.iter().map(|&(x, y)| warp.keypoint_to_input(x, y)))
            .collect()
    };
    let both = keypoint_box(&pts(&[&f0, &f1]), &[0.9; 34], 64, 48).unwrap();
    let first = keypoint_box(&pts(&[&f0]), &[0.9; 17], 64, 48).unwrap();
    assert!(
        both[2] > first[2] + 1.0,
        "frame 1 extends the box: {both:?} vs {first:?}"
    );
    let got = detection(&[f0, f1], &conf, false)
        .expect("a person")
        .person_box;
    assert_eq!(got, both);
}

/// Upstream's no-person rule averages the visibility of EVERY ratio in use: body ratios at 0.12
/// pass alone (mean 0.12), but with the two head ratios at 0 the mean is 0.096 < 0.1 ⇒ no person.
/// Mutation: average only the first `NUM_BODY_RATIOS` ⇒ the head case detects a person ⇒ red.
#[test]
fn no_person_gate_averages_every_ratio_in_use() {
    let f: Vec<(f32, f32)> = (0..17).map(synthetic_keypoint).collect();
    let mut conf = [0.12f32; 17];
    for k in [0, 3, 4] {
        conf[k] = 0.0;
    }
    // Eyes (in no ratio group) are confident, so the box itself exists either way.
    conf[1] = 0.9;
    conf[2] = 0.9;
    assert!(detection(std::slice::from_ref(&f), &conf, false).is_some());
    assert!(detection(&[f], &conf, true).is_none());
}

/// The losses' reference paths follow upstream `encode`, the live paths `forward`: the shape
/// reference betas come from the `int()`-truncated crop while the stored live crop is the rounded
/// one; the normal reference is the native-letterbox `encode_pixels`. Mutations: compute the
/// reference betas on `crop_for` ⇒ red; compute the reference normals with `forward_pixels` ⇒ red.
#[test]
fn reference_paths_follow_upstream_encode() {
    on_cpu(|| {
        let w = fixture();
        let clean = nhwc(&w, "input.a");
        let bx = person_pose()
            .detect(&clean, false)
            .unwrap()
            .expect("a person")
            .person_box;
        let (crop, enc) = (
            HybrikEncoder::crop_for(bx, 40, 30),
            HybrikEncoder::encode_crop_for(bx, 40, 30),
        );
        assert_ne!(crop, enc, "the detected box tells the two crops apart");
        let shape = BodyShapeLoss::new(person_pose(), hybrik(&w), -1.0);
        let r = shape.reference(&clean).unwrap().unwrap();
        let r = r.downcast_ref::<ShapeReference>().unwrap();
        assert_eq!(r.crop, crop);
        close(
            "shape reference",
            &r.betas,
            &hybrik(&w).forward_crop(&clean, enc).unwrap(),
            1e-6,
        );
        let normal = NormalLoss::new(person_pose(), sapiens(&w), false);
        let r = normal.reference(&clean).unwrap().unwrap();
        let r = r.downcast_ref::<NormalReference>().unwrap();
        close(
            "normal reference",
            &r.normals,
            &sapiens(&w).encode_pixels(&clean).unwrap(),
            1e-6,
        );
    });
}
