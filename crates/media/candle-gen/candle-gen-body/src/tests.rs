//! Body-loss tests (sc-24832), candle: reference-implementation parity on the tiny fixture (the
//! same committed file the MLX twin checks — so the two backends are pinned to one reference and
//! to each other), LoRA gradient through the shared path, the no-person skip, the real-weight
//! parity harness.

use std::collections::HashMap;

use super::*;
use candle_gen::candle_core::Var;
use candle_gen::train::perceptual::{AuxLoss, AuxLossSchedule, PerceptualPath, X0Decoder};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../../docs/migration/body-losses-reference/body_losses_tiny.safetensors"
);

fn dev() -> Device {
    Device::Cpu
}

fn fixture() -> Weights {
    Weights::from_file(Path::new(FIXTURE), &dev(), DType::F32)
        .expect("the committed body-loss reference fixture")
}

fn t(w: &Weights, k: &str) -> Tensor {
    w.require(k).unwrap()
}

fn nhwc(w: &Weights, k: &str) -> Tensor {
    t(w, k).permute([0, 2, 3, 1]).unwrap().contiguous().unwrap()
}

fn scalar(a: &Tensor) -> f32 {
    a.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]
}

fn close(name: &str, got: &Tensor, want: &Tensor, tol: f32) {
    assert_eq!(got.dims(), want.dims(), "{name}: shape");
    let d = scalar(
        &(got - want)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap(),
    );
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

/// The fixture's ViTPose with its keypoint head rescaled: heatmaps `gain · conv + bias + shift`.
fn vitpose_with_head(gain: f64, shift: f64) -> VitPose {
    let w = fixture();
    let mut map: HashMap<String, Tensor> = w
        .keys()
        .map(|k| (k.clone(), w.require(k).unwrap()))
        .collect();
    let k = "vitpose.w.head.conv.weight".to_string();
    let cw = (map[&k].clone() * gain).unwrap();
    map.insert(k, cw);
    let k = "vitpose.w.head.conv.bias".to_string();
    let cb = (map[&k].clone() + shift).unwrap();
    map.insert(k, cb);
    vitpose(&Weights::from_map(map))
}

// ---------------------------------------------------------------------------------------------
// AC2: reference-implementation parity (tiny models, committed fixture).
// ---------------------------------------------------------------------------------------------

/// ViTPose vs HF `VitPoseForPoseEstimation` + upstream's encoder. Mutations: drop `+ pos[:, :1]`;
/// swap `(x, y)` in `heatmaps_to_keypoints`; skip `substitute_low_confidence` — each ⇒ red.
#[test]
fn vitpose_matches_the_reference_implementation() {
    let w = fixture();
    let m = vitpose(&w);
    let a = nhwc(&w, "input.a");
    let (hm, _) = m.forward_pixels(&a).unwrap();
    let hm_nchw = hm.permute([0, 3, 1, 2]).unwrap();
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
    let loss = BodyProportionLoss::new(Arc::new(m), true);
    let reference = ProportionReference {
        ratios: ra,
        ratio_vis: va,
    };
    let b = nhwc(&w, "input.b");
    let got = loss.loss(&b, &reference).unwrap();
    close(
        "proportion loss",
        &got.reshape(1).unwrap(),
        &t(&w, "vitpose.out.loss"),
        1e-4,
    );
    let (hb, _) = loss.pose.forward_pixels(&b).unwrap();
    let (cb, fb) = vitpose::heatmaps_to_keypoints(&hb).unwrap();
    let (rb, vb) = body_ratios(&cb, &fb, true).unwrap();
    let (rb, vb) = substitute_low_confidence(&rb, &vb, &reference.ratios).unwrap();
    close("ratios_b", &rb, &t(&w, "vitpose.out.ratios_b"), 1e-4);
    close("ratio_vis_b", &vb, &t(&w, "vitpose.out.ratio_vis_b"), 2e-4);
}

/// HybrIK vs upstream's encoder. Mutations: `align_corners=True` in `forward_crop`; drop
/// `+ init_shape` ⇒ red.
#[test]
fn hybrik_matches_the_reference_implementation() {
    let w = fixture();
    let m = hybrik(&w);
    let b: Vec<f32> = t(&w, "input.person_bbox").to_vec1().unwrap();
    let a = nhwc(&w, "input.a");
    let crop = HybrikEncoder::crop_for([b[0], b[1], b[2], b[3]], 40, 30);
    let ba = m.forward_crop(&a, crop).unwrap();
    close("betas_a", &ba, &t(&w, "hybrik.out.betas_a"), 1e-4);
    let bb = m.forward_crop(&nhwc(&w, "input.b"), crop).unwrap();
    close("betas_b", &bb, &t(&w, "hybrik.out.betas_b"), 1e-4);
    let l1 = t(&w, "hybrik.out.l1");
    close(
        "shape loss",
        &shape_comparison(&ba, &bb, -1.0)
            .unwrap()
            .reshape(1)
            .unwrap(),
        &l1,
        1e-5,
    );
    let cos = scalar(&t(&w, "hybrik.out.cos"));
    assert!(cos > 0.5 && cos < 0.99, "fixture cosine {cos}");
    assert_eq!(
        scalar(&shape_comparison(&ba, &bb, cos + 1e-3).unwrap()),
        0.0
    );
    assert!(scalar(&shape_comparison(&ba, &bb, cos - 1e-3).unwrap()) > 0.0);
}

/// Sapiens vs upstream's estimator. Mutations: drop an InstanceNorm; skip the letterbox pad ⇒ red.
#[test]
fn sapiens_matches_the_reference_implementation() {
    let w = fixture();
    let m = sapiens(&w);
    let na = m.forward_pixels(&nhwc(&w, "input.a")).unwrap();
    let nb = m.forward_pixels(&nhwc(&w, "input.b")).unwrap();
    close("normals_a", &na, &nhwc(&w, "sapiens.out.normals_a"), 2e-4);
    close("normals_b", &nb, &nhwc(&w, "sapiens.out.normals_b"), 2e-4);
    let mask = t(&w, "input.mask").reshape((40, 30)).unwrap();
    let grid = m.mask_to_normal_grid(&mask).unwrap();
    close(
        "mask",
        &grid,
        &t(&w, "sapiens.out.mask").reshape((16, 16)).unwrap(),
        1e-5,
    );
    let l = sapiens::normal_comparison(&na, &nb, None).unwrap();
    close(
        "normal loss",
        &l.reshape(1).unwrap(),
        &t(&w, "sapiens.out.loss"),
        1e-4,
    );
    let lm = sapiens::normal_comparison(&na, &nb, Some(&grid)).unwrap();
    close(
        "masked normal loss",
        &lm.reshape(1).unwrap(),
        &t(&w, "sapiens.out.loss_masked"),
        1e-4,
    );
}

// ---------------------------------------------------------------------------------------------
// AC1: gradient into a LoRA through the shared path; zero on a no-person reference.
// ---------------------------------------------------------------------------------------------

fn normal(seed: u64, shape: &[usize], std: f32) -> Tensor {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|_| {
            let mut s = 0f32;
            for _ in 0..4 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s += ((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
            }
            s * 3f32.sqrt() * std
        })
        .collect();
    Tensor::from_vec(v, shape, &dev()).unwrap()
}

/// The MLX twin's test decoder: fixed 1×1 channel mix → sigmoid → ×8 bilinear upsample (a
/// random-init TAESD saturates its clamp and zeroes the gradient under test).
struct TinyTestDecoder {
    mix: Tensor,
}

impl X0Decoder for TinyTestDecoder {
    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let (_, c, h, w) = latents.dims4()?;
        let flat = latents.reshape((c, h * w))?;
        let rgb = candle_gen::candle_nn::ops::sigmoid(&self.mix.matmul(&flat)?)?
            .reshape((1, 3, h, w))?
            .permute([0, 2, 3, 1])?;
        resample_nhwc(
            &rgb,
            &AxisMatrix::resize(h, 8 * h, false, &dev())?,
            &AxisMatrix::resize(w, 8 * w, false, &dev())?,
        )
    }
}

fn tiny_decoder() -> TinyTestDecoder {
    TinyTestDecoder {
        mix: normal(31, &[3, 4], 0.8),
    }
}

/// The "trainer": a frozen 1×1 channel mix with a rank-2 LoRA (`W + B·A`) applied to a latent.
struct LoraModule {
    base: Tensor,
    a: Tensor,
}

impl LoraModule {
    fn new() -> Self {
        Self {
            base: (normal(11, &[4, 4], 0.1) + Tensor::eye(4, DType::F32, &dev()).unwrap()).unwrap(),
            a: normal(12, &[2, 4], 0.5),
        }
    }

    fn x0(&self, b: &Tensor, latent: &Tensor) -> Result<Tensor> {
        let w = (&self.base + b.matmul(&self.a)?)?;
        let (n, c, h, wd) = latent.dims4()?;
        Ok(w.matmul(&latent.reshape((c, h * wd))?)?
            .reshape((n, c, h, wd))?)
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

fn latent() -> Tensor {
    normal(21, &[1, 4, 4, 3], 1.0)
}

/// `|∂ aux / ∂ B|₁` for the LoRA up-projection (`None` when the plan has no aux term).
fn lora_grad(path: &mut PerceptualPath, mask: Option<&Tensor>) -> Option<f32> {
    let lora = LoraModule::new();
    let clean = latent();
    let b = Var::from_tensor(&normal(13, &[4, 2], 0.3)).unwrap();
    let x_clean = lora.x0(b.as_tensor(), &clean).unwrap();
    path.ensure_reference_with_mask(0, &x_clean, mask).unwrap();
    let plan = path.plan(0, 0, 0.5).unwrap();
    plan.aux.first()?;
    let live = (&clean + normal(22, &[1, 4, 4, 3], 0.2)).unwrap();
    let x0 = lora.x0(b.as_tensor(), &live).unwrap();
    let aux = path.aux_loss(&plan, 0, &x0).unwrap()?.weighted;
    let grads = aux.backward().unwrap();
    let g = grads
        .get(b.as_tensor())
        .expect("a gradient for the LoRA factor");
    Some(scalar(&g.abs().unwrap().sum_all().unwrap()))
}

fn person_pose() -> Arc<VitPose> {
    Arc::new(vitpose_with_head(10.0, 5.0))
}

fn nobody_pose() -> Arc<VitPose> {
    Arc::new(vitpose_with_head(1.0, -50.0))
}

fn full_mask() -> Tensor {
    let mut v = vec![0.0f32; 32 * 24];
    for y in 4..28 {
        for x in 4..20 {
            v[y * 24 + x] = 1.0;
        }
    }
    Tensor::from_vec(v, (32, 24), &dev()).unwrap()
}

/// AC1, per loss: a nonzero, finite gradient reaches the LoRA factor through decoder + frozen
/// model on candle. Mutation: `.detach()` the live heatmaps / betas / normals ⇒ red.
#[test]
fn each_body_loss_trains_the_lora_through_the_shared_path() {
    let w = fixture();
    let losses: Vec<(&str, Box<dyn PerceptualLoss>, Option<Tensor>)> = vec![
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

/// AC1, per loss: no detected person ⇒ no reference, no aux term, diffusion fallback. Mutation:
/// return a reference regardless of `detect` ⇒ red.
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

/// The restricted normal loss refuses a reference without a mask and the mask changes the loss.
/// Mutation: ignore `r.mask` in `NormalLoss::loss` ⇒ red.
#[test]
fn the_subject_restricted_normal_loss_needs_and_uses_the_mask() {
    let w = fixture();
    let restricted = NormalLoss::new(person_pose(), sapiens(&w), true);
    let clean = tiny_decoder().decode(&latent()).unwrap();
    assert!(restricted.reference(&clean).is_err());
    let live = tiny_decoder()
        .decode(&(latent() + normal(5, &[1, 4, 4, 3], 0.5)).unwrap())
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

/// E7: candle arm footprints are the backend-neutral split. Mutation: drop the ViTPose
/// attribution in `body_arm_footprint` ⇒ red.
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

/// The arm builders name a missing checkpoint.
#[test]
fn arm_builders_name_a_missing_checkpoint() {
    let mut cfg = BodyLossesConfig::default();
    cfg.shape.weight = 0.1;
    let e = shape_loss(&cfg, &dev()).err().unwrap().to_string();
    assert!(e.contains("HybrIK") && e.contains("shape_model_dir"), "{e}");
    cfg.shape_model_dir = Some(PathBuf::from("/nonexistent/hybrik"));
    let e = shape_loss(&cfg, &dev()).err().unwrap().to_string();
    assert!(e.contains("HybrIK"), "{e}");
    let e = proportion_loss(&cfg, &dev()).err().unwrap().to_string();
    assert!(e.contains("pose_model_dir"), "{e}");
}

/// Real-weight parity (S9's real-weight phase) — see the MLX twin for the directory layout.
#[test]
#[ignore = "needs the real ViTPose+/HybrIK/Sapiens checkpoints + --real reference outputs \
            (SCENEWORKS_BODY_LOSS_REAL); never downloaded in ordinary test runs"]
fn real_checkpoints_match_the_reference_implementation() {
    let root = PathBuf::from(
        std::env::var("SCENEWORKS_BODY_LOSS_REAL")
            .expect("SCENEWORKS_BODY_LOSS_REAL must name the real-weight directory"),
    );
    let r = Weights::from_file(&root.join("reference.safetensors"), &dev(), DType::F32).unwrap();
    let img = nhwc(&r, "input.a");
    let pose = VitPose::from_dir(root.join("vitpose"), VitPoseConfig::plus_base(), &dev()).unwrap();
    let (hm, _) = pose.forward_pixels(&img).unwrap();
    let (coords, conf) = vitpose::heatmaps_to_keypoints(&hm).unwrap();
    let (ratios, vis) = body_ratios(&coords, &conf, true).unwrap();
    close(
        "real heatmaps",
        &hm.permute([0, 3, 1, 2]).unwrap(),
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
    let b: Vec<f32> = t(&r, "input.person_bbox").to_vec1().unwrap();
    let (_, h, w, _) = img.dims4().unwrap();
    let crop = HybrikEncoder::crop_for([b[0], b[1], b[2], b[3]], h, w);
    let hyb =
        HybrikEncoder::from_dir(root.join("hybrik"), HybrikConfig::resnet34(), &dev()).unwrap();
    close(
        "real betas",
        &hyb.forward_crop(&img, crop).unwrap(),
        &t(&r, "hybrik.out.betas_a"),
        1e-3,
    );
    let sap = SapiensNormal::from_dir(root.join("sapiens"), SapiensConfig::normal_0_3b(), &dev())
        .unwrap();
    close(
        "real normals",
        &sap.forward_pixels(&img).unwrap(),
        &nhwc(&r, "sapiens.out.normals_a"),
        5e-3,
    );
}
