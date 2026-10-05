//! sc-24831 kit tests: the face losses through the shared perceptual path on synthetic models, and
//! the torch-reference parity fixture (AC3; the Candle twin checks the same fixture).

use std::path::PathBuf;

use mlx_gen::train::perceptual::{
    AuxLoss, AuxLossSchedule, AuxModelFootprint, PerceptualPath, X0Decoder,
};
use mlx_rs::transforms::eval;
use serde_json::Value;

use super::*;
use crate::synth;

// ---------------------------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------------------------

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../face_loss_fixtures")
}

/// The committed torch-reference fixture, refusing one whose producer changed without a regen, plus
/// a [`CpuDevice`] guard for the comparing test.
fn fixture() -> (Value, CpuDevice) {
    use sha2::{Digest, Sha256};
    let dir = fixture_dir();
    let f: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("face_loss_fixtures.json")).expect("fixture"),
    )
    .expect("fixture json");
    let producer = std::fs::read(dir.join("produce_face_loss_fixtures.py")).expect("producer");
    let sha: String = Sha256::digest(&producer)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        f["producer_sha256"].as_str().unwrap(),
        sha,
        "the producer changed without regenerating face_loss_fixtures.json"
    );
    (f, CpuDevice::new())
}

/// Run on the MLX **CPU** device for the guard's lifetime. The Metal f32 GEMM of this MLX build
/// rounds (measured: a 112² bilinear resample drifts ~4e-4 from the f64-exact result on GPU, exact
/// on CPU), which would force a tolerance loose enough to hide a real port bug — the parity tests
/// check the port's math, so they run where f32 is f32. The library itself runs wherever the
/// trainer runs.
struct CpuDevice;
impl CpuDevice {
    fn new() -> Self {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        CpuDevice
    }
}
impl Drop for CpuDevice {
    fn drop(&mut self) {
        mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    }
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn read(a: &Array) -> Vec<f32> {
    let a = a.reshape(&[-1]).unwrap();
    eval([&a]).unwrap();
    a.as_slice::<f32>().to_vec()
}

fn scalar(a: &Array) -> f32 {
    eval([a]).unwrap();
    a.item::<f32>()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn arcface_weights(f: &Value) -> Weights {
    let a = &f["arcface"];
    let seed = a["seed"].as_u64().unwrap();
    let mut w = Weights::empty();
    for (k, shape) in a["keys"].as_object().unwrap() {
        let shape: Vec<usize> = shape
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        w.insert(k.clone(), synth::tensor(seed, k, &shape));
    }
    w
}

fn mesh_program(f: &Value) -> Program {
    let m = &f["facemesh"];
    let seed = m["seed"].as_u64().unwrap();
    let mut w = Weights::empty();
    for (k, shape) in m["param_shapes"].as_object().unwrap() {
        let shape: Vec<usize> = shape
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        w.insert(k.clone(), synth::tensor(seed, k, &shape));
    }
    Program::new(
        crate::program::ProgramSpec::from_value(&m["program"]).unwrap(),
        &w,
    )
    .unwrap()
}

fn images(f: &Value) -> (Array, Array) {
    let im = &f["image"];
    let (seed, h, w) = (
        im["seed"].as_u64().unwrap(),
        im["h"].as_u64().unwrap() as usize,
        im["w"].as_u64().unwrap() as usize,
    );
    (
        synth::image(seed, "live", h, w),
        synth::image(seed, "reference", h, w),
    )
}

fn bbox(f: &Value) -> [f32; 4] {
    let b = floats(&f["bbox"]);
    [b[0], b[1], b[2], b[3]]
}

/// A detector that always reports `face` (or nothing).
struct StubDetector(Option<[f32; 4]>);
impl FaceBoxDetector for StubDetector {
    fn largest_face(&self, _: &[u8], _: usize, _: usize) -> Result<Option<[f32; 4]>> {
        Ok(self.0)
    }
}

fn identity_loss(f: &Value, face: Option<[f32; 4]>, min_cos: f32, mode: IdentityReferenceMode) -> IdentityLoss {
    IdentityLoss::new(
        ArcFace::from_weights(&arcface_weights(f)).unwrap(),
        Rc::new(StubDetector(face)),
        min_cos,
        mode,
    )
}

// ---------------------------------------------------------------------------------------------
// AC3 — torch-reference parity (the Candle twin asserts the same numbers)
// ---------------------------------------------------------------------------------------------

/// AC3: the MLX identity path (crop box, zero-pad-to-square + bilinear 112², ArcFace, normalize)
/// reproduces the torch reference embedding and loss. Mutations: `round_ties_even` → `round` in
/// `face_crop_box` ⇒ crop box differs ⇒ red; drop the `.max(0.0)` source clamp in
/// `bilinear_matrix` ⇒ embedding differs ⇒ red; centre the square pad on the wrong side ⇒ red.
#[test]
fn identity_path_matches_the_torch_reference() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let (h, w) = (live.shape()[1] as usize, live.shape()[2] as usize);
    let crop = face_crop_box(bbox(&f), h, w);
    let want_box: Vec<usize> = f["crop_box"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!([crop.x0, crop.y0, crop.x1, crop.y1].to_vec(), want_box);

    let loss = identity_loss(&f, None, -1.0, IdentityReferenceMode::PerImage);
    let e_live = read(&loss.embed(&live, crop).unwrap());
    let e_ref = read(&loss.embed(&reference, crop).unwrap());
    let d_live = max_abs_diff(&e_live, &floats(&f["arcface"]["embedding"]));
    let d_ref = max_abs_diff(&e_ref, &floats(&f["arcface"]["reference_embedding"]));
    assert!(d_live < 2e-5 && d_ref < 2e-5, "embedding drift {d_live} / {d_ref}");

    let r = IdentityReference {
        crop,
        image_hw: (h, w),
        embedding: Array::from_slice(&e_ref, &[e_ref.len() as i32]),
    };
    let (l, cos) = loss.loss_and_cos(&live, &r).unwrap();
    let want_cos = f["arcface"]["cos"].as_f64().unwrap() as f32;
    assert!((scalar(&cos) - want_cos).abs() < 2e-5, "cos {} vs {want_cos}", scalar(&cos));
    let want = f["arcface"]["identity_loss"].as_f64().unwrap() as f32;
    assert!((scalar(&l) - want).abs() < 2e-5, "loss {} vs {want}", scalar(&l));
}

/// AC3 (landmarks): the fx-program executor runs the converter's lowered FaceMesh-shaped program
/// (conv / depthwise / PReLU / max-pool / spatial + channel pad / add / mul / sigmoid / reshape) and
/// the landmark path reproduces torch's normalized landmarks and region-weighted loss. Mutations:
/// map the channel pad onto the wrong NHWC axis ⇒ red; drop the nose centring ⇒ red; swap the
/// jaw/lip weights ⇒ the loss goes red.
#[test]
fn landmark_path_matches_the_torch_reference() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let (h, w) = (live.shape()[1] as usize, live.shape()[2] as usize);
    let crop = face_crop_box(bbox(&f), h, w);
    let loss = FaceLandmarkLoss::new(mesh_program(&f), Rc::new(StubDetector(None)));
    let l_live = loss.landmarks(&live, crop).unwrap();
    let l_ref = loss.landmarks(&reference, crop).unwrap();
    let want = floats(&f["facemesh"]["landmarks"]);
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let d = max_abs_diff(&read(&l_live), &want);
    assert!(d < 1e-4 * scale, "landmark drift {d} (scale {scale})");
    let got = scalar(&landmark_distance(&l_live, &l_ref).unwrap());
    let want = f["facemesh"]["landmark_loss"].as_f64().unwrap() as f32;
    assert!((got - want).abs() < 1e-4 * want.max(1.0), "loss {got} vs {want}");
}

// ---------------------------------------------------------------------------------------------
// AC1 / AC2 — through the shared perceptual path with a tiny LoRA'd module
// ---------------------------------------------------------------------------------------------

/// Identity "decoder": NCHW latents are already pixels.
struct PixelDecoder;
impl X0Decoder for PixelDecoder {
    fn decode(&self, latents: &Array) -> Result<Array> {
        Ok(latents.transpose_axes(&[0, 2, 3, 1])?)
    }
}

fn nchw(px: &Array) -> Array {
    px.transpose_axes(&[0, 3, 1, 2]).unwrap()
}

const WEIGHT: f32 = 0.5;

fn path_with(loss: Box<dyn PerceptualLoss>) -> PerceptualPath {
    PerceptualPath::new(
        Some(Box::new(PixelDecoder)),
        vec![AuxLoss {
            schedule: AuxLossSchedule {
                weight: WEIGHT,
                t_min: 0.0,
                t_max: 1.0,
                every_n: 1,
            },
            loss,
        }],
    )
    .unwrap()
}

/// The tiny LoRA'd module: a per-pixel channel mix `x0 = (I + B·A) · z` over the 3 channels of the
/// NCHW latent `z` (rank 2) — the adapter is `(A, B)`.
fn lora_x0(z: &Array, a: &Array, b: &Array) -> Array {
    let s = z.shape();
    let (h, w) = (s[2], s[3]);
    let eye = Array::from_slice(&[1.0f32, 0., 0., 0., 1., 0., 0., 0., 1.], &[3, 3]);
    let m = add(&eye, &matmul(b, a).unwrap()).unwrap();
    matmul(&m, &z.reshape(&[3, h * w]).unwrap())
        .unwrap()
        .reshape(&[1, 3, h, w])
        .unwrap()
}

fn aux_value_and_grads(
    path: &PerceptualPath,
    z: &Array,
    a: &Array,
    b: &Array,
) -> (f32, Vec<f32>, Vec<f32>) {
    let plan = path.plan(0, 0, 0.5).unwrap();
    assert!(plan.aux == vec![0], "{plan:?}");
    let f = |args: &[Array]| -> mlx_rs::error::Result<Vec<Array>> {
        let x0 = lora_x0(z, &args[0], &args[1]);
        let terms = path
            .aux_loss(&plan, 0, &x0)
            .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))?
            .expect("aux term");
        Ok(vec![terms.weighted])
    };
    let mut vg = mlx_rs::transforms::value_and_grad_with_argnums(f, &[0, 1]);
    let (v, g) = vg(&[a.clone(), b.clone()]).unwrap();
    (scalar(&v[0]), read(&g[0]), read(&g[1]))
}

fn lora_init() -> (Array, Array) {
    // Standard LoRA init: A random, B zero ⇒ x0 == z at step 0, gradient lands on B.
    (
        synth::tensor(7, "lora.a", &[2, 3]),
        Array::from_slice(&[0f32; 6], &[3, 2]),
    )
}

/// AC1: with identity weight > 0 the path's aux term is `weight · (1 − cos(embed(x0_face), ref))` —
/// equal to the torch fixture's `1 − cos` on the same images — and back-propagates a nonzero
/// gradient into the LoRA. Mutations: return `cos` instead of `1 − cos` ⇒ value red; wrap the live
/// embedding in `stop_gradient` ⇒ the LoRA gradient is zero ⇒ red.
#[test]
fn identity_loss_is_one_minus_cos_and_trains_the_lora() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let mut path = path_with(Box::new(identity_loss(
        &f,
        Some(bbox(&f)),
        -1.0,
        IdentityReferenceMode::PerImage,
    )));
    path.ensure_reference(0, &nchw(&reference)).unwrap();
    assert!(path.is_usable(0, 0).unwrap());
    let (a, b) = lora_init();
    let (value, ga, gb) = aux_value_and_grads(&path, &nchw(&live), &a, &b);
    let want = WEIGHT * f["arcface"]["identity_loss"].as_f64().unwrap() as f32;
    assert!((value - want).abs() < 2e-5, "aux {value} vs {want}");
    let gb_norm: f32 = gb.iter().map(|g| g.abs()).sum();
    assert!(gb_norm > 1e-6, "no gradient reached the LoRA B factor: {gb:?}");
    assert!(ga.iter().all(|g| g.is_finite()));
}

/// AC2: an image with no detected face is unusable (the plan falls back to diffusion, no aux term),
/// and a step whose live cosine is at or below `min_cos` contributes exactly zero loss and zero
/// gradient, while one just above it does not. Mutations: drop the `gt(min_cos)` gate ⇒ the gated
/// value is nonzero ⇒ red; return `Some` for a no-face reference ⇒ the plan keeps the aux loss ⇒ red.
#[test]
fn no_face_and_low_cos_samples_contribute_zero() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let cos = f["arcface"]["cos"].as_f64().unwrap() as f32;

    let mut no_face = path_with(Box::new(identity_loss(
        &f,
        None,
        -1.0,
        IdentityReferenceMode::PerImage,
    )));
    no_face.ensure_reference(0, &nchw(&reference)).unwrap();
    assert!(!no_face.is_usable(0, 0).unwrap());
    let plan = no_face.plan(0, 0, 0.5).unwrap();
    assert!(plan.diffusion && plan.aux.is_empty(), "{plan:?}");
    assert!(no_face.aux_loss(&plan, 0, &nchw(&live)).unwrap().is_none());

    let (a, b) = lora_init();
    for (min_cos, gated) in [(cos + 1e-3, true), (cos - 1e-3, false)] {
        let mut p = path_with(Box::new(identity_loss(
            &f,
            Some(bbox(&f)),
            min_cos,
            IdentityReferenceMode::PerImage,
        )));
        p.ensure_reference(0, &nchw(&reference)).unwrap();
        let (value, _, gb) = aux_value_and_grads(&p, &nchw(&live), &a, &b);
        let g: f32 = gb.iter().map(|g| g.abs()).sum();
        if gated {
            assert_eq!(value, 0.0, "min_cos {min_cos} > cos {cos} must gate the loss");
            assert_eq!(g, 0.0, "a gated step must not move the LoRA");
        } else {
            assert!(value > 0.1 && g > 1e-6, "min_cos {min_cos} < cos {cos}: {value} / {g}");
        }
    }
}

/// Dataset-average mode targets the normalized mean of every face-bearing reference (upstream
/// `identity_loss_use_average`), frozen at the first loss; a reference added afterwards is refused.
/// Mutation: target the per-image embedding in average mode ⇒ the value equals the per-image one ⇒
/// red.
#[test]
fn dataset_average_targets_the_mean_reference() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let loss = identity_loss(&f, Some(bbox(&f)), -1.0, IdentityReferenceMode::DatasetAverage);
    let r_live = loss.reference(&live).unwrap().unwrap();
    let r_ref = loss.reference(&reference).unwrap().unwrap();
    let r_ref = reference_as::<IdentityReference>("identity", r_ref.as_ref()).unwrap();
    let r_live_e = &reference_as::<IdentityReference>("identity", r_live.as_ref())
        .unwrap()
        .embedding;
    let mean = l2_normalize(&add(r_live_e, &r_ref.embedding).unwrap()).unwrap();
    let want = 1.0 - scalar(&multiply(&mean, r_live_e).unwrap().sum(None).unwrap());
    let got = scalar(&loss.loss(&live, r_ref).unwrap());
    assert!((got - want).abs() < 1e-5, "average-mode loss {got} vs {want}");
    let per_image = f["arcface"]["identity_loss"].as_f64().unwrap() as f32;
    assert!((got - per_image).abs() > 1e-3, "average mode used the per-image target");
    assert!(loss.reference(&live).is_err(), "a late reference must be refused");
}

/// The landmark loss is differentiable into the LoRA and skips a no-face image. Mutation: wrap the
/// live landmarks in `stop_gradient` ⇒ zero gradient ⇒ red.
#[test]
fn landmark_loss_trains_the_lora_and_skips_no_face() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let mut path = path_with(Box::new(FaceLandmarkLoss::new(
        mesh_program(&f),
        Rc::new(StubDetector(Some(bbox(&f)))),
    )));
    path.ensure_reference(0, &nchw(&reference)).unwrap();
    let (a, b) = lora_init();
    let (value, _, gb) = aux_value_and_grads(&path, &nchw(&live), &a, &b);
    let want = WEIGHT * f["facemesh"]["landmark_loss"].as_f64().unwrap() as f32;
    assert!((value - want).abs() < 1e-4 * want.max(1.0), "{value} vs {want}");
    assert!(gb.iter().map(|g| g.abs()).sum::<f32>() > 1e-6);

    let mut skip = path_with(Box::new(FaceLandmarkLoss::new(
        mesh_program(&f),
        Rc::new(StubDetector(None)),
    )));
    skip.ensure_reference(0, &nchw(&reference)).unwrap();
    assert!(skip.plan(0, 0, 0.5).unwrap().aux.is_empty());
}

/// A live decode at another size than its reference is an error, never a silently misplaced crop.
#[test]
fn a_live_decode_of_another_size_is_refused() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let loss = identity_loss(&f, Some(bbox(&f)), -1.0, IdentityReferenceMode::PerImage);
    let r = loss.reference(&reference).unwrap().unwrap();
    let small = live.index((.., 0..64, .., ..));
    assert!(loss.loss(&small, r.as_ref()).is_err());
}

// ---------------------------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------------------------

/// Python `round` is half-to-even; a box edge on .5 must round to even like upstream.
/// Mutation: `round_ties_even` → `round` ⇒ x0 becomes 3 ⇒ red.
#[test]
fn face_crop_box_rounds_half_to_even_and_falls_back_to_the_frame() {
    // bw = 20 ⇒ pad 3 ⇒ x1 − pad = 2.5 ⇒ 2 (even).
    let b = face_crop_box([5.5, 10.0, 25.5, 30.0], 64, 64);
    assert_eq!((b.x0, b.y0, b.x1, b.y1), (2, 7, 28, 33));
    let full = face_crop_box([70.0, 70.0, 80.0, 80.0], 64, 64);
    assert_eq!((full.x0, full.y0, full.x1, full.y1), (0, 0, 64, 64));
}

#[test]
fn bilinear_matrix_rows_sum_to_one_and_identity_at_equal_size() {
    let m = bilinear_matrix(7, 13);
    for r in m.chunks(13) {
        assert!((r.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    let id = bilinear_matrix(5, 5);
    for (i, r) in id.chunks(5).enumerate() {
        for (j, v) in r.iter().enumerate() {
            assert_eq!(*v, if i == j { 1.0 } else { 0.0 });
        }
    }
}

/// `infer_layers` reads any IResNet depth; the analytic glintr100 count matches the published
/// 65.2 M (261 MB f32 onnx). Mutation: drop the downsample term ⇒ the count leaves the band ⇒ red.
#[test]
fn arcface_depth_is_read_from_the_keys_and_its_size_is_counted() {
    use crate::iresnet::{infer_layers, IRESNET100_LAYERS, IRESNET50_LAYERS};
    for layers in [IRESNET100_LAYERS, IRESNET50_LAYERS] {
        let has = |k: &str| {
            (1..=4).any(|l| (0..layers[l - 1]).any(|b| k == format!("layer{l}.{b}.conv1.weight")))
        };
        assert_eq!(infer_layers(has).unwrap(), layers);
    }
    assert!(infer_layers(|_| false).is_err());
    let r100 = arcface_param_count(IRESNET100_LAYERS);
    assert!((65_000_000..65_400_000).contains(&r100), "{r100}");
    let r50 = arcface_param_count(IRESNET50_LAYERS);
    assert!((43_400_000..43_800_000).contains(&r50), "{r50}");
}

/// E7: off ⇒ no footprint; the shared detector is counted once whichever losses are on, and each
/// enabled loss adds its model. Mutation: push the detector per loss ⇒ 4 entries ⇒ red.
#[test]
fn face_loss_footprints_count_each_model_once() {
    assert!(face_loss_footprints(None, false).is_empty());
    let both = face_loss_footprints(Some(crate::iresnet::IRESNET100_LAYERS), true);
    assert_eq!(both.len(), 3);
    assert_eq!(both[0].param_bytes, SCRFD_10G_PARAMS * 4);
    assert!(both[1].param_bytes > 250_000_000 && both[1].working_set_bytes > 0);
    assert_eq!(face_loss_footprints(None, true).len(), 2);
    let total: u64 = mlx_gen::train::perceptual::perceptual_footprint_bytes(None, &both, 10);
    let sum: u64 = both
        .iter()
        .map(|f: &AuxModelFootprint| {
            f.param_bytes + f.working_set_bytes + 10 * f.reference_bytes_per_image
        })
        .sum();
    assert_eq!(total, sum);
}

/// The executor refuses an op the converter never emits, naming it.
#[test]
fn an_unknown_program_op_is_refused_by_name() {
    let err = crate::program::ProgramSpec::parse(
        r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"op":"gelu","out":"y","inputs":["x"]}]}"#,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("gelu"), "{err}");
}
