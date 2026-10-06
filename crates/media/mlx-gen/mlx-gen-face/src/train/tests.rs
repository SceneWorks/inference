//! sc-24831 kit tests: the face losses through the shared perceptual path on synthetic models, and
//! the torch-reference parity fixture (AC3; the Candle twin checks the same fixture).

use std::path::PathBuf;

use mlx_gen::train::perceptual::{AuxLoss, AuxLossSchedule, PerceptualPath, X0Decoder};
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
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
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
        mlx_gen::gen_core::fx_program::ProgramSpec::from_value(&m["program"]).unwrap(),
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

/// The scorer over the fixture's synthetic ArcFace (its noise mean computed at construction, as at
/// load).
fn scorer(f: &Value, min_cos: f32, mode: IdentityReferenceMode) -> Rc<IdentityScorer> {
    Rc::new(
        IdentityScorer::new(
            ArcFace::from_weights(&arcface_weights(f)).unwrap(),
            min_cos,
            mode,
        )
        .unwrap(),
    )
}

fn identity_loss(
    f: &Value,
    face: Option<[f32; 4]>,
    min_cos: f32,
    mode: IdentityReferenceMode,
) -> IdentityLoss {
    IdentityLoss::new(scorer(f, min_cos, mode), Rc::new(StubDetector(face)))
}

fn crop_of(f: &Value) -> CropBox {
    let im = &f["image"];
    face_crop_box(
        bbox(f),
        im["h"].as_u64().unwrap() as usize,
        im["w"].as_u64().unwrap() as usize,
    )
}

fn expect(v: &Value) -> f32 {
    v.as_f64().unwrap() as f32
}

fn noise_px(key: &str, h: usize, w: usize) -> Array {
    Array::from_slice(
        &mlx_gen::gen_core::train::face_loss::synth::noise_image(IDENTITY_NOISE_SEED, key, h, w),
        &[1, h as i32, w as i32, 3],
    )
}

// ---------------------------------------------------------------------------------------------
// AC3 — torch-reference parity (the Candle twin asserts the same numbers)
// ---------------------------------------------------------------------------------------------

/// AC3: the MLX identity path (crop box, zero-pad-to-square + bilinear 112², ArcFace, normalize)
/// reproduces the torch reference embedding, the bias direction (mean embedding of the 200
/// counter-based noise images) and the bias-centred cosine / loss. Mutations: `round_ties_even` →
/// `round` in `face_crop_box` ⇒ crop box differs ⇒ red; drop the `.max(0.0)` source clamp in
/// `bilinear_matrix` ⇒ embedding differs ⇒ red; skip the bias centring in `frame_score` ⇒ cos red;
/// draw the noise set from a different key ⇒ noise mean red.
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
    let s = loss.scorer();
    assert_eq!(
        f["arcface"]["noise_seed"].as_u64().unwrap(),
        IDENTITY_NOISE_SEED
    );
    assert_eq!(
        f["arcface"]["noise_samples"].as_u64().unwrap() as usize,
        IDENTITY_NOISE_SAMPLES
    );
    let d_mean = max_abs_diff(&read(s.noise_mean()), &floats(&f["arcface"]["noise_mean"]));
    assert!(d_mean < 2e-5, "noise-mean drift {d_mean}");
    let e_live = read(&s.embed(&live, crop).unwrap());
    let e_ref = read(&s.embed(&reference, crop).unwrap());
    let d_live = max_abs_diff(&e_live, &floats(&f["arcface"]["embedding"]));
    let d_ref = max_abs_diff(&e_ref, &floats(&f["arcface"]["reference_embedding"]));
    assert!(
        d_live < 2e-5 && d_ref < 2e-5,
        "embedding drift {d_live} / {d_ref}"
    );

    let r = IdentityReference::single(
        crop,
        (h, w),
        Array::from_slice(&e_ref, &[e_ref.len() as i32]),
    );
    let (l, cos) = loss.loss_and_cos(&live, &r).unwrap();
    let want_cos = expect(&f["arcface"]["cos"]);
    assert!(
        (scalar(&cos) - want_cos).abs() < 2e-5,
        "cos {} vs {want_cos}",
        scalar(&cos)
    );
    let want = expect(&f["arcface"]["identity_loss"]);
    assert!(
        (scalar(&l) - want).abs() < 2e-5,
        "loss {} vs {want}",
        scalar(&l)
    );
}

/// Bias centring takes a non-face toward 0: 112² noise images drawn like the bias set score below
/// upstream's 0.2 gate against the synthetic reference, matching torch. Mutation: score the raw
/// (uncentred) cosine ⇒ values drift from the fixture ⇒ red.
#[test]
fn noise_scores_below_the_gate_against_the_reference() {
    let (f, _cpu) = fixture();
    let (_, reference) = images(&f);
    let s = scorer(&f, -1.0, IdentityReferenceMode::PerImage);
    let e_ref = s
        .center(&s.embed(&reference, crop_of(&f)).unwrap())
        .unwrap();
    let keys = f["arcface"]["noise_probe_keys"].as_array().unwrap();
    let want = floats(&f["arcface"]["noise_probe_cos"]);
    assert_eq!(keys.len(), want.len());
    assert!(!keys.is_empty());
    let full = CropBox {
        x0: 0,
        y0: 0,
        x1: ARCFACE_INPUT,
        y1: ARCFACE_INPUT,
    };
    for (key, want) in keys.iter().zip(want) {
        let probe = noise_px(key.as_str().unwrap(), ARCFACE_INPUT, ARCFACE_INPUT);
        let c = s.center(&s.embed(&probe, full).unwrap()).unwrap();
        let cos = scalar(&multiply(&c, &e_ref).unwrap().sum(None).unwrap());
        assert!((cos - want).abs() < 2e-5, "{key}: cos {cos} vs {want}");
        assert!(cos < 0.2, "{key}: noise scored {cos}");
    }
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
    let crop = crop_of(&f);
    let loss = FaceLandmarkLoss::new(mesh_program(&f), Rc::new(StubDetector(None)), None);
    let l_live = loss.landmarks(&live, crop).unwrap();
    let l_ref = loss.landmarks(&reference, crop).unwrap();
    let want = floats(&f["facemesh"]["landmarks"]);
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let d = max_abs_diff(&read(&l_live), &want);
    assert!(d < 1e-4 * scale, "landmark drift {d} (scale {scale})");
    let got = scalar(&landmark_distance(&l_live, &l_ref).unwrap());
    let want = expect(&f["facemesh"]["landmark_loss"]);
    assert!(
        (got - want).abs() < 1e-4 * want.max(1.0),
        "loss {got} vs {want}"
    );
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
    let m = add(&eye, matmul(b, a).unwrap()).unwrap();
    matmul(&m, z.reshape(&[3, h * w]).unwrap())
        .unwrap()
        .reshape(&[1, 3, h, w])
        .unwrap()
}

/// The weighted aux term and its LoRA gradients at a step, plus that step's noise level.
fn aux_value_and_grads(
    path: &PerceptualPath,
    z: &Array,
    a: &Array,
    b: &Array,
) -> (f32, Vec<f32>, Vec<f32>, f32) {
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
    (scalar(&v[0]), read(&g[0]), read(&g[1]), plan.noise_level)
}

fn lora_init() -> (Array, Array) {
    // Standard LoRA init: A random, B zero ⇒ x0 == z at step 0, gradient lands on B.
    (
        synth::tensor(7, "lora.a", &[2, 3]),
        Array::from_slice(&[0f32; 6], &[3, 2]),
    )
}

/// AC1: with identity weight > 0 the path's aux term is `weight · t · (1 − cos)` — `cos` the
/// bias-centred cosine, equal to the torch fixture's `1 − cos` on the same images, `t` the step's
/// noise level (upstream's `t_ratio`) — and back-propagates a nonzero gradient into the LoRA.
/// Mutations: return `cos` instead of `1 − cos` ⇒ value red; wrap the live embedding in
/// `stop_gradient` ⇒ the LoRA gradient is zero ⇒ red; `timestep_weight` → 1 ⇒ value red.
#[test]
fn identity_loss_is_one_minus_cos_and_trains_the_lora() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let loss = identity_loss(&f, Some(bbox(&f)), -1.0, IdentityReferenceMode::PerImage);
    assert_eq!(loss.timestep_weight(0.3), 0.3);
    let mut path = path_with(Box::new(loss));
    path.ensure_reference(0, &nchw(&reference)).unwrap();
    assert!(path.is_usable(0, 0).unwrap());
    let (a, b) = lora_init();
    let (value, ga, gb, t) = aux_value_and_grads(&path, &nchw(&live), &a, &b);
    assert!(t > 0.05 && t < 0.95, "noise level {t}");
    let want = WEIGHT * t * expect(&f["arcface"]["identity_loss"]);
    assert!((value - want).abs() < 2e-5, "aux {value} vs {want}");
    let gb_norm: f32 = gb.iter().map(|g| g.abs()).sum();
    assert!(
        gb_norm > 1e-6,
        "no gradient reached the LoRA B factor: {gb:?}"
    );
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
    let cos = expect(&f["arcface"]["cos"]);

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
        let (value, _, gb, _) = aux_value_and_grads(&p, &nchw(&live), &a, &b);
        let g: f32 = gb.iter().map(|g| g.abs()).sum();
        if gated {
            assert_eq!(
                value, 0.0,
                "min_cos {min_cos} > cos {cos} must gate the loss"
            );
            assert_eq!(g, 0.0, "a gated step must not move the LoRA");
        } else {
            assert!(
                value > 0.05 && g > 1e-6,
                "min_cos {min_cos} < cos {cos}: {value} / {g}"
            );
        }
    }
}

/// Dataset-average mode (upstream `identity_loss_use_average`) targets the normalized mean of every
/// face-bearing reference, frozen at the first loss, and normalizes each image's loss by its own
/// clean score `max(cos(own, mean), 0.1)` — `max(0, 1 − cos / clean)` — matching torch; a reference
/// added afterwards is refused. Mutations: target the per-image embedding ⇒ cos red; drop the
/// clean-cos normalization ⇒ loss red.
#[test]
fn dataset_average_targets_the_mean_reference() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let im = &f["image"];
    let reference2 = synth::image(
        im["seed"].as_u64().unwrap(),
        f["arcface"]["average"]["reference2_key"].as_str().unwrap(),
        im["h"].as_u64().unwrap() as usize,
        im["w"].as_u64().unwrap() as usize,
    );
    let loss = identity_loss(
        &f,
        Some(bbox(&f)),
        -1.0,
        IdentityReferenceMode::DatasetAverage,
    );
    let r_ref = loss.reference(&reference).unwrap().unwrap();
    loss.reference(&reference2).unwrap().unwrap();
    let r = reference_as::<IdentityReference>("identity", r_ref.as_ref()).unwrap();
    let avg = &f["arcface"]["average"];
    let (cos, clean, _) = loss
        .scorer()
        .frame_score(&live, r.frames[0].as_ref().unwrap())
        .unwrap();
    let (cos, clean) = (scalar(&cos), scalar(&clean));
    assert!((cos - expect(&avg["cos"])).abs() < 2e-5, "cos {cos}");
    assert!(
        (clean - expect(&avg["clean_cos"])).abs() < 2e-5,
        "clean {clean}"
    );
    let got = scalar(&loss.loss(&live, r).unwrap());
    let want = expect(&avg["identity_loss"]);
    assert!(
        (got - want).abs() < 2e-5,
        "average-mode loss {got} vs {want}"
    );
    assert!(
        loss.reference(&live).is_err(),
        "a late reference must be refused"
    );
}

/// The multi-frame mean divides by the number of frames whose gate is OPEN (`max(Σ gate, 1)`):
/// with one frame gated out the loss equals the passing frame's own. Mutations: divide by the
/// face-bearing frame count ⇒ the value halves ⇒ red; drop the gate from the numerator ⇒ red.
#[test]
fn a_gated_frame_leaves_the_multi_frame_mean() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let (h, w) = (live.shape()[1] as usize, live.shape()[2] as usize);
    let other = noise_px("gate-probe", h, w);
    let clip = |a: &Array, b: &Array| mlx_rs::ops::concatenate_axis(&[a, b], 0).unwrap();
    let face = Some(bbox(&f));
    let probe = identity_loss(&f, face, -1.0, IdentityReferenceMode::PerImage);
    let r = probe.reference(&reference).unwrap().unwrap();
    let rf = reference_as::<IdentityReference>("identity", r.as_ref())
        .unwrap()
        .frames[0]
        .as_ref()
        .unwrap();
    let cos_of = |px: &Array| scalar(&probe.scorer().frame_score(px, rf).unwrap().0);
    let (c_live, c_other) = (cos_of(&live), cos_of(&other));
    assert!(
        (c_live - c_other).abs() > 0.02,
        "{c_live} vs {c_other}: need separable frames"
    );
    let min_cos = (c_live + c_other) / 2.0;
    let pass = 1.0 - c_live.max(c_other);
    assert!(pass > 0.05);

    let l = IdentityLoss::new(
        scorer(&f, min_cos, IdentityReferenceMode::PerImage),
        Rc::new(ScriptedDetector(RefCell::new(vec![face, face]))),
    );
    let r = l.reference(&clip(&reference, &reference)).unwrap().unwrap();
    let v = scalar(&l.loss(&clip(&live, &other), r.as_ref()).unwrap());
    assert!((v - pass).abs() < 2e-5, "{v} vs the passing frame's {pass}");
}

/// Upstream's detection retry reaches the reference: a detector that only finds the face on the
/// gray-padded frame yields the same crop (box unpadded) and so the fixture loss. Mutation: skip
/// the retry in `reference_boxes` ⇒ the reference is unusable ⇒ red.
#[test]
fn a_face_found_only_on_the_padded_retry_is_used() {
    struct PaddedOnly([f32; 4], (usize, usize));
    impl FaceBoxDetector for PaddedOnly {
        fn largest_face(&self, _: &[u8], h: usize, w: usize) -> Result<Option<[f32; 4]>> {
            Ok(((h, w) != self.1).then_some(self.0))
        }
    }
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let (h, w) = (live.shape()[1] as usize, live.shape()[2] as usize);
    let pad = (h.max(w) / 4) as f32;
    let b = bbox(&f);
    let loss = IdentityLoss::new(
        scorer(&f, -1.0, IdentityReferenceMode::PerImage),
        Rc::new(PaddedOnly(
            [b[0] + pad, b[1] + pad, b[2] + pad, b[3] + pad],
            (h, w),
        )),
    );
    let r = loss
        .reference(&reference)
        .unwrap()
        .expect("the retry finds the face");
    let crop = reference_as::<IdentityReference>("identity", r.as_ref())
        .unwrap()
        .frames[0]
        .as_ref()
        .unwrap()
        .crop;
    assert_eq!(crop, crop_of(&f));
    let v = scalar(&loss.loss(&live, r.as_ref()).unwrap());
    let want = expect(&f["arcface"]["identity_loss"]);
    assert!((v - want).abs() < 2e-5, "{v} vs {want}");
}

/// The landmark loss is differentiable into the LoRA, weighted by the noise level, and skips a
/// no-face image. Mutations: wrap the live landmarks in `stop_gradient` ⇒ zero gradient ⇒ red;
/// `timestep_weight` → 1 ⇒ value red.
#[test]
fn landmark_loss_trains_the_lora_and_skips_no_face() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let lm = FaceLandmarkLoss::new(
        mesh_program(&f),
        Rc::new(StubDetector(Some(bbox(&f)))),
        None,
    );
    assert_eq!(lm.timestep_weight(0.3), 0.3);
    let mut path = path_with(Box::new(lm));
    path.ensure_reference(0, &nchw(&reference)).unwrap();
    let (a, b) = lora_init();
    let (value, _, gb, t) = aux_value_and_grads(&path, &nchw(&live), &a, &b);
    let want = WEIGHT * t * expect(&f["facemesh"]["landmark_loss"]);
    assert!(
        (value - want).abs() < 1e-4 * want.max(1.0),
        "{value} vs {want}"
    );
    assert!(gb.iter().map(|g| g.abs()).sum::<f32>() > 1e-6);

    let mut skip = path_with(Box::new(FaceLandmarkLoss::new(
        mesh_program(&f),
        Rc::new(StubDetector(None)),
        None,
    )));
    skip.ensure_reference(0, &nchw(&reference)).unwrap();
    assert!(skip.plan(0, 0, 0.5).unwrap().aux.is_empty());
}

/// With the identity loss on, the landmark loss is gated on the identity cosine (upstream reuses
/// it): a frame at or below `min_cos` contributes zero loss and gradient, one above contributes
/// the ungated value. Mutation: ignore the gate in `FaceLandmarkLoss::loss` ⇒ the gated value is
/// nonzero ⇒ red.
#[test]
fn landmark_loss_is_gated_on_the_identity_cosine() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let cos = expect(&f["arcface"]["cos"]);
    let (a, b) = lora_init();
    for (min_cos, gated) in [(cos + 1e-3, true), (cos - 1e-3, false)] {
        let gate = scorer(&f, min_cos, IdentityReferenceMode::PerImage);
        let mut p = path_with(Box::new(FaceLandmarkLoss::new(
            mesh_program(&f),
            Rc::new(StubDetector(Some(bbox(&f)))),
            Some(gate),
        )));
        p.ensure_reference(0, &nchw(&reference)).unwrap();
        let (value, _, gb, t) = aux_value_and_grads(&p, &nchw(&live), &a, &b);
        let g: f32 = gb.iter().map(|g| g.abs()).sum();
        if gated {
            assert_eq!(value, 0.0, "min_cos {min_cos} > cos {cos}");
            assert_eq!(g, 0.0);
        } else {
            let want = WEIGHT * t * expect(&f["facemesh"]["landmark_loss"]);
            assert!(
                (value - want).abs() < 1e-4 * want.max(1.0),
                "{value} vs {want}"
            );
            assert!(g > 1e-6);
        }
    }
}

/// A detector that answers per call from a script (one entry per call, in order).
struct ScriptedDetector(RefCell<Vec<Option<[f32; 4]>>>);
impl FaceBoxDetector for ScriptedDetector {
    fn largest_face(&self, _: &[u8], _: usize, _: usize) -> Result<Option<[f32; 4]>> {
        Ok(self.0.borrow_mut().remove(0))
    }
}

/// A video decoder hands the loss `[F, H, W, 3]`: each frame gets its own reference box, a frame
/// without a face is skipped (its live pixels cannot move the loss), the loss is the mean over the
/// face-bearing frames, and a clip with no face at all is unusable. Mutations: average over every
/// frame (divide by 2) ⇒ the value halves ⇒ red; read frame 0 for every frame ⇒ the frame-1 face
/// is never scored ⇒ red. (A miss is retried on the padded frame, so a no-face frame consumes two
/// script entries.)
#[test]
fn multi_frame_decodes_score_each_face_bearing_frame() {
    let (f, _cpu) = fixture();
    let (live, reference) = images(&f);
    let clip = |a: &Array, b: &Array| mlx_rs::ops::concatenate_axis(&[a, b], 0).unwrap();
    let face = Some(bbox(&f));
    let shared = scorer(&f, -1.0, IdentityReferenceMode::PerImage);
    let loss = |script: Vec<Option<[f32; 4]>>| {
        IdentityLoss::new(
            shared.clone(),
            Rc::new(ScriptedDetector(RefCell::new(script))),
        )
    };
    let single = expect(&f["arcface"]["identity_loss"]);

    // Frames [face, no face]: only frame 0 counts; garbage in live frame 1 changes nothing.
    let l = loss(vec![face, None, None]);
    let r = l.reference(&clip(&reference, &reference)).unwrap().unwrap();
    let noise = synth::image(9, "noise", 72, 88);
    let v = scalar(&l.loss(&clip(&live, &noise), r.as_ref()).unwrap());
    assert!((v - single).abs() < 2e-5, "{v} vs {single}");
    // Frames [no face, face]: frame 1 is the one scored.
    let l = loss(vec![None, None, face]);
    let r = l.reference(&clip(&reference, &reference)).unwrap().unwrap();
    let v = scalar(&l.loss(&clip(&noise, &live), r.as_ref()).unwrap());
    assert!((v - single).abs() < 2e-5, "{v} vs {single}");
    // Both frames: the mean of the two (identical) frame losses.
    let l = loss(vec![face, face]);
    let r = l.reference(&clip(&reference, &reference)).unwrap().unwrap();
    let v = scalar(&l.loss(&clip(&live, &live), r.as_ref()).unwrap());
    assert!((v - single).abs() < 2e-5, "{v} vs {single}");
    // No face anywhere (the retry included) ⇒ unusable; a live clip of another length is refused.
    assert!(loss(vec![None, None, None, None])
        .reference(&clip(&reference, &reference))
        .unwrap()
        .is_none());
    assert!(l.loss(&live, r.as_ref()).is_err());
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

/// `infer_layers` reads any IResNet depth (iresnet100 glintr100, iresnet50 w600k_r50). Mutation:
/// stop counting at the first block ⇒ red.
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
}

/// The on-disk test stand-ins load through the real loaders, and the tiny ArcFace they write is the
/// fixture's architecture (kept in lockstep with the producer). The FaceMesh stand-in honours the
/// real I/O contract (478 landmarks). With the identity loss on, the landmark loss's gate IS the
/// identity loss's scorer (one ArcFace, one dataset mean); off, it is ungated. Mutation: build a
/// fresh scorer per loader (skip the `SCORERS` lookup) ⇒ `ptr_eq` red.
#[test]
fn testing_checkpoints_load_through_the_real_loaders() {
    let (f, _cpu) = fixture();
    let want: std::collections::BTreeMap<String, Vec<usize>> = f["arcface"]["keys"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, s)| {
            (
                k.clone(),
                s.as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect(),
            )
        })
        .collect();
    let got: std::collections::BTreeMap<String, Vec<usize>> =
        mlx_gen::gen_core::train::face_loss::synth::tiny_arcface_shapes()
            .into_iter()
            .collect();
    assert_eq!(got, want);
    let dir = tempfile::tempdir().unwrap();
    let tmp = dir.path().to_path_buf();
    testing::write_face_stack(&tmp).unwrap();
    testing::write_facemesh(&tmp).unwrap();
    let cfg = IdentityLossConfig::default();
    let id = load_identity_loss(&tmp, &cfg).unwrap();
    assert_eq!(id.scorer().arcface.layers(), [1, 2, 1, 1]);
    let lm = load_face_landmark_loss(&tmp, &tmp, Some(&cfg)).unwrap();
    assert!(Rc::ptr_eq(id.scorer(), lm.gate.as_ref().unwrap()));
    assert!(load_face_landmark_loss(&tmp, &tmp, None)
        .unwrap()
        .gate
        .is_none());
    let (live, _) = images(&f);
    let out = lm
        .landmarks(&live, face_crop_box(bbox(&f), 72, 88))
        .unwrap();
    assert_eq!(out.shape(), &[478, 2]);
}

/// The executor refuses an op the converter never emits, naming it.
#[test]
fn an_unknown_program_op_is_refused_by_name() {
    let err = mlx_gen::gen_core::fx_program::ProgramSpec::parse(
        r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"op":"gelu","out":"y","inputs":["x"]}]}"#,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("gelu"), "{err}");
}

/// A max-pool kernel larger than its (padded) input is an error, never an underflow panic.
/// Mutation: drop the size check in `max_pool_nhwc` ⇒ the subtraction underflows ⇒ panic ⇒ red.
#[test]
fn a_max_pool_kernel_larger_than_its_input_is_refused() {
    let spec = mlx_gen::gen_core::fx_program::ProgramSpec::parse(
        r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"op":"maxpool2d","out":"y","inputs":["x"],
            "kernel":[9,9],"stride":[1,1],"padding":[1,1]}]}"#,
    )
    .unwrap();
    let program = Program::new(spec, &Weights::empty()).unwrap();
    let x = Array::from_slice(&[0f32; 4 * 4 * 3], &[1, 4, 4, 3]);
    let err = program.forward(&x).unwrap_err().to_string();
    assert!(err.contains("exceeds"), "{err}");
}

/// The REAL converted FaceMesh-v2 program (`tools/convert_mp_facemesh_v2.py` output) reproduces the
/// upstream torch model's output 0 on a fixed input. Needs the converted checkpoint and the torch
/// I/O pair (`facemesh_torch_io.safetensors`: `input` NCHW, `output0`) in `$FACEMESH_REAL_DIR`.
#[test]
#[ignore = "needs the converted FaceMesh-v2 checkpoint + torch I/O in $FACEMESH_REAL_DIR"]
fn real_facemesh_program_matches_torch() {
    let _cpu = CpuDevice::new();
    let dir = PathBuf::from(std::env::var("FACEMESH_REAL_DIR").expect("FACEMESH_REAL_DIR"));
    let program = Program::from_file(dir.join(FACEMESH_FILE)).unwrap();
    let io = Weights::from_file(dir.join("facemesh_torch_io.safetensors")).unwrap();
    let x = io
        .require("input")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    let got = read(&program.forward(&x).unwrap()[0]);
    let want = read(io.require("output0").unwrap());
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let d = max_abs_diff(&got, &want);
    println!("real FaceMesh max abs {d} (scale {scale})");
    assert!(d <= 1e-4 * scale, "real FaceMesh drift {d} (scale {scale})");
}
