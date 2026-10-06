//! sc-24831 kit tests (Candle twin of `mlx-gen-face`'s): the face losses through the shared
//! perceptual path on synthetic models, and the torch-reference parity fixture (AC3 — the MLX twin
//! asserts the same committed numbers, so passing both pins MLX ≈ torch ≈ Candle).

use std::path::PathBuf;
use std::sync::Arc;

use candle_gen::candle_core::{Device, Tensor, Var};
use candle_gen::train::perceptual::{AuxLoss, AuxLossSchedule, PerceptualPath, X0Decoder};
use serde_json::Value;

use super::*;
use crate::synth;

fn dev() -> Device {
    Device::Cpu
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../face_loss_fixtures")
}

/// The committed torch-reference fixture, refusing one whose producer changed without a regen.
fn fixture() -> Value {
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
    f
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn read(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn scalar(t: &Tensor) -> f32 {
    t.to_scalar::<f32>().unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn shapes(v: &Value) -> Vec<(String, Vec<usize>)> {
    v.as_object()
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
        .collect()
}

fn arcface(f: &Value) -> ArcFace {
    let a = &f["arcface"];
    let seed = a["seed"].as_u64().unwrap();
    let map = shapes(&a["keys"])
        .into_iter()
        .map(|(k, s)| {
            let t = synth::tensor(seed, &k, &s, &dev()).unwrap();
            (k, t)
        })
        .collect();
    ArcFace::from_weights(&Weights::from_map(map).unwrap()).unwrap()
}

fn mesh_program(f: &Value) -> Program {
    let m = &f["facemesh"];
    let seed = m["seed"].as_u64().unwrap();
    let map = shapes(&m["param_shapes"])
        .into_iter()
        .map(|(k, s)| {
            let t = synth::tensor(seed, &k, &s, &dev()).unwrap();
            (k, t)
        })
        .collect();
    Program::new(
        candle_gen::gen_core::fx_program::ProgramSpec::from_value(&m["program"]).unwrap(),
        map,
        &dev(),
    )
    .unwrap()
}

fn images(f: &Value) -> (Tensor, Tensor) {
    let im = &f["image"];
    let (seed, h, w) = (
        im["seed"].as_u64().unwrap(),
        im["h"].as_u64().unwrap() as usize,
        im["w"].as_u64().unwrap() as usize,
    );
    (
        synth::image(seed, "live", h, w, &dev()).unwrap(),
        synth::image(seed, "reference", h, w, &dev()).unwrap(),
    )
}

fn bbox(f: &Value) -> [f32; 4] {
    let b = floats(&f["bbox"]);
    [b[0], b[1], b[2], b[3]]
}

struct StubDetector(Option<[f32; 4]>);
impl FaceBoxDetector for StubDetector {
    fn largest_face(&self, _: &[u8], _: usize, _: usize) -> Result<Option<[f32; 4]>> {
        Ok(self.0)
    }
}

/// The scorer over the fixture's synthetic ArcFace (its noise mean computed at construction, as at
/// load).
fn scorer(f: &Value, min_cos: f32, mode: IdentityReferenceMode) -> Arc<IdentityScorer> {
    Arc::new(IdentityScorer::new(arcface(f), min_cos, mode, &dev()).unwrap())
}

fn identity_loss(
    f: &Value,
    face: Option<[f32; 4]>,
    min_cos: f32,
    mode: IdentityReferenceMode,
) -> IdentityLoss {
    IdentityLoss::new(scorer(f, min_cos, mode), Arc::new(StubDetector(face)))
}

fn crop_of(f: &Value) -> CropBox {
    face_crop_box(bbox(f), 72, 88)
}

fn expect(v: &Value) -> f32 {
    v.as_f64().unwrap() as f32
}

fn noise_px(key: &str, h: usize, w: usize) -> Tensor {
    Tensor::from_vec(
        candle_gen::gen_core::train::face_loss::synth::noise_image(IDENTITY_NOISE_SEED, key, h, w),
        (1, h, w, 3),
        &dev(),
    )
    .unwrap()
}

/// AC3: the Candle identity path reproduces the torch reference embedding, the bias direction
/// (mean embedding of the 200 counter-based noise images) and the bias-centred cosine / loss (the
/// same numbers the MLX twin asserts). Mutations: transpose `rx` wrongly (use `rx` without `.t()`)
/// ⇒ shape/values red; drop the `2·px − 1` normalization ⇒ embedding red; skip the bias centring
/// ⇒ cos red.
#[test]
fn identity_path_matches_the_torch_reference() {
    let f = fixture();
    let (live, reference) = images(&f);
    let crop = crop_of(&f);
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
        (72, 88),
        Tensor::from_vec(e_ref.clone(), e_ref.len(), &dev()).unwrap(),
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

/// Bias centring takes a non-face toward 0: 112² noise images score below upstream's 0.2 gate
/// against the synthetic reference, matching torch. Mutation: score the raw (uncentred) cosine ⇒
/// values drift from the fixture ⇒ red.
#[test]
fn noise_scores_below_the_gate_against_the_reference() {
    let f = fixture();
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
        let cos = scalar(&(&c * &e_ref).unwrap().sum_all().unwrap());
        assert!((cos - want).abs() < 2e-5, "{key}: cos {cos} vs {want}");
        assert!(cos < 0.2, "{key}: noise scored {cos}");
    }
}

/// AC3 (landmarks): the Candle fx-program executor + landmark path reproduce torch's normalized
/// landmarks and region-weighted loss. Mutations: pad the channel axis on the wrong side ⇒ red;
/// drop the inner-eye scaling ⇒ red.
#[test]
fn landmark_path_matches_the_torch_reference() {
    let f = fixture();
    let (live, reference) = images(&f);
    let crop = crop_of(&f);
    let loss = FaceLandmarkLoss::new(mesh_program(&f), Arc::new(StubDetector(None)), None);
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

/// Identity "decoder": NCHW latents are already pixels (→ NHWC).
struct PixelDecoder;
impl X0Decoder for PixelDecoder {
    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        Ok(latents.permute((0, 2, 3, 1))?.contiguous()?)
    }
}

fn nchw(px: &Tensor) -> Tensor {
    px.permute((0, 3, 1, 2)).unwrap().contiguous().unwrap()
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

/// The tiny LoRA'd module `x0 = (I + B·A) · z` over the 3 channels of the NCHW latent `z`.
fn lora_x0(z: &Tensor, a: &Tensor, b: &Tensor) -> Tensor {
    let (_, c, h, w) = z.dims4().unwrap();
    let eye = Tensor::eye(3, candle_gen::candle_core::DType::F32, &dev()).unwrap();
    let m = (eye + b.matmul(a).unwrap()).unwrap();
    m.matmul(&z.reshape((c, h * w)).unwrap())
        .unwrap()
        .reshape((1, c, h, w))
        .unwrap()
}

/// `(aux value, |dL/dA|₁, |dL/dB|₁, noise level)` of one aux step through the path.
fn aux_value_and_grads(path: &PerceptualPath, z: &Tensor) -> (f32, f32, f32, f32) {
    let plan = path.plan(0, 0, 0.5).unwrap();
    assert!(plan.aux == vec![0], "{plan:?}");
    // Standard LoRA init: A random, B zero ⇒ x0 == z at step 0, gradient lands on B.
    let a = Var::from_tensor(&synth::tensor(7, "lora.a", &[2, 3], &dev()).unwrap()).unwrap();
    let b = Var::zeros((3, 2), candle_gen::candle_core::DType::F32, &dev()).unwrap();
    let x0 = lora_x0(z, a.as_tensor(), b.as_tensor());
    let terms = path.aux_loss(&plan, 0, &x0).unwrap().expect("aux term");
    let grads = terms.weighted.backward().unwrap();
    let norm = |v: &Var| {
        grads
            .get(v.as_tensor())
            .map(|g| {
                g.abs()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
            })
            .unwrap_or(0.0)
    };
    (
        scalar(&terms.weighted),
        norm(&a),
        norm(&b),
        plan.noise_level,
    )
}

/// AC1: with identity weight > 0 the aux term is `weight · t · (1 − cos)` — `cos` the bias-centred
/// cosine, equal to the torch fixture's, `t` the step's noise level (upstream's `t_ratio`) — and
/// back-propagates a nonzero gradient into the LoRA. Mutations: return `cos` instead of `1 − cos`
/// ⇒ value red; `detach()` the live embedding ⇒ zero LoRA gradient ⇒ red; `timestep_weight` → 1
/// ⇒ value red.
#[test]
fn identity_loss_is_one_minus_cos_and_trains_the_lora() {
    let f = fixture();
    let (live, reference) = images(&f);
    let loss = identity_loss(&f, Some(bbox(&f)), -1.0, IdentityReferenceMode::PerImage);
    assert_eq!(loss.timestep_weight(0.3), 0.3);
    let mut path = path_with(Box::new(loss));
    path.ensure_reference(0, &nchw(&reference)).unwrap();
    assert!(path.is_usable(0, 0).unwrap());
    let (value, _, gb, t) = aux_value_and_grads(&path, &nchw(&live));
    assert!(t > 0.05 && t < 0.95, "noise level {t}");
    let want = WEIGHT * t * expect(&f["arcface"]["identity_loss"]);
    assert!((value - want).abs() < 2e-5, "aux {value} vs {want}");
    assert!(gb > 1e-6, "no gradient reached the LoRA B factor: {gb}");
}

/// AC2: no detected face ⇒ unusable (diffusion fallback, no aux term); a live cosine at or below
/// `min_cos` ⇒ exactly zero loss and gradient, just above ⇒ not. Mutations: drop the `gt(min_cos)`
/// gate ⇒ red; return `Some` for a no-face reference ⇒ red.
#[test]
fn no_face_and_low_cos_samples_contribute_zero() {
    let f = fixture();
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
    for (min_cos, gated) in [(cos + 1e-3, true), (cos - 1e-3, false)] {
        let mut p = path_with(Box::new(identity_loss(
            &f,
            Some(bbox(&f)),
            min_cos,
            IdentityReferenceMode::PerImage,
        )));
        p.ensure_reference(0, &nchw(&reference)).unwrap();
        let (value, _, gb, _) = aux_value_and_grads(&p, &nchw(&live));
        if gated {
            assert_eq!(
                value, 0.0,
                "min_cos {min_cos} > cos {cos} must gate the loss"
            );
            assert_eq!(gb, 0.0, "a gated step must not move the LoRA");
        } else {
            assert!(value > 0.05 && gb > 1e-6, "{value} / {gb}");
        }
    }
}

/// Dataset-average mode targets the normalized mean of every face-bearing reference, frozen at the
/// first loss, and normalizes each image's loss by its own clean score — `max(0, 1 − cos / clean)`
/// — matching torch; a late reference is refused. Mutations: target the per-image embedding ⇒ cos
/// red; drop the clean-cos normalization ⇒ loss red.
#[test]
fn dataset_average_targets_the_mean_reference() {
    let f = fixture();
    let (live, reference) = images(&f);
    let reference2 = synth::image(
        f["image"]["seed"].as_u64().unwrap(),
        f["arcface"]["average"]["reference2_key"].as_str().unwrap(),
        72,
        88,
        &dev(),
    )
    .unwrap();
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
    assert!((got - want).abs() < 2e-5, "{got} vs {want}");
    assert!(loss.reference(&live).is_err());
}

/// The multi-frame mean divides by the number of frames whose gate is OPEN (`max(Σ gate, 1)`):
/// with one frame gated out the loss equals the passing frame's own. Mutations: divide by the
/// face-bearing frame count ⇒ the value halves ⇒ red; drop the gate from the numerator ⇒ red.
#[test]
fn a_gated_frame_leaves_the_multi_frame_mean() {
    let f = fixture();
    let (live, reference) = images(&f);
    let other = noise_px("gate-probe", 72, 88);
    let clip = |a: &Tensor, b: &Tensor| Tensor::cat(&[a, b], 0).unwrap();
    let face = Some(bbox(&f));
    let probe = identity_loss(&f, face, -1.0, IdentityReferenceMode::PerImage);
    let r = probe.reference(&reference).unwrap().unwrap();
    let rf = reference_as::<IdentityReference>("identity", r.as_ref())
        .unwrap()
        .frames[0]
        .as_ref()
        .unwrap();
    let cos_of = |px: &Tensor| scalar(&probe.scorer().frame_score(px, rf).unwrap().0);
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
        Arc::new(ScriptedDetector(std::sync::Mutex::new(vec![face, face]))),
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
    let f = fixture();
    let (live, reference) = images(&f);
    let pad = (88 / 4) as f32;
    let b = bbox(&f);
    let loss = IdentityLoss::new(
        scorer(&f, -1.0, IdentityReferenceMode::PerImage),
        Arc::new(PaddedOnly(
            [b[0] + pad, b[1] + pad, b[2] + pad, b[3] + pad],
            (72, 88),
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
/// no-face image. Mutations: `detach` the live landmarks ⇒ zero gradient ⇒ red;
/// `timestep_weight` → 1 ⇒ value red.
#[test]
fn landmark_loss_trains_the_lora_and_skips_no_face() {
    let f = fixture();
    let (live, reference) = images(&f);
    let lm = FaceLandmarkLoss::new(
        mesh_program(&f),
        Arc::new(StubDetector(Some(bbox(&f)))),
        None,
    );
    assert_eq!(lm.timestep_weight(0.3), 0.3);
    let mut path = path_with(Box::new(lm));
    path.ensure_reference(0, &nchw(&reference)).unwrap();
    let (value, _, gb, t) = aux_value_and_grads(&path, &nchw(&live));
    let want = WEIGHT * t * expect(&f["facemesh"]["landmark_loss"]);
    assert!(
        (value - want).abs() < 1e-4 * want.max(1.0),
        "{value} vs {want}"
    );
    assert!(gb > 1e-6);
    let mut skip = path_with(Box::new(FaceLandmarkLoss::new(
        mesh_program(&f),
        Arc::new(StubDetector(None)),
        None,
    )));
    skip.ensure_reference(0, &nchw(&reference)).unwrap();
    assert!(skip.plan(0, 0, 0.5).unwrap().aux.is_empty());
}

/// With the identity loss on, the landmark loss is gated on the identity cosine: a frame at or
/// below `min_cos` contributes zero loss and gradient, one above the ungated value. Mutation:
/// ignore the gate in `FaceLandmarkLoss::loss` ⇒ the gated value is nonzero ⇒ red.
#[test]
fn landmark_loss_is_gated_on_the_identity_cosine() {
    let f = fixture();
    let (live, reference) = images(&f);
    let cos = expect(&f["arcface"]["cos"]);
    for (min_cos, gated) in [(cos + 1e-3, true), (cos - 1e-3, false)] {
        let gate = scorer(&f, min_cos, IdentityReferenceMode::PerImage);
        let mut p = path_with(Box::new(FaceLandmarkLoss::new(
            mesh_program(&f),
            Arc::new(StubDetector(Some(bbox(&f)))),
            Some(gate),
        )));
        p.ensure_reference(0, &nchw(&reference)).unwrap();
        let (value, _, gb, t) = aux_value_and_grads(&p, &nchw(&live));
        if gated {
            assert_eq!(value, 0.0, "min_cos {min_cos} > cos {cos}");
            assert_eq!(gb, 0.0);
        } else {
            let want = WEIGHT * t * expect(&f["facemesh"]["landmark_loss"]);
            assert!(
                (value - want).abs() < 1e-4 * want.max(1.0),
                "{value} vs {want}"
            );
            assert!(gb > 1e-6);
        }
    }
}

/// A detector that answers per call from a script (one entry per call, in order).
struct ScriptedDetector(std::sync::Mutex<Vec<Option<[f32; 4]>>>);
impl FaceBoxDetector for ScriptedDetector {
    fn largest_face(&self, _: &[u8], _: usize, _: usize) -> Result<Option<[f32; 4]>> {
        Ok(candle_gen::lock_recover(&self.0).remove(0))
    }
}

/// A video decoder hands the loss `[F, H, W, 3]`: per-frame reference boxes, frames without a face
/// skipped, the mean over face-bearing frames, and a clip with no face unusable (the Candle twin of
/// mlx-gen-face's test; a miss is retried on the padded frame, so it consumes two script entries).
/// Mutations: average over every frame ⇒ red; score frame 0 for every frame ⇒ red.
#[test]
fn multi_frame_decodes_score_each_face_bearing_frame() {
    let f = fixture();
    let (live, reference) = images(&f);
    let clip = |a: &Tensor, b: &Tensor| Tensor::cat(&[a, b], 0).unwrap();
    let face = Some(bbox(&f));
    let shared = scorer(&f, -1.0, IdentityReferenceMode::PerImage);
    let loss = |script: Vec<Option<[f32; 4]>>| {
        IdentityLoss::new(
            shared.clone(),
            Arc::new(ScriptedDetector(std::sync::Mutex::new(script))),
        )
    };
    let single = expect(&f["arcface"]["identity_loss"]);
    let noise = synth::image(9, "noise", 72, 88, &dev()).unwrap();

    let l = loss(vec![face, None, None]);
    let r = l.reference(&clip(&reference, &reference)).unwrap().unwrap();
    let v = scalar(&l.loss(&clip(&live, &noise), r.as_ref()).unwrap());
    assert!((v - single).abs() < 2e-5, "{v} vs {single}");
    let l = loss(vec![None, None, face]);
    let r = l.reference(&clip(&reference, &reference)).unwrap().unwrap();
    let v = scalar(&l.loss(&clip(&noise, &live), r.as_ref()).unwrap());
    assert!((v - single).abs() < 2e-5, "{v} vs {single}");
    let l = loss(vec![face, face]);
    let r = l.reference(&clip(&reference, &reference)).unwrap().unwrap();
    let v = scalar(&l.loss(&clip(&live, &live), r.as_ref()).unwrap());
    assert!((v - single).abs() < 2e-5, "{v} vs {single}");
    assert!(loss(vec![None, None, None, None])
        .reference(&clip(&reference, &reference))
        .unwrap()
        .is_none());
    assert!(l.loss(&live, r.as_ref()).is_err());
}

/// A live decode at another size than its reference is an error.
#[test]
fn a_live_decode_of_another_size_is_refused() {
    let f = fixture();
    let (live, reference) = images(&f);
    let loss = identity_loss(&f, Some(bbox(&f)), -1.0, IdentityReferenceMode::PerImage);
    let r = loss.reference(&reference).unwrap().unwrap();
    let small = live.narrow(1, 0, 64).unwrap();
    assert!(loss.loss(&small, r.as_ref()).is_err());
}

/// The on-disk stand-ins load through the real loaders; the tiny ArcFace is the fixture's
/// architecture; the FaceMesh stand-in honours the real I/O contract. With the identity loss on,
/// the landmark loss's gate IS the identity loss's scorer; off, it is ungated. Mutation: build a
/// fresh scorer per loader (skip the `SCORERS` lookup) ⇒ `ptr_eq` red.
#[test]
fn testing_checkpoints_load_through_the_real_loaders() {
    let f = fixture();
    let want: std::collections::BTreeMap<_, _> =
        shapes(&f["arcface"]["keys"]).into_iter().collect();
    let got: std::collections::BTreeMap<_, _> =
        candle_gen::gen_core::train::face_loss::synth::tiny_arcface_shapes()
            .into_iter()
            .collect();
    assert_eq!(got, want);
    let tmp = tempfile::tempdir().unwrap();
    testing::write_face_stack(tmp.path()).unwrap();
    testing::write_facemesh(tmp.path()).unwrap();
    let cfg = IdentityLossConfig::default();
    let id = load_identity_loss(tmp.path(), &cfg, &dev()).unwrap();
    let lm = load_face_landmark_loss(tmp.path(), tmp.path(), Some(&cfg), &dev()).unwrap();
    assert!(Arc::ptr_eq(id.scorer(), lm.gate.as_ref().unwrap()));
    assert!(
        load_face_landmark_loss(tmp.path(), tmp.path(), None, &dev())
            .unwrap()
            .gate
            .is_none()
    );
    let (live, _) = images(&f);
    let out = lm
        .landmarks(&live, face_crop_box(bbox(&f), 72, 88))
        .unwrap();
    assert_eq!(out.dims(), &[478, 2]);
}

/// A max-pool kernel larger than its (padded) input is an error, never an underflow panic.
/// Mutation: drop the size check in `max_pool` ⇒ the subtraction underflows ⇒ panic ⇒ red.
#[test]
fn a_max_pool_kernel_larger_than_its_input_is_refused() {
    let spec = candle_gen::gen_core::fx_program::ProgramSpec::parse(
        r#"{"inputs":["x"],"outputs":["y"],"nodes":[{"op":"maxpool2d","out":"y","inputs":["x"],
            "kernel":[9,9],"stride":[1,1],"padding":[1,1]}]}"#,
    )
    .unwrap();
    let program = Program::new(spec, std::collections::HashMap::new(), &dev()).unwrap();
    let x = Tensor::zeros((1, 3, 4, 4), candle_gen::candle_core::DType::F32, &dev()).unwrap();
    let err = program.forward(&x).unwrap_err().to_string();
    assert!(err.contains("exceeds"), "{err}");
}

/// The REAL converted FaceMesh-v2 program reproduces the upstream torch model's output 0 on a
/// fixed input (the Candle twin of mlx-gen-face's test; same `$FACEMESH_REAL_DIR` layout).
#[test]
#[ignore = "needs the converted FaceMesh-v2 checkpoint + torch I/O in $FACEMESH_REAL_DIR"]
fn real_facemesh_program_matches_torch() {
    let dir = PathBuf::from(std::env::var("FACEMESH_REAL_DIR").expect("FACEMESH_REAL_DIR"));
    let program = Program::from_file(dir.join(FACEMESH_FILE), &dev()).unwrap();
    let io = candle_gen::candle_core::safetensors::load(
        dir.join("facemesh_torch_io.safetensors"),
        &dev(),
    )
    .unwrap();
    let got = read(&program.forward(&io["input"]).unwrap()[0]);
    let want = read(&io["output0"]);
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let d = max_abs_diff(&got, &want);
    println!("real FaceMesh max abs {d} (scale {scale})");
    assert!(d <= 1e-4 * scale, "real FaceMesh drift {d} (scale {scale})");
}
