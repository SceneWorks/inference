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

fn identity_loss(
    f: &Value,
    face: Option<[f32; 4]>,
    min_cos: f32,
    mode: IdentityReferenceMode,
) -> IdentityLoss {
    IdentityLoss::new(arcface(f), Arc::new(StubDetector(face)), min_cos, mode)
}

/// AC3: the Candle identity path reproduces the torch reference embedding, cosine and loss (the
/// same numbers the MLX twin asserts). Mutations: transpose `rx` wrongly (use `rx` without `.t()`)
/// ⇒ shape/values red; drop the `2·px − 1` normalization ⇒ embedding red.
#[test]
fn identity_path_matches_the_torch_reference() {
    let f = fixture();
    let (live, reference) = images(&f);
    let crop = face_crop_box(bbox(&f), 72, 88);
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
    assert!(
        d_live < 2e-5 && d_ref < 2e-5,
        "embedding drift {d_live} / {d_ref}"
    );
    let r = IdentityReference {
        crop,
        image_hw: (72, 88),
        embedding: Tensor::from_vec(e_ref.clone(), e_ref.len(), &dev()).unwrap(),
    };
    let (l, cos) = loss.loss_and_cos(&live, &r).unwrap();
    let want_cos = f["arcface"]["cos"].as_f64().unwrap() as f32;
    assert!((scalar(&cos) - want_cos).abs() < 2e-5);
    let want = f["arcface"]["identity_loss"].as_f64().unwrap() as f32;
    assert!(
        (scalar(&l) - want).abs() < 2e-5,
        "loss {} vs {want}",
        scalar(&l)
    );
}

/// AC3 (landmarks): the Candle fx-program executor + landmark path reproduce torch's normalized
/// landmarks and region-weighted loss. Mutations: pad the channel axis on the wrong side ⇒ red;
/// drop the inner-eye scaling ⇒ red.
#[test]
fn landmark_path_matches_the_torch_reference() {
    let f = fixture();
    let (live, reference) = images(&f);
    let crop = face_crop_box(bbox(&f), 72, 88);
    let loss = FaceLandmarkLoss::new(mesh_program(&f), Arc::new(StubDetector(None)));
    let l_live = loss.landmarks(&live, crop).unwrap();
    let l_ref = loss.landmarks(&reference, crop).unwrap();
    let want = floats(&f["facemesh"]["landmarks"]);
    let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let d = max_abs_diff(&read(&l_live), &want);
    assert!(d < 1e-4 * scale, "landmark drift {d} (scale {scale})");
    let got = scalar(&landmark_distance(&l_live, &l_ref).unwrap());
    let want = f["facemesh"]["landmark_loss"].as_f64().unwrap() as f32;
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

/// `(aux value, |dL/dA|₁, |dL/dB|₁)` of one aux step through the path.
fn aux_value_and_grads(path: &PerceptualPath, z: &Tensor) -> (f32, f32, f32) {
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
    (scalar(&terms.weighted), norm(&a), norm(&b))
}

/// AC1: with identity weight > 0 the aux term is `weight · (1 − cos(embed(x0_face), ref))` — equal
/// to the torch fixture's `1 − cos` — and back-propagates a nonzero gradient into the LoRA.
/// Mutations: return `cos` instead of `1 − cos` ⇒ value red; `detach()` the live embedding ⇒ zero
/// LoRA gradient ⇒ red.
#[test]
fn identity_loss_is_one_minus_cos_and_trains_the_lora() {
    let f = fixture();
    let (live, reference) = images(&f);
    let mut path = path_with(Box::new(identity_loss(
        &f,
        Some(bbox(&f)),
        -1.0,
        IdentityReferenceMode::PerImage,
    )));
    path.ensure_reference(0, &nchw(&reference)).unwrap();
    assert!(path.is_usable(0, 0).unwrap());
    let (value, _, gb) = aux_value_and_grads(&path, &nchw(&live));
    let want = WEIGHT * f["arcface"]["identity_loss"].as_f64().unwrap() as f32;
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
    for (min_cos, gated) in [(cos + 1e-3, true), (cos - 1e-3, false)] {
        let mut p = path_with(Box::new(identity_loss(
            &f,
            Some(bbox(&f)),
            min_cos,
            IdentityReferenceMode::PerImage,
        )));
        p.ensure_reference(0, &nchw(&reference)).unwrap();
        let (value, _, gb) = aux_value_and_grads(&p, &nchw(&live));
        if gated {
            assert_eq!(
                value, 0.0,
                "min_cos {min_cos} > cos {cos} must gate the loss"
            );
            assert_eq!(gb, 0.0, "a gated step must not move the LoRA");
        } else {
            assert!(value > 0.1 && gb > 1e-6, "{value} / {gb}");
        }
    }
}

/// Dataset-average mode targets the normalized mean reference, frozen at the first loss; a late
/// reference is refused. Mutation: target the per-image embedding in average mode ⇒ red.
#[test]
fn dataset_average_targets_the_mean_reference() {
    let f = fixture();
    let (live, reference) = images(&f);
    let loss = identity_loss(
        &f,
        Some(bbox(&f)),
        -1.0,
        IdentityReferenceMode::DatasetAverage,
    );
    let r_live = loss.reference(&live).unwrap().unwrap();
    let r_ref = loss.reference(&reference).unwrap().unwrap();
    let r_ref = reference_as::<IdentityReference>("identity", r_ref.as_ref()).unwrap();
    let e_live = &reference_as::<IdentityReference>("identity", r_live.as_ref())
        .unwrap()
        .embedding;
    let mean = l2_normalize(&(e_live + &r_ref.embedding).unwrap()).unwrap();
    let want = 1.0 - scalar(&(&mean * e_live).unwrap().sum_all().unwrap());
    let got = scalar(&loss.loss(&live, r_ref).unwrap());
    assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    let per_image = f["arcface"]["identity_loss"].as_f64().unwrap() as f32;
    assert!((got - per_image).abs() > 1e-3);
    assert!(loss.reference(&live).is_err());
}

/// The landmark loss is differentiable into the LoRA and skips a no-face image. Mutation: `detach`
/// the live landmarks ⇒ zero gradient ⇒ red.
#[test]
fn landmark_loss_trains_the_lora_and_skips_no_face() {
    let f = fixture();
    let (live, reference) = images(&f);
    let mut path = path_with(Box::new(FaceLandmarkLoss::new(
        mesh_program(&f),
        Arc::new(StubDetector(Some(bbox(&f)))),
    )));
    path.ensure_reference(0, &nchw(&reference)).unwrap();
    let (value, _, gb) = aux_value_and_grads(&path, &nchw(&live));
    let want = WEIGHT * f["facemesh"]["landmark_loss"].as_f64().unwrap() as f32;
    assert!(
        (value - want).abs() < 1e-4 * want.max(1.0),
        "{value} vs {want}"
    );
    assert!(gb > 1e-6);
    let mut skip = path_with(Box::new(FaceLandmarkLoss::new(
        mesh_program(&f),
        Arc::new(StubDetector(None)),
    )));
    skip.ensure_reference(0, &nchw(&reference)).unwrap();
    assert!(skip.plan(0, 0, 0.5).unwrap().aux.is_empty());
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
/// architecture; the FaceMesh stand-in honours the real I/O contract.
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
    load_identity_loss(tmp.path(), &IdentityLossConfig::default(), &dev()).unwrap();
    let lm = load_face_landmark_loss(tmp.path(), tmp.path(), &dev()).unwrap();
    let (live, _) = images(&f);
    let out = lm
        .landmarks(&live, face_crop_box(bbox(&f), 72, 88))
        .unwrap();
    assert_eq!(out.dims(), &[478, 2]);
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
