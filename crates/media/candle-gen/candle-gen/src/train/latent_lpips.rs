//! **E-LatentLPIPS** (epic 2123, sc-24833) for the Candle trainers — the Candle twin of
//! `mlx_gen::train::latent_lpips`: the learned latent-space perceptual metric of Kang et al. (ECCV
//! 2024, `mingukkang/elatentlpips` @ f64cd65) as a frozen [`PerceptualLoss`] with
//! [`PerceptualInput::Latents`] input on the shared perceptual path ([`super::perceptual`]).
//!
//! Network: VGG16-BN whose first conv takes the family's latent channels and whose first three
//! max-pools are identities, sliced at torchvision feature indices `0-7 / 7-14 / 14-24 / 24-34 /
//! 34-44`; each slice output is unit-normalized over channels (`x / sqrt(Σc x² + 1e-8)`), the squared
//! difference of the two inputs' features goes through a bias-free 1×1 lin head and is spatially
//! averaged; the five terms are summed plus the mean absolute latent difference — exactly
//! ai-toolkit-perceptual's call (`normalize=False`, `add_l1_loss=True`, `ensembling=False`,
//! `augment=None`). BatchNorm runs in eval mode (`eps = 1e-5`).
//!
//! Every op is a candle op with a backward (stride-1 conv, affine, relu, max-pool, sqrt/sum/div,
//! abs), so the distance is differentiable in the live latent. Weights: the published
//! `elatentlpips_ckpt/<family>_latest_vgg16_tuned.pth` state dicts (read with candle's pickle
//! reader), or a same-stem `.safetensors`.

use std::any::Any;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

use crate::gen_core::train::LatentLpipsFamily;

use super::formula::{formula_tensor, Role};
use super::perceptual::{
    reference_as, AuxModelFootprint, LossReference, PerceptualInput, PerceptualLoss,
};
use crate::weights::Weights;
use crate::{CandleError, Result};

/// Output channels of the five trunk slices (`relu1_2 … relu5_3`).
pub const LPIPS_CHANNELS: [usize; 5] = [64, 128, 256, 512, 512];
const SLICE_BOUNDS: [(usize, usize); 5] = [(0, 7), (7, 14), (14, 24), (24, 34), (34, 44)];
/// The `vgg16_bn` "D" config: conv widths, `0` = max-pool.
const VGG16_CFG: [usize; 18] = [
    64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0,
];
const IDENTITY_POOLS: usize = 3;
const BN_EPS: f64 = 1e-5;
const NORMALIZE_EPS: f64 = 1e-8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerKind {
    Conv(usize, usize),
    Bn(usize),
    Relu,
    Identity,
    MaxPool,
}

fn trunk_layers(latent_channels: usize) -> Vec<LayerKind> {
    let mut out = Vec::with_capacity(44);
    let mut cin = latent_channels;
    let mut pools = 0;
    for &v in &VGG16_CFG {
        if v == 0 {
            out.push(if pools < IDENTITY_POOLS {
                LayerKind::Identity
            } else {
                LayerKind::MaxPool
            });
            pools += 1;
        } else {
            out.push(LayerKind::Conv(cin, v));
            out.push(LayerKind::Bn(v));
            out.push(LayerKind::Relu);
            cin = v;
        }
    }
    out
}

fn slice_of(index: usize) -> usize {
    SLICE_BOUNDS
        .iter()
        .position(|&(lo, hi)| (lo..hi).contains(&index))
        .expect("feature index < 44")
        + 1
}

fn layer_key(index: usize) -> String {
    format!("net.slice{}.{index}", slice_of(index))
}

fn checkpoint_tensors(latent_channels: usize) -> Vec<(String, Vec<usize>, Role)> {
    let mut out = Vec::new();
    for (i, layer) in trunk_layers(latent_channels).into_iter().enumerate() {
        let k = layer_key(i);
        match layer {
            LayerKind::Conv(cin, cout) => {
                out.push((format!("{k}.weight"), vec![cout, cin, 3, 3], Role::Conv));
                out.push((format!("{k}.bias"), vec![cout], Role::Bias));
            }
            LayerKind::Bn(c) => {
                out.push((format!("{k}.weight"), vec![c], Role::NormWeight));
                out.push((format!("{k}.bias"), vec![c], Role::Bias));
                out.push((format!("{k}.running_mean"), vec![c], Role::BnMean));
                out.push((format!("{k}.running_var"), vec![c], Role::BnVar));
            }
            _ => {}
        }
    }
    for (k, &c) in LPIPS_CHANNELS.iter().enumerate() {
        out.push((
            format!("lin{k}.model.1.weight"),
            vec![1, c, 1, 1],
            Role::Lin,
        ));
    }
    out
}

/// Exact parameter count of the network for `latent_channels`.
pub fn param_count(latent_channels: usize) -> u64 {
    checkpoint_tensors(latent_channels)
        .iter()
        .map(|(_, s, _)| s.iter().map(|&d| d as u64).product::<u64>())
        .sum()
}

/// The deterministic formula checkpoint (every key the network reads) on `device`.
pub fn formula_weights(latent_channels: usize, device: &Device) -> Result<Weights> {
    let mut map = std::collections::HashMap::new();
    for (k, shape, role) in checkpoint_tensors(latent_channels) {
        let t = formula_tensor(&k, &shape, role, device)?;
        map.insert(k, t);
    }
    Ok(Weights::from_map(map))
}

enum Op {
    Conv(Tensor, Tensor),
    /// Eval-mode BatchNorm folded to `x · scale + shift` (`[1, C, 1, 1]` each).
    Affine(Tensor, Tensor),
    Relu,
    MaxPool,
}

/// The loaded, frozen E-LatentLPIPS network.
pub struct LatentLpips {
    latent_channels: usize,
    slices: Vec<Vec<Op>>,
    /// Lin-head weights, `[1, C, 1, 1]` per slice.
    lins: Vec<Tensor>,
}

fn f32w(w: &Weights, key: &str) -> Result<Tensor> {
    Ok(w.require(key)?.to_dtype(DType::F32)?)
}

impl LatentLpips {
    /// Build from a checkpoint map (`net.slice{s}.{i}.*`, `lin{k}.model.1.weight`, torch layouts).
    pub fn from_weights(w: &Weights, latent_channels: usize) -> Result<Self> {
        let mut slices: Vec<Vec<Op>> = (0..5).map(|_| Vec::new()).collect();
        for (i, layer) in trunk_layers(latent_channels).into_iter().enumerate() {
            let k = layer_key(i);
            let op = match layer {
                LayerKind::Conv(cin, cout) => {
                    let cw = f32w(w, &format!("{k}.weight"))?;
                    if cw.dims() != [cout, cin, 3, 3] {
                        return Err(CandleError::Msg(format!(
                            "E-LatentLPIPS: {k}.weight has shape {:?}, expected [{cout}, {cin}, 3, 3] \
                             (a checkpoint for another latent family?)",
                            cw.dims()
                        )));
                    }
                    let b = f32w(w, &format!("{k}.bias"))?.reshape((1, cout, 1, 1))?;
                    Op::Conv(cw, b)
                }
                LayerKind::Bn(c) => {
                    let g = f32w(w, &format!("{k}.weight"))?;
                    let b = f32w(w, &format!("{k}.bias"))?;
                    let m = f32w(w, &format!("{k}.running_mean"))?;
                    let v = f32w(w, &format!("{k}.running_var"))?;
                    let scale = g.div(&(v + BN_EPS)?.sqrt()?)?;
                    let shift = b.sub(&m.mul(&scale)?)?;
                    Op::Affine(scale.reshape((1, c, 1, 1))?, shift.reshape((1, c, 1, 1))?)
                }
                LayerKind::Relu => Op::Relu,
                LayerKind::MaxPool => Op::MaxPool,
                LayerKind::Identity => continue,
            };
            slices[slice_of(i) - 1].push(op);
        }
        let lins = LPIPS_CHANNELS
            .iter()
            .enumerate()
            .map(|(k, &c)| Ok(f32w(w, &format!("lin{k}.model.1.weight"))?.reshape((1, c, 1, 1))?))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            latent_channels,
            slices,
            lins,
        })
    }

    /// Load `family`'s published checkpoint from `dir` onto `device` (see [`resolve_checkpoint`]).
    pub fn from_dir(
        dir: impl AsRef<Path>,
        family: LatentLpipsFamily,
        device: &Device,
    ) -> Result<Self> {
        let path = resolve_checkpoint(dir.as_ref(), family)?;
        let w = if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            Weights::from_file(&path, device, DType::F32)?
        } else {
            let mut map = std::collections::HashMap::new();
            for (k, t) in candle_core::pickle::read_all(&path)? {
                // Float tensors only (BatchNorm's int64 `num_batches_tracked` is never read).
                if t.dtype().is_float() {
                    map.insert(k, t.to_dtype(DType::F32)?.to_device(device)?);
                }
            }
            Weights::from_map(map)
        };
        Self::from_weights(&w, family.latent_channels()).map_err(|e| {
            CandleError::Msg(format!(
                "E-LatentLPIPS ({}) from {}: {e}",
                family.as_str(),
                path.display()
            ))
        })
    }

    /// Latent channels the first convolution consumes.
    pub fn latent_channels(&self) -> usize {
        self.latent_channels
    }

    fn features(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut h = x.clone();
        let mut out = Vec::with_capacity(5);
        for slice in &self.slices {
            for op in slice {
                h = match op {
                    Op::Conv(w, b) => h.conv2d(w, 1, 1, 1, 1)?.broadcast_add(b)?,
                    Op::Affine(s, t) => h.broadcast_mul(s)?.broadcast_add(t)?,
                    Op::Relu => h.relu()?,
                    Op::MaxPool => h.max_pool2d(2)?,
                };
            }
            let norm = (h.sqr()?.sum_keepdim(1)? + NORMALIZE_EPS)?.sqrt()?;
            out.push(h.broadcast_div(&norm)?);
        }
        Ok(out)
    }

    /// The per-layer terms `[5]` and the L1 term of `in0` vs `in1` (NCHW model-space latents), each
    /// averaged over the batch. Differentiable in both inputs.
    pub fn distance_terms(&self, in0: &Tensor, in1: &Tensor) -> Result<(Vec<Tensor>, Tensor)> {
        if in0.dims() != in1.dims() {
            return Err(CandleError::Msg(format!(
                "E-LatentLPIPS: input shapes differ ({:?} vs {:?})",
                in0.dims(),
                in1.dims()
            )));
        }
        let dims = in0.dims();
        if dims.len() != 4 || dims[1] != self.latent_channels {
            return Err(CandleError::Msg(format!(
                "E-LatentLPIPS: expected an NCHW latent with {} channels, got {dims:?}",
                self.latent_channels
            )));
        }
        let x0 = in0.to_dtype(DType::F32)?;
        let x1 = in1.to_dtype(DType::F32)?;
        let l1 = (&x0 - &x1)?.abs()?.mean_all()?;
        let f0 = self.features(&x0)?;
        let f1 = self.features(&x1)?;
        let mut layers = Vec::with_capacity(5);
        for ((a, b), lin) in f0.iter().zip(&f1).zip(&self.lins) {
            let d = (a - b)?;
            let weighted = d.sqr()?.broadcast_mul(lin)?.sum_keepdim(1)?; // [B, 1, h, w]
            layers.push(weighted.mean_all()?);
        }
        Ok((layers, l1))
    }

    /// E-LatentLPIPS(`in0`, `in1`) with the L1 term (batch-averaged scalar).
    pub fn distance(&self, in0: &Tensor, in1: &Tensor) -> Result<Tensor> {
        let (layers, l1) = self.distance_terms(in0, in1)?;
        let mut total = l1;
        for l in layers {
            total = (total + l)?;
        }
        Ok(total)
    }
}

/// The checkpoint file for `family` under `dir`: [`LatentLpipsFamily::checkpoint_file`], else the
/// same file name at the top level, else a same-stem `.safetensors`.
pub fn resolve_checkpoint(dir: &Path, family: LatentLpipsFamily) -> Result<PathBuf> {
    let rel = family.checkpoint_file();
    let name = Path::new(&rel)
        .file_name()
        .expect("checkpoint_file has a file name")
        .to_owned();
    let stem = Path::new(&name).with_extension("safetensors");
    [dir.join(&rel), dir.join(&name), dir.join(stem)]
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| {
            CandleError::Msg(format!(
                "E-LatentLPIPS: no {} checkpoint under {} (looked for {})",
                family.as_str(),
                dir.display(),
                rel
            ))
        })
}

/// The E-LatentLPIPS auxiliary loss: frozen network + the clean latent as the per-image reference.
pub struct LatentLpipsLoss {
    net: LatentLpips,
}

/// The per-image E-LatentLPIPS reference: the image's clean model-space latent (NCHW, detached).
pub struct LatentLpipsReference {
    pub latent: Tensor,
}

impl LatentLpipsLoss {
    /// Wrap a loaded network.
    pub fn new(net: LatentLpips) -> Self {
        Self { net }
    }

    /// Load `family`'s checkpoint from `dir` onto `device`.
    pub fn from_dir(
        dir: impl AsRef<Path>,
        family: LatentLpipsFamily,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self::new(LatentLpips::from_dir(dir, family, device)?))
    }

    /// The wrapped network.
    pub fn net(&self) -> &LatentLpips {
        &self.net
    }
}

impl PerceptualLoss for LatentLpipsLoss {
    fn name(&self) -> &'static str {
        "latent_lpips"
    }

    fn input(&self) -> PerceptualInput {
        PerceptualInput::Latents
    }

    fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
        let latent = clean.to_dtype(DType::F32)?.detach();
        Ok(Some(Box::new(LatentLpipsReference { latent })))
    }

    fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor> {
        let r = reference_as::<LatentLpipsReference>(self.name(), reference)?;
        self.net.distance(live, &r.latent)
    }
}

/// The pre-load memory figures of E-LatentLPIPS for `family` on a `latent_h × latent_w` latent
/// (epic 2123 E7) — identical arithmetic to the MLX twin: resident f32 weights; every trunk
/// activation (conv, BN, ReLU output per conv layer) for both inputs, ×2 for cotangents; one clean
/// latent per image.
pub fn latent_lpips_footprint(
    family: LatentLpipsFamily,
    latent_h: u32,
    latent_w: u32,
) -> AuxModelFootprint {
    let c = family.latent_channels();
    let mut pixels = latent_h as u64 * latent_w as u64;
    let mut floats = 0u64;
    for layer in trunk_layers(c) {
        match layer {
            LayerKind::Conv(_, cout) => floats += 3 * cout as u64 * pixels,
            LayerKind::MaxPool => pixels /= 4,
            _ => {}
        }
    }
    AuxModelFootprint {
        param_bytes: param_count(c) * 4,
        working_set_bytes: floats * 4 * 2 * 2,
        reference_bytes_per_image: c as u64 * latent_h as u64 * latent_w as u64 * 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::formula::{input_wave, PARITY_FIXTURE_REL};
    use crate::train::perceptual::{AuxLoss, AuxLossSchedule, PerceptualPath};
    use candle_core::Var;

    const CPU: Device = Device::Cpu;

    fn fixture() -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join(PARITY_FIXTURE_REL);
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn scalar(t: &Tensor) -> f32 {
        t.to_dtype(DType::F32).unwrap().to_scalar::<f32>().unwrap()
    }

    fn wave(
        c: usize,
        h: usize,
        w: usize,
        spec: &serde_json::Value,
        base: Option<&[f64]>,
    ) -> (Tensor, Vec<f64>) {
        let p: Vec<f64> = spec
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let mut v = input_wave(c * h * w, p[0], p[1], p[2]);
        if let Some(b) = base {
            for (x, y) in v.iter_mut().zip(b) {
                *x += y;
            }
        }
        let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        (Tensor::from_vec(f, (1, c, h, w), &CPU).unwrap(), v)
    }

    fn net(c: usize) -> LatentLpips {
        LatentLpips::from_weights(&formula_weights(c, &CPU).unwrap(), c).unwrap()
    }

    fn randn(seed: u64, shape: (usize, usize, usize, usize)) -> Tensor {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        let v: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        Tensor::from_vec(v, shape, &CPU).unwrap()
    }

    /// Parity with the upstream PyTorch `ELatentLPIPS.forward` on the formula weights (the same
    /// fixture the MLX twin asserts — so the two backends agree with each other through it): total,
    /// each layer term, the L1 term and the self-distance, for 4- and 16-channel families.
    /// Mutations: `IDENTITY_POOLS = 2`; drop the channel normalization; drop the L1 term ⇒ red.
    #[test]
    fn matches_the_upstream_reference_on_formula_weights() {
        let fx = fixture();
        for case in fx["latent_lpips"].as_array().unwrap() {
            let c = case["latent_channels"].as_u64().unwrap() as usize;
            let (h, w) = (
                case["h"].as_u64().unwrap() as usize,
                case["w"].as_u64().unwrap() as usize,
            );
            let (in0, base) = wave(c, h, w, &case["inputs"]["in0"], None);
            let (in1, _) = wave(c, h, w, &case["inputs"]["in1_delta"], Some(&base));
            let n = net(c);
            let (layers, l1) = n.distance_terms(&in0, &in1).unwrap();
            let close = |got: f32, want: f64, what: &str| {
                let tol = 2e-4 * want.abs().max(1e-3);
                assert!(
                    (got as f64 - want).abs() <= tol,
                    "{} {what}: {got} vs upstream {want}",
                    case["family"]
                );
            };
            for (k, l) in layers.iter().enumerate() {
                close(
                    scalar(l),
                    case["per_layer"][k].as_f64().unwrap(),
                    &format!("layer {k}"),
                );
            }
            close(scalar(&l1), case["l1"].as_f64().unwrap(), "l1");
            close(
                scalar(&n.distance(&in0, &in1).unwrap()),
                case["value"].as_f64().unwrap(),
                "total",
            );
            assert_eq!(scalar(&n.distance(&in0, &in0).unwrap()), 0.0);
        }
    }

    /// AC2: zero for identical latents, positive (its learned layer terms too) and growing with the
    /// perturbation, and differentiable in the live latent through candle's autograd. Mutation:
    /// subtract the layer terms in `distance` ⇒ the distance falls below its L1 term ⇒ red.
    #[test]
    fn is_zero_for_identical_latents_and_positive_when_perturbed() {
        let n = net(4);
        let x = randn(3, (1, 4, 8, 8));
        let d = randn(4, (1, 4, 8, 8));
        assert_eq!(scalar(&n.distance(&x, &x).unwrap()), 0.0);
        let small = (&x + (&d * 0.05).unwrap()).unwrap();
        let big = (&x + (&d * 0.5).unwrap()).unwrap();
        let (s, b) = (
            scalar(&n.distance(&small, &x).unwrap()),
            scalar(&n.distance(&big, &x).unwrap()),
        );
        assert!(s > 0.0 && b > s, "small {s} big {b}");
        let (layers, l1) = n.distance_terms(&big, &x).unwrap();
        for (k, l) in layers.iter().enumerate() {
            assert!(scalar(l) > 0.0, "layer {k}");
        }
        assert!(b > scalar(&l1), "distance {b} must exceed its L1 term");
        let live = Var::from_tensor(&small).unwrap();
        let grads = n
            .distance(live.as_tensor(), &x)
            .unwrap()
            .backward()
            .unwrap();
        let g = grads
            .get(live.as_tensor())
            .expect("gradient reaches the live latent");
        assert!(scalar(&g.abs().unwrap().sum_all().unwrap()) > 0.0);
    }

    /// A checkpoint for another latent family is refused; a missing checkpoint is named.
    #[test]
    fn a_checkpoint_for_another_family_is_refused() {
        let err = LatentLpips::from_weights(&formula_weights(16, &CPU).unwrap(), 4)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("another latent family"), "{err}");
        let tmp = tempfile::tempdir().unwrap();
        let err = LatentLpips::from_dir(tmp.path(), LatentLpipsFamily::Sd3, &CPU)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("no sd3 checkpoint"), "{err}");
    }

    /// The published `.pth` layout loads through candle's pickle reader: a real `torch.save`d
    /// state dict (BatchNorm counters included) placed at the published relative path for a
    /// stand-in family is found by `resolve_checkpoint` and its float tensors read exactly.
    #[test]
    fn reads_the_published_pth_layout() {
        use crate::train::formula::formula_values;
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../../contracts/gen-core/tests/fixtures/latent_perceptual/tiny_bn_state_dict.pth",
        );
        let tmp = tempfile::tempdir().unwrap();
        let dst = tmp.path().join(LatentLpipsFamily::Sdxl.checkpoint_file());
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::fs::copy(&fixture, &dst).unwrap();
        assert_eq!(
            resolve_checkpoint(tmp.path(), LatentLpipsFamily::Sdxl).unwrap(),
            dst
        );
        let sd: std::collections::HashMap<_, _> = candle_core::pickle::read_all(&dst)
            .unwrap()
            .into_iter()
            .collect();
        let w = sd["0.weight"]
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(w, formula_values("0.weight", &[2, 4, 3, 3], Role::Conv));
        // Not an E-LatentLPIPS checkpoint ⇒ a named load error, never a silent run.
        let err = LatentLpips::from_dir(tmp.path(), LatentLpipsFamily::Sdxl, &CPU)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("E-LatentLPIPS (sdxl)"), "{err}");
    }

    /// The parameter count is the published checkpoints' size: both files exceed `param_count · 4`
    /// by the same 35 446 bytes (pickle, zip headers, int64 counters), so the window catches a
    /// missing lin head (5.9 KB) or BN running stat (17 KB). Mutation: drop the lin heads or the
    /// running variances from `checkpoint_tensors` ⇒ red.
    #[test]
    fn param_count_matches_the_published_checkpoint_sizes() {
        for (c, bytes) in [(4usize, 58_969_974u64), (16, 58_997_622)] {
            let ours = param_count(c) * 4;
            assert!(
                ours <= bytes && (30_000..40_000).contains(&(bytes - ours)),
                "{c}: {ours} vs {bytes}"
            );
        }
    }

    /// Runs on the shared path without a decoder; zero at the clean latent, positive off it, and
    /// silent outside its window. Mutation: report `DecodedPixels` ⇒ `PerceptualPath::new` refuses.
    #[test]
    fn runs_on_the_shared_path_without_a_decoder() {
        let mut path = PerceptualPath::new(
            None,
            vec![AuxLoss {
                schedule: AuxLossSchedule {
                    weight: 0.5,
                    t_min: 0.0,
                    t_max: 0.5,
                    every_n: 1,
                },
                loss: Box::new(LatentLpipsLoss::new(net(4))),
            }],
        )
        .unwrap();
        let clean = randn(9, (1, 4, 8, 8));
        path.ensure_reference(0, &clean).unwrap();
        let plan = path.plan(1, 0, 0.3).unwrap();
        assert!(plan.diffusion && plan.aux == vec![0], "{plan:?}");
        assert_eq!(
            scalar(&path.aux_loss(&plan, 0, &clean).unwrap().unwrap().weighted),
            0.0
        );
        let off = (&clean + 0.2).unwrap();
        assert!(scalar(&path.aux_loss(&plan, 0, &off).unwrap().unwrap().weighted) > 0.0);
        assert!(path.plan(1, 0, 0.8).unwrap().aux.is_empty());
    }

    /// E7: weights + a working set that scales with the latent + one latent per image; identical
    /// arithmetic to the MLX twin. Mutation: drop the reference term ⇒ red.
    #[test]
    fn footprint_scales_with_the_latent() {
        let a = latent_lpips_footprint(LatentLpipsFamily::Flux, 64, 64);
        let b = latent_lpips_footprint(LatentLpipsFamily::Flux, 128, 128);
        assert_eq!(a.param_bytes, param_count(16) * 4);
        assert_eq!(a.reference_bytes_per_image, 16 * 64 * 64 * 4);
        assert!(b.working_set_bytes == 4 * a.working_set_bytes && a.working_set_bytes > 0);
    }
}
