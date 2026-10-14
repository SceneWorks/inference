//! The degradation pipeline against HYPIR's own implementation (sc-25688): the committed fixture
//! `mlx-gen-iris/tests/fixtures/iris_restoration_train_golden.safetensors`
//! (`crates/media/mlx-gen/tools/dump_iris_restoration_train.py`) records HYPIR's kernels and three
//! scripted runs of `RealESRGANBatchTransform.__call__` — every op's input and output, every
//! per-pixel draw. Each native op is checked on the recorded input of its reference op (so an
//! error never hides behind an earlier one), then the whole chain end to end on the recorded draws.

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::Value;

use super::*;

struct Fixture {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    meta: HashMap<String, String>,
}

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../media/mlx-gen/mlx-gen-iris/tests/fixtures/iris_restoration_train_golden.safetensors",
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let meta = header.metadata().clone().unwrap_or_default();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let tensors = st
        .tensors()
        .into_iter()
        .map(|(k, v)| {
            let data = v
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            (k, (v.shape().to_vec(), data))
        })
        .collect();
    Fixture { tensors, meta }
}

impl Fixture {
    fn get(&self, k: &str) -> &(Vec<usize>, Vec<f32>) {
        self.tensors
            .get(k)
            .unwrap_or_else(|| panic!("fixture has no {k}"))
    }
    fn planes(&self, k: &str) -> Planes {
        let (shape, data) = self.get(k);
        Planes {
            channels: shape[0],
            height: shape[1],
            width: shape[2],
            data: data.clone(),
        }
    }
    fn json(&self, k: &str) -> Value {
        serde_json::from_str(&self.meta[k]).unwrap()
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// (max |Δ|, mean |Δ|, fraction of |Δ| > 1e-3)
fn stats(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    assert_eq!(a.len(), b.len(), "length");
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
    let n = d.len() as f32;
    (
        d.iter().copied().fold(0.0, f32::max),
        d.iter().sum::<f32>() / n,
        d.iter().filter(|v| **v > 1e-3).count() as f32 / n,
    )
}

#[test]
fn kernels_match_basicsr() {
    let fx = fixture();
    let cases = fx.json("kernels");
    for (i, c) in cases.as_array().unwrap().iter().enumerate() {
        let kind = c["kind"].as_str().unwrap();
        let size = c["size"].as_u64().unwrap() as usize;
        let f = |k: &str| c[k].as_f64().unwrap();
        let k = if kind == "sinc" {
            pad_kernel(&sinc_kernel_f64(f("sx"), size), size, KERNEL_SIDE)
        } else {
            let p = BlurKernelParams {
                kind: KernelType::parse(kind).unwrap(),
                size,
                sigma_x: f("sx"),
                sigma_y: f("sy"),
                theta: f("theta"),
                beta: f("beta"),
            };
            pad_kernel(&blur_kernel_f64(&p), size, KERNEL_SIDE)
        };
        let want = &fx.get(&format!("kernels/{i}")).1;
        let d = max_abs(&k, want);
        assert!(d < 1e-8, "kernel {i} ({kind}): max |Δ| {d}");
    }
    let d = max_abs(&usm_kernel(), &fx.get("kernels/usm").1);
    assert!(d < 1e-9, "usm kernel: {d}");
}

/// Replays the reference's recorded draws in order.
struct Replay {
    normals: Vec<f32>,
    poisson: Vec<f32>,
    n: usize,
    p: usize,
}

impl NoiseSource for Replay {
    fn normals(&mut self, n: usize) -> Vec<f32> {
        let out = self.normals[self.n..self.n + n].to_vec();
        self.n += n;
        out
    }
    fn poisson(&mut self, rates: &[f32]) -> Vec<f32> {
        let out = self.poisson[self.p..self.p + rates.len()].to_vec();
        self.p += rates.len();
        out
    }
}

fn mode(s: &str) -> Interp {
    match s {
        "area" => Interp::Area,
        "bilinear" => Interp::Bilinear,
        "bicubic" => Interp::Bicubic,
        o => panic!("mode {o}"),
    }
}

fn noise_draw(v: &Value) -> NoiseDraw {
    let level = v["level"].as_f64().unwrap() as f32;
    let gray = v["gray"].as_bool().unwrap();
    match v["noise"].as_str().unwrap() {
        "gaussian" => NoiseDraw::Gaussian { sigma: level, gray },
        _ => NoiseDraw::Poisson { scale: level, gray },
    }
}

fn replay(fx: &Fixture, chain: &str) -> Replay {
    Replay {
        normals: fx.get(&format!("chain/{chain}/normals")).1.clone(),
        poisson: fx.get(&format!("chain/{chain}/poisson")).1.clone(),
        n: 0,
        p: 0,
    }
}

#[test]
fn every_pipeline_op_matches_hypir_on_its_recorded_input() {
    let fx = fixture();
    let chains = fx.json("chains");
    let draws = fx.json("draws");
    for (name, ops) in chains.as_object().unwrap() {
        let mut noise = replay(&fx, name);
        let mut noise_stage = 0;
        for (i, op) in ops.as_array().unwrap().iter().enumerate() {
            let pre = format!("chain/{name}/op{i}");
            let input = fx.planes(&format!("{pre}/in"));
            let want = fx.planes(&format!("{pre}/out"));
            let kind = op["kind"].as_str().unwrap();
            let got = match kind {
                "usm" => usm_sharpen(&input).unwrap(),
                "filter2d" => filter2d(&input, &fx.get(&format!("{pre}/kernel")).1).unwrap(),
                "interpolate" => {
                    let m = mode(op["mode"].as_str().unwrap());
                    if let Some(s) = op.get("scale_factor") {
                        interpolate_scale(&input, s.as_f64().unwrap(), m).unwrap()
                    } else {
                        let sz = op["size"].as_array().unwrap();
                        interpolate_size(
                            &input,
                            sz[0].as_u64().unwrap() as usize,
                            sz[1].as_u64().unwrap() as usize,
                            m,
                        )
                        .unwrap()
                    }
                }
                "gaussian" | "poisson" => {
                    noise_stage += 1;
                    let d = noise_draw(&draws[name][format!("s{noise_stage}")]);
                    add_noise(&input, d, &mut noise)
                }
                "jpeg" => jpeg(&input, op["quality"].as_f64().unwrap() as f32),
                o => panic!("op {o}"),
            };
            assert_eq!(
                (got.height, got.width),
                (want.height, want.width),
                "{name} op{i} {kind} size"
            );
            let (max, mean, frac) = stats(&got.data, &want.data);
            // JPEG re-quantizes: a coefficient that lands within float noise of a rounding
            // boundary may round the other way (one quantization step on that block).
            let ok = match kind {
                "jpeg" => mean < 2e-4 && frac < 0.02,
                _ => max < 2e-5,
            };
            assert!(
                ok,
                "{name} op{i} {kind}: max {max:.3e} mean {mean:.3e} frac>1e-3 {frac:.4}"
            );
        }
        assert_eq!(
            noise.n,
            noise.normals.len(),
            "{name}: every normal consumed"
        );
        assert_eq!(
            noise.p,
            noise.poisson.len(),
            "{name}: every poisson consumed"
        );
    }
}

fn stage(v: &Value) -> StageDraw {
    StageDraw {
        resize: ResizeDraw {
            scale: v["scale"].as_f64().unwrap(),
            mode: mode(v["mode"].as_str().unwrap()),
        },
        noise: noise_draw(v),
        jpeg_quality: v["jpeg"].as_f64().unwrap() as f32,
    }
}

#[test]
fn the_whole_chain_matches_hypir_on_the_recorded_draws() {
    let fx = fixture();
    let draws = fx.json("draws");
    for name in ["a", "b", "c"] {
        let d = &draws[name];
        let draw = DegradationDraw {
            hflip: false,
            vflip: false,
            rot90: false,
            kernel1: fx.get(&format!("chain/{name}/k1")).1.clone(),
            kernel2: fx.get(&format!("chain/{name}/k2")).1.clone(),
            sinc: fx.get(&format!("chain/{name}/sinc")).1.clone(),
            stage1: stage(&d["s1"]),
            second_blur: d["second_blur"].as_bool().unwrap(),
            stage2: stage(&d["s2"]),
            sinc_before_jpeg: d["sinc_first"].as_bool().unwrap(),
            back_mode: mode(d["back"].as_str().unwrap()),
            noise_key: 0,
        };
        let hq = fx.planes(&format!("chain/{name}/hq"));
        let pair = degrade_with(
            &DegradationConfig::default(),
            &hq,
            &draw,
            &mut replay(&fx, name),
        )
        .unwrap();
        let gt = fx.planes(&format!("chain/{name}/gt"));
        let lq = fx.planes(&format!("chain/{name}/lq"));
        let (gmax, _, _) = stats(&pair.gt.data, &gt.data);
        assert!(gmax < 2e-5, "{name}: GT (USM) max {gmax:.3e}");
        let (max, mean, frac) = stats(&pair.lq.data, &lq.data);
        // 8-bit output: float drift can move a value across a rounding boundary (1/255).
        assert!(
            mean < 1e-3 && frac < 0.05,
            "{name}: LQ max {max:.3e} mean {mean:.3e} frac>1e-3 {frac:.4}"
        );
        eprintln!("chain {name}: LQ max {max:.3e} mean {mean:.3e} frac>1e-3 {frac:.4}");
    }
}
