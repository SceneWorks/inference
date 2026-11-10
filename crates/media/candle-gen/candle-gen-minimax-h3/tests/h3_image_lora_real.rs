//! SW-7096: MiniMax-H3-Image LoRAs on real weights (candle/CUDA) — the diffusers/PEFT, kohya
//! `lora_unet_` and ComfyUI layouts of the same synthetic factors fold identically onto the real
//! image DiT's projections, the fold moves them, and ten stacked files keep their own scales.
//!
//! `#[ignore]`d: needs `MiniMaxAI/MiniMax-H3-Image` (`MINIMAX_H3_IMAGE_SNAPSHOT`) and CUDA. The
//! `minimax-h3-image` real-weights profile runs it on the `candle-minimax-h3-image` job. It
//! asserts on its inputs rather than skipping. The LoRA is generated deterministically from a seed
//! and written to `RUNNER_TEMP`; nothing is downloaded.
//!
//! The comparison is per projection — each adapted leaf's output on the same input — over the
//! first [`BLOCKS`] blocks. That is where a layout's conversion (the fused qkv split, the SwiGLU
//! half swap, the alpha) can go wrong; the block arithmetic around the leaves is unchanged by a
//! LoRA's spelling.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen_minimax_h3::adapters::{apply_minimax_h3_adapters, BLOCK_TARGETS};
use candle_gen_minimax_h3::{MiniMaxH3Dit, MiniMaxH3DitConfig};

const BLOCKS: usize = 2;
const RANK: usize = 4;
const REL_TOL: f32 = 2e-2;

fn image_snapshot() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_IMAGE_SNAPSHOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_SNAPSHOT must point at a MiniMaxAI/MiniMax-H3-Image snapshot root; this \
         test is #[ignore]d and asserts rather than skips."
    );
    let path = PathBuf::from(raw);
    assert!(
        path.join("transformer_diffusers")
            .join("config.json")
            .is_file(),
        "MINIMAX_H3_IMAGE_SNAPSHOT={} has no transformer_diffusers/config.json",
        path.display()
    );
    path
}

fn scratch() -> tempfile::TempDir {
    match std::env::var("RUNNER_TEMP") {
        Ok(dir) if !dir.is_empty() => tempfile::tempdir_in(dir).expect("scratch in RUNNER_TEMP"),
        _ => tempfile::tempdir().expect("scratch"),
    }
}

fn seeded(shape: &[usize], seed: u32) -> Tensor {
    let n: usize = shape.iter().product();
    let values: Vec<f32> = (0..n as u32)
        .map(|i| {
            let h = i.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503).wrapping_add(17);
            ((h % 2003) as f32 / 2003.0 - 0.5) * 0.02
        })
        .collect();
    Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
}

fn leaf_shape(cfg: &MiniMaxH3DitConfig, leaf: &str) -> (usize, usize) {
    let (hidden, inner, ffn) = (
        cfg.hidden_size,
        cfg.num_attention_heads * cfg.attention_head_dim,
        cfg.ffn_dim,
    );
    match leaf {
        "attn.to_q" | "attn.to_k" | "attn.to_v" => (inner, hidden),
        "attn.to_out.0" => (hidden, inner),
        "ff.net.0.proj" => (2 * ffn, hidden),
        "ff.net.2" => (hidden, ffn),
        _ => unreachable!("{leaf}"),
    }
}

fn factors(cfg: &MiniMaxH3DitConfig, block: usize, leaf: &str, seed: u32) -> (Tensor, Tensor) {
    let (out, inp) = leaf_shape(cfg, leaf);
    let k = seed
        + 1000 * block as u32
        + 10 * BLOCK_TARGETS.iter().position(|t| *t == leaf).unwrap() as u32;
    (seeded(&[RANK, inp], k), seeded(&[out, RANK], k + 1))
}

fn write(path: &Path, tensors: &[(String, Tensor)], meta: &[(&str, &str)]) {
    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(k, t)| {
            let values = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let raw = values.into_iter().flat_map(f32::to_le_bytes).collect();
            (k.clone(), t.dims().to_vec(), raw)
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView)> = bytes
        .iter()
        .map(|(k, shape, raw)| {
            (
                k.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), raw)
                    .unwrap(),
            )
        })
        .collect();
    let meta: HashMap<String, String> = meta
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    safetensors::serialize_to_file(views, &Some(meta), path).expect("write the synthetic LoRA");
}

fn scalar(v: f32) -> Tensor {
    Tensor::new(&[v], &Device::Cpu).unwrap()
}

fn write_peft(cfg: &MiniMaxH3DitConfig, dir: &Path, seed: u32) -> PathBuf {
    let mut t = Vec::new();
    for b in 0..BLOCKS {
        for leaf in BLOCK_TARGETS {
            let (down, up) = factors(cfg, b, leaf, seed);
            let stem = format!("transformer.transformer_blocks.{b}.{leaf}");
            t.push((format!("{stem}.lora_A.weight"), down));
            t.push((format!("{stem}.lora_B.weight"), up));
        }
    }
    let path = dir.join(format!("peft-{seed}.safetensors"));
    write(&path, &t, &[("alpha", &RANK.to_string())]);
    path
}

/// The native-module form: fused qkv (concatenated `A`, block-diagonal `B`, alpha × 3), `mlp.fc1`
/// with native `[gate | value]` halves, and the two renames — kohya-flattened or ComfyUI.
fn write_native(cfg: &MiniMaxH3DitConfig, dir: &Path, seed: u32, kohya: bool) -> PathBuf {
    let stem = |b: usize, leaf: &str| {
        if kohya {
            format!("lora_unet_blocks_{b}_{}", leaf.replace('.', "_"))
        } else {
            format!("diffusion_model.blocks.{b}.{leaf}")
        }
    };
    let (down_s, up_s) = if kohya {
        (".lora_down.weight", ".lora_up.weight")
    } else {
        (".lora_A.weight", ".lora_B.weight")
    };
    let mut t = Vec::new();
    for b in 0..BLOCKS {
        let qkv: Vec<(Tensor, Tensor)> = ["attn.to_q", "attn.to_k", "attn.to_v"]
            .iter()
            .map(|l| factors(cfg, b, l, seed))
            .collect();
        let down = Tensor::cat(&[&qkv[0].0, &qkv[1].0, &qkv[2].0], 0).unwrap();
        let out = qkv[0].1.dims()[0];
        let zero = Tensor::zeros((out, RANK), DType::F32, &Device::Cpu).unwrap();
        let rows: Vec<Tensor> = (0..3)
            .map(|i| {
                let parts: Vec<&Tensor> = (0..3)
                    .map(|j| if i == j { &qkv[i].1 } else { &zero })
                    .collect();
                Tensor::cat(&parts, 1).unwrap()
            })
            .collect();
        let up = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0).unwrap();
        let s = stem(b, "attn.qkv_proj");
        t.push((format!("{s}{down_s}"), down));
        t.push((format!("{s}{up_s}"), up));
        t.push((format!("{s}.alpha"), scalar(3.0 * RANK as f32)));

        let (down, up) = factors(cfg, b, "ff.net.0.proj", seed);
        let half = cfg.ffn_dim;
        let swapped = Tensor::cat(
            &[
                &up.narrow(0, half, half).unwrap(),
                &up.narrow(0, 0, half).unwrap(),
            ],
            0,
        )
        .unwrap();
        let s = stem(b, "mlp.fc1");
        t.push((format!("{s}{down_s}"), down));
        t.push((format!("{s}{up_s}"), swapped));
        t.push((format!("{s}.alpha"), scalar(RANK as f32)));

        for (diffusers, native) in [("attn.to_out.0", "attn.out_proj"), ("ff.net.2", "mlp.fc2")] {
            let (down, up) = factors(cfg, b, diffusers, seed);
            let s = stem(b, native);
            t.push((format!("{s}{down_s}"), down));
            t.push((format!("{s}{up_s}"), up));
            t.push((format!("{s}.alpha"), scalar(RANK as f32)));
        }
    }
    let path = dir.join(format!(
        "{}-{seed}.safetensors",
        if kohya { "kohya" } else { "comfyui" }
    ));
    let meta: &[(&str, &str)] = if kohya {
        &[("ss_network_module", "networks.lora")]
    } else {
        &[]
    };
    write(&path, &t, meta);
    path
}

/// Each adapted leaf's output on a fixed input, in leaf order, on the host as f32.
fn leaf_outputs(dit: &mut MiniMaxH3Dit, device: &Device) -> Vec<Vec<f32>> {
    let cfg = dit.config().clone();
    let mut out = Vec::new();
    for b in 0..BLOCKS {
        for leaf in BLOCK_TARGETS {
            let (_, inp) = leaf_shape(&cfg, leaf);
            let x = seeded(&[1, 8, inp], 7)
                .affine(50.0, 0.0)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_device(device)
                .unwrap();
            let path = format!("transformer_blocks.{b}.{leaf}");
            let segs: Vec<&str> = path.split('.').collect();
            let y = dit.adaptable_mut(&segs).unwrap().forward(&x).unwrap();
            out.push(
                y.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            );
        }
    }
    out
}

fn rel(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let scale = b.iter().fold(1e-12f32, |m, v| m.max(v.abs()));
            a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
        })
        .fold(0f32, f32::max)
}

fn spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path, scale, AdapterKind::Lora)
}

/// AC1 on candle/CUDA.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT) and CUDA; 66 GB bf16 DiT"]
fn h3_image_lora_layouts_fold_identically_on_the_real_dit_cuda() {
    let root = image_snapshot();
    let dir = scratch();
    let device = candle_gen::default_device().expect("a candle device");
    let mut dit =
        MiniMaxH3Dit::load_from_dir(root.join("transformer_diffusers"), &device, DType::BF16)
            .expect("load the image DiT");
    let cfg = dit.config().clone();
    let bare = leaf_outputs(&mut dit, &device);

    let peft = write_peft(&cfg, dir.path(), 1);
    apply_minimax_h3_adapters(&mut dit, &[spec(peft, 1.0)]).expect("PEFT install");
    let with_peft = leaf_outputs(&mut dit, &device);
    let moved = rel(&with_peft, &bare);
    println!("[sw-7096 candle] LoRA moves the leaf outputs by rel-max-abs {moved:.3e}");
    assert!(
        moved > 1e-3,
        "the LoRA must move the outputs, got {moved:.3e}"
    );

    for kohya in [true, false] {
        dit.clear_adapters();
        let native = write_native(&cfg, dir.path(), 1, kohya);
        let report = apply_minimax_h3_adapters(&mut dit, &[spec(native, 1.0)]).expect("install");
        assert_eq!(report.applied, BLOCKS * BLOCK_TARGETS.len());
        let drift = rel(&leaf_outputs(&mut dit, &device), &with_peft);
        let name = if kohya { "kohya" } else { "comfyui" };
        println!("[sw-7096 candle] {name} vs PEFT rel-max-abs {drift:.3e}");
        assert!(
            drift < REL_TOL,
            "{name} must fold like PEFT, rel-max-abs {drift:.3e}"
        );
    }

    dit.clear_adapters();
    let specs: Vec<AdapterSpec> = (0..10u32)
        .map(|i| {
            let seed = 100 + i;
            let path = match i % 3 {
                0 => write_peft(&cfg, dir.path(), seed),
                1 => write_native(&cfg, dir.path(), seed, true),
                _ => write_native(&cfg, dir.path(), seed, false),
            };
            spec(path, 0.1 * (i + 1) as f32)
        })
        .collect();
    apply_minimax_h3_adapters(&mut dit, &specs).expect("ten stacked LoRAs");
    for leaf in BLOCK_TARGETS {
        let path = format!("transformer_blocks.0.{leaf}");
        let segs: Vec<&str> = path.split('.').collect();
        let scales: Vec<f64> = dit
            .adaptable_mut(&segs)
            .unwrap()
            .adapters()
            .iter()
            .map(|r| r.scale())
            .collect();
        let want: Vec<f64> = (0..10).map(|i| (0.1 * (i + 1) as f32) as f64).collect();
        assert_eq!(scales, want, "{path}");
    }
    assert!(rel(&leaf_outputs(&mut dit, &device), &bare) > 1e-3);

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_lora_real::h3_image_lora_layouts_fold_identically_on_the_real_dit_cuda"
    );
}
