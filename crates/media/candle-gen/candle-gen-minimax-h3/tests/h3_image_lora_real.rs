//! SW-7096: MiniMax-H3-Image LoRAs on real weights (candle/CUDA) — the diffusers/PEFT, kohya
//! `lora_unet_` and ComfyUI layouts of the same synthetic factors fold identically onto the real
//! image DiT's projections, the fold moves them, and ten stacked files keep their own scales; and
//! blocks streamed through rung 4's one-block window carry the LoRA exactly as the resident ones.
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

fn write_peft(cfg: &MiniMaxH3DitConfig, dir: &Path, seed: u32, blocks: usize) -> PathBuf {
    let mut t = Vec::new();
    for b in 0..blocks {
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
fn write_native(
    cfg: &MiniMaxH3DitConfig,
    dir: &Path,
    seed: u32,
    kohya: bool,
    blocks: usize,
) -> PathBuf {
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
    for b in 0..blocks {
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

    let peft = write_peft(&cfg, dir.path(), 1, BLOCKS);
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
        let native = write_native(&cfg, dir.path(), 1, kohya, BLOCKS);
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
                0 => write_peft(&cfg, dir.path(), seed, BLOCKS),
                1 => write_native(&cfg, dir.path(), seed, true, BLOCKS),
                _ => write_native(&cfg, dir.path(), seed, false, BLOCKS),
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

/// Blocks the streamed-vs-resident gate runs — the MLX twin's `BLOCKS`.
const STREAM_BLOCKS: usize = 3;
/// The strength the streamed-vs-resident gate applies its block-leaf LoRAs at — the MLX twin's
/// `STRENGTH`: the residual stream carries activations far larger than one leaf's output, so a
/// strength-1 file barely moves a 3-block output; the fold is linear in strength.
const STRENGTH: f32 = 64.0;

/// The first [`STREAM_BLOCKS`] blocks' output at step 0 over a fixed residual stream, from a
/// freshly loaded image DiT carrying `specs` — resident on `device`, or `streamed`: loaded on the
/// host, adapters installed there, and run through rung 4's one-block window inside the request
/// scope the generator enters ([`candle_gen_minimax_h3::StreamedRequestScope`]), each block copied
/// to `device` with its residuals for its turn.
fn block_outputs(root: &Path, device: &Device, specs: &[AdapterSpec], streamed: bool) -> Vec<f32> {
    use candle_gen::gen_core::{
        GenerationMemory, GenerationRequest, LoadShape, TransformerComponent,
    };
    use candle_gen_minimax_h3::dit::ImageDit;
    use candle_gen_minimax_h3::{ImageLayout, StreamedRequestScope};
    let request = GenerationRequest {
        memory: streamed.then(|| GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_size: Some(1),
            transformer_window_component: Some(TransformerComponent::Both),
            ..Default::default()
        }),
        ..Default::default()
    };
    let _scope = StreamedRequestScope::enter(&request, LoadShape::DeferredMaterialization, device)
        .expect("enter the request scope");
    let load_device = if streamed {
        Device::Cpu
    } else {
        device.clone()
    };
    let mut dit = MiniMaxH3Dit::load_from_dir(
        root.join("transformer_diffusers"),
        &load_device,
        DType::BF16,
    )
    .expect("load the image DiT");
    if !specs.is_empty() {
        let report = apply_minimax_h3_adapters(&mut dit, specs).expect("install");
        assert_eq!(
            report.applied,
            specs.len() * BLOCK_TARGETS.len() * STREAM_BLOCKS,
            "every leaf of every LoRA'd block, from each file"
        );
    }
    if streamed {
        dit.prepare_streamed(device, 1)
            .expect("prepare the streamed DiT");
    }
    let cfg = dit.config().clone();
    let layout = ImageLayout::text_to_image(8, 16, 16, cfg.patch_size).expect("layout");
    let context = seeded(&[1, 8, cfg.text_dim], 21)
        .affine(50.0, 0.0)
        .and_then(|t| t.to_dtype(DType::BF16))
        .and_then(|t| t.to_device(device))
        .unwrap();
    let schedule = gen_core_h3::h3_schedule::SigmaSchedule::h3_image_distilled().unwrap();
    let model = ImageDit::new(dit, &layout, &context, schedule.timesteps()).expect("ImageDit");
    let mut x = seeded(&[1, layout.seq_len(), cfg.hidden_size], 7)
        .affine(50.0, 0.0)
        .and_then(|t| t.to_dtype(DType::BF16))
        .and_then(|t| t.to_device(device))
        .unwrap();
    for layer in 0..STREAM_BLOCKS {
        x = model.block_forward(0, layer, &x, None).expect("block");
    }
    let out = x
        .to_dtype(DType::F32)
        .and_then(|v| v.flatten_all())
        .and_then(|v| v.to_vec1::<f32>())
        .unwrap();
    drop(model);
    candle_gen_minimax_h3::release_device_memory(device).unwrap();
    out
}

/// AC2 on candle/CUDA, the MLX gate `h3_image_streamed_blocks_carry_the_lora_like_resident_ones`
/// mirrored: the same synthetic PEFT and kohya LoRAs at the same strengths over the first
/// [`STREAM_BLOCKS`] real image-DiT blocks, installed through the production install on a host
/// load and run through rung 4's one-block window, match the resident blocks carrying the same
/// LoRAs. The drift is measured against **the LoRA's own effect** (`max |resident − bare|`):
/// a streamed window that dropped the residuals would equal the bare output and score 1, so the
/// comparison cannot pass by accident.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT) and CUDA; two resident 66 GB bf16 DiT loads and one host load"]
fn h3_image_streamed_blocks_carry_the_lora_like_resident_ones_cuda() {
    let root = image_snapshot();
    let dir = scratch();
    let device = candle_gen::default_device().expect("a candle device");
    let cfg = MiniMaxH3DitConfig::from_image_diffusers_json(
        &std::fs::read_to_string(root.join("transformer_diffusers").join("config.json")).unwrap(),
    )
    .unwrap();
    let peft = write_peft(&cfg, dir.path(), 3, STREAM_BLOCKS);
    let kohya = write_native(&cfg, dir.path(), 4, true, STREAM_BLOCKS);
    let specs = [spec(peft, 0.75 * STRENGTH), spec(kohya, 0.5 * STRENGTH)];

    let bare = block_outputs(&root, &device, &[], false);
    let resident = block_outputs(&root, &device, &specs, false);
    let streamed = block_outputs(&root, &device, &specs, true);

    let max_diff = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
    };
    let moved = rel1(&resident, &bare);
    println!("[sw-7096 candle] LoRA moved the resident output by rel-max-abs {moved:.3e}");
    assert!(
        moved > 1e-3,
        "the LoRA must move the output, got {moved:.3e}"
    );
    let effect = max_diff(&resident, &bare);
    assert!(
        effect > 0.0,
        "the LoRA changed nothing, so no drift can be judged"
    );
    let drift = max_diff(&streamed, &resident) / effect;
    println!("[sw-7096 candle] streamed vs resident, relative to the LoRA's effect: {drift:.3e}");
    assert!(
        drift < REL_TOL,
        "streamed must match resident, {drift:.3e} of the LoRA's effect"
    );

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_lora_real::h3_image_streamed_blocks_carry_the_lora_like_resident_ones_cuda"
    );
}

/// One PEFT file of rank-[`RANK`] factors on `targets` (`(module path, out, in)`), alpha = rank.
fn write_targets(dir: &Path, name: &str, targets: &[(&str, usize, usize)], seed: u32) -> PathBuf {
    let mut t = Vec::new();
    for (k, (path, out, inp)) in targets.iter().enumerate() {
        let s = seed + 10 * k as u32;
        t.push((
            format!("transformer.{path}.lora_A.weight"),
            seeded(&[RANK, *inp], s),
        ));
        t.push((
            format!("transformer.{path}.lora_B.weight"),
            seeded(&[*out, RANK], s + 1),
        ));
    }
    let path = dir.join(name);
    write(&path, &t, &[("alpha", &RANK.to_string())]);
    path
}

/// Velocities at `steps` from a freshly loaded image DiT carrying `specs` — load, install,
/// precompute (with the adapters applied), forward, release.
fn velocities(
    root: &Path,
    device: &Device,
    specs: &[AdapterSpec],
    steps: &[usize],
) -> Vec<Vec<f32>> {
    use candle_gen_minimax_h3::dit::ImageDit;
    use candle_gen_minimax_h3::{ImageLayout, ImageVelocity};
    let mut dit =
        MiniMaxH3Dit::load_from_dir(root.join("transformer_diffusers"), device, DType::BF16)
            .expect("load the image DiT");
    if !specs.is_empty() {
        apply_minimax_h3_adapters(&mut dit, specs).expect("install");
    }
    let cfg = dit.config().clone();
    let layout = ImageLayout::text_to_image(8, 16, 16, cfg.patch_size).expect("layout");
    let context = seeded(&[1, 8, cfg.text_dim], 21)
        .affine(50.0, 0.0)
        .and_then(|t| t.to_dtype(DType::BF16))
        .and_then(|t| t.to_device(device))
        .unwrap();
    let schedule = gen_core_h3::h3_schedule::SigmaSchedule::h3_image_distilled().unwrap();
    let rows_n = layout.num_image_rows();
    let patch_dim = cfg.in_channels * cfg.patch_size.iter().product::<usize>();
    let rows = seeded(&[1, rows_n, patch_dim], 23)
        .affine(50.0, 0.0)
        .and_then(|t| t.to_device(device))
        .unwrap();
    let mut model = ImageDit::new(dit, &layout, &context, schedule.timesteps()).expect("ImageDit");
    let out = steps
        .iter()
        .map(|&step| {
            let v = model.forward(step, &rows).expect("forward");
            v.to_dtype(DType::F32)
                .and_then(|v| v.flatten_all())
                .and_then(|v| v.to_vec1::<f32>())
                .unwrap()
        })
        .collect();
    drop(model);
    candle_gen_minimax_h3::release_device_memory(device).unwrap();
    out
}

fn rel1(a: &[f32], b: &[f32]) -> f32 {
    rel(&[a.to_vec()], &[b.to_vec()])
}

/// SW-7096, the non-body linears on candle/CUDA: a LoRA on `pdd_proj_out.2` moves step 2 and not
/// step 0; LoRAs on block 0's `adaln_proj.linear`, `time_embedder.linear_2`, `norm_out.linear`
/// (kohya-spelled) plus `proj_in` / `context_embedder` move step 0, so the AdaLN precompute carried
/// them; a `proj_out` target names no module and is refused by name.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT) and CUDA; three 66 GB bf16 DiT loads"]
fn h3_image_adaln_and_output_head_loras_reach_the_denoiser_cuda() {
    let root = image_snapshot();
    let dir = scratch();
    let device = candle_gen::default_device().expect("a candle device");
    let cfg = MiniMaxH3DitConfig::from_image_diffusers_json(
        &std::fs::read_to_string(root.join("transformer_diffusers").join("config.json")).unwrap(),
    )
    .unwrap();
    let hidden = cfg.hidden_size;
    let patch_dim = cfg.in_channels * cfg.patch_size.iter().product::<usize>();
    let bare = velocities(&root, &device, &[], &[0, 2]);

    let head = write_targets(
        dir.path(),
        "head.safetensors",
        &[("pdd_proj_out.2", patch_dim, hidden)],
        31,
    );
    let with_head = velocities(&root, &device, &[spec(head, 20.0)], &[0, 2]);
    let (moved_2, moved_0) = (rel1(&with_head[1], &bare[1]), rel1(&with_head[0], &bare[0]));
    println!("[sw-7096 candle] pdd_proj_out.2 LoRA moves step 2 by {moved_2:.3e}, step 0 by {moved_0:.3e}");
    assert!(
        moved_2 > 1e-3,
        "the head LoRA must move step 2, got {moved_2:.3e}"
    );
    assert!(
        moved_0 <= moved_2 / 100.0,
        "a LoRA on head 2 must not touch step 0 ({moved_0:.3e})"
    );

    let io = write_targets(
        dir.path(),
        "io.safetensors",
        &[
            (
                "time_embedder.linear_2",
                cfg.time_embed_dim,
                cfg.time_embed_hidden_dim,
            ),
            ("proj_in", hidden, patch_dim),
            ("context_embedder", hidden, cfg.text_dim),
        ],
        41,
    );
    let mut t = Vec::new();
    for (k, (stem, out, inp)) in [
        (
            "lora_unet_transformer_blocks_0_adaln_proj_linear",
            cfg.adaln_out_features(),
            cfg.time_embed_dim,
        ),
        ("lora_unet_norm_out_linear", 2 * hidden, cfg.time_embed_dim),
    ]
    .into_iter()
    .enumerate()
    {
        let s = 51 + 10 * k as u32;
        t.push((format!("{stem}.lora_down.weight"), seeded(&[RANK, inp], s)));
        t.push((
            format!("{stem}.lora_up.weight"),
            seeded(&[out, RANK], s + 1),
        ));
        t.push((format!("{stem}.alpha"), scalar(RANK as f32)));
    }
    let adaln = dir.path().join("adaln-kohya.safetensors");
    write(&adaln, &t, &[("ss_network_module", "networks.lora")]);
    let with_adaln = velocities(&root, &device, &[spec(io, 20.0), spec(adaln, 20.0)], &[0]);
    let moved = rel1(&with_adaln[0], &bare[0]);
    println!("[sw-7096 candle] AdaLN + I/O LoRAs move step 0 by {moved:.3e}");
    assert!(
        moved > 1e-3,
        "the AdaLN/I-O LoRAs must move step 0, got {moved:.3e}"
    );

    let missing = write_targets(
        dir.path(),
        "proj-out.safetensors",
        &[("proj_out", patch_dim, hidden)],
        61,
    );
    let mut dit =
        MiniMaxH3Dit::load_from_dir(root.join("transformer_diffusers"), &device, DType::BF16)
            .expect("load the image DiT");
    let error = apply_minimax_h3_adapters(&mut dit, &[spec(missing, 1.0)])
        .unwrap_err()
        .to_string();
    assert!(error.contains("proj_out"), "{error}");

    let mut t = Vec::new();
    for (k, (stem, out, inp)) in [
        ("diffusion_model.final_layer.video_out", patch_dim, hidden),
        ("diffusion_model.condition_proj", hidden, cfg.text_dim),
        ("diffusion_model.video_patch_proj", hidden, patch_dim),
        (
            "diffusion_model.time_embedder.proj_in",
            cfg.time_embed_hidden_dim,
            cfg.freq_dim,
        ),
        (
            "diffusion_model.final_layer.adaln_proj.linear",
            2 * hidden,
            cfg.time_embed_dim,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let s = 71 + 10 * k as u32;
        t.push((format!("{stem}.lora_A.weight"), seeded(&[RANK, inp], s)));
        t.push((format!("{stem}.lora_B.weight"), seeded(&[out, RANK], s + 1)));
    }
    let native = dir.path().join("native-comfy.safetensors");
    write(&native, &t, &[("alpha", &RANK.to_string())]);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(native, 1.0)]).expect("native install");
    assert_eq!(
        report.applied,
        6 + 4,
        "video_out fans out to six heads; four renames"
    );

    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_lora_real::h3_image_adaln_and_output_head_loras_reach_the_denoiser_cuda"
    );
}
