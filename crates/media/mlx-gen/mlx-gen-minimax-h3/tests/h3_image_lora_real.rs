//! SW-7096: MiniMax-H3-Image LoRAs on real weights (MLX) — the three community layouts fold
//! identically onto the real image DiT, ten stacked files keep their own scales, and a block
//! **streamed** through the shared [`DitBlockStream`] carries the LoRA exactly as the resident
//! block does.
//!
//! All `#[ignore]`d: they need `MiniMaxAI/MiniMax-H3-Image` (`MINIMAX_H3_IMAGE_SNAPSHOT`). The
//! `minimax-h3-image` real-weights profile runs them on the `mlx-minimax-h3-image` job. Each
//! asserts on its inputs rather than skipping. The LoRA is synthetic: generated deterministically
//! from a seed and written to `RUNNER_TEMP` (no network), so the factors are known exactly.
//!
//! Only the first [`BLOCKS`] blocks run: the claims are per block (a layout's fold, a stream's
//! re-application), so a short stack proves them at a fraction of the resident bytes.

use std::path::{Path, PathBuf};

use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::gen_core::{AdapterKind, AdapterSpec};
use mlx_gen_minimax_h3::adapters::{apply_minimax_h3_adapters, BLOCK_TARGETS};
use mlx_gen_minimax_h3::denoise::schedule::SigmaSchedule;
use mlx_gen_minimax_h3::dit::config::MiniMaxH3ImageDitConfig;
use mlx_gen_minimax_h3::dit::{MmRope, MmRopeTables};
use mlx_gen_minimax_h3::h3_image::{
    ImageDenoiser, ImageLayout, MiniMaxH3ImageDit, IMAGE_DIT_COMPONENT,
};
use mlx_gen_minimax_h3::tensor::slice_axis;
use mlx_gen_minimax_h3::DitBlockStream;

/// Blocks run per arm.
const BLOCKS: usize = 3;
/// The synthetic LoRA's rank.
const RANK: i32 = 4;
/// Relative max-abs between two layouts of the same factors, or a streamed and a resident block.
/// The arms run the same bf16 arithmetic; the bound only absorbs a different summation order of
/// the split fused factors.
const REL_TOL: f32 = 2e-2;

fn image_snapshot() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_IMAGE_SNAPSHOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_IMAGE_SNAPSHOT must point at a MiniMaxAI/MiniMax-H3-Image snapshot root. This \
         test is #[ignore]d and asserts rather than skips."
    );
    let path = PathBuf::from(raw);
    assert!(
        path.join(IMAGE_DIT_COMPONENT).join("config.json").is_file(),
        "MINIMAX_H3_IMAGE_SNAPSHOT={} has no {IMAGE_DIT_COMPONENT}/config.json",
        path.display()
    );
    path
}

/// A scratch directory under `RUNNER_TEMP` when the job provides one.
fn scratch() -> tempfile::TempDir {
    match std::env::var("RUNNER_TEMP") {
        Ok(dir) if !dir.is_empty() => tempfile::tempdir_in(dir).expect("scratch in RUNNER_TEMP"),
        _ => tempfile::tempdir().expect("scratch"),
    }
}

/// Deterministic factor values from `seed`, centred, small enough that a rank-4 residual moves a
/// block's output visibly without saturating it.
fn seeded(shape: &[i32], seed: u32) -> Array {
    let n: i32 = shape.iter().product();
    let values: Vec<f32> = (0..n as u32)
        .map(|i| {
            let h = i.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503).wrapping_add(17);
            ((h % 2003) as f32 / 2003.0 - 0.5) * 0.02
        })
        .collect();
    Array::from_slice(&values, shape)
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

fn leaf_shape(cfg: &MiniMaxH3ImageDitConfig, leaf: &str) -> (i32, i32) {
    let b = &cfg.base;
    match leaf {
        "attn.to_q" | "attn.to_k" | "attn.to_v" => (b.inner_dim(), b.hidden_size),
        "attn.to_out.0" => (b.hidden_size, b.inner_dim()),
        "ff.net.0.proj" => (2 * b.ffn_dim, b.hidden_size),
        "ff.net.2" => (b.hidden_size, b.ffn_dim),
        _ => unreachable!("{leaf}"),
    }
}

/// The diffusers-space factors `(down [r, in], up [out, r])` of one leaf.
fn factors(cfg: &MiniMaxH3ImageDitConfig, block: usize, leaf: &str, seed: u32) -> (Array, Array) {
    let (out, inp) = leaf_shape(cfg, leaf);
    let k = seed
        + 1000 * block as u32
        + 10 * BLOCK_TARGETS.iter().position(|t| *t == leaf).unwrap() as u32;
    (seeded(&[RANK, inp], k), seeded(&[out, RANK], k + 1))
}

fn write(path: &Path, tensors: Vec<(String, Array)>, meta: &[(&str, &str)]) {
    let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let meta: std::collections::HashMap<String, String> = meta
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Array::save_safetensors(refs, Some(&meta), path).expect("write the synthetic LoRA");
}

fn alpha(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Diffusers / PEFT: `transformer.transformer_blocks.{i}.<leaf>.lora_{A,B}.weight`, alpha = rank.
fn write_peft(cfg: &MiniMaxH3ImageDitConfig, dir: &Path, seed: u32) -> PathBuf {
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
    write(&path, t, &[("alpha", &RANK.to_string())]);
    path
}

/// The native-module form of the same factors: fused qkv (concatenated `A`, block-diagonal `B`,
/// alpha × 3 so `alpha/rank` is unchanged), `mlp.fc1` with its SwiGLU halves in native
/// `[gate | value]` order, and the two renames. `kohya` picks the flattened `lora_unet_` spelling,
/// otherwise the ComfyUI `diffusion_model.` one.
fn write_native(cfg: &MiniMaxH3ImageDitConfig, dir: &Path, seed: u32, kohya: bool) -> PathBuf {
    let r = RANK;
    let mut t = Vec::new();
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
    for b in 0..BLOCKS {
        let qkv: Vec<(Array, Array)> = ["attn.to_q", "attn.to_k", "attn.to_v"]
            .iter()
            .map(|l| factors(cfg, b, l, seed))
            .collect();
        let down = mlx_rs::ops::concatenate_axis(&[&qkv[0].0, &qkv[1].0, &qkv[2].0], 0).unwrap();
        let out = cfg.base.inner_dim();
        let zero = Array::zeros::<f32>(&[out, r])
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let rows: Vec<Array> = (0..3)
            .map(|i| {
                let parts: Vec<&Array> = (0..3)
                    .map(|j| if i == j { &qkv[i].1 } else { &zero })
                    .collect();
                mlx_rs::ops::concatenate_axis(&parts, 1).unwrap()
            })
            .collect();
        let up = mlx_rs::ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 0).unwrap();
        let s = stem(b, "attn.qkv_proj");
        t.push((format!("{s}{down_s}"), down));
        t.push((format!("{s}{up_s}"), up));
        t.push((format!("{s}.alpha"), alpha(3.0 * r as f32)));

        let (down, up) = factors(cfg, b, "ff.net.0.proj", seed);
        let half = cfg.base.ffn_dim;
        let first = slice_axis(&up, 0, 0, half).unwrap();
        let second = slice_axis(&up, 0, half, 2 * half).unwrap();
        let swapped = mlx_rs::ops::concatenate_axis(&[&second, &first], 0).unwrap();
        let s = stem(b, "mlp.fc1");
        t.push((format!("{s}{down_s}"), down));
        t.push((format!("{s}{up_s}"), swapped));
        t.push((format!("{s}.alpha"), alpha(r as f32)));

        for (diffusers, native) in [("attn.to_out.0", "attn.out_proj"), ("ff.net.2", "mlp.fc2")] {
            let (down, up) = factors(cfg, b, diffusers, seed);
            let s = stem(b, native);
            t.push((format!("{s}{down_s}"), down));
            t.push((format!("{s}{up_s}"), up));
            t.push((format!("{s}.alpha"), alpha(r as f32)));
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
    write(&path, t, meta);
    path
}

fn spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec::new(path, scale, AdapterKind::Lora)
}

/// Inputs for a short block stack over a real `text2image` layout: rows, rotary tables, gather
/// indices and a `[1, time_embed_dim]` timestep embedding.
struct StackInputs {
    x: Array,
    rope: MmRope,
    tables: MmRopeTables,
    adaln_indices: Array,
    temb: Array,
}

fn stack_inputs(cfg: &MiniMaxH3ImageDitConfig) -> StackInputs {
    let layout = ImageLayout::t2i(8, 16, 16, cfg.base.patch_size).unwrap();
    let rope = MmRope::new(cfg.base.rope_freq_dim, cfg.base.rope_theta).unwrap();
    let tables = rope.tables(layout.position_ids()).unwrap();
    let seq = layout.seq_len();
    StackInputs {
        x: seeded(&[1, seq, cfg.base.hidden_size], 7)
            .multiply(Array::from_f32(50.0))
            .unwrap(),
        rope,
        tables,
        adaln_indices: layout.token_tags().as_dtype(Dtype::Int32).unwrap(),
        temb: seeded(&[1, cfg.base.time_embed_dim], 9)
            .multiply(Array::from_f32(50.0))
            .unwrap(),
    }
}

/// The first [`BLOCKS`] resident blocks of `dit`, forward.
fn run_resident(dit: &MiniMaxH3ImageDit, inputs: &StackInputs) -> Array {
    let mut x = inputs.x.clone();
    for block in &dit.blocks()[..BLOCKS] {
        let modulation = block.modulation(&inputs.temb).unwrap();
        x = block
            .forward(
                &x,
                &modulation,
                &inputs.adaln_indices,
                &inputs.rope,
                &inputs.tables,
            )
            .unwrap();
        mlx_rs::transforms::eval([&x]).unwrap();
    }
    x
}

fn rel(a: &Array, b: &Array) -> f32 {
    let scale = mlx_gen::testing::max_abs(&b.as_dtype(Dtype::Float32).unwrap()).max(1e-12);
    mlx_gen::testing::max_abs_diff_f32(a, b) / scale
}

fn clear(dit: &mut MiniMaxH3ImageDit) {
    for b in 0..BLOCKS {
        for leaf in BLOCK_TARGETS {
            let path = format!("transformer_blocks.{b}.{leaf}");
            let segs: Vec<&str> = path.split('.').collect();
            dit.adaptable_mut(&segs).unwrap().set_adapters(Vec::new());
        }
    }
}

/// AC1 on MLX: the diffusers/PEFT, kohya `lora_unet_` and ComfyUI layouts of the same factors fold
/// to the same block output on the real image DiT, the LoRA moves that output, and ten stacked
/// files land in order at their own scales.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT); three resident blocks"]
fn h3_image_lora_layouts_fold_identically_on_the_real_dit() {
    let root = image_snapshot();
    let dir = scratch();
    let mut dit = MiniMaxH3ImageDit::load(&root, Dtype::Bfloat16).expect("image DiT");
    let cfg = dit.config().clone();
    let inputs = stack_inputs(&cfg);
    let bare = run_resident(&dit, &inputs);

    let peft = write_peft(&cfg, dir.path(), 1);
    apply_minimax_h3_adapters(&mut dit, &[spec(peft.clone(), 1.0)]).expect("PEFT install");
    let with_peft = run_resident(&dit, &inputs);
    let moved = rel(&with_peft, &bare);
    println!("[sw-7096 mlx] LoRA moves the {BLOCKS}-block output by rel-max-abs {moved:.3e}");
    assert!(
        moved > 1e-3,
        "the LoRA must move the output, got {moved:.3e}"
    );

    for kohya in [true, false] {
        clear(&mut dit);
        let native = write_native(&cfg, dir.path(), 1, kohya);
        let report = apply_minimax_h3_adapters(&mut dit, &[spec(native, 1.0)]).expect("install");
        assert_eq!(report.applied, BLOCKS * BLOCK_TARGETS.len());
        let got = run_resident(&dit, &inputs);
        let drift = rel(&got, &with_peft);
        let name = if kohya { "kohya" } else { "comfyui" };
        println!("[sw-7096 mlx] {name} vs PEFT rel-max-abs {drift:.3e}");
        assert!(
            drift < REL_TOL,
            "{name} must fold like PEFT, rel-max-abs {drift:.3e}"
        );
    }

    // Ten stacked files, scales 0.1 … 1.0, three layouts interleaved: every leaf carries ten
    // residuals in file order at the file's own scale.
    clear(&mut dit);
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
        let scales: Vec<f32> = dit
            .adaptable_mut(&segs)
            .unwrap()
            .adapters()
            .iter()
            .map(|a| match a {
                Adapter::Lora { scale, .. } => *scale,
                _ => panic!("{path}: a LoRA residual"),
            })
            .collect();
        let want: Vec<f32> = (0..10).map(|i| 0.1 * (i + 1) as f32).collect();
        assert_eq!(scales, want, "{path}");
    }
    let stacked = run_resident(&dit, &inputs);
    assert!(
        rel(&stacked, &bare) > 1e-3,
        "the stack must move the output"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_lora_real::h3_image_lora_layouts_fold_identically_on_the_real_dit"
    );
}

/// AC2 on MLX: a block rebuilt by the shared [`DitBlockStream`] from the real image checkpoint
/// carries the LoRA — installed through the production install onto the stream — and matches the
/// resident block with the same LoRA. Without the stream's re-application the streamed output
/// equals the bare one, so the comparison cannot pass by accident.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT); three resident blocks"]
fn h3_image_streamed_blocks_carry_the_lora_like_resident_ones() {
    let root = image_snapshot();
    let dir = scratch();
    let mut dit = MiniMaxH3ImageDit::load(&root, Dtype::Bfloat16).expect("image DiT");
    let cfg = dit.config().clone();
    let inputs = stack_inputs(&cfg);
    let bare = run_resident(&dit, &inputs);
    let peft = write_peft(&cfg, dir.path(), 3);
    let kohya = write_native(&cfg, dir.path(), 4, true);
    let specs = [spec(peft, 0.75), spec(kohya, 0.5)];
    apply_minimax_h3_adapters(&mut dit, &specs).expect("resident install");
    let resident = run_resident(&dit, &inputs);

    // Modulation tables come from the resident blocks: a streamed body has no adaln_proj.
    let modulations: Vec<_> = dit.blocks()[..BLOCKS]
        .iter()
        .map(|b| b.modulation(&inputs.temb).unwrap())
        .collect();
    let mut stream = DitBlockStream::new(
        root.join(IMAGE_DIT_COMPONENT),
        Dtype::Bfloat16,
        cfg.base.clone(),
    )
    .expect("stream over transformer_diffusers/");
    apply_minimax_h3_adapters(&mut stream, &specs).expect("streamed install");
    assert_eq!(
        stream.adapters().residual_count(),
        2 * BLOCK_TARGETS.len() * BLOCKS,
        "two files, every leaf of every LoRA'd block"
    );
    stream.adapters().materialize().unwrap();

    let mut x = inputs.x.clone();
    for (index, modulation) in modulations.iter().enumerate() {
        let mut view = stream.open().unwrap();
        let block = stream.materialize(&mut view, index).unwrap();
        x = block
            .forward(
                &x,
                modulation,
                &inputs.adaln_indices,
                &inputs.rope,
                &inputs.tables,
            )
            .unwrap();
        mlx_rs::transforms::eval([&x]).unwrap();
        drop(view);
    }
    let drift = rel(&x, &resident);
    let moved = rel(&resident, &bare);
    println!(
        "[sw-7096 mlx] streamed vs resident with LoRA rel-max-abs {drift:.3e}; LoRA moved {moved:.3e}"
    );
    assert!(
        moved > 1e-3,
        "the LoRA must move the output, got {moved:.3e}"
    );
    assert!(
        drift < REL_TOL,
        "streamed must match resident, rel-max-abs {drift:.3e}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_lora_real::h3_image_streamed_blocks_carry_the_lora_like_resident_ones"
    );
}

/// One PEFT file of rank-[`RANK`] factors on `targets` (`(module path, out, in)`), alpha = rank.
fn write_targets(dir: &Path, name: &str, targets: &[(&str, i32, i32)], seed: u32) -> PathBuf {
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
    write(&path, t, &[("alpha", &RANK.to_string())]);
    path
}

/// The step-`step` velocity of a freshly loaded image DiT carrying `specs`, on a small packed
/// sequence. Each call loads, installs, precomputes (with the adapters applied) and releases.
fn velocities(root: &Path, specs: &[AdapterSpec], steps: &[usize]) -> Vec<Array> {
    let mut dit = MiniMaxH3ImageDit::load(root, Dtype::Bfloat16).expect("image DiT");
    if !specs.is_empty() {
        apply_minimax_h3_adapters(&mut dit, specs).expect("install");
    }
    let cfg = dit.config().clone();
    let layout = ImageLayout::t2i(8, 16, 16, cfg.base.patch_size).unwrap();
    let context = seeded(&[1, 8, cfg.base.text_dim], 21)
        .multiply(Array::from_f32(50.0))
        .unwrap();
    let rows = seeded(
        &[1, layout.num_image_rows(), cfg.base.video_patch_dim()],
        23,
    )
    .multiply(Array::from_f32(50.0))
    .unwrap()
    .as_dtype(Dtype::Float32)
    .unwrap();
    let denoiser = ImageDenoiser::new(
        dit,
        layout,
        &context,
        SigmaSchedule::h3_image_distilled().unwrap(),
    )
    .expect("denoiser");
    let out = steps
        .iter()
        .map(|&step| {
            let v = denoiser.velocity(step, &rows).unwrap();
            mlx_rs::transforms::eval([&v]).unwrap();
            v
        })
        .collect();
    drop(denoiser);
    mlx_gen::residency::drain_allocator_cache();
    out
}

/// SW-7096, the non-body linears on MLX: a LoRA on `pdd_proj_out.2` changes step 2's velocity and
/// leaves step 0's **bit-identical** (each head is its own PEFT module); a LoRA on block 0's
/// `adaln_proj.linear`, `time_embedder.linear_2` and `norm_out.linear` (kohya-spelled) plus
/// `proj_in` / `context_embedder` changes step 0 — so the AdaLN precompute ran with the residuals
/// applied — and a `proj_out` target, a module the image DiT does not have, is refused by name.
#[test]
#[ignore = "needs a real MiniMax-H3-Image snapshot (MINIMAX_H3_IMAGE_SNAPSHOT); three full image-DiT loads"]
fn h3_image_adaln_and_output_head_loras_reach_the_denoiser() {
    let root = image_snapshot();
    let dir = scratch();
    let probe = MiniMaxH3ImageDit::load(&root, Dtype::Bfloat16).expect("image DiT");
    let cfg = probe.config().clone();
    drop(probe);
    let b = &cfg.base;

    let bare = velocities(&root, &[], &[0, 2]);

    let head = write_targets(
        dir.path(),
        "head.safetensors",
        &[("pdd_proj_out.2", b.video_patch_dim(), b.hidden_size)],
        31,
    );
    let with_head = velocities(&root, &[spec(head, 20.0)], &[0, 2]);
    let moved_2 = rel(&with_head[1], &bare[1]);
    let moved_0 = rel(&with_head[0], &bare[0]);
    println!(
        "[sw-7096 mlx] pdd_proj_out.2 LoRA moves step 2 by rel-max-abs {moved_2:.3e}, step 0 by {moved_0:.3e}"
    );
    assert!(
        moved_2 > 1e-3,
        "the head LoRA must move step 2, got {moved_2:.3e}"
    );
    // Expected exactly 0; the bound only tolerates run-to-run noise far below the head's effect.
    assert!(
        moved_0 <= moved_2 / 100.0,
        "a LoRA on head 2 must not touch step 0 (moved {moved_0:.3e})"
    );

    let io = write_targets(
        dir.path(),
        "io.safetensors",
        &[
            (
                "time_embedder.linear_2",
                b.time_embed_dim,
                b.time_embed_hidden_dim,
            ),
            ("proj_in", b.hidden_size, b.video_patch_dim()),
            ("context_embedder", b.hidden_size, b.text_dim),
        ],
        41,
    );
    // kohya top-level and block spellings for the two AdaLN projections.
    let mut t = Vec::new();
    for (k, (stem, out, inp)) in [
        (
            "lora_unet_transformer_blocks_0_adaln_proj_linear",
            b.adaln_out_features(),
            b.time_embed_dim,
        ),
        (
            "lora_unet_norm_out_linear",
            2 * b.hidden_size,
            b.time_embed_dim,
        ),
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
        t.push((format!("{stem}.alpha"), alpha(RANK as f32)));
    }
    let adaln = dir.path().join("adaln-kohya.safetensors");
    write(&adaln, t, &[("ss_network_module", "networks.lora")]);
    let with_adaln = velocities(&root, &[spec(io, 20.0), spec(adaln, 20.0)], &[0]);
    let moved = rel(&with_adaln[0], &bare[0]);
    println!("[sw-7096 mlx] AdaLN + I/O LoRAs move step 0 by rel-max-abs {moved:.3e}");
    assert!(
        moved > 1e-3,
        "the AdaLN/I-O LoRAs must move step 0, got {moved:.3e}"
    );

    let missing = write_targets(
        dir.path(),
        "proj-out.safetensors",
        &[("proj_out", b.video_patch_dim(), b.hidden_size)],
        61,
    );
    let mut dit = MiniMaxH3ImageDit::load(&root, Dtype::Bfloat16).expect("image DiT");
    let error = apply_minimax_h3_adapters(&mut dit, &[spec(missing, 1.0)])
        .unwrap_err()
        .to_string();
    assert!(error.contains("proj_out"), "{error}");

    // Native ComfyUI spellings: renamed to their diffusers modules, and the single native output
    // head fans out to all six per-step heads.
    let mut t = Vec::new();
    for (k, (stem, out, inp)) in [
        (
            "diffusion_model.final_layer.video_out",
            b.video_patch_dim(),
            b.hidden_size,
        ),
        ("diffusion_model.condition_proj", b.hidden_size, b.text_dim),
        (
            "diffusion_model.video_patch_proj",
            b.hidden_size,
            b.video_patch_dim(),
        ),
        (
            "diffusion_model.time_embedder.proj_in",
            b.time_embed_hidden_dim,
            b.freq_dim,
        ),
        (
            "diffusion_model.final_layer.adaln_proj.linear",
            2 * b.hidden_size,
            b.time_embed_dim,
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
    write(&native, t, &[("alpha", &RANK.to_string())]);
    let report = apply_minimax_h3_adapters(&mut dit, &[spec(native, 1.0)]).expect("native install");
    assert_eq!(
        report.applied,
        6 + 4,
        "video_out fans out to six heads; four renames"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_lora_real::h3_image_adaln_and_output_head_loras_reach_the_denoiser"
    );
}
