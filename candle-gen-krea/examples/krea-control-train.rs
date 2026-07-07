//! Krea 2 **pose-ControlNet control-branch trainer** — the sc-8460 spike harness (epic 8459).
//!
//! Trains the [`candle_gen_krea::control::ControlBranch`] (N trainable copies of the first N
//! single-stream DiT blocks + zero-init residual projections) against the **frozen** Krea 2 Turbo
//! base on (target image, pose skeleton, caption) triples, with the same rectified-flow velocity
//! objective the Krea LoRA trainer uses, sampled across the full noise schedule.
//!
//! ```text
//! cargo run -p candle-gen-krea --example krea-control-train --features cuda --release -- \
//!   --snapshot D:\.cache\huggingface\hub\models--krea--Krea-2-Turbo\snapshots\<sha> \
//!   --data D:\sceneworks-pose-controlnet\spike\train \
//!   --out  D:\krea-control-ckpt --steps 200 --batch 1 --lr 1e-4 --save-every 100
//! ```
//!
//! Dataset layout: `<data>/manifest.jsonl` with rows
//! `{"id","target","pose","caption",...}` (paths relative to the manifest dir), images square.
//! `--synth N` first writes N synthetic (gradient target, stick-figure pose) pairs + manifest into
//! `--data` — the self-contained smoke path when the real COCO-pose data isn't staged yet.
//!
//! Flags: `--snapshot <dir>` `--data <dir>` `--out <dir>` `--steps N` `--batch N` `--lr F`
//! `--save-every N` `--n-blocks N` (default 7) `--resolution N` (default 512) `--seed N`
//! `--branch-dtype bf16|f32` (default bf16) `--resume <ckpt.safetensors>` `--synth N`
//! `--timestep-type sigmoid|uniform|linear|weighted` (default uniform).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::train::flow_match::{
    self, component_vb, sample_noise, sample_unit_timestep, velocity_loss,
};
use candle_gen::train::optim::{accumulate_grads, clip_grad_norm, scale_grads, TrainOptimizer};
use candle_gen_krea::control::{forward_with_control, ControlBranch};
use candle_gen_krea::loader::Weights;
use candle_gen_krea::{Krea2Config, KreaTeConfig, KreaTextEncoder, KreaTokenizer, KreaTrainDit};
use candle_gen_qwen_image::vae::QwenVaeEncoder;

const MAX_TEXT_TOKENS: usize = 1024;

struct Args {
    snapshot: PathBuf,
    data: PathBuf,
    out: PathBuf,
    steps: u32,
    batch: u32,
    lr: f32,
    save_every: u32,
    n_blocks: usize,
    resolution: u32,
    seed: u64,
    branch_dtype: DType,
    resume: Option<PathBuf>,
    synth: usize,
    timestep_type: String,
}

fn parse_args() -> Args {
    let mut a = Args {
        snapshot: PathBuf::from("D:/models/Krea-2-Turbo"),
        data: PathBuf::from("./control-data"),
        out: PathBuf::from("./control-ckpt"),
        steps: 200,
        batch: 1,
        lr: 1e-4,
        save_every: 100,
        n_blocks: 7,
        resolution: 512,
        seed: 42,
        branch_dtype: DType::BF16,
        resume: None,
        synth: 0,
        timestep_type: "uniform".into(),
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let val = || {
            argv.get(i + 1)
                .unwrap_or_else(|| panic!("missing value for {key}"))
                .clone()
        };
        match key {
            "--snapshot" => a.snapshot = val().into(),
            "--data" => a.data = val().into(),
            "--out" => a.out = val().into(),
            "--steps" => a.steps = val().parse().expect("--steps"),
            "--batch" => a.batch = val().parse::<u32>().expect("--batch").max(1),
            "--lr" => a.lr = val().parse().expect("--lr"),
            "--save-every" => a.save_every = val().parse().expect("--save-every"),
            "--n-blocks" => a.n_blocks = val().parse().expect("--n-blocks"),
            "--resolution" => a.resolution = val().parse().expect("--resolution"),
            "--seed" => a.seed = val().parse().expect("--seed"),
            "--branch-dtype" => {
                a.branch_dtype = match val().to_ascii_lowercase().as_str() {
                    "bf16" | "bfloat16" => DType::BF16,
                    "f32" | "fp32" | "float32" => DType::F32,
                    other => panic!("--branch-dtype must be bf16|f32, got {other}"),
                }
            }
            "--resume" => a.resume = Some(val().into()),
            "--synth" => a.synth = val().parse().expect("--synth"),
            "--timestep-type" => a.timestep_type = val(),
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }
    a
}

struct Row {
    target: String,
    pose: String,
    caption: String,
}

fn read_manifest(data: &Path) -> Vec<Row> {
    let path = data.join("manifest.jsonl");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: serde_json::Value =
                serde_json::from_str(l).unwrap_or_else(|e| panic!("manifest row: {e}"));
            let field = |k: &str| {
                v.get(k)
                    .and_then(|s| s.as_str())
                    .unwrap_or_else(|| panic!("manifest row missing string field {k:?}: {l}"))
                    .to_string()
            };
            Row {
                target: field("target"),
                pose: field("pose"),
                caption: field("caption"),
            }
        })
        .collect()
}

// ── synthetic smoke data ─────────────────────────────────────────────────────────────────────

fn draw_line(img: &mut image::RgbImage, p0: (f32, f32), p1: (f32, f32), c: [u8; 3], th: i32) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let steps = ((p1.0 - p0.0).abs().max((p1.1 - p0.1).abs()) as i32).max(1);
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = (p0.0 + (p1.0 - p0.0) * t) as i32;
        let y = (p0.1 + (p1.1 - p0.1) * t) as i32;
        for dx in -th..=th {
            for dy in -th..=th {
                let (px, py) = (x + dx, y + dy);
                if px >= 0 && px < w && py >= 0 && py < h {
                    img.put_pixel(px as u32, py as u32, image::Rgb(c));
                }
            }
        }
    }
}

/// A stick figure at `(cx, cy)` with per-sample limb angles; drawn in `c` on `img`.
fn draw_figure(img: &mut image::RgbImage, cx: f32, cy: f32, scale: f32, i: usize, c: [u8; 3]) {
    let a = (i as f32) * 0.7;
    let head = (cx, cy - 0.35 * scale);
    let hip = (cx, cy + 0.1 * scale);
    // head "circle"
    for k in 0..24 {
        let t0 = (k as f32) / 24.0 * std::f32::consts::TAU;
        let t1 = ((k + 1) as f32) / 24.0 * std::f32::consts::TAU;
        let r = 0.08 * scale;
        draw_line(
            img,
            (head.0 + r * t0.cos(), head.1 + r * t0.sin()),
            (head.0 + r * t1.cos(), head.1 + r * t1.sin()),
            c,
            3,
        );
    }
    let neck = (cx, cy - 0.25 * scale);
    draw_line(img, neck, hip, c, 4); // spine
    let arm = 0.22 * scale;
    draw_line(
        img,
        neck,
        (
            neck.0 - arm * (0.9 + 0.3 * a.sin()),
            neck.1 + arm * a.cos().abs(),
        ),
        c,
        4,
    );
    draw_line(
        img,
        neck,
        (
            neck.0 + arm * (0.9 + 0.3 * a.cos()),
            neck.1 + arm * a.sin().abs(),
        ),
        c,
        4,
    );
    let leg = 0.3 * scale;
    draw_line(img, hip, (hip.0 - leg * 0.5, hip.1 + leg), c, 4);
    draw_line(img, hip, (hip.0 + leg * 0.5, hip.1 + leg), c, 4);
}

/// Write `count` synthetic (target, pose, caption) pairs + `manifest.jsonl` into `dir`.
fn synth_pairs(dir: &Path, count: usize) {
    std::fs::create_dir_all(dir).expect("create --data dir");
    let edge = 1024u32;
    let mut manifest = String::new();
    for i in 0..count {
        let mut target = image::RgbImage::from_fn(edge, edge, |x, y| {
            let r = ((x as f32 / edge as f32) * 200.0) as u8;
            let g = ((y as f32 / edge as f32) * 200.0) as u8;
            let b = ((i * 40) % 255) as u8;
            image::Rgb([r, g, b])
        });
        let cx = edge as f32 * (0.35 + 0.3 * ((i as f32 * 0.37).sin().abs()));
        let cy = edge as f32 * 0.5;
        let scale = edge as f32 * 0.6;
        draw_figure(&mut target, cx, cy, scale, i, [240, 220, 200]);

        let mut pose = image::RgbImage::from_pixel(edge, edge, image::Rgb([0, 0, 0]));
        draw_figure(&mut pose, cx, cy, scale, i, [255, 255, 255]);

        let (tname, pname) = (format!("target_{i}.png"), format!("pose_{i}.png"));
        target.save(dir.join(&tname)).expect("save target");
        pose.save(dir.join(&pname)).expect("save pose");
        manifest.push_str(&format!(
            "{}\n",
            serde_json::json!({
                "id": format!("synth_{i}"),
                "target": tname,
                "pose": pname,
                "caption": format!("a person standing in a colorful room, synthetic pose sample {i}"),
                "coco_image_id": 0,
            })
        ));
    }
    std::fs::write(dir.join("manifest.jsonl"), manifest).expect("write manifest");
    eprintln!("synthesized {count} pairs into {}", dir.display());
}

// ── main ─────────────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = parse_args();
    if a.synth > 0 {
        synth_pairs(&a.data, a.synth);
    }
    let rows = read_manifest(&a.data);
    if rows.is_empty() {
        return Err("manifest is empty".into());
    }
    std::fs::create_dir_all(&a.out)?;
    let device = candle_gen::default_device()?;
    let edge = candle_gen::train::dataset::bucket_resolution(a.resolution);
    eprintln!(
        "krea control trainer: {} items, {} steps, batch {}, lr {}, n_blocks {}, res {edge}, branch dtype {:?}",
        rows.len(),
        a.steps,
        a.batch,
        a.lr,
        a.n_blocks,
        a.branch_dtype
    );

    // ── cache: VAE-encode targets + poses, TE-encode captions (all f32); drop the encoders ──
    let t_cache = Instant::now();
    let vae_enc = QwenVaeEncoder::new(component_vb(
        &a.snapshot,
        "vae",
        &device,
        DType::F32,
        "krea control trainer",
    )?)?;
    let tokenizer = KreaTokenizer::from_snapshot(&a.snapshot, &device)?;
    let te_cfg = KreaTeConfig::from_snapshot(&a.snapshot)?;
    let te_w = Weights::from_dir(&a.snapshot.join("text_encoder"), &device, DType::F32)?;
    let te = KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;

    // (x0 latent, control latent, caption stack), all f32 on device.
    let mut cache: Vec<(Tensor, Tensor, Tensor)> = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let target = candle_gen::train::dataset::load_image_tensor(
            &a.data.join(&row.target),
            edge,
            &device,
        )?;
        let pose =
            candle_gen::train::dataset::load_image_tensor(&a.data.join(&row.pose), edge, &device)?;
        let x0 = vae_enc.encode(&target)?;
        let ctrl = vae_enc.encode(&pose)?;
        let ids = tokenizer.encode_prompt(&row.caption, MAX_TEXT_TOKENS)?;
        let cap = te.forward(&ids)?.to_dtype(DType::F32)?; // (1, L, layers, hidden)
        cache.push((x0, ctrl, cap));
        eprintln!("cached {}/{}", i + 1, rows.len());
    }
    drop(te);
    drop(te_w);
    drop(vae_enc);
    eprintln!("cache done in {:.1}s", t_cache.elapsed().as_secs_f32());

    // ── frozen base DiT (bf16) + trainable branch ──
    let cfg = Krea2Config::from_snapshot(&a.snapshot)?;
    let dit_w = Weights::from_dir(&a.snapshot.join("transformer"), &device, DType::BF16)?;
    let dit = KreaTrainDit::load(&dit_w, &cfg)?;

    let (branch, start_step) = match &a.resume {
        Some(ckpt) => {
            let b = ControlBranch::from_checkpoint(ckpt, &cfg, &device)?;
            let meta = ckpt.with_extension("json");
            let step = std::fs::read_to_string(&meta)
                .ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                .and_then(|v| v.get("step").and_then(|s| s.as_u64()))
                .unwrap_or(0) as u32;
            eprintln!(
                "resumed {} ({} blocks, {} params) at step {step}",
                ckpt.display(),
                b.num_blocks(),
                b.num_params()
            );
            (b, step)
        }
        None => {
            let b = ControlBranch::from_base(&dit_w, &cfg, a.n_blocks, a.branch_dtype)?;
            eprintln!(
                "fresh branch: {} blocks, {:.2}B trainable params",
                b.num_blocks(),
                b.num_params() as f64 / 1e9
            );
            (b, 0)
        }
    };
    drop(dit_w);

    let vars = branch.vars();
    let mut opt = TrainOptimizer::from_config("adamw", vars.clone(), a.lr, 0.0)?;

    let log_path = a.out.join("train_log.jsonl");
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;

    let save = |branch: &ControlBranch, step: u32| -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = a.out.join(format!("control_step{step}.safetensors"));
        branch.save(&path)?;
        std::fs::write(
            path.with_extension("json"),
            serde_json::json!({
                "step": step,
                "n_blocks": branch.num_blocks(),
                "baseModel": "krea_2_turbo",
                "family": "krea_2",
                "kind": "pose_control_branch",
                "resolution": edge,
            })
            .to_string(),
        )?;
        Ok(path)
    };

    // ── train loop: micro-steps over the batch, averaged grads, clip, AdamW ──
    for step in start_step..(start_step + a.steps) {
        let t0 = Instant::now();
        let mut acc = None;
        let mut loss_sum = 0f32;
        for j in 0..a.batch {
            let micro = step * a.batch + j;
            let (x0, ctrl, cap) = &cache[(micro as usize) % cache.len()];
            let sigma = sample_unit_timestep(
                &a.timestep_type,
                "none",
                flow_match::timestep_seed(a.seed, micro),
            );
            let noise = sample_noise(x0.dims(), flow_match::noise_seed(a.seed, micro), &device)?;
            let (x_t, target) = flow_match::build_batch(x0, &noise, sigma as f64)?;
            let t = Tensor::from_vec(vec![sigma], (1,), &device)?;
            let v = forward_with_control(&dit, &branch, &x_t, &t, cap, ctrl, 1.0)?;
            let loss = velocity_loss(&v, &target, false)?;
            loss_sum += loss.to_vec0::<f32>()?;
            let grads = loss.backward()?;
            accumulate_grads(&mut acc, grads, &vars)?;
        }
        let mut grads = acc.expect("batch >= 1");
        scale_grads(&mut grads, &vars, 1.0 / a.batch as f64)?;
        let gnorm = clip_grad_norm(&mut grads, &vars, 1.0)?;
        opt.step(&grads)?;

        let loss = loss_sum / a.batch as f32;
        let secs = t0.elapsed().as_secs_f32();
        println!(
            "step {:>5}/{} loss {loss:.5} grad_norm {gnorm:.3} {secs:.2}s",
            step + 1,
            start_step + a.steps
        );
        writeln!(
            log,
            "{}",
            serde_json::json!({"step": step + 1, "loss": loss, "grad_norm": gnorm, "secs": secs, "lr": a.lr})
        )?;
        if !loss.is_finite() {
            return Err(format!("non-finite loss at step {}", step + 1).into());
        }

        let done = step + 1 == start_step + a.steps;
        if (a.save_every > 0 && (step + 1) % a.save_every == 0) || done {
            let p = save(&branch, step + 1)?;
            eprintln!("checkpoint -> {}", p.display());
        }
    }
    eprintln!("training complete; log at {}", log_path.display());
    Ok(())
}
