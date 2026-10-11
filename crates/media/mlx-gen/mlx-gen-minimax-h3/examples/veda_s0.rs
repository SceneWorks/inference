//! Pre-planning S0 instrument: RANDOM masks, not Veda predictor or image-quality evidence.
//! Build: cargo build --locked --release -p mlx-gen-minimax-h3 --example veda_s0
//! Run --case smoke first; --case all selects both shipped canvases at 124f/512 text rows.
//! The host random mask plan is prepared outside timing (predictor cost is not measured).
//! Device index allocation, gathers, SDPA, concatenation and synchronous eval are timed.
use mlx_gen::attention::{sdpa_bounded_bhsd, slice_axis, AttentionBudget, BoundedAttention};
use mlx_gen_minimax_h3::pipeline::{resolve_geometry, t2va_layout, PATCH_SIZE};
use mlx_rs::{ops::concatenate_axis, Array, Dtype};
use serde_json::{json, Value};
use std::{env, error::Error, path::PathBuf, process::Command, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[derive(Clone)]
struct Case {
    name: String,
    seq: i32,
    global: i32,
    heads: i32,
    dim: i32,
    tile: i32,
    geometry: Value,
}
struct Options {
    case: String,
    repeat: usize,
    seed: u64,
    head_chunk: i32,
    score_budget: u64,
    real_dit: Option<PathBuf>,
}
#[derive(Clone)]
struct Block {
    head: i32,
    start: i32,
    len: i32,
    keys: Vec<i32>,
}
struct Plan {
    blocks: Vec<Block>,
    pairs: u64,
    max_keys: usize,
    host_bytes: usize,
}
fn next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn options() -> Result<Options> {
    let mut o = Options {
        case: "smoke".into(),
        repeat: 3,
        seed: 25140,
        head_chunk: 56,
        score_budget: u64::MAX,
        real_dit: None,
    };
    let mut args = env::args().skip(1);
    while let Some(key) = args.next() {
        let value = args.next().ok_or("every option requires a value")?;
        match key.as_str() {
            "--case" => o.case = value,
            "--repeat" => o.repeat = value.parse()?,
            "--seed" => o.seed = value.parse()?,
            "--head-chunk" => o.head_chunk = value.parse()?,
            "--score-budget" => o.score_budget = value.parse()?,
            "--real-dit" => o.real_dit = Some(value.into()),
            _ => return Err(format!("unknown option {key}").into()),
        }
    }
    if o.repeat == 0 || o.head_chunk <= 0 || o.score_budget == 0 {
        return Err("repeat, head-chunk and score-budget must be positive".into());
    }
    Ok(o)
}
fn cases(name: &str) -> Result<Vec<Case>> {
    if name == "smoke" {
        return Ok(vec![Case {
            name: name.into(),
            seq: 273,
            global: 17,
            heads: 2,
            dim: 16,
            tile: 16,
            geometry: json!({"synthetic": true}),
        }]);
    }
    let sizes = match name {
        "small" => vec![(576, 320)],
        "default" => vec![(1344, 768)],
        "all" => vec![(576, 320), (1344, 768)],
        _ => return Err("--case must be smoke, small, default or all".into()),
    };
    sizes.into_iter().map(|(w,h)| {
        let g = resolve_geometry(w,h,124)?;
        let layout = t2va_layout(&g,512,PATCH_SIZE)?;
        let global = layout.num_text_tokens() + layout.audio_indices().len() as i32;
        Ok(Case { name: format!("{w}x{h}"), seq: layout.seq_len(), global, heads: 56, dim: 128, tile: 128,
            geometry: json!({"width":w,"height":h,"frames":124,"text_tokens":512,"audio_tokens":layout.audio_indices().len(),"video_tokens":layout.video_indices().len(),"latent_frames":g.joint.num_latent_frames,"rows_per_frame":layout.rows_per_frame()}) })
    }).collect()
}
fn plan(c: &Case, mut seed: u64) -> Plan {
    let tiles = (c.seq - c.global + c.tile - 1) / c.tile;
    let keep = ((tiles + 9) / 10).max(1);
    let mut blocks = Vec::new();
    for head in 0..c.heads {
        for start in (0..c.global).step_by(c.tile as usize) {
            blocks.push(Block {
                head,
                start,
                len: c.tile.min(c.global - start),
                keys: (0..c.seq).collect(),
            });
        }
        for diagonal in 0..tiles {
            let start = c.global + diagonal * c.tile;
            let mut candidates: Vec<i32> = (0..tiles).filter(|&t| t != diagonal).collect();
            for i in (1..candidates.len()).rev() {
                let j = next(&mut seed) as usize % (i + 1);
                candidates.swap(i, j);
            }
            let mut selected = vec![diagonal];
            selected.extend(candidates.into_iter().take((keep - 1) as usize));
            selected.sort_unstable();
            let mut keys: Vec<i32> = (0..c.global).collect();
            for t in selected {
                keys.extend(c.global + t * c.tile..(c.global + (t + 1) * c.tile).min(c.seq));
            }
            blocks.push(Block {
                head,
                start,
                len: c.tile.min(c.seq - start),
                keys,
            });
        }
    }
    let pairs = blocks
        .iter()
        .map(|b| b.len as u64 * b.keys.len() as u64)
        .sum();
    let max_keys = blocks.iter().map(|b| b.keys.len()).max().unwrap();
    let host_bytes = blocks
        .iter()
        .map(|b| b.keys.capacity() * 4 + std::mem::size_of::<Block>())
        .sum();
    assert!(
        pairs < c.heads as u64 * c.seq as u64 * c.seq as u64,
        "must execute a non-dense plan"
    );
    for b in &blocks {
        assert!(b.keys.windows(2).all(|p| p[0] < p[1]));
        assert!(
            b.keys.iter().all(|&k| k >= 0 && k < c.seq),
            "no padded keys"
        );
        assert!(
            (b.start..b.start + b.len).all(|q| b.keys.binary_search(&q).is_ok()),
            "forced diagonal missing"
        );
        assert!(
            (0..c.global).all(|k| b.keys.binary_search(&k).is_ok()),
            "global keys must be dense"
        );
    }
    Plan {
        blocks,
        pairs,
        max_keys,
        host_bytes,
    }
}
fn drive(shape: &[i32], seed: u64) -> Result<Array> {
    Ok(
        mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed)?))?
            .as_dtype(Dtype::Bfloat16)?,
    )
}
fn bounded(o: &Options) -> BoundedAttention<'static> {
    BoundedAttention::heads(AttentionBudget::from_score_elements(o.score_budget, true))
}
fn dense(
    c: &Case,
    o: &Options,
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
) -> Result<Array> {
    let mut heads = Vec::new();
    for start in (0..c.heads).step_by(o.head_chunk as usize) {
        let count = o.head_chunk.min(c.heads - start);
        let qs = slice_axis(q, 1, start, count)?;
        let ks = slice_axis(k, 1, start, count)?;
        let vs = slice_axis(v, 1, start, count)?;
        let m = mask.map(|a| slice_axis(a, 1, start, count)).transpose()?;
        let out = sdpa_bounded_bhsd(
            &qs,
            &ks,
            &vs,
            1.0 / (c.dim as f32).sqrt(),
            m.as_ref(),
            bounded(o),
        )?;
        mlx_rs::transforms::eval([&out])?;
        heads.push(out);
    }
    let refs: Vec<&Array> = heads.iter().collect();
    let out = concatenate_axis(&refs, 1)?;
    mlx_rs::transforms::eval([&out])?;
    Ok(out)
}
fn sparse(c: &Case, o: &Options, p: &Plan, q: &Array, k: &Array, v: &Array) -> Result<Array> {
    let mut heads = Vec::new();
    for head in 0..c.heads {
        let qh = slice_axis(q, 1, head, 1)?;
        let kh = slice_axis(k, 1, head, 1)?;
        let vh = slice_axis(v, 1, head, 1)?;
        let mut rows = Vec::new();
        for b in p.blocks.iter().filter(|b| b.head == head) {
            let qb = slice_axis(&qh, 2, b.start, b.len)?;
            let indices = Array::from_slice(&b.keys, &[b.keys.len() as i32]);
            let kb = kh.take_axis(&indices, 2)?;
            let vb = vh.take_axis(&indices, 2)?;
            let out =
                sdpa_bounded_bhsd(&qb, &kb, &vb, 1.0 / (c.dim as f32).sqrt(), None, bounded(o))?;
            mlx_rs::transforms::eval([&out])?;
            rows.push(out);
        }
        let refs: Vec<&Array> = rows.iter().collect();
        let out = concatenate_axis(&refs, 2)?;
        mlx_rs::transforms::eval([&out])?;
        heads.push(out);
    }
    let refs: Vec<&Array> = heads.iter().collect();
    let out = concatenate_axis(&refs, 1)?;
    mlx_rs::transforms::eval([&out])?;
    Ok(out)
}
fn relative(a: &Array, b: &Array) -> Result<f32> {
    let aa = a.as_dtype(Dtype::Float32)?;
    let bb = b.as_dtype(Dtype::Float32)?;
    mlx_rs::transforms::eval([&aa, &bb])?;
    let a = aa.as_slice::<f32>();
    let b = bb.as_slice::<f32>();
    if a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return Err("nonfinite/shape mismatch".into());
    }
    let delta = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let scale = b.iter().map(|x| x.abs()).fold(0.0, f32::max).max(1e-6);
    Ok(delta / scale)
}
fn self_check(o: &Options) -> Result<Value> {
    let c = Case {
        name: "self_check".into(),
        seq: 97,
        global: 3,
        heads: 2,
        dim: 16,
        tile: 8,
        geometry: Value::Null,
    };
    let p = plan(&c, o.seed);
    let shape = [1, c.heads, c.seq, c.dim];
    let (q, k, v) = (drive(&shape, 1)?, drive(&shape, 2)?, drive(&shape, 3)?);
    mlx_rs::transforms::eval([&q, &k, &v])?;
    // Explicit SMALL dense mask, built independently from the selected key lists.
    let mut mask = vec![f32::NEG_INFINITY; (c.heads * c.seq * c.seq) as usize];
    for b in &p.blocks {
        for row in b.start..b.start + b.len {
            for &key in &b.keys {
                mask[((b.head * c.seq + row) * c.seq + key) as usize] = 0.0;
            }
        }
    }
    // MLX additive masks must promote to the attention output dtype; f32 would
    // promote BF16 QKV to f32 and is rejected. Zero and -infinity are exact in BF16.
    let mask = Array::from_slice(&mask, &[1, c.heads, c.seq, c.seq]).as_dtype(q.dtype())?;
    let actual = sparse(&c, o, &p, &q, &k, &v)?;
    let expected = dense(&c, o, &q, &k, &v, Some(&mask))?;
    let error = relative(&actual, &expected)?;
    // BF16 output plus shape-dependent SDPA dispatch: numeric agreement, not bit identity.
    if error > 0.025 {
        return Err(format!("masked reference mismatch: relative max abs {error}").into());
    }
    Ok(
        json!({"passed":true,"mask_dtype":"bf16","relative_max_abs":error,"tolerance":0.025,"retained_pair_fraction":p.pairs as f64/(c.heads as f64*c.seq as f64*c.seq as f64),"not_dense":true,"padded_keys":false}),
    )
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
fn measure(repeat: usize, mut run: impl FnMut() -> Result<Array>) -> Result<Value> {
    drop(run()?); // JIT/allocator warmup outside the timed samples.
    let mut times = Vec::new();
    let mut peaks = Vec::new();
    let mut extra = Vec::new();
    let mut baselines = Vec::new();
    for _ in 0..repeat {
        mlx_rs::memory::clear_cache();
        let baseline = mlx_rs::memory::get_active_memory();
        mlx_rs::memory::reset_peak_memory();
        let start = Instant::now();
        let out = run()?;
        mlx_rs::transforms::eval([&out])?;
        times.push(start.elapsed().as_secs_f64());
        let peak = mlx_rs::memory::get_peak_memory();
        peaks.push(peak);
        extra.push(peak.saturating_sub(baseline));
        baselines.push(baseline);
        drop(out);
    }
    let median = median(&mut times.clone());
    Ok(
        json!({"median_seconds":median,"sample_seconds":times,"peak_bytes":peaks,"peak_increment_bytes":extra,"active_baseline_bytes":baselines,"warmups":1,"repeat":repeat}),
    )
}
fn command(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}
fn main() -> Result<()> {
    let o = options()?;
    let check = self_check(&o)?;
    let git = command("git", &["rev-parse", "HEAD"]);
    let chip = command("sysctl", &["-n", "machdep.cpu.brand_string"]);
    for c in cases(&o.case)? {
        if let Some(dir) = &o.real_dit {
            real_forward(&c, &o, dir)?;
            continue;
        }
        let p = plan(&c, o.seed);
        let shape = [1, c.heads, c.seq, c.dim];
        let (q, k, v) = (
            drive(&shape, o.seed)?,
            drive(&shape, o.seed + 1)?,
            drive(&shape, o.seed + 2)?,
        );
        mlx_rs::transforms::eval([&q, &k, &v])?;
        let dense = measure(o.repeat, || dense(&c, &o, &q, &k, &v, None))?;
        let sparse = measure(o.repeat, || sparse(&c, &o, &p, &q, &k, &v))?;
        let domain = c.heads as u64 * c.seq as u64 * c.seq as u64;
        println!(
            "{}",
            json!({"instrument":"veda_s0_random_gather_sdpa","quality_evidence":false,"predictor":"none_random_mask","chip":chip,"git":git,"pid":std::process::id(),"case":c.name,"geometry":c.geometry,"tokens":c.seq,"global_dense_tokens":c.global,"heads":c.heads,"head_dim":c.dim,"dtype":"bf16","tile":c.tile,"seed":o.seed,"requested_video_tile_sparsity_percent":90,"actual_total_sparsity_percent":100.0*(1.0-p.pairs as f64/domain as f64),"retained_pairs":p.pairs,"dense_pairs":domain,"forced_diagonal":true,"head_chunk":o.head_chunk,"score_budget_elements":o.score_budget,"sparse_head_chunk":1,"workspace":{"host_plan_bytes":p.host_bytes,"max_device_key_index_bytes":p.max_keys*4,"max_gathered_kv_bytes":p.max_keys*c.dim as usize*2*2,"output_bytes":c.heads as usize*c.seq as usize*c.dim as usize*2,"score_budget_is_logical_not_metal_allocation_limit":true},"self_check":check,"dense":dense,"sparse":sparse,"timed":"device indices + gather + SDPA + concat + sync eval; host mask planning excluded"})
        );
    }
    Ok(())
}
#[cfg(not(feature = "veda-s0-profile"))]
fn real_forward(_: &Case, _: &Options, _: &std::path::Path) -> Result<()> {
    Err("--real-dit requires --features veda-s0-profile".into())
}
#[cfg(feature = "veda-s0-profile")]
fn real_forward(c: &Case, o: &Options, dir: &std::path::Path) -> Result<()> {
    use mlx_gen_minimax_h3::dit::layers::s0_profile;
    use mlx_gen_minimax_h3::{
        denoise::{adaln_schedule, JointSchedule, JointStep, JointVelocity, PackedLayout},
        dit::{AdaLnResidency, JointDit, MiniMaxH3Dit, TimestepSchedule},
    };
    if c.heads != 56 {
        return Err("real DiT requires --case small, default or all".into());
    }
    let w = c.geometry["width"].as_u64().unwrap() as u32;
    let h = c.geometry["height"].as_u64().unwrap() as u32;
    let geometry = resolve_geometry(w, h, 124)?;
    let layout = t2va_layout(&geometry, 512, PATCH_SIZE)?;
    // Same loader/drive/residency as bounded_attention_real::Rig, not a new weight-loading path.
    let dit = MiniMaxH3Dit::load_dir(dir, Dtype::Bfloat16)?;
    let cfg = dit.config().clone();
    let joint = JointSchedule::new(3)?;
    let schedule: TimestepSchedule = adaln_schedule(&joint)?;
    let video = drive(
        &[1, layout.video_indices().len() as i32, cfg.in_channels * 4],
        101,
    )?;
    let audio = drive(
        &[
            1,
            layout.audio_indices().len() as i32,
            cfg.audio_in_channels,
        ],
        102,
    )?;
    let context = drive(&[1, 512, cfg.text_dim], 103)?;
    mlx_rs::transforms::eval([&video, &audio, &context])?;
    let mut model = JointDit::new(
        dit,
        layout.clone(),
        &context,
        schedule.clone(),
        AdaLnResidency::PrecomputeAndEvict,
    )?;
    let indices = schedule.adaln_indices(0, layout.row_classes(), layout.token_tags())?;
    let step = JointStep {
        index: 0,
        layout: &layout,
        video_rows: &video,
        audio_rows: &audio,
        adaln_indices: &indices,
        row_timesteps: PackedLayout::row_timesteps(
            joint.video().timesteps()[0],
            joint.audio().timesteps()[0],
        ),
    };
    let mut summaries = Vec::new();
    for profile in [false, true] {
        let mut phases = Vec::new();
        let result = measure(o.repeat, || {
            let scope = s0_profile::Scope::new(profile);
            let (v, a) = model.forward(&step)?;
            mlx_rs::transforms::eval([&v, &a])?;
            let sample = scope.finish();
            phases.push(json!({"attention_seconds":sample.0,"calls":sample.1}));
            if profile && sample.1 != 50 {
                return Err(
                    format!("expected 50 DiT attention calls, observed {}", sample.1).into(),
                );
            }
            Ok(v)
        })?;
        let warmup = phases.remove(0);
        let shares: Vec<f64> = phases
            .iter()
            .zip(result["sample_seconds"].as_array().unwrap())
            .map(|(p, t)| p["attention_seconds"].as_f64().unwrap() / t.as_f64().unwrap())
            .collect();
        summaries.push(json!({"profile_enabled":profile,"whole_forward":result,"phase_warmup":warmup,"phase_samples":phases,"attention_share_of_profiled_forward":shares}));
    }
    let perturbation = summaries[1]["whole_forward"]["median_seconds"]
        .as_f64()
        .unwrap()
        / summaries[0]["whole_forward"]["median_seconds"]
            .as_f64()
            .unwrap()
        - 1.0;
    println!(
        "{}",
        json!({"instrument":"veda_s0_dense_real_dit_phase","profile_whole_forward_perturbation_percent":100.0*perturbation,"case":c.name,"geometry":c.geometry,"tokens":c.seq,"dit_directory":dir,"evaluation_index":0,"schedule_steps":3,"drive_seed":101,"git":command("git",&["rev-parse","HEAD"]),"chip":command("sysctl",&["-n","machdep.cpu.brand_string"]),"pid":std::process::id(),"arms":summaries,"note":"QKV synchronized before SDPA timer, output after; compare whole-forward arms to quantify graph-cut perturbation. Synthetic latents/context; no image quality claim."})
    );
    Ok(())
}
