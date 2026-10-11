//! S0 random tile gather measurement, not learned Veda or image-quality evidence.
//! CPU: cargo test --locked -p candle-gen-minimax-h3 --example veda_s0_cuda
//! CUDA: cargo run --locked --release -p candle-gen-minimax-h3 --features cuda,veda-s0-profile --example veda_s0_cuda -- --case all --repeat 1
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen_minimax_h3::pipeline::{resolve_geometry, t2va_layout, PATCH_SIZE};
use serde_json::{json, Value};
use std::{collections::BTreeMap, env, error::Error, path::PathBuf, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[derive(Clone)]
struct Case {
    name: String,
    seq: usize,
    global: usize,
    heads: usize,
    dim: usize,
    tile: usize,
    geometry: Value,
}
struct Options {
    case: String,
    repeat: usize,
    seed: u64,
    budget: usize,
    batch: usize,
    real_dit: Option<PathBuf>,
    cpu_check: bool,
}
#[derive(Clone)]
struct Block {
    head: usize,
    start: usize,
    len: usize,
    keys: Vec<u32>,
}
struct Plan {
    blocks: Vec<Block>,
    pairs: u64,
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
        repeat: 1,
        seed: 25140,
        budget: candle_gen::ATTN_SCORES_BUDGET,
        batch: 16,
        real_dit: None,
        cpu_check: false,
    };
    let mut args = env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--cpu-check" {
            o.cpu_check = true;
            continue;
        }
        let v = args.next().ok_or("option requires a value")?;
        match key.as_str() {
            "--case" => o.case = v,
            "--repeat" => o.repeat = v.parse()?,
            "--seed" => o.seed = v.parse()?,
            "--score-budget" => o.budget = v.parse()?,
            "--gather-batch" => o.batch = v.parse()?,
            "--real-dit" => o.real_dit = Some(v.into()),
            _ => return Err(format!("unknown option {key}").into()),
        }
    }
    if o.repeat == 0 || o.budget == 0 || o.batch == 0 {
        return Err("positive repeat/budget/batch required".into());
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
            geometry: json!({"synthetic":true}),
        }]);
    }
    let sizes = match name {
        "small" => vec![(576, 320)],
        "default" => vec![(1344, 768)],
        "all" => vec![(576, 320), (1344, 768)],
        _ => return Err("case must be smoke/small/default/all".into()),
    };
    sizes.into_iter().map(|(w,h)| {
        let g = resolve_geometry(w,h,124)?;
        let layout = t2va_layout(&g,512,PATCH_SIZE,&Device::Cpu)?;
        // A permutation groups the globally dense text/audio rows first, as in the MLX S0 example.
        // Random masks are not tied to original packed video positions.
        let expected = if w == 576 { 7586 } else { 38222 };
        if layout.seq_len() != expected { return Err("S0 token geometry changed".into()); }
        Ok(Case { name:format!("{w}x{h}"), seq:layout.seq_len(), global:layout.num_text_tokens()+layout.audio_indices().len(), heads:56, dim:128, tile:128,
            geometry:json!({"width":w,"height":h,"frames":124,"text_tokens":512,"audio_tokens":layout.audio_indices().len(),"video_tokens":layout.video_indices().len(),"latent_frames":g.joint.num_latent_frames,"rows_per_frame":layout.rows_per_frame()}) })
    }).collect()
}
fn plan(c: &Case, mut seed: u64) -> Plan {
    let tiles = (c.seq - c.global).div_ceil(c.tile);
    let keep = tiles.div_ceil(10).max(1);
    let mut blocks = Vec::new();
    for head in 0..c.heads {
        for start in (0..c.global).step_by(c.tile) {
            blocks.push(Block {
                head,
                start,
                len: c.tile.min(c.global - start),
                keys: (0..c.seq as u32).collect(),
            });
        }
        for diagonal in 0..tiles {
            let start = c.global + diagonal * c.tile;
            let mut candidates: Vec<usize> = (0..tiles).filter(|&t| t != diagonal).collect();
            for i in (1..candidates.len()).rev() {
                let j = next(&mut seed) as usize % (i + 1);
                candidates.swap(i, j);
            }
            let mut selected = vec![diagonal];
            selected.extend(candidates.into_iter().take(keep - 1));
            selected.sort_unstable();
            let mut keys: Vec<u32> = (0..c.global as u32).collect();
            for t in selected {
                keys.extend(
                    (c.global + t * c.tile) as u32..(c.global + (t + 1) * c.tile).min(c.seq) as u32,
                );
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
    assert!(
        pairs < c.heads as u64 * c.seq as u64 * c.seq as u64,
        "must execute non-dense plan"
    );
    for b in &blocks {
        assert!(b.keys.windows(2).all(|p| p[0] < p[1]));
        assert!(b.keys.iter().all(|&k| (k as usize) < c.seq));
        assert!((b.start..b.start + b.len).all(|q| b.keys.binary_search(&(q as u32)).is_ok()));
        assert!((0..c.global).all(|k| b.keys.binary_search(&(k as u32)).is_ok()));
        if b.start < c.global {
            assert_eq!(b.keys.len(), c.seq);
        }
    }
    Plan { blocks, pairs }
}
fn drive(shape: &[usize], mut seed: u64, device: &Device, dtype: DType) -> Result<Tensor> {
    // Fixed host Box-Muller normal stream, materialized before timing. No device RNG ambiguity.
    let count = shape.iter().product();
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let u = ((next(&mut seed) >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let v = (next(&mut seed) >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u.ln()).sqrt();
        let theta = std::f64::consts::TAU * v;
        values.push((r * theta.cos()) as f32);
        if values.len() < count {
            values.push((r * theta.sin()) as f32);
        }
    }
    Ok(Tensor::from_vec(values, shape, device)?.to_dtype(dtype)?)
}
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>, o: &Options) -> Result<Tensor> {
    Ok(candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        1.0 / (q.dim(3)? as f64).sqrt(),
        mask,
        candle_gen::candle_nn::ops::softmax_last_dim,
        o.budget,
    )?)
}
fn sparse(c: &Case, o: &Options, p: &Plan, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let flatten = |x: &Tensor| x.reshape((c.heads * c.seq, c.dim));
    let (q, k, v) = (flatten(q)?, flatten(k)?, flatten(v)?);
    // Bucket identical query/key lengths: batched SDPA requires no padding/masking or fake keys.
    let mut buckets: BTreeMap<(usize, usize), Vec<&Block>> = BTreeMap::new();
    for b in &p.blocks {
        buckets.entry((b.len, b.keys.len())).or_default().push(b);
    }
    let mut outputs: BTreeMap<(usize, usize), Tensor> = BTreeMap::new();
    for ((len, nkeys), blocks) in buckets {
        for batch in blocks.chunks(o.batch) {
            let qi: Vec<u32> = batch
                .iter()
                .flat_map(|b| (b.start..b.start + b.len).map(move |q| (b.head * c.seq + q) as u32))
                .collect();
            let ki: Vec<u32> = batch
                .iter()
                .flat_map(|b| {
                    b.keys
                        .iter()
                        .map(move |&k| (b.head * c.seq + k as usize) as u32)
                })
                .collect();
            let qi = Tensor::from_vec(qi, (batch.len() * len,), q.device())?;
            let ki = Tensor::from_vec(ki, (batch.len() * nkeys,), q.device())?;
            let qb = q
                .index_select(&qi, 0)?
                .reshape((batch.len(), 1, len, c.dim))?;
            let kb = k
                .index_select(&ki, 0)?
                .reshape((batch.len(), 1, nkeys, c.dim))?;
            let vb = v
                .index_select(&ki, 0)?
                .reshape((batch.len(), 1, nkeys, c.dim))?;
            let out = sdpa(&qb, &kb, &vb, None, o)?;
            for (i, b) in batch.iter().enumerate() {
                outputs.insert((b.head, b.start), out.narrow(0, i, 1)?);
            }
        }
    }
    let mut heads = Vec::new();
    for head in 0..c.heads {
        let rows: Vec<&Tensor> = outputs
            .range((head, 0)..=(head, usize::MAX))
            .map(|(_, t)| t)
            .collect();
        heads.push(Tensor::cat(&rows, 2)?);
    }
    Ok(Tensor::cat(&heads, 1)?)
}
fn relative(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if a.len() != b.len() || a.iter().chain(&b).any(|x| !x.is_finite()) {
        return Err("nonfinite/shape mismatch".into());
    }
    let delta = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    Ok(delta / b.iter().map(|b| b.abs()).fold(1e-6, f32::max))
}
fn self_check_dim(o: &Options, device: &Device, dtype: DType, dim: usize) -> Result<Value> {
    let mut c = cases("smoke")?.remove(0);
    c.dim = dim;
    let p = plan(&c, o.seed);
    let shape = [1, c.heads, c.seq, c.dim];
    let (q, k, v) = (
        drive(&shape, 1, device, dtype)?,
        drive(&shape, 2, device, dtype)?,
        drive(&shape, 3, device, dtype)?,
    );
    let mut mask = vec![f32::NEG_INFINITY; c.heads * c.seq * c.seq];
    for b in &p.blocks {
        for row in b.start..b.start + b.len {
            for &key in &b.keys {
                mask[(b.head * c.seq + row) * c.seq + key as usize] = 0.0;
            }
        }
    }
    let mask = Tensor::from_vec(mask, (1, c.heads, c.seq, c.seq), device)?.to_dtype(dtype)?;
    let actual = sparse(&c, o, &p, &q, &k, &v)?;
    let expected = sdpa(&q, &k, &v, Some(&mask), o)?;
    let error = relative(&actual, &expected)?;
    let tolerance = if dtype == DType::F32 { 1e-5 } else { 0.025 };
    if error > tolerance {
        return Err(format!("masked oracle mismatch {error} > {tolerance}").into());
    }
    Ok(
        json!({"passed":true,"head_dim":dim,"dtype":format!("{dtype:?}"),"relative_max_abs":error,"tolerance":tolerance,"not_dense":true,"padded_keys":false,"retained_pair_fraction":p.pairs as f64/(c.heads as f64*c.seq as f64*c.seq as f64)}),
    )
}
fn self_check(o: &Options, device: &Device, dtype: DType) -> Result<Value> {
    Ok(
        json!({"passed":true,"oracles":[self_check_dim(o,device,dtype,16)?,self_check_dim(o,device,dtype,128)?]}),
    )
}
fn measure(
    device: &Device,
    repeat: usize,
    mut f: impl FnMut() -> Result<Vec<Tensor>>,
) -> Result<Value> {
    drop(f()?);
    device.synchronize()?;
    let mut times = Vec::new();
    for _ in 0..repeat {
        device.synchronize()?;
        let start = Instant::now();
        let out = f()?;
        device.synchronize()?;
        times.push(start.elapsed().as_secs_f64());
        for output in &out {
            let energy = output
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?;
            if !energy.is_finite() || energy <= 1e-12 {
                return Err("nonfinite or degenerate benchmark output".into());
            }
        }
        drop(out);
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    };
    Ok(json!({"median_seconds":median,"sample_seconds":times,"warmups":1,"repeat":repeat}))
}
fn main() -> Result<()> {
    let o = options()?;
    if o.cpu_check {
        println!("{}", self_check(&o, &Device::Cpu, DType::F32)?);
        return Ok(());
    }
    // Never silently fall back to CPU for claimed CUDA measurements.
    let device = Device::new_cuda(0)?;
    let check = self_check(&o, &device, DType::BF16)?;
    for c in cases(&o.case)? {
        if let Some(dir) = &o.real_dit {
            real_forward(&c, &o, dir, &device)?;
            continue;
        }
        let p = plan(&c, o.seed);
        let shape = [1, c.heads, c.seq, c.dim];
        let (q, k, v) = (
            drive(&shape, o.seed, &device, DType::BF16)?,
            drive(&shape, o.seed + 1, &device, DType::BF16)?,
            drive(&shape, o.seed + 2, &device, DType::BF16)?,
        );
        device.synchronize()?;
        let dense = measure(&device, o.repeat, || Ok(vec![sdpa(&q, &k, &v, None, &o)?]))?;
        let gathered = measure(&device, o.repeat, || {
            Ok(vec![sparse(&c, &o, &p, &q, &k, &v)?])
        })?;
        let domain = c.heads as u64 * c.seq as u64 * c.seq as u64;
        println!(
            "{}",
            json!({"instrument":"veda_s0_cuda_random_batched_gather_sdpa","quality_evidence":false,"predictor":"none_random_mask","case":c.name,"geometry":c.geometry,"tokens":c.seq,"global_dense_tokens":c.global,"heads":c.heads,"head_dim":c.dim,"dtype":"bf16","tile":c.tile,"seed":o.seed,"requested_video_tile_retention_percent":10,"actual_total_sparsity_percent":100.0*(1.0-p.pairs as f64/domain as f64),"retained_pairs":p.pairs,"dense_pairs":domain,"forced_diagonal":true,"score_budget_elements":o.budget,"gather_batch":o.batch,"self_check":check,"dense":dense,"sparse":gathered,"timed":"device index allocation + batched gather + existing budgeted SDPA + concatenation + device synchronization; host plan excluded","rng":"host splitmix64 Box-Muller; fixed normal stream; differs from MLX device RNG","logical_cuda_device":0})
        );
    }
    Ok(())
}
#[cfg(feature = "veda-s0-profile")]
fn validate_q4_headers(dir: &std::path::Path) -> Result<usize> {
    use std::io::Read;
    let mut tensors = serde_json::Map::new();
    for file in candle_gen::loader::sorted_safetensors(dir, "S0 q4 headers")? {
        let mut f = std::fs::File::open(&file)?;
        let mut length = [0u8; 8];
        f.read_exact(&mut length)?;
        let length = u64::from_le_bytes(length);
        if length > 64 * 1024 * 1024 {
            return Err("unreasonable safetensors header size".into());
        }
        let mut header = vec![0; length as usize];
        f.read_exact(&mut header)?;
        let header: Value = serde_json::from_slice(&header)?;
        for (key, value) in header.as_object().ok_or("invalid tensor header")? {
            if key != "__metadata__" && tensors.insert(key.clone(), value.clone()).is_some() {
                return Err(format!("duplicate tensor {key}").into());
            }
        }
    }
    let mut count = 0;
    for (name, scales) in &tensors {
        if let Some(base) = name.strip_suffix(".scales") {
            let weight = tensors
                .get(&format!("{base}.weight"))
                .ok_or("packed scales missing weight")?;
            let groups = scales["shape"][1].as_u64().ok_or("scales rank mismatch")?;
            let codes = weight["shape"][1].as_u64().ok_or("weight rank mismatch")?;
            if weight["dtype"] != "U32" || codes == 0 || groups * 64 != codes * 8 {
                return Err(format!("{base} is not q4 group64 packed weights").into());
            }
            count += 1;
        }
    }
    for i in 0..50 {
        for suffix in [
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ] {
            if !tensors.contains_key(&format!("transformer_blocks.{i}.{suffix}.scales")) {
                return Err(format!("missing packed block {i} {suffix}").into());
            }
        }
    }
    Ok(count)
}
#[cfg(not(feature = "veda-s0-profile"))]
fn real_forward(_: &Case, _: &Options, _: &std::path::Path, _: &Device) -> Result<()> {
    Err("real DiT requires veda-s0-profile".into())
}
#[cfg(feature = "veda-s0-profile")]
fn real_forward(c: &Case, o: &Options, dir: &std::path::Path, device: &Device) -> Result<()> {
    use candle_gen_minimax_h3::{
        denoise::{adaln_schedule, JointSchedule, JointStep, JointVelocity, PackedLayout},
        dit::s0_profile,
        dit::{AdaLnResidency, JointDit, MiniMaxH3Dit},
    };
    if c.heads != 56 {
        return Err("real DiT requires small/default/all".into());
    }
    if !dir.join("config.json").is_file() {
        return Err(format!("real DiT config absent: {}", dir.display()).into());
    }
    // Require genuinely packed q4 projections rather than quietly profiling dense weights.
    let packed_projections = validate_q4_headers(dir)?;
    let g = resolve_geometry(
        c.geometry["width"].as_u64().unwrap() as u32,
        c.geometry["height"].as_u64().unwrap() as u32,
        124,
    )?;
    let layout = t2va_layout(&g, 512, PATCH_SIZE, device)?;
    let dit = MiniMaxH3Dit::load_from_dir(dir, device, DType::BF16)?;
    let cfg = dit.config().clone();
    if cfg.num_layers != 50 {
        return Err("expected 50 real DiT blocks".into());
    }
    let joint = JointSchedule::new(3)?;
    let schedule = adaln_schedule(&joint)?;
    let video = drive(
        &[1, layout.video_indices().len(), cfg.in_channels * 4],
        101,
        device,
        DType::BF16,
    )?;
    let audio = drive(
        &[1, layout.audio_indices().len(), cfg.audio_in_channels],
        102,
        device,
        DType::BF16,
    )?;
    let context = drive(&[1, 512, cfg.text_dim], 103, device, DType::BF16)?;
    let indices = schedule.adaln_indices(0, layout.row_classes(), layout.token_tags(), device)?;
    let mut model = JointDit::new(
        dit,
        layout.clone(),
        &context,
        schedule,
        AdaLnResidency::PrecomputeAndEvict,
    )?;
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
    let mut arms = Vec::new();
    for enabled in [false, true] {
        let mut phases = Vec::new();
        let times = measure(device, o.repeat, || {
            let scope = s0_profile::Scope::new(enabled);
            let (v, a) = model.forward(&step)?;
            device.synchronize()?;
            // Keep readback outside the synchronized model timer.

            let sample = scope.finish();
            if enabled && sample.calls != [50; 4] {
                return Err(format!("expected 50 calls per phase, got {:?}", sample.calls).into());
            }
            phases.push(json!({"seconds":sample.seconds,"calls":sample.calls}));
            Ok(vec![v, a])
        })?;
        let warmup = phases.remove(0);
        arms.push(json!({"profile_enabled":enabled,"whole_forward":times,"phase_warmup":warmup,"phase_samples":phases}));
    }
    let perturbation = arms[1]["whole_forward"]["median_seconds"].as_f64().unwrap()
        / arms[0]["whole_forward"]["median_seconds"].as_f64().unwrap()
        - 1.0;
    println!(
        "{}",
        json!({"instrument":"veda_s0_cuda_q4_real_dit_phase","quality_evidence":false,"case":c.name,"geometry":c.geometry,"tokens":c.seq,"dit_directory":dir,"validated_q4_projections":packed_projections,"schedule_steps":3,"evaluation_index":0,"synthetic_context_and_latents":true,"drive_seeds":[101,102,103],"rng":"host splitmix64 Box-Muller; same shapes/seeds as MLX, different RNG","arms":arms,"phase_order":["qkv_norm_rope_layout","sdpa","out_projection_layout","mlp"],"profile_whole_forward_perturbation_percent":100.0*perturbation,"note":"Real cached q4 weights, synthetic inputs; no image-quality claim. Phase synchronization perturbs execution; compare arms."})
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gathered_attention_matches_masked_dense_cpu() {
        let o = Options {
            case: "smoke".into(),
            repeat: 1,
            seed: 25140,
            budget: 4096,
            batch: 3,
            real_dit: None,
            cpu_check: true,
        };
        assert!(self_check(&o, &Device::Cpu, DType::F32).unwrap()["passed"]
            .as_bool()
            .unwrap());
    }
    #[cfg(feature = "veda-s0-profile")]
    #[test]
    fn q4_asset_guard_rejects_q8_headers() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = serde_json::Map::new();
        for i in 0..50 {
            for suffix in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ] {
                let base = format!("transformer_blocks.{i}.{suffix}");
                tensors.insert(
                    format!("{base}.weight"),
                    json!({"dtype":"U32","shape":[8,8],"data_offsets":[0,0]}),
                );
                tensors.insert(
                    format!("{base}.scales"),
                    json!({"dtype":"BF16","shape":[8,1],"data_offsets":[0,0]}),
                );
            }
        }
        let file = dir.path().join("model.safetensors");
        let write = |tensors: &serde_json::Map<String, Value>| {
            let bytes = serde_json::to_vec(tensors).unwrap();
            let mut f = std::fs::File::create(&file).unwrap();
            f.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
            f.write_all(&bytes).unwrap();
        };
        write(&tensors);
        assert_eq!(validate_q4_headers(dir.path()).unwrap(), 300);
        tensors
            .get_mut("transformer_blocks.0.attn.to_q.weight")
            .unwrap()["shape"][1] = json!(16);
        write(&tensors);
        assert!(validate_q4_headers(dir.path()).is_err());
    }
    #[test]
    fn shipped_token_counts_and_global_dense_plan() {
        for c in cases("all").unwrap() {
            let p = plan(&c, 25140);
            assert_eq!(
                p.blocks.iter().map(|b| b.len).sum::<usize>(),
                c.heads * c.seq
            );
        }
    }
}
