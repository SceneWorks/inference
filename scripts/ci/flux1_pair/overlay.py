"""Identical depth-only constructor substitutions and observation taps for isolated proof builds."""
from pathlib import Path
import subprocess

from scripts.ci.flux1_pair.fixture import ENG4B, ENG4B_REAL, ENG4B_TIERS

HERE = Path(__file__).resolve().parent
MLX = "crates/media/mlx-gen/"
CANDLE = "crates/media/candle-gen/"
BASE = "2b09938e33370915d4f853389961709b8bbf8bf9"
ENG4B_BASE = "e86b8d5fc493c16b2d9d852e0c9ecbdefee58294"
ENG4B_BASE_TREE = "88e1369f11fbc1be2d2fb9b0063db8725463db92"
ENG4B_TERMINAL_BASE = "be45bdf746c5d4a12d4db76f9653b8686e7f33f8"
ENG4B_TERMINAL_BASE_TREE = "4d6eb59e08edcd257fd907247acd5e8f0cc5c545"


def eng4b_baseline(tier):
    if tier == ENG4B_REAL:
        return ENG4B_TERMINAL_BASE, ENG4B_TERMINAL_BASE_TREE
    return ENG4B_BASE, ENG4B_BASE_TREE


def replace_once(root, path, old, new):
    target = root / path
    body = target.read_text(encoding="utf-8")
    if body.count(old) != 1:
        raise ValueError(f"proof overlay no longer binds uniquely in {path}: {old[:64]}")
    target.write_text(body.replace(old, new), encoding="utf-8")


def install(root, backend, arm, tier):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise ValueError("overlay requires a fresh clean proof checkout")
    eng4b = tier in ENG4B_TIERS
    measured = tier == ENG4B_REAL
    if eng4b:
        if backend != "candle-cuda" or arm not in ("base", "head"):
            raise ValueError("ENG-4b overlay requires the Candle CUDA comparison arms")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                       encoding="utf-8").strip()
        baseline, _ = eng4b_baseline(tier)
        if (head == baseline) != (arm == "base"):
            raise ValueError("ENG-4b overlay received the wrong source revision for its arm")
    mlx = backend == "mlx"
    planned = arm == "head" or eng4b
    family = MLX + "mlx-gen-flux" if mlx else CANDLE + "candle-gen-flux"
    if tier in ("tier1", ENG4B):
        if mlx:
            path = family + "/src/transformer.rs"
            replace_once(root, path, "num_layers: 19,\n            num_single_layers: 38,",
                         "num_layers: 1,\n            num_single_layers: 1,")
            path = family + "/src/text_encoder.rs"
            replace_once(root, path, "for i in 0..12 {", "for i in 0..1 {")
            replace_once(root, path, "pub const T5_BLOCKS: usize = 24;", "pub const T5_BLOCKS: usize = 1;")
        else:
            text_encoder_sources = [
                path for path in (
                    family + "/src/packed_te.rs",
                    CANDLE + "candle-gen/src/text_encoder.rs",
                ) if (root / path).is_file()
            ]
            if len(text_encoder_sources) != 1:
                raise ValueError("proof overlay requires exactly one Candle FLUX text encoder source")
            path = text_encoder_sources[0]
            if eng4b and (path.endswith("/packed_te.rs") != (arm == "base")):
                raise ValueError("ENG-4b overlay did not bind the expected text encoder for its arm")
            replace_once(root, path, "num_hidden_layers: 12,", "num_hidden_layers: 1,")
            replace_once(root, path, "num_layers: 24,", "num_layers: 1,")
    path = (MLX if mlx else CANDLE + "candle-gen/") + "src/sampler.rs"
    # Observe the exact input to the existing sampler at each Euler outer
    # step, plus its terminal output just before unpack/decode. No new draws.
    if mlx:
        start = "if step_observer.is_step_start(sigmas, sigma) {\n            on_latent(x, sigma);"
        replace_once(root, path, start, start.replace("on_latent", "pair_observe(x, sigma);\n            on_latent"))
        if arm == "head":
            end = '''if phase_scope.is_none() {
        step_eval(&[&out])?;
    }
    crate::ios_quantum::diagnostic_metadata(
        "curated",'''
        else:
            end = 'which evals only inputs).\n    step_eval(&[&out])?;\n    crate::ios_quantum::diagnostic_metadata(\n        "curated",'
        replace_once(root, path, end, end.replace("    crate::", "    pair_observe(&out, 0.0);\n    crate::", 1))
        helper = """fn pair_observe(x: &Array, sigma: f32) {
    assert_eq!(x.shape(), &[1, 256, 64]);
    let device = mlx_rs::Stream::task_local_or_default().get_device().unwrap().get_type().unwrap();
    assert!(matches!(device, mlx_rs::DeviceType::Gpu));
    let state = crate::array::contiguous(&x.as_dtype(Dtype::Float32).unwrap()).unwrap();
    pair_write(state.as_slice::<f32>(), sigma, &format!("{:?}", x.dtype()), "mlx-metal");
}
"""
    else:
        if planned:
            start = "let mut denoise_fn = |x: &Tensor, sigma: f32| -> gen_core::Result<Tensor> {\n                if cancel.is_cancelled()"
            replace_once(root, path, start, start.replace("                if cancel.", "                pair_observe(x, sigma);\n                if cancel."))
            end = '''    if state.next_step() != sigmas.len().saturating_sub(1) {
        return Err(CandleError::from(gen_core::Error::Unsupported(
            "sampler spans do not cover the executed schedule".into(),
        )));
    }
    Ok(out)
}'''
            replace_once(root, path, end, end.replace("    Ok(out)", "    pair_observe(&out, 0.0);\n    Ok(out)"))
        else:
            start = "let mut denoise_fn = |x: &Tensor, sigma: f32| -> gen_core::Result<Tensor> {\n        if cancel.is_cancelled()"
            replace_once(root, path, start, start.replace("        if cancel.", "        pair_observe(x, sigma);\n        if cancel."))
            end = """    sampler
        .sample(&ops, ms, &mut denoise_fn, latents, sigmas, seed)
        .map_err(CandleError::from)
}"""
            replace_once(root, path, end, end.replace("    sampler", "    let out = sampler").replace(
                ".map_err(CandleError::from)\n}", ".map_err(CandleError::from)?;\n    pair_observe(&out, 0.0);\n    Ok(out)\n}"))
        helper = """fn pair_observe(x: &Tensor, sigma: f32) {
    assert_eq!(x.dims(), &[1, 256, 64]);
    assert!(x.device().is_cuda());
    let state = x.to_dtype(candle_core::DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
    pair_write(&state, sigma, &format!("{:?}", x.dtype()), "candle-cuda");
}
"""
    helper += """
fn pair_write(values: &[f32], sigma: f32, dtype: &str, device: &str) {
    use std::io::Write;
    let root = std::path::PathBuf::from(std::env::var_os("PAIR_OUT").unwrap());
    let id = sigma.to_bits();
    let mut out = std::fs::File::create_new(root.join(format!("state-{id:08x}.bin"))).unwrap();
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values { assert!(v.is_finite()); bytes.extend(v.to_le_bytes()); }
    out.write_all(&bytes).unwrap();
    println!("FLUX1_PAIR_STATE {id:08x} {dtype} {device}");
}
"""
    if measured:
        # No readbacks, hashes or output writes inside the measured sampler.
        start = 'fn pair_observe(x: &Tensor, sigma: f32) {\n'
        helper = helper.replace(start, start + """    static MEASURE: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::args().nth(1).as_deref() == Some("--measure"));
    if *MEASURE {
        pair_measure_step(x, sigma);
        return;
    }
""", 1)
        helper += """thread_local! {
    static PAIR_STEP_CLOCK: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
    static PAIR_STEP_SAMPLE: std::cell::Cell<Option<(u128, candle_core::DType)>> = const { std::cell::Cell::new(None) };
}

fn pair_measure_step(x: &Tensor, sigma: f32) {
    if sigma == 1.0 {
        assert!(x.device().is_cuda());
        assert_eq!(x.dims(), &[1, 256, 64]);
        x.device().synchronize().unwrap();
        PAIR_STEP_CLOCK.with(|clock| assert!(clock.replace(Some(std::time::Instant::now())).is_none()));
    } else if sigma == 0.0 {
        assert!(x.device().is_cuda());
        x.device().synchronize().unwrap();
        let elapsed = PAIR_STEP_CLOCK.with(|clock| clock.take().expect("started step clock").elapsed());
        PAIR_STEP_SAMPLE.with(|sample| assert!(sample.replace(Some((elapsed.as_nanos(), x.dtype()))).is_none()));
    }
}

#[doc(hidden)]
pub fn pair_take_step_sample() -> Option<(u128, candle_core::DType)> {
    PAIR_STEP_SAMPLE.with(|sample| sample.take())
}
"""
    with (root / path).open("a", encoding="utf-8") as handle:
        handle.write("\n" + helper)
    engine = "mlx_gen" if mlx else "candle_gen"
    generate = """let generator = registry.load(&model, &spec)?;
    let output = generator.generate(&request, &mut progress)?;"""
    if planned:
        generate = """let plan = registry.compile_render_plan(&model, &spec, &request)?.ok_or("unregistered plan")?;
    fs::write(dir.join("plan.json"), serde_json::to_vec(&plan)?)?;
    if std::env::args().nth(1).as_deref() == Some("--preflight") {
        println!("FLUX1_PAIR_PREFLIGHT {model}");
        return Ok(());
    }
    let generator = registry.load(&model, &spec)?;
    let report = generator.generate_planned(&request, &plan, None, &mut progress)?;
    let output = report.output.ok_or("missing planned output")?;"""
    if measured:
        generate = generate.replace(
            "    let report =",
            """    let measure = std::env::args().nth(1).as_deref() == Some("--measure");
    let warmed = if measure {
        let mut warm_steps = Vec::new();
        let warm = generator.generate_planned(&request, &plan, None, &mut |event| {
            if let Progress::Step { current, total } = event {
                warm_steps.push([current, total]);
            }
        })?;
        if warm_steps != [[1, 3], [2, 3], [3, 3]] {
            return Err("warmup did not complete three Euler steps".into());
        }
        let Some(GenerationOutput::Images(mut images)) = warm.output else {
            return Err("warmup did not generate images".into());
        };
        if images.len() != 1 || (images[0].width, images[0].height) != (256, 256) {
            return Err("wrong warmup render count or geometry".into());
        }
        emit_step_sample()?;
        Some(images.remove(0).pixels)
    } else {
        None
    };
    let started = std::time::Instant::now();
    let report =""", 1)
        generate = generate.replace(
            "    let output =",
            """    let elapsed = started.elapsed();
    let output =""", 1)
    template = (HERE / "probe.rs.in").read_text(encoding="utf-8")
    cuda_proof = ""
    if not mlx:
        cuda_proof = """let pool = candle_gen::cuda_mempool::MemPool::device_current(0)
        .ok_or("CUDA current memory pool unavailable")?;
    let used_high = pool.used_high().ok_or("CUDA pool used high-water unavailable")?;
    let used = pool.used().ok_or("CUDA pool used bytes unavailable")?;
    let reserved = pool.reserved().ok_or("CUDA pool reserved bytes unavailable")?;
    let reserved_high = pool.reserved_high().ok_or("CUDA pool reserved high-water unavailable")?;
    if used == 0 || reserved == 0 || used > used_high || used > reserved
        || used_high > reserved_high || reserved_high < reserved {
        return Err("CUDA pool did not record coherent positive allocation".into());
    }
    let uuid = env::var("CUDA_VISIBLE_DEVICES")?;
    let proof = serde_json::json!({"pid": std::process::id(), "cudaDeviceUuid": uuid,
        "poolUsedBytes": used, "poolUsedHighBytes": used_high, "poolReservedBytes": reserved,
        "poolReservedHighBytes": reserved_high});
    let pending = dir.join("cuda-pool.pending");
    let final_path = dir.join("cuda-pool.json");
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(&pending)?;
    use std::io::Write;
    file.write_all(&serde_json::to_vec(&proof)?)?;
    file.sync_all()?;
    if final_path.exists() { return Err("CUDA pool proof destination exists".into()); }
    fs::rename(pending, final_path)?;
    // The supervisor polls every second. Keep the owned CUDA context alive across ticks.
    std::thread::sleep(std::time::Duration::from_millis(3500));
    drop(generator);"""
    spec = "let spec = spec.with_quant(mlx_gen::gen_core::Quant::Q4);" if mlx and tier == "tier2" else ""
    body = template.replace("@ENGINE@", engine).replace("@PROVIDER@", engine + "_flux")
    if eng4b:
        # Permit compile-time and native plan-only validation without a CUDA
        # toolchain. Rendering still refuses a non-CUDA build before any model load.
        body = body.replace("    let model =", """    if !cfg!(feature = "cuda") && std::env::args().nth(1).as_deref() != Some("--preflight") {
        return Err("FLUX.1 ENG-4b proof requires CUDA".into());
    }
    let model =""", 1)
        cuda_proof = '#[cfg(feature = "cuda")] {\n' + cuda_proof + '\n}'
    if measured:
        body = body.replace("    @TIER_SPEC@\n", "", 1)
        body = body.replace('    fs::write(dir.join("output.rgb")', '''    if let Some(warmed) = warmed {
        if warmed != images[0].pixels {
            return Err("resident warmup changed the seeded RGB8 result".into());
        }
        emit_step_sample()?;
        println!("FLUX1_PAIR_TIMING {} {}", elapsed.as_nanos(), peak_working_set()?);
    }
    fs::write(dir.join("output.rgb")''', 1)
        body += (HERE / "performance.rs.in").read_text(encoding="utf-8")
    destination = root / family / "src/bin/flux1-pair.rs"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("x", encoding="utf-8") as out:
        out.write(body.replace("@TIER_SPEC@", spec).replace("@GENERATE@", generate)
                  .replace("@CUDA_PROOF@", cuda_proof))
    return family.split("/")[-1]
