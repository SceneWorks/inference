"""Identical depth-only constructor substitutions and observation taps for isolated proof builds."""
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
MLX = "crates/media/mlx-gen/"
CANDLE = "crates/media/candle-gen/"
BASE = "2b09938e33370915d4f853389961709b8bbf8bf9"


def replace_once(root, path, old, new):
    target = root / path
    body = target.read_text(encoding="utf-8")
    if body.count(old) != 1:
        raise ValueError(f"proof overlay no longer binds uniquely in {path}: {old[:64]}")
    target.write_text(body.replace(old, new), encoding="utf-8")


def install(root, backend, arm, tier):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise ValueError("overlay requires a fresh clean proof checkout")
    mlx = backend == "mlx"
    family = MLX + "mlx-gen-flux" if mlx else CANDLE + "candle-gen-flux"
    if tier == "tier1":
        if mlx:
            path = family + "/src/transformer.rs"
            replace_once(root, path, "num_layers: 19,\n            num_single_layers: 38,",
                         "num_layers: 1,\n            num_single_layers: 1,")
            path = family + "/src/text_encoder.rs"
            replace_once(root, path, "for i in 0..12 {", "for i in 0..1 {")
            replace_once(root, path, "pub const T5_BLOCKS: usize = 24;", "pub const T5_BLOCKS: usize = 1;")
        else:
            path = family + "/src/packed_te.rs"
            replace_once(root, path, "num_hidden_layers: 12,", "num_hidden_layers: 1,")
            replace_once(root, path, "num_layers: 24,", "num_layers: 1,")
    path = (MLX if mlx else CANDLE + "candle-gen/") + "src/sampler.rs"
    # Observe the exact input to the existing sampler at each Euler outer
    # step, plus its terminal output just before unpack/decode. No new draws.
    if mlx:
        start = "if step_observer.is_step_start(sigmas, sigma) {\n            on_latent(x, sigma);"
        replace_once(root, path, start, start.replace("on_latent", "pair_observe(x, sigma);\n            on_latent"))
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
    with (root / path).open("a", encoding="utf-8") as handle:
        handle.write("\n" + helper)
    engine = "mlx_gen" if mlx else "candle_gen"
    generate = """let generator = registry.load(&model, &spec)?;
    let output = generator.generate(&request, &mut progress)?;"""
    if arm == "head":
        generate = """let plan = registry.compile_render_plan(&model, &spec, &request)?.ok_or("unregistered plan")?;
    fs::write(dir.join("plan.json"), serde_json::to_vec(&plan)?)?;
    if std::env::args().nth(1).as_deref() == Some("--preflight") {
        println!("FLUX1_PAIR_PREFLIGHT {model}");
        return Ok(());
    }
    let generator = registry.load(&model, &spec)?;
    let report = generator.generate_planned(&request, &plan, None, &mut progress)?;
    let output = report.output.ok_or("missing planned output")?;"""
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
    destination = root / family / "src/bin/flux1-pair.rs"
    destination.parent.mkdir(exist_ok=True)
    with destination.open("x", encoding="utf-8") as out:
        out.write(body.replace("@TIER_SPEC@", spec).replace("@GENERATE@", generate)
                  .replace("@CUDA_PROOF@", cuda_proof))
    return family.split("/")[-1]
