"""Generate provider-owned FLUX.1 conditional probes for manual same-device A/B builds."""
from pathlib import Path
import subprocess

from scripts.ci.flux1_pair import conditional, overlay

HERE = Path(__file__).resolve().parent


def mlx_setup(case):
    variant = conditional.variant(case)
    common = '''use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    let mut request = GenerationRequest {
        prompt: "red kite portrait".into(), width: 256, height: 256, count: 1,
        seed: Some(104729), steps: Some(3), sampler: Some("euler".into()),
        guidance: @GUIDANCE@, ..Default::default()
    };
    let model = @MODEL@.to_owned();'''
    if case.startswith("control-"):
        kind = case.split("-", 1)[1]
        kind = "pose" if kind == "zero" else kind
        control = {"pose": "Pose", "canny": "Canny", "depth": "Depth"}[kind]
        scale = "0.0" if case == "control-zero" else "0.7"
        route = f'''spec = spec.with_control(WeightsSource::File(assets.join("control.safetensors")));
    request.conditioning = vec![Conditioning::Control {{
        image: source, kind: mlx_gen::ControlKind::{control}, scale: Some({scale})
    }}];'''
        model = "flux1_dev_control"
    elif case.startswith("ip-"):
        route = '''spec = spec.with_ip_adapter(WeightsSource::Dir(assets.join("ip")));
    request.conditioning = vec![Conditioning::Reference { image: source, strength: Some(0.7) }];'''
        if case.endswith("-cfg"):
            route += '''
    request.true_cfg = Some(2.0);
    request.negative_prompt = Some("blurred".into());'''
        model = "flux1_" + variant
    else:
        route = '''spec.identity = Some(mlx_gen::IdentityWeights {
        encoder: Some(WeightsSource::File(assets.join("pulid.safetensors"))),
        eva: Some(WeightsSource::File(assets.join("eva.safetensors"))),
        face_dir: Some(WeightsSource::Dir(assets.join("face"))),
    });
    request.conditioning = vec![Conditioning::Reference { image: source, strength: Some(1.0) }];'''
        if case in ("pulid-real", "pulid-delayed"):
            route += '''
    request.true_cfg = Some(2.0);
    request.negative_prompt = Some("blurred".into());'''
        if case == "pulid-delayed":
            route += "\n    request.timestep_to_start_cfg = Some(2);"
        if case == "pulid-pid":
            route += '''
    spec = spec.with_pid(
        WeightsSource::File(assets.join("pid_decoder.safetensors")),
        WeightsSource::Dir(PathBuf::from(env::var("PAIR_PID_GEMMA")?)),
    );
    request.use_pid = true;'''
        model = "pulid_flux"
    guidance = "Some(3.5)" if variant == "dev" else "None"
    return (common.replace("@GUIDANCE@", guidance).replace("@MODEL@", f'"{model}"')
            + "\n    " + route + "\n    let registry = @REGISTRY@::provider_registry()?;")


def mlx_generate(arm):
    if arm == "head":
        prefix = '''let plan = registry.compile_render_plan(&model, &spec, &request)?
        .ok_or("missing conditional pre-load plan")?;
    fs::write(dir.join("plan.json"), serde_json::to_vec(&plan)?)?;
    if env::args().nth(1).as_deref() == Some("--preflight") {
        println!("FLUX1_CONDITIONAL_PREFLIGHT {case}");
        return Ok(());
    }
    let generator = registry.load(&model, &spec)?;
    let report = generator.generate_planned(&request, &plan, None, &mut progress)?;
    let rendered = report.output.ok_or("missing conditional output")?;'''
    else:
        prefix = '''let generator = registry.load(&model, &spec)?;
    let rendered = generator.generate(&request, &mut progress)?;'''
    return prefix + '''
    let GenerationOutput::Images(mut images) = rendered else {
        return Err("expected one image".into());
    };
    if images.len() != 1 { return Err("wrong image count".into()); }
    let output = images.remove(0);'''


def candle_setup(case):
    variant = conditional.variant(case)
    if case.startswith("control-"):
        kind = case.split("-", 1)[1]
        kind = "pose" if kind == "zero" else kind
        scale = "0.0" if case == "control-zero" else "0.7"
        return f'''use candle_gen::gen_core::{{GenerationMemory, LoadSpec, WeightsSource}};
    use candle_gen_flux::control_provider::{{Flux1ControlPaths, Flux1ControlRequest, Flux1DevControl}};
    let paths = Flux1ControlPaths {{ flux_base: root.clone(),
        control: assets.join("control.safetensors"), adapters: Vec::new() }};
    let spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_control(WeightsSource::File(paths.control.clone()));
    let req = Flux1ControlRequest {{ prompt: "red kite portrait".into(), width: 256, height: 256,
        steps: 3, sampler: Some("euler".into()), guidance: 3.5,
        control_kind: "{kind}".into(), control_scale: Some({scale}), seed: 104729,
        ..Default::default() }};'''
    if case.startswith("ip-"):
        guidance = "3.5" if variant == "dev" else "0.0"
        return f'''use candle_gen::gen_core::{{GenerationMemory, LoadSpec, WeightsSource}};
    use candle_gen_flux::ip_provider::{{IpAdapterFluxPaths, IpAdapterFluxRequest, IpAdapterFlux}};
    let paths = IpAdapterFluxPaths {{ flux_base: root.clone(),
        ip_adapter: assets.join("ip_adapter.safetensors"),
        image_encoder: assets.join("image_encoder.safetensors"), adapters: Vec::new() }};
    let spec = LoadSpec::new(WeightsSource::Dir(root))
        .with_ip_adapter(WeightsSource::File(paths.ip_adapter.clone()))
        .with_component("flux_ip_image_encoder", WeightsSource::File(paths.image_encoder.clone()));
    let req = IpAdapterFluxRequest {{ prompt: "red kite portrait".into(), width: 256, height: 256,
        steps: 3, sampler: Some("euler".into()), guidance: {guidance},
        ip_adapter_scale: 0.7, seed: 104729, ..Default::default() }};'''
    pid = case == "pulid-pid"
    pid_setup = '''let pid_weights = candle_gen::gen_core::PidWeights {
        checkpoint: WeightsSource::File(assets.join("pid_decoder.safetensors")),
        gemma: WeightsSource::Dir(PathBuf::from(env::var("PAIR_PID_GEMMA")?)),
    };
    spec.pid = Some(pid_weights.clone());''' if pid else ""
    return f'''use candle_gen::gen_core::{{GenerationMemory, IdentityWeights, LoadSpec, WeightsSource}};
    use candle_gen_pulid::pulid_flux::{{PulidFluxPaths, PulidFluxRequest, PulidFlux}};
    let paths = PulidFluxPaths {{ flux_base: root.clone(), pulid_weights: assets.join("pulid.safetensors"),
        eva_weights: assets.join("eva.safetensors"), face_dir: assets.join("face"),
        adapters: Vec::new() }};
    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    spec.identity = Some(IdentityWeights {{
        encoder: Some(WeightsSource::File(paths.pulid_weights.clone())),
        eva: Some(WeightsSource::File(paths.eva_weights.clone())),
        face_dir: Some(WeightsSource::Dir(paths.face_dir.clone())),
    }});
    {pid_setup}
    let req = PulidFluxRequest {{ prompt: "red kite portrait".into(), width: 256, height: 256,
        steps: 3, guidance: 3.5, id_weight: 1.0, seed: 104729,
        use_pid: {str(pid).lower()}, ..Default::default() }};'''


def candle_generate(case, arm):
    if case.startswith("control-"):
        module, compiler = "candle_gen_flux::control_provider", "compile_control_plan"
        load = ("Flux1DevControl::load_with_memory_spec(&paths, GenerationMemory::default(), &spec)?"
                if arm == "head" else "Flux1DevControl::load_with_memory(&paths, GenerationMemory::default())?")
        base = "generator.generate(&req, &source, &mut progress)?"
        planned = "generator.generate_with_injector_planned(&req, &source, None, &plan, &mut progress)?"
    elif case.startswith("ip-"):
        module, compiler = "candle_gen_flux::ip_provider", "compile_ip_plan"
        load = ("IpAdapterFlux::load_with_memory_spec(&paths, GenerationMemory::default(), &spec)?"
                if arm == "head" else "IpAdapterFlux::load_with_memory(&paths, GenerationMemory::default())?")
        base = "generator.generate(&req, &source, &mut progress)?"
        planned = "generator.generate_planned(&req, &source, &plan, &mut progress)?"
    else:
        module, compiler = "candle_gen_pulid::pulid_flux", "compile_pulid_plan"
        load = ("PulidFlux::load_with_memory_spec(&paths, GenerationMemory::default(), &spec)?"
                if arm == "head" else "PulidFlux::load_with_memory(&paths, GenerationMemory::default())?")
        if case == "pulid-pid":
            load += ".with_pid(&pid_weights)?"
        base = "generator.generate(&req, &source, &mut progress)?"
        planned = "generator.generate_planned(&req, &source, &plan, &mut progress)?"
    if arm == "head":
        prefix = f'''let plan = {module}::{compiler}(&paths, &spec, &req, &source)?;
    fs::write(dir.join("plan.json"), serde_json::to_vec(&plan)?)?;
    if env::args().nth(1).as_deref() == Some("--preflight") {{
        println!("FLUX1_CONDITIONAL_PREFLIGHT {{case}}");
        return Ok(());
    }}'''
        render = planned
    else:
        prefix = "let _ = &spec;"
        render = base
    return f'''{prefix}
    let generator = {load};
    let output = {render};'''


def cuda_proof():
    # Same bounded pool receipt as the existing base-pair probe.
    return '''let pool = candle_gen::cuda_mempool::MemPool::device_current(0)
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
        "poolUsedBytes": used, "poolUsedHighBytes": used_high,
        "poolReservedBytes": reserved, "poolReservedHighBytes": reserved_high});
    let pending = dir.join("cuda-pool.pending");
    let final_path = dir.join("cuda-pool.json");
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(&pending)?;
    use std::io::Write;
    file.write_all(&serde_json::to_vec(&proof)?)?;
    file.sync_all()?;
    if final_path.exists() { return Err("CUDA pool proof destination exists".into()); }
    fs::rename(pending, final_path)?;
    std::thread::sleep(std::time::Duration::from_millis(3500));
    drop(generator);'''


def generate_source(backend, arm, case):
    template = (HERE / "conditional_probe.rs.in").read_text(encoding="utf-8")
    mlx = backend == "mlx"
    setup = mlx_setup(case) if mlx else candle_setup(case)
    if mlx:
        setup = setup.replace("@REGISTRY@", "mlx_gen_pulid" if case.startswith("pulid-") else "mlx_gen_flux")
    return (template.replace("@ENGINE@", "mlx_gen" if mlx else "candle_gen")
            .replace("@SETUP@", setup)
            .replace("@GENERATE@", mlx_generate(arm) if mlx else candle_generate(case, arm))
            .replace("@CUDA_PROOF@", "" if mlx else cuda_proof()))


def proof_manifest(body, package):
    """Promote the existing image dev dependency for this isolated proof binary."""
    dependencies = "image = { workspace = true }\n"
    if package == "mlx-gen-pulid":
        dependencies = "serde_json = { workspace = true }\n" + dependencies
    anchor = "\n[dependencies]\n"
    image_dev = "\nimage = { workspace = true }\n"
    if (body.count(anchor) != 1 or body.count(image_dev) != 1
            or body.index(image_dev) < body.index("[dev-dependencies]")):
        raise ValueError("proof package has no unique image dev dependency: " + package)
    body = body.replace(image_dev, "\n")
    return body.replace(anchor, anchor + dependencies)


def install(root, backend, arm, tier, case):
    if tier != "tier1":
        raise ValueError("conditional FLUX.1 comparison is Tier 1 only")
    # Reuse the audited shallow-depth substitutions and native sampler taps. The base binary is
    # replaced with a route-specific probe in the owning provider package.
    base_package = overlay.install(root, backend, arm, tier)
    family = "mlx-gen" if backend == "mlx" else "candle-gen"
    base_bin = root / "crates/media" / family / base_package / "src/bin/flux1-pair.rs"
    base_bin.unlink()
    package = conditional.family(case, backend)
    manifest = f"crates/media/{family}/{package}/Cargo.toml"
    manifest_path = root / manifest
    manifest_path.write_text(proof_manifest(manifest_path.read_text(encoding="utf-8"), package),
                             encoding="utf-8")
    # Adding proof-only direct dependencies can change this package's lockfile dependency list.
    # Resolve from the committed cache without network, then the actual build stays --locked.
    subprocess.run(["cargo", "metadata", "--offline", "--format-version", "1"], cwd=root,
                   check=True, stdout=subprocess.DEVNULL)
    destination = root / "crates/media" / family / package / "src/bin/flux1-pair.rs"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(generate_source(backend, arm, case), encoding="utf-8")
    return package
