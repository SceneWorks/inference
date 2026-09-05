//! SC-15514 serial real-Metal evidence runner for the exact cached FLUX.1-dev MLX Q4 artifact.
//!
//! This test never downloads weights and runs exactly one independently selected arm per process.
//! Run the Resident arm first so later arms can compare against its persisted RGB output:
//!
//! ```text
//! FLUX1_LADDER_ROOT=/path/to/snapshots/323fd12d79f78ad444e882e8d8e871914584f2b9/q4 \
//! FLUX1_LADDER_ARM=resident cargo test -p mlx-gen-flux --release \
//!   --test integration -- shared_memory_ladder_real_weights:: --ignored --nocapture --test-threads=1
//! FLUX1_LADDER_ROOT=/path/to/snapshots/323fd12d79f78ad444e882e8d8e871914584f2b9/q4 \
//! FLUX1_LADDER_ARM=rung4 cargo test -p mlx-gen-flux --release \
//!   --test integration -- shared_memory_ladder_real_weights:: --ignored --nocapture --test-threads=1
//! ```
//!
//! Arms: `resident`, `staged`, `tile768`, `tile640`, `tile512`, `attention`, `rung4`,
//! `cancel`, and `fault`. `FLUX1_LADDER_ROOT` is required and may name either the exact revision
//! root or its `q4` child; both resolve to and bind `snapshots/<revision>/q4`. Outputs go to
//! `FLUX1_LADDER_OUTPUT_DIR` or a deterministic temporary directory.
//!
//! The provider currently has no integration-test-visible runtime window-event counter. The runner
//! independently enumerates the exact artifact header and requires all 19 joint + 38 single block
//! indices, reports the 57 expected window=1 materializations per denoise step, and relies on the
//! production deferred-load invariant that rejects any retained eager block stack. It labels runtime
//! event counts unavailable rather than presenting the arithmetic as observed instrumentation.

#![cfg(target_os = "macos")]

use mlx_gen::gen_core::{
    weightsmeta::safetensors_path_tensor_headers, GenerationMemory, MemoryPhase, MemoryStrategy,
    MemoryStrategySupport, TransformerComponent,
};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Generator, LoadPhase, LoadShape, LoadSpec, OffloadPolicy,
    Progress, Quant, WeightsSource,
};
use mlx_rs::memory::{
    clear_cache, get_active_memory, get_cache_memory, get_peak_memory, reset_peak_memory,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const REVISION: &str = "323fd12d79f78ad444e882e8d8e871914584f2b9";
const EDGE: u32 = 1024;
const SEED: u64 = 15514;
// Two is the minimum that leaves one clean post-Step(1) denoise interval and lets the timed
// cancellation raised after Step(1) interrupt the second model evaluation rather than decode.
const STEPS: u32 = 2;
const JOINT_BLOCKS: usize = 19;
const SINGLE_BLOCKS: usize = 38;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Resident,
    Staged,
    Tile(u32),
    Attention,
    Rung4,
    Cancel,
    Fault,
}

impl Arm {
    fn from_env() -> Self {
        match std::env::var("FLUX1_LADDER_ARM").as_deref() {
            Ok("resident") => Self::Resident,
            Ok("staged") => Self::Staged,
            Ok("tile768") => Self::Tile(768),
            Ok("tile640") => Self::Tile(640),
            Ok("tile512") => Self::Tile(512),
            Ok("attention") => Self::Attention,
            Ok("rung4") => Self::Rung4,
            Ok("cancel") => Self::Cancel,
            Ok("fault") => Self::Fault,
            Ok(other) => panic!(
                "unknown FLUX1_LADDER_ARM={other}; expected resident|staged|tile768|tile640|tile512|attention|rung4|cancel|fault"
            ),
            Err(_) => panic!("set FLUX1_LADDER_ARM; see test documentation"),
        }
    }

    fn label(self) -> String {
        match self {
            Self::Resident => "resident".to_owned(),
            Self::Staged => "staged".to_owned(),
            Self::Tile(edge) => format!("tile{edge}"),
            Self::Attention => "attention".to_owned(),
            Self::Rung4 => "rung4".to_owned(),
            Self::Cancel => "cancel".to_owned(),
            Self::Fault => "fault".to_owned(),
        }
    }

    fn is_terminal(self) -> bool {
        matches!(self, Self::Cancel | Self::Fault)
    }
}

fn bind_q4_snapshot(requested: &Path) -> PathBuf {
    let resolved = std::fs::canonicalize(requested).unwrap_or_else(|error| {
        panic!(
            "exact cached FLUX.1-dev MLX revision/tier is unavailable at {}: {error}",
            requested.display()
        )
    });
    let canonical = match resolved.file_name().and_then(|part| part.to_str()) {
        Some("q4") => resolved,
        Some(REVISION) => std::fs::canonicalize(resolved.join("q4")).unwrap_or_else(|error| {
            panic!(
                "exact cached FLUX.1-dev MLX Q4 child is unavailable under {}: {error}",
                resolved.display()
            )
        }),
        _ => panic!(
            "FLUX1_LADDER_ROOT must resolve to snapshots/{REVISION} or its q4 child, got {}",
            resolved.display()
        ),
    };
    assert_eq!(
        canonical.file_name().and_then(|part| part.to_str()),
        Some("q4"),
        "resolved artifact root must be the q4 tier"
    );
    assert_eq!(
        canonical
            .parent()
            .and_then(Path::file_name)
            .and_then(|part| part.to_str()),
        Some(REVISION),
        "q4 tier must belong to exact revision {REVISION}, got {}",
        canonical.display()
    );
    assert_eq!(
        canonical
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .and_then(|part| part.to_str()),
        Some("snapshots"),
        "exact tier must be a Hugging Face snapshots/<revision>/q4 directory"
    );
    canonical
}

fn snapshot() -> PathBuf {
    let requested = std::env::var_os("FLUX1_LADDER_ROOT")
        .map(PathBuf::from)
        .expect("set FLUX1_LADDER_ROOT to the exact revision directory or its q4 child");
    bind_q4_snapshot(&requested)
}

fn spec(root: &Path, arm: Arm) -> LoadSpec {
    let (policy, shape) = if arm == Arm::Resident {
        (OffloadPolicy::Resident, LoadShape::EagerMaterialization)
    } else {
        (
            OffloadPolicy::Sequential,
            LoadShape::DeferredMaterialization,
        )
    };
    LoadSpec::new(WeightsSource::Dir(root.to_owned()))
        .with_quant(Quant::Q4)
        .with_offload_policy(policy)
        .with_load_shape(shape)
}

fn validate_exact_inventory(root: &Path) -> String {
    let validation = LoadSpec::new(WeightsSource::Dir(root.to_owned()))
        .with_quant(Quant::Q4)
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    let contract = mlx_gen_flux::memory_strategy::memory_strategy_contract(
        mlx_gen_flux::FLUX1_DEV_ID,
        &validation,
    )
    .expect("exact pinned inventory/content validation");
    assert_eq!(
        contract
            .calibration
            .as_ref()
            .expect("reviewed exact-key production calibration")
            .fingerprint,
        mlx_gen_flux::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
    );
    assert_eq!(
        contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .expect("rung-4 capability")
            .support,
        MemoryStrategySupport::Implemented,
        "rung 4 is implemented only after exact pinned Q4 inventory/content admission"
    );
    assert_eq!(
        contract
            .capability(MemoryStrategy::BoundedDecode)
            .unwrap()
            .parameters
            .decode_tile_edges,
        [768, 640, 512]
    );
    assert_eq!(
        contract
            .capability(MemoryStrategy::BoundedDecode)
            .unwrap()
            .parameters
            .decode_overlaps,
        [64]
    );
    assert_eq!(
        contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap()
            .parameters
            .transformer_window_sizes,
        [1]
    );
    mlx_gen_flux::memory_strategy::verified_runner_artifact(mlx_gen_flux::FLUX1_DEV_ID, &validation)
        .expect("exact runner artifact composite")
}

fn block_inventory(root: &Path) -> (usize, usize) {
    let headers = safetensors_path_tensor_headers(root.join("transformer/model.safetensors"))
        .expect("header-read exact transformer artifact");
    let mut joint = BTreeSet::new();
    let mut single = BTreeSet::new();
    for tensor in headers {
        let mut parts = tensor.name.split('.');
        let stack = parts.next();
        let index = parts.next().and_then(|part| part.parse::<usize>().ok());
        match (stack, index) {
            (Some("transformer_blocks"), Some(index)) => {
                joint.insert(index);
            }
            (Some("single_transformer_blocks"), Some(index)) => {
                single.insert(index);
            }
            _ => {}
        }
    }
    assert_eq!(
        joint,
        (0..JOINT_BLOCKS).collect(),
        "exact artifact must expose joint blocks 0..18"
    );
    assert_eq!(
        single,
        (0..SINGLE_BLOCKS).collect(),
        "exact artifact must expose single blocks 0..37"
    );
    (joint.len(), single.len())
}

fn memory(arm: Arm) -> Option<GenerationMemory> {
    if arm == Arm::Resident {
        return None;
    }
    let mut memory = GenerationMemory {
        stage_residency: true,
        ..Default::default()
    };
    match arm {
        Arm::Resident | Arm::Staged => {}
        Arm::Tile(edge) => {
            memory.tile_vae_decode = true;
            memory.decode_tile_edge = Some(edge);
            memory.decode_overlap = Some(64);
        }
        Arm::Attention => {
            memory.tile_vae_decode = true;
            memory.decode_tile_edge = Some(512);
            memory.decode_overlap = Some(64);
            memory.chunk_attention = true;
            memory.attention_chunk_size = Some(mlx_gen_flux::memory_strategy::ATTENTION_CHUNK_SIZE);
        }
        Arm::Rung4 | Arm::Cancel | Arm::Fault => {
            memory.tile_vae_decode = true;
            memory.decode_tile_edge = Some(512);
            memory.decode_overlap = Some(64);
            memory.chunk_attention = true;
            memory.attention_chunk_size = Some(mlx_gen_flux::memory_strategy::ATTENTION_CHUNK_SIZE);
            memory.stream_transformer_blocks = true;
            memory.transformer_window_size = Some(1);
            memory.transformer_window_component = Some(TransformerComponent::Dit);
        }
    }
    Some(memory)
}

fn request(memory: Option<GenerationMemory>) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox in a snowy pine clearing at dawn, detailed photograph".to_owned(),
        width: EDGE,
        height: EDGE,
        count: 1,
        steps: Some(STEPS),
        guidance: Some(3.5),
        seed: Some(SEED),
        memory,
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Metric {
    active_bytes: usize,
    peak_bytes: usize,
    wall_ms: u128,
}

fn metric(started: Instant) -> Metric {
    Metric {
        active_bytes: get_active_memory(),
        peak_bytes: get_peak_memory(),
        wall_ms: started.elapsed().as_millis(),
    }
}

#[derive(Debug)]
struct Run {
    image: mlx_gen::Image,
    conditioning: Metric,
    first_render: Metric,
    denoise: Metric,
    decode: Metric,
    request_wall_ms: u128,
    immediate_active_bytes: usize,
    immediate_cache_bytes: usize,
}

fn output_image(output: GenerationOutput) -> mlx_gen::Image {
    match output {
        GenerationOutput::Images(mut images) if images.len() == 1 => images.remove(0),
        other => panic!("expected one image, got {other:?}"),
    }
}

fn run_success(generator: &dyn Generator, arm: Arm) -> Run {
    clear_cache();
    reset_peak_memory();
    let request_started = Instant::now();
    let mut phase_started = request_started;
    let mut conditioning = None;
    let mut first_render = None;
    let mut denoise = None;
    let mut on_progress = |progress| match progress {
        Progress::Loading(LoadPhase::TextEncoder) => {
            phase_started = Instant::now();
        }
        Progress::Loading(LoadPhase::Renderer) => {
            conditioning = Some(metric(phase_started));
            reset_peak_memory();
            phase_started = Instant::now();
        }
        Progress::Step { current: 1, .. } => {
            // Progress::Step(1) is emitted after the first denoise evaluation. Sequential therefore
            // samples renderer load + step 1 here; Resident samples conditioning + step 1 because it
            // has no Loading(Renderer) boundary. The post-reset interval is clean step 2.
            first_render = Some(metric(phase_started));
            reset_peak_memory();
            phase_started = Instant::now();
        }
        Progress::Decoding => {
            denoise = Some(metric(phase_started));
            reset_peak_memory();
            phase_started = Instant::now();
        }
        _ => {}
    };
    let output = generator
        .generate(&request(memory(arm)), &mut on_progress)
        .expect("fixed FLUX.1-dev generation");
    let decode = metric(phase_started);
    let image = output_image(output);
    Run {
        image,
        conditioning: conditioning.unwrap_or_default(),
        first_render: first_render.expect("first denoise step boundary"),
        denoise: denoise.expect("denoise/decode phase boundary"),
        decode,
        request_wall_ms: request_started.elapsed().as_millis(),
        immediate_active_bytes: get_active_memory(),
        immediate_cache_bytes: get_cache_memory(),
    }
}

fn output_dir() -> PathBuf {
    std::env::var_os("FLUX1_LADDER_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("inference-flux1-ladder-evidence-v1"))
}

fn persist(arm: Arm, image: &mlx_gen::Image) -> PathBuf {
    let root = output_dir();
    std::fs::create_dir_all(&root).expect("create evidence output directory");
    let path = root.join(format!("{}.rgb", arm.label()));
    std::fs::write(&path, &image.pixels).expect("write evidence RGB");
    path
}

#[derive(Debug)]
struct Drift {
    differing_bytes: usize,
    max_delta: u8,
    mean_delta: f64,
    rmse: f64,
    correlation: f64,
}

fn drift(reference: &[u8], observed: &[u8]) -> Drift {
    assert_eq!(reference.len(), observed.len(), "RGB byte count");
    let mut differing = 0_usize;
    let mut max = 0_u8;
    let mut absolute = 0_f64;
    let mut squared = 0_f64;
    let mut dot = 0_f64;
    let mut aa = 0_f64;
    let mut bb = 0_f64;
    for (&a, &b) in reference.iter().zip(observed) {
        let delta = a.abs_diff(b);
        differing += usize::from(delta != 0);
        max = max.max(delta);
        absolute += f64::from(delta);
        squared += f64::from(delta).powi(2);
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }
    Drift {
        differing_bytes: differing,
        max_delta: max,
        mean_delta: absolute / reference.len() as f64,
        rmse: (squared / reference.len() as f64).sqrt(),
        correlation: dot / (aa.sqrt() * bb.sqrt()).max(1e-12),
    }
}

fn report_success(arm: Arm, artifact_sha256: &str, load: Metric, run: &Run) {
    let sha = format!("{:x}", Sha256::digest(&run.image.pixels));
    let path = persist(arm, &run.image);
    let request_peak = run
        .conditioning
        .peak_bytes
        .max(run.first_render.peak_bytes)
        .max(run.denoise.peak_bytes)
        .max(run.decode.peak_bytes);
    let (conditioning_scope, first_render_scope) = if arm == Arm::Resident {
        ("unavailable", "conditioning_plus_first_step")
    } else {
        ("text_conditioning", "renderer_load_plus_first_step")
    };
    println!(
        "ARM provider={} revision={} artifact_sha256={} tier=q4 arm={} size={} batch=1 seed={} steps={} load_active_bytes={} load_peak_bytes={} load_wall_ms={} conditioning_metric_scope={} conditioning_active_bytes={} conditioning_peak_bytes={} conditioning_wall_ms={} first_render_metric_scope={} first_render_active_bytes={} first_render_peak_bytes={} first_render_wall_ms={} denoise_metric_scope=second_step denoise_active_bytes={} denoise_peak_bytes={} denoise_wall_ms={} decode_active_bytes={} decode_peak_bytes={} decode_wall_ms={} request_peak_bytes={} request_wall_ms={} immediate_active_bytes={} immediate_cache_bytes={} output_sha256={} output_path={}",
        mlx_gen_flux::FLUX1_DEV_ID,
        REVISION,
        artifact_sha256,
        arm.label(),
        EDGE,
        SEED,
        STEPS,
        load.active_bytes,
        load.peak_bytes,
        load.wall_ms,
        conditioning_scope,
        run.conditioning.active_bytes,
        run.conditioning.peak_bytes,
        run.conditioning.wall_ms,
        first_render_scope,
        run.first_render.active_bytes,
        run.first_render.peak_bytes,
        run.first_render.wall_ms,
        run.denoise.active_bytes,
        run.denoise.peak_bytes,
        run.denoise.wall_ms,
        run.decode.active_bytes,
        run.decode.peak_bytes,
        run.decode.wall_ms,
        request_peak,
        run.request_wall_ms,
        run.immediate_active_bytes,
        run.immediate_cache_bytes,
        sha,
        path.display(),
    );

    if arm != Arm::Resident {
        let reference_path = output_dir().join("resident.rgb");
        let reference = std::fs::read(&reference_path).unwrap_or_else(|error| {
            panic!(
                "run FLUX1_LADDER_ARM=resident first; cannot read {}: {error}",
                reference_path.display()
            )
        });
        let drift = drift(&reference, &run.image.pixels);
        println!(
            "PARITY arm={} reference_sha256={:x} output_sha256={} differing_bytes={} max_delta={} mean_delta={:.8} rmse={:.8} correlation={:.10}",
            arm.label(),
            Sha256::digest(&reference),
            sha,
            drift.differing_bytes,
            drift.max_delta,
            drift.mean_delta,
            drift.rmse,
            drift.correlation,
        );
    }
}

fn load_measured(spec: &LoadSpec) -> (Box<dyn Generator>, Metric) {
    clear_cache();
    reset_peak_memory();
    let started = Instant::now();
    let generator = mlx_gen_flux::provider_registry()
        .expect("FLUX registry")
        .load(mlx_gen_flux::FLUX1_DEV_ID, spec)
        .expect("load exact cached FLUX.1-dev Q4");
    (generator, metric(started))
}

fn run_cancel(generator: &dyn Generator) -> (usize, usize, Run) {
    clear_cache();
    reset_peak_memory();
    let request = request(memory(Arm::Cancel));
    let cancel = request.cancel.clone();
    let delay_ms = std::env::var("FLUX1_LADDER_CANCEL_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(250_u64);
    let mut timer = None;
    let error = generator
        .generate(&request, &mut |progress| {
            if matches!(progress, Progress::Step { current: 1, .. }) && timer.is_none() {
                let cancel = cancel.clone();
                timer = Some(std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(delay_ms));
                    cancel.cancel();
                }));
            }
        })
        .expect_err("mid-denoise timed cancellation must interrupt rung 4");
    timer
        .expect("denoise step boundary before cancellation")
        .join()
        .unwrap();
    assert!(
        matches!(error, mlx_gen::gen_core::Error::Canceled),
        "{error:?}"
    );
    clear_cache();
    let floor = (get_active_memory(), get_cache_memory());
    let recovery = run_success(generator, Arm::Rung4);
    (floor.0, floor.1, recovery)
}

fn run_fault(generator: &dyn Generator) -> (usize, usize, Run) {
    let mut fault = memory(Arm::Fault).expect("rung-4 memory");
    fault.authorize_calibration_fault(MemoryPhase::Denoise);
    let error = generator
        .generate(&request(Some(fault)), &mut |_| {})
        .expect_err("authorized mid-denoise block fault must surface");
    assert!(
        error
            .to_string()
            .contains("calibration fault after joint block materialization"),
        "{error}"
    );
    clear_cache();
    let floor = (get_active_memory(), get_cache_memory());
    let recovery = run_success(generator, Arm::Rung4);
    (floor.0, floor.1, recovery)
}

#[test]
fn snapshot_resolver_accepts_revision_or_q4_and_returns_the_bound_tier() {
    let root_tmp = tempfile::tempdir().unwrap();
    let root = root_tmp.path().to_path_buf();
    let revision = root.join("snapshots").join(REVISION);
    let q4 = revision.join("q4");
    std::fs::create_dir_all(&q4).unwrap();
    let canonical_q4 = std::fs::canonicalize(&q4).unwrap();
    assert_eq!(bind_q4_snapshot(&revision), canonical_q4);
    assert_eq!(bind_q4_snapshot(&q4), canonical_q4);
}

#[test]
#[ignore = "needs exact cached SceneWorks/flux1-dev-mlx Q4 revision and exclusive Apple/Metal access"]
fn exact_q4_shared_memory_ladder_arm() {
    let arm = Arm::from_env();
    let root = snapshot();
    let artifact_sha256 = validate_exact_inventory(&root);
    let (joint, single) = block_inventory(&root);
    println!(
        "WINDOW_EVIDENCE joint_blocks={} single_blocks={} window_size=1 expected_joint_windows_per_step={} expected_single_windows_per_step={} expected_total_windows_per_step={} expected_total_windows_request={} runtime_window_event_instrumentation=unavailable no_eager_full_stack_guard=production_deferred_load_assertion",
        joint,
        single,
        joint,
        single,
        joint + single,
        (joint + single) * STEPS as usize,
    );

    let arm_spec = spec(&root, arm);
    let contract = mlx_gen_flux::memory_strategy::memory_strategy_contract(
        mlx_gen_flux::FLUX1_DEV_ID,
        &arm_spec,
    )
    .expect("arm contract");
    if arm == Arm::Resident {
        // sc-22726: the resident baseline is the same (dev, q4) artifact and carries the same
        // identity under its own `EagerMaterialization` load shape; only the optimized arms are
        // additionally bound to the exact composite runner key below.
        let identity = contract
            .calibration
            .as_ref()
            .expect("Resident+Eager publishes the (dev, q4) identity");
        assert_eq!(
            identity.fingerprint,
            mlx_gen_flux::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_eq!(
            identity.load_shape,
            mlx_gen::LoadShape::EagerMaterialization
        );
    } else {
        mlx_gen_flux::memory_strategy::validate_runner_gate(
            mlx_gen_flux::FLUX1_DEV_ID,
            &artifact_sha256,
            &contract,
        )
        .expect("optimized arm remains bound to the exact calibrated key");
    }
    let (generator, load) = load_measured(&arm_spec);

    if arm.is_terminal() {
        let (floor_active, floor_cache, recovery) = match arm {
            Arm::Cancel => run_cancel(generator.as_ref()),
            Arm::Fault => run_fault(generator.as_ref()),
            _ => unreachable!(),
        };
        report_success(arm, &artifact_sha256, load, &recovery);
        clear_cache();
        let recovery_floor = (get_active_memory(), get_cache_memory());
        assert_eq!(
            (floor_active, floor_cache),
            recovery_floor,
            "terminal path did not return to the clean successful rung-4 allocator floor"
        );
        println!(
            "TERMINAL_RESULT arm={} post_clear_active_bytes={} post_clear_cache_bytes={} recovery_post_clear_active_bytes={} recovery_post_clear_cache_bytes={} recovery_output_sha256={:x}",
            arm.label(),
            floor_active,
            floor_cache,
            recovery_floor.0,
            recovery_floor.1,
            Sha256::digest(&recovery.image.pixels),
        );
    } else {
        let run = run_success(generator.as_ref(), arm);
        report_success(arm, &artifact_sha256, load, &run);
    }

    drop(generator);
    clear_cache();
    println!(
        "RESULT status=pass provider={} revision={} artifact_sha256={} tier=q4 arm={} size={} batch=1 seed={} steps={} final_active_bytes={} final_cache_bytes={} calibration_published=true calibration_fingerprint={}",
        mlx_gen_flux::FLUX1_DEV_ID,
        REVISION,
        artifact_sha256,
        arm.label(),
        EDGE,
        SEED,
        STEPS,
        get_active_memory(),
        get_cache_memory(),
        mlx_gen_flux::memory_strategy::MEMORY_CALIBRATION_FINGERPRINT,
    );
}
