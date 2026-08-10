#[cfg(not(target_os = "macos"))]
compile_error!("mlx-perf-bench is supported only on macOS with an MLX Metal device");

use mlx_rs::memory::{clear_cache, get_cache_memory, get_peak_memory, reset_peak_memory};
use runtime_macos::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, WeightsSource,
};
use runtime_macos::media::diagnostics::{
    self, CacheDisposition, CompileDisposition, DiagnosticCounter, DiagnosticReport,
    ToggleDisposition,
};
use runtime_macos::perf_bench::{
    build_summary, ArtifactManifest, ArtifactReceipt, BenchmarkFamily, BenchmarkMatrix,
    BenchmarkSummary, DiagnosticRecord, EnvironmentRecord, MeasurementRecord, ModelTier,
    OutputFingerprint, PhaseMetrics, PhaseSet, RunRecord, VariantPlan, RUN_SCHEMA_VERSION,
};
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_MATRIX: &str = include_str!("../../benchmarks/mlx-perf-matrix-v1.json");

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mlx-perf-bench: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("run") => run_matrix(&args[1..]),
        Some("child") => run_child(&args[1..]),
        Some("validate") => validate_inputs(&args[1..]),
        _ => Err(usage()),
    }
}

fn usage() -> String {
    "usage:\n  mlx-perf-bench validate [--matrix PATH] --artifacts PATH\n  \
     mlx-perf-bench run [--matrix PATH] --artifacts PATH --output-dir PATH \
     [--variants baseline,id,...]\n\nThe output directory must be absolute, outside the \
     checkout, and empty. `run` defaults to the full baseline + independent toggles + all-on \
     matrix; use `--variants baseline` only for a pre-optimization baseline campaign."
        .to_owned()
}

#[derive(Default)]
struct Options {
    matrix: Option<PathBuf>,
    artifacts: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    variants: Option<Vec<String>>,
    case_id: Option<String>,
    variant_id: Option<String>,
    output_file: Option<PathBuf>,
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        index += 1;
        let value = args
            .get(index)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        index += 1;
        match flag {
            "--matrix" => options.matrix = Some(PathBuf::from(value)),
            "--artifacts" => options.artifacts = Some(PathBuf::from(value)),
            "--output-dir" => options.output_dir = Some(PathBuf::from(value)),
            "--variants" => {
                options.variants = Some(
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_owned)
                        .collect(),
                )
            }
            "--case" => options.case_id = Some(value.clone()),
            "--variant" => options.variant_id = Some(value.clone()),
            "--output-file" => options.output_file = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {flag:?}\n{}", usage())),
        }
    }
    Ok(options)
}

fn load_json<T: DeserializeOwned>(path: &Path, label: &str) -> Result<T, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("read {label} {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {label} {}: {error}", path.display()))
}

fn load_matrix(path: Option<&Path>) -> Result<BenchmarkMatrix, String> {
    let matrix: BenchmarkMatrix = match path {
        Some(path) => load_json(path, "matrix")?,
        None => serde_json::from_str(DEFAULT_MATRIX)
            .map_err(|error| format!("parse embedded matrix: {error}"))?,
    };
    matrix
        .validate()
        .map_err(|error| format!("invalid benchmark matrix: {error}"))?;
    Ok(matrix)
}

fn load_inputs(options: &Options) -> Result<(BenchmarkMatrix, ArtifactManifest), String> {
    let matrix = load_matrix(options.matrix.as_deref())?;
    let artifacts_path = options
        .artifacts
        .as_deref()
        .ok_or_else(|| "--artifacts is required".to_owned())?;
    let artifacts: ArtifactManifest = load_json(artifacts_path, "artifact manifest")?;
    artifacts
        .validate_against(&matrix)
        .map_err(|error| format!("invalid artifact manifest: {error}"))?;
    Ok((matrix, artifacts))
}

fn validate_inputs(args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    let (matrix, artifacts) = load_inputs(&options)?;
    println!(
        "validated {}: {} cases, {} variants, {} artifacts",
        matrix.benchmark_id,
        matrix.cases.len(),
        matrix.variants.len(),
        artifacts.artifacts.len()
    );
    Ok(())
}

fn repository_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or_else(|| "cannot resolve repository root from CARGO_MANIFEST_DIR".to_owned())?
        .canonicalize()
        .map_err(|error| format!("canonicalize repository root: {error}"))
}

fn prepare_output_dir(path: &Path, require_empty: bool) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("output path must be absolute: {}", path.display()));
    }
    fs::create_dir_all(path)
        .map_err(|error| format!("create output directory {}: {error}", path.display()))?;
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("canonicalize output directory {}: {error}", path.display()))?;
    let root = repository_root()?;
    if canonical.starts_with(&root) {
        return Err(format!(
            "results must be outside the repository ({} is under {})",
            canonical.display(),
            root.display()
        ));
    }
    if require_empty
        && fs::read_dir(&canonical)
            .map_err(|error| format!("read output directory {}: {error}", canonical.display()))?
            .next()
            .is_some()
    {
        return Err(format!(
            "output directory must be empty to prevent mixed campaigns: {}",
            canonical.display()
        ));
    }
    Ok(canonical)
}

fn selected_variants(
    matrix: &BenchmarkMatrix,
    requested: Option<&[String]>,
) -> Result<Vec<String>, String> {
    let selected = requested.map_or_else(
        || {
            matrix
                .variants
                .iter()
                .map(|variant| variant.id.clone())
                .collect()
        },
        <[String]>::to_vec,
    );
    let unique: BTreeSet<_> = selected.iter().map(String::as_str).collect();
    if selected.is_empty() || unique.len() != selected.len() {
        return Err("--variants must name a non-empty unique list".to_owned());
    }
    if !unique.contains("baseline") {
        return Err("--variants must include baseline".to_owned());
    }
    for id in &selected {
        if matrix.variant(id).is_none() {
            return Err(format!("unknown variant {id:?}"));
        }
    }
    Ok(selected)
}

fn run_matrix(args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    let (matrix, _artifacts) = load_inputs(&options)?;
    let output_dir = prepare_output_dir(
        options
            .output_dir
            .as_deref()
            .ok_or_else(|| "--output-dir is required".to_owned())?,
        true,
    )?;
    let selected = selected_variants(&matrix, options.variants.as_deref())?;
    ensure_clean_checkout()?;
    let executable =
        env::current_exe().map_err(|error| format!("resolve current binary: {error}"))?;
    let artifacts = options
        .artifacts
        .as_deref()
        .expect("load_inputs required --artifacts")
        .canonicalize()
        .map_err(|error| format!("canonicalize artifact manifest: {error}"))?;

    for case in &matrix.cases {
        for variant in &matrix.variants {
            if !selected.iter().any(|id| id == &variant.id) {
                continue;
            }
            let output_file = output_dir.join(format!("{}__{}.json", case.id, variant.id));
            let mut command = Command::new(&executable);
            command
                .arg("child")
                .arg("--artifacts")
                .arg(&artifacts)
                .arg("--case")
                .arg(&case.id)
                .arg("--variant")
                .arg(&variant.id)
                .arg("--output-file")
                .arg(&output_file)
                .stdin(Stdio::null())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit());
            if let Some(matrix_path) = options.matrix.as_deref() {
                command.arg("--matrix").arg(matrix_path);
            }
            println!("run {} / {}", case.id, variant.id);
            let status = command
                .status()
                .map_err(|error| format!("start child benchmark: {error}"))?;
            if !status.success() {
                return Err(format!(
                    "child failed for {} / {}; no summary was published",
                    case.id, variant.id
                ));
            }
        }
    }

    let mut records = Vec::new();
    for case in &matrix.cases {
        for variant in &selected {
            records.push(load_json(
                &output_dir.join(format!("{}__{}.json", case.id, variant)),
                "run record",
            )?);
        }
    }
    let summary = build_summary(&matrix, &records, &selected)
        .map_err(|error| format!("refuse incomplete comparison set: {error}"))?;
    write_json_atomic(&output_dir.join("summary.json"), &summary)?;
    print_summary(&summary);
    Ok(())
}

fn run_child(args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    let (matrix, artifacts) = load_inputs(&options)?;
    let case = matrix
        .case(
            options
                .case_id
                .as_deref()
                .ok_or_else(|| "child requires --case".to_owned())?,
        )
        .ok_or_else(|| "child case is not in the matrix".to_owned())?;
    let variant = matrix
        .variant(
            options
                .variant_id
                .as_deref()
                .ok_or_else(|| "child requires --variant".to_owned())?,
        )
        .ok_or_else(|| "child variant is not in the matrix".to_owned())?;
    let output_file = options
        .output_file
        .as_deref()
        .ok_or_else(|| "child requires --output-file".to_owned())?;
    let output_dir = output_file
        .parent()
        .ok_or_else(|| "output file must have a parent directory".to_owned())?;
    let canonical_dir = prepare_output_dir(output_dir, false)?;
    let file_name = output_file
        .file_name()
        .ok_or_else(|| "output file must have a file name".to_owned())?;
    let output_file = canonical_dir.join(file_name);
    if output_file.exists() {
        return Err(format!("refuse to overwrite {}", output_file.display()));
    }
    ensure_clean_checkout()?;

    let artifact = artifacts
        .artifact(&case.artifact_key)
        .expect("validated artifact manifest covers every case");
    let artifact_path = artifact
        .path
        .canonicalize()
        .map_err(|error| format!("canonicalize artifact {}: {error}", artifact.path.display()))?;
    let environment = environment_record()?;
    let started_at_unix_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock predates Unix epoch: {error}"))?
        .as_millis();

    clear_cache();
    reset_peak_memory();
    let load_start = Instant::now();
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build macOS catalog: {error}"))?;
    let spec = load_spec(case.tier, &artifact_path);
    let generator = catalog
        .media()
        .load(&case.provider, &spec)
        .map_err(|error| format!("load {}: {error}", case.provider))?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let load_active_peak_bytes = get_peak_memory() as u64;
    let load_cache_bytes_after_load = get_cache_memory() as u64;
    if !(load_seconds.is_finite() && load_seconds > 0.0) {
        return Err("cold model load did not produce credible timing evidence".to_owned());
    }

    for warmup in 0..matrix.warmup_runs {
        let request_id = format!("{}/{}/warmup-{warmup}", case.id, variant.id);
        let _ = measure_request(&*generator, case, variant, &request_id, warmup, true)?;
    }
    let mut measurements = Vec::new();
    for repetition in 0..matrix.measured_runs {
        let request_id = format!("{}/{}/measured-{repetition}", case.id, variant.id);
        measurements.push(measure_request(
            &*generator,
            case,
            variant,
            &request_id,
            repetition,
            false,
        )?);
    }

    let record = RunRecord {
        schema_version: RUN_SCHEMA_VERSION.to_owned(),
        benchmark_id: matrix.benchmark_id.clone(),
        case_id: case.id.clone(),
        family: case.family,
        provider: case.provider.clone(),
        artifact: ArtifactReceipt {
            key: artifact.key.clone(),
            repository: artifact.repository.clone(),
            resolved_revision: artifact.resolved_revision.clone(),
            tier: case.tier,
            canonical_path: artifact_path,
        },
        variant: variant.clone(),
        environment,
        started_at_unix_millis,
        load_seconds,
        load_active_peak_bytes,
        load_cache_bytes_after_load,
        warmup_runs_completed: matrix.warmup_runs,
        measurements,
    };
    record
        .validate_against(&matrix)
        .map_err(|error| format!("refuse false-green run record: {error}"))?;
    write_json_atomic(&output_file, &record)?;
    println!("wrote {}", output_file.display());
    Ok(())
}

fn load_spec(tier: ModelTier, path: &Path) -> LoadSpec {
    let spec = LoadSpec::new(WeightsSource::Dir(path.to_path_buf()));
    match tier {
        ModelTier::Bf16 => spec,
        ModelTier::Q4 => spec.with_quant(Quant::Q4),
        ModelTier::Q8 => spec.with_quant(Quant::Q8),
    }
}

fn generation_request(case: &runtime_macos::perf_bench::WorkloadCase) -> GenerationRequest {
    GenerationRequest {
        prompt: case.prompt.clone(),
        width: case.width,
        height: case.height,
        count: 1,
        seed: Some(case.seed),
        steps: Some(case.steps),
        frames: (case.family == BenchmarkFamily::WanVideo).then_some(case.frames),
        ..Default::default()
    }
}

fn measure_request(
    generator: &dyn runtime_macos::gen_core::Generator,
    case: &runtime_macos::perf_bench::WorkloadCase,
    variant: &VariantPlan,
    request_id: &str,
    repetition: u32,
    warmup: bool,
) -> Result<MeasurementRecord, String> {
    let requested: Vec<_> = variant
        .toggles
        .iter()
        .map(|toggle| toggle.as_str())
        .collect();
    let scope =
        diagnostics::begin_request_with_toggles(request_id, case.family.as_str(), &requested)
            .map_err(|error| error.to_string())?;
    let request = generation_request(case);
    generator
        .validate(&request)
        .map_err(|error| format!("validate {}: {error}", case.id))?;

    clear_cache();
    reset_peak_memory();
    let start = Instant::now();
    let mut phases = PhaseRecorder::new(start, case.steps);
    let output = generator
        .generate(&request, &mut |progress| phases.observe(progress))
        .map_err(|error| format!("generate {} / {}: {error}", case.id, variant.id))?;
    let total_seconds = start.elapsed().as_secs_f64();
    let phase_set = phases.finish()?;
    let output = fingerprint_output(case, &output)?;
    let report = scope.finish();
    let diagnostics = diagnostic_records(report);
    let steady_steps = phases_step_intervals(case.steps);
    let denoise_steps_per_second = steady_steps as f64 / phase_set.denoise.seconds;
    let measurement = MeasurementRecord {
        repetition,
        total_seconds,
        denoise_steps_per_second,
        step_events: phases.step_events,
        saw_decode: phases.saw_decode,
        phases: phase_set,
        output,
        diagnostics,
    };
    if !warmup {
        // Validation of applied receipts happens again at the complete-record boundary. Fail here as
        // well so an unavailable toggle never burns the rest of a multi-hour matrix.
        require_toggle_receipts(variant, &measurement)?;
    }
    Ok(measurement)
}

fn phases_step_intervals(steps: u32) -> u32 {
    // Progress::Step is emitted after each step. The first event is the only current
    // encode→denoise boundary, so steady denoise throughput spans the remaining intervals.
    steps.saturating_sub(1).max(1)
}

fn require_toggle_receipts(
    variant: &VariantPlan,
    measurement: &MeasurementRecord,
) -> Result<(), String> {
    for toggle in &variant.toggles {
        if !measurement.diagnostics.iter().any(|diagnostic| {
            diagnostic.domain == "toggle"
                && diagnostic.site == toggle.as_str()
                && diagnostic.outcome == "applied"
                && diagnostic.count > 0
        }) {
            return Err(format!(
                "toggle {} was requested but the provider emitted no applied receipt; refusing a \
                 false comparison (the owning optimization story must wire and acknowledge it)",
                toggle.as_str()
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Encode,
    Denoise,
    Decode,
}

#[derive(Clone, Copy)]
struct OpenPhase {
    phase: Phase,
    started: Instant,
    active_peak: u64,
    cache_peak: u64,
    cache_at_boundary: u64,
    samples: u32,
}

impl OpenPhase {
    fn new(phase: Phase, started: Instant) -> Self {
        Self {
            phase,
            started,
            active_peak: 0,
            cache_peak: 0,
            cache_at_boundary: 0,
            samples: 0,
        }
    }

    fn sample(&mut self) {
        self.active_peak = self.active_peak.max(get_peak_memory() as u64);
        let cache = get_cache_memory() as u64;
        self.cache_peak = self.cache_peak.max(cache);
        self.cache_at_boundary = cache;
        self.samples += 1;
    }

    fn finish(mut self, at: Instant) -> PhaseMetrics {
        self.sample();
        PhaseMetrics {
            seconds: at.duration_since(self.started).as_secs_f64(),
            active_peak_bytes: self.active_peak,
            cache_peak_bytes: self.cache_peak,
            cache_bytes_at_boundary: self.cache_at_boundary,
            samples: self.samples,
        }
    }
}

struct PhaseRecorder {
    current: Option<OpenPhase>,
    encode: Option<PhaseMetrics>,
    denoise: Option<PhaseMetrics>,
    decode: Option<PhaseMetrics>,
    expected_steps: u32,
    step_events: u32,
    saw_decode: bool,
    error: Option<String>,
}

impl PhaseRecorder {
    fn new(started: Instant, expected_steps: u32) -> Self {
        Self {
            current: Some(OpenPhase::new(Phase::Encode, started)),
            encode: None,
            denoise: None,
            decode: None,
            expected_steps,
            step_events: 0,
            saw_decode: false,
            error: None,
        }
    }

    fn transition(&mut self, expected: Phase, next: Phase, now: Instant) {
        let Some(mut current) = self.current.take() else {
            self.error
                .get_or_insert_with(|| "phase recorder is closed".to_owned());
            return;
        };
        if current.phase != expected {
            self.error.get_or_insert_with(|| {
                format!(
                    "invalid progress phase transition: expected {expected:?}, got {:?}",
                    current.phase
                )
            });
            self.current = Some(current);
            return;
        }
        current.sample();
        let metrics = current.finish(now);
        match expected {
            Phase::Encode => self.encode = Some(metrics),
            Phase::Denoise => self.denoise = Some(metrics),
            Phase::Decode => self.decode = Some(metrics),
        }
        reset_peak_memory();
        self.current = Some(OpenPhase::new(next, now));
    }

    fn observe(&mut self, progress: Progress) {
        if let Some(current) = self.current.as_mut() {
            current.sample();
        }
        match progress {
            Progress::Step { current, total } => {
                if current == 0 || current > total || total != self.expected_steps {
                    self.error.get_or_insert_with(|| {
                        format!(
                            "invalid Step progress current={current} total={total}; expected total {}",
                            self.expected_steps
                        )
                    });
                }
                self.step_events += 1;
                if self.step_events == 1 {
                    self.transition(Phase::Encode, Phase::Denoise, Instant::now());
                }
            }
            Progress::Decoding => {
                if self.saw_decode {
                    self.error
                        .get_or_insert_with(|| "duplicate Decoding progress event".to_owned());
                    return;
                }
                self.saw_decode = true;
                self.transition(Phase::Denoise, Phase::Decode, Instant::now());
            }
            Progress::Loading(_) => {}
        }
    }

    fn finish(&mut self) -> Result<PhaseSet, String> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        if self.step_events != self.expected_steps {
            return Err(format!(
                "observed {} Step events, expected {}; a zero/partial run is not benchmark evidence",
                self.step_events, self.expected_steps
            ));
        }
        if !self.saw_decode {
            return Err("generation emitted no Decoding event".to_owned());
        }
        let current = self
            .current
            .take()
            .ok_or_else(|| "phase recorder closed before output".to_owned())?;
        if current.phase != Phase::Decode {
            return Err("generation completed outside the decode phase".to_owned());
        }
        self.decode = Some(current.finish(Instant::now()));
        Ok(PhaseSet {
            encode: self
                .encode
                .take()
                .ok_or_else(|| "missing encode phase".to_owned())?,
            denoise: self
                .denoise
                .take()
                .ok_or_else(|| "missing denoise phase".to_owned())?,
            decode: self
                .decode
                .take()
                .ok_or_else(|| "missing decode phase".to_owned())?,
        })
    }
}

fn fingerprint_output(
    case: &runtime_macos::perf_bench::WorkloadCase,
    output: &GenerationOutput,
) -> Result<OutputFingerprint, String> {
    let mut hash = Sha256::new();
    let mut payload_bytes = 0u64;
    let (kind, items) = match output {
        GenerationOutput::Images(images) => {
            if case.family == BenchmarkFamily::WanVideo || images.len() != 1 {
                return Err(format!(
                    "expected one image for {}, got {}",
                    case.id,
                    images.len()
                ));
            }
            for image in images {
                hash_image(case, image, &mut hash, &mut payload_bytes)?;
            }
            ("images", images.len())
        }
        GenerationOutput::Video { frames, fps, audio } => {
            if case.family != BenchmarkFamily::WanVideo || frames.len() != case.frames as usize {
                return Err(format!(
                    "expected {} Wan frames for {}, got {}",
                    case.frames,
                    case.id,
                    frames.len()
                ));
            }
            hash.update(fps.to_le_bytes());
            for frame in frames {
                hash_image(case, frame, &mut hash, &mut payload_bytes)?;
            }
            if let Some(audio) = audio {
                hash.update(audio.sample_rate.to_le_bytes());
                hash.update(audio.channels.to_le_bytes());
                for sample in &audio.samples {
                    hash.update(sample.to_bits().to_le_bytes());
                }
                payload_bytes = payload_bytes.saturating_add(
                    u64::try_from(audio.samples.len())
                        .unwrap_or(u64::MAX)
                        .saturating_mul(4),
                );
            }
            ("video", frames.len())
        }
        GenerationOutput::Audio(_) => {
            return Err("image/video P6 matrix unexpectedly produced audio-only output".to_owned())
        }
    };
    if payload_bytes == 0 || items == 0 {
        return Err("generation returned an empty output".to_owned());
    }
    Ok(OutputFingerprint {
        kind: kind.to_owned(),
        items: u32::try_from(items).map_err(|_| "output item count exceeds u32".to_owned())?,
        width: case.width,
        height: case.height,
        payload_bytes,
        sha256: format!("{:x}", hash.finalize()),
    })
}

fn hash_image(
    case: &runtime_macos::perf_bench::WorkloadCase,
    image: &runtime_macos::gen_core::Image,
    hash: &mut Sha256,
    payload_bytes: &mut u64,
) -> Result<(), String> {
    if image.width != case.width || image.height != case.height {
        return Err(format!(
            "{} output geometry is {}x{}, expected {}x{}",
            case.id, image.width, image.height, case.width, case.height
        ));
    }
    let expected = u64::from(image.width)
        .checked_mul(u64::from(image.height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "output geometry overflows byte count".to_owned())?;
    if image.pixels.len() as u64 != expected {
        return Err(format!(
            "{} output has {} RGB bytes, expected {expected}",
            case.id,
            image.pixels.len()
        ));
    }
    hash.update(image.width.to_le_bytes());
    hash.update(image.height.to_le_bytes());
    hash.update(&image.pixels);
    *payload_bytes = payload_bytes.saturating_add(expected);
    Ok(())
}

fn diagnostic_records(report: DiagnosticReport) -> Vec<DiagnosticRecord> {
    report
        .counters
        .into_iter()
        .map(|counter| match counter {
            DiagnosticCounter::Compile {
                site,
                disposition,
                count,
            } => DiagnosticRecord {
                domain: "compile".to_owned(),
                site: site.to_owned(),
                outcome: match disposition {
                    CompileDisposition::OneShot => "one_shot",
                    CompileDisposition::RetainedMiss => "retained_miss",
                    CompileDisposition::RetainedHit => "retained_hit",
                }
                .to_owned(),
                count,
                reason: None,
            },
            DiagnosticCounter::Cache {
                site,
                disposition,
                count,
            } => DiagnosticRecord {
                domain: "cache".to_owned(),
                site: site.to_owned(),
                outcome: match disposition {
                    CacheDisposition::Hit => "hit",
                    CacheDisposition::Miss => "miss",
                    CacheDisposition::Bypass => "bypass",
                }
                .to_owned(),
                count,
                reason: None,
            },
            DiagnosticCounter::Fallback {
                site,
                reason,
                count,
            } => DiagnosticRecord {
                domain: "fallback".to_owned(),
                site: site.to_owned(),
                outcome: "fallback".to_owned(),
                count,
                reason: Some(reason.to_owned()),
            },
            DiagnosticCounter::Toggle {
                toggle,
                disposition,
                count,
            } => DiagnosticRecord {
                domain: "toggle".to_owned(),
                site: toggle.to_owned(),
                outcome: match disposition {
                    ToggleDisposition::Applied => "applied",
                    ToggleDisposition::Fallback => "fallback",
                    ToggleDisposition::Unavailable => "unavailable",
                }
                .to_owned(),
                count,
                reason: None,
            },
        })
        .collect()
}

fn command_output(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("start {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn ensure_clean_checkout() -> Result<(), String> {
    let root = repository_root()?;
    let status = Command::new("git")
        .arg("-C")
        .arg(&root)
        .args(["status", "--porcelain", "--untracked-files=all"])
        .output()
        .map_err(|error| format!("inspect checkout status: {error}"))?;
    if !status.status.success() {
        return Err("git status failed while binding benchmark provenance".to_owned());
    }
    if !status.stdout.is_empty() {
        return Err("benchmark evidence requires a clean inference checkout".to_owned());
    }
    Ok(())
}

fn environment_record() -> Result<EnvironmentRecord, String> {
    let root = repository_root()?;
    let root_text = root
        .to_str()
        .ok_or_else(|| "repository root is not UTF-8".to_owned())?;
    let inference_revision = command_output("git", &["-C", root_text, "rev-parse", "HEAD"])?;
    let mlx_revision = mlx_revision(&root.join("Cargo.lock"))?;
    Ok(EnvironmentRecord {
        inference_revision,
        mlx_revision,
        rustc_version: command_output("rustc", &["--version", "--verbose"])?.replace('\n', "; "),
        os_version: format!(
            "{} ({})",
            command_output("/usr/bin/sw_vers", &["-productVersion"])?,
            command_output("/usr/bin/sw_vers", &["-buildVersion"])?
        ),
        hardware_model: hardware_model()?,
        metal_device: metal_device()?,
    })
}

fn hardware_model() -> Result<String, String> {
    let raw = command_output(
        "/usr/sbin/system_profiler",
        &["SPHardwareDataType", "-json"],
    )?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).map_err(|error| format!("parse hardware profile: {error}"))?;
    let hardware = value
        .get("SPHardwareDataType")
        .and_then(serde_json::Value::as_array)
        .and_then(|entries| entries.first())
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "system_profiler did not report hardware identity".to_owned())?;
    let field = |name| {
        hardware
            .get(name)
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
    };
    let machine = field("machine_name").unwrap_or("Mac");
    let model = field("machine_model").unwrap_or("unknown model");
    let chip = field("chip_type").unwrap_or("unknown chip");
    let memory = field("physical_memory").unwrap_or("unknown memory");
    Ok(format!("{machine} ({model}; {chip}; {memory})"))
}

fn mlx_revision(lockfile: &Path) -> Result<String, String> {
    let lock = fs::read_to_string(lockfile)
        .map_err(|error| format!("read {}: {error}", lockfile.display()))?;
    let marker = "git+https://github.com/michaeltrefry/mlx-rs?rev=";
    let source = lock
        .lines()
        .find(|line| line.contains(marker))
        .ok_or_else(|| "Cargo.lock has no pinned pmetal mlx-rs source".to_owned())?;
    let revision = source
        .split("?rev=")
        .nth(1)
        .and_then(|tail| tail.split('#').next())
        .ok_or_else(|| "cannot parse mlx-rs revision from Cargo.lock".to_owned())?;
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("Cargo.lock mlx-rs revision is not a full SHA".to_owned());
    }
    Ok(revision.to_ascii_lowercase())
}

fn metal_device() -> Result<String, String> {
    let raw = command_output(
        "/usr/sbin/system_profiler",
        &["SPDisplaysDataType", "-json"],
    )?;
    let value: serde_json::Value =
        serde_json::from_str(&raw).map_err(|error| format!("parse display profile: {error}"))?;
    fn find(value: &serde_json::Value) -> Option<&str> {
        match value {
            serde_json::Value::Object(map) => map
                .get("sppci_model")
                .and_then(serde_json::Value::as_str)
                .or_else(|| map.values().find_map(find)),
            serde_json::Value::Array(items) => items.iter().find_map(find),
            _ => None,
        }
    }
    find(&value)
        .map(str::to_owned)
        .ok_or_else(|| "system_profiler did not report a Metal device".to_owned())
}

fn write_json_atomic(path: &Path, value: &impl serde::Serialize) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent", path.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} has no UTF-8 file name", path.display()))?;
    let temporary = parent.join(format!(".{file_name}.tmp"));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| format!("create {}: {error}", temporary.display()))?;
    serde_json::to_writer_pretty(&mut file, value)
        .map_err(|error| format!("serialize {}: {error}", temporary.display()))?;
    file.write_all(b"\n")
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("finish {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|error| format!("publish {}: {error}", path.display()))
}

fn print_summary(summary: &BenchmarkSummary) {
    println!("\nTiming comparison");
    println!(
        "{:<26} {:<12} {:<30} {:>10} {:>10} {:>10}",
        "case", "tier", "variant", "seconds", "steps/s", "speedup"
    );
    for row in &summary.rows {
        println!(
            "{:<26} {:<12} {:<30} {:>10.3} {:>10.3} {:>9.3}x",
            row.case_id,
            row.tier.as_str(),
            row.variant_id,
            row.median_total_seconds,
            row.median_denoise_steps_per_second,
            row.speedup_vs_baseline
        );
    }
    println!("\nPhase peaks (active/cache GiB)");
    println!(
        "{:<26} {:<30} {:>15} {:>15} {:>15} {:<8}",
        "case", "variant", "encode", "denoise", "decode", "binds"
    );
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    for row in &summary.rows {
        let pair = |active: u64, cache: u64| {
            format!("{:.2}/{:.2}", active as f64 / GIB, cache as f64 / GIB)
        };
        println!(
            "{:<26} {:<30} {:>15} {:>15} {:>15} {:<8?}",
            row.case_id,
            row.variant_id,
            pair(
                row.median_encode_active_peak_bytes,
                row.median_encode_cache_peak_bytes
            ),
            pair(
                row.median_denoise_active_peak_bytes,
                row.median_denoise_cache_peak_bytes
            ),
            pair(
                row.median_decode_active_peak_bytes,
                row.median_decode_cache_peak_bytes
            ),
            row.binding_phase
        );
    }
    println!("\nJSON summary: schema {}", summary.schema_version);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_intervals_exclude_the_progress_boundary_step() {
        assert_eq!(phases_step_intervals(8), 7);
        assert_eq!(phases_step_intervals(1), 1);
    }

    #[test]
    fn mlx_revision_is_bound_to_the_workspace_lock() {
        let revision = mlx_revision(&repository_root().unwrap().join("Cargo.lock")).unwrap();
        assert_eq!(revision.len(), 40);
        assert!(revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn default_matrix_parses_and_validates() {
        let matrix: BenchmarkMatrix = serde_json::from_str(DEFAULT_MATRIX).unwrap();
        assert_eq!(
            matrix.schema_version,
            runtime_macos::perf_bench::MATRIX_SCHEMA_VERSION
        );
        matrix.validate().unwrap();
    }
}
