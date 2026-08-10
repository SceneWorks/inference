#[cfg(not(target_os = "macos"))]
compile_error!("mlx-perf-bench is supported only on macOS with an MLX Metal device");

use mlx_rs::memory::{clear_cache, get_cache_memory, get_peak_memory, reset_peak_memory};
use runtime_macos::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, WeightsSource,
};
use runtime_macos::media::diagnostics::{
    self, BenchmarkDecodeControl, BenchmarkPhaseBoundary, CacheDisposition, CompileDisposition,
    DecodePathDisposition, DecodePolicyDisposition, DiagnosticCounter, DiagnosticReport,
    ToggleDisposition,
};
use runtime_macos::media::memory_probe::{AllocatorProbe, AllocatorProbeReport};
use runtime_macos::perf_bench::{
    build_summary, inventory_artifact, request_receipt, validate_toggle_diagnostics,
    ArtifactManifest, ArtifactReceipt, BenchmarkFamily, BenchmarkMatrix, BenchmarkSummary,
    BindingPhase, BuildProvenance, DecodeControlMode, DiagnosticRecord, FrozenCampaign,
    HostIdentity, MeasurementRecord, MemoryCoverageReceipt, ModelTier, OptimizationToggle,
    OutputFingerprint, PhaseBoundary, PhaseBoundaryReceipt, PhaseMetrics, PhaseSet,
    ProgressReceipt, ProviderCapabilityReceipt, RunRecord, StepReceipt, VariantPlan, WorkloadCase,
    MEMORY_SAMPLE_INTERVAL_MICROS, RUN_SCHEMA_VERSION,
};
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_MATRIX: &str = include_str!("../../benchmarks/mlx-perf-matrix-v1.json");
const CAMPAIGN_FILE: &str = "campaign.json";
const SUMMARY_FILE: &str = "summary.json";

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
        Some("validate") => validate_inputs(&args[1..]),
        Some("run") => run_matrix(&args[1..]),
        Some("child") => run_child(&args[1..]),
        Some("validate-results") => validate_results(&args[1..]),
        _ => Err(usage()),
    }
}

fn usage() -> String {
    "usage:\n  mlx-perf-bench validate [--matrix PATH] --artifacts PATH\n  \
     mlx-perf-bench run [--matrix PATH] --artifacts PATH --output-dir PATH \
     [--variants baseline,id,...]\n  \
     mlx-perf-bench validate-results --results-dir PATH\n\nThe output directory must be \
     absolute, outside the checkout, and empty. `run` defaults to the required-all matrix. \
     `--variants baseline` creates a runnable baseline campaign but never acceptance evidence."
        .to_owned()
}

#[derive(Default)]
struct Options {
    matrix: Option<PathBuf>,
    artifacts: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    results_dir: Option<PathBuf>,
    variants: Option<Vec<String>>,
    campaign: Option<PathBuf>,
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
            "--results-dir" => options.results_dir = Some(PathBuf::from(value)),
            "--campaign" => options.campaign = Some(PathBuf::from(value)),
            "--case" => options.case_id = Some(value.clone()),
            "--variant" => options.variant_id = Some(value.clone()),
            "--output-file" => options.output_file = Some(PathBuf::from(value)),
            "--variants" => {
                options.variants = Some(
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_owned)
                        .collect(),
                );
            }
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
    verify_executable_provenance(None)?;
    let (matrix, manifest) = load_inputs(&options)?;
    let capabilities = provider_capability_receipts(&matrix)?;
    let campaign = FrozenCampaign::freeze(
        matrix,
        &manifest,
        vec!["baseline".to_owned()],
        build_provenance()?,
        host_identity()?,
        capabilities,
        unix_millis()?,
    )
    .map_err(|error| format!("invalid frozen inputs: {error}"))?;
    println!(
        "validated {}: {} cases, {} exact artifacts; campaign {}",
        campaign.matrix.benchmark_id,
        campaign.matrix.cases.len(),
        campaign.artifacts.len(),
        campaign.campaign_id
    );
    Ok(())
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
    if selected.is_empty() || unique.len() != selected.len() || !unique.contains("baseline") {
        return Err("--variants must be non-empty, unique, and include baseline".to_owned());
    }
    for id in &selected {
        let variant = matrix
            .variant(id)
            .ok_or_else(|| format!("unknown variant {id:?}"))?;
        if !unique.contains(variant.control_variant.as_str()) {
            return Err(format!(
                "variant {:?} requires control {:?} in --variants",
                variant.id, variant.control_variant
            ));
        }
    }
    Ok(selected)
}

fn run_matrix(args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    verify_executable_provenance(None)?;
    let (matrix, manifest) = load_inputs(&options)?;
    let selected = selected_variants(&matrix, options.variants.as_deref())?;
    let output_dir = prepare_new_output_dir(
        options
            .output_dir
            .as_deref()
            .ok_or_else(|| "--output-dir is required".to_owned())?,
    )?;
    let capabilities = provider_capability_receipts(&matrix)?;
    let campaign = FrozenCampaign::freeze(
        matrix,
        &manifest,
        selected,
        build_provenance()?,
        host_identity()?,
        capabilities,
        unix_millis()?,
    )
    .map_err(|error| format!("refuse unavailable or unfrozen campaign: {error}"))?;
    let campaign_file = output_dir.join(CAMPAIGN_FILE);
    write_json_atomic(&campaign_file, &campaign)?;

    let executable =
        env::current_exe().map_err(|error| format!("resolve current binary: {error}"))?;
    for case in &campaign.matrix.cases {
        for variant_id in &campaign.selected_variants {
            let output_file = output_dir.join(run_file_name(&case.id, variant_id));
            println!("run {} / {}", case.id, variant_id);
            let status = Command::new(&executable)
                .arg("child")
                .arg("--campaign")
                .arg(&campaign_file)
                .arg("--case")
                .arg(&case.id)
                .arg("--variant")
                .arg(variant_id)
                .arg("--output-file")
                .arg(&output_file)
                .stdin(Stdio::null())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .status()
                .map_err(|error| format!("start child benchmark: {error}"))?;
            if !status.success() {
                return Err(format!(
                    "child failed for {} / {}; no summary was published",
                    case.id, variant_id
                ));
            }
        }
    }

    let records = load_campaign_records(&output_dir, &campaign, false)?;
    let summary = build_summary(&campaign, &records)
        .map_err(|error| format!("refuse incomplete comparison set: {error}"))?;
    write_json_atomic(&output_dir.join(SUMMARY_FILE), &summary)?;
    print_summary(&summary);
    Ok(())
}

fn validate_results(args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    let results_dir = existing_results_dir(
        options
            .results_dir
            .as_deref()
            .ok_or_else(|| "--results-dir is required".to_owned())?,
    )?;
    let campaign_path = results_dir.join(CAMPAIGN_FILE);
    if !campaign_path.is_file() {
        return Err(format!(
            "legacy/unbound results: missing {} in {}",
            CAMPAIGN_FILE,
            results_dir.display()
        ));
    }
    let campaign: FrozenCampaign = load_json(&campaign_path, "frozen campaign")?;
    campaign
        .validate()
        .map_err(|error| format!("invalid frozen campaign: {error}"))?;
    verify_executable_provenance(Some(&campaign.build))?;
    let records = load_campaign_records(&results_dir, &campaign, true)?;
    let summary = build_summary(&campaign, &records)
        .map_err(|error| format!("invalid comparison set: {error}"))?;
    let stored_summary = results_dir.join(SUMMARY_FILE);
    if stored_summary.is_file() {
        let stored: BenchmarkSummary = load_json(&stored_summary, "stored summary")?;
        if stored != summary {
            return Err("stored summary does not match the validated campaign records".to_owned());
        }
    }
    print_summary(&summary);
    Ok(())
}

fn load_campaign_records(
    output_dir: &Path,
    campaign: &FrozenCampaign,
    allow_summary: bool,
) -> Result<Vec<RunRecord>, String> {
    let mut expected = BTreeSet::from([CAMPAIGN_FILE.to_owned()]);
    if allow_summary {
        expected.insert(SUMMARY_FILE.to_owned());
    }
    for case in &campaign.matrix.cases {
        for variant in &campaign.selected_variants {
            expected.insert(run_file_name(&case.id, variant));
        }
    }
    for entry in fs::read_dir(output_dir)
        .map_err(|error| format!("read results directory {}: {error}", output_dir.display()))?
    {
        let entry = entry.map_err(|error| format!("read results entry: {error}"))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "results contain a non-UTF-8 entry".to_owned())?;
        if !expected.contains(&name) {
            return Err(format!(
                "results contain an unexpected mixed/stale entry {name:?}"
            ));
        }
    }
    let mut records = Vec::new();
    for case in &campaign.matrix.cases {
        for variant in &campaign.selected_variants {
            records.push(load_json(
                &output_dir.join(run_file_name(&case.id, variant)),
                "run record",
            )?);
        }
    }
    Ok(records)
}

fn run_child(args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    let campaign_path = options
        .campaign
        .as_deref()
        .ok_or_else(|| "child requires --campaign".to_owned())?;
    let campaign: FrozenCampaign = load_json(campaign_path, "frozen campaign")?;
    campaign
        .validate()
        .map_err(|error| format!("invalid frozen campaign: {error}"))?;
    verify_executable_provenance(Some(&campaign.build))?;
    let current_host = host_identity()?;
    if current_host != campaign.host {
        return Err("runtime host identity differs from the frozen campaign".to_owned());
    }

    let case = campaign
        .matrix
        .case(
            options
                .case_id
                .as_deref()
                .ok_or_else(|| "child requires --case".to_owned())?,
        )
        .ok_or_else(|| "child case is not in the frozen campaign".to_owned())?;
    let variant = campaign
        .matrix
        .variant(
            options
                .variant_id
                .as_deref()
                .ok_or_else(|| "child requires --variant".to_owned())?,
        )
        .filter(|variant| campaign.selected_variants.contains(&variant.id))
        .ok_or_else(|| "child variant is not selected by the frozen campaign".to_owned())?;
    let output_file = options
        .output_file
        .as_deref()
        .ok_or_else(|| "child requires --output-file".to_owned())?;
    let output_dir = output_file
        .parent()
        .ok_or_else(|| "output file must have a parent directory".to_owned())?;
    let output_dir = existing_results_dir(output_dir)?;
    let campaign_parent = campaign_path
        .parent()
        .ok_or_else(|| "campaign file has no parent".to_owned())?
        .canonicalize()
        .map_err(|error| format!("canonicalize campaign directory: {error}"))?;
    if output_dir != campaign_parent
        || output_file.file_name().and_then(|name| name.to_str())
            != Some(run_file_name(&case.id, &variant.id).as_str())
    {
        return Err("child output must be the exact campaign-owned run filename".to_owned());
    }
    let output_file = output_dir.join(run_file_name(&case.id, &variant.id));
    if output_file.exists() {
        return Err(format!("refuse to overwrite {}", output_file.display()));
    }

    let artifact = campaign
        .artifact(&case.artifact_key)
        .expect("validated campaign covers every case artifact");
    verify_artifact_content(artifact)?;
    let available_toggles = campaign
        .capabilities(&case.provider)
        .ok_or_else(|| "campaign omitted provider capability receipt".to_owned())?;
    if !variant
        .toggles
        .iter()
        .all(|toggle| available_toggles.contains(toggle))
    {
        return Err(format!(
            "provider {} does not declare every toggle requested by {}",
            case.provider, variant.id
        ));
    }

    let started_at_unix_millis = unix_millis()?;
    clear_cache();
    reset_peak_memory();
    let load_start = Instant::now();
    let catalog =
        runtime_macos::catalog().map_err(|error| format!("build macOS catalog: {error}"))?;
    let generator = catalog
        .media()
        .load(
            &case.provider,
            &load_spec(case.tier, &artifact.canonical_path),
        )
        .map_err(|error| format!("load {}: {error}", case.provider))?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let load_active_peak_bytes = get_peak_memory() as u64;
    let load_cache_bytes_after_load = get_cache_memory() as u64;
    if !(load_seconds.is_finite() && load_seconds > 0.0) {
        return Err("cold model load did not produce credible timing evidence".to_owned());
    }

    for warmup in 0..campaign.matrix.warmup_runs {
        let request_id = format!("{}/{}/warmup-{warmup}", case.id, variant.id);
        let _ = measure_request(&*generator, case, variant, &request_id, warmup)?;
    }
    let mut measurements = Vec::new();
    for repetition in 0..campaign.matrix.measured_runs {
        let request_id = format!("{}/{}/measured-{repetition}", case.id, variant.id);
        measurements.push(measure_request(
            &*generator,
            case,
            variant,
            &request_id,
            repetition,
        )?);
    }

    let record = RunRecord {
        schema_version: RUN_SCHEMA_VERSION.to_owned(),
        campaign_id: campaign.campaign_id.clone(),
        benchmark_id: campaign.matrix.benchmark_id.clone(),
        case_id: case.id.clone(),
        family: case.family,
        provider: case.provider.clone(),
        artifact: artifact.clone(),
        variant: variant.clone(),
        request: request_receipt(&campaign, case, artifact).map_err(|error| error.to_string())?,
        build: campaign.build.clone(),
        host: campaign.host.clone(),
        available_toggles: available_toggles.to_vec(),
        started_at_unix_millis,
        load_seconds,
        load_active_peak_bytes,
        load_cache_bytes_after_load,
        warmup_runs_completed: campaign.matrix.warmup_runs,
        measurements,
    };
    record
        .validate_against(&campaign)
        .map_err(|error| format!("refuse false-green run record: {error}"))?;
    write_json_atomic(&output_file, &record)?;
    println!("wrote {}", output_file.display());
    Ok(())
}

fn verify_artifact_content(artifact: &ArtifactReceipt) -> Result<(), String> {
    if !artifact.canonical_path.is_dir() {
        return Err(format!(
            "frozen artifact path is unavailable: {}",
            artifact.canonical_path.display()
        ));
    }
    let actual = inventory_artifact(&artifact.canonical_path).map_err(|error| error.to_string())?;
    if actual != artifact.inventory {
        return Err(format!(
            "artifact {} changed after the campaign was frozen",
            artifact.key
        ));
    }
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

fn generation_request(case: &WorkloadCase) -> GenerationRequest {
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
    case: &WorkloadCase,
    variant: &VariantPlan,
    request_id: &str,
    repetition: u32,
) -> Result<MeasurementRecord, String> {
    let request = generation_request(case);
    generator
        .validate(&request)
        .map_err(|error| format!("validate {}: {error}", case.id))?;
    clear_cache();

    let recorder = Rc::new(RefCell::new(PhaseRecorder::new(case.steps)));
    let observer = Rc::clone(&recorder);
    let requested: Vec<_> = variant
        .toggles
        .iter()
        .map(|toggle| toggle.as_str())
        .collect();
    let scope = diagnostics::begin_benchmark_request(
        request_id,
        case.family.as_str(),
        &requested,
        benchmark_decode_control(case, variant)?,
        move |boundary| observer.borrow_mut().transition(boundary),
    )
    .map_err(|error| error.to_string())?;
    let output = generator
        .generate(&request, &mut |progress| {
            recorder.borrow_mut().observe(progress)
        })
        .map_err(|error| format!("generate {} / {}: {error}", case.id, variant.id))?;
    let report = scope.finish();
    let recorder = Rc::try_unwrap(recorder)
        .map_err(|_| "phase observer remained alive after diagnostics finished".to_owned())?
        .into_inner();
    let finished = recorder.finish()?;
    validate_report_identity_and_boundaries(request_id, case, &report, &finished.phase_boundaries)?;
    let diagnostics = diagnostic_records(report);
    validate_toggle_diagnostics(variant, &diagnostics)
        .map_err(|error| format!("invalid toggle receipts: {error}"))?;
    let output = fingerprint_output(case, &output)?;
    let denoise_steps_per_second = case.steps as f64 / finished.phases.denoise.seconds;
    Ok(MeasurementRecord {
        repetition,
        total_elapsed_nanos: finished.total_elapsed_nanos,
        total_seconds: finished.total_elapsed_nanos as f64 / 1e9,
        denoise_steps_per_second,
        progress: finished.progress,
        phase_boundaries: finished.phase_boundaries,
        phases: finished.phases,
        output,
        diagnostics,
    })
}

fn benchmark_decode_control(
    case: &WorkloadCase,
    variant: &VariantPlan,
) -> Result<Option<BenchmarkDecodeControl>, String> {
    if variant.decode_control == DecodeControlMode::Default {
        return Ok(None);
    }
    let control = case.tiled_decode_control;
    let to_i32 = |value: u32, label: &str| {
        i32::try_from(value).map_err(|_| format!("{label} exceeds the MLX tiling domain"))
    };
    Ok(Some(BenchmarkDecodeControl {
        spatial_tile_px: to_i32(control.spatial_tile_px, "spatial tile")?,
        spatial_overlap_px: to_i32(control.spatial_overlap_px, "spatial overlap")?,
        temporal_tile_frames: control
            .temporal_tile_frames
            .map(|value| to_i32(value, "temporal tile"))
            .transpose()?,
        temporal_overlap_frames: control
            .temporal_overlap_frames
            .map(|value| to_i32(value, "temporal overlap"))
            .transpose()?,
    }))
}

fn validate_report_identity_and_boundaries(
    request_id: &str,
    case: &WorkloadCase,
    report: &DiagnosticReport,
    receipts: &[PhaseBoundaryReceipt],
) -> Result<(), String> {
    if report.request_id != request_id || report.family != case.family.as_str() {
        return Err("diagnostic report escaped its request/family scope".to_owned());
    }
    let reported: Vec<_> = report
        .phase_boundaries
        .iter()
        .map(|record| match record.boundary {
            BenchmarkPhaseBoundary::DenoiseStart => PhaseBoundary::DenoiseStart,
            BenchmarkPhaseBoundary::DecodeStart => PhaseBoundary::DecodeStart,
        })
        .collect();
    let observed: Vec<_> = receipts.iter().map(|receipt| receipt.boundary).collect();
    if reported != observed
        || report
            .phase_boundaries
            .windows(2)
            .any(|window| window[0].elapsed_nanos >= window[1].elapsed_nanos)
    {
        return Err(
            "provider diagnostic phase boundaries were missing, duplicated, or reordered"
                .to_owned(),
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Encode,
    Denoise,
    Decode,
}

struct OpenPhase {
    phase: Phase,
    started: Instant,
    probe: AllocatorProbe,
}

impl OpenPhase {
    fn new(phase: Phase, started: Instant) -> Self {
        reset_peak_memory();
        Self {
            phase,
            started,
            probe: AllocatorProbe::start(Duration::from_micros(MEMORY_SAMPLE_INTERVAL_MICROS)),
        }
    }

    fn finish(self, at: Instant) -> PhaseMetrics {
        let native_active_peak_bytes = get_peak_memory() as u64;
        let report = self.probe.finish();
        phase_metrics(
            at.duration_since(self.started).as_secs_f64(),
            native_active_peak_bytes,
            report,
        )
    }
}

fn phase_metrics(
    seconds: f64,
    native_active_peak_bytes: u64,
    report: AllocatorProbeReport,
) -> PhaseMetrics {
    PhaseMetrics {
        seconds,
        native_active_peak_bytes,
        sampled_active_peak_bytes: report.sampled_active_peak_bytes,
        sampled_cache_peak_bytes: report.sampled_cache_peak_bytes,
        sampled_footprint_peak_bytes: report.sampled_footprint_peak_bytes,
        footprint_peak_active_bytes: report.footprint_peak_active_bytes,
        footprint_peak_cache_bytes: report.footprint_peak_cache_bytes,
        boundary_active_bytes: report.boundary_active_bytes,
        boundary_cache_bytes: report.boundary_cache_bytes,
        coverage: MemoryCoverageReceipt {
            interval_micros: report.interval_micros,
            sample_count: report.sample_count,
            periodic_sample_count: report.periodic_sample_count,
            sampling_span_micros: report.sampling_span_micros,
            max_gap_micros: report.max_gap_micros,
        },
    }
}

struct FinishedPhases {
    total_elapsed_nanos: u64,
    progress: ProgressReceipt,
    phase_boundaries: Vec<PhaseBoundaryReceipt>,
    phases: PhaseSet,
}

struct PhaseRecorder {
    started: Instant,
    current: Option<OpenPhase>,
    encode: Option<PhaseMetrics>,
    denoise: Option<PhaseMetrics>,
    decode: Option<PhaseMetrics>,
    expected_steps: u32,
    steps: Vec<StepReceipt>,
    decoding_elapsed_nanos: Option<u64>,
    phase_boundaries: Vec<PhaseBoundaryReceipt>,
    error: Option<String>,
}

impl PhaseRecorder {
    fn new(expected_steps: u32) -> Self {
        let started = Instant::now();
        Self {
            started,
            current: Some(OpenPhase::new(Phase::Encode, started)),
            encode: None,
            denoise: None,
            decode: None,
            expected_steps,
            steps: Vec::new(),
            decoding_elapsed_nanos: None,
            phase_boundaries: Vec::new(),
            error: None,
        }
    }

    fn elapsed_nanos(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn fail(&mut self, message: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(message.into());
        }
    }

    fn transition(&mut self, boundary: BenchmarkPhaseBoundary) {
        let (expected, next, receipt) = match boundary {
            BenchmarkPhaseBoundary::DenoiseStart => {
                (Phase::Encode, Phase::Denoise, PhaseBoundary::DenoiseStart)
            }
            BenchmarkPhaseBoundary::DecodeStart => {
                if self.steps.len() != self.expected_steps as usize {
                    self.fail(format!(
                        "DecodeStart arrived after {} Step events, expected {}",
                        self.steps.len(),
                        self.expected_steps
                    ));
                }
                (Phase::Denoise, Phase::Decode, PhaseBoundary::DecodeStart)
            }
        };
        let now = Instant::now();
        let elapsed_nanos =
            u64::try_from(now.duration_since(self.started).as_nanos()).unwrap_or(u64::MAX);
        let Some(current) = self.current.take() else {
            self.fail("phase recorder was already closed");
            return;
        };
        if current.phase != expected {
            let actual = current.phase;
            self.current = Some(current);
            self.fail(format!(
                "invalid explicit phase transition: expected {expected:?}, got {actual:?}"
            ));
            return;
        }
        let metrics = current.finish(now);
        match expected {
            Phase::Encode => self.encode = Some(metrics),
            Phase::Denoise => self.denoise = Some(metrics),
            Phase::Decode => self.decode = Some(metrics),
        }
        self.phase_boundaries.push(PhaseBoundaryReceipt {
            boundary: receipt,
            elapsed_nanos,
        });
        self.current = Some(OpenPhase::new(next, now));
    }

    fn observe(&mut self, progress: Progress) {
        match progress {
            Progress::Step { current, total } => {
                let expected_current = self.steps.len() as u32 + 1;
                if self.decoding_elapsed_nanos.is_some() {
                    self.fail("Step progress arrived after Decoding");
                }
                if current != expected_current || total != self.expected_steps {
                    self.fail(format!(
                        "invalid Step current={current} total={total}; expected {expected_current}/{}",
                        self.expected_steps
                    ));
                }
                self.steps.push(StepReceipt {
                    current,
                    total,
                    elapsed_nanos: self.elapsed_nanos(),
                });
            }
            Progress::Decoding => {
                if self.decoding_elapsed_nanos.is_some() {
                    self.fail("duplicate Decoding progress event");
                } else {
                    if self.steps.len() != self.expected_steps as usize {
                        self.fail(format!(
                            "Decoding arrived after {} Step events, expected {}",
                            self.steps.len(),
                            self.expected_steps
                        ));
                    }
                    self.decoding_elapsed_nanos = Some(self.elapsed_nanos());
                }
            }
            Progress::Loading(_) => {}
        }
    }

    fn finish(mut self) -> Result<FinishedPhases, String> {
        let finished_at = Instant::now();
        let total_elapsed_nanos =
            u64::try_from(finished_at.duration_since(self.started).as_nanos()).unwrap_or(u64::MAX);
        if let Some(current) = self.current.take() {
            if current.phase != Phase::Decode {
                self.fail(format!(
                    "generation completed in {:?}, expected Decode",
                    current.phase
                ));
            } else {
                self.decode = Some(current.finish(finished_at));
            }
        } else {
            self.fail("phase recorder closed before generation completed");
        }
        if self.steps.len() != self.expected_steps as usize {
            self.fail(format!(
                "observed {} Step events, expected {}",
                self.steps.len(),
                self.expected_steps
            ));
        }
        if self.phase_boundaries.len() != 2 {
            self.fail(format!(
                "observed {} explicit phase boundaries, expected 2",
                self.phase_boundaries.len()
            ));
        }
        let decoding_elapsed_nanos = match self.decoding_elapsed_nanos {
            Some(value) => value,
            None => {
                self.fail("generation emitted no Decoding event");
                0
            }
        };
        if let Some(error) = self.error {
            return Err(error);
        }
        Ok(FinishedPhases {
            total_elapsed_nanos,
            progress: ProgressReceipt {
                steps: self.steps,
                decoding_elapsed_nanos,
            },
            phase_boundaries: self.phase_boundaries,
            phases: PhaseSet {
                encode: self
                    .encode
                    .ok_or_else(|| "missing encode phase".to_owned())?,
                denoise: self
                    .denoise
                    .ok_or_else(|| "missing denoise phase".to_owned())?,
                decode: self
                    .decode
                    .ok_or_else(|| "missing decode phase".to_owned())?,
            },
        })
    }
}

fn fingerprint_output(
    case: &WorkloadCase,
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
            return Err("image/video P6 matrix unexpectedly produced audio-only output".to_owned());
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
    case: &WorkloadCase,
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
                decode_path: None,
                production_evidence_sha256: None,
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
                decode_path: None,
                production_evidence_sha256: None,
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
                decode_path: None,
                production_evidence_sha256: None,
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
                decode_path: None,
                production_evidence_sha256: None,
            },
            DiagnosticCounter::DecodePolicy {
                disposition,
                decode_path,
                production_evidence_sha256,
                count,
            } => DiagnosticRecord {
                domain: "decode_policy".to_owned(),
                site: diagnostics::GEOMETRY_AWARE_DECODE.to_owned(),
                outcome: match disposition {
                    DecodePolicyDisposition::Unchanged => "unchanged",
                    DecodePolicyDisposition::GeometryTiled => "geometry_tiled",
                }
                .to_owned(),
                count,
                reason: None,
                decode_path: Some(
                    match decode_path {
                        DecodePathDisposition::Dense => "dense",
                        DecodePathDisposition::Tiled => "tiled",
                    }
                    .to_owned(),
                ),
                production_evidence_sha256,
            },
        })
        .collect()
}

fn provider_capability_receipts(
    matrix: &BenchmarkMatrix,
) -> Result<Vec<ProviderCapabilityReceipt>, String> {
    let providers: BTreeSet<_> = matrix
        .cases
        .iter()
        .map(|case| case.provider.as_str())
        .collect();
    providers
        .into_iter()
        .map(|provider| {
            let declared =
                runtime_macos::benchmark_toggle_capabilities(provider).ok_or_else(|| {
                    format!("provider {provider:?} has no benchmark capability contract")
                })?;
            let mut available_toggles: Vec<_> = declared
                .iter()
                .map(|name| {
                    OptimizationToggle::from_name(name).ok_or_else(|| {
                        format!("provider {provider:?} declares unknown benchmark toggle {name:?}")
                    })
                })
                .collect::<Result<_, _>>()?;
            available_toggles.sort_unstable();
            available_toggles.dedup();
            if available_toggles.len() != declared.len() {
                return Err(format!("provider {provider:?} repeats a benchmark toggle"));
            }
            Ok(ProviderCapabilityReceipt {
                provider: provider.to_owned(),
                available_toggles,
            })
        })
        .collect()
}

fn repository_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or_else(|| "cannot resolve repository root from CARGO_MANIFEST_DIR".to_owned())?
        .canonicalize()
        .map_err(|error| format!("canonicalize repository root: {error}"))
}

fn ensure_outside_checkout(path: &Path) -> Result<(), String> {
    let root = repository_root()?;
    if path.starts_with(&root) {
        return Err(format!(
            "benchmark evidence must be outside the repository ({} is under {})",
            path.display(),
            root.display()
        ));
    }
    Ok(())
}

fn prepare_new_output_dir(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("output path must be absolute: {}", path.display()));
    }
    fs::create_dir_all(path)
        .map_err(|error| format!("create output directory {}: {error}", path.display()))?;
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("canonicalize output directory {}: {error}", path.display()))?;
    ensure_outside_checkout(&canonical)?;
    if fs::read_dir(&canonical)
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

fn existing_results_dir(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() || !path.is_dir() {
        return Err(format!(
            "results path must be an existing absolute directory: {}",
            path.display()
        ));
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("canonicalize results directory {}: {error}", path.display()))?;
    ensure_outside_checkout(&canonical)?;
    Ok(canonical)
}

fn run_file_name(case_id: &str, variant_id: &str) -> String {
    format!("{case_id}__{variant_id}.json")
}

fn unix_millis() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .map_err(|error| format!("system clock predates Unix epoch: {error}"))
}

fn build_provenance() -> Result<BuildProvenance, String> {
    let source_dirty = match env!("SCENEWORKS_BENCH_SOURCE_DIRTY") {
        "true" => true,
        "false" => false,
        value => return Err(format!("invalid build-time dirty receipt {value:?}")),
    };
    let executable =
        env::current_exe().map_err(|error| format!("resolve current executable: {error}"))?;
    Ok(BuildProvenance {
        source_revision: env!("SCENEWORKS_BENCH_SOURCE_REVISION").to_owned(),
        mlx_revision: env!("SCENEWORKS_BENCH_MLX_REVISION").to_owned(),
        source_dirty,
        cargo_profile: env!("SCENEWORKS_BENCH_CARGO_PROFILE").to_owned(),
        opt_level: env!("SCENEWORKS_BENCH_OPT_LEVEL").to_owned(),
        debug_assertions: cfg!(debug_assertions),
        target_triple: env!("SCENEWORKS_BENCH_TARGET").to_owned(),
        cargo_features: split_receipt(env!("SCENEWORKS_BENCH_CARGO_FEATURES"), ','),
        target_features: split_receipt(env!("SCENEWORKS_BENCH_TARGET_FEATURES"), ','),
        rustflags: split_receipt(env!("SCENEWORKS_BENCH_RUSTFLAGS"), '\u{241f}'),
        rustc_version: env!("SCENEWORKS_BENCH_RUSTC_VERSION").to_owned(),
        executable_sha256: sha256_file(&executable)?,
    })
}

fn split_receipt(value: &str, separator: char) -> Vec<String> {
    value
        .split(separator)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path)
        .map_err(|error| format!("open executable {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("hash executable {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn verify_executable_provenance(expected: Option<&BuildProvenance>) -> Result<(), String> {
    let build = build_provenance()?;
    if build.source_dirty {
        return Err(
            "benchmark executable was built from a dirty checkout; commit and rebuild it"
                .to_owned(),
        );
    }
    let root = repository_root()?;
    ensure_clean_checkout(&root)?;
    let root_text = root
        .to_str()
        .ok_or_else(|| "repository root is not UTF-8".to_owned())?;
    let runtime_source = command_output("git", &["-C", root_text, "rev-parse", "HEAD"])?;
    let runtime_mlx = mlx_revision(&root.join("Cargo.lock"))?;
    if runtime_source != build.source_revision || runtime_mlx != build.mlx_revision {
        return Err(format!(
            "runtime checkout/lock differs from executable build provenance: build={}/{} runtime={}/{}",
            build.source_revision, build.mlx_revision, runtime_source, runtime_mlx
        ));
    }
    if expected.is_some_and(|expected| expected != &build) {
        return Err("executable build provenance differs from the frozen campaign".to_owned());
    }
    Ok(())
}

fn ensure_clean_checkout(root: &Path) -> Result<(), String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["status", "--porcelain", "--untracked-files=all"])
        .output()
        .map_err(|error| format!("inspect checkout status: {error}"))?;
    if !output.status.success() {
        return Err("git status failed while binding benchmark provenance".to_owned());
    }
    if !output.stdout.is_empty() {
        return Err("benchmark evidence requires a clean inference checkout".to_owned());
    }
    Ok(())
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

fn host_identity() -> Result<HostIdentity, String> {
    Ok(HostIdentity {
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
    Ok(format!(
        "{} ({}; {}; {})",
        field("machine_name").unwrap_or("Mac"),
        field("machine_model").unwrap_or("unknown model"),
        field("chip_type").unwrap_or("unknown chip"),
        field("physical_memory").unwrap_or("unknown memory")
    ))
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
    println!("\nStage timing comparison (seconds)");
    println!(
        "{:<26} {:<24} {:<24} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "case",
        "variant",
        "control",
        "total",
        "encode",
        "denoise",
        "decode",
        "steps/s",
        "vs ctl",
        "vs base"
    );
    for row in &summary.rows {
        println!(
            "{:<26} {:<24} {:<24} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>8.3}x {:>8.3}x",
            row.case_id,
            row.variant_id,
            row.control_variant_id,
            row.median_total_seconds,
            row.median_encode_seconds,
            row.median_denoise_seconds,
            row.median_decode_seconds,
            row.median_denoise_steps_per_second,
            row.speedup_vs_control,
            row.speedup_vs_baseline
        );
    }
    println!("\nPhase allocator peaks (native-active / sampled-cache / paired-footprint GiB)");
    println!(
        "{:<26} {:<24} {:>20} {:>20} {:>20} {:<8}",
        "case", "variant", "encode", "denoise", "decode", "binds"
    );
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let triple = |active: u64, cache: u64, footprint: u64| {
        format!(
            "{:.2}/{:.2}/{:.2}",
            active as f64 / GIB,
            cache as f64 / GIB,
            footprint as f64 / GIB
        )
    };
    for row in &summary.rows {
        println!(
            "{:<26} {:<24} {:>20} {:>20} {:>20} {:<8}",
            row.case_id,
            row.variant_id,
            triple(
                row.median_encode_native_active_peak_bytes,
                row.median_encode_sampled_cache_peak_bytes,
                row.median_encode_sampled_footprint_peak_bytes
            ),
            triple(
                row.median_denoise_native_active_peak_bytes,
                row.median_denoise_sampled_cache_peak_bytes,
                row.median_denoise_sampled_footprint_peak_bytes
            ),
            triple(
                row.median_decode_native_active_peak_bytes,
                row.median_decode_sampled_cache_peak_bytes,
                row.median_decode_sampled_footprint_peak_bytes
            ),
            match row.binding_phase {
                BindingPhase::Encode => "encode",
                BindingPhase::Denoise => "denoise",
                BindingPhase::Decode => "decode",
            }
        );
    }
    println!(
        "\nJSON summary: schema {}, campaign {}, acceptance_complete={}",
        summary.schema_version, summary.campaign_id, summary.acceptance_complete
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matrix_parses_and_validates() {
        let matrix: BenchmarkMatrix = serde_json::from_str(DEFAULT_MATRIX).unwrap();
        assert_eq!(
            matrix.schema_version,
            runtime_macos::perf_bench::MATRIX_SCHEMA_VERSION
        );
        matrix.validate().unwrap();
    }

    #[test]
    fn full_selection_is_default_and_baseline_is_required() {
        let matrix: BenchmarkMatrix = serde_json::from_str(DEFAULT_MATRIX).unwrap();
        assert_eq!(selected_variants(&matrix, None).unwrap().len(), 8);
        assert!(selected_variants(&matrix, Some(&["exact_epilogues".to_owned()])).is_err());
        assert!(selected_variants(
            &matrix,
            Some(&[
                "baseline".to_owned(),
                "indexed_decode_accumulator".to_owned()
            ])
        )
        .is_err());
    }

    #[test]
    fn compile_time_build_receipt_has_exact_revisions() {
        let receipt = build_provenance().unwrap();
        assert_eq!(receipt.source_revision.len(), 40);
        assert_eq!(receipt.mlx_revision.len(), 40);
        assert!(!receipt.cargo_profile.is_empty());
        assert!(!receipt.opt_level.is_empty());
        assert!(!receipt.target_triple.is_empty());
        assert!(!receipt.rustc_version.is_empty());
        assert_eq!(receipt.executable_sha256.len(), 64);
        assert!(receipt
            .cargo_features
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn fixed_tiled_control_is_derived_only_for_controlled_variants() {
        let matrix: BenchmarkMatrix = serde_json::from_str(DEFAULT_MATRIX).unwrap();
        let case = matrix.case("wan-q4-832x480-f81").unwrap();
        let baseline = matrix.variant("baseline").unwrap();
        let control = matrix.variant("tiled_decode_control").unwrap();

        assert_eq!(benchmark_decode_control(case, baseline).unwrap(), None);
        assert_eq!(
            benchmark_decode_control(case, control).unwrap(),
            Some(BenchmarkDecodeControl {
                spatial_tile_px: 256,
                spatial_overlap_px: 64,
                temporal_tile_frames: Some(32),
                temporal_overlap_frames: Some(8),
            })
        );
    }

    #[test]
    fn p9_policy_counter_preserves_production_evidence_identity() {
        let evidence = "a".repeat(64);
        let records = diagnostic_records(DiagnosticReport {
            request_id: "p9".to_owned(),
            family: "image_dit".to_owned(),
            counters: vec![DiagnosticCounter::DecodePolicy {
                disposition: DecodePolicyDisposition::GeometryTiled,
                decode_path: DecodePathDisposition::Tiled,
                production_evidence_sha256: Some(evidence.clone()),
                count: 1,
            }],
            phase_boundaries: Vec::new(),
        });

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].domain, "decode_policy");
        assert_eq!(records[0].site, diagnostics::GEOMETRY_AWARE_DECODE);
        assert_eq!(records[0].outcome, "geometry_tiled");
        assert_eq!(records[0].decode_path.as_deref(), Some("tiled"));
        assert_eq!(
            records[0].production_evidence_sha256.as_deref(),
            Some(evidence.as_str())
        );
    }

    #[test]
    fn mlx_revision_is_bound_to_the_workspace_lock() {
        let revision = mlx_revision(&repository_root().unwrap().join("Cargo.lock")).unwrap();
        assert_eq!(revision.len(), 40);
        assert!(revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }
}
