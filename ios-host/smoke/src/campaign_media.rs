//! Catalog-driven device campaign adapter. Enumeration is weights-free.
//!
//! Two execution paths, never mixed:
//! * ADMITTED (default): runtime-ios's admitted facades against the single reviewed evidence table;
//!   a refusal carries a typed kind (capacity / no_evidence / mode / strategy / artifact).
//! * CAPTURE (`--diagnostic`): an explicitly labelled, bounded device capture through
//!   `runtime_ios::media::capture_permit`. No evidence row is needed (that would be circular); every
//!   quantum checks live OS headroom before constructing its graph, so the first quantum that does
//!   not fit refuses BEFORE allocating, and that typed refusal is the "next size refused before
//!   jetsam" evidence. A capture row is never a production pass and never changes admission: a
//!   human reviews it into `evidence/ios_device_evidence.json`.
//!
//! Execution is never authorized by a host peak, launch headroom or a directory name.
use super::campaign::{self, AdmissionError, Case, ExecuteError, Provider, Status, Surface};
use runtime_ios::gen_core::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::result::Result;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
#[path = "campaign_support.rs"]
mod support;
use support::{case_timeout, classify, Deadline};

#[cfg(test)]
#[path = "campaign_admission_tests.rs"]
mod admission_tests;
#[cfg(test)]
#[path = "campaign_media_tests.rs"]
mod tests;

pub fn surfaces(registry: &ProviderRegistry) -> Vec<Surface> {
    let mut out = Vec::new();
    for registration in registry.generators() {
        let d = (registration.descriptor)();
        let mut modes = vec!["text".into()];
        modes.extend(
            d.capabilities
                .conditioning
                .iter()
                .map(|m| format!("{m:?}").to_ascii_lowercase()),
        );
        if d.capabilities.supports_prompt_batch {
            modes.push("prompt_batch".into());
        }
        let routes = if d.id == "sdxl" {
            runtime_ios::media::sdxl::SDXL_FAMILY_ROUTES
                .iter()
                .map(|s| (*s).into())
                .collect()
        } else {
            vec![d.id.into()]
        };
        out.push(Surface {
            provider: d.id.into(),
            routes,
            modes,
        });
    }
    for r in registry.transforms() {
        let d = (r.descriptor)();
        out.push(Surface {
            provider: d.id.into(),
            routes: vec![d.id.into()],
            modes: vec!["restore".into()],
        });
    }
    // Composed provider-specific tasks are not generators. Discover them through their actual
    // memory registrations, not a second hard-coded catalog. Unknown tasks remain explicit cells.
    for r in registry.memory_strategy_registrations() {
        if !out.iter().any(|s| s.provider == r.provider_id) {
            out.push(Surface {
                provider: r.provider_id.into(),
                routes: vec![r.provider_id.into()],
                modes: vec![if r.provider_id == "iris_3b_depth" {
                    "depth".into()
                } else {
                    "task".into()
                }],
            });
        }
    }
    out
}

pub fn listing() -> Result<String, String> {
    let catalog = runtime_ios::catalog().map_err(|e| e.to_string())?;
    let cases = campaign::enumerate(&surfaces(catalog.media()));
    Ok(json!({"schema": 1, "cases": cases.iter().map(Case::key).collect::<Vec<_>>(),
        "dependencies": [{"provider":"supra2", "epic":24212, "available": cases.iter().any(|c| c.provider.contains("supra")), "reason":"requires native engine and iOS catalog registration; absence is not a passing cell"}]
    }).to_string())
}

fn tier(case: &Case) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: Precision::Bf16,
        quant: match case.tier.as_str() {
            "q4" => Some(Quant::Q4),
            "q8" => Some(Quant::Q8),
            _ => None,
        },
        component_precision_floors: &[],
    }
}

/// A deterministic, non-constant synthetic RGB image: conditioning/task input for capture and
/// admitted cases. Pixels never enter an evidence key (geometry and mode do).
fn synthetic_image(edge: u32) -> Image {
    Image {
        width: edge,
        height: edge,
        pixels: (0..edge as usize * edge as usize * 3)
            .map(|i| ((i * 37) % 251) as u8)
            .collect(),
    }
}

/// Providers that pad their text to a fixed window get a window-filling prompt, so a capture
/// prices the full encoder; SDXL keys the exact CLIP window count, so it uses one window.
fn prompt(provider: &str) -> String {
    const SHORT: &str = "A red ceramic teapot on a wooden table in daylight";
    if provider == "sdxl" {
        SHORT.into()
    } else {
        std::iter::repeat_n(
            "A red ceramic teapot on a wooden table in warm afternoon daylight, soft shadows, \
             detailed glaze reflections, a linen cloth, a window with green plants behind,",
            12,
        )
        .collect::<Vec<_>>()
        .join(" ")
    }
}

/// The actual work of a case's mode. Every registered mode has a carrier; a mode the provider
/// does not serve is refused by admission with a typed `mode` kind, never dropped here.
fn work(c: &Case, cancel: &CancelFlag) -> Result<runtime_ios::media::IosWork, AdmissionError> {
    use runtime_ios::media::IosWork;
    let image = || synthetic_image(c.edge);
    let mut request = GenerationRequest {
        prompt: prompt(&c.provider),
        width: c.edge,
        height: c.edge,
        seed: Some(16774),
        count: 1,
        cancel: cancel.clone(),
        ..Default::default()
    };
    match c.mode.as_str() {
        "text" => {}
        "prompt_batch" => {
            request.prompt_batch = vec![
                std::mem::take(&mut request.prompt),
                "A blue glass vase beside a window at sunset".into(),
            ];
        }
        "reference" => {
            request.conditioning = vec![Conditioning::Reference {
                image: image(),
                strength: Some(0.6),
            }]
        }
        // Inpainting carries its source image and the mask.
        "mask" => {
            request.conditioning = vec![
                Conditioning::Reference {
                    image: image(),
                    strength: Some(0.6),
                },
                Conditioning::Mask { image: image() },
            ]
        }
        "control" => {
            request.conditioning = vec![Conditioning::Control {
                image: image(),
                kind: ControlKind::Other("tile".into()),
                scale: Some(1.0),
            }]
        }
        "restore" => {
            return Ok(IosWork::Restore(TransformRequest {
                image: image(),
                seed: Some(16774),
                cancel: cancel.clone(),
                ..Default::default()
            }))
        }
        "depth" => {
            return Ok(IosWork::Depth(runtime_ios::iris_depth::DepthRequest {
                image: image(),
                cancel: cancel.clone(),
                ..Default::default()
            }))
        }
        other => {
            return Err(AdmissionError::Refused(format!(
                "{other}: no request carrier for this registered task"
            )))
        }
    }
    Ok(IosWork::Generate(request))
}

fn snapshot(root: &Path, c: &Case) -> Result<LoadSpec, String> {
    let path = root.join(&c.route).join(&c.tier);
    if !path.is_dir() {
        return Err(format!(
            "missing provisioned snapshot {}; not admission/refusal evidence",
            path.display()
        ));
    }
    let source = WeightsSource::Dir(path.clone());
    let mut spec = match c.provider.as_str() {
        "sdxl" => {
            runtime_ios::media::sdxl::sdxl_load_spec(&c.route, source).map_err(|e| e.to_string())?
        }
        "anima_turbo" => runtime_ios::media::anima_turbo_load_spec(source),
        id if id.starts_with("iris_") => {
            LoadSpec::new(source).with_load_shape(LoadShape::EagerMaterialization)
        }
        _ => LoadSpec::new(source)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization),
    };
    spec.precision = tier(c).precision;
    spec.quantize = tier(c).quant;
    // Relative caller-provisioned components only. No download, path guessing, or unbound
    // substitution. Preserve lexical source paths and let provider inventory checks bind them.
    let manifest = path.join("components.json");
    if manifest.exists() {
        if std::fs::metadata(&manifest)
            .map_err(|e| e.to_string())?
            .len()
            > 65536
        {
            return Err("component manifest exceeds 64 KiB".into());
        }
        use std::io::Read;
        let mut data = Vec::new();
        std::fs::File::open(&manifest)
            .map_err(|e| e.to_string())?
            .take(65537)
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
        if data.len() > 65536 {
            return Err("component manifest exceeds 64 KiB".into());
        }
        let components: std::collections::BTreeMap<String, String> =
            serde_json::from_slice(&data).map_err(|e| e.to_string())?;
        for (key, relative) in components {
            let rel = Path::new(&relative);
            if rel.is_absolute()
                || rel
                    .components()
                    .any(|p| !matches!(p, std::path::Component::Normal(_)))
            {
                return Err("component paths must be confined relative paths".into());
            }
            spec.components
                .insert(key, WeightsSource::Dir(path.join(rel)));
        }
    }
    Ok(spec)
}

struct DeviceProvider<'a> {
    registry: &'a ProviderRegistry,
    device_allowed: bool,
    artifact: Value,
    refusal_kind: Option<&'static str>,
    cancel: CancelFlag,
    deadline: Instant,
    root: PathBuf,
    candidate: Option<runtime_ios::media::IosCandidate>,
    /// The memory-relevant evidence key of this case's actual work on this artifact/device.
    evidence_key: Option<runtime_ios::media::evidence::EvidenceKey>,
    /// A bounded capture permit (diagnostic runs only).
    permit: Option<(ProviderRegistry, runtime_ios::media::Admission)>,
    /// The typed pre-construction quantum refusal, when a capture stopped at one.
    refused_quantum: Option<Value>,
    refusal_only: bool,
}
impl DeviceProvider<'_> {
    fn prepare(&mut self, c: &Case) -> Result<(), AdmissionError> {
        let spec = snapshot(&self.root, c)?;
        let contract = self
            .registry
            .memory_strategy_contract(&c.provider, &spec)
            .map_err(|e| e.to_string())?
            .ok_or("no provider memory contract")?;
        let strategy = MemoryStrategy::ALL[usize::from(c.rung)];
        if contract
            .capability(strategy)
            .is_none_or(|cap| !matches!(cap.support, MemoryStrategySupport::Implemented))
        {
            self.refusal_kind = Some("strategy");
            return Err(AdmissionError::Refused(format!(
                "provider does not implement selected rung {}",
                c.rung
            )));
        }
        let selection = contract
            .representative_selection(strategy, tier(c), false)
            .map_err(|e| {
                self.refusal_kind = Some("strategy");
                AdmissionError::Refused(e.to_string())
            })?;
        let work = work(c, &self.cancel)?;
        let candidate = runtime_ios::media::build_candidate(
            self.registry,
            &c.provider,
            spec,
            Some(selection),
            strategy,
            work,
        )
        .map_err(|e| e.to_string())?;
        // The memory-relevant key (path-free layout digest, mode, geometry, execution, token
        // window). A malformed artifact is a failure here, before any evidence question.
        let key = runtime_ios::media::evidence_key(
            self.registry,
            &candidate,
            &runtime_ios::device_machine(),
        )
        .map_err(|e| classify(&mut self.refusal_kind, e))?;
        self.artifact = json!({"layout_sha256": key.layout_sha256});
        self.evidence_key = Some(key);
        self.candidate = Some(candidate);
        Ok(())
    }

    fn run_work(
        &mut self,
        c: &Case,
        load: Option<runtime_ios::media::Admission>,
        registry: &ProviderRegistry,
    ) -> Result<(), ExecuteError> {
        use runtime_ios::media::{capture_capacity, IosWork};
        let candidate = self.candidate.as_ref().ok_or("no exact work")?;
        let docs = super::dirs_documents().ok_or("no Documents directory")?;
        // A stale record from an earlier call must never be read as this run's refusal.
        let _ = runtime_ios::media::ios_quantum::take_refusal();
        let outcome = (|| -> Result<(), String> {
            match (&candidate.work, load) {
                (IosWork::Generate(request), permit) => {
                    let generator: Box<dyn Generator> = match permit {
                        Some(permit) => Box::new(
                            permit
                                .load_generator(registry, capture_capacity)
                                .map_err(|e| e.to_string())?,
                        ),
                        None => {
                            let one = std::slice::from_ref(candidate);
                            let loaded = match c.provider.as_str() {
                                "sdxl" => runtime_ios::media::load_sdxl(one),
                                "anima_turbo" => runtime_ios::media::load_anima_turbo(one),
                                "iris_3b" => runtime_ios::media::load_iris(one),
                                _ => runtime_ios::media::load_sana(one),
                            };
                            Box::new(loaded.map_err(|e| e.to_string())?.1)
                        }
                    };
                    generator.validate(request).map_err(|e| e.to_string())?;
                    let GenerationOutput::Images(images) = generator
                        .generate(request, &mut |_| {})
                        .map_err(|e| e.to_string())?
                    else {
                        return Err("expected image output".into());
                    };
                    let expected = request.prompt_batch.len().max(1);
                    if images.len() != expected {
                        return Err("wrong output count".into());
                    }
                    support::persist_images(&docs, c, &images, Some(c.edge))
                }
                (IosWork::Restore(request), permit) => {
                    let restorer = match permit {
                        Some(permit) => permit
                            .load_restorer(registry, capture_capacity)
                            .map_err(|e| e.to_string())?,
                        None => {
                            runtime_ios::media::load_iris_restore(std::slice::from_ref(candidate))
                                .map_err(|e| e.to_string())?
                                .1
                        }
                    };
                    let image = restorer
                        .apply(request, &mut |_| {})
                        .map_err(|e| e.to_string())?;
                    support::persist_images(&docs, c, &[image], None)
                }
                (IosWork::Depth(request), permit) => {
                    use runtime_ios::iris_depth::IrisDepthEstimator;
                    let depth = match permit {
                        Some(permit) => permit
                            .load_depth(capture_capacity)
                            .map_err(|e| e.to_string())?,
                        None => {
                            runtime_ios::media::load_iris_depth(std::slice::from_ref(candidate))
                                .map_err(|e| e.to_string())?
                                .1
                        }
                    };
                    let output = depth
                        .estimate(request, &mut |_| {})
                        .map_err(|e| e.to_string())?;
                    let map = &output.map;
                    if (map.width, map.height) != (c.edge, c.edge)
                        || map.values.iter().any(|v| !v.is_finite())
                    {
                        return Err("depth map has the wrong geometry or non-finite values".into());
                    }
                    let (lo, hi) = map
                        .values
                        .iter()
                        .fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
                    let pixels = map
                        .values
                        .iter()
                        .flat_map(|v| {
                            let g = (((v - lo) / (hi - lo).max(f32::EPSILON)) * 255.0) as u8;
                            [g, g, g]
                        })
                        .collect();
                    let image = Image {
                        width: map.width,
                        height: map.height,
                        pixels,
                    };
                    support::persist_images(&docs, c, &[image], Some(c.edge))
                }
            }
        })();
        outcome.map_err(
            |error| match runtime_ios::media::ios_quantum::take_refusal() {
                // Typed by the quantum mechanism, never by the message.
                Some(refusal) => {
                    self.refusal_kind = Some("capacity");
                    self.refused_quantum = Some(json!({
                        "label": refusal.label,
                        "scratch_bytes": refusal.scratch_bytes,
                        "reserve_bytes": refusal.reserve_bytes,
                        "available_bytes": refusal.available_bytes,
                        "edge": c.edge,
                    }));
                    ExecuteError::QuantumRefused(error)
                }
                None => ExecuteError::Failure(error),
            },
        )
    }
}
impl Provider for DeviceProvider<'_> {
    fn admit(&mut self, c: &Case, diagnostic: bool) -> Result<(), AdmissionError> {
        if !self.device_allowed {
            return Err("campaign execution requires a real iOS device; host/simulator are listing/build only".into());
        }
        self.prepare(c)?;
        let candidate = self.candidate.as_ref().ok_or("no exact work")?;
        if diagnostic {
            // CAPTURE: labelled, bounded per quantum, no evidence row consulted or created.
            let permit = runtime_ios::media::capture_permit(candidate)
                .map_err(|e| classify(&mut self.refusal_kind, e))?;
            self.permit = Some(permit);
            return Ok(());
        }
        // ADMITTED: the same single seam the facades use, against the reviewed table, before load.
        let live = super::available_memory_mib().map(|m| (m * 1048576.0) as u64);
        let budget = runtime_ios::media::ios_admission::capacity_budget(
            u64::from(runtime_ios::IOS_APP_MEMORY_CAP_MIB) << 20,
            0,
            live,
        );
        let admitted = runtime_ios::media::ios_admission::admit(
            self.registry,
            std::slice::from_ref(candidate),
            &runtime_ios::device_machine(),
            budget,
        )
        .map_err(|e| classify(&mut self.refusal_kind, e))?;
        if admitted.index() != 0 {
            return Err("admission silently changed the selected case".into());
        }
        Ok(())
    }
    fn execute(&mut self, c: &Case) -> Result<(), ExecuteError> {
        if self.refusal_only {
            return Err("next-up was admitted but refusal-only probe forbids weight loading; no refusal proven".into());
        }
        if self.cancel.is_cancelled() || Instant::now() >= self.deadline {
            return Err("cooperative case deadline reached before loading".into());
        }
        match self.permit.take() {
            Some((registry, permit)) => self.run_work(c, Some(permit), &registry),
            None => {
                let registry = self.registry;
                self.run_work(c, None, registry)
            }
        }
    }
}

// Darwin task_vm_info's stable prefix: current phys_footprint at words 36–37 and the exact
// ledger lifetime peak at 42–43 (iOS 26.5 SDK). A shorter kernel reply leaves peak unavailable.
// Neither is RSS or MLX allocator accounting. Unavailable readings stay null, never zero evidence.
fn phys_footprint() -> Option<(u64, Option<u64>)> {
    #[cfg(target_vendor = "apple")]
    {
        support::read_footprint(|words, count| {
            #[allow(deprecated)]
            unsafe {
                libc::task_info(libc::mach_task_self(), 22, words.as_mut_ptr().cast(), count)
            }
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        None
    }
}

pub fn report(value: &str, diagnostic: bool) -> String {
    let started = Instant::now();
    let result = (|| -> Result<Value, String> {
        let case = Case::parse(value)?;
        let catalog = runtime_ios::catalog().map_err(|e| e.to_string())?;
        if !campaign::enumerate(&surfaces(catalog.media()))
            .iter()
            .any(|c| {
                c.provider == case.provider
                    && c.route == case.route
                    && c.tier == case.tier
                    && c.mode == case.mode
                    && c.rung == case.rung
            })
        {
            return Err("case not in the actual supported catalog".into());
        }
        if !cfg!(all(target_os = "ios", not(target_abi = "sim"))) {
            return Err("campaign execution requires a real iOS device; host/simulator are listing/build only".into());
        }
        let root = super::dirs_documents()
            .ok_or("no Documents directory")?
            .join("campaign");
        let mut adapter = DeviceProvider {
            registry: catalog.media(),
            device_allowed: cfg!(all(target_os = "ios", not(target_abi = "sim"))),
            artifact: Value::Null,
            refusal_kind: None,
            cancel: CancelFlag::new(),
            deadline: started + Duration::from_secs(case_timeout()?),
            root,
            candidate: None,
            evidence_key: None,
            permit: None,
            refused_quantum: None,
            refusal_only: std::env::var("IOS_CAMPAIGN_REFUSAL_ONLY").as_deref() == Ok("1"),
        };
        // Bind before any source inspection/load. This does not create footprint evidence.
        runtime_ios::bound_mlx_to_platform_limits()
            .ok_or("platform allocator limits could not be bound; refused before load")?;
        if let Some(docs) = super::dirs_documents() {
            std::fs::write(docs.join(super::RENDERS_MANIFEST), "").map_err(|e| e.to_string())?;
        }
        mlx_rs::memory::reset_peak_memory();
        let extrema = Arc::new(Mutex::new((
            0usize,
            0usize,
            0usize,
            None::<u64>,
            None::<f64>,
            None::<u64>,
        )));
        let (stop_tx, stop_rx) = std::sync::mpsc::channel();
        let samples = Arc::clone(&extrema);
        let sample = move || {
            if let Ok(mut e) = samples.lock() {
                e.0 = e.0.max(mlx_rs::memory::get_active_memory());
                e.1 = e.1.max(mlx_rs::memory::get_cache_memory());
                e.2 = e
                    .2
                    .max(mlx_rs::memory::get_active_memory() + mlx_rs::memory::get_cache_memory());
                if let Some((current, peak)) = phys_footprint() {
                    e.3 = Some(e.3.unwrap_or(0).max(current));
                    if let Some(peak) = peak {
                        e.5 = Some(e.5.unwrap_or(0).max(peak));
                    }
                }
                if let Some(a) = super::available_memory_mib() {
                    e.4 = Some(e.4.unwrap_or(a).min(a));
                }
            }
        };
        sample();
        let thread = std::thread::spawn(move || loop {
            sample();
            if !matches!(
                stop_rx.recv_timeout(Duration::from_millis(20)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ) {
                sample();
                break;
            }
        });
        let deadline = Deadline::start(
            adapter.cancel.clone(),
            adapter.deadline.saturating_duration_since(Instant::now()),
        );
        let (mut status, mut reason) = campaign::run(&mut adapter, &case, diagnostic);
        if adapter.cancel.is_cancelled() {
            status = Status::DiagnosticFailure;
            reason = "device cooperative deadline cancelled the case".into();
        }
        drop(deadline);
        let _ = stop_tx.send(());
        let _ = thread.join();
        let e = extrema.lock().map_err(|_| "sampler poisoned")?;
        let (status, reason) = if status.production_success()
            && (e.3.is_none() || e.4.is_none() || e.5.is_none())
        {
            (Status::DiagnosticFailure, "completed but process footprint/headroom telemetry unavailable; not production evidence".into())
        } else {
            (status, reason)
        };
        Ok(
            json!({ "schema":1, "case":case.key(), "status":status.label(), "production_success":status.production_success(), "reason":reason,
            "elapsed_seconds":started.elapsed().as_secs_f64(), "mlx_active_peak_bytes":e.0, "mlx_cache_peak_bytes":e.1,
            "mlx_active_cache_peak_bytes":e.2, "mlx_peak_bytes":mlx_rs::memory::get_peak_memory(), "phys_footprint_peak_bytes":e.3,
            "min_headroom_mib":e.4, "device":runtime_ios::device_machine(), "cap_mib":runtime_ios::IOS_APP_MEMORY_CAP_MIB,
            "cap_validated":false, "diagnostic":diagnostic,
            "kernel_lifetime_phys_footprint_peak_bytes":e.5, "artifact_identity":adapter.artifact,
            "evidence_class": if diagnostic { "capture" } else { "admitted" }, "evidence_key":adapter.evidence_key, "refused_quantum":adapter.refused_quantum,
            "selected_parameters":adapter.candidate.as_ref().map(|c| &c.context).map(|c| json!({"strategy":format!("{:?}",c.selection.strategy),"stage_residency":c.selection.parameters.stage_residency,"decode_tile_edge":c.selection.parameters.decode_tile_edge,"decode_overlap":c.selection.parameters.decode_overlap,"attention_chunk_size":c.selection.parameters.attention_chunk_size,"transformer_window_size":c.selection.parameters.transformer_window_size,"transformer_window_component":format!("{:?}",c.selection.parameters.transformer_window_component),"numeric_tier":format!("{:?}",c.selection.tier)})),
            "evidence_revision":adapter.candidate.as_ref().map(|c| &c.context.evidence_revision),
            "refusal_kind":adapter.refusal_kind, "refusal_only":adapter.refusal_only,
            "expected_render_count":adapter.candidate.as_ref().map(|c| match &c.work { runtime_ios::media::IosWork::Generate(r) => r.prompt_batch.len().max(1), _ => 1 }),
            "build_sha":option_env!("IOS_CAMPAIGN_BUILD_SHA").unwrap_or("unknown"),
            "entitlement":std::env::var("IOS_SMOKE_INCREASED_MEMORY_LIMIT").ok(),
            "ios_version":std::env::var("IOS_CAMPAIGN_IOS_VERSION").ok() }),
        )
    })();
    let mut row = result.unwrap_or_else(|e| json!({"schema":1,"case":value,"status":Status::DiagnosticFailure.label(),"production_success":false,"reason":e,"elapsed_seconds":started.elapsed().as_secs_f64(),"mlx_active_peak_bytes":null,"mlx_cache_peak_bytes":null,"mlx_active_cache_peak_bytes":null,"mlx_peak_bytes":null,"phys_footprint_peak_bytes":null,"min_headroom_mib":null,"diagnostic":diagnostic}));
    row["run_id"] = json!(std::env::var("IOS_CAMPAIGN_RUN_ID").ok());
    row["build_sha"] = json!(option_env!("IOS_CAMPAIGN_BUILD_SHA").unwrap_or("unknown"));
    row["entitlement"] = json!(std::env::var("IOS_SMOKE_INCREASED_MEMORY_LIMIT").ok());
    row["ios_version"] = json!(std::env::var("IOS_CAMPAIGN_IOS_VERSION").ok());
    row["refusal_only"] = json!(std::env::var("IOS_CAMPAIGN_REFUSAL_ONLY").as_deref() == Ok("1"));
    format!(
        "SMOKE: {}\nCAMPAIGN_JSON {}\n",
        if row["production_success"] == true {
            "PASS"
        } else {
            "FAIL"
        },
        row
    )
}
