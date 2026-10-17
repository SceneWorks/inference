//! Catalog-driven device campaign adapter. Enumeration is weights-free. Execution is never
//! authorized by a host peak, launch headroom, a directory name, or the diagnostic flag.
use super::campaign::{self, AdmissionError, Case, Provider, Status, Surface};
use runtime_ios::gen_core::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::result::Result;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
        let data = std::fs::read(&manifest).map_err(|e| e.to_string())?;
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
    root: PathBuf,
    spec: Option<LoadSpec>,
    context: Option<MemoryRunContext>,
    request: Option<GenerationRequest>,
    refusal_only: bool,
    sdxl: Option<runtime_ios::media::sdxl::SdxlCandidate>,
}
impl DeviceProvider<'_> {
    fn prepare(&mut self, c: &Case) -> Result<(), String> {
        let spec = snapshot(&self.root, c)?;
        let contract = self
            .registry
            .memory_strategy_contract(&c.provider, &spec)
            .map_err(|e| e.to_string())?
            .ok_or("no provider memory contract")?;
        let calibration = contract
            .calibration
            .as_ref()
            .ok_or("no provider calibration identity")?;
        let selection = contract
            .representative_selection(MemoryStrategy::ALL[usize::from(c.rung)], tier(c), false)
            .map_err(|e| e.to_string())?;
        let mut request = GenerationRequest {
            prompt: "A red ceramic teapot on a wooden table in daylight".into(),
            width: c.edge,
            height: c.edge,
            seed: Some(16774),
            count: 1,
            memory: contract.generation_memory(&selection),
            ..Default::default()
        };
        if c.mode == "prompt_batch" {
            request.prompt_batch = vec![std::mem::take(&mut request.prompt)];
        }
        // Conditioning/task cases are retained and refused until device-bound evidence covers
        // their exact carrier. Never replace an edit/depth/restoration request with text-to-image.
        let mode = if c.mode == "text" || c.mode == "prompt_batch" {
            MemoryMode::TextToImage
        } else {
            MemoryMode::Other(c.mode.clone())
        };
        let context = MemoryRunContext {
            selection,
            optimization_authority: MemoryOptimizationAuthority::Calibrated,
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: spec.load_shape,
            mode,
            has_reference: false,
            use_pid: false,
            has_phases: true,
            geometry: MemoryGeometry {
                width: c.edge,
                height: c.edge,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: runtime_ios::media::ios_memory_budget(),
            // Unknown is NOT zero. No fabricated estimate is ever supplied to admission.
            predicted_peak_bytes: u64::MAX,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "no-device-capture".into(),
        };
        self.spec = Some(spec);
        self.context = Some(context);
        self.request = Some(request);
        Ok(())
    }
}
impl Provider for DeviceProvider<'_> {
    fn admit(&mut self, c: &Case, diagnostic: bool) -> Result<(), AdmissionError> {
        if !cfg!(all(target_os = "ios", not(target_abi = "sim"))) {
            return Err("campaign execution requires a real iOS device; host/simulator are listing/build only".into());
        }
        self.prepare(c)?;
        if diagnostic {
            return Err("UNADMITTED diagnostic refused: no independently bounded primitive permit for this case; --diagnostic is not permission to load weights or raise the cap".into());
        }
        if c.mode != "text" && c.mode != "prompt_batch" {
            return Err(AdmissionError::Refused(format!("{} remains registered; no artifact/device/carrier-bound footprint authorizes this mode before load", c.mode)));
        }
        let spec = self.spec.as_ref().unwrap();
        let context = self.context.as_mut().unwrap();
        let request = self.request.as_ref().unwrap();
        if c.provider == "sdxl" {
            self.sdxl = Some(runtime_ios::media::sdxl::SdxlCandidate {
                spec: spec.clone(),
                context: context.clone(),
                execution: request.memory.unwrap_or_default(),
                request: request.clone(),
            });
            // This API selects and seals the actual request/source, rechecks live capacity before
            // load and every deferred generation. No raw registry shortcut.
            let live = super::available_memory_mib().map(|m| (m * 1048576.0) as u64);
            let budget = runtime_ios::media::sdxl::capacity_budget(
                u64::from(runtime_ios::IOS_APP_MEMORY_CAP_MIB) << 20,
                0,
                live,
            );
            let admitted = runtime_ios::media::sdxl::select_sdxl_candidate(
                self.registry,
                std::slice::from_ref(self.sdxl.as_ref().unwrap()),
                &runtime_ios::device_machine(),
                budget,
            )
            .map_err(|e| AdmissionError::Refused(e.to_string()))?;
            if admitted.index() != 0 {
                return Err("admission silently changed the selected case".into());
            }
            return Ok(());
        }
        if c.provider.starts_with("sana_") {
            let memory = request.memory.unwrap_or_default();
            if !memory.tile_vae_decode {
                return Err(AdmissionError::Refused(
                    "SANA whole-image decode has no applicable device footprint".into(),
                ));
            }
            let tile = memory
                .decode_tile_edge
                .ok_or("selected decode has no explicit tile edge")?;
            let estimate = runtime_ios::admit_sana_snapshot(&c.provider, spec, c.edge, tile)
                .map_err(|e| AdmissionError::Refused(e.to_string()))?;
            context.predicted_peak_bytes = u64::from(estimate) << 20;
            context.evidence_revision = "artifact-bound-sana".into();
            let contract = self
                .registry
                .memory_strategy_contract(&c.provider, spec)
                .map_err(|e| e.to_string())?
                .ok_or("missing contract")?;
            let registration = self
                .registry
                .memory_strategy_registrations()
                .find(|r| r.provider_id == c.provider)
                .ok_or("missing registered safety check")?;
            match (registration.safety_check)(spec, &contract, context) {
                MemorySafetyDecision::Accept => (),
                MemorySafetyDecision::Reject { reason } => {
                    return Err(AdmissionError::Refused(reason))
                }
            }
            return Ok(());
        }
        // Anima host allocator peaks are not device phys_footprint; Iris has no iPhone capture.
        // The public raw catalog is a diagnostic surface, never an implicit admitted facade.
        Err(AdmissionError::Refused(format!("{} registered; no artifact/device-bound total footprint and admitted task facade; refused before weight load", c.provider)))
    }
    fn execute(&mut self, c: &Case) -> Result<(), String> {
        if self.refusal_only {
            return Err("next-up was admitted but refusal-only probe forbids weight loading; no refusal proven".into());
        }
        let request = self.request.as_ref().ok_or("no exact request")?;
        let generator: Box<dyn Generator> = if let Some(candidate) = &self.sdxl {
            let (_, bound) = runtime_ios::media::load_sdxl(std::slice::from_ref(candidate))
                .map_err(|e| e.to_string())?;
            Box::new(bound)
        } else {
            let spec = self.spec.as_ref().ok_or("missing admitted source")?;
            let tile = request
                .memory
                .and_then(|m| m.decode_tile_edge)
                .ok_or("missing admitted tile")?;
            runtime_ios::media::load_sana(&c.provider, spec, c.edge, tile)
                .map_err(|e| e.to_string())?
        };
        generator.validate(request).map_err(|e| e.to_string())?;
        let result = generator
            .generate(request, &mut |_| {})
            .map_err(|e| e.to_string())?;
        let GenerationOutput::Images(images) = result else {
            return Err("expected image output".into());
        };
        if images.len() != 1 {
            return Err("wrong output count".into());
        }
        let image = &images[0];
        if image.width != c.edge
            || image.height != c.edge
            || image.pixels.len() != (c.edge as usize).pow(2) * 3
            || image.pixels.iter().min() == image.pixels.iter().max()
        {
            return Err("wrong geometry or constant image".into());
        }
        let output = super::dirs_documents()
            .ok_or("no Documents directory")?
            .join("campaign-render.png");
        image::save_buffer_with_format(
            output,
            &image.pixels,
            image.width,
            image.height,
            image::ColorType::Rgb8,
            image::ImageFormat::Png,
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}

// TASK_VM_INFO_REV1: the stable Darwin 4-byte-aligned prefix through phys_footprint. Requesting
// only this prefix avoids assuming a newer kernel's ledger layout. Unlike RSS/MLX this is the
// process quantity charged by jetsam. Unavailable readings stay null, never zero evidence.
fn phys_footprint() -> Option<u64> {
    #[cfg(target_vendor = "apple")]
    {
        let mut words = [0u32; 38];
        let mut count = words.len() as u32;
        #[allow(deprecated)]
        let status = unsafe {
            libc::task_info(
                libc::mach_task_self(),
                22,
                words.as_mut_ptr().cast(),
                &mut count,
            )
        };
        if status == 0 && count >= 38 {
            return Some(u64::from(words[36]) | (u64::from(words[37]) << 32));
        }
    }
    None
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
            root,
            spec: None,
            context: None,
            request: None,
            refusal_only: std::env::var("IOS_CAMPAIGN_REFUSAL_ONLY").as_deref() == Ok("1"),
            sdxl: None,
        };
        // Bind before any source inspection/load. This does not create footprint evidence.
        let _ = runtime_ios::bound_mlx_to_platform_limits();
        mlx_rs::memory::reset_peak_memory();
        let extrema = Arc::new(Mutex::new((
            0usize,
            0usize,
            0usize,
            None::<u64>,
            None::<f64>,
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
                if let Some(p) = phys_footprint() {
                    e.3 = Some(e.3.unwrap_or(0).max(p));
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
        let (status, reason) = campaign::run(&mut adapter, &case, diagnostic);
        let _ = stop_tx.send(());
        let _ = thread.join();
        let e = extrema.lock().map_err(|_| "sampler poisoned")?;
        let (status, reason) = if status.production_success() && (e.3.is_none() || e.4.is_none()) {
            (Status::DiagnosticFailure, "completed but process footprint/headroom telemetry unavailable; not production evidence".into())
        } else {
            (status, reason)
        };
        Ok(
            json!({ "schema":1, "case":case.key(), "status":status.label(), "production_success":status.production_success(), "reason":reason,
            "elapsed_seconds":started.elapsed().as_secs_f64(), "mlx_active_peak_bytes":e.0, "mlx_cache_peak_bytes":e.1,
            "mlx_active_cache_peak_bytes":e.2, "mlx_peak_bytes":mlx_rs::memory::get_peak_memory(), "phys_footprint_peak_bytes":e.3,
            "min_headroom_mib":e.4, "device":runtime_ios::device_machine(), "cap_mib":runtime_ios::IOS_APP_MEMORY_CAP_MIB,
            "cap_validated":false, "diagnostic":diagnostic }),
        )
    })();
    let row = result.unwrap_or_else(|e| json!({"schema":1,"case":value,"status":Status::DiagnosticFailure.label(),"production_success":false,"reason":e,"elapsed_seconds":started.elapsed().as_secs_f64(),"mlx_active_peak_bytes":null,"mlx_cache_peak_bytes":null,"mlx_active_cache_peak_bytes":null,"mlx_peak_bytes":null,"phys_footprint_peak_bytes":null,"min_headroom_mib":null,"diagnostic":diagnostic}));
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
