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
#[path = "campaign_primitive.rs"]
mod primitive;

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
        let conditioning = d
            .capabilities
            .conditioning
            .iter()
            .map(|m| format!("{m:?}").to_ascii_lowercase());
        // A control-REQUIRED route (the catalog's explicit route fact, e.g. Z-Image Turbo's
        // Fun-Controlnet-Union route) has no plain text mode: every one of its modes carries the
        // control image ([`work`]), and each is exercised once per control kind the route accepts
        // (`control_pose`, `reference_canny`, ...).
        let mut modes = if runtime_ios::media::control_required(d.id) {
            let kinds = accepted_kinds(registry, d.id);
            conditioning
                .flat_map(|m| {
                    kinds
                        .iter()
                        .map(move |k| format!("{m}_{}", kind_token(k)))
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            std::iter::once("text".to_string())
                .chain(conditioning)
                .collect::<Vec<_>>()
        };
        if d.capabilities.supports_prompt_batch {
            modes.push("prompt_batch".into());
        }
        // SDXL's IP-Adapter is a distinct memory route (an image encoder + per-block IP keys),
        // reached with a reference image and a provisioned `ip_adapter` component.
        if d.id == "sdxl" {
            modes.push("ip_adapter".into());
        }
        let routes: Vec<String> = if d.id == "sdxl" {
            runtime_ios::media::sdxl::SDXL_FAMILY_ROUTES
                .iter()
                .map(|s| (*s).into())
                .collect()
        } else {
            vec![d.id.into()]
        };
        out.push(Surface {
            selections: selections(registry, d.id, &routes[0]),
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
            selections: selections(registry, d.id, d.id),
        });
    }
    // Composed provider-specific tasks are not generators. Discover them through their actual
    // memory registrations, not a second hard-coded catalog. Unknown tasks remain explicit cells.
    for r in registry.memory_strategy_registrations() {
        if !out.iter().any(|s| s.provider == r.provider_id) {
            out.push(Surface {
                selections: selections(registry, r.provider_id, r.provider_id),
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

/// The weights-free campaign listing (schema 2): every planned ANCHOR case with its purpose and
/// repeat count ([`campaign::enumerate`]), derived from the actual runtime catalog.
pub fn listing() -> Result<String, String> {
    let catalog = runtime_ios::catalog().map_err(|e| e.to_string())?;
    let planned = campaign::enumerate(&surfaces(catalog.media()));
    Ok(json!({"schema": 2,
        "cases": planned.iter().map(|p| json!({"case": p.case.key(), "purpose": p.purpose.label(), "repeats": p.repeats})).collect::<Vec<_>>(),
        "dependencies": [{"provider":"supra2", "epic":24212, "available": planned.iter().any(|p| p.case.provider.contains("supra")), "reason":"requires native engine and iOS catalog registration; absence is not a passing cell"}]
    }).to_string())
}

/// The campaign token of a selection's parameters: `t{tile}`, `o{overlap}`, `c{chunk}`,
/// `w{window}` and the window scope (`dit` / `te` / `both`), in that order, `_`-joined, and `s`
/// first when it adds phase eviction; [`campaign::BASE_SELECTION`] when it has none.
pub fn selection_token(p: &MemoryStrategyParameters) -> String {
    let mut parts = Vec::new();
    if p.stage_residency == Some(true) {
        parts.push("s".to_string());
    }
    if let Some(t) = p.decode_tile_edge {
        parts.push(format!("t{t}"));
    }
    if let Some(o) = p.decode_overlap {
        parts.push(format!("o{o}"));
    }
    if let Some(c) = p.attention_chunk_size {
        parts.push(format!("c{c}"));
    }
    if let Some(w) = p.transformer_window_size {
        parts.push(format!("w{w}"));
    }
    if let Some(scope) = p.transformer_window_component {
        parts.push(
            match scope {
                TransformerComponent::Dit => "dit",
                TransformerComponent::TextEncoder => "te",
                TransformerComponent::Both => "both",
            }
            .into(),
        );
    }
    if parts.is_empty() {
        campaign::BASE_SELECTION.into()
    } else {
        parts.join("_")
    }
}

/// The provider's declared selection space per `(tier, rung)` as quality-first tokens, from its
/// weights-free contract surface at the load shape the campaign actually uses ([`load_spec`]).
fn selections(
    registry: &ProviderRegistry,
    provider: &str,
    route: &str,
) -> std::collections::BTreeMap<(String, u8), Vec<String>> {
    let mut out = std::collections::BTreeMap::new();
    let Ok(contracts) = registry.memory_contract_surfaces() else {
        return out;
    };
    for tier_name in ["q4", "q8", "bf16"] {
        let probe = Case {
            provider: provider.into(),
            route: route.into(),
            tier: tier_name.into(),
            edge: 512,
            mode: String::new(),
            rung: 0,
            selection: campaign::BASE_SELECTION.into(),
            steps: 0,
        };
        let Ok(spec) = load_spec(&probe, WeightsSource::Dir(PathBuf::new())) else {
            continue;
        };
        let Some(surface) = contracts.iter().find(|s| {
            s.contract.provider_id == provider
                && s.selector.offload_policy == spec.offload_policy
                && s.selector.load_shape == spec.load_shape
                && match s.selector.tier {
                    MemoryContractSurfaceTier::Q4 => tier_name == "q4",
                    MemoryContractSurfaceTier::Q8 => tier_name == "q8",
                    MemoryContractSurfaceTier::Bf16 => tier_name == "bf16",
                    _ => false,
                }
        }) else {
            continue;
        };
        for (rung, strategy) in MemoryStrategy::ALL.into_iter().enumerate().take(5) {
            if let Ok(space) = surface
                .contract
                .selection_space(strategy, tier_of(tier_name), false)
            {
                out.insert(
                    (tier_name.to_string(), rung as u8),
                    space
                        .iter()
                        .map(|s| selection_token(&s.parameters))
                        .collect(),
                );
            }
        }
    }
    out
}

fn tier(case: &Case) -> MemoryNumericTier {
    tier_of(&case.tier)
}

fn tier_of(tier: &str) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: Precision::Bf16,
        quant: match tier {
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

/// The control kinds a route declares it accepts (empty: no declared policy).
fn accepted_kinds(registry: &ProviderRegistry, provider: &str) -> Vec<ControlKind> {
    match registry
        .generators()
        .map(|r| (r.descriptor)())
        .find(|d| d.id == provider)
        .and_then(|d| d.control_kinds)
    {
        Some(AcceptedControlKinds::Only(kinds)) => kinds.to_vec(),
        _ => Vec::new(),
    }
}

/// A control kind's mode-suffix token (`pose`, `canny`, `depth`, ...).
fn kind_token(kind: &ControlKind) -> String {
    match kind {
        ControlKind::Other(name) => name.to_ascii_lowercase(),
        other => format!("{other:?}").to_ascii_lowercase(),
    }
}

/// A case's mode and, on a control-required route, the control kind its `{mode}_{kind}` suffix
/// names (a kind the route does not accept is a typed mode refusal).
fn mode_and_kind(
    c: &Case,
    registry: &ProviderRegistry,
) -> Result<(String, Option<ControlKind>), AdmissionError> {
    if !runtime_ios::media::control_required(&c.provider) {
        return Ok((c.mode.clone(), None));
    }
    accepted_kinds(registry, &c.provider)
        .into_iter()
        .find_map(|k| {
            c.mode
                .strip_suffix(&format!("_{}", kind_token(&k)))
                .map(|m| (m.to_string(), Some(k)))
        })
        .ok_or_else(|| {
            AdmissionError::Refused(format!(
                "{}: mode {} names no control kind the route accepts",
                c.provider, c.mode
            ))
        })
}

/// The actual work of a case's mode. Every registered mode has a carrier; a mode the provider
/// does not serve is refused by admission with a typed `mode` kind, never dropped here. A
/// control-required route carries its control image, of the kind its mode names, in every mode
/// (`reference_<kind>` adds an img2img init); SDXL's optional ControlNet is its tile model.
fn work(
    c: &Case,
    cancel: &CancelFlag,
    registry: &ProviderRegistry,
) -> Result<runtime_ios::media::IosWork, AdmissionError> {
    use runtime_ios::media::IosWork;
    let (mode, required_kind) = mode_and_kind(c, registry)?;
    let image = || synthetic_image(c.edge);
    let mut request = GenerationRequest {
        prompt: prompt(&c.provider),
        width: c.edge,
        height: c.edge,
        seed: Some(16774),
        count: 1,
        cancel: cancel.clone(),
        // An anchor captures a short run ([`campaign::ANCHOR_STEPS`]); 0 keeps the provider's own.
        steps: (c.steps > 0).then_some(c.steps),
        ..Default::default()
    };
    match mode.as_str() {
        "text" => {}
        "prompt_batch" => {
            request.prompt_batch = vec![
                std::mem::take(&mut request.prompt),
                "A blue glass vase beside a window at sunset".into(),
            ];
        }
        "reference" | "ip_adapter" => {
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
        // SDXL's optional ControlNet: its provisioned tile model.
        "control" if required_kind.is_none() => {
            request.conditioning = vec![Conditioning::Control {
                image: image(),
                kind: ControlKind::Other("tile".into()),
                scale: Some(1.0),
            }]
        }
        // A control-required route's control-only mode: the control image below.
        "control" => {}
        "restore" => {
            return Ok(IosWork::Restore(TransformRequest {
                image: image(),
                seed: Some(16774),
                cancel: cancel.clone(),
                steps: (c.steps > 0).then_some(c.steps),
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
    if let Some(kind) = required_kind {
        request.conditioning.insert(
            0,
            Conditioning::Control {
                image: image(),
                kind,
                scale: Some(1.0),
            },
        );
    }
    Ok(IosWork::Generate(request))
}

/// The provisioned slot a case loads from: its route's, except Z-Image Turbo's control route, which
/// loads the `z_image_turbo` snapshot (the same base model) with that slot's `control` role.
fn slot_route(c: &Case) -> &str {
    if c.provider == runtime_ios::media::zimage::Z_IMAGE_TURBO_CONTROL_ID {
        runtime_ios::media::zimage::Z_IMAGE_TURBO_ID
    } else {
        &c.route
    }
}

/// Whether `value`'s campaign slot (`Documents/campaign/<slot route>/<tier>/`) is provisioned on
/// this device. A malformed case is "not provisioned".
#[cfg(feature = "zimage")]
pub fn slot_provisioned(value: &str) -> bool {
    let (Ok(c), Some(docs)) = (Case::parse(value), super::dirs_documents()) else {
        return false;
    };
    docs.join("campaign")
        .join(slot_route(&c))
        .join(&c.tier)
        .is_dir()
}

/// A case's load spec over `source`, before any provisioned component: the provider's iOS load
/// policy at the case's tier. The same spec selects the weights-free contract surface the campaign
/// enumerates selections from ([`selections`]), so enumeration and execution agree.
fn load_spec(c: &Case, source: WeightsSource) -> Result<LoadSpec, String> {
    let mut spec = match c.provider.as_str() {
        "sdxl" => {
            runtime_ios::media::sdxl::sdxl_load_spec(&c.route, source).map_err(|e| e.to_string())?
        }
        "anima_turbo" => runtime_ios::media::anima_turbo_load_spec(source),
        id if runtime_ios::media::zimage::is_z_image(id) => {
            runtime_ios::media::zimage::z_image_turbo_load_spec(source, None)
        }
        id if id.starts_with("iris_") => {
            LoadSpec::new(source).with_load_shape(LoadShape::EagerMaterialization)
        }
        _ => LoadSpec::new(source)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization),
    };
    spec.precision = tier(c).precision;
    spec.quantize = tier(c).quant;
    Ok(spec)
}

fn snapshot(root: &Path, c: &Case) -> Result<LoadSpec, String> {
    let path = root.join(slot_route(c)).join(&c.tier);
    if !path.is_dir() {
        return Err(format!(
            "missing provisioned snapshot {}; not admission/refusal evidence",
            path.display()
        ));
    }
    let mut spec = load_spec(c, WeightsSource::Dir(path.clone()))?;
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
            let target = path.join(rel);
            // A single-file artifact is a File source; a directory is a Dir source.
            let source = if target.is_file() {
                WeightsSource::File(target)
            } else {
                WeightsSource::Dir(target)
            };
            // The SDXL ControlNet and IP-Adapter are typed load-spec roles, not named components,
            // attached only for their own mode: a loaded IP-Adapter takes the reference image
            // (so an image-to-image case would silently become IP-Adapter), and either adds
            // weights the other routes do not run.
            // Z-Image Turbo's control route takes its ControlNet in every mode (it is the route).
            let control_route =
                c.mode == "control" || runtime_ios::media::control_required(&c.provider);
            match key.as_str() {
                "control" if control_route => spec.control = Some(source),
                "ip_adapter" if c.mode == "ip_adapter" => spec.ip_adapter = Some(source),
                "control" | "ip_adapter" => {}
                _ => {
                    spec.components.insert(key, source);
                }
            }
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
    /// An admitted case's price: demand, margin and its basis (measured anchor or derived).
    admitted: Option<Value>,
}
impl DeviceProvider<'_> {
    fn prepare(&mut self, c: &Case) -> Result<(), AdmissionError> {
        let spec = snapshot(&self.root, c)?;
        // A control / IP-Adapter case without its provisioned component would silently run as
        // another route: a provisioning failure, never admission or refusal evidence.
        if ((c.provider == "sdxl" && c.mode == "control")
            || runtime_ios::media::control_required(&c.provider))
            && spec.control.is_none()
        {
            return Err(format!("{} {} case needs a provisioned `control` component (components.json); not admission/refusal evidence", c.provider, c.mode).into());
        }
        if c.mode == "ip_adapter" && spec.ip_adapter.is_none() {
            return Err("IP-Adapter case needs a provisioned `ip_adapter` component (components.json); not admission/refusal evidence".into());
        }
        // Z-Image Turbo's stored-tier artifact policy runs before its contract reads any header (the
        // provider's own tier guard would otherwise surface it as an untyped backend failure).
        if runtime_ios::media::zimage::is_z_image(&c.provider) {
            match runtime_ios::media::zimage::check_artifact(&c.provider, &spec) {
                Ok(()) => {}
                Err(Error::Unsupported(reason)) => {
                    self.refusal_kind = Some("artifact");
                    return Err(AdmissionError::Refused(reason));
                }
                Err(e) => return Err(e.to_string().into()),
            }
        }
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
        // The cell's own selection, from the provider's declared space on THIS artifact's
        // contract (build_candidate re-validates it): never a substitute.
        let space = contract
            .selection_space(strategy, tier(c), false)
            .map_err(|e| {
                self.refusal_kind = Some("strategy");
                AdmissionError::Refused(e.to_string())
            })?;
        let Some(selection) = space
            .iter()
            .find(|s| selection_token(&s.parameters) == c.selection)
            .copied()
        else {
            self.refusal_kind = Some("strategy");
            return Err(AdmissionError::Refused(format!(
                "selection {} is not in {}'s declared space for rung {}: {:?}",
                c.selection,
                c.provider,
                c.rung,
                space
                    .iter()
                    .map(|s| selection_token(&s.parameters))
                    .collect::<Vec<_>>()
            )));
        };
        let work = work(c, &self.cancel, self.registry).map_err(|e| {
            if matches!(e, AdmissionError::Refused(_)) {
                self.refusal_kind = Some("mode");
            }
            e
        })?;
        let candidate = runtime_ios::media::build_candidate(
            self.registry,
            &c.provider,
            spec,
            Some(selection),
            strategy,
            work,
        )
        .map_err(|e| classify(&mut self.refusal_kind, e))?;
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
        let _ = runtime_ios::media::take_capacity_refusal();
        // A facade refusal keeps its typed kind; anything else is a message.
        enum Stop {
            Refused(&'static str, String),
            Failed(String),
        }
        impl From<String> for Stop {
            fn from(e: String) -> Self {
                Self::Failed(e)
            }
        }
        impl From<&str> for Stop {
            fn from(e: &str) -> Self {
                Self::Failed(e.into())
            }
        }
        let refused = |e: runtime_ios::media::AdmissionError| match e {
            runtime_ios::media::AdmissionError::Refused(r) => {
                Stop::Refused(r.kind.label(), r.reason)
            }
            runtime_ios::media::AdmissionError::Failed(e) => Stop::Failed(e.to_string()),
        };
        let outcome = (|| -> Result<(), Stop> {
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
                                id if runtime_ios::media::zimage::is_z_image(id) => {
                                    runtime_ios::media::load_z_image_turbo(one)
                                }
                                _ => runtime_ios::media::load_sana(one),
                            };
                            Box::new(loaded.map_err(refused)?.1)
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
                    Ok(support::persist_images(&docs, c, &images, Some(c.edge))?)
                }
                (IosWork::Restore(request), permit) => {
                    let restorer = match permit {
                        Some(permit) => permit
                            .load_restorer(registry, capture_capacity)
                            .map_err(|e| e.to_string())?,
                        None => {
                            runtime_ios::media::load_iris_restore(std::slice::from_ref(candidate))
                                .map_err(refused)?
                                .1
                        }
                    };
                    let image = restorer
                        .apply(request, &mut |_| {})
                        .map_err(|e| e.to_string())?;
                    Ok(support::persist_images(&docs, c, &[image], None)?)
                }
                (IosWork::Depth(request), permit) => {
                    use runtime_ios::iris_depth::IrisDepthEstimator;
                    let depth = match permit {
                        Some(permit) => permit
                            .load_depth(capture_capacity)
                            .map_err(|e| e.to_string())?,
                        None => {
                            runtime_ios::media::load_iris_depth(std::slice::from_ref(candidate))
                                .map_err(refused)?
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
                    Ok(support::persist_images(&docs, c, &[image], Some(c.edge))?)
                }
            }
        })();
        let error = match outcome {
            Ok(()) => return Ok(()),
            Err(Stop::Refused(kind, reason)) => {
                self.refusal_kind = Some(kind);
                return Err(ExecuteError::Refused(reason));
            }
            Err(Stop::Failed(error)) => error,
        };
        // Typed by the quantum mechanism / the capacity recheck, never by the message.
        if let Some(refusal) = runtime_ios::media::ios_quantum::take_refusal() {
            self.refusal_kind = Some("capacity");
            self.refused_quantum = Some(json!({
                "label": refusal.label,
                "scratch_bytes": refusal.scratch_bytes,
                "reserve_bytes": refusal.reserve_bytes,
                "available_bytes": refusal.available_bytes,
                "edge": c.edge,
            }));
            return Err(ExecuteError::QuantumRefused(error));
        }
        if runtime_ios::media::take_capacity_refusal() {
            self.refusal_kind = Some("capacity");
            return Err(ExecuteError::Refused(error));
        }
        Err(ExecuteError::Failure(error))
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
        // What this process already holds is committed exactly once (the facades do the same).
        let now = runtime_ios::process_memory();
        let budget = runtime_ios::media::capacity_budget(
            u64::from(runtime_ios::IOS_APP_MEMORY_CAP_MIB) << 20,
            now.footprint.unwrap_or(0),
            now.available,
        );
        let admitted = runtime_ios::media::admit(
            self.registry,
            std::slice::from_ref(candidate),
            &runtime_ios::device_machine(),
            budget,
        )
        .map_err(|e| classify(&mut self.refusal_kind, e))?;
        if admitted.index() != 0 {
            return Err("admission silently changed the selected case".into());
        }
        self.admitted = Some(json!({
            "demand_bytes": admitted.demand_bytes(),
            "margin_bytes": admitted.margin_bytes(),
            "required_bytes": admitted.required_bytes(),
            "basis": format!("{:?}", admitted.basis()),
        }));
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

// Darwin `task_vm_info`, requested at its full SDK count (`support::TASK_VM_INFO_COUNT`): current
// `phys_footprint` and the exact `ledger_phys_footprint_peak`, which only a REV3+ reply carries. A
// shorter reply is an explicit `PeakReading::Unavailable`. Neither is RSS or MLX allocator
// accounting. Unavailable readings stay null in the row, never zero evidence.
fn phys_footprint() -> Option<(u64, support::PeakReading)> {
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
    let model_case = || -> Result<Value, String> {
        let case = Case::parse(value)?;
        let catalog = runtime_ios::catalog().map_err(|e| e.to_string())?;
        // The catalog's DOMAIN (provider, route, mode): an anchor or any configuration the anchor
        // law derives. The selection is validated against the provider's declared space in
        // `prepare` (a token outside it is a typed strategy refusal).
        if !surfaces(catalog.media()).iter().any(|s| {
            s.provider == case.provider && s.routes.contains(&case.route) && s.modes.contains(&case.mode)
        }) {
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
            admitted: None,
        };
        // Bind before any source inspection/load. This does not create footprint evidence.
        runtime_ios::bound_mlx_to_platform_limits()
            .ok_or("platform allocator limits could not be bound; refused before load")?;
        if let Some(docs) = super::dirs_documents() {
            std::fs::write(docs.join(super::RENDERS_MANIFEST), "").map_err(|e| e.to_string())?;
        }
        mlx_rs::memory::reset_peak_memory();
        // The capture's starting footprint, before any of this case's loads: an evidence row's
        // demand is its peak minus this baseline (admission never credits a product's residents).
        let baseline_footprint = phys_footprint().map(|(current, _)| current);
        let start_peak = phys_footprint().and_then(|(_, peak)| match peak {
            support::PeakReading::Read(peak) => Some(peak),
            support::PeakReading::Unavailable { .. } => None,
        });
        // Every quantum boundary reads the kernel lifetime peak, attributed to the phase the
        // provider says that quantum runs in: the anchor's per-phase split.
        let marks = Arc::new(Mutex::new(Vec::<support::Mark>::new()));
        let sink = Arc::clone(&marks);
        let provider = case.provider.clone();
        runtime_ios::media::ios_quantum::set_quantum_listener(Some(Box::new(move |mark| {
            let (label, opens) = match mark {
                runtime_ios::media::ios_quantum::QuantumMark::Checked { label, .. } => (label, true),
                runtime_ios::media::ios_quantum::QuantumMark::Synchronized { label } => {
                    (label, false)
                }
            };
            if let Some((_, support::PeakReading::Read(lifetime_peak))) = phys_footprint() {
                if let Ok(mut marks) = sink.lock() {
                    marks.push(support::Mark {
                        phase: runtime_ios::media::pricing::quantum_phase(&provider, label),
                        opens,
                        lifetime_peak,
                    });
                }
            }
        })));
        let sampler = Sampler::start();
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
        runtime_ios::media::ios_quantum::set_quantum_listener(None);
        let e = sampler.finish()?;
        let phase_peaks = match (start_peak, e.lifetime_peak, marks.lock()) {
            (Some(start), Some(end), Ok(marks)) => support::phase_peaks(start, &marks, end)
                .into_iter()
                .map(|(phase, (peak, exact))| {
                    (phase.as_key().to_string(), json!({"peak_bytes": peak, "exact": exact}))
                })
                .collect::<serde_json::Map<_, _>>(),
            _ => serde_json::Map::new(),
        };
        let (status, reason) = if status.production_success()
            && (e.footprint.is_none() || e.min_headroom_mib.is_none() || e.lifetime_peak.is_none())
        {
            (Status::DiagnosticFailure, "completed but process footprint/headroom telemetry unavailable; not production evidence".into())
        } else {
            (status, reason)
        };
        let mut row = json!({ "schema":1, "case":case.key(), "status":status.label(), "production_success":status.production_success(), "reason":reason,
            "elapsed_seconds":started.elapsed().as_secs_f64(), "mlx_peak_bytes":mlx_rs::memory::get_peak_memory(),
            "device":runtime_ios::device_machine(), "cap_mib":runtime_ios::IOS_APP_MEMORY_CAP_MIB,
            "cap_validated":false, "diagnostic":diagnostic, "artifact_identity":adapter.artifact,
            "baseline_phys_footprint_bytes":baseline_footprint, "phase_peaks":phase_peaks, "steps":case.steps,
            "anchor_selection":adapter.candidate.as_ref().map(|c| runtime_ios::media::evidence::AnchorSelection::from_parts(c.context.selection.strategy, &c.context.selection.parameters)),
            "admitted":adapter.admitted,
            "evidence_class": if diagnostic { "capture" } else { "admitted" }, "evidence_key":adapter.evidence_key, "refused_quantum":adapter.refused_quantum,
            "selected_parameters":adapter.candidate.as_ref().map(|c| &c.context).map(|c| json!({"strategy":format!("{:?}",c.selection.strategy),"stage_residency":c.selection.parameters.stage_residency,"decode_tile_edge":c.selection.parameters.decode_tile_edge,"decode_overlap":c.selection.parameters.decode_overlap,"attention_chunk_size":c.selection.parameters.attention_chunk_size,"transformer_window_size":c.selection.parameters.transformer_window_size,"transformer_window_component":format!("{:?}",c.selection.parameters.transformer_window_component),"numeric_tier":format!("{:?}",c.selection.tier)})),
            "evidence_revision":adapter.candidate.as_ref().map(|c| &c.context.evidence_revision),
            "refusal_kind":adapter.refusal_kind, "refusal_only":adapter.refusal_only,
            "expected_render_count":adapter.candidate.as_ref().map(|c| match &c.work { runtime_ios::media::IosWork::Generate(r) => r.prompt_batch.len().max(1), _ => 1 }),
            "build_sha":option_env!("IOS_CAMPAIGN_BUILD_SHA").unwrap_or("unknown"),
            "entitlement":std::env::var("IOS_SMOKE_INCREASED_MEMORY_LIMIT").ok(),
            "ios_version":std::env::var("IOS_CAMPAIGN_IOS_VERSION").ok() });
        e.write(&mut row);
        Ok(row)
    };
    let result = if value == primitive::PROBE_KEY {
        primitive::report(diagnostic)
    } else {
        model_case()
    };
    let mut row = result.unwrap_or_else(|e| json!({"schema":1,"case":value,"status":Status::DiagnosticFailure.label(),"production_success":false,"reason":e,"elapsed_seconds":started.elapsed().as_secs_f64(),"mlx_active_peak_bytes":null,"mlx_cache_peak_bytes":null,"mlx_active_cache_peak_bytes":null,"mlx_peak_bytes":null,"phys_footprint_peak_bytes":null,"min_headroom_mib":null,"diagnostic":diagnostic}));
    row["run_id"] = json!(std::env::var("IOS_CAMPAIGN_RUN_ID").ok());
    row["build_sha"] = json!(option_env!("IOS_CAMPAIGN_BUILD_SHA").unwrap_or("unknown"));
    row["entitlement"] = json!(std::env::var("IOS_SMOKE_INCREASED_MEMORY_LIMIT").ok());
    row["ios_version"] = json!(std::env::var("IOS_CAMPAIGN_IOS_VERSION").ok());
    row["refusal_only"] = json!(std::env::var("IOS_CAMPAIGN_REFUSAL_ONLY").as_deref() == Ok("1"));
    if value == primitive::PROBE_KEY {
        // A primitive row is never a smoke/production verdict, so it never prints SMOKE: PASS.
        row["evidence_class"] = json!(primitive::EVIDENCE_CLASS);
        row["model_admission_evidence"] = json!(false);
        row["diagnostic"] = json!(true);
        row["nonce"] = row["run_id"].clone();
        if row.get("allocation_attempted").is_none() {
            row["allocation_attempted"] = json!(false);
        }
        return format!(
            "PRIMITIVE: {}\nCAMPAIGN_JSON {}\n",
            row["status"].as_str().unwrap_or("diagnostic_failure"),
            row
        );
    }
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

/// Extremes sampled every 20 ms across one case (model case or primitive probe). Unavailable
/// OS readings stay `None`, never zero evidence.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Extrema {
    pub active: usize,
    pub cache: usize,
    pub combined: usize,
    pub footprint: Option<u64>,
    pub min_headroom_mib: Option<f64>,
    pub lifetime_peak: Option<u64>,
    /// The kernel's reply word count when it lacked the lifetime peak (below REV3).
    pub peak_unavailable_words: Option<u32>,
}
impl Extrema {
    fn sample(&mut self) {
        let (active, cache) = (
            mlx_rs::memory::get_active_memory(),
            mlx_rs::memory::get_cache_memory(),
        );
        self.active = self.active.max(active);
        self.cache = self.cache.max(cache);
        self.combined = self.combined.max(active + cache);
        if let Some((current, peak)) = phys_footprint() {
            self.footprint = Some(self.footprint.unwrap_or(0).max(current));
            match peak {
                support::PeakReading::Read(peak) => {
                    self.lifetime_peak = Some(self.lifetime_peak.unwrap_or(0).max(peak));
                }
                support::PeakReading::Unavailable { returned_words } => {
                    self.peak_unavailable_words = Some(returned_words);
                }
            }
        }
        if let Some(a) = super::available_memory_mib() {
            self.min_headroom_mib = Some(self.min_headroom_mib.unwrap_or(a).min(a));
        }
    }
    pub(super) fn write(&self, row: &mut Value) {
        row["mlx_active_peak_bytes"] = json!(self.active);
        row["mlx_cache_peak_bytes"] = json!(self.cache);
        row["mlx_active_cache_peak_bytes"] = json!(self.combined);
        row["phys_footprint_peak_bytes"] = json!(self.footprint);
        row["min_headroom_mib"] = json!(self.min_headroom_mib);
        row["kernel_lifetime_phys_footprint_peak_bytes"] = json!(self.lifetime_peak);
        row["kernel_lifetime_peak_unavailable"] = json!(self.peak_unavailable_words.map(|w| {
            json!({"returned_words": w, "required_words": support::TASK_VM_INFO_REV3_COUNT})
        }));
    }
}

pub(super) struct Sampler {
    extrema: Arc<Mutex<Extrema>>,
    stop: std::sync::mpsc::Sender<()>,
    thread: std::thread::JoinHandle<()>,
}
impl Sampler {
    pub(super) fn start() -> Self {
        let extrema = Arc::new(Mutex::new(Extrema::default()));
        if let Ok(mut e) = extrema.lock() {
            e.sample();
        }
        let (stop, stop_rx) = std::sync::mpsc::channel();
        let samples = Arc::clone(&extrema);
        let thread = std::thread::spawn(move || loop {
            if let Ok(mut e) = samples.lock() {
                e.sample();
            }
            if !matches!(
                stop_rx.recv_timeout(Duration::from_millis(20)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ) {
                if let Ok(mut e) = samples.lock() {
                    e.sample();
                }
                break;
            }
        });
        Self {
            extrema,
            stop,
            thread,
        }
    }
    pub(super) fn finish(self) -> Result<Extrema, String> {
        let _ = self.stop.send(());
        let _ = self.thread.join();
        let e = *self.extrema.lock().map_err(|_| "sampler poisoned")?;
        Ok(e)
    }
}
