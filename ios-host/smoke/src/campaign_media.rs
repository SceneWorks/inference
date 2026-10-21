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

fn capture_recipes(
    registry: &ProviderRegistry,
    provider: &str,
    routes: &[String],
    modes: &[String],
    snapshots: Option<&Path>,
) -> std::collections::BTreeMap<(String, String, String), Vec<campaign::CaptureRecipe>> {
    use runtime_ios::media::{sampling, IosWork};
    let mut out = std::collections::BTreeMap::new();
    for route in routes {
        for tier in ["q4", "q8", "bf16"] {
            for mode in modes {
                let case = Case {
                    provider: provider.into(),
                    route: route.clone(),
                    tier: tier.into(),
                    edge: campaign::reference_edge(provider),
                    mode: mode.clone(),
                    rung: 0,
                    selection: campaign::BASE_SELECTION.into(),
                    steps: if provider.starts_with("iris_") {
                        0
                    } else {
                        campaign::ANCHOR_STEPS
                    },
                    recipe: None,
                };
                let carrier = work(&case, &CancelFlag::default(), registry);
                let configured = snapshots.map(|root| snapshot(root, &case));
                let refusal = |token, reason: String| campaign::CaptureRecipe {
                    token,
                    memory_class: None,
                    configured_variant: None,
                    steps: case.steps,
                    refusal: Some(reason),
                };
                let mut recipes = Vec::new();
                match carrier {
                    Err(error) => recipes.push(refusal(
                        None,
                        format!("actual carrier construction failed: {error:?}"),
                    )),
                    Ok(mut carrier) if provider.starts_with("iris_") => {
                        if let Some(Ok(spec)) = &configured {
                            let variants =
                                runtime_ios::media::configured_variant_digest(provider, spec)
                                    .map(|digest| {
                                        digest.map(|digest| runtime_ios::media::hex(&digest))
                                    })
                                    .map_err(|error| error.to_string());
                            let resolved = match &mut carrier {
                                IosWork::Generate(request) => {
                                    sampling::iris_configured_representatives(spec, request)
                                        .map(|items| items.into_iter().map(Ok).collect::<Vec<_>>())
                                }
                                IosWork::Restore(request) => {
                                    sampling::restoration_configured_representatives(spec, request)
                                }
                                IosWork::Depth(request) => {
                                    sampling::depth_configured_representatives(spec, request)
                                }
                            }
                            .map_err(|error| error.to_string());
                            match (variants, resolved) {
                                (Ok(configured_variant), Ok(items)) => {
                                    let mut classes = std::collections::BTreeSet::new();
                                    for item in items {
                                        match item {
                                            Ok(identity) if classes.insert(identity.allocation_class().to_string()) => recipes.push(campaign::CaptureRecipe {
                                                token: Some(identity.token().into()), memory_class: Some(identity.allocation_class().into()),
                                                configured_variant: configured_variant.clone(), steps: identity.steps(), refusal: None,
                                            }),
                                            Ok(_) => {}, // Dedup ONLY actual provider-normalized configured classes.
                                            Err(error) => recipes.push(refusal(None, format!("actual source topology unrepresentable: {error}"))),
                                        }
                                    }
                                }
                                (Err(error), _) | (_, Err(error)) => recipes.push(refusal(
                                    None,
                                    format!("actual configured metadata failed: {error}"),
                                )),
                            }
                        } else {
                            let reason = match &configured {
                                Some(Err(error)) => format!("unresolved controls: {error}"),
                                None => "unresolved controls: actual route/tier checkpoint metadata is not provisioned".into(),
                                Some(Ok(_)) => unreachable!(),
                            };
                            match &carrier {
                                IosWork::Generate(request) => {
                                    match sampling::iris_raw_templates(request) {
                                        Ok(tokens) => {
                                            for token in tokens {
                                                recipes.push(refusal(Some(token), reason.clone()));
                                            }
                                        }
                                        Err(error) => recipes.push(refusal(
                                            None,
                                            format!("{reason}; raw template failure: {error}"),
                                        )),
                                    }
                                }
                                IosWork::Restore(_) | IosWork::Depth(_) => {
                                    recipes.push(refusal(None, reason))
                                }
                            }
                        }
                    }
                    Ok(IosWork::Generate(mut request)) => {
                        let configured_variant = match &configured {
                            Some(Ok(spec)) => {
                                runtime_ios::media::configured_variant_digest(provider, spec)
                                    .map(|digest| {
                                        digest.map(|digest| runtime_ios::media::hex(&digest))
                                    })
                                    .map_err(|error| error.to_string())
                            }
                            Some(Err(error)) => Err(error.clone()),
                            None => Ok(None),
                        };
                        match (
                            sampling::representatives(provider).map_err(|error| error.to_string()),
                            configured_variant,
                        ) {
                            (Ok(items), Ok(configured_variant)) => {
                                for item in items {
                                    match sampling::apply_representative(
                                        provider,
                                        item.token(),
                                        &mut request,
                                    ) {
                                        Ok(actual) => recipes.push(campaign::CaptureRecipe {
                                            token: Some(actual.token().into()),
                                            memory_class: Some(actual.allocation_class().into()),
                                            configured_variant: configured_variant.clone(),
                                            steps: actual.steps(),
                                            refusal: None,
                                        }),
                                        Err(error) => recipes.push(campaign::CaptureRecipe {
                                            token: Some(item.token().into()),
                                            memory_class: Some(item.allocation_class().into()),
                                            configured_variant: configured_variant.clone(),
                                            steps: item.steps(),
                                            refusal: Some(error.to_string()),
                                        }),
                                    }
                                }
                            }
                            (Err(error), _) | (_, Err(error)) => recipes.push(refusal(
                                None,
                                format!("actual sampling metadata failed: {error}"),
                            )),
                        }
                    }
                    Ok(_) => recipes.push(refusal(
                        None,
                        "registered task has no source-owned representative adapter".into(),
                    )),
                }
                if recipes.is_empty() {
                    recipes.push(refusal(None, "source-owned representative set was empty; no capture coverage established".into()));
                }
                out.insert((route.clone(), tier.into(), mode.clone()), recipes);
            }
        }
    }
    out
}

pub fn surfaces(registry: &ProviderRegistry) -> Result<Vec<Surface>, String> {
    surfaces_with_snapshots(registry, None)
}

fn surfaces_with_snapshots(
    registry: &ProviderRegistry,
    snapshots: Option<&Path>,
) -> Result<Vec<Surface>, String> {
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
            recipes: capture_recipes(registry, d.id, &routes, &modes, snapshots),
            selections: selections(registry, d.id, &routes, snapshots)?,
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
            recipes: capture_recipes(
                registry,
                d.id,
                &[d.id.into()],
                &["restore".into()],
                snapshots,
            ),
            selections: selections(registry, d.id, &[d.id.into()], snapshots)?,
        });
    }
    // Composed provider-specific tasks are not generators. Discover them through their actual
    // memory registrations, not a second hard-coded catalog. Unknown tasks remain explicit cells.
    for r in registry.memory_strategy_registrations() {
        if !out.iter().any(|s| s.provider == r.provider_id) {
            out.push(Surface {
                recipes: capture_recipes(
                    registry,
                    r.provider_id,
                    &[r.provider_id.into()],
                    &[if r.provider_id == "iris_3b_depth" {
                        "depth".into()
                    } else {
                        "task".into()
                    }],
                    snapshots,
                ),
                selections: selections(
                    registry,
                    r.provider_id,
                    &[r.provider_id.into()],
                    snapshots,
                )?,
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
    Ok(out)
}

/// The `MemoryMode` key a campaign mode runs as (`work_mode`'s mapping).
pub fn mode_key(mode: &str) -> &str {
    match mode {
        "text" | "prompt_batch" => "text_to_image",
        "reference" => "image_to_image",
        "mask" => "edit",
        "restore" => "restoration",
        m if m == "control" || m.starts_with("control_") || m.starts_with("reference_") => {
            "control"
        }
        other => other,
    }
}

/// TRUE Reference/Mask/Control counts of the ACTUAL carrier constructor below, after accepted
/// control-kind resolution. Geometry reference_count remains their AGGREGATE, not tuple.0.
/// Counts are retained in evidence, not multiplied into a measured family grid.
fn carrier_role_counts(mode: &str, required_control: bool) -> Result<(u32, u32, u32), String> {
    let references = u32::from(matches!(mode, "reference" | "mask" | "ip_adapter"));
    let masks = u32::from(mode == "mask");
    let controls = u32::from(required_control || mode == "control");
    if !matches!(
        mode,
        "text"
            | "prompt_batch"
            | "reference"
            | "mask"
            | "ip_adapter"
            | "control"
            | "restore"
            | "depth"
    ) {
        return Err(format!("no actual campaign carrier for {mode}"));
    }
    Ok((references, masks, controls))
}

fn variant_for_registry(
    registry: &ProviderRegistry,
    provider: &str,
    mode: &str,
    tier: &str,
    class: Option<&str>,
) -> Result<String, String> {
    if provider == runtime_ios::media::sdxl::SDXL_ID {
        return Err(
            "SDXL role family requires its actual configured spec and sampling recipe".into(),
        );
    }
    let case = Case {
        provider: provider.into(),
        route: provider.into(),
        tier: tier.into(),
        edge: campaign::reference_edge(provider),
        mode: mode.into(),
        rung: 0,
        selection: campaign::BASE_SELECTION.into(),
        steps: 0,
        recipe: None,
    };
    // Use the same accepted-kind resolver as work_with_spec, not a lossy mode-prefix guess.
    let (actual, kind) = mode_and_kind(&case, registry).map_err(|error| format!("{error:?}"))?;
    let (references, masks, controls) = carrier_role_counts(&actual, kind.is_some())?;
    let aggregate = references
        .checked_add(masks)
        .and_then(|count| count.checked_add(controls))
        .ok_or_else(|| "campaign carrier role total overflows geometry".to_string())?;
    runtime_ios::media::pricing::anchor_variant_for_roles(
        provider,
        mode_key(&actual),
        tier,
        class,
        aggregate,
        masks,
        controls,
    )
}

/// Actual configured route classification, shared with admission, after recipe replay on the
/// existing carrier. No label, IP-file presence alone, or ignored top-level scale chooses IP.
fn variant_for_work(
    provider: &str,
    tier: &str,
    class: Option<&str>,
    spec: &LoadSpec,
    work: &runtime_ios::media::IosWork,
) -> Result<String, String> {
    let mode =
        runtime_ios::media::work_mode(provider, spec, work).map_err(|error| error.to_string())?;
    let roles = match work {
        runtime_ios::media::IosWork::Generate(request) => (
            request.memory_reference_count(),
            request
                .conditioning
                .iter()
                .filter(|role| matches!(role, Conditioning::Mask { .. }))
                .count() as u32,
            request
                .conditioning
                .iter()
                .filter(|role| matches!(role, Conditioning::Control { .. }))
                .count() as u32,
        ),
        runtime_ios::media::IosWork::Restore(_) | runtime_ios::media::IosWork::Depth(_) => {
            (0, 0, 0)
        }
    };
    runtime_ios::media::pricing::anchor_variant_for_roles(
        provider,
        mode.as_key(),
        tier,
        class,
        roles.0,
        roles.1,
        roles.2,
    )
}

/// The production listing's single configured case path. Keep recipe replay and actual spec
/// together; tests can exercise one case without enumerating a catalog or creating a model.
fn variant_for_case(
    registry: &ProviderRegistry,
    case: &Case,
    spec: &LoadSpec,
    class: Option<&str>,
) -> Result<String, String> {
    let work = work_with_spec(case, &CancelFlag::default(), registry, Some(spec))
        .map_err(|error| format!("{error:?}"))?;
    variant_for_work(&case.provider, &case.tier, class, spec, &work)
}

/// Unresolved carriers remain visible as unresolved families, never borrowed text evidence.
#[cfg(test)]
fn variant(provider: &str, mode: &str, tier: &str, class: Option<&str>) -> String {
    runtime_ios::catalog()
        .map_err(|error| error.to_string())
        .and_then(|catalog| variant_for_registry(catalog.media(), provider, mode, tier, class))
        .unwrap_or_else(|_| format!("unresolved-role-{mode}"))
}

/// The architecture group of `(provider, route, tier)`: on the host, the base-component
/// ARCHITECTURE digest of the provisioned slot under `snapshots` (the same function admission keys
/// the device's anchor families on); a slot that is absent or unreadable is its own group.
fn architecture_group(
    snapshots: &Path,
    provider: &str,
    route: &str,
    tier: &str,
) -> Result<String, String> {
    let case = Case {
        provider: provider.into(),
        route: route.into(),
        tier: tier.into(),
        edge: campaign::reference_edge(provider),
        mode: "text".into(),
        rung: 0,
        selection: campaign::BASE_SELECTION.into(),
        steps: 0,
        recipe: None,
    };
    if !snapshots.join(slot_route(&case)).join(tier).is_dir() {
        return Ok(format!("unprovisioned:{provider}/{route}/{tier}"));
    }
    let spec = snapshot(snapshots, &case)?;
    let digest =
        runtime_ios::media::architecture_digest(&spec).map_err(|error| error.to_string())?;
    Ok(runtime_ios::media::hex(&digest))
}

/// The campaign listing (schema 2): every planned ANCHOR case with its purpose and repeat count,
/// and the anchor FAMILIES ([`campaign::plan`]) — derived from the actual runtime catalog. With
/// `IOS_CAMPAIGN_SNAPSHOTS=<dir of route/tier slots>` (the slots that will be pushed) families are
/// grouped by architecture digest, so checkpoints of one architecture share anchors; without it,
/// every route is its own family (weights-free, conservative).
pub fn listing() -> Result<String, String> {
    let snapshots = std::env::var_os("IOS_CAMPAIGN_SNAPSHOTS")
        .map(PathBuf::from)
        .or_else(|| {
            #[cfg(target_os = "ios")]
            {
                super::dirs_documents().map(|docs| docs.join("campaign"))
            }
            #[cfg(not(target_os = "ios"))]
            {
                None
            }
        });
    listing_with(snapshots)
}

pub fn listing_with(snapshots: Option<PathBuf>) -> Result<String, String> {
    let catalog = runtime_ios::catalog().map_err(|e| e.to_string())?;
    let surfaces = surfaces_with_snapshots(catalog.media(), snapshots.as_deref())?;
    let variant_of =
        |provider: &str, route: &str, mode: &str, tier: &str, recipe: &campaign::CaptureRecipe| {
            let resolved = (|| -> Result<String, String> {
                if recipe.refusal.is_some() {
                    return Err("capture recipe has unresolved actual controls".into());
                }
                if let Some(root) = snapshots.as_deref() {
                    let case = Case {
                        provider: provider.into(),
                        route: route.into(),
                        tier: tier.into(),
                        edge: campaign::reference_edge(provider),
                        mode: mode.into(),
                        rung: 0,
                        selection: campaign::BASE_SELECTION.into(),
                        steps: recipe.steps,
                        recipe: recipe.token.clone(),
                    };
                    let spec = snapshot(root, &case)?;
                    variant_for_case(
                        catalog.media(),
                        &case,
                        &spec,
                        recipe.memory_class.as_deref(),
                    )
                } else {
                    variant_for_registry(
                        catalog.media(),
                        provider,
                        mode,
                        tier,
                        recipe.memory_class.as_deref(),
                    )
                }
            })();
            resolved.unwrap_or_else(|_| format!("unresolved-role-{mode}"))
        };
    let mut architectures = std::collections::BTreeMap::new();
    for surface in &surfaces {
        for route in &surface.routes {
            for tier in ["q4", "q8", "bf16"]
                .into_iter()
                .filter(|tier| surface.serves_route(route, tier))
            {
                let group = match &snapshots {
                    Some(root) => architecture_group(root, &surface.provider, route, tier)?,
                    None => format!("{}/{route}", surface.provider),
                };
                architectures.insert(
                    (surface.provider.clone(), route.clone(), tier.to_string()),
                    group,
                );
            }
        }
    }
    let by_architecture = |provider: &str, route: &str, tier: &str| {
        architectures
            .get(&(provider.into(), route.into(), tier.into()))
            .expect("family traversal uses the same served route/tier surface")
            .clone()
    };
    let grouping = campaign::Grouping {
        architecture: &by_architecture,
        variant: &variant_of,
    };
    let (families, planned) = campaign::plan(&surfaces, &grouping);
    // Visible, never silently dropped: every provider/route/tier it does not serve.
    let unserved: Vec<Value> = surfaces
        .iter()
        .flat_map(|s| {
            s.routes.iter().flat_map(move |route| {
                ["q4", "q8", "bf16"]
                    .into_iter()
                    .filter(|tier| !s.serves_route(route, tier))
                    .map(move |tier| json!({"provider": s.provider, "route": route, "tier": tier}))
            })
        })
        .collect();
    let unserved_sampling: Vec<Value> = surfaces.iter().flat_map(|surface| surface.recipes.iter()
        .flat_map(move |((route, tier, mode), recipes)| recipes.iter().filter_map(move |recipe| recipe.refusal.as_ref()
            .map(|reason| json!({"provider":surface.provider,"route":route,"tier":tier,"mode":mode,"recipe":recipe.token,"memory_class":recipe.memory_class,"configured_variant":recipe.configured_variant,"reason":reason}))))).collect();
    Ok(json!({"schema": 2,
        "grouping": if snapshots.is_some() { "architecture" } else { "per_route" },
        "families": families.iter().map(|f| json!({"architecture": f.architecture, "tier": f.tier, "variant": f.variant, "sampling_recipe": f.recipe,
            "anchor": format!("{}/{}/{}", f.anchor.0, f.anchor.1, f.anchor.2), "members": f.members})).collect::<Vec<_>>(),
        "unserved": unserved,
        "unserved_sampling": unserved_sampling,
        "cases": planned.iter().map(|p| json!({"case": p.case.key(), "purpose": p.purpose.label(), "repeats": p.repeats})).collect::<Vec<_>>(),
        "dependencies": [{"provider":"supra2", "epic":24212, "available": planned.iter().any(|p| p.case.provider.contains("supra")), "reason":"requires native engine and iOS catalog registration; absence is not a passing cell"}]
    }).to_string())
}

/// The slot a `(provider, route)` loads from (Z-Image Turbo's control route loads the base slot).
fn slot_route_of<'a>(provider: &str, route: &'a str) -> &'a str {
    if provider == runtime_ios::media::zimage::Z_IMAGE_TURBO_CONTROL_ID {
        runtime_ios::media::zimage::Z_IMAGE_TURBO_ID
    } else {
        route
    }
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

type CampaignSelections = std::collections::BTreeMap<(String, String, u8), Vec<String>>;

/// The provider's declared selection space per `(tier, rung)` as quality-first tokens, from its
/// weights-free contract surface at the load shape the campaign actually uses ([`load_spec`]).
fn selections(
    registry: &ProviderRegistry,
    provider: &str,
    routes: &[String],
    snapshots: Option<&Path>,
) -> Result<CampaignSelections, String> {
    let mut out = std::collections::BTreeMap::new();
    let contracts = registry
        .memory_contract_surfaces()
        .map_err(|error| error.to_string())?;
    for route in routes {
        for tier_name in ["q4", "q8", "bf16"] {
            for (rung, strategy) in MemoryStrategy::ALL.into_iter().enumerate().take(5) {
                let probe = Case {
                    provider: provider.into(),
                    route: route.clone(),
                    tier: tier_name.into(),
                    edge: campaign::reference_edge(provider),
                    mode: "text".into(),
                    rung: rung as u8,
                    selection: campaign::BASE_SELECTION.into(),
                    steps: 0,
                    recipe: None,
                };
                let configured = snapshots.filter(|root| {
                    root.join(slot_route_of(provider, route))
                        .join(tier_name)
                        .is_dir()
                });
                let space = if let Some(root) = configured {
                    let base = snapshot(root, &probe)?;
                    let spec = match runtime_ios::media::configured_load_form(
                        registry, provider, &base, strategy,
                    ) {
                        Ok(spec) => spec,
                        Err(runtime_ios::media::AdmissionError::Refused(refusal))
                            if refusal.kind == runtime_ios::media::RefusalKind::Strategy =>
                        {
                            continue
                        }
                        Err(error) => return Err(error.to_string()),
                    };
                    let contract = registry.memory_strategy_contract(provider, &spec).map_err(|error| error.to_string())?
                        .ok_or_else(|| format!("{provider}/{route}/{tier_name}: actual configured load has no memory contract"))?;
                    if contract.capability(strategy).is_none_or(|capability| {
                        capability.support != MemoryStrategySupport::Implemented
                    }) {
                        continue;
                    }
                    match contract.selection_space(strategy, tier_of(tier_name), false) {
                        Ok(space) => space,
                        Err(Error::Unsupported(_)) => continue,
                        Err(error) => return Err(error.to_string()),
                    }
                } else {
                    let spec = load_spec(&probe, WeightsSource::Dir(PathBuf::new()))?;
                    let Some(surface) = contracts.iter().find(|surface| {
                        surface.contract.provider_id == provider
                            && surface.selector.offload_policy == spec.offload_policy
                            && surface.selector.load_shape == spec.load_shape
                            && match surface.selector.tier {
                                MemoryContractSurfaceTier::Q4 => tier_name == "q4",
                                MemoryContractSurfaceTier::Q8 => tier_name == "q8",
                                MemoryContractSurfaceTier::Bf16 => tier_name == "bf16",
                                _ => false,
                            }
                    }) else {
                        continue;
                    };
                    match surface
                        .contract
                        .selection_space(strategy, tier_of(tier_name), false)
                    {
                        Ok(space) => space,
                        Err(Error::Unsupported(_)) => continue,
                        Err(error) => return Err(error.to_string()),
                    }
                };
                if !space.is_empty() {
                    out.insert(
                        (route.clone(), tier_name.to_string(), rung as u8),
                        space
                            .iter()
                            .map(|selection| selection_token(&selection.parameters))
                            .collect(),
                    );
                }
            }
        }
    }
    Ok(out)
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
    work_with_spec(c, cancel, registry, None)
}

fn work_with_spec(
    c: &Case,
    cancel: &CancelFlag,
    registry: &ProviderRegistry,
    spec: Option<&LoadSpec>,
) -> Result<runtime_ios::media::IosWork, AdmissionError> {
    use runtime_ios::media::IosWork;
    if c.recipe.is_some() && !runtime_ios::media::sampling::requires_recipe(&c.provider) {
        return Err(AdmissionError::Refused(
            "sampling recipe belongs to another provider".into(),
        ));
    }
    let (mode, required_kind) = mode_and_kind(c, registry)?;
    let expected_roles =
        carrier_role_counts(&mode, required_kind.is_some()).map_err(AdmissionError::Refused)?;
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
        // Generation role construction below shares the family's source role partition.
        "reference" | "ip_adapter" | "mask" | "control" => {}
        "restore" => {
            let mut request = TransformRequest {
                image: image(),
                seed: Some(16774),
                cancel: cancel.clone(),
                steps: (c.steps > 0).then_some(c.steps),
                ..Default::default()
            };
            if let Some(token) = &c.recipe {
                let spec = spec.ok_or_else(|| {
                    AdmissionError::Refused(
                        "restoration replay requires actual configured metadata".into(),
                    )
                })?;
                runtime_ios::media::sampling::apply_transform_for_spec(spec, token, &mut request)
                    .map_err(|error| AdmissionError::Refused(error.to_string()))?;
            }
            return Ok(IosWork::Restore(request));
        }
        "depth" => {
            let mut request = runtime_ios::iris_depth::DepthRequest {
                image: image(),
                cancel: cancel.clone(),
                ..Default::default()
            };
            if let Some(token) = &c.recipe {
                let spec = spec.ok_or_else(|| {
                    AdmissionError::Refused(
                        "depth replay requires actual configured metadata".into(),
                    )
                })?;
                runtime_ios::media::sampling::apply_depth_for_spec(spec, token, &mut request)
                    .map_err(|error| AdmissionError::Refused(error.to_string()))?;
            }
            return Ok(IosWork::Depth(request));
        }
        other => {
            return Err(AdmissionError::Refused(format!(
                "{other}: no request carrier for this registered task"
            )))
        }
    }
    if expected_roles.0 != 0 {
        request.conditioning.push(Conditioning::Reference {
            image: image(),
            strength: Some(0.6),
        });
    }
    if expected_roles.1 != 0 {
        request
            .conditioning
            .push(Conditioning::Mask { image: image() });
    }
    if expected_roles.2 != 0 && required_kind.is_none() {
        request.conditioning.push(Conditioning::Control {
            image: image(),
            kind: ControlKind::Other("tile".into()),
            scale: Some(1.0),
        });
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
    if let Some(token) = &c.recipe {
        // Mode/conditioning carriers exist BEFORE compatibility validation. In particular, the
        // provider's actual IP/Reference rules are not replaced by a planner-side guess.
        if let Some(spec) = spec {
            runtime_ios::media::sampling::apply_for_spec(&c.provider, spec, token, &mut request)
                .map_err(|e| AdmissionError::Refused(e.to_string()))?;
        } else {
            runtime_ios::media::sampling::apply(&c.provider, token, &mut request).map_err(|e| {
                AdmissionError::Refused(format!(
                    "unresolved controls require actual configured metadata: {e}"
                ))
            })?;
        }
        // Steps remain the case's explicit validation axis; 0 requests actual factory defaults.
        request.steps = (c.steps > 0).then_some(c.steps);
        match spec {
            Some(spec) => {
                runtime_ios::media::sampling::resolve_for_spec(&c.provider, spec, &request)
            }
            None => runtime_ios::media::sampling::resolve(&c.provider, &request),
        }
        .map_err(|e| AdmissionError::Refused(e.to_string()))?;
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
        // Deferred: Iris's rung 4 (streamed text-tower layers and backbone blocks) exists only on a
        // deferred load, and the fully optimized anchor is rung 4.
        id if id.starts_with("iris_") => {
            LoadSpec::new(source).with_load_shape(LoadShape::DeferredMaterialization)
        }
        _ => LoadSpec::new(source)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization),
    };
    // Endpoint load form precedes contract lookup, selection, pricing, construction and identity.
    // Ordinary provider defaults above remain unchanged outside this capture adapter.
    let endpoint = if c.rung == 0 {
        runtime_ios::media::evidence::AnchorKind::Resident
    } else {
        runtime_ios::media::evidence::AnchorKind::Optimized
    };
    endpoint.apply_load_form(&mut spec);
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
    // Actual AdapterSpec transport is a separate typed list, never a named-component path hack
    // or a caller-claimed memory class. Header normalization stays in the production provider.
    let adapters = path.join("adapters.json");
    if adapters.exists() {
        use std::io::Read;
        let mut bytes = Vec::new();
        std::fs::File::open(&adapters)
            .map_err(|e| e.to_string())?
            .take(65537)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > 65536 {
            return Err("typed adapter manifest exceeds 64 KiB".into());
        }
        spec.adapters =
            runtime_ios::media::capture_adapters(&bytes, &path).map_err(|e| e.to_string())?;
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
    /// Actual sampler state shared only by the on-device capture, so its second baseline is taken
    /// after real request carriers exist and before model/admit-load work.
    telemetry: Option<Arc<Mutex<Extrema>>>,
}
impl DeviceProvider<'_> {
    fn prepare(&mut self, c: &Case) -> Result<(), AdmissionError> {
        // REAL source boundary BEFORE snapshot/configured-load/contract/work/candidate reads.
        // A later key or permit check cannot retroactively protect this prelude.
        let source_capacity_refused = std::cell::Cell::new(false);
        let mut key_next = |bytes| {
            let result = runtime_ios::media::source_metadata_capacity(bytes);
            if result.is_err() {
                source_capacity_refused.set(true);
            }
            result.map_err(Into::into)
        };
        mlx_gen::ios_pricing::with_core_source_capacity(&mut key_next, |core_next| {
            SourceInventory::preflight_directory_with_capacity(&self.root, core_next)
                .map_err(mlx_gen::Error::from)
        })
        .map_err(|error| {
            let error: runtime_ios::gen_core::Error = error.into();
            let error = if source_capacity_refused.get() {
                runtime_ios::media::AdmissionError::refused(
                    runtime_ios::media::RefusalKind::Capacity,
                    error.to_string(),
                )
            } else {
                match error {
                    Error::Unsupported(reason)
                        if [
                            SourceInventory::READDIR_METADATA_GAP,
                            SourceInventory::PIN_VERIFY_METADATA_GAP,
                            SourceInventory::PIN_CAPTURE_METADATA_GAP,
                            SourceInventory::ABSOLUTE_PATH_METADATA_GAP,
                            SourceInventory::MISSING_INVENTORY_METADATA_GAP,
                        ]
                        .contains(&reason.as_str()) =>
                    {
                        runtime_ios::media::AdmissionError::refused(
                            runtime_ios::media::RefusalKind::NoEvidence,
                            reason,
                        )
                    }
                    other => runtime_ios::media::AdmissionError::Failed(other.into()),
                }
            };
            classify(&mut self.refusal_kind, error)
        })?;
        let configured = snapshot(&self.root, c)?;
        let strategy = MemoryStrategy::ALL[usize::from(c.rung)];
        let spec = runtime_ios::media::configured_load_form(
            self.registry,
            &c.provider,
            &configured,
            strategy,
        )
        .map_err(|error| classify(&mut self.refusal_kind, error))?;
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
        let work = work_with_spec(c, &self.cancel, self.registry, Some(&spec)).map_err(|e| {
            if matches!(e, AdmissionError::Refused(_)) {
                self.refusal_kind = Some("mode");
            }
            e
        })?;
        let mut candidate = runtime_ios::media::build_candidate(
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
        // Reuse the authoritative callback installed before the pre-candidate prelude.
        let mut key = runtime_ios::media::evidence_key_with_key_capacity(
            self.registry,
            &mut candidate,
            &runtime_ios::device_machine(),
            &mut key_next,
        )
        .map_err(|e| classify(&mut self.refusal_kind, e))?;
        if c.provider == "sdxl" {
            let allowance =
                runtime_ios::media::source_metadata_allowance().map_err(|e| e.to_string())?;
            let mut next =
                |bytes| runtime_ios::media::source_metadata_capacity(bytes).map_err(Into::into);
            let mut budget =
                runtime_ios::media::SourceRequestBudget::with_capacity(allowance, &mut next);
            runtime_ios::media::record_source_context(&mut key, &candidate, &mut budget)
                .map_err(|e| e.to_string())?;
        }
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
                    let expected = support::expected_generation_outputs(request)?;
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
        // Provenance storage exists before the real counter boundary. Both capture ledger and
        // SDK sidecar use THIS query, not a second arbitrarily timed footprint or budget guess.
        let baseline_source =
            String::from("TASK_VM_INFO.phys_footprint after prepare, before model admission/load");
        let snapshot = if let Some(telemetry) = &self.telemetry {
            let mut state = telemetry
                .lock()
                .map_err(|_| "capture sampler poisoned before model baseline")?;
            let snapshot = footprint_snapshot();
            state.start_model_coordinate(snapshot);
            snapshot
        } else {
            footprint_snapshot()
        };
        let baseline = runtime_ios::media::IosMemoryBaseline {
            domain: runtime_ios::media::IosMemoryDomain::ProcessGrowth,
            coordinate: runtime_ios::media::IosMemoryCoordinate::BeforeModelLoad,
            bytes: snapshot.current().ok_or_else(|| format!(
                "before-model-load footprint unavailable: status={}, returned_words={}, requested_words={}",
                snapshot.query_status, snapshot.returned_words, snapshot.requested_words))?,
            source: baseline_source,
        };
        // This is an actual snapshot marker, NOT a declared/predicted solver evaluation.
        if let Ok((current, peak)) = snapshot.complete_pair() {
            runtime_ios::media::ios_quantum::diagnostic_metadata(
                "ios_capture",
                "before-model-load",
                &[
                    ("current_bytes", current),
                    ("lifetime_highwater", peak),
                    ("requested_words", u64::from(snapshot.requested_words)),
                    ("returned_words", u64::from(snapshot.returned_words)),
                ],
            );
        } else {
            runtime_ios::media::ios_quantum::diagnostic_metadata(
                "ios_capture",
                "before-model-load",
                &[
                    ("query_valid", 0),
                    ("returned_words", u64::from(snapshot.returned_words)),
                ],
            );
        }
        let candidate = self.candidate.as_ref().ok_or("no exact work")?;
        let (reference_count, mask_count, control_count) = match &candidate.work {
            runtime_ios::media::IosWork::Generate(request) => (
                u64::from(request.memory_reference_count()),
                request
                    .conditioning
                    .iter()
                    .filter(|item| matches!(item, Conditioning::Mask { .. }))
                    .count() as u64,
                request
                    .conditioning
                    .iter()
                    .filter(|item| matches!(item, Conditioning::Control { .. }))
                    .count() as u64,
            ),
            runtime_ios::media::IosWork::Restore(_) | runtime_ios::media::IosWork::Depth(_) => {
                (0, 0, 0)
            }
        };
        // Actual prepared carriers, not the case's mode label or caller-supplied evidence counts.
        runtime_ios::media::ios_quantum::diagnostic_metadata(
            "ios_capture",
            "conditioning-roles",
            &[
                ("reference_count", reference_count),
                ("mask_count", mask_count),
                ("control_count", control_count),
            ],
        );
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
            baseline.bytes,
            now.available,
        );
        // The full SDK boundary: this campaign is the execution owner, so it states the facts of
        // the run it is about to execute, including its OWN callbacks: the no-op progress closures
        // and the capture listener installed for this case (source-derived upper, not zero).
        let callbacks = support::capture_listener_callback_host(&candidate.provider)?;
        let facts = runtime_ios::media::execution_facts(Some(callbacks))
            .map_err(|e| classify(&mut self.refusal_kind, e))?;
        let admitted = runtime_ios::media::admit_with_boundary(
            self.registry,
            std::slice::from_ref(candidate),
            &runtime_ios::device_machine(),
            budget,
            &baseline,
            &facts,
            &runtime_ios::media::PremisePolicy::default(),
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
fn footprint_snapshot() -> runtime_ios::media::evidence::KernelFootprintSnapshot {
    #[cfg(target_vendor = "apple")]
    {
        support::footprint_snapshot(|words, count| {
            #[allow(deprecated)]
            unsafe {
                libc::task_info(libc::mach_task_self(), 22, words.as_mut_ptr().cast(), count)
            }
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        use runtime_ios::media::evidence::{KernelFootprintMethod, KernelFootprintSnapshot};
        KernelFootprintSnapshot {
            method: KernelFootprintMethod::UnsupportedPlatform,
            query_status: -1,
            requested_words: support::TASK_VM_INFO_COUNT,
            returned_words: 0,
            current_required_words: support::TASK_VM_INFO_REV1_COUNT,
            peak_required_words: support::TASK_VM_INFO_REV3_COUNT,
            current_bytes: None,
            lifetime_peak_bytes: None,
        }
    }
}
fn phys_footprint() -> Option<(u64, support::PeakReading)> {
    let snapshot = footprint_snapshot();
    Some((
        snapshot.current_bytes?,
        snapshot.lifetime_peak_bytes.map_or(
            support::PeakReading::Unavailable {
                returned_words: snapshot.returned_words,
            },
            support::PeakReading::Read,
        ),
    ))
}

pub fn report(value: &str, diagnostic: bool) -> String {
    let started = Instant::now();
    let model_case =
        || -> Result<(Value, Option<runtime_ios::media::evidence::EvidenceKey>), String> {
            let case = Case::parse(value)?;
            let catalog = runtime_ios::catalog().map_err(|e| e.to_string())?;
            // The catalog's DOMAIN (provider, route, mode): an anchor or any configuration the anchor
            // law derives. The selection is validated against the provider's declared space in
            // `prepare` (a token outside it is a typed strategy refusal).
            if !surfaces(catalog.media())?.iter().any(|s| {
                s.provider == case.provider
                    && s.routes.contains(&case.route)
                    && s.modes.contains(&case.mode)
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
                telemetry: None,
            };
            // Bind before any source inspection/load. This does not create footprint evidence.
            runtime_ios::bound_mlx_to_platform_limits()
                .ok_or("platform allocator limits could not be bound; refused before load")?;
            if let Some(docs) = super::dirs_documents() {
                std::fs::write(docs.join(super::RENDERS_MANIFEST), "")
                    .map_err(|e| e.to_string())?;
            }
            mlx_rs::memory::reset_peak_memory();
            // The capture's starting footprint, before any of this case's loads: an evidence row's
            // demand is its peak minus this baseline (admission never credits a product's residents).
            let before_request_snapshot = footprint_snapshot();
            let baseline_footprint = before_request_snapshot.current();
            let start_peak = before_request_snapshot
                .complete_pair()
                .ok()
                .map(|(_, peak)| peak);
            // Every quantum boundary reads the kernel lifetime peak, attributed to the phase the
            // provider says that quantum runs in: the anchor's per-phase split.
            let marks = Arc::new(Mutex::new(support::CaptureMarks::default()));
            let sink = Arc::clone(&marks);
            let listener_failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let failed = Arc::clone(&listener_failed);
            let provider = case.provider.clone();
            runtime_ios::media::ios_quantum::set_quantum_listener(Some(Box::new(move |mark| {
                let (label, opens, scratch) = match mark {
                    runtime_ios::media::ios_quantum::QuantumMark::Metadata {
                        provider,
                        kind,
                        fields,
                    } => {
                        match sink.lock() {
                            Ok(mut captured) => captured.metadata(provider, kind, fields),
                            Err(_) => failed.store(true, std::sync::atomic::Ordering::Relaxed),
                        }
                        return;
                    }
                    runtime_ios::media::ios_quantum::QuantumMark::Checked { label, scratch } => {
                        (label, true, scratch)
                    }
                    runtime_ios::media::ios_quantum::QuantumMark::Synchronized { label } => {
                        (label, false, 0)
                    }
                };
                let reading = phys_footprint();
                match sink.lock() {
                    Ok(mut captured) => {
                        let phase = runtime_ios::media::pricing::quantum_phase(&provider, label);
                        // Check/context occurrence is real even when its footprint query failed.
                        if opens {
                            captured.checked(phase, label);
                        }
                        match reading {
                            Some((current, support::PeakReading::Read(lifetime_peak))) => {
                                let preceding_sampling_mark =
                                    if phase == Some(mlx_gen::ios_pricing::IosPhase::Denoise) {
                                        captured.preceding_sampling_mark
                                    } else {
                                        None
                                    };
                                captured.marks.push(support::Mark {
                                    phase,
                                    opens,
                                    lifetime_peak,
                                    label: label.to_string(),
                                    scratch,
                                    current,
                                    preceding_sampling_mark,
                                });
                            }
                            _ => captured.quantum_query_gaps += 1,
                        }
                    }
                    Err(_) => failed.store(true, std::sync::atomic::Ordering::Relaxed),
                }
            })));
            let sampler = Sampler::start();
            adapter.telemetry = Some(Arc::clone(&sampler.extrema));
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
            let ending_snapshot = footprint_snapshot();
            if listener_failed.load(std::sync::atomic::Ordering::Relaxed) {
                return Err("capture observation listener failed".into());
            }
            let captured = marks
                .lock()
                .map_err(|_| "capture observation collector was poisoned")?;
            captured.sampling.validate()?;
            captured.quantum.validate(&captured.sampling)?;
            let observed_current_peak = before_request_snapshot
                .current()
                .into_iter()
                .chain(e.footprint)
                .chain(ending_snapshot.current())
                .chain(captured.marks.iter().map(|mark| mark.current))
                .max();
            let model_marks = captured
                .model_mark_start
                .and_then(|start| captured.marks.get(start..));
            let observed_model_current_peak = e
                .before_model_load
                .and_then(|snapshot| snapshot.current())
                .into_iter()
                .chain(e.model_footprint)
                .chain(ending_snapshot.current())
                .chain(model_marks.into_iter().flatten().map(|mark| mark.current))
                .max();
            let valid_kernel_queries = e.valid_queries
                + captured.marks.len() as u64
                + u64::from(before_request_snapshot.complete_pair().is_ok())
                + u64::from(ending_snapshot.complete_pair().is_ok());
            let valid_model_kernel_queries = e.valid_model_queries
                + model_marks.map_or(0, |marks| marks.len() as u64)
                + u64::from(ending_snapshot.complete_pair().is_ok());
            let growth = |start: Option<runtime_ios::media::evidence::KernelFootprintSnapshot>,
                          current_peak: Option<u64>,
                          queries|
             -> Result<mlx_gen::ios_anchor_law::KernelGrowth, String> {
                let (current, old_peak) = start
                    .ok_or("starting kernel snapshot missing")?
                    .complete_pair()?;
                let (_, end_peak) = ending_snapshot.complete_pair()?;
                mlx_gen::ios_anchor_law::KernelGrowth::from_queries(
                    current,
                    old_peak,
                    end_peak,
                    current_peak.ok_or("observed current footprint missing")?,
                    queries,
                )
                .map_err(|e| e.to_string())
            };
            let original_growth = growth(
                Some(before_request_snapshot),
                observed_current_peak,
                valid_kernel_queries,
            );
            let sdk_growth = growth(
                e.before_model_load,
                observed_model_current_peak,
                valid_model_kernel_queries,
            );
            let sampling_required = adapter.candidate.as_ref().is_some_and(|candidate| {
                matches!(&candidate.work, runtime_ios::media::IosWork::Generate(_))
            });
            let sampling_missing =
                sampling_required && !captured.sampling.has_execution_observation();
            let quantum_missing = sampling_required && captured.quantum.checks.is_empty();
            if matches!(status, Status::Successful | Status::DiagnosticSuccess)
                && (sampling_missing
                    || quantum_missing
                    || captured.sampling.truncated
                    || captured.quantum.truncated)
            {
                status = Status::DiagnosticFailure;
                reason = format!("actual observations missing_sampling={sampling_missing}, missing_quantum={quantum_missing}, sampling_truncated={}, quantum_truncated={}, dropped_metadata={}, dropped_checks={}, malformed={}; not class-coverage evidence",
                captured.sampling.truncated, captured.quantum.truncated, captured.sampling.dropped_marks,
                captured.quantum.dropped_checks, captured.sampling.malformed_marks);
            }
            let phase_peaks = match (start_peak, e.lifetime_peak) {
                (Some(start), Some(end)) => support::phase_peaks(start, &captured.marks, end)
                    .into_iter()
                    .map(|(phase, (peak, exact))| {
                        (
                            phase.as_key().to_string(),
                            json!({"peak_bytes": peak, "exact": exact}),
                        )
                    })
                    .collect::<serde_json::Map<_, _>>(),
                _ => serde_json::Map::new(),
            };
            // The per-formula DEVICE calibration: each quantum's priced scratch vs its observed growth.
            let quanta = match e.lifetime_peak {
            Some(end) => support::quantum_observations(&captured.marks, end)
                .into_iter()
                .map(|(label, o)| {
                    (label, json!({"priced_scratch_bytes": o.priced, "observed_bytes": o.observed, "exact": o.exact, "preceding_sampling_mark": o.preceding_sampling_mark}))
                })
                .collect::<serde_json::Map<_, _>>(),
            _ => serde_json::Map::new(),
        };
            let (status, reason) = if matches!(
                status,
                Status::Successful | Status::DiagnosticSuccess
            ) && (e.footprint.is_none()
                || e.min_headroom_mib.is_none()
                || e.lifetime_peak.is_none()
                || before_request_snapshot.complete_pair().is_err()
                || ending_snapshot.complete_pair().is_err()
                || e.before_model_load
                    .is_none_or(|snapshot| snapshot.complete_pair().is_err())
                || captured.model_mark_start.is_none()
                || original_growth.is_err()
                || sdk_growth.is_err())
            {
                (Status::DiagnosticFailure, "completed but process footprint/headroom telemetry unavailable; not production evidence".into())
            } else {
                (status, reason)
            };
            let sampling_identity = adapter
                .evidence_key
                .as_ref()
                .and_then(|key| key.sampling_recipe.as_deref())
                .map(|token| {
                    runtime_ios::media::sampling::from_token(&case.provider, token)
                        .map_err(|e| e.to_string())
                })
                .transpose()?;
            if sampling_identity.is_none()
                && adapter.candidate.as_ref().is_some_and(|candidate| {
                    matches!(&candidate.work, runtime_ios::media::IosWork::Generate(_))
                })
            {
                return Err(
                    "prepared generation capture is missing its resolved sampling identity".into(),
                );
            }
            let resolved_steps = sampling_identity
                .as_ref()
                .map_or(case.steps, |identity| identity.steps());
            let expected_render_count = adapter
                .candidate
                .as_ref()
                .map(|candidate| match &candidate.work {
                    runtime_ios::media::IosWork::Generate(request) => {
                        support::expected_generation_outputs(request)
                    }
                    _ => Ok(1),
                })
                .transpose()?;
            let mut row = json!({ "schema":1, "case":case.key(), "status":status.label(), "production_success":status.production_success(), "reason":reason,
            "elapsed_seconds":started.elapsed().as_secs_f64(), "mlx_peak_bytes":mlx_rs::memory::get_peak_memory(),
            "device":runtime_ios::device_machine(), "cap_mib":runtime_ios::IOS_APP_MEMORY_CAP_MIB,
            "cap_validated":false, "diagnostic":diagnostic, "artifact_identity":adapter.artifact,
            "baseline_phys_footprint_bytes":baseline_footprint,
            "before_request_snapshot":before_request_snapshot, "before_model_load_snapshot":e.before_model_load,
            "ending_snapshot":ending_snapshot, "observed_current_peak_bytes":observed_current_peak,
            "observed_model_current_peak_bytes":observed_model_current_peak,
            "valid_kernel_queries":valid_kernel_queries, "valid_model_kernel_queries":valid_model_kernel_queries,
            "physical_growth_bounds":{"before_request_construction":support::growth_report(&original_growth),
                "before_model_load":support::growth_report(&sdk_growth)},
            "phase_attribution_gaps":captured.quantum_query_gaps, "phase_peaks":phase_peaks, "quanta":quanta, "steps":resolved_steps, "requested_steps":case.steps,
            "load_form":adapter.candidate.as_ref().map(|c| json!({"offload_policy":format!("{:?}",c.spec.offload_policy),"load_shape":format!("{:?}",c.spec.load_shape)})),
            "anchor_selection":adapter.candidate.as_ref().map(|c| runtime_ios::media::evidence::AnchorSelection::from_parts(c.context.selection.strategy, &c.context.selection.parameters)),
            "sampling_recipe":sampling_identity.as_ref().map(|identity| identity.provenance()),
            "sampling_allocation_class":sampling_identity.as_ref().map(|identity| identity.allocation_class()),
            "sampling_metadata_required":sampling_required, "sampling_metadata_missing":sampling_missing,
            "sampling_coverage":if captured.sampling.total_marks > 0 { Some(&captured.sampling) } else { None },
            "quantum_coverage":Some(&captured.quantum), "quantum_query_gaps":captured.quantum_query_gaps,
            "admitted":adapter.admitted,
            "evidence_class": if diagnostic { "capture" } else { "admitted" }, "refused_quantum":adapter.refused_quantum,
            "selected_parameters":adapter.candidate.as_ref().map(|c| &c.context).map(|c| json!({"strategy":format!("{:?}",c.selection.strategy),"stage_residency":c.selection.parameters.stage_residency,"decode_tile_edge":c.selection.parameters.decode_tile_edge,"decode_overlap":c.selection.parameters.decode_overlap,"attention_chunk_size":c.selection.parameters.attention_chunk_size,"transformer_window_size":c.selection.parameters.transformer_window_size,"transformer_window_component":format!("{:?}",c.selection.parameters.transformer_window_component),"numeric_tier":format!("{:?}",c.selection.tier)})),
            "evidence_revision":adapter.candidate.as_ref().map(|c| &c.context.evidence_revision),
            "refusal_kind":adapter.refusal_kind, "refusal_only":adapter.refusal_only,
            "expected_render_count":expected_render_count,
            "build_sha":option_env!("IOS_CAMPAIGN_BUILD_SHA").unwrap_or("unknown"),
            "entitlement":std::env::var("IOS_SMOKE_INCREASED_MEMORY_LIMIT").ok(),
            "ios_version":std::env::var("IOS_CAMPAIGN_IOS_VERSION").ok() });
            e.write(&mut row);
            // The final same-task kernel query, not a sampled lower/current value, is the run upper.
            row["kernel_lifetime_phys_footprint_peak_bytes"] =
                json!(ending_snapshot.lifetime_peak_bytes);
            Ok((row, adapter.evidence_key))
        };
    let result = if value == primitive::PROBE_KEY {
        primitive::report(diagnostic).map(|row| (row, None))
    } else {
        model_case()
    };
    let (mut row, source_key) = result.unwrap_or_else(|e| (json!({"schema":1,"case":value,"status":Status::DiagnosticFailure.label(),"production_success":false,"reason":e,"elapsed_seconds":started.elapsed().as_secs_f64(),"mlx_active_peak_bytes":null,"mlx_cache_peak_bytes":null,"mlx_active_cache_peak_bytes":null,"mlx_peak_bytes":null,"phys_footprint_peak_bytes":null,"min_headroom_mib":null,"diagnostic":diagnostic}), None));
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
    let prefix = if row["production_success"] == true {
        "SMOKE: PASS\nCAMPAIGN_JSON "
    } else {
        "SMOKE: FAIL\nCAMPAIGN_JSON "
    };
    let serialized = (|| -> Result<String, String> {
        let retained = source_key
            .as_ref()
            .and_then(|key| key.sdxl_work.as_ref())
            .map(|binding| binding.retained_requested_bytes())
            .transpose()
            .map_err(|e| e.to_string())?
            .unwrap_or(0);
        let fresh = runtime_ios::media::source_metadata_allowance().map_err(|e| e.to_string())?;
        let allowance = retained
            .checked_add(fresh)
            .ok_or("source serialization allowance overflow")?;
        let mut next =
            |bytes| runtime_ios::media::source_metadata_capacity(bytes).map_err(Into::into);
        let mut budget =
            runtime_ios::media::SourceRequestBudget::with_capacity(allowance, &mut next);
        budget
            .account_retained(retained)
            .map_err(|e| e.to_string())?;
        runtime_ios::media::serialize_capture_row(prefix, &row, source_key.as_ref(), &mut budget)
            .map_err(|e| e.to_string())
    })();
    match serialized {
        Ok(output) => output,
        // A failed output reservation is not a completed capture. This fixed-size diagnostic has
        // no source key/payload and is never importable as completed evidence.
        Err(_) => {
            let fallback = (|| -> Result<String, String> {
                let retained = source_key
                    .as_ref()
                    .and_then(|key| key.sdxl_work.as_ref())
                    .map(|binding| binding.retained_requested_bytes())
                    .transpose()
                    .map_err(|e| e.to_string())?
                    .unwrap_or(0);
                let fresh =
                    runtime_ios::media::source_metadata_allowance().map_err(|e| e.to_string())?;
                let allowance = retained
                    .checked_add(fresh)
                    .ok_or("source diagnostic allowance overflow")?;
                let mut next =
                    |bytes| runtime_ios::media::source_metadata_capacity(bytes).map_err(Into::into);
                let mut budget =
                    runtime_ios::media::SourceRequestBudget::with_capacity(allowance, &mut next);
                budget
                    .account_retained(retained)
                    .map_err(|e| e.to_string())?;
                runtime_ios::media::serialize_capture_failure(
                    row.get("case").unwrap_or(&Value::Null),
                    &mut budget,
                )
                .map_err(|e| e.to_string())
            })();
            // No output allocation, and no successful evidence line, if even the diagnostic fails.
            fallback.unwrap_or_else(|_| String::new())
        }
    }
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
    pub before_model_load: Option<runtime_ios::media::evidence::KernelFootprintSnapshot>,
    pub model_footprint: Option<u64>,
    pub valid_queries: u64,
    pub valid_model_queries: u64,
    /// The kernel's reply word count when it lacked the lifetime peak (below REV3).
    pub peak_unavailable_words: Option<u32>,
}
impl Extrema {
    fn start_model_coordinate(
        &mut self,
        snapshot: runtime_ios::media::evidence::KernelFootprintSnapshot,
    ) {
        self.before_model_load = Some(snapshot);
        self.model_footprint = snapshot.current();
        self.valid_model_queries = u64::from(snapshot.complete_pair().is_ok());
    }

    fn sample(&mut self) {
        let (active, cache) = (
            mlx_rs::memory::get_active_memory(),
            mlx_rs::memory::get_cache_memory(),
        );
        self.active = self.active.max(active);
        self.cache = self.cache.max(cache);
        self.combined = self.combined.max(active + cache);
        let snapshot = footprint_snapshot();
        if let Some(current) = snapshot.current() {
            self.footprint = Some(
                self.footprint
                    .map_or(current, |previous| previous.max(current)),
            );
            if self.before_model_load.is_some() {
                self.model_footprint = Some(
                    self.model_footprint
                        .map_or(current, |previous| previous.max(current)),
                );
            }
        }
        match snapshot.complete_pair() {
            Ok((_, peak)) => {
                self.lifetime_peak = Some(
                    self.lifetime_peak
                        .map_or(peak, |previous| previous.max(peak)),
                );
                self.valid_queries += 1;
                if self.before_model_load.is_some() {
                    self.valid_model_queries += 1;
                }
            }
            Err(_) => self.peak_unavailable_words = Some(snapshot.returned_words),
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
