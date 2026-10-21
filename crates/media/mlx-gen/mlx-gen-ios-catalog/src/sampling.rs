//! Provider-owned sampling replay dispatch for iOS evidence and capture. Allocation classes name
//! actual factory/input graphs; they do NOT register a component-additivity or residue proof.
use mlx_gen::gen_core::GenerationRequest;
use mlx_gen::sampler::flow_sampling::FlowSamplingRecipe;
use mlx_gen::{Error, Result};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct SamplingIdentity {
    token: String,
    allocation_class: String,
    steps: u32,
    provenance: Value,
}
impl SamplingIdentity {
    pub fn token(&self) -> &str {
        &self.token
    }
    pub fn allocation_class(&self) -> &str {
        &self.allocation_class
    }
    pub fn steps(&self) -> u32 {
        self.steps
    }
    pub fn provenance(&self) -> &Value {
        &self.provenance
    }
}

/// Registered generation routes require a reproducible sampling identity; task-only Iris routes
/// use their separately bound task execution rather than an invented generation recipe.
pub fn requires_generation_recipe(provider: &str) -> bool {
    matches!(
        provider,
        crate::sdxl::SDXL_ID
            | crate::anima::ANIMA_TURBO_ID
            | "sana_1600m"
            | "sana_sprint_1600m"
            | "iris_3b"
    ) || crate::zimage::is_z_image(provider)
}

pub fn requires_recipe(provider: &str) -> bool {
    requires_generation_recipe(provider)
        || matches!(
            provider,
            crate::ios_admission::IRIS_RESTORE_ID | crate::ios_admission::IRIS_DEPTH_ID
        )
}

fn unsupported(provider: &str) -> Error {
    Error::Unsupported(format!(
        "{provider}: no resolved generation sampling adapter"
    ))
}
fn value<T: serde::Serialize>(recipe: &T) -> Result<Value> {
    serde_json::to_value(recipe).map_err(|e| Error::Msg(e.to_string()))
}
fn safe_token(prefix: &str, raw: &str) -> String {
    format!("{prefix}{}", crate::hex(raw.as_bytes()))
}
/// An Iris payload-free anchor's recorded native replay token, parsed exactly as
/// [`price_metadata`] parses it for the same provider.
pub(crate) enum IrisAnchorReplay {
    Generation(mlx_gen_iris::ios_pricing::SamplingToken),
    Restoration(mlx_gen_iris::ios_pricing::RestorationToken),
    Depth(mlx_gen_iris::ios_pricing::DepthToken),
}

impl IrisAnchorReplay {
    pub(crate) fn parse(provider: &str, token: &str) -> Result<Self> {
        Ok(match provider {
            "iris_3b" => Self::Generation(mlx_gen_iris::ios_pricing::SamplingToken::from_token(
                &native_token("iris1-", token)?,
            )?),
            crate::ios_admission::IRIS_RESTORE_ID => {
                Self::Restoration(mlx_gen_iris::ios_pricing::RestorationToken::from_token(
                    &native_token("iris-restore1-", token)?,
                )?)
            }
            crate::ios_admission::IRIS_DEPTH_ID => {
                Self::Depth(mlx_gen_iris::ios_pricing::DepthToken::from_token(
                    &native_token("iris-depth1-", token)?,
                )?)
            }
            _ => return Err(unsupported(provider)),
        })
    }

    pub(crate) fn as_replay(&self) -> mlx_gen_iris::ios_pricing::IrisReplay<'_> {
        match self {
            Self::Generation(token) => mlx_gen_iris::ios_pricing::IrisReplay::Generation(token),
            Self::Restoration(token) => mlx_gen_iris::ios_pricing::IrisReplay::Restoration(token),
            Self::Depth(token) => mlx_gen_iris::ios_pricing::IrisReplay::Depth(token),
        }
    }
}

fn native_token(prefix: &str, token: &str) -> Result<String> {
    let hex = token
        .strip_prefix(prefix)
        .ok_or_else(|| Error::Unsupported("wrong execution recipe prefix".into()))?;
    if hex.len() > 16384
        || hex.len() % 2 != 0
        || !hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Error::Unsupported(
            "invalid bounded SAFE execution recipe".into(),
        ));
    }
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|e| Error::Msg(e.to_string())))
        .collect::<Result<Vec<_>>>()?;
    String::from_utf8(bytes).map_err(|e| Error::Msg(e.to_string()))
}
fn iris_identity(token: mlx_gen_iris::ios_pricing::SamplingToken) -> Result<SamplingIdentity> {
    let class = token.memory_class()?;
    Ok(SamplingIdentity {
        token: safe_token("iris1-", token.as_str()),
        steps: token.steps()?,
        allocation_class: format!(
            "iris1-{}",
            serde_json::json!({"order":class.order,"clean_prediction":class.clean_prediction,
            "prompt_rows":class.prompt_rows,"serial_count":class.serial_count,"epoch_runs":class.epoch_runs})
        ),
        provenance: token.to_value()?,
    })
}

/// Actual configured normalization. Iris defaults/flow f64 metadata come from the REAL artifact,
/// not a fabricated architecture/default config. Other providers own their fixed profile defaults.
pub fn resolve_for_spec(
    provider: &str,
    spec: &mlx_gen::LoadSpec,
    request: &GenerationRequest,
) -> Result<SamplingIdentity> {
    if provider == "iris_3b" {
        let config = mlx_gen_iris::ios_pricing::generation_metadata_config(spec)?;
        return iris_identity(mlx_gen_iris::ios_pricing::canonical_token(
            &config, request,
        )?);
    }
    resolve(provider, request)
}

pub fn resolve_transform(
    provider: &str,
    spec: &mlx_gen::LoadSpec,
    request: &mlx_gen::TransformRequest,
) -> Result<SamplingIdentity> {
    if provider != crate::ios_admission::IRIS_RESTORE_ID {
        return Err(unsupported(provider));
    }
    let token = mlx_gen_iris::ios_pricing::canonical_restoration_token(spec, request)?;
    from_token(provider, &safe_token("iris-restore1-", token.as_str()))
}
pub fn resolve_depth(
    provider: &str,
    spec: &mlx_gen::LoadSpec,
    request: &mlx_gen::gen_core::iris::depth::DepthRequest,
) -> Result<SamplingIdentity> {
    if provider != crate::ios_admission::IRIS_DEPTH_ID {
        return Err(unsupported(provider));
    }
    let token = mlx_gen_iris::ios_pricing::canonical_depth_token(spec, request)?;
    from_token(provider, &safe_token("iris-depth1-", token.as_str()))
}

// Provider-private work envelope. The shared flow1 wire and public DTO/signatures stay unchanged.
const Z_PID_WORK_PREFIX: &str = "zpidwork1-";
// Existing Flow decoder accepts at most16384 hex bytes after its6-byte prefix. Canonical tuple
// [1,"flow1-...",u64] adds7 punctuation/version bytes and at most20 decimal digits. This bounds
// malformed wire allocation, NOT a prompt-length/capability limit.
const Z_PID_WORK_MAX_HEX: usize = 2 * (16_384 + 6 + 7 + 20);
fn z_pid_provider(provider: &str) -> bool {
    matches!(provider, "z_image_turbo" | "z_image")
}
fn z_pid_work_wire(flow_token: &str, prompt_utf8_bytes: u64) -> Result<String> {
    let wire = serde_json::to_vec(&(1u8, flow_token, prompt_utf8_bytes))
        .map_err(|error| Error::Msg(error.to_string()))?;
    Ok(format!("{Z_PID_WORK_PREFIX}{}", crate::hex(&wire)))
}
fn z_pid_pricing_token(
    provider: &str,
    request: &GenerationRequest,
    flow_token: &str,
) -> Result<String> {
    if !z_pid_provider(provider) || !request.use_pid {
        return Ok(flow_token.to_owned());
    }
    // The ACTUAL request's original UTF-8 bytes; never chars/tokens, a synthetic caption,
    // the configured tokenizer limit, a native/OS budget, or a reconstructed metadata DTO.
    let bytes =
        u64::try_from(request.prompt.len()).map_err(|error| Error::Msg(error.to_string()))?;
    z_pid_work_wire(flow_token, bytes)
}
fn z_flow_recipe(provider: &str, token: &str) -> Result<(FlowSamplingRecipe, Option<u64>)> {
    let Some(hex) = token.strip_prefix(Z_PID_WORK_PREFIX) else {
        return Ok((FlowSamplingRecipe::from_token(token)?, None));
    };
    if !z_pid_provider(provider) {
        return Err(Error::Unsupported(
            "PiD work envelope is not a Control/foreign-provider route".into(),
        ));
    }
    if hex.len() > Z_PID_WORK_MAX_HEX
        || hex.len() % 2 != 0
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(Error::Unsupported(
            "invalid bounded provider-private PiD work token".into(),
        ));
    }
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&hex[index..index + 2], 16)
                .map_err(|error| Error::Msg(error.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let (version, flow_token, prompt_utf8_bytes): (u8, String, u64) =
        serde_json::from_slice(&bytes)
            .map_err(|error| Error::Msg(format!("invalid private PiD work tuple: {error}")))?;
    if version != 1 {
        return Err(Error::Unsupported(
            "unsupported provider-private PiD work version".into(),
        ));
    }
    let recipe = FlowSamplingRecipe::from_token(&flow_token)?;
    let value = value(&recipe)?;
    if value.get("use_pid").and_then(Value::as_bool) != Some(true) {
        return Err(Error::Unsupported(
            "PiD work envelope contains an inactive flow recipe".into(),
        ));
    }
    if z_pid_work_wire(&recipe.canonical_token(), prompt_utf8_bytes)? != token {
        return Err(Error::Unsupported(
            "noncanonical provider-private PiD work token".into(),
        ));
    }
    Ok((recipe, Some(prompt_utf8_bytes)))
}

fn flow_identity(
    plan: mlx_gen::sampler::flow_sampling::ResolvedFlowControls,
) -> Result<SamplingIdentity> {
    Ok(SamplingIdentity {
        token: plan.recipe().canonical_token(),
        allocation_class: plan.allocation_class(),
        steps: u32::try_from(plan.steps()).map_err(|e| Error::Msg(e.to_string()))?,
        provenance: value(plan.recipe())?,
    })
}

/// Resolve the actual request through the SAME provider factory metadata used by execution.
/// Unknown generation providers fail closed; they are never assigned a neutral class.
pub fn resolve(provider: &str, request: &GenerationRequest) -> Result<SamplingIdentity> {
    match provider {
        crate::sdxl::SDXL_ID => {
            let plan = mlx_gen_sdxl::sampling::ResolvedSampling::resolve(request)?;
            Ok(SamplingIdentity {
                token: plan.recipe().canonical_token(),
                allocation_class: plan.allocation_class(),
                steps: u32::try_from(plan.steps()).map_err(|e| Error::Msg(e.to_string()))?,
                provenance: plan.recipe().to_value(),
            })
        }
        crate::anima::ANIMA_TURBO_ID => flow_identity(mlx_gen_anima::sampling::resolve(request)?),
        id if crate::zimage::is_z_image(id) => {
            let mut identity = flow_identity(mlx_gen_z_image::sampling::resolve(request)?)?;
            identity.token = z_pid_pricing_token(provider, request, identity.token())?;
            Ok(identity)
        }
        "sana_1600m" => flow_identity(mlx_gen_sana::sampling::resolve_base(request)?),
        "sana_sprint_1600m" => {
            let plan = mlx_gen_sana::sampling::resolve_sprint(request)?;
            Ok(SamplingIdentity {
                token: plan.recipe().canonical_token(),
                allocation_class: plan.allocation_class().into(),
                steps: u32::try_from(plan.steps()).map_err(|e| Error::Msg(e.to_string()))?,
                provenance: value(plan.recipe())?,
            })
        }
        _ => Err(unsupported(provider)),
    }
}

/// Decode typed, canonical replay controls; class identity is derived through the actual provider,
/// never accepted from a caller-supplied class string. No image payload is fabricated here.
pub fn from_token(provider: &str, token: &str) -> Result<SamplingIdentity> {
    if provider == "iris_3b" {
        let raw = native_token("iris1-", token)?;
        let identity = iris_identity(mlx_gen_iris::ios_pricing::SamplingToken::from_token(&raw)?)?;
        if identity.token() != token {
            return Err(Error::Unsupported("noncanonical SAFE Iris token".into()));
        }
        return Ok(identity);
    }
    if provider == crate::ios_admission::IRIS_RESTORE_ID {
        let raw = native_token("iris-restore1-", token)?;
        let native = mlx_gen_iris::ios_pricing::RestorationToken::from_token(&raw)?;
        let class = native.memory_class()?;
        return Ok(SamplingIdentity {
            token: safe_token("iris-restore1-", native.as_str()),
            steps: 1,
            allocation_class: format!(
                "iris-restore1-{}",
                serde_json::json!({"budget_applied":class.budget_applied,
                "enlarged":class.enlarged,"single_pass":class.single_pass,"color_fix":class.color_fix})
            ),
            provenance: native.to_value()?,
        });
    }
    if provider == crate::ios_admission::IRIS_DEPTH_ID {
        let raw = native_token("iris-depth1-", token)?;
        let native = mlx_gen_iris::ios_pricing::DepthToken::from_token(&raw)?;
        return Ok(SamplingIdentity {
            token: safe_token("iris-depth1-", native.as_str()),
            steps: 1,
            allocation_class: format!("iris-depth1-resized{}", native.memory_class()?.resized),
            provenance: native.to_value()?,
        });
    }
    let (request, provenance) = match provider {
        crate::sdxl::SDXL_ID => {
            let recipe = mlx_gen_sdxl::sampling::SamplingRecipe::from_token(token)?;
            (recipe.to_request(), recipe.to_value())
        }
        crate::anima::ANIMA_TURBO_ID | "sana_1600m" => {
            let recipe = FlowSamplingRecipe::from_token(token)?;
            (recipe.to_request(), value(&recipe)?)
        }
        id if crate::zimage::is_z_image(id) => {
            let (recipe, _preserved_basis_in_outer_token) = z_flow_recipe(provider, token)?;
            // Scalar controls only. This projected request has no real caption and MUST NOT
            // supply an actual PiD work basis. Preserve the caller's validated outer token below.
            (recipe.to_request(), value(&recipe)?)
        }
        "sana_sprint_1600m" => {
            let recipe = mlx_gen_sana::sampling::ScmSamplingRecipe::from_token(token)?;
            (recipe.to_request(), value(&recipe)?)
        }
        _ => return Err(unsupported(provider)),
    };
    let mut identity = if crate::zimage::is_z_image(provider) {
        // Direct provider factory normalization, not resolve()'s ACTUAL-request envelope path.
        flow_identity(mlx_gen_z_image::sampling::resolve(&request)?)?
    } else {
        resolve(provider, &request)?
    };
    let mut actual_semantics = identity.provenance.clone();
    let mut supplied_semantics = provenance.clone();
    for value in [&mut actual_semantics, &mut supplied_semantics] {
        let object = value
            .as_object_mut()
            .ok_or_else(|| Error::Unsupported("sampling recipe must be an object".into()))?;
        object.remove("strength");
        object.remove("reference_strength");
    }
    if actual_semantics != supplied_semantics {
        return Err(Error::Unsupported(
            "sampling recipe does not match the provider's actual normalized semantics".into(),
        ));
    }
    // to_request has no media carriers and projects effective reference strength for schedule
    // construction. Preserve the validated original raw/global/reference controls in provenance.
    identity.token = token.into();
    identity.provenance = provenance;
    Ok(identity)
}

/// Apply only replay controls AFTER the actual mode/media carrier exists. The normalized result
/// must retain the whole recipe, so a missing Reference cannot silently discard its override.
fn write_controls(provider: &str, token: &str, request: &mut GenerationRequest) -> Result<()> {
    match provider {
        crate::sdxl::SDXL_ID => {
            mlx_gen_sdxl::sampling::SamplingRecipe::from_token(token)?.apply_to(request)?
        }
        crate::anima::ANIMA_TURBO_ID | "sana_1600m" => {
            FlowSamplingRecipe::from_token(token)?.apply_to(request)?
        }
        id if crate::zimage::is_z_image(id) => {
            // Only the inner flow controls reach the solver; no prompt is fabricated from basis.
            z_flow_recipe(provider, token)?.0.apply_to(request)?
        }
        "sana_sprint_1600m" => {
            mlx_gen_sana::sampling::ScmSamplingRecipe::from_token(token)?.apply_to(request)?
        }
        _ => return Err(unsupported(provider)),
    }
    Ok(())
}

pub fn apply(provider: &str, token: &str, request: &mut GenerationRequest) -> Result<()> {
    if crate::zimage::is_z_image(provider) {
        if let Some(expected) = z_flow_recipe(provider, token)?.1 {
            let actual = u64::try_from(request.prompt.len())
                .map_err(|error| Error::Msg(error.to_string()))?;
            if actual != expected {
                return Err(Error::Unsupported(
                    "PiD strict replay work basis differs from the actual request prompt bytes"
                        .into(),
                ));
            }
        }
    }
    write_controls(provider, token, request)?;
    if resolve(provider, request)?.token() != token {
        return Err(Error::Unsupported(
            "sampling replay does not match the actual mode/controls".into(),
        ));
    }
    Ok(())
}

/// Build a MODE representative from factory controls without erasing that actual carrier's init
/// strength (or IP scale). This is construction, NOT replay: return its new canonical recipe.
/// Strict evidence/anchor replay always uses `apply` instead.
pub fn apply_representative(
    provider: &str,
    token: &str,
    request: &mut GenerationRequest,
) -> Result<SamplingIdentity> {
    let global_strength = request.strength;
    let strengths: Vec<_> = request
        .conditioning
        .iter()
        .filter_map(|c| match c {
            mlx_gen::gen_core::Conditioning::Reference { strength, .. } => Some(*strength),
            _ => None,
        })
        .collect();
    write_controls(provider, token, request)?;
    request.strength = global_strength;
    let mut strengths = strengths.into_iter();
    for conditioning in &mut request.conditioning {
        if let mlx_gen::gen_core::Conditioning::Reference { strength, .. } = conditioning {
            *strength = strengths.next().ok_or_else(|| {
                Error::Unsupported("representative lost a reference carrier".into())
            })?;
        }
    }
    if strengths.next().is_some() {
        return Err(Error::Unsupported(
            "representative changed reference cardinality".into(),
        ));
    }
    resolve(provider, request)
}

/// Factory representatives, not a scheduler/step/float measurement grid. Unsupported combinations
/// are checked against actual mode carriers by capture, rather than erased from the registry.
pub fn representatives(provider: &str) -> Result<Vec<SamplingIdentity>> {
    let tokens: Vec<String> = match provider {
        crate::sdxl::SDXL_ID => mlx_gen_sdxl::sampling::representative_recipes()
            .into_iter()
            .map(|r| r.canonical_token())
            .collect(),
        crate::anima::ANIMA_TURBO_ID => mlx_gen_anima::sampling::representative_recipes()
            .into_iter()
            .map(|r| r.canonical_token())
            .collect(),
        id if crate::zimage::is_z_image(id) => mlx_gen_z_image::sampling::representative_recipes()
            .into_iter()
            .map(|r| r.canonical_token())
            .collect(),
        "sana_1600m" => mlx_gen_sana::sampling::representative_recipes()
            .into_iter()
            .map(|r| r.canonical_token())
            .collect(),
        "sana_sprint_1600m" => {
            vec![
                mlx_gen_sana::sampling::resolve_sprint(&GenerationRequest::default())?
                    .recipe()
                    .canonical_token(),
            ]
        }
        _ => return Err(unsupported(provider)),
    };
    tokens
        .iter()
        .map(|token| from_token(provider, token))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{Conditioning, Image};

    fn reference_request() -> GenerationRequest {
        GenerationRequest {
            steps: Some(8),
            strength: Some(0.4),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 4,
                    height: 4,
                    pixels: (0..48).collect(),
                },
                strength: Some(0.6),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn representative_preserves_actual_mode_strength_and_strict_replay_uses_its_own_controls() {
        let provider = crate::zimage::Z_IMAGE_TURBO_ID;
        let text = resolve(
            provider,
            &GenerationRequest {
                steps: Some(8),
                ..Default::default()
            },
        )
        .unwrap();
        let mut mode = reference_request();
        let actual = apply_representative(provider, text.token(), &mut mode).unwrap();
        assert_eq!(mode.strength, Some(0.4));
        let Conditioning::Reference { strength, .. } = &mode.conditioning[0] else {
            unreachable!()
        };
        assert_eq!(*strength, Some(0.6), "a text representative must not turn reference capture into start-step-zero text execution");
        assert_ne!(actual.token(), text.token());
        assert_eq!(
            from_token(provider, actual.token())
                .unwrap()
                .allocation_class(),
            actual.allocation_class()
        );
        let mut replay = reference_request();
        replay.strength = Some(0.1);
        let Conditioning::Reference { strength, .. } = &mut replay.conditioning[0] else {
            unreachable!()
        };
        *strength = Some(0.2);
        apply(provider, actual.token(), &mut replay).unwrap();
        assert_eq!(replay.strength, Some(0.4));
        let Conditioning::Reference { strength, .. } = &replay.conditioning[0] else {
            unreachable!()
        };
        assert_eq!(
            *strength,
            Some(0.6),
            "anchor replay must not inherit requesting strengths"
        );
        assert_eq!(resolve(provider, &replay).unwrap().token(), actual.token());
    }

    #[test]
    fn cfg_free_recipe_is_not_reinterpreted_as_sana_default_guidance() {
        let turbo = resolve(
            crate::zimage::Z_IMAGE_TURBO_ID,
            &GenerationRequest::default(),
        )
        .unwrap();
        assert!(from_token("sana_1600m", turbo.token()).is_err());
        let sana = resolve("sana_1600m", &GenerationRequest::default()).unwrap();
        assert!(from_token(crate::zimage::Z_IMAGE_TURBO_ID, sana.token()).is_err());
        assert!(resolve("unknown_generation_provider", &GenerationRequest::default()).is_err());
    }
}

/// Reprice sealed anchor metadata through the real provider path. No GenerationRequest, RGB,
/// prompt or task image is constructed or cloned. Original conditioning roles stay explicit;
/// provider code alone decides whether a reference is an init or an IP/control input.
pub(crate) fn price_metadata(
    req: &mlx_gen::ios_pricing::IosPricingRequest<'_>,
    token: &str,
    mask_count: u32,
    control_count: u32,
    source: Option<&mlx_gen_sdxl::sampling::SamplingSourceFacts>,
) -> Result<mlx_gen::ios_pricing::IosPricing> {
    price_metadata_using_source_capacity(req, token, mask_count, control_count, source, None)
}

/// The SAME anchor metadata pricing, with a real borrowed guard at existing Iris reads.
pub(crate) fn price_metadata_with_source_capacity(
    req: &mlx_gen::ios_pricing::IosPricingRequest<'_>,
    token: &str,
    mask_count: u32,
    control_count: u32,
    source: Option<&mlx_gen_sdxl::sampling::SamplingSourceFacts>,
    next: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<mlx_gen::ios_pricing::IosPricing> {
    price_metadata_using_source_capacity(req, token, mask_count, control_count, source, Some(next))
}

fn price_metadata_using_source_capacity(
    req: &mlx_gen::ios_pricing::IosPricingRequest<'_>,
    token: &str,
    mask_count: u32,
    control_count: u32,
    source: Option<&mlx_gen_sdxl::sampling::SamplingSourceFacts>,
    next: Option<&mut dyn FnMut(u64) -> Result<()>>,
) -> Result<mlx_gen::ios_pricing::IosPricing> {
    if req.generation.is_some() || req.transform.is_some() || req.depth.is_some() {
        return Err(Error::Unsupported(
            "anchor metadata replay rejects a competing actual work context".into(),
        ));
    }
    // Crate tests that install a synthetic pricing price the ANCHOR replay with it too, so a
    // fixture's anchor and request profiles come from the same counted law.
    #[cfg(test)]
    if crate::pricing::test_pricing_active() {
        let (pricer, _) = crate::pricing::provider_pricing(req.provider).ok_or_else(|| {
            Error::Unsupported("anchor replay under a test pricing that prices nothing".into())
        })?;
        return pricer(req);
    }
    if req.geometry.batch == 0 {
        return Err(Error::Unsupported(
            "anchor metadata replay requires positive output cardinality".into(),
        ));
    }
    if req.provider != crate::sdxl::SDXL_ID && source.is_some() {
        return Err(Error::Unsupported(
            "SDXL source context belongs to another provider".into(),
        ));
    }
    if let (true, Some(next)) = (
        matches!(
            req.provider,
            "iris_3b" | crate::ios_admission::IRIS_RESTORE_ID | crate::ios_admission::IRIS_DEPTH_ID
        ),
        next,
    ) {
        mlx_gen_iris::ios_pricing::check_replay_source_capacity_for_provider(
            req.provider,
            token,
            next,
        )?;
        let replay = IrisAnchorReplay::parse(req.provider, token)?;
        return mlx_gen_iris::ios_pricing::pricing_facts_with_source_capacity(
            req,
            Some(replay.as_replay()),
            next,
        )
        .map(|facts| facts.pricing);
    }
    match req.provider {
        crate::sdxl::SDXL_ID => {
            use mlx_gen_sdxl::sampling::{ResolvedSampling, SamplingRecipe};
            let recipe = SamplingRecipe::from_token(token)?;
            let source = source.ok_or_else(|| {
                Error::Unsupported(
                    "SDXL metadata replay lacks actual image/text source context".into(),
                )
            })?;
            source.validate_roles()?;
            let work = source.work();
            if (work.reference_count, work.mask_count, work.control_count)
                != (
                    crate::pricing::reference_role_count(
                        req.geometry.reference_count,
                        mask_count,
                        control_count,
                    )? as usize,
                    mask_count as usize,
                    control_count as usize,
                )
            {
                return Err(Error::Unsupported(
                    "SDXL source context and replay roles differ".into(),
                ));
            }
            let resolved = ResolvedSampling::resolve_recipe(&recipe, work)?;
            let mut execution = *req.execution;
            execution.cfg_batching = Some(resolved.cfg_batching());
            let replay = mlx_gen::ios_pricing::IosPricingRequest {
                execution: &execution,
                ..*req
            };
            mlx_gen_sdxl::ios_pricing::ios_pricing_with_source(&replay, &recipe, source)
        }
        crate::anima::ANIMA_TURBO_ID => {
            mlx_gen_anima::ios_pricing::ios_pricing_with_token(req, token, req.geometry.batch)
        }
        id if crate::zimage::is_z_image(id) => {
            let references = crate::pricing::reference_role_count(
                req.geometry.reference_count,
                mask_count,
                control_count,
            )?;
            mlx_gen_z_image::ios_pricing::ios_pricing_with_token(
                req,
                token,
                req.geometry.batch,
                references,
                mask_count,
                control_count,
            )
        }
        "sana_1600m" | "sana_sprint_1600m" => {
            mlx_gen_sana::ios_pricing::ios_pricing_with_token(req, token, req.geometry.batch)
        }
        "iris_3b" => {
            let raw = native_token("iris1-", token)?;
            let token = mlx_gen_iris::ios_pricing::SamplingToken::from_token(&raw)?;
            mlx_gen_iris::ios_pricing::ios_pricing_with_token(req, &token)
        }
        crate::ios_admission::IRIS_RESTORE_ID => {
            let raw = native_token("iris-restore1-", token)?;
            let token = mlx_gen_iris::ios_pricing::RestorationToken::from_token(&raw)?;
            Ok(
                mlx_gen_iris::ios_pricing::pricing_facts_with_restoration_token(req, &token)?
                    .pricing,
            )
        }
        crate::ios_admission::IRIS_DEPTH_ID => {
            let raw = native_token("iris-depth1-", token)?;
            let token = mlx_gen_iris::ios_pricing::DepthToken::from_token(&raw)?;
            Ok(mlx_gen_iris::ios_pricing::pricing_facts_with_depth_token(req, &token)?.pricing)
        }
        _ => Err(unsupported(req.provider)),
    }
}

/// Unresolved Iris controls are visible templates, never a guessed checkpoint/configuration or
/// allocation class. Only `apply_for_spec` may turn one into the actual configured identity.
pub fn iris_raw_templates(carrier: &GenerationRequest) -> Result<Vec<String>> {
    mlx_gen_iris::ios_pricing::sampling_representative_templates(carrier)
        .into_iter()
        .map(|template| {
            template
                .to_token()
                .map(|raw| safe_token("iris-template1-", &raw))
        })
        .collect()
}

/// Configured Iris representatives use the actual provider's indexed plan and exact checkpoint
/// defaults. They do not reuse an unresolved/default-shift class or construct media carriers.
pub fn iris_configured_representatives(
    spec: &mlx_gen::LoadSpec,
    carrier: &mut GenerationRequest,
) -> Result<Vec<SamplingIdentity>> {
    let config = mlx_gen_iris::ios_pricing::generation_metadata_config(spec)?;
    mlx_gen_iris::ios_pricing::sampling_representatives_in_place(&config, carrier)?
        .into_iter()
        .map(iris_identity)
        .collect()
}

/// Apply captured controls to the actual owned request with the real provisioned configuration.
/// No payload/request clone, metadata-config override, or caller-claimed class authorizes work.
pub fn apply_for_spec(
    provider: &str,
    spec: &mlx_gen::LoadSpec,
    token: &str,
    request: &mut GenerationRequest,
) -> Result<SamplingIdentity> {
    if provider != "iris_3b" {
        apply(provider, token, request)?;
        return resolve_for_spec(provider, spec, request);
    }
    let config = mlx_gen_iris::ios_pricing::generation_metadata_config(spec)?;
    if token.starts_with("iris-template1-") {
        let raw = native_token("iris-template1-", token)?;
        let template = mlx_gen_iris::ios_pricing::SamplingTemplate::from_token(&raw)?;
        return iris_identity(template.apply_to(&config, request)?);
    }
    let raw = native_token("iris1-", token)?;
    let captured = mlx_gen_iris::ios_pricing::SamplingToken::from_token(&raw)?;
    captured.apply_runtime_to(&config, request)?;
    iris_identity(mlx_gen_iris::ios_pricing::canonical_token(
        &config, request,
    )?)
}

pub fn restoration_configured_representatives(
    spec: &mlx_gen::LoadSpec,
    carrier: &mlx_gen::gen_core::TransformRequest,
) -> Result<Vec<Result<SamplingIdentity>>> {
    Ok(
        mlx_gen_iris::ios_pricing::restoration_representatives(spec, carrier)?
            .into_iter()
            .map(|token| {
                let token = token?;
                from_token(
                    crate::ios_admission::IRIS_RESTORE_ID,
                    &safe_token("iris-restore1-", token.as_str()),
                )
            })
            .collect(),
    )
}
pub fn depth_configured_representatives(
    spec: &mlx_gen::LoadSpec,
    carrier: &mlx_gen::gen_core::iris::depth::DepthRequest,
) -> Result<Vec<Result<SamplingIdentity>>> {
    Ok(
        mlx_gen_iris::ios_pricing::depth_representatives(spec, carrier)?
            .into_iter()
            .map(|token| {
                let token = token?;
                from_token(
                    crate::ios_admission::IRIS_DEPTH_ID,
                    &safe_token("iris-depth1-", token.as_str()),
                )
            })
            .collect(),
    )
}
pub fn apply_transform_for_spec(
    spec: &mlx_gen::LoadSpec,
    token: &str,
    request: &mut mlx_gen::gen_core::TransformRequest,
) -> Result<SamplingIdentity> {
    let raw = native_token("iris-restore1-", token)?;
    mlx_gen_iris::ios_pricing::RestorationToken::from_token(&raw)?
        .apply_runtime_to(spec, request)?;
    resolve_transform(crate::ios_admission::IRIS_RESTORE_ID, spec, request)
}
pub fn apply_depth_for_spec(
    spec: &mlx_gen::LoadSpec,
    token: &str,
    request: &mut mlx_gen::gen_core::iris::depth::DepthRequest,
) -> Result<SamplingIdentity> {
    let raw = native_token("iris-depth1-", token)?;
    mlx_gen_iris::ios_pricing::DepthToken::from_token(&raw)?.apply_runtime_to(spec, request)?;
    resolve_depth(crate::ios_admission::IRIS_DEPTH_ID, spec, request)
}

#[cfg(test)]
mod z_pid_work_token_tests {
    use super::*;
    use crate::ios_admission::IosWork;

    fn request(prompt: &str, use_pid: bool) -> GenerationRequest {
        GenerationRequest {
            prompt: prompt.into(),
            use_pid,
            steps: Some(4),
            ..Default::default()
        }
    }
    fn independent_wire(flow: &str, bytes: u64) -> String {
        let tuple = serde_json::to_vec(&(1u8, flow, bytes)).unwrap();
        let lower = tuple
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        format!("zpidwork1-{lower}")
    }
    #[test]
    fn actual_utf8_basis_roundtrips_without_treating_recipe_projection_as_real_prompt() {
        let req = request("é界", true);
        let flow = mlx_gen_z_image::sampling::resolve(&req)
            .unwrap()
            .recipe()
            .canonical_token();
        let identity = resolve(crate::zimage::Z_IMAGE_TURBO_ID, &req).unwrap();
        assert_eq!(
            identity.token(),
            independent_wire(&flow, 5),
            "ACTUAL_UTF8_NOT_CHAR_OR_PADDED_TOKEN_COUNT"
        );
        let (recipe, basis) =
            z_flow_recipe(crate::zimage::Z_IMAGE_TURBO_ID, identity.token()).unwrap();
        assert_eq!(recipe.canonical_token(), flow);
        assert_eq!(basis, Some(5));
        let replay = from_token(crate::zimage::Z_IMAGE_TURBO_ID, identity.token()).unwrap();
        assert_eq!(
            replay.token(),
            identity.token(),
            "OUTER_BASIS_NOT_REENCODED_FROM_EMPTY_SCALAR_DTO"
        );
        assert_eq!(replay.allocation_class(), identity.allocation_class());
        assert_eq!(replay.provenance(), identity.provenance());
        let maximum = independent_wire(&flow, u64::MAX);
        assert_eq!(
            from_token(crate::zimage::Z_IMAGE_TURBO_ID, &maximum)
                .unwrap()
                .token(),
            maximum,
            "METADATA_BASIS_PRESERVED_NOT_INVENTED_OR_CLIPPED"
        );
    }
    #[test]
    fn strict_apply_checks_real_bytes_and_full_work_binding_still_checks_content() {
        let provider = crate::zimage::Z_IMAGE_TURBO_ID;
        let mut actual = request("é界", true);
        let token = resolve(provider, &actual).unwrap().token().to_owned();
        apply(provider, &token, &mut actual).unwrap();
        let mut wrong = actual.clone();
        wrong.prompt = "four".into();
        let before_steps = wrong.steps;
        assert!(
            apply(provider, &token, &mut wrong).is_err(),
            "STRICT_REPLAY_CANNOT_INVENT_PROMPT_FROM_LENGTH"
        );
        assert_eq!(wrong.prompt, "four");
        assert_eq!(wrong.steps, before_steps);
        let mut same_bytes = actual.clone();
        same_bytes.prompt = "abcde".into();
        apply(provider, &token, &mut same_bytes).unwrap();
        assert_ne!(
            crate::binding::work_binding(&IosWork::Generate(actual)),
            crate::binding::work_binding(&IosWork::Generate(same_bytes)),
            "BYTE_BASIS_IS_NOT_PROMPT_CONTENT_OR_LIFETIME_IDENTITY"
        );
    }
    #[test]
    fn representative_mints_its_actual_work_basis_but_legacy_replay_does_not_invent_one() {
        let provider = crate::zimage::Z_IMAGE_TURBO_ID;
        let source = request("a", true);
        let old = resolve(provider, &source).unwrap().token().to_owned();
        let mut target = request("é界", false);
        let constructed = apply_representative(provider, &old, &mut target).unwrap();
        assert_eq!(
            z_flow_recipe(provider, constructed.token()).unwrap().1,
            Some(5)
        );
        assert_eq!(target.prompt, "é界");
        let legacy = mlx_gen_z_image::sampling::resolve(&source)
            .unwrap()
            .recipe()
            .canonical_token();
        assert_eq!(
            z_flow_recipe(provider, &legacy).unwrap().1,
            None,
            "LEGACY_MISSING_IS_NOT_SOME_ZERO"
        );
        assert_eq!(from_token(provider, &legacy).unwrap().token(), legacy);
        assert!(
            apply(provider, &legacy, &mut target).is_err(),
            "STRICT_ACTIVE_PID_REPLAY_NEEDS_REAL_BASIS"
        );
        let empty = resolve(provider, &request("", true)).unwrap();
        assert_eq!(
            z_flow_recipe(provider, empty.token()).unwrap().1,
            Some(0),
            "ZERO_ONLY_FROM_ACTUAL_EMPTY_REQUEST"
        );
    }
    #[test]
    fn non_pid_control_and_existing_flow_semantics_keep_exact_old_tokens() {
        for provider in [
            crate::zimage::Z_IMAGE_TURBO_ID,
            crate::zimage::Z_IMAGE_TURBO_CONTROL_ID,
        ] {
            for active in [false, true] {
                let req = request("actual prompt", active);
                let flow = mlx_gen_z_image::sampling::resolve(&req)
                    .unwrap()
                    .recipe()
                    .canonical_token();
                if provider == crate::zimage::Z_IMAGE_TURBO_ID && active {
                    continue;
                }
                assert_eq!(resolve(provider, &req).unwrap().token(), flow);
                assert_eq!(from_token(provider, &flow).unwrap().token(), flow);
            }
        }
        let req = request("é界", true);
        let wrapped = resolve(crate::zimage::Z_IMAGE_TURBO_ID, &req).unwrap();
        assert!(
            from_token(crate::zimage::Z_IMAGE_TURBO_CONTROL_ID, wrapped.token()).is_err(),
            "OPAQUE_WORK_NAMESPACE_CANNOT_CREATE_CONTROL_PID_ROUTE"
        );
    }
    #[test]
    fn private_wire_is_canonical_bounded_and_does_not_cap_the_prompt_dimension() {
        let provider = crate::zimage::Z_IMAGE_TURBO_ID;
        let req = request("é界", true);
        let flow = mlx_gen_z_image::sampling::resolve(&req)
            .unwrap()
            .recipe()
            .canonical_token();
        let valid = independent_wire(&flow, u64::MAX);
        assert_eq!(z_flow_recipe(provider, &valid).unwrap().1, Some(u64::MAX));
        let version2 = serde_json::to_vec(&(2u8, &flow, 5u64)).unwrap();
        assert!(z_flow_recipe(provider, &format!("zpidwork1-{}", crate::hex(&version2))).is_err());
        let extra = serde_json::to_vec(&(1u8, &flow, 5u64, 9u64)).unwrap();
        assert!(z_flow_recipe(provider, &format!("zpidwork1-{}", crate::hex(&extra))).is_err());
        assert!(z_flow_recipe(provider, "zpidwork1-a").is_err());
        assert!(z_flow_recipe(provider, "zpidwork1-AA").is_err());
        let huge = format!("zpidwork1-{}", "0".repeat(Z_PID_WORK_MAX_HEX + 2));
        assert!(
            z_flow_recipe(provider, &huge).is_err(),
            "OUTER_BOUND_PRECEDES_VEC_AND_SERDE"
        );
        let inactive = mlx_gen_z_image::sampling::resolve(&request("actual", false))
            .unwrap()
            .recipe()
            .canonical_token();
        assert!(z_flow_recipe(provider, &independent_wire(&inactive, 6)).is_err());
    }
}
