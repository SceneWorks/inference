//! Link-time provider registry, id-based routing, and **model-first** resolution.
//!
//! Backends register a provider with [`inventory::submit!`]; consumers discover and load providers
//! by id ([`load_textllm`]) without a central match statement (additive, like the mlx-gen
//! registries). A registration stores the descriptor constructor separately from `load`, so the
//! registry can be introspected cheaply without loading any weights.
//!
//! On top of id-based routing this module adds **[`load_for_model`]**: a caller hands over a
//! [`LoadSpec`] (a snapshot path) and gets a working [`TextLlm`] back, naming no provider id,
//! family, or backend. The resolver does NOT match on [`TextLlmDescriptor::family`] — a generic
//! backend provider serves many architectures behind one registration whose *static* family is just
//! `"llama"` (the true family is only known post-load). Instead each registration carries a cheap,
//! **weightless** [`can_load`](TextLlmRegistration::can_load) probe that reads only the snapshot's
//! `config.json`; the resolver picks the first registered provider whose probe accepts the model and
//! whose declared capabilities meet the caller's [`ModelRequirements`].

use crate::capabilities::{TextLlmCapabilities, TextLlmDescriptor};
use crate::constraint::Constraint;
use crate::error::{Error, Result};
use crate::request::{LoadSpec, TextLlmRequest};
use crate::text_llm::TextLlm;

/// A registered provider: how to describe it (cheap), whether it can serve a given model (cheap,
/// weightless), and how to load it (loads weights).
pub struct TextLlmRegistration {
    /// Build the provider's descriptor without loading weights.
    pub descriptor: fn() -> TextLlmDescriptor,
    /// Load an instance from a [`LoadSpec`].
    pub load: fn(&LoadSpec) -> Result<Box<dyn TextLlm>>,
    /// **Weightless** probe: can this provider serve the model at `spec.source`? Implemented by the
    /// backend by running its own architecture dispatch (`Architecture::from_config` or equivalent)
    /// over the snapshot's `config.json` — it MUST NOT read safetensors / weight shards. Drives
    /// [`load_for_model`]; the architecture knowledge stays in the backend, never in `core-llm`.
    pub can_load: fn(&LoadSpec) -> bool,
}

inventory::collect!(TextLlmRegistration);

/// What a caller needs a provider to support, used to disambiguate when more than one registered
/// provider accepts a model (e.g. a text and a vision provider both serve a multimodal snapshot).
/// The [`Default`] is "no special needs": any architecture-matching provider qualifies.
#[derive(Clone, Debug, Default)]
pub struct ModelRequirements {
    /// The provider must accept image (vision) input.
    pub vision: bool,
    /// The provider must be able to enforce each of these output constraints.
    pub constraints: Vec<Constraint>,
}

impl ModelRequirements {
    /// Derive the requirements implied by a concrete request: vision if any message carries an
    /// image, plus the request's output constraint (if any). This is the bridge the worker uses —
    /// `load_for_model_with(spec, &ModelRequirements::from_request(req))`.
    pub fn from_request(req: &TextLlmRequest) -> Self {
        Self {
            vision: req.has_image(),
            constraints: req.constraint.iter().copied().collect(),
        }
    }

    /// Require image (vision) input support.
    pub fn with_vision(mut self) -> Self {
        self.vision = true;
        self
    }

    /// Require support for an output constraint (e.g. [`Constraint::Json`]).
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        if !self.constraints.contains(&constraint) {
            self.constraints.push(constraint);
        }
        self
    }
}

/// Iterate every registered provider (link-time collected).
pub fn textllms() -> impl Iterator<Item = &'static TextLlmRegistration> {
    inventory::iter::<TextLlmRegistration>.into_iter()
}

/// Look up a registered provider by its descriptor id.
pub fn find(id: &str) -> Option<&'static TextLlmRegistration> {
    textllms().find(|r| (r.descriptor)().id == id)
}

/// Load a provider by id. First-wins on duplicate ids (a `debug_assert!` flags the collision).
pub fn load_textllm(id: &str, spec: &LoadSpec) -> Result<Box<dyn TextLlm>> {
    let mut matches = textllms().filter(|r| (r.descriptor)().id == id);
    let reg = matches
        .next()
        .ok_or_else(|| Error::Msg(format!("no textllm registered for id '{id}'")))?;
    debug_assert!(
        matches.next().is_none(),
        "duplicate textllm id '{id}' registered (first-wins shadows the rest)"
    );
    (reg.load)(spec)
}

/// **Model-first** load: select the registered provider that can serve the model at `spec.source`
/// and load it — naming no provider id, family, or backend. Equivalent to
/// [`load_for_model_with`] with default (no-special-needs) requirements.
///
/// Resolution reads only `config.json` (tensor-free) via each provider's weightless
/// [`can_load`](TextLlmRegistration::can_load) probe; an unknown / unsupported architecture yields a
/// clear [`Error::Unsupported`] naming the model and the linked providers — never a panic, never a
/// silent default.
///
/// ```ignore
/// // The caller links exactly one backend crate (e.g. mlx-llm) and switches models by path alone:
/// let llm = core_llm::load_for_model(&core_llm::LoadSpec::dense("/models/qwen3-0.6b"))?;
/// ```
pub fn load_for_model(spec: &LoadSpec) -> Result<Box<dyn TextLlm>> {
    load_for_model_with(spec, &ModelRequirements::default())
}

/// [`load_for_model`] with explicit capability requirements used to break ties when more than one
/// registered provider accepts the model (vision requested ⇒ the vision-capable provider; a JSON
/// constraint requested ⇒ a provider whose `supported_constraints` includes it).
pub fn load_for_model_with(spec: &LoadSpec, reqs: &ModelRequirements) -> Result<Box<dyn TextLlm>> {
    let reg = select(textllms(), spec, reqs)?;
    (reg.load)(spec)
}

/// Resolve the registration to load: architecture match (`can_load`) first, then a capability
/// filter, then a deterministic tie-break (prefer a non-vision provider when vision is not
/// requested; otherwise first-registered). Pure over the supplied registrations so it is unit
/// testable without the global inventory.
fn select<'a>(
    regs: impl Iterator<Item = &'a TextLlmRegistration>,
    spec: &LoadSpec,
    reqs: &ModelRequirements,
) -> Result<&'a TextLlmRegistration> {
    let all: Vec<&TextLlmRegistration> = regs.collect();

    // 1. Architecture match: the weightless probe accepts the snapshot.
    let accepting: Vec<&TextLlmRegistration> =
        all.iter().copied().filter(|r| (r.can_load)(spec)).collect();
    if accepting.is_empty() {
        return Err(Error::Unsupported(no_provider_msg(spec, &all)));
    }

    // 2. Capability filter: keep only providers whose declared capabilities meet the request surface.
    let viable: Vec<&TextLlmRegistration> = accepting
        .iter()
        .copied()
        .filter(|r| meets(&(r.descriptor)().capabilities, reqs))
        .collect();
    if viable.is_empty() {
        return Err(Error::Unsupported(unmet_caps_msg(spec, reqs, &accepting)));
    }

    // 3. Tie-break. When vision was not requested and several providers match, prefer a text
    //    (non-vision) provider so a plain load never hands back a model that expects an image;
    //    otherwise take the first-registered viable provider.
    if viable.len() > 1 && !reqs.vision {
        if let Some(text) = viable
            .iter()
            .copied()
            .find(|r| !(r.descriptor)().capabilities.supports_vision)
        {
            return Ok(text);
        }
    }
    Ok(viable[0])
}

/// Whether a provider's declared capabilities satisfy the caller's requirements.
fn meets(caps: &TextLlmCapabilities, reqs: &ModelRequirements) -> bool {
    if reqs.vision && !caps.supports_vision {
        return false;
    }
    reqs.constraints
        .iter()
        .all(|c| caps.supports_constraint(*c))
}

/// `id (backend)` summary of a set of registrations, for diagnostics.
fn summary(regs: &[&TextLlmRegistration]) -> String {
    if regs.is_empty() {
        return "(none)".to_string();
    }
    let mut v: Vec<String> = regs
        .iter()
        .map(|r| {
            let d = (r.descriptor)();
            format!("{} ({})", d.id, d.backend)
        })
        .collect();
    v.sort();
    v.dedup();
    v.join(", ")
}

/// Best-effort raw architecture echo from `config.json` (the literal `architectures` / `model_type`
/// fields, NOT a mapped family) for error messages. Reads only `config.json`; `None` when the source
/// is not a readable snapshot config (e.g. a `*.gguf` path). This stays generic — `core-llm` never
/// interprets the architecture, it only surfaces what the file says.
///
/// For a **VLM-wrapped** config the top-level `model_type` is the wrapper (e.g. Qwen3.6's
/// `qwen3_5` / `Qwen3_5ForConditionalGeneration`), while the actual decoder type is nested under
/// `text_config.model_type` (`qwen3_5_text`). Surfacing the nested type means an unknown-architecture
/// error names the real decoder a provider would dispatch on, not just the multimodal wrapper.
fn raw_arch_hint(spec: &LoadSpec) -> Option<String> {
    let p = std::path::Path::new(&spec.source);
    let cfg = if p.is_dir() { p.join("config.json") } else { p.to_path_buf() };
    let text = std::fs::read_to_string(&cfg).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let str_at = |val: &serde_json::Value, key: &str| -> Option<String> {
        val.get(key).and_then(|s| s.as_str()).map(String::from)
    };
    let mut parts: Vec<String> = Vec::new();
    if let Some(a) = v
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|s| s.as_str())
    {
        parts.push(format!("architectures={a}"));
    }
    if let Some(m) = str_at(&v, "model_type") {
        parts.push(format!("model_type={m}"));
    }
    if let Some(tm) = v.get("text_config").and_then(|tc| str_at(tc, "model_type")) {
        parts.push(format!("text_config.model_type={tm}"));
    }
    (!parts.is_empty()).then(|| parts.join(", "))
}

fn no_provider_msg(spec: &LoadSpec, all: &[&TextLlmRegistration]) -> String {
    let arch = raw_arch_hint(spec)
        .map(|a| format!(" ({a})"))
        .unwrap_or_default();
    format!(
        "no registered provider can serve model '{}'{arch}; linked providers: {}",
        spec.source,
        summary(all),
    )
}

fn unmet_caps_msg(
    spec: &LoadSpec,
    reqs: &ModelRequirements,
    accepting: &[&TextLlmRegistration],
) -> String {
    format!(
        "model '{}' is loadable, but no linked provider meets the requested capabilities \
         (vision={}, constraints={:?}); providers that match the architecture: {}",
        spec.source,
        reqs.vision,
        reqs.constraints,
        summary(accepting),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::{StreamEvent, TextLlmOutput};

    // --- A throwaway provider whose `load` is never invoked by `select` (resolution only). ---
    struct Dummy;
    impl TextLlm for Dummy {
        fn descriptor(&self) -> &TextLlmDescriptor {
            unreachable!("select never constructs a provider")
        }
        fn validate(&self, _req: &TextLlmRequest) -> Result<()> {
            unreachable!()
        }
        fn generate(
            &self,
            _req: &TextLlmRequest,
            _on_event: &mut dyn FnMut(StreamEvent),
        ) -> Result<TextLlmOutput> {
            unreachable!()
        }
    }
    fn never_loads(_spec: &LoadSpec) -> Result<Box<dyn TextLlm>> {
        Ok(Box::new(Dummy))
    }

    fn caps(vision: bool, constraints: &[Constraint]) -> TextLlmCapabilities {
        TextLlmCapabilities {
            max_context_tokens: 0,
            max_new_tokens: 0,
            supports_system_prompt: true,
            supports_vision: vision,
            supports_thinking: false,
            supports_tools: false,
            supported_constraints: constraints.to_vec(),
        }
    }

    // Distinct descriptor constructors (the field is `fn() -> _`, so one fn per provider shape).
    fn text_desc() -> TextLlmDescriptor {
        TextLlmDescriptor {
            id: "text".into(),
            family: "llama".into(),
            backend: "test".into(),
            capabilities: caps(false, &[Constraint::Json]),
        }
    }
    fn vision_desc() -> TextLlmDescriptor {
        TextLlmDescriptor {
            id: "vision".into(),
            family: "llava".into(),
            backend: "test".into(),
            capabilities: caps(true, &[]),
        }
    }

    fn yes(_spec: &LoadSpec) -> bool {
        true
    }
    fn no(_spec: &LoadSpec) -> bool {
        false
    }

    fn reg(
        descriptor: fn() -> TextLlmDescriptor,
        can_load: fn(&LoadSpec) -> bool,
    ) -> TextLlmRegistration {
        TextLlmRegistration {
            descriptor,
            load: never_loads,
            can_load,
        }
    }

    fn picked<'a>(
        regs: &'a [&'a TextLlmRegistration],
        reqs: &ModelRequirements,
    ) -> Result<String> {
        let spec = LoadSpec::dense("/no/such/snapshot");
        select(regs.iter().copied(), &spec, reqs).map(|r| (r.descriptor)().id)
    }

    #[test]
    fn can_load_filters_non_matching_providers() {
        let text = reg(text_desc, yes);
        let declines = reg(vision_desc, no);
        let id = picked(&[&text, &declines], &ModelRequirements::default()).unwrap();
        assert_eq!(id, "text");
    }

    #[test]
    fn unknown_architecture_is_a_typed_error() {
        let text = reg(text_desc, no);
        let vision = reg(vision_desc, no);
        let err = picked(&[&text, &vision], &ModelRequirements::default()).unwrap_err();
        match err {
            Error::Unsupported(m) => {
                assert!(m.contains("no registered provider can serve"), "{m}");
                // The linked providers are surfaced so the caller sees what IS available.
                assert!(m.contains("text (test)") && m.contains("vision (test)"), "{m}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn raw_arch_hint_surfaces_nested_text_config_model_type() {
        // A VLM-wrapped config (Qwen3.6 shape): the top-level `model_type` is the multimodal wrapper,
        // while the real decoder type a provider dispatches on is nested under `text_config`. The hint
        // must surface BOTH so an unknown-architecture error names the actual decoder, not just the
        // wrapper — otherwise the message points the reader at `qwen3_5` when the gap is `qwen3_5_text`.
        let dir = std::env::temp_dir().join(format!("core-llm-archhint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            br#"{"architectures":["Qwen3_5ForConditionalGeneration"],
                "model_type":"qwen3_5",
                "text_config":{"model_type":"qwen3_5_text"},
                "vision_config":{"model_type":"qwen3_5"}}"#,
        )
        .unwrap();
        let hint = raw_arch_hint(&LoadSpec::dense(dir.to_str().unwrap().to_string()))
            .expect("a readable config yields a hint");
        assert_eq!(
            hint,
            "architectures=Qwen3_5ForConditionalGeneration, model_type=qwen3_5, \
             text_config.model_type=qwen3_5_text"
        );
        let _ = std::fs::remove_dir_all(&dir);

        // A flat (non-wrapped) config still works and omits the nested part.
        let flat = std::env::temp_dir().join(format!("core-llm-archhint-flat-{}", std::process::id()));
        std::fs::create_dir_all(&flat).unwrap();
        std::fs::write(
            flat.join("config.json"),
            br#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#,
        )
        .unwrap();
        let hint = raw_arch_hint(&LoadSpec::dense(flat.to_str().unwrap().to_string())).unwrap();
        assert_eq!(hint, "architectures=LlamaForCausalLM, model_type=llama");
        let _ = std::fs::remove_dir_all(&flat);
    }

    #[test]
    fn vision_request_prefers_the_vision_provider() {
        // Both accept the model; the request needs vision.
        let text = reg(text_desc, yes);
        let vision = reg(vision_desc, yes);
        let reqs = ModelRequirements::default().with_vision();
        let id = picked(&[&text, &vision], &reqs).unwrap();
        assert_eq!(id, "vision");
    }

    #[test]
    fn json_request_prefers_a_json_capable_provider() {
        // Both accept; the request needs the Json constraint, which only the text provider enforces.
        let text = reg(text_desc, yes);
        let vision = reg(vision_desc, yes);
        let reqs = ModelRequirements::default().with_constraint(Constraint::Json);
        let id = picked(&[&text, &vision], &reqs).unwrap();
        assert_eq!(id, "text");
    }

    #[test]
    fn default_request_prefers_text_over_vision_on_a_tie() {
        // Vision registered first, but with no special needs a plain load should not hand back a
        // model that expects an image.
        let vision = reg(vision_desc, yes);
        let text = reg(text_desc, yes);
        let id = picked(&[&vision, &text], &ModelRequirements::default()).unwrap();
        assert_eq!(id, "text");
    }

    #[test]
    fn only_vision_accepts_default_request_still_loads_it() {
        // A multimodal-only snapshot: just the vision provider matches; default reqs must take it.
        let text = reg(text_desc, no);
        let vision = reg(vision_desc, yes);
        let id = picked(&[&text, &vision], &ModelRequirements::default()).unwrap();
        assert_eq!(id, "vision");
    }

    #[test]
    fn requested_capability_unmet_is_a_typed_error() {
        // Only the vision provider matches the architecture, but the caller needs the Json
        // constraint it cannot enforce.
        let text = reg(text_desc, no);
        let vision = reg(vision_desc, yes);
        let reqs = ModelRequirements::default().with_constraint(Constraint::Json);
        let err = picked(&[&text, &vision], &reqs).unwrap_err();
        match err {
            Error::Unsupported(m) => {
                assert!(m.contains("is loadable, but no linked provider meets"), "{m}");
                assert!(m.contains("vision (test)"), "{m}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
