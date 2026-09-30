//! Provider identity and declared capabilities.

use crate::constraint::{Constraint, ConstraintKind};
use crate::error::{Error, Result};
use crate::request::{ReasoningEffort, Sampling, TextLlmRequest};

/// Limits advertised by a loaded model with an in-checkpoint multi-token predictor (MTP).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpCapabilities {
    /// Maximum draft-token count the native provider supports per verification pass.
    pub max_draft_tokens: u32,
    /// Upstream-recommended draft-token count used by [`MtpMode::Auto`](crate::MtpMode::Auto).
    pub recommended_draft_tokens: u32,
}

/// One speculative proposer a loaded model can run, and its depth limits (epic sc-24432 E4).
/// Advertised in [`TextLlmCapabilities::speculative`]; read through
/// [`TextLlmCapabilities::proposer`], which also answers for a provider that still advertises MTP
/// only through the legacy [`TextLlmCapabilities::mtp`] field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProposerCapabilities {
    /// The proposal source.
    pub proposer: crate::SpeculativeProposer,
    /// The largest depth (draft tokens per verification pass) the backend runs for this proposer —
    /// a finite, backend-true bound, never `u32::MAX` (epic sc-24432 E4). A request above it is
    /// clamped to it by [`resolve_speculative`](crate::resolve_speculative), and the clamp is named
    /// in [`DecodeReport::fallbacks`](crate::DecodeReport::fallbacks) (sc-24438).
    pub max_depth: u32,
    /// The depth [`Speculative::Auto`](crate::Speculative::Auto) runs.
    pub recommended_depth: u32,
}

/// Upstream-recommended sampling presets for a model whose thinking and non-thinking modes use
/// different distributions. These are discovery metadata: a client may initialize controls from
/// the applicable preset, then sends an ordinary explicit [`Sampling`] request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelSamplingDefaults {
    pub thinking: Sampling,
    pub non_thinking: Sampling,
}

/// What a provider supports. Used both for honest advertisement and to validate requests up front.
#[derive(Clone, Debug, Default)]
pub struct TextLlmCapabilities {
    /// Maximum context length in tokens (`0` = unspecified / unbounded).
    pub max_context_tokens: usize,
    /// Maximum tokens that may be requested per generation (`0` = unspecified).
    pub max_new_tokens: u32,
    /// Whether a system turn is honored.
    pub supports_system_prompt: bool,
    /// Whether image (vision) content is accepted.
    pub supports_vision: bool,
    /// Whether video content (sampled frames + per-frame timestamps) is accepted. Independent of
    /// [`supports_vision`](Self::supports_vision): a model may take images but not video. `false` ⇒ a
    /// request carrying [`Content::Video`](crate::message::Content::Video) is rejected, never silently
    /// dropped.
    pub supports_video: bool,
    /// Whether audio content (decoded mono PCM) is accepted. Independent of
    /// [`supports_vision`](Self::supports_vision) and [`supports_video`](Self::supports_video):
    /// Gemma 4's audio path is a separate projector over raw sample frames and has nothing to do with
    /// its vision embedder, so a model may take images but not audio (and, in principle, the
    /// reverse). `false` ⇒ a request carrying [`Content::Audio`](crate::message::Content::Audio) is
    /// rejected, never silently dropped.
    pub supports_audio: bool,
    /// Whether the model has a controllable reasoning ("thinking") mode — i.e. it honors the
    /// [`thinking`](crate::TextLlmRequest::thinking) request control (its chat template gates an
    /// `enable_thinking` kwarg). `false` ⇒ the model never reasons, and an explicit
    /// [`ThinkingMode::Enabled`](crate::request::ThinkingMode::Enabled) request is rejected.
    pub supports_thinking: bool,
    /// Whether the model supports Qwen's `reasoning_effort` chat-template kwarg. This is separate
    /// from [`supports_thinking`](Self::supports_thinking): a generic thinking template can honor
    /// `enable_thinking` without recognizing Qwen's effort levels.
    pub supports_reasoning_effort: bool,
    /// Reasoning efforts a UI should offer for this loaded model. This is an advertised selection
    /// surface, not the parser's compatibility set: a provider may accept a legacy effort as an
    /// alias while omitting it here when the model does not implement that effort distinctly.
    pub reasoning_efforts: Vec<ReasoningEffort>,
    /// Model-card sampling recommendations, when the loaded artifact publishes them. Providers do
    /// not silently replace an explicit request with these values.
    pub model_sampling_defaults: Option<ModelSamplingDefaults>,
    /// Whether the model supports Qwen's `preserve_thinking` chat-template kwarg. This is separate
    /// from [`supports_thinking`](Self::supports_thinking): generic thinking history handling must
    /// not be advertised as Qwen-compatible preservation control.
    pub supports_preserve_thinking: bool,
    /// Whether the model supports tool / function calling — i.e. its chat template renders a `tools`
    /// section and it emits parseable `<tool_call>` blocks. `false` ⇒ a request carrying
    /// [`tools`](crate::TextLlmRequest::tools) is rejected (never silently dropped).
    pub supports_tools: bool,
    /// Native in-checkpoint MTP support and limits — the legacy advertisement, still read for
    /// MTP when [`speculative`](Self::speculative) carries no MTP entry. `None` there too means an
    /// explicit MTP request is rejected.
    pub mtp: Option<MtpCapabilities>,
    /// The speculative proposers this loaded model runs and their depth limits (epic sc-24432
    /// E4). Empty = none beyond a legacy [`mtp`](Self::mtp) advertisement. Query one proposer with
    /// [`proposer`](Self::proposer), all of them with
    /// [`speculative_proposers`](Self::speculative_proposers).
    pub speculative: Vec<ProposerCapabilities>,
    /// The output constraint KINDS this provider can enforce (empty = none).
    ///
    /// Kinds, not [`Constraint`] values: a provider cannot know in advance which schema or
    /// grammar a caller will send, so advertising concrete values would make any payload-carrying
    /// constraint permanently unsupportable. See [`ConstraintKind`].
    pub supported_constraints: Vec<ConstraintKind>,
}

impl TextLlmCapabilities {
    /// Whether a given constraint is supported, compared by [`ConstraintKind`].
    pub fn supports_constraint(&self, c: &Constraint) -> bool {
        self.supported_constraints.contains(&c.kind())
    }

    /// What this model advertises for `proposer`: its [`speculative`](Self::speculative) entry,
    /// else — for MTP only — the legacy [`mtp`](Self::mtp) advertisement.
    pub fn proposer(&self, proposer: crate::SpeculativeProposer) -> Option<ProposerCapabilities> {
        let legacy_mtp = match (proposer, self.mtp) {
            (crate::SpeculativeProposer::Mtp, Some(mtp)) => Some(ProposerCapabilities {
                proposer,
                max_depth: mtp.max_draft_tokens,
                recommended_depth: mtp.recommended_draft_tokens,
            }),
            _ => None,
        };
        self.speculative
            .iter()
            .find(|c| c.proposer == proposer)
            .copied()
            .or(legacy_mtp)
    }

    /// Advertise an in-checkpoint MTP head at `max_depth` drafts per verify step, running
    /// `recommended_depth` (clamped into `1..=max_depth`) under
    /// [`Speculative::Auto`](crate::Speculative::Auto) — on the per-proposer
    /// [`speculative`](Self::speculative) list **and** the legacy [`mtp`](Self::mtp) field, so the
    /// two advertisements always agree (sc-24438, E4). Replaces any earlier MTP advertisement.
    pub fn advertise_mtp(&mut self, max_depth: u32, recommended_depth: u32) {
        let max_depth = max_depth.max(1);
        let recommended_depth = recommended_depth.clamp(1, max_depth);
        self.mtp = Some(MtpCapabilities {
            max_draft_tokens: max_depth,
            recommended_draft_tokens: recommended_depth,
        });
        self.speculative
            .retain(|c| c.proposer != crate::SpeculativeProposer::Mtp);
        self.speculative.push(ProposerCapabilities {
            proposer: crate::SpeculativeProposer::Mtp,
            max_depth,
            recommended_depth,
        });
    }

    /// Every advertised proposer (the legacy MTP advertisement included), in
    /// [`SpeculativeProposer::ALL`](crate::SpeculativeProposer::ALL) order.
    pub fn speculative_proposers(&self) -> Vec<ProposerCapabilities> {
        crate::SpeculativeProposer::ALL
            .into_iter()
            .filter_map(|p| self.proposer(p))
            .collect()
    }

    /// Validate a request against these capabilities. Providers call this from
    /// [`TextLlm::validate`](crate::TextLlm::validate). Rejects (rather than silently ignoring)
    /// anything outside the declared surface.
    pub fn validate_request(&self, id: &str, req: &TextLlmRequest) -> Result<()> {
        self.validate_request_with_skipped_mtp(id, req, false)
    }

    /// [`validate_request`](Self::validate_request) for a loaded model whose checkpoint carries an
    /// MTP head the backend loaded past without running (`mtp_skipped`, sc-24438 — e.g. a
    /// sparse-MoE predictor layer). An explicit `mtp` request is then admitted rather than refused
    /// as unsupported: it decodes without a proposer, the skipped head named in its report
    /// ([`resolve_speculative_with_skipped_mtp`](crate::resolve_speculative_with_skipped_mtp),
    /// epic sc-24432 E2). Everything else is validated exactly as `validate_request` does.
    pub fn validate_request_with_skipped_mtp(
        &self,
        id: &str,
        req: &TextLlmRequest,
        mtp_skipped: bool,
    ) -> Result<()> {
        let reject = |msg: String| Err(Error::InvalidRequest(format!("[{id}] {msg}")));

        if req.messages.is_empty() {
            return reject("request has no messages".into());
        }
        if req.messages.iter().all(|m| {
            m.text_content().trim().is_empty() && !m.has_image() && !m.has_video() && !m.has_audio()
        }) {
            return reject("request has no non-empty content".into());
        }

        if !self.supports_system_prompt
            && req
                .messages
                .iter()
                .any(|m| m.role == crate::message::Role::System)
        {
            return Err(Error::Unsupported(format!(
                "[{id}] provider does not support a system prompt"
            )));
        }

        if !self.supports_vision && req.has_image() {
            return Err(Error::Unsupported(format!(
                "[{id}] provider does not support image (vision) input"
            )));
        }

        if !self.supports_video && req.has_video() {
            return Err(Error::Unsupported(format!(
                "[{id}] provider does not support video input"
            )));
        }

        if !self.supports_audio && req.has_audio() {
            return Err(Error::Unsupported(format!(
                "[{id}] provider does not support audio input"
            )));
        }

        // Reject only an explicit *enable*: a model with no reasoning mode cannot satisfy it. A
        // no-think (Disabled) request is trivially satisfied (the model never thinks) and Auto
        // defers to the template, so both are accepted regardless of support.
        if !self.supports_thinking && req.thinking == crate::request::ThinkingMode::Enabled {
            return Err(Error::Unsupported(format!(
                "[{id}] provider does not support a thinking (reasoning) mode"
            )));
        }

        if req.reasoning_effort.is_some() && !self.supports_reasoning_effort {
            return Err(Error::Unsupported(format!(
                "[{id}] provider does not support the reasoning_effort template control"
            )));
        }

        if req.preserve_thinking.is_some() && !self.supports_preserve_thinking {
            return Err(Error::Unsupported(format!(
                "[{id}] provider does not support the preserve_thinking template control"
            )));
        }

        if req.thinking == crate::request::ThinkingMode::Disabled && req.reasoning_effort.is_some()
        {
            return reject("reasoning_effort cannot be set when thinking is disabled".to_string());
        }

        // Offered tools the provider cannot render/parse are rejected, not silently dropped.
        if !self.supports_tools && !req.tools.is_empty() {
            return Err(Error::Unsupported(format!(
                "[{id}] provider does not support tool (function) calling"
            )));
        }

        if req.speculative.is_some() && req.mtp != crate::MtpMode::Off {
            return reject(
                "set either `speculative` or the legacy `mtp`, not both (legacy `mtp` maps onto \
                 `speculative`)"
                    .into(),
            );
        }
        // An explicit proposer must be advertised (or be a checkpoint MTP head the backend
        // skipped, which falls back by name) and ask for at least one draft. A depth above the
        // advertised maximum is not refused: `resolve_speculative` clamps it and names the clamp
        // in the report (sc-24438). `auto` and `off` never refuse (auto resolves to what the model
        // offers, or runs plain with the reason named).
        if let crate::Speculative::Proposer { proposer, depth } = req.speculative_mode() {
            let skipped_head = mtp_skipped && proposer == crate::SpeculativeProposer::Mtp;
            if self.proposer(proposer).is_none() && !skipped_head {
                return Err(Error::Unsupported(format!(
                    "[{id}] provider does not support speculative decoding with the \
                     `{proposer}` proposer"
                )));
            }
            if depth == 0 {
                return reject(format!("speculative `{proposer}` depth must be >= 1"));
            }
        }

        if let Some(c) = &req.constraint {
            if !self.supports_constraint(c) {
                return Err(Error::Unsupported(format!(
                    "[{id}] provider does not support the {c:?} constraint"
                )));
            }
        }

        let s = &req.sampling;
        if !(0.0..=2.0).contains(&s.temperature) {
            return reject(format!("temperature {} out of [0, 2]", s.temperature));
        }
        if !(0.0..=1.0).contains(&s.top_p) {
            return reject(format!("top_p {} out of [0, 1]", s.top_p));
        }
        if req.max_new_tokens == 0 {
            return reject("max_new_tokens must be >= 1".into());
        }
        if self.max_new_tokens > 0 && req.max_new_tokens > self.max_new_tokens {
            return reject(format!(
                "max_new_tokens {} exceeds cap {}",
                req.max_new_tokens, self.max_new_tokens
            ));
        }
        Ok(())
    }
}

/// A provider's identity plus its capabilities.
#[derive(Clone, Debug)]
pub struct TextLlmDescriptor {
    /// Unique provider id used for registry routing (e.g. `"mlx-llama"`).
    pub id: String,
    /// Model family (e.g. `"llama"`, `"qwen3"`).
    pub family: String,
    /// Tensor backend tag (`"mlx"` | `"candle"`).
    pub backend: String,
    /// Declared capabilities.
    pub capabilities: TextLlmCapabilities,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, MtpMode, ReasoningEffort, ThinkingMode};

    fn request() -> TextLlmRequest {
        TextLlmRequest::new(vec![Message::user("hello")], 8)
    }

    #[test]
    fn qwen_template_controls_are_not_implied_by_generic_thinking() {
        let generic_thinking = TextLlmCapabilities {
            supports_thinking: true,
            ..Default::default()
        };
        let mut effort = request();
        effort.reasoning_effort = Some(ReasoningEffort::Low);
        let err = generic_thinking
            .validate_request("test", &effort)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
        assert!(err.to_string().contains("reasoning_effort"));

        let mut preserve = request();
        preserve.preserve_thinking = Some(false);
        let err = generic_thinking
            .validate_request("test", &preserve)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
        assert!(err.to_string().contains("preserve_thinking"));
    }

    #[test]
    fn qwen_thinking_controls_validate_when_explicitly_advertised() {
        let qwen = TextLlmCapabilities {
            supports_thinking: true,
            supports_reasoning_effort: true,
            reasoning_efforts: vec![
                ReasoningEffort::XHigh,
                ReasoningEffort::Medium,
                ReasoningEffort::Low,
            ],
            model_sampling_defaults: None,
            supports_preserve_thinking: true,
            ..Default::default()
        };
        let mut req = request();
        req.reasoning_effort = Some(ReasoningEffort::XHigh);
        req.preserve_thinking = Some(true);
        qwen.validate_request("qwen38", &req).unwrap();
        assert_eq!(
            qwen.reasoning_efforts,
            [
                ReasoningEffort::XHigh,
                ReasoningEffort::Medium,
                ReasoningEffort::Low
            ]
        );
    }

    #[test]
    fn explicit_mtp_requires_capability_and_valid_draft_count() {
        let mut req = request();
        req.mtp = MtpMode::Enabled { draft_tokens: 3 };
        let err = TextLlmCapabilities::default()
            .validate_request("test", &req)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));

        let caps = TextLlmCapabilities {
            mtp: Some(MtpCapabilities {
                max_draft_tokens: 4,
                recommended_draft_tokens: 3,
            }),
            ..Default::default()
        };
        caps.validate_request("test", &req).unwrap();

        req.mtp = MtpMode::Enabled { draft_tokens: 0 };
        assert!(matches!(
            caps.validate_request("test", &req),
            Err(Error::InvalidRequest(_))
        ));
        // sc-24438: a depth above the advertised maximum is admitted — the resolver clamps it and
        // names the clamp in the report — rather than refused.
        req.mtp = MtpMode::Enabled { draft_tokens: 5 };
        caps.validate_request("test", &req).unwrap();
    }

    /// sc-24438 (E4): one call advertises MTP on both the per-proposer list and the legacy field,
    /// with the same finite numbers, and replaces an earlier advertisement.
    #[test]
    fn advertising_mtp_keeps_the_legacy_field_and_the_proposer_list_in_agreement() {
        use crate::SpeculativeProposer;
        let mut caps = TextLlmCapabilities::default();
        caps.advertise_mtp(7, 3);
        let legacy = caps.mtp.unwrap();
        let listed = caps
            .speculative
            .iter()
            .find(|c| c.proposer == SpeculativeProposer::Mtp)
            .copied()
            .unwrap();
        assert_eq!(legacy.max_draft_tokens, listed.max_depth);
        assert_eq!(legacy.recommended_draft_tokens, listed.recommended_depth);
        assert_eq!((listed.max_depth, listed.recommended_depth), (7, 3));
        assert_eq!(caps.proposer(SpeculativeProposer::Mtp), Some(listed));

        // Re-advertising replaces (one MTP entry), and the recommendation never exceeds the max.
        caps.advertise_mtp(2, 3);
        assert_eq!(
            caps.speculative
                .iter()
                .filter(|c| c.proposer == SpeculativeProposer::Mtp)
                .count(),
            1
        );
        let mtp = caps.proposer(SpeculativeProposer::Mtp).unwrap();
        assert_eq!((mtp.max_depth, mtp.recommended_depth), (2, 2));
        assert_eq!(
            caps.mtp,
            Some(MtpCapabilities {
                max_draft_tokens: 2,
                recommended_draft_tokens: 2
            })
        );
    }

    /// sc-24438 (E2): a checkpoint MTP head the backend skipped admits an explicit `mtp` request
    /// (it falls back by name at generate) instead of refusing it as unsupported; nothing else
    /// about validation changes.
    #[test]
    fn a_skipped_mtp_head_admits_an_explicit_mtp_request() {
        use crate::{Speculative, SpeculativeProposer};
        let caps = TextLlmCapabilities::default();
        let mut req = request();
        req.speculative = Some(Speculative::proposer(SpeculativeProposer::Mtp, 3));
        assert!(matches!(
            caps.validate_request("test", &req),
            Err(Error::Unsupported(_))
        ));
        caps.validate_request_with_skipped_mtp("test", &req, true)
            .unwrap();
        req.speculative = None;
        req.mtp = MtpMode::Enabled { draft_tokens: 3 };
        caps.validate_request_with_skipped_mtp("test", &req, true)
            .unwrap();
        req.mtp = MtpMode::Enabled { draft_tokens: 0 };
        assert!(matches!(
            caps.validate_request_with_skipped_mtp("test", &req, true),
            Err(Error::InvalidRequest(_))
        ));
        req.mtp = MtpMode::Off;
        req.speculative = Some(Speculative::proposer(SpeculativeProposer::DraftModel, 2));
        assert!(matches!(
            caps.validate_request_with_skipped_mtp("test", &req, true),
            Err(Error::Unsupported(_))
        ));
    }

    /// sc-24433: an explicit proposer is checked against the per-proposer advertisement (legacy
    /// MTP included), `auto` never refuses, and the new option and the legacy field cannot both
    /// be set. (Since sc-24438 a too-deep request is clamped by the resolver, not refused.)
    #[test]
    fn explicit_proposers_are_checked_per_proposer_and_the_two_fields_cannot_both_be_set() {
        use crate::{Speculative, SpeculativeProposer};
        let lookup = ProposerCapabilities {
            proposer: SpeculativeProposer::PromptLookup,
            max_depth: 8,
            recommended_depth: 4,
        };
        let caps = TextLlmCapabilities {
            speculative: vec![lookup],
            mtp: Some(MtpCapabilities {
                max_draft_tokens: 5,
                recommended_draft_tokens: 3,
            }),
            ..Default::default()
        };
        let mtp = ProposerCapabilities {
            proposer: SpeculativeProposer::Mtp,
            max_depth: 5,
            recommended_depth: 3,
        };
        assert_eq!(
            caps.proposer(SpeculativeProposer::PromptLookup),
            Some(lookup)
        );
        assert_eq!(
            caps.proposer(SpeculativeProposer::Mtp),
            Some(mtp),
            "legacy mirror"
        );
        assert_eq!(caps.proposer(SpeculativeProposer::DraftModel), None);
        assert_eq!(caps.speculative_proposers(), vec![mtp, lookup]);

        let mut req = request();
        req.speculative = Some(Speculative::proposer(SpeculativeProposer::PromptLookup, 4));
        caps.validate_request("test", &req).unwrap();
        // Above the advertised max: admitted, the resolver clamps it by name (sc-24438).
        req.speculative = Some(Speculative::proposer(SpeculativeProposer::PromptLookup, 9));
        caps.validate_request("test", &req).unwrap();
        req.speculative = Some(Speculative::proposer(SpeculativeProposer::PromptLookup, 0));
        assert!(matches!(
            caps.validate_request("test", &req),
            Err(Error::InvalidRequest(_))
        ));
        req.speculative = Some(Speculative::proposer(SpeculativeProposer::DraftModel, 2));
        let err = caps.validate_request("test", &req).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
        assert!(err.to_string().contains("`draft_model`"), "{err}");
        req.speculative = Some(Speculative::proposer(SpeculativeProposer::Mtp, 5));
        caps.validate_request("test", &req).unwrap();
        req.speculative = Some(Speculative::Auto);
        TextLlmCapabilities::default()
            .validate_request("test", &req)
            .unwrap();

        req.mtp = MtpMode::Auto;
        let err = caps.validate_request("test", &req).unwrap_err();
        assert!(err.to_string().contains("not both"), "{err}");
        req.speculative = None;
        caps.validate_request("test", &req).unwrap();
    }

    #[test]
    fn auto_mtp_is_a_safe_fallback_without_capability() {
        let mut req = request();
        req.mtp = MtpMode::Auto;
        TextLlmCapabilities::default()
            .validate_request("test", &req)
            .unwrap();
    }

    #[test]
    fn reasoning_effort_is_rejected_when_thinking_is_disabled() {
        let caps = TextLlmCapabilities {
            supports_thinking: true,
            supports_reasoning_effort: true,
            reasoning_efforts: Vec::new(),
            model_sampling_defaults: None,
            ..Default::default()
        };
        let mut req = request();
        req.thinking = ThinkingMode::Disabled;
        req.reasoning_effort = Some(ReasoningEffort::Medium);
        let err = caps.validate_request("test", &req).unwrap_err();
        assert!(matches!(err, Error::InvalidRequest(_)));
        assert!(err.to_string().contains("thinking is disabled"));
    }
}
