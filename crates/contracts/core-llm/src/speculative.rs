//! Backend-neutral speculative-decoding policy (epic 7153, stories 7171 + 7172).
//!
//! Speculative decoding generates several tokens per target forward by **proposing** a short
//! continuation cheaply, **verifying** it in one batched target forward, and **accepting** the
//! longest prefix the target agrees with. The proposal source differs by story — an n-gram match of
//! the context ([`ngram_propose`], story 7171) or a small draft model (story 7172) — but the
//! acceptance is the same distribution-preserving rule, so it lives here once, tensor-free, and both
//! backends ([`mlx-llm`], later `candle-llm`) reuse it.
//!
//! ## Distribution preservation
//! [`accept_token`] is the Leviathan et al. / Chen et al. speculative-sampling step: given the target
//! distribution `p` and the draft distribution `q` from which the proposed token was drawn, accept the
//! proposal with probability `min(1, p(t)/q(t))`, and on rejection resample from the normalized
//! residual `max(0, p − q)`. The committed token is then distributed **exactly** as `p` — speculative
//! decoding changes the *speed*, not the *output distribution*. Greedy decoding is the special case
//! where `p` is a point mass at the argmax: [`accept_greedy_run`] is the efficient form (accept the
//! longest draft prefix equal to the target's argmax).
//!
//! The acceptance functions take their random draws as parameters rather than owning an RNG, so this
//! module stays tensor-free *and* RNG-free — deterministic and exhaustively unit-testable, with the
//! backend feeding draws from its own seeded PRNG.
//!
//! ## The unified engine's policy (epic sc-24128, story sc-24130)
//! The Candle engine runs **one** speculative loop over its step-model seam with pluggable
//! proposers; the parts of that loop that are policy rather than tensor work live here so MLX can
//! adopt them unchanged: which proposer a request resolves to ([`resolve_speculative`],
//! [`ProposerKind`]; the proposer-agnostic request option since sc-24433) and the verify decision ([`greedy_commit`] for greedy, [`accept_token`] for
//! stochastic — the same acceptance rule as before, unchanged).
//!
//! [`mlx-llm`]: https://github.com/SceneWorks/mlx-llm

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Which proposal source a speculative run used (epic sc-24128, story sc-24130). Every decode
/// record names one, so "which proposer ran" is visible per request and a request that resolved
/// to no proposer says so (`none`) rather than silently downgrading. The labels are the request
/// vocabulary of [`SpeculativeProposer`](crate::SpeculativeProposer) plus `none` (sc-24433).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProposerKind {
    /// No proposer: the token-at-a-time path (also what a request whose speculation resolved to
    /// nothing runs — its [`DecodeReport::fallbacks`](crate::DecodeReport::fallbacks) says why).
    #[default]
    None,
    /// The checkpoint-native multi-token-prediction head.
    Mtp,
    /// Prompt lookup: [`ngram_propose`] over the context.
    PromptLookup,
    /// A separate draft model.
    DraftModel,
}

impl ProposerKind {
    /// Stable lower-case label for logs and evidence rows (`none`, `mtp`, `prompt_lookup`,
    /// `draft_model`).
    pub fn label(self) -> &'static str {
        match self {
            ProposerKind::None => "none",
            ProposerKind::Mtp => "mtp",
            ProposerKind::PromptLookup => "prompt_lookup",
            ProposerKind::DraftModel => "draft_model",
        }
    }
}

impl From<crate::SpeculativeProposer> for ProposerKind {
    fn from(proposer: crate::SpeculativeProposer) -> Self {
        match proposer {
            crate::SpeculativeProposer::Mtp => ProposerKind::Mtp,
            crate::SpeculativeProposer::PromptLookup => ProposerKind::PromptLookup,
            crate::SpeculativeProposer::DraftModel => ProposerKind::DraftModel,
        }
    }
}

/// What a request's [`Speculative`](crate::Speculative) option resolves to against the loaded
/// model ([`resolve_speculative`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpeculativePlan {
    /// Decode token-at-a-time; the record names `proposer=none`.
    #[default]
    Off,
    /// Run `proposer` with `depth` drafts per verify step.
    Run {
        /// The proposal source.
        proposer: crate::SpeculativeProposer,
        /// Drafts proposed per target verification pass (`>= 1`).
        depth: u32,
    },
}

impl SpeculativePlan {
    /// The proposer this plan runs (`none` when off).
    pub fn proposer(self) -> ProposerKind {
        match self {
            SpeculativePlan::Off => ProposerKind::None,
            SpeculativePlan::Run { proposer, .. } => proposer.into(),
        }
    }

    /// The draft width, or `None` when off.
    pub fn depth(self) -> Option<u32> {
        match self {
            SpeculativePlan::Off => None,
            SpeculativePlan::Run { depth, .. } => Some(depth),
        }
    }
}

/// A resolved speculative option: the plan to run and, when the request asked for more than the
/// plan delivers, the named reason ([`DecodeReport::fallbacks`](crate::DecodeReport::fallbacks),
/// epic sc-24432 E2) — never a silent downgrade.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpeculativeResolution {
    /// What to run.
    pub plan: SpeculativePlan,
    /// Why the plan is less than the request asked for, when it is.
    pub fallback: Option<String>,
}

/// Resolve a request's speculative option against what the loaded model advertises — the
/// backend-neutral policy both engines apply (epic sc-24432 E4):
///
/// * `off` never speculates;
/// * `auto` runs MTP at its recommended depth where the model has a head, else prompt lookup at
///   its recommended depth, else decodes plainly with the reason named;
/// * `{proposer, depth}` runs that proposer. `depth >= 1` is checked by
///   [`TextLlmCapabilities::validate_request`](crate::TextLlmCapabilities::validate_request)
///   first; a proposer the model does not advertise is **not** refused there (E2: explicit
///   fallback, never failure): it resolves here to `off` with the reason named, and the request
///   decodes plainly. A depth above the advertised `max_depth` — the backend-true bound — is
///   clamped to it and the clamp named in the fallback (sc-24438), never refused and never run
///   past the bound.
pub fn resolve_speculative(
    mode: crate::Speculative,
    capabilities: &crate::TextLlmCapabilities,
) -> SpeculativeResolution {
    use crate::{Speculative, SpeculativeProposer};
    let run = |proposer, depth| SpeculativeResolution {
        plan: SpeculativePlan::Run { proposer, depth },
        fallback: None,
    };
    match mode {
        Speculative::Off => SpeculativeResolution::default(),
        Speculative::Auto => [SpeculativeProposer::Mtp, SpeculativeProposer::PromptLookup]
            .into_iter()
            .find_map(|p| capabilities.proposer(p))
            .map(|cap| {
                run(
                    cap.proposer,
                    cap.recommended_depth.clamp(1, cap.max_depth.max(1)),
                )
            })
            .unwrap_or_else(|| SpeculativeResolution {
                plan: SpeculativePlan::Off,
                fallback: Some(format!(
                    "{AUTO_FOUND_NO_PROPOSER} ({NO_PROPOSER_ADVERTISED})"
                )),
            }),
        Speculative::Proposer { proposer, depth } => match capabilities.proposer(proposer) {
            None => SpeculativeResolution {
                plan: SpeculativePlan::Off,
                fallback: Some(proposer_unavailable(proposer, NO_SUCH_PROPOSER_ADVERTISED)),
            },
            Some(cap) => {
                let clamped = depth.clamp(1, cap.max_depth.max(1));
                SpeculativeResolution {
                    plan: SpeculativePlan::Run {
                        proposer,
                        depth: clamped,
                    },
                    fallback: (clamped != depth).then(|| {
                        format!(
                            "speculative: `{proposer}` depth {depth} clamped to {clamped} \
                             (advertised 1..={})",
                            cap.max_depth
                        )
                    }),
                }
            }
        },
    }
}

/// How an `auto` fallback with no runnable proposer begins; the reason follows in parentheses.
const AUTO_FOUND_NO_PROPOSER: &str = "speculative: auto found no proposer this model can run";

/// Why `auto` ran no proposer on a model that advertises none (the generic reason; a family with
/// its own reason names it through [`no_proposer_fallback`]).
const NO_PROPOSER_ADVERTISED: &str = "it advertises none; decoded without a proposer";

/// Why an explicit proposer the model does not advertise ran plain (the generic reason).
const NO_SUCH_PROPOSER_ADVERTISED: &str =
    "this model does not advertise it; decoded without a proposer";

/// The fallback for an explicit `proposer` that cannot run, with `why`.
fn proposer_unavailable(proposer: crate::SpeculativeProposer, why: &str) -> String {
    format!("speculative: `{proposer}` is not available for this model ({why})")
}

/// Why a captioner or SVG decoder (JoyCaption / LLaVA, StarVector — both backends) runs no
/// proposer: its continuation decodes token-at-a-time after a multimodal prefill, and the family
/// advertises no proposer for it. One string, so both backends say the same thing (E8).
pub const CAPTIONER_NO_PROPOSER: &str = "this captioner / SVG decoder advertises no proposer: \
     its continuation decodes token-at-a-time after a multimodal prefill";

/// Why a captioner or SVG decoder reports the cross-turn prefix cache as `none` (story sc-24437):
/// the cache is not wired to it, and its prompt's image rows are not in the token key. One string
/// for both backends (E8).
pub const CAPTIONER_NO_PREFIX_CACHE: &str = "prefix_cache: not wired to this captioner / SVG \
     decoder — its prompt's image rows are not in the token key";

/// The fallback a family that advertises **no** proposer names for `mode`, with the family's own
/// `why` (e.g. [`CAPTIONER_NO_PROPOSER`]) instead of the generic reason
/// [`resolve_speculative`] gives (E2/E3): `None` for `off`; for `auto` and an explicit proposer,
/// the same leading words `resolve_speculative` uses, so a product matches one vocabulary.
pub fn no_proposer_fallback(mode: crate::Speculative, why: &str) -> Option<String> {
    match mode {
        crate::Speculative::Off => None,
        crate::Speculative::Auto => Some(format!("{AUTO_FOUND_NO_PROPOSER} ({why})")),
        crate::Speculative::Proposer { proposer, .. } => Some(proposer_unavailable(proposer, why)),
    }
}

/// The MTP depth [`Speculative::Auto`](crate::Speculative::Auto) runs on a Qwen3.8 head — native
/// or companion — on both backends (the upstream recommendation). Recorded per backend in
/// [`crate::defaults`].
pub const MTP_RECOMMENDED_DEPTH: u32 = 3;

/// The prompt-lookup depth [`Speculative::Auto`](crate::Speculative::Auto) runs on a model without
/// an MTP head (sc-24433), on both backends. A lookup that finds no match proposes nothing and the
/// step is a single-token verify, so the depth only prices a match; 4 keeps the Qwen3.5 hybrid's
/// DeltaNet checkpoint ring at `K + 2 = 6` states. Such a step is *not* free on MLX: the proposer
/// must read the committed token on the host before it can look up the next match, so the step is
/// never pipelined the way the plain token-at-a-time loop is — the cost [`AcceptanceMonitor`]
/// weighs when it demotes `auto`. Recorded per backend in [`crate::defaults`].
pub const PROMPT_LOOKUP_RECOMMENDED_DEPTH: u32 = 4;

/// The prompt-lookup advertisement every text decoder carries (both backends' decoders run the
/// proposer through their step engine, so it needs nothing from the checkpoint), at the
/// provider's per-model verify bound `max_depth`, recommending the backend's defaults-table depth
/// (`depths`, [`DecodeDefaults::recommended_depths`](crate::DecodeDefaults::recommended_depths))
/// or `max_depth` when shallower.
pub fn prompt_lookup_capabilities(
    max_depth: u32,
    depths: &crate::RecommendedDepths,
) -> crate::ProposerCapabilities {
    crate::ProposerCapabilities {
        proposer: crate::SpeculativeProposer::PromptLookup,
        max_depth,
        recommended_depth: depths.prompt_lookup.min(max_depth),
    }
}

/// A model's verify-depth bound: the depth its descriptor advertises for prompt lookup (the
/// per-model bound, sc-24438), which `draft_model` advertises too — every draft is one more
/// verify row whichever proposer drew it (sc-24436). Every text decoder advertises prompt lookup;
/// one draft per step otherwise.
pub fn verify_depth_bound(capabilities: &crate::TextLlmCapabilities) -> u32 {
    capabilities
        .proposer(crate::SpeculativeProposer::PromptLookup)
        .map_or(1, |lookup| lookup.max_depth)
}

/// The fallback a `draft_model` plan names on a provider with no resident draft (both backends).
pub const DRAFT_MODEL_NOT_LOADED: &str = "speculative: `draft_model` has no draft model loaded \
     on this provider; decoded without a proposer";

/// The draft-model depth advertised as recommended (sc-24436): four drafts per verify step, or
/// the provider's bound when that is shallower.
pub const DRAFT_MODEL_RECOMMENDED_DEPTH: u32 = 4;

/// The `draft_model` advertisement a provider carries while a compatible draft is resident
/// (sc-24436): `1..=max_depth` drafts per verify step, recommending the backend's defaults-table
/// depth (`depths`, [`DecodeDefaults::recommended_depths`](crate::DecodeDefaults::recommended_depths))
/// or `max_depth` when shallower. `max_depth` is the same per-model verify bound the provider
/// advertises for its other proposers — every draft is one more verify row, whichever proposer
/// drew it — so a provider passes that one value here.
pub fn draft_model_capabilities(
    max_depth: u32,
    depths: &crate::RecommendedDepths,
) -> crate::ProposerCapabilities {
    crate::ProposerCapabilities {
        proposer: crate::SpeculativeProposer::DraftModel,
        max_depth,
        recommended_depth: depths.draft_model.min(max_depth),
    }
}

/// Whether a draft model can propose for a target (sc-24436), or the named reason it cannot.
/// The target verifies each draft id as its own token and the acceptance rule compares the two
/// models' distributions row for row, so the draft must share the target's tokenizer vocabulary
/// token for token ([`Tokenizer::vocabulary_mismatch`](crate::Tokenizer::vocabulary_mismatch))
/// and score no ids the target does not (`*_logits` — the models' `vocab_size`, which may exceed
/// the tokenizer's by padding rows). A draft padded less than its target is compatible: it
/// scores every real token. `Ok` carries how many leading draft ids a proposer may draw — the
/// tokenizer's tokens the draft scores — so a padding row is never proposed. The reason leads
/// with `draft model:`.
pub fn draft_compatibility(
    target: &crate::Tokenizer,
    target_logits: usize,
    draft: &crate::Tokenizer,
    draft_logits: usize,
) -> Result<usize, String> {
    if let Some(why) = target.vocabulary_mismatch(draft) {
        return Err(vocabulary_refusal(&why));
    }
    if draft_logits > target_logits {
        return Err(format!(
            "draft model: it scores {draft_logits} token ids, more than the target's {target_logits}"
        ));
    }
    Ok(draft_logits.min(draft.vocab_size()))
}

fn vocabulary_refusal(why: &str) -> String {
    format!("draft model: its tokenizer vocabulary is not the target's ({why})")
}

/// The draft's own `tokenizer.json` (beside the weights at `draft_source`) checked against the
/// target's before any draft weight is read (sc-24436): the named refusal when the vocabularies
/// differ, `None` when they agree or the draft ships no readable tokenizer (its loaded tokenizer
/// is then checked by [`draft_compatibility`]).
pub fn draft_tokenizer_refusal(
    target: &crate::Tokenizer,
    draft_source: &std::path::Path,
) -> Option<String> {
    crate::Tokenizer::from_file(draft_source.join("tokenizer.json"))
        .ok()
        .and_then(|draft| target.vocabulary_mismatch(&draft))
        .map(|why| vocabulary_refusal(&why))
}

/// A named draft refused for `error` (an unreadable source, a refused device, a load-time format
/// the draft cannot take), leading with `draft model:`.
pub fn draft_refusal(error: impl std::fmt::Display) -> String {
    format!("draft model: {error}")
}

/// A named draft whose load estimate failed (`error`) — it cannot be priced beside the target.
pub fn draft_unpriced_refusal(error: impl std::fmt::Display) -> String {
    format!("draft model: its load cannot be priced ({error})")
}

/// A named draft whose load failed (`error`) after admission.
pub fn draft_load_refusal(error: impl std::fmt::Display) -> String {
    format!("draft model: its load failed ({error})")
}

/// Settle what became of a load's named draft (sc-24436) — the model-agnostic half of a
/// backend's `attach_draft`: a compatible draft (`Ok`) is kept and `draft_model` advertised on
/// `capabilities` at the model's [`verify_depth_bound`], recommending the backend row's `depths`;
/// a refused one (`Err`, its reason leading
/// with `draft model:`) is named in the load's `fallbacks` (E2) as well as in the returned
/// [`DraftReport`](crate::DraftReport), and the target loads alone. Returns the draft to keep
/// resident and the report.
pub fn settle_draft<D>(
    source: impl Into<String>,
    outcome: std::result::Result<D, String>,
    capabilities: &mut crate::TextLlmCapabilities,
    fallbacks: &mut Vec<String>,
    depths: &crate::RecommendedDepths,
) -> (Option<D>, crate::DraftReport) {
    let source = source.into();
    match outcome {
        Ok(draft) => {
            let max_depth = verify_depth_bound(capabilities);
            capabilities
                .speculative
                .push(draft_model_capabilities(max_depth, depths));
            (Some(draft), crate::DraftReport::resident(source))
        }
        Err(why) => (
            None,
            crate::DraftReport::refused(source, why).named_in(fallbacks),
        ),
    }
}

/// A `draft_model` plan the resident draft cannot cover (sc-24436, E2): when the positions its
/// cache must hold — the request's `prompt_tokens + max_new_tokens` plus the `depth + 1` a
/// draft step writes past the committed tokens — outrun the draft's own context window
/// (`draft_context`; `0` is unbounded), the request runs what
/// [`Speculative::Auto`](crate::Speculative::Auto) resolves to on these `capabilities` instead —
/// MTP or prompt lookup, else plain decoding — with the reason named, rather than driving the
/// draft past the positions it was built for. Any other resolution is returned unchanged.
pub fn fit_draft_context(
    resolution: SpeculativeResolution,
    capabilities: &crate::TextLlmCapabilities,
    draft_context: usize,
    prompt_tokens: usize,
    max_new_tokens: u32,
) -> SpeculativeResolution {
    let SpeculativePlan::Run {
        proposer: crate::SpeculativeProposer::DraftModel,
        depth,
    } = resolution.plan
    else {
        return resolution;
    };
    let reach = prompt_tokens
        .saturating_add(max_new_tokens as usize)
        .saturating_add(depth as usize)
        .saturating_add(1);
    if draft_context == 0 || reach <= draft_context {
        return resolution;
    }
    let auto = resolve_speculative(crate::Speculative::Auto, capabilities);
    let runs = match auto.plan {
        SpeculativePlan::Run { proposer, depth } => format!("`{proposer}` at depth {depth}"),
        SpeculativePlan::Off => "without a proposer".into(),
    };
    let why = format!(
        "speculative: `draft_model` cannot cover this request: prompt ({prompt_tokens} tokens) + \
         requested generation ({max_new_tokens}) + a depth-{depth} draft step's {} positions \
         exceeds the draft model's context window {draft_context}; decoded as `auto` would, \
         {runs}",
        depth + 1
    );
    SpeculativeResolution {
        plan: auto.plan,
        fallback: Some(match auto.fallback {
            Some(more) => format!("{why}; {more}"),
            None => why,
        }),
    }
}

// ---------------------------------------------------------------------------------------------
// Adaptive demotion of `auto` (sc-24446, epic sc-24432 E5).
// ---------------------------------------------------------------------------------------------

/// Verify steps in each window [`AcceptanceMonitor`] judges `auto`'s proposer over (sc-24446).
/// Long enough that one lucky or unlucky step does not decide (a prompt-lookup step accepts `0` or
/// up to its depth), short enough that a request whose proposer stops paying pays for at most
/// this many slow steps before it is demoted.
pub const ACCEPTANCE_PROBE_VERIFIES: u32 = 16;

// The break-even thresholds below (E5: the justification sits next to the value). A verify step
// commits `1 + accepted` tokens for the cost of `r` plain single-token steps, so speculation pays
// only while the mean accepted length per verify `mal` exceeds `r − 1`; a run's `r` is
// `(1 + mal) / speedup`, its speedup being its decode tok/s over its own `off` twin's.
//
// MLX evidence: the sc-24446 campaign documents mlx-campaign-2 (epic commit 9b310c4da) and
// mlx-campaign-3 (epic commit 8a886fce6), attached to Shortcut epic 24432 — the per-process
// `DecodeReport`s of rows `f1-qwen38-*`, `f2-bonsai-*`, `f3-qwen3-8b-*` and `f4-gemma4-*`
// (`-auto-s`, `-full`, `-cache`), each `auto` / explicit option against its own `off` twin. They
// measured `r` against MLX's **pipelined** plain loop:
//
// * MTP: a verify step costs 1.25–1.6× a pipelined step at 1 draft and 2.2–2.9× at depth 3 (each
//   draft is one more sequential head forward). Open-ended prompts on a companion MTP head
//   (`f2-bonsai`) measured mal 0.66–0.85 and lost 22–32 %; native MTP heads (`f1-qwen38`, the
//   Qwen3.6 MoE) measured mal 1.38–2.8 and won 5–49 %. The head forwards are most of that cost
//   and are paid whichever MLX plain loop a demoted request falls back to, so the MLX MTP
//   thresholds hold pipelined or not.
// * Prompt lookup (`f3-qwen3-8b`, `f4-gemma4`): the drafts are free (a host n-gram search); its
//   runs' implied ratio `(1 + mal) / (1 + speedup)` is 1.42–1.58 (open-ended: acceptance 1–9 %,
//   mal 0.02–0.5, lost 5–28 %; grounded code / RAG / summary: acceptance 43–71 %, up to +122 %).
//   Much of that cost is a lookup step forfeiting pipelining (the proposer reads each token on
//   the host): mlx-campaign-2's `f3-qwen3-8b-pipe-on` / `-pipe-off` pair measured the pipelined
//   plain loop 3–15 % faster than the unpipelined one. Where the demoted MLX request would
//   continue **unpipelined** anyway — a constraint, a history-reading (penalized) sampler,
//   `Pipelining::Off` or `MLX_LLM_PIPELINING=0` — that part of the cost does not exist, and no
//   campaign row measured lookup against an unpipelined `off` (the pipe-on / pipe-off rows ran
//   `off` only). So MLX lookup is **not demoted** there (E5: off only on a measured regression).
//
// Candle evidence: the sc-24446 CUDA campaign cuda-campaign-a @1bcb23e6a, attached to Shortcut
// epic 24432 — the per-process `DecodeReport`s of rows `f1-qwen38-*` (Qwen3.8-27B bf16, native
// MTP head) and `f2-bonsai-*` (Ternary-Bonsai-2-27B mlx-2bit; companion MTP head, or prompt
// lookup on the `f2-bonsai-graphs-*` / `-positions-off` rows, where `auto` resolved to it), each
// against its own `off` twin. Candle's verify steps are priced against its own token-at-a-time
// step, not MLX's, and cost a different multiple of it per weight format:
//
// * MTP: Qwen3.8 bf16 `r` 1.27–1.28 at 1 draft (`f1-qwen38-full`), 1.50–1.71 at depth 3 (every
//   `f1-qwen38` row; 1.50 on `-graphs-on` creative), 2.12–2.17 at 7 — its `auto` won 41–127 % at
//   mal 1.24–2.70. Bonsai's companion head: 1.45–1.58, 2.26–2.58 and 3.96–4.25 — its open-ended
//   `auto` (chat / creative, mal 0.65–0.85) lost 23–35 %. The MLX depth-3 floor (2.2) is above
//   Qwen3.8's measured Candle cost, so the MLX MTP thresholds would demote Candle requests that
//   win: Candle's MTP thresholds are its own cheapest costs (below).
// * Prompt lookup: Qwen3.8 bf16 `r` 1.10–1.20 at 1 draft, 1.20–1.37 at 3, 1.35–1.66 at 7
//   (`f1-qwen38-full`, explicit depths; at depth 3 it won 8 % at mal 0.40); Bonsai 1.17–1.38,
//   1.45–1.89, 1.94–2.84 (`f2-bonsai-full`) and 1.59–2.71 under `auto` at depth 4
//   (`f2-bonsai-graphs-on` / `-off` / `-positions-off`, `auto` vs `off`: open-ended mal 0.15–0.20
//   lost 27–44 %, code mal 0.39–0.40 lost 27–40 %). Candle has no pipelined loop, so a lookup step
//   forfeits no pipelining — its cost is the wider verify forward alone, growing with depth.
//
// Demotion is irreversible for the request, so every threshold is the break-even at the
// **cheapest** measured cost on that plain loop: a request is demoted only when it is losing even
// if its verify steps are as cheap as any the campaign measured there — no measured winning
// request falls below its threshold, and the measured open-ended losers (companion MTP on MLX
// 0.66–0.85, lookup ≤ 0.4 on MLX and ≤ 0.2 on Candle) do. A loser whose verify steps cost more
// than the cheapest measured (Bonsai's companion head on Candle, mal 0.65–0.85 against a 2.26+
// verify) is above its static threshold on average: telling it from Qwen3.8 bf16 at the same
// acceptance (which would win) needs the request's own verify cost — which a timed monitor
// measures (below); the static thresholds are its fallback where timing cannot decide.

/// Mean accepted drafts per verify below which `auto`'s prompt lookup is demoted when the plain
/// loop it would fall back to is MLX's pipelined one ([`PlainDecode::MlxPipelined`], the only
/// measured MLX regime): the cheapest measured prompt-lookup verify cost `r = 1.4` (the 1.42 floor
/// of the campaign's implied ratios, rounded down), minus the one token every verify commits
/// anyway.
pub const PROMPT_LOOKUP_DEMOTE_BELOW: f64 = 0.4;

/// Mean accepted drafts per verify below which `auto`'s MTP head at **one** draft is demoted: the
/// cheapest measured one-draft verify cost `r = 1.25` (MLX; Candle's is 1.27), minus one.
pub const MTP_DEMOTE_BELOW_AT_ONE_DRAFT: f64 = 0.25;

/// What each MTP draft past the first adds to [`MTP_DEMOTE_BELOW_AT_ONE_DRAFT`] on MLX: the
/// cheapest measured verify cost grows from 1.25 at one draft to 2.2 at depth 3, `(2.2 − 1.25) / 2`
/// per sequential head forward — so depth 3 (the recommended depth) demotes below mal 1.2.
pub const MTP_DEMOTE_BELOW_PER_EXTRA_DRAFT: f64 = 0.475;

/// What each MTP draft past the first adds to [`MTP_DEMOTE_BELOW_AT_ONE_DRAFT`] on Candle
/// ([`PlainDecode::Candle`]): the cheapest measured Candle verify cost grows from 1.25 at one draft
/// to 1.50 at depth 3 (Qwen3.8 bf16, `f1-qwen38-graphs-on` creative), `(1.50 − 1.25) / 2` — so
/// depth 3 demotes below mal 0.5, and depth 7 below 1.0 (its cheapest measured cost is 2.12).
pub const CANDLE_MTP_DEMOTE_BELOW_PER_EXTRA_DRAFT: f64 = 0.125;

/// Mean accepted drafts per verify below which `auto`'s prompt lookup at **one** draft is demoted
/// on Candle ([`PlainDecode::Candle`]): the cheapest measured one-draft lookup verify cost
/// `r = 1.099` (Qwen3.8 bf16, `f1-qwen38-full` creative), rounded down, minus one.
pub const CANDLE_PROMPT_LOOKUP_DEMOTE_BELOW_AT_ONE_DRAFT: f64 = 0.09;

/// What each lookup draft past the first adds to
/// [`CANDLE_PROMPT_LOOKUP_DEMOTE_BELOW_AT_ONE_DRAFT`]: the cheapest measured Candle lookup verify
/// cost grows from 1.099 at one draft to 1.197 at depth 3 and 1.349 at depth 7 (Qwen3.8 bf16
/// creative each time); `0.04` per draft keeps every measured depth at or below its floor
/// (0.17 ≤ 0.197 at 3, 0.33 ≤ 0.349 at 7) — so [`PROMPT_LOOKUP_RECOMMENDED_DEPTH`] (4, `auto`'s
/// depth) demotes below mal 0.21. Bonsai's measured depth-4 cost (1.59+) is well above that floor.
pub const CANDLE_PROMPT_LOOKUP_DEMOTE_BELOW_PER_EXTRA_DRAFT: f64 = 0.04;

/// The plain decoding a demoted request would continue on (sc-24446) — what a proposer's verify
/// cost is weighed against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlainDecode {
    /// MLX's pipelined token-at-a-time loop: device-resident draws, no constraint, a sampler that
    /// does not read the history, `Pipelining::Auto` and the pipelining switch on.
    MlxPipelined,
    /// MLX's unpipelined loop: a constraint, a history-reading (penalized) sampler,
    /// `Pipelining::Off` or `MLX_LLM_PIPELINING=0`.
    MlxUnpipelined,
    /// Candle's token-at-a-time engine steps (Candle has no pipelined loop).
    Candle,
}

/// The mean accepted length per verify below which `proposer` at `depth` loses to `plain`
/// decoding at the cheapest measured verify cost (see the constants above), or `None` when no
/// demotion applies: no proposer, a zero depth, prompt lookup falling back to MLX's unpipelined
/// loop (no measured regression there), or `draft_model` — which [`Speculative::Auto`] never
/// resolves to ([`resolve_speculative`] picks MTP, else prompt lookup) and whose cost the campaign
/// did not measure.
///
/// [`Speculative::Auto`]: crate::Speculative::Auto
pub fn demotion_threshold(proposer: ProposerKind, depth: u32, plain: PlainDecode) -> Option<f64> {
    if depth == 0 {
        return None;
    }
    let extra = f64::from(depth - 1);
    match (proposer, plain) {
        (ProposerKind::PromptLookup, PlainDecode::MlxPipelined) => Some(PROMPT_LOOKUP_DEMOTE_BELOW),
        (ProposerKind::PromptLookup, PlainDecode::MlxUnpipelined) => None,
        (ProposerKind::PromptLookup, PlainDecode::Candle) => Some(
            CANDLE_PROMPT_LOOKUP_DEMOTE_BELOW_AT_ONE_DRAFT
                + CANDLE_PROMPT_LOOKUP_DEMOTE_BELOW_PER_EXTRA_DRAFT * extra,
        ),
        (ProposerKind::Mtp, PlainDecode::Candle) => {
            Some(MTP_DEMOTE_BELOW_AT_ONE_DRAFT + CANDLE_MTP_DEMOTE_BELOW_PER_EXTRA_DRAFT * extra)
        }
        (ProposerKind::Mtp, PlainDecode::MlxPipelined | PlainDecode::MlxUnpipelined) => {
            Some(MTP_DEMOTE_BELOW_AT_ONE_DRAFT + MTP_DEMOTE_BELOW_PER_EXTRA_DRAFT * extra)
        }
        (ProposerKind::None | ProposerKind::DraftModel, _) => None,
    }
}

/// A monotonic clock the engines time decode steps with (sc-24446): `auto`'s acceptance monitor
/// weighs each request's own measured verify cost against its own measured plain-step cost. A
/// trait so tests drive the monitor with a deterministic clock ([`with_decode_clock`]).
pub trait DecodeClock {
    /// Time since an arbitrary fixed origin; never decreases.
    fn now(&self) -> Duration;
}

/// The host wall clock ([`Instant`]) — what every engine times with unless a
/// [`with_decode_clock`] scope says otherwise.
#[derive(Clone, Copy, Debug, Default)]
pub struct WallClock;

impl DecodeClock for WallClock {
    fn now(&self) -> Duration {
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        ORIGIN.get_or_init(Instant::now).elapsed()
    }
}

thread_local! {
    /// A [`with_decode_clock`] scope's clock: `Some(clock)` while one is active on this thread
    /// (`clock` itself `None` for untimed steps).
    static DECODE_CLOCK: RefCell<Option<Option<Rc<dyn DecodeClock>>>> = const { RefCell::new(None) };
}

/// The clock the engines time this thread's decode steps with: [`WallClock`], unless a
/// [`with_decode_clock`] scope on this thread names another — or `None`, where steps are not
/// timed and the monitor decides on its static thresholds alone (no plain probe).
pub fn decode_clock() -> Option<Rc<dyn DecodeClock>> {
    DECODE_CLOCK
        .with(|c| c.borrow().clone())
        .unwrap_or_else(|| Some(Rc::new(WallClock)))
}

/// Run `f` with this thread's decode clock ([`decode_clock`]) set to `clock` — a test's
/// deterministic clock, or `None` for untimed steps — restoring the previous one afterwards
/// (also when `f` panics).
pub fn with_decode_clock<R>(clock: Option<Rc<dyn DecodeClock>>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Option<Rc<dyn DecodeClock>>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            DECODE_CLOCK.with(|c| *c.borrow_mut() = previous);
        }
    }
    let _restore = Restore(DECODE_CLOCK.with(|c| c.borrow_mut().replace(clock)));
    f()
}

// The cost-aware decision (sc-24446): the static thresholds above are the break-even at the
// cheapest cost a campaign measured, so they cannot tell a request whose verify steps are cheap
// (Qwen3.8 bf16 on Candle CUDA, r 1.27–1.71) from one at the same acceptance whose verify steps
// are dear (Bonsai's 2-bit weights, r 2.26–2.71). A timed monitor therefore measures the
// request's own costs, with host wall-clock marks around each engine step — the step's propose,
// its verify forward, the host readback that decides it, the cache recovery and the proposer
// commit; the event emission is outside the mark — and judges each window on its measured gain:
//
//   gain = (tokens the window's timed steps committed) × t_plain / (their summed wall time)
//        = (1 + mal) × t_plain / t_step,
//
// demoting when `gain < 1 − MEASURED_GAIN_MARGIN`; a stalled step (`STALL_STEP_FACTOR` × the
// stretch's median step) is left out of both sums. Work a step enqueues but does not wait for
// (CUDA's asynchronous launches after the readback) lands in the next step's mark, so in steady
// state every step is charged one step's work.
//
// `t_plain` is the median of the request's own timed single-token steps: the first
// `SHAPE_WARMUP_STEPS + PLAIN_PROBE_STEPS` decode steps of a timed `auto` request run plain
// (`k = 0`, the plain loop's draw — the output is unchanged) before the proposer is used, and a
// later step whose proposer found nothing is one more sample. A probe step of an MTP request
// still asks for the target's hidden row (the head is caught up on the probed tokens before its
// first proposal, `Proposer::catch_up`), which makes it marginally dearer than a plain step —
// `t_plain` errs high, the gain errs high, the monitor errs towards keeping the proposer. No
// forward runs only to be timed.
//
// A step is timed only once its shape — the verify token count `1 + drafts` — has run
// `SHAPE_WARMUP_STEPS` times in the request: Candle's CUDA-graph runner spends a shape's first
// three steps on an eager warm-up, the capture (with an eager self-check) and a verified first
// replay (`candle-llm` `decode/graph.rs`), and every backend pays a shape's kernel selection /
// compilation and allocator growth on its first use. The want-hidden flag is part of a graph's
// key too, but it is constant over a request's speculative phase (the proposer's), so the width
// is the shape.
//
// Where timing cannot decide — an untimed run (no clock), fewer than `PLAIN_PROBE_STEPS` plain
// samples, or a window with fewer than `MIN_TIMED_WINDOW_STEPS` timed steps (a lookup whose
// draft count keeps changing shape) — the window falls back to the static threshold. On MLX's
// pipelined path a demoted request continues on the **pipelined** loop, which the probe cannot
// time (its plain steps run unpipelined inside the speculative loop, 3–15 % dearer — mlx-campaign-2
// `f3-qwen3-8b-pipe-on` / `-off`): the measured gain then overstates the true one, so a measured
// loss there is a real loss, but a measured win is not proven — MLX-pipelined windows demote on a
// measured loss **or** the static threshold.

/// Times a verify shape (`1 + drafts` tokens) must have run in a request before a step of that
/// shape is timed: Candle's CUDA-graph runner spends a shape's first three steps on an eager
/// warm-up, the capture and a verified first replay; other backends need one (kernel selection
/// or compilation, allocator growth), so three is the common bound.
pub const SHAPE_WARMUP_STEPS: u32 = 3;

/// Timed plain steps a timed `auto` request takes before its first proposal: the samples its
/// plain-step cost `t_plain` (their median) starts from. Plain decode steps are the least noisy
/// thing the engine times (the CUDA campaign's in-process decode-rate stddev has a median of
/// 0.26 %), so four bound the median well; with the shape warm-up the probe is
/// [`PLAIN_PROBE_MAX_STEPS`] steps — about 1–3 % of a 256-token request's speculative gain.
pub const PLAIN_PROBE_STEPS: u32 = 4;

/// The plain probe's length bound: its shape warm-up plus [`PLAIN_PROBE_STEPS`] timed steps. A
/// probe whose steps are not timed (a zero-length mark) ends here all the same.
pub const PLAIN_PROBE_MAX_STEPS: u32 = SHAPE_WARMUP_STEPS + PLAIN_PROBE_STEPS;

/// Timed steps a window needs for its measured gain to decide (half the window); a window with
/// fewer falls back to the static threshold.
pub const MIN_TIMED_WINDOW_STEPS: u32 = ACCEPTANCE_PROBE_VERIFIES / 2;

/// How far below break-even a window's measured gain must fall before it demotes: `gain < 1 −
/// 0.05`. Both timings come from the same request seconds apart, so the noise that matters is
/// in-process: the campaigns' run-to-run stddev of a request's decode rate within one process is
/// 0.26 % (median) / 1.0 % (p90) / 2.2 % (p95) on CUDA (cuda-campaign-a, 160 `off` rows) and
/// 0.51 % / 5.0 % / 10 % on MLX (mlx-campaign-4, 320 `off` rows, a shared desktop GPU). 5 % is
/// above 2σ of CUDA's p95 and at MLX's p90 — and it bounds the error either way: a request kept
/// below break-even loses at most 5 % on that window, one demoted by noise had at most that much
/// left to gain.
pub const MEASURED_GAIN_MARGIN: f64 = 0.05;

/// A measured gain below which `auto`'s proposer is a **clear** loser, demoted at the first step
/// its measurement can decide (sc-24446) — once it holds [`CLEAR_LOSS_MIN_TIMED_STEPS`] timed
/// speculative steps, before the request's first window measured — rather than at the end of a
/// [`ACCEPTANCE_PROBE_VERIFIES`]-step window. The gain checked is an **optimistic** one: the
/// timed steps' tokens per step at [`CLEAR_LOSS_CONFIDENCE_Z`] standard errors above their mean,
/// so a winner's unlucky stretch of acceptance does not read as a loss.
///
/// Why (cuda-campaign-b4 @0826a16cf, attached to Shortcut epic 24432): under CUDA graphs Bonsai's
/// `auto` prompt lookup lost 5.6–18.3 % on every prompt against its `off` twin even though each
/// request was demoted on a measured gain of 0.66–0.87 — the window's 16 slow verify steps (at
/// 2.0–3.1 plain steps each) and their shapes' warm-ups are most of that loss. Where the verify
/// widths keep changing (a lookup's draft count), the first window's steps are mostly shape
/// warm-ups, so it had too few timed steps to measure and fell back to the static 0.21 the
/// grounded `rag_answer` passed (mal 1.31) — demoted one window later (step 78) at gain 0.66.
///
/// 0.8 is four times [`MEASURED_GAIN_MARGIN`] below break-even, so in-process timing noise
/// (CUDA p95 2.2 %, MLX p95 10 %) cannot reach it from a winner. A request whose verify step
/// costs `r` measures a gain of at least `1 / r`, so no proposer with `r < 1.25` can fall below
/// it — the campaigns' measured winners (Qwen3-8B lookup on Candle, `r` 1.02–1.18, gain
/// 1.13–1.24; Qwen3.8 MTP, gain 1.16–1.56; Qwen3.6 MTP, gain 1.08–1.72) sit at or above 1.08;
/// every request between 0.8 and 0.95 is still judged on its whole window.
///
/// Acceptance is lumpy, so a point estimate over a short stretch is not enough: a Qwen3.6-like
/// MTP request (`r` 2.0, mal 1.25 at depth 3) accepts at most 4 drafts in 8 steps about 5 % of
/// the time, which reads as gain < 0.8, and re-testing a running mean every step from the eighth
/// demoted such a winner in about half of 150-step requests (seeded simulation, sc-24446
/// review). Hence the optimistic gain, a longer stretch ([`CLEAR_LOSS_MIN_TIMED_STEPS`]) and a
/// check that stops once a window has measured (from then on its window decides): the same
/// simulation demotes well under 1 % of such winners, while a clear loser still goes before its
/// second window ends.
pub const CLEAR_LOSS_GAIN: f64 = 0.8;

/// Timed speculative steps the clear-loss check needs before it can decide: three quarters of a
/// window.
pub const CLEAR_LOSS_MIN_TIMED_STEPS: u32 = ACCEPTANCE_PROBE_VERIFIES * 3 / 4;

/// The most timed steps the clear-loss check holds: four windows' worth, the newest. It judges
/// only until a window measures; a request whose windows keep falling short of
/// [`MIN_TIMED_WINDOW_STEPS`] is judged on its most recent steps, never a growing history.
pub const CLEAR_LOSS_MAX_TIMED_STEPS: usize = 4 * ACCEPTANCE_PROBE_VERIFIES as usize;

/// Standard errors (of the timed steps' tokens per step, from their sample variance) the
/// clear-loss check adds to the mean before comparing the gain with [`CLEAR_LOSS_GAIN`]: the
/// one-sided 95 % normal quantile.
pub const CLEAR_LOSS_CONFIDENCE_Z: f64 = 1.645;

/// A timed speculative step whose wall time **per verified token** (`1 + drafts`) is longer than
/// this many times the stretch's median per-token step time (a window, or the clear-loss
/// stretch) is a **stall** — a host hiccup, not the proposer's cost — and is left out of that
/// stretch's measured gain, its tokens with its time (sc-24446). Per verified token, so a stretch
/// of mostly single-token steps (a lookup that mostly finds nothing) cannot read its wide verify
/// steps as stalls: a verify costs at most about one plain step per token it verifies.
///
/// Why (cuda-campaign-a4f1 `f1-qwen38-graphs-off` epic-2, `creative`): a Qwen3.8 MTP request
/// whose windows measured `r` 1.66–1.82 and gains 1.46–2.26 had one window's mean verify step
/// pushed from ~122 ms to 227 ms (`r` 3.12, at least 1.7 s of stall in sixteen steps); its gain
/// read 0.70 and the window demoted a winner, which then decoded at 11.6 tok/s against 14.6 plain.
/// A mean step cost lets one stalled step decide a window; the median of the stretch does not
/// move with it. The plain side is already a median ([`PLAIN_PROBE_STEPS`]).
///
/// 4 sits above the spread of legitimate per-token costs inside one stretch: the campaigns'
/// verify steps cost 1.02–3.1 plain steps over widths 1–5, between ~0.4 and ~1.0 plain step per
/// verified token, so no real step reaches 4× the stretch's median while most of its steps are
/// real; a stall of a second or more in a ~100 ms step does.
pub const STALL_STEP_FACTOR: u64 = 4;

/// One engine step as the monitor sees it (sc-24446).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepObservation {
    /// Drafts the step's verify accepted.
    pub accepted: usize,
    /// Drafts the step verified (`0`: a single-token step — a plain probe step, or a proposal that
    /// found nothing).
    pub drafts: usize,
    /// The step's wall time between its marks; `None` when the engine does not time.
    pub elapsed: Option<Duration>,
}

/// What decided a window ([`MonitorDecision::basis`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemotionBasis {
    /// The window's measured gain against the request's measured plain-step cost.
    Measured,
    /// The static break-even threshold ([`demotion_threshold`]): the run was not timed, or the
    /// window had too few timed steps / plain samples — or, on MLX's pipelined path, the
    /// threshold demoted where the measured gain did not.
    Static,
}

impl DemotionBasis {
    /// The wire label: `measured` / `static`.
    pub fn label(self) -> &'static str {
        match self {
            DemotionBasis::Measured => "measured",
            DemotionBasis::Static => "static",
        }
    }
}

/// The inputs and outcome of the last window `auto`'s acceptance monitor judged — the demoting
/// window on a demoted request (sc-24446). Carried on
/// [`DecodeReport::speculative_monitor`](crate::DecodeReport::speculative_monitor) so a campaign
/// row can check the decision against its own timings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MonitorDecision {
    /// The window's 1-based index among the request's speculative windows.
    pub window: u32,
    /// Verify steps in the window — for a clear-loss decision ([`CLEAR_LOSS_GAIN`]), in the
    /// stretch it judged (since the speculative phase began).
    pub verifies: u32,
    /// Drafts accepted over those verify steps.
    pub accepted: u64,
    /// The timed steps among them (past their shape's warm-up, with a non-zero mark), less any
    /// stall ([`STALL_STEP_FACTOR`]).
    pub timed_steps: u32,
    /// Tokens those steps committed (`1 + accepted` each).
    pub timed_tokens: u64,
    /// Their summed wall time, in nanoseconds.
    pub timed_ns: u64,
    /// The request's plain-step cost when the window closed (the median timed single-token step,
    /// nanoseconds); `None` with fewer than [`PLAIN_PROBE_STEPS`] samples.
    pub plain_step_ns: Option<u64>,
    /// What decided the window.
    pub basis: DemotionBasis,
    /// Whether the window demoted the request.
    pub demoted: bool,
}

impl MonitorDecision {
    /// The window's mean accepted drafts per verify.
    pub fn mean_accepted_length(&self) -> f64 {
        self.accepted as f64 / f64::from(self.verifies.max(1))
    }

    /// The window's measured verify cost in plain steps, `r` (its mean timed step over the plain
    /// step); `None` without timed steps or a plain-step cost.
    pub fn verify_cost_ratio(&self) -> Option<f64> {
        let plain = self.plain_step_ns? as f64;
        (self.timed_steps > 0 && plain > 0.0)
            .then(|| self.timed_ns as f64 / f64::from(self.timed_steps) / plain)
    }

    /// The window's measured gain over plain decoding: tokens its timed steps committed per
    /// plain-step time, `(1 + mal) / r`; `None` where [`verify_cost_ratio`](Self::verify_cost_ratio)
    /// is.
    pub fn gain(&self) -> Option<f64> {
        let plain = self.plain_step_ns? as f64;
        (self.timed_ns > 0).then(|| self.timed_tokens as f64 * plain / self.timed_ns as f64)
    }
}

/// `auto`'s acceptance monitor (sc-24446, E5): the backend-neutral policy that stops a request's
/// proposer once it is measurably slower than plain decoding. Both engines feed it every step
/// ([`observe_step`](Self::observe_step): drafts accepted, drafts verified, the step's wall time —
/// a step whose proposer found nothing counts as `0` accepted, as it does in
/// [`DecodeReport::mean_accepted_length`](crate::DecodeReport::mean_accepted_length)).
///
/// It judges **rolling windows**: every [`ACCEPTANCE_PROBE_VERIFIES`] speculative steps it
/// decides on the window just closed — the most recent [`ACCEPTANCE_PROBE_VERIFIES`] steps, not
/// the request so far. A timed monitor ([`for_request`](Self::for_request) with `timed`) first
/// runs a short plain probe ([`probing`](Self::probing), [`PLAIN_PROBE_MAX_STEPS`] steps) and
/// then decides each window on its **measured gain** against the request's own plain-step cost
/// (see the cost-aware notes above [`SHAPE_WARMUP_STEPS`]); where timing cannot decide, and for
/// an untimed monitor, a window below the proposer's static [`demotion_threshold`] demotes. A
/// clear loser — measured gain below [`CLEAR_LOSS_GAIN`] over the timed steps since the last
/// measured window — is demoted at the first step that measurement holds
/// [`MIN_TIMED_WINDOW_STEPS`] timed steps, without waiting for its window to close. The
/// first demoting decision ends the proposer for the request — no more proposals, no more draft
/// rows, the rest decoded token-at-a-time (on MLX through the pipelined loop) — and the report
/// records where ([`DecodeReport::speculative_demoted_at`](crate::DecodeReport::speculative_demoted_at))
/// and on what ([`last_decision`](Self::last_decision)). The demotion is irreversible for the
/// request.
///
/// Why a rolling window rather than one decision at the first window or a cumulative mean: with
/// one decision at the first window, requests that passed it lost for the rest of the request.
/// On MLX (mlx-campaign-4 @1bcb23e6a, every process undemoted) `f3-qwen3-8b` open-ended lookup
/// accepted 29–32 drafts over 223–235 verifies (mal 0.09–0.14 against 0.4, −25 %): its first
/// window passed with at least 7, leaving at most 25 for the next 200+ verifies — a rolling
/// window demotes it by its fifth window at the latest (four passing windows need 28). `f2-bonsai`
/// open-ended companion MTP (mal 0.66–0.85 against 1.2, −22..−42 %) is demoted by its sixth. A
/// cumulative mean lets an early window's credit hide a later loss for as many verifies as it
/// banked, where a window bounds the slow steps after acceptance collapses to
/// [`ACCEPTANCE_PROBE_VERIFIES`]. It also judges a request by what its proposer is doing now: a
/// grounded stretch keeps the proposer however its preamble went (as long as no earlier window
/// demoted it).
///
/// Only [`Speculative::Auto`](crate::Speculative::Auto) is monitored
/// ([`for_request`](Self::for_request)): `auto` is the engine's choice of proposer, so the engine
/// may withdraw it; an explicit `{proposer, depth}` is the caller's choice and runs as asked.
#[derive(Clone, Debug, PartialEq)]
pub struct AcceptanceMonitor {
    /// The static break-even ([`demotion_threshold`]); `None` where none was measured.
    threshold: Option<f64>,
    plain: PlainDecode,
    /// Whether the engine times its steps (a plain probe, measured decisions).
    timed: bool,
    /// Plain probe steps taken.
    probe_steps: u32,
    /// Timed single-token step samples, nanoseconds.
    plain_ns: Vec<u64>,
    /// Steps seen per verify width (`1 + drafts`).
    shapes: Vec<(usize, u32)>,
    /// Windows judged.
    windows: u32,
    /// Verify steps observed in the current window.
    verifies: u32,
    /// Drafts accepted in the current window.
    accepted: u64,
    /// The current window's timed steps.
    window_timed: Vec<TimedStep>,
    /// Since the request's speculative phase began — a window with too few timed steps to
    /// measure carries its own forward — the verify steps and the drafts they accepted, and the
    /// newest [`CLEAR_LOSS_MAX_TIMED_STEPS`] timed steps among them: what [`CLEAR_LOSS_GAIN`] is
    /// checked against after every step until a window measures.
    clear_verifies: u32,
    clear_accepted: u64,
    clear_timed: Vec<TimedStep>,
    /// Whether a window has decided on its measured gain (the clear-loss check stops there).
    measured: bool,
    demoted: bool,
    last: Option<MonitorDecision>,
}

impl AcceptanceMonitor {
    /// The monitor a request runs under: `Some` only when the request asked for
    /// [`Speculative::Auto`](crate::Speculative::Auto) and it resolved to a proposer (`proposer` /
    /// `depth` are what the engine will actually run, after any route fallback) the monitor can
    /// judge against the `plain` loop a demotion would fall back to — one with a static
    /// [`demotion_threshold`] there, or, when the engine times its steps (`timed`), MTP or prompt
    /// lookup at a non-zero depth (its own measured costs decide).
    pub fn for_request(
        mode: crate::Speculative,
        proposer: ProposerKind,
        depth: u32,
        plain: PlainDecode,
        timed: bool,
    ) -> Option<Self> {
        if mode != crate::Speculative::Auto {
            return None;
        }
        let threshold = demotion_threshold(proposer, depth, plain);
        let measurable = timed
            && depth > 0
            && matches!(proposer, ProposerKind::Mtp | ProposerKind::PromptLookup);
        if threshold.is_none() && !measurable {
            return None;
        }
        Some(Self {
            plain,
            timed: measurable,
            ..Self::untimed(threshold)
        })
    }

    /// An untimed monitor demoting below `threshold` mean accepted drafts per verify (a finite,
    /// non-negative value; `None` otherwise).
    pub fn with_threshold(threshold: f64) -> Option<Self> {
        (threshold.is_finite() && threshold >= 0.0).then(|| Self::untimed(Some(threshold)))
    }

    fn untimed(threshold: Option<f64>) -> Self {
        Self {
            threshold,
            plain: PlainDecode::Candle,
            timed: false,
            probe_steps: 0,
            plain_ns: Vec::new(),
            shapes: Vec::new(),
            windows: 0,
            verifies: 0,
            accepted: 0,
            window_timed: Vec::new(),
            clear_verifies: 0,
            clear_accepted: 0,
            clear_timed: Vec::new(),
            measured: false,
            demoted: false,
            last: None,
        }
    }

    /// The static break-even mean accepted length this monitor falls back to; `None` where none
    /// was measured (only measured gains decide).
    pub fn threshold(&self) -> Option<f64> {
        self.threshold
    }

    /// Whether the next step is a plain probe step: the engine runs it without drafts (`k = 0`,
    /// the plain loop's draw) and does not propose. Only a timed monitor probes, before its first
    /// window, until it holds [`PLAIN_PROBE_STEPS`] timed plain samples or has taken
    /// [`PLAIN_PROBE_MAX_STEPS`] probe steps.
    pub fn probing(&self) -> bool {
        self.timed
            && !self.demoted
            && self.plain_ns.len() < PLAIN_PROBE_STEPS as usize
            && self.probe_steps < PLAIN_PROBE_MAX_STEPS
    }

    /// The last window judged — the demoting one on a demoted request; `None` before the first.
    pub fn last_decision(&self) -> Option<MonitorDecision> {
        self.last
    }

    /// [`observe_step`](Self::observe_step) for an untimed step that accepted `accepted` drafts.
    pub fn observe(&mut self, accepted: usize) -> bool {
        self.observe_step(StepObservation {
            accepted,
            drafts: accepted,
            elapsed: None,
        })
    }

    /// Observe one engine step. Returns `true` exactly once — on the step that closes the first
    /// demoting window, or the earlier step at which a clear loser's measured gain falls below
    /// [`CLEAR_LOSS_GAIN`] — when the engine must demote; `false` otherwise, including every probe
    /// step and every step after the demotion.
    pub fn observe_step(&mut self, step: StepObservation) -> bool {
        if self.demoted {
            return false;
        }
        let probe = self.probing();
        let width = 1 + step.drafts;
        let seen = match self.shapes.iter_mut().find(|(w, _)| *w == width) {
            Some((_, n)) => {
                *n += 1;
                *n - 1
            }
            None => {
                self.shapes.push((width, 1));
                0
            }
        };
        let elapsed_ns = step
            .elapsed
            .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
            .filter(|&ns| self.timed && ns > 0 && seen >= SHAPE_WARMUP_STEPS);
        if step.drafts == 0 {
            self.plain_ns.extend(elapsed_ns);
        }
        if probe {
            self.probe_steps += 1;
            return false;
        }
        self.verifies += 1;
        self.accepted += step.accepted as u64;
        self.clear_verifies += 1;
        self.clear_accepted += step.accepted as u64;
        if let Some(ns) = elapsed_ns {
            let timed = TimedStep {
                tokens: 1 + step.accepted as u64,
                width: width as u64,
                ns,
            };
            self.window_timed.push(timed);
            if !self.measured {
                if self.clear_timed.len() == CLEAR_LOSS_MAX_TIMED_STEPS {
                    self.clear_timed.remove(0);
                }
                self.clear_timed.push(timed);
            }
        }
        let decision = if let Some(clear) = self.clear_loss() {
            clear
        } else if self.verifies < ACCEPTANCE_PROBE_VERIFIES {
            return false;
        } else {
            self.decide()
        };
        if decision.basis == DemotionBasis::Measured {
            self.measured = true;
        }
        self.last = Some(decision);
        self.demoted = decision.demoted;
        self.verifies = 0;
        self.accepted = 0;
        self.window_timed.clear();
        if self.measured {
            self.clear_timed = Vec::new();
        }
        self.demoted
    }

    /// The request's plain-step cost: the median timed single-token step, once it holds
    /// [`PLAIN_PROBE_STEPS`] samples.
    fn plain_step_ns(&self) -> Option<u64> {
        (self.plain_ns.len() >= PLAIN_PROBE_STEPS as usize).then(|| {
            let mut sorted = self.plain_ns.clone();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        })
    }

    /// The demoting decision for a **clear** loser ([`CLEAR_LOSS_GAIN`]): `Some` as soon as no
    /// window has measured yet, the timed steps since the speculative phase began number
    /// [`CLEAR_LOSS_MIN_TIMED_STEPS`] and even their optimistic gain
    /// ([`CLEAR_LOSS_CONFIDENCE_Z`]) is below it — mid-window; the decision reports those steps:
    /// their verify and acceptance counts with the timed steps among them. `None` otherwise (the
    /// window decides at its end as usual).
    fn clear_loss(&mut self) -> Option<MonitorDecision> {
        if self.measured || self.clear_timed.len() < CLEAR_LOSS_MIN_TIMED_STEPS as usize {
            return None;
        }
        let plain_step_ns = self.plain_step_ns()?;
        let stretch = Stretch::of(&self.clear_timed);
        if stretch.steps < CLEAR_LOSS_MIN_TIMED_STEPS {
            return None;
        }
        let decision = MonitorDecision {
            window: self.windows + 1,
            verifies: self.clear_verifies,
            accepted: self.clear_accepted,
            timed_steps: stretch.steps,
            timed_tokens: stretch.tokens,
            timed_ns: stretch.ns,
            plain_step_ns: Some(plain_step_ns),
            basis: DemotionBasis::Measured,
            demoted: true,
        };
        let gain = decision.gain()?;
        let optimistic = gain * stretch.optimistic_tokens_factor();
        if optimistic.is_nan() || optimistic >= CLEAR_LOSS_GAIN {
            return None;
        }
        self.windows += 1;
        Some(decision)
    }

    /// Judge the window just closed (see the cost-aware notes above [`SHAPE_WARMUP_STEPS`]).
    fn decide(&mut self) -> MonitorDecision {
        self.windows += 1;
        let plain_step_ns = self.plain_step_ns();
        let stretch = Stretch::of(&self.window_timed);
        let mut decision = MonitorDecision {
            window: self.windows,
            verifies: self.verifies,
            accepted: self.accepted,
            timed_steps: stretch.steps,
            timed_tokens: stretch.tokens,
            timed_ns: stretch.ns,
            plain_step_ns,
            basis: DemotionBasis::Static,
            demoted: false,
        };
        let measured_loss = decision
            .gain()
            .filter(|_| stretch.steps >= MIN_TIMED_WINDOW_STEPS)
            .map(|gain| gain < 1.0 - MEASURED_GAIN_MARGIN);
        let static_loss = self
            .threshold
            .is_some_and(|t| (self.accepted as f64) < t * f64::from(self.verifies));
        (decision.demoted, decision.basis) = match (self.plain, measured_loss) {
            // The probe times unpipelined plain steps, dearer than the pipelined loop a demoted
            // request continues on: a measured loss is real, a measured win unproven.
            (PlainDecode::MlxPipelined, Some(true)) => (true, DemotionBasis::Measured),
            (PlainDecode::MlxPipelined, measured) if static_loss || measured.is_none() => {
                (static_loss, DemotionBasis::Static)
            }
            (_, Some(loss)) => (loss, DemotionBasis::Measured),
            (_, None) => (static_loss, DemotionBasis::Static),
        };
        decision
    }
}

/// One timed speculative step: the tokens it committed, its verify width (`1 + drafts`) and its
/// wall time.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TimedStep {
    tokens: u64,
    width: u64,
    ns: u64,
}

/// A stretch's timed steps less its stalls ([`STALL_STEP_FACTOR`]): their count, the tokens they
/// committed (and the squares, for their variance) and their summed wall time.
struct Stretch {
    steps: u32,
    tokens: u64,
    tokens_sq: u64,
    ns: u64,
}

impl Stretch {
    fn of(timed: &[TimedStep]) -> Self {
        // Wall time per verified token, the median of which bounds a real step's.
        let per_token = |s: &TimedStep| s.ns / s.width.max(1);
        let mut costs: Vec<u64> = timed.iter().map(per_token).collect();
        costs.sort_unstable();
        let stall = costs
            .get(costs.len() / 2)
            .map_or(u64::MAX, |median| median.saturating_mul(STALL_STEP_FACTOR));
        let mut stretch = Self {
            steps: 0,
            tokens: 0,
            tokens_sq: 0,
            ns: 0,
        };
        for s in timed.iter().filter(|s| per_token(s) <= stall) {
            stretch.steps += 1;
            stretch.tokens += s.tokens;
            stretch.tokens_sq += s.tokens * s.tokens;
            stretch.ns = stretch.ns.saturating_add(s.ns);
        }
        stretch
    }

    /// The clear-loss check's optimism: the steps' mean tokens per step plus
    /// [`CLEAR_LOSS_CONFIDENCE_Z`] standard errors (their sample variance), over the mean.
    fn optimistic_tokens_factor(&self) -> f64 {
        let n = f64::from(self.steps);
        let mean = self.tokens as f64 / n;
        let variance = ((self.tokens_sq as f64 / n - mean * mean) * n / (n - 1.0)).max(0.0);
        (mean + CLEAR_LOSS_CONFIDENCE_Z * (variance / n).sqrt()) / mean
    }
}

/// The greedy verify decision in one call: the committed run (accepted drafts + the bonus token)
/// and how many drafts were accepted, from the target's per-position argmax
/// (`target_argmax.len() == drafts.len() + 1`, see [`accept_greedy_run`]). Every committed token is
/// the target's own greedy choice, so a greedy speculative run commits exactly what token-at-a-time
/// greedy decoding would have chosen at each position.
pub fn greedy_commit(target_argmax: &[i32], drafts: &[i32]) -> (Vec<i32>, usize) {
    let accepted = accept_greedy_run(target_argmax, drafts);
    let mut committed = drafts[..accepted].to_vec();
    committed.push(target_argmax[accepted]);
    (committed, accepted)
}

/// Propose a continuation by **prompt lookup**: find the most recent earlier occurrence of the
/// sequence's trailing n-gram and return the tokens that followed it (story 7171).
///
/// Tries the longest n-gram first (down to length 1): for each size it looks for the rightmost match
/// of the last `n` tokens *strictly before* the trailing copy, and returns up to `max_proposal` tokens
/// that followed that earlier match. Returns empty when nothing matches or inputs are degenerate —
/// the caller then falls back to a single-token step. No draft model, no tensors.
pub fn ngram_propose(tokens: &[i32], max_ngram: usize, max_proposal: usize) -> Vec<i32> {
    let len = tokens.len();
    if len < 2 || max_ngram == 0 || max_proposal == 0 {
        return Vec::new();
    }
    let max_n = max_ngram.min(len - 1);
    for n in (1..=max_n).rev() {
        let suffix = &tokens[len - n..];
        // Search earlier start positions, most-recent first; the match must end before the suffix.
        for start in (0..=len - n - 1).rev() {
            if &tokens[start..start + n] == suffix {
                let from = start + n;
                let to = (from + max_proposal).min(len);
                if from < to {
                    return tokens[from..to].to_vec();
                }
            }
        }
    }
    Vec::new()
}

/// How many leading draft tokens a **greedy** target accepts: the length of the prefix of `drafts`
/// equal, position by position, to the target's argmax (`target_argmax[i]` is the target's greedy
/// token at draft position `i`). The committed run is then those accepted drafts followed by the
/// bonus token `target_argmax[accepted]` — every committed token equals the target's own greedy
/// choice, so greedy speculative output is identical to non-speculative greedy output.
///
/// `target_argmax.len()` must be `drafts.len() + 1` (one extra for the always-present bonus). Returns
/// a value in `0..=drafts.len()`.
pub fn accept_greedy_run(target_argmax: &[i32], drafts: &[i32]) -> usize {
    debug_assert_eq!(
        target_argmax.len(),
        drafts.len() + 1,
        "need a bonus slot past the drafts"
    );
    let mut accepted = 0;
    while accepted < drafts.len() && target_argmax[accepted] == drafts[accepted] {
        accepted += 1;
    }
    accepted
}

/// The outcome of a stochastic [`accept_token`] step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Acceptance {
    /// The proposed token was accepted; the run may continue to the next position.
    Accepted(i32),
    /// The proposal was rejected; `i32` is the bonus token resampled from the residual, and the run
    /// ends here.
    Rejected(i32),
}

impl Acceptance {
    /// The committed token regardless of outcome.
    pub fn token(self) -> i32 {
        match self {
            Acceptance::Accepted(t) | Acceptance::Rejected(t) => t,
        }
    }

    /// Whether the proposal was accepted (the run continues).
    pub fn is_accepted(self) -> bool {
        matches!(self, Acceptance::Accepted(_))
    }
}

/// One distribution-preserving speculative-sampling step.
///
/// `target` and `draft` are `(token, weight)` candidate sets (weights need not be normalized — each
/// is normalized over its own set). `proposed` is the token the draft sampled. `u_accept` and
/// `u_resample` are independent uniform `[0, 1)` draws from the backend's RNG.
///
/// Accepts `proposed` with probability `min(1, p(proposed)/q(proposed))`; on rejection resamples a
/// bonus token from the normalized residual `max(0, p − q)`. The committed token is distributed
/// exactly as the (normalized) `target`.
pub fn accept_token(
    target: &[(i32, f32)],
    draft: &[(i32, f32)],
    proposed: i32,
    u_accept: f32,
    u_resample: f32,
) -> Acceptance {
    let p_total: f32 = target.iter().map(|&(_, w)| w.max(0.0)).sum();
    let q_total: f32 = draft.iter().map(|&(_, w)| w.max(0.0)).sum();
    let p_of = |t: i32| weight_of(target, t).max(0.0) / p_total.max(f32::MIN_POSITIVE);
    let q_of = |t: i32| weight_of(draft, t).max(0.0) / q_total.max(f32::MIN_POSITIVE);

    let p_t = p_of(proposed);
    let q_t = q_of(proposed);
    // q_t should be > 0 (proposed was drawn from q); guard anyway. Accept w.p. min(1, p/q).
    let accept_prob = if q_t > 0.0 { (p_t / q_t).min(1.0) } else { 1.0 };
    if u_accept < accept_prob {
        return Acceptance::Accepted(proposed);
    }

    // Reject: resample from the normalized residual max(0, p - q) over the union of supports.
    let mut residual: Vec<(i32, f32)> = Vec::with_capacity(target.len() + draft.len());
    for &(t, _) in target.iter().chain(draft.iter()) {
        if residual.iter().all(|&(s, _)| s != t) {
            let r = p_of(t) - q_of(t);
            if r > 0.0 {
                residual.push((t, r));
            }
        }
    }
    Acceptance::Rejected(sample_weighted(&residual, u_resample, proposed))
}

/// Draw a token from a `(token, weight)` candidate set by inverse-CDF over the normalized weights.
/// `fallback` is returned only if the set is empty or all-zero (degenerate). Public so a backend can
/// draw the post-all-accepted bonus from the final target distribution with the same policy.
pub fn sample_weighted(candidates: &[(i32, f32)], u: f32, fallback: i32) -> i32 {
    let total: f32 = candidates.iter().map(|&(_, w)| w.max(0.0)).sum();
    if total <= 0.0 {
        return fallback;
    }
    let mut target = u.clamp(0.0, 1.0) * total;
    for &(t, w) in candidates {
        target -= w.max(0.0);
        if target <= 0.0 {
            return t;
        }
    }
    candidates.last().map(|&(t, _)| t).unwrap_or(fallback)
}

/// Weight of `token` in a candidate set (`0` if absent).
fn weight_of(candidates: &[(i32, f32)], token: i32) -> f32 {
    candidates
        .iter()
        .find(|&&(t, _)| t == token)
        .map(|&(_, w)| w)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests;
