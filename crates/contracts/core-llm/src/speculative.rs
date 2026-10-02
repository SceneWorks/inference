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
// verify) is above its threshold on average and is demoted only by a window that falls below it:
// telling it from Qwen3.8 bf16 at the same acceptance (which would win) needs the request's own
// verify cost, which the monitor does not have.

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

/// `auto`'s acceptance monitor (sc-24446, E5): the backend-neutral policy that stops a request's
/// proposer once it is measurably slower than plain decoding. Both engines feed it the accepted
/// draft count of every verify step (a step whose proposer found nothing counts as `0`, as it does
/// in [`DecodeReport::mean_accepted_length`](crate::DecodeReport::mean_accepted_length)).
///
/// It judges **rolling windows**: every [`ACCEPTANCE_PROBE_VERIFIES`] verify steps it decides on
/// the window just closed — the most recent [`ACCEPTANCE_PROBE_VERIFIES`] steps, not the request
/// so far. The first window below the proposer's [`demotion_threshold`] demotes the request — no
/// more proposals, no more draft rows, the rest decoded token-at-a-time (on MLX through the
/// pipelined loop) — and the report records where
/// ([`DecodeReport::speculative_demoted_at`](crate::DecodeReport::speculative_demoted_at)). A
/// window at or above it keeps the proposer for the next window. The demotion is irreversible for
/// the request.
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
/// demoted it). The price is that every window is a fresh chance for noise to fall below the
/// threshold, which is why each threshold is its plain loop's own cheapest-cost break-even.
///
/// Only [`Speculative::Auto`](crate::Speculative::Auto) is monitored
/// ([`for_request`](Self::for_request)): `auto` is the engine's choice of proposer, so the engine
/// may withdraw it; an explicit `{proposer, depth}` is the caller's choice and runs as asked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcceptanceMonitor {
    threshold: f64,
    /// Verify steps observed in the current window.
    verifies: u32,
    /// Drafts accepted in the current window.
    accepted: u64,
    demoted: bool,
}

impl AcceptanceMonitor {
    /// The monitor a request runs under: `Some` only when the request asked for
    /// [`Speculative::Auto`](crate::Speculative::Auto) and it resolved to a proposer with a
    /// [`demotion_threshold`] against the `plain` loop a demotion would fall back to (`proposer` /
    /// `depth` are what the engine will actually run, after any route fallback).
    pub fn for_request(
        mode: crate::Speculative,
        proposer: ProposerKind,
        depth: u32,
        plain: PlainDecode,
    ) -> Option<Self> {
        if mode != crate::Speculative::Auto {
            return None;
        }
        Self::with_threshold(demotion_threshold(proposer, depth, plain)?)
    }

    /// A monitor demoting below `threshold` mean accepted drafts per verify (a finite,
    /// non-negative value; `None` otherwise).
    pub fn with_threshold(threshold: f64) -> Option<Self> {
        (threshold.is_finite() && threshold >= 0.0).then_some(Self {
            threshold,
            verifies: 0,
            accepted: 0,
            demoted: false,
        })
    }

    /// The break-even mean accepted length this monitor demotes below.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Observe one verify step that accepted `accepted` drafts. Returns `true` exactly once — on
    /// the step that closes the first window whose mean is below the threshold — when the engine
    /// must demote; `false` otherwise, including every step after the demotion.
    pub fn observe(&mut self, accepted: usize) -> bool {
        if self.demoted {
            return false;
        }
        self.verifies += 1;
        self.accepted += accepted as u64;
        if self.verifies < ACCEPTANCE_PROBE_VERIFIES {
            return false;
        }
        self.demoted = (self.accepted as f64) < self.threshold * f64::from(self.verifies);
        self.verifies = 0;
        self.accepted = 0;
        self.demoted
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
mod tests {
    use super::*;

    /// A defaults-table row's recommended depths, spelled out so these tests do not move with the
    /// table.
    const DEPTHS: crate::RecommendedDepths = crate::RecommendedDepths {
        mtp: 3,
        prompt_lookup: 4,
        draft_model: 4,
    };

    /// E5: the shared advertisements recommend the depth of the defaults-table row they are handed
    /// — a different row advertises a different depth (still clamped to the verify bound) — and
    /// every backend's row is what the providers hand them.
    #[test]
    fn the_advertised_recommended_depth_is_the_rows() {
        let row = crate::RecommendedDepths {
            mtp: 2,
            prompt_lookup: 6,
            draft_model: 5,
        };
        assert_eq!(prompt_lookup_capabilities(8, &row).recommended_depth, 6);
        assert_eq!(prompt_lookup_capabilities(8, &DEPTHS).recommended_depth, 4);
        assert_eq!(draft_model_capabilities(8, &row).recommended_depth, 5);
        assert_eq!(draft_model_capabilities(8, &DEPTHS).recommended_depth, 4);
        assert_eq!(prompt_lookup_capabilities(3, &row).recommended_depth, 3);
        let mut caps = crate::TextLlmCapabilities {
            speculative: vec![prompt_lookup_capabilities(8, &row)],
            ..Default::default()
        };
        let mut fallbacks = Vec::new();
        settle_draft("/d", Ok(()), &mut caps, &mut fallbacks, &row);
        assert_eq!(
            caps.proposer(crate::SpeculativeProposer::DraftModel)
                .map(|c| c.recommended_depth),
            Some(5)
        );
    }

    // --- n-gram proposer ---

    #[test]
    fn ngram_proposes_continuation_of_recent_match() {
        // "1 2 3 1 2": last bigram "1 2" recurs at index 0, followed by "3" (then the trailing 1 2).
        let toks = [1, 2, 3, 1, 2];
        assert_eq!(ngram_propose(&toks, 3, 1), vec![3]);
        assert_eq!(ngram_propose(&toks, 3, 3), vec![3, 1, 2]);
    }

    #[test]
    fn ngram_prefers_longer_match() {
        // Longer n-gram "x a b" recurs; its continuation is "y", not the "a b"->"z" of the bigram.
        let toks = [9, 1, 2, 4, 7, 9, 1, 2];
        // last trigram is "9 1 2", earlier at index 0 -> followed by "4".
        assert_eq!(ngram_propose(&toks, 3, 1), vec![4]);
    }

    #[test]
    fn ngram_returns_empty_without_a_match() {
        assert_eq!(ngram_propose(&[1, 2, 3, 4], 3, 4), Vec::<i32>::new());
        assert_eq!(ngram_propose(&[1], 3, 4), Vec::<i32>::new());
        assert_eq!(ngram_propose(&[], 3, 4), Vec::<i32>::new());
    }

    // --- mode resolution and the greedy commit (sc-24130) ---

    /// sc-24433 AC2: `auto` resolves to MTP where a head exists, else prompt lookup, else off
    /// with a named reason; an explicit proposer runs as asked; nothing resolves silently.
    #[test]
    fn speculative_option_resolves_against_the_advertised_proposers() {
        use crate::{
            MtpCapabilities, MtpMode, ProposerCapabilities, Speculative, SpeculativeProposer,
            TextLlmCapabilities,
        };
        let lookup = ProposerCapabilities {
            proposer: SpeculativeProposer::PromptLookup,
            max_depth: 8,
            recommended_depth: 4,
        };
        let head = TextLlmCapabilities {
            mtp: Some(MtpCapabilities {
                max_draft_tokens: 5,
                recommended_draft_tokens: 3,
            }),
            speculative: vec![lookup],
            ..Default::default()
        };
        let no_head = TextLlmCapabilities {
            speculative: vec![lookup],
            ..Default::default()
        };
        let nothing = TextLlmCapabilities::default();
        let plan = |mode, caps: &TextLlmCapabilities| resolve_speculative(mode, caps);
        let run = |proposer, depth| SpeculativePlan::Run { proposer, depth };

        assert_eq!(
            plan(Speculative::Off, &head),
            SpeculativeResolution::default()
        );
        let auto = plan(Speculative::Auto, &head);
        assert_eq!(auto.plan, run(SpeculativeProposer::Mtp, 3));
        assert_eq!(auto.fallback, None);
        assert_eq!(auto.plan.proposer(), ProposerKind::Mtp);
        let auto = plan(Speculative::Auto, &no_head);
        assert_eq!(auto.plan, run(SpeculativeProposer::PromptLookup, 4));
        assert_eq!(auto.plan.proposer().label(), "prompt_lookup");
        assert_eq!(auto.plan.depth(), Some(4));
        let auto = plan(Speculative::Auto, &nothing);
        assert_eq!(auto.plan, SpeculativePlan::Off);
        assert_eq!(auto.plan.proposer(), ProposerKind::None);
        assert_eq!(
            auto.fallback.as_deref(),
            Some(
                "speculative: auto found no proposer this model can run (it advertises none; \
                 decoded without a proposer)"
            )
        );

        // The legacy mode is the same option.
        assert_eq!(
            plan(MtpMode::Enabled { draft_tokens: 3 }.into(), &head).plan,
            run(SpeculativeProposer::Mtp, 3)
        );
        assert_eq!(
            plan(MtpMode::Auto.into(), &no_head).plan,
            auto_plan(&no_head)
        );

        let explicit = Speculative::proposer(SpeculativeProposer::PromptLookup, 2);
        assert_eq!(
            plan(explicit, &head).plan,
            run(SpeculativeProposer::PromptLookup, 2)
        );
        let unadvertised = plan(
            Speculative::proposer(SpeculativeProposer::DraftModel, 2),
            &head,
        );
        assert_eq!(unadvertised.plan, SpeculativePlan::Off);
        assert!(unadvertised
            .fallback
            .unwrap()
            .contains("`draft_model` is not available"));
        let too_deep = plan(
            Speculative::proposer(SpeculativeProposer::PromptLookup, 20),
            &head,
        );
        assert_eq!(too_deep.plan, run(SpeculativeProposer::PromptLookup, 8));
        assert!(too_deep.fallback.unwrap().contains("depth 20 clamped to 8"));

        assert_eq!(ProposerKind::default(), ProposerKind::None);
        assert_eq!(ProposerKind::PromptLookup.label(), "prompt_lookup");
        assert_eq!(ProposerKind::DraftModel.label(), "draft_model");
        assert_eq!(ProposerKind::Mtp.label(), "mtp");
        assert_eq!(
            ProposerKind::from(SpeculativeProposer::DraftModel),
            ProposerKind::DraftModel
        );
    }

    /// A family that advertises no proposer names its own reason for `auto` and for an explicit
    /// proposer, with the same leading words the resolver uses; `off` names nothing (E2/E8).
    #[test]
    fn a_family_without_proposers_names_its_own_reason() {
        use crate::{Speculative, SpeculativeProposer};
        assert_eq!(
            no_proposer_fallback(Speculative::Off, CAPTIONER_NO_PROPOSER),
            None
        );
        assert_eq!(
            no_proposer_fallback(Speculative::Auto, CAPTIONER_NO_PROPOSER).as_deref(),
            Some(
                "speculative: auto found no proposer this model can run (this captioner / SVG \
                 decoder advertises no proposer: its continuation decodes token-at-a-time after \
                 a multimodal prefill)"
            )
        );
        let explicit = no_proposer_fallback(
            Speculative::proposer(SpeculativeProposer::Mtp, 2),
            CAPTIONER_NO_PROPOSER,
        )
        .unwrap();
        assert!(
            explicit.starts_with("speculative: `mtp` is not available for this model (")
                && explicit.contains(CAPTIONER_NO_PROPOSER),
            "{explicit}"
        );
    }

    /// E8: the shared prompt-lookup advertisement and verify bound — the recommendation never
    /// exceeds the bound, and `draft_model` is advertised at the prompt-lookup bound.
    #[test]
    fn the_shared_advertisements_respect_the_verify_bound() {
        let depths = &crate::DecodeBackend::Mlx.defaults().recommended_depths;
        let deep = prompt_lookup_capabilities(7, depths);
        assert_eq!(
            (deep.max_depth, deep.recommended_depth),
            (7, depths.prompt_lookup)
        );
        let shallow = prompt_lookup_capabilities(2, depths);
        assert_eq!((shallow.max_depth, shallow.recommended_depth), (2, 2));
        let mut caps = crate::TextLlmCapabilities::default();
        assert_eq!(verify_depth_bound(&caps), 1, "no prompt lookup: one draft");
        caps.speculative.push(deep);
        assert_eq!(verify_depth_bound(&caps), 7);
    }

    /// E2 (sc-24436): a refused draft is named in both `LoadReport::draft` and
    /// `LoadReport::fallbacks` and advertises nothing; a resident one advertises `draft_model` at
    /// the verify bound and names no fallback.
    #[test]
    fn settling_a_draft_names_a_refusal_and_advertises_a_resident_draft() {
        use crate::SpeculativeProposer;
        let mut caps = crate::TextLlmCapabilities {
            speculative: vec![prompt_lookup_capabilities(5, &DEPTHS)],
            ..Default::default()
        };
        let mut fallbacks = Vec::new();
        let why = draft_load_refusal("no such file");
        assert_eq!(why, "draft model: its load failed (no such file)");
        let (kept, report): (Option<()>, _) =
            settle_draft("/d", Err(why.clone()), &mut caps, &mut fallbacks, &DEPTHS);
        assert!(kept.is_none());
        assert_eq!(caps.proposer(SpeculativeProposer::DraftModel), None);
        assert_eq!(report, crate::DraftReport::refused("/d", why.clone()));
        assert_eq!(fallbacks, vec![why.clone()]);
        // The load report's own entry point names an admission refusal the same way.
        let mut load = crate::LoadReport::default();
        load.record_draft(crate::DraftReport::refused("/d", why.clone()));
        assert_eq!(load.fallbacks, vec![why]);

        let mut fallbacks = Vec::new();
        let (kept, report) = settle_draft("/d", Ok(7u8), &mut caps, &mut fallbacks, &DEPTHS);
        assert_eq!(kept, Some(7));
        assert_eq!(
            caps.proposer(SpeculativeProposer::DraftModel),
            Some(draft_model_capabilities(5, &DEPTHS))
        );
        assert_eq!(report, crate::DraftReport::resident("/d"));
        assert!(fallbacks.is_empty());
        assert_eq!(
            draft_unpriced_refusal("x"),
            "draft model: its load cannot be priced (x)"
        );
        assert_eq!(draft_refusal("x"), "draft model: x");
    }

    fn word_tokenizer(prefix: &str, vocab: usize) -> crate::Tokenizer {
        let entries: Vec<String> = (0..vocab)
            .map(|i| format!("\"{prefix}{i}\": {i}"))
            .collect();
        crate::Tokenizer::from_json(&format!(
            r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                "decoder": null,
                "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "{prefix}0" }} }}"#,
            entries.join(", ")
        ))
        .unwrap()
    }

    #[test]
    fn a_draft_must_share_the_vocabulary_and_the_logits_width() {
        use crate::{Speculative, SpeculativeProposer};
        let target = word_tokenizer("t", 8);
        assert_eq!(
            draft_compatibility(&target, 8, &word_tokenizer("t", 8), 8),
            Ok(8)
        );
        // A tokenizer of the same size over different tokens is refused by name.
        let why = draft_compatibility(&target, 8, &word_tokenizer("w", 8), 8).unwrap_err();
        assert!(
            why.starts_with("draft model: its tokenizer vocabulary is not the target's"),
            "{why}"
        );
        // A draft padded less than its target (Qwen2.5-0.5B's 151936 rows beside 7B's 152064) is
        // compatible; one padded past its tokenizer proposes only the tokenizer's ids.
        assert_eq!(
            draft_compatibility(&target, 16, &word_tokenizer("t", 8), 12),
            Ok(8)
        );
        assert_eq!(
            draft_compatibility(&target, 16, &word_tokenizer("t", 8), 8),
            Ok(8)
        );
        // A draft scoring more ids than its target is refused.
        let why = draft_compatibility(&target, 8, &word_tokenizer("t", 8), 16).unwrap_err();
        assert!(
            why.contains("scores 16 token ids, more than the target's 8"),
            "{why}"
        );

        let caps = draft_model_capabilities(8, &DEPTHS);
        assert_eq!(caps.proposer, SpeculativeProposer::DraftModel);
        assert_eq!((caps.max_depth, caps.recommended_depth), (8, 4));
        // The provider's bound is the advertisement's; the recommendation never exceeds it.
        let shallow = draft_model_capabilities(3, &DEPTHS);
        assert_eq!((shallow.max_depth, shallow.recommended_depth), (3, 3));
        // Advertised, `{proposer: draft_model}` runs at the requested depth.
        let resident = crate::TextLlmCapabilities {
            speculative: vec![caps],
            ..Default::default()
        };
        assert_eq!(
            resolve_speculative(
                Speculative::proposer(SpeculativeProposer::DraftModel, 3),
                &resident
            ),
            SpeculativeResolution {
                plan: SpeculativePlan::Run {
                    proposer: SpeculativeProposer::DraftModel,
                    depth: 3
                },
                fallback: None
            }
        );
    }

    /// sc-24436 E2: a `draft_model` request past the draft's context window runs what `auto`
    /// resolves to, by name; within it (or with an unbounded draft) the plan is unchanged.
    #[test]
    fn a_request_past_the_draft_context_runs_auto_by_name() {
        use crate::{ProposerCapabilities, Speculative, SpeculativeProposer};
        let lookup = ProposerCapabilities {
            proposer: SpeculativeProposer::PromptLookup,
            max_depth: 8,
            recommended_depth: 4,
        };
        let caps = crate::TextLlmCapabilities {
            speculative: vec![lookup, draft_model_capabilities(8, &DEPTHS)],
            ..Default::default()
        };
        let asked = resolve_speculative(
            Speculative::proposer(SpeculativeProposer::DraftModel, 3),
            &caps,
        );
        // Depth 3: the draft cache holds the prompt, the budget and a step's 3 + 1 positions.
        assert_eq!(fit_draft_context(asked.clone(), &caps, 64, 40, 20), asked);
        assert_eq!(fit_draft_context(asked.clone(), &caps, 0, 4000, 24), asked);
        let fit = fit_draft_context(asked.clone(), &caps, 64, 40, 21);
        assert_eq!(
            fit.plan,
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::PromptLookup,
                depth: 4
            }
        );
        let why = fit.fallback.unwrap();
        assert!(
            why.contains("exceeds the draft model's context window 64")
                && why.contains("`prompt_lookup` at depth 4"),
            "{why}"
        );
        // No proposer for `auto` either: plain decoding, both reasons named.
        let only_draft = crate::TextLlmCapabilities {
            speculative: vec![draft_model_capabilities(8, &DEPTHS)],
            ..Default::default()
        };
        let fit = fit_draft_context(asked, &only_draft, 64, 40, 21);
        assert_eq!(fit.plan, SpeculativePlan::Off);
        let why = fit.fallback.unwrap();
        assert!(
            why.contains("without a proposer") && why.contains("auto found no proposer"),
            "{why}"
        );
        // Any other plan is untouched.
        let lookup_plan = resolve_speculative(
            Speculative::proposer(SpeculativeProposer::PromptLookup, 3),
            &caps,
        );
        assert_eq!(
            fit_draft_context(lookup_plan.clone(), &caps, 64, 40, 25),
            lookup_plan
        );
    }

    /// sc-24438 AC1: a request above the advertised max depth — new option or legacy `mtp` —
    /// runs at the max, and the clamp is named; a request within it is untouched.
    #[test]
    fn a_too_deep_request_is_clamped_to_the_advertised_max_and_named() {
        use crate::{MtpMode, Speculative, SpeculativeProposer, TextLlmCapabilities};
        let mut caps = TextLlmCapabilities::default();
        caps.advertise_mtp(7, 3);
        let legacy = resolve_speculative(MtpMode::Enabled { draft_tokens: 40 }.into(), &caps);
        assert_eq!(
            legacy.plan,
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::Mtp,
                depth: 7
            }
        );
        assert_eq!(
            legacy.fallback.as_deref(),
            Some("speculative: `mtp` depth 40 clamped to 7 (advertised 1..=7)")
        );
        let within = resolve_speculative(Speculative::proposer(SpeculativeProposer::Mtp, 7), &caps);
        assert_eq!(within.plan.depth(), Some(7));
        assert_eq!(within.fallback, None);
    }

    fn auto_plan(caps: &crate::TextLlmCapabilities) -> SpeculativePlan {
        resolve_speculative(crate::Speculative::Auto, caps).plan
    }

    // --- auto's acceptance monitor (sc-24446) ---

    /// E5: the thresholds are each plain loop's cheapest-measured break-evens — on MLX prompt
    /// lookup 0.4 at any depth but only where the demoted request would continue on the pipelined
    /// loop (the measured regime), MTP 0.25 at one draft growing 0.475 per extra draft (1.2 at the
    /// recommended depth 3); on Candle (cuda-campaign-a) lookup 0.09 growing 0.04 per draft (0.21
    /// at `auto`'s depth 4), MTP 0.25 growing 0.125 (0.5 at depth 3) — and nothing is monitored
    /// without a measured cost (no proposer, a zero depth, `draft_model`, lookup falling back to
    /// MLX's unpipelined loop).
    #[test]
    fn the_demotion_thresholds_are_the_measured_break_evens() {
        use PlainDecode::{Candle, MlxPipelined, MlxUnpipelined};
        let close = |a: Option<f64>, b: f64| (a.unwrap() - b).abs() < 1e-12;
        let lookup = |depth, plain| demotion_threshold(ProposerKind::PromptLookup, depth, plain);
        let mtp = |depth, plain| demotion_threshold(ProposerKind::Mtp, depth, plain);
        assert!(close(lookup(4, MlxPipelined), 0.4));
        assert!(close(lookup(1, MlxPipelined), 0.4));
        assert_eq!(lookup(4, MlxUnpipelined), None);
        assert!(close(lookup(1, Candle), 0.09));
        assert!(close(lookup(3, Candle), 0.17));
        assert!(close(lookup(PROMPT_LOOKUP_RECOMMENDED_DEPTH, Candle), 0.21));
        assert!(close(lookup(7, Candle), 0.33));
        for plain in [MlxPipelined, MlxUnpipelined] {
            assert!(close(mtp(1, plain), 0.25));
            assert!(close(mtp(3, plain), 1.2));
        }
        assert!(close(mtp(1, Candle), 0.25));
        assert!(close(mtp(MTP_RECOMMENDED_DEPTH, Candle), 0.5));
        assert!(close(mtp(7, Candle), 1.0));
        for plain in [MlxPipelined, MlxUnpipelined, Candle] {
            assert_eq!(mtp(0, plain), None);
            assert_eq!(lookup(0, plain), None);
            assert_eq!(demotion_threshold(ProposerKind::DraftModel, 4, plain), None);
            assert_eq!(demotion_threshold(ProposerKind::None, 4, plain), None);
        }
        // The measured families land on the documented sides. MLX: a companion head's open-ended
        // mal (0.66–0.85) demotes at depth 3, a native head's (1.38+) does not.
        let mlx3 = mtp(3, MlxPipelined).unwrap();
        assert!(0.85 < mlx3 && mlx3 < 1.38);
        // Candle: Qwen3.8's native head (mal 1.24+ at depth 3, +41..+127 %) is far above;
        // Candle lookup's measured open-ended losers (Bonsai, mal 0.15–0.20 at depth 4) demote,
        // and no measured Candle lookup winner (Qwen3.8 at depth 3, mal 0.40+) does.
        assert!(mtp(3, Candle).unwrap() < 1.24);
        let candle4 = lookup(4, Candle).unwrap();
        assert!(0.20 < candle4 && candle4 < 0.40);
        assert!(lookup(3, Candle).unwrap() < 0.40);
        // Each Candle threshold is at or below its depth's cheapest measured cost minus one.
        for (depth, floor) in [(1, 0.099), (3, 0.197), (7, 0.349)] {
            assert!(
                lookup(depth, Candle).unwrap() <= floor,
                "lookup depth {depth}"
            );
        }
        for (depth, floor) in [(1, 0.27), (3, 0.50), (7, 1.12)] {
            assert!(
                mtp(depth, Candle).unwrap() <= floor + 1e-12,
                "mtp depth {depth}"
            );
        }
    }

    /// The monitor judges rolling windows of [`ACCEPTANCE_PROBE_VERIFIES`] verify steps: the first
    /// window below the threshold demotes (on exactly its last step, once); a window at or above
    /// it keeps the proposer for the next window; after the demotion nothing fires again.
    #[test]
    fn the_monitor_demotes_at_the_first_window_below_the_threshold() {
        let window = ACCEPTANCE_PROBE_VERIFIES as usize;
        // Below: 0 accepted every step demotes on the first window's last step, never before or
        // after.
        let mut low = AcceptanceMonitor::with_threshold(0.4).unwrap();
        let fired: Vec<bool> = (0..3 * window).map(|_| low.observe(0)).collect();
        assert_eq!(fired.iter().position(|&f| f), Some(window - 1));
        assert_eq!(fired.iter().filter(|&&f| f).count(), 1);
        // Just below the break-even (6 / 16 = 0.375 < 0.4) still demotes.
        let mut under = AcceptanceMonitor::with_threshold(0.4).unwrap();
        let fired = (0..window).map(|i| under.observe(usize::from(i < 6)));
        assert!(fired.last().unwrap());
        // At it (0.4 × 16 = 6.4 → 7 accepted) a window keeps the proposer.
        let mut at = AcceptanceMonitor::with_threshold(0.4).unwrap();
        assert!(!(0..4 * window).any(|i| at.observe(usize::from(i % window < 7))));
        // Exactly at it (0.25 × 16 = 4 accepted, break-even: not losing) a window keeps it too.
        let mut even = AcceptanceMonitor::with_threshold(0.25).unwrap();
        assert!(!(0..4 * window).any(|i| even.observe(usize::from(i % window < 4))));
        // Consistently above it, never.
        let mut high = AcceptanceMonitor::with_threshold(1.2).unwrap();
        assert!(!(0..8 * window).any(|_| high.observe(3)));
        assert_eq!(AcceptanceMonitor::with_threshold(f64::NAN), None);
        assert_eq!(AcceptanceMonitor::with_threshold(-1.0), None);
    }

    /// A late loser: two windows well above the threshold, then acceptance collapses. The monitor
    /// demotes at the end of the first failing window — the third — not at the first window, and
    /// not later as a cumulative mean would (the request's mean is still 2.0 there, far above
    /// 1.2). The earlier windows' credit is not carried: a window is judged on its own steps.
    #[test]
    fn an_early_high_then_low_request_demotes_at_its_first_failing_window() {
        let window = ACCEPTANCE_PROBE_VERIFIES as usize;
        let mut late = AcceptanceMonitor::with_threshold(1.2).unwrap();
        let accepted = |i: usize| if i < 2 * window { 3 } else { 0 };
        let fired: Vec<bool> = (0..6 * window).map(|i| late.observe(accepted(i))).collect();
        assert_eq!(fired.iter().position(|&f| f), Some(3 * window - 1));
        assert_eq!(fired.iter().filter(|&&f| f).count(), 1);
        // A window straddling the collapse is judged on its own steps: 6 high steps (18 accepted
        // < 1.2 × 16 = 19.2) still demote, 7 (21) do not — and the next all-zero window does.
        for (high_steps, demoted_at) in [(6, 2 * window - 1), (7, 3 * window - 1)] {
            let mut straddle = AcceptanceMonitor::with_threshold(1.2).unwrap();
            let accepted = |i: usize| if i < window + high_steps { 3 } else { 0 };
            let fired: Vec<bool> = (0..4 * window)
                .map(|i| straddle.observe(accepted(i)))
                .collect();
            assert_eq!(
                fired.iter().position(|&f| f),
                Some(demoted_at),
                "{high_steps} high steps"
            );
        }
        // Low early, high later: still demoted at the first window — irreversible.
        let mut early = AcceptanceMonitor::with_threshold(1.2).unwrap();
        let fired: Vec<bool> = (0..4 * window)
            .map(|i| early.observe(if i < window { 0 } else { 3 }))
            .collect();
        assert_eq!(fired.iter().position(|&f| f), Some(window - 1));
        assert_eq!(fired.iter().filter(|&&f| f).count(), 1);
    }

    /// Only `auto` is monitored: an explicit `{proposer, depth}` is the caller's choice and runs
    /// as asked; `off` and a proposer without a measured cost are never monitored.
    #[test]
    fn only_auto_is_monitored() {
        use crate::{Speculative, SpeculativeProposer};
        use PlainDecode::{Candle, MlxPipelined, MlxUnpipelined};
        let monitor = |mode, proposer, depth, plain| {
            AcceptanceMonitor::for_request(mode, proposer, depth, plain)
        };
        let auto = monitor(Speculative::Auto, ProposerKind::Mtp, 3, MlxPipelined).unwrap();
        assert!((auto.threshold() - 1.2).abs() < 1e-12);
        let candle = monitor(Speculative::Auto, ProposerKind::PromptLookup, 4, Candle).unwrap();
        assert!((candle.threshold() - 0.21).abs() < 1e-12);
        assert!(monitor(
            Speculative::Auto,
            ProposerKind::PromptLookup,
            4,
            MlxPipelined
        )
        .is_some());
        assert_eq!(
            monitor(
                Speculative::Auto,
                ProposerKind::PromptLookup,
                4,
                MlxUnpipelined
            ),
            None,
            "lookup is not demoted where nothing measured a regression"
        );
        for plain in [MlxPipelined, MlxUnpipelined, Candle] {
            for explicit in [
                Speculative::proposer(SpeculativeProposer::PromptLookup, 4),
                Speculative::proposer(SpeculativeProposer::Mtp, 3),
                Speculative::Off,
            ] {
                for (kind, depth) in [(ProposerKind::PromptLookup, 4), (ProposerKind::Mtp, 3)] {
                    assert_eq!(
                        monitor(explicit, kind, depth, plain),
                        None,
                        "{explicit:?} {kind:?} {plain:?}"
                    );
                }
            }
            assert_eq!(
                monitor(Speculative::Auto, ProposerKind::None, 0, plain),
                None
            );
            assert_eq!(
                monitor(Speculative::Auto, ProposerKind::DraftModel, 4, plain),
                None
            );
        }
    }

    #[test]
    fn greedy_commit_is_the_accepted_prefix_plus_the_bonus() {
        assert_eq!(greedy_commit(&[1, 2, 9], &[1, 2]), (vec![1, 2, 9], 2));
        assert_eq!(greedy_commit(&[1, 5, 9], &[1, 2]), (vec![1, 5], 1));
        assert_eq!(greedy_commit(&[7, 5, 9], &[1, 2]), (vec![7], 0));
        assert_eq!(greedy_commit(&[9], &[]), (vec![9], 0));
    }

    // --- greedy acceptance ---

    #[test]
    fn greedy_accepts_matching_prefix() {
        // drafts d, target argmax (with bonus slot).
        assert_eq!(accept_greedy_run(&[1, 2, 9], &[1, 2]), 2); // both match
        assert_eq!(accept_greedy_run(&[1, 5, 9], &[1, 2]), 1); // 2nd diverges
        assert_eq!(accept_greedy_run(&[7, 5, 9], &[1, 2]), 0); // 1st diverges
        assert_eq!(accept_greedy_run(&[9], &[]), 0); // no drafts -> bonus only
    }

    // --- stochastic acceptance: basic behaviour ---

    #[test]
    fn accept_when_target_dominates() {
        // p strongly favours the proposed token -> accept_prob ~1.
        let p = [(0, 0.9f32), (1, 0.1)];
        let q = [(0, 1.0f32)]; // point mass at 0 (prompt-lookup style)
        assert_eq!(accept_token(&p, &q, 0, 0.5, 0.0), Acceptance::Accepted(0));
    }

    #[test]
    fn reject_resamples_from_residual_excluding_proposed_for_point_mass() {
        // Point-mass draft at 0; on rejection the residual is p with 0 removed -> must yield 1 or 2.
        let p = [(0, 0.2f32), (1, 0.5), (2, 0.3)];
        let q = [(0, 1.0f32)];
        // Force rejection with u_accept above accept_prob (=p(0)=0.2).
        match accept_token(&p, &q, 0, 0.99, 0.1) {
            Acceptance::Rejected(t) => assert!(t == 1 || t == 2, "got {t}"),
            other => panic!("expected rejection, got {other:?}"),
        }
    }

    /// A tiny deterministic PRNG for the Monte-Carlo test (xorshift64*).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / ((1u64 << 24) as f32)
        }
    }

    #[test]
    fn monte_carlo_output_matches_target_distribution() {
        // The headline guarantee: drawing the proposal from q and running accept_token yields the
        // committed token distributed as p, for arbitrary p, q.
        let p = [(0, 0.10f32), (1, 0.35), (2, 0.05), (3, 0.30), (4, 0.20)];
        let q = [(0, 0.30f32), (1, 0.10), (2, 0.25), (3, 0.20), (4, 0.15)];
        let q_total: f32 = q.iter().map(|&(_, w)| w).sum();

        let mut rng = Rng(0x1234_5678_9abc_def0);
        let n = 400_000;
        let mut counts = [0u64; 5];
        for _ in 0..n {
            let proposed = sample_weighted(&q, rng.next(), 0);
            let committed = accept_token(&p, &q, proposed, rng.next(), rng.next()).token();
            counts[committed as usize] += 1;
        }
        // q is a proper distribution (sums to 1) so the draw is unbiased.
        assert!((q_total - 1.0).abs() < 1e-6);
        for (i, &(_, pi)) in p.iter().enumerate() {
            let emp = counts[i] as f32 / n as f32;
            assert!(
                (emp - pi).abs() < 0.01,
                "token {i}: empirical {emp} vs target {pi}"
            );
        }
    }

    #[test]
    fn greedy_is_the_point_mass_special_case() {
        // Target point mass at argmax=3; proposing 3 accepts, proposing anything else rejects to 3.
        let p = [(3, 1.0f32)];
        let q = [(3, 1.0f32)];
        assert_eq!(accept_token(&p, &q, 3, 0.999, 0.5), Acceptance::Accepted(3));
        let p2 = [(3, 1.0f32)];
        let q2 = [(1, 1.0f32)];
        assert_eq!(
            accept_token(&p2, &q2, 1, 0.999, 0.5),
            Acceptance::Rejected(3)
        );
    }
}
