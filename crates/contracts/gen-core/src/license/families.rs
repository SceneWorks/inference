//! The licence **family** table (sc-16662, extended by sc-16665) — nineteen upstream licence texts,
//! transcribed.
//!
//! # PROVISIONAL — gathered by an agent, not yet signed off by a human
//!
//! Every value here was transcribed from a primary-source licence text retrieved on **2026-08-02**
//! by an automated agent, and each one is backed by a verbatim quote recorded in
//! `docs/licensing/sc-16662-licence-family-evidence.md` (the sixteen sc-16662 families) or
//! `docs/licensing/sc-16665-checkpoint-licence-evidence.md` (the three sc-16665 added). Those notes
//! are the sign-off documents: they carry the quote, the source URL and the retrieval date behind
//! every term below, plus the open items a human still has to decide. **No human has yet checked the
//! quotes.** Treat these values as provisional until the notes' sign-off tables are ticked.
//!
//! # Disclosure only
//!
//! Nothing here blocks, gates, degrades or withholds anything, and nothing added here ever should.
//! Each term records what a licence text **names**. Whether a given use is permitted is the
//! consumer's evaluation of these facts against its own situation — its revenue, whether it
//! redistributes weights, which agreements it has with its own users — and this crate has none of
//! that information. The doc comments below therefore report what a text states and stop there.
//!
//! # Silence is recorded as silence
//!
//! Where a text says nothing, the honest transcription is nothing. Four families restrict use of
//! the *weights* to non-commercial purposes and say nothing at all about **outputs**
//! (`insightface-research-only`, `nvidia-nsclv1`, `apple-mlr`, `cc-by-nc-4-0`), so none of them
//! carries [`LicenseTerm::NonCommercialOutputs`] — inferring one from a use restriction would be a
//! legal reading, and there is no quote for it. Contrast FLUX.1 \[dev\] §2(d) and CircleStone
//! §2(e), which address outputs explicitly and in the opposite direction; those are quoted, and
//! still not transcribed as a term, because the vocabulary has no "outputs are unrestricted"
//! variant and does not need one.
//!
//! # Scope
//!
//! Families only. The component rows that point at them live beside this module in
//! [`super::components`] (sc-16665); the provider→component mappings
//! ([`super::ProviderComponents`]) are per-backend and belong to the two media catalogs (sc-16666,
//! sc-16667). A family being listed here is a statement that its text was read, **not** that a
//! shipped checkpoint declares it.

use super::{CeilingBoundary, LicenseFamily, LicenseTerm};

/// Apache License 2.0.
///
/// Text read at <https://www.apache.org/licenses/LICENSE-2.0.txt> on 2026-08-02.
pub const APACHE_2_0: LicenseFamily = LicenseFamily {
    id: "apache-2-0",
    spdx_id: "Apache-2.0",
    name: "Apache License 2.0",
    text_url: "https://www.apache.org/licenses/LICENSE-2.0.txt",
    terms: &[
        // §4(c) "You must retain, in the Source form of any Derivative Works that You distribute,
        // all copyright, patent, trademark, and attribution notices from the Source form of the
        // Work".
        LicenseTerm::AttributionRequired,
        // §4(d) "any Derivative Works that You distribute must include a readable copy of the
        // attribution notices contained within such NOTICE file".
        LicenseTerm::NoticeFileRequired,
        // §4(a) "You must give any other recipients of the Work or Derivative Works a copy of this
        // License". Recorded as the copy-shaped flow-down and not the restrictions-shaped one:
        // Apache-2.0 states no use restrictions for a downstream agreement to reproduce.
        LicenseTerm::DownstreamLicenseCopy {
            family: "apache-2-0",
        },
    ],
};

/// `MiniMaxAI/MiniMax-H3` — the **MiniMax H3 Community License Agreement**, dated 2 August 2026.
///
/// # The term this vocabulary cannot express: the licence is TERRITORIALLY EXCLUSIVE
///
/// §I.3 defines the "Applicable Territory" as **worldwide excluding** the "Excluded Territories",
/// and §I.5 names those as **the European Union, the United Kingdom, the Republic of Korea and the
/// United States of America**. §V.4: *"You may not use, reproduce, modify, distribute, or display
/// the MiniMax H3 Works or any of their Outputs or results outside the Applicable Territory."*
/// Exhibit A item 1 repeats it as a use restriction. §II offers a bespoke licence to anyone in an
/// Excluded Territory who contacts MiniMax.
///
/// [`LicenseTerm`] has **no territorial-restriction variant**, so the clause is carried below as a
/// [`LicenseTerm::DeployerObligation`] quoting §V.4 verbatim rather than being dropped — a term this
/// consequential must appear in a derived term union even if it appears under an imprecise kind. A
/// first-class variant is tracked separately; until it exists, read this doc comment.
///
/// # The encoder is a different licence
///
/// The LICENSE's own closing note: *"the encoder of MiniMax H3 uses Qwen3-VL-32B, which is licensed
/// under Apache 2.0 License"*. sc-17143 established that the shipped `text_encoder/` shards are
/// byte-identical to `Qwen/Qwen3-VL-32B-Instruct`, so that is a **separate component**
/// ([`QWEN3_VL_32B_INSTRUCT`](super::components::QWEN3_VL_32B_INSTRUCT)) and not covered by
/// this family.
///
/// Text read at <https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE> on 2026-08-12.
pub const MINIMAX_H3_COMMUNITY: LicenseFamily = LicenseFamily {
    id: "minimax-h3-community",
    spdx_id: "LicenseRef-MiniMax-H3-Community",
    name: "MiniMax H3 Community License Agreement",
    text_url: "https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE",
    terms: &[
        // §IV.1 "if your commercial products and services generate MORE THAN 20 million US dollars
        // (or equivalent in other currencies) in yearly revenue" — "more than", so the amount itself
        // is below the threshold and the boundary is Exclusive (LTX's "at least" is Inclusive).
        LicenseTerm::RevenueCeiling {
            amount_usd: 20_000_000,
            boundary: CeilingBoundary::Exclusive,
        },
        // §IV.1 names the address and the subject line, unlike LTX's contactless clause.
        LicenseTerm::RegistrationRequired {
            contact: Some("api@minimax.io"),
        },
        // §V.2 "you must bind each recipient or user to enforceable terms at least as protective as
        // the use restrictions in this Section V and Exhibit A".
        LicenseTerm::DownstreamRestrictions {
            family: "minimax-h3-community",
        },
        // §III.1 "You must provide a copy of this Agreement to all such Third Parties".
        LicenseTerm::DownstreamLicenseCopy {
            family: "minimax-h3-community",
        },
        // §IV.2 "You shall prominently display 'MiniMax H3' on the user interface of commercial
        // product or service that uses MiniMax H3". Mandatory, unlike §III.3's encouraged
        // "Powered by MiniMax H3" notice.
        LicenseTerm::AttributionRequired,
        // §III.4 "must be accompanied by a 'NOTICE' text file containing the following notice".
        LicenseTerm::NoticeFileRequired,
        // §V.4, the territorial restriction. See the type docs: this is the wrong KIND for the
        // clause, and it is recorded here anyway so it cannot vanish from a derived union.
        LicenseTerm::DeployerObligation {
            text: "You may not use, reproduce, modify, distribute, or display the MiniMax H3 \
                   Works or any of their Outputs or results outside the Applicable Territory \
                   (worldwide EXCLUDING the European Union, the United Kingdom, the Republic of \
                   Korea and the United States of America)",
        },
        // §V.5, the safeguards obligation on anyone offering generation to a third party.
        LicenseTerm::DeployerObligation {
            text: "you must implement, maintain, test, and periodically review reasonable and \
                   proportionate technical and organizational safeguards designed to prevent and \
                   mitigate access, uses, and Outputs that violate this Section V or Exhibit A",
        },
        // Exhibit A is inside the LICENSE itself, so there is no separate URL to cite.
        LicenseTerm::AcceptableUsePolicy { url: None },
    ],
};

/// MIT License.
///
/// Text read at <https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt> on
/// 2026-08-02.
pub const MIT: LicenseFamily = LicenseFamily {
    id: "mit",
    spdx_id: "MIT",
    name: "MIT License",
    text_url: "https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt",
    // "The above copyright notice and this permission notice shall be included in all copies or
    // substantial portions of the Software." The sole condition the text states.
    terms: &[LicenseTerm::AttributionRequired],
};

/// Creative Commons Attribution-NonCommercial 4.0 International.
///
/// Text read at <https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt> on 2026-08-02.
pub const CC_BY_NC_4_0: LicenseFamily = LicenseFamily {
    id: "cc-by-nc-4-0",
    spdx_id: "CC-BY-NC-4.0",
    name: "Creative Commons Attribution-NonCommercial 4.0 International",
    text_url: "https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt",
    terms: &[
        // §3(a)(1) "If You Share the Licensed Material (including in modified form), You must: …
        // retain … identification of the creator(s) of the Licensed Material …; a copyright notice;
        // a notice that refers to this Public License".
        LicenseTerm::AttributionRequired,
        // §2(a)(1), the operative grant, bounded on its face: the Licensor "grants You a …
        // license to exercise the Licensed Rights in the Licensed Material to: a. reproduce and
        // Share the Licensed Material, in whole or in part, for NonCommercial purposes only; and
        // b. produce, reproduce, and Share Adapted Material for NonCommercial purposes only."
        //
        // §1(i) is only the *definition* — "NonCommercial means not primarily intended for or
        // directed towards commercial advantage or monetary compensation" — and states no
        // restriction of its own, so the restriction is quoted from §2(a)(1) instead. Both are
        // scoped to the Licensed Material; the text says nothing about material generated by
        // running a model, so no outputs term — see the module note on silence.
        LicenseTerm::NonCommercialWeights,
    ],
};

/// CreativeML Open RAIL++-M License, dated July 26, 2023.
///
/// Text read at
/// <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/LICENSE.md> on
/// 2026-08-02.
pub const CREATIVEML_OPENRAIL_PP_M: LicenseFamily = LicenseFamily {
    id: "creativeml-openrail-pp-m",
    spdx_id: "LicenseRef-CreativeML-OpenRAIL-PP-M",
    name: "CreativeML Open RAIL++-M License",
    text_url: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/LICENSE.md",
    terms: &[
        // Section III "Use-based restrictions as referenced in paragraph 5 MUST be included as an
        // enforceable provision by You in any type of legal agreement … and You shall give notice
        // to subsequent users You Distribute to, that the Model or Derivatives of the Model are
        // subject to paragraph 5."
        LicenseTerm::DownstreamRestrictions {
            family: "creativeml-openrail-pp-m",
        },
        // Section III "You must retain all copyright, patent, trademark, and attribution notices
        // excluding those notices that do not pertain to any part of the Model".
        LicenseTerm::AttributionRequired,
        // Section III "You must cause any modified files to carry prominent notices stating that
        // You changed the files".
        LicenseTerm::NoticeFileRequired,
        // Section III "The restrictions set forth in Attachment A are considered Use-based
        // restrictions." The restrictions are enumerated inside the licence and the text contains
        // no URL of any kind, so there is no address to record.
        LicenseTerm::AcceptableUsePolicy { url: None },
    ],
};

/// Stability AI Community License Agreement, Last Updated July 5, 2024.
///
/// Text read at <https://huggingface.co/stabilityai/stable-diffusion-3.5-large/raw/main/LICENSE.md>
/// on 2026-08-02 under an authenticated Hugging Face session, at the revision that was then `main`.
///
/// One text, several declared identifier strings: `stabilityai-ai-community`,
/// `stable-video-diffusion-community` and `stable-audio-community` are the same agreement, verified
/// by comparing the gated SD3.5 file against the ungated
/// `stable-video-diffusion-img2vid-xt` copy — the differences are typographic and the
/// revenue-threshold sentence is byte-identical. The declared strings belong on the component rows,
/// not in separate families.
pub const STABILITY_AI_COMMUNITY: LicenseFamily = LicenseFamily {
    id: "stability-ai-community",
    spdx_id: "LicenseRef-Stability-AI-Community",
    name: "Stability AI Community License Agreement",
    text_url: "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/raw/main/LICENSE.md",
    terms: &[
        // §III "If at any time You or Your Affiliate(s), either individually or in aggregate,
        // generate more than USD $1,000,000 in annual revenue …, any licenses granted to You under
        // this Agreement shall terminate as of such date." "more than" — the amount itself sits
        // below the threshold the text names, so the boundary is Exclusive. The string "at least"
        // occurs nowhere in the file; cf. LTX_2_COMMUNITY, which reads the other way.
        LicenseTerm::RevenueCeiling {
            amount_usd: 1_000_000,
            boundary: CeilingBoundary::Exclusive,
        },
        // §III "If You are using or distributing the Stability AI Materials for a Commercial
        // Purpose, You must register with Stability AI at (https://stability.ai/community-license)."
        LicenseTerm::RegistrationRequired {
            contact: Some("https://stability.ai/community-license"),
        },
        // §IV(a) "prominently display \"Powered by Stability AI\" on a related website, user
        // interface, blogpost, about page, or product documentation".
        LicenseTerm::AttributionRequired,
        // §IV(a) "retain the following attribution notice within a \"Notice\" text file distributed
        // as a part of such copies".
        LicenseTerm::NoticeFileRequired,
        // §IV(a) "If You distribute or make available the Stability AI Materials or a Derivative
        // Work to a third party, … You shall: (i) provide a copy of this Agreement to that third
        // party".
        LicenseTerm::DownstreamLicenseCopy {
            family: "stability-ai-community",
        },
        // §V "\"AUP\" means the Stability AI Acceptable Use Policy available at
        // (https://stability.ai/use-policy), as may be updated from time to time."
        LicenseTerm::AcceptableUsePolicy {
            url: Some("https://stability.ai/use-policy"),
        },
    ],
};

/// FLUX.1 \[dev\] Non-Commercial License v1.1.1.
///
/// Text read at
/// <https://raw.githubusercontent.com/black-forest-labs/flux/main/model_licenses/LICENSE-FLUX1-dev>
/// on 2026-08-02, and confirmed identical to the gated Hugging Face copy read the same day under an
/// authenticated session.
pub const FLUX_1_DEV_NON_COMMERCIAL: LicenseFamily = LicenseFamily {
    id: "flux-1-dev-non-commercial",
    spdx_id: "LicenseRef-FLUX-1-dev-Non-Commercial",
    name: "FLUX.1 [dev] Non-Commercial License v1.1.1",
    text_url:
        "https://raw.githubusercontent.com/black-forest-labs/flux/main/model_licenses/LICENSE-FLUX1-dev",
    terms: &[
        // §2(b) "You may only access, use, Distribute, or create Derivatives of the FLUX.1 [dev]
        // Model or Derivatives for Non-Commercial Purposes."
        //
        // §2(d) addresses outputs in the opposite direction — "You may use Output for any purpose
        // (including for commercial purposes), except as expressly prohibited herein" — so
        // NonCommercialOutputs is absent, and would be wrong rather than merely unquoted.
        LicenseTerm::NonCommercialWeights,
        // §3(a) "you must make available a copy of this License to third-party recipients of the
        // FLUX.1 [dev] Models and/or Derivatives you Distribute, and specify that any rights … shall
        // be directly granted by Company to said third-party recipients pursuant to this License".
        LicenseTerm::DownstreamLicenseCopy {
            family: "flux-1-dev-non-commercial",
        },
        // §2(e), quoted.
        LicenseTerm::DeployerObligation {
            text: "implement and maintain content filtering measures (\"Content Filters\") for your \
                   use of the FLUX.1 [dev] Model or Derivatives to prevent the creation, display, \
                   transmission, generation, or dissemination of unlawful or infringing content",
        },
        // Prohibited uses are enumerated inline in §4; the phrases "acceptable use", "use policy"
        // and "AUP" do not occur anywhere in the text. The model card's gate prompt cites a
        // `POLICY.md` in the same repository, and that file is not published — confirmed absent
        // from the repository's file listing under an authenticated read on 2026-08-02. There is no
        // address to record. Notably `https://blackforestlabs.ai/aup` is NOT it: that URL 404s and
        // appears nowhere in the licence.
        LicenseTerm::AcceptableUsePolicy { url: None },
    ],
};

/// FLUX Non-Commercial License v2.1 — the text Black Forest Labs ships with its FLUX.2 checkpoints
/// (sc-16665).
///
/// Text read at <https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md> on
/// 2026-08-02 under an authenticated session (the repository is gated), at `sha`
/// `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`. `black-forest-labs/FLUX.2-klein-9B`'s copy is
/// **byte-identical** to it; `FLUX.2-klein-9b-kv`'s is a different blob whose first line is the same
/// title.
///
/// # Why this is not [`FLUX_1_DEV_NON_COMMERCIAL`]
///
/// A different document with a different title, a different `Models` definition, and two obligations
/// the v1.1.1 text does not impose in the same shape — the self-named *Attribution Notice* of §3(b)
/// and the AI-generation disclosure duty of §2(e). Stretching the FLUX.1 transcription over FLUX.2
/// would have mis-stated both.
///
/// A **third** BFL text is in circulation and is deliberately not merged here: `FLUX [dev]
/// Non-Commercial License v2.0`, shipped beside `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`, whose
/// own `Models` definition enumerates FLUX.2. Whether v2.0 and v2.1 are one family or two is open
/// item **X9** in `docs/licensing/sc-16665-checkpoint-licence-evidence.md`, so that ControlNet has
/// **no component row** rather than a guessed one.
///
/// # U11 linkage
///
/// §3(b) is one clause and it is transcribed as **two** terms, matching [`NVIDIA_OPEN_MODEL`] rather
/// than [`GEMMA_TERMS`], because the clause names itself an *Attribution* Notice — which is exactly
/// the textual hook sc-16662's open item **U11** proposes as the distinguisher. If U11 is settled
/// the other way, this family and [`IDEOGRAM_4_NON_COMMERCIAL`] drop
/// [`LicenseTerm::AttributionRequired`] with the two landed families that share the shape.
pub const FLUX_NON_COMMERCIAL_V2_1: LicenseFamily = LicenseFamily {
    id: "flux-non-commercial-v2-1",
    spdx_id: "LicenseRef-FLUX-Non-Commercial-v2.1",
    name: "FLUX Non-Commercial License v2.1",
    text_url: "https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.md",
    terms: &[
        // §2(b) "You may only access, use, Distribute, or create Derivatives of the FLUX Model or
        // Derivatives for Non-Commercial Purposes."
        //
        // §2(d) addresses outputs in the opposite direction — "You may use Output for any purpose
        // (including for commercial purposes), except as expressly prohibited herein" — so
        // NonCommercialOutputs is absent, and would be wrong rather than merely unquoted.
        LicenseTerm::NonCommercialWeights,
        // §3(a) "you must make available a copy of this License to third-party recipients of the
        // FLUX Mode and/or Derivatives you Distribute, and specify that any rights to use the FLUX
        // Model and/or Derivatives shall be directly granted by Company to said third-party
        // recipients pursuant to this License". (The "Mode" typo is the source's.)
        LicenseTerm::DownstreamLicenseCopy {
            family: "flux-non-commercial-v2-1",
        },
        // §3(b) "you must prominently display the following notice alongside the Distribution of the
        // FLUX Model or Derivative (such as via a \"Notice\" text file distributed as part of such
        // FLUX Model or Derivative) (the \"Attribution Notice\")".
        LicenseTerm::NoticeFileRequired,
        // Same §3(b) clause, which names itself the "Attribution Notice" — see the U11 note above.
        LicenseTerm::AttributionRequired,
        // §2(e), first of two duties in one sentence.
        LicenseTerm::DeployerObligation {
            text:
                "implement and maintain content filtering measures (\"Content Filters\") for your \
                   use of the FLUX Model or Derivatives to prevent the creation, display, \
                   transmission, generation, or dissemination of unlawful or infringing content",
        },
        // §2(e), second duty — new relative to the v1.1.1 text.
        LicenseTerm::DeployerObligation {
            text: "ensure Output includes disclosure (or other indication) that the Output was \
                   generated or modified using artificial intelligence technologies to the extent \
                   required under applicable law.",
        },
        // The licence text enumerates prohibited uses inline in §4 and names no policy; the address
        // is in the model card's `extra_gated_prompt`, verbatim: "you agree to the [FLUX
        // Non-Commercial License Agreement](…) and acknowledge the [Acceptable Use
        // Policy](https://bfl.ai/legal/usage-policy)". Fetched 2026-08-02: HTTP 200. Same evidence
        // shape sc-16662 accepted for `krea-2-community`.
        //
        // This address is for FLUX.2 ONLY. FLUX.1 [dev]'s gate prompt still cites a `POLICY.md`
        // that is not published — re-verified 2026-08-02 — so [`FLUX_1_DEV_NON_COMMERCIAL`] keeps
        // `url: None` and the live URL must never be back-ported onto it.
        LicenseTerm::AcceptableUsePolicy {
            url: Some("https://bfl.ai/legal/usage-policy"),
        },
    ],
};

/// Krea 2 Community License Agreement v.1, dated June 22, 2026.
///
/// Text read on 2026-08-02 from the PDF the model card's `license_link` names, at a content-pinned
/// CDN revision. The in-repository copy is behind an access list this project is not on.
///
/// [`LicenseTerm::AttributionRequired`] here rests on §3.1(b)'s model-name-prefix duty, which is
/// attribution-shaped but not a notice string. It is quoted, and it is the closest variant the
/// vocabulary carries; whether it is the right one is the same class of judgement as the
/// NVIDIA/Gemma notice clauses and is held open with them as **U11** — see [`NVIDIA_OPEN_MODEL`] and
/// `docs/licensing/sc-16662-licence-family-evidence.md`.
pub const KREA_2_COMMUNITY: LicenseFamily = LicenseFamily {
    id: "krea-2-community",
    spdx_id: "LicenseRef-Krea-2-Community",
    name: "Krea 2 Community License Agreement v.1",
    text_url:
        "https://cdn.jsdelivr.net/gh/krea-ai/krea-2@db3984fbc6e13b34c0064990fc2d95ac64d00058/assets/hf_samples/LICENSE.pdf",
    terms: &[
        // §2.3 names the threshold from below — "permitted only if you … have total company-wide
        // annual revenue of less than one million United States dollars ($1,000,000 USD)" — and
        // then from above: "If you meet or exceed this threshold, you must obtain a separate
        // enterprise license". "meet or exceed" puts the amount itself AT the threshold, so the
        // boundary is Inclusive. Same amount as STABILITY_AI_COMMUNITY, opposite reading of it.
        LicenseTerm::RevenueCeiling {
            amount_usd: 1_000_000,
            boundary: CeilingBoundary::Inclusive,
        },
        // §2.3 "Enterprise license inquiries may be directed to opensource@krea.ai."
        LicenseTerm::RegistrationRequired {
            contact: Some("opensource@krea.ai"),
        },
        // §3.1 "you shall (a) provide a copy of this Agreement and require each recipient to be
        // bound by the Terms of this Agreement".
        LicenseTerm::DownstreamLicenseCopy {
            family: "krea-2-community",
        },
        // §3.1(b) "include \"Krea\" at the beginning of any such AI model name" — a model-naming
        // duty, recorded as the closest variant the vocabulary carries. Whether that is the right
        // variant is the same class of judgement as the NVIDIA/Gemma notice clauses and is held
        // open with them as U11; see the doc comment above.
        LicenseTerm::AttributionRequired,
        // §3.1(c) "retain the following attribution notice within a \"Notice\" text file
        // distributed as part of such copies".
        LicenseTerm::NoticeFileRequired,
        // §4.2, quoted.
        LicenseTerm::DeployerObligation {
            text: "You must implement reasonable and appropriate Content Filter measures to detect, \
                   prevent, and mitigate the generation or distribution of prohibited, harmful, or \
                   unlawful content through your deployment of the Krea Model or any Derivative.",
        },
        // §4.3, quoted.
        LicenseTerm::DeployerObligation {
            text: "Where required by applicable law, regulation, or platform policy, you must \
                   clearly disclose that Outputs were generated using artificial intelligence.",
        },
        // §4.4 "You must comply with the Acceptable Use Policy, which is incorporated herein by
        // reference." The address is not in the PDF; it is in the model card's gate prompt, and it
        // resolves to a page titled "Krea Acceptable Use Policy", dated the same day as the licence
        // and scoped to these weights.
        LicenseTerm::AcceptableUsePolicy {
            url: Some("https://www.krea.ai/krea-2-use-policy"),
        },
    ],
};

/// LTX-2 Community License Agreement, License date January 5, 2026.
///
/// # Two upstream texts exist and they differ
///
/// `text_url` points at the copy committed **beside the weights** in `Lightricks/LTX-2.3`, which is
/// what a user who downloads the checkpoint actually receives. The licensor also publishes a
/// different `LICENSE` in its GitHub repository — the file the model card's `license_link` names —
/// and the two differ in two operative places, both read on 2026-08-02:
///
/// * the GitHub copy links an address at the registration sentence and the shipped copy does not,
///   which is why [`LicenseTerm::RegistrationRequired`] below carries `contact: None`; and
/// * the definition of "Control" is *"fifty percent (50%) or more"* in the shipped copy and
///   *"more than fifty percent (50%)"* in the GitHub copy, so at exactly 50% the two texts disagree
///   about which entities aggregate as Affiliates.
///
/// Which text governs is unresolved and recorded as such in the sign-off note. Everything else in
/// the two files agrees, so every other term below transcribes identically from either.
pub const LTX_2_COMMUNITY: LicenseFamily = LicenseFamily {
    id: "ltx-2-community",
    spdx_id: "LicenseRef-LTX-2-Community",
    name: "LTX-2 Community License Agreement",
    text_url: "https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE",
    terms: &[
        // §2 "Entities with annual revenues of at least $10,000,000 (the \"Commercial Entities\")
        // are required to obtain a paid commercial use license". "at least" — the amount itself is
        // AT the threshold, so the boundary is Inclusive. Identical in both upstream copies.
        LicenseTerm::RevenueCeiling {
            amount_usd: 10_000_000,
            boundary: CeilingBoundary::Inclusive,
        },
        // §2, shipped copy: "Commercial Entities interested in such a commercial license are
        // required to contact Licensor." — full stop, no address. The GitHub copy links
        // `https://ltx.io/model/licensing` at the same words; that address is not in the text
        // distributed with the weights, so it is not transcribed here.
        LicenseTerm::RegistrationRequired { contact: None },
        // §3(a) "Use-based restrictions as referenced in paragraph 4 and all provisions of
        // Attachment A MUST be included as an enforceable provision by you in any type of legal
        // agreement … governing the use and/or distribution of LTX-2".
        LicenseTerm::DownstreamRestrictions {
            family: "ltx-2-community",
        },
        // §3(b) "You must provide any third party recipients of LTX-2 or Derivatives of LTX-2 a
        // copy of this Agreement, including all attachments and use policies." This licence imposes
        // BOTH flow-down shapes; they are two duties and stay two elements of a union.
        LicenseTerm::DownstreamLicenseCopy {
            family: "ltx-2-community",
        },
        // §3(d) "You must retain all copyright, patent, trademark, and attribution notices
        // excluding those notices that do not pertain to any part of LTX-2".
        LicenseTerm::AttributionRequired,
        // §3(c) "You must cause any modified files to carry prominent notices stating that you
        // changed the files".
        LicenseTerm::NoticeFileRequired,
        // §3(b), quoted (the source quote elides the subject clause). The "…" is the source's own
        // elision and is carried into the string: this text is shown to a user, so joining across
        // it would present two fragments as one continuous sentence. A constraint neither
        // CreativeML Open RAIL++-M nor the Gemma Terms state, so it is carried alongside the
        // flow-down rather than folded into it.
        LicenseTerm::DeployerObligation {
            text:
                "Any Derivative of LTX-2 … must be distributed exclusively under the terms of this \
                   Agreement with a complete copy of this license included",
        },
        // Attachment A "When using the Outputs, LTX-2 and any Derivatives thereof, you will comply
        // with the Acceptable Use Policy. In addition, you agree not to use the Outputs …" — the
        // "In addition" reads the policy as a document separate from the enumerated list that
        // follows, yet the term is capitalised without ever being defined, the licence's only URL is
        // the licensor's GitHub repository, and that repository publishes no policy file. No
        // address exists to record.
        LicenseTerm::AcceptableUsePolicy { url: None },
    ],
};

/// CircleStone Labs Non-Commercial License v1.2.
///
/// Text read at <https://huggingface.co/circlestone-labs/Anima/raw/main/LICENSE.md> on 2026-08-02.
pub const CIRCLESTONE_LABS_NON_COMMERCIAL: LicenseFamily = LicenseFamily {
    id: "circlestone-labs-non-commercial",
    spdx_id: "LicenseRef-CircleStone-Labs-Non-Commercial",
    name: "CircleStone Labs Non-Commercial License v1.2",
    text_url: "https://huggingface.co/circlestone-labs/Anima/raw/main/LICENSE.md",
    terms: &[
        // §2(b) "You may only access, use, Distribute, or create Derivatives of the CircleStone
        // Model or Derivatives for Non-Commercial Purposes, unless otherwise expressly granted by
        // this License."
        //
        // §2(e) addresses outputs the other way — "You may use Outputs for any purpose (including
        // for commercial purposes), except as expressly prohibited herein" — so no outputs term.
        LicenseTerm::NonCommercialWeights,
        // §3(a) "you must make available a copy of this License to third-party recipients of the
        // CircleStone Models and/or Derivatives you Distribute, and specify that any rights … shall
        // be directly granted by Company to said third-party recipients pursuant to this License".
        LicenseTerm::DownstreamLicenseCopy {
            family: "circlestone-labs-non-commercial",
        },
        // §3 attribution string: "The CircleStone Model is licensed by CircleStone Labs LLC under
        // the CircleStone Non-Commercial License. Copyright CircleStone Labs LLC."
        LicenseTerm::AttributionRequired,
    ],
};

/// Gemma Terms of Use.
///
/// Text read at <https://ai.google.dev/gemma/terms> on 2026-08-02.
///
/// # Open item U11 — one clause, one term or two?
///
/// [`LicenseTerm::AttributionRequired`] is deliberately **not** transcribed here. §3.1's `"Notice"`
/// string is the only attribution-shaped obligation the text states, and recording one clause as
/// two terms is a modelling choice rather than a transcription. [`NVIDIA_OPEN_MODEL`] takes the
/// opposite course on its own §3, and [`KREA_2_COMMUNITY`] records a third shape again — the three
/// are held together, undecided, as open item **U11** in
/// `docs/licensing/sc-16662-licence-family-evidence.md`. The candidate distinguisher is textual:
/// NVIDIA's clause names "the following **attribution** notice", Gemma's names a `"Notice"` text
/// file and no attribution. Whether that difference is enough to justify the different treatment is
/// what U11 asks; nothing in this module argues it either way.
pub const GEMMA_TERMS: LicenseFamily = LicenseFamily {
    id: "gemma-terms",
    spdx_id: "LicenseRef-Gemma-Terms",
    name: "Gemma Terms of Use",
    text_url: "https://ai.google.dev/gemma/terms",
    terms: &[
        // §3.1 "You must include the use restrictions referenced in Section 3.2 as an enforceable
        // provision in any agreement … and you must provide notice to subsequent users you
        // Distribute to that Gemma or Model Derivatives are subject to the use restrictions in
        // Section 3.2."
        LicenseTerm::DownstreamRestrictions {
            family: "gemma-terms",
        },
        // §3.1 "You must provide all third party recipients of Gemma or Model Derivatives a copy of
        // this Agreement." Both shapes, from the same section.
        LicenseTerm::DownstreamLicenseCopy {
            family: "gemma-terms",
        },
        // §3.1 "All Distributions (other than through a Hosted Service) must be accompanied by a
        // \"Notice\" text file that contains the following notice: \"Gemma is provided under and
        // subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\"." Note the
        // Hosted-Service carve-out, which the flow-down obligations above do not share.
        LicenseTerm::NoticeFileRequired,
        // §3.2 "for the restricted uses set forth in the Gemma Prohibited Use Policy at
        // ai.google.dev/gemma/prohibited_use_policy (\"Prohibited Use Policy\"), which is hereby
        // incorporated by reference into this Agreement".
        LicenseTerm::AcceptableUsePolicy {
            url: Some("https://ai.google.dev/gemma/prohibited_use_policy"),
        },
        // §3.3 states "Google claims no rights in Outputs you generate using Gemma", so no
        // outputs term. AttributionRequired is deliberately absent: the §3.1 Notice string is the
        // only attribution-shaped obligation in the text, and whether that is also a standalone
        // attribution duty is a reading, not a transcription — see the sign-off note.
    ],
};

/// NVIDIA Open Model License Agreement.
///
/// Text read at
/// <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/> on
/// 2026-08-02.
///
/// A different licence from [`NVIDIA_NSCLV1`], which is what `nvidia/PiD` declares. The two are
/// separate texts with different terms and share no family id.
///
/// # Open item U11 — one clause, one term or two?
///
/// The §3 notice clause below is transcribed as **both** [`LicenseTerm::NoticeFileRequired`] and
/// [`LicenseTerm::AttributionRequired`]. That is the same modelling move deliberately withheld from
/// [`GEMMA_TERMS`], whose §3.1 notice clause carries `NoticeFileRequired` only, and it sits beside a
/// third shape in [`KREA_2_COMMUNITY`]. The candidate distinguisher is textual: NVIDIA's clause
/// names "the following **attribution** notice", Gemma's names a `"Notice"` text file and no
/// attribution. That reading is not asserted here — the three families are held together, undecided,
/// as open item **U11** in `docs/licensing/sc-16662-licence-family-evidence.md`, so that one
/// decision settles all three rather than one being answered silently by transcription order.
pub const NVIDIA_OPEN_MODEL: LicenseFamily = LicenseFamily {
    id: "nvidia-open-model",
    spdx_id: "LicenseRef-NVIDIA-Open-Model",
    name: "NVIDIA Open Model License Agreement",
    text_url: "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
    terms: &[
        // §2, quoted (the source quote elides a clause).
        LicenseTerm::DeployerObligation {
            text: "If You bypass, disable, reduce the efficacy of, or circumvent any technical \
                   limitation, safety guardrail or associated safety guardrail hyperparameter, \
                   encryption, security, digital rights management, or authentication mechanism … \
                   contained in the Model without a substantially similar Guardrail appropriate for \
                   your use case, your rights under this Agreement will automatically terminate.",
        },
        // §3 "If you distribute the Model, You must give any other recipients of the Model a copy
        // of this Agreement".
        LicenseTerm::DownstreamLicenseCopy {
            family: "nvidia-open-model",
        },
        // §3 "include the following attribution notice within a \"Notice\" text file with such
        // copies: \"Licensed by NVIDIA Corporation under the NVIDIA Open Model License\"".
        LicenseTerm::NoticeFileRequired,
        // Same §3 clause, which names "the following attribution notice" in its own words. Two
        // terms from one clause — the modelling move withheld from GEMMA_TERMS, held open as U11;
        // see the doc comment above.
        LicenseTerm::AttributionRequired,
        // §3 "AI Ethics. Use of the Models under the Agreement must be consistent with NVIDIA's
        // Trustworthy AI terms found at
        // https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/."
        LicenseTerm::AcceptableUsePolicy {
            url: Some("https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/"),
        },
    ],
};

/// NVIDIA License (NSCLv1) — the licence `nvidia/PiD` declares.
///
/// Text read at <https://huggingface.co/nvidia/PixelDiT-1300M-1024px/raw/main/LICENSE> on
/// 2026-08-02, which is the file the PiD model card links as its licence.
///
/// Distinct from [`NVIDIA_OPEN_MODEL`] in both text and terms.
pub const NVIDIA_NSCLV1: LicenseFamily = LicenseFamily {
    id: "nvidia-nsclv1",
    spdx_id: "LicenseRef-NVIDIA-NSCLv1",
    name: "NVIDIA License (NSCLv1)",
    text_url: "https://huggingface.co/nvidia/PixelDiT-1300M-1024px/raw/main/LICENSE",
    terms: &[
        // §3.3 "The Work and any derivative works thereof only may be used or intended for use
        // non-commercially. … As used herein, \"non-commercially\" means for research or evaluation
        // purposes only." The text says nothing about outputs — see the module note on silence.
        LicenseTerm::NonCommercialWeights,
        // §3.1 "You may reproduce or distribute the Work only if (a) you do so under this license,
        // (b) you include a complete copy of this license with your distribution".
        LicenseTerm::DownstreamLicenseCopy {
            family: "nvidia-nsclv1",
        },
        // §3.2 "Your Terms provide that the use limitation in Section 3.3 applies to your
        // derivative works" — the restriction must survive into the deployer's own terms, which is
        // the heavier shape, imposed here alongside the copy duty.
        LicenseTerm::DownstreamRestrictions {
            family: "nvidia-nsclv1",
        },
        // §3.1(c) "you retain without modification any copyright, patent, trademark, or attribution
        // notices that are present in the Work".
        LicenseTerm::AttributionRequired,
    ],
};

/// InsightFace non-commercial research use only.
///
/// The upstream publishes **no licence document for the models**; the only statement is prose in the
/// repository README, read at
/// <https://raw.githubusercontent.com/deepinsight/insightface/master/README.md> on 2026-08-02. The
/// project's MIT grant covers its *code* and explicitly does not reach the models. This is the
/// thinnest evidence in the table and the sign-off note flags it as such.
pub const INSIGHTFACE_RESEARCH_ONLY: LicenseFamily = LicenseFamily {
    id: "insightface-research-only",
    spdx_id: "LicenseRef-InsightFace-Research-Only",
    name: "InsightFace non-commercial research use only",
    text_url: "https://raw.githubusercontent.com/deepinsight/insightface/master/README.md",
    // README §License "The training data containing the annotation (and the models trained with
    // these data) are available for non-commercial research purposes only." The prose restricts the
    // models; it says nothing about generated images, so no outputs term — see the module note on
    // silence.
    terms: &[LicenseTerm::NonCommercialWeights],
};

/// The ChatGLM3-6B License.
///
/// Text read at <https://raw.githubusercontent.com/THUDM/ChatGLM3/main/MODEL_LICENSE> on
/// 2026-08-02. The Hugging Face card carries no licence metadata field; this `MODEL_LICENSE` file in
/// the upstream repository is the governing document.
pub const CHATGLM3_MODEL_LICENSE: LicenseFamily = LicenseFamily {
    id: "chatglm3-model-license",
    spdx_id: "LicenseRef-ChatGLM3-6B-Model-License",
    name: "The ChatGLM3-6B License",
    text_url: "https://raw.githubusercontent.com/THUDM/ChatGLM3/main/MODEL_LICENSE",
    terms: &[
        // §2 "Users who wish to use the models for commercial purposes must register [here]
        // (https://open.bigmodel.cn/mla/form)."
        LicenseTerm::RegistrationRequired {
            contact: Some("https://open.bigmodel.cn/mla/form"),
        },
        // §2 "The license notice shall be included in all copies or substantial portions of the
        // Software."
        LicenseTerm::AttributionRequired,
    ],
};

/// Apple Machine Learning Research Model License Agreement.
///
/// Text read at <https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384/raw/main/LICENSE> on
/// 2026-08-02.
pub const APPLE_MLR: LicenseFamily = LicenseFamily {
    id: "apple-mlr",
    spdx_id: "LicenseRef-Apple-MLR",
    name: "Apple Machine Learning Research Model License Agreement",
    text_url: "https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-384/raw/main/LICENSE",
    terms: &[
        // §1 "limited license, to use, copy, modify, distribute, and create Model Derivatives …
        // exclusively for Research Purposes. … \"Research Purposes\" does not include any commercial
        // exploitation, product development or use in any commercial product or service." The text
        // says nothing about outputs — see the module note on silence.
        LicenseTerm::NonCommercialWeights,
        // §2 "If you choose to redistribute Apple Machine Learning Research Model or its Model
        // Derivatives, you must provide a copy of this Agreement to such third party".
        LicenseTerm::DownstreamLicenseCopy {
            family: "apple-mlr",
        },
        // §2 "ensure that the following attribution notice be provided: \"Apple Machine Learning
        // Research Model is licensed under the Apple Machine Learning Research Model License
        // Agreement.\""
        LicenseTerm::AttributionRequired,
    ],
};

/// Llama 3.1 Community License Agreement.
///
/// Text read at
/// <https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE> on
/// 2026-08-02.
pub const LLAMA_3_1_COMMUNITY: LicenseFamily = LicenseFamily {
    id: "llama-3-1-community",
    spdx_id: "LicenseRef-Llama-3.1-Community",
    name: "Llama 3.1 Community License Agreement",
    text_url:
        "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE",
    terms: &[
        // §1(b)(i) "you shall (A) provide a copy of this Agreement with any such Llama Materials".
        LicenseTerm::DownstreamLicenseCopy {
            family: "llama-3-1-community",
        },
        // §1(b)(i) "prominently display \"Built with Llama\" on a related website, user interface,
        // blogpost, about page, or product documentation".
        LicenseTerm::AttributionRequired,
        // §1(b)(iii) "You must retain in all copies of the Llama Materials that you distribute the
        // following attribution notice within a \"Notice\" text file distributed as a part of such
        // copies".
        LicenseTerm::NoticeFileRequired,
        // §1(b)(iv) "adhere to the Acceptable Use Policy for the Llama Materials (available at
        // https://llama.meta.com/llama3_1/use-policy), which is hereby incorporated by reference
        // into this Agreement".
        LicenseTerm::AcceptableUsePolicy {
            url: Some("https://llama.meta.com/llama3_1/use-policy"),
        },
        // §2, quoted (the source quote elides the subject clause). The threshold is denominated in
        // monthly active users, not revenue, so RevenueCeiling would be a false transcription and
        // the condition is disclosed verbatim instead — see LicenseTerm::DeployerObligation.
        LicenseTerm::DeployerObligation {
            text:
                "If, on the Llama 3.1 version release date, the monthly active users … is greater \
                   than 700 million monthly active users in the preceding calendar month, you must \
                   request a license from Meta",
        },
    ],
};

/// Ideogram Non-Commercial Model Agreement, last updated June 3, 2026 (sc-16665).
///
/// Text read at <https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md> on
/// 2026-08-02 under an authenticated session (the repository is gated), at `sha`
/// `ee79a7237b519f1402ceacf952f30c8a31ec5073`.
///
/// # Outputs are addressed, and not restricted commercially
///
/// §7 "We claim no rights in outputs you generate using the Model. … You may not use any Output to
/// develop, train, fine-tune or distill a model or other product or services that is competitive
/// with the Model" — an anti-competitive-training restriction, not a commercial one, and the
/// vocabulary has no variant for it.
///
/// §1(d) does fold a *use of Outputs* into the **definition** of Non-Commercial Purposes: "any use …
/// that involves generating Output to include in, or to advertise or promote, revenue-generating
/// products or services, in each case, is not a Non-Commercial Purpose." Whether a definition
/// reaches Outputs as a licence term is a legal read, so [`LicenseTerm::NonCommercialOutputs`] is
/// **not** transcribed and the passage is recorded in
/// `docs/licensing/sc-16665-checkpoint-licence-evidence.md` instead.
///
/// # U11 linkage
///
/// §3(iii) is transcribed as two terms for the same reason as [`FLUX_NON_COMMERCIAL_V2_1`] — the
/// clause names itself an *attribution* notice. See that family's U11 note.
pub const IDEOGRAM_4_NON_COMMERCIAL: LicenseFamily = LicenseFamily {
    id: "ideogram-4-non-commercial",
    spdx_id: "LicenseRef-Ideogram-4-Non-Commercial",
    name: "Ideogram Non-Commercial Model Agreement",
    text_url: "https://huggingface.co/ideogram-ai/ideogram-4-fp8/blob/main/LICENSE.md",
    terms: &[
        // §2 "We hereby permit you to use, reproduce, Distribute, copy, create derivative works of
        // (including Model Derivatives), and make modifications to the Model for Non-Commercial
        // Purposes subject to the terms of this Agreement".
        LicenseTerm::NonCommercialWeights,
        // §3(i) "all permitted use of the reproduced and re-Distributed Model or Model Derivatives
        // must be on terms that are no less restrictive than those set forth in this Agreement for
        // the Model" — the heavier flow-down shape.
        LicenseTerm::DownstreamRestrictions {
            family: "ideogram-4-non-commercial",
        },
        // §3(ii) "you provide all third party recipients of the Model or Model Derivative a copy of
        // this Agreement".
        LicenseTerm::DownstreamLicenseCopy {
            family: "ideogram-4-non-commercial",
        },
        // §3(iii) "you retain in all copies of the Model or Model Derivatives that you Distribute
        // the following attribution notice within a \"Notice\" text file that accompanies such
        // copy". The clause prescribes the notice's exact wording; the component rows carry it
        // complete — see `IDEOGRAM_4_PRESCRIBED_NOTICE` in `license::components`.
        LicenseTerm::NoticeFileRequired,
        // Same §3(iii) clause, which names it an "attribution notice" — see the U11 note above.
        LicenseTerm::AttributionRequired,
        // §4 "adhere to the Acceptable Use Policy available at
        // https://ideogram.ai/legal/usage-policy, which is hereby incorporated by reference into
        // this Agreement". The address is in the licence text itself, not only in a gate prompt.
        // Fetched 2026-08-02: HTTP 200.
        LicenseTerm::AcceptableUsePolicy {
            url: Some("https://ideogram.ai/legal/usage-policy"),
        },
        // §4, quoted.
        LicenseTerm::DeployerObligation {
            text: "You are responsible for implementing appropriate safety measures, including \
                   content filters and human oversight, suitable for your use case and to prevent \
                   the creation, display, generation or reproduction of unlawful or infringing \
                   content",
        },
    ],
};

/// SAM License, last updated November 19, 2025 — Meta's bespoke licence for SAM 3 (sc-16665).
///
/// Text read at <https://huggingface.co/facebook/sam3/blob/main/LICENSE> on 2026-08-02 under an
/// authenticated session (the repository is gated for manual approval), at `sha`
/// `3c879f39826c281e95690f02c7821c4de09afae7`.
///
/// # This licence is **not** non-commercial
///
/// Worth stating plainly, because a bespoke vendor licence is easy to assume otherwise. §1(a) grants
/// a "non-exclusive, worldwide, non-transferable and royalty-free limited license … to use,
/// reproduce, distribute, copy, create derivative works of, and make modifications to the SAM
/// Materials" with no purpose bound, so [`LicenseTerm::NonCommercialWeights`] is absent.
///
/// # SAM 2.1 is a different licence
///
/// `facebook/sam2.1-hiera-large` and `facebook/sam2.1-hiera-base-plus` declare plain `apache-2.0`
/// and are ungated. Meta licenses SAM 2.1 and SAM 3 differently; this family covers `facebook/sam3`
/// **only**.
///
/// # The acknowledgement duty is conditional — U8's shape, applied
///
/// [`LicenseTerm::AttributionRequired`] is deliberately **not** transcribed. §1(b)(ii)'s
/// acknowledgement duty binds only on submitting research for publication, while the typed term
/// reads as an unconditional duty on every use — so transcribing it that way would make every SAM 3
/// render's derived union name an obligation this text does not impose on it. That is the same
/// defect sc-16662's open item **U8** settled for [`LLAMA_3_1_COMMUNITY`]: a 700M-MAU threshold is
/// not a [`LicenseTerm::RevenueCeiling`] because the typed term would be a false transcription, so
/// the condition is disclosed verbatim as a [`LicenseTerm::DeployerObligation`] instead. This family
/// applies that existing decision; it does not make a new one. The duty is disclosed in full, with
/// its condition intact, rather than dropped.
///
/// # What the text does not say
///
/// No notice file (`Notice` does not occur), no acceptable-use policy (§1(b)(iii)–(v) enumerate
/// trade-control and reverse-engineering restrictions inline and reference no external document), no
/// revenue ceiling, no registration.
pub const META_SAM_LICENSE: LicenseFamily = LicenseFamily {
    id: "meta-sam-license",
    spdx_id: "LicenseRef-Meta-SAM",
    name: "SAM License",
    text_url: "https://huggingface.co/facebook/sam3/blob/main/LICENSE",
    terms: &[
        // §1(b)(i) "If you distribute or make the SAM Materials, or any derivative works thereof,
        // available to a third party, you may only do so under the terms of this Agreement and you
        // shall provide a copy of this Agreement with any such SAM Materials." One sentence, both
        // flow-down shapes.
        LicenseTerm::DownstreamRestrictions {
            family: "meta-sam-license",
        },
        LicenseTerm::DownstreamLicenseCopy {
            family: "meta-sam-license",
        },
        // §1(b)(ii), quoted in full (the source sentence carries no elision). The duty is real but
        // conditional on publishing research, and the vocabulary carries no conditional attribution
        // variant — so it is disclosed verbatim rather than flattened into AttributionRequired,
        // which would assert an unconditional duty the text does not impose. Same move, same reason
        // as LLAMA_3_1_COMMUNITY's MAU threshold (U8); see the doc comment above.
        LicenseTerm::DeployerObligation {
            text:
                "If you submit for publication the results of research you perform on, using, or \
                   otherwise in connection with SAM Materials, you must acknowledge the use of SAM \
                   Materials in your publication.",
        },
    ],
};

/// Every licence family whose text has been read, ordered by [`LicenseFamily::id`].
///
/// The reviewed unit of the licence surface. Nineteen entries — the sixteen sc-16662 landed, plus
/// the three the media checkpoint census forced (sc-16665).
///
/// sc-16662 landed sixteen: the fourteen the story sketched, with `stable-video-diffusion-community`
/// merged into [`STABILITY_AI_COMMUNITY`] (one text, three declared strings), the draft's single
/// NVIDIA row split into [`NVIDIA_OPEN_MODEL`] and [`NVIDIA_NSCLV1`] (two different texts), and
/// [`APPLE_MLR`] and [`LLAMA_3_1_COMMUNITY`] added because shipped checkpoints declare them.
///
/// sc-16665 added three, each because a shipped checkpoint declares a text none of the sixteen
/// carries: [`FLUX_NON_COMMERCIAL_V2_1`] (the three FLUX.2 repositories),
/// [`IDEOGRAM_4_NON_COMMERCIAL`] (`ideogram-ai/ideogram-4-fp8` and the ostris TurboTime LoRA) and
/// [`META_SAM_LICENSE`] (`facebook/sam3` alone). A fourth candidate, `kolors-model-license`, was
/// **withheld**: whether `Kwai-Kolors/Kolors-diffusers` is governed by its card's `apache-2.0` or by
/// the `MODEL_LICENSE` committed beside its weights is sc-16662's open item **U6**, and landing a
/// family would answer it silently.
///
/// Membership means "this text was read on the date its doc comment records". It does **not** mean a
/// shipped checkpoint declares it: [`NVIDIA_OPEN_MODEL`] still has no confirmed checkpoint in this
/// repository (sc-16662 **U4**, which the census recommends closing as *no shipped checkpoint*), and
/// several families are reachable only through components sc-16665 deliberately left unwritten.
pub const LICENSE_FAMILIES: &[LicenseFamily] = &[
    APACHE_2_0,
    APPLE_MLR,
    CC_BY_NC_4_0,
    CHATGLM3_MODEL_LICENSE,
    CIRCLESTONE_LABS_NON_COMMERCIAL,
    CREATIVEML_OPENRAIL_PP_M,
    FLUX_1_DEV_NON_COMMERCIAL,
    FLUX_NON_COMMERCIAL_V2_1,
    GEMMA_TERMS,
    IDEOGRAM_4_NON_COMMERCIAL,
    INSIGHTFACE_RESEARCH_ONLY,
    KREA_2_COMMUNITY,
    LLAMA_3_1_COMMUNITY,
    LTX_2_COMMUNITY,
    META_SAM_LICENSE,
    MINIMAX_H3_COMMUNITY,
    MIT,
    NVIDIA_NSCLV1,
    NVIDIA_OPEN_MODEL,
    STABILITY_AI_COMMUNITY,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::license::{
        component_licenses_manifest_json, license_table_conformance_errors, resolve_family,
    };

    /// Every const is reachable through the public slice, and through the one sanctioned lookup
    /// path. A family declared but left out of `LICENSE_FAMILIES` would be invisible to every
    /// consumer while still looking landed in source.
    #[test]
    fn every_family_is_in_the_slice_and_resolves_by_id() {
        const DECLARED: &[LicenseFamily] = &[
            APACHE_2_0,
            APPLE_MLR,
            CC_BY_NC_4_0,
            CHATGLM3_MODEL_LICENSE,
            CIRCLESTONE_LABS_NON_COMMERCIAL,
            CREATIVEML_OPENRAIL_PP_M,
            FLUX_1_DEV_NON_COMMERCIAL,
            FLUX_NON_COMMERCIAL_V2_1,
            GEMMA_TERMS,
            IDEOGRAM_4_NON_COMMERCIAL,
            INSIGHTFACE_RESEARCH_ONLY,
            KREA_2_COMMUNITY,
            LLAMA_3_1_COMMUNITY,
            LTX_2_COMMUNITY,
            META_SAM_LICENSE,
            MINIMAX_H3_COMMUNITY,
            MIT,
            NVIDIA_NSCLV1,
            NVIDIA_OPEN_MODEL,
            STABILITY_AI_COMMUNITY,
        ];
        assert_eq!(LICENSE_FAMILIES.len(), 20);
        assert_eq!(LICENSE_FAMILIES, DECLARED);

        for family in LICENSE_FAMILIES {
            assert_eq!(
                resolve_family(LICENSE_FAMILIES, family.id),
                Some(family),
                "{:?} must resolve to itself",
                family.id
            );
        }
        // The slice is sorted by id, so a reader can find a row and a future diff stays local.
        let mut ids: Vec<&str> = LICENSE_FAMILIES.iter().map(|f| f.id).collect();
        let declared_order = ids.clone();
        ids.sort_unstable();
        assert_eq!(ids, declared_order, "LICENSE_FAMILIES must be sorted by id");
    }

    /// The ship-gate over the whole family set. Component and provider rows are a later slice, so
    /// the table is checked with those sections empty — which is exactly how a catalog will call it
    /// while they are being filled in.
    #[test]
    fn family_table_is_conformant() {
        assert_eq!(
            license_table_conformance_errors(LICENSE_FAMILIES, &[], &[]),
            Vec::<String>::new()
        );
    }

    /// Gating is a per-checkpoint distribution setting, not a clause in any of these sixteen texts.
    /// The conformance checker rejects it on a family; assert it directly too, because the reason it
    /// is absent is evidentiary — no licence read for this table says "the weights are gated" — and
    /// that should fail loudly rather than as a generic conformance message.
    #[test]
    fn no_family_declares_gated_access() {
        for family in LICENSE_FAMILIES {
            assert!(
                !family.imposes(LicenseTerm::GatedAccess),
                "{:?} declares gated_access; gating belongs on ComponentLicense::gated",
                family.id
            );
        }
    }

    /// Both evidence packs found **no quote anywhere** supporting an outputs restriction, in any
    /// family. Four texts restrict non-commercial *use of the weights* and are silent on outputs
    /// (`insightface-research-only`, `nvidia-nsclv1`, `apple-mlr`, `cc-by-nc-4-0`); four others
    /// address outputs and permit them commercially (`flux-1-dev-non-commercial` §2(d),
    /// `circlestone-labs-non-commercial` §2(e), `flux-non-commercial-v2-1` §2(d),
    /// `ideogram-4-non-commercial` §7). Inferring a restriction from silence would be a legal
    /// reading this surface does not make, so the term appears nowhere — and re-adding it must
    /// require producing a quote, which is what this test is here to force.
    ///
    /// The nearest miss is Ideogram §1(d), which folds a use of Outputs into the *definition* of
    /// Non-Commercial Purposes. That is recorded in
    /// `docs/licensing/sc-16665-checkpoint-licence-evidence.md` and deliberately not transcribed:
    /// a definition is not an outputs restriction without a legal reading.
    #[test]
    fn no_family_infers_an_outputs_restriction_from_a_use_restriction() {
        for family in LICENSE_FAMILIES {
            assert!(
                !family.imposes(LicenseTerm::NonCommercialOutputs),
                "{:?} declares non_commercial_outputs; the evidence pack records no quote for it \
                 in any family — add the quote to docs/licensing/ before adding the term",
                family.id
            );
        }
        // The weights-side restriction is not what is being suppressed: four families do carry it.
        let non_commercial: Vec<&str> = LICENSE_FAMILIES
            .iter()
            .filter(|f| f.imposes(LicenseTerm::NonCommercialWeights))
            .map(|f| f.id)
            .collect();
        assert_eq!(
            non_commercial,
            vec![
                "apple-mlr",
                "cc-by-nc-4-0",
                "circlestone-labs-non-commercial",
                "flux-1-dev-non-commercial",
                "flux-non-commercial-v2-1",
                "ideogram-4-non-commercial",
                "insightface-research-only",
                "nvidia-nsclv1",
            ]
        );
    }

    /// Two licences name a *different* amount and two name the *same* amount with opposite
    /// readings. Stability says "more than USD $1,000,000" and Krea says "meet or exceed
    /// $1,000,000": at exactly one million dollars the two texts say different things, and the
    /// boundary is the only field that records it. Flattening it would put revenue equal to the
    /// amount on the wrong side of one of them.
    #[test]
    fn ceiling_boundaries_distinguish_the_texts_at_the_amount() {
        fn ceiling(family: &LicenseFamily) -> (u64, CeilingBoundary) {
            family
                .terms
                .iter()
                .find_map(|term| match *term {
                    LicenseTerm::RevenueCeiling {
                        amount_usd,
                        boundary,
                    } => Some((amount_usd, boundary)),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{:?} should name a revenue ceiling", family.id))
        }

        assert_eq!(
            ceiling(&STABILITY_AI_COMMUNITY),
            (1_000_000, CeilingBoundary::Exclusive)
        );
        assert_eq!(
            ceiling(&LTX_2_COMMUNITY),
            (10_000_000, CeilingBoundary::Inclusive)
        );
        // Same amount as Stability, opposite reading — the case a bare amount cannot express.
        assert_eq!(
            ceiling(&KREA_2_COMMUNITY),
            (1_000_000, CeilingBoundary::Inclusive)
        );
        assert_ne!(ceiling(&STABILITY_AI_COMMUNITY), ceiling(&KREA_2_COMMUNITY));

        // And exactly those three families name one.
        let with_ceiling: Vec<&str> = LICENSE_FAMILIES
            .iter()
            .filter(|f| f.terms.iter().any(|t| t.tag() == "revenue_ceiling"))
            .map(|f| f.id)
            .collect();
        assert_eq!(
            with_ceiling,
            vec![
                "krea-2-community",
                "ltx-2-community",
                "minimax-h3-community",
                "stability-ai-community"
            ]
        );
    }

    /// The two flow-down shapes are not interchangeable, and the table has to keep them apart:
    /// handing a recipient the licence text is a different duty from writing the licence's use
    /// restrictions into your own agreement with your users. Four families impose the heavier
    /// shape, and three of those impose *both* — which the union must preserve as two elements.
    #[test]
    fn the_two_flow_down_shapes_stay_distinct() {
        fn ids(tag: &str) -> Vec<&'static str> {
            LICENSE_FAMILIES
                .iter()
                .filter(|f| f.terms.iter().any(|t| t.tag() == tag))
                .map(|f| f.id)
                .collect()
        }

        assert_eq!(
            ids("downstream_restrictions"),
            vec![
                "creativeml-openrail-pp-m",
                "gemma-terms",
                "ideogram-4-non-commercial",
                "ltx-2-community",
                "meta-sam-license",
                "minimax-h3-community",
                "nvidia-nsclv1",
            ],
            "restrictions-as-enforceable-provisions is the heavier duty and only these state it"
        );
        assert_eq!(
            ids("downstream_license_copy"),
            vec![
                "apache-2-0",
                "apple-mlr",
                "circlestone-labs-non-commercial",
                "flux-1-dev-non-commercial",
                "flux-non-commercial-v2-1",
                "gemma-terms",
                "ideogram-4-non-commercial",
                "krea-2-community",
                "llama-3-1-community",
                "ltx-2-community",
                "meta-sam-license",
                "minimax-h3-community",
                "nvidia-nsclv1",
                "nvidia-open-model",
                "stability-ai-community",
            ]
        );

        // Five texts impose both, and CreativeML Open RAIL++-M imposes only the heavier one — so
        // neither list is a subset of the other and neither variant can stand in for the other.
        for id in [
            "gemma-terms",
            "ideogram-4-non-commercial",
            "ltx-2-community",
            "meta-sam-license",
            "nvidia-nsclv1",
        ] {
            let family = resolve_family(LICENSE_FAMILIES, id).unwrap();
            assert!(family.imposes(LicenseTerm::DownstreamLicenseCopy { family: id }));
            assert!(family.imposes(LicenseTerm::DownstreamRestrictions { family: id }));
        }
        assert!(
            !CREATIVEML_OPENRAIL_PP_M.imposes(LicenseTerm::DownstreamLicenseCopy {
                family: "creativeml-openrail-pp-m"
            })
        );

        // Every flow-down term names the family that declares it, so resolving one reaches the text
        // that actually imposed the duty.
        for family in LICENSE_FAMILIES {
            for term in family.terms {
                if let Some(named) = term.flow_down_family() {
                    assert_eq!(named, family.id);
                    assert_eq!(
                        resolve_family(LICENSE_FAMILIES, named).map(|f| f.text_url),
                        Some(family.text_url)
                    );
                }
            }
        }
    }

    /// Three texts name an acceptable-use policy and publish no address, and one names a
    /// registration with no address. `None` is the recorded fact; an invented URL would be a
    /// fabricated disclosure and a dropped term would lose the fact that the licence points
    /// somewhere. In particular `https://blackforestlabs.ai/aup` — a guess that appeared in an
    /// earlier fixture — 404s and occurs nowhere in the FLUX.1 [dev] text, so it must never appear
    /// in this table.
    #[test]
    fn addressless_references_are_recorded_as_absent_not_invented() {
        let mut policy_without_url = Vec::new();
        let mut registration_without_contact = Vec::new();
        for family in LICENSE_FAMILIES {
            for term in family.terms {
                match *term {
                    LicenseTerm::AcceptableUsePolicy { url: None } => {
                        policy_without_url.push(family.id)
                    }
                    LicenseTerm::RegistrationRequired { contact: None } => {
                        registration_without_contact.push(family.id)
                    }
                    _ => {}
                }
            }
        }
        assert_eq!(
            policy_without_url,
            vec![
                "creativeml-openrail-pp-m",
                "flux-1-dev-non-commercial",
                "ltx-2-community",
                "minimax-h3-community",
            ]
        );
        assert_eq!(registration_without_contact, vec!["ltx-2-community"]);

        // The addressed policies, and the addressed registrations.
        assert!(GEMMA_TERMS.imposes(LicenseTerm::AcceptableUsePolicy {
            url: Some("https://ai.google.dev/gemma/prohibited_use_policy")
        }));
        assert!(KREA_2_COMMUNITY.imposes(LicenseTerm::AcceptableUsePolicy {
            url: Some("https://www.krea.ai/krea-2-use-policy")
        }));

        // sc-16665 found that BFL now publishes a live usage policy — but only for FLUX.2. FLUX.1
        // [dev]'s gate prompt still cites a `POLICY.md` that does not exist (re-verified
        // 2026-08-02), so back-porting the live address onto the FLUX.1 family would invent an
        // address for a text that names none. The two families must disagree here.
        assert!(
            FLUX_NON_COMMERCIAL_V2_1.imposes(LicenseTerm::AcceptableUsePolicy {
                url: Some("https://bfl.ai/legal/usage-policy")
            })
        );
        assert!(
            FLUX_1_DEV_NON_COMMERCIAL.imposes(LicenseTerm::AcceptableUsePolicy { url: None }),
            "sc-16662's U2 finding for FLUX.1 [dev] stands: the v2.1 address must not be back-ported"
        );
        assert!(!FLUX_1_DEV_NON_COMMERCIAL
            .terms
            .iter()
            .any(|t| matches!(t, LicenseTerm::AcceptableUsePolicy { url: Some(_) })));

        let json = component_licenses_manifest_json(LICENSE_FAMILIES, &[], &[]);
        assert!(
            !json.contains("blackforestlabs.ai"),
            "the fixture's guessed FLUX policy URL 404s and is in no licence text"
        );
        // An absent address is an explicit null a consumer can render, not a missing key.
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let flux = value["families"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["id"] == "flux-1-dev-non-commercial")
            .unwrap();
        let policy = flux["terms"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["term"] == "acceptable_use_policy")
            .unwrap();
        assert!(policy.get("url").is_some(), "the key must be present");
        assert!(policy["url"].is_null(), "and carry an explicit null");
    }

    /// LTX-2's transcription comes from the copy committed **beside the weights**, which is the text
    /// a user who downloads the checkpoint receives. The licensor also publishes a different
    /// `LICENSE` in its GitHub repository, and that copy links an address at the registration
    /// sentence the shipped copy leaves bare. Pinning the choice here means a later edit to the more
    /// convenient text is a deliberate, reviewable change rather than a quiet one.
    #[test]
    fn ltx_2_transcribes_the_text_shipped_with_the_weights() {
        assert_eq!(
            LTX_2_COMMUNITY.text_url,
            "https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE"
        );
        assert!(
            LTX_2_COMMUNITY.imposes(LicenseTerm::RegistrationRequired { contact: None }),
            "the shipped copy names the registration and no address"
        );
        assert!(
            !LTX_2_COMMUNITY
                .terms
                .iter()
                .any(|t| matches!(t, LicenseTerm::RegistrationRequired { contact: Some(_) })),
            "the GitHub copy's address is not in the text shipped with the weights"
        );
    }

    /// The tripwire for "no term without a quote".
    ///
    /// Each row below is the exact term set transcribed for one family, and every term in it is
    /// backed by a verbatim quote in `docs/licensing/sc-16662-licence-family-evidence.md` or, for
    /// the three families sc-16665 added, `docs/licensing/sc-16665-checkpoint-licence-evidence.md`
    /// — the source comments beside each const carry the same quote. Full linkage is not mechanisable
    /// from here (the quotes live in Markdown, and this crate reads no files at test time), so this
    /// census is the mechanism: adding, removing or re-parameterising any term fails this test and
    /// sends the author back to the note to record or produce the quote.
    #[test]
    fn every_transcribed_term_is_pinned_to_the_evidence_pack() {
        fn render(term: &LicenseTerm) -> String {
            match *term {
                LicenseTerm::RevenueCeiling {
                    amount_usd,
                    boundary,
                } => format!("revenue_ceiling:{amount_usd}:{}", boundary.tag()),
                LicenseTerm::RegistrationRequired { contact } => {
                    format!("registration_required:{}", contact.unwrap_or("<none>"))
                }
                LicenseTerm::AcceptableUsePolicy { url } => {
                    format!("acceptable_use_policy:{}", url.unwrap_or("<none>"))
                }
                // Obligations are quoted at length; the leading words identify the clause without
                // making the expectation table unreadable.
                LicenseTerm::DeployerObligation { text } => {
                    let head: String = text.chars().take(48).collect();
                    format!("deployer_obligation:{head}")
                }
                LicenseTerm::DownstreamLicenseCopy { family }
                | LicenseTerm::DownstreamRestrictions { family } => {
                    format!("{}:{family}", term.tag())
                }
                _ => term.tag().to_string(),
            }
        }

        let census: Vec<(&str, Vec<String>)> = LICENSE_FAMILIES
            .iter()
            .map(|family| (family.id, family.terms.iter().map(render).collect()))
            .collect();

        let expected: Vec<(&str, Vec<&str>)> = vec![
            (
                "apache-2-0",
                vec![
                    "attribution_required",
                    "notice_file_required",
                    "downstream_license_copy:apache-2-0",
                ],
            ),
            (
                "apple-mlr",
                vec![
                    "non_commercial_weights",
                    "downstream_license_copy:apple-mlr",
                    "attribution_required",
                ],
            ),
            (
                "cc-by-nc-4-0",
                vec!["attribution_required", "non_commercial_weights"],
            ),
            (
                "chatglm3-model-license",
                vec![
                    "registration_required:https://open.bigmodel.cn/mla/form",
                    "attribution_required",
                ],
            ),
            (
                "circlestone-labs-non-commercial",
                vec![
                    "non_commercial_weights",
                    "downstream_license_copy:circlestone-labs-non-commercial",
                    "attribution_required",
                ],
            ),
            (
                "creativeml-openrail-pp-m",
                vec![
                    "downstream_restrictions:creativeml-openrail-pp-m",
                    "attribution_required",
                    "notice_file_required",
                    "acceptable_use_policy:<none>",
                ],
            ),
            (
                "flux-1-dev-non-commercial",
                vec![
                    "non_commercial_weights",
                    "downstream_license_copy:flux-1-dev-non-commercial",
                    "deployer_obligation:implement and maintain content filtering measure",
                    "acceptable_use_policy:<none>",
                ],
            ),
            (
                "flux-non-commercial-v2-1",
                vec![
                    "non_commercial_weights",
                    "downstream_license_copy:flux-non-commercial-v2-1",
                    "notice_file_required",
                    "attribution_required",
                    "deployer_obligation:implement and maintain content filtering measure",
                    "deployer_obligation:ensure Output includes disclosure (or other indi",
                    "acceptable_use_policy:https://bfl.ai/legal/usage-policy",
                ],
            ),
            (
                "gemma-terms",
                vec![
                    "downstream_restrictions:gemma-terms",
                    "downstream_license_copy:gemma-terms",
                    "notice_file_required",
                    "acceptable_use_policy:https://ai.google.dev/gemma/prohibited_use_policy",
                ],
            ),
            (
                "ideogram-4-non-commercial",
                vec![
                    "non_commercial_weights",
                    "downstream_restrictions:ideogram-4-non-commercial",
                    "downstream_license_copy:ideogram-4-non-commercial",
                    "notice_file_required",
                    "attribution_required",
                    "acceptable_use_policy:https://ideogram.ai/legal/usage-policy",
                    "deployer_obligation:You are responsible for implementing appropriate",
                ],
            ),
            ("insightface-research-only", vec!["non_commercial_weights"]),
            (
                "krea-2-community",
                vec![
                    "revenue_ceiling:1000000:inclusive",
                    "registration_required:opensource@krea.ai",
                    "downstream_license_copy:krea-2-community",
                    "attribution_required",
                    "notice_file_required",
                    "deployer_obligation:You must implement reasonable and appropriate Co",
                    "deployer_obligation:Where required by applicable law, regulation, or",
                    "acceptable_use_policy:https://www.krea.ai/krea-2-use-policy",
                ],
            ),
            (
                "llama-3-1-community",
                vec![
                    "downstream_license_copy:llama-3-1-community",
                    "attribution_required",
                    "notice_file_required",
                    "acceptable_use_policy:https://llama.meta.com/llama3_1/use-policy",
                    "deployer_obligation:If, on the Llama 3.1 version release date, the m",
                ],
            ),
            (
                "ltx-2-community",
                vec![
                    "revenue_ceiling:10000000:inclusive",
                    "registration_required:<none>",
                    "downstream_restrictions:ltx-2-community",
                    "downstream_license_copy:ltx-2-community",
                    "attribution_required",
                    "notice_file_required",
                    // The "…" is the source's own elision, preserved in the landed string.
                    "deployer_obligation:Any Derivative of LTX-2 … must be distributed ex",
                    "acceptable_use_policy:<none>",
                ],
            ),
            (
                "meta-sam-license",
                vec![
                    "downstream_restrictions:meta-sam-license",
                    "downstream_license_copy:meta-sam-license",
                    // §1(b)(ii)'s acknowledgement duty, conditional on publishing research. Not
                    // `attribution_required`: the typed term would assert an unconditional duty —
                    // U8's shape, applied. See META_SAM_LICENSE's doc comment.
                    "deployer_obligation:If you submit for publication the results of res",
                ],
            ),
            (
                "minimax-h3-community",
                vec![
                    "revenue_ceiling:20000000:exclusive",
                    "registration_required:api@minimax.io",
                    "downstream_restrictions:minimax-h3-community",
                    "downstream_license_copy:minimax-h3-community",
                    "attribution_required",
                    "notice_file_required",
                    // §V.4, the territorial exclusion (EU / UK / Republic of Korea / USA).
                    "deployer_obligation:You may not use, reproduce, modify, distribute, ",
                    "deployer_obligation:you must implement, maintain, test, and periodic",
                    "acceptable_use_policy:<none>",
                ],
            ),
            ("mit", vec!["attribution_required"]),
            (
                "nvidia-nsclv1",
                vec![
                    "non_commercial_weights",
                    "downstream_license_copy:nvidia-nsclv1",
                    "downstream_restrictions:nvidia-nsclv1",
                    "attribution_required",
                ],
            ),
            (
                "nvidia-open-model",
                vec![
                    "deployer_obligation:If You bypass, disable, reduce the efficacy of, ",
                    "downstream_license_copy:nvidia-open-model",
                    "notice_file_required",
                    "attribution_required",
                    "acceptable_use_policy:https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/",
                ],
            ),
            (
                "stability-ai-community",
                vec![
                    "revenue_ceiling:1000000:exclusive",
                    "registration_required:https://stability.ai/community-license",
                    "attribution_required",
                    "notice_file_required",
                    "downstream_license_copy:stability-ai-community",
                    "acceptable_use_policy:https://stability.ai/use-policy",
                ],
            ),
        ];

        let expected: Vec<(&str, Vec<String>)> = expected
            .into_iter()
            .map(|(id, terms)| (id, terms.into_iter().map(str::to_string).collect()))
            .collect();
        assert_eq!(census, expected);
    }
}
