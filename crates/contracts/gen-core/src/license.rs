//! Machine-readable **model-weight licence** surface (sc-13332, schema 3 since sc-16661).
//!
//! The crate/source licence axis is already captured by the release tooling's SPDX SBOM
//! (`scripts/release/build_release.py`, one entry per resolved Cargo package). The licence of a
//! provider's **model weights** — the pinned checkpoint each provider loads — is a *separate axis*
//! that cargo tooling never sees, and this module is the tensor-free contract for it.
//!
//! ## Disclosure only
//!
//! Everything here exists so a consumer can **show** a user what the upstream texts say. No value
//! in this module blocks, gates, degrades or withholds anything, and none ever should — this
//! surface describes, it does not decide. Whether a given use is permitted is the consumer's
//! evaluation of these facts against its own situation: its revenue, whether it redistributes
//! weights, which agreements it has with its own users. This crate has none of that information.
//!
//! ## Three layers
//!
//! * [`LicenseFamily`] — one upstream licence text, reviewed once by a human (19 rows);
//! * [`LicenseTerm`] — typed obligations, so a licence join is a **set union** rather than a
//!   boolean AND, and the consumer applies its own profile;
//! * [`ComponentLicense`] — one row per loaded artifact, carrying the provenance
//!   ([`source_url`](ComponentLicense::source_url), [`declared`](ComponentLicense::declared),
//!   [`retrieved`](ComponentLicense::retrieved)) that makes review a quote check and makes upstream
//!   re-licensing detectable.
//!
//! A provider's terms are **derived** from its components ([`provider_terms`]) and never
//! hand-authored: a hand-typed composite is a second place to be wrong and can drift from its own
//! components. [`license_table_conformance_errors`] is the catalog ship-gate, and
//! [`component_licenses_manifest_json`] emits the `schema_version: 3` manifest the release tooling
//! ships beside the SPDX SBOM.
//!
//! ## What schema 2 stored, and why it is gone (sc-16663)
//!
//! The retired v2 surface recorded a legal **conclusion**, `WeightLicense::commercial_use`, that
//! depends on facts inference does not have. Several shipped checkpoints had no correct boolean:
//! FLUX.1 \[dev\] (weights non-commercial, outputs explicitly commercial-OK), SD3.5 and every
//! Stable Audio 3 registration (no prohibition at all — a revenue threshold and a registration),
//! and Kolors (commercial weight use only after registering with Kuaishou). Whichever value was
//! written was silently wrong for half of callers, and a join computed over it read as
//! authoritative. A wrong bool is worse than an absent field. sc-16663 migrated the audio lane onto
//! the layers above and deleted `WeightLicense`, `WeightLicenseEntry`, `commercial_use` and the
//! `schema_version: 2` emitter outright.
//!
//! ## The tables themselves (sc-16662, sc-16665)
//!
//! [`families`] carries the transcribed [`LicenseFamily`] rows — nineteen upstream texts, each term
//! backed by a verbatim quote in `docs/licensing/sc-16662-licence-family-evidence.md` or
//! `docs/licensing/sc-16665-checkpoint-licence-evidence.md`.
//!
//! [`components`] carries the media lane's shared [`ComponentLicense`] rows — one per upstream
//! checkpoint, read by **both** media catalogs, because a licence is a property of the checkpoint
//! and the MLX and Candle engines load the same checkpoints. Only the provider→component mapping
//! ([`ProviderComponents`]) differs per backend, and that lives in each catalog. The audio lane
//! keeps its rows in its provider crates instead, which predates this table and is not affected by
//! it.
//!
//! Both tables are **provisional**: an agent gathered the quotes and no human has yet signed them
//! off. The component table is also deliberately **incomplete** — a checkpoint whose licence the
//! evidence could not settle has no row rather than a guessed one, and the gaps are enumerated in
//! [`components`] and in its evidence note.

pub mod components;
pub mod families;

// =================================================================================================
// v3 — licence families, typed terms, and per-component rows (sc-16661), amended by sc-16898.
//
// DISCLOSURE ONLY. Everything below exists so a consumer can SHOW a user what the upstream texts
// say. No value here blocks, gates, degrades, or withholds anything, and none ever should — this
// surface describes, it does not decide.
//
// The retired v2 surface recorded a legal CONCLUSION (`WeightLicense::commercial_use`) that depends
// on facts inference does not have: the consumer's revenue, whether it redistributes weights or only
// sells renders, whether it registered with the upstream. Several shipped checkpoints had no correct
// boolean — FLUX.1-dev (weights non-commercial, outputs commercially usable), SD3.5 and every Stable
// Audio 3 registration (no prohibition at all: a revenue threshold and a registration), and Kolors
// (commercial weights use only after registering with Kuaishou). Whichever value was written was
// silently wrong for half of callers, and a join computed over it read as authoritative.
//
// v3 stores facts instead, in three layers:
//   * [`LicenseFamily`] — the reviewed unit, ~14 rows, read by a human once;
//   * [`LicenseTerm`]   — typed obligations, so a licence join is a set union rather than a
//                         boolean AND, and the consumer applies its own profile;
//   * [`ComponentLicense`] — one row per loaded artifact, carrying the provenance that makes review
//                         a quote check and makes upstream re-licensing detectable.
//
// A provider's terms are DERIVED from its components ([`provider_terms`]) and never hand-authored:
// v2's hand-typed composite row is a second place to be wrong and can drift from its own components.
//
// sc-16663 migrated the audio lane (43 rows / 18 providers) onto these layers and deleted the v2
// types, `commercial_use` and the `schema_version: 2` emitter. There is no compatibility shim: a
// wrong bool is worse than an absent field, because a join computed over it looks authoritative.
//
// sc-16898 amended three shapes, each forced by reading the actual licence texts
// (docs/licensing/sc-16662-licence-family-evidence.md) rather than by a design preference:
//
//   * `DownstreamFlowDown` was a bare variant, so the eleven families that impose a flow-down
//     deduped to ONE element of the union. It is now two variants — `DownstreamLicenseCopy` and
//     `DownstreamRestrictions` — each carrying the family whose text it points at.
//   * `GatedAccess` sat in `LicenseFamily::terms`, but SVD-XT (ungated) and SD3.5 (gated) are
//     governed by the SAME Stability text; it moved to `ComponentLicense::gated` and is raised into
//     the derived union from there.
//   * `RevenueCeiling` carried an amount with no boundary reading, while Stability says "more than
//     USD $1,000,000" and LTX-2 says "at least $10,000,000" — different answers at exactly the
//     amount. It now carries a `CeilingBoundary`.
//
// sc-16662 landed the family table itself (`license::families`) and amended one more shape for the
// same reason — the texts, not a preference. `AcceptableUsePolicy::url` and
// `RegistrationRequired::contact` are now `Option`: four of the sixteen families name a policy or a
// registration and publish no address for it, so a required address left only a fabricated URL or a
// dropped fact. See the `LicenseTerm` type note.
// =================================================================================================

/// Which side of a [`LicenseTerm::RevenueCeiling`] the named amount itself falls on.
///
/// The licences genuinely differ here and the difference is not cosmetic, so the type carries it
/// rather than picking a house convention and mis-transcribing half the table. Both readings appear
/// in the catalog at time of writing: the Stability AI Community License names its threshold as
/// *"more than USD $1,000,000"*, while the LTX-2 Community License names its own as *"at least
/// $10,000,000"*. Flattening them would put revenue exactly equal to the amount on the wrong side of
/// one of the two.
///
/// This is a description of what the text says, not a computation: nothing in this crate compares a
/// consumer's revenue against a ceiling, and nothing here changes behaviour based on one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CeilingBoundary {
    /// The licence states its threshold with a strict comparison — "more than $N" — so the amount
    /// itself is **below** the threshold the text names. (Stability AI Community License.)
    Exclusive,
    /// The licence states its threshold inclusively — "at least $N", "$N or more" — so the amount
    /// itself is **at** the threshold the text names. (LTX-2 Community License.)
    Inclusive,
}

impl CeilingBoundary {
    /// The stable serialized discriminator for this boundary.
    pub const fn tag(&self) -> &'static str {
        match self {
            Self::Exclusive => "exclusive",
            Self::Inclusive => "inclusive",
        }
    }
}

/// A typed obligation or condition a [`LicenseFamily`]'s text states, or — for
/// [`GatedAccess`](Self::GatedAccess) — a distribution fact a [`ComponentLicense`] records.
///
/// Deliberately **not** a permissive/restrictive flag. Each variant is a fact about the licence
/// text; whether a given use is permitted is the consumer's evaluation of the union of these
/// against its own situation.
///
/// ## Disclosure only
///
/// This vocabulary exists so a consumer can **show** a user what the upstream texts say. Nothing in
/// this crate blocks, gates, degrades, or withholds anything on the strength of a term, and nothing
/// added here ever should: a term describes, it does not decide. Doc comments on the variants
/// therefore report what a licence *states* and avoid asserting that a use is or is not allowed —
/// that conclusion belongs to the consumer, who alone knows its own revenue, whether it
/// redistributes weights, and which agreements it has with its own users.
///
/// ## A licence can name a thing without naming its address (sc-16662)
///
/// [`AcceptableUsePolicy::url`](Self::AcceptableUsePolicy::url) and
/// [`RegistrationRequired::contact`](Self::RegistrationRequired::contact) are `Option`, because the
/// texts in the catalog do this and a required address forces one of two dishonest outcomes:
/// invent a URL, or drop the fact that the licence points somewhere. Four of the sixteen families
/// read this way — CreativeML Open RAIL++-M, FLUX.1 \[dev\] and LTX-2 Community name a policy with
/// no published address, and the LTX-2 copy shipped beside the weights names a registration with
/// no address. `None` is the recorded fact "the text names this and gives no address", which is
/// what a consumer shows a user; it is not "no policy" and not "unknown".
///
/// [`license_table_conformance_errors`] rejects `Some("")` and `Some("   ")` for either field, so
/// the absence is spelled one way rather than two.
///
/// ## Serialized order is keyed on [`tag`](Self::tag), not on declaration order
///
/// A derived `Ord` would make the emitted order of a [`provider_terms`] union a function of where
/// each variant happens to sit in this `enum`, so inserting a future variant mid-enum would silently
/// reorder every `terms` array in the manifest and trip the committed-manifest drift gate for
/// reasons unrelated to any licence change. `Ord` is therefore deliberately **not** derived: the
/// union is ordered by a private `sort_key` — the stable string [`tag`](Self::tag), then the
/// variant's payload — so adding, removing, or reordering variants moves nothing that a consumer
/// already reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LicenseTerm {
    /// An attribution / copyright notice must be reproduced by the product.
    AttributionRequired,
    /// Distribution must carry a NOTICE file, and modified files must be marked.
    NoticeFileRequired,
    /// Redistribution of the **weights** is restricted to non-commercial use.
    NonCommercialWeights,
    /// Use of the **outputs** (renders) is restricted to non-commercial use. Strictly stronger than
    /// [`NonCommercialWeights`](Self::NonCommercialWeights) for a product that sells renders.
    NonCommercialOutputs,
    /// The licence names an annual-revenue threshold above which different terms apply.
    ///
    /// A union may carry more than one ceiling, and they are **not** collapsed to a minimum: each
    /// one is a separate disclosure a consumer surfaces, and two licences can name the same amount
    /// with different [`boundary`](Self::RevenueCeiling::boundary) readings, which is a real
    /// difference at exactly that amount.
    ///
    /// Thresholds that are not denominated in revenue have no typed variant — Llama 3.1
    /// Community's trigger is *700 million monthly active users* — and are transcribed verbatim as
    /// a [`DeployerObligation`](Self::DeployerObligation) instead; see that variant's note.
    RevenueCeiling {
        /// The amount the licence names, in whole US dollars.
        amount_usd: u64,
        /// Whether the licence's wording puts `amount_usd` itself below the threshold
        /// ([`Exclusive`](CeilingBoundary::Exclusive), "more than $N") or at it
        /// ([`Inclusive`](CeilingBoundary::Inclusive), "at least $N").
        boundary: CeilingBoundary,
    },
    /// The licence names an out-of-band registration or approval with the upstream.
    RegistrationRequired {
        /// Where the registration is made (an email address or URL), when the text names one.
        ///
        /// `None` records that the licence names the registration **without naming an address** —
        /// see the type-level note on addressless references. The LTX-2 Community License copy
        /// distributed beside the `Lightricks/LTX-2.3` weights reads *"required to contact
        /// Licensor."* full stop, while the copy in the licensor's GitHub repository links an
        /// address at the same sentence.
        contact: Option<&'static str>,
    },
    /// The licence names an acceptable-use / prohibited-use policy.
    AcceptableUsePolicy {
        /// Where the policy is published, when the text names an address.
        ///
        /// `None` records that the licence names a policy and **gives no address** — see the
        /// type-level note. Three families in the catalog read this way: CreativeML Open RAIL++-M
        /// (whose text contains no URL of any kind and enumerates its restrictions in its own
        /// Attachment A), FLUX.1 \[dev\] (which enumerates prohibited uses in §4 and never uses the
        /// phrase at all, while its model card cites a `POLICY.md` that is not published), and
        /// LTX-2 Community (which capitalises the term in Attachment A but never defines it and
        /// publishes no policy document).
        url: Option<&'static str>,
    },
    /// A concrete duty the licence puts on the deployer (e.g. content filtering on generated
    /// media), in the licence's own words.
    ///
    /// Also the landing spot for any condition no typed variant carries — most notably a threshold
    /// denominated in something other than revenue, such as Llama 3.1 Community's *"greater than
    /// 700 million monthly active users"*. A free-text quote is a complete disclosure precisely
    /// because nothing in this system computes against a term: the consumer reads the sentence,
    /// which is what it would have to do with a typed variant anyway.
    DeployerObligation {
        /// The duty, quoted or closely paraphrased from the licence.
        text: &'static str,
    },
    /// Downstream recipients must be handed **a copy of the licence text itself**.
    ///
    /// The lighter of the two flow-down shapes: Apache-2.0 §4(a), Stability §IV(a), FLUX.1 \[dev\]
    /// §3(a), Krea §3.1, CircleStone §3(a), the two NVIDIA licences, Apple MLR §2 and Llama 3.1
    /// §1(b)(i) all read this way. It carries the family whose text travels, because two licences
    /// imposing it are **two** duties — a distributor hands over two documents, not one — and a
    /// union that deduped them to a single element would show a user one obligation where the
    /// catalog carries several. Resolve [`family`](Self::DownstreamLicenseCopy::family) through
    /// [`resolve_family`] to reach the [`LicenseFamily::text_url`] the duty points at.
    DownstreamLicenseCopy {
        /// The [`LicenseFamily::id`] whose text must reach the downstream recipient. Always the id
        /// of the family declaring the term — [`license_table_conformance_errors`] rejects a term
        /// naming any other family, since a mis-transcribed id would point a consumer at the wrong
        /// text.
        family: &'static str,
    },
    /// The licence's own **use restrictions must be written into the deployer's agreement with its
    /// users** as enforceable provisions, and notice given to subsequent users.
    ///
    /// Structurally heavier than [`DownstreamLicenseCopy`](Self::DownstreamLicenseCopy), and
    /// materially different licence to licence: CreativeML Open RAIL++-M §III requires its
    /// paragraph 5; LTX-2 §3(a) requires its paragraph 4 *plus the whole of Attachment A*; Gemma
    /// §3.1 requires its §3.2, which incorporates an externally hosted, unilaterally updatable
    /// policy. The population bound differs too — OpenRAIL++ and LTX-2 count hosted-service/API
    /// consumers as recipients, while Gemma §3.1 carves Hosted Services out of its Notice
    /// requirement. None of that is interchangeable, so the term carries its family for the same
    /// reason the copy variant does.
    ///
    /// A constraint one licence adds and its neighbours do not — LTX-2 §3(b)'s
    /// "distributed exclusively under the terms of this Agreement" — is transcribed alongside as
    /// its own [`DeployerObligation`](Self::DeployerObligation) rather than folded in here.
    DownstreamRestrictions {
        /// The [`LicenseFamily::id`] whose restrictions must be reproduced. Same self-naming rule
        /// as [`DownstreamLicenseCopy::family`](Self::DownstreamLicenseCopy::family).
        family: &'static str,
    },
    /// The upstream distributes the checkpoint behind a gate — its terms have to be accepted before
    /// the artifact can be obtained at all.
    ///
    /// **A per-checkpoint distribution fact, not a clause in any licence text**, and therefore
    /// recorded on [`ComponentLicense::gated`] rather than in a [`LicenseFamily::terms`] list;
    /// [`license_table_conformance_errors`] rejects a family that declares it. Stability AI's
    /// Community License governs both `stable-video-diffusion-img2vid-xt` (ungated) and
    /// `stable-diffusion-3.5-large` (gated) — one text, two distribution settings — so a family
    /// carrying gating would have to be split in two that differ in no legal respect.
    ///
    /// [`provider_terms`] still raises it into the derived union, because a consumer needs to know a
    /// render touched a gated checkpoint.
    GatedAccess,
}

impl LicenseTerm {
    /// The stable serialized discriminator for this term.
    pub const fn tag(&self) -> &'static str {
        match self {
            Self::AttributionRequired => "attribution_required",
            Self::NoticeFileRequired => "notice_file_required",
            Self::NonCommercialWeights => "non_commercial_weights",
            Self::NonCommercialOutputs => "non_commercial_outputs",
            Self::RevenueCeiling { .. } => "revenue_ceiling",
            Self::RegistrationRequired { .. } => "registration_required",
            Self::AcceptableUsePolicy { .. } => "acceptable_use_policy",
            Self::DeployerObligation { .. } => "deployer_obligation",
            Self::DownstreamLicenseCopy { .. } => "downstream_license_copy",
            Self::DownstreamRestrictions { .. } => "downstream_restrictions",
            Self::GatedAccess => "gated_access",
        }
    }

    /// The [`LicenseFamily::id`] a flow-down term names — the route from a term in a
    /// [`provider_terms`] union back to the specific text it points at, via [`resolve_family`] and
    /// [`LicenseFamily::text_url`]. `None` for every other variant.
    pub const fn flow_down_family(&self) -> Option<&'static str> {
        match *self {
            Self::DownstreamLicenseCopy { family } | Self::DownstreamRestrictions { family } => {
                Some(family)
            }
            _ => None,
        }
    }

    /// The total order the serialized union is emitted in: the stable [`tag`](Self::tag) first, then
    /// the variant's payload (numeric ceilings ascending, then string payloads lexicographic). Two
    /// distinct terms never share a key — every field that distinguishes two values of a variant
    /// appears in the key — so sorting by it is deterministic without depending on the declaration
    /// order of the `enum`; see the type-level note.
    ///
    /// The middle slot carries an optional payload's **presence** for the two `Option` fields, not
    /// only its contents: `None` and `Some("")` are different disclosures ("the text names no
    /// address" versus a malformed row), and flattening both to `""` would let
    /// [`provider_terms`]'s dedup drop one of them.
    fn sort_key(&self) -> (&'static str, u64, &'static str) {
        match *self {
            Self::RevenueCeiling {
                amount_usd,
                boundary,
            } => (self.tag(), amount_usd, boundary.tag()),
            Self::RegistrationRequired { contact } => {
                (self.tag(), contact.is_some() as u64, contact.unwrap_or(""))
            }
            Self::AcceptableUsePolicy { url } => {
                (self.tag(), url.is_some() as u64, url.unwrap_or(""))
            }
            Self::DeployerObligation { text } => (self.tag(), 0, text),
            Self::DownstreamLicenseCopy { family } | Self::DownstreamRestrictions { family } => {
                (self.tag(), 0, family)
            }
            Self::AttributionRequired
            | Self::NoticeFileRequired
            | Self::NonCommercialWeights
            | Self::NonCommercialOutputs
            | Self::GatedAccess => (self.tag(), 0, ""),
        }
    }

    fn to_json(self) -> serde_json::Value {
        let mut value = serde_json::json!({ "term": self.tag() });
        let object = value
            .as_object_mut()
            .expect("term is serialized as an object");
        match self {
            Self::RevenueCeiling {
                amount_usd,
                boundary,
            } => {
                object.insert("amount_usd".into(), amount_usd.into());
                object.insert("boundary".into(), boundary.tag().into());
            }
            Self::DownstreamLicenseCopy { family } | Self::DownstreamRestrictions { family } => {
                object.insert("family".into(), family.into());
            }
            // `None` is emitted as an explicit JSON `null` rather than an absent key: the fact
            // being disclosed is "the licence names this and gives no address", and a key that
            // simply vanished would read to a consumer as "no such term was recorded".
            Self::RegistrationRequired { contact } => {
                object.insert("contact".into(), serde_json::json!(contact));
            }
            Self::AcceptableUsePolicy { url } => {
                object.insert("url".into(), serde_json::json!(url));
            }
            Self::DeployerObligation { text } => {
                object.insert("text".into(), text.into());
            }
            // Listed rather than a wildcard, deliberately: this is the only layer a consumer reads,
            // so a new payload-carrying variant falling through here would serialize as a bare
            // `{"term": …}` and collapse into another term inside one `terms` array — silently, and
            // after `sort_key` had already separated them. Adding a variant must break this match.
            Self::AttributionRequired
            | Self::NoticeFileRequired
            | Self::NonCommercialWeights
            | Self::NonCommercialOutputs
            | Self::GatedAccess => {}
        }
        value
    }
}

/// A licence **family** — one upstream licence text, reviewed once by a human.
///
/// Roughly fourteen families cover the whole catalog. Normalizing checkpoints onto families is what
/// keeps the surface maintainable: adding a model is a transcription of which family it declares
/// (mechanical, and verifiable against a quote), not another licence read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LicenseFamily {
    /// Stable key referenced by [`ComponentLicense::family`], e.g. `"flux-1-dev-non-commercial"`.
    pub id: &'static str,
    /// SPDX identifier, or a `LicenseRef-…` id where the licence has no SPDX entry.
    pub spdx_id: &'static str,
    /// Human-readable licence name.
    pub name: &'static str,
    /// The canonical licence text this family was read from.
    pub text_url: &'static str,
    /// The typed conditions this licence's **text** states.
    ///
    /// Facts about the text only. A fact about how one upstream chose to *distribute* a particular
    /// checkpoint belongs on the [`ComponentLicense`] row instead — that is why
    /// [`LicenseTerm::GatedAccess`] is rejected here by
    /// [`license_table_conformance_errors`] and lives on [`ComponentLicense::gated`]. Mixing the two
    /// axes forces one licence text into two families that differ in no legal respect, and the
    /// reviewed unit is the family.
    pub terms: &'static [LicenseTerm],
}

impl LicenseFamily {
    /// Whether this family imposes `term`.
    pub fn imposes(&self, term: LicenseTerm) -> bool {
        self.terms.contains(&term)
    }

    /// Whether this family requires attribution (the invariant [`ComponentLicense`] rows are
    /// checked against).
    pub fn requires_attribution(&self) -> bool {
        self.imposes(LicenseTerm::AttributionRequired)
    }
}

/// One **loaded artifact** and the licence it declares — the fact layer.
///
/// The unit that carries a licence is a checkpoint, not a provider id: `boogu_image` is a Boogu DiT
/// plus a Qwen3-VL-8B encoder plus a FLUX.1 VAE, three licences with different terms. Keying rows
/// by component also lets the surface represent artifacts that register no provider id at all —
/// the face / depth / segmentation / decoder stacks an identity or PiD render pulls in, which is
/// where the strictest terms in the catalog live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentLicense {
    /// Stable component key, e.g. `"arcface_antelopev2"`. Unique across the table.
    pub component: &'static str,
    /// **The document [`declared`](Self::declared) was transcribed from** — the one place a drift
    /// job can re-read it.
    ///
    /// The rule is re-readability, not provenance-in-general: this is not "the upstream project",
    /// not "where we downloaded the weights", and not "the licence text" (that lives once on the
    /// family, as [`LicenseFamily::text_url`]). Pick whichever document actually carries the string
    /// in [`declared`](Self::declared) for *this artifact*, so that fetching this URL and comparing
    /// is a mechanical check. In practice that resolves to one of two shapes, and which one applies
    /// is decided by where the declaration lives rather than by taste:
    ///
    /// * **A model-card / repository URL**, when the declaration is the card's own licence tag —
    ///   `"apache-2.0"` on `openai/whisper-base`, `"stable-audio-community"` on
    ///   `stabilityai/stable-audio-3-small-music`. For an artifact *bundled inside* another party's
    ///   repository this is the **upstream** card, not the redistributor's: the redistributor's tag
    ///   declares the bundle's licence, not the bundled component's, so ACE-Step's bundled
    ///   `Qwen3-Embedding-0.6B` row points at `Qwen/Qwen3-Embedding-0.6B`, where `"apache-2.0"` is
    ///   the value actually published.
    /// * **A blob URL for a licence file shipped beside the weights**, when the artifact's
    ///   declaration exists nowhere else — Stable Audio 3's bundled T5Gemma is declared only by the
    ///   `LICENSE_GEMMA.md` committed next to the checkpoint, whose title *is* the
    ///   `"Gemma Terms of Use"` string in [`declared`](Self::declared). The repository's own tag
    ///   says `"stable-audio-community"` and would be the wrong document for that row.
    ///
    /// Prefer the artifact's **current** location, unpinned, because a re-read that cannot observe a
    /// change cannot detect one — that is the entire purpose of the field. The blob shape above is
    /// the exception the rule tolerates rather than a second convention: a revision-pinned URL still
    /// makes the transcription auditable (the quote check), but it is frozen, so re-licensing after
    /// [`retrieved`](Self::retrieved) is invisible to it. A gate that wants both properties for such
    /// a row has to fetch the unpinned path itself; nothing in this table can supply it.
    pub source_url: &'static str,
    /// Whether the upstream distributes **this checkpoint** behind an access gate, as observed at
    /// [`retrieved`](Self::retrieved).
    ///
    /// A property of the artifact, like [`source_url`](Self::source_url) — not of its licence.
    /// Stability AI's Community License governs `stable-video-diffusion-img2vid-xt` (ungated) and
    /// `stable-diffusion-3.5-large` (gated) alike, so recording gating on the family would split
    /// one reviewed text into two rows differing in no legal respect. [`provider_terms`] raises a
    /// `true` here into the derived union as [`LicenseTerm::GatedAccess`], because a consumer needs
    /// to know a render touched a gated checkpoint.
    ///
    /// A plain `bool`: *whether*, not *how*. Hugging Face additionally distinguishes click-through
    /// (`auto`) from human approval (`manual`); if a consumer ever needs to show which, that is a
    /// second field rather than an overload of this one.
    pub gated: bool,
    /// The licence identifier **as declared upstream**, verbatim (e.g. `"apache-2.0"`,
    /// `"flux-1-dev-non-commercial-license"`). The drift gate re-reads `source_url` and compares
    /// against this string, so it must be transcribed rather than normalized.
    pub declared: &'static str,
    /// The [`LicenseFamily::id`] this declaration normalizes to.
    pub family: &'static str,
    /// The attribution the family requires. Mandatory when the family sets
    /// [`LicenseTerm::AttributionRequired`].
    pub attribution: Option<&'static str>,
    /// ISO `YYYY-MM-DD` date `declared` was read from `source_url`. Without an as-of date a
    /// re-licensed upstream rots the table invisibly.
    pub retrieved: &'static str,
}

impl ComponentLicense {
    /// Whether this row is well-formed against `families`: the identity fields
    /// ([`component`](Self::component), [`source_url`](Self::source_url),
    /// [`declared`](Self::declared)) are non-blank, [`family`](Self::family) resolves to a known
    /// [`LicenseFamily`], [`retrieved`](Self::retrieved) parses as a real ISO `YYYY-MM-DD` calendar
    /// date, [`attribution`](Self::attribution) is not `Some("")` or `Some("   ")`, and a family
    /// imposing [`LicenseTerm::AttributionRequired`] implies a **non-blank**
    /// [`attribution`](Self::attribution).
    ///
    /// "Non-blank" throughout means neither empty nor whitespace-only: a placeholder that renders as
    /// nothing on a licences page discharges no obligation, so it is rejected exactly like an
    /// absent value.
    ///
    /// Row-local counterpart to [`license_table_conformance_errors`], which runs exactly these
    /// checks plus the table-level ones (family-id and component-key uniqueness, provider-id
    /// uniqueness, provider→component resolution) and reports *why* rather than just *whether*.
    /// Prefer the table function at a catalog boundary; this is for checking a single row in
    /// isolation.
    pub fn is_well_formed(&self, families: &[LicenseFamily]) -> bool {
        self.row_errors(families).is_empty()
    }

    /// The row-local conformance failures, as human-readable messages. Single definition of the
    /// per-row rules, shared by [`is_well_formed`](Self::is_well_formed) and
    /// [`license_table_conformance_errors`] so the predicate and the reporter cannot drift.
    fn row_errors(&self, families: &[LicenseFamily]) -> Vec<String> {
        let key = self.component;
        let mut errors = Vec::new();
        if is_blank(self.source_url) {
            errors.push(format!("component {key:?} has no source_url"));
        }
        if is_blank(self.declared) {
            errors.push(format!(
                "component {key:?} has no declared licence identifier"
            ));
        }
        if !is_iso_date(self.retrieved) {
            errors.push(format!(
                "component {key:?} has a non-ISO retrieved date {:?}",
                self.retrieved
            ));
        }
        // An `attribution: Some("")` — or `Some("   ")` — is not an attribution: it satisfies
        // nothing, and treating it as present would let a CC-BY-* obligation ship unrecorded behind
        // a placeholder.
        let attribution = self.attribution.filter(|text| !is_blank(text));
        if self.attribution.is_some() && attribution.is_none() {
            errors.push(format!(
                "component {key:?} records an empty attribution string"
            ));
        }
        match resolve_family(families, self.family) {
            None => errors.push(format!(
                "component {key:?} references unknown licence family {:?}",
                self.family
            )),
            Some(family) => {
                if family.requires_attribution() && attribution.is_none() {
                    errors.push(format!(
                        "component {key:?} resolves to {:?}, which requires attribution, but \
                         records none",
                        family.id
                    ));
                }
            }
        }
        if is_blank(key) {
            // Reported by the table checker, which sees the key before anything else; keep the
            // row-local predicate honest about it too.
            errors.push("component row has an empty component key".to_string());
        }
        errors
    }
}

/// The components a registered provider id loads. The per-backend part of the surface — the two
/// media catalogs ship different id sets, but a component row itself is backend-independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderComponents {
    /// The registry id, matching the provider descriptor's `id`.
    pub provider_id: &'static str,
    /// Component keys resolving into the [`ComponentLicense`] table.
    pub components: &'static [&'static str],
}

/// Whether `value` carries no information — empty, **or nothing but whitespace**.
///
/// Every identity check in this module goes through here rather than through `str::is_empty`, so the
/// gate cannot be satisfied by a placeholder. `"   "` passes `!is_empty()` while rendering as nothing
/// on a licences page, which is the same hole `Some("")` opened, one space wider; a single predicate
/// keeps a future field from re-opening it. The reported messages say "empty" for both cases — the
/// distinction is not one a table author needs to act on differently.
fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

/// Whether `value` is an ISO `YYYY-MM-DD` **calendar** date. Dependency-free on purpose — the
/// contracts crate takes no date dependency for one field.
///
/// Calendar-accurate, not merely range-checked: month lengths and the Gregorian leap rule are
/// applied, so `2026-02-31`, `2026-04-31` and `2026-02-29` are rejected while `2024-02-29` is
/// accepted, and year `0000` is rejected. That matters because `retrieved` is a hand-transcribed
/// provenance stamp that downstream tooling parses — Python's `date.fromisoformat` *raises* on
/// `2026-02-31`, so a date this gate blessed but the calendar rejects would surface as a stack trace
/// in the sc-16664 validator or the sc-16670 drift job instead of as a message here.
fn is_iso_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    if bytes
        .iter()
        .enumerate()
        .any(|(i, b)| !matches!(i, 4 | 7) && !b.is_ascii_digit())
    {
        return false;
    }
    let number = |from: usize, to: usize| value[from..to].parse::<u32>().unwrap_or(0);
    let (year, month, day) = (number(0, 4), number(5, 7), number(8, 10));
    if year == 0 || !(1..=12).contains(&month) || day == 0 {
        return false;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if leap => 29,
        _ => 28,
    };
    day <= days_in_month
}

/// Look up a family by id — the **single** family-resolution path, so the id-uniqueness invariant has
/// one enforcement point.
///
/// Returns the first match in input order. That is only unambiguous because
/// [`license_table_conformance_errors`] rejects a table carrying two families with the same id: a
/// shadowed family would otherwise make an entire set of obligations vanish from the
/// [`provider_terms`] union depending on which order the catalogs were concatenated in. Resolve
/// through this function rather than open-coding a scan, so that guarantee cannot be bypassed.
pub fn resolve_family<'a>(families: &'a [LicenseFamily], id: &str) -> Option<&'a LicenseFamily> {
    families.iter().find(|family| family.id == id)
}

/// Look up a component row by key — the **single** component-resolution path, mirroring
/// [`resolve_family`]. Component-key uniqueness is likewise enforced by
/// [`license_table_conformance_errors`].
pub fn resolve_component<'a>(
    components: &'a [ComponentLicense],
    key: &str,
) -> Option<&'a ComponentLicense> {
    components.iter().find(|row| row.component == key)
}

/// The union of every term the components `provider` loads bring with them — sorted and
/// deduplicated, so it is deterministic and comparable.
///
/// **Derived, never hand-authored.** This is the "effective terms" answer a consumer joins over,
/// and computing it from the component rows is what keeps it from drifting away from them.
/// A component key that does not resolve contributes nothing; use
/// [`license_table_conformance_errors`] to reject that state at the catalog boundary rather than
/// silently under-reporting here.
///
/// Two sources feed the union:
///
/// * every term in the [`LicenseFamily::terms`] each resolved row normalizes to, and
/// * [`LicenseTerm::GatedAccess`] for each resolved row whose [`ComponentLicense::gated`] is set —
///   a distribution fact that lives on the checkpoint, not in any licence text, but that a consumer
///   still has to be able to show.
///
/// A gated row contributes its `GatedAccess` even if its `family` fails to resolve: gating is not a
/// property of the family, so losing it to an unrelated table defect would under-report.
///
/// Deduplication is by value, so two flow-down duties naming different families stay two elements
/// and two ceilings at the same amount with different [`CeilingBoundary`] readings stay two
/// elements — a union that collapsed either would show one obligation where the catalog carries
/// several.
///
/// Ordering is by [`LicenseTerm::tag`] then payload, **not** by variant declaration order, so a
/// future variant inserted mid-`enum` cannot reorder an already-committed manifest.
pub fn provider_terms(
    provider: &ProviderComponents,
    components: &[ComponentLicense],
    families: &[LicenseFamily],
) -> Vec<LicenseTerm> {
    let mut terms: Vec<LicenseTerm> = Vec::new();
    for row in provider
        .components
        .iter()
        .filter_map(|key| resolve_component(components, key))
    {
        if row.gated {
            terms.push(LicenseTerm::GatedAccess);
        }
        if let Some(family) = resolve_family(families, row.family) {
            terms.extend(family.terms.iter().copied());
        }
    }
    terms.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
    terms.dedup();
    terms
}

/// Every way the licence table can be malformed, as human-readable messages — the catalog ship-gate
/// asserts this is empty so no registered provider escapes a resolved, well-formed licence.
///
/// Checks, per section:
///
/// * **Families** — ids are non-blank and unique (a shadowed family would make an entire set of
///   obligations vanish from the [`provider_terms`] union depending on input order, since
///   [`resolve_family`] takes the first match); `spdx_id`, `name` and `text_url` are populated; the
///   `terms` list states only facts about the *text*, so [`LicenseTerm::GatedAccess`] — a
///   per-checkpoint distribution setting — is rejected there and belongs on
///   [`ComponentLicense::gated`]; and a flow-down term names its own family, so the text a consumer
///   resolves through it is the text that actually imposed the duty.
/// * **Components** — keys are non-blank and unique; `source_url` and `declared` are populated;
///   every `family` resolves; `retrieved` is a real ISO calendar date; an attribution-requiring
///   family implies a non-blank attribution, and `Some("")` never counts as one.
/// * **Providers** — `provider_id` is non-blank and unique (duplicates are not byte-stable under
///   the manifest's stable sort); every provider maps to at least one component; a provider does
///   not list the same component twice; every referenced component exists.
///
/// Every "non-blank" above rejects whitespace-only as well as empty: `"   "` satisfies `!is_empty()`
/// but is a placeholder, not a value, and would otherwise pass the gate carrying no information.
///
/// Uniqueness is checked here rather than at the lookup sites deliberately: the resolvers stay
/// total-order-free single scans, and this is the one boundary that has the whole table in hand.
pub fn license_table_conformance_errors(
    families: &[LicenseFamily],
    components: &[ComponentLicense],
    providers: &[ProviderComponents],
) -> Vec<String> {
    let mut errors = Vec::new();

    let mut seen_families: Vec<&str> = Vec::new();
    for family in families {
        let id = family.id;
        if is_blank(id) {
            errors.push("licence family has an empty id".to_string());
        } else if seen_families.contains(&id) {
            errors.push(format!("duplicate licence family {id:?}"));
        } else {
            seen_families.push(id);
        }
        if is_blank(family.spdx_id) {
            errors.push(format!("licence family {id:?} has no spdx_id"));
        }
        if is_blank(family.name) {
            errors.push(format!("licence family {id:?} has no name"));
        }
        if is_blank(family.text_url) {
            errors.push(format!("licence family {id:?} has no text_url"));
        }
        for term in family.terms {
            if matches!(term, LicenseTerm::GatedAccess) {
                errors.push(format!(
                    "licence family {id:?} declares gated_access, which is a per-checkpoint \
                     distribution fact recorded on ComponentLicense::gated, not a licence term"
                ));
            }
            if let Some(named) = term.flow_down_family() {
                if named != id {
                    errors.push(format!(
                        "licence family {id:?} declares a {} term naming {named:?}; a flow-down \
                         term must name its own family so the text it points at is the text that \
                         imposed the duty",
                        term.tag()
                    ));
                }
            }
            // "The text names no address" is spelled `None`. A blank `Some` renders as nothing on a
            // licences page while reading as an address in the JSON, so it is neither disclosure.
            let blank_address = match term {
                LicenseTerm::AcceptableUsePolicy { url } => url.map(is_blank),
                LicenseTerm::RegistrationRequired { contact } => contact.map(is_blank),
                _ => None,
            };
            if blank_address == Some(true) {
                errors.push(format!(
                    "licence family {id:?} declares a {} term with an empty address; record a \
                     licence that names no address as None, not as an empty string",
                    term.tag()
                ));
            }
        }
    }

    let mut seen: Vec<&str> = Vec::new();
    for row in components {
        let key = row.component;
        if is_blank(key) {
            errors.push("component row has an empty component key".to_string());
            continue;
        }
        if seen.contains(&key) {
            errors.push(format!("duplicate component row {key:?}"));
        }
        seen.push(key);

        errors.extend(row.row_errors(families));
    }

    let mut seen_providers: Vec<&str> = Vec::new();
    for provider in providers {
        let id = provider.provider_id;
        if is_blank(id) {
            errors.push("provider row has an empty provider_id".to_string());
        } else if seen_providers.contains(&id) {
            errors.push(format!("duplicate provider row {id:?}"));
        } else {
            seen_providers.push(id);
        }
        if provider.components.is_empty() {
            errors.push(format!("provider {id:?} maps to no components"));
        }
        let mut seen_keys: Vec<&str> = Vec::new();
        for key in provider.components {
            if seen_keys.contains(key) {
                errors.push(format!("provider {id:?} lists component {key:?} twice"));
            } else {
                seen_keys.push(key);
            }
            if resolve_component(components, key).is_none() {
                errors.push(format!(
                    "provider {id:?} references unknown component {key:?}"
                ));
            }
        }
    }

    errors
}

/// Serialize the licence table into the canonical **model-licenses manifest** JSON at
/// `schema_version` 3 — the file the release tooling emits beside the SPDX SBOM.
///
/// Three fact sections (`families`, `components`, `providers`) plus, on each provider, the
/// **derived** term union. Output is deterministic: every section is sorted by its key, so the
/// committed manifest and the catalog-generated value compare byte-for-byte regardless of
/// registration order. A trailing newline matches `write_json`'s convention in the release tooling.
pub fn component_licenses_manifest_json(
    families: &[LicenseFamily],
    components: &[ComponentLicense],
    providers: &[ProviderComponents],
) -> String {
    let mut sorted_families: Vec<&LicenseFamily> = families.iter().collect();
    sorted_families.sort_by(|a, b| a.id.cmp(b.id));
    let families_json: Vec<serde_json::Value> = sorted_families
        .iter()
        .map(|family| {
            serde_json::json!({
                "id": family.id,
                "spdx_id": family.spdx_id,
                "name": family.name,
                "text_url": family.text_url,
                "terms": family.terms.iter().map(|t| t.to_json()).collect::<Vec<_>>(),
            })
        })
        .collect();

    let mut sorted_components: Vec<&ComponentLicense> = components.iter().collect();
    sorted_components.sort_by(|a, b| a.component.cmp(b.component));
    let components_json: Vec<serde_json::Value> = sorted_components
        .iter()
        .map(|row| {
            serde_json::json!({
                "component": row.component,
                "source_url": row.source_url,
                "gated": row.gated,
                "declared": row.declared,
                "family": row.family,
                "attribution": row.attribution,
                "retrieved": row.retrieved,
            })
        })
        .collect();

    let mut sorted_providers: Vec<&ProviderComponents> = providers.iter().collect();
    sorted_providers.sort_by(|a, b| a.provider_id.cmp(b.provider_id));
    let providers_json: Vec<serde_json::Value> = sorted_providers
        .iter()
        .map(|provider| {
            let mut keys: Vec<&str> = provider.components.to_vec();
            keys.sort_unstable();
            serde_json::json!({
                "provider_id": provider.provider_id,
                "components": keys,
                "terms": provider_terms(provider, components, families)
                    .into_iter()
                    .map(|t| t.to_json())
                    .collect::<Vec<_>>(),
            })
        })
        .collect();

    let document = serde_json::json!({
        "schema_version": 3,
        "kind": "model-weight-licenses",
        "families": families_json,
        "components": components_json,
        "providers": providers_json,
    });
    let mut rendered = serde_json::to_string_pretty(&document)
        .expect("weight-license manifest is always serializable");
    rendered.push('\n');
    rendered
}

#[cfg(test)]
mod v3_tests {
    use super::*;

    // A miniature of the real table: a permissive DiT, a permissive encoder, and a research-only
    // face model whose terms are the strictest in the set.
    const FAMILIES: &[LicenseFamily] = &[
        LicenseFamily {
            id: "apache-2-0",
            spdx_id: "Apache-2.0",
            name: "Apache License 2.0",
            text_url: "https://www.apache.org/licenses/LICENSE-2.0",
            terms: &[
                LicenseTerm::AttributionRequired,
                LicenseTerm::NoticeFileRequired,
            ],
        },
        LicenseFamily {
            id: "flux-1-dev-non-commercial",
            spdx_id: "LicenseRef-FLUX-1-dev-NC",
            name: "FLUX.1 [dev] Non-Commercial License",
            text_url: "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
            // Weights are non-commercial; OUTPUTS are explicitly commercially usable, so
            // NonCommercialOutputs is deliberately absent. Gating is NOT here — the checkpoint is
            // gated upstream, which the `flux1_dev_dit` row records; see A2 below.
            terms: &[
                LicenseTerm::NonCommercialWeights,
                LicenseTerm::DownstreamLicenseCopy {
                    family: "flux-1-dev-non-commercial",
                },
                LicenseTerm::AcceptableUsePolicy {
                    url: Some("https://example.invalid/fixture-aup"),
                },
            ],
        },
        LicenseFamily {
            id: "insightface-research-only",
            spdx_id: "LicenseRef-InsightFace-NC",
            name: "InsightFace non-commercial research use only",
            text_url: "https://github.com/deepinsight/insightface/tree/master/model_zoo",
            terms: &[
                LicenseTerm::NonCommercialWeights,
                LicenseTerm::NonCommercialOutputs,
            ],
        },
    ];

    const COMPONENTS: &[ComponentLicense] = &[
        ComponentLicense {
            component: "flux1_dev_dit",
            source_url: "https://huggingface.co/black-forest-labs/FLUX.1-dev",
            gated: true,
            declared: "flux-1-dev-non-commercial-license",
            family: "flux-1-dev-non-commercial",
            attribution: None,
            retrieved: "2026-08-01",
        },
        ComponentLicense {
            component: "t5_xxl",
            source_url: "https://huggingface.co/google/t5-v1_1-xxl",
            gated: false,
            declared: "apache-2.0",
            family: "apache-2-0",
            attribution: Some("T5 v1.1 © Google — Apache-2.0"),
            retrieved: "2026-08-01",
        },
        ComponentLicense {
            component: "arcface_antelopev2",
            source_url: "https://github.com/deepinsight/insightface/tree/master/model_zoo",
            gated: false,
            declared: "non-commercial research purposes only",
            family: "insightface-research-only",
            attribution: None,
            retrieved: "2026-08-01",
        },
    ];

    const PLAIN_FLUX: ProviderComponents = ProviderComponents {
        provider_id: "flux1_dev",
        components: &["flux1_dev_dit", "t5_xxl"],
    };
    const IDENTITY_FLUX: ProviderComponents = ProviderComponents {
        provider_id: "pulid_flux",
        components: &["flux1_dev_dit", "t5_xxl", "arcface_antelopev2"],
    };

    #[test]
    fn table_is_conformant() {
        assert_eq!(
            license_table_conformance_errors(FAMILIES, COMPONENTS, &[PLAIN_FLUX, IDENTITY_FLUX]),
            Vec::<String>::new()
        );
    }

    /// The strictest term comes from a component that is not the headline model: the identity
    /// render's output restriction is contributed by the face stack, not by the DiT. Deriving the
    /// provider view is what surfaces it — a hand-authored composite is exactly where this is
    /// missed.
    #[test]
    fn strictest_term_is_derived_from_a_non_obvious_component() {
        let plain = provider_terms(&PLAIN_FLUX, COMPONENTS, FAMILIES);
        let identity = provider_terms(&IDENTITY_FLUX, COMPONENTS, FAMILIES);

        // The DiT restricts weights, so both carry that.
        assert!(plain.contains(&LicenseTerm::NonCommercialWeights));
        assert!(identity.contains(&LicenseTerm::NonCommercialWeights));

        // Only the identity route restricts OUTPUTS, and it does so via the face model.
        assert!(
            !plain.contains(&LicenseTerm::NonCommercialOutputs),
            "a plain FLUX render must not inherit an output restriction"
        );
        assert!(
            identity.contains(&LicenseTerm::NonCommercialOutputs),
            "the identity route must inherit the face stack's output restriction"
        );

        // Two renders over the same base model have materially different terms — the join the
        // downstream product has to be able to compute.
        assert_ne!(plain, identity);
    }

    /// FLUX.1-dev's weights-restricted / outputs-permitted split is the case a `commercial_use`
    /// boolean cannot express. It must survive a manifest round trip intact.
    #[test]
    fn weights_nc_outputs_ok_split_survives_a_round_trip() {
        let json =
            component_licenses_manifest_json(FAMILIES, COMPONENTS, &[PLAIN_FLUX, IDENTITY_FLUX]);
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        let flux = value["providers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["provider_id"] == "flux1_dev")
            .unwrap();
        let tags: Vec<&str> = flux["terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["term"].as_str().unwrap())
            .collect();

        assert!(tags.contains(&"non_commercial_weights"));
        assert!(
            !tags.contains(&"non_commercial_outputs"),
            "FLUX.1-dev outputs are commercially usable; the split must not collapse"
        );
        // And no boolean conclusion is emitted anywhere.
        assert!(!json.contains("commercial_use"));
    }

    #[test]
    fn parameterized_terms_round_trip_their_payload() {
        const STABILITY: &[LicenseFamily] = &[LicenseFamily {
            id: "stability-ai-community",
            spdx_id: "LicenseRef-Stability-Community",
            name: "Stability AI Community License",
            text_url: "https://stability.ai/license",
            terms: &[
                LicenseTerm::RevenueCeiling {
                    amount_usd: 1_000_000,
                    boundary: CeilingBoundary::Exclusive,
                },
                LicenseTerm::DownstreamLicenseCopy {
                    family: "stability-ai-community",
                },
            ],
        }];
        const ROWS: &[ComponentLicense] = &[ComponentLicense {
            component: "sd3_5_large_dit",
            source_url: "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
            gated: true,
            declared: "stabilityai-ai-community",
            family: "stability-ai-community",
            attribution: None,
            retrieved: "2026-08-01",
        }];
        const PROVIDER: ProviderComponents = ProviderComponents {
            provider_id: "sd3_5_large",
            components: &["sd3_5_large_dit"],
        };

        let value: serde_json::Value = serde_json::from_str(&component_licenses_manifest_json(
            STABILITY,
            ROWS,
            &[PROVIDER],
        ))
        .unwrap();
        let terms = value["providers"][0]["terms"].as_array().unwrap();

        // Tag order: downstream_license_copy, gated_access, revenue_ceiling.
        let flow_down = &terms[0];
        assert_eq!(flow_down["term"], "downstream_license_copy");
        assert_eq!(flow_down["family"], "stability-ai-community");
        // The family the term names resolves back to the text that imposed the duty.
        assert_eq!(
            resolve_family(STABILITY, flow_down["family"].as_str().unwrap())
                .unwrap()
                .text_url,
            "https://stability.ai/license"
        );

        // Gating rides the component row, not the family, and still reaches the union.
        assert_eq!(terms[1]["term"], "gated_access");
        assert!(!STABILITY[0].imposes(LicenseTerm::GatedAccess));

        let ceiling = &terms[2];
        assert_eq!(ceiling["term"], "revenue_ceiling");
        assert_eq!(ceiling["amount_usd"], 1_000_000);
        assert_eq!(ceiling["boundary"], "exclusive");
    }

    /// Byte-stability under permutation of **every** input axis, not just the provider slice:
    /// `FAMILIES` (which arrives pre-sorted by id, so permuting it is the only thing that exercises
    /// the families sort at all), `COMPONENTS`, the provider slice, and the inner
    /// `provider.components` list. sc-16664 concatenates three catalog manifests, so every one of
    /// these arrives in an order nobody controls.
    #[test]
    fn manifest_is_deterministic_across_input_order() {
        const PLAIN_FLUX_PERMUTED: ProviderComponents = ProviderComponents {
            provider_id: "flux1_dev",
            components: &["t5_xxl", "flux1_dev_dit"],
        };
        const IDENTITY_FLUX_PERMUTED: ProviderComponents = ProviderComponents {
            provider_id: "pulid_flux",
            components: &["arcface_antelopev2", "t5_xxl", "flux1_dev_dit"],
        };

        let forward =
            component_licenses_manifest_json(FAMILIES, COMPONENTS, &[PLAIN_FLUX, IDENTITY_FLUX]);

        let permuted_families: Vec<LicenseFamily> = FAMILIES.iter().rev().copied().collect();
        let permuted_components: Vec<ComponentLicense> = COMPONENTS.iter().rev().copied().collect();
        let reversed = component_licenses_manifest_json(
            &permuted_families,
            &permuted_components,
            &[IDENTITY_FLUX_PERMUTED, PLAIN_FLUX_PERMUTED],
        );
        assert_eq!(
            forward, reversed,
            "permuting families, components, providers and a provider's own component list must \
             not change one byte of the manifest"
        );
        assert!(forward.ends_with("}\n"));

        let value: serde_json::Value = serde_json::from_str(&forward).unwrap();
        assert_eq!(value["schema_version"], 3);
        assert_eq!(value["kind"], "model-weight-licenses");
        // Providers are sorted by id: flux1_dev precedes pulid_flux.
        assert_eq!(value["providers"][0]["provider_id"], "flux1_dev");
        assert_eq!(value["providers"][1]["provider_id"], "pulid_flux");
        // Families are sorted by id, and the permuted run above proves the sort does the work.
        let family_ids: Vec<&str> = value["families"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f["id"].as_str().unwrap())
            .collect();
        let mut expected = family_ids.clone();
        expected.sort_unstable();
        assert_eq!(family_ids, expected);
        // Components are sorted by key.
        let component_keys: Vec<&str> = value["components"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["component"].as_str().unwrap())
            .collect();
        let mut expected_keys = component_keys.clone();
        expected_keys.sort_unstable();
        assert_eq!(component_keys, expected_keys);
    }

    /// A family shadowed by a second row with the same id silently changes which obligations a
    /// provider inherits, because [`resolve_family`] takes the first match. The gate must reject the
    /// table rather than let the join answer differently depending on concatenation order — the
    /// exact shape sc-16662 (hand-authored families) and sc-16664 (three merged catalogs) hit.
    #[test]
    fn duplicate_family_id_is_an_error() {
        const SHADOWED: &[LicenseFamily] = &[
            LicenseFamily {
                id: "ambiguous",
                spdx_id: "LicenseRef-A",
                name: "A",
                text_url: "https://example.invalid/a",
                terms: &[LicenseTerm::NonCommercialOutputs],
            },
            LicenseFamily {
                id: "ambiguous",
                spdx_id: "LicenseRef-B",
                name: "B",
                text_url: "https://example.invalid/b",
                terms: &[LicenseTerm::NonCommercialWeights],
            },
        ];
        const ROWS: &[ComponentLicense] = &[ComponentLicense {
            component: "shadowed",
            source_url: "https://example.invalid/model",
            gated: false,
            declared: "ambiguous",
            family: "ambiguous",
            attribution: None,
            retrieved: "2026-08-01",
        }];
        const PROVIDER: ProviderComponents = ProviderComponents {
            provider_id: "shadowy",
            components: &["shadowed"],
        };

        let errors = license_table_conformance_errors(SHADOWED, ROWS, &[PROVIDER]);
        assert!(
            errors
                .iter()
                .any(|e| e.contains("duplicate licence family")),
            "{errors:?}"
        );

        // And this is why: the derived union — and therefore the manifest — depends on which of the
        // two rows came first.
        let reversed: Vec<LicenseFamily> = SHADOWED.iter().rev().copied().collect();
        assert_ne!(
            provider_terms(&PROVIDER, ROWS, SHADOWED),
            provider_terms(&PROVIDER, ROWS, &reversed)
        );
        assert_ne!(
            component_licenses_manifest_json(SHADOWED, ROWS, &[PROVIDER]),
            component_licenses_manifest_json(&reversed, ROWS, &[PROVIDER])
        );
    }

    #[test]
    fn empty_family_identity_fields_are_errors() {
        let good = FAMILIES[0];
        let cases: [(&str, LicenseFamily, &str); 4] = [
            ("id", LicenseFamily { id: "", ..good }, "empty id"),
            (
                "spdx_id",
                LicenseFamily {
                    spdx_id: "",
                    ..good
                },
                "no spdx_id",
            ),
            ("name", LicenseFamily { name: "", ..good }, "no name"),
            (
                "text_url",
                LicenseFamily {
                    text_url: "",
                    ..good
                },
                "no text_url",
            ),
        ];
        for (label, family, needle) in cases {
            let errors = license_table_conformance_errors(&[family], &[], &[]);
            assert!(
                errors.iter().any(|e| e.contains(needle)),
                "empty {label} must be reported, got {errors:?}"
            );
        }

        // A component pointing at the empty-id family must not launder it into a conformant table.
        const ANONYMOUS: &[LicenseFamily] = &[LicenseFamily {
            id: "",
            spdx_id: "LicenseRef-Anon",
            name: "Anon",
            text_url: "https://example.invalid/anon",
            terms: &[],
        }];
        const ROW: &[ComponentLicense] = &[ComponentLicense {
            component: "anon",
            source_url: "https://example.invalid/model",
            gated: false,
            declared: "anon",
            family: "",
            attribution: None,
            retrieved: "2026-08-01",
        }];
        assert!(
            !license_table_conformance_errors(ANONYMOUS, ROW, &[]).is_empty(),
            "a component resolving against an empty family id must not pass the gate"
        );
    }

    #[test]
    fn duplicate_provider_id_is_an_error() {
        const A: ProviderComponents = ProviderComponents {
            provider_id: "twice",
            components: &["t5_xxl"],
        };
        const B: ProviderComponents = ProviderComponents {
            provider_id: "twice",
            components: &["flux1_dev_dit"],
        };
        let errors = license_table_conformance_errors(FAMILIES, COMPONENTS, &[A, B]);
        assert!(
            errors.iter().any(|e| e.contains("duplicate provider row")),
            "{errors:?}"
        );

        // The stable sort keeps input order for equal keys, so the manifest is not byte-stable —
        // which is what the gate above exists to prevent reaching a release.
        assert_ne!(
            component_licenses_manifest_json(FAMILIES, COMPONENTS, &[A, B]),
            component_licenses_manifest_json(FAMILIES, COMPONENTS, &[B, A])
        );
    }

    #[test]
    fn empty_provider_id_is_an_error() {
        const ANONYMOUS: ProviderComponents = ProviderComponents {
            provider_id: "",
            components: &["t5_xxl"],
        };
        let errors = license_table_conformance_errors(FAMILIES, COMPONENTS, &[ANONYMOUS]);
        assert!(
            errors.iter().any(|e| e.contains("empty provider_id")),
            "{errors:?}"
        );
    }

    #[test]
    fn provider_listing_a_component_twice_is_an_error() {
        const DOUBLED: ProviderComponents = ProviderComponents {
            provider_id: "stutter",
            components: &["t5_xxl", "t5_xxl"],
        };
        let errors = license_table_conformance_errors(FAMILIES, COMPONENTS, &[DOUBLED]);
        assert!(
            errors.iter().any(|e| e.contains("twice")),
            "a duplicated component key must be reported, got {errors:?}"
        );

        // It is visible in the manifest too: the emitted list repeats the key.
        let value: serde_json::Value = serde_json::from_str(&component_licenses_manifest_json(
            FAMILIES,
            COMPONENTS,
            &[DOUBLED],
        ))
        .unwrap();
        assert_eq!(
            value["providers"][0]["components"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn empty_attribution_does_not_satisfy_attribution_required() {
        // t5_xxl's family imposes AttributionRequired.
        let placeholder = ComponentLicense {
            attribution: Some(""),
            ..COMPONENTS[1]
        };
        assert!(
            !placeholder.is_well_formed(FAMILIES),
            "an empty attribution string is not an attribution"
        );
        let errors = license_table_conformance_errors(FAMILIES, &[placeholder], &[]);
        assert!(
            errors.iter().any(|e| e.contains("requires attribution")),
            "{errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.contains("empty attribution")),
            "{errors:?}"
        );

        // Even where the family requires none, an empty string is still not a value.
        let flux_placeholder = ComponentLicense {
            attribution: Some(""),
            ..COMPONENTS[0]
        };
        assert!(!flux_placeholder.is_well_formed(FAMILIES));
    }

    /// The `Some("")` hole, one space wider: `"   "` satisfies `!is_empty()` while printing as
    /// nothing on a licences page, so a CC-BY-* obligation could ship unrecorded behind it. Both the
    /// row-local predicate and the table checker must reject it.
    #[test]
    fn whitespace_only_attribution_does_not_satisfy_attribution_required() {
        // t5_xxl's family imposes AttributionRequired.
        let placeholder = ComponentLicense {
            attribution: Some("   "),
            ..COMPONENTS[1]
        };
        assert!(
            !placeholder.is_well_formed(FAMILIES),
            "a whitespace-only attribution string is not an attribution"
        );
        let errors = license_table_conformance_errors(FAMILIES, &[placeholder], &[]);
        assert!(
            errors.iter().any(|e| e.contains("requires attribution")),
            "{errors:?}"
        );
        assert!(
            errors.iter().any(|e| e.contains("empty attribution")),
            "{errors:?}"
        );

        // Tabs and newlines are whitespace too, and a family requiring no attribution still must not
        // record a blank one.
        let flux_placeholder = ComponentLicense {
            attribution: Some("\t\n "),
            ..COMPONENTS[0]
        };
        assert!(!flux_placeholder.is_well_formed(FAMILIES));
        assert!(
            license_table_conformance_errors(FAMILIES, &[flux_placeholder], &[])
                .iter()
                .any(|e| e.contains("empty attribution")),
        );
    }

    /// The blank-placeholder rule is not attribution-specific: every identity field the gate checks
    /// goes through the same predicate, so none of them can be satisfied by whitespace.
    #[test]
    fn whitespace_only_identity_fields_are_rejected() {
        let good_family = FAMILIES[0];
        let family_cases: [(&str, LicenseFamily, &str); 4] = [
            (
                "id",
                LicenseFamily {
                    id: "   ",
                    ..good_family
                },
                "empty id",
            ),
            (
                "spdx_id",
                LicenseFamily {
                    spdx_id: " ",
                    ..good_family
                },
                "no spdx_id",
            ),
            (
                "name",
                LicenseFamily {
                    name: "\t",
                    ..good_family
                },
                "no name",
            ),
            (
                "text_url",
                LicenseFamily {
                    text_url: "\n",
                    ..good_family
                },
                "no text_url",
            ),
        ];
        for (label, family, needle) in family_cases {
            let errors = license_table_conformance_errors(&[family], &[], &[]);
            assert!(
                errors.iter().any(|e| e.contains(needle)),
                "whitespace-only {label} must be reported, got {errors:?}"
            );
        }

        let good_row = COMPONENTS[1]; // t5_xxl.
        let row_cases: [(&str, ComponentLicense); 3] = [
            (
                "component",
                ComponentLicense {
                    component: "   ",
                    ..good_row
                },
            ),
            (
                "source_url",
                ComponentLicense {
                    source_url: " ",
                    ..good_row
                },
            ),
            (
                "declared",
                ComponentLicense {
                    declared: "\t",
                    ..good_row
                },
            ),
        ];
        for (label, row) in row_cases {
            assert!(
                !row.is_well_formed(FAMILIES),
                "whitespace-only {label} must be rejected row-locally"
            );
            assert!(
                !license_table_conformance_errors(FAMILIES, &[row], &[]).is_empty(),
                "whitespace-only {label} must also be reported by the table checker"
            );
        }

        const BLANK_PROVIDER: ProviderComponents = ProviderComponents {
            provider_id: "   ",
            components: &["t5_xxl"],
        };
        let errors = license_table_conformance_errors(FAMILIES, COMPONENTS, &[BLANK_PROVIDER]);
        assert!(
            errors.iter().any(|e| e.contains("empty provider_id")),
            "{errors:?}"
        );
    }

    /// The emitted order of a term union is keyed on [`LicenseTerm::tag`], not on where each variant
    /// sits in the `enum`. Pinning it here means inserting a future variant mid-declaration cannot
    /// silently reorder an already-committed manifest and trip the drift gate.
    #[test]
    fn serialized_term_order_follows_tag_not_declaration_order() {
        const MIXED: &[LicenseFamily] = &[LicenseFamily {
            id: "mixed",
            spdx_id: "LicenseRef-Mixed",
            name: "Mixed",
            text_url: "https://example.invalid/mixed",
            // Declaration order in the enum is: AttributionRequired, NoticeFileRequired,
            // NonCommercialWeights, …, GatedAccess (last), and `gated_access` reaches the union
            // last of all — it is contributed by the component row below, after the family's own
            // terms. Tag order is alphabetical, which puts gated_access second and
            // notice_file_required last — a different sequence from either.
            terms: &[
                LicenseTerm::NoticeFileRequired,
                LicenseTerm::AttributionRequired,
                LicenseTerm::NonCommercialWeights,
            ],
        }];
        const ROWS: &[ComponentLicense] = &[ComponentLicense {
            component: "mixed",
            source_url: "https://example.invalid/model",
            gated: true,
            declared: "mixed",
            family: "mixed",
            attribution: Some("© Example"),
            retrieved: "2026-08-01",
        }];
        const PROVIDER: ProviderComponents = ProviderComponents {
            provider_id: "mixed",
            components: &["mixed"],
        };

        assert!(license_table_conformance_errors(MIXED, ROWS, &[PROVIDER]).is_empty());

        let value: serde_json::Value =
            serde_json::from_str(&component_licenses_manifest_json(MIXED, ROWS, &[PROVIDER]))
                .unwrap();
        let tags: Vec<&str> = value["providers"][0]["terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["term"].as_str().unwrap())
            .collect();
        assert_eq!(
            tags,
            vec![
                "attribution_required",
                "gated_access",
                "non_commercial_weights",
                "notice_file_required",
            ],
            "terms serialize in tag order; declaration order would put notice_file_required second \
             and gated_access last"
        );
    }

    /// `is_well_formed` enforces exactly the documented row conditions, and agrees with the table
    /// checker on every one of them (they share `row_errors`, and this pins that they stay tied).
    /// The empty-attribution condition has its own test below.
    #[test]
    fn is_well_formed_enforces_the_documented_row_conditions() {
        let good = COMPONENTS[1]; // t5_xxl: attribution-requiring family, attribution present.
        assert!(good.is_well_formed(FAMILIES));

        let cases: [(&str, ComponentLicense); 5] = [
            (
                "empty identity: component",
                ComponentLicense {
                    component: "",
                    ..good
                },
            ),
            (
                "empty identity: source_url",
                ComponentLicense {
                    source_url: "",
                    ..good
                },
            ),
            (
                "empty identity: declared",
                ComponentLicense {
                    declared: "",
                    ..good
                },
            ),
            (
                "family does not resolve",
                ComponentLicense {
                    family: "not-a-family",
                    ..good
                },
            ),
            (
                "retrieved is not an ISO date",
                ComponentLicense {
                    retrieved: "last tuesday",
                    ..good
                },
            ),
        ];
        for (label, row) in cases {
            assert!(!row.is_well_formed(FAMILIES), "{label} must be rejected");
            assert!(
                !license_table_conformance_errors(FAMILIES, &[row], &[]).is_empty(),
                "{label} must also be reported by the table checker"
            );
        }

        // AttributionRequired in the family implies attribution.is_some().
        let unattributed = ComponentLicense {
            attribution: None,
            ..good
        };
        assert!(resolve_family(FAMILIES, good.family)
            .unwrap()
            .requires_attribution());
        assert!(!unattributed.is_well_formed(FAMILIES));

        // A family that does NOT require attribution leaves the row well-formed without one.
        let flux_dit = COMPONENTS[0];
        assert!(flux_dit.attribution.is_none());
        assert!(!resolve_family(FAMILIES, flux_dit.family)
            .unwrap()
            .requires_attribution());
        assert!(flux_dit.is_well_formed(FAMILIES));
    }

    #[test]
    fn unresolved_family_is_an_error() {
        const ORPHAN: &[ComponentLicense] = &[ComponentLicense {
            component: "mystery",
            source_url: "https://example.invalid/model",
            gated: false,
            declared: "who-knows",
            family: "not-a-family",
            attribution: None,
            retrieved: "2026-08-01",
        }];
        let errors = license_table_conformance_errors(FAMILIES, ORPHAN, &[]);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("unknown licence family"), "{errors:?}");
    }

    #[test]
    fn attribution_requiring_family_without_attribution_is_an_error() {
        const MISSING: &[ComponentLicense] = &[ComponentLicense {
            component: "t5_xxl",
            source_url: "https://huggingface.co/google/t5-v1_1-xxl",
            gated: false,
            declared: "apache-2.0",
            family: "apache-2-0",
            attribution: None,
            retrieved: "2026-08-01",
        }];
        let errors = license_table_conformance_errors(FAMILIES, MISSING, &[]);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("requires attribution"), "{errors:?}");
    }

    #[test]
    fn provenance_must_be_a_real_iso_date() {
        assert!(is_iso_date("2026-08-01"));
        assert!(!is_iso_date("2026-8-1"));
        assert!(!is_iso_date("01-08-2026"));
        assert!(!is_iso_date("2026-13-01"), "month 13 is not a date");
        assert!(!is_iso_date("2026-08-32"), "day 32 is not a date");
        assert!(!is_iso_date(""));

        const STALE: &[ComponentLicense] = &[ComponentLicense {
            component: "undated",
            source_url: "https://example.invalid/model",
            gated: false,
            declared: "apache-2.0",
            family: "apache-2-0",
            attribution: Some("© Example"),
            retrieved: "sometime",
        }];
        let errors = license_table_conformance_errors(FAMILIES, STALE, &[]);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("non-ISO retrieved date"), "{errors:?}");
    }

    /// A range check on month 1-12 / day 1-31 is not calendar validation. `retrieved` is
    /// hand-transcribed and downstream tooling *parses* it — Python's `date.fromisoformat` raises on
    /// `2026-02-31` — so a date this gate blesses but no calendar contains would fail sc-16664's
    /// validator with a stack trace instead of a message here.
    #[test]
    fn impossible_calendar_dates_are_rejected() {
        // Month lengths.
        assert!(!is_iso_date("2026-02-31"), "February has no 31st");
        assert!(!is_iso_date("2026-02-30"), "February has no 30th");
        assert!(!is_iso_date("2026-04-31"), "April has 30 days");
        assert!(!is_iso_date("2026-06-31"));
        assert!(!is_iso_date("2026-09-31"));
        assert!(!is_iso_date("2026-11-31"));
        assert!(is_iso_date("2026-04-30"));
        assert!(is_iso_date("2026-01-31"));
        assert!(is_iso_date("2026-12-31"));

        // Leap years: the Gregorian rule, not just "divisible by four".
        assert!(!is_iso_date("2026-02-29"), "2026 is not a leap year");
        assert!(is_iso_date("2024-02-29"), "2024 is a leap year");
        assert!(is_iso_date("2026-02-28"));
        assert!(
            !is_iso_date("1900-02-29"),
            "1900 is a century non-leap year"
        );
        assert!(is_iso_date("2000-02-29"), "2000 is a 400-year leap year");

        // Zero components are not dates.
        assert!(!is_iso_date("0000-01-01"), "there is no year zero here");
        assert!(!is_iso_date("2026-00-01"), "month 0 is not a date");
        assert!(!is_iso_date("2026-01-00"), "day 0 is not a date");

        // And the gate reports it rather than passing it downstream.
        const IMPOSSIBLE: &[ComponentLicense] = &[ComponentLicense {
            component: "impossible",
            source_url: "https://example.invalid/model",
            gated: false,
            declared: "apache-2.0",
            family: "apache-2-0",
            attribution: Some("© Example"),
            retrieved: "2026-02-31",
        }];
        let errors = license_table_conformance_errors(FAMILIES, IMPOSSIBLE, &[]);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("non-ISO retrieved date"), "{errors:?}");
    }

    #[test]
    fn provider_referencing_a_missing_component_is_an_error() {
        const DANGLING: ProviderComponents = ProviderComponents {
            provider_id: "ghost",
            components: &["nope"],
        };
        let errors = license_table_conformance_errors(FAMILIES, COMPONENTS, &[DANGLING]);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("unknown component"), "{errors:?}");
    }

    #[test]
    fn provider_with_no_components_is_an_error() {
        const EMPTY: ProviderComponents = ProviderComponents {
            provider_id: "hollow",
            components: &[],
        };
        let errors = license_table_conformance_errors(FAMILIES, COMPONENTS, &[EMPTY]);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("maps to no components"), "{errors:?}");
    }

    #[test]
    fn duplicate_component_rows_are_rejected() {
        const DUPES: &[ComponentLicense] = &[
            ComponentLicense {
                component: "t5_xxl",
                source_url: "https://huggingface.co/google/t5-v1_1-xxl",
                gated: false,
                declared: "apache-2.0",
                family: "apache-2-0",
                attribution: Some("© Google"),
                retrieved: "2026-08-01",
            },
            ComponentLicense {
                component: "t5_xxl",
                source_url: "https://huggingface.co/google/t5-v1_1-xxl",
                gated: false,
                declared: "apache-2.0",
                family: "apache-2-0",
                attribution: Some("© Google"),
                retrieved: "2026-08-01",
            },
        ];
        let errors = license_table_conformance_errors(FAMILIES, DUPES, &[]);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("duplicate component row"), "{errors:?}");
    }

    /// Two ceilings in one union are two separate disclosures — they are not collapsed to a
    /// minimum, so a consumer showing a user the terms of a render shows both amounts.
    #[test]
    fn multiple_revenue_ceilings_are_both_retained() {
        const TWO: &[LicenseFamily] = &[
            LicenseFamily {
                id: "a",
                spdx_id: "LicenseRef-A",
                name: "A",
                text_url: "https://example.invalid/a",
                terms: &[LicenseTerm::RevenueCeiling {
                    amount_usd: 1_000_000,
                    boundary: CeilingBoundary::Exclusive,
                }],
            },
            LicenseFamily {
                id: "b",
                spdx_id: "LicenseRef-B",
                name: "B",
                text_url: "https://example.invalid/b",
                terms: &[LicenseTerm::RevenueCeiling {
                    amount_usd: 10_000_000,
                    boundary: CeilingBoundary::Inclusive,
                }],
            },
        ];
        const ROWS: &[ComponentLicense] = &[
            ComponentLicense {
                component: "a",
                source_url: "https://example.invalid/a",
                gated: false,
                declared: "a",
                family: "a",
                attribution: None,
                retrieved: "2026-08-01",
            },
            ComponentLicense {
                component: "b",
                source_url: "https://example.invalid/b",
                gated: false,
                declared: "b",
                family: "b",
                attribution: None,
                retrieved: "2026-08-01",
            },
        ];
        const BOTH: ProviderComponents = ProviderComponents {
            provider_id: "both",
            components: &["a", "b"],
        };
        let terms = provider_terms(&BOTH, ROWS, TWO);
        assert_eq!(
            terms,
            vec![
                LicenseTerm::RevenueCeiling {
                    amount_usd: 1_000_000,
                    boundary: CeilingBoundary::Exclusive,
                },
                LicenseTerm::RevenueCeiling {
                    amount_usd: 10_000_000,
                    boundary: CeilingBoundary::Inclusive,
                },
            ]
        );
    }
}

/// The sc-16898 amendments: the three shapes the sc-16662 licence read contradicted.
///
/// Fixtures here are miniatures of the real families named in
/// `docs/licensing/sc-16662-licence-family-evidence.md`, because each amendment exists for a
/// specific pair of licences that the previous shape could not tell apart.
#[cfg(test)]
mod v3_amendment_tests {
    use super::*;

    // Five families drawn from the evidence, covering both flow-down kinds and both ceiling
    // boundaries. `text_url`s are fixtures; the ids and clause shapes are the evidenced ones.
    const FAMILIES: &[LicenseFamily] = &[
        // "copy of the licence" flow-down (evidence §IV(a)), plus the exclusive ceiling.
        LicenseFamily {
            id: "stability-ai-community",
            spdx_id: "LicenseRef-Stability-AI-Community",
            name: "Stability AI Community License Agreement",
            text_url: "https://example.invalid/stability",
            terms: &[
                LicenseTerm::DownstreamLicenseCopy {
                    family: "stability-ai-community",
                },
                // "more than USD $1,000,000 in annual revenue".
                LicenseTerm::RevenueCeiling {
                    amount_usd: 1_000_000,
                    boundary: CeilingBoundary::Exclusive,
                },
            ],
        },
        // A second, textually unrelated "copy of the licence" flow-down (evidence §2).
        LicenseFamily {
            id: "apple-mlr",
            spdx_id: "LicenseRef-Apple-MLR",
            name: "Apple Machine Learning Research Model License Agreement",
            text_url: "https://example.invalid/apple",
            terms: &[
                LicenseTerm::DownstreamLicenseCopy {
                    family: "apple-mlr",
                },
                LicenseTerm::NonCommercialWeights,
            ],
        },
        // "restrictions as enforceable provisions" flow-down — paragraph 5 (evidence §III).
        LicenseFamily {
            id: "creativeml-openrail-pp-m",
            spdx_id: "LicenseRef-OpenRAIL-PP-M",
            name: "CreativeML Open RAIL++-M License",
            text_url: "https://example.invalid/openrail",
            terms: &[LicenseTerm::DownstreamRestrictions {
                family: "creativeml-openrail-pp-m",
            }],
        },
        // The same *kind* as OpenRAIL++, but a different body of text: paragraph 4 plus the whole of
        // Attachment A, plus a no-relicensing constraint that rides its own DeployerObligation. And
        // an INCLUSIVE ceiling, unlike Stability's.
        LicenseFamily {
            id: "ltx-2-community",
            spdx_id: "LicenseRef-LTX-2-Community",
            name: "LTX-2 Community License Agreement",
            text_url: "https://example.invalid/ltx",
            terms: &[
                LicenseTerm::DownstreamRestrictions {
                    family: "ltx-2-community",
                },
                LicenseTerm::DownstreamLicenseCopy {
                    family: "ltx-2-community",
                },
                LicenseTerm::DeployerObligation {
                    text: "Derivatives must be distributed exclusively under the terms of this \
                           Agreement with a complete copy of this license included.",
                },
                // "annual revenues of at least $10,000,000".
                LicenseTerm::RevenueCeiling {
                    amount_usd: 10_000_000,
                    boundary: CeilingBoundary::Inclusive,
                },
            ],
        },
        // Llama 3.1's threshold is denominated in monthly active users, not revenue — A3.
        LicenseFamily {
            id: "llama-3-1-community",
            spdx_id: "LicenseRef-Llama-3-1-Community",
            name: "Llama 3.1 Community License Agreement",
            text_url: "https://example.invalid/llama",
            terms: &[
                LicenseTerm::DownstreamLicenseCopy {
                    family: "llama-3-1-community",
                },
                LicenseTerm::DeployerObligation {
                    text: "If, on the Llama 3.1 version release date, the monthly active users of \
                           the products or services made available by or for Licensee is greater \
                           than 700 million monthly active users in the preceding calendar month, \
                           you must request a license from Meta.",
                },
            ],
        },
    ];

    // SVD-XT and SD3.5 are the A2 pair: ONE licence text, two upstream distribution settings.
    const COMPONENTS: &[ComponentLicense] = &[
        ComponentLicense {
            component: "svd_xt",
            source_url: "https://example.invalid/stable-video-diffusion-img2vid-xt",
            gated: false,
            declared: "stable-video-diffusion-community",
            family: "stability-ai-community",
            attribution: None,
            retrieved: "2026-08-02",
        },
        ComponentLicense {
            component: "sd3_5_large_dit",
            source_url: "https://example.invalid/stable-diffusion-3.5-large",
            gated: true,
            declared: "stabilityai-ai-community",
            family: "stability-ai-community",
            attribution: None,
            retrieved: "2026-08-02",
        },
        ComponentLicense {
            component: "sdxl_unet",
            source_url: "https://example.invalid/stable-diffusion-xl-base-1.0",
            gated: false,
            declared: "openrail++",
            family: "creativeml-openrail-pp-m",
            attribution: None,
            retrieved: "2026-08-02",
        },
        ComponentLicense {
            component: "ltx_dit",
            source_url: "https://example.invalid/ltx-2.3",
            gated: false,
            declared: "ltx-2-community-license-agreement",
            family: "ltx-2-community",
            attribution: None,
            retrieved: "2026-08-02",
        },
        ComponentLicense {
            component: "dfn5b_clip",
            source_url: "https://example.invalid/DFN5B-CLIP-ViT-H-14-378",
            gated: false,
            declared: "apple-ascl",
            family: "apple-mlr",
            attribution: None,
            retrieved: "2026-08-02",
        },
        ComponentLicense {
            component: "joycaption_llama",
            source_url: "https://example.invalid/llama-joycaption-beta-one-hf-llava",
            gated: false,
            declared: "Llama 3.1 Community License",
            family: "llama-3-1-community",
            attribution: None,
            retrieved: "2026-08-02",
        },
    ];

    #[test]
    fn amendment_fixture_table_is_conformant() {
        const PROVIDERS: &[ProviderComponents] = &[ProviderComponents {
            provider_id: "everything",
            components: &[
                "svd_xt",
                "sd3_5_large_dit",
                "sdxl_unet",
                "ltx_dit",
                "dfn5b_clip",
                "joycaption_llama",
            ],
        }];
        assert_eq!(
            license_table_conformance_errors(FAMILIES, COMPONENTS, PROVIDERS),
            Vec::<String>::new()
        );
    }

    // ---------------------------------------------------------------------------------------
    // A1 — flow-down duties stay distinct through the union.
    // ---------------------------------------------------------------------------------------

    /// The defect the bare `DownstreamFlowDown` variant carried: eleven of sixteen families impose a
    /// flow-down, and a payload-free variant deduped every one of them to a SINGLE element of the
    /// union. A consumer surfacing the join then showed a user one obligation where the catalog
    /// carried several.
    ///
    /// Three axes are pinned here: two *kinds* stay distinct, two families of the *same* kind stay
    /// distinct, and every survivor resolves back to the text that imposed it.
    #[test]
    fn distinct_flow_down_duties_do_not_dedupe_to_one_element() {
        // Stability (hand over a copy) + OpenRAIL++ (write the restrictions into your own user
        // agreement) — the two structurally different kinds.
        const TWO_KINDS: ProviderComponents = ProviderComponents {
            provider_id: "sdxl_then_svd",
            components: &["sdxl_unet", "svd_xt"],
        };
        let terms = provider_terms(&TWO_KINDS, COMPONENTS, FAMILIES);
        let flow_downs: Vec<LicenseTerm> = terms
            .iter()
            .copied()
            .filter(|term| term.flow_down_family().is_some())
            .collect();
        assert_eq!(
            flow_downs,
            vec![
                LicenseTerm::DownstreamLicenseCopy {
                    family: "stability-ai-community"
                },
                LicenseTerm::DownstreamRestrictions {
                    family: "creativeml-openrail-pp-m"
                },
            ],
            "a copy-of-licence duty and a restrictions-as-provisions duty are two obligations"
        );

        // Two families imposing the SAME kind are still two duties: a distributor hands over two
        // documents, not one. This is the case a two-variant split alone would still collapse.
        const TWO_COPIES: ProviderComponents = ProviderComponents {
            provider_id: "svd_with_apple_clip",
            components: &["svd_xt", "dfn5b_clip"],
        };
        let copies: Vec<&'static str> = provider_terms(&TWO_COPIES, COMPONENTS, FAMILIES)
            .iter()
            .filter_map(|term| match term {
                LicenseTerm::DownstreamLicenseCopy { family } => Some(*family),
                _ => None,
            })
            .collect();
        assert_eq!(copies, vec!["apple-mlr", "stability-ai-community"]);

        // Two families imposing the same RESTRICTIONS kind likewise — OpenRAIL++ requires its
        // paragraph 5, LTX-2 requires its paragraph 4 plus all of Attachment A.
        const TWO_RESTRICTIONS: ProviderComponents = ProviderComponents {
            provider_id: "sdxl_then_ltx",
            components: &["sdxl_unet", "ltx_dit"],
        };
        let restrictions: Vec<&'static str> =
            provider_terms(&TWO_RESTRICTIONS, COMPONENTS, FAMILIES)
                .iter()
                .filter_map(|term| match term {
                    LicenseTerm::DownstreamRestrictions { family } => Some(*family),
                    _ => None,
                })
                .collect();
        assert_eq!(
            restrictions,
            vec!["creativeml-openrail-pp-m", "ltx-2-community"]
        );

        // And every surviving duty routes back to the specific text, which is what makes the
        // disclosure actionable rather than a label.
        for term in flow_downs {
            let family = resolve_family(FAMILIES, term.flow_down_family().unwrap())
                .expect("a flow-down term resolves to the family whose text it points at");
            assert!(!family.text_url.is_empty());
        }
    }

    /// The other half of the same property: dedup must still fire where the duty really is one
    /// duty. Two components under the SAME family contribute one flow-down, not two.
    #[test]
    fn one_family_reached_twice_contributes_one_flow_down() {
        const TWICE: ProviderComponents = ProviderComponents {
            provider_id: "svd_and_sd35",
            components: &["svd_xt", "sd3_5_large_dit"],
        };
        let copies = provider_terms(&TWICE, COMPONENTS, FAMILIES)
            .into_iter()
            .filter(|term| term.flow_down_family().is_some())
            .count();
        assert_eq!(
            copies, 1,
            "one licence text imposes one flow-down however many of its checkpoints are loaded"
        );
    }

    /// The flow-down payload is the route from a term back to its text, so a term naming a family
    /// other than its own would point a consumer at the wrong licence. The gate rejects it.
    #[test]
    fn flow_down_term_naming_another_family_is_an_error() {
        const MISNAMED: &[LicenseFamily] = &[LicenseFamily {
            id: "ltx-2-community",
            spdx_id: "LicenseRef-LTX-2-Community",
            name: "LTX-2 Community License Agreement",
            text_url: "https://example.invalid/ltx",
            terms: &[LicenseTerm::DownstreamRestrictions {
                // Copy-paste from the OpenRAIL++ row — the transcription slip this catches.
                family: "creativeml-openrail-pp-m",
            }],
        }];
        let errors = license_table_conformance_errors(MISNAMED, &[], &[]);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("must name its own family"), "{errors:?}");

        // A blank id is caught by the same rule rather than silently resolving to nothing.
        const BLANK: &[LicenseFamily] = &[LicenseFamily {
            id: "apple-mlr",
            spdx_id: "LicenseRef-Apple-MLR",
            name: "Apple MLR",
            text_url: "https://example.invalid/apple",
            terms: &[LicenseTerm::DownstreamLicenseCopy { family: "" }],
        }];
        assert!(license_table_conformance_errors(BLANK, &[], &[])
            .iter()
            .any(|e| e.contains("must name its own family")));
    }

    // ---------------------------------------------------------------------------------------
    // A2 — gating is a per-checkpoint fact, so one family covers both gating states.
    // ---------------------------------------------------------------------------------------

    /// `stable-video-diffusion-img2vid-xt` (ungated) and `stable-diffusion-3.5-large` (gated) are
    /// governed by the SAME Stability AI Community License text. With `GatedAccess` on
    /// `LicenseFamily::terms` the only way to express that was two families differing in no legal
    /// respect — defeating the "reviewed unit is the family" principle and manufacturing the exact
    /// duplicate-family hazard the sc-16661 conformance check exists to catch.
    #[test]
    fn one_family_covers_a_gated_and_an_ungated_checkpoint() {
        const SVD: ProviderComponents = ProviderComponents {
            provider_id: "svd_xt",
            components: &["svd_xt"],
        };
        const SD35: ProviderComponents = ProviderComponents {
            provider_id: "sd3_5_large",
            components: &["sd3_5_large_dit"],
        };
        assert!(license_table_conformance_errors(FAMILIES, COMPONENTS, &[SVD, SD35]).is_empty());

        // Both checkpoints resolve to one family — no split, no duplicate id.
        let svd_family = resolve_component(COMPONENTS, "svd_xt").unwrap().family;
        let sd35_family = resolve_component(COMPONENTS, "sd3_5_large_dit")
            .unwrap()
            .family;
        assert_eq!(svd_family, sd35_family);

        // Yet the derived unions differ exactly where the upstream distribution differs.
        let svd = provider_terms(&SVD, COMPONENTS, FAMILIES);
        let sd35 = provider_terms(&SD35, COMPONENTS, FAMILIES);
        assert!(
            !svd.contains(&LicenseTerm::GatedAccess),
            "SVD-XT is distributed ungated, {svd:?}"
        );
        assert!(
            sd35.contains(&LicenseTerm::GatedAccess),
            "a consumer must still learn the render touched a gated checkpoint, {sd35:?}"
        );
        // The licence terms themselves are identical — the only difference is the gate.
        let without_gate: Vec<LicenseTerm> = sd35
            .iter()
            .copied()
            .filter(|term| *term != LicenseTerm::GatedAccess)
            .collect();
        assert_eq!(without_gate, svd);

        // And the manifest carries one family row, with the gate recorded per component.
        let value: serde_json::Value = serde_json::from_str(&component_licenses_manifest_json(
            FAMILIES,
            COMPONENTS,
            &[SVD, SD35],
        ))
        .unwrap();
        let stability_rows = value["families"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["id"] == "stability-ai-community")
            .count();
        assert_eq!(stability_rows, 1, "one licence text, one reviewed family");
        let component_rows = value["components"].as_array().unwrap();
        let gated_of = |key: &str| {
            component_rows
                .iter()
                .find(|c| c["component"] == key)
                .unwrap()["gated"]
                .as_bool()
                .unwrap()
        };
        assert!(!gated_of("svd_xt"));
        assert!(gated_of("sd3_5_large_dit"));
    }

    /// The corollary rule: `gated_access` is not a licence term, so a family may not declare it.
    /// Without this the old shape is reachable again one hand-authored table at a time.
    #[test]
    fn family_declaring_gated_access_is_an_error() {
        const GATING_FAMILY: &[LicenseFamily] = &[LicenseFamily {
            id: "stability-ai-community",
            spdx_id: "LicenseRef-Stability-AI-Community",
            name: "Stability AI Community License Agreement",
            text_url: "https://example.invalid/stability",
            terms: &[LicenseTerm::GatedAccess],
        }];
        let errors = license_table_conformance_errors(GATING_FAMILY, &[], &[]);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].contains("per-checkpoint distribution fact"),
            "{errors:?}"
        );
    }

    /// sc-16662: an addressless policy or registration is spelled `None`. A blank `Some` reads as
    /// an address in the JSON and renders as nothing on a licences page, so it is neither the
    /// "here is where to look" disclosure nor the "the text names no address" one — and it would
    /// sort and serialize next to the honest `None` rather than being visibly wrong.
    #[test]
    fn a_blank_address_is_an_error_but_an_absent_one_is_not() {
        const BLANK: &[LicenseFamily] = &[LicenseFamily {
            id: "blank-addresses",
            spdx_id: "LicenseRef-Blank",
            name: "Fixture with blank addresses",
            text_url: "https://example.invalid/blank",
            terms: &[
                LicenseTerm::AcceptableUsePolicy { url: Some("") },
                LicenseTerm::RegistrationRequired {
                    contact: Some("   "),
                },
            ],
        }];
        let errors = license_table_conformance_errors(BLANK, &[], &[]);
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(errors[0].contains("acceptable_use_policy"), "{errors:?}");
        assert!(errors[1].contains("registration_required"), "{errors:?}");
        assert!(
            errors
                .iter()
                .all(|e| e.contains("record a licence that names no address as None")),
            "{errors:?}"
        );

        // The honest shape passes, and so does the addressed one.
        const HONEST: &[LicenseFamily] = &[LicenseFamily {
            id: "ltx-2-community",
            spdx_id: "LicenseRef-LTX-2-Community",
            name: "LTX-2 Community License Agreement",
            text_url: "https://example.invalid/ltx",
            terms: &[
                LicenseTerm::AcceptableUsePolicy { url: None },
                LicenseTerm::RegistrationRequired { contact: None },
            ],
        }];
        assert_eq!(
            license_table_conformance_errors(HONEST, &[], &[]),
            Vec::<String>::new()
        );

        // And the two absences stay distinguishable from a blank one all the way into the union,
        // which is the reason `sort_key` carries presence rather than only contents.
        assert_ne!(
            LicenseTerm::AcceptableUsePolicy { url: None }.to_json(),
            LicenseTerm::AcceptableUsePolicy { url: Some("") }.to_json()
        );
    }

    /// Gating is not a property of the family, so a table defect in the family section must not
    /// swallow it — the consumer still needs to know the checkpoint is gated.
    #[test]
    fn gating_survives_a_component_whose_family_does_not_resolve() {
        const ORPHANED: &[ComponentLicense] = &[ComponentLicense {
            component: "mystery_gated",
            source_url: "https://example.invalid/mystery",
            gated: true,
            declared: "who-knows",
            family: "not-a-family",
            attribution: None,
            retrieved: "2026-08-02",
        }];
        const PROVIDER: ProviderComponents = ProviderComponents {
            provider_id: "mystery",
            components: &["mystery_gated"],
        };
        assert_eq!(
            provider_terms(&PROVIDER, ORPHANED, FAMILIES),
            vec![LicenseTerm::GatedAccess]
        );
        // The unresolved family is still a table error — this is under-reporting insurance, not a
        // licence to ship a dangling row.
        assert!(!license_table_conformance_errors(FAMILIES, ORPHANED, &[PROVIDER]).is_empty());
    }

    // ---------------------------------------------------------------------------------------
    // A3 — threshold representability.
    // ---------------------------------------------------------------------------------------

    /// `RevenueCeiling`'s boundary is load-bearing, not decoration: Stability terminates *above*
    /// USD $1,000,000 ("more than"), LTX-2 requires a paid licence *at* $10,000,000 ("at least").
    /// Revenue exactly equal to the named amount therefore falls on opposite sides of the two.
    #[test]
    fn revenue_ceiling_boundary_semantics_are_pinned() {
        // The reading each boundary denotes, spelled out as an executable statement so the meaning
        // is pinned by more than prose. It lives HERE, in a test: nothing in the contract compares a
        // consumer's revenue against a ceiling, and nothing in this system changes behaviour on one.
        fn revenue_is_below_the_named_threshold(term: LicenseTerm, revenue_usd: u64) -> bool {
            match term {
                LicenseTerm::RevenueCeiling {
                    amount_usd,
                    boundary,
                } => match boundary {
                    // "more than $N" — $N itself has not exceeded it.
                    CeilingBoundary::Exclusive => revenue_usd <= amount_usd,
                    // "at least $N" — $N itself has reached it.
                    CeilingBoundary::Inclusive => revenue_usd < amount_usd,
                },
                other => panic!("not a ceiling: {other:?}"),
            }
        }

        const STABILITY: LicenseTerm = LicenseTerm::RevenueCeiling {
            amount_usd: 1_000_000,
            boundary: CeilingBoundary::Exclusive,
        };
        const LTX: LicenseTerm = LicenseTerm::RevenueCeiling {
            amount_usd: 10_000_000,
            boundary: CeilingBoundary::Inclusive,
        };

        assert!(revenue_is_below_the_named_threshold(STABILITY, 1_000_000));
        assert!(!revenue_is_below_the_named_threshold(STABILITY, 1_000_001));
        assert!(!revenue_is_below_the_named_threshold(LTX, 10_000_000));
        assert!(revenue_is_below_the_named_threshold(LTX, 9_999_999));

        // Same amount, different reading — two distinct disclosures that must not dedupe, because
        // at exactly that amount they say opposite things.
        const EXCLUSIVE_1M: LicenseTerm = LicenseTerm::RevenueCeiling {
            amount_usd: 1_000_000,
            boundary: CeilingBoundary::Exclusive,
        };
        const INCLUSIVE_1M: LicenseTerm = LicenseTerm::RevenueCeiling {
            amount_usd: 1_000_000,
            boundary: CeilingBoundary::Inclusive,
        };
        assert_ne!(EXCLUSIVE_1M, INCLUSIVE_1M);
        assert_ne!(EXCLUSIVE_1M.sort_key(), INCLUSIVE_1M.sort_key());
        assert!(revenue_is_below_the_named_threshold(
            EXCLUSIVE_1M,
            1_000_000
        ));
        assert!(!revenue_is_below_the_named_threshold(
            INCLUSIVE_1M,
            1_000_000
        ));

        // Both readings reach the manifest verbatim.
        const BOTH: ProviderComponents = ProviderComponents {
            provider_id: "svd_then_ltx",
            components: &["svd_xt", "ltx_dit"],
        };
        let value: serde_json::Value = serde_json::from_str(&component_licenses_manifest_json(
            FAMILIES,
            COMPONENTS,
            &[BOTH],
        ))
        .unwrap();
        let ceilings: Vec<(u64, &str)> = value["providers"][0]["terms"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| t["term"] == "revenue_ceiling")
            .map(|t| {
                (
                    t["amount_usd"].as_u64().unwrap(),
                    t["boundary"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            ceilings,
            vec![(1_000_000, "exclusive"), (10_000_000, "inclusive")]
        );
    }

    /// A3's resolution for thresholds `RevenueCeiling` cannot express: Llama 3.1 Community's
    /// *700 million monthly active users* trigger is transcribed verbatim as a
    /// `DeployerObligation`, which is a complete disclosure because nothing computes against a term.
    /// Writing it as `RevenueCeiling { amount_usd: 700_000_000 }` would be a false transcription —
    /// the number is users, not dollars — so no typed variant was added.
    #[test]
    fn a_non_revenue_threshold_is_disclosed_verbatim_as_a_deployer_obligation() {
        const JOYCAPTION: ProviderComponents = ProviderComponents {
            provider_id: "joycaption",
            components: &["joycaption_llama"],
        };
        let terms = provider_terms(&JOYCAPTION, COMPONENTS, FAMILIES);
        let quoted: Vec<&'static str> = terms
            .iter()
            .filter_map(|term| match term {
                LicenseTerm::DeployerObligation { text } => Some(*text),
                _ => None,
            })
            .collect();
        assert_eq!(quoted.len(), 1, "{terms:?}");
        assert!(
            quoted[0].contains("700 million monthly active users"),
            "the trigger must survive as the licence's own words, got {:?}",
            quoted[0]
        );
        // And it is emphatically NOT laundered into a dollar figure.
        assert!(
            !terms.iter().any(|term| term.tag() == "revenue_ceiling"),
            "a user-count threshold is not a revenue ceiling, {terms:?}"
        );
    }

    // ---------------------------------------------------------------------------------------
    // Invariants the amendments must not break.
    // ---------------------------------------------------------------------------------------

    /// `sort_key` is the total order the manifest is emitted in, and [`provider_terms`] dedups the
    /// sorted list. Two distinct terms sharing a key would therefore make one of them VANISH from a
    /// consumer's disclosure — so injectivity is a disclosure-accuracy property, not a tidiness one.
    /// It matters more after sc-16898 than before it: three variants now carry a field that was not
    /// in the key at all. The same assertion runs over `to_json`, because separating two terms in
    /// the sort order buys nothing if they serialize to the same bytes.
    #[test]
    fn sort_key_is_injective_across_every_variant() {
        // No wildcard arm: adding a `LicenseTerm` variant stops this compiling, which is the prompt
        // to add a sample for it to `SAMPLES` below (the tag assertion then checks that you did).
        fn tag_of(term: LicenseTerm) -> &'static str {
            match term {
                LicenseTerm::AttributionRequired => "attribution_required",
                LicenseTerm::NoticeFileRequired => "notice_file_required",
                LicenseTerm::NonCommercialWeights => "non_commercial_weights",
                LicenseTerm::NonCommercialOutputs => "non_commercial_outputs",
                LicenseTerm::RevenueCeiling { .. } => "revenue_ceiling",
                LicenseTerm::RegistrationRequired { .. } => "registration_required",
                LicenseTerm::AcceptableUsePolicy { .. } => "acceptable_use_policy",
                LicenseTerm::DeployerObligation { .. } => "deployer_obligation",
                LicenseTerm::DownstreamLicenseCopy { .. } => "downstream_license_copy",
                LicenseTerm::DownstreamRestrictions { .. } => "downstream_restrictions",
                LicenseTerm::GatedAccess => "gated_access",
            }
        }

        // Every variant at least once, and every payload-carrying variant at least twice so the
        // payload's contribution to the key is exercised — including the two pairs that differ in
        // ONLY the field sc-16898 added.
        const SAMPLES: &[LicenseTerm] = &[
            LicenseTerm::AttributionRequired,
            LicenseTerm::NoticeFileRequired,
            LicenseTerm::NonCommercialWeights,
            LicenseTerm::NonCommercialOutputs,
            LicenseTerm::GatedAccess,
            LicenseTerm::RevenueCeiling {
                amount_usd: 1_000_000,
                boundary: CeilingBoundary::Exclusive,
            },
            // Same amount, different boundary.
            LicenseTerm::RevenueCeiling {
                amount_usd: 1_000_000,
                boundary: CeilingBoundary::Inclusive,
            },
            LicenseTerm::RevenueCeiling {
                amount_usd: 10_000_000,
                boundary: CeilingBoundary::Inclusive,
            },
            LicenseTerm::RegistrationRequired {
                contact: Some("https://example.invalid/register"),
            },
            LicenseTerm::RegistrationRequired {
                contact: Some("opensource@example.invalid"),
            },
            // The sc-16662 shape: a registration the text names with no address. It must not
            // collide with either of the addressed ones, nor with the addressless AUP below.
            LicenseTerm::RegistrationRequired { contact: None },
            LicenseTerm::AcceptableUsePolicy {
                url: Some("https://example.invalid/aup"),
            },
            LicenseTerm::AcceptableUsePolicy {
                url: Some("https://example.invalid/prohibited_use_policy"),
            },
            LicenseTerm::AcceptableUsePolicy { url: None },
            LicenseTerm::DeployerObligation {
                text: "implement and maintain content filtering measures",
            },
            LicenseTerm::DeployerObligation {
                text: "disclose that Outputs were generated using artificial intelligence",
            },
            // Same kind, different family.
            LicenseTerm::DownstreamLicenseCopy {
                family: "stability-ai-community",
            },
            LicenseTerm::DownstreamLicenseCopy {
                family: "apple-mlr",
            },
            // Same family, different kind.
            LicenseTerm::DownstreamRestrictions {
                family: "ltx-2-community",
            },
            LicenseTerm::DownstreamLicenseCopy {
                family: "ltx-2-community",
            },
            LicenseTerm::DownstreamRestrictions {
                family: "creativeml-openrail-pp-m",
            },
        ];

        let mut tags: Vec<&str> = SAMPLES.iter().copied().map(tag_of).collect();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(
            tags,
            vec![
                "acceptable_use_policy",
                "attribution_required",
                "deployer_obligation",
                "downstream_license_copy",
                "downstream_restrictions",
                "gated_access",
                "non_commercial_outputs",
                "non_commercial_weights",
                "notice_file_required",
                "registration_required",
                "revenue_ceiling",
            ],
            "every LicenseTerm variant needs a sample here"
        );
        // tag() and the exhaustive match agree, so the tripwire cannot rot into a second opinion.
        for term in SAMPLES {
            assert_eq!(term.tag(), tag_of(*term));
        }

        for a in SAMPLES {
            for b in SAMPLES {
                assert_eq!(
                    a.sort_key() == b.sort_key(),
                    a == b,
                    "sort_key must separate {a:?} from {b:?}"
                );
            }
        }

        // The same property one layer down, at the only layer a consumer actually reads: a payload
        // dropped on the way into JSON collapses two distinct terms into byte-identical objects
        // inside one `terms` array, even though `sort_key` kept them apart. `to_json` matches
        // exhaustively so a new variant cannot fall through, and this catches it anyway if someone
        // reintroduces a wildcard there.
        for a in SAMPLES {
            for b in SAMPLES {
                assert_eq!(
                    a.to_json() == b.to_json(),
                    a == b,
                    "to_json must separate {a:?} from {b:?}"
                );
            }
        }

        // The consequence, end to end: a sorted-then-deduped union keeps all of them.
        let mut union: Vec<LicenseTerm> = SAMPLES.to_vec();
        union.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
        union.dedup();
        assert_eq!(union.len(), SAMPLES.len());
    }

    /// The sc-16661 byte-stability guarantee, re-checked over the amended shape: none of the new
    /// payloads reintroduce an input-order dependency, and a gated component contributes the same
    /// bytes wherever it sits in the provider's list.
    #[test]
    fn amended_manifest_is_deterministic_across_input_order() {
        const FORWARD: &[ProviderComponents] = &[
            ProviderComponents {
                provider_id: "sd3_5_large",
                components: &["sd3_5_large_dit", "sdxl_unet"],
            },
            ProviderComponents {
                provider_id: "svd_then_ltx",
                components: &["svd_xt", "ltx_dit", "dfn5b_clip"],
            },
        ];
        const REVERSED: &[ProviderComponents] = &[
            ProviderComponents {
                provider_id: "svd_then_ltx",
                components: &["dfn5b_clip", "ltx_dit", "svd_xt"],
            },
            ProviderComponents {
                provider_id: "sd3_5_large",
                components: &["sdxl_unet", "sd3_5_large_dit"],
            },
        ];

        let forward = component_licenses_manifest_json(FAMILIES, COMPONENTS, FORWARD);
        let permuted_families: Vec<LicenseFamily> = FAMILIES.iter().rev().copied().collect();
        let permuted_components: Vec<ComponentLicense> = COMPONENTS.iter().rev().copied().collect();
        assert_eq!(
            forward,
            component_licenses_manifest_json(&permuted_families, &permuted_components, REVERSED),
            "permuting families, components, providers and a provider's own component list must \
             not change one byte of the manifest"
        );
        assert!(forward.ends_with("}\n"));
        let value: serde_json::Value = serde_json::from_str(&forward).unwrap();
        assert_eq!(value["schema_version"], 3);
        // Still no legal conclusion anywhere in the document.
        assert!(!forward.contains("commercial_use"));
    }
}
