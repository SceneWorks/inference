//! Machine-readable **model-weight license** surface (sc-13332).
//!
//! The crate/source license axis is already captured by the release tooling's SPDX SBOM
//! (`scripts/release/build_release.py`, one entry per resolved Cargo package). The license of a
//! provider's **model weights** — the pinned Hugging Face checkpoint each provider resolves at an
//! immutable revision — is a *separate axis* that cargo tooling never sees: Kokoro's weights are
//! Apache-2.0, Whisper's are Apache-2.0, OpenVoice's are MIT, and a checkpoint that lands later
//! (e.g. an MMAudio video→audio model) may be CC-BY-NC. SceneWorks is a **non-commercial** product,
//! so it may lawfully use non-commercially-licensed weights — but every weight license MUST be
//! surfaced so the product can list it on its licenses page (attribution is mandatory for CC-BY-*
//! and good practice for MIT/Apache too).
//!
//! This module is the tensor-free contract for that surface: each provider records a
//! [`WeightLicense`] as source-of-truth (it travels with the provider crate, next to its pinned
//! `HUB_REPO`/`HUB_REVISION`), a catalog aggregates the registered providers' licenses into
//! [`WeightLicenseEntry`] rows, and the release tooling serializes them into a model-licenses
//! manifest ([`weight_licenses_manifest_json`]) beside the SPDX SBOM so a consumer reads exactly
//! one file.
//!
//! ## Restriction discipline
//!
//! [`WeightLicense::commercial_use`] is the permissive flag. A `false` entry (CC-BY-NC,
//! research-only, or otherwise non-commercial) is admissible for the non-commercial product but
//! MUST also carry a human-readable [`WeightLicense::restriction`] note describing the terms the
//! product has to surface — [`WeightLicense::is_well_formed`] enforces that invariant so a
//! restricted checkpoint can never ship with its restriction unrecorded.
//!
//! ## Schema 3 supersedes the above (sc-16661)
//!
//! The current surface is the licence **table** further down this module — [`LicenseFamily`],
//! [`LicenseTerm`], [`ComponentLicense`], [`ProviderComponents`], the derived [`provider_terms`]
//! union and [`component_licenses_manifest_json`]. It records licence *facts* instead of the legal
//! *conclusion* `commercial_use` encodes, and keys rows by loaded artifact rather than by provider.
//! The v2 types in the next section — [`WeightLicense`], [`WeightLicenseEntry`] and the
//! `schema_version: 2` emitter [`weight_licenses_manifest_json`] — remain only until the audio lane
//! migrates (sc-16663) and the release tooling moves to schema 3 (sc-16664), which retires
//! [`weight_licenses_manifest_json`] in favour of [`component_licenses_manifest_json`].
//! **Do not add v2 callers.**

/// The license under which a provider's **model weights** (its pinned checkpoint) are distributed.
///
/// A separate axis from the crate/source license the SPDX SBOM records. Constructible without
/// loading weights — every field is `&'static str` / `bool` so the value is a `const` a provider
/// declares beside its pinned `HUB_REPO` / `HUB_REVISION`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightLicense {
    /// SPDX license identifier, e.g. `"Apache-2.0"`, `"MIT"`, `"CC-BY-NC-4.0"`.
    pub spdx_id: &'static str,
    /// Human-readable license name, e.g. `"Apache License 2.0"`.
    pub name: &'static str,
    /// The pinned checkpoint the license applies to — the Hugging Face repository URL (the
    /// provider also pins an immutable revision, recorded separately as its `HUB_REVISION`).
    pub source_url: &'static str,
    /// Attribution / copyright notice the license requires the product to surface (the Apache/MIT
    /// copyright line; the CC-BY-* attribution string). `None` only for a public-domain-equivalent
    /// dedication (CC0) that requires none.
    pub attribution: Option<&'static str>,
    /// Whether the weights may be used **commercially**. `false` flags a non-commercial
    /// (CC-BY-NC), research-only, or otherwise commercially-restricted checkpoint. SceneWorks is
    /// non-commercial, so `false` is admissible — but the product must surface the terms, so a
    /// `false` entry MUST also carry a [`restriction`](Self::restriction) note
    /// ([`is_well_formed`](Self::is_well_formed)).
    pub commercial_use: bool,
    /// A note carrying any additional restriction / terms the product must surface
    /// (non-commercial, research-only, gated / acceptable-use, or a mixed-component note). Required
    /// whenever [`commercial_use`](Self::commercial_use) is `false`; optional otherwise (e.g. to
    /// note that one sub-component of a checkpoint carries a different permissive license).
    pub restriction: Option<&'static str>,
}

impl WeightLicense {
    /// Whether the weights are permissively (commercially) usable.
    pub const fn is_permissive(&self) -> bool {
        self.commercial_use
    }

    /// Whether this record honors the restriction discipline: a non-commercial checkpoint
    /// (`commercial_use == false`) must carry a [`restriction`](Self::restriction) note, and the
    /// identity fields must be non-empty. The catalog ship-gate asserts this for every shipped
    /// provider so a restricted checkpoint can never ship with its terms unrecorded.
    pub fn is_well_formed(&self) -> bool {
        !self.spdx_id.is_empty()
            && !self.name.is_empty()
            && !self.source_url.is_empty()
            && (self.commercial_use || self.restriction.is_some())
    }
}

/// A `(provider_id, component, WeightLicense)` pairing — the aggregated unit a catalog surfaces and
/// the release tooling serializes into the model-licenses manifest. `provider_id` is the same stable
/// registry id the provider's descriptor advertises (e.g. `"kokoro_82m"`).
///
/// ## One provider, one *or more* rows (sc-13493)
///
/// A single-checkpoint provider contributes exactly one row with [`component`](Self::component) =
/// `None` — its license is both the attribution AND the effective restriction. A provider assembled
/// from **multiple** checkpoints (e.g. MMAudio's video→audio Foley, which pulls a CLIP conditioner,
/// a sync encoder, a DiT, a VAE and a vocoder — each under its own upstream license) contributes:
///
/// * one **composite** row (`component == None`) whose [`WeightLicense::restriction`] is the *effective*
///   (strictest-terms) restriction — the at-a-glance "can we use this provider" signal, and
/// * one **component** row per checkpoint (`component == Some(name)`) carrying that checkpoint's own
///   SPDX id, source URL and attribution — the per-upstream attribution obligation CC-BY-* imposes.
///
/// Rows are therefore keyed by the `(provider_id, component)` pair, not by `provider_id` alone. A
/// consumer that only wants the effective restriction reads the `component == None` row per provider;
/// a consumer building an attributions page reads every row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightLicenseEntry {
    /// The registry id of the provider these weights belong to (matches the provider descriptor's
    /// `id`).
    pub provider_id: &'static str,
    /// The checkpoint/component discriminator within a multi-checkpoint provider. `None` for a
    /// single-checkpoint provider's sole row, and for a multi-checkpoint provider's **composite**
    /// (effective-restriction) row; `Some(name)` for each per-checkpoint attribution row. The
    /// `(provider_id, component)` pair is the manifest's unique key.
    pub component: Option<&'static str>,
    /// The license of that provider's pinned weight checkpoint (or, for a composite row, the
    /// effective/strictest-terms license across the provider's checkpoints).
    pub license: WeightLicense,
}

/// Serialize weight-license entries into the canonical **model-licenses manifest** JSON — the file
/// the release tooling emits beside the SPDX SBOM and SceneWorks aggregates for its licenses page.
///
/// The output is deterministic: rows are sorted by the `(provider_id, component)` key (a provider's
/// composite `component == None` row sorts first, ahead of its `Some(_)` component rows), so the
/// committed manifest and the catalog-generated value compare byte-for-byte (the drift ship-gate)
/// regardless of the order providers were registered in. Every row carries a `component` field
/// (`null` for a single-checkpoint or composite row). A trailing newline is included so the file
/// matches `write_json`'s convention in the release tooling.
pub fn weight_licenses_manifest_json(entries: &[WeightLicenseEntry]) -> String {
    let mut sorted: Vec<&WeightLicenseEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| {
        a.provider_id
            .cmp(b.provider_id)
            .then_with(|| a.component.cmp(&b.component))
    });
    let providers: Vec<serde_json::Value> = sorted
        .iter()
        .map(|entry| {
            serde_json::json!({
                "provider_id": entry.provider_id,
                "component": entry.component,
                "spdx_id": entry.license.spdx_id,
                "license_name": entry.license.name,
                "source_url": entry.license.source_url,
                "commercial_use": entry.license.commercial_use,
                "attribution": entry.license.attribution,
                "restriction": entry.license.restriction,
            })
        })
        .collect();
    let document = serde_json::json!({
        "schema_version": 2,
        "kind": "model-weight-licenses",
        "providers": providers,
    });
    let mut rendered = serde_json::to_string_pretty(&document)
        .expect("weight-license manifest is always serializable");
    rendered.push('\n');
    rendered
}

// =================================================================================================
// v3 — licence families, typed terms, and per-component rows (sc-16661).
//
// The v2 surface above records a legal CONCLUSION (`WeightLicense::commercial_use`) that depends on
// facts inference does not have: the consumer's revenue, whether it redistributes weights or only
// sells renders, whether it registered with the upstream. Three checkpoints already in the media
// catalog have no correct boolean — FLUX.1-dev (weights non-commercial, outputs commercially
// usable), SD3.5 (commercial below $1M annual revenue only), and Kolors (commercial weights use
// only after registering with Kuaishou). Whichever value is written is silently wrong for half of
// callers, and a join computed over it reads as authoritative.
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
// The v2 types remain until the audio lane migrates (sc-16663); `commercial_use` occurs on 48 lines
// across 18 files in 13 crates (12 candle-audio providers plus candle-audio-catalog), so deleting it
// here would break the tree. sc-16664 additionally retires the `schema_version: 2` emitter
// [`weight_licenses_manifest_json`], moving the release tooling onto
// [`component_licenses_manifest_json`]. v2 is superseded, not supported — do not add callers.
// =================================================================================================

/// A typed obligation or condition imposed by a [`LicenseFamily`].
///
/// Deliberately **not** a permissive/restrictive flag. Each variant is a fact about the licence
/// text; whether a given use is permitted is the consumer's evaluation of the union of these
/// against its own situation.
///
/// ## Serialized order is keyed on [`tag`](Self::tag), not on declaration order
///
/// A derived `Ord` would make the emitted order of a [`provider_terms`] union a function of where
/// each variant happens to sit in this `enum`, so inserting a future variant mid-enum would silently
/// reorder every `terms` array in the manifest and trip the committed-manifest drift gate for
/// reasons unrelated to any licence change. `Ord` is therefore deliberately **not** derived: the
/// union is ordered by [`sort_key`](Self::sort_key) — the stable string [`tag`](Self::tag), then the
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
    /// Commercial use is permitted only below an annual-revenue ceiling. A union may carry more
    /// than one ceiling; every one binds, so a consumer at revenue `r` is clear only when `r` is
    /// below **all** of them. That is why these are not collapsed to a minimum here.
    RevenueCeiling {
        /// The ceiling in whole US dollars.
        amount_usd: u64,
    },
    /// Commercial use requires an out-of-band registration or approval with the upstream.
    RegistrationRequired {
        /// Where the registration is made (an email address or URL).
        contact: &'static str,
    },
    /// An acceptable-use / prohibited-use policy binds the deployer.
    AcceptableUsePolicy {
        /// The policy text.
        url: &'static str,
    },
    /// A concrete duty the deployer must discharge (e.g. content filtering on generated media).
    DeployerObligation {
        /// What the deployer must do, in the licence's own terms.
        text: &'static str,
    },
    /// The licence's restrictions must be passed to downstream recipients as enforceable
    /// provisions. Note that several licences impose this with materially different texts, so a
    /// union containing it more than once is not redundant at the product layer.
    DownstreamFlowDown,
    /// The weights are gated upstream — the terms must be accepted to obtain them at all.
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
            Self::DownstreamFlowDown => "downstream_flow_down",
            Self::GatedAccess => "gated_access",
        }
    }

    /// The total order the serialized union is emitted in: the stable [`tag`](Self::tag) first, then
    /// the variant's payload (numeric ceilings ascending, string payloads lexicographic). Two
    /// distinct terms never share a key, so sorting by it is deterministic without depending on the
    /// declaration order of the `enum` — see the type-level note.
    fn sort_key(&self) -> (&'static str, u64, &'static str) {
        match *self {
            Self::RevenueCeiling { amount_usd } => (self.tag(), amount_usd, ""),
            Self::RegistrationRequired { contact } => (self.tag(), 0, contact),
            Self::AcceptableUsePolicy { url } => (self.tag(), 0, url),
            Self::DeployerObligation { text } => (self.tag(), 0, text),
            Self::AttributionRequired
            | Self::NoticeFileRequired
            | Self::NonCommercialWeights
            | Self::NonCommercialOutputs
            | Self::DownstreamFlowDown
            | Self::GatedAccess => (self.tag(), 0, ""),
        }
    }

    fn to_json(self) -> serde_json::Value {
        let mut value = serde_json::json!({ "term": self.tag() });
        let object = value
            .as_object_mut()
            .expect("term is serialized as an object");
        match self {
            Self::RevenueCeiling { amount_usd } => {
                object.insert("amount_usd".into(), amount_usd.into());
            }
            Self::RegistrationRequired { contact } => {
                object.insert("contact".into(), contact.into());
            }
            Self::AcceptableUsePolicy { url } => {
                object.insert("url".into(), url.into());
            }
            Self::DeployerObligation { text } => {
                object.insert("text".into(), text.into());
            }
            _ => {}
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
    /// The typed obligations this licence imposes.
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
    /// The upstream artifact this row describes.
    pub source_url: &'static str,
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
    /// [`declared`](Self::declared)) are non-empty, [`family`](Self::family) resolves to a known
    /// [`LicenseFamily`], [`retrieved`](Self::retrieved) parses as a real ISO `YYYY-MM-DD` calendar
    /// date, [`attribution`](Self::attribution) is not `Some("")`, and a family imposing
    /// [`LicenseTerm::AttributionRequired`] implies a **non-empty**
    /// [`attribution`](Self::attribution).
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
        if self.source_url.is_empty() {
            errors.push(format!("component {key:?} has no source_url"));
        }
        if self.declared.is_empty() {
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
        // An `attribution: Some("")` is not an attribution — it satisfies nothing, and treating it
        // as present would let a CC-BY-* obligation ship unrecorded behind a placeholder.
        let attribution = self.attribution.filter(|text| !text.is_empty());
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
        if key.is_empty() {
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

/// The union of every term imposed on `provider` by the components it loads — sorted and
/// deduplicated, so it is deterministic and comparable.
///
/// **Derived, never hand-authored.** This is the "effective terms" answer a consumer joins over,
/// and computing it from the component rows is what keeps it from drifting away from them.
/// A component that does not resolve contributes nothing; use
/// [`license_table_conformance_errors`] to reject that state at the catalog boundary rather than
/// silently under-reporting here.
///
/// Ordering is by [`LicenseTerm::tag`] then payload, **not** by variant declaration order, so a
/// future variant inserted mid-`enum` cannot reorder an already-committed manifest.
pub fn provider_terms(
    provider: &ProviderComponents,
    components: &[ComponentLicense],
    families: &[LicenseFamily],
) -> Vec<LicenseTerm> {
    let mut terms: Vec<LicenseTerm> = provider
        .components
        .iter()
        .filter_map(|key| resolve_component(components, key))
        .filter_map(|row| resolve_family(families, row.family))
        .flat_map(|family| family.terms.iter().copied())
        .collect();
    terms.sort_by(|a, b| a.sort_key().cmp(&b.sort_key()));
    terms.dedup();
    terms
}

/// Every way the licence table can be malformed, as human-readable messages — the catalog ship-gate
/// asserts this is empty so no registered provider escapes a resolved, well-formed licence.
///
/// Checks, per section:
///
/// * **Families** — ids are non-empty and unique (a shadowed family would make an entire set of
///   obligations vanish from the [`provider_terms`] union depending on input order, since
///   [`resolve_family`] takes the first match); `spdx_id`, `name` and `text_url` are populated.
/// * **Components** — keys are non-empty and unique; `source_url` and `declared` are populated;
///   every `family` resolves; `retrieved` is a real ISO calendar date; an attribution-requiring
///   family implies a non-empty attribution, and `Some("")` never counts as one.
/// * **Providers** — `provider_id` is non-empty and unique (duplicates are not byte-stable under
///   the manifest's stable sort); every provider maps to at least one component; a provider does
///   not list the same component twice; every referenced component exists.
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
        if id.is_empty() {
            errors.push("licence family has an empty id".to_string());
        } else if seen_families.contains(&id) {
            errors.push(format!("duplicate licence family {id:?}"));
        } else {
            seen_families.push(id);
        }
        if family.spdx_id.is_empty() {
            errors.push(format!("licence family {id:?} has no spdx_id"));
        }
        if family.name.is_empty() {
            errors.push(format!("licence family {id:?} has no name"));
        }
        if family.text_url.is_empty() {
            errors.push(format!("licence family {id:?} has no text_url"));
        }
    }

    let mut seen: Vec<&str> = Vec::new();
    for row in components {
        let key = row.component;
        if key.is_empty() {
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
        if id.is_empty() {
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
mod tests {
    use super::*;

    const APACHE: WeightLicense = WeightLicense {
        spdx_id: "Apache-2.0",
        name: "Apache License 2.0",
        source_url: "https://huggingface.co/example/model",
        attribution: Some("© Example"),
        commercial_use: true,
        restriction: None,
    };

    #[test]
    fn permissive_entry_is_well_formed() {
        assert!(APACHE.is_well_formed());
        assert!(APACHE.is_permissive());
    }

    #[test]
    fn non_commercial_without_restriction_is_not_well_formed() {
        let nc = WeightLicense {
            spdx_id: "CC-BY-NC-4.0",
            name: "Creative Commons Attribution-NonCommercial 4.0",
            source_url: "https://huggingface.co/example/nc-model",
            attribution: Some("© Example"),
            commercial_use: false,
            restriction: None,
        };
        assert!(!nc.is_permissive());
        assert!(
            !nc.is_well_formed(),
            "a non-commercial license must record its restriction"
        );

        let fixed = WeightLicense {
            restriction: Some("Non-commercial use only (CC-BY-NC-4.0)."),
            ..nc
        };
        assert!(fixed.is_well_formed());
    }

    #[test]
    fn manifest_json_is_deterministic_and_sorted() {
        let entries = [
            WeightLicenseEntry {
                provider_id: "zeta",
                component: None,
                license: APACHE,
            },
            WeightLicenseEntry {
                provider_id: "alpha",
                component: None,
                license: APACHE,
            },
        ];
        let json = weight_licenses_manifest_json(&entries);
        // Stable across input order.
        let reversed = [
            WeightLicenseEntry {
                provider_id: "alpha",
                component: None,
                license: APACHE,
            },
            WeightLicenseEntry {
                provider_id: "zeta",
                component: None,
                license: APACHE,
            },
        ];
        assert_eq!(json, weight_licenses_manifest_json(&reversed));
        // Sorted: alpha precedes zeta.
        assert!(json.find("alpha").unwrap() < json.find("zeta").unwrap());
        // Trailing newline for file-write parity.
        assert!(json.ends_with("}\n"));
        // Parses back and carries the schema envelope.
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["schema_version"], 2);
        assert_eq!(value["kind"], "model-weight-licenses");
        assert_eq!(value["providers"].as_array().unwrap().len(), 2);
        // Every row carries a `component` field; a single-checkpoint row's is null.
        assert!(value["providers"][0]["component"].is_null());
        // A `None` optional serializes as JSON null (surfaced, not omitted).
        assert!(value["providers"][0]["restriction"].is_null());
    }

    #[test]
    fn multi_component_provider_emits_ordered_rows_with_composite_first() {
        // A provider assembled from two checkpoints: one composite (effective restriction) row and
        // two per-checkpoint attribution rows, all sharing the provider id (sc-13493).
        let composite = WeightLicense {
            spdx_id: "LicenseRef-Foo-composite",
            name: "Foo composite",
            source_url: "https://huggingface.co/example/foo",
            attribution: Some("Foo assembles two checkpoints"),
            commercial_use: false,
            restriction: Some("Research / non-commercial only — strictest of two components."),
        };
        let component_b = WeightLicense {
            spdx_id: "MIT",
            name: "MIT License",
            source_url: "https://github.com/example/b",
            attribution: Some("Component B © Example — MIT"),
            commercial_use: true,
            restriction: None,
        };
        let entries = [
            WeightLicenseEntry {
                provider_id: "foo",
                component: Some("b_checkpoint"),
                license: component_b,
            },
            WeightLicenseEntry {
                provider_id: "foo",
                component: None,
                license: composite,
            },
            WeightLicenseEntry {
                provider_id: "foo",
                component: Some("a_checkpoint"),
                license: APACHE,
            },
        ];
        let value: serde_json::Value =
            serde_json::from_str(&weight_licenses_manifest_json(&entries)).unwrap();
        let rows = value["providers"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        // Composite (component == null) sorts first, then components alphabetically.
        assert!(rows[0]["component"].is_null());
        assert_eq!(rows[0]["spdx_id"], "LicenseRef-Foo-composite");
        assert_eq!(rows[1]["component"], "a_checkpoint");
        assert_eq!(rows[2]["component"], "b_checkpoint");
        // All three share the provider id.
        assert!(rows.iter().all(|r| r["provider_id"] == "foo"));
    }
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
            // NonCommercialOutputs is deliberately absent.
            terms: &[
                LicenseTerm::NonCommercialWeights,
                LicenseTerm::GatedAccess,
                LicenseTerm::AcceptableUsePolicy {
                    url: "https://blackforestlabs.ai/aup",
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
            declared: "flux-1-dev-non-commercial-license",
            family: "flux-1-dev-non-commercial",
            attribution: None,
            retrieved: "2026-08-01",
        },
        ComponentLicense {
            component: "t5_xxl",
            source_url: "https://huggingface.co/google/t5-v1_1-xxl",
            declared: "apache-2.0",
            family: "apache-2-0",
            attribution: Some("T5 v1.1 © Google — Apache-2.0"),
            retrieved: "2026-08-01",
        },
        ComponentLicense {
            component: "arcface_antelopev2",
            source_url: "https://github.com/deepinsight/insightface/tree/master/model_zoo",
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
            terms: &[LicenseTerm::RevenueCeiling {
                amount_usd: 1_000_000,
            }],
        }];
        const ROWS: &[ComponentLicense] = &[ComponentLicense {
            component: "sd3_5_large_dit",
            source_url: "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
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
        let term = &value["providers"][0]["terms"][0];
        assert_eq!(term["term"], "revenue_ceiling");
        assert_eq!(term["amount_usd"], 1_000_000);
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
                terms: &[LicenseTerm::GatedAccess],
            },
        ];
        const ROWS: &[ComponentLicense] = &[ComponentLicense {
            component: "shadowed",
            source_url: "https://example.invalid/model",
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
            // NonCommercialWeights, …, GatedAccess (last). Tag order is alphabetical, which puts
            // gated_access second and notice_file_required last — a different sequence.
            terms: &[
                LicenseTerm::GatedAccess,
                LicenseTerm::NoticeFileRequired,
                LicenseTerm::AttributionRequired,
                LicenseTerm::NonCommercialWeights,
            ],
        }];
        const ROWS: &[ComponentLicense] = &[ComponentLicense {
            component: "mixed",
            source_url: "https://example.invalid/model",
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
                declared: "apache-2.0",
                family: "apache-2-0",
                attribution: Some("© Google"),
                retrieved: "2026-08-01",
            },
            ComponentLicense {
                component: "t5_xxl",
                source_url: "https://huggingface.co/google/t5-v1_1-xxl",
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

    /// Two ceilings in one union both bind — they are not collapsed to a minimum, so a consumer
    /// checking "am I below every ceiling" gets the right answer without special-casing.
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
                }],
            },
            LicenseFamily {
                id: "b",
                spdx_id: "LicenseRef-B",
                name: "B",
                text_url: "https://example.invalid/b",
                terms: &[LicenseTerm::RevenueCeiling {
                    amount_usd: 10_000_000,
                }],
            },
        ];
        const ROWS: &[ComponentLicense] = &[
            ComponentLicense {
                component: "a",
                source_url: "https://example.invalid/a",
                declared: "a",
                family: "a",
                attribution: None,
                retrieved: "2026-08-01",
            },
            ComponentLicense {
                component: "b",
                source_url: "https://example.invalid/b",
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
                    amount_usd: 1_000_000
                },
                LicenseTerm::RevenueCeiling {
                    amount_usd: 10_000_000
                },
            ]
        );
    }
}
