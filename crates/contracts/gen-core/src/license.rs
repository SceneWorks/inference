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
