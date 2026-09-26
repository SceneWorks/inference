//! The YuE2 licence policy (sc-22989; epic E2, E7).
//!
//! Four different sets of terms meet in the YuE2 closure, and they are kept apart here rather than
//! collapsed into one "YuE2 licence":
//!
//! 1. **First-party source** — `multimodal-art-projection/YuE` at the pinned commit is Apache-2.0
//!    ([`SOURCE_TERMS`]). It is the only source a native port of the YuE2 LM, VAE and text
//!    tokenizer may derive from. Its Oobleck VAE / SnakeBeta code carries two MIT notices of its
//!    own. It contains **no** SheetSage2 or MERT2 code: the cover closure's only code is the remote
//!    Python inside those model repositories, which carries no code licence, so code derived from it
//!    is treated as CC BY-NC 4.0 and a port is gated until an owner records a basis
//!    ([`CODE_TERMS`]).
//! 2. **Model weights** — YuE2-3B, both VAEs, SheetSage2 and MERT-v2-FullSong are CC BY-NC 4.0
//!    ([`COMPONENT_LICENSES`]): noncommercial use only, with attribution.
//! 3. **`qwen.tiktoken`** — byte-identical to the Qwen-7B tokenizer file, so it carries the Tongyi
//!    Qianwen License Agreement; YuE2's `MODEL_LICENSE` says the weight licence "does not replace
//!    separately applicable licenses for … text tokenization files".
//! 4. **Bundled archive and third-party terms** — the earlier `yue2-v0.1.6` release archives and the
//!    wheels in the YuE2-3B repository license their *code* under CC BY-NC 4.0 (not Apache-2.0), and
//!    SheetSage2's rendering assets carry MIT / font / CC BY 3.0 US terms ([`BUNDLED_TERMS`]). None
//!    of it is in a closure, and none of it is an ungated port source.
//!
//! # The gate
//!
//! Unlike the gen-core licence surface — which is disclosure only by design — this module decides:
//! [`authorize`] permits an [`IntendedUse`] only when **every** component involved has a recorded
//! compatible basis for it ([`UseDisposition::Permitted`]), and otherwise returns [`UseRefused`]
//! naming each gated component, why, and what would unblock it. Recorded today:
//!
//! | Use | Weights (CC BY-NC 4.0) | `qwen.tiktoken` (Tongyi Qianwen) |
//! | --- | --- | --- |
//! | [`IntendedUse::NoncommercialExperimentation`] | permitted, CC BY-NC §2(a)(1) | permitted, Tongyi §2 |
//! | [`IntendedUse::CommercialUse`] | **gated** — the licence grants NonCommercial use only | **gated** — no reviewed basis |
//! | [`IntendedUse::Redistribution`] | **gated** — no owner-recorded distribution basis | **gated** — no owner-recorded basis |
//!
//! Commercial routes therefore can never run YuE2 (they use YuE1), and nothing may rehost, bundle or
//! share a YuE2 file until an owner records a basis here. The gate never strips an upstream access
//! gate and never relabels a licence.
//!
//! Every licence text this module cites is vendored verbatim under the crate's `licenses/`
//! directory and pinned by SHA-256 in [`LICENSE_TEXTS`].

use crate::gen_core;
use crate::inventory::{self, Closure, ComponentId};

/// What a caller intends to do with the closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntendedUse {
    /// Local, explicitly selected, noncommercial experimentation: generating, transcribing and
    /// editing on the user's own machine with snapshots the user acquired from the upstream.
    NoncommercialExperimentation,
    /// Any commercial use of the models or of what they produce (a commercial-use route).
    CommercialUse,
    /// Rehosting, bundling or otherwise sharing any YuE2 file with others.
    Redistribution,
}

impl IntendedUse {
    /// Every intended use.
    pub const ALL: [IntendedUse; 3] = [
        IntendedUse::NoncommercialExperimentation,
        IntendedUse::CommercialUse,
        IntendedUse::Redistribution,
    ];
}

/// A recorded compatible basis: the licence clause that permits a use, and where its text is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Basis {
    /// The [`gen_core::LicenseFamily::id`] the clause belongs to.
    pub family: &'static str,
    /// The clause, quoted.
    pub clause: &'static str,
    /// The vendored licence text, relative to the crate root (an entry of [`LICENSE_TEXTS`]).
    pub evidence: &'static str,
}

/// Whether a use is permitted for one component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UseDisposition {
    /// Permitted on a recorded basis.
    Permitted(Basis),
    /// Gated: no compatible basis is recorded.
    Gated {
        /// Why the use is gated.
        reason: &'static str,
        /// What would have to be recorded (by the owner) to unblock it.
        unblock: &'static str,
    },
}

/// The policy for one component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentPolicy {
    /// The component.
    pub component: ComponentId,
    /// Its schema-3 licence row (declaration, family, attribution, provenance).
    pub license: gen_core::ComponentLicense,
    /// Vendored licence / notice texts that must accompany any copy (entries of [`LICENSE_TEXTS`]).
    pub texts: &'static [&'static str],
    /// A licence fact about this component that the row cannot carry, if any.
    pub note: Option<&'static str>,
    /// [`IntendedUse::NoncommercialExperimentation`].
    pub noncommercial_experimentation: UseDisposition,
    /// [`IntendedUse::CommercialUse`].
    pub commercial_use: UseDisposition,
    /// [`IntendedUse::Redistribution`].
    pub redistribution: UseDisposition,
}

impl ComponentPolicy {
    /// The disposition of `intended` for this component.
    pub fn disposition(&self, intended: IntendedUse) -> UseDisposition {
        match intended {
            IntendedUse::NoncommercialExperimentation => self.noncommercial_experimentation,
            IntendedUse::CommercialUse => self.commercial_use,
            IntendedUse::Redistribution => self.redistribution,
        }
    }
}

/// A permitted use of a set of components.
///
/// A capability, not a record: its fields are private, so [`authorize`] (and
/// [`authorize_closure`], which calls it) is the **only** way to obtain one. Code that requires an
/// `Authorization` for, say, [`IntendedUse::CommercialUse`] therefore cannot be handed a forged
/// one. A struct literal outside this crate does not compile:
///
/// ```compile_fail
/// use candle_audio_yue2::license::{Authorization, IntendedUse};
///
/// let forged = Authorization {
///     intended: IntendedUse::CommercialUse,
///     grants: Vec::new(),
///     attributions: Vec::new(),
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Authorization {
    intended: IntendedUse,
    grants: Vec<(ComponentId, Basis)>,
    attributions: Vec<&'static str>,
}

impl Authorization {
    /// The use that was authorized.
    pub fn intended(&self) -> IntendedUse {
        self.intended
    }

    /// The basis each component was permitted on, in the order the components were named.
    pub fn grants(&self) -> &[(ComponentId, Basis)] {
        &self.grants
    }

    /// The attributions the components' licences require, deduplicated, in component order — to be
    /// shown to the user and retained in provenance and exports.
    pub fn attributions(&self) -> &[&'static str] {
        &self.attributions
    }
}

/// A refused use.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UseRefused {
    /// No component was named, so there is nothing a basis could have been recorded for.
    #[error("{intended:?} is not permitted for YuE2: no components were named")]
    NoComponents {
        /// The use that was refused.
        intended: IntendedUse,
    },
    /// At least one named component has no recorded compatible basis for the use.
    #[error("{intended:?} is not permitted for YuE2: {}", describe(gated))]
    Gated {
        /// The use that was refused.
        intended: IntendedUse,
        /// `(component, reason, unblock)` for every gated component.
        gated: Vec<(ComponentId, &'static str, &'static str)>,
    },
}

impl UseRefused {
    /// The use that was refused.
    pub fn intended(&self) -> IntendedUse {
        match self {
            UseRefused::NoComponents { intended } | UseRefused::Gated { intended, .. } => *intended,
        }
    }
}

fn describe(gated: &[(ComponentId, &'static str, &'static str)]) -> String {
    gated
        .iter()
        .map(|(c, reason, unblock)| format!("{c:?}: {reason} (unblock: {unblock})"))
        .collect::<Vec<_>>()
        .join("; ")
}

/// Permit `intended` for `components` only if every one of them has a recorded compatible basis.
/// An empty component list is refused: there is nothing a basis could have been recorded for.
pub fn authorize(
    components: &[ComponentId],
    intended: IntendedUse,
) -> Result<Authorization, UseRefused> {
    if components.is_empty() {
        return Err(UseRefused::NoComponents { intended });
    }
    let mut grants = Vec::new();
    let mut gated = Vec::new();
    let mut attributions: Vec<&'static str> = Vec::new();
    for &id in components {
        let p = policy(id);
        match p.disposition(intended) {
            UseDisposition::Permitted(basis) => {
                grants.push((id, basis));
                if let Some(a) = p.license.attribution {
                    if !attributions.contains(&a) {
                        attributions.push(a);
                    }
                }
            }
            UseDisposition::Gated { reason, unblock } => gated.push((id, reason, unblock)),
        }
    }
    if gated.is_empty() {
        Ok(Authorization {
            intended,
            grants,
            attributions,
        })
    } else {
        Err(UseRefused::Gated { intended, gated })
    }
}

/// [`authorize`] over every component of `closure`.
pub fn authorize_closure(
    closure: Closure,
    intended: IntendedUse,
) -> Result<Authorization, UseRefused> {
    authorize(closure.components(), intended)
}

/// The policy for component `id`.
pub fn policy(id: ComponentId) -> &'static ComponentPolicy {
    POLICIES
        .iter()
        .find(|p| p.component == id)
        .expect("every component has a policy (checked by tests)")
}

// -------------------------------------------------------------------------------------------------
// Licence rows (gen-core schema 3). Declarations read from each card on `inventory::RETRIEVED`.
// -------------------------------------------------------------------------------------------------

/// Licence row for `m-a-p/YuE2-3B`.
pub const LICENSE_YUE2_3B: gen_core::ComponentLicense = gen_core::ComponentLicense {
    component: "yue2_3b",
    source_url: "https://huggingface.co/m-a-p/YuE2-3B",
    gated: false,
    declared: "cc-by-nc-4.0",
    family: "cc-by-nc-4-0",
    attribution: Some(
        "YuE2-3B — YuE2 by the YuE2 authors (Multimodal Art Projection), \
         https://huggingface.co/m-a-p/YuE2-3B — weights licensed under CC BY-NC 4.0 \
         (https://creativecommons.org/licenses/by-nc/4.0/)",
    ),
    retrieved: inventory::RETRIEVED,
};

/// Licence row for `qwen.tiktoken` — declared by its **origin** (Qwen/Qwen-7B, byte-identical),
/// not by the redistributing YuE2-3B card, whose `MODEL_LICENSE` excludes tokenization files.
pub const LICENSE_QWEN_TIKTOKEN: gen_core::ComponentLicense = gen_core::ComponentLicense {
    component: "yue2_qwen_tiktoken",
    source_url: "https://huggingface.co/Qwen/Qwen-7B",
    gated: false,
    declared: "tongyi-qianwen-license-agreement",
    family: "tongyi-qianwen",
    attribution: Some(
        "Tongyi Qianwen is licensed under the Tongyi Qianwen LICENSE AGREEMENT, Copyright (c) \
         Alibaba Cloud. All Rights Reserved.",
    ),
    retrieved: inventory::RETRIEVED,
};

/// Licence row for `m-a-p/YuE2-Vae`.
pub const LICENSE_YUE2_VAE: gen_core::ComponentLicense = gen_core::ComponentLicense {
    component: "yue2_vae",
    source_url: "https://huggingface.co/m-a-p/YuE2-Vae",
    gated: false,
    declared: "cc-by-nc-4.0",
    family: "cc-by-nc-4-0",
    attribution: Some(
        "YuE2-Vae — YuE2 by the YuE2 authors (Multimodal Art Projection), \
         https://huggingface.co/m-a-p/YuE2-Vae — weights licensed under CC BY-NC 4.0 \
         (https://creativecommons.org/licenses/by-nc/4.0/)",
    ),
    retrieved: inventory::RETRIEVED,
};

/// Licence row for `m-a-p/YuE2-Vae-legacy`.
pub const LICENSE_YUE2_VAE_LEGACY: gen_core::ComponentLicense = gen_core::ComponentLicense {
    component: "yue2_vae_legacy",
    source_url: "https://huggingface.co/m-a-p/YuE2-Vae-legacy",
    gated: false,
    declared: "cc-by-nc-4.0",
    family: "cc-by-nc-4-0",
    attribution: Some(
        "YuE2-Vae-legacy — YuE2 by the YuE2 authors (Multimodal Art Projection), \
         https://huggingface.co/m-a-p/YuE2-Vae-legacy — weights licensed under CC BY-NC 4.0 \
         (https://creativecommons.org/licenses/by-nc/4.0/)",
    ),
    retrieved: inventory::RETRIEVED,
};

/// Licence row for `m-a-p/SheetSage2`.
pub const LICENSE_SHEETSAGE2: gen_core::ComponentLicense = gen_core::ComponentLicense {
    component: "yue2_sheetsage2",
    source_url: "https://huggingface.co/m-a-p/SheetSage2",
    gated: false,
    declared: "cc-by-nc-4.0",
    family: "cc-by-nc-4-0",
    attribution: Some(
        "SheetSage2 by Multimodal Art Projection, https://huggingface.co/m-a-p/SheetSage2 \
         (adapts MERT-v2-FullSong) — weights licensed under CC BY-NC 4.0 \
         (https://creativecommons.org/licenses/by-nc/4.0/)",
    ),
    retrieved: inventory::RETRIEVED,
};

/// Licence row for `m-a-p/MERT-v2-FullSong`.
pub const LICENSE_MERT_V2_FULLSONG: gen_core::ComponentLicense = gen_core::ComponentLicense {
    component: "yue2_mert_v2_fullsong",
    source_url: "https://huggingface.co/m-a-p/MERT-v2-FullSong",
    gated: false,
    declared: "cc-by-nc-4.0",
    family: "cc-by-nc-4-0",
    attribution: Some(
        "MERT-v2-FullSong — MERT2 by Multimodal Art Projection, \
         https://huggingface.co/m-a-p/MERT-v2-FullSong — weights licensed under CC BY-NC 4.0 \
         (https://creativecommons.org/licenses/by-nc/4.0/)",
    ),
    retrieved: inventory::RETRIEVED,
};

/// Every artifact the YuE2 closures load — one row each, in [`ComponentId::ALL`] order. For the
/// catalog registration (a later slice); nothing registers a provider yet.
pub const COMPONENT_LICENSES: &[gen_core::ComponentLicense] = &[
    LICENSE_YUE2_3B,
    LICENSE_QWEN_TIKTOKEN,
    LICENSE_YUE2_VAE,
    LICENSE_YUE2_VAE_LEGACY,
    LICENSE_SHEETSAGE2,
    LICENSE_MERT_V2_FULLSONG,
];

// -------------------------------------------------------------------------------------------------
// Vendored licence texts.
// -------------------------------------------------------------------------------------------------

/// A licence or notice text vendored verbatim under the crate's `licenses/` directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LicenseText {
    /// Path relative to the crate root.
    pub path: &'static str,
    /// SHA-256 of the vendored bytes (equal to the upstream file's).
    pub sha256: &'static str,
    /// Where the bytes came from.
    pub origin: &'static str,
}

/// Every vendored licence / notice text. The YuE2-weights and MERT2 entries are byte-identical to
/// the `LICENSE` / notice files pinned in the inventory, so a verified snapshot proves the copy it
/// ships beside the weights is this text.
pub const LICENSE_TEXTS: &[LicenseText] = &[
    LicenseText {
        path: "licenses/yue2-source/LICENSE",
        sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        origin: "github.com/multimodal-art-projection/YuE@92a73cc7652fcc1f937855e4b765e0a0edd7ff2e:LICENSE",
    },
    LicenseText {
        path: "licenses/yue2-source/MODEL_LICENSE",
        sha256: "623b5b238853e2ab79d87933ee5ea7ff474ec31e348f128ca4063cda64b901bc",
        origin: "github.com/multimodal-art-projection/YuE@92a73cc7652fcc1f937855e4b765e0a0edd7ff2e:MODEL_LICENSE",
    },
    LicenseText {
        path: "licenses/yue2-source/THIRD_PARTY_NOTICES.md",
        sha256: "395f9073249ef34782e5ec16bbbae4ff3a49226d506ee8e166806024de00386b",
        origin: "github.com/multimodal-art-projection/YuE@92a73cc7652fcc1f937855e4b765e0a0edd7ff2e:THIRD_PARTY_NOTICES.md",
    },
    LicenseText {
        path: "licenses/yue2-source/licenses/SnakeBeta-NVIDIA-MIT.txt",
        sha256: "e2f86126a8a56dfd0882e9112e282c7599a34d2c4479f6f5c461b8c2121a579f",
        origin: "github.com/multimodal-art-projection/YuE@92a73cc7652fcc1f937855e4b765e0a0edd7ff2e:licenses/SnakeBeta-NVIDIA-MIT.txt",
    },
    LicenseText {
        path: "licenses/yue2-source/licenses/stable-audio-tools-MIT.txt",
        sha256: "a1fac33b7bcd791b74fb33aeb439f825e7277e239fc119fb7d2ab6f084a0c101",
        origin: "github.com/multimodal-art-projection/YuE@92a73cc7652fcc1f937855e4b765e0a0edd7ff2e:licenses/stable-audio-tools-MIT.txt",
    },
    LicenseText {
        path: "licenses/yue2-weights/LICENSE",
        sha256: "060985741d20e70613b4c189c7de106cabd3fb2109fbbfb6d705c9d619417dd0",
        origin: "huggingface.co/m-a-p/YuE2-3B, YuE2-Vae, YuE2-Vae-legacy (pinned revisions):LICENSE",
    },
    LicenseText {
        path: "licenses/yue2-weights/THIRD_PARTY_NOTICES.md",
        sha256: "14d3fd9f6fee86b4260b69b0979735b99ffec8c6b4db254567047a4215cd9af3",
        origin: "huggingface.co/m-a-p/YuE2-3B, YuE2-Vae, YuE2-Vae-legacy (pinned revisions):THIRD_PARTY_NOTICES.md",
    },
    LicenseText {
        path: "licenses/yue2-weights/licenses/SnakeBeta-NVIDIA-MIT.txt",
        sha256: "da9858d516047d82096d01c112a61bd67f26d289039464d668a1d45f91738ecc",
        origin: "huggingface.co/m-a-p/YuE2-3B, YuE2-Vae, YuE2-Vae-legacy (pinned revisions):licenses/SnakeBeta-NVIDIA-MIT.txt",
    },
    LicenseText {
        path: "licenses/yue2-weights/licenses/stable-audio-tools-MIT.txt",
        sha256: "a1fac33b7bcd791b74fb33aeb439f825e7277e239fc119fb7d2ab6f084a0c101",
        origin: "huggingface.co/m-a-p/YuE2-3B, YuE2-Vae, YuE2-Vae-legacy (pinned revisions):licenses/stable-audio-tools-MIT.txt",
    },
    LicenseText {
        path: "licenses/mert2-weights/LICENSE",
        sha256: "73593d6cad4ca80f3ab1448a4908f92ab802dfffe029561e3b12fa7d07c8b87c",
        origin: "huggingface.co/m-a-p/MERT-v2-FullSong and m-a-p/SheetSage2 (pinned revisions):LICENSE",
    },
    LicenseText {
        path: "licenses/sheetsage2/THIRD_PARTY_NOTICES.md",
        sha256: "ab8b195c2c1475f254a04181a9841db2f2581c579b95858c78ee0040eb495a29",
        origin: "huggingface.co/m-a-p/SheetSage2@eab522a8168e8b8b8c4856bf8609cd86198f01fe:THIRD_PARTY_NOTICES.md",
    },
    LicenseText {
        path: "licenses/mert-v2-fullsong/THIRD_PARTY_NOTICES.md",
        sha256: "ce8fd578969f1bdad9a0be5e20cab0251dc4253bd4dbfaa9abeb1c11d68bafdc",
        origin: "huggingface.co/m-a-p/MERT-v2-FullSong@d8ba1c745e733b3908ce6ad16ebeb17ac7600a42:THIRD_PARTY_NOTICES.md",
    },
    LicenseText {
        path: "licenses/qwen-tiktoken/LICENSE",
        sha256: "7c7b8e244f6aa1ac8c32b74f56d42c41a0364dd2dabed8d9c6030a862e805b54",
        origin: "huggingface.co/Qwen/Qwen-7B@ef3c5c9c57b252f3149c1408daf4d649ec8b6c85:LICENSE",
    },
    LicenseText {
        path: "licenses/qwen-tiktoken/NOTICE",
        sha256: "3187bff5c804c33b314509ecdee669e33d90d1ece2fbd93d5fdbd8a9926b7a59",
        origin: "huggingface.co/Qwen/Qwen-7B@ef3c5c9c57b252f3149c1408daf4d649ec8b6c85:NOTICE",
    },
];

// -------------------------------------------------------------------------------------------------
// Per-component policy.
// -------------------------------------------------------------------------------------------------

const CC_BY_NC_GRANT: &str =
    "§2(a)(1): the Licensor grants a licence to \"reproduce and Share the \
                              Licensed Material, in whole or in part, for NonCommercial purposes \
                              only\"";

const YUE2_WEIGHTS_TEXTS: &[&str] = &[
    "licenses/yue2-weights/LICENSE",
    "licenses/yue2-weights/THIRD_PARTY_NOTICES.md",
    "licenses/yue2-weights/licenses/SnakeBeta-NVIDIA-MIT.txt",
    "licenses/yue2-weights/licenses/stable-audio-tools-MIT.txt",
];

const fn cc_by_nc_noncommercial(evidence: &'static str) -> UseDisposition {
    UseDisposition::Permitted(Basis {
        family: "cc-by-nc-4-0",
        clause: CC_BY_NC_GRANT,
        evidence,
    })
}

const CC_BY_NC_COMMERCIAL: UseDisposition = UseDisposition::Gated {
    reason: "the weights are CC BY-NC 4.0, which licenses NonCommercial purposes only; commercial \
             routes must use YuE1",
    unblock: "a separate written licence from the rights holder (Multimodal Art Projection) \
              permitting commercial use of these weights, recorded here",
};

const CC_BY_NC_REDISTRIBUTION: UseDisposition = UseDisposition::Gated {
    reason: "no SceneWorks distribution basis is recorded; CC BY-NC 4.0 permits Sharing only for \
             NonCommercial purposes with §3(a) attribution, and whether a given distribution is \
             NonCommercial is an owner decision",
    unblock: "an owner-recorded compatible distribution basis; any copy then ships the pinned \
              LICENSE / THIRD_PARTY_NOTICES.md / licenses/ unmodified with the upstream revision \
              stamp, keeps any upstream access gate, and is never relicensed",
};

/// The per-component policies, in [`ComponentId::ALL`] order.
pub const POLICIES: &[ComponentPolicy] = &[
    ComponentPolicy {
        component: ComponentId::Lm,
        license: LICENSE_YUE2_3B,
        texts: YUE2_WEIGHTS_TEXTS,
        note: None,
        noncommercial_experimentation: cc_by_nc_noncommercial("licenses/yue2-weights/LICENSE"),
        commercial_use: CC_BY_NC_COMMERCIAL,
        redistribution: CC_BY_NC_REDISTRIBUTION,
    },
    ComponentPolicy {
        component: ComponentId::QwenTiktoken,
        license: LICENSE_QWEN_TIKTOKEN,
        texts: &["licenses/qwen-tiktoken/LICENSE", "licenses/qwen-tiktoken/NOTICE"],
        note: Some(
            "byte-identical (git blob 9b9b0e0416d84d7c88333eb261c77e5fe2d7f7be) to \
             Qwen/Qwen-7B's qwen.tiktoken. m-a-p/YuE2-3B redistributes it under its card tag \
             cc-by-nc-4.0 but ships neither the Tongyi Qianwen Agreement nor its §3(c) Notice, and \
             YuE2's MODEL_LICENSE states the weight licence does not replace separately applicable \
             licences for text tokenization files",
        ),
        noncommercial_experimentation: UseDisposition::Permitted(Basis {
            family: "tongyi-qianwen",
            clause: "§2: \"You are granted a non-exclusive, worldwide, non-transferable and \
                     royalty-free limited license … to use, reproduce, distribute, copy, create \
                     derivative works of, and make modifications to the Materials\"",
            evidence: "licenses/qwen-tiktoken/LICENSE",
        }),
        commercial_use: UseDisposition::Gated {
            reason: "no commercial-use basis is recorded: Tongyi Qianwen §4 (commercial use above \
                     100 million monthly active users) and §5(b) (no use to improve other large \
                     language models) have not been reviewed for this product — and the YuE2 \
                     weights it serves are noncommercial regardless",
            unblock: "an owner review of Tongyi Qianwen §4 and §5(b) recorded here",
        },
        redistribution: UseDisposition::Gated {
            reason: "no distribution basis is recorded; Tongyi Qianwen §3 requires a copy of the \
                     Agreement and a Notice file carrying the §3(c) attribution with every copy, \
                     which the upstream YuE2-3B copy does not ship",
            unblock: "an owner-recorded basis; any copy then ships licenses/qwen-tiktoken/LICENSE \
                      and a Notice file with the §3(c) attribution",
        },
    },
    ComponentPolicy {
        component: ComponentId::VaeStandard,
        license: LICENSE_YUE2_VAE,
        texts: YUE2_WEIGHTS_TEXTS,
        note: None,
        noncommercial_experimentation: cc_by_nc_noncommercial("licenses/yue2-weights/LICENSE"),
        commercial_use: CC_BY_NC_COMMERCIAL,
        redistribution: CC_BY_NC_REDISTRIBUTION,
    },
    ComponentPolicy {
        component: ComponentId::VaeLegacy,
        license: LICENSE_YUE2_VAE_LEGACY,
        texts: YUE2_WEIGHTS_TEXTS,
        note: None,
        noncommercial_experimentation: cc_by_nc_noncommercial("licenses/yue2-weights/LICENSE"),
        commercial_use: CC_BY_NC_COMMERCIAL,
        redistribution: CC_BY_NC_REDISTRIBUTION,
    },
    ComponentPolicy {
        component: ComponentId::SheetSage2,
        license: LICENSE_SHEETSAGE2,
        texts: &[
            "licenses/mert2-weights/LICENSE",
            "licenses/sheetsage2/THIRD_PARTY_NOTICES.md",
        ],
        note: Some(
            "the card declares cc-by-nc-4.0, but the LICENSE file the repository ships is the \
             MERT2 weight licence verbatim (byte-identical to MERT-v2-FullSong's): its scope \
             preamble names only the MERT2-30s and MERT2-FS checkpoints, not SheetSage2. The CC BY-NC \
             4.0 text that follows is unmodified, so the card tag is the SheetSage2 declaration",
        ),
        noncommercial_experimentation: cc_by_nc_noncommercial("licenses/mert2-weights/LICENSE"),
        commercial_use: CC_BY_NC_COMMERCIAL,
        redistribution: CC_BY_NC_REDISTRIBUTION,
    },
    ComponentPolicy {
        component: ComponentId::MertV2FullSong,
        license: LICENSE_MERT_V2_FULLSONG,
        texts: &[
            "licenses/mert2-weights/LICENSE",
            "licenses/mert-v2-fullsong/THIRD_PARTY_NOTICES.md",
        ],
        note: None,
        noncommercial_experimentation: cc_by_nc_noncommercial("licenses/mert2-weights/LICENSE"),
        commercial_use: CC_BY_NC_COMMERCIAL,
        redistribution: CC_BY_NC_REDISTRIBUTION,
    },
];

// -------------------------------------------------------------------------------------------------
// Source and bundled terms.
// -------------------------------------------------------------------------------------------------

/// A third-party notice the first-party source carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThirdPartyNotice {
    /// What the notice covers.
    pub covers: &'static str,
    /// SPDX id.
    pub spdx: &'static str,
    /// The copyright line to retain.
    pub copyright: &'static str,
    /// The vendored text (an entry of [`LICENSE_TEXTS`]).
    pub text: &'static str,
}

/// The licence terms of the pinned first-party source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceTerms {
    /// The repository.
    pub repo: &'static str,
    /// The pinned commit.
    pub commit: &'static str,
    /// SPDX id of the first-party source at that commit.
    pub spdx: &'static str,
    /// The copyright line to retain.
    pub copyright: &'static str,
    /// The vendored licence text.
    pub license_text: &'static str,
    /// The vendored `MODEL_LICENSE`, which scopes the weight licence away from code.
    pub model_license_text: &'static str,
    /// Third-party code notices that travel with any port of the covered code.
    pub third_party: &'static [ThirdPartyNotice],
    /// The rule native ports follow.
    pub port_rule: &'static str,
}

/// The pinned YuE2 GitHub source: Apache-2.0.
pub const SOURCE_TERMS: SourceTerms = SourceTerms {
    repo: inventory::YUE2_SOURCE_REPO,
    commit: inventory::YUE2_SOURCE_COMMIT,
    spdx: "Apache-2.0",
    copyright: "Copyright (c) 2026 the YuE2 authors",
    license_text: "licenses/yue2-source/LICENSE",
    model_license_text: "licenses/yue2-source/MODEL_LICENSE",
    third_party: &[
        ThirdPartyNotice {
            covers: "the Oobleck VAE implementation in src/yue2/modeling_vae.py (derived from \
                     stable-audio-tools a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45)",
            spdx: "MIT",
            copyright: "Copyright (c) 2023 Stability AI",
            text: "licenses/yue2-source/licenses/stable-audio-tools-MIT.txt",
        },
        ThirdPartyNotice {
            covers: "the SnakeBeta activation in src/yue2/modeling_vae.py (from BigVGAN)",
            spdx: "MIT",
            copyright: "Copyright (c) 2022 NVIDIA CORPORATION",
            text: "licenses/yue2-source/licenses/SnakeBeta-NVIDIA-MIT.txt",
        },
    ],
    port_rule: "native ports of the YuE2 language model (src/yue2/modeling_yue2.py), VAE \
                (src/yue2/modeling_vae.py) and text tokenizer (src/yue2/tokenization_yue2.py) derive \
                only from this repository at this commit (Apache-2.0), retaining its copyright line \
                and the two MIT notices for any ported VAE / SnakeBeta code; never from the \
                yue2-v0.1.6 release archives, the wheels in m-a-p/YuE2-3B, or the remote-code copies \
                in the YuE2 model repositories, whose bundled terms differ. This repository contains \
                no SheetSage2 or MERT2 code: the cover closure's code terms and port disposition are \
                recorded separately in CODE_TERMS",
};

/// The terms of the code a native port of one component would derive from, and whether porting it
/// into this Apache-2.0 crate has a recorded basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CodeTerms {
    /// The component the code implements.
    pub component: ComponentId,
    /// The upstream code a port derives from.
    pub code: &'static str,
    /// Where that code is published, at which revision.
    pub location: &'static str,
    /// The terms the code carries, as inspected on [`inventory::RETRIEVED`].
    pub terms: &'static str,
    /// SPDX id that code derived from it is treated as carrying.
    pub treated_as: &'static str,
    /// Whether a native port may be written into this crate (Apache-2.0): permitted on a recorded
    /// basis, or gated with its unblock condition.
    pub port: UseDisposition,
}

const APACHE_PORT: UseDisposition = UseDisposition::Permitted(Basis {
    family: "apache-2-0",
    clause: "§2: \"each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, \
             no-charge, royalty-free, irrevocable copyright license to reproduce, prepare \
             Derivative Works of, publicly display, publicly perform, sublicense, and distribute \
             the Work and such Derivative Works\"",
    evidence: "licenses/yue2-source/LICENSE",
});

const COVER_PORT: UseDisposition = UseDisposition::Gated {
    reason: "the only SheetSage2 / MERT2 code is the remote Python published inside the model \
             repositories, which carries no code licence of its own; the repositories declare \
             cc-by-nc-4.0, so code derived from it is treated as CC BY-NC 4.0 — noncommercial — \
             and must not be relicensed into this Apache-2.0 crate. The native port (sc-22996) \
             therefore lives in the separate workspace crate `candle-audio-sheetsage2`, whose \
             Cargo licence is CC-BY-NC-4.0 (admitted by one scoped `deny.toml` exception); it is \
             not composed into the audio catalog or any runtime bundle",
    unblock: "an owner-recorded basis for the cover port, recorded here: an explicit code licence \
              from the rights holder (Multimodal Art Projection), or an owner decision to ship the \
              derived code under CC BY-NC 4.0 with attribution, outside the crate's Apache-2.0 \
              grant",
};

/// Per-component code terms, in [`ComponentId::ALL`] order. Every component a native port will be
/// written for has an entry, so no port can land without its code terms being on record.
pub const CODE_TERMS: &[CodeTerms] = &[
    CodeTerms {
        component: ComponentId::Lm,
        code: "src/yue2/modeling_yue2.py",
        location: "github.com/multimodal-art-projection/YuE@92a73cc7652fcc1f937855e4b765e0a0edd7ff2e",
        terms: "Apache-2.0 (root LICENSE, pyproject license = \"Apache-2.0\")",
        treated_as: "Apache-2.0",
        port: APACHE_PORT,
    },
    CodeTerms {
        component: ComponentId::QwenTiktoken,
        code: "src/yue2/tokenization_yue2.py (the tokenizer data itself is covered by the \
               Tongyi Qianwen row, not by this code licence)",
        location: "github.com/multimodal-art-projection/YuE@92a73cc7652fcc1f937855e4b765e0a0edd7ff2e",
        terms: "Apache-2.0 (root LICENSE, pyproject license = \"Apache-2.0\")",
        treated_as: "Apache-2.0",
        port: APACHE_PORT,
    },
    CodeTerms {
        component: ComponentId::VaeStandard,
        code: "src/yue2/modeling_vae.py",
        location: "github.com/multimodal-art-projection/YuE@92a73cc7652fcc1f937855e4b765e0a0edd7ff2e",
        terms: "Apache-2.0, with the stable-audio-tools (Oobleck) and BigVGAN (SnakeBeta) MIT \
                notices in SOURCE_TERMS.third_party",
        treated_as: "Apache-2.0",
        port: APACHE_PORT,
    },
    CodeTerms {
        component: ComponentId::VaeLegacy,
        code: "src/yue2/modeling_vae.py",
        location: "github.com/multimodal-art-projection/YuE@92a73cc7652fcc1f937855e4b765e0a0edd7ff2e",
        terms: "Apache-2.0, with the stable-audio-tools (Oobleck) and BigVGAN (SnakeBeta) MIT \
                notices in SOURCE_TERMS.third_party",
        treated_as: "Apache-2.0",
        port: APACHE_PORT,
    },
    CodeTerms {
        component: ComponentId::SheetSage2,
        code: "the 22 remote Python files of m-a-p/SheetSage2 (modeling_sheetsage2.py, \
               tokenization_sheetsage2.py, notation_sheetsage2.py, …, plus copies of \
               modeling_mert2.py / configuration_mert2.py)",
        location: "https://huggingface.co/m-a-p/SheetSage2@eab522a8168e8b8b8c4856bf8609cd86198f01fe",
        terms: "no licence header, SPDX tag or code licence file in any of them; the card declares \
                cc-by-nc-4.0 and the shipped LICENSE is the CC BY-NC 4.0 MERT2 weight licence, whose \
                preamble says it \"does not replace separately applicable licenses for code or \
                dependencies\"; THIRD_PARTY_NOTICES.md says the BART decoder uses Hugging Face \
                Transformers (Apache 2.0) — imported (transformers BartDecoder), not vendored",
        treated_as: "CC-BY-NC-4.0",
        port: COVER_PORT,
    },
    CodeTerms {
        component: ComponentId::MertV2FullSong,
        code: "modeling_mert2.py and configuration_mert2.py of m-a-p/MERT-v2-FullSong",
        location: "https://huggingface.co/m-a-p/MERT-v2-FullSong@d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
        terms: "no licence header, SPDX tag or code licence file; the card declares cc-by-nc-4.0 \
                and the shipped LICENSE is the CC BY-NC 4.0 MERT2 weight licence, whose preamble \
                says it \"does not replace separately applicable licenses for code or \
                dependencies\"; THIRD_PARTY_NOTICES.md lists PyTorch, torchaudio, Transformers, \
                huggingface_hub and safetensors as separately installed dependencies",
        treated_as: "CC-BY-NC-4.0",
        port: COVER_PORT,
    },
];

/// The code terms for component `id`.
pub fn code_terms(id: ComponentId) -> Option<&'static CodeTerms> {
    CODE_TERMS.iter().find(|t| t.component == id)
}

/// A bundled archive or third-party asset outside every closure, with the terms it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BundledTerms {
    /// The artifact.
    pub artifact: &'static str,
    /// Where it is published.
    pub location: &'static str,
    /// The terms it carries, as inspected.
    pub terms: &'static str,
    /// What this crate does with it.
    pub disposition: &'static str,
}

/// Bundled archive and third-party terms, as inspected on [`inventory::RETRIEVED`].
pub const BUNDLED_TERMS: &[BundledTerms] = &[
    BundledTerms {
        artifact: "yue2_infer-0.1.6 wheel and sdist, yue2-music.zip (release yue2-v0.1.6, built \
                   from 9c6c4b349be978b06a9d0d958471a07a6cdeff4d)",
        location: "https://github.com/multimodal-art-projection/YuE/releases/tag/yue2-v0.1.6",
        terms: "\"YuE2 source code and agent skill license\": the source code, agent skill and \
                documentation in that release are CC BY-NC 4.0 (wheel METADATA \
                License-Expression: CC-BY-NC-4.0), plus MODEL_LICENSE and the two MIT notices",
        disposition: "excluded; retains its bundled terms; never a port source",
    },
    BundledTerms {
        artifact: "yue2_infer-0.1.3 and 0.1.5 wheels",
        location: "https://huggingface.co/m-a-p/YuE2-3B (pinned revision)",
        terms: "dist-info licence is the CC BY-NC 4.0 YuE2 model licence text plus the two MIT \
                notices",
        disposition: "excluded; retains its bundled terms; never a port source",
    },
    BundledTerms {
        artifact: "modeling_yue2.py / modeling_vae.py remote code",
        location: "m-a-p/YuE2-3B, m-a-p/YuE2-Vae, m-a-p/YuE2-Vae-legacy (pinned revisions)",
        terms: "byte-identical to src/yue2/*.py at the pinned source commit, but published inside \
                repositories whose card declares cc-by-nc-4.0",
        disposition: "excluded (no remote code at runtime); ports use the Apache-2.0 GitHub copy",
    },
    BundledTerms {
        artifact: "skills/yue2-music agent skill",
        location: "github.com/multimodal-art-projection/YuE at the pinned commit",
        terms: "Apache-2.0 (skills/yue2-music/LICENSE is byte-identical to the root LICENSE)",
        disposition: "not used",
    },
    BundledTerms {
        artifact: "SheetSage2 render_assets/ (abcjs 6.6.3, DejaVu Sans font, FluidR3 piano \
                   samples)",
        location: "https://huggingface.co/m-a-p/SheetSage2 (pinned revision)",
        terms: "abcjs MIT; the DejaVu font licence; FluidR3 samples by Frank Wen CC BY 3.0 US \
                (per the repository's THIRD_PARTY_NOTICES.md)",
        disposition: "excluded; score/audio preview rendering is not a transcription dependency",
    },
    BundledTerms {
        artifact: "SheetSage2 remote code (22 *.py files, including copies of modeling_mert2.py \
                   and configuration_mert2.py byte-identical to MERT-v2-FullSong's)",
        location: "https://huggingface.co/m-a-p/SheetSage2 (pinned revision)",
        terms: "no explicit code licence: no header or SPDX tag in any file and no code licence \
                file; the repository card declares cc-by-nc-4.0; THIRD_PARTY_NOTICES.md cites \
                Hugging Face Transformers (Apache 2.0) for the BART decoder, which the code \
                imports rather than vendors",
        disposition: "excluded at runtime; treated as CC BY-NC 4.0 and any native port gated \
                      (see CODE_TERMS)",
    },
    BundledTerms {
        artifact: "MERT-v2-FullSong remote code (modeling_mert2.py, configuration_mert2.py)",
        location: "https://huggingface.co/m-a-p/MERT-v2-FullSong (pinned revision)",
        terms: "no explicit code licence: no header or SPDX tag and no code licence file; the \
                repository card declares cc-by-nc-4.0; THIRD_PARTY_NOTICES.md lists only \
                separately installed dependencies",
        disposition: "excluded at runtime; treated as CC BY-NC 4.0 and any native port gated \
                      (see CODE_TERMS)",
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::path::Path;

    fn crate_file(rel: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
    }

    /// Every vendored licence text is present and byte-identical to what it was pinned as.
    #[test]
    fn vendored_licence_texts_match_their_pins() {
        for t in LICENSE_TEXTS {
            let bytes = std::fs::read(crate_file(t.path))
                .unwrap_or_else(|e| panic!("{} is not vendored: {e}", t.path));
            let sha: String = Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(sha, t.sha256, "{} changed", t.path);
        }
        // And nothing is vendored without a pin (the directory README aside).
        fn walk(dir: &Path, out: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, out);
                } else {
                    let rel = path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap();
                    out.push(rel.to_string_lossy().replace('\\', "/"));
                }
            }
        }
        let mut found = Vec::new();
        walk(&crate_file("licenses"), &mut found);
        for rel in found {
            assert!(
                rel == "licenses/README.md" || LICENSE_TEXTS.iter().any(|t| t.path == rel),
                "{rel} is vendored but not pinned in LICENSE_TEXTS"
            );
        }
    }

    /// The licence and notice files pinned in the inventory are exactly the vendored texts — so a
    /// verified snapshot carries these texts beside its weights, and the policy's evidence is the
    /// text actually distributed with them.
    #[test]
    fn inventory_licence_files_are_the_vendored_texts() {
        let vendored: Vec<&str> = LICENSE_TEXTS.iter().map(|t| t.sha256).collect();
        for id in ComponentId::ALL {
            for f in id.component().files {
                if matches!(
                    f.role,
                    inventory::FileRole::License | inventory::FileRole::Notice
                ) {
                    assert!(
                        vendored.contains(&f.sha256),
                        "{:?} {} is not a vendored text",
                        id,
                        f.path
                    );
                }
            }
        }
    }

    /// Each component has exactly one policy whose row names it; each row is well-formed against
    /// the gen-core family table (so it carries the attribution its family requires); every cited
    /// text is vendored.
    #[test]
    fn every_component_has_a_well_formed_policy() {
        assert_eq!(POLICIES.len(), ComponentId::ALL.len());
        let texts: Vec<&str> = LICENSE_TEXTS.iter().map(|t| t.path).collect();
        for (p, id) in POLICIES.iter().zip(ComponentId::ALL) {
            assert_eq!(p.component, id);
            assert_eq!(p.license.component, id.component().key);
            assert_eq!(
                COMPONENT_LICENSES
                    .iter()
                    .filter(|r| **r == p.license)
                    .count(),
                1
            );
            assert!(
                p.license.is_well_formed(gen_core::LICENSE_FAMILIES),
                "{:?} row is malformed",
                id
            );
            assert!(
                p.license.attribution.is_some(),
                "{id:?} records no attribution"
            );
            for t in p.texts {
                assert!(texts.contains(t), "{id:?} cites unvendored {t}");
            }
            for u in IntendedUse::ALL {
                if let UseDisposition::Permitted(b) = p.disposition(u) {
                    assert!(
                        texts.contains(&b.evidence),
                        "{id:?} basis cites {}",
                        b.evidence
                    );
                    assert!(
                        gen_core::resolve_family(gen_core::LICENSE_FAMILIES, b.family).is_some(),
                        "{id:?} basis names unknown family {}",
                        b.family
                    );
                }
            }
        }
        for n in SOURCE_TERMS.third_party {
            assert!(texts.contains(&n.text));
        }
        assert!(texts.contains(&SOURCE_TERMS.license_text));
        assert!(texts.contains(&SOURCE_TERMS.model_license_text));
    }

    /// Every weights component declares the CC BY-NC 4.0 family, which imposes the noncommercial
    /// term; the tokenizer declares the Tongyi Qianwen family (not the redistributor's tag). The
    /// source is Apache-2.0 — three distinct families, never collapsed.
    #[test]
    fn weights_source_and_tokenizer_licences_stay_distinct() {
        let nc = gen_core::resolve_family(gen_core::LICENSE_FAMILIES, "cc-by-nc-4-0").unwrap();
        assert!(nc.imposes(gen_core::LicenseTerm::NonCommercialWeights));
        for id in ComponentId::ALL {
            let row = policy(id).license;
            if id == ComponentId::QwenTiktoken {
                assert_eq!(row.family, "tongyi-qianwen");
                assert_eq!(row.source_url, "https://huggingface.co/Qwen/Qwen-7B");
            } else {
                assert_eq!(row.family, "cc-by-nc-4-0", "{id:?}");
                assert_eq!(row.declared, id.component().repo.card_license, "{id:?}");
                assert_eq!(row.gated, id.component().repo.gated, "{id:?}");
            }
        }
        assert_eq!(SOURCE_TERMS.spdx, "Apache-2.0");
        let text = std::fs::read_to_string(crate_file(SOURCE_TERMS.license_text)).unwrap();
        assert!(text.trim_start().starts_with("Apache License"));
        let model = std::fs::read_to_string(crate_file(SOURCE_TERMS.model_license_text)).unwrap();
        assert!(model.contains("(CC BY-NC 4.0)"));
        assert!(model.contains("does not replace separately applicable licenses for code"));
    }

    /// Noncommercial experimentation is permitted for both closures, with every attribution.
    #[test]
    fn noncommercial_experimentation_is_permitted_with_attribution() {
        for closure in [
            Closure::Generation {
                vae: inventory::VaeVariant::Standard,
            },
            Closure::Generation {
                vae: inventory::VaeVariant::Legacy,
            },
            Closure::Cover,
        ] {
            let a = authorize_closure(closure, IntendedUse::NoncommercialExperimentation)
                .unwrap_or_else(|e| panic!("{closure:?}: {e}"));
            assert_eq!(a.intended(), IntendedUse::NoncommercialExperimentation);
            let granted: Vec<ComponentId> = a.grants().iter().map(|(id, _)| *id).collect();
            assert_eq!(granted, closure.components());
            assert_eq!(a.attributions().len(), closure.components().len());
        }
    }

    fn gated(err: UseRefused) -> Vec<(ComponentId, &'static str, &'static str)> {
        match err {
            UseRefused::Gated { gated, .. } => gated,
            other => panic!("expected a Gated refusal, got {other:?}"),
        }
    }

    /// Commercial use and redistribution are refused for every closure and every single component,
    /// and the refusal names each component with its unblock condition.
    #[test]
    fn commercial_use_and_redistribution_are_gated() {
        for intended in [IntendedUse::CommercialUse, IntendedUse::Redistribution] {
            for id in ComponentId::ALL {
                let err = authorize(&[id], intended).unwrap_err();
                assert_eq!(err.intended(), intended);
                let g = gated(err);
                assert_eq!(g.len(), 1);
                assert_eq!(g[0].0, id);
                assert!(!g[0].2.is_empty());
            }
            let closure = Closure::Generation {
                vae: inventory::VaeVariant::Standard,
            };
            let err = authorize_closure(closure, intended).unwrap_err();
            assert!(err.to_string().contains("unblock"), "{err}");
            assert_eq!(gated(err).len(), closure.components().len());
        }
    }

    /// An empty list gets its own refusal — it is not blamed on any component.
    #[test]
    fn an_empty_component_list_is_refused() {
        for intended in IntendedUse::ALL {
            let err = authorize(&[], intended).unwrap_err();
            assert_eq!(err, UseRefused::NoComponents { intended });
            assert!(
                err.to_string().contains("no components were named"),
                "{err}"
            );
        }
    }

    /// Every component — in particular every cover-closure component, whose only code is remote
    /// Python with no code licence — has recorded code terms. Cover code is treated as CC BY-NC 4.0
    /// with its port gated behind an owner-recorded basis; the YuE2 LM / tokenizer / VAE ports are
    /// permitted only on the Apache-2.0 GitHub source.
    #[test]
    fn every_component_has_recorded_code_terms_and_cover_ports_are_gated() {
        let texts: Vec<&str> = LICENSE_TEXTS.iter().map(|t| t.path).collect();
        for id in Closure::Cover.components() {
            let t = code_terms(*id).unwrap_or_else(|| panic!("{id:?} has no recorded code terms"));
            assert_eq!(t.treated_as, "CC-BY-NC-4.0", "{id:?}");
            match t.port {
                UseDisposition::Gated { reason, unblock } => {
                    assert!(reason.contains("CC BY-NC 4.0"), "{id:?}: {reason}");
                    assert!(
                        unblock.contains("owner-recorded basis"),
                        "{id:?}: {unblock}"
                    );
                }
                UseDisposition::Permitted(b) => panic!("{id:?} port is permitted on {b:?}"),
            }
            assert!(
                BUNDLED_TERMS
                    .iter()
                    .any(|b| b.location.contains(id.component().repo.id)
                        && b.artifact.contains("remote code")),
                "{id:?}: its remote code has no BUNDLED_TERMS entry"
            );
        }
        for id in ComponentId::ALL {
            let t = code_terms(id).unwrap_or_else(|| panic!("{id:?} has no recorded code terms"));
            if Closure::Cover.components().contains(&id) {
                continue;
            }
            assert_eq!(t.treated_as, SOURCE_TERMS.spdx, "{id:?}");
            assert!(t.location.contains(SOURCE_TERMS.commit), "{id:?}");
            match t.port {
                UseDisposition::Permitted(b) => {
                    assert_eq!(b.family, "apache-2-0");
                    assert!(texts.contains(&b.evidence));
                }
                UseDisposition::Gated { .. } => panic!("{id:?} port is gated"),
            }
        }
        assert_eq!(CODE_TERMS.len(), ComponentId::ALL.len());
        // The port rule is scoped to the three YuE2 source files and points the cover closure at
        // its own recorded terms.
        for needle in [
            "modeling_yue2.py",
            "modeling_vae.py",
            "tokenization_yue2.py",
            "no SheetSage2 or MERT2 code",
            "CODE_TERMS",
        ] {
            assert!(
                SOURCE_TERMS.port_rule.contains(needle),
                "port_rule does not mention {needle}"
            );
        }
    }
}
