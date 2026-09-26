//! The pinned YuE2 asset inventory (sc-22989).
//!
//! Every upstream artifact YuE2 loads is pinned here by repository, **revision** and, per file, byte
//! size and SHA-256. The values were read from the pinned revisions on 2026-09-26 and cross-checked
//! three ways: streamed from the downloaded snapshot, compared with the Hugging Face revision tree
//! (LFS SHA-256 for the weights, git blob ids for everything else), and — for the YuE2 and MERT
//! repositories — checked by upstream's own `weights_manifest.json` routine
//! (`scripts/reference/yue2/asset_manifest.py`).
//!
//! # Components and closures
//!
//! A [`Component`] is a set of files inside one pinned repository. `qwen.tiktoken` lives in the
//! `m-a-p/YuE2-3B` repository but is its own component, because its licence provenance differs from
//! the weights beside it (see [`crate::license`]).
//!
//! * [`Closure::Generation`] — YuE2-3B + `qwen.tiktoken` + exactly one VAE (standard or legacy).
//! * [`Closure::Cover`] — SheetSage2 + MERT-v2-FullSong, the source-recording transcription closure.
//!   A conditional dependency: generation never requires it (epic E7).
//!
//! # Conversion manifests
//!
//! Each component carries a committed manifest (`manifests/<key>.json`, see
//! [`crate::manifest::ConversionManifest`]) recording the source files, the conversion (the
//! identity for every YuE2 component: all weights are published as BF16/F32 safetensors, which
//! Candle loads as-is) and, for weights, every tensor's name, dtype, shape and value digest.
//!
//! # What is deliberately not in the closure
//!
//! [`EXCLUDED`] lists the upstream files a snapshot may contain that are never loaded — remote
//! Python model code (production runs natively, epic E3), the bundled Python wheels, demo audio,
//! images, and SheetSage2's rendering assets — each with the reason.

use crate::manifest::ConversionManifest;
use crate::snapshot::AssetError;

/// The date every pin and licence declaration in this module was read from its upstream.
pub const RETRIEVED: &str = "2026-09-26";

/// One pinned upstream Hugging Face model repository.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UpstreamRepo {
    /// `owner/name` on huggingface.co.
    pub id: &'static str,
    /// The full 40-hex commit the snapshot must be at.
    pub revision: &'static str,
    /// Whether the repository gates access (observed at [`RETRIEVED`]). A gate is never stripped.
    pub gated: bool,
    /// The model card's `license:` tag, verbatim.
    pub card_license: &'static str,
}

/// `m-a-p/YuE2-3B` — the MoT language model, its generation configs and `qwen.tiktoken`.
pub const YUE2_3B_REPO: UpstreamRepo = UpstreamRepo {
    id: "m-a-p/YuE2-3B",
    revision: "1a96eca688d6ae5d7f0feb88573fec89920fcd19",
    gated: false,
    card_license: "cc-by-nc-4.0",
};

/// `m-a-p/YuE2-Vae` — the standard acoustic-latent decoder.
pub const YUE2_VAE_REPO: UpstreamRepo = UpstreamRepo {
    id: "m-a-p/YuE2-Vae",
    revision: "95535e72a97bc0f09b8ada125d26b4009428c0e8",
    gated: false,
    card_license: "cc-by-nc-4.0",
};

/// `m-a-p/YuE2-Vae-legacy` — the legacy acoustic-latent decoder.
pub const YUE2_VAE_LEGACY_REPO: UpstreamRepo = UpstreamRepo {
    id: "m-a-p/YuE2-Vae-legacy",
    revision: "b54118f0fc462f08999d1ec07e88817f4ee3f770",
    gated: false,
    card_license: "cc-by-nc-4.0",
};

/// `m-a-p/SheetSage2` — the audio-to-score transcription head (cover closure).
pub const SHEETSAGE2_REPO: UpstreamRepo = UpstreamRepo {
    id: "m-a-p/SheetSage2",
    revision: "eab522a8168e8b8b8c4856bf8609cd86198f01fe",
    gated: false,
    card_license: "cc-by-nc-4.0",
};

/// `m-a-p/MERT-v2-FullSong` — the music encoder SheetSage2 adapts (cover closure).
pub const MERT_V2_FULLSONG_REPO: UpstreamRepo = UpstreamRepo {
    id: "m-a-p/MERT-v2-FullSong",
    revision: "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
    gated: false,
    card_license: "cc-by-nc-4.0",
};

/// Every pinned repository, generation first.
pub const REPOS: [UpstreamRepo; 5] = [
    YUE2_3B_REPO,
    YUE2_VAE_REPO,
    YUE2_VAE_LEGACY_REPO,
    SHEETSAGE2_REPO,
    MERT_V2_FULLSONG_REPO,
];

/// The pinned first-party GitHub source: `multimodal-art-projection/YuE` at this commit is the only
/// source native ports may derive from (Apache-2.0; see [`crate::license::SOURCE_TERMS`]).
pub const YUE2_SOURCE_REPO: &str = "https://github.com/multimodal-art-projection/YuE";
/// The pinned upstream source commit (`yue2-infer` 0.1.6).
pub const YUE2_SOURCE_COMMIT: &str = "92a73cc7652fcc1f937855e4b765e0a0edd7ff2e";

/// What a pinned file is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FileRole {
    /// A safetensors weights file.
    Weights,
    /// A tokenizer data file (`qwen.tiktoken`).
    Tokenizer,
    /// A model / generation / processor configuration file.
    Config,
    /// Upstream's own weights manifest (`weights_manifest.json`).
    UpstreamManifest,
    /// The model card (`README.md`) — the document that declares the card licence tag.
    ModelCard,
    /// A licence text shipped beside the weights; travels with every copy.
    License,
    /// A third-party notice file shipped beside the weights; travels with every copy.
    Notice,
}

/// One file of a component, pinned by size and SHA-256.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedFile {
    /// Path relative to the repository snapshot root (`/`-separated).
    pub path: &'static str,
    /// Exact byte size.
    pub bytes: u64,
    /// Lower-case hex SHA-256 of the file's bytes.
    pub sha256: &'static str,
    /// What the file is for.
    pub role: FileRole,
}

const fn pinned(
    path: &'static str,
    bytes: u64,
    sha256: &'static str,
    role: FileRole,
) -> PinnedFile {
    PinnedFile {
        path,
        bytes,
        sha256,
        role,
    }
}

/// A YuE2 closure component.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ComponentId {
    /// The YuE2-3B MoT language model (weights + configs + the repository's licence files).
    Lm,
    /// `qwen.tiktoken`, the frozen text/ABC BPE (inside the YuE2-3B repository).
    QwenTiktoken,
    /// The standard YuE2 VAE decoder.
    VaeStandard,
    /// The legacy YuE2 VAE decoder.
    VaeLegacy,
    /// The SheetSage2 transcription head (cover closure).
    SheetSage2,
    /// The MERT-v2-FullSong music encoder (cover closure).
    MertV2FullSong,
}

impl ComponentId {
    /// Every component, generation first.
    pub const ALL: [ComponentId; 6] = [
        ComponentId::Lm,
        ComponentId::QwenTiktoken,
        ComponentId::VaeStandard,
        ComponentId::VaeLegacy,
        ComponentId::SheetSage2,
        ComponentId::MertV2FullSong,
    ];

    /// The pinned inventory entry for this component.
    pub fn component(self) -> &'static Component {
        match self {
            ComponentId::Lm => &LM,
            ComponentId::QwenTiktoken => &QWEN_TIKTOKEN,
            ComponentId::VaeStandard => &VAE_STANDARD,
            ComponentId::VaeLegacy => &VAE_LEGACY,
            ComponentId::SheetSage2 => &SHEETSAGE2,
            ComponentId::MertV2FullSong => &MERT_V2_FULLSONG,
        }
    }
}

/// One pinned component: a file set inside one pinned repository, plus its conversion manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Component {
    /// Which component this is.
    pub id: ComponentId,
    /// Stable key (`yue2_…`), shared with the licence rows and the manifest file name.
    pub key: &'static str,
    /// The pinned repository the files come from.
    pub repo: UpstreamRepo,
    /// Every file of the component, with its pinned size and SHA-256.
    pub files: &'static [PinnedFile],
    /// The committed conversion manifest (`manifests/<key>.json`), verbatim.
    pub manifest_json: &'static str,
}

impl Component {
    /// The component's single weights file, if it has one.
    pub fn weights(&self) -> Option<&'static PinnedFile> {
        self.files.iter().find(|f| f.role == FileRole::Weights)
    }

    /// The pinned file at `path`, if the component has one.
    pub fn file(&self, path: &str) -> Option<&'static PinnedFile> {
        self.files.iter().find(|f| f.path == path)
    }

    /// Parse the committed conversion manifest. A malformed manifest is an error, never a skipped
    /// check.
    pub fn conversion_manifest(&self) -> Result<ConversionManifest, AssetError> {
        ConversionManifest::parse(self.key, self.manifest_json)
    }
}

const YUE2_3B_FILES: &[PinnedFile] = &[
    pinned(
        "config.json",
        959,
        "ad3477bbef890bf98ae196c1e4b44779494a6231c4ab66f32708eabadf265329",
        FileRole::Config,
    ),
    pinned(
        "generation_config.json",
        204,
        "ccc842772f029049fadf9b288393bcdd9f79271b361c0f1875ec1218e4ba5170",
        FileRole::Config,
    ),
    pinned(
        "yue2_generation_config.json",
        466,
        "203830cebde6e3644eb291925362990d198e66c3b6006cd11f1aaa21904bcc61",
        FileRole::Config,
    ),
    pinned(
        "weights_manifest.json",
        179,
        "2296f82c29dcb11aeefde07e216403a32fc86c8f6da2ec15a70102abfd936a93",
        FileRole::UpstreamManifest,
    ),
    pinned(
        "model.safetensors",
        7_261_441_640,
        "1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9",
        FileRole::Weights,
    ),
    pinned(
        "README.md",
        19050,
        "70bba7556a6f873eaddcaa80b5514a798154c0744dda5a8611adf66778eb99c2",
        FileRole::ModelCard,
    ),
    pinned(
        "LICENSE",
        20309,
        "060985741d20e70613b4c189c7de106cabd3fb2109fbbfb6d705c9d619417dd0",
        FileRole::License,
    ),
    pinned(
        "THIRD_PARTY_NOTICES.md",
        793,
        "14d3fd9f6fee86b4260b69b0979735b99ffec8c6b4db254567047a4215cd9af3",
        FileRole::Notice,
    ),
    pinned(
        "licenses/SnakeBeta-NVIDIA-MIT.txt",
        1076,
        "da9858d516047d82096d01c112a61bd67f26d289039464d668a1d45f91738ecc",
        FileRole::License,
    ),
    pinned(
        "licenses/stable-audio-tools-MIT.txt",
        1069,
        "a1fac33b7bcd791b74fb33aeb439f825e7277e239fc119fb7d2ab6f084a0c101",
        FileRole::License,
    ),
];

const QWEN_TIKTOKEN_FILES: &[PinnedFile] = &[pinned(
    "qwen.tiktoken",
    2_561_218,
    "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186",
    FileRole::Tokenizer,
)];

const VAE_STANDARD_FILES: &[PinnedFile] = &[
    pinned(
        "config.json",
        1378,
        "f0191bb9694009956de44e0c361a6f1334760be4c8f848e599bde242a54a0970",
        FileRole::Config,
    ),
    pinned(
        "weights_manifest.json",
        178,
        "017d64a4d288217a43fcac3866a0e1d55f796fa891d4dfb6de0bd170ab750026",
        FileRole::UpstreamManifest,
    ),
    pinned(
        "model.safetensors",
        530_512_720,
        "807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346",
        FileRole::Weights,
    ),
    pinned(
        "README.md",
        8975,
        "4ff58df7b4b6a4a51fb2a709d21a5bba1b9cb4f142d50767c3efa40e53b6e88d",
        FileRole::ModelCard,
    ),
    pinned(
        "LICENSE",
        20309,
        "060985741d20e70613b4c189c7de106cabd3fb2109fbbfb6d705c9d619417dd0",
        FileRole::License,
    ),
    pinned(
        "THIRD_PARTY_NOTICES.md",
        793,
        "14d3fd9f6fee86b4260b69b0979735b99ffec8c6b4db254567047a4215cd9af3",
        FileRole::Notice,
    ),
    pinned(
        "licenses/SnakeBeta-NVIDIA-MIT.txt",
        1076,
        "da9858d516047d82096d01c112a61bd67f26d289039464d668a1d45f91738ecc",
        FileRole::License,
    ),
    pinned(
        "licenses/stable-audio-tools-MIT.txt",
        1069,
        "a1fac33b7bcd791b74fb33aeb439f825e7277e239fc119fb7d2ab6f084a0c101",
        FileRole::License,
    ),
];

const VAE_LEGACY_FILES: &[PinnedFile] = &[
    pinned(
        "config.json",
        1376,
        "a5053282e619d241155769b9b26751a5badd1f5644707ea3ebb0c6ed849a038c",
        FileRole::Config,
    ),
    pinned(
        "weights_manifest.json",
        178,
        "92652174b1218d24303dbad37cc420058de3d90229c218e0cf9a57c1114ed0aa",
        FileRole::UpstreamManifest,
    ),
    pinned(
        "model.safetensors",
        530_512_720,
        "b6d283628913bb41145ba99e2314eef613905ee95f690eb70e8212d5f4965044",
        FileRole::Weights,
    ),
    pinned(
        "README.md",
        8612,
        "c7b723920bac92a334810dde8d975195cf186ec2f82c2c21be0c4273f6d14a5e",
        FileRole::ModelCard,
    ),
    pinned(
        "LICENSE",
        20309,
        "060985741d20e70613b4c189c7de106cabd3fb2109fbbfb6d705c9d619417dd0",
        FileRole::License,
    ),
    pinned(
        "THIRD_PARTY_NOTICES.md",
        793,
        "14d3fd9f6fee86b4260b69b0979735b99ffec8c6b4db254567047a4215cd9af3",
        FileRole::Notice,
    ),
    pinned(
        "licenses/SnakeBeta-NVIDIA-MIT.txt",
        1076,
        "da9858d516047d82096d01c112a61bd67f26d289039464d668a1d45f91738ecc",
        FileRole::License,
    ),
    pinned(
        "licenses/stable-audio-tools-MIT.txt",
        1069,
        "a1fac33b7bcd791b74fb33aeb439f825e7277e239fc119fb7d2ab6f084a0c101",
        FileRole::License,
    ),
];

const SHEETSAGE2_FILES: &[PinnedFile] = &[
    pinned(
        "config.json",
        2060,
        "a986e63f5d831ecb823c11d19cfb371d763f25ae2e614ec9acd714c0b8bb87fd",
        FileRole::Config,
    ),
    pinned(
        "processor_config.json",
        277,
        "1eed73513a74029fe74c71faba70f8cff002ae7be23ac1c926125846d1b29817",
        FileRole::Config,
    ),
    pinned(
        "model.safetensors",
        228_738_564,
        "b235f68091a5f5b644000f2b5acb57d1e70432aca2b34ab1b9cf27236e1f4274",
        FileRole::Weights,
    ),
    pinned(
        "README.md",
        10929,
        "30c377847af4131b84c9725ab0c6f39538709b974c9fa9001d38092c8962803d",
        FileRole::ModelCard,
    ),
    pinned(
        "LICENSE",
        20000,
        "73593d6cad4ca80f3ab1448a4908f92ab802dfffe029561e3b12fa7d07c8b87c",
        FileRole::License,
    ),
    pinned(
        "THIRD_PARTY_NOTICES.md",
        589,
        "ab8b195c2c1475f254a04181a9841db2f2581c579b95858c78ee0040eb495a29",
        FileRole::Notice,
    ),
];

const MERT_V2_FULLSONG_FILES: &[PinnedFile] = &[
    pinned(
        "config.json",
        882,
        "f2e194895f58be3ddba327255db129ff0e3bee550cc0ecf08e4d22d79ce3bca3",
        FileRole::Config,
    ),
    pinned(
        "preprocessor_config.json",
        215,
        "fc7337f113b71062b8efd03f8a43a07aa769ce85c6a53fdc0b3bb90c299fe63f",
        FileRole::Config,
    ),
    pinned(
        "weights_manifest.json",
        241,
        "413e2ddfa71ae5bb364bfd15364e221249111e14c2f0843ea6162afb776a8300",
        FileRole::UpstreamManifest,
    ),
    pinned(
        "model.safetensors",
        2_529_812_848,
        "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d",
        FileRole::Weights,
    ),
    pinned(
        "README.md",
        10419,
        "2b025166f640a0416b5544f4c3775aced46bf0c467af4ef7b302d52496335da6",
        FileRole::ModelCard,
    ),
    pinned(
        "LICENSE",
        20000,
        "73593d6cad4ca80f3ab1448a4908f92ab802dfffe029561e3b12fa7d07c8b87c",
        FileRole::License,
    ),
    pinned(
        "THIRD_PARTY_NOTICES.md",
        541,
        "ce8fd578969f1bdad9a0be5e20cab0251dc4253bd4dbfaa9abeb1c11d68bafdc",
        FileRole::Notice,
    ),
];

/// The YuE2-3B language model.
pub const LM: Component = Component {
    id: ComponentId::Lm,
    key: "yue2_3b",
    repo: YUE2_3B_REPO,
    files: YUE2_3B_FILES,
    manifest_json: include_str!("../manifests/yue2_3b.json"),
};

/// `qwen.tiktoken` from the YuE2-3B repository.
pub const QWEN_TIKTOKEN: Component = Component {
    id: ComponentId::QwenTiktoken,
    key: "yue2_qwen_tiktoken",
    repo: YUE2_3B_REPO,
    files: QWEN_TIKTOKEN_FILES,
    manifest_json: include_str!("../manifests/yue2_qwen_tiktoken.json"),
};

/// The standard VAE decoder.
pub const VAE_STANDARD: Component = Component {
    id: ComponentId::VaeStandard,
    key: "yue2_vae",
    repo: YUE2_VAE_REPO,
    files: VAE_STANDARD_FILES,
    manifest_json: include_str!("../manifests/yue2_vae.json"),
};

/// The legacy VAE decoder.
pub const VAE_LEGACY: Component = Component {
    id: ComponentId::VaeLegacy,
    key: "yue2_vae_legacy",
    repo: YUE2_VAE_LEGACY_REPO,
    files: VAE_LEGACY_FILES,
    manifest_json: include_str!("../manifests/yue2_vae_legacy.json"),
};

/// The SheetSage2 transcription head.
pub const SHEETSAGE2: Component = Component {
    id: ComponentId::SheetSage2,
    key: "yue2_sheetsage2",
    repo: SHEETSAGE2_REPO,
    files: SHEETSAGE2_FILES,
    manifest_json: include_str!("../manifests/yue2_sheetsage2.json"),
};

/// The MERT-v2-FullSong music encoder.
pub const MERT_V2_FULLSONG: Component = Component {
    id: ComponentId::MertV2FullSong,
    key: "yue2_mert_v2_fullsong",
    repo: MERT_V2_FULLSONG_REPO,
    files: MERT_V2_FULLSONG_FILES,
    manifest_json: include_str!("../manifests/yue2_mert_v2_fullsong.json"),
};

/// Which of the two published VAE decoders a generation closure decodes with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VaeVariant {
    /// `m-a-p/YuE2-Vae`.
    Standard,
    /// `m-a-p/YuE2-Vae-legacy`.
    Legacy,
}

/// A set of components that is resolved and verified together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Closure {
    /// Song generation: YuE2-3B, `qwen.tiktoken` and the selected VAE. Never includes the cover
    /// closure.
    Generation {
        /// The decoder this generation uses.
        vae: VaeVariant,
    },
    /// Source-recording transcription for covers: SheetSage2 + MERT-v2-FullSong.
    Cover,
}

impl Closure {
    /// The components of this closure, in resolution order.
    pub fn components(self) -> &'static [ComponentId] {
        match self {
            Closure::Generation {
                vae: VaeVariant::Standard,
            } => &[
                ComponentId::Lm,
                ComponentId::QwenTiktoken,
                ComponentId::VaeStandard,
            ],
            Closure::Generation {
                vae: VaeVariant::Legacy,
            } => &[
                ComponentId::Lm,
                ComponentId::QwenTiktoken,
                ComponentId::VaeLegacy,
            ],
            Closure::Cover => &[ComponentId::SheetSage2, ComponentId::MertV2FullSong],
        }
    }
}

/// An upstream file a pinned snapshot may contain that is deliberately **not** part of any closure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExcludedFile {
    /// The repository it appears in.
    pub repo: &'static str,
    /// The path, or a `…/` prefix for a whole directory.
    pub path: &'static str,
    /// Why it is never loaded.
    pub reason: &'static str,
}

const REMOTE_CODE: &str = "remote Python model code; production runs natively (epic E3) and ports \
                           derive only from the Apache-2.0 GitHub source at the pinned commit";

const COVER_REMOTE_CODE: &str =
    "remote Python model / processing code with no code licence of its \
                                 own; production runs natively (epic E3); provisionally CC BY-NC 4.0 \
                                 and any native port gated (license::CODE_TERMS) — the pinned \
                                 GitHub source has no SheetSage2 / MERT2 code";

/// Files the pinned repositories ship that no closure loads.
pub const EXCLUDED: &[ExcludedFile] = &[
    ExcludedFile {
        repo: "m-a-p/YuE2-3B",
        path: "modeling_yue2.py",
        reason: REMOTE_CODE,
    },
    ExcludedFile {
        repo: "m-a-p/YuE2-3B",
        path: "yue2_infer-0.1.3-py3-none-any.whl",
        reason: "bundled Python wheel; retains its bundled terms (its dist-info LICENSE is the CC \
                 BY-NC 4.0 model licence) and is never a port source",
    },
    ExcludedFile {
        repo: "m-a-p/YuE2-3B",
        path: "yue2_infer-0.1.5-py3-none-any.whl",
        reason: "bundled Python wheel; retains its bundled terms (its dist-info LICENSE is the CC \
                 BY-NC 4.0 model licence) and is never a port source",
    },
    ExcludedFile {
        repo: "m-a-p/YuE2-3B",
        path: "assets/",
        reason: "demo audio and images",
    },
    ExcludedFile {
        repo: "m-a-p/YuE2-3B",
        path: "examples/",
        reason: "example request",
    },
    ExcludedFile {
        repo: "m-a-p/YuE2-Vae",
        path: "modeling_vae.py",
        reason: REMOTE_CODE,
    },
    ExcludedFile {
        repo: "m-a-p/YuE2-Vae",
        path: "assets/",
        reason: "images",
    },
    ExcludedFile {
        repo: "m-a-p/YuE2-Vae-legacy",
        path: "modeling_vae.py",
        reason: REMOTE_CODE,
    },
    ExcludedFile {
        repo: "m-a-p/YuE2-Vae-legacy",
        path: "assets/",
        reason: "images",
    },
    ExcludedFile {
        repo: "m-a-p/SheetSage2",
        path: "*.py",
        reason: COVER_REMOTE_CODE,
    },
    ExcludedFile {
        repo: "m-a-p/SheetSage2",
        path: "render_assets/",
        reason: "score/audio preview rendering assets under their own bundled third-party terms \
                 (abcjs MIT, the DejaVu font licence, FluidR3 piano samples CC BY 3.0 US); not a \
                 transcription dependency",
    },
    ExcludedFile {
        repo: "m-a-p/SheetSage2",
        path: "requirements.txt",
        reason: "Python dependency list",
    },
    ExcludedFile {
        repo: "m-a-p/SheetSage2",
        path: "requirements-render.txt",
        reason: "Python dependency list",
    },
    ExcludedFile {
        repo: "m-a-p/SheetSage2",
        path: "benchmark_results.json",
        reason: "report",
    },
    ExcludedFile {
        repo: "m-a-p/SheetSage2",
        path: "assets/",
        reason: "images",
    },
    ExcludedFile {
        repo: "m-a-p/MERT-v2-FullSong",
        path: "*.py",
        reason: COVER_REMOTE_CODE,
    },
    ExcludedFile {
        repo: "m-a-p/MERT-v2-FullSong",
        path: "marble_results.json",
        reason: "report",
    },
    ExcludedFile {
        repo: "m-a-p/MERT-v2-FullSong",
        path: "assets/",
        reason: "images",
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn is_hex(s: &str, len: usize) -> bool {
        s.len() == len
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    /// Every pin is a full revision / full SHA-256, keys are unique and `yue2_`-prefixed, and no
    /// repository is shared with YuE1 (whose repos are all `m-a-p/YuE-s…` / `xcodec_mini_infer`).
    #[test]
    fn pins_are_complete_and_distinct_from_yue1() {
        let mut keys = BTreeSet::new();
        for id in ComponentId::ALL {
            let c = id.component();
            assert_eq!(c.id, id);
            assert!(keys.insert(c.key), "duplicate key {}", c.key);
            assert!(
                c.key.starts_with("yue2_"),
                "{} is not yue2_-namespaced",
                c.key
            );
            assert!(
                is_hex(c.repo.revision, 40),
                "{}: revision {}",
                c.key,
                c.repo.revision
            );
            assert!(c.repo.id.starts_with("m-a-p/"), "{}", c.repo.id);
            assert!(
                !c.repo.id.starts_with("m-a-p/YuE-") && !c.repo.id.contains("xcodec"),
                "{} collides with a YuE1 repository",
                c.repo.id
            );
            let mut paths = BTreeSet::new();
            for f in c.files {
                assert!(paths.insert(f.path), "{}: duplicate {}", c.key, f.path);
                assert!(
                    is_hex(f.sha256, 64),
                    "{}: {} sha {}",
                    c.key,
                    f.path,
                    f.sha256
                );
                assert!(f.bytes > 0, "{}: {}", c.key, f.path);
            }
            assert!(
                c.files
                    .iter()
                    .filter(|f| f.role == FileRole::Weights)
                    .count()
                    <= 1,
                "{} pins more than one weights file",
                c.key
            );
        }
        assert!(is_hex(YUE2_SOURCE_COMMIT, 40));
    }

    /// Generation needs the LM, the tokenizer and exactly the selected VAE; the cover closure is
    /// separate and never pulled in by generation (epic E7).
    #[test]
    fn closures_have_the_epic_shape() {
        for vae in [VaeVariant::Standard, VaeVariant::Legacy] {
            let g = Closure::Generation { vae }.components();
            assert!(g.contains(&ComponentId::Lm) && g.contains(&ComponentId::QwenTiktoken));
            let vaes: Vec<_> = g
                .iter()
                .filter(|c| matches!(c, ComponentId::VaeStandard | ComponentId::VaeLegacy))
                .collect();
            let want = match vae {
                VaeVariant::Standard => ComponentId::VaeStandard,
                VaeVariant::Legacy => ComponentId::VaeLegacy,
            };
            assert_eq!(vaes, vec![&want]);
            for cover in Closure::Cover.components() {
                assert!(
                    !g.contains(cover),
                    "generation pulls in cover dependency {cover:?}"
                );
            }
        }
        assert_eq!(
            Closure::Cover.components(),
            &[ComponentId::SheetSage2, ComponentId::MertV2FullSong]
        );
    }

    /// Every component that loads weights carries its licence and notice files, so a verified
    /// snapshot always has them beside the weights (epic E7: preserve licence/NOTICE material).
    #[test]
    fn weights_components_carry_their_licence_files() {
        for id in ComponentId::ALL {
            let c = id.component();
            if c.weights().is_some() {
                assert!(c.file("LICENSE").is_some(), "{} has no LICENSE", c.key);
                assert!(
                    c.file("THIRD_PARTY_NOTICES.md").is_some(),
                    "{} has no THIRD_PARTY_NOTICES.md",
                    c.key
                );
            }
        }
    }

    /// No excluded path is also a pinned closure path.
    #[test]
    fn excluded_files_are_not_in_any_closure() {
        for e in EXCLUDED {
            for id in ComponentId::ALL {
                let c = id.component();
                if c.repo.id != e.repo {
                    continue;
                }
                for f in c.files {
                    let hit = if let Some(prefix) = e.path.strip_suffix('/') {
                        f.path.starts_with(prefix)
                    } else if let Some(ext) = e.path.strip_prefix('*') {
                        f.path.ends_with(ext)
                    } else {
                        f.path == e.path
                    };
                    assert!(!hit, "{} is both pinned ({}) and excluded", f.path, c.key);
                }
            }
        }
    }
}
