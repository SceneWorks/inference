//! YuE2 — the **noncommercial, experimental** music-generation provider for the SceneWorks Candle
//! audio lane (epic sc-22988).
//!
//! YuE2 coexists with YuE1 (`candle-audio-yue`, epic sc-19373) and never replaces it: every
//! component key here is `yue2_`-prefixed, every upstream repository and revision is distinct from
//! YuE1's, and the two crates share no cache namespace (epic E1). YuE1 is the commercially licensed
//! option (Apache-2.0 weights); YuE2's weights are CC BY-NC 4.0 (epic E2).
//!
//! This walking-skeleton slice (sc-22989) is the pinned asset closure every later slice loads from:
//!
//! * [`inventory`] — the exact upstream repositories and revisions, every closure file with its
//!   size and SHA-256, the two closures ([`Closure::Generation`] needs YuE2-3B, `qwen.tiktoken` and
//!   one VAE; [`Closure::Cover`] is the separate SheetSage2 + MERT-v2-FullSong transcription
//!   closure), what is deliberately excluded, and the committed conversion manifests.
//! * [`snapshot`] — offline resolution of a component to a local snapshot directory and full
//!   integrity verification of the bytes that will be loaded; a cache miss, a missing, truncated,
//!   corrupt or unexpected shard, or a tensor table that disagrees with the conversion manifest is
//!   an explicit [`AssetError`]. [`snapshot::native_tensor_digests`] re-reads every tensor through
//!   Candle's safetensors loader so the native values can be compared with the pinned originals.
//! * [`license`] — the licence policy: Apache-2.0 GitHub source vs CC BY-NC 4.0 weights vs the
//!   Tongyi Qianwen terms of the bundled `qwen.tiktoken` vs bundled archive / third-party terms;
//!   the per-artifact [`gen_core::ComponentLicense`] rows with attribution; and
//!   [`license::authorize`], which refuses any intended use (commercial use, redistribution) that
//!   has no recorded compatible basis. [`license::CODE_TERMS`] records which upstream code each
//!   component's native port may derive from: the Apache-2.0 GitHub source for the LM, VAE and
//!   tokenizer; the cover closure's code is treated as CC BY-NC 4.0 and its port is gated.
//!
//! The request slice (sc-22990) is the native text side of generation:
//!
//! * [`tokenizer`] — the frozen `qwen.tiktoken` text/ABC BPE ([`Yue2TextTokenizer`]), loaded only
//!   through verification.
//! * [`protocol`] — the `yue2-native-v1` request protocol: [`SongRequest`], per-phase
//!   [`Sampling`], [`GenerationConfig`], the exact positive / CFG-negative token prefixes, and the
//!   explicit context-budget refusals.
//! * [`plan`] — exact symbolic plans ([`SymbolicPlan`]): planned, saved and restored as token IDs
//!   with integrity checks; an edited ABC is a new request.
//!
//! The autoregressive stages (sc-22991) run on that closure:
//!
//! * [`model`] — the YuE2-3B Mixture-of-Transformers backbone (AR path, and the NAR twins on
//!   request), loaded from the verified snapshot, with a bounded, preallocated KV cache;
//! * [`sampling`] — the released token-range masks, stop ids, windowed repetition penalty,
//!   temperature / top-k / top-p, classifier-free guidance and the categorical draw;
//! * [`generate`] — score planning and semantic-token generation over token ids, with guidance
//!   that keeps the exact planned score in its negative branch, truthful truncation, and
//!   cancellation at bounded boundaries.
//!
//! Nothing here downloads anything: acquiring a snapshot is the application's job, and this crate
//! only ever reads a snapshot that is already on disk.
//!
//! # Rule for loader slices: verify at the load boundary
//!
//! Every loader in this crate must call [`snapshot::resolve_closure`] /
//! [`snapshot::resolve_component`] **immediately before loading**, and load only the paths the
//! returned [`VerifiedComponent`] names. The integrity guarantee holds only for the moment of the
//! call; a verification done earlier (at start-up, or cached from a previous request) does not
//! cover the bytes a later load reads.

#![deny(rustdoc::private_intra_doc_links)]

pub use candle_audio::gen_core;

pub mod generate;
pub mod inventory;
pub mod license;
pub mod manifest;
pub mod model;
#[cfg(test)]
mod parity;
pub mod sampling;
pub mod plan;
pub mod protocol;
pub mod snapshot;
pub mod tokenizer;

#[cfg(test)]
mod test_fixtures;

pub use inventory::{Closure, Component, ComponentId, VaeVariant};
pub use plan::{PlanError, PlanIdentity, PlanStep, SemanticConditioning, SymbolicPlan};
pub use protocol::{CotMode, GenerationConfig, ProtocolError, Sampling, SongRequest};
pub use snapshot::{AssetError, SnapshotDirs, VerifiedClosure, VerifiedComponent};
pub use tokenizer::{TokenizerError, Yue2TextTokenizer};
