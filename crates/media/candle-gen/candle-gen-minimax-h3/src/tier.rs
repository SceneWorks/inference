//! MiniMax-H3 **pre-quantized tier** path resolution on the candle lane (sc-20267, epic sc-17137) —
//! the twin of the layout `mlx_gen_minimax_h3::model` resolves, expressed as one reusable type.
//!
//! [`crate::quant`] answers "is *this tensor* packed?" from a `.scales` sibling. This module answers
//! the question one level up: **which directories** does a given tier's DiT, reference DiT and text
//! encoder come from, and does the tier the caller asked for match the tier actually on disk. It is
//! inert until the `supported_quants` flip (PR 3 of sc-20267); nothing in [`crate::model`] calls it
//! yet, and its whole surface is exercised by the unit tests below against tempdir fixtures.
//!
//! # The layout is NOT `root/{tier}/component`, and mirroring that is the point
//!
//! The obvious resolver — join the tier name onto the snapshot root — would be wrong, and it would
//! disagree with the MLX lane on the one thing the two lanes must agree about. `SceneWorks/minimax-h3-mlx`
//! does publish self-contained tier dirs — SceneWorks' `builtin.models.jsonc` ships
//! `{q4,q8,bf16}/transformer/*`, `{q4,q8,bf16}/transformer_ref/*` and `{q4,q8}/text_encoder/*` at
//! revision `137ce668`, with exact hosted byte counts — but **neither backend ever constructs those
//! paths**. The tier dir is *staged* by the caller as a component override, and the engine resolves
//! against the override:
//!
//! | component | resolution | tier-dependent |
//! |---|---|---|
//! | `transformer` | staged [`DIT_COMPONENT`] override, else `root/transformer` | yes |
//! | `transformer_ref` | **sibling of the resolved DiT dir** | yes |
//! | `text_encoder` | staged [`TEXT_ENCODER_COMPONENT`] override, else `root/text_encoder` | yes, *independently* |
//! | `vae`, `audio_vae`, `tokenizer`, `FL2VA` | always `root/...` | no |
//!
//! That is `mlx_gen_minimax_h3::model`'s `resolve_dit_dir` / `resolve_text_encoder_dir` /
//! `dit_dir.with_file_name(REFERENCE_DIT_PARTITION)` shape, reproduced here call for call. Three of
//! those rows are load-bearing:
//!
//! * **The reference DiT is a SIBLING, never `root/transformer_ref`.** A split install stages the
//!   tier's `transformer` somewhere outside the snapshot and has no `root/transformer_ref` at all;
//!   resolving against the root would look in a directory that does not exist while the correct one
//!   sat next to the staged DiT. On a flat snapshot `dit_dir == root/transformer`, whose sibling is
//!   exactly `root/transformer_ref`, so the one rule serves both layouts.
//! * **The engine does not couple the text encoder's tier to the DiT's** (sc-19120), so
//!   [`MiniMaxH3TierPaths::require_dit`] takes a [`Tier`] and asserts it against the staged DiT
//!   while [`MiniMaxH3TierPaths::require_text_encoder`] deliberately takes **none** — it validates
//!   only the group size. That mirrors `mlx_gen_minimax_h3::model`, whose `reconcile_text_encoder`
//!   pointedly omits the `spec.quantize` comparison its `reconcile_tier` makes.
//!
//!   Note the *rationale* that decision was originally written under — "the manifest pairs one dense
//!   TE with all three DiT tiers" — no longer describes the manifest: sc-19120 added per-tier TE
//!   rows, so `builtin.models.jsonc` now ships `q4/text_encoder` and `q8/text_encoder` from the
//!   rehost and the `bf16` TE from upstream `MiniMaxAI/MiniMax-H3`, variant-scoped so
//!   `resolve_co_requisites_for_tier` picks exactly one per tier. The *decision* survives the
//!   correction on a better reason: the TE's tier is recoverable from its own `{base}.scales`
//!   ([`crate::quant::lin`] auto-detects it), the three rows do not all come from the same repo, and
//!   installs predating sc-19120 carry a dense TE beside a packed DiT. Asserting a pairing the
//!   loader can simply observe would reject those installs for no gain.
//! * **The VAEs and tokenizer are tier-agnostic siblings** resolved against the upstream root, so a
//!   `q4` install holds one packed DiT and one packed TE with no second copy of anything else.
//!
//! # The reference partition is probed LAZILY here and EAGERLY on MLX
//!
//! This is the one place the two lanes deliberately disagree, so it is stated in full rather than
//! left to be rediscovered.
//!
//! `mlx_gen_minimax_h3::model::load` opens `transformer/config.json` **and**
//! `transformer_ref/config.json` on **every** load regardless of task, and fails the whole load if
//! either is missing. That is safe on macOS because the manifest guarantees the sibling is there:
//! `builtin.models.jsonc` ships `{q4,q8,bf16}/transformer_ref/*` as **per-tier `coRequisite` rows**
//! (18.78 GB at q4, 35.30 at q8, 66.28 at bf16), with the manifest recording that these bytes "are
//! not an optional Ref2VA extra, they are part of the minimum loadable set".
//!
//! This lane probes it only when a request resolves to `ref2va`
//! ([`crate::model::task_component_dirs`](crate::model)). The justification is deliberately confined
//! to **three checkable facts about the artifacts**, with no claim about what either engine permits:
//!
//! 1. every `transformer_ref` row in `builtin.models.jsonc` is `"platforms": ["macos"]`;
//! 2. the off-Mac artifact set sc-19558 defines carries **no `transformer_ref` rows at all**; and
//! 3. SceneWorks' `crates/sceneworks-worker/src/video_jobs/minimax_h3.rs` is
//!    `#[cfg(target_os = "macos")]` end to end, so no off-Mac dispatch arm exists today.
//!
//! An off-Mac install therefore has no catalog route to the partition, and an every-load probe would
//! fail **every** off-Mac load over a directory the catalog never offered.
//!
//! **The trigger to close the divergence** is the artifacts, not the engine: if the manifest gains
//! off-Mac `transformer_ref` rows, this should move to MLX's every-load probe. Why those rows are
//! absent today is a **catalog** question — confirm it with the catalog owner rather than inferring
//! it from this crate.
//!
//! > **Two retracted premises, so neither is reintroduced.** (a) "`SceneWorks/minimax-h3-mlx`
//! > publishes no `q4/transformer_ref` (sc-19517)" — false; the sc-19573 block ships it with an exact
//! > hosted byte count, and the manifest flags that reasoning as coming "from a premise the engine
//! > has never honoured". (b) "this provider default-denies `ref2va` until sc-17157 lands the port" —
//! > also false; sc-17157 **landed** (`32204c935`), and [`crate::model`] advertises and admits all
//! > three reference conditioning kinds. That second claim was paraphrased from a manifest comment
//! > written before the port, which makes the comment stale too. Neither premise is load-bearing
//! > above: facts 1-3 are all directly greppable.
//!
//! # The tier is an ASSERTION about what is staged, never an instruction
//!
//! **MiniMax-H3 never quantizes at load.** Quantizing the DiT at load would materialize its
//! 66_280_430_080 dense bytes *and* the growing packed output at once. So every tier ships
//! pre-quantized (`mlx_gen_minimax_h3::convert`) and a caller's requested tier is reconciled
//! against the `quantization` marker the converter writes into each component's `config.json` — the
//! same marker `mlx_gen::quant::packed_quant_bits_at` reads, parsed here through the shared
//! [`candle_gen::quant::PackedConfig`]. A disagreement is a hard error rather than a silent
//! downgrade, because an unmarked packed component and a genuinely dense one are indistinguishable
//! to a loader that guesses.
//!
//! [`Tier::Bf16`] is the dense layout: no marker, no group size, nothing to validate.
//!
//! # Why the group size is validated but the bit width is not
//!
//! [`candle_gen::quant::QLinear::from_packed_gs`] *infers* the packed bit width from the
//! weight/scales column ratio **assuming** [`GROUP_SIZE`]. A component packed at a different group
//! therefore does not fail cleanly: Mage's q8 vision tower (packed at 32, read at 64) derived a
//! legal-looking "bits 16" from a perfectly good artifact and was diagnosed as a bad upload
//! (sc-15154). Reading the declared group size and rejecting a mismatch by name turns that into one
//! actionable line. The bit width needs no such check — it is *recovered* from the artifact, so a
//! q4 and a q8 tier both load correctly without being told which they are.
//!
//! **This lane validates the group size on more components than MLX does, deliberately.** MLX runs
//! the check only on the text encoder (`reconcile_text_encoder`); its `reconcile_tier` compares bit
//! widths and never reads `group_size`, so a DiT tier packed at 32 reaches its loaders unchallenged.
//! [`MiniMaxH3TierPaths::require_dit`] and [`MiniMaxH3TierPaths::require_reference_dit`] validate it
//! too. That is a departure in the safe direction — it can only reject an artifact that would have
//! decoded to garbage — and it costs nothing on a well-formed tier, since every published H3 tier
//! packs at 64. It is recorded here rather than silently tightened so a future reader comparing the
//! two lanes finds an intent rather than an inconsistency.
//!
//! The group size is **imported under an alias**, exactly as [`crate::quant`] imports it, and is
//! deliberately not re-exported as a crate-local `pub const`. A second declaration that drifted
//! would derive an illegal width from a perfectly good artifact instead of failing cleanly — and
//! `scripts/check-workspace.py`'s cross-backend geometry gate rejects the `pub const` form outright
//! for that reason (it reads as this backend declaring its own geometry rather than consuming the
//! shared one).

use std::path::{Path, PathBuf};

use candle_gen::quant::{PackedConfig, MLX_GROUP_SIZE as GROUP_SIZE};
use candle_gen::{CandleError, Result};

use crate::model::{BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION};
use crate::MODEL_ID;

/// The staged-component id for the tiered DiT directory (sc-17150) — `mlx_gen_minimax_h3::model`'s
/// `DIT_COMPONENT`.
pub const DIT_COMPONENT: &str = BASE_DIT_PARTITION;

/// The staged-component id for the tiered text encoder (sc-19120) — `mlx_gen_minimax_h3::model`'s
/// `TEXT_ENCODER_COMPONENT`.
pub const TEXT_ENCODER_COMPONENT: &str = "text_encoder";

/// A published MiniMax-H3 weight tier.
///
/// # ⚠️ PR 3 must not map an absent request onto [`Tier::Bf16`]
///
/// This enum can express only an *asserted* tier, and gen-core's `spec.quantize` is an
/// `Option<Quant>` whose `None` means **"the caller asserted nothing"**. MLX honours that third
/// state: its `reconcile_tier` admits a packed tier under `(Some(_), None)` and only refuses a
/// *dense* component under an explicit request. The nearest thing here, [`Tier::Bf16`], is not that
/// state — it is a positive assertion of denseness, so
/// [`MiniMaxH3TierPaths::require_dit`] refuses a packed tier under it (see
/// `a_packed_component_under_a_bf16_request_is_refused`).
///
/// So the `supported_quants` flip must map `spec.quantize` deliberately, **not** by defaulting
/// `None` to `Bf16` — that would turn "I did not ask" into "I demand dense" and reject every packed
/// install that loads fine on MLX today. Either add an `Unasserted` variant, or skip the reconcile
/// entirely when `spec.quantize` is `None` and let the loaders auto-detect (which they already do —
/// nothing downstream of here reads a tier). Deliberately not wired now: this PR ships the resolver
/// inert, and picking the shape without the call site in front of you invites the wrong default.
///
/// The three the rehost publishes, and the complete set: `SceneWorks/minimax-h3-mlx` ships `q4`,
/// `q8` and `bf16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// 4-bit MLX affine pack.
    Q4,
    /// 8-bit MLX affine pack.
    Q8,
    /// The dense tier — no pack, no marker.
    Bf16,
}

impl Tier {
    /// The tier's packed bit width, or `None` for the dense [`Tier::Bf16`].
    pub fn bits(self) -> Option<i32> {
        match self {
            Self::Q4 => Some(4),
            Self::Q8 => Some(8),
            Self::Bf16 => None,
        }
    }

    /// The tier's published directory name (`q4` / `q8` / `bf16`) — used to *name* a tier in an
    /// error message. It is deliberately not used to build a path; see the module docs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q4 => "q4",
            Self::Q8 => "q8",
            Self::Bf16 => "bf16",
        }
    }
}

/// The component directories one MiniMax-H3 load reads, resolved for a tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxH3TierPaths {
    /// The `t2va` / `fl2va` DiT partition — the staged tier dir, else `root/transformer`.
    pub dit_dir: PathBuf,
    /// The `ref2va` DiT partition — always [`Self::dit_dir`]'s **sibling**, never `root`-relative.
    ///
    /// Resolved unconditionally, but [`MiniMaxH3TierPaths::require_reference_dit`] is called only
    /// when a request resolves to `ref2va`. **This is a deliberate divergence from the MLX lane, and
    /// it is conditional — see the module docs' "the reference partition is probed lazily here and
    /// eagerly on MLX" section for the verified reason and the trigger to close it.**
    pub reference_dit_dir: PathBuf,
    /// The condition encoder — the staged tier dir, else `root/text_encoder`. Its tier is
    /// auto-detected rather than asserted (sc-19120); see the module docs.
    pub text_encoder_dir: PathBuf,
    /// Where the **tier-agnostic** components live: `vae/`, `audio_vae/`, `tokenizer/`, `FL2VA/`.
    /// Always the upstream snapshot root.
    pub shared_root: PathBuf,
}

impl MiniMaxH3TierPaths {
    /// Resolve every component directory for a load.
    ///
    /// `staged_dit` / `staged_text_encoder` are the caller's component overrides (gen-core's
    /// `LoadSpec::components` entries for [`DIT_COMPONENT`] / [`TEXT_ENCODER_COMPONENT`]); `None`
    /// falls back to the flat upstream layout under `root`. This performs **no I/O** — it is pure
    /// path arithmetic, so a caller can resolve and then decide what to require.
    pub fn resolve(
        root: &Path,
        staged_dit: Option<&Path>,
        staged_text_encoder: Option<&Path>,
    ) -> Self {
        let dit_dir = staged_dit.map_or_else(|| root.join(DIT_COMPONENT), Path::to_path_buf);
        // The SIBLING rule — see the module docs. `with_file_name` on `<staged>/transformer` yields
        // `<staged>/transformer_ref`, and on `root/transformer` yields `root/transformer_ref`.
        let reference_dit_dir = dit_dir.with_file_name(REFERENCE_DIT_PARTITION);
        let text_encoder_dir = staged_text_encoder
            .map_or_else(|| root.join(TEXT_ENCODER_COMPONENT), Path::to_path_buf);
        Self {
            dit_dir,
            reference_dit_dir,
            text_encoder_dir,
            shared_root: root.to_path_buf(),
        }
    }

    /// Require the DiT partition for `tier`: the directory exists and carries a `config.json`, and
    /// its on-disk quantization marker agrees with `tier`.
    pub fn require_dit(&self, tier: Tier) -> Result<()> {
        require_component_dir(&self.dit_dir, tier, DIT_COMPONENT)?;
        reconcile_tier(&self.dit_dir, tier, DIT_COMPONENT)?;
        validate_group_size(&self.dit_dir, DIT_COMPONENT)
    }

    /// Require the `ref2va` partition for `tier` — called only when a request resolves to
    /// [`MiniMaxH3Task::Ref2va`](crate::model::MiniMaxH3Task), never at load — see the module docs
    /// for why this lane probes lazily where MLX probes eagerly.
    pub fn require_reference_dit(&self, tier: Tier) -> Result<()> {
        require_component_dir(&self.reference_dit_dir, tier, REFERENCE_DIT_PARTITION)?;
        reconcile_tier(&self.reference_dit_dir, tier, REFERENCE_DIT_PARTITION)?;
        validate_group_size(&self.reference_dit_dir, REFERENCE_DIT_PARTITION)
    }

    /// Require the condition encoder: the directory exists and carries a `config.json`, and — if it
    /// is packed — declares [`GROUP_SIZE`].
    ///
    /// Takes **no** `Tier`, and that asymmetry is the sc-19120 contract, not an oversight: the
    /// shipped manifest pairs one dense text encoder with all three DiT tiers, so the TE's tier is
    /// whatever the `{base}.scales` in it say and the loaders auto-detect it ([`crate::quant::lin`]).
    pub fn require_text_encoder(&self) -> Result<()> {
        let dir = &self.text_encoder_dir;
        if !dir.join("config.json").is_file() {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID}: missing {} — stage the tier's `{TEXT_ENCODER_COMPONENT}` directory as \
                 the '{TEXT_ENCODER_COMPONENT}' component, or point at a snapshot root that holds \
                 `{TEXT_ENCODER_COMPONENT}/`",
                dir.join("config.json").display()
            )));
        }
        validate_group_size(dir, TEXT_ENCODER_COMPONENT)
    }

    /// The tier a component directory actually is, read from its `config.json` marker — `None` when
    /// the component is dense (the `bf16` tier writes no marker).
    pub fn staged_bits(dir: &Path) -> Result<Option<i32>> {
        Ok(packed_config(dir)?.map(|c| c.bits))
    }
}

/// The component's `config.json` `quantization` block, or `None` when the component is dense (no
/// `config.json`, or a `config.json` with no marker — the `bf16` tier).
fn packed_config(dir: &Path) -> Result<Option<PackedConfig>> {
    let path = dir.join("config.json");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(CandleError::Msg(format!(
                "{MODEL_ID} tier: read {}: {err}",
                path.display()
            )))
        }
    };
    let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|err| {
        CandleError::Msg(format!("{MODEL_ID} tier: parse {}: {err}", path.display()))
    })?;
    Ok(PackedConfig::from_config(&json))
}

/// The directory exists and carries the `config.json` every component ships — the probe
/// `mlx_gen_minimax_h3::model` runs, with the tier named so the message says which install is wrong.
fn require_component_dir(dir: &Path, tier: Tier, component: &str) -> Result<()> {
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "{MODEL_ID}: the `{}` tier has no `{component}/` component at {} — stage the tier's \
             `{component}` directory as the '{component}' component, or point at a snapshot root \
             that holds `{component}/`",
            tier.as_str(),
            dir.display()
        )));
    }
    let config = dir.join("config.json");
    if !config.is_file() {
        return Err(CandleError::Msg(format!(
            "{MODEL_ID}: the `{}` tier's `{component}/` at {} has no config.json — an incomplete \
             component directory loads clean and fails in the middle of the render",
            tier.as_str(),
            dir.display()
        )));
    }
    Ok(())
}

/// Reconcile a caller's requested `tier` against the tier actually staged in `dir`.
///
/// **MiniMax-H3 never quantizes at load** (see the module docs), so this is an assertion, not an
/// instruction. Both disagreements are hard errors: a packed component under a `bf16` request would
/// otherwise render from a tier the caller did not ask for, and a dense component under a `q4`
/// request cannot be satisfied at all.
fn reconcile_tier(dir: &Path, tier: Tier, component: &str) -> Result<()> {
    let staged = MiniMaxH3TierPaths::staged_bits(dir)?;
    match (staged, tier.bits()) {
        (Some(staged_bits), Some(want)) if staged_bits != want => Err(CandleError::Msg(format!(
            "{MODEL_ID}: the `{}` tier was requested (bits {want}), but the `{component}/` staged at \
             {} declares quantization bits {staged_bits} in its config.json — the on-disk tier is \
             authoritative; stage the tier you asked for",
            tier.as_str(),
            dir.display()
        ))),
        (Some(staged_bits), None) => Err(CandleError::Msg(format!(
            "{MODEL_ID}: the dense `{}` tier was requested, but the `{component}/` staged at {} \
             declares quantization bits {staged_bits} in its config.json — stage the dense tier's \
             `{component}` directory, or request the packed tier that is staged",
            tier.as_str(),
            dir.display()
        ))),
        (None, Some(want)) => Err(CandleError::Msg(format!(
            "{MODEL_ID}: the `{}` tier was requested (bits {want}), but the `{component}/` staged at \
             {} carries no quantization marker — MiniMax-H3 does not quantize at load (the DiT's \
             66_280_430_080 dense bytes plus the packed output will not co-reside); stage the \
             pre-quantized tier's `{component}` directory",
            tier.as_str(),
            dir.display()
        ))),
        // Packed and agreed, or dense and agreed: the loaders build from `.scales` or from the
        // dense weight, and nothing reads a manifest to decide.
        _ => Ok(()),
    }
}

/// Validate a packed component's declared group size against [`GROUP_SIZE`] — a no-op for a dense
/// component. See the module docs for why this check exists and the bit-width one does not.
fn validate_group_size(dir: &Path, component: &str) -> Result<()> {
    let Some(cfg) = packed_config(dir)? else {
        return Ok(());
    };
    let declared = cfg.group_size as usize;
    if declared != GROUP_SIZE {
        return Err(CandleError::Msg(format!(
            "{MODEL_ID}: the `{component}/` staged at {} declares quantization group_size \
             {declared}, but this engine reads packed weights at {GROUP_SIZE} — a mismatched group \
             size does not fail cleanly (the inferred bit width comes out legal-looking and wrong, \
             sc-15154), so it is rejected here; re-pack the tier at {GROUP_SIZE}",
            dir.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a component dir carrying a packed `config.json` marker.
    fn packed_component(root: &Path, name: &str, bits: i32, group_size: usize) -> PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"quantization": {{"bits": {bits}, "group_size": {group_size}}}}}"#),
        )
        .unwrap();
        dir
    }

    /// Write a component dir carrying a dense `config.json` (no `quantization` marker).
    fn dense_component(root: &Path, name: &str) -> PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"num_layers": 50}"#).unwrap();
        dir
    }

    /// The flat upstream layout: every component under the root, the reference partition a sibling
    /// of the base one.
    #[test]
    fn a_flat_snapshot_resolves_every_component_under_the_root() {
        let root = Path::new("/snap");
        let paths = MiniMaxH3TierPaths::resolve(root, None, None);
        assert_eq!(paths.dit_dir, PathBuf::from("/snap/transformer"));
        assert_eq!(
            paths.reference_dit_dir,
            PathBuf::from("/snap/transformer_ref"),
            "on a flat snapshot the sibling rule lands under the root"
        );
        assert_eq!(paths.text_encoder_dir, PathBuf::from("/snap/text_encoder"));
        assert_eq!(paths.shared_root, PathBuf::from("/snap"));
    }

    /// **The sibling rule.** A staged tier DiT puts the reference partition next to the STAGED dir,
    /// not under the snapshot root — a split install has no `root/transformer_ref` at all, and the
    /// VAEs/tokenizer still resolve against the root.
    #[test]
    fn a_staged_tier_puts_the_reference_partition_beside_the_staged_dit() {
        let root = Path::new("/snap");
        let paths =
            MiniMaxH3TierPaths::resolve(root, Some(Path::new("/tiers/q4/transformer")), None);
        assert_eq!(paths.dit_dir, PathBuf::from("/tiers/q4/transformer"));
        assert_eq!(
            paths.reference_dit_dir,
            PathBuf::from("/tiers/q4/transformer_ref"),
            "the reference DiT is the staged DiT's sibling, never root/transformer_ref"
        );
        assert_eq!(
            paths.text_encoder_dir,
            PathBuf::from("/snap/text_encoder"),
            "an unstaged text encoder still resolves against the root — the two are independent"
        );
        assert_eq!(
            paths.shared_root,
            PathBuf::from("/snap"),
            "the VAEs and tokenizer are tier-agnostic"
        );
    }

    /// The text encoder is staged **independently** of the DiT (sc-19120) — a packed TE beside a
    /// dense DiT, and vice versa, are both expressible.
    #[test]
    fn the_text_encoder_is_staged_independently_of_the_dit() {
        let paths = MiniMaxH3TierPaths::resolve(
            Path::new("/snap"),
            None,
            Some(Path::new("/tiers/q4/text_encoder")),
        );
        assert_eq!(paths.dit_dir, PathBuf::from("/snap/transformer"));
        assert_eq!(
            paths.text_encoder_dir,
            PathBuf::from("/tiers/q4/text_encoder")
        );
    }

    /// A `q4` request against a staged `q4` DiT resolves and reconciles cleanly.
    #[test]
    fn a_matching_packed_tier_is_admitted() {
        let tmp = tempfile::tempdir().unwrap();
        let staged = packed_component(tmp.path(), "transformer", 4, 64);
        let paths = MiniMaxH3TierPaths::resolve(tmp.path(), Some(&staged), None);
        paths
            .require_dit(Tier::Q4)
            .expect("a q4 tier under a q4 request");
        assert_eq!(
            MiniMaxH3TierPaths::staged_bits(&staged).unwrap(),
            Some(4),
            "the marker is read back off the staged dir"
        );
    }

    /// **`bf16` resolves to the dense layout** — no marker, nothing to validate.
    #[test]
    fn bf16_resolves_to_the_dense_layout() {
        let tmp = tempfile::tempdir().unwrap();
        let staged = dense_component(tmp.path(), "transformer");
        dense_component(tmp.path(), "text_encoder");
        let paths = MiniMaxH3TierPaths::resolve(tmp.path(), None, None);
        paths
            .require_dit(Tier::Bf16)
            .expect("a dense component under a bf16 request");
        paths
            .require_text_encoder()
            .expect("a dense text encoder validates trivially");
        assert_eq!(
            MiniMaxH3TierPaths::staged_bits(&staged).unwrap(),
            None,
            "the dense tier carries no quantization marker"
        );
    }

    /// **A missing tier dir is a typed error naming the tier and the path.**
    ///
    /// The message is asserted on the *absent-directory* wording specifically, not merely on "some
    /// error mentioning q4". Both refusal branches in `require_component_dir` name the tier, the
    /// component and the path, so a weaker assertion passes with the `is_dir` check deleted — the
    /// mutation that survived this test's first draft.
    #[test]
    fn a_missing_tier_dir_names_the_tier_and_the_path() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = MiniMaxH3TierPaths::resolve(tmp.path(), None, None);
        let err = paths
            .require_dit(Tier::Q4)
            .expect_err("a missing component must be refused");
        let msg = err.to_string();
        assert!(msg.contains("q4"), "the message must name the tier: {msg}");
        assert!(
            msg.contains("has no `transformer/` component at"),
            "an ABSENT directory must be distinguished from a present-but-incomplete one: {msg}"
        );
        assert!(
            msg.contains(&tmp.path().display().to_string()),
            "the message must name the path: {msg}"
        );
    }

    /// A component directory present but missing its `config.json` is refused — the shape an
    /// interrupted download leaves behind.
    #[test]
    fn a_component_dir_without_a_config_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("transformer")).unwrap();
        let paths = MiniMaxH3TierPaths::resolve(tmp.path(), None, None);
        let err = paths.require_dit(Tier::Bf16).expect_err("no config.json");
        assert!(err.to_string().contains("no config.json"), "{err}");
    }

    /// A `q4` request against a staged `q8` tier is a hard error, not a silent downgrade.
    #[test]
    fn a_tier_disagreement_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        packed_component(tmp.path(), "transformer", 8, 64);
        let paths = MiniMaxH3TierPaths::resolve(tmp.path(), None, None);
        let err = paths
            .require_dit(Tier::Q4)
            .expect_err("q4 requested, q8 staged");
        let msg = err.to_string();
        assert!(msg.contains("bits 8"), "{msg}");
        assert!(msg.contains("on-disk tier is authoritative"), "{msg}");
    }

    /// A packed component under a **dense** request is refused too — the direction a resolver that
    /// only checked "did I get at least what I asked for" would miss.
    #[test]
    fn a_packed_component_under_a_bf16_request_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        packed_component(tmp.path(), "transformer", 4, 64);
        let paths = MiniMaxH3TierPaths::resolve(tmp.path(), None, None);
        let err = paths
            .require_dit(Tier::Bf16)
            .expect_err("bf16 requested, q4 staged");
        assert!(err.to_string().contains("dense `bf16` tier"), "{err}");
    }

    /// A dense component under a `q4` request names the reason H3 cannot simply quantize at load.
    #[test]
    fn a_dense_component_under_a_packed_request_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        dense_component(tmp.path(), "transformer");
        let paths = MiniMaxH3TierPaths::resolve(tmp.path(), None, None);
        let err = paths
            .require_dit(Tier::Q4)
            .expect_err("q4 requested, dense staged");
        assert!(
            err.to_string().contains("does not quantize at load"),
            "{err}"
        );
    }

    /// **The group-size gate** (sc-15154): a tier packed at 32 and read at 64 derives a
    /// legal-looking, wrong bit width, so it is rejected by name.
    #[test]
    fn a_non_64_group_size_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        packed_component(tmp.path(), "transformer", 4, 32);
        let paths = MiniMaxH3TierPaths::resolve(tmp.path(), None, None);
        let err = paths
            .require_dit(Tier::Q4)
            .expect_err("a group-32 tier must be refused");
        let msg = err.to_string();
        assert!(msg.contains("group_size 32"), "{msg}");
        assert!(msg.contains("64"), "{msg}");
    }

    /// The text encoder's group size is validated even though its **tier** is not reconciled — the
    /// sc-19120 asymmetry, held by a test so it cannot be "tidied" into symmetry.
    #[test]
    fn the_text_encoder_validates_group_size_but_not_tier() {
        let tmp = tempfile::tempdir().unwrap();
        // A packed TE with no DiT tier in sight validates fine — the two are independent.
        packed_component(tmp.path(), "text_encoder", 4, 64);
        let paths = MiniMaxH3TierPaths::resolve(tmp.path(), None, None);
        paths
            .require_text_encoder()
            .expect("a packed TE needs no matching DiT tier");

        // ...but a group-32 TE is still refused.
        let bad = tempfile::tempdir().unwrap();
        packed_component(bad.path(), "text_encoder", 4, 32);
        let paths = MiniMaxH3TierPaths::resolve(bad.path(), None, None);
        assert!(
            paths.require_text_encoder().is_err(),
            "a group-32 text encoder must be refused"
        );
    }

    /// The reference partition reconciles at the same tier as the base one — it is pre-quantized
    /// alongside it (`crate::convert` packs `transformer/` and `transformer_ref/` alike).
    #[test]
    fn the_reference_partition_reconciles_at_the_base_tier() {
        let tmp = tempfile::tempdir().unwrap();
        packed_component(tmp.path(), "transformer", 4, 64);
        packed_component(tmp.path(), "transformer_ref", 4, 64);
        let paths = MiniMaxH3TierPaths::resolve(tmp.path(), None, None);
        paths.require_dit(Tier::Q4).unwrap();
        paths.require_reference_dit(Tier::Q4).unwrap();

        // A base-only install (the normal off-Mac shape) fails ONLY the reference probe.
        let partial = tempfile::tempdir().unwrap();
        packed_component(partial.path(), "transformer", 4, 64);
        let paths = MiniMaxH3TierPaths::resolve(partial.path(), None, None);
        paths
            .require_dit(Tier::Q4)
            .expect("t2va/fl2va stay online without transformer_ref");
        assert!(
            paths.require_reference_dit(Tier::Q4).is_err(),
            "a ref2va request against a snapshot with no transformer_ref must be refused"
        );
    }

    /// The tier↔bits mapping, pinned.
    #[test]
    fn tier_bits_map_the_published_tier_names() {
        assert_eq!(Tier::Q4.bits(), Some(4));
        assert_eq!(Tier::Q8.bits(), Some(8));
        assert_eq!(Tier::Bf16.bits(), None);
        assert_eq!(Tier::Q4.as_str(), "q4");
        assert_eq!(Tier::Q8.as_str(), "q8");
        assert_eq!(Tier::Bf16.as_str(), "bf16");
    }

    /// The group size is the shared constant, not a crate-local re-declaration.
    #[test]
    fn group_size_is_the_shared_constant() {
        assert_eq!(GROUP_SIZE, candle_gen::quant::MLX_GROUP_SIZE);
        assert_eq!(GROUP_SIZE, 64);
    }
}
