//! MiniMax-H3's shared-ladder `MemoryProviderContract` on Candle/CUDA (sc-18659).
//!
//! The sibling declaration to `mlx_gen_minimax_h3::memory_strategy`. The contract carries a
//! [`MemoryBackendRealization`], so the two backends are *allowed* to differ — and here they
//! genuinely do, in three ways that are worth stating before the code:
//!
//! 1. **No fitted curve.** [`MemoryProviderContract::calibration`] is `None`. A provider with no
//!    calibration identity can run its resident path and can never claim a verified optimized fit —
//!    which is exactly the truth here, and is enforced: `standard_memory_strategy_safety_check`
//!    refuses any optimized selection when the identity is absent.
//!
//!    The formula is [`MemoryFormulaKind::ComponentPhaseEnvelope`] as of sc-18665, but over a single
//!    phase and the single [`MemoryFormulaVariable::AssetBytes`] input — it is the floor arm plus
//!    the AdaLN exclusion, not MLX's fitted five-variable envelope. The variant changed because it
//!    is the *only* one that carries `resident_components`; the absent calibration identity, not the
//!    formula shape, is what keeps every optimized selection failing closed.
//! 2. **Staged residency is implemented but under-declared (sc-17156 / sc-18660).** MLX declares
//!    [`MemoryStrategy::StagedResidency`] `Implemented`. This lane now *does the same thing* —
//!    `MiniMaxH3::generate_impl` releases each heavy component before mapping the next, and the
//!    provider **forces** `OffloadPolicy::Sequential` so a caller cannot opt out — but the rung
//!    stays declared `Missing` until its executable behavior seam lands, and that seam is itself
//!    gated on a `calibration` identity this backend does not have. An implemented optimized rung
//!    without a seam fails catalog conformance, and under-declaring is the safe direction: the
//!    staging still happens, and nothing is admitted on an unexercisable claim.
//! 3. **No fused streaming SDPA.** The MLX verdict that attention scratch is already streamed —
//!    peak tracking `4·B·H·S·D` with no materialized score tensor — is a property of *MLX's* fused
//!    kernel and **must not be copied here**. Candle materializes scores, so bounding attention
//!    (sc-18661) may buy real memory on this backend even if it buys none on MLX.
//!
//! # Why the asset facts are the full four components anyway
//!
//! `conditioning_bytes` charges the 66.71 GB Qwen3-VL-32B text encoder, which this crate executes
//! through [`crate::text_encoder`] (sc-17155). Asset facts are the render's byte floor, not a
//! capability claim: a candle render of this family needs the conditioner, and a contract that
//! declared zero there would publish a floor small enough to admit a request that cannot possibly
//! run. Capability lives in `strategies`, where every optimized rung is honestly `Missing`.
//!
//! # Stage attribution
//!
//! The ~53 GB memory floor measured for this family is the **conditioning** stage — the dense text
//! encoder in isolation — not the DiT and not activation pressure. The DiT's own denoise-resident
//! cost is a separate, genuinely tiered quantity. Those measurements were taken on MLX; sc-17156
//! landed this backend's pipeline but no fitted curve of its own, which is the other reason
//! `calibration` is `None`.
//!
//! # The packed `q4`/`q8` tiers are live on this lane (sc-20267)
//!
//! **This crate is no longer bf16-only, and several notes in this file used to say that it was.**
//! [`crate::quant::lin`] auto-detects the MLX affine triple and builds a packed base with no dense
//! transient, [`crate::tier`] resolves and reconciles the published `q4` / `q8` / `bf16` subtrees, and
//! [`crate::dit::block::AdaLnProjection`] is one of the two DiT loaders that pack. Three consequences
//! land in this module, each carried by code below rather than left in prose:
//!
//! * the per-tier asset sizes are declared ([`DIT_Q4_BYTES`] and its three siblings) — **from the
//!   manifest, not measured**, and their section note says exactly what that costs;
//! * `resolved_adaln_bytes` finally owes and pays the **marker leg** its old doc promised: it reads
//!   the staged DiT's own `quantization.bits` and re-derives the projection stack at that width,
//!   taking `min(marker, footprint)` the way the MLX sibling does;
//! * **the container is candle's, not MLX's.** A 4-bit MLX pack repacks into GGUF `Q4_1` and an
//!   8-bit one into `Q8_0` (`candle_gen::quant::repack_packed_weight`), so the packed stack sizes are
//!   *not* the MLX sibling's at `q4` — see `adaln_stack_bytes`, where that divergence is derived
//!   rather than reconciled away.
//!
//! What has **not** changed: there is still no fitted curve, no calibration identity and no behavior
//! seam on this lane, so every optimized rung stays `Missing` and no tier makes anything new
//! admissible. The tiers change what a render *costs*, not what this provider *claims*.
//!
//! # The configured LoRA stack is charged (sc-18650)
//!
//! `overlay_bytes` was the literal `0` from sc-18659 until this story, and — unlike the MLX
//! sibling's identical defect, whose contract was still unpublished — **this one shipped**. sc-18724
//! landed the adapter seam on 2026-08-13 and nothing brought the declaration with it, so a LoRA
//! render's resident factors were charged nothing on a live path. They are now a typed
//! [`MemoryComponentKind::AdapterStack`] at [`MemoryComponentResidency::WholeRender`], sized through
//! the shared `adapter_stack_resident_bytes` at [`AdapterResidencyMode::Additive`] — see
//! [`adapter_overlay_bytes`], which is also where the probe that makes the bytes *knowable* lives.
//!
//! **That probe is a fourth genuine divergence from MLX**, and it is forced by the two lanes' `load`
//! boundaries rather than chosen. MLX's `load` refuses a nonexistent adapter outright, so it can
//! size with a flat `ok_or_else(Err)`; this one never has, and two shipped guards
//! (`crate::model::tests::a_staged_lora_survives_load_rather_than_being_dropped` and its knob-refusal
//! sibling) drive a path that does not exist because what they assert is retention and refusal, not
//! sizing. So the absent case is charged an **exact** 0 — nothing can become resident from a file
//! `read_adapter` cannot even `stat` — while a file that *is* there and cannot be sized fails closed.
//! The declaration and the variable are both conditional on a non-zero overlay, so a render with no
//! adapter publishes byte-for-byte the contract this provider always has.

use candle_gen::candle_core::quantized::GgmlDType;
use candle_gen::candle_core::DType;
use candle_gen::gen_core::{
    adapter_stack_resident_bytes, safetensors_path_bytes, AdapterResidencyMode, Error as CoreError,
    LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase, MemoryProviderContract,
    MemoryResidentComponent, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategySupport, MemoryWindowMaterialization,
    ResidentRequestMemory, TransformerComponent, WeightsSource,
};

use crate::model::{BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION};
use crate::MODEL_ID;

// --- measured asset facts -------------------------------------------------------------------
//
// Exact `.safetensors` bytes under each component directory of the upstream bf16 snapshot
// (`MiniMaxAI/MiniMax-H3` @ `939557dc`). Identical to the MLX sibling's, deliberately: these are
// facts about the checkpoint, not about a backend, and the two must not drift apart.

/// Qwen3-VL-32B text encoder — 14 shards, the conditioning component. 66.71 GB.
pub const TEXT_ENCODER_BYTES: u64 = 66_714_912_872;

/// One 33 B DiT partition at bf16 — 14 shards. 66.28 GB. A render loads exactly one of
/// `transformer` / `transformer_ref`, so it is charged once.
pub const DIT_BF16_BYTES: u64 = 66_280_504_216;

/// Video VAE — 3 shards; the decoder is a 36-layer transformer. 10.42 GB.
pub const VIDEO_VAE_BYTES: u64 = 10_415_558_888;

/// Audio VAE — 1 shard. 0.61 GB.
pub const AUDIO_VAE_BYTES: u64 = 605_429_340;

// --- DECLARED per-tier asset sizes (sc-20267) --------------------------------------------------
//
// **Declared, NOT measured, and NOT runtime peaks.** Every figure in this section is an
// `estimatedSizeBytes` row copied out of SceneWorks' `config/manifests/builtin.models.jsonc` for
// repo `SceneWorks/minimax-h3-mlx` at revision `137ce668c55a20bc0935fd1cf2a3de8448abb7f4`. Nothing
// in this crate measured any of them, no test in this crate can measure them (the tiers are 18-35 GB
// hosted artifacts), and none of them may be cited as a measurement.
//
// # They are HOSTED SUBTREE bytes, so they slightly OVER-declare the safetensors footprint
//
// The section above is `.safetensors`-only — precisely what `safetensors_path_bytes` sums at load.
// The manifest rows are **whole-subtree** bytes, including each tier's sidecars: `config.json`, the
// seven tokenizer/preprocessor files every tier subtree carries verbatim, and the shard index. The
// bf16 pair measures that gap exactly rather than leaving it asserted — the manifest's bf16 DiT row
// is [`MANIFEST_DIT_BF16_SUBTREE_BYTES`] against this crate's measured [`DIT_BF16_BYTES`], i.e.
// 65,034 B of sidecars on a 66.28 GB component (0.98 ppm). So each figure below is an **upper
// bound** on its tier's safetensors bytes, by about that much.
//
// **Over-declaring an asset floor is the SAFE direction**, and that is why the rows are carried as
// published rather than adjusted down by a guessed sidecar allowance. Under-declaring a floor admits
// a render that then OOMs — the failure the whole ladder exists to prevent, and the one the
// [`DIT_PARTITIONS`] note records this provider having already shipped once. Over-declaring one only
// leaves a little admission headroom on the table. A hand-guessed correction would trade a bounded,
// documented, sub-ppm over-declaration for an unbounded error in the direction that OOMs.
//
// # These are ON-DISK tier sizes, NOT runtime peaks
//
// No stage peak, resident figure or activation transient is declared here, and none may be derived
// from these numbers. The MLX sibling's `DENOISE_RESIDENT_Q4_BYTES` /
// `CONDITIONING_STAGE_PEAK_Q4_BYTES` / `CONDITIONING_STAGE_PEAK_Q8_BYTES` constants are deliberately
// **not** mirrored into this crate: they were measured on MLX/Metal, against MLX's allocator and its
// lazy materialization, so they are facts about that backend rather than about this one. The CUDA
// vram probe is what measures this lane's real runtime peaks, later and off this crate; until it has,
// this lane declares no peak at all — which is the same absence that keeps `calibration` `None`.

/// One 33 B DiT partition on the published **`q4`** tier — 18.78 GB. **Declared from the manifest**;
/// see the section note above for why it is an upper bound on the safetensors bytes.
pub const DIT_Q4_BYTES: u64 = 18_780_109_783;

/// One 33 B DiT partition on the published **`q8`** tier — 35.30 GB. **Declared from the manifest**;
/// see the section note above.
pub const DIT_Q8_BYTES: u64 = 35_302_064_357;

/// The packed **`q4`** Qwen3-VL-32B text encoder (sc-19120) — 18.72 GB. **Declared from the
/// manifest**; see the section note above.
///
/// Staged **independently of the DiT's tier** — `crate::tier::MiniMaxH3TierPaths::require_text_encoder`
/// takes no [`crate::tier::Tier`] at all — so a `q4` DiT beside a dense encoder is a legal install and
/// this constant is not implied by [`DIT_Q4_BYTES`].
pub const TEXT_ENCODER_Q4_BYTES: u64 = 18_722_713_964;

/// The packed **`q8`** Qwen3-VL-32B text encoder (sc-19120) — 33.72 GB. **Declared from the
/// manifest**; see [`TEXT_ENCODER_Q4_BYTES`] for the independence note.
pub const TEXT_ENCODER_Q8_BYTES: u64 = 33_723_765_614;

/// The manifest's **bf16 DiT subtree** row, declared for exactly one purpose: to size the sidecar gap
/// between a whole-subtree manifest figure and this crate's measured `.safetensors` sum.
///
/// It is the one tier where both accountings exist, so it is the only place the over-declaration in
/// the four constants above can be *quantified* rather than merely admitted:
/// `MANIFEST_DIT_BF16_SUBTREE_BYTES − DIT_BF16_BYTES = 65_034` B. Not an asset fact — the contract
/// never reads it, and [`DIT_BF16_BYTES`] remains the bf16 figure everything here uses.
pub const MANIFEST_DIT_BF16_SUBTREE_BYTES: u64 = 66_280_569_250;

/// Exact bytes the AdaLN precompute-and-evict drops, asserted against the loader in
/// [`crate::dit::adaln`]. Declared here so both backends' contracts carry the same figure, and
/// carried into the contract as a typed resident-component exclusion by the private `adaln_component`
/// (sc-18665).
///
/// This is the **bf16** figure, and as of sc-20267 it is one of three: the packed `q4` and `q8` tiers
/// hold the same 50 projections in a GGUF container, and the private `adaln_stack_bytes` derives each
/// width from the shipped configuration. The private `resolved_adaln_bytes` is what the contract
/// declares, and it is `min(marker, footprint)` over both legs.
pub const ADALN_EVICTED_BYTES: u64 = 26_020_915_200;

/// Bytes of the modulation table the precompute keeps in the projections' place, at the longest
/// schedule this model admits.
///
/// `MODULATION_PARAMS · modulation_rows · hidden_size · num_layers` elements at the block dtype,
/// with `modulation_rows` read off a real [`crate::MAX_STEPS`]-evaluation schedule rather than
/// derived by hand — `the_retained_table_is_the_worst_case_over_the_admitted_schedule` pins that.
///
/// **The evict is not free, and this is the price.** The contract declares the *net* difference,
/// because declaring the gross [`ADALN_EVICTED_BYTES`] claims a saving the runtime does not
/// deliver. Identical to the MLX sibling's figure, and for the same reason the two DiT footprints
/// are identical: this is a property of the checkpoint's schedule domain, not of a backend.
///
/// **It does NOT scale with a tier, and sc-20267 leaves that claim standing rather than revising
/// it.** The table is the projections' *output*, so its element type is the block's **compute** dtype
/// rather than any packed width — [`crate::dit::block::AdaLnProjection`] holds that dtype explicitly
/// and casts the activation to it on either tier, so a packed base emits the identical bf16 table a
/// dense one does. The resident side genuinely scales now that the packed tiers are live (26.02 GB
/// bf16 → 13.83 GB `q8` → 8.14 GB `q4`, all derived by the private `adaln_stack_bytes`) while this
/// figure stays at 3.87 GB across all three, which is exactly why the two quantities are declared
/// separately and why one factor must never be applied to both. See the private
/// `resolved_adaln_bytes`.
pub const ADALN_MODULATION_TABLE_MAX_BYTES: u64 = 3_870_720_000;

/// Contract-stable identity of the evictable AdaLN sub-stack, so a consumer can find the
/// declaration by name rather than by matching on bytes. Shared spelling with the MLX sibling.
pub const ADALN_COMPONENT_ID: &str = "dit_adaln_proj_stack";

/// Contract-stable identity of the configured LoRA stack (sc-18650), declared only when one is
/// configured. Deliberately **not** `pub`: it is the MLX sibling's literal spelling, and publishing
/// it would put a second copy of the same string under `scripts/check-workspace.py`'s cross-backend
/// `pub const` comparison for no consumer that needs the name.
const ADAPTER_STACK_COMPONENT_ID: &str = "adapter_stack";

/// The load shape this loader actually has, pinned rather than mirrored from the spec.
///
/// [`LoadShape::DeferredMaterialization`] means transformer blocks are materialized through a block
/// schedule. `MiniMaxH3Dit::load` builds the whole stack, so this provider is
/// [`LoadShape::EagerMaterialization`] whatever a caller asks for. sc-18662 changes it.
pub const LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;

/// The DiT partition directories a flat snapshot may carry, in the order a tie is broken.
///
/// **Both, not just the base one.** This read `const DIT_COMPONENT: &str = "transformer"` and sized
/// `root.join("transformer")` unconditionally, so a `ref2va` render — which denoises from
/// [`REFERENCE_DIT_PARTITION`] and never opens `transformer/` — was charged for a directory it does
/// not read, and a snapshot carrying only the reference partition was charged **zero** for its DiT.
/// A contract that under-reports 66 GB admits a render that then OOMs, which is the failure the
/// ladder exists to prevent.
///
/// The contract is built once per *load*, from a [`LoadSpec`] alone, and the task is a property of
/// each later *request* — so there is no single partition that is "the" DiT here. `resolve` takes
/// the **larger** of the two present: a render loads exactly one of them, so charging the larger is
/// both a true bound for whichever task arrives and still "charged once", which is what
/// [`DIT_BF16_BYTES`] means. An absent partition measures 0 and simply loses the max, so the
/// base-only install (the normal off-Mac shape) is charged exactly as it was.
///
/// Named through `crate::model`'s own `pub const`s rather than restating the literals, so
/// `tests/ref2va_checkpoint.rs`'s bare-literal ban has exactly one declaration site to exempt.
const DIT_PARTITIONS: [&str; 2] = [BASE_DIT_PARTITION, REFERENCE_DIT_PARTITION];

struct ComponentBytes {
    text_encoder: u64,
    dit: u64,
    /// The AdaLN sub-stack's residency **on the DiT that was actually resolved** — see
    /// [`resolved_adaln_bytes`].
    adaln: u64,
    video_vae: u64,
    audio_vae: u64,
    /// Load-exact bytes of the configured LoRA stack, held **resident for the whole render**
    /// (sc-18650). See [`adapter_overlay_bytes`], which resolves it.
    overlay: u64,
}

impl ComponentBytes {
    fn resolve(spec: &LoadSpec) -> candle_gen::gen_core::Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root.clone(),
            WeightsSource::File(path) => path.parent().unwrap_or(path).to_path_buf(),
        };
        // The larger of the two DiT partitions this snapshot carries — see [`DIT_PARTITIONS`]. A
        // staged override is honored per partition, exactly as the single-partition form did.
        //
        // **The winning partition's DIRECTORY is carried out alongside its bytes, and that is a
        // requirement rather than a convenience** (sc-20267): `resolved_adaln_bytes` now reads a
        // `quantization` marker out of the resolved DiT dir, so handing it an arbitrary one of the two
        // would read a tier marker off a partition that is not the one being charged. The pair is
        // taken as a unit for that reason.
        //
        // `.rev()` before `max_by_key` breaks a tie toward [`BASE_DIT_PARTITION`], which is the order
        // [`DIT_PARTITIONS`] documents: `Iterator::max_by_key` keeps the LAST maximum, and the two
        // partitions are byte-identical on a flat snapshot. It costs nothing on a well-formed install
        // — `crate::convert` packs `transformer/` and `transformer_ref/` at the same width, so their
        // markers agree — but leaving the choice to iteration order would make the declaration depend
        // on it.
        let (dit_dir, dit) = DIT_PARTITIONS
            .iter()
            .rev()
            .map(|partition| {
                let dir = match spec.components.get(*partition) {
                    Some(WeightsSource::Dir(staged)) => staged.clone(),
                    _ => root.join(partition),
                };
                let bytes = safetensors_path_bytes(&dir);
                (dir, bytes)
            })
            .max_by_key(|(_, bytes)| *bytes)
            .expect("DIT_PARTITIONS is a non-empty array");
        // The condition encoder honors its own staged override too (sc-19120 / sc-20267). It did NOT
        // before, and the omission was the [`DIT_PARTITIONS`] failure in a second place: the TE's tier
        // is staged **independently** of the DiT's, so a split `q4` install stages
        // `crate::tier::TEXT_ENCODER_COMPONENT` outside the snapshot and may carry no
        // `root/text_encoder` at all — which measured 0 and published an 18.72 GB under-declaration of
        // the largest single component this family loads. `safetensors_path_bytes` on an absent
        // directory returns 0, so the flat-snapshot path is unchanged.
        let text_encoder = match spec.components.get(crate::tier::TEXT_ENCODER_COMPONENT) {
            Some(WeightsSource::Dir(staged)) => staged.clone(),
            _ => root.join(crate::tier::TEXT_ENCODER_COMPONENT),
        };
        Ok(Self {
            text_encoder: safetensors_path_bytes(text_encoder),
            dit,
            // Resolved against the partition that was actually charged, whichever of the two won the
            // max — see [`resolved_adaln_bytes`] for both legs. Declaring the flat bf16 figure against
            // a smaller staged winner would declare a sub-stack larger than the stack containing it.
            adaln: resolved_adaln_bytes(&dit_dir, dit),
            video_vae: safetensors_path_bytes(root.join("vae")),
            audio_vae: safetensors_path_bytes(root.join("audio_vae")),
            overlay: adapter_overlay_bytes(spec)?,
        })
    }

    /// The two decoders are one contract field; H3 is the first family with two of them.
    fn decoder(&self) -> u64 {
        self.video_vae.saturating_add(self.audio_vae)
    }

    fn base(&self) -> u64 {
        self.text_encoder
            .saturating_add(self.dit)
            .saturating_add(self.decoder())
    }
}

/// **The adapter-file probe** — load-exact resident bytes of the configured LoRA stack (sc-18650).
///
/// Until this story `overlay_bytes` was the literal `0` and nothing in `load` had ever touched an
/// adapter path, so the two facts propped each other up. Both were wrong, and unlike the MLX
/// sibling's — whose contract was still unpublished — **this one shipped**:
/// [`crate::model::descriptor`] declares `supports_lora: true`, and since sc-18724
/// [`crate::model::MiniMaxH3::load_task_dit`] installs the published `lightx2v/Minimax-h3-Turbo`
/// exports through [`crate::adapters::apply_minimax_h3_adapters`]. A LoRA render's factors were
/// charged nothing on a live path.
///
/// [`AdapterResidencyMode::Additive`], not `Folded`, and that is read off this lane's own install
/// rather than copied: [`crate::adapters`] declares a **forward-time residual** over
/// [`crate::dit::layers::LinearNoBias`], "never a merged weight", deliberately tier-*blind* so it
/// composes over a packed base too. Nothing is ever folded away, so the factors are still resident at
/// the last denoise step. `Folded` would declare zero and re-open the same under-declaration under a
/// typed name.
///
/// # Why an ABSENT file is charged 0 while a PRESENT one must be sizable
///
/// The shared helper returns `None` for both, and a flat `ok_or_else(Err)` — the MLX shape — is
/// wrong here, because the two lanes' `load` boundaries genuinely differ. MLX's refuses a
/// nonexistent adapter outright (`!adapter.path.is_file()`); this one never has, and two shipped
/// tests depend on that — `crate::model::tests::a_staged_lora_survives_load_rather_than_being_dropped`
/// and `…::lokr_and_the_two_foreign_adapter_knobs_are_each_refused_individually` both drive a
/// `/turbo.safetensors` that does not exist, because what they assert is **retention** and **knob
/// refusal**, neither of which needs a real file. Turning their scenario into a load error would
/// delete two correct guards to satisfy a sizing concern that does not apply to them.
///
/// It does not apply because an absent path can put **no bytes anywhere**: `read_adapter`'s first
/// act is `std::fs::metadata`, so [`crate::adapters::apply_minimax_h3_adapters`] refuses the render
/// before a single factor is materialized. `0` there is *exact*, not a guess.
///
/// A **present** file is the opposite, and it is the case that makes this a fail-closed check rather
/// than a formality. `read_adapter` is extension-blind — it reads the bytes and parses safetensors
/// out of the buffer — while `safetensors_path_bytes` gates on the `.safetensors` extension and skips
/// dotfiles. So a perfectly loadable adapter named `turbo.bin`, `turbo.st` or `.turbo.safetensors`
/// installs 312 modules' worth of factors and sizes to **zero**: a live under-declaration in the OOM
/// direction, reachable with no malformed input at all. That is refused here rather than charged 0,
/// which is also what the MLX lane does with the same file.
pub fn adapter_overlay_bytes(spec: &LoadSpec) -> candle_gen::gen_core::Result<u64> {
    // `try_exists` rather than `exists`: the skip is taken only when the filesystem says the path is
    // definitely not there. An I/O error means we could not tell, which falls through to the sizing
    // helper and fails closed — `exists()` would have folded that into "absent" and charged 0.
    let present: Vec<_> = spec
        .adapters
        .iter()
        .filter(|adapter| !adapter.path.try_exists().is_ok_and(|there| !there))
        .cloned()
        .collect();
    adapter_stack_resident_bytes(&present, AdapterResidencyMode::Additive).ok_or_else(|| {
        CoreError::Unsupported(format!(
            "{MODEL_ID}: every adapter present on disk must have a non-zero load-exact safetensors \
             size before the memory contract can declare its resident overlay — {} sized to 0. The \
             render seam parses safetensors out of the file's bytes whatever it is named, but the \
             shared sizer counts only non-hidden `.safetensors`, so an adapter named otherwise would \
             load and be charged nothing. Rename it to `<name>.safetensors`.",
            spec.adapters
                .iter()
                .map(|adapter| adapter.path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))
    })
}

/// The AdaLN sub-stack's resident bytes on a DiT whose resolved footprint is `dit_bytes`.
///
/// [`ADALN_EVICTED_BYTES`] is a bf16 figure, and `ComponentBytes::resolve` already honours a staged
/// override on either [`DIT_PARTITIONS`] entry, charging whichever partition measures larger
/// (sc-17157). `dit_bytes` is therefore that winner, and this scales to it: the AdaLN stack is the
/// same architectural fraction of `transformer` and `transformer_ref` alike, so scaling to the
/// charged partition is correct for whichever task later arrives. Declaring the flat 26.02 GB
/// against a smaller staged DiT would declare a
/// sub-stack larger than the stack containing it, which `conformance_errors` refuses and
/// `Registry::memory_strategy_contract` turns into a hard error — a render that cannot resolve a
/// contract at all. `a_staged_dit_component_is_charged_at_its_own_size` proves this lane really can
/// resolve a smaller staged DiT, so the hazard is reachable here and not merely theoretical.
///
/// **Two legs as of sc-20267, and the SMALLER wins.** The old note here said this crate had "no packed
/// tier at all" and named this function as the one that would owe a marker leg if one ever landed. One
/// has: [`crate::quant::lin`] builds a packed base from the MLX affine triple and
/// [`crate::tier`] resolves the published `q4` / `q8` subtrees, so the debt is paid rather than
/// restated.
///
/// * the **marker** leg reads the staged DiT dir's own `config.json` `quantization.bits` — through
///   `crate::tier::MiniMaxH3TierPaths::staged_bits`, which is the *same* parse
///   `crate::tier::reconcile_tier` treats as decisive about which tier is staged, so the contract and
///   the loader cannot disagree about a tier — and re-derives the stack at that width through
///   [`adaln_stack_bytes`]. Exact at every published tier. A **missing or unreadable** marker falls
///   back to the bf16 [`ADALN_EVICTED_BYTES`], never to zero and never to an error: this function is
///   called while building a contract, an `Err` here would mean a render that cannot resolve a
///   contract at all, and a zero would declare an eviction that excludes nothing.
/// * the **footprint** leg is unchanged — the bf16 stack scaled by the resolved DiT's share of the
///   bf16 DiT. Never exact: the f32 I/O heads, `context_embedder`, `norm_out.linear` and every norm
///   that `mlx_gen_minimax_h3::convert::DENSE_BY_POLICY` leaves dense **in every tier** do not shrink,
///   so a whole-DiT ratio over-declares the projections' own share. But it reads nothing except a
///   number `ComponentBytes::resolve` already holds.
///
/// Taking the minimum is what makes this **safe** rather than merely accurate, because each leg closes
/// the other's failure: marker alone reports the full 26.02 GB for a packed tier whose marker is
/// missing, and footprint alone over-declares the eviction — and an over-declared *saving* is the OOM
/// direction, the same asymmetry [`ADALN_MODULATION_TABLE_MAX_BYTES`] is chosen on.
///
/// Containment holds unconditionally either way: [`ADALN_EVICTED_BYTES`] is 39.3 % of
/// [`DIT_BF16_BYTES`], so the footprint leg is always strictly below `dit_bytes` and a minimum taken
/// against it is therefore below it too.
///
/// # Which leg actually binds — and it is NOT the same leg the MLX sibling's binds at `q4`
///
/// Both legs are live at both packed tiers, and the winner differs by tier because the two legs scale
/// differently. At the published footprints:
///
/// | tier | marker leg | footprint leg | this lane declares |
/// |---|---:|---:|---|
/// | `q8` | 13,828,147,200 | 13,859,158,645 | **marker**, by 31.0 MB |
/// | `q4` | 8,138,188,800 | 7,372,841,379 | **footprint**, by 765.3 MB |
///
/// So at `q4` the marker leg is the *larger* of the two here, and `min` discards it. That is not the
/// marker leg failing — it is the safe direction working. `resident_bytes` is the quantity the
/// eviction is netted out of, so a smaller value declares a *smaller* saving and therefore a *larger*
/// steady-state charge; taking the tighter bound can only make admission more conservative.
///
/// The MLX sibling's `q4` marker is 7,325,337,600 — 47.5 MB **below** the same footprint leg — so its
/// `q4` declaration comes from its marker where this one comes from its footprint. The two lanes
/// therefore disagree about which leg binds at `q4`, for exactly the container reason
/// [`adaln_stack_bytes`] documents, and neither is wrong. Do not "reconcile" them.
fn resolved_adaln_bytes(dit_dir: &std::path::Path, dit_bytes: u64) -> u64 {
    if dit_bytes == 0 {
        // Nothing was resolved, so there is no footprint to scale to and the declaration falls back
        // to the architecture fact. `conformance_errors` skips sub-stack containment against zero
        // asset facts for exactly this case.
        return ADALN_EVICTED_BYTES;
    }
    // `.ok().flatten()`: an unreadable or unparseable config.json and a dense tier both mean "no
    // packed width to derive", and both take the bf16 fallback. A contract build has no channel for
    // the difference, and the fallback is the conservative answer for either.
    let marked = crate::tier::MiniMaxH3TierPaths::staged_bits(dit_dir)
        .ok()
        .flatten()
        .map_or(ADALN_EVICTED_BYTES, adaln_stack_bytes);
    // u128 because `ADALN_EVICTED_BYTES · DIT_BF16_BYTES` is ~1.7e21 and overflows u64.
    let scaled = (u128::from(ADALN_EVICTED_BYTES) * u128::from(dit_bytes)
        / u128::from(DIT_BF16_BYTES)) as u64;
    marked.min(scaled)
}

/// The GGUF container candle **repacks** an MLX pack of `bits` width into — `Q4_1` at 4 bits, `Q8_0`
/// at 8, and `None` for a width the repack does not serve.
///
/// Mirrors `candle_gen::quant::repack_packed_weight`'s own match, which is the only authority on this:
/// it is the function every packed load on this lane goes through. Deliberately **not**
/// `candle_gen::quant::ggml_dtype`, whose `Quant::Q4` arm is `Q4_0` — that is the in-place dense fold's
/// container, and its own doc records that "the **packed** path uses `Q4_1` instead". Confusing the two
/// would under-declare a `q4` stack by the f16 minimum `Q4_1` carries per block and `Q4_0` does not.
fn packed_container(bits: i32) -> Option<GgmlDType> {
    match bits {
        4 => Some(GgmlDType::Q4_1),
        8 => Some(GgmlDType::Q8_0),
        _ => None,
    }
}

/// Device bytes the 50-block `adaln_proj` stack holds on a tier packed at `bits`.
///
/// The same accounting `crate::quant::lin` records into `TieredLinear::base_bytes` and
/// [`crate::dit::block::AdaLnProjection::nbytes`] reports on a *loaded* projection, done from the
/// shipped configuration instead of from a device tensor, because a contract is resolved before
/// anything is loaded. Per block: the repacked GGUF blocks over the `[out, in]` weight, plus the dense
/// `{base}.bias` row, which `crate::quant::lin` loads at the compute dtype on **either** tier.
/// `bits >= 16` is the dense arm — the unpacked bf16 `weight` + `bias`, i.e. [`ADALN_EVICTED_BYTES`],
/// derived here rather than returned as the constant so a configuration change moves both together.
///
/// The block size and per-block byte cost are read off [`GgmlDType`] rather than typed, so the figures
/// track whatever candle's container actually costs.
///
/// # This backend's container is NOT the MLX triple, and that is the point
///
/// The MLX lane's packed stack is `out · in · bits / 8` code bytes plus a bf16 `scales` **and**
/// `biases` entry per 64-element input group. Candle holds neither: the repack folds the MLX
/// scales/biases *into* the GGUF blocks, so what is resident is `blocks · type_size` and nothing else.
/// The two agree at `q8` and diverge at `q4`, both for the same reason:
///
/// | tier | candle container | per 32 weights | this crate | MLX sibling |
/// |---|---|---:|---:|---:|
/// | `q8` | `Q8_0` | 32 codes + one f16 scale = 34 B | 13,828,147,200 B | 13,828,147,200 B |
/// | `q4` | `Q4_1` | 16 codes + f16 scale + f16 min = 20 B | 8,138,188,800 B | 7,325,337,600 B |
///
/// `q8` coinciding is arithmetic luck: `Q8_0`'s 2 B per 32 elements happens to equal MLX's 4 B per
/// 64-element group summed across its two metadata tensors. `q4` does **not** coincide, because `Q4_1`
/// carries its f16 scale and f16 minimum per **32** elements where MLX carries them per **64** — twice
/// the metadata density over the same codes, so candle's `q4` stack is ~11.1 % larger.
///
/// **That divergence is real and correct, and must not be "fixed" to match MLX.** It is a fact about
/// which container each backend keeps resident, not a discrepancy between two accounts of one thing —
/// the contract carries a [`MemoryBackendRealization`] precisely so the two lanes may differ here.
/// Copying MLX's 7.33 GB across would under-declare what this backend actually holds, and an
/// under-declared resident is the OOM direction.
fn adaln_stack_bytes(bits: i32) -> u64 {
    let config = crate::dit::MiniMaxH3DitConfig::default();
    let out = config.adaln_out_features() as u64;
    let inp = config.time_embed_dim as u64;
    let blocks = config.num_layers as u64;
    // `bits >= 16` lands here through [`packed_container`]'s `None`, alongside every width
    // `repack_packed_weight` would refuse at load. Either way the dense bf16 stack is the conservative
    // answer, and it is the one `resolved_adaln_bytes`'s marker fallback needs.
    let Some(dtype) = packed_container(bits) else {
        return blocks * (out * inp + out) * compute_dtype_bytes();
    };
    blocks * packed_projection_bytes(dtype, out, inp)
}

/// Bytes one **packed** `[out, in]` projection holds on this lane: the repacked GGUF blocks, plus the
/// dense `{base}.bias` row that `crate::quant::lin` loads at the compute dtype on either tier.
///
/// Factored out of [`adaln_stack_bytes`] — which is only this, times the block count — for one reason:
/// it is the quantity `crate::quant::lin` records into `TieredLinear::base_bytes` at load, so
/// `the_packed_stack_arithmetic_is_the_loaders_own_accounting` can drive a **real** packed load at a
/// geometry a unit test can materialize and compare against this function. The shipped 96768x2688
/// stack is 8-14 GB of codes and cannot be built in-process, so the agreement has to be shown on the
/// formula rather than on the shipped tensors; sharing the formula is what makes that a property
/// instead of two restatements of the same guess.
fn packed_projection_bytes(dtype: GgmlDType, out: u64, inp: u64) -> u64 {
    let codes = (out * inp / dtype.block_size() as u64) * dtype.type_size() as u64;
    codes + out * compute_dtype_bytes()
}

/// Bytes per element of the DiT's compute dtype.
///
/// bf16, which is both the dense weight's element size (the `· 2` [`ADALN_EVICTED_BYTES`] is derived
/// with) and the width `crate::quant::lin` casts the dense `{base}.bias` to on the packed path. Read
/// off [`DType`] rather than typed as `2` so the declaration names the dtype it means. A function
/// rather than a `const` only because `DType::size_in_bytes` is not `const fn`.
fn compute_dtype_bytes() -> u64 {
    DType::BF16.size_in_bytes() as u64
}

/// The AdaLN projection stack as a typed, evictable intra-transformer component (sc-18665).
///
/// The candle lane runs the same precompute-and-evict the MLX lane does — `crate::model` passes
/// [`crate::dit::adaln::AdaLnResidency::PrecomputeAndEvict`] as a **literal**, and
/// `AdaLnCache::precompute_and_evict` refuses `Resident` outright — so the exclusion is an
/// unconditional property of this provider rather than a lever, and `bounded_by` is `None`.
///
/// Before this, the candle contract carried the full 26.02 GB over-charge: `transformer_bytes` is a
/// single load-exact scalar, and [`MemoryFormulaKind::AssetBytesPlusHeadroom`] has nowhere to record
/// that the projections do not survive into the denoise steady state.
///
/// * `kind` is [`MemoryComponentKind::TransformerSubStack`], not `Transformer`: these bytes are
///   already inside `asset_facts.transformer_bytes`, and a whole-transformer kind would charge them
///   twice. They are not auxiliary either, so they contribute nothing to `overlay_bytes` — which
///   carries the configured LoRA stack and only that (sc-18650), and is 0 on a render with no
///   adapter.
/// * `retained_bytes` makes the declaration **net**. The precompute keeps a modulation table in the
///   projections' place; declaring the gross figure claims a saving the runtime does not deliver.
fn adaln_component(resident_bytes: u64) -> MemoryResidentComponent {
    MemoryResidentComponent {
        id: ADALN_COMPONENT_ID.to_owned(),
        kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
        resident_bytes,
        bounded_by: None,
        residency: MemoryComponentResidency::PrecomputedThenEvicted {
            precomputed_in: MemoryPhase::Denoise,
            retained_bytes: ADALN_MODULATION_TABLE_MAX_BYTES,
            // Only tests that exist in THIS crate are named. The MLX sibling's evidence cites
            // `tests/adaln_evict_real_weights.rs` and `common::assert_adaln_phase_envelope`
            // (sc-19449); neither exists on this lane, and citing them here would be a claim about
            // coverage this backend does not have.
            evidence: "crate::dit::adaln::AdaLnCache::precompute_and_evict returns the bytes it \
                       released and crate::dit::block::DitBlock::evict_adaln performs the drop; \
                       tests/adaln_evict_memory.rs drives it under a counting global allocator, \
                       and crate::dit::adaln's own cache_bytes_are_independent_of_resolution_and_\
                       duration pins both the 26,020,915,200 B stack and the retained table's \
                       shape"
                .to_owned(),
        },
    }
}

/// The five capability entries. **Exactly one is `Implemented` — rung 0, `Resident`. Rungs 1-4 are
/// all `Missing`**, which is what the `match` below returns and what the ladder reads.
///
/// Rung 1 is the one that needs stating carefully, because the pipeline and the declaration disagree
/// **on purpose**. sc-17156 made the staging real — `MiniMaxH3::generate_impl` drops each heavy
/// component and synchronizes the device before mapping the next, and `MiniMaxH3` forces
/// [`OffloadPolicy::Sequential`](candle_gen::gen_core::OffloadPolicy) whatever the caller asks for,
/// so a caller cannot opt out. The *declaration* deliberately did not follow: the rung stays
/// `Missing` until its executable weights-free behavior seam lands, and that seam is gated on a
/// `calibration` identity this crate does not have (sc-18660 pinned that chain in a test). The long
/// note on the arm below gives the full reasoning, and the module header says the same thing.
///
/// So "implemented in the pipeline" and "declared `Implemented`" are two different facts here, and
/// only the first is true. Reading this as rung 1 being available to the ladder would be wrong in
/// the direction that matters: nothing is admitted on the strength of it.
///
/// Every entry publishes an empty [`MemoryParameterRanges`], which is correct in both directions:
/// rung 0 owns no numeric parameters, and a `Missing` rung must not publish a domain it cannot
/// honor. Flipping any of rungs 1-4 to `Implemented` without filling its domain is a conformance
/// error, not a silent under-declaration.
fn strategies() -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
                // Rung 1 is **implemented in code** as of sc-17156 — `MiniMaxH3::generate_impl`
                // stages all three phases and the offload policy is forced — but it stays declared
                // `Missing` here, and that is a deliberate under-declaration, not an oversight.
                //
                // `check_memory_strategy_contract` requires every *implemented optimized* rung to
                // carry an executable weights-free **behavior seam** (a `MemoryBehaviorRegistration`
                // with per-route fixtures, the MLX sibling's `MEMORY_BEHAVIOR`). Declaring the rung
                // without it fails catalog conformance — correctly: an optimized declaration with no
                // seam is a capability claim nothing can execute against.
                //
                // Under-declaring is the safe direction. `Missing` means the ladder never *selects*
                // staged residency for this provider, so the staging still happens (it is forced)
                // and nothing is admitted on the strength of an unproven declaration. The reverse —
                // declaring it and having the seam absent — would let admission believe a rung it
                // cannot exercise.
                //
                // Rung 2 (sc-18660) is `Missing` for the very same reason, and it too is **not** a
                // porting gap: the mechanism is present — `MiniMaxH3VideoVae::decode_clip` tiles at
                // the reference 256/64 through `crate::spatial_tiling::BoundedStitch`, the same
                // seam the MLX sibling declares `Implemented` on. What this backend lacks is one
                // link further back than the seam: a weights-free `MemoryBehaviorRegistration`
                // needs a `MemoryBehaviorFixture`, and that needs a `calibration` identity this
                // crate does not have (see `contract.calibration`). So the blocker for **both**
                // rungs is the absent calibration identity — not the seam, and not the mechanism.
                // A seam cannot be written before the identity exists.
                // `an_optimized_rung_here_is_blocked_by_the_missing_calibration_identity_not_the_\
                // seam` drives that chain and fails the moment it is removed.
                //
                // Rungs 3/4: sc-18661 / sc-18662.
                _ => MemoryStrategySupport::Missing,
            },
            parameters: MemoryParameterRanges::default(),
        })
        .collect()
}

fn build_contract(components: &ComponentBytes) -> MemoryProviderContract {
    MemoryProviderContract {
        provider_id: MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            // There is no block-wise host→device materialization path in this crate at all: the
            // DiT loads as a whole stack. sc-18662 builds one.
            host_to_device_block_materialization: false,
            // Answered even though rung 4 is `Missing`, because the field is deliberately not
            // optional.
            //
            // **The premise this used to rest on is gone.** It read "this crate has no packed tier
            // and therefore no MLX-affine → GGML repack seam", and named a packed candle tier
            // landing as "the change that turns a conforming realization into a
            // `HostFormatConversion` one". sc-20267 landed that tier: `crate::quant::lin` routes
            // every packed projection through `candle_gen::quant::repack_packed_weight`, which is
            // host-side work proportional to the weight.
            //
            // The declaration is unchanged all the same, on two grounds that are stated here rather
            // than left implicit, because the old comment's own prediction says otherwise:
            //
            // 1. **there is no window path here to characterize.**
            //    `host_to_device_block_materialization` above is `false` and rung 4 is `Missing` —
            //    the DiT loads as a whole stack — so this field describes a counterfactual either
            //    way, and `conformance_errors` only reads it when rung 4 is `Implemented`.
            // 2. **`DeviceFormatTransfer` is the obligation, and it is reachable.** The enum's own
            //    doc calls `HostFormatConversion` a transitional state a conforming rung SHOULD not
            //    be in, and says of exactly this repack that "nothing about Candle prevents it: the
            //    conversion is content-addressed by the source tensor and can be done once, ahead of
            //    the render, into a device-format artifact a window then maps and copies". Every
            //    other packed-tier candle provider in this workspace declares the same value.
            //
            // **sc-18662 still owns re-verifying this**, and the obligation is now sharper rather
            // than conditional: whatever window loader it builds must do the repack ahead of the
            // render, or it must change this field to `HostFormatConversion` and name its own
            // removal story. Building a per-window repack and leaving this value is the one
            // outcome the field exists to prevent.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: strategies(),
        // No decode-quality geometry table is declared for this route, and that is a statement of
        // fact rather than a declared-but-refused authority — identical to every other candle
        // video provider after the sc-18325 sweep.
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: LOAD_SHAPE,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        // Empty, and paired with the `Missing` rung-1 declaration above for the same reason: the
        // phases ARE implemented (`generate_impl` releases each component and synchronizes the
        // device via `crate::dit::release_device_memory`), but the rung they belong to cannot be
        // declared until its behavior seam exists, and a lifecycle block attached to a `Missing`
        // rung is a claim with no rung to hang on. The seam waits on a `calibration` identity, so
        // the rung and this block flip together when one lands.
        //
        // `phases` is no longer empty, and it declares exactly ONE entry. sc-18665's exclusion is
        // `PrecomputedThenEvicted { precomputed_in: Denoise }`, and `conformance_errors` requires
        // that phase to be a declared lifecycle phase — so `Denoise` is the minimum the eviction
        // needs, and declaring it is what ties the component's ordering back to `lifecycle` rather
        // than inventing an axis for it. `Conditioning` and `Decode` stay undeclared: they belong to
        // staged residency, which is still `Missing`, and declaring them would be exactly the false
        // phase hook `no_lifecycle_phase_is_declared_without_an_implementation` refuses.
        //
        // Safe in both conformance directions: the forward rule fires only when StagedResidency is
        // `Implemented`, and the converse fires only on `StructurallyNotApplicable`. Rung 1 here is
        // `Missing`, which is neither. Every other hook stays false.
        lifecycle: MemoryLifecycleCapabilities {
            phases: vec![MemoryPhase::Denoise],
            ..Default::default()
        },
        // sc-18665. **This is no longer `AssetBytesPlusHeadroom`, and the change is forced.**
        // `ComponentPhaseEnvelope` is the only formula variant that carries `resident_components`,
        // so it is the only shape in which the AdaLN exclusion can be declared at all — see that
        // variant's own doc, which records the decision that the component axis exists as a formula
        // shape "instead of a field on `MemoryAssetFacts`".
        //
        // It does NOT smuggle in a fitted curve. `variables` is `AssetBytes` alone — the same single
        // input the floor arm had — `phases` carries only the phase the exclusion is ordered
        // against, and `calibration` is still `None`, which is what actually fails every optimized
        // selection closed at admission. A one-phase, one-variable envelope IS the floor arm, minus
        // bytes the runtime provably does not hold through the denoise steady state.
        formula: MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: vec![MemoryPhase::Denoise],
            variables: {
                let mut variables = vec![MemoryFormulaVariable::AssetBytes];
                // Declared **only when an adapter is configured** (sc-18650), so an ordinary load's
                // formula is byte-for-byte the one this provider has always published and no
                // existing calibration record is disturbed. `conformance_errors` requires this
                // variable exactly when a typed auxiliary component is present, and the two are
                // built from the same condition here so they cannot drift.
                if components.overlay > 0 {
                    variables.push(MemoryFormulaVariable::OverlayBytes);
                }
                variables
            },
            resident_components: {
                let mut resident = vec![adaln_component(components.adaln)];
                // The LoRA stack as a typed auxiliary component, the spelling `mlx-gen-ltx` uses for
                // the same forward-time-residual install. `WholeRender` is literal here:
                // `crate::model::MiniMaxH3::load_task_dit` installs the factors onto the task's DiT
                // before the first step and nothing releases them until the DiT itself goes at the
                // end of denoise — see `crate::adapters`, where the residual is declared never to be
                // folded on any tier.
                if components.overlay > 0 {
                    resident.push(MemoryResidentComponent {
                        id: ADAPTER_STACK_COMPONENT_ID.to_owned(),
                        kind: MemoryComponentKind::AdapterStack,
                        resident_bytes: components.overlay,
                        bounded_by: None,
                        residency: MemoryComponentResidency::WholeRender,
                    });
                }
                resident
            },
        },
        // No fitted curve exists for this backend. `None` is the honest state, and it is load
        // bearing: it makes every optimized selection fail closed at admission.
        calibration: None,
        asset_facts: MemoryAssetFacts {
            base_bytes: components.base(),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.decoder(),
            // **The configured LoRA stack, and nothing else** (sc-18650).
            //
            // This was the literal `0` until this story, and unlike the MLX sibling's the same
            // mistake here was **live rather than latent**: this contract has been published since
            // sc-18659, so a LoRA render's resident factors were charged zero on a shipped path.
            // sc-18724 landed the adapter seam on 2026-08-13 and nothing brought the declaration
            // with it.
            //
            // There is no ControlNet, no IP-adapter and no identity encoder on this family —
            // `reject_unread_slots` refuses all three at the registry entry point — so the LoRA
            // stack is the only auxiliary residency there is.
            overlay_bytes: components.overlay,
        },
        runtime: Default::default(),
    }
}

/// The production contract: asset facts read off the resolved snapshot.
pub fn contract_for(spec: &LoadSpec) -> candle_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(&ComponentBytes::resolve(spec)?))
}

/// The weights-free fixture contract: the identical route declaration with zero asset facts and no
/// filesystem traversal.
pub fn weights_free_contract(
    _spec: &LoadSpec,
) -> candle_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(&ComponentBytes {
        text_encoder: 0,
        dit: 0,
        // The declaration-only footprint resolves no DiT, so the sub-stack states the architecture's
        // own bf16 figure. `fixture_contract_conforms_weights_free` requires the fixture and
        // production formulas to be equally shaped, and gen-core skips sub-stack containment against
        // zero asset facts for exactly this case.
        adaln: ADALN_EVICTED_BYTES,
        video_vae: 0,
        audio_vae: 0,
        // The declaration-only footprint charges no adapter, and the fixture specs
        // (`candle_memory_contract_surface_specs`) carry none — so the fixture formula stays the
        // no-adapter shape `fixture_contract_conforms_weights_free` compares against. Same choice
        // the MLX sibling's `ComponentBytes::weights_free` makes.
        overlay: 0,
    }))
}

/// The provider's real admission check, callable before any weight file is opened.
///
/// The shared check is sufficient here and a route gate would be a lie: with no optimized rung and
/// no calibration identity, every admission this provider can accept is the resident baseline, and
/// the geometry gates belong to the pipeline sc-17156 landed (`crate::denoise::geometry`), not to
/// this contract.
pub fn safety_check(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    candle_gen::gen_core::default_memory_strategy_safety_check(contract, context)
}

/// The memory-strategy registration for `minimax_h3` on candle.
///
/// **Reachable from `candle-gen-catalog` as of sc-17156.** It was not before, and that was
/// deliberate rather than an oversight: `ProviderRegistryBuilder::build` rejects a memory-strategy
/// registration whose `provider_id` has no matching generator, and this crate shipped no generator.
/// Wiring it through `register_composed_memory_strategy` would have satisfied the builder by
/// declaring a composition root that did not exist — the exact "provider contract with no
/// executable owner" that seam exists to prevent. The generator landed with sc-17156, so the
/// catalog line landed with it.
///
/// That coupling was a guard rather than a hope, and it still holds in the other direction: this
/// constant is registered by [`crate::register_providers`], and
/// `tests::a_generator_landing_here_forces_the_catalog_line` builds *that* inventory, so a
/// generator registered around it — leaving this registration behind — fails there.
pub const MEMORY_REGISTRATION: candle_gen::gen_core::MemoryRegistration =
    candle_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID,
        contract: contract_for,
        safety_check,
    };

/// The weights-free contract fixture paired with [`MEMORY_REGISTRATION`].
pub const MEMORY_CONTRACT_FIXTURE: candle_gen::gen_core::MemoryContractFixtureRegistration =
    candle_gen::gen_core::MemoryContractFixtureRegistration {
        provider_id: MODEL_ID,
        contract: weights_free_contract,
        surface_specs: memory_contract_surface_specs,
    };

/// The shared Bf16/Q4/Q8 witness set: exactly the tiers this crate ships (bf16 base plus the
/// sc-20267 packed Q4/Q8 tiers). A local name (the wan/ltx idiom) so the fixture registration
/// spells identically across the two backends for the cross-backend geometry gate.
fn memory_contract_surface_specs() -> Vec<candle_gen::gen_core::MemoryContractSurfaceSpec> {
    candle_gen::gen_core::candle_memory_contract_surface_specs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        LoadShape, MemoryBudget, MemoryCacheState, MemoryCalibrationIdentity, MemoryGeometry,
        MemoryMode, MemoryNumericTier, MemoryPhase, MemoryRunContext, MemorySelection,
        MemoryStrategyParameters, ProviderRegistryBuilder,
    };
    use std::path::{Path, PathBuf};

    /// One named, independently applied mutation of a known-good contract.
    type ContractMutation = (&'static str, Box<dyn Fn(&mut MemoryProviderContract)>);

    fn weightless_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    fn declared() -> MemoryProviderContract {
        weights_free_contract(&weightless_spec()).expect("weights-free contract")
    }

    fn candle_backend() -> MemoryBackendRealization {
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        }
    }

    /// Sparse `.safetensors` shards of exact sizes. `safetensors_path_bytes` stats rather than
    /// parses, so this costs no disk and still exercises the real directory-name wiring.
    fn sparse_snapshot(root: &Path, sizes: &[(&str, u64)]) {
        for (component, bytes) in sizes {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).expect("component dir");
            let file = std::fs::File::create(dir.join("model.safetensors")).expect("shard");
            file.set_len(*bytes).expect("sparse shard");
        }
    }

    fn full_snapshot(root: &Path) {
        sparse_snapshot(
            root,
            &[
                ("text_encoder", TEXT_ENCODER_BYTES),
                ("transformer", DIT_BF16_BYTES),
                ("vae", VIDEO_VAE_BYTES),
                ("audio_vae", AUDIO_VAE_BYTES),
                // Byte-identical to `transformer`; a render loads exactly one, so it must not be
                // charged on top.
                ("transformer_ref", DIT_BF16_BYTES),
            ],
        );
    }

    // --- AC1: declared Resident, not fallen-back Resident ---------------------------------------

    /// **The honest state of this backend is resident-only**, so the strategy table alone cannot
    /// distinguish this declaration from `compatibility_default` — asserting on it would be the
    /// false green AC1 warns about. The distinguisher is the one that actually matters to a
    /// consumer: a fallback contract publishes a **zero** byte floor for a 144 GB family, and this
    /// one publishes the measured components split across the fields the formula reads.
    #[test]
    fn resolved_contract_is_declared_and_not_the_compatibility_default() {
        let root = tempfile::tempdir().expect("tempdir");
        full_snapshot(root.path());
        let contract =
            (MEMORY_REGISTRATION.contract)(&LoadSpec::new(WeightsSource::Dir(root.path().into())))
                .expect("registered contract");

        let fallback = MemoryProviderContract::compatibility_default(MODEL_ID, candle_backend());
        assert_ne!(contract, fallback);
        assert_eq!(
            fallback.asset_facts.base_bytes, 0,
            "the fallback publishes a zero floor — that is what makes it unsafe here"
        );
        assert_eq!(
            contract.asset_facts.base_bytes,
            TEXT_ENCODER_BYTES + DIT_BF16_BYTES + VIDEO_VAE_BYTES + AUDIO_VAE_BYTES
        );
        assert_eq!(contract.asset_facts.conditioning_bytes, TEXT_ENCODER_BYTES);
        assert_eq!(contract.asset_facts.transformer_bytes, DIT_BF16_BYTES);
        assert_eq!(
            contract.asset_facts.decoder_bytes,
            VIDEO_VAE_BYTES + AUDIO_VAE_BYTES,
            "both decoders are charged, and only the decoders"
        );
        assert!(contract.conformance_errors().is_empty());
    }

    /// A misspelled provider id would produce a contract for a family that does not exist. The two
    /// registration constants and the contract must all agree on one id.
    #[test]
    fn every_declaration_agrees_on_one_provider_id() {
        assert_eq!(MODEL_ID, "minimax_h3");
        assert_eq!(declared().provider_id, MODEL_ID);
        assert_eq!(MEMORY_REGISTRATION.provider_id, MODEL_ID);
        assert_eq!(MEMORY_CONTRACT_FIXTURE.provider_id, MODEL_ID);
        assert_eq!(
            (MEMORY_CONTRACT_FIXTURE.contract)(&weightless_spec())
                .expect("fixture")
                .provider_id,
            MEMORY_REGISTRATION.provider_id
        );
    }

    /// This crate's package name, as `cargo metadata` reports it.
    const THIS_CRATE: &str = "candle-gen-minimax-h3";

    /// The catalog crate whose dependency edge on this one is the tripwire.
    const CATALOG_CRATE: &str = "candle-gen-catalog";

    /// Does `candle-gen-catalog` declare a dependency on this crate?
    ///
    /// Asked of `cargo metadata --no-deps --offline`, which is Cargo's own parse of the workspace
    /// manifests. Every workspace member reports its declared dependencies with each dependency's
    /// **package** name already resolved, so renames, quoting style, dotted keys,
    /// `workspace = true` inheritance and the `dev-`/`build-`/`target.'cfg(…)'` sections all arrive
    /// in the same `name` field and fall to one comparison. Lexing the TOML by hand instead is what
    /// let four spellings of the documented rename route go green (sc-18659).
    ///
    /// `--no-deps` scopes the answer to what the catalog *itself* declares: a route that reaches
    /// this crate transitively, through an intermediate crate that depends on it, is not covered.
    fn catalog_depends_on_this_crate() -> bool {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(4)
            .expect("candle-gen-minimax-h3 sits four levels under the workspace root");
        let output = std::process::Command::new(env!("CARGO"))
            .args([
                "metadata",
                "--no-deps",
                "--offline",
                "--format-version",
                "1",
                "--manifest-path",
            ])
            .arg(root.join("Cargo.toml"))
            .output()
            .unwrap_or_else(|error| panic!("cargo metadata: {error}"));
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("cargo metadata emits JSON");
        let catalog = metadata["packages"]
            .as_array()
            .expect("cargo metadata reports a `packages` array")
            .iter()
            .find(|package| package["name"] == CATALOG_CRATE)
            .unwrap_or_else(|| {
                panic!("{CATALOG_CRATE} is not a workspace member — this check would answer `false` for the wrong reason")
            });
        catalog["dependencies"]
            .as_array()
            .expect("a package reports a `dependencies` array")
            .iter()
            .any(|dependency| dependency["name"] == THIS_CRATE)
    }

    /// Every `.rs` under `candle-gen-catalog/src`, at any depth, concatenated — used only to check
    /// that the registration *call* landed, never to decide reachability. Reachability is
    /// [`catalog_depends_on_this_crate`]; scanning sources for that would reintroduce the bypass.
    ///
    /// The walk recurses because the call may land in `src/<subdir>/*.rs`, which a top-level-only
    /// `read_dir` would miss, making `wired` false on the `Ok` arm even though the call landed.
    fn catalog_sources() -> String {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("candle-gen-minimax-h3 sits inside crates/media/candle-gen")
            .join("candle-gen-catalog/src");
        let mut sources = Vec::new();
        push_rs_sources(&dir, &mut sources);
        assert!(
            !sources.is_empty(),
            "{} holds no .rs file — the source scan would vacuously pass",
            dir.display()
        );
        sources.join("\n")
    }

    /// Appends the text of every `.rs` file at or below `dir`.
    ///
    /// Symlinks are skipped. `Path::is_dir` resolves them, so a link would be followed into whatever
    /// tree it points at, and a broken `*.rs` link would panic in `read_to_string`.
    fn push_rs_sources(dir: &Path, sources: &mut Vec<String>) {
        let entries = std::fs::read_dir(dir)
            .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap_or_else(|error| panic!("{}: {error}", dir.display()));
        for entry in &entries {
            let path = entry.path();
            let file_type = entry
                .file_type()
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                push_rs_sources(&path, sources);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                sources.push(
                    std::fs::read_to_string(&path)
                        .unwrap_or_else(|error| panic!("{}: {error}", path.display())),
                );
            }
        }
    }

    /// **The tripwire on the missing catalog line**, and it reads the crate's *real* registration
    /// inventory — [`crate::register_providers`], the one function a catalog line would call —
    /// rather than a builder assembled inside the test, which could never contain a generator and
    /// so could never detect one arriving.
    ///
    /// sc-17156 added `.register_generator(…)` to `register_providers`, so `build()` now succeeds
    /// and this test takes its `Ok` arm: it demands the catalog line, and that line exists. Before
    /// then, `build()` rejected the memory strategy for want of a generator and the `Err` arm
    /// asserted the catalog must **not** carry the line (it would break
    /// `candle_gen_catalog::provider_registry()`), so wiring the line early failed here too.
    ///
    /// Both arms stay live, which is the safety property the declaration's doc comment claims,
    /// actually guarded: removing the catalog line while the generator is still registered fails
    /// here, and the failure message names the line to add.
    ///
    /// The `Err` arm's needle is the catalog's declared *dependencies*, not its sources — see
    /// [`catalog_depends_on_this_crate`] for what that covers and the one route it does not.
    /// Sources are the wrong needle here: a generator registered straight from the catalog
    /// would never pass through this crate's inventory and would leave the memory contract
    /// unregistered, and it could be written in any catalog module, so a scan of `lib.rs` alone
    /// goes green in exactly the state this test exists to catch.
    ///
    /// A known, accepted wart: a bare `pub use candle_gen_minimax_h3 as …;` re-export — which needs
    /// the dependency but registers nothing — false-reds the `Err` arm. That is the correct
    /// trade-off for a needle with no false *greens*, and the message says what to check.
    #[test]
    fn a_generator_landing_here_forces_the_catalog_line() {
        let reached = catalog_depends_on_this_crate();
        let wired =
            reached && catalog_sources().contains("candle_gen_minimax_h3::register_providers");
        match crate::register_providers(ProviderRegistryBuilder::new()).build() {
            Err(error) => {
                assert!(
                    error.to_string().contains("no matching generator"),
                    "unexpected rejection: {error}"
                );
                assert!(
                    !reached,
                    "candle-gen-catalog's Cargo.toml now depends on candle-gen-minimax-h3, but \
                     this crate's own `register_providers` still registers no generator. Either \
                     the catalog line landed early — `provider_registry()` cannot build a memory \
                     strategy with no matching generator — or a generator was registered straight \
                     from the catalog (in any module) and bypassed this crate's inventory, leaving \
                     MEMORY_REGISTRATION unregistered. Register generators in \
                     `candle_gen_minimax_h3::register_providers` (sc-17156). A dependency added \
                     for a bare re-export with no registration trips this too — drop it until the \
                     generator exists"
                );
            }
            Ok(registry) => {
                let generators: Vec<&str> = registry
                    .generators()
                    .map(|registration| (registration.descriptor)().id)
                    .collect();
                assert!(
                    wired,
                    "this crate now registers generator(s) {generators:?}, so its memory contract \
                     is finally wirable — add `candle-gen-minimax-h3` to candle-gen-catalog's \
                     Cargo.toml (reached: {reached}) and `let registry = \
                     candle_gen_minimax_h3::register_providers(registry);` to its \
                     `register_providers` (and the matching `ProviderCrate` row), or the contract \
                     stays unreachable at runtime (sc-17156)"
                );
            }
        }
    }

    // --- AC2: nothing optimized is declared, and nothing optimized is reachable ------------------

    /// **Rungs 1-4 are all declared `Missing`, and each is independently refused at selection.**
    ///
    /// Rung 1 is the interesting member: sc-17156 made the staging real in `generate_impl` and
    /// forced the offload policy, but the declaration deliberately did not follow (see
    /// `strategies`), so it is refused here exactly like 2-4. **Two** independent things would each
    /// have to change before it became admittable — the declaration itself, and the `calibration`
    /// identity that is `None` — and the assertions below check the declaration and the admission
    /// decision separately so a change to either is visible rather than absorbed by the other.
    #[test]
    fn no_optimized_rung_is_declared_or_selectable() {
        let contract = declared();
        assert!(
            contract.calibration.is_none(),
            "candle has no fitted curve for this family"
        );
        for strategy in MemoryStrategy::ALL {
            // Only rung 0 is declared `Implemented`; 1-4 are `Missing`. `is_optimized()` — which
            // decides which arms below run — is `!matches!(self, Resident)`, so `StagedResidency`
            // IS optimized and the refusal arms DO exercise rung 1. That is the point of the test,
            // not an exception to it: the rung whose mechanism exists is refused all the same.
            let expected = if strategy == MemoryStrategy::Resident {
                MemoryStrategySupport::Implemented
            } else {
                MemoryStrategySupport::Missing
            };
            assert_eq!(
                contract.capability(strategy).expect("entry").support,
                expected,
                "{strategy:?}"
            );
            if strategy.is_optimized() {
                assert!(
                    contract
                        .validate_selection(&MemorySelection {
                            strategy,
                            tier: MemoryNumericTier {
                                precision: candle_gen::gen_core::Precision::Bf16,
                                quant: None,
                                component_precision_floors: &[],
                            },
                            parameters: MemoryStrategyParameters::default(),
                        })
                        .is_err(),
                    "{strategy:?} must not be selectable"
                );
                assert!(
                    matches!(
                        safety_check(&weightless_spec(), &contract, &context(strategy)),
                        MemorySafetyDecision::Reject { .. }
                    ),
                    "{strategy:?} must be refused at admission"
                );
            }
        }
        // The control arm: the one rung this backend does implement is admitted, so the rejections
        // above are not an always-rejecting check.
        assert_eq!(
            safety_check(
                &weightless_spec(),
                &declared(),
                &context(MemoryStrategy::Resident)
            ),
            MemorySafetyDecision::Accept
        );
    }

    fn context(strategy: MemoryStrategy) -> MemoryRunContext {
        MemoryRunContext {
            // Inert for the Resident probe this fixture serves (the authority gate fires only for
            // optimized strategies); Calibrated matches every other provider's weights-free
            // safety-check context.
            optimization_authority: candle_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: MemorySelection {
                strategy,
                tier: MemoryNumericTier {
                    precision: candle_gen::gen_core::Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
                parameters: MemoryStrategyParameters::default(),
            },
            calibration_abi: candle_gen::gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: String::new(),
            load_shape: LOAD_SHAPE,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1344,
                height: 768,
                batch: 1,
                frames: 124,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 256 * 1024 * 1024 * 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1024 * 1024 * 1024,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "unit-test".to_owned(),
        }
    }

    // --- AC3: weights-free conformance, and a malformed contract fails ---------------------------

    /// The same check the registry path runs, on the same fixture factory, in the default lane.
    #[test]
    fn fixture_contract_conforms_weights_free() {
        let fixture = (MEMORY_CONTRACT_FIXTURE.contract)(&weightless_spec()).expect("fixture");
        assert_eq!(
            fixture.asset_facts,
            MemoryAssetFacts::default(),
            "the fixture must inject zero asset facts without touching the filesystem"
        );
        gen_core_testkit::memory_strategy_conformance(&fixture);

        // ...and it must not diverge from the production declaration in anything but bytes.
        let root = tempfile::tempdir().expect("tempdir");
        full_snapshot(root.path());
        let production =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert_eq!(fixture.strategies, production.strategies);
        assert_eq!(fixture.lifecycle, production.lifecycle);
        assert_eq!(fixture.formula, production.formula);
        assert_eq!(fixture.calibration, production.calibration);
        assert_eq!(fixture.load_shape, production.load_shape);
        assert_eq!(fixture.backend, production.backend);
    }

    /// Each mutation is applied **alone** to a known-good contract, so each guard is proven to
    /// detect its own breakage rather than the set proving itself.
    #[test]
    fn each_contract_mutation_is_independently_detected() {
        assert!(
            gen_core_testkit::check_memory_strategy_contract(&declared()).is_ok(),
            "the shipped contract must conform, or every mutation below is vacuous"
        );

        let mutations: Vec<ContractMutation> = vec![
            (
                "a dropped strategy entry",
                Box::new(|c: &mut MemoryProviderContract| {
                    c.strategies
                        .retain(|entry| entry.strategy != MemoryStrategy::BoundedDecode);
                }),
            ),
            (
                "a duplicated strategy entry",
                Box::new(|c: &mut MemoryProviderContract| {
                    let first = c.strategies[0].clone();
                    c.strategies.push(first);
                }),
            ),
            (
                "a Resident baseline that is not implemented",
                Box::new(|c: &mut MemoryProviderContract| {
                    for entry in &mut c.strategies {
                        if entry.strategy == MemoryStrategy::Resident {
                            entry.support = MemoryStrategySupport::Missing;
                        }
                    }
                }),
            ),
            (
                "an empty StructurallyNotApplicable reason",
                Box::new(|c: &mut MemoryProviderContract| {
                    for entry in &mut c.strategies {
                        if entry.strategy == MemoryStrategy::BoundedAttention {
                            entry.support = MemoryStrategySupport::StructurallyNotApplicable {
                                reason: "   ".to_owned(),
                            };
                        }
                    }
                }),
            ),
            (
                "StagedResidency implemented with no lifecycle phases",
                Box::new(|c: &mut MemoryProviderContract| {
                    for entry in &mut c.strategies {
                        if entry.strategy == MemoryStrategy::StagedResidency {
                            entry.support = MemoryStrategySupport::Implemented;
                        }
                    }
                }),
            ),
            (
                "BoundedDecode implemented with no tile domain",
                Box::new(|c: &mut MemoryProviderContract| {
                    implement_without_range(c, MemoryStrategy::BoundedDecode)
                }),
            ),
            (
                "BoundedAttention implemented with no chunk domain",
                Box::new(|c: &mut MemoryProviderContract| {
                    implement_without_range(c, MemoryStrategy::BoundedAttention)
                }),
            ),
            (
                "BoundedTransformerResidency implemented with no window domain",
                Box::new(|c: &mut MemoryProviderContract| {
                    implement_without_range(c, MemoryStrategy::BoundedTransformerResidency)
                }),
            ),
            (
                "base_bytes that does not equal its components",
                Box::new(|c: &mut MemoryProviderContract| c.asset_facts.base_bytes += 1),
            ),
            (
                "a malformed calibration fingerprint",
                Box::new(|c: &mut MemoryProviderContract| {
                    c.calibration = Some(MemoryCalibrationIdentity::new("No_Version", LOAD_SHAPE));
                }),
            ),
            (
                "a calibration load shape that disagrees with the contract",
                Box::new(|c: &mut MemoryProviderContract| {
                    c.calibration = Some(MemoryCalibrationIdentity::new(
                        "minimax-h3-candle-v1",
                        LoadShape::DeferredMaterialization,
                    ));
                }),
            ),
        ];

        for (name, mutate) in mutations {
            let mut contract = declared();
            mutate(&mut contract);
            assert!(
                gen_core_testkit::check_memory_strategy_contract(&contract).is_err(),
                "conformance must reject {name}"
            );
        }
    }

    /// AC5's failure shape: a lever declared `Implemented` with no [`MemoryParameterRanges`].
    fn implement_without_range(contract: &mut MemoryProviderContract, strategy: MemoryStrategy) {
        for entry in &mut contract.strategies {
            if entry.strategy == strategy {
                entry.support = MemoryStrategySupport::Implemented;
                entry.parameters = MemoryParameterRanges::default();
            }
        }
        match strategy {
            MemoryStrategy::BoundedDecode => contract.lifecycle.decode_tiling = true,
            MemoryStrategy::BoundedAttention => contract.lifecycle.attention_chunking = true,
            MemoryStrategy::BoundedTransformerResidency => {
                contract.lifecycle.transformer_window_materialization = true
            }
            _ => {}
        }
    }

    // --- AC5: parameter ranges are declared exactly where they are owned ------------------------

    #[test]
    fn parameter_ranges_are_owned_by_the_rung_that_consumes_them() {
        let contract = declared();
        assert!(contract.conformance_errors().is_empty());
        for capability in &contract.strategies {
            assert_eq!(
                capability.parameters,
                MemoryParameterRanges::default(),
                "{:?} owns no numeric parameters on this backend",
                capability.strategy
            );
        }
        for strategy in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let mut mutated = declared();
            implement_without_range(&mut mutated, strategy);
            assert!(
                !mutated.conformance_errors().is_empty(),
                "{strategy:?} implemented with no MemoryParameterRanges must fail"
            );
        }
    }

    // --- AC4: measured asset facts --------------------------------------------------------------

    /// Tolerance for the GB figures recorded on sc-18659, in bytes. The story quotes 66.73 GB for
    /// the text encoder; the measured on-disk total is 66,714,912,872 B = 66.715 GB, so the byte
    /// constants are authoritative and the GB figures are held only to this window.
    const GB_TOLERANCE_BYTES: u64 = 20_000_000;

    fn assert_within(measured: u64, story_gb: f64, what: &str) {
        let story_bytes = (story_gb * 1e9) as u64;
        let delta = measured.abs_diff(story_bytes);
        assert!(
            delta <= GB_TOLERANCE_BYTES,
            "{what}: measured {measured} B ({:.3} GB) is {delta} B from the recorded {story_gb} GB, \
             outside the {GB_TOLERANCE_BYTES} B tolerance",
            measured as f64 / 1e9
        );
    }

    #[test]
    fn measured_component_bytes_match_the_recorded_footprints() {
        assert_within(DIT_BF16_BYTES, 66.28, "33 B DiT partition at bf16");
        assert_within(TEXT_ENCODER_BYTES, 66.73, "Qwen3-VL-32B text encoder");
        assert_within(VIDEO_VAE_BYTES, 10.42, "video VAE");
        assert_within(AUDIO_VAE_BYTES, 0.61, "audio VAE");
        assert_eq!(
            ADALN_EVICTED_BYTES, 26_020_915_200,
            "the exact bytes crate::dit::adaln releases"
        );
    }

    /// The two backends must not drift apart on facts about the same checkpoint.
    #[test]
    fn the_measured_facts_are_the_same_numbers_the_mlx_sibling_declares() {
        // Mirrored deliberately rather than shared through a dependency: `candle-gen-*` crates do
        // not depend on `mlx-gen-*`. These literals are the same ones
        // `mlx_gen_minimax_h3::memory_strategy` declares, and both are asserted against the story's
        // recorded GB figures above, so a drift on either side fails there first.
        assert_eq!(TEXT_ENCODER_BYTES, 66_714_912_872);
        assert_eq!(DIT_BF16_BYTES, 66_280_504_216);
        assert_eq!(VIDEO_VAE_BYTES, 10_415_558_888);
        assert_eq!(AUDIO_VAE_BYTES, 605_429_340);
    }

    // --- sc-20267: the DECLARED per-tier asset sizes ---------------------------------------------

    /// The four per-tier figures are the manifest's `estimatedSizeBytes` rows, byte for byte, and
    /// they are held to their GB figures through the same window the measured facts are.
    ///
    /// **Nothing here measures anything.** These are `SceneWorks/minimax-h3-mlx` @ `137ce668` manifest
    /// rows; a test in this crate cannot verify an 18-35 GB hosted subtree, so what is pinned is that
    /// the transcription has not drifted and that the byte and GB spellings agree.
    #[test]
    fn the_declared_tier_sizes_are_the_manifest_rows() {
        assert_eq!(DIT_Q4_BYTES, 18_780_109_783);
        assert_eq!(DIT_Q8_BYTES, 35_302_064_357);
        assert_eq!(TEXT_ENCODER_Q4_BYTES, 18_722_713_964);
        assert_eq!(TEXT_ENCODER_Q8_BYTES, 33_723_765_614);
        assert_within(DIT_Q4_BYTES, 18.78, "q4 DiT partition (declared)");
        assert_within(DIT_Q8_BYTES, 35.30, "q8 DiT partition (declared)");
        assert_within(TEXT_ENCODER_Q4_BYTES, 18.72, "q4 text encoder (declared)");
        assert_within(TEXT_ENCODER_Q8_BYTES, 33.72, "q8 text encoder (declared)");
    }

    /// **The over-declaration is quantified rather than merely admitted.**
    ///
    /// The declared tier figures are whole-subtree bytes while every measured constant in this module
    /// is `.safetensors`-only, so the declared ones are upper bounds. bf16 is the one tier where both
    /// accountings exist, and the gap there is the sidecar allowance the packed rows also carry: if a
    /// re-capture ever moved this materially, the "sub-ppm upper bound" claim in the section note would
    /// be wrong and the four constants would need re-deriving rather than nudging.
    #[test]
    fn the_declared_tier_sizes_over_declare_only_by_the_sidecars() {
        let gap = MANIFEST_DIT_BF16_SUBTREE_BYTES - DIT_BF16_BYTES;
        assert_eq!(gap, 65_034, "the bf16 DiT subtree's sidecar allowance");
        const {
            assert!(
                MANIFEST_DIT_BF16_SUBTREE_BYTES > DIT_BF16_BYTES,
                "a whole-subtree figure cannot be below the safetensors sum inside it"
            );
        }
        // Sub-ppm, which is what makes carrying the rows unadjusted the right trade — see the section
        // note on why the over-declaration direction is the safe one.
        assert!(
            gap * 1_000_000 < DIT_BF16_BYTES,
            "{gap} B on {DIT_BF16_BYTES} B is no longer sub-ppm; re-read the section note"
        );
    }

    /// Coherence: `q4 < q8 < bf16` on both tiered components. A transposed pair or a copy-paste
    /// between rows would publish a floor for the wrong tier, and no other assertion here would see it.
    #[test]
    fn the_declared_tier_sizes_are_ordered_by_tier() {
        const {
            assert!(DIT_Q4_BYTES < DIT_Q8_BYTES);
            assert!(DIT_Q8_BYTES < DIT_BF16_BYTES);
            assert!(TEXT_ENCODER_Q4_BYTES < TEXT_ENCODER_Q8_BYTES);
            assert!(
                TEXT_ENCODER_Q8_BYTES < TEXT_ENCODER_BYTES,
                "the packed q8 encoder must sit below the dense one it replaces"
            );
        }
        // ...and the two components are independently tiered (sc-19120), so their rows must not have
        // been filled in from each other.
        assert_ne!(DIT_Q4_BYTES, TEXT_ENCODER_Q4_BYTES);
        assert_ne!(DIT_Q8_BYTES, TEXT_ENCODER_Q8_BYTES);
    }

    /// **The MLX lane's `q4` AdaLN figure, mirrored here for one purpose: to assert this lane does NOT
    /// equal it.** See [`adaln_stack_bytes`] for why.
    const MLX_ADALN_Q4_BYTES: u64 = 7_325_337_600;

    /// The packed AdaLN stack sizes, derived from the GGUF container candle actually keeps resident.
    #[test]
    fn the_packed_adaln_stack_sizes_come_from_the_gguf_container() {
        // The dense arm is derived, not returned as the constant, so a config change moves both.
        assert_eq!(
            adaln_stack_bytes(16),
            ADALN_EVICTED_BYTES,
            "bits >= 16 is the dense bf16 stack"
        );
        assert_eq!(adaln_stack_bytes(32), ADALN_EVICTED_BYTES);
        // A width the repack does not serve takes the same conservative arm rather than deriving a
        // container that does not exist.
        assert_eq!(adaln_stack_bytes(2), ADALN_EVICTED_BYTES);
        assert_eq!(adaln_stack_bytes(0), ADALN_EVICTED_BYTES);

        // Q8_0: 32 codes + one f16 scale per 32 weights = 34 B/block, plus the dense bias row.
        assert_eq!(adaln_stack_bytes(8), 13_828_147_200);
        // Q4_1: 16 code bytes + an f16 scale AND an f16 minimum per 32 weights = 20 B/block.
        assert_eq!(adaln_stack_bytes(4), 8_138_188_800);

        assert!(adaln_stack_bytes(4) < adaln_stack_bytes(8));
        assert!(adaln_stack_bytes(8) < adaln_stack_bytes(16));

        // The q8 figure COINCIDES with the MLX sibling's `ADALN_EVICTED_Q8_BYTES`, and that is
        // arithmetic luck rather than a shared derivation: Q8_0's 2 B of metadata per 32 elements
        // happens to equal MLX's two bf16 metadata tensors at 4 B per 64-element group.
        assert_eq!(
            adaln_stack_bytes(8),
            13_828_147_200,
            "the same number mlx_gen_minimax_h3::memory_strategy::ADALN_EVICTED_Q8_BYTES declares"
        );
        // The q4 figure DIVERGES, and the divergence is the correct answer rather than a defect to
        // reconcile: Q4_1 carries its f16 scale and f16 minimum per 32 elements where the MLX pack
        // carries them per 64, so candle holds ~11.1% more metadata over the same codes. Copying MLX's
        // figure across would under-declare what this backend actually keeps resident.
        assert!(
            adaln_stack_bytes(4) > MLX_ADALN_Q4_BYTES,
            "candle's Q4_1 stack ({}) must exceed the MLX triple's ({MLX_ADALN_Q4_BYTES}) — twice the \
             per-group metadata density over the same codes",
            adaln_stack_bytes(4)
        );

        // The containers are read off GgmlDType rather than hand-typed, and the mapping is the
        // repack's, NOT `candle_gen::quant::ggml_dtype`'s (whose Q4 arm is the dense fold's Q4_0).
        assert_eq!(packed_container(4), Some(GgmlDType::Q4_1));
        assert_eq!(packed_container(8), Some(GgmlDType::Q8_0));
        assert_eq!(packed_container(16), None);
        assert_ne!(
            packed_container(4),
            Some(GgmlDType::Q4_0),
            "Q4_0 is the in-place dense fold's container and carries no per-block minimum"
        );
    }

    /// **The stack arithmetic is the LOADER's own accounting, not a parallel restatement of it.**
    ///
    /// The shipped 96768x2688 projection is 8-14 GB of codes and cannot be materialized in-process, so
    /// the property is shown on the shared [`packed_projection_bytes`] at a geometry that can be: a real
    /// `crate::quant::lin` load of a synthetic MLX Q4 triple, whose `TieredLinear::base_bytes` is the
    /// figure [`crate::dit::block::AdaLnProjection::nbytes`] reports at render time. Because
    /// [`adaln_stack_bytes`] is exactly that function times the block count, agreement here is
    /// agreement at the shipped geometry too.
    #[test]
    fn the_packed_stack_arithmetic_is_the_loaders_own_accounting() {
        use candle_gen::candle_core::{Device, Tensor};
        use candle_gen::Weights;

        let (out, inp) = (64usize, 128usize);
        let mut map = std::collections::HashMap::new();
        crate::quant::testkit::insert_packed(&mut map, "adaln_proj", out, inp, 1);
        map.insert(
            "adaln_proj.bias".to_owned(),
            Tensor::zeros(out, DType::BF16, &Device::Cpu).expect("bias row"),
        );
        let loaded = crate::quant::lin(
            &Weights::from_map(map),
            crate::quant::DIT,
            "adaln_proj",
            true,
            DType::BF16,
        )
        .expect("a packed projection load");
        assert!(loaded.linear.is_packed(), "the fixture must load packed");
        assert_eq!(
            loaded.linear.quant_dtype(),
            Some(GgmlDType::Q4_1),
            "a 4-bit MLX pack repacks into Q4_1 — the container adaln_stack_bytes derives from"
        );
        assert_eq!(
            loaded.base_bytes as u64,
            packed_projection_bytes(GgmlDType::Q4_1, out as u64, inp as u64),
            "the declaration's per-projection arithmetic must equal what the loader records"
        );
        // ...and the shipped stack really is that function times the block count, so the agreement
        // above transfers.
        let config = crate::dit::MiniMaxH3DitConfig::default();
        assert_eq!(
            adaln_stack_bytes(4),
            config.num_layers as u64
                * packed_projection_bytes(
                    GgmlDType::Q4_1,
                    config.adaln_out_features() as u64,
                    config.time_embed_dim as u64,
                )
        );
        assert_eq!(compute_dtype_bytes(), 2, "the DiT computes at bf16");
    }

    /// Write the `quantization` marker `mlx_gen_minimax_h3::convert` puts in a packed component's
    /// `config.json` — the exact shape `crate::tier`'s own fixtures write, and the only thing the
    /// marker leg reads.
    fn write_tier_marker(component_dir: &Path, bits: i32) {
        std::fs::write(
            component_dir.join("config.json"),
            format!(r#"{{"quantization": {{"bits": {bits}, "group_size": 64}}}}"#),
        )
        .expect("tier marker");
    }

    /// A snapshot carrying one DiT partition at `bytes`, optionally marked at a packed width.
    fn staged_contract(
        bytes: u64,
        bits: Option<i32>,
    ) -> (tempfile::TempDir, MemoryProviderContract) {
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[(BASE_DIT_PARTITION, bytes)]);
        if let Some(bits) = bits {
            write_tier_marker(&root.path().join(BASE_DIT_PARTITION), bits);
        }
        let contract =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        (root, contract)
    }

    /// **The marker leg, driven end to end** — and it binds at `q8`, where the derived stack is
    /// genuinely tighter than the footprint ratio.
    ///
    /// `min(marker, footprint)` is asymmetric on purpose, so this pins *which* leg wins at each shipped
    /// tier rather than only that the pair is wired: at `q8` the marker wins by 31 MB, and at `q4` the
    /// footprint wins by 765 MB because candle's `Q4_1` container is denser in metadata than the MLX
    /// pack the ratio was calibrated on. Both are the tighter bound, which is the whole contract of
    /// this function.
    #[test]
    fn the_marker_leg_binds_where_it_is_the_tighter_bound() {
        // q8, marked: the marker leg wins.
        let (_root, marked_q8) = staged_contract(DIT_Q8_BYTES, Some(8));
        let (_bare_root, bare_q8) = staged_contract(DIT_Q8_BYTES, None);
        assert_eq!(
            marked_q8.asset_facts.transformer_bytes, bare_q8.asset_facts.transformer_bytes,
            "the two differ ONLY in the marker, never in the footprint"
        );
        assert_eq!(
            marked_q8.resident_components()[0].resident_bytes,
            adaln_stack_bytes(8)
        );
        assert!(
            marked_q8.resident_components()[0].resident_bytes
                < bare_q8.resident_components()[0].resident_bytes,
            "a marked q8 tier must declare a SMALLER stack than the footprint ratio alone: {} vs {}",
            marked_q8.resident_components()[0].resident_bytes,
            bare_q8.resident_components()[0].resident_bytes
        );

        // q4, marked: the FOOTPRINT leg wins, because candle's Q4_1 stack is the larger of the two.
        // This is `min` doing its job — the smaller declaration excludes less and so charges more.
        let (_q4_root, marked_q4) = staged_contract(DIT_Q4_BYTES, Some(4));
        let (_bare_q4_root, bare_q4) = staged_contract(DIT_Q4_BYTES, None);
        assert!(
            adaln_stack_bytes(4) > bare_q4.resident_components()[0].resident_bytes,
            "the q4 marker leg is expected to be the LARGER one on this lane"
        );
        assert_eq!(
            marked_q4.resident_components()[0].resident_bytes,
            bare_q4.resident_components()[0].resident_bytes,
            "at q4 the footprint leg is tighter, so the marker must be discarded"
        );

        // A marker that is decisive in the other direction: a q4 marker over a bf16-sized footprint is
        // the mislabelled/partially-staged install the footprint leg alone would charge 26.02 GB for.
        let (_mixed_root, mixed) = staged_contract(DIT_BF16_BYTES, Some(4));
        assert_eq!(
            mixed.resident_components()[0].resident_bytes,
            adaln_stack_bytes(4),
            "the marker is authoritative about the tier; the footprint is only a ratio"
        );
        assert!(mixed.resident_components()[0].resident_bytes < ADALN_EVICTED_BYTES);

        // An unreadable/absent marker falls back to bf16, NEVER to zero and never to an error — a
        // contract that cannot resolve is a render that cannot run.
        assert_eq!(
            resolved_adaln_bytes(Path::new("/nonexistent/transformer"), DIT_BF16_BYTES),
            ADALN_EVICTED_BYTES
        );
        assert_eq!(
            resolved_adaln_bytes(Path::new("/nonexistent/transformer"), 0),
            ADALN_EVICTED_BYTES,
            "the weights-free arm keeps the architecture fact"
        );
    }

    /// **Containment at every tier**, which is a hard contract failure rather than a tidiness one:
    /// `conformance_errors` refuses a sub-stack larger than the stack containing it, and
    /// `Registry::memory_strategy_contract` turns that into a render that cannot resolve a contract at
    /// all. The eviction must also stay clear of the floor where it would exclude nothing.
    #[test]
    fn the_resolved_adaln_stack_stays_inside_the_resolved_dit_at_every_tier() {
        for (label, bytes, bits) in [
            ("q4 marked", DIT_Q4_BYTES, Some(4)),
            ("q4 unmarked", DIT_Q4_BYTES, None),
            ("q8 marked", DIT_Q8_BYTES, Some(8)),
            ("q8 unmarked", DIT_Q8_BYTES, None),
            ("bf16", DIT_BF16_BYTES, None),
        ] {
            let (_root, contract) = staged_contract(bytes, bits);
            let resident = contract.resident_components()[0].resident_bytes;
            assert_eq!(contract.asset_facts.transformer_bytes, bytes, "{label}");
            assert!(
                resident < contract.asset_facts.transformer_bytes,
                "{label}: a sub-stack ({resident}) cannot exceed the stack containing it ({bytes})"
            );
            assert!(
                ADALN_MODULATION_TABLE_MAX_BYTES < resident,
                "{label}: the eviction must stay clear of the floor where it excludes nothing"
            );
            assert!(
                contract.conformance_errors().is_empty(),
                "{label}: {:?}",
                contract.conformance_errors()
            );
        }
    }

    /// **A staged text encoder is charged at its own size** (sc-19120 / sc-20267).
    ///
    /// The TE's tier is staged **independently** of the DiT's, so a split `q4` install stages
    /// `text_encoder` outside the snapshot and may carry no `root/text_encoder` at all. `resolve` used
    /// to size `root.join("text_encoder")` unconditionally, so that install was charged **zero** for the
    /// largest single component this family loads — the same 66 GB under-declaration the
    /// [`DIT_PARTITIONS`] note records, in a second place.
    #[test]
    fn a_staged_text_encoder_is_charged_at_its_own_size() {
        // A split install: no `text_encoder/` under the root at all.
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[(BASE_DIT_PARTITION, DIT_Q4_BYTES)]);
        let staged = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(
            staged.path(),
            &[(crate::tier::TEXT_ENCODER_COMPONENT, TEXT_ENCODER_Q4_BYTES)],
        );

        let unstaged =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert_eq!(
            unstaged.asset_facts.conditioning_bytes, 0,
            "the snapshot genuinely carries no text encoder, so the staged case below is not vacuous"
        );

        let contract = contract_for(
            &LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
                crate::tier::TEXT_ENCODER_COMPONENT,
                WeightsSource::Dir(staged.path().join(crate::tier::TEXT_ENCODER_COMPONENT)),
            ),
        )
        .expect("contract");
        assert_eq!(
            contract.asset_facts.conditioning_bytes, TEXT_ENCODER_Q4_BYTES,
            "a staged packed encoder must be charged, not measured as absent"
        );
        assert_eq!(
            contract.asset_facts.base_bytes,
            TEXT_ENCODER_Q4_BYTES + DIT_Q4_BYTES,
            "and it must reach the floor the formula reads"
        );
        assert!(contract.conformance_errors().is_empty());
    }

    #[test]
    fn a_staged_dit_component_is_charged_at_its_own_size() {
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[(BASE_DIT_PARTITION, DIT_BF16_BYTES)]);
        let staged = tempfile::tempdir().expect("tempdir");
        const Q4_BYTES: u64 = 18_779_970_678;
        sparse_snapshot(staged.path(), &[(BASE_DIT_PARTITION, Q4_BYTES)]);

        let contract = contract_for(
            &LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
                BASE_DIT_PARTITION,
                WeightsSource::Dir(staged.path().join(BASE_DIT_PARTITION)),
            ),
        )
        .expect("contract");
        assert_eq!(contract.asset_facts.transformer_bytes, Q4_BYTES);
    }

    /// **A `ref2va` snapshot is charged for the partition a `ref2va` render actually reads.**
    ///
    /// `ComponentBytes::resolve` sized `root.join("transformer")` unconditionally, so a snapshot
    /// carrying only `transformer_ref/` was charged ZERO for its 66 GB DiT — a contract that
    /// under-reports by 66 GB admits a render that then OOMs. The three cases below are the ones
    /// that distinguish "the base partition" from "the partition this snapshot can serve".
    #[test]
    fn the_dit_charge_covers_the_reference_partition_too() {
        const REF_BYTES: u64 = 66_280_504_216;
        const SMALL: u64 = 1_000_000;

        // Reference partition ALONE: charged for it, not zero.
        let only_ref = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(only_ref.path(), &[(REFERENCE_DIT_PARTITION, REF_BYTES)]);
        let c = contract_for(&LoadSpec::new(WeightsSource::Dir(only_ref.path().into())))
            .expect("contract");
        assert_eq!(
            c.asset_facts.transformer_bytes, REF_BYTES,
            "a ref2va-only snapshot was charged {} for a {REF_BYTES}-byte DiT",
            c.asset_facts.transformer_bytes
        );

        // Base partition alone is unchanged — the base-only (off-Mac) install shape.
        let only_base = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(only_base.path(), &[(BASE_DIT_PARTITION, SMALL)]);
        let c = contract_for(&LoadSpec::new(WeightsSource::Dir(only_base.path().into())))
            .expect("contract");
        assert_eq!(c.asset_facts.transformer_bytes, SMALL);

        // Both present: a render loads exactly ONE, so the larger is the bound, and it is charged
        // ONCE rather than summed. Asymmetric sizes, so "max" and "sum" cannot coincide.
        let both = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(
            both.path(),
            &[
                (BASE_DIT_PARTITION, SMALL),
                (REFERENCE_DIT_PARTITION, REF_BYTES),
            ],
        );
        let c =
            contract_for(&LoadSpec::new(WeightsSource::Dir(both.path().into()))).expect("contract");
        assert_eq!(
            c.asset_facts.transformer_bytes, REF_BYTES,
            "with both partitions present the charge is the larger, once"
        );
        assert_ne!(
            c.asset_facts.transformer_bytes,
            SMALL + REF_BYTES,
            "the two partitions must not be summed; only one is ever resident"
        );
    }

    // --- the declaration facts that are easy to get wrong ----------------------------------------

    /// The load shape is pinned to the loader, not mirrored from the request.
    #[test]
    fn load_shape_is_pinned_to_the_loader_not_taken_from_the_spec() {
        let spec = weightless_spec();
        assert_eq!(spec.load_shape, LoadShape::DeferredMaterialization);
        assert_eq!(
            contract_for(&spec).expect("contract").load_shape,
            LoadShape::EagerMaterialization
        );
        // ...and the spec IS read: the asset facts come off it.
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[("audio_vae", AUDIO_VAE_BYTES)]);
        let resolved =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert_eq!(resolved.asset_facts.decoder_bytes, AUDIO_VAE_BYTES);
    }

    /// **What actually blocks an optimized rung on this backend, executed rather than asserted**
    /// (sc-18660).
    ///
    /// The MLX sibling declares rung 2 `Implemented` because its spatial tiling is reachable from a
    /// request. **The mechanism exists here too** — `MiniMaxH3VideoVae::decode_clip` tiles at the
    /// same reference geometry, through the same `BoundedStitch` — so the natural conclusion is
    /// that this crate under-declares, and the natural fix is to add the weights-free
    /// `MemoryBehaviorRegistration` seam the MLX sibling has.
    ///
    /// **That fix is not available on this tree, and the reason is one link further back than the
    /// seam.** The chain, each link driven below rather than described:
    ///
    /// 1. this backend has no fitted curve of its own — sc-17156 landed the pipeline, not
    ///    measurements — so `calibration` is `None`;
    /// 2. `standard_memory_behavior_context` **requires** a calibration identity, so no
    ///    `MemoryBehaviorFixture` can be constructed at all;
    /// 3. without a fixture there is no `MemoryBehaviorRegistration`;
    /// 4. `check_memory_strategy_contract` rejects a contract implementing any *optimized* rung
    ///    with no behavior seam.
    ///
    /// So the blocker is the absent calibration identity — not the seam, and not the mechanism. The
    /// seam cannot be written before the identity exists. This test **fails the moment a
    /// calibration identity lands**, which is exactly when the flip becomes possible, and it names
    /// the rungs waiting on it.
    #[test]
    fn an_optimized_rung_here_is_blocked_by_the_missing_calibration_identity_not_the_seam() {
        let contract = declared();

        // Link 1 — load-bearing, not incidental.
        assert!(
            contract.calibration.is_none(),
            "a calibration identity landed: rungs 1 and 2 can now be declared here. Add the \
             weights-free MemoryBehaviorRegistration seam (model it on \
             mlx_gen_minimax_h3::memory_strategy::MEMORY_BEHAVIOR) and flip them — the mechanisms \
             already exist in this crate."
        );

        // Link 2 — executed. This is the call every behavior seam must make, and it errors here.
        let refused = candle_gen::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedDecode,
            MemoryNumericTier {
                precision: candle_gen::gen_core::Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            candle_gen::gen_core::MemoryBehaviorRoute {
                mode: candle_gen::gen_core::MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .expect_err("a behavior context must not be constructible without a calibration identity");
        assert!(
            refused.to_string().contains("calibration identity"),
            "expected the identity refusal, got: {refused}"
        );

        // …while the MECHANISM rung 2 would declare is genuinely present in this crate today, so
        // the gap is a declaration gap and not a porting gap.
        // Read through `Default`, not off the constants: comparing a constant with itself would
        // pass with the mechanism deleted.
        let tiling = crate::spatial_tiling::SpatialTiling::default();
        assert!(
            tiling.enabled,
            "the shipped candle VAE must tile by default"
        );
        assert_eq!((tiling.tile_height, tiling.tile_width), (256, 256));
        assert_eq!((tiling.overlap_height, tiling.overlap_width), (64, 64));
        // …and the stitcher rung 2 would declare is constructible here today.
        assert!(crate::spatial_tiling::BoundedStitch::new(2, 2, &[1], &[1]).is_ok());
    }

    /// The three ways this backend's declaration diverges from the MLX sibling's, pinned so a later
    /// slice cannot copy an MLX verdict across without the test noticing.
    #[test]
    fn the_candle_declaration_differs_from_mlx_where_the_backends_differ() {
        let contract = declared();
        // 1. No fitted curve, and no calibration identity.
        //
        //    The formula is a component envelope as of sc-18665 rather than the floor arm, because
        //    that is the ONLY variant carrying `resident_components` and therefore the only shape
        //    the AdaLN exclusion can be declared in. That is not MLX's fitted envelope and this
        //    asserts the difference rather than the variant name: one phase and the floor arm's
        //    single `AssetBytes` input, against MLX's three phases and five variables.
        let MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components,
        } = &contract.formula
        else {
            panic!(
                "the AdaLN exclusion requires a component envelope, got {:?}",
                contract.formula
            );
        };
        assert_eq!(
            variables,
            &vec![MemoryFormulaVariable::AssetBytes],
            "no fitted curve on this lane: the floor arm's single input, and nothing else"
        );
        assert_eq!(phases, &vec![MemoryPhase::Denoise]);
        assert_eq!(resident_components.len(), 1);
        assert_eq!(resident_components[0].id, ADALN_COMPONENT_ID);
        assert!(contract.calibration.is_none());
        // 2. Staged residency is IMPLEMENTED IN CODE as of sc-17156 but still declared `Missing`,
        //    because an implemented optimized rung needs a behavior seam this lane does not have
        //    yet, and that seam waits on a `calibration` identity (see
        //    `an_optimized_rung_here_is_blocked_by_the_missing_calibration_identity_not_the_seam`).
        //    Under-declaring is the safe direction — see `strategies`.
        //
        //    `lifecycle.phases` carries `Denoise` alone, which is the eviction's requirement and not
        //    a staged-residency claim: `synchronized_phase_release` and all three mechanism hooks
        //    stay false, and those are what a staged declaration would need.
        assert_eq!(
            contract.lifecycle.phases,
            vec![MemoryPhase::Denoise],
            "only the phase the AdaLN exclusion is ordered against"
        );
        assert!(!contract.lifecycle.synchronized_phase_release);
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .expect("entry")
                .support,
            MemoryStrategySupport::Missing
        );
        // 3. No fused streaming SDPA: rung 3 stays open here on its own evidence, never on MLX's.
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .expect("entry")
                .support,
            MemoryStrategySupport::Missing,
            "candle materializes attention scores; the MLX streaming verdict does not carry over"
        );
        assert!(matches!(
            contract.backend,
            MemoryBackendRealization::CandleCuda { .. }
        ));
        assert_eq!(
            contract.backend.backend_id(),
            "candle",
            "the evidence key must not be recorded under the MLX backend"
        );
    }

    /// A phase hook declared without an implementation is the false declaration conformance cannot
    /// catch. This pins which phases are declared and, for the one that is, what implements it.
    ///
    /// `Denoise` is declared as of sc-18665 and **only** because the AdaLN exclusion is ordered
    /// against it: `conformance_errors` requires a `PrecomputedThenEvicted` component's
    /// `precomputed_in` to be a declared lifecycle phase. That eviction is implemented — the
    /// literal in `crate::model` is unconditional — so the phase is backed by a mechanism, which is
    /// the property this test exists to hold.
    ///
    /// `Conditioning` and `Decode` stay undeclared. They belong to staged residency, whose rung is
    /// still `Missing`, and there is nothing else on this lane that needs them. Adding either would
    /// be the false hook.
    ///
    /// The release primitive the remaining phases WILL declare is exercised here anyway, so this
    /// test also fails if it is removed while that declaration is pending.
    #[test]
    fn no_lifecycle_phase_is_declared_without_an_implementation() {
        let contract = declared();
        for phase in [MemoryPhase::Conditioning, MemoryPhase::Decode] {
            assert!(
                !contract.lifecycle.phases.contains(&phase),
                "{phase:?} must not be declared while StagedResidency is Missing and nothing else \
                 needs it"
            );
        }
        // Declared, and tied to the thing that requires it rather than to a rung.
        assert!(contract.lifecycle.phases.contains(&MemoryPhase::Denoise));
        let component = &contract.resident_components()[0];
        let MemoryComponentResidency::PrecomputedThenEvicted { precomputed_in, .. } =
            &component.residency
        else {
            panic!("the AdaLN component declares an eviction");
        };
        assert_eq!(
            precomputed_in,
            &MemoryPhase::Denoise,
            "Denoise is declared because the exclusion is precomputed there; if the exclusion moved \
             phase, this phase list would have to move with it"
        );
        // Strip the component and the declared phase loses its justification — which is what makes
        // the assertion above a statement about this lane rather than about a constant.
        let mut stripped = declared();
        stripped.formula = MemoryFormulaKind::AssetBytesPlusHeadroom;
        assert!(stripped.resident_components().is_empty());

        crate::dit::release_device_memory(&candle_gen::candle_core::Device::Cpu)
            .expect("the phase-release primitive must exist and succeed");
        assert!(!contract.lifecycle.synchronized_phase_release);
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    /// The candle AdaLN exclusion: net, tier-safe, and conformant on a resolved snapshot.
    ///
    /// The candle lane carried the full 26.02 GB over-charge until sc-18665 — `transformer_bytes`
    /// is a single load-exact scalar and `AssetBytesPlusHeadroom` had nowhere to record that the
    /// projections do not survive into the denoise steady state.
    #[test]
    fn the_candle_contract_declares_the_adaln_exclusion_net_and_scaled() {
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
        let contract =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );

        let net = ADALN_EVICTED_BYTES - ADALN_MODULATION_TABLE_MAX_BYTES;
        assert_eq!(
            contract.resident_components()[0].resident_bytes,
            ADALN_EVICTED_BYTES
        );
        assert_eq!(contract.evicted_component_bytes(), net);
        assert_eq!(
            contract.asset_facts.transformer_bytes - contract.steady_state_transformer_bytes(),
            net,
            "the post-precompute charge is the load-exact one minus the NET exclusion"
        );
        // Declaring the gross figure would claim a saving the precompute does not deliver: it keeps
        // a modulation table in the projections' place.
        assert!(net < ADALN_EVICTED_BYTES);
        // A sub-stack is not auxiliary — those bytes are already inside transformer_bytes.
        assert_eq!(contract.auxiliary_resident_bytes(), 0);
        assert_eq!(contract.asset_facts.overlay_bytes, 0);

        // A staged, smaller DiT scales the sub-stack. Declaring the flat bf16 figure against it
        // would declare a sub-stack larger than the stack containing it, which conformance refuses
        // and the registry turns into a render that cannot resolve a contract at all.
        let staged = tempfile::tempdir().expect("tempdir");
        const Q4_BYTES: u64 = 18_779_970_678;
        sparse_snapshot(staged.path(), &[("transformer", Q4_BYTES)]);
        let tiered = contract_for(
            &LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
                BASE_DIT_PARTITION,
                WeightsSource::Dir(staged.path().join(BASE_DIT_PARTITION)),
            ),
        )
        .expect("contract");
        assert_eq!(tiered.asset_facts.transformer_bytes, Q4_BYTES);
        assert_eq!(
            tiered.resident_components()[0].resident_bytes,
            7_372_786_768,
            "the footprint leg: ADALN_EVICTED_BYTES scaled by the resolved DiT's share of the bf16 \
             DiT. The staged dir here carries NO quantization marker, so the marker leg falls back to \
             the bf16 figure and `min` takes the footprint — see resolved_adaln_bytes, and \
             the_marker_leg_binds_where_it_is_the_tighter_bound for the case where the marker wins"
        );
        assert!(
            tiered.conformance_errors().is_empty(),
            "{:?}",
            tiered.conformance_errors()
        );
        assert!(
            tiered.resident_components()[0].resident_bytes < tiered.asset_facts.transformer_bytes,
            "a sub-stack cannot exceed the stack that contains it"
        );
        assert!(
            ADALN_MODULATION_TABLE_MAX_BYTES < tiered.resident_components()[0].resident_bytes,
            "and it must stay clear of the floor where the eviction would exclude nothing"
        );
    }

    // --- sc-18650: the configured LoRA stack is charged ------------------------------------------

    /// A sparse `.safetensors` file of exactly `bytes`, and the LoRA spec that stages it.
    fn lora_at(path: PathBuf) -> candle_gen::gen_core::AdapterSpec {
        candle_gen::gen_core::AdapterSpec {
            path,
            scale: 1.0,
            kind: candle_gen::gen_core::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }
    }

    fn sparse_file(path: &Path, bytes: u64) -> PathBuf {
        let file = std::fs::File::create(path).expect("adapter file");
        file.set_len(bytes).expect("sparse adapter");
        path.to_path_buf()
    }

    /// Roughly the published `lightx2v/Minimax-h3-Turbo` 8-step diffusers export — 624 bf16 factor
    /// tensors. The exact figure does not matter; that it is *not zero* is the whole point.
    const TURBO_LORA_BYTES: u64 = 636_512_768;

    /// **A configured LoRA is charged as a typed, additive, whole-render overlay** (sc-18650).
    ///
    /// This lane published `overlay_bytes: 0` from sc-18659 until this story while
    /// [`crate::model::descriptor`] declared `supports_lora: true` and sc-18724's seam installed the
    /// factors — so a LoRA render's resident tensors were charged nothing on a **shipped** path.
    /// [`MemoryComponentResidency::WholeRender`] rather than an eviction, and
    /// [`AdapterResidencyMode::Additive`] rather than `Folded`, because [`crate::adapters`] installs
    /// a forward-time residual that is never merged into the base on any tier.
    ///
    /// The no-adapter control at the top is what makes the arm attributable *and* is a guard in its
    /// own right: the whole declaration is conditional on `overlay > 0` so that an ordinary render's
    /// contract is byte-for-byte the one this provider has always published.
    #[test]
    fn the_configured_lora_stack_is_charged_as_a_typed_additive_overlay() {
        let root = tempfile::tempdir().expect("tempdir");
        full_snapshot(root.path());
        let bare =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");

        // Control: no adapter, and the contract is unmoved from its pre-sc-18650 shape.
        assert_eq!(bare.asset_facts.overlay_bytes, 0);
        assert_eq!(bare.auxiliary_resident_bytes(), 0);
        assert_eq!(
            bare.resident_components().len(),
            1,
            "the AdaLN sub-stack and nothing else"
        );
        assert!(!bare.formula.uses(MemoryFormulaVariable::OverlayBytes));
        assert!(
            bare.conformance_errors().is_empty(),
            "{:?}",
            bare.conformance_errors()
        );

        let lora = sparse_file(&root.path().join("turbo.safetensors"), TURBO_LORA_BYTES);
        let adapted = contract_for(
            &LoadSpec::new(WeightsSource::Dir(root.path().into()))
                .with_adapters(vec![lora_at(lora)]),
        )
        .expect("contract");

        assert_eq!(
            adapted.asset_facts.overlay_bytes, TURBO_LORA_BYTES,
            "the forward-time residual factors are resident for the whole render, so their \
             load-exact bytes are the overlay"
        );
        assert!(
            adapted.formula.uses(MemoryFormulaVariable::OverlayBytes),
            "a typed auxiliary component without the variable is a conformance error, and a \
             non-zero overlay without the component is the other half of the same rule"
        );
        let stack = adapted
            .resident_components()
            .iter()
            .find(|component| component.kind == MemoryComponentKind::AdapterStack)
            .expect("the LoRA stack must be declared as a TYPED auxiliary component");
        assert_eq!(stack.id, ADAPTER_STACK_COMPONENT_ID);
        assert_eq!(stack.resident_bytes, TURBO_LORA_BYTES);
        assert_eq!(
            stack.residency,
            MemoryComponentResidency::WholeRender,
            "never folded on any tier, so the factors live to the last denoise step"
        );
        assert_eq!(stack.bounded_by, None, "no rung bounds the adapter stack");
        assert_eq!(adapted.auxiliary_resident_bytes(), TURBO_LORA_BYTES);
        assert!(
            adapted.conformance_errors().is_empty(),
            "{:?}",
            adapted.conformance_errors()
        );

        // The AdaLN sub-stack is untouched: it is inside `transformer_bytes` and is not auxiliary,
        // so the two components must not contaminate each other.
        assert_eq!(
            adapted.evicted_component_bytes(),
            bare.evicted_component_bytes()
        );
        assert_eq!(
            adapted.asset_facts.base_bytes, bare.asset_facts.base_bytes,
            "`base_bytes` must EXCLUDE the overlay — conformance refuses the double charge"
        );
    }

    /// **An adapter that is not on disk leaves the contract byte-identical** (sc-18650) — the leg
    /// that lets the sizing fail closed without inventing a load error.
    ///
    /// The MLX sibling refuses a nonexistent adapter in `load` outright, so it can size with a flat
    /// `ok_or_else(Err)`. This lane never has, and two shipped guards depend on that:
    /// `crate::model::tests::a_staged_lora_survives_load_rather_than_being_dropped` and
    /// `…::lokr_and_the_two_foreign_adapter_knobs_are_each_refused_individually` both stage a
    /// `/turbo.safetensors` that does not exist, because they are about **retention** and **knob
    /// refusal**. `0` is exact for them rather than a guess: `read_adapter` stats the path first, so
    /// [`crate::adapters::apply_minimax_h3_adapters`] refuses before a factor is materialized.
    ///
    /// Pinned here so the absent leg is a decision rather than an accident — a later "tidy-up" that
    /// makes it an error deletes those two guards.
    #[test]
    fn an_adapter_that_is_not_on_disk_leaves_the_contract_byte_identical() {
        let root = tempfile::tempdir().expect("tempdir");
        full_snapshot(root.path());
        let bare =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        let absent = contract_for(
            &LoadSpec::new(WeightsSource::Dir(root.path().into()))
                .with_adapters(vec![lora_at(PathBuf::from("/turbo.safetensors"))]),
        )
        .expect("an absent adapter must not become a contract error");
        assert_eq!(
            absent, bare,
            "an adapter that can put no bytes anywhere must publish the no-adapter contract"
        );
    }

    /// **An adapter that IS on disk but sizes to zero fails closed** (sc-18650).
    ///
    /// The case that makes the probe a fail-closed check rather than a formality, and it needs no
    /// malformed input at all. `candle_gen::train::merge::read_adapter` is **extension-blind** — it
    /// reads the bytes and parses safetensors out of the buffer — while `safetensors_path_bytes`
    /// counts only non-hidden `.safetensors`. So a perfectly loadable adapter named `turbo.bin`
    /// installs all 312 modules' factors and would be charged **zero**: an under-declaration in the
    /// OOM direction, on the live path.
    #[test]
    fn an_adapter_present_on_disk_but_unsizable_is_refused_rather_than_charged_zero() {
        let root = tempfile::tempdir().expect("tempdir");
        full_snapshot(root.path());
        let misnamed = sparse_file(&root.path().join("turbo.bin"), TURBO_LORA_BYTES);
        let err = contract_for(
            &LoadSpec::new(WeightsSource::Dir(root.path().into()))
                .with_adapters(vec![lora_at(misnamed)]),
        )
        .expect_err("an adapter that loads but cannot be sized must not be charged 0");
        assert!(
            matches!(err, candle_gen::gen_core::Error::Unsupported(_)),
            "must be the contract-load-bearing Unsupported, not an opaque Msg: {err:?}"
        );
        assert!(
            err.to_string().contains("turbo.bin"),
            "the refusal must name the file it could not size: {err}"
        );

        // ...and a hidden `.safetensors` is the same hole with a different spelling.
        let hidden = sparse_file(&root.path().join(".turbo.safetensors"), TURBO_LORA_BYTES);
        assert!(contract_for(
            &LoadSpec::new(WeightsSource::Dir(root.path().into()))
                .with_adapters(vec![lora_at(hidden)]),
        )
        .is_err());
    }

    /// [`ADALN_MODULATION_TABLE_MAX_BYTES`] is read off a **real** worst-case schedule rather than
    /// typed twice, so a schedule or config change moves it or reds here.
    ///
    /// The MLX sibling pins the identical figure from its own schedule types. That the two agree is
    /// the point: the retained table is a property of the checkpoint's schedule domain, not of a
    /// backend.
    #[test]
    fn the_retained_table_is_the_worst_case_over_the_admitted_schedule() {
        let config = crate::dit::MiniMaxH3DitConfig::default();
        // `num_inference_steps` counts the terminal sigma = 0, at which the model is never
        // evaluated, so the longest admitted run is `MAX_STEPS + 1` inference steps.
        let longest = crate::denoise::JointSchedule::new(crate::MAX_STEPS as usize + 1)
            .expect("the longest admitted schedule");
        assert_eq!(longest.num_evals(), crate::MAX_STEPS as usize);
        let rows = crate::denoise::adaln_schedule(&longest)
            .expect("adaln schedule")
            .modulation_rows() as u64;
        let widest = crate::dit::config::MODULATION_PARAMS as u64
            * rows
            * config.hidden_size as u64
            * config.num_layers as u64
            * 2;
        assert_eq!(
            widest, ADALN_MODULATION_TABLE_MAX_BYTES,
            "the declared retained table must be the widest the admitted schedule can produce"
        );

        // …and it really is the worst case: a shorter schedule keeps strictly less.
        let default = crate::denoise::JointSchedule::new(crate::DEFAULT_STEPS as usize + 1)
            .expect("the default schedule");
        let default_rows = crate::denoise::adaln_schedule(&default)
            .expect("adaln schedule")
            .modulation_rows() as u64;
        assert!(
            default_rows < rows,
            "the default schedule's {default_rows} rows must sit below the admitted maximum's \
             {rows}, or the declaration is not conservative"
        );
        // The gross figure is the shipped config's own projection bytes.
        let out = config.adaln_out_features() as u64;
        assert_eq!(
            config.num_layers as u64 * (out * config.time_embed_dim as u64 + out) * 2,
            ADALN_EVICTED_BYTES
        );
    }
}
