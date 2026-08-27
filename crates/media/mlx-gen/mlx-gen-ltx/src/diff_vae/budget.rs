//! sc-18799 — **memory-budgeted DiffVAE decode selection.**
//!
//! [`super::NaDiffusionDecoder::decode_tiled`] shipped in sc-18766 with real neighbourhood-window
//! tiling arithmetic, but nothing chose a tile for it: every caller either passed a hand-built
//! [`DiffVaeTiling`] or ran the single-pass [`decode`](super::NaDiffusionDecoder::decode). The conv
//! decoder has had a budgeted auto-selector since sc-6894
//! ([`crate::pipeline::auto_tiling_budgeted_ltx`]); this is the DiffVAE's, in the same shape:
//!
//! ```text
//! auto_*  → read the process memory limit → × SAFE_FRACTION → pure planner
//!         → Ok(None)      the single-pass decode already fits
//!         → Ok(Some(cfg)) this tile fits
//!         → Err(..)       catchable, *before* the decode, instead of a SIGKILL
//! ```
//!
//! What differs is the axis count. The conv selector picks one **spatial edge** plus a temporal
//! `(tile, overlap)` pair off two fixed candidate lists. The DiffVAE tiles three axes
//! *independently* — [`DiffVaeTiling::tile`] is `[T, H, W]` in stage-4-input cells — so the planner
//! enumerates a per-axis size grid and scores the product, which is exactly what upstream's own
//! `recommended_decode_tiling_config` does.
//!
//! ## The four upstream modes
//!
//! Upstream exposes `--diffvae-optimization` with four presets ([`DiffVaeMode`]); each resolves to
//! an *install recipe* (`DiffVAEConfig`) whose memory-relevant content is a stage-5 working-set
//! coefficient and a budget-safety withhold. Both are declared here as data
//! ([`DiffVaeMode::declared_stage5_coef`], [`DiffVaeMode::declared_budget_safety_bytes`]) so the
//! ladder can reason about every mode, and both are then put through upstream's own host resolve
//! ([`DiffVaeMode::resolve_for_host`]) before a plan is costed — because upstream's
//! `stage5_mem_coef` also resolves the host first, and on a host without NATTEN it *downgrades the
//! chunked coefficients and refuses `combined_compile` outright*.
//!
//! This port ships exactly one neighbourhood-attention kernel: [`super::na3d`], the eager tiled
//! SDPA (upstream `NAttentionKind::EAGER_SDPA`; see the `NA_TILE_BUDGET` comment). There is no
//! NATTEN, no Triton `na3d`, and no fused CuTe DSL kernel on Metal at all. So:
//!
//! | upstream mode | resolves to | on this host |
//! | --- | --- | --- |
//! | `chunked_eager` | eager SDPA, coef 5, safety 1 GiB | **runs** |
//! | `chunked_compile` | eager SDPA, coef 5, safety 1 GiB (upstream's own no-NATTEN remap) | **runs** |
//! | `combined_compile` | — | **refused**: upstream raises without NATTEN |
//! | `blackwell_dsl` | — | **refused**: needs a datacenter Blackwell GPU + CuTe DSL |
//!
//! The two refusals are upstream's refusals, moved verbatim; they are not this port declining to
//! implement something. A `blackwell_dsl` decode that *ran* here would be an eager-SDPA decode
//! wearing a fused-kernel budget — a plan that under-predicts its own peak by the ratio between
//! coefficients 2.5 and 5, which is precisely the failure this module exists to prevent.
//!
//! ## The cost model
//!
//! Upstream's estimate is written for its own decoder, which streams temporal groups and keeps a
//! `2 * tile_t`-deep accumulator in bf16. [`super::NaDiffusionDecoder::decode_tiled`] does neither:
//! it holds the **whole stage-5 canvas** in f32 and adds each padded tile into it. Reusing
//! upstream's accumulator term would under-predict our peak by the ratio between a temporal slab
//! and the full clip. The three terms below are therefore this decode's, with upstream's *shape*:
//!
//! * `feature` — the stages-1-3 output, resident for the whole tiled decode.
//! * the pixel-space accumulators — [`ACCUM_LIVE_CHANNEL_CANVASES`] full canvases in f32.
//! * the per-tile stage-5 working set — `tokens × stage5_channels × coef ×`
//!   [`STAGE5_BYTES_PER_TOKEN_CHANNEL_COEF_UNIT`].
//!
//! Reference: `Lightricks/LTX-2` `d151147788a9284cca791edc6ce898007e727fe6`,
//! `packages/ltx-core/src/ltx_core/model/video_vae/diffusion_tiling.py`
//! (`recommended_decode_tiling_config`, `stage5_mem_coef`, `budget_safety_bytes`) and
//! `.../video_vae/transformer/{config,apply}.py` (`DiffVAEMode`, `resolve_attention_for_host`).

use mlx_rs::memory::get_memory_limit;

use mlx_gen::{Error, Result};

use super::{split_by_size, DiffVaeTiling, NaDiffusionDecoderConfig};

/// Fraction of the process memory limit a decode may plan against — the same 0.85 the conv
/// selector uses ([`crate::pipeline::auto_tiling_budgeted_ltx`]), so the two decoders on one
/// machine agree about what "safe" means.
pub const SAFE_FRACTION: f64 = 0.85;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---------------------------------------------------------------------------------------------
// The four upstream modes
// ---------------------------------------------------------------------------------------------

/// Upstream's user-facing DiffVAE decode presets (`DiffVAEMode`, `--diffvae-optimization`).
///
/// The variants carry upstream's semantics, not this port's: `Chunked*` defer stage 4 and split
/// attention into four W-chunks, `CombinedCompile` uses a combined context buffer with full-volume
/// attention, and `BlackwellDsl` is a fused CuTe DSL neighbourhood-attention + stage-5 kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DiffVaeMode {
    /// `chunked_eager` — upstream's default. Deferred stage-4 inject, W-chunks 4, no compile.
    ChunkedEager,
    /// `chunked_compile` — the same chunked pathway plus `torch.compile` on attn+mlp.
    ChunkedCompile,
    /// `combined_compile` — combined context, full-volume attention, everything compiled.
    /// **Requires NATTEN upstream**; without it upstream's `resolve_attention_for_host` raises.
    CombinedCompile,
    /// `blackwell_dsl` — deferred stage-4 plus a fused CuTe DSL kernel. Datacenter Blackwell only;
    /// upstream explicitly does *not* use it on consumer Blackwell (`sm_120`).
    BlackwellDsl,
}

/// Which neighbourhood-attention kernel serves a resolved mode (upstream `NAttentionKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NaKind {
    /// CUTLASS-FNA via the `natten` package.
    Natten,
    /// The fused CuTe DSL kernel.
    BlackwellDsl,
    /// Upstream's Triton `na3d` compatibility fallback.
    Triton,
    /// Eager tiled SDPA — what [`super::na3d`] is.
    EagerSdpa,
}

/// Upstream `_MEM_COEF_BY_MODE[CHUNKED_EAGER]`.
const COEF_CHUNKED_EAGER: f64 = 5.0;
/// Upstream `_MEM_COEF_BY_MODE[CHUNKED_COMPILE]` (assumes NATTEN; remapped hosts use the eager one).
const COEF_CHUNKED_COMPILE: f64 = 7.0;
/// Upstream `_MEM_COEF_BY_MODE[COMBINED_COMPILE]`.
const COEF_COMBINED_COMPILE: f64 = 11.0;
/// Upstream `_MEM_COEF_BY_MODE[BLACKWELL_DSL]`.
const COEF_BLACKWELL_DSL: f64 = 2.5;

/// Upstream `_BUDGET_SAFETY_BYTES_EAGER`.
const SAFETY_BYTES_EAGER: u64 = 1 << 30;
/// Upstream `_BUDGET_SAFETY_BYTES_COMPILED`.
const SAFETY_BYTES_COMPILED: u64 = 2 << 30;

impl DiffVaeMode {
    /// Every upstream preset, in the order `--diffvae-optimization` documents them.
    pub const ALL: [Self; 4] = [
        Self::ChunkedEager,
        Self::ChunkedCompile,
        Self::CombinedCompile,
        Self::BlackwellDsl,
    ];

    /// The upstream CLI spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ChunkedEager => "chunked_eager",
            Self::ChunkedCompile => "chunked_compile",
            Self::CombinedCompile => "combined_compile",
            Self::BlackwellDsl => "blackwell_dsl",
        }
    }

    /// Parse the upstream CLI spelling.
    pub fn parse(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|m| m.as_str() == s)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "ltx diffvae: unknown diffvae optimization {s:?} (expected one of {})",
                    Self::ALL.map(Self::as_str).join(" | ")
                ))
            })
    }

    /// The stage-5 working-set multiplicity upstream declares for this preset, **before** the host
    /// resolve (upstream `_MEM_COEF_BY_MODE`). This is the mode's own number; what a plan is costed
    /// against is [`ResolvedDiffVaeMode::stage5_coef`].
    pub fn declared_stage5_coef(self) -> f64 {
        match self {
            Self::ChunkedEager => COEF_CHUNKED_EAGER,
            Self::ChunkedCompile => COEF_CHUNKED_COMPILE,
            Self::CombinedCompile => COEF_COMBINED_COMPILE,
            Self::BlackwellDsl => COEF_BLACKWELL_DSL,
        }
    }

    /// Bytes upstream withholds from the recommend budget for this preset, before the host resolve
    /// (upstream `_BUDGET_SAFETY_BYTES_EAGER` / `_COMPILED`): a compiled pathway pays for its own
    /// Dynamo/kernel-cache residency, an eager one does not.
    pub fn declared_budget_safety_bytes(self) -> u64 {
        match self {
            Self::ChunkedEager => SAFETY_BYTES_EAGER,
            Self::ChunkedCompile | Self::CombinedCompile | Self::BlackwellDsl => {
                SAFETY_BYTES_COMPILED
            }
        }
    }

    /// The neighbourhood-attention kernel this preset asks for, before the host resolve
    /// (upstream `DiffVAEMode.resolve().attention`).
    pub fn declared_attention(self) -> NaKind {
        match self {
            Self::ChunkedEager | Self::ChunkedCompile | Self::CombinedCompile => NaKind::Natten,
            Self::BlackwellDsl => NaKind::BlackwellDsl,
        }
    }

    /// Upstream's `resolve_attention_for_host` + `stage5_mem_coef` + `budget_safety_bytes`, in one
    /// step, against a host that declares which kernels it can actually serve.
    ///
    /// * A NATTEN-asking preset on a host with NATTEN keeps its declared coefficient.
    /// * A **chunked** NATTEN preset on a host without it is remapped to Triton/eager, which also
    ///   drops `torch.compile` — so it takes the `chunked_eager` coefficient *and* the eager safety
    ///   withhold. Upstream does exactly this; taking the declared coefficient 7 on a host running
    ///   the eager kernel would under-predict nothing but *over*-predict, and taking 7 while
    ///   running a heavier kernel would under-predict — either way the resolved number is the only
    ///   honest one.
    /// * A **combined** NATTEN preset on a host without it is refused (upstream raises).
    /// * `blackwell_dsl` on a host without the fused kernel is refused.
    pub fn resolve_for_host(self, host: HostNaSupport) -> Result<ResolvedDiffVaeMode> {
        let attention = match self.declared_attention() {
            NaKind::BlackwellDsl => {
                if !host.blackwell_dsl {
                    return Err(Error::Msg(format!(
                        "ltx diffvae: diffvae_optimization={} needs the fused CuTe DSL \
                         neighborhood-attention kernel on a datacenter Blackwell GPU (sm_10x, e.g. \
                         B200); {}. Use chunked_eager or chunked_compile.",
                        self.as_str(),
                        host.blackwell_reason
                    )));
                }
                NaKind::BlackwellDsl
            }
            NaKind::Natten if host.natten => NaKind::Natten,
            NaKind::Natten if self == Self::CombinedCompile => {
                return Err(Error::Msg(format!(
                    "ltx diffvae: diffvae_optimization={} requires NATTEN (combined context + \
                     full-volume neighborhood attention); this build serves neighborhood attention \
                     with the eager tiled SDPA `na3d` only. Use chunked_eager or chunked_compile, \
                     which upstream remaps onto the same fallback kernel.",
                    self.as_str()
                )));
            }
            NaKind::Natten if host.triton => NaKind::Triton,
            NaKind::Natten => NaKind::EagerSdpa,
            // The port never *declares* Triton/eager; they are only ever resolve results.
            kind @ (NaKind::Triton | NaKind::EagerSdpa) => kind,
        };
        // Upstream: a chunked mode remapped onto the fallback kernel is "effectively eager for peak
        // VRAM" — both the coefficient and the safety withhold fall back with it.
        let remapped = matches!(attention, NaKind::Triton | NaKind::EagerSdpa);
        let (stage5_coef, budget_safety_bytes) = if remapped {
            (
                Self::ChunkedEager.declared_stage5_coef(),
                Self::ChunkedEager.declared_budget_safety_bytes(),
            )
        } else {
            (
                self.declared_stage5_coef(),
                self.declared_budget_safety_bytes(),
            )
        };
        Ok(ResolvedDiffVaeMode {
            mode: self,
            attention,
            stage5_coef,
            budget_safety_bytes,
        })
    }
}

/// A mode after the host resolve — the only form a plan may be costed against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedDiffVaeMode {
    /// The preset the caller asked for.
    pub mode: DiffVaeMode,
    /// The kernel that will actually serve neighbourhood attention.
    pub attention: NaKind,
    /// Stage-5 working-set multiplicity after the resolve (upstream `stage5_mem_coef`).
    pub stage5_coef: f64,
    /// Bytes withheld from the budget after the resolve (upstream `budget_safety_bytes`).
    pub budget_safety_bytes: u64,
}

/// Which neighbourhood-attention kernels the running host can serve.
///
/// Built by [`HostNaSupport::detect`] in production and by hand in tests, so the *arithmetic* of a
/// mode that cannot run here is still exercisable without pretending the kernel exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostNaSupport {
    /// CUTLASS-FNA through the `natten` package.
    pub natten: bool,
    /// Upstream's Triton `na3d`.
    pub triton: bool,
    /// The fused CuTe DSL kernel on a datacenter Blackwell GPU.
    pub blackwell_dsl: bool,
    /// Why `blackwell_dsl` is what it is — quoted into the refusal so it names the real reason
    /// rather than "unsupported".
    pub blackwell_reason: &'static str,
}

/// The compute-capability **major** of *datacenter* Blackwell — B200/GB200 `sm_100` and B300
/// `sm_103`. Consumer Blackwell is `sm_120`, i.e. major 12, and is deliberately **not** this: the
/// CuTe DSL path is documented upstream as "not used on consumer Blackwell (sm_120)", and treating
/// 12.0 as "Blackwell" is how a host ends up running a kernel that returns plausible garbage.
pub const DATACENTER_BLACKWELL_CC_MAJOR: i32 = 10;

/// Whether a `(major, minor)` CUDA compute capability is a datacenter Blackwell part.
pub fn compute_cap_is_datacenter_blackwell(cap: (i32, i32)) -> bool {
    cap.0 == DATACENTER_BLACKWELL_CC_MAJOR
}

impl HostNaSupport {
    /// What this build can actually serve.
    ///
    /// All three are `false` and stay `false`: this is the MLX/Metal backend, which has no CUDA
    /// device at all, so NATTEN (a CUDA extension), Triton `na3d` (CUDA), and the CuTe DSL kernel
    /// (CUDA, datacenter Blackwell) are all absent by construction rather than by configuration.
    /// [`super::na3d`] is upstream's eager tiled SDPA and is what every resolved mode runs on.
    pub fn detect() -> Self {
        Self {
            natten: false,
            triton: false,
            blackwell_dsl: false,
            blackwell_reason:
                "this is the MLX/Metal backend, which binds no CUDA device (the CuTe DSL kernel is \
                 CUDA-only, and consumer Blackwell sm_120 would not qualify either)",
        }
    }

    /// A host declaration for tests — lets the *budget arithmetic* of a mode be exercised without
    /// claiming the kernel runs here.
    #[cfg(test)]
    pub(crate) fn with(natten: bool, triton: bool, blackwell_dsl: bool) -> Self {
        Self {
            natten,
            triton,
            blackwell_dsl,
            blackwell_reason: "test host declaration",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The cost model
// ---------------------------------------------------------------------------------------------

/// Bytes per element everywhere in this decoder: it runs f32 end to end (see the module header of
/// [`super`] — the pmetal bf16 SDPA/GEMM hazards are what a 22-block attention stack amplifies).
/// Upstream's own estimate uses 2 because its features are bf16; using 2 here would halve every
/// term.
const ELEMENT_BYTES: u64 = 4;

/// How many single-channel full-canvas f32 buffers [`super::NaDiffusionDecoder::decode_tiled`] has
/// live at its peak.
///
/// Counted off the driver rather than assumed. Inside the tile loop the live set is
/// `accumulator` + `placed` + the `add` result, each `[1, out_channels, F, H, W]` — the previous
/// accumulator is only released once the sum exists. So `3 × out_channels` channel-canvases, and
/// `out_channels = 3` makes that 9. Afterwards the loop is done and the live set is
/// `accumulator` + `weight_3d` (1 channel) + `blended`, i.e. `2 × out_channels + 1 = 7`, which is
/// smaller. The `+ 1` below carries the weight canvas anyway so the constant bounds both phases.
const ACCUM_LIVE_CHANNEL_CANVASES: u64 = 3 * 3 + 1;

/// Bytes of concurrent decoder working set per `(stage-5 token × stage-5 channel × coefficient
/// unit)`, for **this** backend's neighbourhood attention.
///
/// Upstream's coefficients are multiplicities over a *fused* NATTEN kernel's working set; this
/// port's [`super::na3d`] is the eager tiled SDPA, whose per-token cost is its own thing — the
/// additive window mask alone is up to `NA_TILE_BUDGET` f32 per tile-row, and the SwiGLU hidden
/// buffer is `4 × dim` wide at up to `SWIGLU_TILE_TOKENS` tokens. So the *mode* contributes the
/// coefficient and the *backend* contributes this unit, exactly as the conv model splits its fixed
/// floor from its per-voxel slopes.
///
/// Fit from the real-weight anchors in `tests/ltx_2_5_diffvae_parity.rs` (see
/// [`DIFFVAE_FIXED_BYTES`] for the fit and its provenance) and rounded **up**: the model must never
/// under-predict, which `budgeted_diffvae_estimate_never_under_predicts_the_measured_peak` asserts
/// directly against measured peaks.
const STAGE5_BYTES_PER_TOKEN_CHANNEL_COEF_UNIT: f64 = 17.0;

/// Fixed floor (bytes): resident decoder weights (~0.83 GB) plus the MLX base working set, paid
/// whatever the geometry.
///
/// Fit together with [`STAGE5_BYTES_PER_TOKEN_CHANNEL_COEF_UNIT`] from four real-weight anchors
/// measured on this Mac (M-series, f32, 2026-08-19 and re-measured under sc-18799); the two
/// single-pass anchors fix the slope and the intercept, the two tiled anchors then have to be
/// covered by the same line. Rounded **up** from the ~1.56 GiB intercept the two single-pass
/// anchors imply. These are conservative upper bounds, not goldens: the committed assertion is the
/// inequality "estimate ≥ measured", never an equality against a machine-dependent number.
const DIFFVAE_FIXED_BYTES: u64 = 2_400_000_000;

/// What a decode of a given latent will do — the two shapes
/// [`super::NaDiffusionDecoder::decode_seeded`] can take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePlan {
    /// [`super::NaDiffusionDecoder::decode`] — one pass over the whole volume, no accumulators.
    SinglePass,
    /// [`super::NaDiffusionDecoder::decode_tiled`] with this tiling.
    Tiled(DiffVaeTiling),
}

/// The geometry a plan is costed over, derived once from the config and the latent extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeGeometry {
    /// Stage-4-input grid `[T, H, W]` the tiling is expressed in.
    pub stage4: [i32; 3],
    /// Full stage-5 pixel canvas `[F, H, W]` the accumulators span.
    pub canvas: [i32; 3],
    /// Channels of the resident stages-1-3 output.
    pub stage4_channels: i32,
    /// Stage-5 feature width.
    pub stage5_channels: i32,
    /// Pixel channels of the decoded video.
    pub out_channels: i32,
    /// Last upsample hop's per-axis stride — stage-4 cells to stage-5 tokens.
    pub last_stride: [i32; 3],
}

impl DecodeGeometry {
    /// The geometry [`super::NaDiffusionDecoder::decode`] / `decode_tiled` will actually run at for
    /// this latent — the latent size floor applied first, exactly as those two do.
    pub fn new(
        cfg: &NaDiffusionDecoderConfig,
        latent_t: i32,
        latent_h: i32,
        latent_w: i32,
    ) -> Self {
        let min = cfg.min_latent_shape();
        let stage4 = cfg.stage4_shape(
            latent_t.max(min[0]),
            latent_h.max(min[1]),
            latent_w.max(min[2]),
        );
        let last = cfg.upsamples.len() - 1;
        Self {
            stage4,
            canvas: cfg.stage5_pixel_shape(stage4, true, true),
            // The stages-1-3 output leaves the stage before the last upsample hop, so it carries
            // that stage's channel count (upstream `stage4_channels`).
            stage4_channels: cfg.stage_channels[last],
            stage5_channels: cfg.stage5_width(),
            out_channels: cfg.out_channels,
            last_stride: cfg.upsamples[last].0,
        }
    }

    /// Stage-5 tokens a stage-4 tile expands to, pre-unpatchify (upstream
    /// `stage5_tokens_for_pixel_tile`, in this port's units). The causal duplicate-frame drop is
    /// deliberately *not* subtracted — one frame less is one frame of headroom.
    fn stage5_tokens(&self, tile: [i32; 3]) -> u64 {
        (0..3)
            .map(|axis| {
                u64::from(tile[axis].max(0) as u32) * u64::from(self.last_stride[axis] as u32)
            })
            .product()
    }

    /// The resident stages-1-3 output.
    fn feature_bytes(&self) -> u64 {
        u64::from(self.stage4[0].max(0) as u32)
            * u64::from(self.stage4[1].max(0) as u32)
            * u64::from(self.stage4[2].max(0) as u32)
            * u64::from(self.stage4_channels.max(0) as u32)
            * ELEMENT_BYTES
    }

    /// One single-channel full stage-5 canvas in f32.
    fn channel_canvas_bytes(&self) -> u64 {
        u64::from(self.canvas[0].max(0) as u32)
            * u64::from(self.canvas[1].max(0) as u32)
            * u64::from(self.canvas[2].max(0) as u32)
            * ELEMENT_BYTES
    }
}

/// Estimated concurrent peak (bytes) of a DiffVAE decode.
///
/// Pure — no global state, no device — so it is unit-testable and so the same function costs both
/// the candidate grid and the regression assertion against measured peaks.
pub fn estimated_diffvae_decode_peak_bytes(
    geometry: &DecodeGeometry,
    plan: DecodePlan,
    resolved: &ResolvedDiffVaeMode,
) -> u64 {
    let (tile, canvases) = match plan {
        // The single-pass decode runs one tile over the whole grid and writes its pixels straight
        // out: no accumulator, no weight canvas, just the decoded volume itself.
        DecodePlan::SinglePass => (
            geometry.stage4,
            u64::from(geometry.out_channels.max(0) as u32),
        ),
        DecodePlan::Tiled(t) => (t.tile, ACCUM_LIVE_CHANNEL_CANVASES),
    };
    let stage5 = geometry.stage5_tokens(tile) as f64
        * f64::from(geometry.stage5_channels.max(0))
        * resolved.stage5_coef
        * STAGE5_BYTES_PER_TOKEN_CHANNEL_COEF_UNIT;
    DIFFVAE_FIXED_BYTES
        .saturating_add(geometry.feature_bytes())
        .saturating_add(geometry.channel_canvas_bytes().saturating_mul(canvases))
        .saturating_add(stage5 as u64)
}

// ---------------------------------------------------------------------------------------------
// The selector
// ---------------------------------------------------------------------------------------------

/// Why a geometry could not be planned — the DiffVAE analogue of the conv selector's
/// `TilingBudgetError`, mapped to a catchable [`Error`] by [`plan_diffvae_tiling`].
#[derive(Clone, Copy, Debug, PartialEq)]
enum BudgetFailure {
    /// Even the accumulators plus the resident feature exceed the budget — no tile can help.
    FloorExceedsBudget { floor_gib: f64 },
    /// The smallest legal tile still exceeds the budget.
    SmallestTileExceedsBudget { smallest_gib: f64 },
}

/// **Memory-budgeted** tiling for the DiffVAE decode: the DiffVAE's
/// [`crate::pipeline::auto_tiling_budgeted_ltx`].
///
/// `Ok(None)` → the single-pass [`super::NaDiffusionDecoder::decode`] already fits. `Ok(Some(t))` →
/// run [`super::NaDiffusionDecoder::decode_tiled`] with `t`. `Err` → a catchable over-budget signal
/// returned *before* the decode rather than a SIGKILL inside it.
pub fn auto_diffvae_tiling_budgeted_ltx(
    cfg: &NaDiffusionDecoderConfig,
    latent_t: i32,
    latent_h: i32,
    latent_w: i32,
    mode: DiffVaeMode,
) -> Result<Option<DiffVaeTiling>> {
    let resolved = mode.resolve_for_host(HostNaSupport::detect())?;
    let budget_gib = get_memory_limit() as f64 / GIB * SAFE_FRACTION;
    plan_diffvae_tiling(cfg, latent_t, latent_h, latent_w, budget_gib, &resolved)
}

/// The pure planner behind [`auto_diffvae_tiling_budgeted_ltx`] — the `safe_gib` ceiling and the
/// resolved mode are injected so the selection is testable without a device and without the global
/// memory limit.
pub fn plan_diffvae_tiling(
    cfg: &NaDiffusionDecoderConfig,
    latent_t: i32,
    latent_h: i32,
    latent_w: i32,
    safe_gib: f64,
    resolved: &ResolvedDiffVaeMode,
) -> Result<Option<DiffVaeTiling>> {
    let geometry = DecodeGeometry::new(cfg, latent_t, latent_h, latent_w);
    let usable = (safe_gib * GIB) as u64;
    let usable = usable.saturating_sub(resolved.budget_safety_bytes);

    if estimated_diffvae_decode_peak_bytes(&geometry, DecodePlan::SinglePass, resolved) <= usable {
        return Ok(None);
    }

    let overlap = cfg.tile_halo();
    let min_tile = cfg.min_tile_shape();
    match select_tiling(&geometry, min_tile, overlap, usable, resolved) {
        Ok(tiling) => Ok(Some(tiling)),
        Err(failure) => Err(budget_error(&geometry, safe_gib, resolved, failure)),
    }
}

/// The three-axis size grid, scored the way upstream scores it.
fn select_tiling(
    geometry: &DecodeGeometry,
    min_tile: [i32; 3],
    overlap: [i32; 3],
    usable: u64,
    resolved: &ResolvedDiffVaeMode,
) -> std::result::Result<DiffVaeTiling, BudgetFailure> {
    let floor = estimated_diffvae_decode_peak_bytes(
        geometry,
        DecodePlan::Tiled(DiffVaeTiling {
            tile: [0, 0, 0],
            overlap,
        }),
        resolved,
    );
    if floor > usable {
        return Err(BudgetFailure::FloorExceedsBudget {
            floor_gib: floor as f64 / GIB,
        });
    }

    let candidates: Vec<Vec<i32>> = (0..3)
        .map(|axis| {
            axis_candidates(
                geometry.stage4[axis],
                min_tile[axis],
                overlap[axis],
                geometry.last_stride[axis],
            )
        })
        .collect();

    let mut best: Option<(f64, i64, i64, [i32; 3])> = None;
    let mut smallest: Option<u64> = None;
    for &t in &candidates[0] {
        for &h in &candidates[1] {
            for &w in &candidates[2] {
                let tile = [t, h, w];
                let bytes = estimated_diffvae_decode_peak_bytes(
                    geometry,
                    DecodePlan::Tiled(DiffVaeTiling { tile, overlap }),
                    resolved,
                );
                smallest = Some(smallest.map_or(bytes, |s: u64| s.min(bytes)));
                if bytes > usable {
                    continue;
                }
                // Upstream `volumetric_overlap_waste`: processed / unique volume, derived from the
                // *decode's own* splitter so the tile count is the one decode will produce.
                let counts: [usize; 3] = [0, 1, 2].map(|axis| {
                    split_by_size(
                        geometry.stage4[axis],
                        tile[axis],
                        overlap[axis],
                        min_tile[axis],
                    )
                    .len()
                });
                let processed = counts.iter().map(|&n| n as f64).product::<f64>()
                    * (t as f64 * h as f64 * w as f64);
                let unique = (geometry.stage4[0] as f64
                    * geometry.stage4[1] as f64
                    * geometry.stage4[2] as f64)
                    .max(1.0);
                let waste = processed / unique;
                let key = (
                    waste,
                    -(t as i64 * h as i64 * w as i64),
                    counts.iter().map(|&n| n as i64).product::<i64>(),
                    tile,
                );
                let better = match best {
                    None => true,
                    Some(ref b) => (key.0, key.1, key.2) < (b.0, b.1, b.2),
                };
                if better {
                    best = Some(key);
                }
            }
        }
    }

    best.map(|(_, _, _, tile)| DiffVaeTiling { tile, overlap })
        .ok_or(BudgetFailure::SmallestTileExceedsBudget {
            smallest_gib: smallest.unwrap_or(floor) as f64 / GIB,
        })
}

/// Legal tile sizes on one stage-4 axis.
///
/// Floored at `max(min_tile, 2 * overlap)` — upstream floors at `2 × overlap` so a tile's left and
/// right ramps do not multiply into a non-partition-of-unity blend — and stepped on the axis' own
/// last-upsample stride so a tile boundary lands on a whole stage-5 pixel group. The full extent is
/// always a candidate: an axis that need not split should not be forced to.
fn axis_candidates(extent: i32, min_tile: i32, overlap: i32, stride: i32) -> Vec<i32> {
    let step = stride.max(1);
    let floor = min_tile.max(2 * overlap).max(1);
    if extent <= floor {
        return vec![extent.max(1)];
    }
    let mut out: Vec<i32> = (0..)
        .map(|i| floor + i * step)
        .take_while(|&size| size < extent)
        .collect();
    out.push(extent);
    out
}

fn budget_error(
    geometry: &DecodeGeometry,
    safe_gib: f64,
    resolved: &ResolvedDiffVaeMode,
    failure: BudgetFailure,
) -> Error {
    let [f, h, w] = geometry.canvas;
    let mode = resolved.mode.as_str();
    match failure {
        BudgetFailure::FloorExceedsBudget { floor_gib } => Error::Msg(format!(
            "ltx diffvae decode: assembling a {w}x{h}x{f} video needs ~{floor_gib:.1} GB for the \
             stage-4 feature and the pixel accumulators alone, over this machine's ~{safe_gib:.1} \
             GB safe budget (mode {mode}). Reduce the resolution or frame count."
        )),
        BudgetFailure::SmallestTileExceedsBudget { smallest_gib } => Error::Msg(format!(
            "ltx diffvae decode: a {w}x{h}x{f} video peaks at ~{smallest_gib:.1} GB even with the \
             smallest legal tile, over this machine's ~{safe_gib:.1} GB safe budget (mode {mode}). \
             Reduce the resolution or frame count."
        )),
    }
}
