//! sc-18799 — **memory-budgeted DiffVAE decode selection** (candle twin of
//! `mlx-gen-ltx/src/diff_vae/budget.rs`).
//!
//! [`super::NaDiffusionDecoder::decode_tiled`] shipped in sc-18767 with real neighbourhood-window
//! tiling arithmetic, but nothing chose a tile for it. The conv decoder has had a budgeted auto
//! selector since sc-7076 ([`crate::vae::auto_tiling_budgeted_ltx`]); this is the DiffVAE's, in the
//! same shape:
//!
//! ```text
//! auto_*  → resolve the safe VRAM budget → pure planner
//!         → Ok(None)      the single-pass decode already fits
//!         → Ok(Some(cfg)) this tile fits
//!         → Err(..)       catchable, *before* the decode, instead of an OOM
//! ```
//!
//! What differs from the conv selector is the axis count: the conv one picks a **spatial edge**
//! plus a temporal `(tile, overlap)` pair off two fixed lists, while the DiffVAE tiles three axes
//! *independently* — [`DiffVaeTiling::tile`] is `[T, H, W]` in stage-4-input cells — so the planner
//! enumerates a per-axis size grid and scores the product, which is what upstream's own
//! `recommended_decode_tiling_config` does.
//!
//! ## The four upstream modes
//!
//! `--diffvae-optimization` has four presets ([`DiffVaeMode`]); each resolves to an install recipe
//! whose memory-relevant content is a stage-5 working-set coefficient and a budget-safety withhold.
//! Both are declared as data and then put through upstream's own host resolve
//! ([`DiffVaeMode::resolve_for_host`]) before a plan is costed, because upstream's `stage5_mem_coef`
//! resolves the host first too — and on a host without NATTEN it downgrades the chunked
//! coefficients and refuses `combined_compile` outright.
//!
//! This port ships exactly one neighbourhood-attention kernel: [`super::na3d`], the eager tiled
//! SDPA (upstream `NAttentionKind::EAGER_SDPA`). There is no NATTEN binding and no Triton `na3d`,
//! and the fused CuTe DSL kernel needs a **datacenter** Blackwell GPU, which
//! [`HostNaSupport::detect`] probes for on the bound device.
//!
//! | upstream mode | resolves to | here |
//! | --- | --- | --- |
//! | `chunked_eager` | eager SDPA, coef 5, safety 1 GiB | **runs** |
//! | `chunked_compile` | eager SDPA, coef 5, safety 1 GiB (upstream's own no-NATTEN remap) | **runs** |
//! | `combined_compile` | — | **refused**: upstream raises without NATTEN |
//! | `blackwell_dsl` | — | **refused** unless the bound CUDA device is `sm_10x` |
//!
//! Reference: `Lightricks/LTX-2` `d151147788a9284cca791edc6ce898007e727fe6`,
//! `packages/ltx-core/src/ltx_core/model/video_vae/diffusion_tiling.py` and
//! `.../video_vae/transformer/{config,apply}.py`.

use candle_gen::candle_core::{Device, Error, Result};

use super::{split_by_size, DiffVaeTiling, NaDiffusionDecoderConfig};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---------------------------------------------------------------------------------------------
// The four upstream modes
// ---------------------------------------------------------------------------------------------

/// Upstream's user-facing DiffVAE decode presets (`DiffVAEMode`, `--diffvae-optimization`).
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
    pub const ALL_MODES: [Self; 4] = [
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
        Self::ALL_MODES
            .into_iter()
            .find(|m| m.as_str() == s)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "ltx diffvae: unknown diffvae optimization {s:?} (expected one of {})",
                    Self::ALL_MODES.map(Self::as_str).join(" | ")
                ))
            })
    }

    /// The stage-5 working-set multiplicity upstream declares for this preset, **before** the host
    /// resolve (upstream `_MEM_COEF_BY_MODE`).
    pub fn declared_stage5_coef(self) -> f64 {
        match self {
            Self::ChunkedEager => COEF_CHUNKED_EAGER,
            Self::ChunkedCompile => COEF_CHUNKED_COMPILE,
            Self::CombinedCompile => COEF_COMBINED_COMPILE,
            Self::BlackwellDsl => COEF_BLACKWELL_DSL,
        }
    }

    /// Bytes upstream withholds from the recommend budget for this preset, before the host resolve
    /// (upstream `_BUDGET_SAFETY_BYTES_EAGER` / `_COMPILED`).
    pub fn declared_budget_safety_bytes(self) -> u64 {
        match self {
            Self::ChunkedEager => SAFETY_BYTES_EAGER,
            Self::ChunkedCompile | Self::CombinedCompile | Self::BlackwellDsl => {
                SAFETY_BYTES_COMPILED
            }
        }
    }

    /// The neighbourhood-attention kernel this preset asks for, before the host resolve.
    pub fn declared_attention(self) -> NaKind {
        match self {
            Self::ChunkedEager | Self::ChunkedCompile | Self::CombinedCompile => NaKind::Natten,
            Self::BlackwellDsl => NaKind::BlackwellDsl,
        }
    }

    /// Upstream's `resolve_attention_for_host` + `stage5_mem_coef` + `budget_safety_bytes`, in one
    /// step, against a host that declares which kernels it can actually serve.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostNaSupport {
    /// CUTLASS-FNA through the `natten` package.
    pub natten: bool,
    /// Upstream's Triton `na3d`.
    pub triton: bool,
    /// The fused CuTe DSL kernel on a datacenter Blackwell GPU.
    pub blackwell_dsl: bool,
    /// Why `blackwell_dsl` is what it is — quoted into the refusal so it names the real reason.
    pub blackwell_reason: &'static str,
}

/// The compute-capability **major** of *datacenter* Blackwell — B200/GB200 `sm_100` and B300
/// `sm_103`. Consumer Blackwell is `sm_120`, i.e. major 12, and is deliberately **not** this: the
/// CuTe DSL path is documented upstream as "not used on consumer Blackwell (sm_120)", and a `>=`
/// floor would accept 12.0 and run a kernel that is not validated there — the same shape as the
/// sm_120 hazard `candle_gen::quant`'s NVFP4 gate guards.
pub const DATACENTER_BLACKWELL_CC_MAJOR: i32 = 10;

/// Whether a `(major, minor)` CUDA compute capability is a datacenter Blackwell part.
pub fn compute_cap_is_datacenter_blackwell(cap: (i32, i32)) -> bool {
    cap.0 == DATACENTER_BLACKWELL_CC_MAJOR
}

impl HostNaSupport {
    /// What this build can serve on `device`.
    ///
    /// `natten` and `triton` are unconditionally `false`: both are PyTorch extensions with no Rust
    /// binding here, so [`super::na3d`] — upstream's eager tiled SDPA — is what every resolved mode
    /// runs on. `blackwell_dsl` reads the bound device's compute capability and applies
    /// [`compute_cap_is_datacenter_blackwell`]; the threshold is not re-derived anywhere else.
    pub fn detect(device: &Device) -> Self {
        let (blackwell_dsl, blackwell_reason) = detect_datacenter_blackwell(device);
        Self {
            natten: false,
            triton: false,
            blackwell_dsl,
            blackwell_reason,
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

/// The hardware gate. Fail-closed on every path that cannot *prove* a datacenter Blackwell part:
/// a non-CUDA device, a driver query that errors, and consumer Blackwell all answer `false` with a
/// reason naming what was actually found.
#[cfg(feature = "cuda")]
fn detect_datacenter_blackwell(device: &Device) -> (bool, &'static str) {
    use candle_gen::candle_core::cuda::cudarc::driver::sys::CUdevice_attribute as Attr;
    let Device::Cuda(cuda) = device else {
        return (false, "the bound device is not a CUDA device");
    };
    let stream = cuda.cuda_stream();
    let ctx = stream.context();
    let (Ok(major), Ok(minor)) = (
        ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
    ) else {
        return (
            false,
            "the CUDA driver would not report the device's compute capability",
        );
    };
    if compute_cap_is_datacenter_blackwell((major, minor)) {
        return (true, "the bound device is a datacenter Blackwell part");
    }
    if major == 12 {
        // The scar, spelled out: sm_120 *is* Blackwell, just not the one the fused kernel targets.
        return (
            false,
            "the bound device is consumer Blackwell (sm_120), which upstream explicitly excludes \
             from the CuTe DSL path",
        );
    }
    (
        false,
        "the bound CUDA device is below the datacenter-Blackwell (sm_10x) line",
    )
}

#[cfg(not(feature = "cuda"))]
fn detect_datacenter_blackwell(_device: &Device) -> (bool, &'static str) {
    (
        false,
        "this build has no CUDA feature, so no datacenter Blackwell device can be bound",
    )
}

// ---------------------------------------------------------------------------------------------
// The cost model
// ---------------------------------------------------------------------------------------------

/// Bytes per element everywhere in this decoder: it runs f32 end to end
/// ([`super::NaDiffusionDecoder::decode_tiled`] casts the latent and the noise before stage 1).
const ELEMENT_BYTES: u64 = 4;

/// How many single-channel full-canvas f32 buffers [`super::NaDiffusionDecoder::decode_tiled`] has
/// live at its peak.
///
/// Counted off the driver, which is structurally the same as the MLX twin's: inside the tile loop
/// the live set is `accumulator` + `placed` + the `acc + placed` result, each
/// `[1, out_channels, F, H, W]`, so `3 × out_channels` channel-canvases; afterwards it is
/// `accumulator` + `weight_3d` (1 channel) + `blended`, which is smaller. The `+ 1` carries the
/// weight canvas so the constant bounds both phases.
const ACCUM_LIVE_CHANNEL_CANVASES: u64 = 3 * 3 + 1;

/// Bytes of concurrent decoder working set per `(stage-5 token × stage-5 channel × coefficient
/// unit)`.
///
/// **PROVISIONAL — cuda-gated.** This is the MLX-fit unit (17) scaled by 2 for the candle/CUDA
/// allocator, because this repo has measured that exact jump once already for the same model
/// family: the conv LTX VAE's MLX-fit per-voxel constants under-predicted the CUDA peak by ~1.9x
/// and were re-fit from 40/300 to 80/620 (`candle-gen-ltx/src/vae.rs`, sc-7148). Scaling up is the
/// fail-closed direction — an over-predicting model tiles more than it must, an under-predicting
/// one OOMs. Re-fit from `tests/ltx_2_5_diffvae_budget_sweep.rs` on the first CUDA host that can
/// run it; that sweep asserts the model never under-predicts, so a wrong constant here is a red
/// test rather than a silent OOM.
const STAGE5_BYTES_PER_TOKEN_CHANNEL_COEF_UNIT: f64 = 34.0;

/// Fixed floor (bytes): resident decoder weights (~0.83 GB) plus the CUDA context and allocator
/// working set, paid whatever the geometry.
///
/// **PROVISIONAL — cuda-gated**, on the same footing as
/// [`STAGE5_BYTES_PER_TOKEN_CHANNEL_COEF_UNIT`]: the MLX-fit 2.4 GB floor plus the ~0.5 GB a CUDA
/// context costs over an MLX one (`candle-gen-ltx/src/vae.rs` fits the conv decoder's CUDA baseline
/// at ~2.2 GiB with no resident DiffVAE weights at all). Re-fit on the first CUDA host.
const DIFFVAE_FIXED_BYTES: u64 = 3_000_000_000;

/// What a decode of a given latent will do.
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
    pub stage4: [usize; 3],
    /// Full stage-5 pixel canvas `[F, H, W]` the accumulators span.
    pub canvas: [usize; 3],
    /// Channels of the resident stages-1-3 output.
    pub stage4_channels: usize,
    /// Stage-5 feature width.
    pub stage5_channels: usize,
    /// Pixel channels of the decoded video.
    pub out_channels: usize,
    /// Last upsample hop's per-axis stride — stage-4 cells to stage-5 tokens.
    pub last_stride: [usize; 3],
}

impl DecodeGeometry {
    /// The geometry `decode` / `decode_tiled` will actually run at for this latent — the latent
    /// size floor applied first, exactly as those two do.
    pub fn new(
        cfg: &NaDiffusionDecoderConfig,
        latent_t: usize,
        latent_h: usize,
        latent_w: usize,
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
            stage4_channels: cfg.stage_channels[last],
            stage5_channels: cfg.stage5_width(),
            out_channels: cfg.out_channels,
            last_stride: cfg.upsamples[last].0,
        }
    }

    /// Stage-5 tokens a stage-4 tile expands to, pre-unpatchify. The causal duplicate-frame drop is
    /// deliberately not subtracted — one frame less is one frame of headroom.
    fn stage5_tokens(&self, tile: [usize; 3]) -> u64 {
        (0..3)
            .map(|axis| (tile[axis] as u64) * (self.last_stride[axis] as u64))
            .product()
    }

    /// The resident stages-1-3 output.
    fn feature_bytes(&self) -> u64 {
        self.stage4[0] as u64
            * self.stage4[1] as u64
            * self.stage4[2] as u64
            * self.stage4_channels as u64
            * ELEMENT_BYTES
    }

    /// One single-channel full stage-5 canvas in f32.
    fn channel_canvas_bytes(&self) -> u64 {
        self.canvas[0] as u64 * self.canvas[1] as u64 * self.canvas[2] as u64 * ELEMENT_BYTES
    }
}

/// Estimated concurrent peak (bytes) of a DiffVAE decode. Pure — no device, no global state.
pub fn estimated_diffvae_decode_peak_bytes(
    geometry: &DecodeGeometry,
    plan: DecodePlan,
    resolved: &ResolvedDiffVaeMode,
) -> u64 {
    let (tile, canvases) = match plan {
        DecodePlan::SinglePass => (geometry.stage4, geometry.out_channels as u64),
        DecodePlan::Tiled(t) => (t.tile, ACCUM_LIVE_CHANNEL_CANVASES),
    };
    let stage5 = geometry.stage5_tokens(tile) as f64
        * geometry.stage5_channels as f64
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

/// Why a geometry could not be planned.
#[derive(Clone, Copy, Debug, PartialEq)]
enum BudgetFailure {
    /// Even the accumulators plus the resident feature exceed the budget — no tile can help.
    FloorExceedsBudget { floor_gib: f64 },
    /// The smallest legal tile still exceeds the budget.
    SmallestTileExceedsBudget { smallest_gib: f64 },
}

/// **Memory-budgeted** tiling for the DiffVAE decode: the DiffVAE's
/// [`crate::vae::auto_tiling_budgeted_ltx`].
///
/// `Ok(None)` → the single-pass decode already fits. `Ok(Some(t))` → run `decode_tiled` with `t`.
/// `Err` → a catchable over-budget signal returned *before* the decode rather than an OOM inside
/// it. The budget comes from [`crate::vae::ltx_vae_safe_budget_gib`], the same resolver the conv
/// selector uses, so both decoders on one machine agree about what "safe" means.
pub fn auto_diffvae_tiling_budgeted_ltx(
    cfg: &NaDiffusionDecoderConfig,
    device: &Device,
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
    mode: DiffVaeMode,
) -> Result<Option<DiffVaeTiling>> {
    let resolved = mode.resolve_for_host(HostNaSupport::detect(device))?;
    plan_diffvae_tiling(
        cfg,
        latent_t,
        latent_h,
        latent_w,
        crate::vae::ltx_vae_safe_budget_gib(),
        &resolved,
    )
}

/// The pure planner behind [`auto_diffvae_tiling_budgeted_ltx`] — the `safe_gib` ceiling and the
/// resolved mode are injected so the selection is testable without a device.
pub fn plan_diffvae_tiling(
    cfg: &NaDiffusionDecoderConfig,
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
    safe_gib: f64,
    resolved: &ResolvedDiffVaeMode,
) -> Result<Option<DiffVaeTiling>> {
    let geometry = DecodeGeometry::new(cfg, latent_t, latent_h, latent_w);
    let usable = ((safe_gib * GIB) as u64).saturating_sub(resolved.budget_safety_bytes);

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
    min_tile: [usize; 3],
    overlap: [usize; 3],
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

    let candidates: Vec<Vec<usize>> = (0..3)
        .map(|axis| {
            axis_candidates(
                geometry.stage4[axis],
                min_tile[axis],
                overlap[axis],
                geometry.last_stride[axis],
            )
        })
        .collect();

    let mut best: Option<(f64, i64, i64, [usize; 3])> = None;
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
                let key = (
                    processed / unique,
                    -((t * h * w) as i64),
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
fn axis_candidates(extent: usize, min_tile: usize, overlap: usize, stride: usize) -> Vec<usize> {
    let step = stride.max(1);
    let floor = min_tile.max(2 * overlap).max(1);
    if extent <= floor {
        return vec![extent.max(1)];
    }
    let mut out: Vec<usize> = (0..)
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
