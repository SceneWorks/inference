//! Quantization, precision, offload, and load-shape knobs.

/// Quantization tier a load may request. [`Q4`](Self::Q4)/[`Q8`](Self::Q8) are the group-wise
/// affine int tiers; [`Nvfp4`](Self::Nvfp4) is the NVFP4 FP4 tensor-core tier
/// (epic 11037).
///
/// **A quant tier is a creative choice — a distinct, additive tier, never a silent numerics swap
/// (epic 11037 SC#5).** [`Nvfp4`](Self::Nvfp4) was added under the **Option A** packaging decision of
/// sc-11042: NVFP4 is exposed as its *own* user-selectable tier, **not** a Blackwell execution backend
/// auto-substituted for [`Q4`](Self::Q4). NVFP4's numerics differ from int4-affine `q4` (E2M1 4-bit
/// elements + FP8-E4M3 block scales, W4A4 regime), so auto-swapping `q4` → NVFP4 on `sm_120` would
/// silently change a picked tier's output — the SC#5 violation Option A avoids. Adding this variant
/// changes **no** existing tier's numerics or behavior; each of `Q4`/`Q8` maps exactly as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    Q4,
    Q8,
    /// The **NVFP4 FP4** tier (epic 11037, sc-11042 Option A) — E2M1 4-bit elements over 16-element
    /// blocks with FP8-E4M3 micro-scales + an FP32 per-tensor scale (~4.5 effective bits/weight). A
    /// *distinct* creative-choice tier, not an int4-affine equivalent. Served **natively packed** by
    /// candle-gen's `Nvfp4Linear` (the packed-forward path, resident at the NVFP4 footprint — never a
    /// dequant→bf16 dense expansion, epic 11037 SC#6) through the sc-11039 cuBLASLt FP4 GEMM on
    /// consumer Blackwell `sm_120`; on other hardware it falls back cleanly. Surfaced through the
    /// candle-gen catalog only under the `cuda` feature — the MLX/macOS runtime (no FP4 hardware) and
    /// the CPU candle bundle (no FP4 compute) do not offer it.
    Nvfp4,
    /// FLUX3 source rowwise E4M3 W8A8: per-output weight scales, per-token
    /// activation scales and one final BF16 rounding. NOT affine integer Q8.
    /// Providers must explicitly implement/advertise this format; element width
    /// is not permission to dispatch an existing eight-bit affine quantizer.
    Fp8Rowwise,
}

/// YuE2's AR execution mode, separate from the loaded weight tier. Experimental FP8 replaces
/// only AR projections during generation and restores their BF16 originals before acoustic work;
/// it is not a whole-model [`Quant`] tier. Other providers must refuse a non-native mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Yue2ArMode {
    /// Run AR with the loaded weight tier's ordinary projection kernels.
    #[default]
    Native,
    /// Run only the AR projections in experimental E4M3 on CUDA sm_89+ over BF16 weights.
    ExperimentalFp8,
}

/// YuE2's stage compute policy, distinct from its quantized weight tier and from the shared
/// [`Precision`] provider-default sentinel. Legacy preserves already queued loads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Yue2ComputePolicy {
    #[default]
    Legacy,
    /// Explicitly allow the provider's mixed stage dtypes (BF16 MoT, FP32 VAE on a GPU).
    Auto,
    /// Run the MoT and both VAE variants in BF16; unsupported on the CPU.
    Bf16,
    /// Run the MoT and both VAE variants in FP32.
    Fp32,
}

impl Quant {
    /// Element bit-width of the tier. For [`Q4`](Self::Q4)/[`Q8`](Self::Q8) this is the width passed to
    /// the MLX affine quantizer. [`Nvfp4`](Self::Nvfp4) reports `4` (its E2M1 elements are 4-bit) but is
    /// **not** an MLX-quantizer tier — it carries FP8 block scales + an FP32 per-tensor scale
    /// (~4.5 *effective* bits/weight) and is served by candle-gen's NVFP4 packed path, not the MLX
    /// affine quantizer; do not route `Nvfp4` through an MLX `quantize(bits)` call on this width alone.
    /// The same rule applies to `Fp8Rowwise` (8-bit E4M3, not Q8). Affine loaders
    /// must use [`Self::affine_bits`] rather than dispatching on this metadata.
    pub fn bits(self) -> i32 {
        match self {
            Quant::Q4 => 4,
            Quant::Q8 | Quant::Fp8Rowwise => 8,
            Quant::Nvfp4 => 4,
        }
    }

    /// Exact wire/cache tier identity. Equal element widths never alias formats.
    pub const fn key(self) -> &'static str {
        match self {
            Self::Q4 => "q4",
            Self::Q8 => "q8",
            Self::Nvfp4 => "nvfp4",
            Self::Fp8Rowwise => "fp8r",
        }
    }

    /// Tier label of an optional quant: `None` (the dense tier) is `"bf16"`, anything else is its
    /// [`Self::key`] (sc-27254: the providers' shared `None => "bf16", Some(Q4) => "q4", …` arms).
    pub const fn tier_label(quant: Option<Quant>) -> &'static str {
        match quant {
            None => "bf16",
            Some(quant) => quant.key(),
        }
    }

    /// [`Self::tier_label`] restricted to the dense and affine tiers (`"bf16"`, `"q4"`, `"q8"`):
    /// `None` for a packed floating-point tier, which a provider shipping only those tiers cannot name.
    pub const fn affine_tier_label(quant: Option<Quant>) -> Option<&'static str> {
        match quant {
            None | Some(Self::Q4 | Self::Q8) => Some(Self::tier_label(quant)),
            Some(Self::Nvfp4 | Self::Fp8Rowwise) => None,
        }
    }

    pub fn from_key(key: &str) -> crate::Result<Self> {
        match key {
            "q4" => Ok(Self::Q4),
            "q8" => Ok(Self::Q8),
            "nvfp4" => Ok(Self::Nvfp4),
            "fp8r" => Ok(Self::Fp8Rowwise),
            _ => Err(crate::Error::Unsupported(format!(
                "unknown quantization format {key:?}; expected q4, q8, nvfp4 or fp8r"
            ))),
        }
    }

    /// Width accepted by an AFFINE integer quantizer. Packed floating-point
    /// formats must not reach that quantizer just because their width is 4/8.
    pub fn affine_bits(self) -> crate::Result<i32> {
        match self {
            Self::Q4 => Ok(4),
            Self::Q8 => Ok(8),
            _ => Err(crate::Error::Unsupported(format!(
                "{} is not an affine integer quantization format",
                self.key()
            ))),
        }
    }
}

/// Compute precision for dense (non-quantized) weights.
///
/// [`Bf16`](Self::Bf16) doubles as the registry's **"dense default / no precision override"
/// sentinel**, not a literal request for bf16 tensors: each provider maps it to its own native
/// dense dtype. Most providers do run bf16 under it (e.g. sensenova), but the SDXL-family loaders
/// (kolors, instantid) run **fp16** — they still gate on `Bf16` and reject `Fp32` because a
/// precision override is not wired, then load at `Dtype::Float16`. So an audit of dtype behavior
/// through `LoadSpec` must read `Bf16` as "the provider's default dense dtype", which is not
/// universally bf16. (A distinct `Precision::Default`/`Dense` sentinel would make this explicit but
/// would touch every provider's match arm — deferred; this note is the documented contract.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// Dense default — the provider's native dense dtype (bf16 for most, fp16 for the SDXL family).
    /// See the type-level note: this is the "no override" sentinel, not a literal bf16 request.
    #[default]
    Bf16,
    /// Full-precision override, honored only by providers that wire it (others reject it at `load`).
    Fp32,
}

/// Component-residency strategy for a load (epic 10765 Phase 1, sc-10769/sc-10821). The default keeps
/// every model component resident for the whole generation (fast, cross-request cached). `Sequential`
/// asks a provider that supports it to load→use→DROP each heavy component in phase order (text encoder →
/// transformer/UNet → VAE) so peak VRAM is bounded to the largest single working set instead of the sum,
/// letting a small card run a model that would OOM resident — at the cost of the cross-request weight
/// cache. Advisory, never an error: a provider that has not wired it treats `Sequential` as `Resident`.
/// Whether a given engine actually honors it is not FLUX/backend-specific — it is advertised per model
/// via [`Capabilities::supports_sequential_offload`](crate::generator::Capabilities::supports_sequential_offload),
/// which a consumer reads to tell "bounds peak memory here" from "no-op fallback" (sc-11126).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OffloadPolicy {
    /// All components co-resident for the whole generation (today's behavior). Fast; keeps the cache.
    #[default]
    Resident,
    /// Load→use→drop each heavy component in phase order to minimize peak VRAM. Advisory: a provider
    /// that has not wired it falls back to `Resident`.
    Sequential,
}

/// How model weights are materialized within an already-loaded generator (SC-15998).
///
/// This is intentionally independent from [`OffloadPolicy`]:
///
/// - [`OffloadPolicy`] controls **inter-phase residency** — whether whole components are released
///   between conditioning, denoise, and decode.
/// - [`LoadShape`] controls **intra-phase materialization** — whether a transformer keeps a bulk
///   resident stack or re-opens its blocks through a deferred schedule.
///
/// A caller can therefore request a resident, cross-request-cached generator whose mmap-backed
/// transformer blocks remain deferred, without also asking the provider to unload whole components
/// at phase boundaries. Providers that do not implement the requested shape may reject it or
/// advertise the corresponding memory strategy as unavailable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LoadShape {
    /// The historical fast path: components may bulk-materialize their complete weight stacks and
    /// retain them across requests.
    #[default]
    EagerMaterialization,
    /// Keep eligible mmap/re-openable transformer weights deferred and materialize them through a
    /// block schedule. The schedule may be all-covering when no bound is selected; this says nothing
    /// about phase-level component release.
    DeferredMaterialization,
}

/// Authoritative result of a caller-owned declaration check for [`LoadShape`].
///
/// This is deliberately separate from [`LoadShape`] itself. A declaration-aware caller may refuse
/// deferred materialization and keep the historical eager shape; carrying that refusal prevents a
/// later legacy shaper from treating the eager value as "not evaluated yet" and re-admitting the
/// same route. Providers continue to consume [`LoadSpec::load_shape`](crate::LoadSpec::load_shape);
/// orchestration layers own and propagate this authority marker across cache and cold-load
/// boundaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LoadShapeDeclarationResult {
    /// No authoritative declaration check has run. Legacy shaping may still apply.
    #[default]
    NotEvaluated,
    /// The declaration and provider accepted deferred materialization, but the required residency
    /// mode has not yet been selected. This marker is intentionally unbound and non-authorizing:
    /// the current load shape remains eager, generic shapers must ignore it, and only a fresh exact
    /// declaration evaluation may transition the load to [`Self::Applied`].
    Eligible,
    /// The declaration check admitted and applied its selected load shape.
    Applied,
    /// The declaration check authoritatively refused its selected load shape. Legacy shaping must
    /// not override this result.
    Refused,
}
