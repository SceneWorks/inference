//! Resident component kinds, adapter-stack residency arithmetic, and peak breakdowns.

use super::*;

/// Typed identity of one resident network component.
///
/// [`TransformerComponent`] keeps its original three meanings exactly; the wrapper lets the same
/// component axis name a network that is not one of the base model's transformers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryComponentKind {
    Transformer(TransformerComponent),
    /// A named sub-stack **inside** one of the base model's transformers: a set of layers or
    /// projections whose bytes are already part of [`MemoryAssetFacts::transformer_bytes`] rather
    /// than a network standing beside it (SC-18665).
    ///
    /// [`Self::Transformer`] cannot express this — it names a whole transformer, so declaring a
    /// projection stack with it would charge the same bytes twice on a provider that also declares
    /// the transformer. It is not auxiliary for the same reason: the bytes are inside the
    /// base-model total already, so `overlay_bytes` must not move for one.
    TransformerSubStack(TransformerComponent),
    ControlBranch,
    AdapterStack,
    IpAdapter,
    IdentityEncoder,
    /// The autoencoder's **encoder** half, materialized only when a request carries a reference
    /// image to encode (img2img / `ConditioningKind::Reference`) — never by a plain text-to-image
    /// render of the same checkpoint (SC-22667).
    ///
    /// It is auxiliary for exactly the reason a control branch is: a request opts into it, so a
    /// fit decision for a request without one must not carry it. [`MemoryAssetFacts::decoder_bytes`]
    /// stays the autoencoder weights the render always holds. A loader whose resident autoencoder
    /// object *does* build both halves at load charges those in `decoder_bytes` and declares here
    /// only the encoder copy it builds lazily on the first reference.
    ReferenceEncoder,
    /// A separately loaded optional caption-rewriting VLM, outside the image model's encoder.
    PromptUpsampler,
}

/// How long one declared component's bytes stay resident within a single render (SC-18665).
///
/// Before this existed a declaration had exactly one meaning — resident from load until the render
/// finishes — and [`Self::WholeRender`] *is* that meaning, spelled out rather than assumed.
///
/// [`Self::PrecomputedThenEvicted`] exists because MiniMax-H3's AdaLN projections are the first
/// component on the ladder that is materialized, consumed to derive something smaller, and then
/// dropped: 26_020_915_200 B of a 66 GB DiT partition, released before the denoise loop's first
/// step. Charging that as resident leaves every estimate for the work that follows wrong by 26 GB
/// in the conservative direction, which suppresses configurations that do in fact fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryComponentResidency {
    /// Resident from load until the render finishes.
    WholeRender,
    /// Materialized at the head of `precomputed_in`, consumed to derive a smaller artifact, and
    /// released before that phase's steady-state work runs. The phase's own peak therefore covers
    /// the full `resident_bytes`, while its steady state — and every later phase — holds only
    /// `retained_bytes`.
    ///
    /// Deliberately **not** a [`MemoryStrategy`]. A rung is something a selection engages or
    /// declines; a provider may perform this drop unconditionally, as MiniMax-H3 does.
    /// [`MemoryResidentComponent::bounded_by`] remains the seam for a rung-selected residency, and
    /// the two are independent.
    PrecomputedThenEvicted {
        /// The phase whose steady state runs without this component. Conformance requires it to be
        /// a declared lifecycle phase, so the drop is ordered against a phase the provider runs
        /// rather than against a stage nobody else can see.
        precomputed_in: MemoryPhase,
        /// Bytes of the derived artifact kept in the component's place. **The evict is not free**;
        /// declaring it as free overstates the win by exactly this much.
        retained_bytes: u64,
        /// The executable measurement or loader assertion that established the drop. Required for
        /// the same reason [`MemoryStrategyEngagementExclusion::evidence`] is: this removes bytes
        /// from an estimate, and an unevidenced removal is how an estimate stops being
        /// conservative. Its **truth** cannot be checked here, only its presence.
        evidence: String,
    },
}

/// Whether adapter factors remain independently resident after a provider finishes loading.
///
/// Dense providers may fold the factors into their base weights, while packed quantized providers
/// commonly preserve them as forward-time residuals because the packed base is immutable. Keeping
/// this distinction typed prevents a fit gate from charging every adapter file as resident bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterResidencyMode {
    Folded,
    Additive,
}

/// Load-exact resident bytes for a single-stack adapter installation that keeps each factor at its
/// **file** dtype: the stack is priced at its safetensors file lengths. An installer that upcasts
/// the factors (candle's) is priced by [`adapter_stack_upcast_resident_bytes`] instead.
///
/// `Some(0)` is positive evidence that factors are folded and therefore add no independent
/// residency. `None` means an additive stack was requested but at least one source could not be
/// sized; callers must fail closed instead of guessing. Multi-expert providers should account for
/// shared adapters once per simultaneously resident expert rather than calling this helper once for
/// the whole model.
pub fn adapter_stack_resident_bytes(
    adapters: &[AdapterSpec],
    mode: AdapterResidencyMode,
) -> Option<u64> {
    if adapters.is_empty() || mode == AdapterResidencyMode::Folded {
        return Some(0);
    }
    adapters.iter().try_fold(0_u64, |total, adapter| {
        let bytes = safetensors_path_bytes(&adapter.path);
        (bytes > 0).then(|| total.saturating_add(bytes))
    })
}

/// Width of the f32 every float factor is upcast to by an [`adapter_stack_upcast_resident_bytes`]
/// installer.
const UPCAST_FACTOR_WIDTH: u64 = 4;

/// Whether an upcasting installer's LoRA residual keeps a compute-dtype copy of its f32 factors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpcastLoraCopy {
    /// The first forward at a narrower compute dtype caches a copy for the life of the adapter
    /// (`candle_gen::quant::AdaptLinear`'s `PreparedLora`).
    Cached,
    /// Each forward casts the factors and drops the cast (`candle_gen::train::lora::LoraLinear`'s
    /// additive residual, the SDXL packed host). Only the f32 factors stay resident.
    PerForward,
}

/// The Kronecker dimensions `w1 [a, c]` ⊗ `w2 [b, d]` of one LoKr module (sc-24163).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LokrKroneckerDims {
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub d: u64,
}

impl LokrKroneckerDims {
    /// Read the dimensions off a module's factor shapes. `factor` maps a factor name (`lokr_w1`,
    /// `lokr_w1_a`, `lokr_w1_b`, `lokr_w2`, `lokr_w2_a`, `lokr_w2_b`, `lokr_t2`) to its shape. The
    /// forms are a full `w1 [a, c]` or a low-rank `w1_a [a, r]`·`w1_b [r, c]`, and a full
    /// `w2 [b, d]`, a low-rank `w2_a [b, r]`·`w2_b [r, d]`, or a tucker `w2_a [r, b]` /
    /// `w2_b [r, d]` beside `lokr_t2`. `None` when a factor the module needs is missing or not 2-D.
    pub fn from_factor_shapes<'a>(factor: impl Fn(&str) -> Option<&'a [usize]>) -> Option<Self> {
        let dim = |name: &str, axis: usize| -> Option<u64> {
            let shape = factor(name)?;
            (shape.len() == 2).then(|| shape[axis] as u64)
        };
        let (a, c) = match factor("lokr_w1") {
            Some(_) => (dim("lokr_w1", 0)?, dim("lokr_w1", 1)?),
            None => (dim("lokr_w1_a", 0)?, dim("lokr_w1_b", 1)?),
        };
        let (b, d) = match (factor("lokr_w2"), factor("lokr_t2")) {
            (Some(_), _) => (dim("lokr_w2", 0)?, dim("lokr_w2", 1)?),
            (None, Some(_)) => (dim("lokr_w2_a", 1)?, dim("lokr_w2_b", 1)?),
            (None, None) => (dim("lokr_w2_a", 0)?, dim("lokr_w2_b", 1)?),
        };
        Some(Self { a, b, c, d })
    }

    /// `a·c + b·d`: the elements of the two Kronecker factors the structured residual keeps.
    pub fn factor_elements(&self) -> u64 {
        self.a
            .saturating_mul(self.c)
            .saturating_add(self.b.saturating_mul(self.d))
    }

    /// Device bytes `candle_gen::quant::LokrFactors` holds for this module once a forward has run at
    /// `compute_width`. Both factors are materialized in f32 (a low-rank or tucker leg is multiplied
    /// out to its full `[b, d]`, so a file's stored elements can be far fewer). The forward then
    /// caches `w1` at the compute width, which is an `Arc` clone at f32, and `w2ᵀ`, which
    /// `contiguous` copies at any width. So the cost is `(a·c + b·d)·4`, plus `a·c·w` when `w ≠ 4`,
    /// plus `b·d·w`.
    pub fn resident_bytes(&self, compute_width: u64) -> u64 {
        let w1 = self.a.saturating_mul(self.c);
        let w2 = self.b.saturating_mul(self.d);
        let w1_copy = if compute_width == UPCAST_FACTOR_WIDTH {
            0
        } else {
            w1.saturating_mul(compute_width)
        };
        self.factor_elements()
            .saturating_mul(UPCAST_FACTOR_WIDTH)
            .saturating_add(w1_copy)
            .saturating_add(w2.saturating_mul(compute_width))
    }
}

/// The LoKr factor suffixes a module groups by, longest first so `.lokr_w1_a` is not read as
/// `.lokr_w1`.
const LOKR_FACTOR_SUFFIXES: [&str; 7] = [
    ".lokr_w1_a",
    ".lokr_w1_b",
    ".lokr_w2_a",
    ".lokr_w2_b",
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_t2",
];

/// Split a LoKr factor tensor name into `(module, factor)`, e.g. `("m", "lokr_w2_a")`.
fn lokr_factor(name: &str) -> Option<(&str, &'static str)> {
    LOKR_FACTOR_SUFFIXES.iter().find_map(|suffix| {
        name.strip_suffix(suffix)
            .map(|module| (module, &suffix[1..]))
    })
}

/// Load-exact resident bytes for an **additive** stack whose installer upcasts every factor to f32
/// on device (sc-24163).
///
/// That is the shared candle installer (`candle_gen::quant::install_dotted_adapters`) and the
/// provider installers built the same way. Each factor is read to the host, cast `to_dtype(F32)`
/// and uploaded, so on device it holds 4 bytes per element whatever the file stores.
///
/// * **LoRA** (any file without LoKr keys): every float element costs 4 bytes, plus `compute_width`
///   when the residual caches a narrower compute copy ([`UpcastLoraCopy::Cached`]; an f32 compute
///   dtype shares the factors' storage). A non-float tensor costs its stored bytes.
/// * **LoKr**: each module is priced structurally ([`LokrKroneckerDims::resident_bytes`]), because
///   the installer materializes a low-rank `w2_a·w2_b` to its full `[b, d]`. Pricing the stored
///   elements would under-price a low-rank module many times over. Non-factor tensors (`alpha`)
///   cost their stored bytes.
///
/// [`adapter_stack_resident_bytes`] prices the file length instead. That is right for an installer
/// that keeps the file dtype and wrong for this one. Measured on CUDA (sc-24163, Qwen-Image 2.1 at
/// 768², `CU_MEMPOOL_ATTR_USED_MEM_HIGH`, one process per cell): a 167.9 MB f32 LoRA and an
/// 83.9 MB bf16 LoRA with the same factor count both added 167.8 MB right after load and 251.7 MB
/// at the render peak. That is 6 bytes per element at bf16 compute; the file length priced them at
/// 167.9 MB and 83.9 MB.
///
/// Fails closed like [`adapter_stack_resident_bytes`]: `None` when a file has no safetensors
/// residency, its header cannot be read, it holds no tensor bytes, or a LoKr module's dimensions
/// cannot be read. A `Folded` stack is `Some(0)`.
pub fn adapter_stack_upcast_resident_bytes(
    adapters: &[AdapterSpec],
    mode: AdapterResidencyMode,
    compute_width: u64,
    lora_copy: UpcastLoraCopy,
) -> Option<u64> {
    if adapters.is_empty() || mode == AdapterResidencyMode::Folded {
        return Some(0);
    }
    let lora_copy_width = match lora_copy {
        UpcastLoraCopy::Cached if compute_width != UPCAST_FACTOR_WIDTH => compute_width,
        _ => 0,
    };
    let lora_per_element = UPCAST_FACTOR_WIDTH.saturating_add(lora_copy_width);
    adapters.iter().try_fold(0_u64, |total, adapter| {
        if safetensors_path_bytes(&adapter.path) == 0 {
            return None;
        }
        let headers = crate::weightsmeta::safetensors_path_tensor_headers(&adapter.path).ok()?;
        let is_lokr = headers.iter().any(|h| lokr_factor(&h.name).is_some());
        let mut bytes = 0_u64;
        let mut modules: std::collections::BTreeMap<&str, Vec<(&str, &[usize])>> =
            std::collections::BTreeMap::new();
        for header in &headers {
            match (is_lokr, lokr_factor(&header.name)) {
                (true, Some((module, factor))) => modules
                    .entry(module)
                    .or_default()
                    .push((factor, header.shape.as_slice())),
                (true, None) => bytes = bytes.saturating_add(header.data_bytes),
                (false, _) if header.is_float() => {
                    let elements = header.element_count().ok()?;
                    bytes = bytes.saturating_add(elements.saturating_mul(lora_per_element));
                }
                (false, _) => bytes = bytes.saturating_add(header.data_bytes),
            }
        }
        for factors in modules.values() {
            let dims = LokrKroneckerDims::from_factor_shapes(|name| {
                factors.iter().find(|(f, _)| *f == name).map(|(_, s)| *s)
            })?;
            bytes = bytes.saturating_add(dims.resident_bytes(compute_width));
        }
        (bytes > 0).then(|| total.saturating_add(bytes))
    })
}

/// Exact ordered identity of the load-time adapter stack used by provider request handshakes.
///
/// The digest binds every adapter's native path representation, LoRA/LoKr kind, and exact IEEE-754
/// scale bits. It deliberately does not inspect mutable file contents: callers that require byte
/// identity must prepare/pin those files separately before constructing the [`AdapterSpec`].
pub fn adapter_stack_identity(adapters: &[AdapterSpec]) -> Option<String> {
    if adapters.is_empty() {
        return None;
    }
    let mut digest = Sha256::new();
    digest.update(b"gen-core-adapter-stack-v1");
    for adapter in adapters {
        let path = format!("{:?}", adapter.path.as_os_str());
        digest.update((path.len() as u64).to_le_bytes());
        digest.update(path.as_bytes());
        digest.update([match adapter.kind {
            crate::AdapterKind::Lora => 0,
            crate::AdapterKind::Lokr => 1,
        }]);
        digest.update(adapter.scale.to_bits().to_le_bytes());
    }
    Some(format!("adapters:{:x}", digest.finalize()))
}

impl MemoryComponentKind {
    pub const fn is_auxiliary(self) -> bool {
        matches!(
            self,
            Self::ControlBranch
                | Self::AdapterStack
                | Self::IpAdapter
                | Self::IdentityEncoder
                | Self::ReferenceEncoder
                | Self::PromptUpsampler
        )
    }
}

/// Load-exact resident bytes for one named component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryResidentComponent {
    /// Stable provider-local identity. The kind is intentionally not the identity: MultiControlNet
    /// may hold several distinct [`MemoryComponentKind::ControlBranch`] components at once.
    pub id: String,
    pub kind: MemoryComponentKind,
    /// Bytes at this component's **widest** instant: what it costs while fully materialized.
    pub resident_bytes: u64,
    /// The rung that bounds this component's residency, if any. `None` means no *rung* bounds it —
    /// which is not the same as staying resident, since [`Self::residency`] may declare an
    /// unconditional drop that no selection can decline (SC-18665).
    pub bounded_by: Option<MemoryStrategy>,
    /// How long those bytes actually stay (SC-18665).
    ///
    /// Required rather than defaulted, following the SC-16090 precedent on
    /// [`MemoryBackendRealization::CandleCuda::block_materialization`]: a default is an opt-out
    /// that reads as a declaration, and every existing site states
    /// [`MemoryComponentResidency::WholeRender`] explicitly, which is byte-for-byte the meaning it
    /// already had.
    pub residency: MemoryComponentResidency,
}

impl MemoryResidentComponent {
    /// Bytes this component still holds once its phase reaches steady state.
    pub fn steady_state_bytes(&self) -> u64 {
        match &self.residency {
            MemoryComponentResidency::WholeRender => self.resident_bytes,
            MemoryComponentResidency::PrecomputedThenEvicted { retained_bytes, .. } => {
                (*retained_bytes).min(self.resident_bytes)
            }
        }
    }

    /// Bytes it drops getting there — **zero** for every [`MemoryComponentResidency::WholeRender`]
    /// declaration, which is why adopting this field cannot move an existing provider's arithmetic.
    pub fn evicted_bytes(&self) -> u64 {
        self.resident_bytes
            .saturating_sub(self.steady_state_bytes())
    }
}

/// A scalar predicted peak with the resident auxiliary contributions exposed separately.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryPeakBreakdown {
    pub(super) predicted_peak_bytes: u64,
    /// The part of the peak not attributable to a separately declared auxiliary network. This
    /// includes base-model residency and calibrated transients; the contract does not invent a split
    /// that the provider did not declare.
    pub unattributed_bytes: u64,
    pub components: Vec<MemoryResidentComponent>,
}

impl MemoryPeakBreakdown {
    pub const fn from_unattributed(predicted_peak_bytes: u64) -> Self {
        Self {
            predicted_peak_bytes,
            unattributed_bytes: predicted_peak_bytes,
            components: Vec::new(),
        }
    }

    pub const fn predicted_peak_bytes(&self) -> u64 {
        self.predicted_peak_bytes
    }
}
