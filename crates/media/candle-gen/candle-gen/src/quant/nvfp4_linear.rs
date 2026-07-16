//! `Nvfp4Linear` — the NVFP4 (FP4) linear layer (sc-11041, epic 11037).
//!
//! Serves a packed [`Nvfp4Tensor`] weight (E2M1 nibbles + UE4M3 block scales + FP32 per-tensor scale,
//! ~4.5 effective bits/weight — sc-11040) resident in VRAM, forwarding through the sc-11039 cuBLASLt
//! `matmul_nvfp4_staged` on consumer Blackwell `sm_120`. This is the candle-gen linear layer that a
//! provider crate swaps in for an NVFP4 compute tier, the FP4 twin of the `Fp8Linear`/`Int8Linear`
//! layers (`super::eight_bit_linear`, cuda-only).
//!
//! # Mixed-precision policy (spike sc-11038 / sc-7702)
//!
//! Two activation regimes, selected per layer by [`ActPrecision`]:
//!
//! - **W4A4** (the default): both operands FP4. This is the *only* regime that lights up the FP4
//!   tensor cores for the ~2× compute win (SC#1) — the FP4 MMA requires both operands in E2M1, so a
//!   bf16 activation cannot feed it. The weight is staged resident on-device as a packed `DevNvfp4`;
//!   the activation is packed per forward and the GEMM runs on the FP4 cores. **This is the SC#6
//!   packed-forward path — the weight never full-dequants to bf16; resident VRAM stays at the NVFP4
//!   footprint** (`Nvfp4Linear::resident_device_bytes`).
//! - **W4A16** (the per-layer override for the outlier class): FP4 weight × bf16 activation. W4A4
//!   collapses on layers with dense activation outliers (the sc-7702 mechanism: an outlier sharing a
//!   16-block crushes its co-located channels to E2M1 zero), so the spike keeps bf16 activation on the
//!   outlier class — text→DiT `caption_projection`, the cross-attention block, first & last DiT blocks
//!   ([`ActPrecision::for_outlier_layer`]). There is no FP4-weight×bf16-activation tensor-core MMA, so
//!   W4A16 is realized by **dequantizing the FP4 weight to bf16 and running a dense bf16 matmul** —
//!   **no FP4 compute win, and no footprint win either**: the dequantized bf16 weight is materialized
//!   once at construction and stays resident, so a W4A16 layer costs the **full dense bf16 footprint**
//!   in VRAM (`Nvfp4Linear::resident_dequant_bf16_bytes`). It buys *numerical stability* on the outlier
//!   class, nothing else. Reports [`Nvfp4Regime::DequantBf16`].
//!
//! **Read the footprint accounting regime-aware.** `nvfp4_footprint_bytes` is a property of the packed
//! *format* (the host container, present in every regime); `resident_weight_bytes` is a property of the
//! *run*. Only W4A4 makes them equal. A W4A16 run holds dense bf16 — never report it as an NVFP4
//! footprint.
//!
//! # Capability gate + fallback (sc-11041 AC, sc-12078 policy)
//!
//! W4A4 requires the `cuda` feature, a CUDA device, `sm_120`+ (`CublasLt::meets_nvfp4_floor`), a shape
//! the cuBLASLt FP4 path accepts (padded K a multiple of `NVFP4_K_ALIGN`, N a multiple of 16), **and
//! the fused activation quantizer to compile** (`CublasLt::nvfp4_fused_quantizer_available`). When any
//! of these do not hold — a `<sm_120` GPU, a CPU device, a non-cuda build, an ineligible shape, no
//! fused quantizer, or an explicit W4A16 override — the layer **transparently falls back** to the
//! [`Nvfp4Regime::DequantBf16`] dense path (no crash). The non-cuda build compiles this whole module
//! (the FP4 compute leg is cfg-gated); it only ever takes the fallback.
//!
//! **Every gate is settled at construction, and the regime is then fixed for the layer's life.** A
//! forward never re-decides and never silently downgrades: `regime()` is what actually ran. This is
//! why the fused-quantizer probe lives in the gate rather than in `forward_fp4` — a per-forward
//! `Err(_) => unfused` would have made `lights_up_fp4() == true` while serving a regime measured at
//! **0.01× vs bf16**, with no log line and nothing in the reported footprint to show for it. Once the
//! gate passes, a fused-quantizer error is a genuine fault and propagates.
//!
//! # M (token-row) alignment (sc-11039 handoff)
//!
//! sc-11039's `check_nvfp4_alignment` guards K/N but **not** M. cuBLASLt's FP4 block-scaled path can
//! return `NOT_SUPPORTED` for an unaligned free dimension, so the W4A4 forward **pads the token rows**
//! up to a multiple of [`NVFP4_M_ALIGN`] (zero rows, which contribute nothing to any real output) and
//! slices the result back to the logical `M`. Arbitrary token counts therefore run without a runtime
//! `NOT_SUPPORTED`.

use super::nvfp4::Nvfp4Tensor;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

/// Token-row (M) alignment the W4A4 forward pads to before the cuBLASLt FP4 GEMM (sc-11041). The
/// contraction K and output N are aligned by the packer / sc-11039 (`NVFP4_K_ALIGN` = 32, N a multiple
/// of 16); M is the free dimension the merged primitive does **not** guard, so the layer pads it here
/// and slices the padded rows back off. 16 matches the general cuBLASLt 8-bit leading-dim alignment
/// (`check_alignment`) and is the multiple the live-GPU M-alignment test confirms cuBLASLt accepts.
pub const NVFP4_M_ALIGN: usize = 16;

// Only the W4A4 FP4 forward (cuda-only) needs M-row rounding; dead on a non-cuda build.
#[cfg(feature = "cuda")]
#[inline]
fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// The activation-precision regime for an [`Nvfp4Linear`] — the mixed-precision policy flag
/// (sc-11041). Default **W4A4**; **W4A16** is the per-layer override for the outlier class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ActPrecision {
    /// W4A4 — both operands FP4. Lights up the FP4 tensor cores (~2×). The compute default.
    #[default]
    W4A4,
    /// W4A16 — FP4 weight × bf16 activation. Dequantizes the weight to bf16 (no FP4 compute win); the
    /// per-layer override for the outlier class where W4A4 collapses (sc-7702 / spike sc-11038).
    W4A16,
}

/// Conservative name anchor for the trunk's **final** output projection — the fallback
/// [`ActPrecision::for_outlier_layer`] uses when the provider does not thread an explicit
/// `is_final_proj` (see [`ActPrecision::for_outlier_layer_with`], which is the authoritative form).
///
/// True iff `l` (already lowercased) names a `proj_out` that is **not nested inside a transformer
/// block** — i.e. its last dotted segment is `proj_out` (tolerating a trailing `.weight`/`.bias`
/// tensor suffix) and it carries no block-nesting marker. That keeps SANA's top-level `proj_out`
/// classified while leaving LTX's `transformer_blocks.{i}.ff.proj_out` and Flux/Chroma's
/// `single_transformer_blocks.{i}.proj_out` — per-block, benign, W4A4 — alone.
fn names_final_proj(l: &str) -> bool {
    // Nested under a transformer block ⇒ a per-block projection, never the trunk head.
    if l.contains("blocks.") || l.contains("block_") {
        return false;
    }
    let mut segs = l.rsplit('.');
    let last = segs.next().unwrap_or("");
    // Tolerate a trailing tensor-param suffix (`proj_out.weight`) as well as a bare prefix key.
    let tail = if matches!(last, "weight" | "bias") {
        segs.next().unwrap_or("")
    } else {
        last
    };
    tail == "proj_out"
}

impl ActPrecision {
    /// The **default per-layer policy**: the outlier-carrying layer class runs **W4A16** (bf16
    /// activation), everything else **W4A4**. Matched by substring on the layer path so a provider can
    /// thread its own dotted key names; callers wanting a blanket regime pass [`ActPrecision::W4A4`] /
    /// [`ActPrecision::W4A16`] directly instead of consulting this.
    ///
    /// The outlier class — where a dense activation outlier collapses W4A4 (the sc-7702 mechanism) — is
    /// the text→DiT `caption_projection`, **the whole cross-attention block** (`attn2` / `cross_att*`),
    /// the trunk's **final** `proj_out` head, and the first & last DiT blocks.
    ///
    /// **Widened in sc-11045 from real measurements — read before narrowing it again.** The spike
    /// (sc-11038) specified "W4A4 on the compute-bulk benign layers (**self-attn** + FF); bf16
    /// activation on the outlier class", and named that class "cross-attn K/V". This function
    /// originally took the K/V wording literally, leaving cross-attn **Q** and **`to_out`** on W4A4.
    /// Capturing per-layer activation-outlier sparsity across a **real Sana-1.6B denoise** (sc-11045's
    /// `ActProbe`) refuted that reading: of 109 projections the old policy sent to W4A4, **27 measured
    /// [`OutlierClass::Dense`](super::OutlierClass::Dense)** — 17 × `attn2.to_out.0`, 6 × `attn2.to_q`
    /// (per-block crush ratios up to **5124×**), plus `proj_out` (438×). The whole cross-attention block
    /// consumes caption-derived context, so it carries the caption's massive activations regardless of
    /// which projection you name. Guarding `attn2` wholesale restores the spike's actual intent (W4A4 ==
    /// self-attn + FF).
    ///
    /// # The widening is a strict superset of the pre-sc-11045 rule — and is tested as one
    ///
    /// Every clause the pre-sc-11045 policy guarded is still guarded here **verbatim** (including its
    /// `cross` + K/V clause, retained below), so this rule can only ever move a layer
    /// **W4A4 → W4A16** — the safe direction, since W4A16 is already the shipping throughput default
    /// (sc-12078). That is not a claim, it is a property: `differential_widening_is_strictly_safe`
    /// re-implements the old rule and asserts the superset over a cross-provider name corpus.
    ///
    /// The retained `cross` + K/V clause matters because the token clauses alone are **not** a
    /// superset: `cross_attn` does not match `cross_attention` (position 10 is `e`, not `n`), so
    /// matching only `cross_attn`/`crossattn` would silently regress `cross_attention.to_k` from
    /// W4A16 back to W4A4 — the collapse-prone direction, on the exact K/V layers the spike named.
    /// Match `cross_att`/`crossatt` (both spellings) *and* keep the old clause.
    pub fn for_outlier_layer(layer_name: &str) -> Self {
        Self::for_outlier_layer_with(layer_name, false)
    }

    /// [`Self::for_outlier_layer`] with the provider's **explicit** structural knowledge threaded in.
    ///
    /// `is_final_proj` marks the trunk's **final output projection** (the head). This is the same seam
    /// as `candle-gen-sana`'s `is_edge_block`: a dotted key alone cannot distinguish the trunk's final
    /// `proj_out` from a *per-block* layer that merely spells itself `proj_out`, so the provider — which
    /// knows — says so. When it does not, [`Self::for_outlier_layer`] falls back to
    /// `names_final_proj`, a conservative name anchor.
    ///
    /// **Why this is not a bare `contains("proj_out")`.** The measurement behind the `proj_out` clause
    /// (Dense, crush 438× — sc-11045) is SANA's single *top-level* head. A bare substring also matches
    /// per-block layers elsewhere in this workspace that are the spike's explicitly **benign** W4A4
    /// class:
    ///
    /// * `candle-gen-ltx` remaps `ff.net.2` → `ff.proj_out` (its `tier.rs`), i.e. the **FF output
    ///   projection** of all 48 blocks — the compute bulk W4A4 exists to capture;
    /// * `candle-gen-flux` / `candle-gen-chroma` name `single_transformer_blocks.{i}.proj_out` — the
    ///   fused attn+MLP output `[5·hidden → hidden]`, the largest GEMM in each of 38 single blocks.
    ///
    /// Firing on those would be the same doc-says-"final"/code-says-"any" mis-encoding this policy was
    /// corrected to remove. Pinned by `final_proj_anchor_does_not_fire_on_per_block_names`.
    pub fn for_outlier_layer_with(layer_name: &str, is_final_proj: bool) -> Self {
        let l = layer_name.to_ascii_lowercase();
        let is_outlier = l.contains("caption_projection")
            || l.contains("caption_proj")
            // The ENTIRE cross-attention block (attn2 = cross-attn in the diffusers DiT naming), not
            // just K/V — Q and to_out read caption-derived context and measure Dense on real
            // activations (sc-11045).
            || l.contains("attn2")
            // Both underscore spellings (cross_attn, cross_attention) and both unseparated ones
            // (crossattn, crossattention).
            || l.contains("cross_att")
            || l.contains("crossatt")
            // Retained VERBATIM from the pre-sc-11045 rule so the widening is provably a superset:
            // any `cross*` K/V spelling the old policy guarded stays guarded, whatever its separator.
            || (l.contains("cross")
                && (l.contains("_k") || l.contains("_v") || l.contains(".k") || l.contains(".v")))
            // The FINAL output projection (measured Dense, crush 438× — sc-11045) — anchored to the
            // trunk head, never a per-block layer that merely spells itself `proj_out`.
            || is_final_proj
            || names_final_proj(&l)
            // first & last DiT blocks (blocks.0 / block_0 and an explicit last-block marker).
            || l.contains("blocks.0.")
            || l.contains("block_0.")
            || l.contains("last_block")
            || l.contains("final_block");
        if is_outlier {
            ActPrecision::W4A16
        } else {
            ActPrecision::W4A4
        }
    }

    /// Partition a set of layer names into the W4A4 (benign, FP4 activation) and W4A16 (outlier class,
    /// bf16 activation) classes per [`Self::for_outlier_layer`] — the **explicit, testable** form of
    /// the spike sc-11038 mixed-precision policy (sc-11044 AC 2). Returns each name paired with its
    /// regime plus the partition counts, so a loader/report can show exactly which projections light
    /// the FP4 cores and which stay bf16-activation.
    pub fn partition_layers<'a, I>(names: I) -> Nvfp4Partition
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut assignments = Vec::new();
        let (mut n_w4a4, mut n_w4a16) = (0usize, 0usize);
        for name in names {
            let p = Self::for_outlier_layer(name);
            match p {
                ActPrecision::W4A4 => n_w4a4 += 1,
                ActPrecision::W4A16 => n_w4a16 += 1,
            }
            assignments.push((name.to_string(), p));
        }
        Nvfp4Partition {
            assignments,
            n_w4a4,
            n_w4a16,
        }
    }
}

/// The result of [`ActPrecision::partition_layers`] — the mixed-precision policy applied to a concrete
/// set of layers (sc-11044). `n_w4a4` benign projections run the FP4 W4A4 compute path; `n_w4a16`
/// outlier-class projections keep bf16 activation (W4A16).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nvfp4Partition {
    /// Each layer name paired with the regime the policy assigns it.
    pub assignments: Vec<(String, ActPrecision)>,
    /// Count assigned W4A4 (benign, FP4 activation).
    pub n_w4a4: usize,
    /// Count assigned W4A16 (outlier class, bf16 activation).
    pub n_w4a16: usize,
}

impl Nvfp4Partition {
    /// Fraction of layers on the FP4 W4A4 compute path (the compute-bulk the ~2× win rides).
    pub fn w4a4_fraction(&self) -> f64 {
        let total = self.n_w4a4 + self.n_w4a16;
        if total == 0 {
            0.0
        } else {
            self.n_w4a4 as f64 / total as f64
        }
    }
}

/// Which compute path an [`Nvfp4Linear`] actually runs — surfaced so a bench/report can state whether
/// the FP4 cores are lit (sc-11041 AC: the report must name the A-precision regime).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Regime {
    /// W4A4 FP4 tensor-core GEMM on `sm_120` via cuBLASLt (`matmul_nvfp4_staged`) — the real FP4
    /// compute win, weight resident at the NVFP4 footprint (the SC#6 packed-forward path).
    Fp4W4A4,
    /// Weight dequantized to bf16, dense bf16 matmul (full-precision activation) — the W4A16 outlier
    /// override **or** the `<sm_120` / CPU / non-cuda / ineligible-shape / no-fused-quantizer
    /// capability fallback. **No FP4 compute** (storage/parity tier).
    ///
    /// # Why an unavailable fused quantizer lands here (sc-12078 fallback policy)
    ///
    /// W4A4 needs to quantize the activation every projection, and there are two implementations: the
    /// fused nvrtc kernel (~0.38 ms/projection at K=6144, M=4118) and the unfused candle reference
    /// chain (~19 ms). Those are not two speeds of the same regime — they are different products:
    ///
    /// | regime | ms/step (Krea 2 Turbo, 1024², exclusive rig) | vs bf16 |
    /// |---|---:|---:|
    /// | dense bf16 | 893.7 | 1.00× |
    /// | W4A16 (this regime) | 895.7 | 1.00× |
    /// | W4A4, fused quantizer | 716.3 | **1.25×** |
    /// | W4A4, unfused quantizer | ~90 000 | **0.01×** |
    ///
    /// So when the fused kernel is unavailable, W4A4-via-unfused is ~100× *worse than never having
    /// used NVFP4*, while W4A16 costs ~nothing. Routing the gate here is not a compromise; it is
    /// ~100× the better answer, and the only thing it gives up is the packed footprint (0.28× → 1.00×,
    /// itself a defect tracked by sc-12121, which makes W4A16 packed-resident). Correctness and
    /// quality are identical in both fallbacks — this is purely about not shipping a 100× cliff behind
    /// a silent `Err(_)`.
    DequantBf16,
}

/// Emit the "fused quantizer unavailable → W4A16" note **once per process** (sc-12078).
///
/// Deliberately not per layer. The original reason was that every [`Nvfp4Linear`] built its own
/// `CublasLt` and the compile failure is cached per *handle*, so an unloud version would print once
/// per projection — 260× on a Krea trunk. **sc-12274 removed that specific hazard** (one shared handle
/// per device ⇒ one cached failure ⇒ one print), but the `Once` stays: it is now what keeps the note
/// to one block across *multiple* trunks/devices in a process, and it costs nothing. One block that
/// explains the whole run still beats N that bury it.
#[cfg(feature = "cuda")]
fn warn_fused_quantizer_unavailable() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[sc-12078] Nvfp4Linear: the fused NVFP4 activation quantizer is UNAVAILABLE on this \
             device (nvrtc failed to compile it). W4A4 is disabled for this process: every layer that \
             asked for it now runs W4A16 (dequant→bf16, ~1.00× vs dense bf16) instead.\n\
             \x20 This run loses BOTH NVFP4 wins — the FP4 compute win (1.10× mixed / 1.25× blanket) \
             and the packed VRAM footprint (0.28× → 1.00×, i.e. dense bf16 resident). Output quality \
             is unaffected.\n\
             \x20 W4A4 is gated on this kernel rather than falling back to the unfused reference \
             quantizer because that path costs ~19 ms/projection vs ~0.38 ms fused, and measured 0.01× \
             vs dense bf16 end-to-end — ~100× slower than not using NVFP4 at all. Degrading to W4A16 \
             is ~100× better than degrading to unfused W4A4.\n\
             \x20 Restore nvrtc to get the FP4 lane back."
        );
    });
}

/// The FP4 compute leg — a resident on-device NVFP4 weight + a shared cuBLASLt handle. Cuda-only.
#[cfg(feature = "cuda")]
struct Fp4Resident {
    lt: std::sync::Arc<super::cublaslt::CublasLt>,
    w_staged: super::cublaslt::DevNvfp4,
}

/// The shared handle + the device it is bound to (sc-12274). Cuda-only.
#[cfg(feature = "cuda")]
#[derive(Clone)]
struct Fp4Ctx {
    lt: std::sync::Arc<super::cublaslt::CublasLt>,
    device: Device,
}

/// A **shared, per-device cuBLASLt compute context** for [`Nvfp4Linear`] (sc-12274).
///
/// # Why this type exists
///
/// `CublasLt::new` eagerly allocates a **32 MiB workspace** and holds it for the handle's life, and
/// the handle's three caches (`nvfp4_algos` by `(m,k,n)`, `nvfp4_act_scale_gather_idx` by
/// `(rows, n_blocks)`, `nvfp4_quant_kernels` by nothing) are keyed **by shape or not at all** —
/// nothing on it is per-layer. Its own module doc says so: *"a real integration caches one per
/// device."* Building one per layer was therefore never a design choice, just an oversight — and it
/// cost, **measured on a 260-projection blanket-W4A4 Krea trunk** (sc-12274):
///
/// | | |
/// |---|---:|
/// | real resident VRAM | 15.41 GiB |
/// | dense bf16 baseline | 25.56 GiB |
/// | **real footprint ratio** | **0.603×** |
/// | SC#6 weights-only reported | 0.2813× |
///
/// i.e. the headline SC#6 figure was **2.14× optimistic** on the exact regime it was claimed for,
/// because ~6.6 GiB of duplicated workspace is not weight bytes and so is invisible to
/// `resident_weight_bytes`. One handle per device makes that ~32 MiB total, and lets the shape-keyed
/// caches actually hit across layers (with a handle per layer they never saw a second layer at all).
///
/// # Contract
///
/// Cfg-neutral **by design** — a zero-sized type on a non-cuda build, so `*_in` constructors have one
/// signature everywhere. An **empty** context is always valid and always safe: every layer built with
/// it takes the transparent dequant→bf16 fallback, exactly as [`Nvfp4Linear`] already does on a CPU
/// device, a `<sm_120` GPU, or an ineligible shape. [`Self::new`] therefore returns an empty context
/// rather than an error for all of those cases — it never fails just because FP4 is unavailable.
///
/// Safe to share across layers: all handles on a device already resolve to the **same** stream
/// (`CublasLt::new` takes `device.cuda_stream()`), so sharing introduces no stream coupling that did
/// not already exist. The one genuinely shared mutable resource is the 32 MiB workspace scratch; a
/// denoise is sequential on that one stream, so cuBLASLt serializes access to it.
#[derive(Clone, Default)]
pub struct Nvfp4Context {
    #[cfg(feature = "cuda")]
    inner: Option<Fp4Ctx>,
}

impl Nvfp4Context {
    /// The **empty** context: no shared handle, so every layer built with it serves dequant→bf16.
    /// The honest choice on any non-FP4 device — and what [`Self::new`] returns there.
    pub fn none() -> Self {
        Self::default()
    }

    /// Build **one** cuBLASLt handle for `device`, to be shared by every [`Nvfp4Linear`] on it.
    ///
    /// Resolves the whole capability gate **once** instead of per layer: a non-CUDA device, a
    /// `<sm_120` GPU, a non-cuda build, or a handle-creation failure all yield [`Self::none`] (not an
    /// error), and the caller's layers then fall back transparently. Below the NVFP4 floor the probe
    /// handle is **dropped**, so a non-Blackwell box does not hold 32 MiB for a path it cannot take.
    pub fn new(device: &Device) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if matches!(device, Device::Cuda(_)) {
                match super::cublaslt::CublasLt::new(device) {
                    Ok(lt) => match lt.meets_nvfp4_floor() {
                        Ok(true) => {
                            return Ok(Self {
                                inner: Some(Fp4Ctx {
                                    lt: std::sync::Arc::new(lt),
                                    device: device.clone(),
                                }),
                            })
                        }
                        // <sm_120 → FP4 is unavailable; drop the probe handle rather than pay its
                        // workspace for a path no layer will take.
                        _ => return Ok(Self::none()),
                    },
                    Err(e) => {
                        eprintln!(
                            "[sc-12274] Nvfp4Context: cuBLASLt handle unavailable ({e}); every layer \
                             built with this context takes the bf16 fallback"
                        );
                        return Ok(Self::none());
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        Ok(Self::none())
    }

    /// True iff this context carries a live FP4 handle (i.e. layers built with it can light the cores).
    pub fn is_fp4(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            return self.inner.is_some();
        }
        #[allow(unreachable_code)]
        false
    }

    /// The shared handle, **iff** it is bound to `device`.
    ///
    /// The device check is the one hazard sharing introduces that per-layer handles could not have: a
    /// context built on `cuda:0` handed to a layer on `cuda:1` would stage the weight through the
    /// wrong device's stream. Mismatch is loud and falls back rather than corrupting the layer.
    #[cfg(feature = "cuda")]
    fn handle_for(&self, device: &Device) -> Option<&std::sync::Arc<super::cublaslt::CublasLt>> {
        let c = self.inner.as_ref()?;
        if c.device.same_device(device) {
            Some(&c.lt)
        } else {
            eprintln!(
                "[sc-12274] Nvfp4Linear: shared cuBLASLt context is bound to {:?} but this layer is on \
                 {:?}; using bf16 fallback rather than staging through the wrong device",
                c.device.location(),
                device.location()
            );
            None
        }
    }
}

/// An NVFP4 linear projection `y = x·Wᵀ (+ b)` over a packed [`Nvfp4Tensor`] weight (sc-11041).
///
/// Built from packed weights ([`Self::from_packed`]) or a dense weight ([`Self::from_dense`]). On
/// `sm_120` with the default W4A4 regime it serves the weight resident-packed and runs the cuBLASLt FP4
/// GEMM (the SC#6 packed-forward path); otherwise it transparently falls back to a dequant→bf16 dense
/// matmul (see the [module docs](self)). The packed [`Nvfp4Tensor`] is always retained (the source of
/// truth for the *format* accounting + a re-stage), so [`Self::nvfp4_footprint_bytes`] reports the
/// packed NVFP4 size regardless of regime — **which is exactly why it is not the SC#6 number**. What a
/// run actually holds in VRAM is [`Self::resident_weight_bytes`]: the packed buffers under W4A4, but a
/// full dense **bf16** weight under [`Nvfp4Regime::DequantBf16`].
pub struct Nvfp4Linear {
    /// The packed NVFP4 weight (host container — the canonical ~4.5-bit representation). Retained in
    /// every regime for footprint accounting and to (re)stage the FP4 device weight.
    weight: Nvfp4Tensor,
    bias: Option<Tensor>,
    device: Device,
    act: ActPrecision,
    regime: Nvfp4Regime,
    /// The resident bf16 dense weight `[out, in]` for the [`Nvfp4Regime::DequantBf16`] path (dequantized
    /// once at construction). `None` in the FP4 regime.
    dequant_w: Option<Tensor>,
    /// The resident FP4 compute leg (staged device weight + handle) for [`Nvfp4Regime::Fp4W4A4`].
    #[cfg(feature = "cuda")]
    fp4: Option<Fp4Resident>,
}

impl Nvfp4Linear {
    /// Build from an already-packed [`Nvfp4Tensor`] weight, choosing the regime by the capability gate +
    /// the requested [`ActPrecision`]. `bias` is kept full-precision (added back in the activation dtype).
    ///
    /// W4A4 on a `sm_120` CUDA device with an eligible shape → the FP4 packed-forward path; anything
    /// else → the transparent dequant→bf16 fallback. Never panics on an ineligible device/shape.
    /// **Builds a private cuBLASLt handle for this one layer** (a 32 MiB workspace). Correct for a
    /// one-off layer or a test; **a trunk must use [`Self::from_packed_in`]** with one shared
    /// [`Nvfp4Context`], or it pays 32 MiB *per projection* — the sc-12274 defect (measured: 6.6 GiB
    /// across a 260-projection Krea trunk, which took the reported SC#6 footprint from 0.2813× to a
    /// real 0.603×).
    pub fn from_packed(
        weight: Nvfp4Tensor,
        bias: Option<Tensor>,
        device: &Device,
        act: ActPrecision,
    ) -> Result<Self> {
        // Only pay for a handle at all if this layer could actually use one.
        let ctx = if act == ActPrecision::W4A4 {
            Nvfp4Context::new(device)?
        } else {
            Nvfp4Context::none()
        };
        Self::from_packed_in(weight, bias, device, act, &ctx)
    }

    /// [`Self::from_packed`] sharing an existing per-device [`Nvfp4Context`] — **the constructor a
    /// trunk loader wants** (sc-12274). Every layer built with one context shares its single cuBLASLt
    /// handle, so the 32 MiB workspace and the shape-keyed algo / kernel caches are paid once per
    /// device instead of once per layer.
    ///
    /// An empty context (CPU, `<sm_120`, non-cuda build, handle failure) simply means every layer
    /// takes the dequant→bf16 fallback — never an error.
    pub fn from_packed_in(
        weight: Nvfp4Tensor,
        bias: Option<Tensor>,
        device: &Device,
        act: ActPrecision,
        ctx: &Nvfp4Context,
    ) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if act == ActPrecision::W4A4 {
                if let Some(built) = Self::try_build_fp4(&weight, &bias, device, ctx)? {
                    return Ok(built);
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = ctx;
        // W4A16 override, or the <sm_120 / CPU / non-cuda / ineligible-shape fallback.
        Self::new_dequant(weight, bias, device, act)
    }

    /// Pack a dense `[out, in]` weight (bf16/f32, any device) to NVFP4 and build (see [`Self::from_packed`]).
    /// **Builds a private handle** — a trunk wants [`Self::from_dense_in`].
    pub fn from_dense(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        act: ActPrecision,
    ) -> Result<Self> {
        let packed = Nvfp4Tensor::pack(weight)?;
        Self::from_packed(packed, bias, device, act)
    }

    /// [`Self::from_dense`] sharing an existing per-device [`Nvfp4Context`] (sc-12274) — the
    /// from-dense twin of [`Self::from_packed_in`].
    pub fn from_dense_in(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        act: ActPrecision,
        ctx: &Nvfp4Context,
    ) -> Result<Self> {
        let packed = Nvfp4Tensor::pack(weight)?;
        Self::from_packed_in(packed, bias, device, act, ctx)
    }

    /// Build applying the **default per-layer policy** ([`ActPrecision::for_outlier_layer`]): the
    /// outlier class runs W4A16, everything else W4A4. The convenience entry a loader calls with the
    /// projection's dotted key name to get the mixed-precision default without classifying by hand.
    pub fn from_dense_for_layer(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        layer_name: &str,
    ) -> Result<Self> {
        Self::from_dense(weight, bias, device, ActPrecision::for_outlier_layer(layer_name))
    }

    /// Attempt to build the resident FP4 (W4A4) compute leg **against a shared handle**. `Ok(None)`
    /// when the device is not CUDA, the shape is ineligible for the cuBLASLt FP4 path, the fused
    /// activation quantizer will not compile (sc-12078), or `ctx` carries no handle for this device
    /// (not `sm_120`+, or handle creation failed) → caller falls back. A weight-staging failure also
    /// degrades to `Ok(None)` (transparent fallback) with a note.
    ///
    /// **sc-12274:** this used to call `CublasLt::new(device)` itself, giving every W4A4 layer its own
    /// 32 MiB workspace. The handle now arrives from [`Nvfp4Context`], which resolved the CUDA +
    /// `sm_120` capability gate **once per device** — so here an absent handle simply *is* that
    /// fallback signal, and the explicit `meets_nvfp4_floor` probe that used to sit here is gone with
    /// the per-layer handle it belonged to.
    #[cfg(feature = "cuda")]
    fn try_build_fp4(
        weight: &Nvfp4Tensor,
        bias: &Option<Tensor>,
        device: &Device,
        ctx: &Nvfp4Context,
    ) -> Result<Option<Self>> {
        use super::cublaslt::NVFP4_K_ALIGN;
        if !matches!(device, Device::Cuda(_)) {
            return Ok(None);
        }
        // Shape gate: the cuBLASLt FP4 path needs padded-K a multiple of NVFP4_K_ALIGN and N a
        // multiple of 16 (sc-11039). An ineligible shape falls back rather than erroring at runtime.
        if !weight.cols_padded.is_multiple_of(NVFP4_K_ALIGN) || !weight.rows.is_multiple_of(16) {
            eprintln!(
                "[sc-11041] Nvfp4Linear: shape [{}, {} (K_pad {})] ineligible for the cuBLASLt FP4 \
                 path (need K_pad % {NVFP4_K_ALIGN} == 0 and N % 16 == 0); using bf16 fallback",
                weight.rows, weight.cols, weight.cols_padded
            );
            return Ok(None);
        }
        let Some(lt) = ctx.handle_for(device) else {
            return Ok(None); // no shared handle for this device (<sm_120 / unavailable) → fallback
        };
        // The <sm_120 floor is already settled: `Nvfp4Context::new` probes `meets_nvfp4_floor` once per
        // device and holds NO handle below it (sc-12274), so reaching here with a handle means the
        // device is Blackwell. Re-probing per layer would be the per-layer-work habit this story removed.
        //
        // Fused-quantizer gate (sc-12078 fallback policy). W4A4 needs an activation quantizer, and the
        // fused kernel is the only one fast enough to make the regime worth running: the unfused
        // reference chain costs ~19 ms/projection against its ~0.38 ms, which measured 0.01× vs dense
        // bf16 end-to-end. So "nvrtc cannot compile the fused kernel" is a capability miss exactly like
        // <sm_120 — the honest response is the same one, W4A16 — and it is settled HERE, at construction,
        // rather than per forward. Probing costs nothing: the compile is cached on the handle, so the
        // first forward would have paid it anyway — and now that the handle is shared per device
        // (sc-12274), the whole trunk pays that compile once instead of once per projection.
        if !lt.nvfp4_fused_quantizer_available() {
            warn_fused_quantizer_unavailable();
            return Ok(None);
        }
        let w_staged = match lt.stage_nvfp4(weight) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[sc-11041] Nvfp4Linear: FP4 weight stage failed ({e}); bf16 fallback");
                return Ok(None);
            }
        };
        Ok(Some(Self {
            weight: weight.clone(),
            bias: bias.clone(),
            device: device.clone(),
            act: ActPrecision::W4A4,
            regime: Nvfp4Regime::Fp4W4A4,
            dequant_w: None,
            fp4: Some(Fp4Resident {
                // The Arc was always here — it just never had a second owner (sc-12274).
                lt: std::sync::Arc::clone(lt),
                w_staged,
            }),
        }))
    }

    /// Build the [`Nvfp4Regime::DequantBf16`] path: dequantize the packed weight to a resident bf16
    /// `[out, in]` tensor on `device` **once**. The W4A16 override and the capability fallback share
    /// this (they differ only in *why* they are here — recorded in `act`).
    fn new_dequant(
        weight: Nvfp4Tensor,
        bias: Option<Tensor>,
        device: &Device,
        act: ActPrecision,
    ) -> Result<Self> {
        // `Nvfp4Tensor::dequantize` returns a CPU f32 [rows, cols]; store it resident as bf16 on device.
        let w = weight
            .dequantize()?
            .to_dtype(DType::BF16)?
            .to_device(device)?;
        Ok(Self {
            weight,
            bias,
            device: device.clone(),
            act,
            regime: Nvfp4Regime::DequantBf16,
            dequant_w: Some(w),
            #[cfg(feature = "cuda")]
            fp4: None,
        })
    }

    /// `y = x·Wᵀ (+ b)`. Routes to the FP4 W4A4 GEMM ([`Nvfp4Regime::Fp4W4A4`]) or the dequant→bf16
    /// dense matmul ([`Nvfp4Regime::DequantBf16`]). Accepts a rank-≥1 activation `[..., in]`; the output
    /// is `[..., out]` cast back to the input dtype.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self.regime {
            #[cfg(feature = "cuda")]
            Nvfp4Regime::Fp4W4A4 => self.forward_fp4(x),
            // The FP4 regime is only ever constructed under the `cuda` feature (`try_build_fp4`), so on
            // a non-cuda build this arm is unreachable — but the match must still compile.
            #[cfg(not(feature = "cuda"))]
            Nvfp4Regime::Fp4W4A4 => {
                unreachable!("Nvfp4Regime::Fp4W4A4 is only constructed with the cuda feature")
            }
            Nvfp4Regime::DequantBf16 => self.forward_dequant(x),
        }
    }

    /// The dequant→bf16 dense forward (W4A16 / fallback). The resident bf16 weight is cast to the
    /// activation dtype and run through a plain `candle_nn::Linear` (which broadcast-matmuls rank-≥2
    /// inputs), keeping the activation full-precision.
    fn forward_dequant(&self, x: &Tensor) -> Result<Tensor> {
        let w = self
            .dequant_w
            .as_ref()
            .expect("DequantBf16 regime holds a resident weight")
            .to_dtype(x.dtype())?;
        let bias = match &self.bias {
            Some(b) => Some(b.to_dtype(x.dtype())?),
            None => None,
        };
        Linear::new(w, bias).forward(x)
    }

    /// The W4A4 FP4 forward (cuda-only): flatten leading dims to `[M, K]`, pad M to [`NVFP4_M_ALIGN`],
    /// pack the activation to NVFP4, run the resident-weight cuBLASLt FP4 GEMM, slice the padded rows
    /// off, reshape back, add bias, cast to the input dtype. The weight never dequantizes — this is the
    /// SC#6 packed-forward path.
    #[cfg(feature = "cuda")]
    fn forward_fp4(&self, x: &Tensor) -> Result<Tensor> {
        let fp4 = self.fp4.as_ref().expect("Fp4W4A4 regime holds a staged weight");
        let dims = x.dims().to_vec();
        let k = *dims.last().expect("linear input has a last dim");
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((m, k))?;

        // M-alignment (sc-11039 handoff): pad the token rows up to a multiple of NVFP4_M_ALIGN with
        // zero rows (they add nothing to any real output) so cuBLASLt does not return NOT_SUPPORTED,
        // then slice the padding back off the result.
        let m_pad = round_up(m.max(1), NVFP4_M_ALIGN);
        let x_pad = if m_pad != m {
            x2.pad_with_zeros(0, 0, m_pad - m)?
        } else {
            x2
        };

        // W4A4 (sc-11044): quantize the activation to NVFP4 **on-device** — no CPU round-trip — and run
        // the FP4 GEMM against the resident weight. `cols_padded` matches the resident weight so the two
        // operands share the padded contraction width.
        //
        // The fused quantizer (sc-12078) is called UNCONDITIONALLY, and its errors propagate. This
        // regime only exists because `try_build_fp4` already proved the kernel compiles on this handle,
        // so there is no "nvrtc is missing" case left to catch here. What remains — a shape/storage
        // mismatch, an OOM, a launch failure — is a real fault, and rerouting it to the unfused
        // reference chain would answer a bug with a ~50×-slower projection while hiding the bug
        // (the unfused path allocates too, so it would not survive an OOM either). Fail loud instead.
        let cols_padded = fp4.w_staged.shape_padded().1;
        let x_stg = fp4.lt.quantize_nvfp4_activation_fused(&x_pad, cols_padded)?;
        let y = fp4.lt.matmul_nvfp4_staged(&fp4.w_staged, &x_stg)?; // [m_pad, N] bf16
        let y = if m_pad != m { y.narrow(0, 0, m)? } else { y };

        let n = y.dim(1)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(n);
        let mut y = y.reshape(out_shape)?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(&b.to_dtype(y.dtype())?)?;
        }
        y.to_dtype(x.dtype())
    }

    /// [`Self::forward`] plus a **NaN/inf guard** (sc-11044 AC): asserts the output is finite and
    /// **fails loud** rather than letting a collapsed/garbage tensor propagate silently through the
    /// denoise. The W4A4 quantizer itself is NaN-free by construction (E2M1 saturates, divisors are
    /// clamped positive — spike sc-11038), so this guards the *accuracy* failure mode (signal collapse
    /// over steps surfacing as an inf/NaN downstream), not the quantizer. Uses a single sum-of-squares
    /// scalar reduction (a NaN/inf in any element makes the sum non-finite), so it is cheap enough to
    /// leave on around a denoise step. Callers wanting max throughput use [`Self::forward`] directly.
    pub fn forward_checked(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.forward(x)?;
        let energy = y.to_dtype(DType::F32)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        if !energy.is_finite() {
            candle_core::bail!(
                "Nvfp4Linear::forward_checked: non-finite output (NaN/inf) from the {:?} regime — \
                 W4A4 signal collapse or a bad activation; failing loud (sc-11044 NaN guard)",
                self.regime
            );
        }
        Ok(y)
    }

    /// The active compute path (whether the FP4 cores are lit) — for a bench/report.
    pub fn regime(&self) -> Nvfp4Regime {
        self.regime
    }

    /// The requested activation-precision regime (W4A4 default / W4A16 override).
    pub fn act_precision(&self) -> ActPrecision {
        self.act
    }

    /// True iff this layer runs the real FP4 tensor-core GEMM (W4A4 on sm_120) — as opposed to the
    /// dequant→bf16 storage/fallback path.
    pub fn lights_up_fp4(&self) -> bool {
        matches!(self.regime, Nvfp4Regime::Fp4W4A4)
    }

    /// The **NVFP4 footprint** in bytes of the weight — E2M1 nibble bytes + UE4M3 block-scale bytes
    /// (the packed host container's actual size). The resident device weight in the FP4 regime matches
    /// this (see `resident_device_bytes`); the SC#6 gate asserts this is ≈ the ~4.5-bit
    /// footprint and far below the bf16 size.
    pub fn nvfp4_footprint_bytes(&self) -> usize {
        self.weight.packed.len() + self.weight.scales.len()
    }

    /// The bf16 footprint the weight *would* occupy dense (`rows * cols * 2`) — the baseline the SC#6
    /// resident-VRAM assertion compares against.
    pub fn bf16_footprint_bytes(&self) -> usize {
        self.weight.rows * self.weight.cols * 2
    }

    /// The logical `[out, in]` weight shape.
    pub fn shape(&self) -> (usize, usize) {
        (self.weight.rows, self.weight.cols)
    }

    /// The device the layer's resident weight lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Resident **device** bytes of the FP4 weight when in the [`Nvfp4Regime::Fp4W4A4`] path — the
    /// E2M1 nibble buffer + UE4M3 block-scale buffer actually uploaded to VRAM (the SC#6 packed-forward
    /// proof). `None` in the dequant→bf16 regime (there the resident device weight is the bf16 tensor).
    #[cfg(feature = "cuda")]
    pub fn resident_device_bytes(&self) -> Option<usize> {
        self.fp4.as_ref().map(|f| f.w_staged.resident_bytes())
    }

    /// Resident **device** bytes of the dequantized **bf16** weight when in the
    /// [`Nvfp4Regime::DequantBf16`] path — the dense `[out, in]` bf16 tensor `new_dequant` materialized
    /// and holds for the layer's whole life. `None` in the FP4 regime (nothing is dequantized there).
    ///
    /// **This is the honest cost of W4A16** and the reason the footprint accounting must be
    /// regime-aware: a W4A16 layer's packed container is a *host* container: what sits in VRAM is a
    /// full dense bf16 weight, i.e. **1.0× the bf16 baseline, not 0.28×**. Reporting the packed size as
    /// though it were the device footprint would let a run with nothing packed on-device still claim an
    /// NVFP4 footprint (the sc-11045 review's MAJOR 3).
    pub fn resident_dequant_bf16_bytes(&self) -> Option<usize> {
        self.dequant_w
            .as_ref()
            .map(|w| w.elem_count() * w.dtype().size_in_bytes())
    }

    /// Total bytes this layer actually holds **resident on-device for its weight**, whatever regime it
    /// resolved to: the staged E2M1 + UE4M3 buffers under [`Nvfp4Regime::Fp4W4A4`], or the dense bf16
    /// tensor under [`Nvfp4Regime::DequantBf16`].
    ///
    /// Unlike [`Self::nvfp4_footprint_bytes`] (a property of the *format*), this is a property of the
    /// **run** — it is what SC#6 is actually about.
    pub fn resident_weight_bytes(&self) -> usize {
        #[cfg(feature = "cuda")]
        if let Some(b) = self.resident_device_bytes() {
            return b;
        }
        self.resident_dequant_bf16_bytes().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 in ~[-1, 1) — no `rand` dep.
    fn prng(seed: &mut u64) -> f32 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn rel_rms(a: &[f32], b: &[f32]) -> f32 {
        let (mut num, mut den) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            num += ((*x - *y) as f64).powi(2);
            den += (*x as f64).powi(2);
        }
        (num / (den + 1e-30)).sqrt() as f32
    }

    /// **The sc-12078 fallback policy, as a gate the CPU lane can run.**
    ///
    /// The W4A4 forward must never reach the unfused reference quantizer. It once did, via an
    /// `Err(_) => quantize_nvfp4_activation(..)` arm that silently served W4A4 at ~19 ms/projection
    /// (0.01× vs dense bf16 — ~100× slower than not using NVFP4) whenever nvrtc failed. That is now a
    /// construction-time capability gate that falls back to W4A16 (~1.00×) instead.
    ///
    /// This is a source-level assertion on purpose. The regression it guards is **invisible to every
    /// numeric gate we have** — the unfused path is bit-identical to the fused one, so rel-RMS reports
    /// 0.000000 either way and only the clock knows. The behavioural tests that *can* see it are
    /// `#[ignore]`d and need an exclusive sm_120 rig, so nothing in a normal CI run would notice the
    /// arm coming back as a well-meaning "robustness" fix. Cheap guard, expensive silence.
    #[test]
    fn w4a4_forward_never_routes_to_the_unfused_quantizer() {
        // Production source only: split at the test module, whose own text contains these needles.
        let src = include_str!("nvfp4_linear.rs");
        let production = src.split("#[cfg(test)]").next().expect("split yields a prefix");
        assert!(
            !production.contains(".quantize_nvfp4_activation("),
            "the W4A4 forward calls the UNFUSED quantizer. It must not: that path measured 0.01× vs \
             dense bf16 end-to-end (~19 ms/projection vs ~0.38 ms fused), so reaching it is ~100× \
             worse than the W4A16 fallback the capability gate is supposed to select. If the fused \
             kernel is unavailable, gate W4A4 off in `try_build_fp4` — do not reroute the forward."
        );
        // ...and not vacuously: the fused call must actually be there, so a rename can't pass this.
        assert!(
            production.contains(".quantize_nvfp4_activation_fused("),
            "the W4A4 forward no longer calls the fused quantizer — the assertion above is vacuous"
        );
    }

    /// On a CPU device (no cuda / not sm_120) `from_packed` always selects the dequant→bf16 fallback,
    /// never crashes, and forwards a coherent result matching the packed weight's dequant reference.
    #[test]
    fn cpu_selects_dequant_fallback_and_forwards() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let mut seed = 0x11FED00Du64;
        let w: Vec<f32> = (0..out_dim * in_dim).map(|_| prng(&mut seed) * 0.4).collect();
        let w_t = Tensor::from_vec(w, (out_dim, in_dim), &dev)?;

        let lin = Nvfp4Linear::from_dense(&w_t, None, &dev, ActPrecision::W4A4)?;
        // No sm_120 here → transparent fallback (the AC), not a crash.
        assert_eq!(lin.regime(), Nvfp4Regime::DequantBf16);
        assert!(!lin.lights_up_fp4());

        // Forward matches x · (dequant W)ᵀ within a small bf16-cast tolerance.
        let x = Tensor::from_vec(
            (0..4 * in_dim).map(|_| prng(&mut seed)).collect::<Vec<_>>(),
            (4, in_dim),
            &dev,
        )?;
        let packed = Nvfp4Tensor::pack(&w_t)?;
        let w_dq = packed.dequantize()?; // [out, in] f32
        let reference = x.matmul(&w_dq.t()?)?;
        let got = lin.forward(&x)?;
        assert_eq!(got.dims(), &[4, out_dim]);
        let rr = rel_rms(
            &got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
            &reference.flatten_all()?.to_vec1::<f32>()?,
        );
        assert!(rr < 0.02, "dequant fallback forward rel-RMS {rr} vs its own dequant ref");
        Ok(())
    }

    /// The dequant fallback forward preserves the input dtype and handles a rank-3 activation (leading
    /// dims broadcast through `candle_nn::Linear`), with a bias added back.
    #[test]
    fn dequant_forward_rank3_and_bias_and_dtype() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (48usize, 64usize);
        let mut seed = 0xABCDEF01u64;
        let w = Tensor::from_vec(
            (0..out_dim * in_dim).map(|_| prng(&mut seed) * 0.3).collect::<Vec<_>>(),
            (out_dim, in_dim),
            &dev,
        )?;
        let b = Tensor::from_vec(
            (0..out_dim).map(|_| prng(&mut seed)).collect::<Vec<_>>(),
            (out_dim,),
            &dev,
        )?;
        let lin = Nvfp4Linear::from_dense(&w, Some(b.clone()), &dev, ActPrecision::W4A16)?;
        assert_eq!(lin.regime(), Nvfp4Regime::DequantBf16);
        assert_eq!(lin.act_precision(), ActPrecision::W4A16);

        let x = Tensor::from_vec(
            (0..2 * 5 * in_dim).map(|_| prng(&mut seed)).collect::<Vec<_>>(),
            (2, 5, in_dim),
            &dev,
        )?;
        let y = lin.forward(&x)?;
        assert_eq!(y.dims(), &[2, 5, out_dim]);
        assert_eq!(y.dtype(), x.dtype());
        Ok(())
    }

    /// The per-layer policy defaults the outlier class to W4A16 and everything else to W4A4.
    ///
    /// The cross-attn / `proj_out` cases are pinned from **real measured activations** (sc-11045), not
    /// the spike's prose: `attn2.to_q`, `attn2.to_out.0` and `proj_out` all measured Dense-outlier on a
    /// live Sana-1.6B denoise. See [`ActPrecision::for_outlier_layer`].
    #[test]
    fn outlier_policy_classification() {
        for name in [
            "transformer.caption_projection.linear_1",
            "blocks.0.attn.to_q",
            "transformer_blocks.12.attn2.to_k", // cross-attn K
            "transformer_blocks.7.attn2.to_v",  // cross-attn V
            "transformer_blocks.7.attn2.to_q",  // cross-attn Q — measured Dense (sc-11045)
            "blocks.12.attn2.to_out.0",         // cross-attn OUT — measured Dense (sc-11045)
            "proj_out",                         // final output proj — measured Dense (sc-11045)
            "dit.final_block.proj",
        ] {
            assert_eq!(
                ActPrecision::for_outlier_layer(name),
                ActPrecision::W4A16,
                "{name} must be classified outlier → W4A16"
            );
        }
        for name in [
            "transformer_blocks.7.attn1.to_q", // self-attn
            "transformer_blocks.7.attn1.to_out.0", // self-attn OUTPUT proj stays benign
            "transformer_blocks.7.ff.net.0.proj",
        ] {
            assert_eq!(
                ActPrecision::for_outlier_layer(name),
                ActPrecision::W4A4,
                "{name} must be classified benign → W4A4"
            );
        }
    }

    /// The **pre-sc-11045 policy**, re-implemented verbatim as the differential baseline for
    /// [`differential_widening_is_strictly_safe`]. Do not "fix" this — it is a frozen historical
    /// record of what shipped before the widening, and the test's whole value is that it is exact.
    fn old_rule_pre_sc11045(layer_name: &str) -> ActPrecision {
        let l = layer_name.to_ascii_lowercase();
        let is_outlier = l.contains("caption_projection")
            || l.contains("caption_proj")
            || (l.contains("attn2") && (l.contains(".to_k") || l.contains(".to_v")))
            || (l.contains("cross")
                && (l.contains("_k") || l.contains("_v") || l.contains(".k") || l.contains(".v")))
            || l.contains("blocks.0.")
            || l.contains("block_0.")
            || l.contains("last_block")
            || l.contains("final_block");
        if is_outlier {
            ActPrecision::W4A16
        } else {
            ActPrecision::W4A4
        }
    }

    /// A cross-provider layer-name corpus: every naming convention in this workspace that the shared
    /// substring policy can plausibly be handed, including the ones that motivated sc-11045's
    /// corrections. Used by the differential and anchor tests below.
    const NAME_CORPUS: &[&str] = &[
        // --- cross-attention, every spelling in the wild -------------------------------------
        "transformer_blocks.7.attn2.to_q",
        "transformer_blocks.7.attn2.to_k",
        "transformer_blocks.7.attn2.to_v",
        "transformer_blocks.7.attn2.to_out.0",
        "transformer_blocks.7.cross_attn.to_k",
        "transformer_blocks.7.cross_attn.to_v",
        "transformer_blocks.7.cross_attention.to_k", // `cross_attn` does NOT match this
        "transformer_blocks.7.cross_attention.to_v",
        "transformer_blocks.7.crossattn.to_k",
        "transformer_blocks.7.crossattention.to_v",
        "blocks.7.cross.k", // the old rule's bare `cross` + `.k` shape
        "blocks.7.cross.v",
        "blocks.7.cross_q_k",
        // --- caption / text→DiT ---------------------------------------------------------------
        "transformer.caption_projection.linear_1",
        "transformer.caption_projection.linear_2",
        "caption_proj.linear_1",
        // --- self-attention + FF: the benign compute bulk ------------------------------------
        "transformer_blocks.7.attn1.to_q",
        "transformer_blocks.7.attn1.to_k",
        "transformer_blocks.7.attn1.to_v",
        "transformer_blocks.7.attn1.to_out.0",
        "transformer_blocks.7.ff.net.0.proj",
        "transformer_blocks.7.ff.net.2",
        // --- `proj_out` spellings: final head vs per-block (MAJOR 1) --------------------------
        "proj_out",        // SANA's top-level head — measured Dense (438×)
        "proj_out.weight", // ...with a tensor suffix
        "transformer.proj_out",
        "transformer_blocks.5.ff.proj_out", // LTX: `ff.net.2` remapped — benign FF output
        "transformer_blocks.5.ff.proj_out.weight",
        "transformer_blocks.5.audio_ff.proj_out", // LTX audio stream, same shape
        "single_transformer_blocks.3.proj_out",   // Flux/Chroma fused attn+MLP output
        "single_transformer_blocks.37.proj_out.weight",
        // --- edge blocks ----------------------------------------------------------------------
        "transformer_blocks.0.attn1.to_q",
        "block_0.attn.to_q",
        "dit.final_block.proj",
        "dit.last_block.proj",
    ];

    /// **The safe-direction property, proven rather than asserted (sc-11045).**
    ///
    /// The PR/README/commit claim that the sc-11045 widening "only ever moves a layer W4A4 → W4A16"
    /// is a *falsifiable* statement about two functions, so test it as one: for every name in the
    /// corpus, whatever the pre-sc-11045 rule sent to **W4A16** must still be **W4A16** today. A
    /// regression in the other direction (W4A16 → W4A4) is the collapse-prone one — it would put a
    /// measured-Dense layer back on FP4 activations — and this test is what makes that impossible to
    /// land silently.
    ///
    /// This is exactly what caught the `cross_attn`/`cross_attention` narrowing: matching the token
    /// `cross_attn` alone regressed `cross_attention.to_k` W4A16 → W4A4 while the surrounding prose
    /// claimed the opposite.
    #[test]
    fn differential_widening_is_strictly_safe() {
        let mut widened = Vec::new();
        for name in NAME_CORPUS {
            let old = old_rule_pre_sc11045(name);
            let new = ActPrecision::for_outlier_layer(name);
            assert!(
                !(old == ActPrecision::W4A16 && new == ActPrecision::W4A4),
                "UNSAFE DIRECTION: {name} regressed W4A16 → W4A4. The sc-11045 policy must only ever \
                 widen the outlier class; moving a layer back onto FP4 activations is the \
                 collapse-prone direction (sc-7702)."
            );
            if old == ActPrecision::W4A4 && new == ActPrecision::W4A16 {
                widened.push(*name);
            }
        }
        // ...and the widening is real, not vacuous: the layers sc-11045 measured Dense moved.
        for name in [
            "transformer_blocks.7.attn2.to_q",
            "transformer_blocks.7.attn2.to_out.0",
            "proj_out",
        ] {
            assert!(
                widened.contains(&name),
                "{name} measured Dense on real activations (sc-11045) and must have widened to W4A16"
            );
        }
    }

    /// **MAJOR 1 (sc-11045 review): the `proj_out` clause is anchored to the trunk's FINAL head.**
    ///
    /// The 438× Dense measurement behind the clause is SANA's single top-level `proj_out`. A bare
    /// `contains("proj_out")` also matches per-block layers that are the spike's *explicitly benign*
    /// W4A4 class — LTX's remapped `ff.proj_out` (its FF output, ×48 blocks) and Flux/Chroma's
    /// `single_transformer_blocks.{i}.proj_out` (fused attn+MLP output, ×38). Those must stay W4A4.
    #[test]
    fn final_proj_anchor_does_not_fire_on_per_block_names() {
        // The trunk head — fires (measured Dense, 438×).
        for name in ["proj_out", "proj_out.weight", "transformer.proj_out"] {
            assert_eq!(
                ActPrecision::for_outlier_layer(name),
                ActPrecision::W4A16,
                "{name} is the trunk's final output projection → W4A16"
            );
        }
        // Per-block `proj_out` spellings — the benign compute bulk, must NOT fire.
        for name in [
            "transformer_blocks.5.ff.proj_out", // LTX FF output (ff.net.2 remapped)
            "transformer_blocks.5.ff.proj_out.weight",
            "transformer_blocks.5.audio_ff.proj_out",
            "single_transformer_blocks.3.proj_out", // Flux/Chroma fused attn+MLP output
            "single_transformer_blocks.37.proj_out.weight",
        ] {
            assert_eq!(
                ActPrecision::for_outlier_layer(name),
                ActPrecision::W4A4,
                "{name} is a PER-BLOCK projection (the spike's benign W4A4 compute bulk), not the \
                 final head — the `proj_out` clause must not fire on it"
            );
        }
        // The same semantic layer under two spellings must classify the same way. This is the
        // concrete bug: LTX renames `ff.net.2` → `ff.proj_out`, and the bare substring made an
        // identical FF output projection flip class on spelling alone.
        assert_eq!(
            ActPrecision::for_outlier_layer("transformer_blocks.4.ff.net.2"),
            ActPrecision::for_outlier_layer("transformer_blocks.4.ff.proj_out"),
            "LTX's `ff.proj_out` IS `ff.net.2` renamed — the policy must not classify it differently"
        );
        // The explicit provider flag is authoritative: a provider that knows a name is its head says
        // so, and is believed even when the name anchor cannot tell.
        assert_eq!(
            ActPrecision::for_outlier_layer_with("some_head.out_layer", true),
            ActPrecision::W4A16
        );
        assert_eq!(
            ActPrecision::for_outlier_layer_with("some_head.out_layer", false),
            ActPrecision::W4A4
        );
    }

    /// The explicit mixed-precision partition (sc-11044 AC 2) over a Sana-1.6B-DiT-style layer list:
    /// **self-attn + FF** (the compute bulk) → W4A4; caption_projection, the **whole cross-attention
    /// block**, `proj_out`, and the first & last blocks → W4A16.
    ///
    /// The cross-attn assignments were widened in sc-11045 from real measured activations — see
    /// [`ActPrecision::for_outlier_layer`].
    #[test]
    fn partition_layers_matches_spike_policy() {
        let layers = [
            // benign compute bulk (W4A4) — self-attn + FF, exactly what the spike specified.
            "transformer_blocks.4.attn1.to_q",
            "transformer_blocks.4.attn1.to_k",
            "transformer_blocks.4.attn1.to_v",
            "transformer_blocks.4.attn1.to_out.0",
            "transformer_blocks.4.ff.net.0.proj",
            "transformer_blocks.4.ff.net.2",
            // outlier class (W4A16)
            "transformer.caption_projection.linear_1",
            "transformer_blocks.4.attn2.to_q", // cross-attn Q — Dense on real activations
            "transformer_blocks.4.attn2.to_k", // cross-attn K
            "transformer_blocks.4.attn2.to_v", // cross-attn V
            "transformer_blocks.4.attn2.to_out.0", // cross-attn OUT — Dense on real activations
            "transformer_blocks.0.attn1.to_q", // first block
            "final_block.proj",                // last block
        ];
        let part = ActPrecision::partition_layers(layers);
        assert_eq!(part.n_w4a16, 7, "7 outlier-class projections must be W4A16");
        assert_eq!(part.n_w4a4, 6, "6 benign projections must be W4A4");
        // Spot-check assignments across both classes.
        let by_name: std::collections::HashMap<_, _> = part
            .assignments
            .iter()
            .map(|(n, p)| (n.as_str(), *p))
            .collect();
        assert_eq!(by_name["transformer_blocks.4.attn1.to_q"], ActPrecision::W4A4);
        assert_eq!(by_name["transformer_blocks.4.attn2.to_k"], ActPrecision::W4A16);
        assert_eq!(by_name["transformer_blocks.4.attn2.to_q"], ActPrecision::W4A16);
        assert_eq!(
            by_name["transformer_blocks.4.attn2.to_out.0"],
            ActPrecision::W4A16
        );
        assert_eq!(by_name["transformer.caption_projection.linear_1"], ActPrecision::W4A16);
    }

    /// Footprint accounting: the NVFP4 footprint is far below the bf16 size (~4.5 vs 16 bits/weight).
    #[test]
    fn footprint_is_far_below_bf16() -> Result<()> {
        let dev = Device::Cpu;
        // A realistically large weight so the 128-row scale-atom padding overhead is negligible.
        let (out_dim, in_dim) = (1024usize, 4096usize);
        let mut seed = 0x5EED_1234u64;
        let w = Tensor::from_vec(
            (0..out_dim * in_dim).map(|_| prng(&mut seed) * 0.2).collect::<Vec<_>>(),
            (out_dim, in_dim),
            &dev,
        )?;
        let lin = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A4)?;
        let nvfp4 = lin.nvfp4_footprint_bytes();
        let bf16 = lin.bf16_footprint_bytes();
        // ~4.5 bits/weight vs 16 → ratio ~0.281; assert comfortably below a third of bf16.
        let ratio = nvfp4 as f64 / bf16 as f64;
        assert!(
            ratio < 0.32,
            "NVFP4 footprint {nvfp4} B / bf16 {bf16} B = {ratio:.3}, expected ≈0.28 (~4.5 bit)"
        );
        // And close to the 4.5-bit ideal (nibble 0.5 B/wt + scale 1 B/16 wt = 0.5625 B/wt).
        let per_wt = nvfp4 as f64 / (out_dim * in_dim) as f64;
        assert!(
            (per_wt - 0.5625).abs() < 0.02,
            "bytes/weight {per_wt:.4}, expected ≈0.5625 (4.5 bits)"
        );
        Ok(())
    }
}
