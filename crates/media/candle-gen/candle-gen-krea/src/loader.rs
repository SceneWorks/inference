//! Weight loading for the Krea 2 DiT + Qwen3-VL-4B condition encoder — a thin shape-inferring wrapper
//! over candle's [`MmapedSafetensors`], mirroring `candle-gen-boogu`/`candle-gen-ideogram`'s `Weights`
//! interface so the port stays a near-1:1 translation of `mlx-gen-krea` (whose `Weights::from_dir`
//! loads the identity-keyed diffusers checkpoint directly). [`linear`] builds a [`Linear`] from the
//! actual `{base}.weight` (+ optional `{base}.bias`) tensor shapes.
//!
//! **Packed-tier detect (sc-9411).** When a component dir is an MLX-packed q4/q8 snapshot
//! (`SceneWorks/krea-2-turbo-mlx`, group size 64), each quantized projection is stored as the triple
//! `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`, and the component `config.json`
//! carries a `quantization: { bits, group_size }` block ([`candle_gen::quant::PackedConfig`]).
//! [`Weights::from_dir`] reads that block; [`linear_detect`] / [`embedding_detect`] then packed-**detect**
//! the `.scales` sibling and build the quantized module straight from the packed parts through the shared
//! group-size-aware loaders (no dense staging — see [`crate::quant`]). Absent the block / `.scales`, the
//! dense path is unchanged.
//!
//! **Adapter compose (sc-9411).** The DiT's `set_overlay` (adapter merge, sc-7836) installs dense
//! CPU-side weights that take priority over the mmap. [`linear_detect`] checks the **overlay first**: a
//! projection the adapter merge targeted resolves to its merged **dense** weight (the merge
//! reconstructs the dense base from the packed parts before folding, [`crate::adapters`]), while an
//! untargeted packed projection stays packed. So the packed base and the dense adapter overlay compose.
//! [`dequant_packed_base`] is the reconstruction the merge uses to build a mergeable dense base off the
//! packed triple.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::{Embedding, Linear};
use candle_gen::quant::{
    dequant_mlx_q4_reference_gs, dequant_mlx_q8_gs, mlx_packed_bits_gs, Int8Context, Nvfp4Linear,
    PackedConfig,
};

use crate::nvfp4_dit::{DitPlan, Nvfp4Proj, Nvfp4Quant, ProbedProj};
use crate::quant::{QEmbedding, QLinear};

/// An mmaped component-directory of `.safetensors`, loading tensors at a fixed compute dtype.
///
/// An optional in-memory `overlay` (installed by `set_overlay`) takes priority
/// over the mmap for the keys it holds — the inference-side LoRA/LoKr adapter merge (sc-7836) folds its
/// deltas into the targeted dense weights on the CPU in f32, then installs them here so
/// [`crate::transformer::Krea2Transformer::load`] reads the **merged** weight without re-mmapping or
/// touching the untargeted bulk of the model. Overlay tensors are stored CPU-side (where the merge runs)
/// and moved to `device` / cast to the requested dtype on read, exactly like the mmap path.
pub struct Weights {
    st: MmapedSafetensors,
    device: Device,
    dtype: DType,
    overlay: HashMap<String, Tensor>,
    /// The component's `quantization` manifest, `Some` for a packed q4/q8 tier (carries the group size
    /// the packed shapes can't disambiguate), `None` for a dense bf16 tier.
    packed: Option<PackedConfig>,
    /// True for **any native-mmdit-keyed** checkpoint (sc-9300 INT8-ConvRot *and* sc-14022 dense-bf16
    /// single file): the file stores the *reference* tensor names, so every diffusers-key lookup is
    /// translated to its native counterpart ([`convrot_diffusers_to_native`], optionally under a
    /// [`native_prefix`](Weights::native_prefix)) at read time in [`resolve`](Weights::resolve). This is
    /// the **key-remap** concern only — it is deliberately independent of the ConvRot int8/rotation legs
    /// ([`convrot`](Weights::convrot)) so a plain dense-bf16 native file (e.g. the community
    /// `kreamania_variant5` merge) reads through the remap as ordinary dense bf16 with **no** inverse
    /// rotation or int8 dequant (which would corrupt it — sc-14022).
    native_keys: bool,
    /// A namespace every native key sits under, detected at load ([`detect_native_prefix`]). Empty for the
    /// ComfyUI INT8-ConvRot export (its keys are bare `blocks.N.…`); `"model.diffusion_model."` for the
    /// community dense single file, whose DiT is namespaced under that prefix. Prepended to the remapped
    /// native key in [`resolve`](Weights::resolve). Only consulted when [`native_keys`](Weights::native_keys).
    native_prefix: String,
    /// True **only** for a community **INT8-ConvRot** checkpoint (sc-9300): its quantized projections carry
    /// a `{native_base}.weight_scale` + int8 `.weight` whose stored weight is the *rotated* `W·R`, so
    /// [`linear_detect`] engages the int8 IGEMM + the online group-256 Hadamard **rotation** legs
    /// ([`crate::quant::ConvRotInt8`]). Split out from [`native_keys`](Weights::native_keys) (sc-14022): a
    /// dense-bf16 native file is `native_keys: true, convrot: false` — remap ON, rotation/int8 OFF.
    convrot: bool,
    /// True only for a native-keyed, non-rotated ComfyUI
    /// `{"format":"int8_tensorwise","per_row":true}` checkpoint (sc-14023).
    ///
    /// Unlike [`convrot`](Weights::convrot), these codes store the canonical `W`, not `W·R`.
    /// Reads therefore dequantize each projection as `codes.i8 * weight_scale` and never rotate
    /// either the weight or the activation. [`from_native_file`](Self::from_native_file) sets this
    /// only after validating every descriptor and per-row scale from the safetensors data section.
    plain_int8: bool,
    /// **One** cuBLASLt handle shared by every INT8-ConvRot projection loaded from this weight set
    /// (sc-12301) — see [`Weights::int8_context`]. `OnceLock` because [`linear_detect`] takes `&Weights`
    /// and the handle must be built at most once for the whole trunk, on first int8 projection.
    ///
    /// This weight set is the right owner and the right *lifetime*: the handle is a per-device compute
    /// resource for exactly these weights, so it dies when they do — no process-global outliving an
    /// unloaded model. `pipeline::load_components_convrot` seeds it up front via
    /// [`with_int8_context`](Weights::with_int8_context) so the sm_89 floor probe's handle *becomes* the
    /// trunk's shared handle instead of being built and thrown away.
    int8: OnceLock<Int8Context>,
}

impl Weights {
    /// mmap every `*.safetensors` in `dir` (sorted; later files win on name collision), reading the
    /// component `config.json`'s `quantization` block (if any) for the packed-tier path.
    pub fn from_dir(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let files = candle_gen::sorted_safetensors(dir, "krea")
            .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
        // SAFETY: read-only mmap of weight files; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::multi(&files)? };
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
            packed: read_packed_config(dir)?,
            native_keys: false,
            native_prefix: String::new(),
            convrot: false,
            plain_int8: false,
            int8: OnceLock::new(),
        })
    }

    /// mmap a single `.safetensors` file (used by the committed parity fixtures). Dense-only (no
    /// packed config), so the packed path is never taken for a single-file fixture. Diffusers-keyed —
    /// for a **native-mmdit-keyed** dense single file use [`from_native_file`](Self::from_native_file).
    pub fn from_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: read-only mmap of a weight file; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::new(path)? };
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
            packed: None,
            native_keys: false,
            native_prefix: String::new(),
            convrot: false,
            plain_int8: false,
            int8: OnceLock::new(),
        })
    }

    /// mmap a **single-file INT8-ConvRot checkpoint** (sc-9300) — the ComfyUI-exported, native-mmdit-keyed
    /// `krea2_turbo_int8_convrot.safetensors`. `convrot` is set, so every diffusers-key lookup is
    /// translated to the native key ([`convrot_diffusers_to_native`]) at read time and quantized
    /// projections are int8 (per-output-row `.weight_scale`). Dense bf16 tensors (`first`/`last`/`tmlp`
    /// /`tproj`/`txtfusion`/`txtmlp` + norms) load unchanged through the remap.
    pub fn from_convrot_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: read-only mmap of a weight file; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::new(path)? };
        validate_convrot_descriptors(&st)?;
        let native_prefix = detect_native_prefix(&st);
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
            packed: None,
            native_keys: true,
            native_prefix,
            convrot: true,
            plain_int8: false,
            int8: OnceLock::new(),
        })
    }

    /// mmap a **single-file native-mmdit-keyed checkpoint** (sc-14022/sc-14023) — either a dense-bf16
    /// community merge (for example `kreamania_variant5.safetensors`) or the non-rotated
    /// int8-per-row variant (`kreamania_variant4.safetensors`). The DiT is stored under the reference
    /// tensor names, typically namespaced beneath `model.diffusion_model.`.
    ///
    /// `native_keys` is set so every diffusers-key lookup is translated to its native key
    /// ([`convrot_diffusers_to_native`], under the auto-detected namespace prefix) at
    /// read time — the DiT reads as **ordinary dense bf16** through the remap. `convrot` is **false**: no
    /// `convrot` is **false** for both forms. For int8, every real `.comfy_quant` descriptor must say
    /// `format=int8_tensorwise`, `per_row=true`, and omit `convrot`; every code tensor must be I8 and
    /// have an F32 `[out]` or `[out,1]` scale. The constructor validates that data before any
    /// dequantization. A present `convrot` field is rejected here and belongs to
    /// [`from_convrot_file`](Self::from_convrot_file), preventing the old group-size-256 fallback from
    /// silently rotating a plain file.
    pub fn from_native_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        // SAFETY: read-only mmap of a weight file; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::new(path)? };
        let native_prefix = detect_native_prefix(&st);
        let plain_int8 = validate_plain_int8_tensorwise(&st)?;
        Ok(Self {
            st,
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
            packed: None,
            native_keys: true,
            native_prefix,
            convrot: false,
            plain_int8,
            int8: OnceLock::new(),
        })
    }

    /// Install a pre-built [`Int8Context`] as this weight set's shared handle (sc-12301).
    ///
    /// The seam for scope 5 of the story: `pipeline::ensure_int8_floor` must build a cuBLASLt handle
    /// anyway to read the device's compute capability against the sm_89 floor, so it hands that handle
    /// here instead of dropping it — the probe *becomes* the trunk's shared handle. Absent this call,
    /// [`int8_context`](Self::int8_context) simply builds one lazily on the first int8 projection.
    ///
    /// Takes `self` by value (called at construction, before any projection reads the cell), so seeding
    /// a fresh `OnceLock` cannot lose a race with a lazy build.
    pub fn with_int8_context(mut self, ctx: Int8Context) -> Self {
        let cell = OnceLock::new();
        // Infallible: the cell was created one line above and nothing else holds it.
        let _ = cell.set(ctx);
        self.int8 = cell;
        self
    }

    /// The **one** [`Int8Context`] every INT8-ConvRot projection from this weight set shares (sc-12301),
    /// built on first use if [`with_int8_context`](Self::with_int8_context) did not seed it.
    ///
    /// This is the fix for the defect the story names: `QLinear::convrot_int8` built a fresh cuBLASLt
    /// handle — and its eager 32 MiB workspace — for *every* int8 projection, so a ConvRot DiT's ~224 of
    /// them carried ~7 GiB of duplicated scratch that a weights-only footprint sum cannot see.
    ///
    /// Errors (rather than caching a failure) if the handle cannot be built on a CUDA device, so the
    /// F-121 / sc-11208 typed-error-at-load property survives the move to a shared handle: the error
    /// surfaces from the first `linear_detect` that reaches an int8 projection, still inside load, where
    /// `?` is available.
    pub fn int8_context(&self) -> Result<&Int8Context> {
        if let Some(ctx) = self.int8.get() {
            return Ok(ctx);
        }
        // Built OUTSIDE the cell so a failure propagates instead of being cached as a poisoned context
        // (`OnceLock` has no stable `get_or_try_init`). A lost race here just drops the loser's handle;
        // the winner's is the one every projection then shares.
        let ctx = Int8Context::new(&self.device)?;
        Ok(self.int8.get_or_init(|| ctx))
    }

    /// Whether this checkpoint applies the **int8 + online-rotation** legs, i.e. it is a genuine
    /// **INT8-ConvRot** export (sc-9300). This is the narrow ConvRot flag (sc-14022): it is **false** for
    /// a dense-bf16 native single file, which is native-keyed but never rotated/int8. Gate the int8
    /// `linear_detect` arm on this, NOT on [`uses_native_keys`](Self::uses_native_keys).
    pub fn is_convrot(&self) -> bool {
        self.convrot
    }

    /// Whether this native checkpoint uses the descriptor-validated, non-rotated int8-per-row
    /// convention added in sc-14023.
    pub fn is_plain_int8(&self) -> bool {
        self.plain_int8
    }

    /// Whether this checkpoint is **native-mmdit-keyed** — its tensors are stored under the reference
    /// names, so `resolve` remaps every diffusers-key lookup to its native counterpart
    /// (sc-14022). True for BOTH the INT8-ConvRot export ([`from_convrot_file`](Self::from_convrot_file))
    /// and the dense-bf16 single file ([`from_native_file`](Self::from_native_file)). Distinct from
    /// [`is_convrot`](Self::is_convrot) (the int8/rotation legs). Coverage/shape validation
    /// ([`crate::convert`]) branches on this so a native file is validated by resolving each expected
    /// diffusers key to a present native tensor rather than a literal key diff.
    pub fn uses_native_keys(&self) -> bool {
        self.native_keys
    }

    /// Resolve a **diffusers** key to the actual on-disk key: for a native-mmdit-keyed checkpoint
    /// (sc-9300 ConvRot or sc-14022 dense) the native key ([`convrot_diffusers_to_native`]) under the
    /// detected [`native_prefix`](Self::native_prefix); else the key unchanged. A native key with no
    /// diffusers counterpart resolves to itself, so the subsequent mmap load errors on the
    /// genuinely-missing tensor (as it would for a truncated dense download) rather than silently
    /// succeeding.
    fn resolve(&self, name: &str) -> String {
        if self.native_keys {
            match convrot_diffusers_to_native(name) {
                Some(native) => format!("{}{native}", self.native_prefix),
                None => name.to_string(),
            }
        } else {
            name.to_string()
        }
    }

    /// Load `name` at the component dtype — from the `overlay` if present
    /// (adapter-merged weight), else the mmap (native-key-resolved for a ConvRot checkpoint).
    pub fn get(&self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.overlay.get(name) {
            return t.to_device(&self.device)?.to_dtype(self.dtype);
        }
        if self.plain_int8 {
            let resolved = self.resolve(name);
            if self
                .st
                .get(&resolved)
                .is_ok_and(|view| view.dtype() == ::safetensors::Dtype::I8)
            {
                return self.dequant_plain_int8(&resolved, self.dtype);
            }
        }
        self.st
            .load(&self.resolve(name), &self.device)?
            .to_dtype(self.dtype)
    }

    /// Load `name` preserving its on-disk dtype (e.g. int `input_ids` in a parity fixture). The overlay
    /// only ever holds merged DiT weights (never raw-dtype tensors), so this stays the mmap path.
    pub fn get_raw(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)
    }

    /// Load `name` at its **native** stored dtype (no cast) on the component device — used for the
    /// packed triple's u32 codes (casting would reinterpret the bit-packed nibbles) and the ConvRot
    /// int8 `.weight` codes. The overlay only holds merged dense DiT weights, so this stays the mmap
    /// path (native-key-resolved for a ConvRot checkpoint).
    pub fn get_native(&self, name: &str) -> Result<Tensor> {
        self.st.load(&self.resolve(name), &self.device)
    }

    /// Load `name` forcing f32 (the `+1` norm weights and other precision-sensitive scalars) — from the
    /// overlay if present, else the mmap (native-key-resolved for a ConvRot checkpoint).
    pub fn get_f32(&self, name: &str) -> Result<Tensor> {
        if let Some(t) = self.overlay.get(name) {
            return t.to_device(&self.device)?.to_dtype(DType::F32);
        }
        if self.plain_int8 {
            let resolved = self.resolve(name);
            if self
                .st
                .get(&resolved)
                .is_ok_and(|view| view.dtype() == ::safetensors::Dtype::I8)
            {
                return self.dequant_plain_int8(&resolved, DType::F32);
            }
        }
        self.st
            .load(&self.resolve(name), &self.device)?
            .to_dtype(DType::F32)
    }

    /// Load `name` onto the **CPU** at its on-disk dtype. Used by the inference-side adapter merge
    /// ([`crate::adapters`]), which reconstructs LoRA/LoKr deltas on the CPU (matching the CPU-loaded
    /// adapter factors) and folds them into the base weight before installing the [`overlay`](Weights::set_overlay).
    pub(crate) fn get_cpu(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &Device::Cpu)
    }

    /// Install an in-memory `overlay` of (CPU-resident) tensors that take priority over the mmap for the
    /// keys they cover — the adapter-merged dense weights (sc-7836). Replaces any prior overlay.
    pub(crate) fn set_overlay(&mut self, overlay: HashMap<String, Tensor>) {
        self.overlay = overlay;
    }

    pub fn contains(&self, name: &str) -> bool {
        self.overlay.contains_key(name) || self.st.get(&self.resolve(name)).is_ok()
    }

    /// Whether a **raw** (already-native) key is present on-disk, bypassing the ConvRot diffusers→native
    /// remap — used to detect a ConvRot int8 projection's `{native_base}.weight_scale` sibling (sc-9300),
    /// which is a native-only key with no diffusers counterpart in the remap.
    fn contains_native(&self, name: &str) -> bool {
        self.st.get(name).is_ok()
    }

    /// Load a **raw** (already-native) key forcing f32, bypassing the diffusers→native remap — the
    /// ConvRot per-output-row `weight_scale` (sc-9300).
    fn get_native_f32(&self, name: &str) -> Result<Tensor> {
        self.st.load(name, &self.device)?.to_dtype(DType::F32)
    }

    /// Load an INT8-ConvRot weight's int8 codes as an `I64` `[out, in]` tensor (sc-9300). `diffusers_key`
    /// is the diffusers `{base}.weight`, resolved to its native key. candle's `DType` at our pin has **no
    /// I8 variant** (only U8/U32/I16/I32/I64), so `st.load` can't decode an `I8` tensor — this reads the
    /// raw `TensorView` bytes and reinterprets them as signed `i8 → i64` codes (the dtype the int8 stage
    /// narrows back down). A test fixture may store the codes as `I64` directly (safetensors save has no
    /// I8); that path loads through `st.load` unchanged.
    fn get_int8_codes(&self, diffusers_key: &str) -> Result<Tensor> {
        let native = self.resolve(diffusers_key);
        let view = self.st.get(&native)?;
        // Build the codes on the **CPU**: the caller (Int8Linear::from_per_channel_parts) stages them
        // to a resident native-`i8` device buffer (1 byte/elem), so materializing the wider I64 form on
        // the GPU first would 8× the VRAM (a 12B DiT's 224 projections OOM). The CPU I64 is transient.
        match view.dtype() {
            // Real ComfyUI export: raw I8 bytes reinterpreted as signed codes (candle can't decode I8).
            ::safetensors::Dtype::I8 => {
                let shape = view.shape().to_vec();
                let codes: Vec<i64> = view.data().iter().map(|&b| b as i8 as i64).collect();
                Tensor::from_vec(codes, shape, &Device::Cpu)
            }
            // Test / any-int fixture: load whatever integer dtype it is, then widen to I64.
            _ => self.st.load(&native, &Device::Cpu)?.to_dtype(DType::I64),
        }
    }

    /// Dequantize one already-validated plain-int8 projection as
    /// `W[out, in] = codes.i8[out, in] * weight_scale[out]`, with no Hadamard rotation.
    fn dequant_plain_int8(&self, native_weight: &str, dtype: DType) -> Result<Tensor> {
        let view = self.st.get(native_weight)?;
        let shape = view.shape().to_vec();
        let rows = shape[0];
        let codes: Vec<i64> = view.data().iter().map(|&byte| byte as i8 as i64).collect();
        let codes = Tensor::from_vec(codes, shape, &Device::Cpu)?
            .to_device(&self.device)?
            .to_dtype(DType::F32)?;

        let base = native_weight.strip_suffix(".weight").ok_or_else(|| {
            candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: I8 tensor `{native_weight}` is not a `.weight`"
            ))
        })?;
        let scale = self
            .st
            .load(&format!("{base}.weight_scale"), &self.device)?
            .to_dtype(DType::F32)?;
        let scale = match scale.dims() {
            [_] => scale.reshape((rows, 1))?,
            [_, 1] => scale,
            dims => {
                return Err(candle_gen::candle_core::Error::Msg(format!(
                    "krea plain int8: `{base}.weight_scale` has invalid shape {dims:?}"
                )));
            }
        };
        codes.broadcast_mul(&scale)?.to_dtype(dtype)
    }

    /// Read the ConvRot `convrot_groupsize` (the regular-Hadamard order `R` was folded at) from a
    /// projection's native `{native_base}.comfy_quant` descriptor — a small U8 JSON blob
    /// (`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`) the ComfyUI export writes
    /// alongside each quantized weight (sc-9601). `None` when the blob is absent or lacks the field (an
    /// older/plain int8 export); the caller then falls back to the checkpoint default (256).
    fn get_convrot_groupsize(&self, native_base: &str) -> Option<usize> {
        let view = self.st.get(&format!("{native_base}.comfy_quant")).ok()?;
        let j: serde_json::Value = serde_json::from_slice(view.data()).ok()?;
        j.get("convrot_groupsize")?.as_u64().map(|g| g as usize)
    }

    /// All tensor keys in the component (for architecture validation). For a ConvRot checkpoint these
    /// are the **native** keys as stored; [`crate::convert::validate_transformer`] uses the ConvRot arm
    /// (diffusers-key resolve) rather than diffing these directly.
    pub fn keys(&self) -> Vec<String> {
        self.st.tensors().into_iter().map(|(k, _)| k).collect()
    }

    /// On-disk shape of `name` (for architecture validation), or `None` if absent (native-key-resolved
    /// for a ConvRot checkpoint). The overlay never changes a weight's shape, so the mmap is
    /// authoritative.
    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.st
            .get(&self.resolve(name))
            .ok()
            .map(|v| v.shape().to_vec())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The MLX `quantization` block when this component is a packed q4/q8 tier, else `None`.
    pub fn packed(&self) -> Option<PackedConfig> {
        self.packed
    }

    /// Whether the [`overlay`](Weights::set_overlay) holds a (dense, adapter-merged) tensor for `name`.
    /// The packed detectors read this first so an adapter-targeted projection resolves to its merged
    /// dense weight rather than the packed triple (sc-9411 adapter compose).
    fn overlay_has(&self, name: &str) -> bool {
        self.overlay.contains_key(name)
    }

    /// The **dense** CPU base weight for an adapter merge target `weight_key` (`{base}.weight`) — the
    /// adapter-compose seam (sc-9411). On a dense tier this is the on-disk weight loaded onto the CPU
    /// (exactly [`Self::get_cpu`]). On a **packed** tier whose `{base}.scales` sibling is present, the
    /// weight is u32 codes, so the dense grid is reconstructed from the packed triple at the tier's
    /// group size ([`dequant_packed_base`], f32) — the mergeable base the LoRA/LoKr delta folds into.
    /// The resulting merged weight is installed in the overlay, so [`linear_detect`] then loads it
    /// dense (the packed base stays packed for untargeted projections).
    pub(crate) fn get_cpu_merge_base(&self, weight_key: &str) -> Result<Tensor> {
        // Resolve the diffusers key to its on-disk name — identity except on a native-keyed INT8-ConvRot
        // checkpoint (sc-9300), where a dense baseline weight a diff-patch folds into (e.g.
        // `text_fusion.projector` → `txtfusion.projector.weight`) would otherwise 404. On the MLX-packed
        // path resolution is a no-op (that tier is diffusers-keyed), so this is behavior-preserving there.
        let key = self.resolve(weight_key);
        if let Some(base) = key.strip_suffix(".weight") {
            let scales_key = format!("{base}.scales");
            if let (Some(cfg), true) = (self.packed, self.st.get(&scales_key).is_ok()) {
                let wq = self.st.load(&key, &Device::Cpu)?;
                let scales = self
                    .st
                    .load(&scales_key, &Device::Cpu)?
                    .to_dtype(DType::F32)?;
                let biases = self
                    .st
                    .load(&format!("{base}.biases"), &Device::Cpu)?
                    .to_dtype(DType::F32)?;
                return dequant_packed_base(&wq, &scales, &biases, cfg.group_size as usize);
            }
        }
        self.get_cpu(&key)
    }

    /// The on-device base weight for a **dense/composable** projection ([`linear`]) at the component
    /// dtype. On a dense tier this is exactly [`Self::get`]. On a **packed** q4/q8 tier whose
    /// `{base}.scales` sibling is present (and the weight is NOT adapter-merged into the overlay), the
    /// stored `{base}.weight` is u32 codes — casting them would reinterpret the bit-packed nibbles — so
    /// the dense grid is reconstructed from the packed triple ([`dequant_packed_base`], f32) and moved to
    /// the component device/dtype. This lets the composable [`KreaTrainDit`](crate::KreaTrainDit) (the
    /// control / train forward, which loads every projection via dense [`linear`], not the packed-detecting
    /// [`linear_detect`]) consume a packed base by dequantizing on load — the mergeable-base seam
    /// [`get_cpu_merge_base`](Self::get_cpu_merge_base) already uses, minus the CPU pin (spike:
    /// packed-base pose-control lane).
    pub(crate) fn get_dense_or_dequant(&self, weight_key: &str) -> Result<Tensor> {
        // An adapter-merged dense weight in the overlay wins (mirrors `get`'s overlay-first read).
        if self.overlay.contains_key(weight_key) {
            return self.get(weight_key);
        }
        if let Some(base) = weight_key.strip_suffix(".weight") {
            let scales_key = format!("{base}.scales");
            if let (Some(cfg), true) = (self.packed, self.st.get(&scales_key).is_ok()) {
                let wq = self.st.load(weight_key, &Device::Cpu)?;
                let scales = self
                    .st
                    .load(&scales_key, &Device::Cpu)?
                    .to_dtype(DType::F32)?;
                let biases = self
                    .st
                    .load(&format!("{base}.biases"), &Device::Cpu)?
                    .to_dtype(DType::F32)?;
                let dense = dequant_packed_base(&wq, &scales, &biases, cfg.group_size as usize)?;
                return dense.to_device(&self.device)?.to_dtype(self.dtype);
            }
        }
        self.get(weight_key)
    }
}

/// Validate the real data-section contract for a non-rotated ComfyUI int8-tensorwise checkpoint.
///
/// The app-side detector can only see header dtypes and key names. That is enough to put a file in
/// the int8-per-row bucket, but not enough to prove that its U8 descriptor actually says
/// `int8_tensorwise`, that `per_row` is true, or that the scale is one F32 value per output row.
/// Perform those checks here, before any projection is dequantized. A `convrot` field at all selects
/// the distinct rotated convention and is rejected from this constructor rather than falling back to
/// the old implicit group-size-256 ConvRot path.
fn validate_plain_int8_tensorwise(st: &MmapedSafetensors) -> Result<bool> {
    let tensors = st.tensors();
    let int8_weights: Vec<String> = tensors
        .iter()
        .filter(|(_, view)| view.dtype() == ::safetensors::Dtype::I8)
        .map(|(name, _)| name.clone())
        .collect();
    let descriptors: Vec<String> = tensors
        .iter()
        .filter(|(name, _)| name.ends_with(".comfy_quant"))
        .map(|(name, _)| name.clone())
        .collect();

    if int8_weights.is_empty() {
        if descriptors.is_empty() {
            return Ok(false);
        }
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "krea plain int8: found {} `.comfy_quant` descriptor(s) but no I8 weight tensors",
            descriptors.len()
        )));
    }

    for weight_key in &int8_weights {
        let Some(base) = weight_key.strip_suffix(".weight") else {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: I8 tensor `{weight_key}` is not a projection `.weight`"
            )));
        };
        let weight = st.get(weight_key)?;
        let [rows, _cols] = weight.shape() else {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{weight_key}` must be rank-2 [out,in], got {:?}",
                weight.shape()
            )));
        };

        let descriptor_key = format!("{base}.comfy_quant");
        let descriptor = st.get(&descriptor_key).map_err(|_| {
            candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{weight_key}` is missing `{descriptor_key}`"
            ))
        })?;
        if descriptor.dtype() != ::safetensors::Dtype::U8 || descriptor.shape().len() != 1 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` must be a rank-1 U8 JSON blob"
            )));
        }
        let json: serde_json::Value =
            serde_json::from_slice(descriptor.data()).map_err(|error| {
                candle_gen::candle_core::Error::Msg(format!(
                    "krea plain int8: `{descriptor_key}` is not valid JSON: {error}"
                ))
            })?;
        if json.get("format").and_then(serde_json::Value::as_str) != Some("int8_tensorwise") {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` must declare format `int8_tensorwise`"
            )));
        }
        if json.get("per_row").and_then(serde_json::Value::as_bool) != Some(true) {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` must declare `per_row: true`"
            )));
        }
        if json.get("convrot").is_some() {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` contains `convrot`; route rotated checkpoints \
                 through the ConvRot loader"
            )));
        }

        let scale_key = format!("{base}.weight_scale");
        let scale = st.get(&scale_key).map_err(|_| {
            candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{weight_key}` is missing `{scale_key}`"
            ))
        })?;
        if scale.dtype() != ::safetensors::Dtype::F32 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{scale_key}` must be F32, got {:?}",
                scale.dtype()
            )));
        }
        if scale.shape() != [*rows] && scale.shape() != [*rows, 1] {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{scale_key}` must be [{rows}] or [{rows},1], got {:?}",
                scale.shape()
            )));
        }
    }

    for descriptor_key in descriptors {
        let base = descriptor_key
            .strip_suffix(".comfy_quant")
            .expect("filtered suffix");
        let weight_key = format!("{base}.weight");
        if !matches!(
            st.get(&weight_key),
            Ok(view) if view.dtype() == ::safetensors::Dtype::I8
        ) {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` does not describe an I8 `{weight_key}`"
            )));
        }
    }

    Ok(true)
}

/// Validate the other side of the descriptor gate: the existing ConvRot constructor may only accept
/// descriptors that explicitly opt into `convrot: true`. This closes the historical fallback where a
/// plain int8 file handed to the rotated entrypoint could inherit group size 256 and silently corrupt
/// its weights.
fn validate_convrot_descriptors(st: &MmapedSafetensors) -> Result<()> {
    let descriptors: Vec<(String, ::safetensors::tensor::TensorView<'_>)> = st
        .tensors()
        .into_iter()
        .filter(|(name, _)| name.ends_with(".comfy_quant"))
        .collect();
    if descriptors.is_empty() {
        return Err(candle_gen::candle_core::Error::Msg(
            "krea convrot: checkpoint has no `.comfy_quant` descriptor with `convrot: true`"
                .to_owned(),
        ));
    }
    for (descriptor_key, descriptor) in descriptors {
        if descriptor.dtype() != ::safetensors::Dtype::U8 || descriptor.shape().len() != 1 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea convrot: `{descriptor_key}` must be a rank-1 U8 JSON blob"
            )));
        }
        let json: serde_json::Value =
            serde_json::from_slice(descriptor.data()).map_err(|error| {
                candle_gen::candle_core::Error::Msg(format!(
                    "krea convrot: `{descriptor_key}` is not valid JSON: {error}"
                ))
            })?;
        if json.get("format").and_then(serde_json::Value::as_str) != Some("int8_tensorwise")
            || json.get("convrot").and_then(serde_json::Value::as_bool) != Some(true)
        {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea convrot: `{descriptor_key}` must declare `format: int8_tensorwise` and \
                 `convrot: true`"
            )));
        }
    }
    Ok(())
}

// ===================================================================================================
// INT8-ConvRot native-key remap (sc-9300)
// ===================================================================================================
//
// The community INT8-ConvRot checkpoint (`krea2_turbo_int8_convrot.safetensors`, a ComfyUI export) is
// **native-mmdit-keyed**, not diffusers-keyed like the published `krea/Krea-2-Turbo` this crate's DiT
// loads. The DiT `load()` / `validate_transformer` read diffusers keys (`transformer_blocks.N.attn.to_q`,
// `norm_q`, `ff.gate`, `norm1`, `time_mod_proj`, `img_in`, `final_layer.*`, `text_fusion.*.ff.*`); the
// ConvRot file stores the *reference* names (`blocks.N.attn.wq`, `qknorm.qnorm`, `mlp.gate`, `prenorm`,
// `tproj`, `first`, `last`, `tmlp`, `txtfusion.*.mlp.*`, `txtmlp`). So a ConvRot `Weights` translates
// every diffusers-key lookup to its native counterpart at read time — the DiT stays byte-for-byte the
// diffusers-key module tree, and only this remap + the int8 detect arm are ConvRot-aware.
//
// The map was validated exhaustively against the real 878-tensor header: all 430 diffusers keys map to
// a present native key, 224 of them to a quantized (`.weight_scale` sibling) projection, with no native
// key left uncovered (the format-spike remap, verified — see the sc-9300 PR).
//
// **Coherent as of sc-9601.** The remap + per-channel int8 loader (sc-9300) are correct but not enough:
// the stored int8 weight is the *rotated* `W·R` (dequantized `blocks.0.attn.wq` has cosine ≈ 0.07 with
// the canonical `to_q`), so reconstructing `X·Wᵀ` needs the matching **online activation rotation**
// `RHT(x)` — the regular-Hadamard (group 256) leg from arXiv 2512.03673 (clean-room from the paper +
// the `comfy_quant` descriptor). The ConvRot projection now applies it before the int8 IGEMM
// ([`crate::quant::ConvRotInt8`]), lifting the render from the sc-9300 NO-GO's noise (PSNR ≈ 8 dB) to
// coherent (verified cosine 0.99991 vs the f32 reference linear).

/// Translate a **diffusers** tensor key to the **native-mmdit** key the INT8-ConvRot checkpoint stores.
/// Returns `None` for a key with no native counterpart (a caller then errors on the missing tensor,
/// exactly as it would for a truncated dense download). Shapes line up 1:1 under this map — the only
/// reshapes (`time_mod_proj`/`scale_shift_table` flatten identically row-major) are done by the DiT.
pub fn convrot_diffusers_to_native(key: &str) -> Option<String> {
    // Top-level (non-block) tensors.
    let top = match key {
        "img_in.weight" => Some("first.weight"),
        "img_in.bias" => Some("first.bias"),
        "txt_in.norm.weight" => Some("txtmlp.0.scale"),
        "txt_in.linear_1.weight" => Some("txtmlp.1.weight"),
        "txt_in.linear_1.bias" => Some("txtmlp.1.bias"),
        "txt_in.linear_2.weight" => Some("txtmlp.3.weight"),
        "txt_in.linear_2.bias" => Some("txtmlp.3.bias"),
        "time_embed.linear_1.weight" => Some("tmlp.0.weight"),
        "time_embed.linear_1.bias" => Some("tmlp.0.bias"),
        "time_embed.linear_2.weight" => Some("tmlp.2.weight"),
        "time_embed.linear_2.bias" => Some("tmlp.2.bias"),
        "time_mod_proj.weight" => Some("tproj.1.weight"),
        "time_mod_proj.bias" => Some("tproj.1.bias"),
        "text_fusion.projector.weight" => Some("txtfusion.projector.weight"),
        "final_layer.linear.weight" => Some("last.linear.weight"),
        "final_layer.linear.bias" => Some("last.linear.bias"),
        "final_layer.norm.weight" => Some("last.norm.scale"),
        "final_layer.scale_shift_table" => Some("last.modulation.lin"),
        _ => None,
    };
    if let Some(t) = top {
        return Some(t.to_string());
    }
    // Per-block leaf remap (shared by single-stream `transformer_blocks` and the two text-fusion stacks).
    let leaf = |rest: &str| -> Option<&'static str> {
        Some(match rest {
            "attn.norm_q.weight" => "attn.qknorm.qnorm.scale",
            "attn.norm_k.weight" => "attn.qknorm.knorm.scale",
            "attn.to_q.weight" => "attn.wq.weight",
            "attn.to_k.weight" => "attn.wk.weight",
            "attn.to_v.weight" => "attn.wv.weight",
            "attn.to_out.0.weight" => "attn.wo.weight",
            "attn.to_gate.weight" => "attn.gate.weight",
            "ff.gate.weight" => "mlp.gate.weight",
            "ff.up.weight" => "mlp.up.weight",
            "ff.down.weight" => "mlp.down.weight",
            "norm1.weight" => "prenorm.scale",
            "norm2.weight" => "postnorm.scale",
            "scale_shift_table" => "mod.lin",
            _ => return None,
        })
    };
    // `transformer_blocks.N.<leaf>` → `blocks.N.<native-leaf>`.
    if let Some(rest) = key.strip_prefix("transformer_blocks.") {
        if let Some((idx, tail)) = rest.split_once('.') {
            if idx.chars().all(|c| c.is_ascii_digit()) {
                return leaf(tail).map(|nl| format!("blocks.{idx}.{nl}"));
            }
        }
    }
    // `text_fusion.{layerwise,refiner}_blocks.N.<leaf>` → `txtfusion.{...}.N.<native-leaf>`.
    if let Some(rest) = key.strip_prefix("text_fusion.") {
        for kind in ["layerwise_blocks.", "refiner_blocks."] {
            if let Some(after) = rest.strip_prefix(kind) {
                if let Some((idx, tail)) = after.split_once('.') {
                    if idx.chars().all(|c| c.is_ascii_digit()) {
                        return leaf(tail).map(|nl| format!("txtfusion.{}{idx}.{nl}", kind));
                    }
                }
            }
        }
    }
    None
}

/// Detect the namespace prefix every native DiT tensor sits under (sc-14022). The community dense single
/// file (`kreamania_variant5`) namespaces its whole DiT beneath `model.diffusion_model.`, whereas the
/// ComfyUI INT8-ConvRot export stores bare `blocks.N.…` keys. [`Weights::resolve`] prepends whatever this
/// returns to the remapped native key, so one native loader serves both layouts. Empty string ⇒ no prefix
/// (the ConvRot export, and the reason [`from_convrot_file`](Weights::from_convrot_file) is behavior-
/// preserving: it detects an empty prefix and remaps exactly as before).
fn detect_native_prefix(st: &MmapedSafetensors) -> String {
    const PREFIX: &str = "model.diffusion_model.";
    if st.tensors().iter().any(|(k, _)| k.starts_with(PREFIX)) {
        PREFIX.to_string()
    } else {
        String::new()
    }
}

/// Read `{dir}/config.json`'s `quantization` block: `Ok(Some(cfg))` for a packed tier, `Ok(None)` for
/// a dense tier (a genuinely-absent `config.json` — a single-file fixture — still loads dense).
///
/// A **present-but-corrupt** `config.json` (I/O error or malformed JSON — e.g. a partial download)
/// returns an `Err` naming the file rather than silently swallowing to the dense path, so a damaged
/// packed snapshot surfaces instead of loading the wrong (dense) tier with no diagnostic (sc-9426,
/// F-073 sibling — the `component_is_packed` twin in flux2). Mirrors boogu's `read_packed_config`
/// (sc-9410) and z-image's `component_is_packed` (sc-9408).
fn read_packed_config(dir: &Path) -> Result<Option<PackedConfig>> {
    let path = dir.join("config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        // No config.json at all → legitimate dense / single-file fixture tier.
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        // Present but unreadable (permissions, partial download) → surface, don't swallow.
        Err(e) => {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea: read {}: {e}",
                path.display()
            )))
        }
    };
    // Present but malformed JSON → corrupt snapshot, error rather than fall to dense.
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        candle_gen::candle_core::Error::Msg(format!(
            "krea: parse {} (corrupt snapshot?): {e}",
            path.display()
        ))
    })?;
    Ok(PackedConfig::from_config(&v))
}

/// Reconstruct the **dense** f32 grid a packed triple (`{base}.weight` u32 codes + `.scales` +
/// `.biases`) represents, at the tier's `group_size` — the adapter-merge base (sc-9411). The
/// `krea_2_raw` adapter merge folds its delta into this reconstructed dense weight (CPU, f32, matching
/// the trainer's math) and installs the result in the overlay, so the merged projection loads dense
/// while the untargeted bulk stays packed. Bit-width is inferred from the packed shapes (Q4 → the
/// lossless affine grid; Q8 → its exact grid), mirroring the shared `repack_packed_weight` dispatch.
pub fn dequant_packed_base(
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    let wq_cols = wq.dim(1)?;
    let s_cols = scales.dim(1)?;
    match mlx_packed_bits_gs(wq_cols, s_cols, group_size) {
        4 => dequant_mlx_q4_reference_gs(wq, scales, biases, group_size),
        8 => dequant_mlx_q8_gs(wq, scales, biases, group_size),
        b => Err(candle_gen::candle_core::Error::Msg(format!(
            "krea: unsupported MLX packed bit-width {b} (wq cols {wq_cols}, scales cols {s_cols}, \
             group {group_size})"
        ))),
    }
}

/// Build a [`Linear`] from `{base}.weight` (+ `{base}.bias` when `bias`), inferring in/out dims from
/// the stored tensor shape (`[out, in]`, PyTorch/HF convention).
pub fn linear(w: &Weights, base: &str, bias: bool) -> Result<Linear> {
    let weight = w.get_dense_or_dequant(&format!("{base}.weight"))?;
    let bias = if bias {
        Some(w.get(&format!("{base}.bias"))?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias))
}

/// **Packed-detecting** [`QLinear`] loader (sc-9411) with adapter-overlay priority. In order:
///
/// 1. **Overlay** (`{base}.weight` is adapter-merged): the merge already reconstructed a dense weight
///    (from the packed parts if the tier is packed, [`crate::adapters`]) and installed it, so load
///    that **dense** merged weight — a `Dense` `QLinear`. The packed base composes with the adapter.
/// 2. **Packed** (a packed tier + `{base}.scales` present, no overlay): build a `Packed` projection
///    straight from the MLX packed triple at the tier's group size — **no dense weight materialized**.
/// 3. **Dense** (otherwise): the exact [`linear`] behavior (`{base}.weight` [+ `{base}.bias`]).
///
/// `base` is the full dotted key prefix (e.g. `attn.to_out.0`), so the `.scales`/`.biases` siblings
/// survive any `to_out.0`-style nesting — build the base string first, then detect (the key-remap trap
/// the `linear_detect_fires_on_to_out_remap` test pins on the real Krea `to_out.0` layout).
pub fn linear_detect(w: &Weights, base: &str, bias: bool) -> Result<QLinear> {
    let weight_key = format!("{base}.weight");
    let scales_key = format!("{base}.scales");
    // (1) An adapter-merged dense weight in the overlay wins — load it dense (adapter compose).
    if w.overlay_has(&weight_key) {
        return Ok(QLinear::dense(linear(w, base, bias)?));
    }
    // (1.5) INT8-ConvRot (sc-9300 loader + sc-9601 rotation): a ConvRot checkpoint whose native
    // `{base}.weight_scale` sibling is present → build a per-output-channel int8 projection from the
    // stored int8 codes + row scale + the `convrot_groupsize` in the `comfy_quant` descriptor. Detect on
    // the *native* base derived from the diffusers `{base}.weight` remap. The stored codes are the
    // *rotated* weight `W·R`; the projection's forward applies the matching online `RHT(x)` so the GEMM
    // reconstructs `X·Wᵀ` (the sc-9601 fix that makes this consume path render coherently).
    if w.is_convrot() {
        if let Some(native_weight) = convrot_diffusers_to_native(&weight_key) {
            if let Some(native_base) = native_weight.strip_suffix(".weight") {
                let scale_key = format!("{native_base}.weight_scale");
                if w.contains_native(&scale_key) {
                    let w_i8 = w.get_int8_codes(&weight_key)?; // raw I8 → I64 codes
                    let scale = w
                        .get_native_f32(&scale_key)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    // The regular-Hadamard order the export rotated at (default 256 per the arXiv
                    // 2512.03673 ConvRot default / this checkpoint) when the descriptor is absent.
                    let group_size = w.get_convrot_groupsize(native_base).unwrap_or(256);
                    let dense_bias = if bias {
                        Some(w.get(&format!("{base}.bias"))?)
                    } else {
                        None
                    };
                    // Pass the model's resident COMPUTE device (where activations live), NOT
                    // `w_i8.device()` — the codes are CPU-materialized here to save VRAM, but the int8
                    // IGEMM leg must be built on the CUDA compute device (F-121 / sc-11208).
                    //
                    // `convrot_int8_in`, NOT `convrot_int8` (sc-12301): this runs once per int8
                    // projection (~224 on a ConvRot DiT), and the private-handle constructor would give
                    // each its own eager 32 MiB cuBLASLt workspace — ~7 GiB of duplicated scratch. The
                    // weight set owns one shared handle for the whole trunk.
                    return QLinear::convrot_int8_in(
                        w_i8,
                        scale,
                        group_size,
                        dense_bias,
                        w.device(),
                        w.int8_context()?,
                    );
                }
            }
        }
    }
    // (2) A packed tier with a `.scales` sibling → build straight from the packed parts.
    if let (Some(cfg), true) = (w.packed(), w.contains(&scales_key)) {
        let wq = w.get_native(&weight_key)?;
        let scales = w.get_f32(&scales_key)?;
        let biases = w.get_f32(&format!("{base}.biases"))?;
        let dense_bias = if bias {
            Some(w.get(&format!("{base}.bias"))?)
        } else {
            None
        };
        return QLinear::packed(&wq, &scales, &biases, dense_bias, cfg.group_size as usize);
    }
    // (3) Dense path unchanged.
    Ok(QLinear::dense(linear(w, base, bias)?))
}

/// [`linear_detect`] under an NVFP4 [`DitPlan`] (sc-12110, epic 11037) — the seam that lets the Krea
/// trunk serve one projection through [`candle_gen::quant::Nvfp4Linear`] instead of its dense/packed
/// baseline leg.
///
/// Three outcomes, in order:
///
/// 1. **NVFP4** (`plan.is_nvfp4()`): pack `{base}.weight` to NVFP4 and build an [`Nvfp4Linear`] at the
///    activation precision the plan assigns this layer ([`DitPlan::act_for_layer`], which derives the
///    [`crate::nvfp4_dit::LayerRole`] from the dotted key + the trunk's block count). Never fails on an
///    ineligible device or shape — [`Nvfp4Linear`] resolves the `sm_120` capability gate itself and
///    transparently serves dequant→bf16 (sc-11041), so this is safe to call on any backend.
/// 2. **Probed baseline** (a probe attached, no NVFP4): the exact [`linear_detect`] leg, wrapped to
///    record its input activation's outlier sparsity. This is how the partition gate measures the
///    trunk's *unperturbed* real activations; the stamped precision is what the **shipping mixed policy
///    would assign**, so a summary can cross measured-vs-assumed without re-deriving roles.
/// 3. **Baseline**: [`linear_detect`], byte-unchanged.
///
/// # The NVFP4 arm requires a dense (bf16) tier — by design
///
/// NVFP4 is packed from the **bf16 master weight**, exactly as the offline packer (sc-11040) would.
/// Packing from an already-quantized q4/q8 tier would measure NVFP4-of-Q4 — a double quantization whose
/// error is not the format's, and which would quietly corrupt SC#2's like-for-like comparison (NVFP4 vs
/// Q4, both from the same master). So a packed tier is a hard error here rather than a silent
/// `get_dense_or_dequant` round-trip.
pub fn linear_detect_planned(
    w: &Weights,
    base: &str,
    bias: bool,
    plan: &DitPlan,
) -> Result<QLinear> {
    if !plan.is_nvfp4() {
        let inner = linear_detect(w, base, bias)?;
        return Ok(match plan.probe() {
            // The stamped precision is the SHIPPING policy's verdict, not this (baseline) plan's — the
            // gate asks "does the class the policy assumed match the class the live model measures?".
            Some(probe) => QLinear::Probed(ProbedProj::new(
                inner,
                base,
                probe.clone(),
                DitPlan::nvfp4(Nvfp4Quant::Mixed)
                    .with_num_layers(plan.num_layers())
                    .act_for_layer(base),
            )),
            None => inner,
        });
    }
    if w.packed().is_some() {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "krea nvfp4: refusing to pack `{base}` from an already-quantized tier — NVFP4 must be \
             packed from the bf16 master (else SC#2 compares NVFP4-of-Q4 against Q4). Load the bf16 \
             snapshot for the NVFP4 lane."
        )));
    }
    let act = plan.act_for_layer(base);
    let weight = w.get(&format!("{base}.weight"))?;
    let dense_bias = if bias {
        Some(w.get(&format!("{base}.bias"))?)
    } else {
        None
    };
    let device = weight.device().clone();
    // sc-12274: build against the plan's ONE shared per-device cuBLASLt handle. `from_dense` would
    // construct a private handle here — and its eager 32 MiB workspace — for every one of the trunk's
    // 260 projections.
    let lin = Nvfp4Linear::from_dense_in(&weight, dense_bias, &device, act, plan.nvfp4_context())?;
    Ok(QLinear::Nvfp4(Nvfp4Proj::new(lin, base, plan, act)))
}

/// **Packed-detecting** [`QEmbedding`] loader (sc-9411): packed straight from the MLX triple when the
/// component is a packed tier and `{base}.scales` is present (dequantized to the component dtype — dtype
/// parity with the dense table), else a dense [`Embedding`] from `{base}.weight` (`hidden` inferred from
/// the stored `[vocab, hidden]` shape). The Krea Qwen3-VL TE keeps `embed_tokens` **dense** in the
/// hosted q4/q8 tiers, so today this takes the dense arm; the packed arm is the future-proof path (and
/// guards against a silent dense read of u32 codes should a tier ever pack the table).
pub fn embedding_detect(w: &Weights, base: &str) -> Result<QEmbedding> {
    let scales_key = format!("{base}.scales");
    if let (Some(cfg), true) = (w.packed(), w.contains(&scales_key)) {
        let wq = w.get_native(&format!("{base}.weight"))?;
        let scales = w.get_f32(&scales_key)?;
        let biases = w.get_f32(&format!("{base}.biases"))?;
        // Dequantize the packed table to **f32** (the encoder's compute dtype), not `w.dtype()`
        // (sc-12828): the TE now stores its weights bf16, but the embedding is upcast to f32 in the
        // forward, so a packed embed must dequantize to f32 to stay bit-identical to the old f32 store
        // (a dequant to bf16 would round the rows before the widen). Uniform with the sibling
        // boogu/ideogram ports, which pack this table on their MLX tiers.
        return QEmbedding::packed(&wq, &scales, &biases, DType::F32, cfg.group_size as usize);
    }
    let weight = w.get(&format!("{base}.weight"))?;
    let hidden = weight.dim(1)?;
    Ok(QEmbedding::dense(Embedding::new(weight, hidden)))
}

/// Standard RMSNorm over the last dim with weight `w` and eps (candle's fused op). Used by the Qwen3-VL
/// text encoder (whose norm weight is applied directly, NOT the DiT's `+1` convention).
pub(crate) fn rmsnorm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    candle_gen::candle_nn::ops::rms_norm(&x.contiguous()?, w, eps as f32)
}

/// Load a `+1` RMSNorm weight (the reference `RMSNorm(weight = scale + 1.0)`): the on-disk `scale` is
/// centered at 0, so pre-fold the `+1` into an **f32** weight at load. Pairs with [`rms_scale`], which
/// always reduces in f32. Mirrors `mlx-gen-krea`'s `RmsScale`.
pub(crate) fn rms_scale_weight(w: &Weights, key: &str) -> Result<Tensor> {
    w.get_f32(key)? + 1.0
}

/// Apply a pre-folded `+1` RMSNorm (`weight` already = `scale + 1`, f32) over the last dim, computing
/// in f32 and casting back to `x`'s dtype — the byte-equivalent of the reference
/// `F.rms_norm(x.float(), weight).to(dtype)`.
pub(crate) fn rms_scale(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let y = candle_gen::candle_nn::ops::rms_norm(
        &x.to_dtype(DType::F32)?.contiguous()?,
        weight,
        eps as f32,
    )?;
    y.to_dtype(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::safetensors;
    use candle_gen::candle_nn::Module;
    use std::collections::HashMap;

    /// The Krea MLX tier's group size (64) — the one carried from `config.json`.
    const G: usize = 64;

    /// Build an MLX group-64 Q4 packed triple for an `[out, in]` weight — `(wq u32, scales, biases,
    /// affine grid)`. The affine grid is the exact dense weight the pack represents.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Tensor) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / G;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        let gpr = in_dim / G;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let g = row * gpr + col / G;
                scales[g] * codes[i] as f32 + biases[g]
            })
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        (
            Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap(),
            Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap(),
            Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap(),
            Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap(),
        )
    }

    fn write_component(dir: &Path, tensors: HashMap<String, Tensor>, quant: bool) {
        std::fs::create_dir_all(dir).unwrap();
        safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
        let cfg = if quant {
            serde_json::json!({ "quantization": { "bits": 4, "group_size": G } })
        } else {
            serde_json::json!({ "hidden_size": 6144 })
        };
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f64 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    /// **Packed-detect fires on the Krea key layout, incl. the `attn.to_out.0` nesting (sc-9411).** A
    /// packed q4 component (`quantization` block present) whose `attn.to_out.0` is a group-64 packed
    /// triple must `linear_detect` to a `Packed` projection — the `.scales`/`.biases` siblings surviving
    /// the `to_out.0` base — while a dense sibling (`attn.to_q`, no `.scales`) stays `Dense`. The packed
    /// forward reproduces the affine grid (proving the group-64 repack + threading is correct, not a
    /// silent dense fallback).
    #[test]
    fn linear_detect_fires_on_to_out_remap_and_leaves_dense_unchanged() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let dense_w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("attn.to_out.0.weight".into(), wq);
        map.insert("attn.to_out.0.scales".into(), s);
        map.insert("attn.to_out.0.biases".into(), b);
        map.insert("attn.to_q.weight".into(), dense_w);

        let dir = std::env::temp_dir().join(format!("sc9411_detect_{}", std::process::id()));
        write_component(&dir, map, true);
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        assert_eq!(w.packed().map(|c| c.group_size), Some(G as i32));

        let packed = linear_detect(&w, "attn.to_out.0", false)?;
        assert!(
            packed.is_packed(),
            "`.scales` under to_out.0 + quant config ⇒ packed load, not a silent dense fallback"
        );
        let dense = linear_detect(&w, "attn.to_q", false)?;
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense path unchanged");

        // The packed forward reproduces the affine grid (group-64 repack + dequant-on-forward).
        let grid_lin = QLinear::dense(Linear::new(grid, None));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?);
        assert!(cos > 0.99999, "group-64 packed vs grid cosine {cos:.6}");

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// A **dense bf16 component** (config.json has no `quantization` block) takes the dense path — the
    /// loader gates on the config, so `Weights::packed()` is `None` and every `linear_detect` stays
    /// `Dense`. The one-crate-serves-both contract.
    #[test]
    fn dense_component_takes_dense_path() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert(
            "attn.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );
        let dir = std::env::temp_dir().join(format!("sc9411_dense_{}", std::process::id()));
        write_component(&dir, map, false);

        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        assert!(w.packed().is_none(), "no quantization block ⇒ dense tier");
        assert!(!linear_detect(&w, "attn.to_q", false)?.is_packed());
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// The packed-detecting **embedding** loader fires on a group-64 packed `embed_tokens` triple and
    /// reproduces its affine grid rows (the future-proof path — the Krea tier keeps this table dense).
    #[test]
    fn embedding_detect_group64() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (128usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("embed_tokens.weight".into(), wq);
        map.insert("embed_tokens.scales".into(), s);
        map.insert("embed_tokens.biases".into(), b);
        let dir = std::env::temp_dir().join(format!("sc9411_emb_{}", std::process::id()));
        write_component(&dir, map, true);

        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        let emb = embedding_detect(&w, "embed_tokens")?;
        assert!(
            emb.is_packed(),
            "`.scales` + quant config ⇒ packed embedding"
        );

        let dense = QEmbedding::dense(Embedding::new(grid, hidden));
        let idx = Tensor::from_vec(vec![0u32, 5, 127, 12, 5], (5,), &dev)?;
        let dev_max = (emb.forward(&idx)?.sub(&dense.forward(&idx)?)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "packed embedding deviates from the grid");
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// **Adapter overlay wins over the packed base (sc-9411 adapter compose).** With a packed
    /// `attn.to_q` triple in the component AND an overlay-installed dense `attn.to_q.weight` (the
    /// adapter-merged weight), `linear_detect` must take the **dense** overlay path — not the packed
    /// triple — and its forward must reproduce the overlay weight exactly. This is the seam that lets a
    /// LoRA merge into a packed tier: the adapted projection loads dense, the rest stays packed.
    #[test]
    fn overlay_shadows_packed_base_for_adapter_compose() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("attn.to_q.weight".into(), wq);
        map.insert("attn.to_q.scales".into(), s);
        map.insert("attn.to_q.biases".into(), b);
        let dir = std::env::temp_dir().join(format!("sc9411_overlay_{}", std::process::id()));
        write_component(&dir, map, true);
        let mut w = Weights::from_dir(&dir, &dev, DType::F32)?;

        // Without an overlay, `attn.to_q` loads packed.
        assert!(linear_detect(&w, "attn.to_q", false)?.is_packed());

        // Install a distinctive dense "merged" weight in the overlay; `linear_detect` must load it dense.
        let merged = Tensor::randn(3f32, 0.5f32, (out_dim, in_dim), &dev)?;
        let mut overlay = HashMap::new();
        overlay.insert("attn.to_q.weight".to_string(), merged.clone());
        w.set_overlay(overlay);

        let lin = linear_detect(&w, "attn.to_q", false)?;
        assert!(
            !lin.is_packed(),
            "an overlaid (adapter-merged) weight must take the dense path, shadowing the packed triple"
        );
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let want = Linear::new(merged, None).forward(&x)?;
        let dev_max = (lin.forward(&x)?.sub(&want)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "overlay forward must equal the merged dense weight"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// **`get_cpu_merge_base` reconstructs the dense grid from the packed triple (sc-9411).** The
    /// adapter merge folds its delta into this reconstructed base; on a packed tier the base must be the
    /// exact affine grid the pack represents (f32), NOT the u32 codes. A dense tier returns the on-disk
    /// weight unchanged.
    #[test]
    fn get_cpu_merge_base_dequantizes_packed_and_passes_dense() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        // Packed tier: base is the reconstructed dense grid.
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("attn.to_q.weight".into(), wq);
        map.insert("attn.to_q.scales".into(), s);
        map.insert("attn.to_q.biases".into(), b);
        let dir = std::env::temp_dir().join(format!("sc9411_base_{}", std::process::id()));
        write_component(&dir, map, true);
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        let base = w.get_cpu_merge_base("attn.to_q.weight")?;
        assert_eq!(base.dims(), &[out_dim, in_dim], "reconstructed dense shape");
        assert!(
            cosine(&base, &grid) > 0.99999,
            "reconstructed base must equal the affine grid"
        );
        std::fs::remove_dir_all(&dir).ok();

        // Dense tier: base is the on-disk weight (identity round-trip).
        let dense_w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let mut dmap: HashMap<String, Tensor> = HashMap::new();
        dmap.insert("attn.to_q.weight".into(), dense_w.clone());
        let ddir = std::env::temp_dir().join(format!("sc9411_base_dense_{}", std::process::id()));
        write_component(&ddir, dmap, false);
        let dw = Weights::from_dir(&ddir, &dev, DType::F32)?;
        let dbase = dw.get_cpu_merge_base("attn.to_q.weight")?;
        let dev_max = (dbase.sub(&dense_w)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "dense tier base is the on-disk weight");
        std::fs::remove_dir_all(&ddir).ok();
        Ok(())
    }

    /// **`linear()` dequantizes a packed tier instead of casting the u32 codes (sc-11727).** The
    /// composable [`crate::KreaTrainDit`] (the control / train forward) loads every base projection
    /// through the dense [`linear`], NOT the packed-detecting [`linear_detect`]. On a packed q4/q8 tier
    /// `{base}.weight` is bit-packed u32 codes, so a plain cast would be garbage; `get_dense_or_dequant`
    /// reconstructs the affine grid from the packed triple. This is what lets the Krea pose-control lane
    /// run on the q8/q4 base (GPU-proven — q8 ≈ bf16, q4 pose-locked with mild haze).
    #[test]
    fn linear_dequantizes_packed_tier() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("attn.to_q.weight".into(), wq);
        map.insert("attn.to_q.scales".into(), s);
        map.insert("attn.to_q.biases".into(), b);
        let dir =
            std::env::temp_dir().join(format!("sc11727_linear_packed_{}", std::process::id()));
        write_component(&dir, map, true);
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;

        // Packed tier: `linear` reconstructs the dense affine grid (NOT the u32 codes).
        let lin = linear(&w, "attn.to_q", false)?;
        assert_eq!(
            lin.weight().dims(),
            &[out_dim, in_dim],
            "reconstructed dense shape"
        );
        assert!(
            cosine(lin.weight(), &grid) > 0.99999,
            "linear() on a packed tier must reconstruct the affine grid, not cast the u32 codes"
        );
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let want = Linear::new(grid, None).forward(&x)?;
        let dev_max = (lin.forward(&x)?.sub(&want)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            dev_max < 1e-4,
            "packed linear forward must match the dense grid (max dev {dev_max})"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// `read_packed_config` distinguishes absent-vs-corrupt (sc-9426, F-073 sibling — the flux2
    /// `component_is_packed` twin): a `quantization` block → packed `Some`, a plain config or a
    /// genuinely-absent `config.json` → dense `None` (unchanged), but a *present-but-corrupt*
    /// `config.json` (malformed JSON, e.g. a partial download) errors loudly naming the file instead
    /// of silently swallowing to the dense path.
    #[test]
    fn read_packed_config_absent_vs_corrupt() {
        let dir = std::env::temp_dir().join(format!("sc9426_krea_cfg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // A `quantization` block → packed tier.
        let packed = dir.join("packed");
        std::fs::create_dir_all(&packed).unwrap();
        std::fs::write(
            packed.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();
        assert!(
            read_packed_config(&packed).unwrap().is_some(),
            "a `quantization` block ⇒ packed tier"
        );

        // A plain config with no `quantization` block → dense.
        let dense = dir.join("dense");
        std::fs::create_dir_all(&dense).unwrap();
        std::fs::write(dense.join("config.json"), r#"{"hidden_size": 6144}"#).unwrap();
        assert!(
            read_packed_config(&dense).unwrap().is_none(),
            "no `quantization` block ⇒ dense tier"
        );

        // No `config.json` at all → dense (single-file fixtures still load).
        let absent = dir.join("absent");
        std::fs::create_dir_all(&absent).unwrap();
        assert!(
            read_packed_config(&absent).unwrap().is_none(),
            "absent config.json ⇒ dense (unchanged)"
        );

        // A config.json that is *present but corrupt* (malformed JSON) → error naming the file, NOT a
        // silent dense fallback.
        let corrupt = dir.join("corrupt");
        std::fs::create_dir_all(&corrupt).unwrap();
        std::fs::write(corrupt.join("config.json"), b"{ not json").unwrap();
        let err = read_packed_config(&corrupt)
            .expect_err("corrupt config.json must error, not fall to dense");
        assert!(
            format!("{err}").contains("config.json"),
            "the error should name the offending file, got: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── INT8-ConvRot (sc-9300) ──────────────────────────────────────────────────────────────────

    /// A byte-exact slice of the diffusers→native remap (sc-9300), pinned so a future edit to the map
    /// can't silently drift a key. Covers the top-level renames, the single-stream block leaves, and
    /// both text-fusion stacks — the traps (`to_out.0 → wo`, `norm1 → prenorm`, `ff.gate → mlp.gate`,
    /// `scale_shift_table → mod.lin`, `time_mod_proj → tproj.1`).
    #[test]
    fn convrot_remap_pins_the_key_map() {
        let cases = [
            ("img_in.weight", "first.weight"),
            ("time_mod_proj.weight", "tproj.1.weight"),
            ("time_embed.linear_1.weight", "tmlp.0.weight"),
            ("txt_in.norm.weight", "txtmlp.0.scale"),
            ("txt_in.linear_2.bias", "txtmlp.3.bias"),
            ("final_layer.linear.weight", "last.linear.weight"),
            ("final_layer.scale_shift_table", "last.modulation.lin"),
            (
                "transformer_blocks.7.attn.to_q.weight",
                "blocks.7.attn.wq.weight",
            ),
            (
                "transformer_blocks.7.attn.to_out.0.weight",
                "blocks.7.attn.wo.weight",
            ),
            (
                "transformer_blocks.7.attn.to_gate.weight",
                "blocks.7.attn.gate.weight",
            ),
            (
                "transformer_blocks.7.attn.norm_q.weight",
                "blocks.7.attn.qknorm.qnorm.scale",
            ),
            (
                "transformer_blocks.7.ff.gate.weight",
                "blocks.7.mlp.gate.weight",
            ),
            (
                "transformer_blocks.7.norm1.weight",
                "blocks.7.prenorm.scale",
            ),
            ("transformer_blocks.7.scale_shift_table", "blocks.7.mod.lin"),
            (
                "text_fusion.layerwise_blocks.1.attn.to_v.weight",
                "txtfusion.layerwise_blocks.1.attn.wv.weight",
            ),
            (
                "text_fusion.refiner_blocks.0.ff.down.weight",
                "txtfusion.refiner_blocks.0.mlp.down.weight",
            ),
            ("text_fusion.projector.weight", "txtfusion.projector.weight"),
        ];
        for (d, n) in cases {
            assert_eq!(
                convrot_diffusers_to_native(d).as_deref(),
                Some(n),
                "remap {d} → {n}"
            );
        }
        // A key with no native counterpart returns None (the caller then errors on the missing tensor).
        assert!(convrot_diffusers_to_native("transformer_blocks.0.attn.to_q.bias").is_none());
        assert!(convrot_diffusers_to_native("nonsense.key").is_none());
    }

    /// The ConvRot regular-Hadamard order the fixtures rotate at (`64 = 4³`; the real checkpoint uses
    /// 256, but 64 keeps the tiny `in_dim = 128` fixtures at 2 groups).
    const CONVROT_G: usize = 64;

    /// Build a tiny **native-mmdit-keyed** ConvRot component the way the real ComfyUI export does: one
    /// single-stream block's attn `wq` as an int8 projection of the **rotated** weight `W·R` (int8 codes
    /// of `RHT(W)` + per-row `weight_scale` + a real `comfy_quant` JSON carrying `convrot_groupsize`),
    /// plus a dense bf16 `prenorm.scale`. Returns the **canonical un-rotated** `W` (f32) — the parity
    /// reference the online-rotation forward must reconstruct (`RHT(x)·RHT(W)ᵀ = x·Wᵀ`). `in_dim` must be
    /// a multiple of [`CONVROT_G`].
    fn convrot_int8_weight(out_dim: usize, in_dim: usize) -> (HashMap<String, Tensor>, Tensor) {
        let dev = Device::Cpu;
        // A ragged f32 weight (rows of different magnitude) → distinct per-row scales.
        let mut wv = vec![0f32; out_dim * in_dim];
        for o in 0..out_dim {
            let mag = 1.0 + o as f32 * 0.3;
            for j in 0..in_dim {
                wv[o * in_dim + j] = (((o * 7 + j * 3) % 51) as f32 / 25.0 - 1.0) * mag;
            }
        }
        let w = Tensor::from_vec(wv, (out_dim, in_dim), &dev).unwrap();
        // Rotate the weight block-diagonally by the regular Hadamard (what the export stores): W·R.
        let r = candle_gen::quant::regular_hadamard(CONVROT_G, &dev).unwrap();
        let rw = candle_gen::quant::convrot_rotate(&w, &r).unwrap();
        // Per-output-row int8 quant of the *rotated* weight (the checkpoint's stored granularity).
        let pc = candle_gen::quant::quantize_weight_int8_per_channel(&rw).unwrap();
        let scale_col = Tensor::from_vec(pc.scale.clone(), (out_dim, 1), &dev).unwrap();
        // On disk: I8 codes of W·R, F32 [out,1] weight_scale, U8 comfy_quant JSON descriptor.
        let codes_i8 = pc.q.to_dtype(DType::I64).unwrap(); // safetensors save has no I8; I64 codes round-trip identically at the int8 stage
        let cq = format!(
            "{{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": {CONVROT_G}}}"
        );
        let cq_bytes = cq.into_bytes();
        let cq_len = cq_bytes.len();
        let mut map = HashMap::new();
        map.insert("blocks.0.attn.wq.weight".into(), codes_i8);
        map.insert("blocks.0.attn.wq.weight_scale".into(), scale_col);
        map.insert(
            "blocks.0.attn.wq.comfy_quant".into(),
            Tensor::from_vec(cq_bytes, (cq_len,), &dev).unwrap(),
        );
        map.insert(
            "blocks.0.prenorm.scale".into(),
            Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap(),
        );
        (map, w) // the canonical un-rotated weight is the parity reference
    }

    fn write_single_file(path: &Path, tensors: HashMap<String, Tensor>) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        safetensors::save(&tensors, path).unwrap();
    }

    /// **ConvRot detect fires on the native int8 layout and the online rotation reconstructs the
    /// canonical weight (sc-9300 loader + sc-9601 rotation).** `linear_detect(w, "…attn.to_q", …)` on a
    /// ConvRot checkpoint must resolve to the native `blocks.0.attn.wq`, see its `weight_scale` sibling,
    /// read `convrot_groupsize` from the `comfy_quant` blob, and build an int8-ConvRot projection whose
    /// forward applies the online `RHT(x)` so it reproduces `X·Wᵀ` for the **canonical un-rotated** `W`
    /// (not the stored `W·R`). There is no `.bias`.
    #[test]
    fn convrot_detect_fires_and_reconstructs_canonical_weight() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (map, ref_w) = convrot_int8_weight(out_dim, in_dim);

        let path = std::env::temp_dir()
            .join(format!("sc9601_convrot_{}", std::process::id()))
            .join("krea2_int8_convrot.safetensors");
        write_single_file(&path, map);

        let w = Weights::from_convrot_file(&path, &dev, DType::F32)?;
        assert!(w.is_convrot(), "from_convrot_file ⇒ convrot mode");

        // Detect via the diffusers key — must resolve to native + fire the int8 arm.
        let lin = linear_detect(&w, "transformer_blocks.0.attn.to_q", false)?;
        assert!(
            lin.is_convrot_int8(),
            "a ConvRot int8 projection with a weight_scale sibling ⇒ int8 arm, not a dense fallback"
        );

        // The online-rotation forward reconstructs X·Wᵀ for the CANONICAL weight within the int8 budget.
        // Without the rotation this would be ~0.1 (the sc-9300 NO-GO); the sc-9601 leg lifts it to ~1.
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let got = lin.forward(&x)?.to_dtype(DType::F32)?;
        let want = Linear::new(ref_w, None).forward(&x)?;
        let cos = cosine(&got, &want);
        assert!(
            cos > 0.99,
            "convrot online-rotation vs canonical cosine {cos:.5}"
        );

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
        Ok(())
    }

    /// A **dense bf16 tensor** in a ConvRot checkpoint (no `weight_scale` sibling) still loads dense —
    /// only the quantized surface goes int8. `prenorm.scale` (→ `norm1.weight` in diffusers) resolves
    /// and loads as a plain tensor; a dense projection detects to `Dense`.
    #[test]
    fn convrot_dense_tensors_load_through_remap() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (mut map, _ref) = convrot_int8_weight(out_dim, in_dim);
        // Add a dense (non-quantized) native projection: no weight_scale sibling.
        map.insert(
            "blocks.0.attn.wk.weight".into(),
            Tensor::randn(0f32, 1f32, (32, in_dim), &dev)?,
        );
        let path = std::env::temp_dir()
            .join(format!("sc9300_convrot_dense_{}", std::process::id()))
            .join("m.safetensors");
        write_single_file(&path, map);
        let w = Weights::from_convrot_file(&path, &dev, DType::F32)?;

        // The dense native norm resolves through the diffusers key `norm1.weight` → `prenorm.scale`.
        let normw = w.get("transformer_blocks.0.norm1.weight")?;
        assert_eq!(normw.dims(), &[out_dim]);

        // A projection with no weight_scale sibling stays Dense (to_k → wk, no scale).
        let dense = linear_detect(&w, "transformer_blocks.0.attn.to_k", false)?;
        assert!(
            !dense.is_convrot_int8() && !dense.is_packed(),
            "a native projection with no weight_scale sibling stays dense"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
        Ok(())
    }

    /// A non-ConvRot (dense/packed) `Weights` never remaps and never fires the int8 arm — the ConvRot
    /// path is fully gated on the `convrot` flag, so the existing dense/packed tiers are unchanged.
    #[test]
    fn non_convrot_weights_never_remap_or_int8() -> Result<()> {
        let dev = Device::Cpu;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        // A diffusers-keyed dense weight (as a normal tier would store it).
        map.insert(
            "transformer_blocks.0.attn.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (64, 128), &dev)?,
        );
        let dir = std::env::temp_dir().join(format!("sc9300_plain_{}", std::process::id()));
        write_component(&dir, map, false);
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        assert!(!w.is_convrot());
        // `resolve` is the identity here: the diffusers key loads directly, no native translation.
        assert!(w.contains("transformer_blocks.0.attn.to_q.weight"));
        let lin = linear_detect(&w, "transformer_blocks.0.attn.to_q", false)?;
        assert!(
            !lin.is_convrot_int8() && !lin.is_packed(),
            "plain tier stays dense"
        );
        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    // ── dense-bf16 native single file (sc-14022) ────────────────────────────────────────────────────

    /// Build a tiny **dense-bf16 native-mmdit-keyed** single file the way the community merge stores it —
    /// under the `model.diffusion_model.` namespace prefix, no `.weight_scale`/`.comfy_quant` siblings and
    /// no int8 codes. One attn projection (`blocks.0.attn.wq`) + a norm (`blocks.0.prenorm.scale`).
    /// Returns the on-disk `wq` weight (the parity reference a faithful dense load must reproduce verbatim).
    fn dense_native_file(out_dim: usize, in_dim: usize) -> (HashMap<String, Tensor>, Tensor) {
        let dev = Device::Cpu;
        let wq = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let mut map = HashMap::new();
        map.insert(
            "model.diffusion_model.blocks.0.attn.wq.weight".into(),
            wq.clone(),
        );
        map.insert(
            "model.diffusion_model.blocks.0.prenorm.scale".into(),
            Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap(),
        );
        (map, wq)
    }

    fn write_plain_int8_native_file(
        path: &Path,
        descriptor: &str,
        scale_shape: Vec<usize>,
        scales: &[f32],
    ) {
        let codes = [
            1_u8,
            (-2_i8) as u8,
            3_u8,
            (-4_i8) as u8,
            5_u8,
            (-6_i8) as u8,
        ];
        let scale_bytes: Vec<u8> = scales
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let descriptor_bytes = descriptor.as_bytes();
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            "model.diffusion_model.blocks.0.attn.wq.weight",
            ::safetensors::tensor::TensorView::new(::safetensors::Dtype::I8, vec![2, 3], &codes)
                .unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.blocks.0.attn.wq.weight_scale",
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::F32,
                scale_shape,
                &scale_bytes,
            )
            .unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.blocks.0.attn.wq.comfy_quant",
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::U8,
                vec![descriptor_bytes.len()],
                descriptor_bytes,
            )
            .unwrap(),
        );
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        ::safetensors::serialize_to_file(tensors, None, path).unwrap();
    }

    /// sc-14023: the non-rotated descriptor arm reconstructs exactly `codes * row_scale` through the
    /// shared native-key remap. The expected values deliberately are not rotation-invariant, so any
    /// accidental ConvRot/Hadamard leg changes the result.
    #[test]
    fn plain_int8_native_file_dequants_per_row_without_rotation() -> Result<()> {
        let dev = Device::Cpu;
        let path = std::env::temp_dir()
            .join(format!("sc14023_plain_int8_{}", std::process::id()))
            .join("kreamania_variant4.safetensors");
        write_plain_int8_native_file(
            &path,
            r#"{"format":"int8_tensorwise","per_row":true}"#,
            vec![2, 1],
            &[0.5, 2.0],
        );

        let w = Weights::from_native_file(&path, &dev, DType::F32)?;
        assert!(w.uses_native_keys());
        assert!(w.is_plain_int8());
        assert!(!w.is_convrot(), "plain int8 must never enable rotation");
        let got = w.get("transformer_blocks.0.attn.to_q.weight")?;
        assert_eq!(
            got.flatten_all()?.to_vec1::<f32>()?,
            vec![0.5, -1.0, 1.5, -8.0, 10.0, -12.0]
        );
        let lin = linear_detect(&w, "transformer_blocks.0.attn.to_q", false)?;
        assert!(
            !lin.is_convrot_int8() && !lin.is_packed(),
            "plain int8 dequantizes to a dense, non-rotated projection"
        );

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
        Ok(())
    }

    #[test]
    fn plain_int8_native_file_rejects_convrot_or_wrong_descriptor() {
        let cases = [
            (
                r#"{"format":"int8_tensorwise","per_row":true,"convrot":true}"#,
                "convrot",
            ),
            (r#"{"format":"mxfp4","per_row":true}"#, "int8_tensorwise"),
            (r#"{"format":"int8_tensorwise","per_row":false}"#, "per_row"),
        ];
        for (index, (descriptor, expected)) in cases.into_iter().enumerate() {
            let path = std::env::temp_dir()
                .join(format!(
                    "sc14023_plain_int8_bad_desc_{}_{}",
                    std::process::id(),
                    index
                ))
                .join("bad.safetensors");
            write_plain_int8_native_file(&path, descriptor, vec![2], &[0.5, 2.0]);
            let error = match Weights::from_native_file(&path, &Device::Cpu, DType::F32) {
                Ok(_) => panic!("invalid descriptor must fail"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains(expected), "{error}");
            std::fs::remove_dir_all(path.parent().unwrap()).ok();
        }
    }

    #[test]
    fn convrot_constructor_rejects_plain_int8_descriptor() {
        let path = std::env::temp_dir()
            .join(format!(
                "sc14023_plain_int8_wrong_route_{}",
                std::process::id()
            ))
            .join("plain.safetensors");
        write_plain_int8_native_file(
            &path,
            r#"{"format":"int8_tensorwise","per_row":true}"#,
            vec![2],
            &[0.5, 2.0],
        );
        let error = match Weights::from_convrot_file(&path, &Device::Cpu, DType::F32) {
            Ok(_) => panic!("plain descriptor must not enter ConvRot"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("convrot: true"), "{error}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn plain_int8_native_file_rejects_non_per_row_scale_shape() {
        let path = std::env::temp_dir()
            .join(format!(
                "sc14023_plain_int8_bad_scale_{}",
                std::process::id()
            ))
            .join("bad.safetensors");
        write_plain_int8_native_file(
            &path,
            r#"{"format":"int8_tensorwise","per_row":true}"#,
            vec![1],
            &[0.5],
        );
        let error = match Weights::from_native_file(&path, &Device::Cpu, DType::F32) {
            Ok(_) => panic!("wrong scale shape must fail"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("weight_scale") && error.contains("[2]"),
            "{error}"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// **The dense-bf16 native path loads through the remap with NO rotation and NO int8 — the corruption
    /// path is closed (sc-14022).** `from_native_file` sets `native_keys` (so the `model.diffusion_model.`
    /// prefixed native keys resolve from diffusers lookups) but leaves `convrot` OFF, so `linear_detect`
    /// stays on the plain **Dense** arm and the loaded weight is byte-for-byte the on-disk dense weight —
    /// NOT an inverse-rotated `W·Rᵀ`. The contrast: `from_convrot_file` reports rotation ON.
    #[test]
    fn from_native_file_loads_dense_without_rotation() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (map, ref_wq) = dense_native_file(out_dim, in_dim);
        let path = std::env::temp_dir()
            .join(format!("sc14022_native_dense_{}", std::process::id()))
            .join("kreamania_variant5.safetensors");
        write_single_file(&path, map);

        let w = Weights::from_native_file(&path, &dev, DType::F32)?;
        // native_keys ON (remap), convrot/rotation OFF — the whole point of the split.
        assert!(
            w.uses_native_keys(),
            "from_native_file ⇒ native-key remap ON"
        );
        assert!(
            !w.is_convrot(),
            "from_native_file must leave the int8/rotation legs OFF (else it corrupts dense weights)"
        );

        // The diffusers key resolves through the remap AND the detected `model.diffusion_model.` prefix,
        // and the value is the on-disk dense weight unchanged (no rotation, no dequant).
        let got = w.get("transformer_blocks.0.attn.to_q.weight")?;
        let dev_max = (got.sub(&ref_wq)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "dense native weight must load verbatim (no inverse rotation applied)"
        );

        // `linear_detect` takes the plain Dense arm — not convrot_int8, not packed.
        let lin = linear_detect(&w, "transformer_blocks.0.attn.to_q", false)?;
        assert!(
            !lin.is_convrot_int8() && !lin.is_packed(),
            "a dense native projection must detect Dense — no rotation, no int8"
        );
        // And its forward equals the un-rotated dense linear (the reconstruction a rotation would break).
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let want = Linear::new(ref_wq, None).forward(&x)?;
        let dev_max = (lin.forward(&x)?.sub(&want)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "dense native forward must equal X·Wᵀ verbatim"
        );

        // The norm resolves through the diffusers key `norm1.weight` → prefixed `…blocks.0.prenorm.scale`.
        let normw = w.get("transformer_blocks.0.norm1.weight")?;
        assert_eq!(normw.dims(), &[out_dim]);

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
        Ok(())
    }

    /// **The int8/rotation flag is the discriminator (sc-14022).** On the SAME logical projection, a
    /// `from_convrot_file` (int8 `W·R`) reports `is_convrot()` — rotation applies — while a
    /// `from_native_file` (dense `W`) does not. Both are `uses_native_keys()` (both native-mmdit-keyed);
    /// the split is exactly `native_keys` (shared) vs `convrot` (int8/rotation, ConvRot only).
    #[test]
    fn native_keys_and_rotation_are_independent_flags() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);

        // Dense native file (prefixed): native_keys ON, rotation OFF.
        let (dmap, _wq) = dense_native_file(out_dim, in_dim);
        let dpath = std::env::temp_dir()
            .join(format!("sc14022_flags_dense_{}", std::process::id()))
            .join("dense.safetensors");
        write_single_file(&dpath, dmap);
        let dense = Weights::from_native_file(&dpath, &dev, DType::F32)?;
        assert!(dense.uses_native_keys());
        assert!(!dense.is_convrot(), "dense native ⇒ rotation OFF");

        // INT8-ConvRot file (bare native keys): native_keys ON, rotation ON.
        let (cmap, _ref) = convrot_int8_weight(out_dim, in_dim);
        let cpath = std::env::temp_dir()
            .join(format!("sc14022_flags_convrot_{}", std::process::id()))
            .join("convrot.safetensors");
        write_single_file(&cpath, cmap);
        let convrot = Weights::from_convrot_file(&cpath, &dev, DType::F32)?;
        assert!(convrot.uses_native_keys());
        assert!(convrot.is_convrot(), "INT8-ConvRot ⇒ rotation ON");

        std::fs::remove_dir_all(dpath.parent().unwrap()).ok();
        std::fs::remove_dir_all(cpath.parent().unwrap()).ok();
        Ok(())
    }
}
