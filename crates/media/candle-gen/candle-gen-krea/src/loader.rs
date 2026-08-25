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
//! [`Weights::from_dir`] reads that block and prepares content-addressed GGML sidecars once.
//! [`linear_detect`] / [`embedding_detect`] then packed-**detect** the `.scales` sibling, map the
//! device-format artifact, and transfer those bytes directly to the compute device (no dense staging
//! and no repeated source conversion — see [`crate::quant`]). Absent the block / `.scales`, the dense
//! path is unchanged.
//!
//! **Adapter compose (sc-9411).** The DiT's `set_overlay` (adapter merge, sc-7836) installs dense
//! CPU-side weights that take priority over the mmap. [`linear_detect`] checks the **overlay first**: a
//! projection the adapter merge targeted resolves to its merged **dense** weight (the merge
//! reconstructs the dense base from the packed parts before folding, [`crate::adapters`]), while an
//! untargeted packed projection stays packed. So the packed base and the dense adapter overlay compose.
//! [`dequant_packed_base`] is the reconstruction the merge uses to build a mergeable dense base off the
//! packed triple.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::OnceLock;

use candle_gen::gen_core::checkpoint_codec::{
    LogicalTensorPlan, LogicalWeightPlan, LogicalWeightReceipt, NVFP4_CODEC,
};
use candle_gen::gen_core::checkpoint_facts::CheckpointWeightFacts;
use candle_gen::logical_weights::{
    plan_logical_weights, CandleCodecResidency, LogicalTensor, LogicalWeightReader,
};

use candle_gen::candle_core::quantized::QTensor;
use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen::candle_nn::{Embedding, Linear};
use candle_gen::quant::{
    dequant_mlx_q4_reference_gs, dequant_mlx_q8_gs, mlx_packed_bits_gs, Int8Context, Nvfp4Fallback,
    Nvfp4Linear, Nvfp4Regime, PackedConfig, PackedWeightSidecars,
};

use crate::native_mapping::{DeclaredLogicalShapes, KreaNativeToDiffusersMapping};
use crate::nvfp4_dit::{
    DitPlan, ExecutionRole, Nvfp4Capability, Nvfp4Proj, Nvfp4Quant, ProbedProj,
};
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
    /// Present for a single-file source. The mmap keeps reads lazy; this lstat/target fingerprint adds
    /// the mutation guard required when the same mapping is revisited for successive block windows.
    pinned_source: Option<candle_gen::gen_core::PinnedWeightsFile>,
    device: Device,
    dtype: DType,
    overlay: HashMap<String, Tensor>,
    /// The component's `quantization` manifest, `Some` for a packed q4/q8 tier (carries the group size
    /// the packed shapes can't disambiguate), `None` for a dense bf16 tier.
    packed: Option<PackedConfig>,
    /// Content-addressed GGML q4/q8 artifacts prepared once at component open. A materialization maps
    /// these bytes and transfers them directly; it never re-reads or converts the MLX affine triple.
    sidecars: Option<PackedWeightSidecars>,
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
    /// True only for a native-keyed ComfyUI Kitchen NVFP4 checkpoint.
    ///
    /// Since sc-20651 this is a **plan** fact, not a dtype guess: it is set when
    /// [`logical_plan`](Self::logical_plan) contains at least one tensor whose planned codec is
    /// `nvfp4-v1`, i.e. when the checkpoint's own `.comfy_quant` / `_quantization_metadata`
    /// descriptor says NVFP4. Since sc-21482 the registered codec is the *whole* authority: there
    /// is no provider-owned structural classifier on top, and every payload contract (scale sizes,
    /// scale values, geometry) is enforced by the shared reader's codec decode. This is distinct
    /// from the MLX affine [`packed`](Self::packed) tier and inline FP8.
    native_nvfp4: bool,
    /// The shared logical-weight reader over the compiled mapped-read plan, for a **single-file
    /// native** checkpoint (`None` for the directory/packed-tier path, which is diffusers-keyed
    /// and has its own tier machinery).
    ///
    /// The plan it holds is the descriptor authority the import reads: which layers are NVFP4,
    /// which companion tensors carry their two scale levels, what stored/logical geometry each
    /// has, and — priced per layer by [`CandleCodecResidency`] — whether each row materializes
    /// packed-native or as the declared dense fallback. Materialization itself goes through
    /// [`LogicalWeightReader::read`] (sc-21482): the provider owns **mapping and construction**,
    /// the codec owns classification and decode, and the reader measures the receipt.
    logical: Option<LogicalWeightReader>,
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
    ///
    /// Packed q4/q8 projections use content-addressed, file-backed Candle-format sidecars. A writable
    /// component keeps those artifacts beside the model; a read-only component automatically uses the
    /// shared per-user external cache documented by [`PackedWeightSidecars`]. A complete valid cache is
    /// reused read-only without creating or acquiring its preparation lock.
    pub fn from_dir(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        Self::from_dir_impl(dir, device, dtype, None, None)
    }

    /// Request-time component open that preserves the shared sidecar preparation cancellation seam.
    pub fn from_dir_cancelable(
        dir: &Path,
        device: &Device,
        dtype: DType,
        cancel: &candle_gen::gen_core::CancelFlag,
    ) -> candle_gen::Result<Self> {
        Self::from_dir_impl(dir, device, dtype, None, Some(cancel)).map_err(|error| {
            if cancel.is_cancelled() {
                candle_gen::CandleError::Canceled
            } else {
                error.into()
            }
        })
    }

    /// Load a component while choosing the non-model cache root used when `dir` is read-only.
    ///
    /// This gives embedders an explicit disk-placement policy instead of requiring writes beside a
    /// caller-provisioned snapshot or relying on `SCENEWORKS_CANDLE_DEVICE_CACHE_DIR`.
    pub fn from_dir_with_external_cache_root(
        dir: &Path,
        device: &Device,
        dtype: DType,
        external_cache_root: &Path,
    ) -> Result<Self> {
        Self::from_dir_impl(dir, device, dtype, Some(external_cache_root), None)
    }

    fn from_dir_impl(
        dir: &Path,
        device: &Device,
        dtype: DType,
        external_cache_root: Option<&Path>,
        cancel: Option<&candle_gen::gen_core::CancelFlag>,
    ) -> Result<Self> {
        let files = candle_gen::sorted_safetensors(dir, "krea")
            .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
        let packed = read_packed_config(dir)?;
        let (st, sidecars) = match packed {
            Some(cfg) => {
                let (st, sidecars) = match (external_cache_root, cancel) {
                    (Some(root), None) => {
                        PackedWeightSidecars::open_and_prepare_with_external_cache_root(
                            &files, dir, cfg, device, root,
                        )
                    }
                    (None, Some(cancel)) => PackedWeightSidecars::open_and_prepare_cancelable(
                        &files, dir, cfg, device, cancel,
                    ),
                    (None, None) => {
                        PackedWeightSidecars::open_and_prepare(&files, dir, cfg, device)
                    }
                    (Some(_), Some(_)) => unreachable!("no public API combines these policies"),
                }?;
                (st, Some(sidecars))
            }
            None => {
                // SAFETY: read-only mmap of weight files; the standard dense loading path.
                (unsafe { MmapedSafetensors::multi(&files)? }, None)
            }
        };
        Ok(Self {
            st,
            pinned_source: None,
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
            packed,
            sidecars,
            native_keys: false,
            native_prefix: String::new(),
            convrot: false,
            plain_int8: false,
            native_nvfp4: false,
            logical: None,
            int8: OnceLock::new(),
        })
    }

    /// mmap a single `.safetensors` file (used by the committed parity fixtures). Dense-only (no
    /// packed config), so the packed path is never taken for a single-file fixture. Diffusers-keyed —
    /// for a **native-mmdit-keyed** dense single file use [`from_native_file`](Self::from_native_file).
    pub fn from_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let pinned_source = candle_gen::gen_core::PinnedWeightsFile::pin(path)
            .map_err(|error| Error::Msg(error.to_string()))?;
        // SAFETY: read-only mmap of a weight file; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::new(pinned_source.loader_path())? };
        Ok(Self {
            st,
            pinned_source: Some(pinned_source),
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
            packed: None,
            sidecars: None,
            native_keys: false,
            native_prefix: String::new(),
            convrot: false,
            plain_int8: false,
            native_nvfp4: false,
            logical: None,
            int8: OnceLock::new(),
        })
    }

    /// mmap a **single-file INT8-ConvRot checkpoint** (sc-9300) — the ComfyUI-exported, native-mmdit-keyed
    /// `krea2_turbo_int8_convrot.safetensors`. `convrot` is set, so every diffusers-key lookup is
    /// translated to the native key ([`convrot_diffusers_to_native`]) at read time and quantized
    /// projections are int8 (per-output-row `.weight_scale`). Dense bf16 tensors (`first`/`last`/`tmlp`
    /// /`tproj`/`txtfusion`/`txtmlp` + norms) load unchanged through the remap.
    pub fn from_convrot_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let pinned_source = candle_gen::gen_core::PinnedWeightsFile::pin(path)
            .map_err(|error| Error::Msg(error.to_string()))?;
        // SAFETY: read-only mmap of a weight file; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::new(pinned_source.loader_path())? };
        validate_convrot_descriptors(&st)?;
        let native_prefix = detect_native_prefix(&st);
        Ok(Self {
            st,
            pinned_source: Some(pinned_source),
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
            packed: None,
            sidecars: None,
            native_keys: true,
            native_prefix,
            convrot: true,
            plain_int8: false,
            native_nvfp4: false,
            logical: None,
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
    /// have an F32 `[out]` or `[out,1]` scale (or scalar when `out == 1`). The constructor validates
    /// that data before any dequantization. A present `convrot` field is rejected here and belongs to
    /// [`from_convrot_file`](Self::from_convrot_file), preventing the old group-size-256 fallback from
    /// silently rotating a plain file.
    pub fn from_native_file(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        Self::from_native_file_for(path, device, dtype, DeclaredLogicalShapes::NotInScope)
    }

    /// [`from_native_file`](Self::from_native_file) with the architecture config in scope, so the
    /// compiled plan can unpad a block-padded (NVFP4/MXFP8) layer to its true geometry.
    ///
    /// A **padded** import needs this form: with no declared logical shape the plan can only carry
    /// the padded stored grid forward, and `gen_core` refuses to materialize that rather than turn
    /// pad rows/columns into weights.
    pub fn from_native_file_for(
        path: &Path,
        device: &Device,
        dtype: DType,
        shapes: DeclaredLogicalShapes<'_>,
    ) -> Result<Self> {
        let pinned_source = candle_gen::gen_core::PinnedWeightsFile::pin(path)
            .map_err(|error| Error::Msg(error.to_string()))?;
        Self::from_pinned_native_file_for(&pinned_source, device, dtype, shapes)
    }

    /// Open a native Krea file through a caller-owned pin. Registry-backed lazy/sequential loads keep
    /// this exact pin from generator construction through every later materialization instead of
    /// re-pinning whatever happens to occupy the path at request time.
    /// Open a native Krea file through a caller-owned pin, with the architecture config in scope.
    ///
    /// # The plan is compiled here, before any tensor is read (sc-20651)
    ///
    /// A single-file native checkpoint is a **checkpoint import**, so it goes through the epic's
    /// one import seam: [`plan_logical_weights`] reads the header plus this file's `.comfy_quant` /
    /// `__metadata__._quantization_metadata` descriptors and compiles them against the engine's
    /// codec registry and residency policy. That is what decides which layers are NVFP4 — the
    /// producer's own declaration — and it refuses, by name and before any tensor exists, on an
    /// unmapped key, a key collision, a descriptor that disagrees with the stored dtype, a missing
    /// or mis-shaped scale companion, or bad NVFP4 block geometry.
    ///
    /// The plan is compiled for **every** native file, not only NVFP4 ones: a dense or int8
    /// checkpoint that plans is a checkpoint whose whole key surface the dialect recognises, which
    /// is the same property `validate_native_transformer` enforces later — just established before
    /// the first read instead of after.
    pub(crate) fn from_pinned_native_file_for(
        pinned_source: &candle_gen::gen_core::PinnedWeightsFile,
        device: &Device,
        dtype: DType,
        shapes: DeclaredLogicalShapes<'_>,
    ) -> Result<Self> {
        // The engine's own residency policy: a CUDA device at the NVFP4 `sm_120` floor prices the
        // packed rows packed, everything else prices the dense fallback. Passing the real device
        // (rather than a fixed policy) keeps the plan's pricing the pricing of *this* load.
        let residency = CandleCodecResidency::probe(device);
        Self::from_pinned_native_file_with_residency(
            pinned_source,
            device,
            dtype,
            shapes,
            residency,
        )
    }

    /// Test-only: [`Self::from_native_file_for`] with the NVFP4 packed residency **forced on**, so
    /// the packed-native construction route is exercisable on a CPU lane. Sound to exercise there:
    /// the [`candle_gen::logical_weights::LogicalTensor::PackedNvfp4`] container is host-side, and
    /// only the GEMM behind `Nvfp4Linear` is `cuda`-gated (it serves the dequant fallback
    /// elsewhere, sc-11041).
    #[cfg(test)]
    pub(crate) fn from_native_file_forcing_packed_nvfp4(
        path: &Path,
        device: &Device,
        dtype: DType,
        shapes: DeclaredLogicalShapes<'_>,
    ) -> Result<Self> {
        let pinned_source = candle_gen::gen_core::PinnedWeightsFile::pin(path)
            .map_err(|error| Error::Msg(error.to_string()))?;
        Self::from_pinned_native_file_with_residency(
            &pinned_source,
            device,
            dtype,
            shapes,
            CandleCodecResidency {
                fp8_e4m3_native: false,
                nvfp4_native: true,
            },
        )
    }

    fn from_pinned_native_file_with_residency(
        pinned_source: &candle_gen::gen_core::PinnedWeightsFile,
        device: &Device,
        dtype: DType,
        shapes: DeclaredLogicalShapes<'_>,
        residency: CandleCodecResidency,
    ) -> Result<Self> {
        pinned_source
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
        // SAFETY: read-only mmap of a weight file; the standard candle loading path.
        let st = unsafe { MmapedSafetensors::new(pinned_source.loader_path())? };
        let native_prefix = detect_native_prefix(&st);
        let mapping = KreaNativeToDiffusersMapping::new(&native_prefix, shapes);
        let logical_plan = plan_logical_weights(pinned_source.loader_path(), &mapping, &residency)
            .map_err(|error| Error::Msg(error.to_string()))?;
        let plain_int8 = validate_plain_int8_tensorwise(&st)?;
        // The descriptor — resolved through the registered `nvfp4-v1` codec row — is the whole
        // authority on NVFP4-ness (sc-21482). Payload contracts (scale-buffer geometry, scale
        // values, packed byte counts) are enforced by the shared reader's codec decode when each
        // layer materializes; there is no provider-owned structural classifier to disagree with.
        let native_nvfp4 = logical_plan
            .tensors
            .iter()
            .any(|tensor| tensor.codec_id == NVFP4_CODEC.codec_id);
        if plain_int8 && native_nvfp4 {
            return Err(Error::Msg(
                "krea native checkpoint mixes plain int8 and NVFP4 projection formats".into(),
            ));
        }
        // The shared reader over the same pinned bytes: opening it re-verifies the on-disk tensor
        // surface against the plan, so source drift between planning and reading refuses here —
        // before any generator construction begins.
        // The capability the facts are validated against is rendered from the *same* residency
        // policy that priced the plan (sc-21484), so a forced-packed test route and a real
        // `sm_120` probe stay consistent, and a dense-only host cannot produce a native row.
        let logical = LogicalWeightReader::open_with_capability(
            pinned_source.loader_path(),
            logical_plan,
            device,
            residency.native_execution_capability(),
        )
        .map_err(|error| Error::Msg(error.to_string()))?;
        let result = Self {
            st,
            pinned_source: Some(pinned_source.clone()),
            device: device.clone(),
            dtype,
            overlay: HashMap::new(),
            packed: None,
            sidecars: None,
            native_keys: true,
            native_prefix,
            convrot: false,
            plain_int8,
            native_nvfp4,
            logical: Some(logical),
            int8: OnceLock::new(),
        };
        pinned_source
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
        Ok(result)
    }

    /// The compiled mapped-read plan for a single-file native checkpoint — the descriptor authority
    /// the NVFP4 import consults. `None` for the directory/packed-tier path.
    pub fn logical_plan(&self) -> Option<&LogicalWeightPlan> {
        self.logical.as_ref().map(LogicalWeightReader::plan)
    }

    /// The planned entry for one **diffusers** (logical) key, or `None` when the file has no plan or
    /// the plan does not contain that key.
    fn planned(&self, logical_key: &str) -> Option<&LogicalTensorPlan> {
        self.logical.as_ref()?.planned(logical_key)
    }

    /// Materialize one planned logical tensor through the **shared reader** (sc-21482): the
    /// registered codec decodes it, the reader measures what it left resident, and the caller gets
    /// back exactly what the plan declared — [`LogicalTensor::Dense`] for dense rows and every
    /// dense-fallback row, [`LogicalTensor::PackedNvfp4`] for a row the residency policy priced
    /// packed on this device.
    pub fn read_planned(&self, logical_key: &str) -> Result<LogicalTensor> {
        let reader = self.logical.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "krea: `{logical_key}` cannot be read through a logical-weight plan; this source \
                 has none (open a single-file native checkpoint through \
                 `Weights::from_native_file*`/`from_pinned_native_file_for`, which compiles one)"
            ))
        })?;
        reader.read(logical_key).map_err(|error| {
            let hint = self
                .planned(logical_key)
                .and_then(|tensor| tensor.undeclared_padded_storage_refusal())
                .map(|_| {
                    " (open the checkpoint with \
                     `Weights::from_native_file_for(.., DeclaredLogicalShapes::FromConfig(cfg))`)"
                })
                .unwrap_or_default();
            Error::Msg(format!("{error}{hint}"))
        })
    }

    /// The NVFP4 capability facts this import carries for one **logical** weight key (sc-12121).
    ///
    /// Every field is read off something real — the compiled logical plan's codec spec for the
    /// checkpoint facts, the residency this import was priced under plus the trunk's shared cuBLASLt
    /// context for the device facts. Nothing is inferred from a key's spelling or a tensor's dtype,
    /// which is the whole point: [`DitPlan::representation`] must be able to say *which* fact
    /// decided a projection's representation, and a re-derivation could only ever restate the
    /// verdict it is supposed to explain.
    ///
    /// The NVFP4 **validation** lane (a dense bf16 tier this import packs itself) has no plan row for
    /// the key; there the checkpoint facts are vacuously clear and the grid question is answered by
    /// [`crate::nvfp4_dit::dense_shape_is_fp4_eligible`] on the dense shape the caller passes.
    ///
    /// Public so a GPU harness can anchor an assertion on the **live** probe rather than on a
    /// hardcoded [`Nvfp4Capability::ELIGIBLE`] (sc-12121 review fix).
    pub fn nvfp4_capability(
        &self,
        weight_key: &str,
        dense_shape: Option<[usize; 2]>,
        ctx: &candle_gen::quant::Nvfp4Context,
    ) -> Nvfp4Capability {
        use candle_gen::gen_core::checkpoint_codec::TensorCodecSpec;
        let planned = self.planned(weight_key);
        let (checkpoint_offers_nvfp4, full_precision_declared, storage_unpadded, layout_native) =
            match planned.map(|tensor| &tensor.codec) {
                Some(TensorCodecSpec::Nvfp4 {
                    stored_shape,
                    logical_shape,
                    full_precision_matrix_mult,
                    ..
                }) => (
                    !*full_precision_matrix_mult,
                    *full_precision_matrix_mult,
                    logical_shape == stored_shape,
                    candle_gen::logical_weights::nvfp4_layout_is_native(*stored_shape),
                ),
                // A planned row that is NOT an NVFP4 codec row is one Kitchen deliberately exported
                // dense; the import preserves that rather than requantizing at load.
                Some(_) => (false, false, true, true),
                // No plan row at all: the dense validation tier, packed by this import. The grid
                // question is then answered by the dense shape the caller passes — and there MUST be
                // one. `dense_shape: None` means the caller had a plan-backed row in mind (the
                // native arm passes `None` because the codec spec answers the question); combined
                // with no plan row there is nothing real left to read `layout_native` off, so
                // claiming `true` would predict PackedW4A4 on grounds never checked (sc-12121 review
                // fix). Report it as ineligible instead — the dense fallback — and say so.
                None => {
                    let layout_native = match dense_shape {
                        Some([rows, cols]) => {
                            crate::nvfp4_dit::dense_shape_is_fp4_eligible(rows, cols)
                        }
                        None => {
                            debug_assert!(
                                false,
                                "krea nvfp4 (sc-12121): `{weight_key}` has no plan row AND no dense \
                                 shape — `layout_native` would be asserted on nothing. A native \
                                 component must have a plan row for every key it serves."
                            );
                            eprintln!(
                                "[sc-12121] krea nvfp4: `{weight_key}` has no plan row and no dense \
                                 shape; reporting the grid as ineligible rather than assuming it"
                            );
                            false
                        }
                    };
                    (true, false, true, layout_native)
                }
            };
        Nvfp4Capability {
            checkpoint_offers_nvfp4,
            full_precision_declared,
            storage_unpadded,
            layout_native,
            // The same probe that gates `Nvfp4Linear::try_build_fp4`: a context holds a handle only
            // above the `sm_120` floor (`Nvfp4Context::new`).
            nvfp4_device: ctx.is_fp4(),
            fused_quantizer: ctx.fused_quantizer_available(),
        }
    }

    /// A [`LogicalWeightReceipt`] **measured** from what this import actually materialized through
    /// the shared reader — every codec row, dense and packed alike (sc-21482). `resident_bytes`
    /// comes off the decoded values themselves (for a packed NVFP4 row: nibbles plus block scales
    /// plus the retained `F32` global scale), never copied from `plan.resident_bytes()`, so the
    /// pair stays an independent cross-check rather than a restatement.
    ///
    /// `None` when the file has no plan. A file none of whose tensors has been materialized yet
    /// reports zero tensors, which is the truth about this instant, not a claim that the load was
    /// free.
    pub fn logical_weight_receipt(&self) -> Option<LogicalWeightReceipt> {
        Some(self.logical.as_ref()?.receipt())
    }

    /// The **three correlated facts** about this import (sc-21484), tied to the verified source
    /// binding: what the source stores (per-codec tensor counts and source bytes), what this host
    /// can execute natively, and what actually materialized — split per execution representation
    /// and measured, never copied from the plan.
    ///
    /// This is the surface the SceneWorks half consumes to distinguish a source stored `nvfp4-v1`
    /// from a run that executed it as packed W4A4 or as dense BF16. `Ok(None)` when the source has
    /// no plan (the directory/packed-tier path); `Err` when the pinned file changed under the load,
    /// so facts are never reported about bytes that are gone.
    pub fn checkpoint_weight_facts(&self) -> Result<Option<CheckpointWeightFacts>> {
        let Some(logical) = self.logical.as_ref() else {
            return Ok(None);
        };
        let facts = logical
            .checkpoint_weight_facts()
            .map_err(|error| Error::Msg(error.to_string()))?;
        let facts = match self.pinned_source.as_ref() {
            Some(pin) => facts
                .with_verified_source(pin)
                .map_err(|error| Error::Msg(error.to_string()))?,
            None => facts,
        };
        Ok(Some(facts))
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

    /// Revalidate a retained single-file source before a new streamed window materializes tensors.
    /// Directory-backed components have no single entry to pin and therefore no-op here.
    pub(crate) fn ensure_source_unchanged(&self) -> Result<()> {
        self.pinned_source
            .as_ref()
            .map(candle_gen::gen_core::PinnedWeightsFile::ensure_unchanged)
            .transpose()
            .map(|_| ())
            .map_err(|error| Error::Msg(error.to_string()))
    }

    /// Run one actual tensor-consumption/materialization operation under the retained File pin.
    /// Directory sources have no single entry to pin and execute the operation directly.
    pub(crate) fn read_source_unchanged<T>(&self, read: impl FnOnce() -> Result<T>) -> Result<T> {
        let Some(pin) = self.pinned_source.as_ref() else {
            return read();
        };
        pin.ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
        let result = read();
        // Mutation identity is the stronger diagnosis, so retain `read_unchanged` semantics and let
        // the post-consumption pin failure win even if materialization also failed.
        pin.ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
        result
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

    /// Whether this native checkpoint carries prepacked ComfyUI Kitchen NVFP4 projection triplets.
    pub fn is_native_nvfp4(&self) -> bool {
        self.native_nvfp4
    }

    /// Number of descriptor-declared NVFP4 projections in this checkpoint, counted from the
    /// compiled plan (the codec is the classifier — sc-21482; the pre-plan version counted
    /// `.weight_scale_2` keys structurally). Used by the exact native-key surface validator to
    /// account for the two scale companions per packed model weight.
    pub(crate) fn native_nvfp4_projection_count(&self) -> usize {
        self.logical_plan()
            .map(|plan| {
                // Distinct PHYSICAL keys: `plan.tensors` is "not a set" (sc-21547) — an
                // adapter-declared transform contributes one entry per logical output, all naming
                // the same physical tensor, and the caller is accounting for on-disk companions.
                plan.tensors
                    .iter()
                    .filter(|tensor| tensor.codec_id == NVFP4_CODEC.codec_id)
                    .map(|tensor| tensor.physical_key.as_str())
                    .collect::<BTreeSet<_>>()
                    .len()
            })
            .unwrap_or(0)
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

    /// True when `diffusers_weight` is one of this file's **descriptor-declared** NVFP4 weights
    /// *and* the descriptor permits a quantized matmul on it.
    ///
    /// The authority is the compiled plan: the producer's `.comfy_quant` /
    /// `_quantization_metadata` declaration for that layer resolved to the `nvfp4-v1` codec row.
    /// The predicate this replaced was `dtype == U8`, which is not a format at all — every packed
    /// container, every byte buffer and every future `U8`-stored codec answers it — and which, by
    /// construction, could never see `full_precision_matrix_mult` or any other descriptor field.
    ///
    /// `full_precision_matrix_mult` is answered **here**, not only in the plan's residency, and
    /// that placement is the whole point: `linear_detect_planned` routes a `false` to
    /// `QLinear::dense(linear(..))`, so a layer the producer flagged takes the dense fallback the
    /// flag is asking for (the residency policy prices such a layer dense too, so the shared
    /// reader materializes exactly that fallback).
    fn is_native_nvfp4_weight(&self, diffusers_weight: &str) -> bool {
        self.native_nvfp4
            && self.planned(diffusers_weight).is_some_and(|tensor| {
                tensor.codec_id == NVFP4_CODEC.codec_id
                    && !tensor.codec.full_precision_matrix_mult()
            })
    }

    /// Load `name` at the component dtype — from the `overlay` if present
    /// (adapter-merged weight), else the mmap (native-key-resolved for a ConvRot checkpoint).
    ///
    /// On a native NVFP4 checkpoint every read goes through the **shared logical reader**
    /// (sc-21482): a dense row is the byte-preserving codec read it always was, while a
    /// descriptor-quantized row asked for densely (e.g. a `full_precision_matrix_mult` projection
    /// through [`linear`]) comes back as the codec's exact dense decode instead of raw stored
    /// bytes reinterpreted at the component dtype.
    ///
    /// # A row the plan priced `Packed` refuses here (sc-21482 review)
    ///
    /// On an eligible device (`sm_120`) the plan prices NVFP4 projections `Packed`, and this
    /// accessor refuses them by name rather than serving one. Two reasons it refuses instead of
    /// quietly re-decoding the row densely:
    ///
    /// * The plan/receipt pair is this story's contract — everything resident was priced. A dense
    ///   back-door read would leave `logical_weight_receipt()` reporting a residency the plan never
    ///   priced, which is exactly the drift the receipt exists to catch.
    /// * The old behaviour was **not** a working escape hatch: before sc-21482 this same call
    ///   returned the stored `U8` nibble buffer at shape `[rows, cols / 2]`, cast to the component
    ///   dtype — silently wrong weights, not a dense decode. Refusing is strictly better.
    ///
    /// Production reachability was audited: the only non-`linear_detect_planned` reads of a
    /// projection `.weight` on a native file are `KreaTrainDit::load_inference`'s `lora_proj`
    /// (reached from [`crate::control_provider`]'s `ControlDitSource::Native`, which does not yet
    /// support an NVFP4 control DiT) and [`crate::control`]'s `ControlBranch::from_base` — whose
    /// every production caller sources `Weights::from_dir`, the directory/packed-tier path, which
    /// compiles no plan at all and so cannot reach this arm. The refusal names the escape hatch
    /// for anything new: re-open the checkpoint under a `CandleCodecResidency` whose
    /// `nvfp4_native` is `false`, so the plan prices — and the receipt reports — the dense
    /// fallback the caller wants.
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
        if self.native_nvfp4 && self.planned(name).is_some() {
            return match self.read_planned(name)? {
                LogicalTensor::Dense(tensor) => tensor.to_dtype(self.dtype),
                LogicalTensor::PackedNvfp4 { .. } | LogicalTensor::PackedFp8E4M3 { .. } => {
                    Err(Error::Msg(format!(
                        "krea: `{name}` was planned packed-native; construct it through \
                         `linear_detect_planned`, not a dense read — or re-open the checkpoint \
                         under a residency whose `nvfp4_native` is false, so the plan prices this \
                         row `CandleCodecResidency::DENSE` and the receipt reports what a dense \
                         read costs"
                    )))
                }
            };
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
    /// overlay if present, else the mmap (native-key-resolved for a ConvRot checkpoint). Native NVFP4
    /// checkpoints route through the shared logical reader exactly as [`get`](Self::get) does.
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
        if self.native_nvfp4 && self.planned(name).is_some() {
            return match self.read_planned(name)? {
                LogicalTensor::Dense(tensor) => tensor.to_dtype(DType::F32),
                LogicalTensor::PackedNvfp4 { .. } | LogicalTensor::PackedFp8E4M3 { .. } => {
                    Err(Error::Msg(format!(
                        "krea: `{name}` was planned packed-native; construct it through \
                         `linear_detect_planned`, not a dense read — or re-open the checkpoint \
                         under a residency whose `nvfp4_native` is false, so the plan prices this \
                         row `CandleCodecResidency::DENSE` and the receipt reports what a dense \
                         read costs"
                    )))
                }
            };
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
            [] if rows == 1 => scale.reshape((1, 1))?,
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
        // A native NVFP4 checkpoint's plan carries the LOGICAL shape of every row — the compiled
        // codec fact, not a `dtype == U8` storage guess (sc-21482) — so architecture validation
        // sees the layer's true geometry rather than its packed byte grid.
        if self.native_nvfp4 {
            if let Some(tensor) = self.planned(name) {
                return Some(tensor.shape.clone());
            }
        }
        let view = self.st.get(&self.resolve(name)).ok()?;
        Some(view.shape().to_vec())
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

    /// Load one packed projection from its mapped device-format artifact. The source mmap is
    /// intentionally not passed through this seam, making per-window format conversion impossible.
    fn load_packed_device_format(&self, base: &str) -> Result<QTensor> {
        self.sidecars
            .as_ref()
            .ok_or_else(|| {
                Error::Msg(format!(
                    "krea: packed projection `{base}` has no prepared device-format sidecar"
                ))
            })?
            .load(base, &self.device)
    }

    /// Sidecar preparation diagnostics used by the real-weight evidence harness.
    pub fn packed_sidecars(&self) -> Option<&PackedWeightSidecars> {
        self.sidecars.as_ref()
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
        let scalar_single_row = *rows == 1 && scale.shape().is_empty();
        if !scalar_single_row && scale.shape() != [*rows] && scale.shape() != [*rows, 1] {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "krea plain int8: `{scale_key}` must be [{rows}] or [{rows},1]{}; got {:?}",
                if *rows == 1 { " or scalar" } else { "" },
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
pub(crate) fn read_packed_config(dir: &Path) -> Result<Option<PackedConfig>> {
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
/// 2. **Packed** (a packed tier + `{base}.scales` present, no overlay): map its prepared GGML sidecar
///    and transfer device-format bytes — **no source conversion or dense weight materialized**.
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
    // (2) A packed tier with a `.scales` sibling → transfer the prepared device-format artifact.
    if let (Some(_cfg), true) = (w.packed(), w.contains(&scales_key)) {
        let dense_bias = if bias {
            Some(w.get(&format!("{base}.bias"))?)
        } else {
            None
        };
        let qtensor = w.load_packed_device_format(base)?;
        return QLinear::packed_device_format(qtensor, dense_bias);
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
/// 1. **NVFP4** (`plan.is_nvfp4()`): consume a validated native Kitchen triplet directly when this
///    projection is prepacked, otherwise pack `{base}.weight` from a dense NVFP4 validation tier.
///    Kitchen's deliberately preserved BF16 projections stay BF16. The plan assigns activation
///    precision by layer role; [`Nvfp4Linear`] resolves the `sm_120` capability gate.
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
    let dense_bias = if bias {
        Some(w.get(&format!("{base}.bias"))?)
    } else {
        None
    };
    let weight_key = format!("{base}.weight");
    if w.is_native_nvfp4() {
        // sc-12121: what the role table + this import's capability facts say this projection will be
        // served as. It is a REPORT of the decisions the plan and `Nvfp4Linear` already own, not a
        // second decider — so it is crossed against the constructed layer below and a disagreement
        // is a defect, never a silent downgrade (epic E4).
        let expected = plan.representation(
            base,
            w.nvfp4_capability(&weight_key, None, plan.nvfp4_context()),
        );
        // Kitchen's Krea profile intentionally leaves sensitive first/last/time/text-fusion layers
        // dense BF16. Preserve that producer decision instead of quantizing them during import.
        // (`full_precision_matrix_mult` rows take this arm too — the descriptor asks for the dense
        // fallback, and the residency policy priced them dense, so `linear`'s reader-backed `get`
        // materializes exactly the codec's dense decode.)
        if !w.is_native_nvfp4_weight(&weight_key) {
            check_representation(base, expected, false, None)?;
            return Ok(QLinear::dense(linear(w, base, bias)?));
        }
        // The shared reader materializes what the plan DECLARED for this row (sc-21482):
        // packed-native on an eligible `sm_120` device with an eligible stored layout, the dense
        // bf16 fallback everywhere else (CPU, pre-`sm_120`, padded or misaligned storage). The
        // provider constructs from that declaration instead of re-deciding it.
        return match w.read_planned(&weight_key)? {
            LogicalTensor::PackedNvfp4 { tensor, .. } => {
                let lin = Nvfp4Linear::from_packed_in(
                    *tensor,
                    dense_bias,
                    w.device(),
                    act,
                    plan.nvfp4_context(),
                )?;
                check_representation(
                    base,
                    expected,
                    lin.regime() == Nvfp4Regime::Fp4W4A4,
                    lin.fallback_cause(),
                )?;
                Ok(QLinear::Nvfp4(Nvfp4Proj::new(lin, base, plan, act)))
            }
            LogicalTensor::Dense(weight) => {
                check_representation(base, expected, false, None)?;
                let weight = weight.to_dtype(w.dtype())?;
                Ok(QLinear::dense(Linear::new(weight, dense_bias)))
            }
            LogicalTensor::PackedFp8E4M3 { .. } => {
                Err(candle_gen::candle_core::Error::Msg(format!(
                    "krea NVFP4: `{weight_key}` is planned as `{}` but the reader returned a \
                     packed fp8 container; the plan and reader disagree about this row's codec",
                    NVFP4_CODEC.codec_id
                )))
            }
        };
    }

    let weight = w.get(&weight_key)?;
    let device = weight.device().clone();
    let expected = plan.representation(
        base,
        w.nvfp4_capability(
            &weight_key,
            Some([weight.dim(0)?, weight.dim(1)?]),
            plan.nvfp4_context(),
        ),
    );
    // sc-12274: build against the plan's ONE shared per-device cuBLASLt handle. `from_dense` would
    // construct a private handle here — and its eager 32 MiB workspace — for every one of the trunk's
    // 260 projections.
    let lin = Nvfp4Linear::from_dense_in(&weight, dense_bias, &device, act, plan.nvfp4_context())?;
    check_representation(
        base,
        expected,
        lin.regime() == Nvfp4Regime::Fp4W4A4,
        lin.fallback_cause(),
    )?;
    Ok(QLinear::Nvfp4(Nvfp4Proj::new(lin, base, plan, act)))
}

/// Cross the representation the role table + capability facts predicted for `base` against the one
/// the constructed projection actually serves (sc-12121, epic E4).
///
/// The prediction never *routes* anything — the logical plan owns residency and `Nvfp4Linear` owns
/// the construction gate — so this is the seam that keeps the reported role honest. A disagreement
/// means one of the two drifted, and the only safe response is to say so by name: a run that
/// silently serves dense BF16 while reporting native NVFP4 is exactly the dishonest reporting E4
/// forbids.
///
/// # Reconciling the two unpredictable causes (sc-12121 review fix)
///
/// `fallback` is the constructed layer's **own** report of why it is dense
/// ([`Nvfp4Linear::fallback_cause`]), `None` for a layer that is not an `Nvfp4Linear` at all (the
/// preserved-dense / reader-dense arms). Two of its causes —
/// [`Nvfp4Fallback::DeviceMismatch`] and [`Nvfp4Fallback::StagingFailed`] — are runtime accidents
/// that **no** [`Nvfp4Capability`] field can see, so a prediction of `PackedW4A4` against them is
/// not table drift; it is a transparent degradation that has always been legal. Those are
/// reconciled by rewriting the expectation via [`dense_reason_for_fallback`] before comparing,
/// which keeps the receipt honest (the projection reports `DenseBf16(DeviceMismatch)`, not a
/// fictional `PackedW4A4`) without aborting a whole trunk load over a driver accident.
///
/// Every other cause is one the capability facts *do* model, so a disagreement there is still a
/// hard error — in both directions.
fn check_representation(
    base: &str,
    expected: ExecutionRole,
    packed_w4a4: bool,
    fallback: Option<Nvfp4Fallback>,
) -> Result<()> {
    if expected.is_packed_w4a4() == packed_w4a4 {
        return Ok(());
    }
    // Safe direction only, and only for a cause no capability probe could have predicted.
    if !packed_w4a4 {
        if let Some(reason) = fallback.and_then(crate::nvfp4_dit::dense_reason_for_fallback) {
            eprintln!(
                "[sc-12121] krea nvfp4: `{base}` was predicted {expected:?} but the layer fell back \
                 to dense bf16 ({reason:?}); this cause is invisible to the plan-time capability \
                 facts, so the load continues. `Nvfp4Report` reads the layer's own regime, so the \
                 run still accounts this projection as dense, never as fp4-lit"
            );
            return Ok(());
        }
    }
    let served = if packed_w4a4 {
        "packed W4A4"
    } else {
        "dense bf16"
    };
    Err(Error::Msg(format!(
        "krea nvfp4 (sc-12121): `{base}` was constructed as {served}, but its execution role is \
         {expected:?}; the role table and the constructed projection disagree about this layer's \
         representation"
    )))
}

/// **Packed-detecting** [`QEmbedding`] loader (sc-9411): transfer its prepared device-format table when
/// the component is a packed tier and `{base}.scales` is present (dequantized to the component dtype —
/// dtype parity with the dense table), else a dense [`Embedding`] from `{base}.weight` (`hidden`
/// inferred from the stored `[vocab, hidden]` shape). The Krea Qwen3-VL TE keeps `embed_tokens` **dense** in the
/// hosted q4/q8 tiers, so today this takes the dense arm; the packed arm is the future-proof path (and
/// guards against a silent dense read of u32 codes should a tier ever pack the table).
pub fn embedding_detect(w: &Weights, base: &str) -> Result<QEmbedding> {
    let scales_key = format!("{base}.scales");
    if let (Some(_cfg), true) = (w.packed(), w.contains(&scales_key)) {
        // Dequantize the packed table to **f32** (the encoder's compute dtype), not `w.dtype()`
        // (sc-12828): the TE now stores its weights bf16, but the embedding is upcast to f32 in the
        // forward, so a packed embed must dequantize to f32 to stay bit-identical to the old f32 store
        // (a dequant to bf16 would round the rows before the widen). Uniform with the sibling
        // boogu/ideogram ports, which pack this table on their MLX tiers.
        let qtensor = w.load_packed_device_format(base)?;
        return QEmbedding::packed_device_format(qtensor, DType::F32);
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
    use crate::nvfp4_dit::DenseReason;
    use candle_gen::candle_core::safetensors;
    use candle_gen::candle_nn::Module;
    use candle_gen::gen_core::checkpoint_codec::TensorCodecSpec;
    use candle_gen::gen_core::checkpoint_facts::ExecutionRepresentation;
    use candle_gen::quant::Nvfp4Tensor;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};

    /// The Krea MLX tier's group size (64) — the one carried from `config.json`.
    const G: usize = 64;

    /// A `Weights` over one dense `[64, 64]` f32 tensor with **no logical plan** — so
    /// `planned(key)` is `None` for every key.
    fn unplanned_weights(tmp: &tempfile::TempDir, key: &str) -> Weights {
        let dev = Device::Cpu;
        let path = tmp.path().join("unplanned.safetensors");
        safetensors::save(
            &HashMap::from([(
                key.to_owned(),
                Tensor::zeros((64, 64), DType::F32, &dev).unwrap(),
            )]),
            &path,
        )
        .unwrap();
        Weights::from_file(&path, &dev, DType::F32).unwrap()
    }

    /// **No plan row AND no dense shape asserts `layout_native` on nothing** (sc-12121 review fix).
    ///
    /// The `None`-plan-row arm used to answer the grid question with `dense_shape.is_none_or(..)`,
    /// which short-circuits to `true` when the caller passes no shape — so a native component with
    /// no plan row would have been predicted `PackedW4A4` on grounds nothing ever checked, breaking
    /// the "every field is read off something real" claim in `nvfp4_capability`'s own doc. The
    /// combination is now explicitly refused.
    #[test]
    #[should_panic(expected = "no plan row AND no dense shape")]
    fn capability_refuses_to_assume_a_grid_with_neither_a_plan_row_nor_a_shape() {
        let tmp = tempfile::tempdir().unwrap();
        let w = unplanned_weights(&tmp, "transformer_blocks.0.attn.to_q.weight");
        let plan = DitPlan::nvfp4(Nvfp4Quant::Mixed);
        let _ = w.nvfp4_capability(
            "transformer_blocks.0.attn.to_q.weight",
            None,
            plan.nvfp4_context(),
        );
    }

    /// The same key **with** a dense shape is answered off something real — the shape — and an
    /// eligible `[64, 64]` grid reports `layout_native: true`, an ineligible `[64, 16]` one `false`.
    #[test]
    fn capability_reads_layout_native_off_the_dense_shape_when_there_is_no_plan_row() {
        let tmp = tempfile::tempdir().unwrap();
        let w = unplanned_weights(&tmp, "transformer_blocks.0.attn.to_q.weight");
        let plan = DitPlan::nvfp4(Nvfp4Quant::Mixed);
        let cap = |shape: [usize; 2]| {
            w.nvfp4_capability(
                "transformer_blocks.0.attn.to_q.weight",
                Some(shape),
                plan.nvfp4_context(),
            )
            .layout_native
        };
        assert!(cap([64, 64]), "K_pad 64 % 32 == 0 and N 64 % 16 == 0");
        assert!(
            !cap([64, 16]),
            "K_pad 16 is not a multiple of NVFP4_K_ALIGN"
        );
    }

    /// **The hard refusal keeps the unsafe direction, and only the two runtime accidents soften the
    /// safe one** (sc-12121 review fix).
    ///
    /// Before the fix `check_representation` erred on *any* disagreement, which aborted a whole
    /// trunk load — blaming the role table — for a shared-context device mismatch or a weight-staging
    /// failure, neither of which any `Nvfp4Capability` field models. After it:
    ///
    /// * constructed **packed** while the table predicted dense — the dishonest-reporting direction —
    ///   is still an `Err`, with **no** softening for any fallback cause (a packed layer reports
    ///   `fallback_cause() == None` anyway);
    /// * constructed **dense** while the table predicted packed is an `Err` unless the layer itself
    ///   names `DeviceMismatch` or `StagingFailed`.
    #[test]
    fn check_representation_refuses_drift_but_not_the_two_runtime_accidents() {
        let packed = ExecutionRole::PackedW4A4;
        let dense = ExecutionRole::DenseBf16(DenseReason::PostNonlinearity);

        // Agreement, either way round.
        assert!(check_representation("k", packed, true, None).is_ok());
        assert!(check_representation("k", dense, false, None).is_ok());

        // UNSAFE direction: constructed packed, table said dense. Never softened.
        let e = check_representation("k", dense, true, None).unwrap_err();
        assert!(format!("{e}").contains("constructed as packed W4A4"), "{e}");
        for cause in [Nvfp4Fallback::DeviceMismatch, Nvfp4Fallback::StagingFailed] {
            assert!(
                check_representation("k", dense, true, Some(cause)).is_err(),
                "{cause:?} must not excuse a layer that actually lit the FP4 cores"
            );
        }

        // SAFE direction: constructed dense, table said packed.
        let e = check_representation("k", packed, false, None).unwrap_err();
        assert!(format!("{e}").contains("constructed as dense bf16"), "{e}");
        for cause in [
            Nvfp4Fallback::W4A16Requested,
            Nvfp4Fallback::NotCudaDevice,
            Nvfp4Fallback::ShapeIneligible,
            Nvfp4Fallback::NoDeviceHandle,
            Nvfp4Fallback::NoFusedQuantizer,
        ] {
            assert!(
                check_representation("k", packed, false, Some(cause)).is_err(),
                "{cause:?} IS modelled by Nvfp4Capability — a disagreement here is a real defect"
            );
        }
        for cause in [Nvfp4Fallback::DeviceMismatch, Nvfp4Fallback::StagingFailed] {
            assert!(
                check_representation("k", packed, false, Some(cause)).is_ok(),
                "{cause:?} is invisible to every capability fact — aborting the load would blame \
                 the role table for a driver accident"
            );
        }
    }

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

    /// Replace `target` while a provider has the original file mapped, exercising the exact
    /// production failure mode on every platform: the open mmap keeps consuming the original object
    /// while the selected path comes to name a new one. A replacing rename is the only swap that can
    /// do that — Windows refuses to truncate or write a file with an open mapped section
    /// (ERROR_USER_MAPPED_FILE, 1224), so the read+write form this used off-Unix could only ever
    /// fail. The retained fingerprint still rejects the swap on Windows, via the file id and change
    /// time, which differ even when the two files share a size and mtime.
    ///
    /// Returns the outcome rather than asserting it: callers run this between two barriers, where an
    /// early return would strand the reader on one and hang the test binary.
    fn replace_mapped_fixture(replacement: &Path, target: &Path) -> std::io::Result<()> {
        std::fs::rename(replacement, target)
    }

    /// The Krea File loader's guard must end only after Candle has copied the last requested tensor
    /// out of the retained mmap (and synchronized any asynchronous device transfer). A pre-only check
    /// would let this barrier-controlled replacement return a mixed-provenance materialization.
    #[test]
    fn file_pin_postchecks_after_actual_krea_tensor_materialization() -> Result<()> {
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().expect("temp dir");
        let source = dir.path().join("krea.safetensors");
        let replacement = dir.path().join("krea.replacement.safetensors");
        let elements = 64 * 1024;

        safetensors::save(
            &HashMap::from([
                (
                    "phase.first".to_owned(),
                    Tensor::from_vec(vec![0.25_f32; elements], elements, &dev)?,
                ),
                (
                    "phase.last".to_owned(),
                    Tensor::from_vec(vec![0.75_f32; elements], elements, &dev)?,
                ),
            ]),
            &source,
        )?;
        safetensors::save(
            &HashMap::from([
                (
                    "phase.first".to_owned(),
                    Tensor::from_vec(vec![-0.25_f32; elements], elements, &dev)?,
                ),
                (
                    "phase.last".to_owned(),
                    Tensor::from_vec(vec![-0.75_f32; elements], elements, &dev)?,
                ),
            ]),
            &replacement,
        )?;

        let weights = Weights::from_file(&source, &dev, DType::F32)?;
        let first_consumed = Arc::new(Barrier::new(2));
        let replacement_done = Arc::new(Barrier::new(2));
        let last_consumed = Arc::new(AtomicBool::new(false));

        let writer_source = source.clone();
        let writer_replacement = replacement.clone();
        let writer_first = Arc::clone(&first_consumed);
        let writer_done = Arc::clone(&replacement_done);
        let writer = std::thread::spawn(move || {
            writer_first.wait();
            let swapped = replace_mapped_fixture(&writer_replacement, &writer_source);
            // Release the reader whatever the swap did; the outcome is asserted on `join` below.
            writer_done.wait();
            swapped
        });

        let consumed = Arc::clone(&last_consumed);
        let outcome = weights.read_source_unchanged(|| {
            let first = weights.get("phase.first")?.to_vec1::<f32>()?;
            assert!(first.iter().all(|value| *value == 0.25));
            first_consumed.wait();
            replacement_done.wait();

            // This is an actual Krea provider `Weights::get`, not a raw File read. Converting the
            // tensor to a Vec forces CPU payload consumption; synchronize is the same final boundary
            // the CUDA loader uses before allowing the pin's post-check to run.
            let last = weights.get("phase.last")?.to_vec1::<f32>()?;
            assert_eq!(last.len(), elements);
            dev.synchronize()?;
            consumed.store(true, Ordering::SeqCst);
            Ok(())
        });
        writer
            .join()
            .expect("replacement writer")
            .expect("replace the mapped source mid-read");

        let error = match outcome {
            Ok(()) => {
                panic!("replacement during materialization must invalidate the Krea file pin")
            }
            Err(error) => error.to_string(),
        };

        assert!(
            last_consumed.load(Ordering::SeqCst),
            "the provider must consume the final tensor before the post-check rejects the mutation"
        );
        assert!(error.contains("changed after load"), "unexpected: {error}");
        Ok(())
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

        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
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

        Ok(())
    }

    /// F-189: a caller-provisioned packed Krea component may be an immutable snapshot. The loader
    /// keeps the packed projection active and places its file-backed Candle representation under the
    /// configured external cache root instead of requiring a write beside the model.
    #[cfg(unix)]
    #[test]
    fn packed_krea_loads_from_a_cold_read_only_snapshot() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let dev = Device::Cpu;
        let (out_dim, in_dim) = (128usize, 256usize);
        let (wq, scales, biases, grid) = q4_packed(out_dim, in_dim);
        let map = HashMap::from([
            ("attn.to_q.weight".to_owned(), wq),
            ("attn.to_q.scales".to_owned(), scales),
            ("attn.to_q.biases".to_owned(), biases),
        ]);
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let component = root.join("snapshot/transformer");
        let external = root.join("external-cache");
        write_component(&component, map, true);
        // Keep this branch deterministic under root-capable test runners; chmod is the production
        // condition, while the regular-file obstruction prevents privileged writes at the cache path.
        std::fs::write(
            component.join(".candle-device-format-v1"),
            b"immutable snapshot entry",
        )?;
        std::fs::set_permissions(&component, std::fs::Permissions::from_mode(0o555))?;
        let result =
            Weights::from_dir_with_external_cache_root(&component, &dev, DType::F32, &external);
        std::fs::set_permissions(&component, std::fs::Permissions::from_mode(0o755))?;

        let weights = result?;
        let sidecars = weights
            .packed_sidecars()
            .expect("packed Krea component prepares sidecars");
        assert!(sidecars.cache_dir().starts_with(&external));
        assert_eq!(sidecars.created_count(), 1);
        let packed = linear_detect(&weights, "attn.to_q", false)?;
        assert!(
            packed.is_packed(),
            "read-only load must retain packed behavior"
        );
        let dense = QLinear::dense(Linear::new(grid, None));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        assert!(cosine(&packed.forward(&x)?, &dense.forward(&x)?) > 0.99999);

        drop(weights);
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
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        write_component(&dir, map, false);

        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        assert!(w.packed().is_none(), "no quantization block ⇒ dense tier");
        assert!(!linear_detect(&w, "attn.to_q", false)?.is_packed());
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
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
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
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
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
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        write_component(&dir, map, true);
        let w = Weights::from_dir(&dir, &dev, DType::F32)?;
        let base = w.get_cpu_merge_base("attn.to_q.weight")?;
        assert_eq!(base.dims(), &[out_dim, in_dim], "reconstructed dense shape");
        assert!(
            cosine(&base, &grid) > 0.99999,
            "reconstructed base must equal the affine grid"
        );

        // Dense tier: base is the on-disk weight (identity round-trip).
        let dense_w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let mut dmap: HashMap<String, Tensor> = HashMap::new();
        dmap.insert("attn.to_q.weight".into(), dense_w.clone());
        let ddir_tmp = tempfile::tempdir().unwrap();
        let ddir = ddir_tmp.path().to_path_buf();
        write_component(&ddir, dmap, false);
        let dw = Weights::from_dir(&ddir, &dev, DType::F32)?;
        let dbase = dw.get_cpu_merge_base("attn.to_q.weight")?;
        let dev_max = (dbase.sub(&dense_w)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "dense tier base is the on-disk weight");
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
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
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
        Ok(())
    }

    /// `read_packed_config` distinguishes absent-vs-corrupt (sc-9426, F-073 sibling — the flux2
    /// `component_is_packed` twin): a `quantization` block → packed `Some`, a plain config or a
    /// genuinely-absent `config.json` → dense `None` (unchanged), but a *present-but-corrupt*
    /// `config.json` (malformed JSON, e.g. a partial download) errors loudly naming the file instead
    /// of silently swallowing to the dense path.
    #[test]
    fn read_packed_config_absent_vs_corrupt() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();

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

        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("krea2_int8_convrot.safetensors");
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

        // SC-16453: the composable control DiT must select this same native ConvRot projection rather
        // than handing the I8 tensor to its dense/adapter loader. This is the exact boundary the first
        // real strict-pose probe exposed (`unsupported safetensor dtype I8`). Forward parity proves the
        // wrapper retains the online rotation rather than merely carrying a ConvRot-looking enum tag.
        let control =
            crate::train_dit::lora_proj_packed(&w, "transformer_blocks.0.attn.to_q", false)?;
        assert!(
            control.is_convrot(),
            "the control-inference projection loader must preserve ConvRot identity"
        );
        let control_got = control.forward(&x)?.to_dtype(DType::F32)?;
        let control_cos = cosine(&control_got, &want);
        assert!(
            control_cos > 0.99,
            "control ConvRot online-rotation vs canonical cosine {control_cos:.5}"
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
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("m.safetensors");
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
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
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
        write_plain_int8_native_file_with_shape(
            path,
            descriptor,
            vec![2, 3],
            &[
                1_u8,
                (-2_i8) as u8,
                3_u8,
                (-4_i8) as u8,
                5_u8,
                (-6_i8) as u8,
            ],
            scale_shape,
            scales,
        );
    }

    fn write_plain_int8_native_file_with_shape(
        path: &Path,
        descriptor: &str,
        weight_shape: Vec<usize>,
        codes: &[u8],
        scale_shape: Vec<usize>,
        scales: &[f32],
    ) {
        let scale_bytes: Vec<u8> = scales
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let descriptor_bytes = descriptor.as_bytes();
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            "model.diffusion_model.blocks.0.attn.wq.weight",
            ::safetensors::tensor::TensorView::new(::safetensors::Dtype::I8, weight_shape, codes)
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

    /// A small but **architecturally coherent** Krea 2 config the NVFP4 fixture below is built to.
    ///
    /// Coherence is load-bearing since sc-20651: the import declares its logical shapes from a
    /// config, so a fixture whose tensors do not belong to one architecture cannot be planned at
    /// all. `hidden = heads · head_dim = 4 · 16`, `head_dim = sum(axes_dims_rope)`, `heads % kv_heads
    /// == 0`, `text_hidden = text_heads · head_dim` — every invariant `Krea2Config::validate`
    /// enforces, at the smallest widths that are still NVFP4-legal (both stored axes 16-aligned).
    fn kitchen_nvfp4_config() -> crate::Krea2Config {
        let cfg = crate::Krea2Config {
            in_channels: 16,
            patch_size: 2,
            hidden_size: 64,
            num_attention_heads: 4,
            num_kv_heads: 2,
            attention_head_dim: 16,
            num_layers: 1,
            intermediate_size: 128,
            norm_eps: 1e-5,
            axes_dims_rope: [4, 6, 6],
            rope_theta: 1000.0,
            timestep_embed_dim: 32,
            num_text_layers: 2,
            num_layerwise_text_blocks: 1,
            num_refiner_text_blocks: 1,
            text_hidden_dim: 32,
            text_intermediate_size: 64,
            text_num_attention_heads: 2,
            text_num_kv_heads: 1,
        };
        cfg.validate()
            .expect("the fixture architecture is coherent");
        cfg
    }

    /// A single-projection Kitchen NVFP4 native file plus one dense sibling, carrying the
    /// **descriptor** a real Kitchen export carries (`__metadata__._quantization_metadata`).
    ///
    /// The descriptor is not decoration: it is what makes the layer NVFP4. Before sc-20651 this
    /// fixture had none and the Candle import still took the NVFP4 path — off `dtype == U8` alone —
    /// which is exactly the defect the epic's codec seam exists to remove.
    fn write_kitchen_nvfp4_native_file(path: &Path) {
        let cfg = kitchen_nvfp4_config();
        // `attn.to_q` is `[q_dim, hidden]`, and Krea's `q_dim == hidden_size`, so the projection is
        // square at the fixture's width. Both axes are 16-aligned, so the stored grid IS the layer
        // (no ComfyUI padding) and the packed container can express it.
        let (rows, cols) = (cfg.q_dim(), cfg.hidden_size);
        let mut packed = vec![0u8; rows * cols / 2];
        packed[0] = 0x12; // Kitchen hi-first: even code 1, odd code 2.
        let mut block_scales = vec![0u8; Nvfp4Tensor::scale_tensor_len(rows, cols)];
        block_scales[Nvfp4Tensor::scale_offset_for(0, 0, rows)] = 0x38; // E4M3 1.0.
        let global_scale = 2.0f32.to_le_bytes();
        // The swizzled `to_blocked` scale grid, not `[rows, blocks]`: the plan validates the stored
        // companion against `gen_core::nvfp4_scale_shape`, which is the 128×4-atom padded shape.
        let scale_shape = candle_gen::gen_core::nvfp4_scale_shape([rows, cols]).to_vec();
        let dense = vec![0u8; cfg.hidden_size * cfg.in_channels * 4];

        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            "model.diffusion_model.blocks.0.attn.wq.weight",
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::U8,
                vec![rows, cols / 2],
                &packed,
            )
            .unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.blocks.0.attn.wq.weight_scale",
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::F8_E4M3,
                scale_shape,
                &block_scales,
            )
            .unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.blocks.0.attn.wq.weight_scale_2",
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::F32,
                vec![],
                &global_scale,
            )
            .unwrap(),
        );
        tensors.insert(
            "model.diffusion_model.first.weight",
            ::safetensors::tensor::TensorView::new(
                ::safetensors::Dtype::F32,
                vec![cfg.hidden_size, cfg.in_channels],
                &dense,
            )
            .unwrap(),
        );
        // Kitchen declares NVFP4 file-wide rather than per-tensor; the layer names are the file's
        // own (native, prefixed) `{layer}` bases.
        let metadata = std::collections::HashMap::from([(
            "_quantization_metadata".to_string(),
            r#"{"format_version": "1.0", "layers": {"model.diffusion_model.blocks.0.attn.wq": {"format": "nvfp4"}}}"#
                .to_string(),
        )]);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        ::safetensors::serialize_to_file(tensors, Some(metadata), path).unwrap();
    }

    /// [`write_kitchen_nvfp4_native_file`] whose one NVFP4 layer additionally declares
    /// `full_precision_matrix_mult` — the producer saying "do not run a quantized matmul here".
    fn write_kitchen_nvfp4_native_file_full_precision(path: &Path) {
        let declared = path.with_extension("declared.safetensors");
        write_kitchen_nvfp4_native_file(&declared);
        // SAFETY: read-only mmap of a file this test just wrote.
        let st = unsafe { MmapedSafetensors::new(&declared) }.expect("fixture opens");
        let owned: Vec<(String, ::safetensors::Dtype, Vec<usize>, Vec<u8>)> = st
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                (
                    name,
                    view.dtype(),
                    view.shape().to_vec(),
                    view.data().to_vec(),
                )
            })
            .collect();
        let tensors: std::collections::BTreeMap<&str, ::safetensors::tensor::TensorView<'_>> =
            owned
                .iter()
                .map(|(name, dtype, shape, data)| {
                    (
                        name.as_str(),
                        ::safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                            .unwrap(),
                    )
                })
                .collect();
        let metadata = std::collections::HashMap::from([(
            "_quantization_metadata".to_string(),
            r#"{"format_version": "1.0", "layers": {"model.diffusion_model.blocks.0.attn.wq": {"format": "nvfp4", "full_precision_matrix_mult": true}}}"#
                .to_string(),
        )]);
        ::safetensors::serialize_to_file(tensors, Some(metadata), path).unwrap();
        std::fs::remove_file(&declared).ok();
    }

    /// [`write_kitchen_nvfp4_native_file`] with the `_quantization_metadata` declaration REMOVED and
    /// nothing else changed — the structural NVFP4 triplet is still on disk, so a storage-shape
    /// predicate still reads it as NVFP4.
    fn write_kitchen_nvfp4_native_file_without_declaration(path: &Path) {
        let declared = path.with_extension("declared.safetensors");
        write_kitchen_nvfp4_native_file(&declared);
        // SAFETY: read-only mmap of a file this test just wrote.
        let st = unsafe { MmapedSafetensors::new(&declared) }.expect("fixture opens");
        let owned: Vec<(String, ::safetensors::Dtype, Vec<usize>, Vec<u8>)> = st
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                (
                    name,
                    view.dtype(),
                    view.shape().to_vec(),
                    view.data().to_vec(),
                )
            })
            .collect();
        let tensors: std::collections::BTreeMap<&str, ::safetensors::tensor::TensorView<'_>> =
            owned
                .iter()
                .map(|(name, dtype, shape, data)| {
                    (
                        name.as_str(),
                        ::safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                            .unwrap(),
                    )
                })
                .collect();
        ::safetensors::serialize_to_file(tensors, None, path).unwrap();
        std::fs::remove_file(&declared).ok();
    }

    #[test]
    fn kitchen_nvfp4_native_file_uses_prepacked_projection_and_preserves_dense_layers() -> Result<()>
    {
        let dev = Device::Cpu;
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("kreamania_variant7.safetensors");
        write_kitchen_nvfp4_native_file(&path);

        let cfg = kitchen_nvfp4_config();
        let w = Weights::from_native_file_for(
            &path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )?;
        assert!(w.uses_native_keys());
        assert!(w.is_native_nvfp4());
        assert!(!w.is_plain_int8());
        assert!(!w.is_convrot());
        assert_eq!(
            w.shape("transformer_blocks.0.attn.to_q.weight"),
            Some(vec![cfg.q_dim(), cfg.hidden_size]),
            "architecture validation must see the logical, not packed, shape"
        );

        // sc-20651: the DESCRIPTOR is what makes this layer NVFP4, and the plan is where that lives.
        let planned = w
            .logical_plan()
            .expect("a native single file compiles a plan")
            .tensors
            .iter()
            .find(|t| t.logical_key == "transformer_blocks.0.attn.to_q.weight")
            .expect("the projection is planned");
        assert_eq!(planned.codec_id, NVFP4_CODEC.codec_id);
        assert!(
            matches!(&planned.codec, TensorCodecSpec::Nvfp4 { block_scale, global_scale, .. }
                if block_scale == "model.diffusion_model.blocks.0.attn.wq.weight_scale"
                    && global_scale == "model.diffusion_model.blocks.0.attn.wq.weight_scale_2"),
            "the companion keys must come from the plan, not from concatenation at the read site: \
             {:?}",
            planned.codec
        );

        // sc-21482: this load is on the CPU, so the residency policy priced every NVFP4 row as the
        // DENSE fallback — and the shared reader materializes exactly that declaration: the
        // codec's reference dequant, not a packed container an ineligible device cannot serve.
        assert_eq!(
            planned.residency.mode,
            candle_gen::gen_core::checkpoint_codec::ResidencyMode::Dense,
            "a CPU load must price the NVFP4 row as the dense fallback"
        );
        let plan = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(1);
        let quantized = linear_detect_planned(&w, "transformer_blocks.0.attn.to_q", false, &plan)?;
        assert!(
            quantized.nvfp4().is_none(),
            "an ineligible (CPU) device must materialize the declared dense fallback"
        );
        // Golden: the dense fallback is the codec's exact two-level decode (codes 1, 2 × block
        // scale 1.0 × global 2.0), not the stored bytes reinterpreted.
        let got = w.get("transformer_blocks.0.attn.to_q.weight")?;
        let got = got.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(&got[..2], &[1.0, 2.0]);

        // The receipt is MEASURED off what really materialized, and it matches the plan's pricing
        // for the rows read so far (AC: dense-fallback rows have matching receipts).
        let receipt = w
            .logical_weight_receipt()
            .expect("the plan yields a receipt");
        assert_eq!(receipt.mapping_id, "krea-native-to-diffusers-v1");
        let nvfp4_row = receipt
            .residency
            .iter()
            // Keyed on the FULL row identity. Matching on `codec_id` alone would silently accept
            // a `NativePacked` row here — the exact confusion sc-21484 split the rows to prevent —
            // so this half must name `DenseFallback` and fail if the reader produced anything else.
            .find(|row| {
                row.codec_id == NVFP4_CODEC.codec_id
                    && row.representation == ExecutionRepresentation::DenseFallback
            })
            .expect("the materialized NVFP4 row is reported as a dense fallback");
        assert_eq!(nvfp4_row.tensor_count, 1);
        assert_eq!(
            nvfp4_row.resident_bytes, planned.residency.resident_bytes,
            "the measured dense-fallback residency must equal the plan's declared pricing"
        );
        assert!(
            receipt
                .residency
                .iter()
                .all(|row| row.representation == ExecutionRepresentation::DenseFallback),
            "a CPU load has no native-packed row at all"
        );

        // sc-12121: the plan (`Dense`), the constructed projection (`QLinear::dense`) and the role
        // table all name the same representation, and the table names WHICH fact decided it.
        assert_eq!(
            plan.representation(
                "transformer_blocks.0.attn.to_q",
                w.nvfp4_capability(
                    "transformer_blocks.0.attn.to_q.weight",
                    None,
                    plan.nvfp4_context()
                )
            ),
            ExecutionRole::DenseBf16(crate::nvfp4_dit::DenseReason::NoNvfp4Hardware),
            "a CPU load is below the sm_120 floor — that, not the layer's name, is the reason"
        );

        let preserved = linear_detect_planned(&w, "img_in", false, &plan)?;
        assert!(
            preserved.nvfp4().is_none(),
            "Kitchen profile's dense BF16/F32 projections must not be requantized"
        );
        // `img_in` is not an NVFP4 row at all in Kitchen's profile — a distinct dense reason from
        // the hardware one above, and the one that must survive even on `sm_120`.
        assert_eq!(
            plan.representation(
                "img_in",
                Nvfp4Capability {
                    nvfp4_device: true,
                    fused_quantizer: true,
                    ..w.nvfp4_capability("img_in.weight", None, plan.nvfp4_context())
                }
            ),
            ExecutionRole::DenseBf16(crate::nvfp4_dit::DenseReason::PreservedDense)
        );
        Ok(())
    }

    /// **sc-21484: the loaded checkpoint exposes the three facts, tied to the verified source
    /// binding.** The provider-level handoff the SceneWorks half consumes — source-codec inventory,
    /// host capability, measured receipt — asserted on the same Kitchen fixture through both the
    /// CPU (dense-fallback) and the forced-packed route.
    #[test]
    fn checkpoint_weight_facts_report_the_source_the_capability_and_the_representation(
    ) -> Result<()> {
        let dev = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("kreamania_variant7.safetensors");
        write_kitchen_nvfp4_native_file(&path);
        let cfg = kitchen_nvfp4_config();
        let dit = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(1);

        // ---- the CPU host: NVFP4 source, dense-fallback execution ---------------------------
        let w = Weights::from_native_file_for(
            &path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )?;
        for base in ["transformer_blocks.0.attn.to_q", "img_in"] {
            linear_detect_planned(&w, base, false, &dit)?;
        }
        let facts = w
            .checkpoint_weight_facts()?
            .expect("a single-file native import has a plan");
        assert!(
            facts.source().declares(NVFP4_CODEC.codec_id),
            "the source stores nvfp4-v1 whatever this host can run"
        );
        assert!(
            !facts.executes_natively(NVFP4_CODEC.codec_id),
            "a CPU host executes the declared dense fallback and must never be labelled native"
        );
        assert!(facts.capability().is_dense_only());
        assert!(facts.is_complete(), "every projection was constructed");
        // AC1: tied to the verified source binding, not to a path string.
        let binding = facts.source_binding().expect("the pin is carried through");
        assert_eq!(
            binding.canonical_path(),
            std::fs::canonicalize(&path).unwrap()
        );
        // The **stable** renderable surface (sc-21484 review): size and a token that mean the same
        // thing on every OS, unlike the cfg-gated `FileStatFingerprint`.
        assert_eq!(
            binding.size_bytes(),
            std::fs::metadata(&path).unwrap().len(),
            "size_bytes is the target's real size"
        );
        assert_eq!(
            binding.stable_token(),
            format!("kreamania_variant7.safetensors@{}", binding.size_bytes()),
            "the token is `<file-name>@<size>` — no separators, no inode, no machine-local path"
        );
        assert_eq!(binding.to_string(), binding.stable_token());
        assert!(
            !binding.stable_token().contains(std::path::MAIN_SEPARATOR),
            "a stable token must not carry a platform-specific path separator"
        );

        // ---- the forced-packed (sm_120-equivalent) host: same source, native execution -------
        let packed = Weights::from_native_file_forcing_packed_nvfp4(
            &path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )?;
        for base in ["transformer_blocks.0.attn.to_q", "img_in"] {
            linear_detect_planned(&packed, base, false, &dit)?;
        }
        let packed_facts = packed
            .checkpoint_weight_facts()?
            .expect("a single-file native import has a plan");
        assert!(packed_facts.executes_natively(NVFP4_CODEC.codec_id));
        assert!(packed_facts
            .capability()
            .executes_natively(NVFP4_CODEC.codec_id));
        // Fact 2 is the same file on both hosts; only fact 3 moved.
        assert_eq!(
            packed_facts
                .source()
                .entry(NVFP4_CODEC.codec_id)
                .unwrap()
                .tensor_count,
            facts
                .source()
                .entry(NVFP4_CODEC.codec_id)
                .unwrap()
                .tensor_count
        );
        assert!(packed_facts.resident_bytes() < facts.resident_bytes());
        Ok(())
    }

    /// **The packed-native route, plan-declared (sc-21482).** With the `sm_120` residency forced
    /// on (the container is host-side; only the GEMM is hardware-gated), the SAME fixture's plan
    /// prices the projection `Packed`, the shared reader hands back the canonical repacked
    /// container (Kitchen's hi-first nibbles swapped), `linear_detect_planned` constructs an NVFP4
    /// projection from it, and the measured receipt equals the plan's packed pricing.
    #[test]
    fn kitchen_nvfp4_native_file_packs_when_the_plan_prices_packed() -> Result<()> {
        let dev = Device::Cpu;
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("kreamania_variant7.safetensors");
        write_kitchen_nvfp4_native_file(&path);
        let cfg = kitchen_nvfp4_config();
        let w = Weights::from_native_file_forcing_packed_nvfp4(
            &path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )?;
        let planned = w
            .logical_plan()
            .expect("a native single file compiles a plan")
            .tensors
            .iter()
            .find(|t| t.logical_key == "transformer_blocks.0.attn.to_q.weight")
            .expect("the projection is planned")
            .clone();
        assert_eq!(
            planned.residency.mode,
            candle_gen::gen_core::checkpoint_codec::ResidencyMode::Packed
        );

        // The shared reader returns the packed container the plan declared…
        let LogicalTensor::PackedNvfp4 { tensor, .. } =
            w.read_planned("transformer_blocks.0.attn.to_q.weight")?
        else {
            panic!("a packed-planned row must materialize PackedNvfp4");
        };
        assert_eq!(
            tensor.packed[0], 0x21,
            "Kitchen nibble order must be swapped"
        );
        assert_eq!(&tensor.dequantize_to_vec()[..2], &[1.0, 2.0]);

        // …`linear_detect_planned` constructs the NVFP4 projection from it…
        let plan = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(1);
        let quantized = linear_detect_planned(&w, "transformer_blocks.0.attn.to_q", false, &plan)?;
        assert!(quantized.nvfp4().is_some());

        // sc-12121, AC2: the plan priced this row PACKED, but the bound device is a CPU — so the
        // constructed linear serves dense bf16, and the role table says so for the same reason
        // (`NoNvfp4Hardware`). All three agree that this run holds dense bf16, and the execution
        // receipt reports the full 1.0 footprint rather than claiming a native NVFP4 one.
        let projection = quantized.nvfp4().expect("the NVFP4 arm was taken");
        assert_eq!(projection.regime(), Nvfp4Regime::DequantBf16);
        assert_eq!(
            plan.representation(
                "transformer_blocks.0.attn.to_q",
                w.nvfp4_capability(
                    "transformer_blocks.0.attn.to_q.weight",
                    None,
                    plan.nvfp4_context()
                )
            ),
            ExecutionRole::DenseBf16(crate::nvfp4_dit::DenseReason::NoNvfp4Hardware)
        );
        let mut report = crate::nvfp4_dit::Nvfp4Report::default();
        report.add(projection);
        assert_eq!((report.fp4_lit, report.dequant_bf16), (0, 1));
        assert!(
            (report.footprint_ratio() - 1.0).abs() < 1e-6,
            "a dense-bf16 run must not report an NVFP4 footprint (got {:.4})",
            report.footprint_ratio()
        );

        // …and the receipt row measured off the container equals the plan's packed pricing
        // (stored nibbles + retained scale companions).
        let receipt = w
            .logical_weight_receipt()
            .expect("the plan yields a receipt");
        let nvfp4_row = receipt
            .residency
            .iter()
            // The packed arm, named explicitly: `codec_id` alone would pass on a dense-fallback
            // row and this test's whole point is that the forced-packed route produced a
            // *native* one.
            .find(|row| {
                row.codec_id == NVFP4_CODEC.codec_id
                    && row.representation == ExecutionRepresentation::NativePacked
            })
            .expect("the materialized NVFP4 row is reported as native-packed");
        assert_eq!(nvfp4_row.tensor_count, 1);
        let companions: u64 = w
            .logical_plan()
            .unwrap()
            .companions
            .iter()
            .filter(|companion| companion.owner_physical_key == planned.physical_key)
            .map(|companion| companion.resident_bytes)
            .sum();
        assert_eq!(
            nvfp4_row.resident_bytes,
            planned.residency.resident_bytes + companions,
            "the measured packed residency must equal the plan's packed pricing"
        );
        Ok(())
    }

    /// **Corrupted scale payloads refuse before any projection is served (sc-21482).** The
    /// provider-owned whole-file scale scan is gone; the refusal is now the CODEC's, at the shared
    /// reader's materialization — same fixture, one block-scale byte given the E4M3 sign bit.
    ///
    /// Each corruption is run through **both** residency arms, because they are two different
    /// call sites of the same codec-owned check and only one of them lives inside `decode_nvfp4`:
    ///
    /// * `Dense` (a plain CPU open) — the refusal comes from inside `decode_nvfp4`.
    /// * `Packed` (residency forced on) — the repack copies scale bytes **verbatim**, so the only
    ///   thing between a sign-bit/NaN E4M3 scale and the FP4 GEMM is the explicit
    ///   `validate_nvfp4_block_scale_payload` call in `logical_weights.rs`'s `Packed` arm. Delete
    ///   that call and this half goes red (sc-21482 review).
    #[test]
    fn nvfp4_negative_block_scale_refuses_at_materialization() {
        for (corrupt_byte, expected) in [(0xB8u8, "sign bit"), (0x7Fu8, "NaN")] {
            let dev = Device::Cpu;
            let path_tmp = tempfile::tempdir().unwrap();
            let path = path_tmp.path().join("kreamania_variant7.safetensors");
            write_kitchen_nvfp4_native_file(&path);
            // Corrupt the first (real, element-governing) block scale in place.
            corrupt_first_block_scale(&path, corrupt_byte);
            let cfg = kitchen_nvfp4_config();
            let plan = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(1);

            // ── the dense-fallback arm (an ineligible device) ──
            // Header-level planning still succeeds — the plan never reads payload bytes…
            let dense = Weights::from_native_file_for(
                &path,
                &dev,
                DType::F32,
                DeclaredLogicalShapes::FromConfig(&cfg),
            )
            .expect("planning is header-level; payload corruption is a materialization refusal");
            assert_eq!(
                dense
                    .logical_plan()
                    .unwrap()
                    .tensors
                    .iter()
                    .find(|t| t.logical_key == "transformer_blocks.0.attn.to_q.weight")
                    .unwrap()
                    .residency
                    .mode,
                candle_gen::gen_core::checkpoint_codec::ResidencyMode::Dense,
                "this half must exercise the DENSE arm"
            );
            // …but materializing the projection refuses.
            let error =
                match linear_detect_planned(&dense, "transformer_blocks.0.attn.to_q", false, &plan)
                {
                    Ok(_) => panic!("a corrupted block scale must refuse on the dense arm"),
                    Err(error) => error.to_string(),
                };
            assert!(
                error.contains(expected),
                "the dense refusal must name the corruption ({expected}): {error}"
            );

            // ── the packed-native arm (an `sm_120`-eligible device, forced on here) ──
            let packed = Weights::from_native_file_forcing_packed_nvfp4(
                &path,
                &dev,
                DType::F32,
                DeclaredLogicalShapes::FromConfig(&cfg),
            )
            .expect("planning is header-level on the packed arm too");
            assert_eq!(
                packed
                    .logical_plan()
                    .unwrap()
                    .tensors
                    .iter()
                    .find(|t| t.logical_key == "transformer_blocks.0.attn.to_q.weight")
                    .unwrap()
                    .residency
                    .mode,
                candle_gen::gen_core::checkpoint_codec::ResidencyMode::Packed,
                "this half must exercise the PACKED arm — otherwise it witnesses nothing new"
            );
            let error = match linear_detect_planned(
                &packed,
                "transformer_blocks.0.attn.to_q",
                false,
                &plan,
            ) {
                Ok(_) => panic!(
                    "a corrupted block scale must refuse on the packed arm too — the repack \
                     copies scale bytes verbatim into the GEMM's operand"
                ),
                Err(error) => error.to_string(),
            };
            assert!(
                error.contains(expected),
                "the packed refusal must name the corruption ({expected}): {error}"
            );
            // The container must not have materialized at all: nothing corrupt is resident.
            assert!(
                packed
                    .logical_weight_receipt()
                    .expect("the plan yields a receipt")
                    .residency
                    .iter()
                    .all(|row| row.codec_id != NVFP4_CODEC.codec_id),
                "a refused packed row must leave no measured NVFP4 residency behind"
            );
        }
    }

    /// **AC1, inference half: a LINKED copy and a MANAGED copy compile the same plan and
    /// materialize the same values (sc-21482).**
    ///
    /// The real-artifact half of AC1 needs weights and hardware and lives in the `#[ignore]`d
    /// `nvfp4_shared_reader_real_weights` matrix. The *plan-equality* half needs neither: a
    /// hard link is exactly what "the same bytes reached by a second path" means, and the plan is
    /// compiled from the file's own header and descriptors. So it is asserted here, on the
    /// synthetic fixture, in milliseconds — and it would catch any path-dependence (a mapping keyed
    /// off the file name, a residency probed from the directory, a pin that varies by route) that
    /// the terminal lane would otherwise be the first to see.
    ///
    /// AC1's *semantic import plan* (`ImportPlanV1`) clause has no artifact in this repo — it is
    /// SceneWorks-side and is asserted at the epic's terminal story.
    #[test]
    fn linked_and_managed_copies_compile_the_same_plan_and_read_the_same_values() -> Result<()> {
        let dev = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let managed = tmp.path().join("managed/kreamania_variant7.safetensors");
        write_kitchen_nvfp4_native_file(&managed);
        // The "linked" route: the identical bytes reached through a second path in the same
        // filesystem. `hard_link` (not a copy) so there is genuinely one artifact, two names.
        let linked = tmp.path().join("linked/kreamania_variant7.safetensors");
        std::fs::create_dir_all(linked.parent().unwrap()).unwrap();
        std::fs::hard_link(&managed, &linked).expect("same-volume hard link");

        let cfg = kitchen_nvfp4_config();
        let managed_w = Weights::from_native_file_for(
            &managed,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )?;
        let linked_w = Weights::from_native_file_for(
            &linked,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )?;

        // The whole compiled plan, not a summary: mapping id, every tensor's codec/companions/
        // geometry/residency pricing, every companion row, and the source-byte total.
        assert_eq!(
            managed_w.logical_plan(),
            linked_w.logical_plan(),
            "the same artifact reached by two paths must compile one plan"
        );

        // …and the reader materializes identical values through both, for the NVFP4 projection
        // and the dense sibling alike.
        for key in ["transformer_blocks.0.attn.to_q.weight", "img_in.weight"] {
            let a = managed_w.get(key)?.flatten_all()?.to_vec1::<f32>()?;
            let b = linked_w.get(key)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(a, b, "`{key}` must read identically through both paths");
        }
        // The projection really is the NVFP4 one (a fixture that stopped being quantized would
        // make the equality above vacuous).
        assert!(managed_w.is_native_nvfp4());
        assert_eq!(
            managed_w
                .get("transformer_blocks.0.attn.to_q.weight")?
                .flatten_all()?
                .to_vec1::<f32>()?[..2],
            [1.0, 2.0]
        );
        Ok(())
    }

    /// **AC2: refusal happens before a generator is returned — which only holds while trunk
    /// construction materializes EVERY planned row (sc-21482 review).**
    ///
    /// The deleted `validate_native_nvfp4` scanned the whole file at open, so a payload defect in
    /// a layer nothing read still refused. The shared reader is incremental instead: a defect is
    /// found when its row materializes. That is equivalent *only* because the trunk reads the
    /// plan's whole surface. This asserts exactly that equivalence on the host, so a future read
    /// path that bypasses the reader for some row (and would silently stop payload-checking it) is
    /// caught here rather than by the terminal real-weight lane.
    #[test]
    fn constructing_every_projection_materializes_the_plans_whole_surface() -> Result<()> {
        let dev = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("kreamania_variant7.safetensors");
        write_kitchen_nvfp4_native_file(&path);
        let cfg = kitchen_nvfp4_config();
        let w = Weights::from_native_file_for(
            &path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )?;
        // Nothing read yet: the receipt is honest about the instant, not about the plan.
        assert_eq!(
            w.logical_weight_receipt().unwrap().tensor_count,
            0,
            "an unread import must report zero materialized tensors"
        );

        let dit_plan = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(1);
        // Construct the fixture's whole surface exactly as the trunk does — every projection
        // through `linear_detect_planned`.
        for base in ["transformer_blocks.0.attn.to_q", "img_in"] {
            linear_detect_planned(&w, base, false, &dit_plan)?;
        }

        let plan_count = w.logical_plan().unwrap().tensor_count();
        assert_eq!(
            w.logical_weight_receipt().unwrap().tensor_count,
            plan_count,
            "constructing every projection must materialize every PLANNED row — otherwise a row \
             the trunk skips is never payload-checked and AC2's \"refuses before a generator is \
             returned\" no longer holds"
        );
        assert!(plan_count >= 2, "the fixture must plan more than one row");
        Ok(())
    }

    /// **The `Packed`-plan dense-read refusal is reachable and names its escape hatch
    /// (sc-21482 review).** On `sm_120` the plan prices NVFP4 projections `Packed`, and a dense
    /// `get`/`get_f32` of such a row refuses rather than serving stored nibbles reinterpreted at
    /// the component dtype (which is what it silently did before this story). Forcing the packed
    /// residency makes that arm reachable on a CPU lane.
    #[test]
    fn dense_read_of_a_packed_planned_row_refuses_and_names_the_escape_hatch() -> Result<()> {
        let dev = Device::Cpu;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("kreamania_variant7.safetensors");
        write_kitchen_nvfp4_native_file(&path);
        let cfg = kitchen_nvfp4_config();
        let packed = Weights::from_native_file_forcing_packed_nvfp4(
            &path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )?;
        for error in [
            packed
                .get("transformer_blocks.0.attn.to_q.weight")
                .expect_err("a packed-planned row has no dense read"),
            packed
                .get_f32("transformer_blocks.0.attn.to_q.weight")
                .expect_err("…and `get_f32` refuses identically"),
        ] {
            let error = error.to_string();
            assert!(
                error.contains("planned packed-native"),
                "the refusal must say why: {error}"
            );
            assert!(
                error.contains("nvfp4_native") && error.contains("DENSE"),
                "the refusal must name the escape hatch — re-plan the row dense: {error}"
            );
        }
        // The dense sibling still reads normally through the same accessor.
        assert_eq!(
            packed.get("img_in.weight")?.dims(),
            [cfg.hidden_size, cfg.in_channels]
        );
        Ok(())
    }

    /// Overwrite the byte of the block-scale companion that governs (row 0, block 0) — the swizzled
    /// slot the fixture's real scale occupies.
    fn corrupt_first_block_scale(path: &Path, value: u8) {
        let cfg = kitchen_nvfp4_config();
        let (rows, cols) = (cfg.q_dim(), cfg.hidden_size);
        // SAFETY: read-only mmap of a file this test just wrote.
        let st = unsafe { MmapedSafetensors::new(path) }.expect("fixture opens");
        let owned: Vec<(String, ::safetensors::Dtype, Vec<usize>, Vec<u8>)> = st
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                let mut data = view.data().to_vec();
                if name.ends_with(".weight_scale") {
                    let index =
                        candle_gen::gen_core::nvfp4_swizzled_scale_index([rows, cols], 0, 0);
                    data[index] = value;
                }
                (name, view.dtype(), view.shape().to_vec(), data)
            })
            .collect();
        drop(st);
        let tensors: std::collections::BTreeMap<&str, ::safetensors::tensor::TensorView<'_>> =
            owned
                .iter()
                .map(|(name, dtype, shape, data)| {
                    (
                        name.as_str(),
                        ::safetensors::tensor::TensorView::new(*dtype, shape.clone(), data)
                            .unwrap(),
                    )
                })
                .collect();
        let metadata = std::collections::HashMap::from([(
            "_quantization_metadata".to_string(),
            r#"{"format_version": "1.0", "layers": {"model.diffusion_model.blocks.0.attn.wq": {"format": "nvfp4"}}}"#
                .to_string(),
        )]);
        ::safetensors::serialize_to_file(tensors, Some(metadata), path).unwrap();
    }

    /// **A dense sibling is not NVFP4 just because the file is.** `img_in` carries no NVFP4
    /// declaration, so the plan gives it a dense codec and the NVFP4 read site refuses it by name
    /// and by codec id rather than reaching for `{base}.weight_scale` and reporting a missing
    /// tensor. (This half does not by itself distinguish the old `dtype == U8` predicate — `img_in`
    /// is `F32` — that is what
    /// [`tests::an_undeclared_u8_projection_is_refused_at_open_instead_of_imported_as_nvfp4`]
    /// pins.)
    #[test]
    fn nvfp4_read_refuses_a_layer_the_descriptor_does_not_declare() {
        let dev = Device::Cpu;
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("kreamania_variant7.safetensors");
        write_kitchen_nvfp4_native_file(&path);
        let cfg = kitchen_nvfp4_config();
        let w = Weights::from_native_file_for(
            &path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )
        .expect("the fixture plans");

        assert!(!w.is_native_nvfp4_weight("img_in.weight"));
        // The shared reader materializes the dense sibling as exactly the DENSE codec row the plan
        // declared — never a packed container, even with the packed residency forced on.
        let forced = Weights::from_native_file_forcing_packed_nvfp4(
            &path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&kitchen_nvfp4_config()),
        )
        .expect("the fixture plans");
        assert!(matches!(
            forced.read_planned("img_in.weight").expect("dense read"),
            LogicalTensor::Dense(_)
        ));
    }

    /// **sc-20651 blocker 1: an UNDECLARED `U8` projection is refused at open, not imported as
    /// NVFP4.**
    ///
    /// This is the divergence the bespoke predicate could not see. The fixture is byte-identical to
    /// the Kitchen one except that its `_quantization_metadata` is gone: the structural triplet
    /// (`U8 [out, in/2]` + `F8_E4M3` blocked scales + `F32 weight_scale_2`) is still there, so
    /// `dtype == U8` still answers "NVFP4" and the old import decoded the layer as NVFP4 nibbles on
    /// nothing but storage shape. Some other producer's `U8` payload in that same shape would have
    /// been read as NVFP4 too, silently.
    ///
    /// With the plan compiled at open, the producer's declaration is the authority: an undescribed
    /// `U8` weight matches no registered safetensors codec row and the import refuses, naming the
    /// tensor, before any weight is read.
    #[test]
    fn an_undeclared_u8_projection_is_refused_at_open_instead_of_imported_as_nvfp4() {
        let dev = Device::Cpu;
        let path_tmp = tempfile::tempdir().unwrap();
        let declared = path_tmp.path().join("declared.safetensors");
        let undeclared = path_tmp.path().join("undeclared.safetensors");
        write_kitchen_nvfp4_native_file(&declared);
        write_kitchen_nvfp4_native_file_without_declaration(&undeclared);
        let cfg = kitchen_nvfp4_config();

        // The declared file imports.
        Weights::from_native_file_for(
            &declared,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )
        .expect("a declared Kitchen NVFP4 file imports");

        // The SAME tensors without the declaration do not.
        let error = Weights::from_native_file_for(
            &undeclared,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )
        .err()
        .expect("an undescribed U8 projection must not be imported as NVFP4 on storage shape alone")
        .to_string();
        assert!(
            error.contains("model.diffusion_model.blocks.0.attn.wq.weight"),
            "the refusal must name the tensor: {error}"
        );
        assert!(
            error.contains("uint8") && error.contains("no checkpoint codec is registered"),
            "the refusal must be the plan's unregistered-format one: {error}"
        );
    }

    /// **A `full_precision_matrix_mult` layer takes the DENSE fallback, it does not fail the load.**
    ///
    /// The flag is the producer saying "this layer must not run a quantized matmul" — a routing
    /// instruction the checkpoint supplies, not an error. So it is answered by the *predicate*
    /// (`is_native_nvfp4_weight`), which is what `linear_detect_planned` branches on, and the layer
    /// resolves to a dense `QLinear` exactly as Kitchen's deliberately-dense projections do. The
    /// read site refuses it as well, for a caller that reaches past the predicate.
    ///
    /// This is a descriptor field the `dtype == U8` predicate could not see at all: it would have
    /// packed the layer and handed it to the NVFP4 GEMM.
    #[test]
    fn a_full_precision_matrix_mult_layer_routes_dense_instead_of_failing_the_load() {
        let dev = Device::Cpu;
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("variant7_fpmm.safetensors");
        write_kitchen_nvfp4_native_file_full_precision(&path);
        let cfg = kitchen_nvfp4_config();

        let w = Weights::from_native_file_for(
            &path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )
        .expect("a full-precision-flagged NVFP4 layer must not stop the import");
        assert!(w.is_native_nvfp4(), "the file is still an NVFP4 checkpoint");
        assert!(
            !w.is_native_nvfp4_weight("transformer_blocks.0.attn.to_q.weight"),
            "a full_precision_matrix_mult layer must not be claimed for the packed route"
        );

        // The route therefore builds a DENSE projection for it.
        let plan = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(1);
        let projection = linear_detect_planned(&w, "transformer_blocks.0.attn.to_q", false, &plan)
            .expect("the flagged layer loads through the dense fallback the descriptor asks for");
        assert!(
            projection.nvfp4().is_none(),
            "a full_precision_matrix_mult layer must not become an NVFP4 projection"
        );
        // sc-12121: and the role table names the producer's declaration as the deciding fact — even
        // on a fully eligible `sm_120` device, which is the case a device-only reason would miss.
        assert_eq!(
            plan.representation(
                "transformer_blocks.0.attn.to_q",
                Nvfp4Capability {
                    nvfp4_device: true,
                    fused_quantizer: true,
                    ..w.nvfp4_capability(
                        "transformer_blocks.0.attn.to_q.weight",
                        None,
                        plan.nvfp4_context()
                    )
                }
            ),
            ExecutionRole::DenseBf16(crate::nvfp4_dit::DenseReason::FullPrecisionDeclared)
        );

        // And the RESIDENCY POLICY prices the flagged layer dense even where packed is available,
        // so the shared reader can only ever hand its dense decode to a caller (sc-21482): the
        // flag is answered at plan time, before any read site exists to get it wrong.
        let forced = Weights::from_native_file_forcing_packed_nvfp4(
            &path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )
        .expect("the flagged fixture plans");
        assert!(matches!(
            forced
                .read_planned("transformer_blocks.0.attn.to_q.weight")
                .expect("the flagged layer materializes its dense fallback"),
            LogicalTensor::Dense(_)
        ));

        // Contrast: the same fixture WITHOUT the flag is claimed for the NVFP4 route.
        let unflagged_path = path_tmp.path().join("variant7.safetensors");
        write_kitchen_nvfp4_native_file(&unflagged_path);
        let unflagged = Weights::from_native_file_for(
            &unflagged_path,
            &dev,
            DType::F32,
            DeclaredLogicalShapes::FromConfig(&cfg),
        )
        .expect("the unflagged fixture imports");
        assert!(unflagged.is_native_nvfp4_weight("transformer_blocks.0.attn.to_q.weight"));
    }

    /// **AC2, plan half (sc-12121): the residency policy and the Krea role table never disagree
    /// about a row's representation.**
    ///
    /// `CandleCodecResidency` is what the *logical plan* consults, and `DitPlan::representation` is
    /// what the provider reports. They are separate code in separate crates, so the property that
    /// matters is that they answer the same question the same way for every capability miss AC2
    /// names — padded storage, an ineligible grid, `full_precision_matrix_mult`, and a device below
    /// the `sm_120` floor — while a benign structural role on a fully eligible row reaches packed on
    /// both sides.
    #[test]
    fn the_residency_policy_and_the_role_table_agree_on_every_capability_miss() {
        use crate::nvfp4_dit::DenseReason;
        use candle_gen::gen_core::checkpoint_codec::{
            CodecResidencyPolicy, ResidencyMode, TensorCodecSpec,
        };

        let spec = |stored: [usize; 2], logical: [usize; 2], full_precision: bool| {
            TensorCodecSpec::Nvfp4 {
                block_scale: "w.weight_scale".into(),
                global_scale: "w.weight_scale_2".into(),
                input_scale: None,
                stored_shape: stored,
                logical_shape: logical,
                logical_shape_declared: true,
                full_precision_matrix_mult: full_precision,
            }
        };
        // A benign interior compute-bulk projection: nothing structural forces dense, so the row's
        // own geometry and the device are the only things left to decide it.
        let plan = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(28);
        let name = "transformer_blocks.7.attn.to_q";
        let sm120 = CandleCodecResidency {
            fp8_e4m3_native: false,
            nvfp4_native: true,
        };

        let cases: [(
            TensorCodecSpec,
            CandleCodecResidency,
            ResidencyMode,
            ExecutionRole,
        ); 5] = [
            // Eligible everywhere: both sides say packed.
            (
                spec([64, 64], [64, 64], false),
                sm120,
                ResidencyMode::Packed,
                ExecutionRole::PackedW4A4,
            ),
            // ComfyUI padded the stored grid.
            (
                spec([64, 64], [64, 60], false),
                sm120,
                ResidencyMode::Dense,
                ExecutionRole::DenseBf16(DenseReason::PaddedStorage),
            ),
            // A legitimate NVFP4 layer the FP4 GEMM's alignment cannot run (K = 48, not 32-aligned).
            (
                spec([64, 48], [64, 48], false),
                sm120,
                ResidencyMode::Dense,
                ExecutionRole::DenseBf16(DenseReason::ShapeIneligible),
            ),
            // The producer said "no quantized matmul here".
            (
                spec([64, 64], [64, 64], true),
                sm120,
                ResidencyMode::Dense,
                ExecutionRole::DenseBf16(DenseReason::FullPrecisionDeclared),
            ),
            // Below the NVFP4 floor (a CPU lane, a pre-Blackwell GPU, or a non-cuda build).
            (
                spec([64, 64], [64, 64], false),
                CandleCodecResidency::DENSE,
                ResidencyMode::Dense,
                ExecutionRole::DenseBf16(DenseReason::NoNvfp4Hardware),
            ),
        ];

        for (codec, residency, expected_mode, expected_role) in cases {
            let TensorCodecSpec::Nvfp4 {
                stored_shape,
                logical_shape,
                full_precision_matrix_mult,
                ..
            } = &codec
            else {
                unreachable!("the fixture builds NVFP4 specs only");
            };
            // The stored BYTE shape the compiler hands the policy is `[rows, cols / 2]`.
            let stored_bytes = [stored_shape[0], stored_shape[1] / 2];
            assert_eq!(
                residency.residency(&NVFP4_CODEC, &codec, &stored_bytes),
                expected_mode,
                "plan side disagreed for {codec:?}"
            );
            let cap = Nvfp4Capability {
                checkpoint_offers_nvfp4: !*full_precision_matrix_mult,
                full_precision_declared: *full_precision_matrix_mult,
                storage_unpadded: logical_shape == stored_shape,
                layout_native: candle_gen::logical_weights::nvfp4_layout_is_native(*stored_shape),
                nvfp4_device: residency.nvfp4_native,
                fused_quantizer: residency.nvfp4_native,
            };
            assert_eq!(
                plan.representation(name, cap),
                expected_role,
                "role-table side disagreed for {codec:?}"
            );
            // The two sides cannot drift: `Packed` on the plan iff packed W4A4 in the report.
            assert_eq!(
                expected_mode == ResidencyMode::Packed,
                plan.representation(name, cap).is_packed_w4a4()
            );
        }
    }

    /// **Materializing a padded NVFP4 layer without a declared logical shape is refused.** With no
    /// architecture config the plan can only carry the stored grid forward, and decoding that grid
    /// would turn ComfyUI's pad columns into weights. Planning still succeeds (a plan is also a
    /// pricing artifact); the *read* is where it stops.
    #[test]
    fn nvfp4_read_refuses_an_undeclared_logical_shape() {
        let dev = Device::Cpu;
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("kreamania_variant7.safetensors");
        write_kitchen_nvfp4_native_file(&path);

        // No config in scope: the same file, planned with nothing declared.
        let w = Weights::from_native_file(&path, &dev, DType::F32)
            .expect("an undeclared padded checkpoint still PLANS — pricing is not materialization");
        assert!(w.is_native_nvfp4(), "the descriptor still declares NVFP4");

        let plan = DitPlan::nvfp4(Nvfp4Quant::Mixed).with_num_layers(1);
        let error = match linear_detect_planned(&w, "transformer_blocks.0.attn.to_q", false, &plan)
        {
            Ok(_) => panic!("materializing an undeclared padded grid must be refused"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("block-padded") && error.contains("declares no logical shape"),
            "the refusal must be gen-core's padded-storage one: {error}"
        );
        assert!(
            error.contains("DeclaredLogicalShapes::FromConfig"),
            "the refusal must name the fix the caller controls: {error}"
        );
    }

    /// sc-14023: the non-rotated descriptor arm reconstructs exactly `codes * row_scale` through the
    /// shared native-key remap. The expected values deliberately are not rotation-invariant, so any
    /// accidental ConvRot/Hadamard leg changes the result.
    #[test]
    fn plain_int8_native_file_dequants_per_row_without_rotation() -> Result<()> {
        let dev = Device::Cpu;
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("kreamania_variant4.safetensors");
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
    fn plain_int8_native_file_accepts_scalar_scale_for_single_row() -> Result<()> {
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("kreamania_variant4.safetensors");
        write_plain_int8_native_file_with_shape(
            &path,
            r#"{"format":"int8_tensorwise","per_row":true}"#,
            vec![1, 3],
            &[1_u8, (-2_i8) as u8, 3_u8],
            vec![],
            &[0.5],
        );

        let weights = Weights::from_native_file(&path, &Device::Cpu, DType::F32)?;
        let got = weights.get("transformer_blocks.0.attn.to_q.weight")?;
        assert_eq!(got.flatten_all()?.to_vec1::<f32>()?, vec![0.5, -1.0, 1.5]);

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
        for (descriptor, expected) in cases {
            let path_tmp = tempfile::tempdir().unwrap();
            let path = path_tmp.path().join("bad.safetensors");
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
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("plain.safetensors");
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
        for scale_shape in [vec![1], vec![]] {
            let path_tmp = tempfile::tempdir().unwrap();
            let path = path_tmp.path().join("bad.safetensors");
            write_plain_int8_native_file(
                &path,
                r#"{"format":"int8_tensorwise","per_row":true}"#,
                scale_shape,
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
        let path_tmp = tempfile::tempdir().unwrap();
        let path = path_tmp.path().join("kreamania_variant5.safetensors");
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
        let dpath_tmp = tempfile::tempdir().unwrap();
        let dpath = dpath_tmp.path().join("dense.safetensors");
        write_single_file(&dpath, dmap);
        let dense = Weights::from_native_file(&dpath, &dev, DType::F32)?;
        assert!(dense.uses_native_keys());
        assert!(!dense.is_convrot(), "dense native ⇒ rotation OFF");

        // INT8-ConvRot file (bare native keys): native_keys ON, rotation ON.
        let (cmap, _ref) = convrot_int8_weight(out_dim, in_dim);
        let cpath_tmp = tempfile::tempdir().unwrap();
        let cpath = cpath_tmp.path().join("convrot.safetensors");
        write_single_file(&cpath, cmap);
        let convrot = Weights::from_convrot_file(&cpath, &dev, DType::F32)?;
        assert!(convrot.uses_native_keys());
        assert!(convrot.is_convrot(), "INT8-ConvRot ⇒ rotation ON");

        std::fs::remove_dir_all(dpath.parent().unwrap()).ok();
        std::fs::remove_dir_all(cpath.parent().unwrap()).ok();
        Ok(())
    }
}
