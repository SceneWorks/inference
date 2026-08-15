//! Shared load/exec types used by both [`Generator`](crate::generator::Generator) and
//! [`Transform`](crate::transform::Transform): where weights come from, quantization +
//! precision knobs, adapter specs, cooperative cancellation, and progress events.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use crate::memory_strategy::{
    MemoryBackend, MemoryDecodeGeometryPolicy, MemoryDecodeQualityRuntimeIdentity,
    MemoryNumericTier,
};

/// Conditional [`LoadSpec::components`] id used when [`LoadSpec::weights`] is a single-file DiT.
///
/// A provider whose ordinary form is a multi-component snapshot can keep the same provider id and
/// accept `WeightsSource::File(dit)` by taking the tokenizer/text-encoder/VAE/config companion from
/// `components[BASE_SNAPSHOT_COMPONENT] = WeightsSource::Dir(snapshot)`. The component is conditional
/// on the `File` source form, so it is intentionally not listed in a descriptor's unconditional
/// `required_components` set.
pub const BASE_SNAPSHOT_COMPONENT: &str = "base_snapshot";

/// Optional request-selected decoder weights. A consumer stages this only when it selects an
/// alternate decoder; an absent entry preserves the provider's native decoder exactly.
pub const VAE_COMPONENT: &str = "vae";

/// Optional in-place ComfyUI text-encoder file paired with a single-file DiT.
pub const COMFYUI_TEXT_ENCODER_COMPONENT: &str = "comfyui_text_encoder";

/// Optional in-place ComfyUI VAE file paired with a single-file DiT.
pub const COMFYUI_VAE_COMPONENT: &str = "comfyui_vae";

/// Candle Krea's optional community INT8-ConvRot DiT checkpoint.
///
/// This is a DiT replacement, not a text encoder. Older Krea loads overloaded
/// [`LoadSpec::text_encoder`] with this file; keeping it in the named component map preserves the
/// provider-specific optionality while leaving the typed text-encoder slot available for its declared
/// purpose.
pub const KREA_CONVROT_DIT_COMPONENT: &str = "krea_convrot_dit";

/// Where a model's weights come from — **always a local, already-provisioned path**. There is
/// deliberately **no** hub-fetch variant: inference never self-fetches weights and has no knowledge
/// of any download cache (epic 13657). A consumer resolves and stages every path — the base
/// `weights`, each typed overlay (control / ip_adapter / …), and every [`LoadSpec::components`]
/// entry — before calling `load`, and a missing component is a load-time contract error
/// ([`crate::control::require_component`]), never a mid-render fetch. (The previously-reserved
/// sc-2340 hub-fetch direction is permanently rejected.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightsSource {
    /// A directory of (possibly sharded) `.safetensors`.
    Dir(PathBuf),
    /// A single `.safetensors` file.
    File(PathBuf),
}

/// Mutation-sensitive identity for one filesystem entry or its resolved file target.
///
/// The source entry and target are intentionally separate. Hugging Face snapshots commonly expose
/// an extension-bearing symlink whose target is an extensionless blob; a streamed loader must keep
/// opening the source path (format dispatch depends on it) while detecting replacement of either the
/// link or the blob. `symlink_metadata` supplies the entry half and `metadata` the target half.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FileStatFingerprint {
    pub size: u64,
    pub modified: Option<SystemTime>,
    pub is_symlink: bool,
    pub symlink_target: Option<PathBuf>,
    #[cfg(unix)]
    pub device: u64,
    #[cfg(unix)]
    pub inode: u64,
    #[cfg(unix)]
    pub changed_seconds: i64,
    #[cfg(unix)]
    pub changed_nanoseconds: i64,
    #[cfg(windows)]
    pub volume_serial_number: u64,
    #[cfg(windows)]
    pub file_id: [u8; 16],
    #[cfg(windows)]
    pub change_time: i64,
    #[cfg(not(any(unix, windows)))]
    pub created: Option<SystemTime>,
}

#[cfg(windows)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WindowsFileStamp {
    volume_serial_number: u64,
    file_id: [u8; 16],
    change_time: i64,
}

#[cfg(windows)]
fn windows_path_stamp(path: &Path, follow: bool) -> std::io::Result<WindowsFileStamp> {
    use std::mem::{size_of, MaybeUninit};
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FileBasicInfo, FileIdInfo, GetFileInformationByHandleEx, FILE_BASIC_INFO,
        FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_ID_INFO, FILE_SHARE_DELETE,
        FILE_SHARE_READ, FILE_SHARE_WRITE,
    };

    let mut flags = FILE_FLAG_BACKUP_SEMANTICS;
    if !follow {
        flags |= FILE_FLAG_OPEN_REPARSE_POINT;
    }
    let file = std::fs::OpenOptions::new()
        .access_mode(0)
        // Prepared pins validate replacement rather than locking it out.
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .custom_flags(flags)
        .open(path)?;
    let handle = file.as_raw_handle();
    let mut basic = MaybeUninit::<FILE_BASIC_INFO>::uninit();
    // SAFETY: `file` owns a valid handle and the output buffer has the exact Win32 structure size.
    let basic_ok = unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileBasicInfo,
            basic.as_mut_ptr().cast(),
            size_of::<FILE_BASIC_INFO>() as u32,
        )
    };
    if basic_ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut id = MaybeUninit::<FILE_ID_INFO>::uninit();
    // SAFETY: same valid handle and exact output-buffer contract as above.
    let id_ok = unsafe {
        GetFileInformationByHandleEx(
            handle,
            FileIdInfo,
            id.as_mut_ptr().cast(),
            size_of::<FILE_ID_INFO>() as u32,
        )
    };
    if id_ok == 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: both successful calls initialized their complete output structures.
    let (basic, id) = unsafe { (basic.assume_init(), id.assume_init()) };
    Ok(WindowsFileStamp {
        volume_serial_number: id.VolumeSerialNumber,
        file_id: id.FileId.Identifier,
        change_time: basic.ChangeTime,
    })
}

fn stat_fingerprint(path: &Path, follow: bool) -> std::io::Result<FileStatFingerprint> {
    let metadata = if follow {
        std::fs::metadata(path)?
    } else {
        std::fs::symlink_metadata(path)?
    };
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    #[cfg(windows)]
    let windows_stamp = windows_path_stamp(path, follow)?;
    let is_symlink = !follow && metadata.file_type().is_symlink();
    Ok(FileStatFingerprint {
        size: metadata.len(),
        modified: metadata.modified().ok(),
        is_symlink,
        symlink_target: is_symlink.then(|| std::fs::read_link(path)).transpose()?,
        #[cfg(unix)]
        device: metadata.dev(),
        #[cfg(unix)]
        inode: metadata.ino(),
        #[cfg(unix)]
        changed_seconds: metadata.ctime(),
        #[cfg(unix)]
        changed_nanoseconds: metadata.ctime_nsec(),
        #[cfg(windows)]
        volume_serial_number: windows_stamp.volume_serial_number,
        #[cfg(windows)]
        file_id: windows_stamp.file_id,
        #[cfg(windows)]
        change_time: windows_stamp.change_time,
        #[cfg(not(any(unix, windows)))]
        created: metadata.created().ok(),
    })
}

/// Replacement-sensitive identity for one lexical parent component.
///
/// Unlike [`FileStatFingerprint`], this intentionally omits directory size/timestamps: ordinary
/// sibling creation must not invalidate a model token. Device + inode detect persistent directory
/// or symlink replacement on Unix, while the explicit link target detects a persistent retarget.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PathComponentFingerprint {
    path: PathBuf,
    is_symlink: bool,
    symlink_target: Option<PathBuf>,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(windows)]
    volume_serial_number: u64,
    #[cfg(windows)]
    file_id: [u8; 16],
    #[cfg(not(any(unix, windows)))]
    created: Option<SystemTime>,
}

fn path_component_fingerprints(path: &Path) -> std::io::Result<Vec<PathComponentFingerprint>> {
    let mut parents: Vec<PathBuf> = path
        .ancestors()
        .skip(1)
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .collect();
    parents.reverse();
    parents
        .into_iter()
        .map(|parent| {
            let metadata = std::fs::symlink_metadata(&parent)?;
            #[cfg(unix)]
            use std::os::unix::fs::MetadataExt;
            #[cfg(windows)]
            let windows_stamp = windows_path_stamp(&parent, false)?;
            let is_symlink = metadata.file_type().is_symlink();
            Ok(PathComponentFingerprint {
                path: parent.clone(),
                is_symlink,
                symlink_target: is_symlink
                    .then(|| std::fs::read_link(&parent))
                    .transpose()?,
                #[cfg(unix)]
                device: metadata.dev(),
                #[cfg(unix)]
                inode: metadata.ino(),
                #[cfg(windows)]
                volume_serial_number: windows_stamp.volume_serial_number,
                #[cfg(windows)]
                file_id: windows_stamp.file_id,
                #[cfg(not(any(unix, windows)))]
                created: metadata.created().ok(),
            })
        })
        .collect()
}

/// A re-openable single-file weights source pinned without canonicalizing its loader path.
///
/// The retained path is absolute but otherwise lexical: an extension-bearing snapshot symlink stays
/// extension-bearing. Every window can call [`ensure_unchanged`](Self::ensure_unchanged) before
/// re-opening it, which detects persistent or identity-changing mutation/replacement of the lexical
/// parent chain, entry (`lstat`), resolution, and resolved target. The fingerprints are also suitable
/// for cache/provenance identity without hashing a multi-gigabyte checkpoint on each request. This
/// remains a path-token model; see [`read_unchanged`](Self::read_unchanged) for its active-swap bound.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PinnedWeightsFile {
    loader_path: PathBuf,
    canonical_target_path: PathBuf,
    path_component_fingerprints: Vec<PathComponentFingerprint>,
    entry_fingerprint: FileStatFingerprint,
    target_fingerprint: FileStatFingerprint,
}

/// Read-only collection of caller-prepared File identities carried by a [`LoadSpec`].
///
/// The private `prepared` bit distinguishes ordinary compatibility mode from an intentionally
/// prepared spec even when the spec has no File sources. Callers can inspect the map for cache-key
/// construction, but can only transition or add tokens through [`LoadSpec`] methods; in particular,
/// there is no mutable-map `clear` that could silently downgrade a prepared spec to compatibility
/// mode. Besides configured [`WeightsSource::File`] slots, a prepared spec may retain exact files
/// below a configured [`WeightsSource::Dir`] (for example a transformer directory's config and
/// weights). Those directory-member tokens participate in cache identity and guard the provider
/// callback without changing the provider's directory-shaped load contract.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PreparedFilePins {
    prepared: bool,
    finalized: bool,
    pins: BTreeMap<PathBuf, PinnedWeightsFile>,
}

impl PreparedFilePins {
    /// Whether this spec has explicitly entered prepared mode, including a prepared Dir-only spec.
    pub fn is_prepared(&self) -> bool {
        self.prepared
    }

    /// Whether the caller has installed the complete token set and finalized it for consumption.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    pub fn is_empty(&self) -> bool {
        self.pins.is_empty()
    }

    pub fn len(&self) -> usize {
        self.pins.len()
    }

    pub fn get(&self, path: &Path) -> Option<&PinnedWeightsFile> {
        self.pins.get(path)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&PathBuf, &PinnedWeightsFile)> {
        self.pins.iter()
    }

    pub fn keys(&self) -> impl Iterator<Item = &PathBuf> {
        self.pins.keys()
    }
}

impl PinnedWeightsFile {
    pub fn pin(path: impl AsRef<Path>) -> crate::Result<Self> {
        let loader_path = std::path::absolute(path.as_ref())?;
        let canonical_target_path = std::fs::canonicalize(&loader_path)?;
        let path_component_fingerprints = path_component_fingerprints(&loader_path)?;
        let entry_fingerprint = stat_fingerprint(&loader_path, false)?;
        let target_fingerprint = stat_fingerprint(&loader_path, true)?;
        if target_fingerprint.is_symlink || !std::fs::metadata(&loader_path)?.is_file() {
            return Err(crate::Error::Msg(format!(
                "weights source is not a regular file: {}",
                loader_path.display()
            )));
        }
        let pinned = Self {
            loader_path,
            canonical_target_path,
            path_component_fingerprints,
            entry_fingerprint,
            target_fingerprint,
        };
        // Prove the entry, parent chain, resolution, and target still agree after the multi-stat
        // capture. This closes persistent changes during pin construction itself.
        pinned.ensure_unchanged()?;
        Ok(pinned)
    }

    pub fn loader_path(&self) -> &Path {
        &self.loader_path
    }

    /// Canonical target resolved by the same pinning operation that captured this token.
    ///
    /// A caller can root-confine this path before installing the token on a [`LoadSpec`], avoiding
    /// a second resolution that could authorize a different filesystem object.
    pub fn canonical_target_path(&self) -> &Path {
        &self.canonical_target_path
    }

    pub fn entry_fingerprint(&self) -> &FileStatFingerprint {
        &self.entry_fingerprint
    }

    pub fn target_fingerprint(&self) -> &FileStatFingerprint {
        &self.target_fingerprint
    }

    pub fn ensure_unchanged(&self) -> crate::Result<()> {
        let path_components = path_component_fingerprints(&self.loader_path)?;
        if path_components != self.path_component_fingerprints {
            return Err(crate::Error::Unsupported(format!(
                "pinned weights path component changed after load: {}",
                self.loader_path.display()
            )));
        }
        let entry = stat_fingerprint(&self.loader_path, false)?;
        if entry != self.entry_fingerprint {
            return Err(crate::Error::Unsupported(format!(
                "pinned weights entry changed after load: {}",
                self.loader_path.display()
            )));
        }
        let canonical_target_path = std::fs::canonicalize(&self.loader_path)?;
        if canonical_target_path != self.canonical_target_path {
            return Err(crate::Error::Unsupported(format!(
                "pinned weights resolution changed after load: {}",
                self.loader_path.display()
            )));
        }
        let target = stat_fingerprint(&self.loader_path, true)?;
        if target != self.target_fingerprint {
            return Err(crate::Error::Unsupported(format!(
                "pinned weights target changed after load: {}",
                self.loader_path.display()
            )));
        }
        Ok(())
    }

    /// Run one source read against the retained loader path, validating the pin both immediately
    /// before the path is opened and immediately after the read completes.
    ///
    /// The second check detects replacement that remains visible in the lexical entry, parent
    /// components, resolution, or target identity after the callback. Callers should keep this same
    /// pin for every lazy or sequential reopen. This is a path-token guard, not an opened-handle
    /// lease: an active actor that swaps in B and restores the *original* A pathname object wholly
    /// between the two checks is outside this guarantee.
    ///
    /// The post-read check runs even when `read` returns an error. If both the read and the pin fail,
    /// the mutation error wins because it is the stronger source-identity diagnosis.
    pub fn read_unchanged<T, E>(
        &self,
        read: impl FnOnce(&Path) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<crate::Error>,
    {
        self.ensure_unchanged().map_err(E::from)?;
        let result = read(&self.loader_path);
        self.ensure_unchanged().map_err(E::from)?;
        result
    }
}

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
}

impl Quant {
    /// Element bit-width of the tier. For [`Q4`](Self::Q4)/[`Q8`](Self::Q8) this is the width passed to
    /// the MLX affine quantizer. [`Nvfp4`](Self::Nvfp4) reports `4` (its E2M1 elements are 4-bit) but is
    /// **not** an MLX-quantizer tier — it carries FP8 block scales + an FP32 per-tensor scale
    /// (~4.5 *effective* bits/weight) and is served by candle-gen's NVFP4 packed path, not the MLX
    /// affine quantizer; do not route `Nvfp4` through an MLX `quantize(bits)` call on this width alone.
    pub fn bits(self) -> i32 {
        match self {
            Quant::Q4 => 4,
            Quant::Q8 => 8,
            Quant::Nvfp4 => 4,
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
/// same route. Providers continue to consume [`LoadSpec::load_shape`]; orchestration layers own and
/// propagate this authority marker across cache and cold-load boundaries.
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

/// How to load a model. `weights` is required; everything else defaults to dense bf16. The
/// device is the process-default Metal GPU — the crate runs single-device (the MLX default
/// device is not thread-safe; the worker serializes jobs per thread).
#[derive(Clone, Debug)]
pub struct LoadSpec {
    pub weights: WeightsSource,
    /// Exact caller-resolved catalog route served by a shared provider. `None` preserves the
    /// historical provider-id identity; callers that route several catalog entries through one
    /// engine (for example the five SDXL entries) set this so route-specific quality evidence can
    /// never authorize a sibling checkpoint.
    pub resolved_route: Option<String>,
    /// Sealed production-latent quality rows packaged by the caller for this exact loaded route.
    /// Empty preserves every provider's historical bounded-decode declaration. Providers accept
    /// rows only when their route/backend/numeric-tier contract validates them; the semantic records
    /// are independent from model-memory calibration evidence.
    pub decode_geometry_policies: Vec<MemoryDecodeGeometryPolicy>,
    /// True once a caller declares quality policy for this route, even if exact tier/load-shape
    /// projection or an independently audited binding leaves zero usable rows. Providers use this
    /// to distinguish fail-closed semantic scope from a genuinely legacy empty table.
    pub decode_geometry_policy_authoritative: bool,
    /// Independently resolved runtime identity for the packaged quality rows. A non-empty table
    /// without this binding is malformed and fails closed before provider construction.
    pub decode_quality_runtime_identity: Option<MemoryDecodeQualityRuntimeIdentity>,
    /// Caller-prepared identities for every single-file source carried by this spec.
    ///
    /// SceneWorks resolves and confines primary weights, controls, adapters, typed auxiliaries, and
    /// named components before it derives cache identity. Keeping those exact
    /// [`PinnedWeightsFile`] tokens on the load spec lets the cache key and provider consume one
    /// identity per lexical path instead of independently re-pinning whatever occupies that path at
    /// two different times. Keys are absolute but otherwise lexical, so an extension-bearing HF
    /// snapshot link is never canonicalized to its extensionless blob target.
    ///
    /// [`PreparedFilePins`] carries an explicit mode bit: an ordinary caller stays in compatibility
    /// mode and [`Self::file_pin_for`] pins the requested source on demand, while an intentionally
    /// prepared spec remains prepared even when it has no File sources. In prepared mode every
    /// configured File source must have a matching token; a missing or orphaned token fails closed.
    /// Prefer [`Self::prepare_file_sources`] or [`Self::with_prepared_file_pin`] to enter that mode.
    prepared_file_pins: PreparedFilePins,
    /// Additional exact files retained by a validated component receipt. These paths are not
    /// ordinary provider slots: a single-file encoder's sibling config and a complete snapshot's
    /// tokenizer are behavior-bearing inputs even though the backend only receives the encoder
    /// source. Only crate-owned validation APIs may populate this set.
    prepared_receipt_paths: BTreeSet<PathBuf>,
    /// Exact selected-encoder source shape and direct-shard inventory paired with
    /// [`prepared_receipt_paths`](Self::prepared_receipt_paths). Known file tokens reject mutation;
    /// this receipt additionally rejects directory additions, removals, renames, and type changes.
    prepared_encoder_receipt: Option<crate::encoder_contract::PreparedEncoderLoadReceipt>,
    pub quantize: Option<Quant>,
    pub precision: Precision,
    /// Auxiliary control-branch weights overlaid onto the base model at load time — a ControlNet
    /// checkpoint applied on top of `weights` (e.g. Z-Image's Fun-Controlnet-Union safetensors).
    /// `None` for the plain base model; a control-variant loader requires it. A load-time model
    /// *component* (it alters the graph), distinct from [`adapters`](Self::adapters) below, which
    /// are forward-time residual overlays on existing linears.
    pub control: Option<WeightsSource>,
    /// **Additional** ControlNet checkpoints for MultiControlNet (sc-3378) — used by providers that
    /// sum several control branches (the SDXL provider). These are loaded *after* [`control`](Self::control)
    /// and paired, in order, with the request's `Conditioning::Control` images (the diffusers
    /// `MultiControlNetModel` order semantics: branch *i* ← the *i*-th `Control`). Empty for the
    /// single-branch case (then only `control` is used); providers that do not support multi-control
    /// (Z-Image / Qwen union checkpoints) ignore this field.
    pub extra_controls: Vec<WeightsSource>,
    /// Auxiliary **IP-Adapter** weights overlaid at load time (sc-3059) — the image-prompt
    /// conditioning checkpoint (image encoder + Resampler + decoupled cross-attn K/V), e.g. an
    /// `h94/IP-Adapter`-layout snapshot dir. `None` for the plain base model. Like
    /// [`control`](Self::control), a load-time graph *component* (it adds K/V projections to the
    /// cross-attention), distinct from forward-time [`adapters`](Self::adapters).
    pub ip_adapter: Option<WeightsSource>,
    /// LoRA/LoKr adapters baked onto the model at load time. Multiples + mixed LoRA/LoKr stack by
    /// construction (see the provider `adapters` modules). Applied during `load` on the still-mutable
    /// model — the seam, since `Generator::generate`/`Transform::apply` take `&self` and the frozen
    /// fork likewise applies adapters in its initializer. Changing the adapter set means reloading.
    pub adapters: Vec<AdapterSpec>,
    /// Auxiliary **PiD** (NVIDIA Pixel-Diffusion) decoder weights overlaid at load time (epic 7840) —
    /// the optional super-resolving replacement for the model's VAE decode step. `None` for the plain
    /// VAE-decoding model; `Some` makes the PiD decoder *available*, after which the per-generation
    /// [`crate::GenerationRequest::use_pid`] flag selects it at the decode call site. Like
    /// [`control`](Self::control)/[`ip_adapter`](Self::ip_adapter) it is a load-time component (PiD's
    /// net + Gemma-2 caption encoder are heavy, so they load once and the toggle rides each request);
    /// only providers whose latent space has a PiD backbone read it (Qwen-Image / Krea today —
    /// sc-7845), and they ignore it when the request does not request PiD.
    pub pid: Option<PidWeights>,
    /// Auxiliary **identity-conditioning** sub-model weights (PuLID / InstantID family, sc-8827) — the
    /// EVA-CLIP tower, the identity encoder checkpoint, and the native face-analysis weight dir that a
    /// face-ID provider needs on top of its diffusion backbone. `None` for a plain base model. A
    /// face-ID provider that reads this slot **requires** it: the caller drives every identity path
    /// through the spec (backend-neutral — just paths), and an absent slot (or an absent sub-field) is a
    /// **load-time** error, never a fetch from an env var or a derived on-disk cache (epic 13657, sc-13664 —
    /// the PuLID-FLUX loader dropped its historical `PULID_*` env / HF-cache-derived fallbacks).
    pub identity: Option<IdentityWeights>,
    /// Auxiliary **external text-encoder** snapshot directory (sc-8827) — a separate TE snapshot a
    /// provider loads alongside its main checkpoint, e.g. LTX-2.3's Gemma-3-12B encoder (which is not
    /// bundled in the checkpoint dir). A provider that reads this slot **requires** it: the caller
    /// drives the TE location through the spec (backend-neutral — just a path), and an absent slot is a
    /// **load-time** error, never a process-global env var or a derived on-disk cache scan (epic 13657,
    /// sc-13664 — LTX-2.3 dropped its historical `$LTX_GEMMA_DIR` / HF-cache-derived fallbacks).
    pub text_encoder: Option<WeightsSource>,
    /// Component-residency strategy (epic 10765, sc-10821). [`OffloadPolicy::Resident`] (default) keeps
    /// every component resident for the whole generation; [`OffloadPolicy::Sequential`] asks a supporting
    /// provider to load→use→drop each heavy component after its phase so peak VRAM is the largest single
    /// working set, not the sum. Advisory — a provider that has not wired the residency lifecycle
    /// ignores it and stays `Resident`; [`Capabilities::supports_sequential_offload`](crate::generator::Capabilities::supports_sequential_offload)
    /// advertises which engines honor it (sc-11126). Backend-neutral.
    pub offload_policy: OffloadPolicy,
    /// Weight materialization shape, independent from [`offload_policy`](Self::offload_policy).
    /// The default preserves the historical eager/warm path. A deferred shape is meaningful only
    /// for providers with a re-openable source such as a snapshot directory.
    pub load_shape: LoadShape,
    /// Caller-owned declaration authority for [`load_shape`](Self::load_shape). This participates
    /// in request-scoped execution-policy identity whenever orchestration reconstructs a
    /// [`LoadSpec`], so an authoritative refusal cannot alias a not-yet-evaluated eager load. It is
    /// not immutable weight identity: residency and materialization policy do not change the model
    /// bytes.
    pub load_shape_declaration_result: LoadShapeDeclarationResult,
    /// **Named, caller-provisioned model components** (epic 13657) — the generic, additive home for
    /// the extra weight artifacts a model needs beyond its base `weights` and the typed overlays
    /// above, keyed by a stable component id. The complement of
    /// [`ModelDescriptor::required_components`](crate::generator::ModelDescriptor::required_components):
    /// the descriptor *advertises* which ids a model requires (weights-free, so a consumer knows what
    /// to stage), and this map *carries* the resolved local path for each. A provider reads each id at
    /// load time via [`require_component`](crate::control::require_component); a required id absent
    /// here is a **load-time** contract error, not a mid-render fetch (the whole point of the seam —
    /// it converts e.g. perth's mid-render watermark-weight fetch into a load contract error), and an
    /// unrecognized id is rejected via
    /// [`reject_unknown_components`](crate::control::reject_unknown_components). Default empty; set
    /// with [`with_component`](Self::with_component), mirroring [`with_control`](Self::with_control).
    ///
    /// This is deliberately a `BTreeMap<String, WeightsSource>` (not a typed slot per component and
    /// not a new [`WeightsSource`] hub-fetch variant — both alternatives were rejected in the sc-13591
    /// research): components are model-specific and open-ended, so a generic keyed map lets a new
    /// model declare new ids without a contract edit, while the descriptor's `required_components`
    /// keeps the set discoverable and conformance-checked.
    ///
    /// ## Provider → component-id registry (the reserved ids downstream stories consume)
    ///
    /// This map is the registry of record for component ids. The provisional set (epic 13657):
    ///
    /// | Model | Component ids |
    /// |-------|---------------|
    /// | chatterbox (TTS) | `perth`, `voice_embedding` |
    /// | MOSS tts / tts-realtime | `codec` |
    /// | SDXL | `tokenizer_clip_l`, `tokenizer_clip_bigg`, `vae_fp16_fix` |
    /// | mmaudio | `clip`, `synchformer`, `dit`, `vae`, `vocoder` |
    /// | sensenova (fast) | `distill_lora` |
    /// | LTX-2.3 | `uncensored_enhancer` |
    /// | acestep (Cover) | `sft_cover` |
    ///
    /// sc-13664 wired sensenova's `distill_lora` (the 8-step distill LoRA for `sensenova_u1_8b_fast`,
    /// with a co-located-in-snapshot fallback; **not** a universally-`required_components` id, because a
    /// pre-merged turnkey tier bakes the merge in and needs no LoRA) and LTX-2.3's optional
    /// `uncensored_enhancer` (the amoral 4-bit Gemma enhancer, read on demand when a request sets
    /// `use_uncensored_enhancer`). acestep's `sft_cover` follows the same on-demand shape: the ~7.8 GB
    /// sft Cover snapshot dir, read only for a `Cover` audio-edit request, so it is likewise **not** a
    /// `required_components` id — text2music and the region edit modes load without it. LTX-2.3's
    /// *main* Gemma text encoder rides the typed [`text_encoder`](Self::text_encoder) slot (now
    /// required), not this map. Ids are lowercase
    /// `snake_case` registry identifiers (same shape as a descriptor `id`); a model's declared
    /// `required_components` ids are validated non-empty and unique by the descriptor conformance sweep
    /// ([`model_descriptor_errors`](crate::registry::model_descriptor_errors)).
    pub components: BTreeMap<String, WeightsSource>,
}

/// Where the optional PiD decoder's weights come from (epic 7840). A PiD decoder is tied to a
/// *latent space*, not a model, so a provider in an eligible space points at the converted
/// per-latent-space checkpoint plus the shared Gemma-2-2b caption encoder. Backend-neutral (just
/// paths); the tensor load lives in `mlx-gen-pid`.
#[derive(Clone, Debug)]
pub struct PidWeights {
    /// The converted PiD student checkpoint — a single `.safetensors`
    /// ([`WeightsSource::File`]; `tools/convert_pid.py` output for this latent space).
    pub checkpoint: WeightsSource,
    /// The `gemma-2-2b-it` snapshot **directory** ([`WeightsSource::Dir`]) — the caption encoder PiD
    /// conditions on (must contain the weights + `tokenizer.json`).
    pub gemma: WeightsSource,
}

/// The identity-conditioning sub-model weights a face-ID provider (PuLID / InstantID family) needs on
/// top of its diffusion backbone (F-114). Backend-neutral paths; the tensor load lives in the provider
/// crate.
///
/// Each field is `Option` only so the struct can be built incrementally / defaulted; a provider that
/// reads a field **requires** it — the caller supplies every path through this struct, and an absent
/// field is a **load-time** error (epic 13657, sc-13664). There is no env-var or cache fallback:
/// the old "optional field ⇒ provider `PULID_*` env / HF-cache-derived default" convention was
/// deleted, so a `None` a provider needs fails fast at load rather than silently scanning the disk.
#[derive(Clone, Debug, Default)]
pub struct IdentityWeights {
    /// The identity-encoder checkpoint — a single `.safetensors` (PuLID's
    /// `pulid_flux_v0.9.1.safetensors`). Required by the PuLID-FLUX loader (`None` ⇒ load-time error).
    pub encoder: Option<WeightsSource>,
    /// The converted EVA-CLIP vision tower — a single `.safetensors`. Required by the PuLID-FLUX loader
    /// (`None` ⇒ load-time error).
    pub eva: Option<WeightsSource>,
    /// The native face-analysis weight **directory** ([`WeightsSource::Dir`]) — must contain
    /// `scrfd_10g` / `arcface_iresnet100` / `bisenet_parsing` safetensors. Required by the PuLID-FLUX
    /// loader (`None` ⇒ load-time error).
    pub face_dir: Option<WeightsSource>,
}

impl LoadSpec {
    /// Dense bf16 load from the given source.
    pub fn new(weights: WeightsSource) -> Self {
        Self {
            weights,
            resolved_route: None,
            decode_geometry_policies: Vec::new(),
            decode_geometry_policy_authoritative: false,
            decode_quality_runtime_identity: None,
            prepared_file_pins: PreparedFilePins::default(),
            prepared_receipt_paths: BTreeSet::new(),
            prepared_encoder_receipt: None,
            quantize: None,
            precision: Precision::Bf16,
            control: None,
            extra_controls: Vec::new(),
            ip_adapter: None,
            adapters: Vec::new(),
            pid: None,
            identity: None,
            text_encoder: None,
            offload_policy: OffloadPolicy::Resident,
            load_shape: LoadShape::EagerMaterialization,
            load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
            components: BTreeMap::new(),
        }
    }

    /// Bind this load to the exact caller-resolved catalog route.
    pub fn with_resolved_route(mut self, resolved_route: impl Into<String>) -> Self {
        self.resolved_route = Some(resolved_route.into());
        self
    }

    /// Attach caller-packaged, sealed decode-quality rows for this load.
    pub fn with_decode_geometry_policies(
        mut self,
        policies: Vec<MemoryDecodeGeometryPolicy>,
    ) -> Self {
        self.decode_geometry_policy_authoritative |= !policies.is_empty();
        self.decode_geometry_policies = policies;
        self
    }

    /// Preserve a caller-audited semantic scope when its rows were stripped after a typed binding
    /// refusal. Setting this false never erases authority already established by attached rows.
    pub fn with_decode_geometry_policy_authority(mut self, authoritative: bool) -> Self {
        self.decode_geometry_policy_authoritative |= authoritative;
        self
    }

    /// Bind packaged decode-quality rows to the artifact and source closure actually being loaded.
    pub fn with_decode_quality_runtime_identity(
        mut self,
        identity: MemoryDecodeQualityRuntimeIdentity,
    ) -> Self {
        self.decode_quality_runtime_identity = Some(identity);
        self
    }

    /// Return only a route-bound quality table. A shared engine must never infer the catalog route
    /// from its provider id when semantic decode evidence is present.
    pub fn route_bound_decode_geometry_policies(
        &self,
    ) -> crate::Result<Vec<MemoryDecodeGeometryPolicy>> {
        if self.decode_geometry_policies.is_empty() {
            return Ok(Vec::new());
        }
        let policies = self.route_and_family_bound_decode_geometry_policies()?;
        self.validate_decode_quality_runtime_identity(&policies)?;
        Ok(policies)
    }

    fn route_and_family_bound_decode_geometry_policies(
        &self,
    ) -> crate::Result<Vec<MemoryDecodeGeometryPolicy>> {
        self.decode_quality_runtime_identity.as_ref().ok_or_else(|| {
            crate::Error::Unsupported(
                "decode-quality policies require independently resolved artifact and implementation identity"
                    .to_owned(),
            )
        })?;
        let route = self.resolved_route.as_deref().ok_or_else(|| {
            crate::Error::Unsupported(
                "decode-quality policies require an exact caller-resolved route".to_owned(),
            )
        })?;
        if route.trim().is_empty() {
            return Err(crate::Error::Unsupported(
                "decode-quality policies require a non-empty caller-resolved route".to_owned(),
            ));
        }
        if let Some(foreign) = self
            .decode_geometry_policies
            .iter()
            .find(|policy| policy.resolved_route != route)
        {
            return Err(crate::Error::Unsupported(format!(
                "decode-quality policy route {:?} cannot authorize loaded route {:?}",
                foreign.resolved_route, route
            )));
        }
        if let Some(first) = self.decode_geometry_policies.first() {
            if let Some(foreign) = self
                .decode_geometry_policies
                .iter()
                .find(|policy| policy.family != first.family)
            {
                return Err(crate::Error::Unsupported(format!(
                    "decode-quality table for route {:?} cannot mix families {:?} and {:?}",
                    route, first.family, foreign.family
                )));
            }
        }
        Ok(self.decode_geometry_policies.clone())
    }

    fn validate_decode_quality_runtime_identity(
        &self,
        policies: &[MemoryDecodeGeometryPolicy],
    ) -> crate::Result<()> {
        let identity = self.decode_quality_runtime_identity.as_ref().ok_or_else(|| {
            crate::Error::Unsupported(
                "decode-quality policies require independently resolved artifact and implementation identity"
                    .to_owned(),
            )
        })?;
        if let Some(foreign) = policies.iter().find(|policy| {
            policy.artifact != identity.artifact
                || policy.implementation_fingerprint != identity.implementation_fingerprint
        }) {
            return Err(crate::Error::Unsupported(format!(
                "decode-quality policy artifact/source identity {:?}/{} cannot authorize loaded identity {:?}/{}",
                foreign.artifact,
                foreign.implementation_fingerprint,
                identity.artifact,
                identity.implementation_fingerprint,
            )));
        }
        Ok(())
    }

    /// Select the rows applicable to one concrete loaded provider contract. A manifest table may
    /// contain several shipped numeric tiers and load shapes for the same catalog route; only the
    /// exact loaded tier and shape enter the runtime contract. A different backend is a malformed
    /// table, not an unrelated row to ignore.
    pub fn decode_geometry_policies_for_loaded_contract(
        &self,
        backend: MemoryBackend,
        tier: MemoryNumericTier,
    ) -> crate::Result<Vec<MemoryDecodeGeometryPolicy>> {
        if self.decode_geometry_policies.is_empty() {
            return Ok(Vec::new());
        }
        let policies = self.route_and_family_bound_decode_geometry_policies()?;
        if let Some(foreign) = policies.iter().find(|policy| policy.backend != backend) {
            return Err(crate::Error::Unsupported(format!(
                "decode-quality policy backend {:?} cannot authorize loaded backend {:?}",
                foreign.backend, backend
            )));
        }
        let selected = policies
            .into_iter()
            .filter(|policy| policy.tier == tier && policy.load_shape == self.load_shape)
            .collect::<Vec<_>>();
        self.validate_decode_quality_runtime_identity(&selected)?;
        Ok(selected)
    }

    /// Read-only view of the prepared File identities carried by this spec.
    ///
    /// The whole field is private so a finalized spec cannot be silently downgraded by replacing
    /// its collection with [`PreparedFilePins::default`].
    ///
    /// ```
    /// use gen_core::PreparedFilePins;
    /// assert!(!PreparedFilePins::default().is_prepared());
    /// ```
    ///
    /// ```compile_fail
    /// use gen_core::{LoadSpec, PreparedFilePins, WeightsSource};
    /// let mut spec = LoadSpec::new(WeightsSource::Dir("snapshot".into()));
    /// spec.prepared_file_pins = PreparedFilePins::default();
    /// ```
    pub fn prepared_file_pins(&self) -> &PreparedFilePins {
        &self.prepared_file_pins
    }

    /// Atomically install an already-pinned, complete File identity set without re-pinning.
    ///
    /// The caller can pin each source first, root-confine both its lexical
    /// [`PinnedWeightsFile::loader_path`] and exact [`PinnedWeightsFile::canonical_target_path`],
    /// then pass those same tokens here. The candidate set is fully checked before this spec is
    /// mutated. Tokens must name either configured File slots or descendants of configured Dir
    /// slots. An empty iterator intentionally finalizes a Dir-only spec.
    pub fn prepare_with_file_pins(
        &mut self,
        prepared: impl IntoIterator<Item = PinnedWeightsFile>,
    ) -> crate::Result<()> {
        let mut pins = BTreeMap::new();
        for pin in prepared {
            pin.ensure_unchanged()?;
            let path = pin.loader_path().to_path_buf();
            if pins.insert(path.clone(), pin).is_some() {
                return Err(crate::Error::Unsupported(format!(
                    "duplicate prepared file token for {}",
                    path.display()
                )));
            }
        }
        let candidate = PreparedFilePins {
            prepared: true,
            finalized: true,
            pins,
        };
        if !self.prepared_receipt_paths.is_empty() || self.prepared_encoder_receipt.is_some() {
            return Err(crate::Error::Unsupported(
                "validated receipt identity must be installed through its owning contract".into(),
            ));
        }
        self.validate_prepared_file_pin_set_for(&candidate)?;
        if self.prepared_file_pins.is_finalized() && self.prepared_file_pins != candidate {
            return Err(crate::Error::Unsupported(
                "cannot replace finalized LoadSpec File identity".into(),
            ));
        }
        self.prepared_file_pins = candidate;
        Ok(())
    }

    /// Install the complete exact-file set retained by a crate-validated component receipt.
    ///
    /// Kept crate-private so an external caller cannot bless arbitrary paths as cache/load identity.
    pub(crate) fn prepare_with_validated_receipt_pins(
        &mut self,
        prepared: impl IntoIterator<Item = PinnedWeightsFile>,
        receipt_paths: impl IntoIterator<Item = PathBuf>,
        encoder_receipt: crate::encoder_contract::PreparedEncoderLoadReceipt,
    ) -> crate::Result<()> {
        let mut pins = BTreeMap::new();
        for pin in prepared {
            pin.ensure_unchanged()?;
            let path = pin.loader_path().to_path_buf();
            if let Some(existing) = pins.insert(path.clone(), pin) {
                if existing != pins[&path] {
                    return Err(crate::Error::Unsupported(format!(
                        "conflicting prepared file tokens for {}",
                        path.display()
                    )));
                }
            }
        }
        let receipt_paths = receipt_paths.into_iter().collect::<BTreeSet<_>>();
        let candidate = PreparedFilePins {
            prepared: true,
            finalized: true,
            pins,
        };
        let previous_receipt_paths =
            std::mem::replace(&mut self.prepared_receipt_paths, receipt_paths);
        let previous_encoder_receipt = self.prepared_encoder_receipt.replace(encoder_receipt);
        if let Err(error) = self.validate_prepared_file_pin_set_for(&candidate) {
            self.prepared_receipt_paths = previous_receipt_paths;
            self.prepared_encoder_receipt = previous_encoder_receipt;
            return Err(error);
        }
        if self.prepared_file_pins.is_finalized()
            && (self.prepared_file_pins != candidate
                || self.prepared_receipt_paths != previous_receipt_paths
                || self.prepared_encoder_receipt != previous_encoder_receipt)
        {
            self.prepared_receipt_paths = previous_receipt_paths;
            self.prepared_encoder_receipt = previous_encoder_receipt;
            return Err(crate::Error::Unsupported(
                "cannot replace finalized LoadSpec receipt identity".into(),
            ));
        }
        self.prepared_file_pins = candidate;
        Ok(())
    }

    /// Every File path configured anywhere on this load spec, in deterministic field order.
    ///
    /// This includes primary weights, typed overlays, adapters, PiD/identity/text-encoder files, and
    /// named components. Duplicate lexical paths occur once in [`Self::prepared_file_pins`].
    pub fn file_source_paths(&self) -> Vec<&Path> {
        fn push_source<'a>(paths: &mut Vec<&'a Path>, source: &'a WeightsSource) {
            if let WeightsSource::File(path) = source {
                paths.push(path.as_path());
            }
        }

        let mut paths = Vec::new();
        push_source(&mut paths, &self.weights);
        if let Some(source) = &self.control {
            push_source(&mut paths, source);
        }
        for source in &self.extra_controls {
            push_source(&mut paths, source);
        }
        if let Some(source) = &self.ip_adapter {
            push_source(&mut paths, source);
        }
        paths.extend(self.adapters.iter().map(|adapter| adapter.path.as_path()));
        if let Some(pid) = &self.pid {
            push_source(&mut paths, &pid.checkpoint);
            push_source(&mut paths, &pid.gemma);
        }
        if let Some(identity) = &self.identity {
            for source in [&identity.encoder, &identity.eva, &identity.face_dir]
                .into_iter()
                .flatten()
            {
                push_source(&mut paths, source);
            }
        }
        if let Some(source) = &self.text_encoder {
            push_source(&mut paths, source);
        }
        for source in self.components.values() {
            push_source(&mut paths, source);
        }
        paths
    }

    /// Every configured directory source, used to validate prepared member-file tokens.
    pub fn directory_source_paths(&self) -> Vec<&Path> {
        fn push_source<'a>(paths: &mut Vec<&'a Path>, source: &'a WeightsSource) {
            if let WeightsSource::Dir(path) = source {
                paths.push(path.as_path());
            }
        }

        let mut paths = Vec::new();
        push_source(&mut paths, &self.weights);
        if let Some(source) = &self.control {
            push_source(&mut paths, source);
        }
        for source in &self.extra_controls {
            push_source(&mut paths, source);
        }
        if let Some(source) = &self.ip_adapter {
            push_source(&mut paths, source);
        }
        if let Some(pid) = &self.pid {
            push_source(&mut paths, &pid.checkpoint);
            push_source(&mut paths, &pid.gemma);
        }
        if let Some(identity) = &self.identity {
            for source in [&identity.encoder, &identity.eva, &identity.face_dir]
                .into_iter()
                .flatten()
            {
                push_source(&mut paths, source);
            }
        }
        if let Some(source) = &self.text_encoder {
            push_source(&mut paths, source);
        }
        for source in self.components.values() {
            push_source(&mut paths, source);
        }
        paths
    }

    /// Attach one caller-prepared token to its expected configured File path.
    pub fn with_prepared_file_pin(
        mut self,
        expected_path: impl AsRef<Path>,
        prepared: PinnedWeightsFile,
    ) -> crate::Result<Self> {
        self.set_prepared_file_pin(expected_path, prepared)?;
        Ok(self)
    }

    /// Mutable counterpart to [`Self::with_prepared_file_pin`].
    pub fn set_prepared_file_pin(
        &mut self,
        expected_path: impl AsRef<Path>,
        prepared: PinnedWeightsFile,
    ) -> crate::Result<()> {
        let expected = std::path::absolute(expected_path.as_ref())?;
        let configured = self
            .file_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        if !configured.contains(&expected) {
            return Err(crate::Error::Unsupported(format!(
                "prepared file token path is not configured on this LoadSpec: {}",
                expected.display()
            )));
        }
        if prepared.loader_path() != expected {
            return Err(crate::Error::Unsupported(format!(
                "prepared file token path mismatch: LoadSpec names {}, token names {}",
                expected.display(),
                prepared.loader_path().display()
            )));
        }
        prepared.ensure_unchanged()?;
        if let Some(existing) = self.prepared_file_pins.get(&expected) {
            if existing != &prepared {
                return Err(crate::Error::Unsupported(format!(
                    "prepared file token was replaced for {}",
                    expected.display()
                )));
            }
            return Ok(());
        }
        if self.prepared_file_pins.finalized {
            return Err(crate::Error::Unsupported(format!(
                "cannot add prepared file token after LoadSpec File identity was finalized: {}",
                expected.display()
            )));
        }
        self.prepared_file_pins.prepared = true;
        self.prepared_file_pins.pins.insert(expected, prepared);
        Ok(())
    }

    /// Pin every configured File source once, retaining any already attached caller token.
    ///
    /// Consumers call this after root confinement and before cache-key construction. Repeated calls
    /// validate existing tokens rather than replacing them.
    pub fn prepare_file_sources(&mut self) -> crate::Result<()> {
        if self.prepared_file_pins.is_finalized() {
            return self.validate_prepared_file_pins();
        }
        // Transition before touching the filesystem. A mid-prepare error leaves a sticky, partial
        // prepared spec that fails closed instead of falling back to on-demand re-pinning.
        self.prepared_file_pins.prepared = true;
        let paths: Vec<PathBuf> = self
            .file_source_paths()
            .into_iter()
            .map(Path::to_path_buf)
            .collect();
        for path in paths {
            let absolute = std::path::absolute(&path)?;
            if let Some(prepared) = self.prepared_file_pins.get(&absolute) {
                if prepared.loader_path() != absolute {
                    return Err(crate::Error::Unsupported(format!(
                        "prepared file token path mismatch: LoadSpec names {}, token names {}",
                        absolute.display(),
                        prepared.loader_path().display()
                    )));
                }
                prepared.ensure_unchanged()?;
            } else {
                let prepared = PinnedWeightsFile::pin(&path)?;
                self.set_prepared_file_pin(&path, prepared)?;
            }
        }
        self.finish_file_source_preparation()
    }

    /// Finalize a complete caller-installed token set without re-pinning any source.
    ///
    /// This is the manual counterpart to [`Self::prepare_file_sources`]: a consumer can pin one
    /// source, validate [`PinnedWeightsFile::canonical_target_path`] against its allowed root, install
    /// that exact token with [`Self::set_prepared_file_pin`], repeat for every File slot, and call
    /// this method once. It is also how a caller explicitly prepares a Dir-only spec.
    pub fn finish_file_source_preparation(&mut self) -> crate::Result<()> {
        if self.prepared_file_pins.is_finalized() {
            return self.validate_prepared_file_pins();
        }
        self.prepared_file_pins.prepared = true;
        self.validate_prepared_file_pin_set_for(&self.prepared_file_pins)?;
        self.prepared_file_pins.finalized = true;
        Ok(())
    }

    /// Validate that prepared mode covers exactly the File sources currently configured by the spec.
    pub fn validate_prepared_file_pins(&self) -> crate::Result<()> {
        if !self.prepared_file_pins.is_prepared() {
            return Ok(());
        }
        if !self.prepared_file_pins.is_finalized() {
            return Err(crate::Error::Unsupported(
                "LoadSpec File identity preparation has not been finalized".into(),
            ));
        }
        self.validate_prepared_file_pin_set_for(&self.prepared_file_pins)
    }

    fn validate_prepared_file_pin_set_for(&self, prepared: &PreparedFilePins) -> crate::Result<()> {
        if let Some(receipt) = &self.prepared_encoder_receipt {
            receipt.ensure_unchanged_for(self.text_encoder.as_ref())?;
        }
        let mut expected: Vec<PathBuf> = self
            .file_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<_>>()?;
        expected.extend(self.prepared_receipt_paths.iter().cloned());
        expected.sort();
        expected.dedup();
        let configured_directories = self
            .directory_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        let actual: Vec<PathBuf> = prepared.keys().cloned().collect();
        let missing = expected
            .iter()
            .filter(|path| !actual.contains(path))
            .cloned()
            .collect::<Vec<_>>();
        let orphaned = actual
            .iter()
            .filter(|path| {
                !expected.contains(path)
                    && !configured_directories
                        .iter()
                        .any(|directory| *path != directory && path.starts_with(directory))
            })
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() || !orphaned.is_empty() {
            return Err(crate::Error::Unsupported(format!(
                "prepared file-token set does not match the LoadSpec sources: missing {missing:?}, orphaned {orphaned:?}"
            )));
        }
        for path in actual {
            let prepared = prepared.get(&path).ok_or_else(|| {
                crate::Error::Unsupported(format!(
                    "prepared file-token set is missing configured source {}",
                    path.display()
                ))
            })?;
            if prepared.loader_path() != path {
                return Err(crate::Error::Unsupported(format!(
                    "prepared file token path mismatch: LoadSpec names {}, token names {}",
                    path.display(),
                    prepared.loader_path().display()
                )));
            }
            prepared.ensure_unchanged()?;
        }
        Ok(())
    }

    /// Return the exact caller-prepared token for `path`, if this is a prepared spec.
    ///
    /// Once the spec enters prepared mode, a missing token for a configured File fails closed. The
    /// expected path and token loader path are compared as absolute-but-lexical paths on every use.
    pub fn prepared_file_pin_for(
        &self,
        path: impl AsRef<Path>,
    ) -> crate::Result<Option<&PinnedWeightsFile>> {
        let expected = std::path::absolute(path.as_ref())?;
        let configured = self
            .file_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        if !configured.contains(&expected) {
            return Err(crate::Error::Unsupported(format!(
                "requested file pin is not configured on this LoadSpec: {}",
                expected.display()
            )));
        }
        if !self.prepared_file_pins.is_prepared() {
            return Ok(None);
        }
        if !self.prepared_file_pins.is_finalized() {
            return Err(crate::Error::Unsupported(
                "LoadSpec File identity preparation has not been finalized".into(),
            ));
        }
        let prepared = self.prepared_file_pins.get(&expected).ok_or_else(|| {
            crate::Error::Unsupported(format!(
                "prepared file-token set is missing configured source {}",
                expected.display()
            ))
        })?;
        if prepared.loader_path() != expected {
            return Err(crate::Error::Unsupported(format!(
                "prepared file token path mismatch: LoadSpec names {}, token names {}",
                expected.display(),
                prepared.loader_path().display()
            )));
        }
        prepared.ensure_unchanged()?;
        Ok(Some(prepared))
    }

    /// Resolve the exact file token a provider must retain or guard while it opens `path`.
    ///
    /// Prepared mode clones the caller token. An ordinary, unprepared spec pins the current source
    /// on demand for backward compatibility.
    pub fn file_pin_for(&self, path: impl AsRef<Path>) -> crate::Result<PinnedWeightsFile> {
        match self.prepared_file_pin_for(path.as_ref())? {
            Some(prepared) => Ok(prepared.clone()),
            None => PinnedWeightsFile::pin(path),
        }
    }

    /// Read `path` through its exact prepared token when this spec has been prepared.
    ///
    /// Ordinary compatibility-mode specs invoke `read` directly. This preserves providers' legacy
    /// validation and error ordering while making every prepared caller consume the token installed
    /// by the resolver instead of re-pinning the current pathname.
    pub fn read_file_unchanged_if_prepared<T, E>(
        &self,
        path: impl AsRef<Path>,
        read: impl FnOnce(&Path) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<crate::Error>,
    {
        match self.prepared_file_pin_for(path.as_ref()).map_err(E::from)? {
            Some(prepared) => prepared.read_unchanged(read),
            None => read(path.as_ref()),
        }
    }

    /// Resolve the primary single-file token; directory-backed specs return `Ok(None)`.
    pub fn weights_file_pin(&self) -> crate::Result<Option<PinnedWeightsFile>> {
        match &self.weights {
            WeightsSource::Dir(_) => Ok(None),
            WeightsSource::File(path) => self.file_pin_for(path).map(Some),
        }
    }

    /// Execute a provider read while every listed File source is guarded by this spec's exact pins.
    ///
    /// All pins are checked before and after `read`; a post-read mutation error wins over a provider
    /// error, matching [`PinnedWeightsFile::read_unchanged`]. This lets batch adapter loaders retain
    /// their all-files ordering and zero-match semantics while consuming the same tokens as cache
    /// identity.
    pub fn read_files_unchanged<T, E, P>(
        &self,
        paths: impl IntoIterator<Item = P>,
        read: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        P: AsRef<Path>,
        E: From<crate::Error>,
    {
        let pins = paths
            .into_iter()
            .map(|path| self.file_pin_for(path).map_err(E::from))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for pin in &pins {
            pin.ensure_unchanged().map_err(E::from)?;
        }
        let result = read();
        for pin in &pins {
            pin.ensure_unchanged().map_err(E::from)?;
        }
        result
    }

    /// Execute a registry/provider callback under the complete prepared File identity set.
    ///
    /// Ordinary compatibility-mode specs invoke `read` directly, preserving historical callback
    /// behavior and error ordering without introducing eager filesystem access. Prepared specs must
    /// cover every current File slot, may additionally cover configured directory members, and every
    /// token is checked before and after the callback so persistent and identity-changing replacement
    /// during the callback fails closed.
    /// As with [`PinnedWeightsFile::read_unchanged`], a fully restored original pathname object is
    /// outside this path-token guarantee.
    pub fn read_prepared_files_unchanged<T, E>(
        &self,
        read: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<crate::Error>,
    {
        if !self.prepared_file_pins.is_prepared() {
            return read();
        }
        self.validate_prepared_file_pins().map_err(E::from)?;
        let pins = self
            .prepared_file_pins
            .iter()
            .map(|(_, pin)| pin)
            .collect::<Vec<_>>();
        for pin in &pins {
            pin.ensure_unchanged().map_err(E::from)?;
        }
        let result = read();
        for pin in &pins {
            pin.ensure_unchanged().map_err(E::from)?;
        }
        result
    }

    /// Builder-style quantization override.
    pub fn with_quant(mut self, quant: Quant) -> Self {
        self.quantize = Some(quant);
        self
    }

    /// Builder-style external text-encoder substitution.
    pub fn with_text_encoder(mut self, text_encoder: WeightsSource) -> Self {
        self.text_encoder = Some(text_encoder);
        self
    }

    /// Builder-style component-residency override (epic 10765, sc-10821). [`OffloadPolicy::Sequential`]
    /// asks a supporting provider to load→use→drop each heavy component to cap peak VRAM; the default
    /// [`OffloadPolicy::Resident`] keeps everything co-resident. Which engines honor it is advertised by
    /// [`Capabilities::supports_sequential_offload`](crate::generator::Capabilities::supports_sequential_offload).
    pub fn with_offload_policy(mut self, offload_policy: OffloadPolicy) -> Self {
        self.offload_policy = offload_policy;
        self
    }

    /// Builder-style materialization-shape override (SC-15998).
    ///
    /// This does not alter [`Self::offload_policy`]: all four combinations are representable, so a
    /// resident generator may defer transformer blocks and a staged generator may use an eager
    /// per-phase load.
    pub fn with_load_shape(mut self, load_shape: LoadShape) -> Self {
        self.load_shape = load_shape;
        self
    }

    /// Atomically record declaration eligibility while retaining eager materialization.
    ///
    /// This state is used when the provider's deferred surface additionally requires a later exact
    /// residency choice. It is not transition authority: a generic shaper must ignore it. The
    /// caller may use
    /// [`with_applied_load_shape_declaration`](Self::with_applied_load_shape_declaration) only after
    /// re-evaluating the exact declaration and provider with that requirement satisfied.
    pub fn with_eligible_load_shape_declaration(mut self) -> Self {
        self.load_shape = LoadShape::EagerMaterialization;
        self.load_shape_declaration_result = LoadShapeDeclarationResult::Eligible;
        self
    }

    /// Atomically record that an exact declaration evaluation admitted deferred materialization.
    ///
    /// [`LoadShapeDeclarationResult::Eligible`] alone never authorizes this transition; callers
    /// must re-evaluate the exact declaration/provider coordinates after any later policy choice.
    pub fn with_applied_load_shape_declaration(mut self) -> Self {
        self.load_shape = LoadShape::DeferredMaterialization;
        self.load_shape_declaration_result = LoadShapeDeclarationResult::Applied;
        self
    }

    /// Atomically record that a declaration refused deferred materialization.
    ///
    /// The eager compatibility shape is retained, but the authoritative refusal prevents a later
    /// legacy shaper from re-admitting the route.
    pub fn with_refused_load_shape_declaration(mut self) -> Self {
        self.load_shape = LoadShape::EagerMaterialization;
        self.load_shape_declaration_result = LoadShapeDeclarationResult::Refused;
        self
    }

    /// Validate the declaration authority and materialization shape as one fail-closed contract.
    ///
    /// [`LoadShapeDeclarationResult::NotEvaluated`] permits either historical shape because legacy
    /// shapers may already have run. Declaration-aware callers must use the atomic builders above
    /// and may validate again after cache or cold-load reconstruction.
    pub fn validate_load_shape_declaration(&self) -> crate::Result<()> {
        match (self.load_shape_declaration_result, self.load_shape) {
            (LoadShapeDeclarationResult::NotEvaluated, _)
            | (LoadShapeDeclarationResult::Eligible, LoadShape::EagerMaterialization)
            | (LoadShapeDeclarationResult::Applied, LoadShape::DeferredMaterialization)
            | (LoadShapeDeclarationResult::Refused, LoadShape::EagerMaterialization) => Ok(()),
            (LoadShapeDeclarationResult::Eligible, LoadShape::DeferredMaterialization) => {
                Err("eligible load-shape declaration must retain eager materialization until its exact policy requirement is satisfied".into())
            }
            (LoadShapeDeclarationResult::Applied, LoadShape::EagerMaterialization) => {
                Err("applied load-shape declaration must use deferred materialization".into())
            }
            (LoadShapeDeclarationResult::Refused, LoadShape::DeferredMaterialization) => {
                Err("refused load-shape declaration must retain eager materialization".into())
            }
        }
    }

    /// Builder-style control-branch overlay (the ControlNet checkpoint over the base `weights`).
    pub fn with_control(mut self, control: WeightsSource) -> Self {
        self.control = Some(control);
        self
    }

    /// Builder-style named component (epic 13657) — stage the caller-provisioned local path for the
    /// component `id` into [`components`](Self::components). Mirrors [`with_control`](Self::with_control);
    /// the id is the stable key a provider reads at load via
    /// [`require_component`](crate::control::require_component). Re-inserting the same id replaces the
    /// prior path (last write wins). See [`components`](Self::components) for the id registry.
    pub fn with_component(mut self, id: impl Into<String>, src: WeightsSource) -> Self {
        self.components.insert(id.into(), src);
        self
    }

    /// Builder-style **additional** ControlNet checkpoint for MultiControlNet (sc-3378) — appends to
    /// [`extra_controls`](Self::extra_controls). Call after [`with_control`](Self::with_control); each
    /// extra branch pairs, in order, with the request's `Conditioning::Control` images. Supported by
    /// the SDXL provider.
    pub fn with_extra_control(mut self, control: WeightsSource) -> Self {
        self.extra_controls.push(control);
        self
    }

    /// Builder-style IP-Adapter overlay (the image-prompt checkpoint dir over the base `weights`).
    pub fn with_ip_adapter(mut self, ip_adapter: WeightsSource) -> Self {
        self.ip_adapter = Some(ip_adapter);
        self
    }

    /// Builder-style LoRA/LoKr adapters to bake on at load time (replaces any already set).
    pub fn with_adapters(mut self, adapters: Vec<AdapterSpec>) -> Self {
        self.adapters = adapters;
        self
    }

    /// Builder-style optional PiD decoder overlay (epic 7840) — the converted per-latent-space PiD
    /// checkpoint + the Gemma-2 caption-encoder snapshot dir. Makes PiD *available*; the per-request
    /// [`crate::GenerationRequest::use_pid`] flag then selects it at decode.
    pub fn with_pid(mut self, checkpoint: WeightsSource, gemma: WeightsSource) -> Self {
        self.pid = Some(PidWeights { checkpoint, gemma });
        self
    }
}

/// A single adapter to stack at load time. Multiples + mixed LoRA/LoKr are supported by
/// construction — see the provider `adapters` modules. Carried by [`LoadSpec::adapters`].
#[derive(Clone, Debug)]
pub struct AdapterSpec {
    pub path: PathBuf,
    pub scale: f32,
    pub kind: AdapterKind,
    /// Per-denoise-pass strength override (LTX-2.3 only). When `Some`, the slice gives this
    /// adapter's strength for each distilled stage (LTX runs a 2-stage denoise, so a length-2
    /// `[stage1, stage2]`); when `None`, [`scale`](Self::scale) is applied uniformly to every pass.
    /// This is the LTX "per-pass strength" feature (sc-2687) — the reference has no per-stage
    /// schedule, so it is net-new. Like [`LoadSpec::control`], it is a model-specific knob on the
    /// shared spec: **only LTX reads it**; every other model ignores it (its denoise is single-pass).
    pub pass_scales: Option<Vec<f32>>,
    /// Which expert of a dual-expert MoE model (Wan2.2 A14B) this adapter targets (sc-2683).
    /// `None` = shared: merged onto **both** the high- and low-noise experts (the reference
    /// `--lora` file → `(loras)+(loras_high/low)`); `Some(High)`/`Some(Low)` = one expert only
    /// (`--lora-high` / `--lora-low`). Like [`pass_scales`](Self::pass_scales), this is a
    /// model-specific knob on the shared spec: **only the Wan MoE models read it**; every
    /// single-stream model ignores it (a `Some(_)` there is surfaced, not silently honored).
    pub moe_expert: Option<MoeExpert>,
}

/// One adapter file's actual provider-side install outcome.
///
/// Providers that can partially accept an adapter publish these reports through
/// [`Generator::adapter_apply_reports`](crate::Generator::adapter_apply_reports) after generation.
/// The report is deliberately tensor-free and additive to the existing generator contract: providers
/// that do not opt in return an empty list, preserving their current behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterApplyReport {
    /// The adapter file the report describes. This is the resolved local path from
    /// [`AdapterSpec::path`], allowing a consumer to correlate reports with load-order specs without
    /// guessing from file headers.
    pub adapter_path: PathBuf,
    /// Target deltas/factors the provider accepted for this adapter.
    pub applied: usize,
    /// Target stems the provider did not apply. Empty means the provider accepted the whole adapter
    /// surface it recognized.
    pub skipped: Vec<String>,
}

impl AdapterSpec {
    /// A uniform-strength adapter (the common case): [`scale`](Self::scale) on every denoise pass,
    /// no per-pass override, shared across both MoE experts. Equivalent to a literal with
    /// `pass_scales: None, moe_expert: None`.
    pub fn new(path: PathBuf, scale: f32, kind: AdapterKind) -> Self {
        Self {
            path,
            scale,
            kind,
            pass_scales: None,
            moe_expert: None,
        }
    }

    /// Builder-style per-pass strength override (LTX only — see [`pass_scales`](Self::pass_scales)).
    pub fn with_pass_scales(mut self, pass_scales: Vec<f32>) -> Self {
        self.pass_scales = Some(pass_scales);
        self
    }

    /// Builder-style MoE expert target (Wan2.2 A14B only — see [`moe_expert`](Self::moe_expert)).
    pub fn with_moe_expert(mut self, expert: MoeExpert) -> Self {
        self.moe_expert = Some(expert);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterKind {
    Lora,
    Lokr,
}

/// One expert of a dual-expert MoE denoiser (Wan2.2 A14B), naming which checkpoint an adapter
/// merges onto. The A14B splits denoising at a noise `boundary` between a **high**-noise expert
/// (early, noisy steps) and a **low**-noise expert (late steps); a trained Wan MoE LoRA ships as a
/// high/low pair (e.g. `*_wan22_high` + `*_wan22_low`). See [`AdapterSpec::moe_expert`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeExpert {
    High,
    Low,
}

/// Cooperative cancellation handle threaded into a request; a model checks it between steps
/// and bails early. Cloneable — the caller keeps a handle to cancel an in-flight job.
#[derive(Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation of the in-flight generation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Adopt a caller-owned cancellation token, so a consumer's existing
    /// `Arc<AtomicBool>` **is** the flag the engine polls rather than something a bridge has to
    /// mirror into.
    ///
    /// Without this, a consumer that already has a cancellation token (a job queue, a request
    /// scope) can only forward cancellation from inside the progress callback, which means a
    /// cancel is not observed until the next progress event. Sharing the atomic removes that
    /// hop. Cancellation stays cooperative and unchanged: whoever sets the flag, the engine
    /// still only observes it where it checks.
    ///
    /// **Where the engine checks is the real bound, and it is coarser than this handle.** Polling
    /// happens at denoise step boundaries; component loading does not poll at all, so a cancel
    /// raised during a multi-second weight load is not observed until the load finishes. Sharing
    /// the token does not change that, and a consumer that needs to abandon a load must do it
    /// above this seam.
    pub fn from_arc(flag: Arc<AtomicBool>) -> Self {
        Self(flag)
    }

    /// The underlying token, so a consumer can observe (or set) the same atomic the engine polls.
    /// The inverse of [`from_arc`](Self::from_arc); see its note on where the engine checks.
    pub fn as_arc(&self) -> &Arc<AtomicBool> {
        &self.0
    }
}

impl std::fmt::Debug for CancelFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CancelFlag")
            .field(&self.is_cancelled())
            .finish()
    }
}

/// One frame of the developing image, handed to a [`PreviewSink`] during denoise.
///
/// The image is a **linear latent→RGB approximation at latent resolution** — for a VAE with an 8×
/// spatial scale that is `width/8 × height/8` (128×128 for a 1024² render), RGB8, three bytes per
/// pixel. It is explicitly **not** a VAE decode: producing it costs one small matmul instead of a
/// full decoder forward, which is what makes a per-step preview affordable. Consumers should
/// upscale it for display and must not treat it as the render's output.
///
/// `current` / `total` are 1-based and mirror [`Progress::Step`], so a consumer can drive one
/// progress indicator from either. `current` advances monotonically and never exceeds `total`; a
/// solver that evaluates the model more than once per step (Heun-family) emits at most one frame
/// per schedule position rather than overrunning the count.
#[derive(Clone, Debug)]
pub struct PreviewFrame {
    /// 1-based schedule position this frame was projected from.
    pub current: u32,
    /// Total denoise steps in this trajectory.
    pub total: u32,
    /// Latent-resolution RGB8 approximation of the developing image.
    pub image: crate::media::Image,
}

/// Per-step preview sink threaded onto a request ([`GenerationRequest::preview`]).
///
/// The [`CancelFlag`] pattern: a cheap cloneable handle carried as a request **field**, not a
/// [`Progress`] variant — `Progress` stays `Copy` and no exhaustive match downstream changes. The
/// inert [`default`](Default::default) is free: a supporting engine gates the projection on
/// [`is_active`](Self::is_active) and does no work at all when nobody is listening.
///
/// **Emission is synchronous and on the denoise thread**, so the closure must return promptly —
/// forward the frame to a channel rather than encoding or rendering inside it. Support is
/// per-engine and opt-in; an engine that never emits is indistinguishable from an inert sink.
///
/// [`GenerationRequest::preview`]: crate::GenerationRequest::preview
#[derive(Clone, Default)]
pub struct PreviewSink(Option<Arc<dyn Fn(PreviewFrame) + Send + Sync>>);

impl PreviewSink {
    /// Build an active sink from a callback. The callback runs on the denoise thread, once per
    /// emitted frame.
    pub fn new(sink: impl Fn(PreviewFrame) + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(sink)))
    }

    /// Whether anyone is listening. Engines gate the projection work on this so an inert sink
    /// costs one branch per denoise evaluation.
    pub fn is_active(&self) -> bool {
        self.0.is_some()
    }

    /// Deliver one frame. A no-op on an inert sink.
    pub fn emit(&self, frame: PreviewFrame) {
        if let Some(sink) = &self.0 {
            sink(frame);
        }
    }
}

impl std::fmt::Debug for PreviewSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PreviewSink")
            .field(&self.is_active())
            .finish()
    }
}

/// The request-local outcome of optional prompt enhancement.
///
/// This is deliberately a small, tensor-free fact rather than a log convention. Consumers use it
/// to distinguish a prompt that was actually rewritten from a safe fallback and from the identity
/// path where enhancement was not requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptEnhancementOutcome {
    /// The provider produced and consumed a non-empty rewritten prompt.
    Enhanced,
    /// Enhancement was requested, but the provider safely consumed the original prompt instead.
    Fallback,
    /// Enhancement was not requested; the effective prompt is the original prompt byte-for-byte.
    Absent,
}

/// Engine-owned prompt provenance emitted for one generation request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptEnhancementReport {
    pub outcome: PromptEnhancementOutcome,
    /// The user-authored prompt passed into the request. Consumers retain this for replay/editing.
    pub original_prompt: String,
    /// The exact prompt encoded by the diffusion/video model.
    pub effective_prompt: String,
    /// Stable, non-sensitive reason code for a fallback. `None` for enhanced/absent outcomes.
    pub fallback_reason: Option<String>,
}

impl PromptEnhancementReport {
    pub fn absent(prompt: impl Into<String>) -> Self {
        let prompt = prompt.into();
        Self {
            outcome: PromptEnhancementOutcome::Absent,
            original_prompt: prompt.clone(),
            effective_prompt: prompt,
            fallback_reason: None,
        }
    }

    pub fn enhanced(
        original_prompt: impl Into<String>,
        effective_prompt: impl Into<String>,
    ) -> Self {
        Self {
            outcome: PromptEnhancementOutcome::Enhanced,
            original_prompt: original_prompt.into(),
            effective_prompt: effective_prompt.into(),
            fallback_reason: None,
        }
    }

    pub fn fallback(prompt: impl Into<String>, reason: impl Into<String>) -> Self {
        let prompt = prompt.into();
        Self {
            outcome: PromptEnhancementOutcome::Fallback,
            original_prompt: prompt.clone(),
            effective_prompt: prompt,
            fallback_reason: Some(reason.into()),
        }
    }
}

/// Synchronous request-local sink for [`PromptEnhancementReport`].
///
/// Like [`PreviewSink`], the inert default is free and an active consumer should forward the fact
/// promptly. Supporting providers emit exactly one report before encoding the effective prompt.
#[derive(Clone, Default)]
pub struct PromptEnhancementSink(Option<Arc<dyn Fn(PromptEnhancementReport) + Send + Sync>>);

impl PromptEnhancementSink {
    pub fn new(sink: impl Fn(PromptEnhancementReport) + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(sink)))
    }

    pub fn is_active(&self) -> bool {
        self.0.is_some()
    }

    pub fn emit(&self, report: PromptEnhancementReport) {
        if let Some(sink) = &self.0 {
            sink(report);
        }
    }
}

impl std::fmt::Debug for PromptEnhancementSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PromptEnhancementSink")
            .field(&self.is_active())
            .finish()
    }
}

/// A progress event streamed to the caller during a long `generate` / `apply`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// Denoising step `current` of `total` (1-based).
    Step { current: u32, total: u32 },
    /// VAE decode underway (post-denoise).
    Decoding,
    /// A heavy model component is (re)loading (epic 10765, sc-11126). Emitted only under
    /// [`OffloadPolicy::Sequential`], where the residency seam load→use→drops each component *inside*
    /// `generate` — a multi-second, multi-GB step during which no `Step`/`Decoding` event fires, so
    /// without this the UI would freeze silently while a component streams from disk (F-179). The
    /// [`Resident`](OffloadPolicy::Resident) path loads everything before `generate` and never emits it.
    Loading(LoadPhase),
}

/// Which component the residency seam is loading when it emits [`Progress::Loading`] (sc-11126). The
/// `Sequential` lifecycle has two in-`generate` load phases: the phase-A text/vision encoder, then the
/// heavy render bundle (transformer/U-Net + VAE + any control/PiD overlay).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadPhase {
    /// The phase-A prompt encoder (text or vision-language), loaded first and dropped before the
    /// render bundle materializes.
    TextEncoder,
    /// The heavy render bundle — the transformer/U-Net, the VAE, and any control/PiD overlay.
    Renderer,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::io::Read;
    use std::sync::{Barrier, Mutex};

    #[test]
    fn load_shape_declaration_result_defaults_clones_hashes_and_validates() {
        let spec = LoadSpec::new(WeightsSource::Dir("snapshot".into()));
        assert_eq!(
            spec.load_shape_declaration_result,
            LoadShapeDeclarationResult::NotEvaluated
        );
        assert_eq!(spec.load_shape, LoadShape::EagerMaterialization);

        spec.validate_load_shape_declaration()
            .expect("default declaration state is valid");

        let eligible = spec.clone().with_eligible_load_shape_declaration();
        assert_eq!(
            eligible.load_shape_declaration_result,
            LoadShapeDeclarationResult::Eligible
        );
        assert_eq!(eligible.load_shape, LoadShape::EagerMaterialization);
        eligible
            .validate_load_shape_declaration()
            .expect("eligible declaration is valid while eager");

        let applied = spec.clone().with_applied_load_shape_declaration();
        assert_eq!(
            applied.load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied
        );
        assert_eq!(applied.load_shape, LoadShape::DeferredMaterialization);
        applied
            .validate_load_shape_declaration()
            .expect("applied declaration is valid");
        assert_eq!(
            applied.clone().load_shape_declaration_result,
            LoadShapeDeclarationResult::Applied,
            "clone must carry declaration authority across a cold-load boundary"
        );

        let refused = spec.with_refused_load_shape_declaration();
        assert_eq!(
            refused.load_shape_declaration_result,
            LoadShapeDeclarationResult::Refused
        );
        assert_eq!(
            refused.load_shape,
            LoadShape::EagerMaterialization,
            "an authoritative refusal keeps the compatibility shape"
        );
        refused
            .validate_load_shape_declaration()
            .expect("refused declaration is valid");

        let cache_axes: HashSet<_> = [
            LoadShapeDeclarationResult::NotEvaluated,
            LoadShapeDeclarationResult::Eligible,
            LoadShapeDeclarationResult::Applied,
            LoadShapeDeclarationResult::Refused,
        ]
        .into_iter()
        .collect();
        assert_eq!(cache_axes.len(), 4);

        let mut contradictory_eligible = eligible;
        contradictory_eligible.load_shape = LoadShape::DeferredMaterialization;
        assert!(contradictory_eligible
            .validate_load_shape_declaration()
            .is_err());

        let mut contradictory_applied = applied;
        contradictory_applied.load_shape = LoadShape::EagerMaterialization;
        assert!(contradictory_applied
            .validate_load_shape_declaration()
            .is_err());

        let mut contradictory_refused = refused;
        contradictory_refused.load_shape = LoadShape::DeferredMaterialization;
        assert!(contradictory_refused
            .validate_load_shape_declaration()
            .is_err());
    }

    #[test]
    fn pinned_weights_file_detects_target_mutation() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("model.safetensors");
        std::fs::write(&file, b"original").expect("write fixture");
        let pinned = PinnedWeightsFile::pin(&file).expect("pin regular file");

        assert_eq!(pinned.loader_path(), file.as_path());
        pinned.ensure_unchanged().expect("unchanged file");

        std::fs::write(&file, b"replacement bytes").expect("mutate fixture");
        let error = pinned
            .ensure_unchanged()
            .expect_err("mutation must invalidate the pin")
            .to_string();
        assert!(error.contains("changed after load"), "got: {error}");
    }

    #[test]
    fn pinned_weights_file_rejects_a_barrier_controlled_mid_read_replacement() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("model.safetensors");
        let original = vec![0x5a; 128 * 1024];
        std::fs::write(&file, &original).expect("write fixture");
        let pinned = PinnedWeightsFile::pin(&file).expect("pin regular file");
        let consumed_first_chunk = Arc::new(Barrier::new(2));
        let replacement_done = Arc::new(Barrier::new(2));

        let writer_file = file.clone();
        let writer_entered = Arc::clone(&consumed_first_chunk);
        let writer_done = Arc::clone(&replacement_done);
        let writer = std::thread::spawn(move || {
            writer_entered.wait();
            let replacement = writer_file.with_extension("replacement");
            std::fs::write(&replacement, vec![0xa5; 128 * 1024])
                .expect("write replacement beside source");
            #[cfg(unix)]
            std::fs::rename(replacement, writer_file).expect("atomically replace during read");
            #[cfg(not(unix))]
            {
                let bytes = std::fs::read(replacement).expect("read replacement fixture");
                std::fs::write(writer_file, bytes).expect("overwrite source during read");
            }
            writer_done.wait();
        });

        let error = pinned
            .read_unchanged::<_, crate::Error>(|path| {
                // Open and consume part of the original payload before allowing replacement. The
                // remainder is then read from the already-open original inode, proving the post-read
                // check—not a pre-open race—is what rejects the mixed-provenance operation.
                let mut source = std::fs::File::open(path)?;
                let mut bytes = vec![0; 4096];
                source.read_exact(&mut bytes)?;
                assert!(bytes.iter().all(|byte| *byte == 0x5a));
                consumed_first_chunk.wait();
                replacement_done.wait();
                source.read_to_end(&mut bytes)?;
                assert_eq!(bytes.len(), original.len());
                #[cfg(unix)]
                assert!(bytes.iter().all(|byte| *byte == 0x5a));
                Ok(bytes)
            })
            .expect_err("a replacement between the two checks must fail")
            .to_string();
        writer.join().expect("writer thread");
        assert!(error.contains("changed after load"), "got: {error}");
    }

    #[cfg(unix)]
    #[test]
    fn pinned_weights_file_preserves_and_pins_the_symlink_entry() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().expect("temp dir");
        let first = dir.path().join("content-addressed-blob-a");
        let second = dir.path().join("content-addressed-blob-b");
        let link = dir.path().join("selected-model.safetensors");
        std::fs::write(&first, b"same-size-a").expect("write first target");
        std::fs::write(&second, b"same-size-b").expect("write second target");
        symlink(&first, &link).expect("create extension-bearing symlink");

        let pinned = PinnedWeightsFile::pin(&link).expect("pin symlinked file");
        assert_eq!(
            pinned.loader_path(),
            link.as_path(),
            "the loader path must stay lexical rather than canonicalizing to the blob"
        );
        pinned.ensure_unchanged().expect("unchanged symlink");

        std::fs::remove_file(&link).expect("remove old link");
        symlink(&second, &link).expect("retarget symlink");
        let error = pinned
            .ensure_unchanged()
            .expect_err("retargeting the entry must invalidate the pin")
            .to_string();
        assert!(error.contains("entry changed"), "got: {error}");
    }

    #[cfg(unix)]
    #[test]
    fn pinned_weights_file_detects_a_recreated_intermediate_symlink() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().expect("temp dir");
        let first = dir.path().join("snapshot-a");
        let second = dir.path().join("snapshot-b");
        std::fs::create_dir(&first).expect("create A snapshot");
        std::fs::create_dir(&second).expect("create B snapshot");
        std::fs::write(first.join("model.safetensors"), b"same-size-a").expect("write A");
        std::fs::write(second.join("model.safetensors"), b"same-size-b").expect("write B");
        let selected = dir.path().join("selected");
        let staged_b = dir.path().join("staged-b");
        let recreated_a = dir.path().join("recreated-a");
        symlink(&first, &selected).expect("select A directory");
        symlink(&second, &staged_b).expect("stage B directory link");
        symlink(&first, &recreated_a).expect("stage recreated A directory link");
        let lexical_file = selected.join("model.safetensors");
        let pinned = PinnedWeightsFile::pin(&lexical_file).expect("pin through intermediate link");
        assert_eq!(
            pinned.canonical_target_path(),
            std::fs::canonicalize(first.join("model.safetensors")).unwrap()
        );

        std::fs::rename(staged_b, &selected).expect("select B directory");
        std::fs::rename(recreated_a, &selected).expect("select recreated A directory link");
        assert_eq!(std::fs::read(&lexical_file).unwrap(), b"same-size-a");
        let error = pinned
            .ensure_unchanged()
            .expect_err("recreating an intermediate symlink must change component identity")
            .to_string();
        assert!(error.contains("path component changed"), "got: {error}");
    }

    #[cfg(windows)]
    #[test]
    fn pinned_weights_file_detects_a_same_size_windows_file_replacement() {
        use std::os::windows::fs::FileTimesExt;

        let dir = tempfile::tempdir().expect("temp dir");
        let selected = dir.path().join("model.safetensors");
        let replacement = dir.path().join("replacement.safetensors");
        std::fs::write(&selected, b"same-size-a").expect("write A");
        std::fs::write(&replacement, b"same-size-b").expect("write B");
        let timestamp = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        let times = std::fs::FileTimes::new()
            .set_modified(timestamp)
            .set_created(timestamp);
        std::fs::OpenOptions::new()
            .write(true)
            .open(&selected)
            .unwrap()
            .set_times(times)
            .expect("set A times");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_times(times)
            .expect("set B times");
        let selected_metadata = std::fs::metadata(&selected).unwrap();
        let replacement_metadata = std::fs::metadata(&replacement).unwrap();
        assert_eq!(
            (
                selected_metadata.len(),
                selected_metadata.modified().ok(),
                selected_metadata.created().ok(),
            ),
            (
                replacement_metadata.len(),
                replacement_metadata.modified().ok(),
                replacement_metadata.created().ok(),
            ),
            "the legacy size/mtime/created tuple must collide"
        );
        let pinned = PinnedWeightsFile::pin(&selected).expect("pin A");

        std::fs::remove_file(&selected).expect("remove A");
        std::fs::rename(&replacement, &selected).expect("install same-size B");
        let error = pinned
            .ensure_unchanged()
            .expect_err("Windows file ID/change time must reject same-size replacement")
            .to_string();
        assert!(
            error.contains("entry changed") || error.contains("target changed"),
            "got: {error}"
        );
    }

    #[cfg(windows)]
    #[test]
    fn pinned_weights_file_ignores_unrelated_windows_sibling_changes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let selected = dir.path().join("model.safetensors");
        let sibling = dir.path().join("unrelated.tmp");
        std::fs::write(&selected, b"weights").expect("write model");
        let pinned = PinnedWeightsFile::pin(&selected).expect("pin model");

        std::fs::write(&sibling, b"unrelated").expect("create sibling");
        std::fs::remove_file(&sibling).expect("remove sibling");
        pinned
            .ensure_unchanged()
            .expect("parent identity must ignore ordinary child-entry timestamp changes");
    }

    #[cfg(windows)]
    #[test]
    fn pinned_weights_file_distinguishes_windows_symlink_entry_and_target() {
        use std::os::windows::fs::symlink_file;

        let dir = tempfile::tempdir().expect("temp dir");
        let first = dir.path().join("blob-a");
        let second = dir.path().join("blob-b");
        let selected = dir.path().join("selected.safetensors");
        std::fs::write(&first, b"same-size-a").expect("write A");
        std::fs::write(&second, b"same-size-b").expect("write B");
        if let Err(error) = symlink_file(&first, &selected) {
            if error.kind() == std::io::ErrorKind::PermissionDenied {
                eprintln!(
                    "skipping Windows symlink identity fixture because this host lacks symlink privilege: {error}"
                );
                return;
            }
            panic!("create Windows file symlink: {error}");
        }

        let pinned = PinnedWeightsFile::pin(&selected).expect("pin selected A link");
        assert_eq!(
            pinned.loader_path(),
            std::path::absolute(&selected).unwrap()
        );
        assert_eq!(
            pinned.canonical_target_path(),
            std::fs::canonicalize(&first).unwrap()
        );
        assert_ne!(
            pinned.entry_fingerprint().file_id,
            pinned.target_fingerprint().file_id,
            "OPEN_REPARSE_POINT must fingerprint the link entry, not its target"
        );

        std::fs::remove_file(&selected).expect("remove A link");
        symlink_file(&second, &selected).expect("retarget selected link to B");
        let error = pinned
            .ensure_unchanged()
            .expect_err("retargeting the Windows link must invalidate its entry token")
            .to_string();
        assert!(
            error.contains("entry changed") || error.contains("resolution changed"),
            "got: {error}"
        );
    }

    #[test]
    fn load_spec_rejects_a_prepared_token_for_another_lexical_path() {
        let dir = tempfile::tempdir().expect("temp dir");
        let expected = dir.path().join("expected.safetensors");
        let other = dir.path().join("other.safetensors");
        std::fs::write(&expected, b"expected").expect("write expected file");
        std::fs::write(&other, b"other").expect("write other file");

        let token = PinnedWeightsFile::pin(&other).expect("pin other file");
        let error = LoadSpec::new(WeightsSource::File(expected.clone()))
            .with_prepared_file_pin(&expected, token)
            .expect_err("a token for another file must fail closed")
            .to_string();

        assert!(error.contains("path mismatch"), "got: {error}");
    }

    #[test]
    fn load_spec_reuses_the_exact_prepared_primary_file_token() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("model.safetensors");
        std::fs::write(&file, b"weights").expect("write weights");
        let prepared = PinnedWeightsFile::pin(&file).expect("prepare file");
        let mut spec = LoadSpec::new(WeightsSource::File(file.clone()))
            .with_prepared_file_pin(&file, prepared.clone())
            .expect("attach matching token");
        spec.finish_file_source_preparation()
            .expect("finalize matching token set");

        let provider_pin = spec
            .weights_file_pin()
            .expect("resolve provider pin")
            .expect("file pin");
        assert_eq!(provider_pin, prepared);
        assert_eq!(provider_pin.loader_path(), file.as_path());
        assert_eq!(
            spec.prepared_file_pin_for(&file)
                .expect("validate prepared token")
                .expect("prepared token"),
            &prepared
        );
    }

    #[test]
    fn unprepared_file_callers_still_receive_a_primary_file_pin() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("model.safetensors");
        std::fs::write(&file, b"weights").expect("write weights");
        let spec = LoadSpec::new(WeightsSource::File(file.clone()));

        assert!(spec.prepared_file_pins.is_empty());
        let provider_pin = spec
            .weights_file_pin()
            .expect("pin ordinary file caller")
            .expect("file pin");
        assert_eq!(provider_pin.loader_path(), file.as_path());
    }

    #[test]
    fn unprepared_callback_guard_preserves_raw_callback_behavior() {
        let spec = LoadSpec::new(WeightsSource::File(
            "/a/compatibility/caller/need/not/open/this.safetensors".into(),
        ));

        let value = spec
            .read_prepared_files_unchanged(|| Ok::<_, crate::Error>(17))
            .expect("an unprepared callback must run without eager filesystem access");
        assert_eq!(value, 17);
        assert!(!spec.prepared_file_pins.is_prepared());
    }

    #[test]
    fn prepared_only_file_guard_preserves_compatibility_and_rejects_a_stale_token() {
        let missing = PathBuf::from("/a/compatibility/caller/need/not/open/this-file.safetensors");
        let ordinary = LoadSpec::new(WeightsSource::File(missing.clone()));
        let value = ordinary
            .read_file_unchanged_if_prepared(&missing, |path| {
                assert_eq!(path, missing);
                Ok::<_, crate::Error>(23)
            })
            .expect("an unprepared file read must not eagerly touch the filesystem");
        assert_eq!(value, 23);

        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("prepared.safetensors");
        std::fs::write(&file, b"original").expect("write original file");
        let mut prepared = LoadSpec::new(WeightsSource::File(file.clone()));
        prepared.prepare_file_sources().expect("prepare file token");
        std::fs::write(&file, b"replacement-with-another-size").expect("replace prepared file");

        let error = prepared
            .read_file_unchanged_if_prepared(&file, |_| Ok::<_, crate::Error>(()))
            .expect_err("a prepared read must validate the caller-installed token")
            .to_string();
        assert!(error.contains("changed after load"), "got: {error}");
    }

    #[test]
    fn prepared_dir_only_mode_stays_sticky_when_a_file_slot_is_added() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("late-control.safetensors");
        std::fs::write(&file, b"control").expect("write control");
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().join("snapshot")));

        spec.prepare_file_sources().expect("prepare Dir-only spec");
        assert!(spec.prepared_file_pins.is_prepared());
        assert!(spec.prepared_file_pins.is_empty());

        spec.control = Some(WeightsSource::File(file.clone()));
        let error = spec
            .file_pin_for(&file)
            .expect_err("a later File slot must not downgrade to compatibility repinning")
            .to_string();
        assert!(error.contains("missing configured source"), "got: {error}");
        assert!(
            spec.prepare_file_sources().is_err(),
            "re-preparing a sticky spec must validate rather than fill a newly added slot"
        );
        assert!(spec.prepared_file_pins.is_prepared());
        assert!(spec.prepared_file_pins.is_empty());
    }

    #[test]
    fn atomic_prepared_token_install_uses_exact_tokens_and_supports_zero_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        let primary = dir.path().join("primary.safetensors");
        let control = dir.path().join("control.safetensors");
        std::fs::write(&primary, b"primary").expect("write primary");
        std::fs::write(&control, b"control").expect("write control");
        let primary_pin = PinnedWeightsFile::pin(&primary).expect("pin primary");
        let control_pin = PinnedWeightsFile::pin(&control).expect("pin control");
        let mut spec = LoadSpec::new(WeightsSource::File(primary.clone()))
            .with_control(WeightsSource::File(control.clone()));

        spec.prepare_with_file_pins([primary_pin.clone(), control_pin.clone()])
            .expect("atomically install exact set");
        assert!(spec.prepared_file_pins().is_prepared());
        assert!(spec.prepared_file_pins().is_finalized());
        assert_eq!(spec.file_pin_for(&primary).unwrap(), primary_pin);
        assert_eq!(spec.file_pin_for(&control).unwrap(), control_pin);

        let mut dir_only = LoadSpec::new(WeightsSource::Dir(dir.path().join("snapshot")));
        dir_only
            .prepare_with_file_pins(std::iter::empty())
            .expect("zero-token set explicitly prepares a Dir-only spec");
        assert!(dir_only.prepared_file_pins().is_prepared());
        assert!(dir_only.prepared_file_pins().is_finalized());
        assert!(dir_only.prepared_file_pins().is_empty());
    }

    #[test]
    fn prepared_directory_members_are_guarded_and_orphans_fail_closed() {
        let dir = tempfile::tempdir().expect("temp dir");
        let transformer = dir.path().join("transformer");
        std::fs::create_dir(&transformer).expect("create transformer dir");
        let config = transformer.join("config.json");
        let weights = transformer.join("diffusion_pytorch_model.safetensors");
        std::fs::write(&config, b"{\"kind\":\"mage\"}").expect("write config");
        std::fs::write(&weights, b"weights-v1").expect("write weights");

        let mut spec = LoadSpec::new(WeightsSource::Dir(transformer));
        spec.prepare_with_file_pins([
            PinnedWeightsFile::pin(&config).expect("pin config"),
            PinnedWeightsFile::pin(&weights).expect("pin weights"),
        ])
        .expect("directory members are valid prepared identities");
        assert_eq!(spec.prepared_file_pins().len(), 2);
        spec.read_prepared_files_unchanged(|| Ok::<_, crate::Error>(()))
            .expect("unchanged directory members pass the callback guard");

        std::fs::write(&weights, b"weights-v2").expect("replace weights in place");
        spec.read_prepared_files_unchanged(|| Ok::<_, crate::Error>(()))
            .expect_err("a changed directory member must fail before provider reuse");

        let outside = dir.path().join("outside.safetensors");
        std::fs::write(&outside, b"outside").expect("write outside file");
        let mut orphaned = LoadSpec::new(WeightsSource::Dir(dir.path().join("other")));
        orphaned
            .prepare_with_file_pins([PinnedWeightsFile::pin(outside).expect("pin outside")])
            .expect_err("a token outside every configured directory must fail closed");
    }

    #[test]
    fn prepared_mode_rejects_an_orphan_after_a_file_slot_is_removed() {
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("primary.safetensors");
        std::fs::write(&file, b"primary").expect("write primary");
        let mut spec = LoadSpec::new(WeightsSource::File(file));
        spec.prepare_file_sources().expect("prepare File spec");

        spec.weights = WeightsSource::Dir(dir.path().join("snapshot"));
        let error = spec
            .validate_prepared_file_pins()
            .expect_err("removing a prepared File slot must leave an invalid orphan")
            .to_string();
        assert!(error.contains("does not match"), "got: {error}");
        assert!(spec.prepared_file_pins.is_prepared());
        assert_eq!(spec.prepared_file_pins.len(), 1);
    }

    #[test]
    fn file_source_paths_covers_the_complete_load_spec_slot_matrix() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = |name: &str| dir.path().join(format!("{name}.safetensors"));
        let primary = path("primary");
        let control = path("control");
        let extra_control_0 = path("extra-control-0");
        let extra_control_1 = path("extra-control-1");
        let ip_adapter = path("ip-adapter");
        let adapter = path("adapter");
        let pid_checkpoint = path("pid-checkpoint");
        let pid_gemma = path("pid-gemma");
        let identity_encoder = path("identity-encoder");
        let identity_eva = path("identity-eva");
        let identity_face = path("identity-face");
        let text_encoder = path("text-encoder");
        let component_alpha = path("component-alpha");
        let component_zeta = path("component-zeta");
        let expected = vec![
            primary.clone(),
            control.clone(),
            extra_control_0.clone(),
            extra_control_1.clone(),
            ip_adapter.clone(),
            adapter.clone(),
            pid_checkpoint.clone(),
            pid_gemma.clone(),
            identity_encoder.clone(),
            identity_eva.clone(),
            identity_face.clone(),
            text_encoder.clone(),
            component_alpha.clone(),
            component_zeta.clone(),
        ];
        for file in &expected {
            std::fs::write(file, b"fixture").expect("write slot fixture");
        }

        let mut spec = LoadSpec::new(WeightsSource::File(primary));
        spec.control = Some(WeightsSource::File(control));
        spec.extra_controls = vec![
            WeightsSource::File(extra_control_0),
            WeightsSource::File(extra_control_1),
        ];
        spec.ip_adapter = Some(WeightsSource::File(ip_adapter));
        spec.adapters = vec![AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)];
        spec.pid = Some(PidWeights {
            checkpoint: WeightsSource::File(pid_checkpoint),
            gemma: WeightsSource::File(pid_gemma),
        });
        spec.identity = Some(IdentityWeights {
            encoder: Some(WeightsSource::File(identity_encoder)),
            eva: Some(WeightsSource::File(identity_eva)),
            face_dir: Some(WeightsSource::File(identity_face)),
        });
        spec.text_encoder = Some(WeightsSource::File(text_encoder));
        // BTreeMap value iteration is key-sorted, independently of insertion order.
        spec.components
            .insert("zeta".into(), WeightsSource::File(component_zeta));
        spec.components
            .insert("alpha".into(), WeightsSource::File(component_alpha));

        let actual: Vec<PathBuf> = spec
            .file_source_paths()
            .into_iter()
            .map(Path::to_path_buf)
            .collect();
        assert_eq!(actual, expected);

        spec.prepare_file_sources()
            .expect("prepare every File slot");
        spec.validate_prepared_file_pins()
            .expect("prepared map exactly covers the slot matrix");
        assert_eq!(spec.prepared_file_pins.len(), expected.len());
        for file in expected {
            let token = spec
                .file_pin_for(&file)
                .expect("resolve prepared slot token");
            assert_eq!(token.loader_path(), file.as_path());
        }
    }

    #[cfg(unix)]
    #[test]
    fn prepared_load_spec_rejects_a_to_b_to_recreated_a_rebinding() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().expect("temp dir");
        let first = dir.path().join("blob-a");
        let second = dir.path().join("blob-b");
        let selected = dir.path().join("selected.safetensors");
        let staged_b = dir.path().join("staged-b.safetensors");
        let staged_a = dir.path().join("staged-a.safetensors");
        std::fs::write(&first, b"same-size-a").expect("write A");
        std::fs::write(&second, b"same-size-b").expect("write B");
        symlink(&first, &selected).expect("select A");
        // Create both replacement entries before the race so their inodes are provably distinct
        // from the original selected entry, even after the final path once again resolves to A.
        symlink(&second, &staged_b).expect("stage B link");
        symlink(&first, &staged_a).expect("stage replacement A link");

        let mut spec = LoadSpec::new(WeightsSource::File(selected.clone()));
        spec.prepare_file_sources().expect("prepare A identity");
        let key_pin = spec
            .prepared_file_pin_for(&selected)
            .expect("validate cache-key token")
            .expect("prepared token")
            .clone();
        let start_mutation = Arc::new(Barrier::new(2));
        let mutation_done = Arc::new(Barrier::new(2));
        let writer_start = Arc::clone(&start_mutation);
        let writer_done = Arc::clone(&mutation_done);
        let writer_selected = selected.clone();
        let writer = std::thread::spawn(move || {
            writer_start.wait();
            std::fs::rename(staged_b, &writer_selected).expect("rebind selected path to B");
            std::fs::rename(staged_a, &writer_selected)
                .expect("replace selected path with a recreated A link");
            writer_done.wait();
        });

        start_mutation.wait();
        mutation_done.wait();
        writer.join().expect("writer thread");
        assert_eq!(
            std::fs::read(&selected).expect("read final selected target"),
            b"same-size-a",
            "the lexical path must resolve to A again before the provider consumes the token"
        );
        let error = spec
            .weights_file_pin()
            .expect_err("the provider must reject the stale cache-key token")
            .to_string();
        assert!(error.contains("entry changed"), "got: {error}");
        assert_eq!(
            key_pin.loader_path(),
            selected.as_path(),
            "cache identity must retain the extension-bearing lexical path"
        );
    }

    #[test]
    fn prepared_mode_rejects_a_missing_token_instead_of_repinning() {
        let dir = tempfile::tempdir().expect("temp dir");
        let primary = dir.path().join("primary.safetensors");
        let adapter = dir.path().join("adapter.safetensors");
        std::fs::write(&primary, b"primary").expect("write primary");
        std::fs::write(&adapter, b"adapter").expect("write adapter");
        let mut spec = LoadSpec::new(WeightsSource::File(primary.clone())).with_adapters(vec![
            AdapterSpec::new(adapter.clone(), 1.0, AdapterKind::Lora),
        ]);
        spec.set_prepared_file_pin(
            &primary,
            PinnedWeightsFile::pin(&primary).expect("pin primary"),
        )
        .expect("attach primary token");

        let error = spec
            .finish_file_source_preparation()
            .expect_err("a partial caller-installed set must not finalize")
            .to_string();
        assert!(error.contains("does not match"), "got: {error}");
        let use_error = spec
            .file_pin_for(&adapter)
            .expect_err("an unfinished prepared set must not silently re-pin")
            .to_string();
        assert!(use_error.contains("not been finalized"), "got: {use_error}");
        assert!(spec.validate_prepared_file_pins().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn prepared_adapter_component_and_control_reject_a_to_b_to_recreated_a_rebinding() {
        use std::os::unix::fs::symlink;

        for role in ["adapter", "component", "control"] {
            let dir = tempfile::tempdir().expect("temp dir");
            let first = dir.path().join("blob-a");
            let second = dir.path().join("blob-b");
            let selected = dir.path().join(format!("{role}.safetensors"));
            let staged_b = dir.path().join(format!("{role}-staged-b.safetensors"));
            let staged_a = dir.path().join(format!("{role}-staged-a.safetensors"));
            std::fs::write(&first, b"same-size-a").expect("write A");
            std::fs::write(&second, b"same-size-b").expect("write B");
            symlink(&first, &selected).expect("select A");
            symlink(&second, &staged_b).expect("stage B link");
            symlink(&first, &staged_a).expect("stage replacement A link");

            let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().join("base")));
            match role {
                "adapter" => {
                    spec.adapters
                        .push(AdapterSpec::new(selected.clone(), 1.0, AdapterKind::Lora));
                }
                "component" => {
                    spec.components
                        .insert("vae".into(), WeightsSource::File(selected.clone()));
                }
                "control" => spec.control = Some(WeightsSource::File(selected.clone())),
                _ => unreachable!(),
            }
            spec.prepare_file_sources()
                .expect("prepare composite load identity");

            let start_mutation = Arc::new(Barrier::new(2));
            let mutation_done = Arc::new(Barrier::new(2));
            let writer_start = Arc::clone(&start_mutation);
            let writer_done = Arc::clone(&mutation_done);
            let writer_selected = selected.clone();
            let writer = std::thread::spawn(move || {
                writer_start.wait();
                std::fs::rename(staged_b, &writer_selected).expect("rebind selected path to B");
                std::fs::rename(staged_a, &writer_selected)
                    .expect("replace selected path with a recreated A link");
                writer_done.wait();
            });

            start_mutation.wait();
            mutation_done.wait();
            writer.join().expect("writer thread");
            assert_eq!(
                std::fs::read(&selected).expect("read final selected target"),
                b"same-size-a"
            );
            let error = spec
                .file_pin_for(&selected)
                .expect_err("provider must consume the stale prepared token")
                .to_string();
            assert!(
                error.contains("entry changed"),
                "{role} should fail on the prepared A token, got: {error}"
            );
        }
    }

    fn frame(current: u32, total: u32) -> PreviewFrame {
        PreviewFrame {
            current,
            total,
            image: crate::media::Image {
                width: 2,
                height: 1,
                pixels: vec![0, 0, 0, 255, 255, 255],
            },
        }
    }

    /// A caller-owned token IS the flag the engine polls: cancelling through the consumer's own
    /// `Arc` is observed by the engine's handle, with no bridge and no progress-callback hop.
    #[test]
    fn from_arc_shares_the_callers_token() {
        let token = Arc::new(AtomicBool::new(false));
        let flag = CancelFlag::from_arc(Arc::clone(&token));

        assert!(!flag.is_cancelled());
        token.store(true, Ordering::Relaxed);
        assert!(
            flag.is_cancelled(),
            "the engine handle must see the caller's cancel"
        );
    }

    /// And the reverse direction: `cancel()` on the engine handle is visible on the caller's token,
    /// so a consumer can share one token across several in-flight requests.
    #[test]
    fn cancel_is_visible_on_the_shared_token() {
        let token = Arc::new(AtomicBool::new(false));
        let flag = CancelFlag::from_arc(Arc::clone(&token));

        flag.cancel();
        assert!(token.load(Ordering::Relaxed));
        assert!(
            Arc::ptr_eq(flag.as_arc(), &token),
            "as_arc must expose the same allocation"
        );
    }

    /// The inert default is the zero-cost path: engines gate their projection on `is_active`, and
    /// `emit` on an inert sink must not panic (a provider may emit unconditionally).
    #[test]
    fn default_preview_sink_is_inert() {
        let sink = PreviewSink::default();
        assert!(!sink.is_active());
        sink.emit(frame(1, 8)); // must be a no-op, not a panic
    }

    /// An active sink receives every emitted frame, in order, through a clone of the handle — the
    /// `CancelFlag` pattern: cloning the request must not detach the sink.
    #[test]
    fn active_preview_sink_receives_frames_through_a_clone() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);
        let sink = PreviewSink::new(move |f: PreviewFrame| {
            recorder
                .lock()
                .unwrap()
                .push((f.current, f.total, f.image.pixels.len()));
        });
        assert!(sink.is_active());

        let cloned = sink.clone();
        sink.emit(frame(1, 8));
        cloned.emit(frame(2, 8));

        assert_eq!(*seen.lock().unwrap(), vec![(1, 8, 6), (2, 8, 6)]);
    }

    /// `Debug` reports liveness, never the closure — matching `CancelFlag`'s shape so a request
    /// stays printable.
    #[test]
    fn preview_sink_debug_reports_liveness() {
        assert_eq!(
            format!("{:?}", PreviewSink::default()),
            "PreviewSink(false)"
        );
        assert_eq!(
            format!("{:?}", PreviewSink::new(|_| {})),
            "PreviewSink(true)"
        );
    }

    #[test]
    fn prompt_enhancement_sink_reports_typed_outcomes_through_a_clone() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);
        let sink = PromptEnhancementSink::new(move |report| {
            recorder.lock().unwrap().push(report);
        });
        sink.clone()
            .emit(PromptEnhancementReport::enhanced("a cat", "a detailed cat"));
        sink.emit(PromptEnhancementReport::fallback("a dog", "empty_output"));
        sink.emit(PromptEnhancementReport::absent("unchanged bytes\n"));

        assert_eq!(
            *seen.lock().unwrap(),
            vec![
                PromptEnhancementReport::enhanced("a cat", "a detailed cat"),
                PromptEnhancementReport::fallback("a dog", "empty_output"),
                PromptEnhancementReport::absent("unchanged bytes\n"),
            ]
        );
        assert_eq!(
            format!("{:?}", PromptEnhancementSink::default()),
            "PromptEnhancementSink(false)"
        );
    }
}
