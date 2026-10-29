//! Where weights come from: `WeightsSource`, component ids, and bounded file-identity fingerprints.

use super::*;

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

/// Optional learned two-stage LTX latent upscaler. Providers may also resolve
/// the canonical co-located `upsampler.safetensors`, so this is not an
/// unconditional descriptor requirement.
pub const LTX_SPATIAL_UPSCALER_COMPONENT: &str = "spatial_upscaler";

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

/// The canonical object a path resolves to, plus that object's stat fingerprint.
///
/// A replacement-sensitive identity for a resolved file or directory when a full [`ArtifactSeal`]
/// is not wanted (a directory symlink, or a digest other than whole-file SHA-256). Two captures are
/// equal only when the path still resolves to the same, unmodified object.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResolvedFileIdentity {
    pub canonical_path: PathBuf,
    pub fingerprint: FileStatFingerprint,
}

impl ResolvedFileIdentity {
    pub fn capture(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let canonical_path = std::fs::canonicalize(path)?;
        let fingerprint = stat_fingerprint(&canonical_path, true)?;
        Ok(Self {
            canonical_path,
            fingerprint,
        })
    }
}

pub(super) fn stat_fingerprint(path: &Path, follow: bool) -> std::io::Result<FileStatFingerprint> {
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
pub(super) struct PathComponentFingerprint {
    pub(super) path: PathBuf,
    pub(super) is_symlink: bool,
    pub(super) symlink_target: Option<PathBuf>,
    #[cfg(unix)]
    pub(super) device: u64,
    #[cfg(unix)]
    pub(super) inode: u64,
    #[cfg(windows)]
    volume_serial_number: u64,
    #[cfg(windows)]
    file_id: [u8; 16],
    #[cfg(not(any(unix, windows)))]
    created: Option<SystemTime>,
}

pub(super) fn path_component_fingerprints(
    path: &Path,
) -> std::io::Result<Vec<PathComponentFingerprint>> {
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

// These helpers are private to the Apple guarded seal path. The native cross-platform
// algorithms above remain unchanged, including their metadata/error ordering.
#[cfg(target_vendor = "apple")]
pub(super) fn bounded_stat_fingerprint(
    path: &Path,
    follow: bool,
    next: &mut dyn FnMut(u64) -> crate::Result<()>,
) -> crate::Result<FileStatFingerprint> {
    use crate::source_inventory::bounded_fs as fs;
    let stat = fs::stat(path, follow, next)?;
    let is_symlink = !follow && fs::is_link(&stat);
    let seconds = stat.st_mtime;
    let modified = if seconds >= 0 {
        std::time::UNIX_EPOCH.checked_add(std::time::Duration::from_secs(seconds as u64))
    } else {
        std::time::UNIX_EPOCH.checked_sub(std::time::Duration::from_secs(seconds.unsigned_abs()))
    }
    .and_then(|time| time.checked_add(std::time::Duration::from_nanos(stat.st_mtime_nsec as u64)));
    Ok(FileStatFingerprint {
        size: stat.st_size as u64,
        modified,
        is_symlink,
        symlink_target: if is_symlink {
            Some(fs::read_link(path, next)?)
        } else {
            None
        },
        device: stat.st_dev as u64,
        inode: stat.st_ino,
        changed_seconds: stat.st_ctime,
        changed_nanoseconds: stat.st_ctime_nsec,
    })
}
#[cfg(target_vendor = "apple")]
pub(super) fn bounded_clone_stat(
    source: &FileStatFingerprint,
    next: &mut dyn FnMut(u64) -> crate::Result<()>,
) -> crate::Result<FileStatFingerprint> {
    use crate::source_inventory::bounded_fs as fs;
    Ok(FileStatFingerprint {
        size: source.size,
        modified: source.modified,
        is_symlink: source.is_symlink,
        symlink_target: source
            .symlink_target
            .as_deref()
            .map(|p| fs::copy_path(p, next))
            .transpose()?,
        device: source.device,
        inode: source.inode,
        changed_seconds: source.changed_seconds,
        changed_nanoseconds: source.changed_nanoseconds,
    })
}
#[cfg(target_vendor = "apple")]
pub(super) fn bounded_path_components(
    path: &Path,
    next: &mut dyn FnMut(u64) -> crate::Result<()>,
) -> crate::Result<Vec<PathComponentFingerprint>> {
    use crate::source_inventory::bounded_fs as fs;
    let count = path
        .ancestors()
        .skip(1)
        .filter(|p| !p.as_os_str().is_empty())
        .count();
    let mut result = fs::vec(count, next)?;
    for parent in path
        .ancestors()
        .skip(1)
        .filter(|p| !p.as_os_str().is_empty())
    {
        let metadata = fs::stat(parent, false, next)?;
        let is_symlink = fs::is_link(&metadata);
        result.push(PathComponentFingerprint {
            path: fs::copy_path(parent, next)?,
            is_symlink,
            symlink_target: if is_symlink {
                Some(fs::read_link(parent, next)?)
            } else {
                None
            },
            device: metadata.st_dev as u64,
            inode: metadata.st_ino,
        });
    }
    result.reverse();
    Ok(result)
}

/// ArcInner is repr(C) { strong: AtomicUsize, weak: AtomicUsize, data: T }
/// (alloc/src/sync.rs). This is its actual payload layout, NOT Layout<Arc<T>>.
#[cfg(target_vendor = "apple")]
pub(crate) fn bounded_arc_layout<T>() -> crate::Result<u64> {
    std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(std::alloc::Layout::new::<T>())
        .map(|(layout, _)| layout.pad_to_align().size() as u64)
        .map_err(|_| crate::source_inventory::bounded_fs::overflow())
}
