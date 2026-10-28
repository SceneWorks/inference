//! Caller-owned filesystem metadata on Darwin (macOS and iOS).
//!
//! No `ReadDir`, `DirEntry`, `opendir`, `getcwd`, null-output `realpath`, or std path
//! encoding wrapper is used. All heap owners are Vecs whose *requested* Layout is
//! checked before reserve_exact; conversion Vec -> OsString -> PathBuf transfers
//! ownership. Kernel memory and allocator overhead are not requested heap layouts.
//!
//! Authority: Apple SDK sys/attr.h + sys/unistd.h and getattrlistbulk(2) (public
//! macOS 10.10 / iOS 8.0). ATTR_BULK_REQUIRED returns length, attribute_set_t,
//! attrreference_t, then the referenced NUL-terminated name. The buffer belongs to
//! the caller, not libc; ERANGE grows that buffer, never a guessed entry/name cap.
//! XNU bsd/kern/syscalls.master + libsyscall/xcodescripts/create-syscalls.pl generate
//! the direct getattrlistbulk syscall stub. Apple Libc stdlib/FreeBSD/realpath.c:
//! absolute input with nonnull PATH_MAX output uses stack arrays and syscalls,
//! and its only malloc/free branch is conditional on null output. Keeping that
//! implementation (including case normalization and mountpoints) matters on Apple.
//! Relative lexical roots use public F_GETPATH instead of getcwd's allocating
//! fdopendir fallback. These laws are deliberately Apple-qualified; ordinary
//! SourceInventory behavior on every other target is unchanged.
//!
//! Pinned source authority: apple-oss-distributions/Libc at
//! 71bbe350ab79eef58113991d817ccc6165061a64; xnu at
//! f6217f891ac0bb64f3d375211650a4c1ff8ca1ea. XNU vfs_attrlist.c falls back
//! to kernel readdirattr when a filesystem lacks VNOP_GETATTRLISTBULK, so
//! this is not an APFS-only discovery path. libsyscall's open-base.c,
//! system-version-compat.c and cancelable/fcntl-base.c also show that the
//! optional macOS SystemVersion compatibility shim uses stack buffers/syscalls,
//! not a hidden heap owner. No assumption that the shim is disabled is needed.
//! These are published implementation/SDK laws, not a binary attestation of
//! libSystem or a claim about native allocator/process physical footprint.
use crate::{Error, Result};
use std::ffi::{OsStr, OsString};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

pub(crate) type Check<'a> = &'a mut dyn FnMut(u64) -> Result<()>;

pub(crate) fn overflow() -> Error {
    std::io::Error::from_raw_os_error(libc::EOVERFLOW).into()
}
fn os_error() -> Error {
    std::io::Error::last_os_error().into()
}
pub(crate) fn vec<T>(len: usize, next: Check<'_>) -> Result<Vec<T>> {
    let layout = std::alloc::Layout::array::<T>(len).map_err(|_| overflow())?;
    next(layout.size() as u64)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(len)
        .map_err(|_| Error::Io(std::io::Error::from_raw_os_error(libc::ENOMEM)))?;
    Ok(result)
}
pub(crate) fn push<T>(values: &mut Vec<T>, value: T, next: Check<'_>) -> Result<()> {
    if values.len() == values.capacity() {
        let capacity = values
            .capacity()
            .checked_mul(2)
            .ok_or_else(overflow)?
            .max(4);
        let layout = std::alloc::Layout::array::<T>(capacity).map_err(|_| overflow())?;
        next(layout.size() as u64)?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|_| Error::Io(std::io::Error::from_raw_os_error(libc::ENOMEM)))?;
    }
    values.push(value);
    Ok(())
}
pub(crate) fn path(bytes: &[u8], next: Check<'_>) -> Result<PathBuf> {
    let mut result = vec(bytes.len(), next)?;
    result.extend_from_slice(bytes);
    Ok(PathBuf::from(OsString::from_vec(result)))
}
pub(crate) fn copy_path(value: &Path, next: Check<'_>) -> Result<PathBuf> {
    path(value.as_os_str().as_bytes(), next)
}
fn c_path(path: &Path, next: Check<'_>) -> Result<Vec<u8>> {
    let bytes = path.as_os_str().as_bytes();
    if bytes.contains(&0) {
        return Err(std::io::Error::from_raw_os_error(libc::EINVAL).into());
    }
    let mut result = vec(bytes.len().checked_add(1).ok_or_else(overflow)?, next)?;
    result.extend_from_slice(bytes);
    result.push(0);
    Ok(result)
}
fn open(path: &Path, flags: libc::c_int, next: Check<'_>) -> Result<OwnedFd> {
    let name = c_path(path, next)?;
    next(0)?; // open's public syscall wrapper has no userspace heap owner.
              // SAFETY: c_path owns a terminated input with no interior NUL; a mode
              // argument is supplied even though all current callers only open for reads.
    let fd = unsafe {
        libc::open(
            name.as_ptr().cast(),
            flags | libc::O_CLOEXEC,
            0 as libc::c_int,
        )
    };
    if fd < 0 {
        return Err(os_error());
    }
    // SAFETY: successful open returned a new descriptor, transferred exactly
    // once to OwnedFd; its error/early-return cleanup only closes that descriptor.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

pub(crate) fn absolute(input: &Path, next: Check<'_>) -> Result<PathBuf> {
    if input.as_os_str().is_empty() {
        return Err(std::io::Error::from_raw_os_error(libc::EINVAL).into());
    }
    let bytes = input.as_os_str().as_bytes();
    let mut components = input.strip_prefix(".").unwrap_or(input).components();
    let cwd;
    let prefix = if input.is_absolute() {
        if bytes.starts_with(b"//") && !bytes.starts_with(b"///") {
            components.next();
            b"//".as_slice()
        } else {
            b"".as_slice()
        }
    } else {
        let directory = open(Path::new("."), libc::O_EVTONLY, next)?;
        let mut buffer = vec::<u8>(libc::PATH_MAX as usize, next)?;
        buffer.resize(libc::PATH_MAX as usize, 0);
        next(0)?;
        if unsafe { libc::fcntl(directory.as_raw_fd(), libc::F_GETPATH, buffer.as_mut_ptr()) } < 0 {
            return Err(os_error());
        }
        let len = buffer.iter().position(|b| *b == 0).ok_or_else(overflow)?;
        buffer.truncate(len);
        cwd = buffer;
        &cwd
    };
    let capacity = prefix
        .len()
        .checked_add(bytes.len())
        .and_then(|n| n.checked_add(2))
        .ok_or_else(overflow)?;
    let mut buffer = vec(capacity, next)?;
    buffer.extend_from_slice(prefix);
    let mut result = PathBuf::from(OsString::from_vec(buffer));
    // Same lexical components / double-leading-slash / trailing-slash convention
    // as std::sys::path::unix::absolute; enough capacity for all original bytes.
    for component in components {
        result.push(component.as_os_str());
    }
    if bytes.ends_with(b"/") {
        result.push("");
    }
    Ok(result)
}

pub(crate) fn join(parent: &Path, name: &OsStr, next: Check<'_>) -> Result<PathBuf> {
    let parent = parent.as_os_str().as_bytes();
    let name = name.as_bytes();
    let capacity = parent
        .len()
        .checked_add(name.len())
        .and_then(|n| n.checked_add(1))
        .ok_or_else(overflow)?;
    let mut bytes = vec(capacity, next)?;
    bytes.extend_from_slice(parent);
    if !bytes.ends_with(b"/") {
        bytes.push(b'/');
    }
    bytes.extend_from_slice(name);
    Ok(PathBuf::from(OsString::from_vec(bytes)))
}

pub(crate) fn try_stat(
    path: &Path,
    follow: bool,
    next: Check<'_>,
) -> Result<std::io::Result<libc::stat>> {
    let path = c_path(path, next)?;
    let mut stat = std::mem::MaybeUninit::<libc::stat>::zeroed();
    next(0)?;
    let rc = unsafe {
        if follow {
            libc::stat(path.as_ptr().cast(), stat.as_mut_ptr())
        } else {
            libc::lstat(path.as_ptr().cast(), stat.as_mut_ptr())
        }
    };
    Ok(if rc < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(unsafe { stat.assume_init() })
    })
}
pub(crate) fn stat(path: &Path, follow: bool, next: Check<'_>) -> Result<libc::stat> {
    Ok(try_stat(path, follow, next)??)
}
/// Match Path::is_file's OS-error-to-false semantics WITHOUT swallowing a
/// capacity callback error (even when that callback itself returns Error::Io).
pub(crate) fn is_file_at(path: &Path, next: Check<'_>) -> Result<bool> {
    Ok(try_stat(path, true, next)?.is_ok_and(|stat| is_file(&stat)))
}
pub(crate) fn is_file(stat: &libc::stat) -> bool {
    stat.st_mode & libc::S_IFMT == libc::S_IFREG
}
pub(crate) fn is_dir(stat: &libc::stat) -> bool {
    stat.st_mode & libc::S_IFMT == libc::S_IFDIR
}
pub(crate) fn is_link(stat: &libc::stat) -> bool {
    stat.st_mode & libc::S_IFMT == libc::S_IFLNK
}

pub(crate) fn canonical(path: &Path, next: Check<'_>) -> Result<PathBuf> {
    debug_assert!(path.is_absolute());
    let name = c_path(path, next)?;
    let mut result = vec::<u8>(libc::PATH_MAX as usize, next)?;
    result.resize(libc::PATH_MAX as usize, 0);
    next(0)?;
    if unsafe { libc::realpath(name.as_ptr().cast(), result.as_mut_ptr().cast()) }.is_null() {
        return Err(os_error());
    }
    let len = result
        .iter()
        .position(|byte| *byte == 0)
        .ok_or_else(overflow)?;
    result.truncate(len);
    Ok(PathBuf::from(OsString::from_vec(result)))
}
pub(crate) fn read_link(path: &Path, next: Check<'_>) -> Result<PathBuf> {
    let name = c_path(path, next)?;
    let mut capacity = libc::PATH_MAX as usize;
    loop {
        let mut result = vec::<u8>(capacity, next)?;
        result.resize(capacity, 0);
        next(0)?;
        let len =
            unsafe { libc::readlink(name.as_ptr().cast(), result.as_mut_ptr().cast(), capacity) };
        if len < 0 {
            return Err(os_error());
        }
        if (len as usize) < capacity {
            result.truncate(len as usize);
            return Ok(PathBuf::from(OsString::from_vec(result)));
        }
        capacity = capacity.checked_mul(2).ok_or_else(overflow)?;
    }
}

/// Every name is borrowed from our buffer until the next syscall. No private
/// libc directory owner, per-entry allocation, cardinality premise or name cap.
pub(crate) struct Directory {
    fd: OwnedFd,
    buffer: Vec<u64>, // documented 8-byte alignment for attribute record groups
    offset: usize,
    remaining: usize,
}
impl Directory {
    pub(crate) fn open(directory: &Path, next: Check<'_>) -> Result<Self> {
        let fd = open(directory, libc::O_RDONLY | libc::O_DIRECTORY, next)?;
        let mut buffer = vec::<u64>(2048, next)?;
        buffer.resize(2048, 0);
        Ok(Self {
            fd,
            buffer,
            offset: 0,
            remaining: 0,
        })
    }
    pub(crate) fn next<'a>(&'a mut self, next: Check<'_>) -> Result<Option<&'a OsStr>> {
        if self.remaining == 0 {
            loop {
                let mut attrs = libc::attrlist {
                    bitmapcount: libc::ATTR_BIT_MAP_COUNT,
                    reserved: 0,
                    commonattr: libc::ATTR_CMN_NAME | libc::ATTR_CMN_RETURNED_ATTRS,
                    volattr: 0,
                    dirattr: 0,
                    fileattr: 0,
                    forkattr: 0,
                };
                next(0)?;
                let count = unsafe {
                    libc::getattrlistbulk(
                        self.fd.as_raw_fd(),
                        (&mut attrs as *mut libc::attrlist).cast(),
                        self.buffer.as_mut_ptr().cast(),
                        self.buffer.len() * 8,
                        0,
                    )
                };
                if count >= 0 {
                    self.remaining = count as usize;
                    self.offset = 0;
                    break;
                }
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ERANGE) {
                    return Err(error.into());
                }
                let len = self.buffer.len().checked_mul(2).ok_or_else(overflow)?;
                let mut buffer = vec(len, next)?;
                buffer.resize(len, 0);
                self.buffer = buffer;
            }
        }
        if self.remaining == 0 {
            return Ok(None);
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(self.buffer.as_ptr().cast::<u8>(), self.buffer.len() * 8)
        };
        let record = bytes.get(self.offset..).ok_or_else(overflow)?;
        fn word(bytes: &[u8], offset: usize) -> Result<u32> {
            Ok(u32::from_ne_bytes(
                bytes
                    .get(offset..offset + 4)
                    .ok_or_else(overflow)?
                    .try_into()
                    .map_err(|_| overflow())?,
            ))
        }
        let len = word(record, 0)? as usize;
        let record = record.get(..len).ok_or_else(overflow)?;
        // uint32 length, attribute_set_t (5 u32), then attrreference_t.
        if word(record, 4)? & (libc::ATTR_CMN_NAME | libc::ATTR_CMN_RETURNED_ATTRS)
            != libc::ATTR_CMN_NAME | libc::ATTR_CMN_RETURNED_ATTRS
        {
            return Err(std::io::Error::from_raw_os_error(libc::EIO).into());
        }
        let displacement = word(record, 24)? as i32;
        let start = 24usize
            .checked_add_signed(displacement as isize)
            .ok_or_else(overflow)?;
        let end = start
            .checked_add(word(record, 28)? as usize)
            .ok_or_else(overflow)?;
        let name = record.get(start..end).ok_or_else(overflow)?;
        if len < 32 || name.last() != Some(&0) || name[..name.len() - 1].contains(&0) {
            return Err(std::io::Error::from_raw_os_error(libc::EIO).into());
        }
        self.offset = self.offset.checked_add(len).ok_or_else(overflow)?;
        self.remaining -= 1;
        // This operation borrows, it does NOT own or truncate an entry name.
        next(0)?;
        Ok(Some(OsStr::from_bytes(&name[..name.len() - 1])))
    }
}

pub(crate) fn hash(path: &Path, next: Check<'_>) -> Result<[u8; 32]> {
    use sha2::{Digest, Sha256};
    let fd = open(path, libc::O_RDONLY, next)?;
    let mut buffer = vec::<u8>(64 * 1024, next)?;
    buffer.resize(64 * 1024, 0);
    let mut hash = Sha256::new();
    loop {
        next(0)?;
        let count = unsafe { libc::read(fd.as_raw_fd(), buffer.as_mut_ptr().cast(), buffer.len()) };
        if count < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error.into());
        }
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count as usize]);
    }
    Ok(hash.finalize().into())
}
