//! Requested-layout law for existing owning Clone implementations on Apple.
//!
//! This is NOT allocator/RSS/physical accounting and never changes evidence currency.
//! Rust 1.96.0 / 1.97.1 alloc sources: Vec/String/Unix OsString clone their length,
//! Option/tuples clone contained values, Arc::clone only increments its count.
//! BTreeMap::clone (alloc/src/collections/btree/map.rs:clone_subtree) allocates
//! exactly one boxed leaf/internal payload per nonempty source node, with no
//! heap traversal stack. node.rs has B=6, a leaf containing a parent pointer,
//! two u16 fields, [MaybeUninit<K>;11], [MaybeUninit<V>;11]; repr(C) internal
//! adds [MaybeUninit<NonNull<LeafNode<K,V>>>;12]. SetValZST has unit layout.
//!
//! We bound the WHOLE payload from those fields, rounding EACH field to the
//! largest field alignment. This upper covers every natural-alignment field
//! permutation of the pinned compilers' Rust struct layout, not merely one
//! guessed private order. An internal node adds all twelve child pointers.
//! Every nonempty node owns >=1 key, so node count <= element count. We charge
//! every node at that internal upper and separately clone every owned key/value.
//! Empty maps clone without a root allocation. These are source-derived bounds,
//! NOT size_of<BTreeMap>, an opaque pointer stand-in, or an entry-name bound.
//! Actual allocator-policing tests exercise empty, split and deep maps, so the
//! supported compiler's requested allocations are checked independently as well.
use crate::{source_inventory::bounded_fs::overflow, Result};
use std::alloc::Layout;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

pub(crate) trait CloneLaw {
    fn clone_bytes(&self) -> Result<u64>;
}
pub(crate) fn sum<const N: usize>(parts: [Result<u64>; N]) -> Result<u64> {
    parts.into_iter().try_fold(0u64, |total, part| {
        total.checked_add(part?).ok_or_else(overflow)
    })
}
pub(crate) fn scalar<T: Copy>(_: &T) -> Result<u64> {
    Ok(0)
}
pub(crate) fn array<T>(len: usize) -> Result<u64> {
    Ok(Layout::array::<T>(len).map_err(|_| overflow())?.size() as u64)
}
macro_rules! scalars {
    ($($ty:ty),*) => { $(impl CloneLaw for $ty {
        fn clone_bytes(&self) -> Result<u64> { scalar(self) }
    })* };
}
scalars!((), bool, u8, u16, u32, u64, usize, i32, i64, f32, f64);
impl CloneLaw for String {
    fn clone_bytes(&self) -> Result<u64> {
        array::<u8>(self.len())
    }
}
impl CloneLaw for PathBuf {
    fn clone_bytes(&self) -> Result<u64> {
        array::<u8>(self.as_os_str().as_encoded_bytes().len())
    }
}
impl<T> CloneLaw for Arc<T> {
    fn clone_bytes(&self) -> Result<u64> {
        Ok(0)
    }
}
impl<T: CloneLaw> CloneLaw for Option<T> {
    fn clone_bytes(&self) -> Result<u64> {
        self.as_ref().map_or(Ok(0), CloneLaw::clone_bytes)
    }
}
impl<A: CloneLaw, B: CloneLaw> CloneLaw for (A, B) {
    fn clone_bytes(&self) -> Result<u64> {
        sum([self.0.clone_bytes(), self.1.clone_bytes()])
    }
}
impl<T: CloneLaw> CloneLaw for Vec<T> {
    fn clone_bytes(&self) -> Result<u64> {
        self.iter().try_fold(array::<T>(self.len())?, |n, item| {
            n.checked_add(item.clone_bytes()?).ok_or_else(overflow)
        })
    }
}
fn node_upper<K, V>() -> Result<u64> {
    let fields = [
        Layout::new::<Option<std::ptr::NonNull<()>>>(),
        Layout::new::<std::mem::MaybeUninit<u16>>(),
        Layout::new::<u16>(),
        Layout::array::<std::mem::MaybeUninit<K>>(11).map_err(|_| overflow())?,
        Layout::array::<std::mem::MaybeUninit<V>>(11).map_err(|_| overflow())?,
    ];
    let alignment = fields.iter().map(Layout::align).max().unwrap();
    let leaf = fields.into_iter().try_fold(0usize, |n, field| {
        let size = field
            .align_to(alignment)
            .map_err(|_| overflow())?
            .pad_to_align()
            .size();
        n.checked_add(size).ok_or_else(overflow)
    })?;
    let leaf = Layout::from_size_align(leaf, alignment).map_err(|_| overflow())?;
    let children = Layout::array::<std::ptr::NonNull<()>>(12).map_err(|_| overflow())?;
    Ok(leaf
        .extend(children)
        .map_err(|_| overflow())?
        .0
        .pad_to_align()
        .size() as u64)
}
impl<K: CloneLaw, V: CloneLaw> CloneLaw for BTreeMap<K, V> {
    fn clone_bytes(&self) -> Result<u64> {
        let nodes = node_upper::<K, V>()?
            .checked_mul(self.len() as u64)
            .ok_or_else(overflow)?;
        self.iter().try_fold(nodes, |n, (key, value)| {
            sum([Ok(n), key.clone_bytes(), value.clone_bytes()])
        })
    }
}
impl<T: CloneLaw> CloneLaw for BTreeSet<T> {
    fn clone_bytes(&self) -> Result<u64> {
        let nodes = node_upper::<T, ()>()?
            .checked_mul(self.len() as u64)
            .ok_or_else(overflow)?;
        self.iter()
            .try_fold(nodes, |n, value| sum([Ok(n), value.clone_bytes()]))
    }
}

// Exhaustive structural patterns deliberately have NO '..'. Adding any field or
// making a scalar non-Copy forces this law to be revisited at compile time.
use crate::memory_strategy::{
    MemoryDecodeArtifactIdentity, MemoryDecodeGeometryPolicy, MemoryDecodeQualityDisposition,
    MemoryDecodeQualityFixture, MemoryDecodeQualityRuntimeIdentity, MemoryMode,
};
impl CloneLaw for MemoryMode {
    fn clone_bytes(&self) -> Result<u64> {
        match self {
            Self::TextToImage | Self::ImageToImage | Self::Edit => Ok(0),
            Self::Other(mode) => mode.clone_bytes(),
        }
    }
}
impl CloneLaw for MemoryDecodeArtifactIdentity {
    fn clone_bytes(&self) -> Result<u64> {
        let Self {
            repository,
            revision,
            variant,
            fingerprint,
        } = self;
        sum([
            repository.clone_bytes(),
            revision.clone_bytes(),
            variant.clone_bytes(),
            fingerprint.clone_bytes(),
        ])
    }
}
impl CloneLaw for MemoryDecodeQualityRuntimeIdentity {
    fn clone_bytes(&self) -> Result<u64> {
        let Self { artifact } = self;
        artifact.clone_bytes()
    }
}
impl CloneLaw for MemoryDecodeQualityFixture {
    fn clone_bytes(&self) -> Result<u64> {
        let Self {
            seed,
            production_latent_provenance,
            production_latent_sha256,
            dense_output_sha256,
            tiled_output_sha256,
            observed_error,
        } = self;
        sum([
            scalar(seed),
            production_latent_provenance.clone_bytes(),
            production_latent_sha256.clone_bytes(),
            dense_output_sha256.clone_bytes(),
            tiled_output_sha256.clone_bytes(),
            scalar(observed_error),
        ])
    }
}
impl CloneLaw for MemoryDecodeQualityDisposition {
    fn clone_bytes(&self) -> Result<u64> {
        match self {
            Self::Admitted => Ok(0),
            Self::Refused { reason } => reason.clone_bytes(),
        }
    }
}
impl CloneLaw for MemoryDecodeGeometryPolicy {
    fn clone_bytes(&self) -> Result<u64> {
        let Self {
            quality_abi,
            family,
            resolved_route,
            backend,
            tier,
            load_shape,
            artifact,
            implementation_fingerprint,
            mode,
            overlay,
            geometry,
            use_pid,
            tile_edge,
            overlap,
            metric,
            maximum_error,
            fixtures,
            production_evidence_sha256,
            disposition,
        } = self;
        sum([
            scalar(quality_abi),
            family.clone_bytes(),
            resolved_route.clone_bytes(),
            scalar(backend),
            scalar(tier),
            scalar(load_shape),
            artifact.clone_bytes(),
            implementation_fingerprint.clone_bytes(),
            mode.clone_bytes(),
            overlay.clone_bytes(),
            scalar(geometry),
            scalar(use_pid),
            scalar(tile_edge),
            scalar(overlap),
            metric.clone_bytes(),
            scalar(maximum_error),
            fixtures.clone_bytes(),
            production_evidence_sha256.clone_bytes(),
            disposition.clone_bytes(),
        ])
    }
}
