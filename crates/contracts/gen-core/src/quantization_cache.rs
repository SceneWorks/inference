//! Durable, backend-neutral storage for the result of a weight quantizer.
//!
//! Backends own the quantization and payload format; the consumer owns artifact storage,
//! publication, leases and retention. The key includes the actual input bytes, shape and the
//! versioned quantizer configuration, so an adapted or replaced weight cannot reuse stale codes.
use crate::Result;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizationArtifactKey {
    digest: String,
    extension: &'static str,
}
impl QuantizationArtifactKey {
    pub fn new(recipe: &str, shape: &[usize], input: &[u8], extension: &'static str) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"sceneworks-weight-quantization-v1\0");
        hash.update((extension.len() as u64).to_le_bytes());
        hash.update(extension.as_bytes());
        hash.update((recipe.len() as u64).to_le_bytes());
        hash.update(recipe.as_bytes());
        hash.update((shape.len() as u64).to_le_bytes());
        for dimension in shape {
            hash.update((*dimension as u64).to_le_bytes());
        }
        hash.update((input.len() as u64).to_le_bytes());
        hash.update(input);
        Self {
            digest: format!("{:x}", hash.finalize()),
            extension,
        }
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub fn extension(&self) -> &'static str {
        self.extension
    }
}

/// Keeps a verified immutable artifact leased while the backend consumes its packed payload.
pub trait QuantizationArtifact {
    fn path(&self) -> &Path;
}
/// A consumer-provided durable store. A hit must not invoke `produce`. A miss must publish
/// the complete output atomically before returning, or return an error without a partial hit.
pub trait QuantizationArtifactCache: Send + Sync {
    fn materialize(
        &self,
        key: &QuantizationArtifactKey,
        produce: &mut dyn FnMut(&Path) -> Result<()>,
    ) -> Result<Box<dyn QuantizationArtifact>>;

    /// Discard an unreadable completed artifact so the shared loader can prepare it again.
    /// Stores retaining immutable artifacts may decline; callers then return the read error.
    fn invalidate(&self, _key: &QuantizationArtifactKey) -> Result<bool> {
        Ok(false)
    }
}

thread_local! {
    static CURRENT: RefCell<Option<Arc<dyn QuantizationArtifactCache>>> = const { RefCell::new(None) };
}
/// Captured and propagated with the existing worker request context, including blocking tasks.
/// Installation is thread-scoped and restores the previous request on drop (also after panic).
#[derive(Clone, Default)]
pub struct QuantizationCacheContext(Option<Arc<dyn QuantizationArtifactCache>>);
impl QuantizationCacheContext {
    pub fn new(cache: Arc<dyn QuantizationArtifactCache>) -> Self {
        Self(Some(cache))
    }
    pub fn capture() -> Self {
        Self(CURRENT.with(|slot| slot.borrow().clone()))
    }
    pub fn cache(&self) -> Option<&Arc<dyn QuantizationArtifactCache>> {
        self.0.as_ref()
    }
    pub fn install(&self) -> impl Drop {
        struct Guard(
            Option<Arc<dyn QuantizationArtifactCache>>,
            std::marker::PhantomData<std::rc::Rc<()>>,
        );
        impl Drop for Guard {
            fn drop(&mut self) {
                CURRENT.with(|slot| *slot.borrow_mut() = self.0.take());
            }
        }
        Guard(
            CURRENT.with(|slot| slot.replace(self.0.clone())),
            std::marker::PhantomData,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct NeverProduce;
    impl QuantizationArtifactCache for NeverProduce {
        fn materialize(
            &self,
            _: &QuantizationArtifactKey,
            _: &mut dyn FnMut(&Path) -> Result<()>,
        ) -> Result<Box<dyn QuantizationArtifact>> {
            Err(crate::Error::Unsupported("context-only fixture".into()))
        }
    }
    #[test]
    fn contexts_restore_and_propagate_without_cross_thread_leaks() {
        assert!(QuantizationCacheContext::capture().cache().is_none());
        let context = QuantizationCacheContext::new(Arc::new(NeverProduce));
        {
            let _guard = context.install();
            let captured = QuantizationCacheContext::capture();
            assert!(captured.cache().is_some());
            std::thread::spawn(move || {
                assert!(QuantizationCacheContext::capture().cache().is_none());
                {
                    let _guard = captured.install();
                    assert!(QuantizationCacheContext::capture().cache().is_some());
                    let _nested = QuantizationCacheContext::default().install();
                    assert!(QuantizationCacheContext::capture().cache().is_none());
                }
                assert!(QuantizationCacheContext::capture().cache().is_none());
            })
            .join()
            .unwrap();
            assert!(QuantizationCacheContext::capture().cache().is_some());
        }
        assert!(QuantizationCacheContext::capture().cache().is_none());
    }
    #[test]
    fn identity_covers_bytes_shape_recipe_and_boundaries() {
        let key = |recipe, shape: &[usize], bytes: &[u8]| {
            QuantizationArtifactKey::new(recipe, shape, bytes, "bin").digest
        };
        let original = key("candle-q8-v1", &[2, 2], b"abcd");
        assert_eq!(original, key("candle-q8-v1", &[2, 2], b"abcd"));
        for changed in [
            key("candle-q4-v1", &[2, 2], b"abcd"),
            key("candle-q8-v2", &[2, 2], b"abcd"),
            key("candle-q8-v1", &[1, 4], b"abcd"),
            key("candle-q8-v1", &[2, 2], b"abce"),
        ] {
            assert_ne!(original, changed);
        }
    }
}
