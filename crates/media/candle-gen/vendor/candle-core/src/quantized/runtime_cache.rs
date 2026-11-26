//! Thread-scoped runtime weight cache interface. Backends supply storage; core owns the common
//! quantizer entry points. Cache installation is request-wide, never model-specific.
use super::{GgmlDType, QTensor};
use crate::{Device, Result, Tensor};
use std::{cell::RefCell, sync::Arc};

pub trait RuntimeQuantizationCache: Send + Sync {
    fn quantize(
        &self,
        source: &Tensor,
        dtype: GgmlDType,
        device: &Device,
        mode: &'static str,
        imatrix: Option<&[f32]>,
        produce: &mut dyn FnMut() -> Result<QTensor>,
    ) -> Result<QTensor>;
    fn repack_affine(
        &self,
        _codes: &Tensor,
        _scales: &Tensor,
        _biases: &Tensor,
        _group_size: usize,
        _device: &Device,
        produce: &mut dyn FnMut() -> Result<QTensor>,
    ) -> Result<QTensor> {
        produce()
    }
}
#[derive(Clone, Default)]
enum CacheScope {
    #[default]
    Inherit,
    Disabled,
    Active(Arc<dyn RuntimeQuantizationCache>),
}
thread_local! { static CURRENT: RefCell<CacheScope> = const { RefCell::new(CacheScope::Inherit) }; }
/// Capture the active request store when an immutable source tensor is read. It can then follow
/// that operand to parallel loader threads without requiring model-specific context propagation.
pub fn capture() -> Option<Arc<dyn RuntimeQuantizationCache>> {
    CURRENT.with(|slot| match &*slot.borrow() {
        CacheScope::Active(cache) => Some(cache.clone()),
        _ => None,
    })
}
pub fn for_tensor(source: &Tensor) -> Option<Arc<dyn RuntimeQuantizationCache>> {
    CURRENT.with(|slot| match &*slot.borrow() {
        CacheScope::Active(cache) => Some(cache.clone()),
        CacheScope::Disabled => None,
        CacheScope::Inherit => source.checkpoint_runtime_cache(),
    })
}
pub fn install(cache: Option<Arc<dyn RuntimeQuantizationCache>>) -> impl Drop {
    struct Guard(CacheScope, std::marker::PhantomData<std::rc::Rc<()>>);
    impl Drop for Guard {
        fn drop(&mut self) {
            CURRENT.with(|slot| *slot.borrow_mut() = std::mem::take(&mut self.0));
        }
    }
    let scope = cache
        .map(CacheScope::Active)
        .unwrap_or(CacheScope::Disabled);
    Guard(
        CURRENT.with(|slot| slot.replace(scope)),
        std::marker::PhantomData,
    )
}
/// A producer must run the core algorithm once, without recursively acquiring its own cache key.
pub fn without_cache<T>(run: impl FnOnce() -> T) -> T {
    let _guard = install(None);
    run()
}
pub(super) fn quantize(
    source: &Tensor,
    dtype: GgmlDType,
    device: &Device,
    mode: &'static str,
    imatrix: Option<&[f32]>,
    produce: impl FnOnce() -> Result<QTensor>,
) -> Result<QTensor> {
    let Some(cache) = for_tensor(source) else {
        return produce();
    };
    let mut produce = Some(produce);
    without_cache(|| {
        cache.quantize(source, dtype, device, mode, imatrix, &mut || {
            produce
                .take()
                .ok_or_else(|| crate::Error::Msg("runtime quantizer invoked twice".into()))?(
            )
        })
    })
}
