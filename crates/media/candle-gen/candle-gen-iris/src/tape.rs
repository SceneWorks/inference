//! One autograd-tape node for a multi-op computation (sc-25690).
//!
//! Candle's tape keeps the output of **every** primitive alive until the loss drops: each op node
//! holds its operands, needed by its backward or not. A fused kernel (the inference `rms_norm`,
//! `softmax_last_dim`, a grouped convolution) is one node — its input and its output. Training
//! swaps those kernels for the same math in composable ops (they have no backward), and each
//! swap turns that one node into a chain whose every link is retained: five attention-score-sized
//! tensors per attention, three activation-sized ones per RMSNorm, 97 feature maps per depthwise
//! 7×7. On Iris-3B that was most of the activation memory of a training step.
//!
//! [`fold1`] / [`fold3`] put such a chain back on the tape as **one** node. The value is computed
//! off the tape (its intermediates die as they are produced); the node's backward replays the chain
//! on the tape for just that node — fresh leaves at the operands, the chain, a backward seeded with
//! the node's cotangent — and hands the operand gradients on. That is the chain rule over the
//! node's boundary, with the chain's own ops in the chain's own order: the gradients are the plain
//! graph's **bit for bit** as long as each operand's contributions land where the plain graph put
//! them, which the call sites arrange (an operand read twice inside the chain is passed twice, in
//! the order the plain backward reaches its two readers). The cost is one extra evaluation of the
//! chain per backward and one copy of its output.

use std::cell::Cell;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{
    self as candle, CpuStorage, CudaStorage, CustomOp1, CustomOp3, DType, Layout, MetalStorage,
    Shape, Storage, Tensor, Var,
};

thread_local! {
    static UNFOLDED: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with every [`fold1`] / [`fold3`] left as its plain chain on this thread — the reference
/// the folded tape is checked against (gradient identity, retained bytes).
#[doc(hidden)]
pub fn with_unfolded_tape<T>(f: impl FnOnce() -> T) -> T {
    let prev = UNFOLDED.with(|c| c.replace(true));
    let out = f();
    UNFOLDED.with(|c| c.set(prev));
    out
}

type Body = dyn Fn(&[Tensor]) -> candle::Result<Tensor> + Send + Sync;

struct Folded {
    name: &'static str,
    body: Arc<Body>,
    /// The value computed off the tape, handed to the node's forward once.
    value: Mutex<Option<Tensor>>,
}

impl Folded {
    /// The node's storage: a copy of the off-tape value (the original then drops).
    fn emit(&self) -> candle::Result<(Storage, Shape)> {
        let value = self
            .value
            .lock()
            .map_err(|_| candle::Error::msg(format!("{}: poisoned value", self.name)))?
            .take()
            .ok_or_else(|| candle::Error::msg(format!("{}: forward ran twice", self.name)))?;
        // Every backend's `try_clone` copies the whole buffer, so the value must own exactly its
        // elements — not a view into a larger one (a prefix `narrow` is contiguous at offset 0).
        // Only the CPU backend exposes a buffer's length; elsewhere the value is compacted
        // unconditionally (one more transient copy of the node's output).
        let exact = {
            let (storage, layout) = value.storage_and_layout();
            layout.is_contiguous()
                && layout.start_offset() == 0
                && cpu_len(&storage) == Some(layout.shape().elem_count())
        };
        let value = if exact {
            value
        } else {
            value.force_contiguous()?
        };
        let (storage, layout) = value.storage_and_layout();
        Ok((storage.try_clone(layout)?, value.shape().clone()))
    }

    /// The operand gradients of `Σ value · grad`: the chain replayed on the tape from fresh leaves.
    fn pull(&self, args: &[&Tensor], grad: &Tensor) -> candle::Result<Vec<Option<Tensor>>> {
        let leaves = args
            .iter()
            .map(|a| {
                Ok(if a.track_op() {
                    Var::from_tensor(&a.detach())?.as_tensor().clone()
                } else {
                    (*a).clone()
                })
            })
            .collect::<candle::Result<Vec<Tensor>>>()?;
        let value = (self.body)(&leaves)?;
        let seeded = value
            .to_dtype(DType::F32)?
            .mul(&grad.to_dtype(DType::F32)?)?
            .sum_all()?;
        let store = seeded.backward()?;
        // A tracked operand the chain never read still gets a (zero) gradient: candle's backward
        // expects one for every tracked node it visits, and nothing else may have supplied it.
        leaves
            .iter()
            .zip(args)
            .map(|(leaf, arg)| {
                Ok(match (arg.track_op(), store.get(leaf)) {
                    (false, _) => None,
                    (true, Some(g)) => Some(g.clone()),
                    (true, None) => Some(arg.zeros_like()?.detach()),
                })
            })
            .collect()
    }

    fn wrong_device(&self) -> candle::Error {
        candle::Error::msg(format!(
            "{}: the folded value is not on the operands' device",
            self.name
        ))
    }
}

struct Fold1(Folded);

impl CustomOp1 for Fold1 {
    fn name(&self) -> &'static str {
        self.0.name
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle::Result<(CpuStorage, Shape)> {
        match self.0.emit()? {
            (Storage::Cpu(s), shape) => Ok((s, shape)),
            _ => Err(self.0.wrong_device()),
        }
    }
    fn cuda_fwd(&self, _: &CudaStorage, _: &Layout) -> candle::Result<(CudaStorage, Shape)> {
        match self.0.emit()? {
            (Storage::Cuda(s), shape) => Ok((s, shape)),
            _ => Err(self.0.wrong_device()),
        }
    }
    fn metal_fwd(&self, _: &MetalStorage, _: &Layout) -> candle::Result<(MetalStorage, Shape)> {
        match self.0.emit()? {
            (Storage::Metal(s), shape) => Ok((s, shape)),
            _ => Err(self.0.wrong_device()),
        }
    }
    fn bwd(&self, arg: &Tensor, _: &Tensor, grad: &Tensor) -> candle::Result<Option<Tensor>> {
        Ok(self.0.pull(&[arg], grad)?.pop().flatten())
    }
}

struct Fold3(Folded);

impl CustomOp3 for Fold3 {
    fn name(&self) -> &'static str {
        self.0.name
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle::Result<(CpuStorage, Shape)> {
        match self.0.emit()? {
            (Storage::Cpu(s), shape) => Ok((s, shape)),
            _ => Err(self.0.wrong_device()),
        }
    }
    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> candle::Result<(CudaStorage, Shape)> {
        match self.0.emit()? {
            (Storage::Cuda(s), shape) => Ok((s, shape)),
            _ => Err(self.0.wrong_device()),
        }
    }
    fn metal_fwd(
        &self,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> candle::Result<(MetalStorage, Shape)> {
        match self.0.emit()? {
            (Storage::Metal(s), shape) => Ok((s, shape)),
            _ => Err(self.0.wrong_device()),
        }
    }
    fn bwd(
        &self,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        _: &Tensor,
        grad: &Tensor,
    ) -> candle::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let mut g = self.0.pull(&[a, b, c], grad)?.into_iter();
        Ok((g.next().flatten(), g.next().flatten(), g.next().flatten()))
    }
}

/// The element count of a CPU buffer (`None` on a device whose storage does not expose one).
fn cpu_len(storage: &Storage) -> Option<usize> {
    let Storage::Cpu(cpu) = storage else {
        return None;
    };
    Some(match cpu {
        CpuStorage::U8(v) => v.len(),
        CpuStorage::U32(v) => v.len(),
        CpuStorage::I16(v) => v.len(),
        CpuStorage::I32(v) => v.len(),
        CpuStorage::I64(v) => v.len(),
        CpuStorage::BF16(v) => v.len(),
        CpuStorage::F16(v) => v.len(),
        CpuStorage::F32(v) => v.len(),
        CpuStorage::F64(v) => v.len(),
        _ => return None,
    })
}

fn folded(name: &'static str, body: Arc<Body>, args: &[&Tensor]) -> candle::Result<Folded> {
    let detached: Vec<Tensor> = args.iter().map(|a| a.detach()).collect();
    let value = body(&detached)?.detach();
    Ok(Folded {
        name,
        body,
        value: Mutex::new(Some(value)),
    })
}

fn plain(args: &[&Tensor]) -> bool {
    UNFOLDED.with(|c| c.get()) || !args.iter().any(|a| a.track_op())
}

/// `body(x)` as one tape node (the plain chain when `x` is off the tape). `x` must have no other
/// reader than this node, and `body` must return a tensor it computed (not a view of `x`).
pub fn fold1(
    name: &'static str,
    x: &Tensor,
    body: impl Fn(&Tensor) -> candle::Result<Tensor> + Send + Sync + 'static,
) -> candle::Result<Tensor> {
    if plain(&[x]) {
        return body(x);
    }
    let body: Arc<Body> = Arc::new(move |a: &[Tensor]| body(&a[0]));
    x.apply_op1(Fold1(folded(name, body, &[x])?))
}

/// `body(a, b, c)` as one tape node (the plain chain when no operand is on the tape). The plain
/// backward must reach `a`'s reader, then `b`'s, then `c`'s; pass one tensor twice when the chain
/// reads it twice.
pub fn fold3(
    name: &'static str,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    body: impl Fn(&Tensor, &Tensor, &Tensor) -> candle::Result<Tensor> + Send + Sync + 'static,
) -> candle::Result<Tensor> {
    if plain(&[a, b, c]) {
        return body(a, b, c);
    }
    let body: Arc<Body> = Arc::new(move |t: &[Tensor]| body(&t[0], &t[1], &t[2]));
    a.apply_op3(b, c, Fold3(folded(name, body, &[a, b, c])?))
}
