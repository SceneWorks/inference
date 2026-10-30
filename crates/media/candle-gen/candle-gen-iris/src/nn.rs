//! The leaf modules of the Iris graph and the precision policy they share — the Candle twin of
//! `mlx-gen-iris`'s `nn` module, op for op.
//!
//! **Precision policy** (mirrors upstream's CUDA path, `torch.autocast(bfloat16)` over FP32
//! parameters, and degenerates to upstream's FP32 CPU path when the compute dtype is f32):
//!
//! * every `Linear` casts its input to the compute dtype and returns the compute dtype (autocast's
//!   matmul rule) — its weight and bias are stored in the compute dtype;
//! * every `RMSNorm` normalises in f32 and multiplies by an f32 gain, so it returns f32 (upstream's
//!   `RMSNorm` computes in fp32 and its fp32 gain promotes the product);
//! * attention runs in the compute dtype (`scaled_dot_product_attention` is on autocast's
//!   low-precision list); RoPE rotates in f32 and returns the dtype it was given;
//! * every other elementwise op follows ordinary **type promotion** — candle has none, so the
//!   binary helpers here ([`add`], [`mul`], [`cat`], …) promote to f32 whenever their operands
//!   disagree, exactly as torch (and MLX) do. The residual streams therefore stay f32 where
//!   upstream's do and the PiT pixel stream stays in the compute dtype where upstream's does.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use candle_gen::attention::repeat_kv;
use candle_gen::attention::AttentionPlan;
use candle_gen::candle_core::quantized::{GgmlDType, QTensor};
use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::ops::{softmax, softmax_last_dim};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{CancelFlag, Quant};
use candle_gen::quant::{LokrFactors, QLinear};
use candle_gen::{CandleError as Error, Result};
use gen_core_iris as iris;

/// The dtype two operands meet at: themselves when they agree, f32 otherwise (the only two dtypes
/// this graph carries are f32 and the compute dtype).
fn common(a: DType, b: DType) -> DType {
    if a == b {
        a
    } else {
        DType::F32
    }
}

fn to(x: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(if x.dtype() == dtype {
        x.clone()
    } else {
        x.to_dtype(dtype)?
    })
}

/// Promoting broadcast `a + b`.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dt = common(a.dtype(), b.dtype());
    Ok(to(a, dt)?.broadcast_add(&to(b, dt)?)?)
}

/// Promoting broadcast `a − b`.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dt = common(a.dtype(), b.dtype());
    Ok(to(a, dt)?.broadcast_sub(&to(b, dt)?)?)
}

/// Promoting broadcast `a · b`.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let dt = common(a.dtype(), b.dtype());
    Ok(to(a, dt)?.broadcast_mul(&to(b, dt)?)?)
}

/// `x · (1 + scale) + shift`, promoting like torch.
pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let one_plus = (scale + 1.0)?;
    add(&mul(x, &one_plus)?, shift)
}

/// Concatenate along `axis`, promoting every part to the common dtype first.
pub fn cat(parts: &[&Tensor], axis: usize) -> Result<Tensor> {
    let dt = parts
        .iter()
        .skip(1)
        .fold(parts[0].dtype(), |acc, p| common(acc, p.dtype()));
    let parts = parts
        .iter()
        .map(|p| to(p, dt))
        .collect::<Result<Vec<_>>>()?;
    Ok(Tensor::cat(&parts, axis)?)
}

/// Split the last axis into `n` equal chunks (`tensor.chunk(n, dim=-1)`).
pub fn chunk_last(x: &Tensor, n: usize) -> Result<Vec<Tensor>> {
    Ok(x.chunk(n, D::Minus1)?)
}

/// `x[.., :at, ..]` and `x[.., at:, ..]` along `axis`.
pub fn split_at(x: &Tensor, axis: usize, at: usize) -> Result<(Tensor, Tensor)> {
    let len = x.dim(axis)?;
    Ok((x.narrow(axis, 0, at)?, x.narrow(axis, at, len - at)?))
}

/// Where a [`Checkpoint`] reads its tensors from.
enum Source {
    /// A `'static` mmap [`VarBuilder`] over one `model.safetensors` (tensors read at f32).
    Mmap(VarBuilder<'static>),
    /// In-memory tensors handed out **as stored** (dtype and autograd identity preserved) — the
    /// training backbone (sc-25686): f32 `Var` masters of a full run, or a frozen base already at
    /// the provider's load dtypes under an adapter run.
    Map(HashMap<String, Tensor>),
}

/// The checkpoint the backbone modules are built from: an mmap over the single
/// `model.safetensors` (or, for training, an in-memory tensor map) plus its key list, so every key
/// a module consumes is recorded and a leftover (or missing) key is a load error naming it.
pub struct Checkpoint {
    source: Source,
    device: Device,
    /// The same file mmapped onto the host, when `device` is not the CPU: the source a packed
    /// projection is quantized from (sc-26266), so its f32 weight never reaches the device.
    host: Option<VarBuilder<'static>>,
    /// Key prefix of this view (`""` for the whole file); `keys` are stored without it.
    prefix: String,
    keys: BTreeSet<String>,
    /// Each key's stored shape, read from the header (a tensor map's own shapes).
    shapes: BTreeMap<String, Vec<usize>>,
    used: RefCell<BTreeSet<String>>,
    /// The dtype each key a [`Loader`] consumed is held in once its module is built — the
    /// load-exact width inventory `crate::memory_strategy` prices a backbone against.
    stored: RefCell<BTreeMap<String, DType>>,
    /// Forward-time adapter residuals keyed by projection path (`blocks.3.attn_proj`), handed to the
    /// [`Loader`] when it builds that projection.
    residuals: RefCell<BTreeMap<String, Vec<Residual>>>,
    /// Every tensor a [`Loader`] built from an in-memory map, at the dtype it was built in
    /// ([`loaded`](Self::loaded)).
    loaded: RefCell<HashMap<String, Tensor>>,
    /// Bytes of every parameter a [`Loader`] handed a module, as the module holds it — dense
    /// tensors at their dtype, packed projections at their GGML storage size
    /// ([`resident_bytes`](Self::resident_bytes)).
    resident: std::cell::Cell<u64>,
}

thread_local! {
    static LOADER_CAST_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Bytes this thread's [`Loader`]s produced by casting a checkpoint tensor to another dtype — a
/// test probe: a second copy of a backbone's weights is otherwise invisible.
#[doc(hidden)]
pub fn loader_cast_bytes() -> usize {
    LOADER_CAST_BYTES.with(|c| c.get())
}

fn missing(key: &str) -> Error {
    Error::Msg(format!(
        "iris: the backbone checkpoint is missing `{key}` (config.yaml and model.safetensors \
         disagree)"
    ))
}

impl Checkpoint {
    /// mmap `file` onto `device` (tensors are read one at a time at f32, then cast by the caller).
    pub fn open(file: &Path, device: &Device) -> Result<Self> {
        Self::open_prefixed(file, device, "")
    }

    /// The view of `file` under `prefix` (e.g. the depth export's `pixel.` backbone): keys are
    /// taken and reported without the prefix, and keys outside it are not part of this view.
    pub fn open_prefixed(file: &Path, device: &Device, prefix: &str) -> Result<Self> {
        let shapes: BTreeMap<String, Vec<usize>> = safetensors_shapes(file)?
            .into_iter()
            .filter_map(|(k, shape)| k.strip_prefix(prefix).map(|k| (k.to_owned(), shape)))
            .collect();
        let keys = shapes.keys().cloned().collect();
        let files = [file.to_path_buf()];
        let vb = candle_gen::mmap_var_builder(&files, DType::F32, device)?;
        let host = if device.is_cpu() {
            None
        } else {
            Some(candle_gen::mmap_var_builder(
                &files,
                DType::F32,
                &Device::Cpu,
            )?)
        };
        Ok(Self {
            source: Source::Mmap(vb),
            host,
            device: device.clone(),
            prefix: prefix.to_owned(),
            keys,
            shapes,
            used: RefCell::new(BTreeSet::new()),
            stored: RefCell::new(BTreeMap::new()),
            residuals: RefCell::new(BTreeMap::new()),
            loaded: RefCell::new(HashMap::new()),
            resident: std::cell::Cell::new(0),
        })
    }

    /// A checkpoint over in-memory `tensors` (upstream key names) on `device`: [`take`](Self::take)
    /// returns each tensor exactly as stored — no dtype change, the same autograd node — so a
    /// backbone built from `Var` tensors is differentiable with respect to them.
    pub fn from_tensors(tensors: HashMap<String, Tensor>, device: &Device) -> Self {
        Self {
            keys: tensors.keys().cloned().collect(),
            shapes: tensors
                .iter()
                .map(|(k, t)| (k.clone(), t.dims().to_vec()))
                .collect(),
            source: Source::Map(tensors),
            host: None,
            device: device.clone(),
            prefix: String::new(),
            used: RefCell::new(BTreeSet::new()),
            stored: RefCell::new(BTreeMap::new()),
            residuals: RefCell::new(BTreeMap::new()),
            loaded: RefCell::new(HashMap::new()),
            resident: std::cell::Cell::new(0),
        }
    }

    /// Install forward-time adapter `residuals` (keyed by projection path). Every path must be a
    /// `‹path›.weight` of the checkpoint, and the model build must consume every one
    /// ([`unconsumed_residuals`](Self::unconsumed_residuals)).
    pub fn with_residuals(self, residuals: BTreeMap<String, Vec<Residual>>) -> Result<Self> {
        if let Some(path) = residuals
            .keys()
            .find(|p| !self.keys.contains(&format!("{p}.weight")))
        {
            return Err(Error::Msg(format!(
                "iris: an adapter targets `{path}`, which the backbone checkpoint does not carry"
            )));
        }
        *self.residuals.borrow_mut() = residuals;
        Ok(self)
    }

    /// Read one tensor (as f32 from a file; as stored from a tensor map), recording the key as
    /// consumed.
    pub fn take(&self, key: &str) -> Result<Tensor> {
        self.consume(key)?;
        Ok(match &self.source {
            Source::Mmap(vb) => vb.get_unchecked(&format!("{}{key}", self.prefix))?,
            Source::Map(map) => map[key].clone(),
        })
    }

    /// [`take`](Self::take), read onto the host when this checkpoint is mmapped onto another
    /// device (a tensor map's tensor as stored) — the source a packed projection is quantized
    /// from, so its f32 weight never occupies the device.
    fn take_host(&self, key: &str) -> Result<Tensor> {
        match &self.host {
            Some(host) => {
                self.consume(key)?;
                Ok(host.get_unchecked(&format!("{}{key}", self.prefix))?)
            }
            None => self.take(key),
        }
    }

    /// The stored shape of `key`, read without touching its bytes.
    fn shape_of(&self, key: &str) -> Result<&[usize]> {
        self.shapes
            .get(key)
            .map(Vec::as_slice)
            .ok_or_else(|| missing(key))
    }

    /// Record `key` as consumed (refused when the checkpoint does not carry it).
    fn consume(&self, key: &str) -> Result<()> {
        if !self.keys.contains(key) {
            return Err(missing(key));
        }
        self.used.borrow_mut().insert(key.to_owned());
        Ok(())
    }

    /// [`take`](Self::take) at `dtype`, as the modules hold it.
    fn take_as(&self, key: &str, dtype: DType) -> Result<Tensor> {
        let stored = self.take(key)?;
        self.held_as(key, stored, dtype)
    }

    /// The tensor `key` (already [`take`](Self::take)n as `stored`) as a module holds it at
    /// `dtype`.
    fn held_as(&self, key: &str, stored: Tensor, dtype: DType) -> Result<Tensor> {
        let cast = to(&stored, dtype)?;
        self.hold((cast.elem_count() * dtype.size_in_bytes()) as u64);
        if stored.dtype() != dtype {
            let bytes = cast.elem_count() * dtype.size_in_bytes();
            LOADER_CAST_BYTES.with(|c| c.set(c.get() + bytes));
        }
        if matches!(self.source, Source::Map(_)) {
            self.loaded
                .borrow_mut()
                .insert(key.to_owned(), cast.clone());
        }
        Ok(cast)
    }

    /// The in-memory tensors of this checkpoint as the modules built from it hold them: each one a
    /// [`Loader`] read at the dtype it was read in (the very tensor, on the tape if its source is),
    /// the rest as stored. A second model built from these shares the first one's casts instead of
    /// casting the source again (sc-25690).
    pub fn loaded(&self) -> HashMap<String, Tensor> {
        let loaded = self.loaded.borrow();
        match &self.source {
            Source::Mmap(_) => HashMap::new(),
            Source::Map(map) => map
                .iter()
                .map(|(k, v)| (k.clone(), loaded.get(k).unwrap_or(v).clone()))
                .collect(),
        }
    }

    /// Count `bytes` a module now holds.
    fn hold(&self, bytes: u64) {
        self.resident.set(self.resident.get() + bytes);
    }

    /// Bytes of every parameter the modules built from this checkpoint hold, as they hold them:
    /// the dense tensors at their dtype and each packed projection at its GGML storage size,
    /// summed from the tensors themselves as they were built (adapter residuals excluded — they
    /// are an overlay).
    pub fn resident_bytes(&self) -> u64 {
        self.resident.get()
    }

    /// Record the dtype `key` is held in by the module built from it.
    fn record_stored(&self, key: &str, dtype: DType) {
        self.stored.borrow_mut().insert(key.to_owned(), dtype);
    }

    /// The dtype every [`Loader`]-built key is held in (keys without this view's prefix): the
    /// compute dtype for a projection's weight and bias, f32 for everything else.
    pub fn stored_dtypes(&self) -> BTreeMap<String, DType> {
        self.stored.borrow().clone()
    }

    /// The residuals installed on projection `path` (handed out once).
    fn take_residuals(&self, path: &str) -> Vec<Residual> {
        self.residuals.borrow_mut().remove(path).unwrap_or_default()
    }

    /// Keys the checkpoint carries that no module consumed, sorted.
    pub fn unused_keys(&self) -> Vec<String> {
        let used = self.used.borrow();
        self.keys.difference(&used).cloned().collect()
    }

    /// Adapter targets no module built as a projection — empty after a successful model build (an
    /// adapter on a 2-D tensor the graph does not read as a Linear is a load error, never a silently
    /// dropped residual).
    pub fn unconsumed_residuals(&self) -> Vec<String> {
        self.residuals.borrow().keys().cloned().collect()
    }
}

/// The tensor names in a `.safetensors` header (`__metadata__` excluded), read without touching the
/// tensor bytes.
pub fn safetensors_keys(file: &Path) -> Result<BTreeSet<String>> {
    Ok(safetensors_shapes(file)?.into_keys().collect())
}

/// The tensor names and shapes in a `.safetensors` header (`__metadata__` excluded), read without
/// touching the tensor bytes.
pub fn safetensors_shapes(file: &Path) -> Result<BTreeMap<String, Vec<usize>>> {
    // gen-core's capped, validated header reader (sc-27387): it also checks dtypes and payload
    // offsets, which the former hand-rolled parser skipped.
    Ok(
        candle_gen::gen_core::safetensors_file_tensor_locations(file)
            .map_err(|e| Error::Msg(format!("iris: reading {}: {e}", file.display())))?
            .tensors
            .into_iter()
            .map(|location| (location.header.name, location.header.shape))
            .collect(),
    )
}

/// The eligibility and padding granularity of a quantized backbone projection (sc-26266): the MLX
/// twin's affine group size, so the two backends pack the same projections over the same padded
/// widths. The GGML blocks this backend stores them in (32 wide, [`GgmlDType::Q4_0`] /
/// [`GgmlDType::Q8_0`]) divide it, and cost the same 4.5 / 8.5 bits per weight as MLX's group-64
/// affine scales and biases.
pub const QUANT_GROUP_SIZE: usize = candle_gen::quant::MLX_GROUP_SIZE;

/// The load-time quantization of one backbone build (sc-26266) — the Candle twin of
/// `mlx_gen_iris::nn::BackboneQuant`: the tier.
///
/// A projection whose input width is not a multiple of [`QUANT_GROUP_SIZE`] is packed over a
/// zero-padded input. The release has exactly one such family — the SwiGLU down projections
/// (`*.mlp.w2`, `[2560, 6826]`). Zero-padding their input to the next group multiple is exact
/// (`[x, 0] · [W, 0]ᵀ = x · Wᵀ`) and lets them quantize like every other projection — adapted or
/// not: an adapter residual on such a projection reads the unpadded input while the packed base
/// reads the padded one ([`Linear::forward`]), so the tier is one layout for the whole backbone
/// whatever adapters a load installs, as on the MLX twin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackboneQuant {
    /// The tier ([`Quant::Q4`] or [`Quant::Q8`]).
    pub quant: Quant,
}

impl BackboneQuant {
    /// The quantization a load with `quant` applies: `None` is the dense tier. Only the q8 / q4
    /// tiers exist on this route; NVFP4 is a typed refusal.
    pub fn for_load(quant: Option<Quant>) -> Result<Option<Self>> {
        match quant {
            None => Ok(None),
            Some(quant @ (Quant::Q4 | Quant::Q8)) => Ok(Some(Self { quant })),
            Some(other) => Err(Error::Unsupported(format!(
                "iris: LoadSpec::quantize {other:?} is not a tier of the Candle Iris-3B route; it \
                 packs the q8 and q4 tiers"
            ))),
        }
    }

    /// The GGML block type the tier's projections are stored in.
    pub fn ggml_dtype(self) -> GgmlDType {
        match self.quant {
            Quant::Q8 => GgmlDType::Q8_0,
            _ => GgmlDType::Q4_0,
        }
    }
}

/// How one `[out, in]` projection is held under a [`BackboneQuant`] — the single rule
/// [`Loader::linear`] applies and the memory contract prices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectionLayout {
    /// Stored in the compute dtype.
    Dense,
    /// GGML-packed as `dtype` over an input zero-padded to `padded_in` (`== in` when aligned).
    Quantized { dtype: GgmlDType, padded_in: usize },
}

impl ProjectionLayout {
    /// The layout of a projection with `in_features` inputs. A projection narrower than one group
    /// (the release's 3-, 12- and 16-wide pixel and pooling heads, a few KB in all) stays dense.
    pub fn of(in_features: usize, quant: Option<BackboneQuant>) -> Self {
        let Some(quant) = quant else {
            return Self::Dense;
        };
        if in_features < QUANT_GROUP_SIZE {
            return Self::Dense;
        }
        let padded_in = in_features.div_ceil(QUANT_GROUP_SIZE) * QUANT_GROUP_SIZE;
        Self::Quantized {
            dtype: quant.ggml_dtype(),
            padded_in,
        }
    }

    /// Resident bytes of an `[out, in]` weight held this way, `dense_width` bytes per element when
    /// dense: the GGML blocks of the (padded) rows when packed.
    pub fn weight_bytes(self, out: u64, in_features: u64, dense_width: u64) -> u64 {
        match self {
            Self::Dense => out * in_features * dense_width,
            Self::Quantized { dtype, padded_in } => {
                out * (padded_in / dtype.block_size()) as u64 * dtype.type_size() as u64
            }
        }
    }
}

/// Builds modules from a checkpoint at one compute dtype.
pub struct Loader<'a> {
    pub weights: &'a Checkpoint,
    pub compute: DType,
    /// The load-time quantization of every projection ([`ProjectionLayout::of`]); `None` is the
    /// dense tier.
    pub quant: Option<BackboneQuant>,
}

impl Loader<'_> {
    /// A tensor kept in f32 (norm gains, adaLN biases, the text position table).
    pub fn f32(&self, key: &str) -> Result<Tensor> {
        self.weights.record_stored(key, DType::F32);
        self.weights.take_as(key, DType::F32)
    }

    pub fn linear(&self, prefix: &str, bias: bool) -> Result<Linear> {
        let weight_key = format!("{prefix}.weight");
        let residuals = self.weights.take_residuals(prefix);
        let in_features = self
            .weights
            .shape_of(&weight_key)?
            .last()
            .copied()
            .unwrap_or(0);
        // An adapted projection packs like any other; its residuals stay forward-time terms over
        // the packed base, reading the checkpoint-width input ([`Linear::forward`]).
        if let ProjectionLayout::Quantized { dtype, padded_in } =
            ProjectionLayout::of(in_features, self.quant)
        {
            let stored = self.weights.take_host(&weight_key)?;
            return self.quantized_linear(prefix, stored, bias, residuals, dtype, padded_in);
        }
        let stored = self.weights.take(&weight_key)?;
        self.weights.record_stored(&weight_key, self.compute);
        if bias {
            self.weights
                .record_stored(&format!("{prefix}.bias"), self.compute);
        }
        let weight = self.weights.held_as(&weight_key, stored, self.compute)?;
        let bias = if bias {
            Some(
                self.weights
                    .take_as(&format!("{prefix}.bias"), self.compute)?,
            )
        } else {
            None
        };
        Ok(Linear {
            base: Base::Dense { weight, bias },
            residuals: self.prepare(residuals)?,
        })
    }

    fn prepare(&self, residuals: Vec<Residual>) -> Result<Vec<Residual>> {
        residuals
            .into_iter()
            .map(|r| r.prepare(self.compute, &self.weights.device))
            .collect()
    }

    /// A projection packed at load (sc-26266): the checkpoint's weight — read from the host mmap
    /// ([`Checkpoint::take_host`]), never staged on the device — zero-padded to `padded_in`
    /// columns, is quantized on the host to `dtype` straight from its stored (f32) values and
    /// placed on the checkpoint's device — the compute-dtype copy is never built. The
    /// bias stays in the compute dtype.
    fn quantized_linear(
        &self,
        prefix: &str,
        weight: Tensor,
        bias: bool,
        residuals: Vec<Residual>,
        dtype: GgmlDType,
        padded_in: usize,
    ) -> Result<Linear> {
        let (out, in_features) = weight.dims2()?;
        let mut host = weight.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        drop(weight);
        if padded_in > in_features {
            host = host.pad_with_zeros(1, 0, padded_in - in_features)?;
        }
        // Candle's quantizer reads the source's backing storage from its start, so hand it owned,
        // zero-offset storage.
        let host = host.force_contiguous()?;
        let packed = QTensor::quantize_onto(&host, dtype, &self.weights.device)?;
        drop(host);
        self.weights.hold(packed.storage_size_in_bytes() as u64);
        let bias = if bias {
            let key = format!("{prefix}.bias");
            self.weights.record_stored(&key, self.compute);
            Some(self.weights.take_as(&key, self.compute)?)
        } else {
            None
        };
        Ok(Linear {
            base: Base::Quantized {
                weight: QLinear::from_qtensor_dequant(std::sync::Arc::new(packed), bias),
                compute: self.compute,
                out,
                in_features,
                padded_in,
            },
            residuals: self.prepare(residuals)?,
        })
    }

    pub fn norm(&self, prefix: &str, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: self.f32(&format!("{prefix}.weight"))?,
            eps,
        })
    }
}

/// A forward-time adapter residual on one projection (sc-25686): `y = base(x) + r(x)`, the residual
/// narrowed to the projection's output dtype — never merged into the (bf16) base weight, so a delta
/// below the base's bf16 ulp still reaches the forward and its gradient, and the trainer's forward
/// is the provider's. The MLX twin's `AdaptableLinear` residuals:
///
/// * a LoRA runs `(x · Aᵀ) · (Bᵀ · s)` with f32 factors (the compute-dtype input promoted to f32, as
///   MLX promotes a bf16 × f32 product);
/// * a LoKr runs the structured Kronecker product `Y = w1 · X · w2ᵀ` over `X = reshape(x, [.., c, d])`
///   ([`LokrFactors`], the MLX twin's `LokrStructured`) — two GEMMs over the small factor shapes,
///   never an `[out, in]` delta, so it costs a LoRA's memory, not a second base weight;
/// * a LyCORIS LoHa `s · (A₁B₁) ⊙ (A₂B₂)` (ranks `r₁`, `r₂`) runs as its exact rank-`r₁r₂`
///   (Khatri–Rao) factorisation `(x · V) · U` when that is smaller than the delta,
///   `r₁r₂·(in + out) < in·out` ([`Residual::LowRank`]); above that rank the factorisation costs more
///   than the dense delta in both memory and GEMM work, so the `[out, in]` delta is held in the compute
///   dtype and run as `x · δᵀ` ([`Residual::Delta`], one extra base-sized weight and GEMM per adapted
///   projection — the MLX twin materialises every LoHa this way). A LoKr whose factors do not
///   decompose its projection's `[out, in]` falls back to the same dense delta.
#[derive(Clone, Debug)]
pub enum Residual {
    /// `a = Aᵀ` `[in, rank]`, `b = Bᵀ · (alpha/rank) · scale` `[rank, out]`, both f32.
    Lora { a: Tensor, b: Tensor },
    /// The structured LoKr: the Kronecker factors with the full scale baked into `w2`, f32.
    Kron(LokrFactors),
    /// A LoHa's exact low-rank factorisation: `a = V` `[in, r₁r₂]`, `b = U · scale` `[r₁r₂, out]`
    /// (f32 as built; held in the compute dtype once installed).
    LowRank { a: Tensor, b: Tensor },
    /// The `[out, in]` delta (f32 as reconstructed; held in the compute dtype once installed).
    Delta(Tensor),
}

impl Residual {
    /// The f32 `[out, in]` weight delta this residual adds — the merged view (the forward never
    /// forms it for a LoRA, a structured LoKr or a factored LoHa).
    pub fn delta(&self) -> Result<Tensor> {
        Ok(match self {
            Residual::Lora { a, b } | Residual::LowRank { a, b } => b
                .to_dtype(DType::F32)?
                .t()?
                .matmul(&a.to_dtype(DType::F32)?.t()?)?,
            Residual::Kron(f) => f.dense_delta()?,
            Residual::Delta(d) => d.to_dtype(DType::F32)?,
        })
    }

    /// The `[out, in]` shape this residual adapts, and whether its inner dimensions agree.
    pub fn shape(&self) -> Result<((usize, usize), bool)> {
        Ok(match self {
            Residual::Lora { a, b } | Residual::LowRank { a, b } => {
                ((b.dim(1)?, a.dim(0)?), a.dim(1)? == b.dim(0)?)
            }
            Residual::Kron(f) => (f.shape(), true),
            Residual::Delta(d) => (d.dims2()?, true),
        })
    }

    /// The residual as the forward holds it: LoRA and LoKr factors f32 on `device` (a LoKr's GEMMs
    /// cast them to the activation dtype); a LoHa factorisation or a delta in `compute`.
    fn prepare(self, compute: DType, device: &Device) -> Result<Self> {
        Ok(match self {
            Residual::Lora { a, b } => Residual::Lora {
                a: to(&a.to_device(device)?, DType::F32)?,
                b: to(&b.to_device(device)?, DType::F32)?,
            },
            Residual::Kron(f) => Residual::Kron(f.to_device(device)?),
            Residual::LowRank { a, b } => Residual::LowRank {
                a: to(&a.to_device(device)?, compute)?,
                b: to(&b.to_device(device)?, compute)?,
            },
            Residual::Delta(d) => Residual::Delta(to(&d.to_device(device)?, compute)?),
        })
    }

    /// `r(x)` for a flattened `[rows, in]` input in the compute dtype. A half-precision GEMM on the
    /// Candle CPU backend runs as one does elsewhere — f32 accumulation, each product rounded once.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gemm = gemm_dtype(x);
        Ok(match self {
            Residual::Lora { a, b } => to(x, DType::F32)?.matmul(a)?.matmul(b)?,
            Residual::Kron(f) => to(&f.residual(&to(x, gemm)?)?, x.dtype())?,
            Residual::LowRank { a, b } => {
                let h = to(&to(x, gemm)?.matmul(&to(a, gemm)?)?, a.dtype())?;
                to(&to(&h, gemm)?.matmul(&to(b, gemm)?)?, b.dtype())?
            }
            Residual::Delta(d) => {
                let gemm = gemm_dtype(d);
                let y = to(x, gemm)?.matmul(&to(d, gemm)?.t()?)?;
                to(&y, d.dtype())?
            }
        })
    }
}

/// `nn.Linear` (`[out, in]` weight) plus any forward-time adapter residuals.
pub struct Linear {
    base: Base,
    residuals: Vec<Residual>,
}

/// The base projection of a [`Linear`].
enum Base {
    /// `[out, in]` weight and optional `[out]` bias in the compute dtype.
    Dense {
        weight: Tensor,
        bias: Option<Tensor>,
    },
    /// A GGML-packed weight (sc-26266), dequantized to the GEMM dtype per forward and multiplied
    /// densely — the activation is never quantized (the shared `DequantDense` forward, sc-7702).
    /// Its rows may be zero-padded past `in_features` ([`ProjectionLayout::Quantized`]).
    Quantized {
        weight: QLinear,
        compute: DType,
        out: usize,
        in_features: usize,
        padded_in: usize,
    },
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = self.compute_dtype();
        let x = to(x, dtype)?;
        let dims = x.dims().to_vec();
        let (last, lead) = dims
            .split_last()
            .expect("linear input has at least one axis");
        let rows: usize = lead.iter().product();
        let (out_dim, _) = self.logical_shape()?;
        let flat = x.reshape((rows, *last))?;
        // Candle's CPU backend has no half-precision GEMM; there the bf16 policy is run as what a
        // bf16 GEMM is — bf16 operands, f32 accumulation, one rounding of the result.
        let gemm = kernel_dtype(x.device(), dtype);
        let y = match &self.base {
            Base::Dense { weight, bias } => {
                let mut y =
                    __model_proof_output!(to(&flat, gemm)?.matmul(&to(weight, gemm)?.t()?))?;
                if let Some(b) = bias {
                    y = __model_proof_output!(y.broadcast_add(&to(b, gemm)?))?;
                }
                y
            }
            Base::Quantized {
                weight,
                in_features,
                padded_in,
                ..
            } => {
                let mut input = to(&flat, gemm)?;
                if padded_in > in_features {
                    input = input.pad_with_zeros(1, 0, padded_in - in_features)?;
                }
                __model_proof_output!(weight.forward(&input))?
            }
        };
        let mut y = to(&y, dtype)?;
        for r in &self.residuals {
            y = y.add(&to(&r.forward(&flat)?, dtype)?)?;
        }
        let mut shape = lead.to_vec();
        shape.push(out_dim);
        Ok(y.reshape(shape)?)
    }

    /// The dtype the projection returns (its compute dtype).
    fn compute_dtype(&self) -> DType {
        match &self.base {
            Base::Dense { weight, .. } => weight.dtype(),
            Base::Quantized { compute, .. } => *compute,
        }
    }

    /// The checkpoint's `[out, in]`, whatever the projection is held as.
    pub fn logical_shape(&self) -> Result<(usize, usize)> {
        match &self.base {
            Base::Dense { weight, .. } => Ok(weight.dims2()?),
            Base::Quantized {
                out, in_features, ..
            } => Ok((*out, *in_features)),
        }
    }

    /// The dense `[out, in]` weight, or `None` for a packed projection.
    pub fn dense_weight(&self) -> Option<&Tensor> {
        match &self.base {
            Base::Dense { weight, .. } => Some(weight),
            Base::Quantized { .. } => None,
        }
    }

    /// The GGML block type of a packed projection (`None` when dense).
    pub fn quant_dtype(&self) -> Option<GgmlDType> {
        match &self.base {
            Base::Dense { .. } => None,
            Base::Quantized { weight, .. } => weight.quant_dtype(),
        }
    }

    /// A linear layer from an already-shaped `[out, in]` weight (and optional `[out]` bias), stored
    /// as given — e.g. the depth task's 1×1-conv reducer, whose `[1, C, 1, 1]` kernel is reshaped.
    pub fn from_parts(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            base: Base::Dense { weight, bias },
            residuals: Vec::new(),
        }
    }
}

/// The dtype a GEMM over `t`'s dtype runs in on `t`'s device: itself, except a half-precision
/// operand on the Candle CPU backend (which has no bf16/f16 matmul kernel) accumulates in f32 and
/// is rounded back by the caller.
fn gemm_dtype(t: &Tensor) -> DType {
    kernel_dtype(t.device(), t.dtype())
}

/// See [`gemm_dtype`].
fn kernel_dtype(device: &Device, dtype: DType) -> DType {
    if device.is_cpu() && matches!(dtype, DType::BF16 | DType::F16) {
        DType::F32
    } else {
        dtype
    }
}

/// `iris3b.nn.norms.RMSNorm`: f32 normalisation, f32 gain.
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = to(x, DType::F32)?.contiguous()?;
        if x.track_op() || self.weight.track_op() || composable_forced() {
            // Under autograd (training, sc-25686): candle's fused `rms_norm` is a CustomOp with no
            // backward, which would silently zero every upstream gradient — the same f32 math in
            // composable ops instead.
            // One tape node (sc-25690): the chain reads `x` twice — the quotient, which the plain
            // backward reaches first, then the mean square.
            let eps = self.eps as f64;
            return Ok(__model_proof_output!(crate::tape::fold3(
                "iris-rms-norm",
                &x,
                &x,
                &self.weight,
                move |x, squared, gain| {
                    let dim = squared.dim(D::Minus1)? as f64;
                    let ms = (squared.sqr()?.sum_keepdim(D::Minus1)? / dim)?;
                    let rms = (ms + eps)?.sqrt()?;
                    x.broadcast_div(&rms)?.broadcast_mul(gain)
                },
            ))?);
        }
        __model_proof_output!(Ok(candle_gen::candle_nn::ops::rms_norm(
            &x,
            &self.weight,
            self.eps,
        )?))
    }
}

thread_local! {
    static FORCE_COMPOSABLE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the composable (autograd) forms of the fused RMSNorm / softmax are forced on this thread.
pub fn composable_forced() -> bool {
    FORCE_COMPOSABLE.with(|c| c.get())
}

/// Run `f` with the composable forms of the fused kernels forced even on untracked tensors (sc-25689):
/// an activation-checkpointing region's tape-free forward must compute exactly what its tracked
/// recompute computes.
pub fn with_composable<T>(f: impl FnOnce() -> T) -> T {
    let prev = FORCE_COMPOSABLE.with(|c| c.replace(true));
    let out = f();
    FORCE_COMPOSABLE.with(|c| c.set(prev));
    out
}

/// Rotation tables `(cos, sin)`, `[n, pairs]` f32, for [`Rope::apply`].
#[derive(Clone)]
pub struct Rope {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl Rope {
    fn from_angles(angles: &[f32], n: usize, pairs: usize, device: &Device) -> Result<Self> {
        let cos: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
        Ok(Self {
            cos: Tensor::from_vec(cos, (n, pairs), device)?,
            sin: Tensor::from_vec(sin, (n, pairs), device)?,
        })
    }

    /// `rope_2d(head_dim, height, width, theta, scale, aspect="isotropic", frame_pairs=0)`:
    /// `head_dim/4` frequencies, interleaved (x, y) per pair along the last axis, row-major grid,
    /// both axes at one step `scale / (max(h, w) − 1)`.
    pub fn grid_2d(
        head_dim: usize,
        height: usize,
        width: usize,
        theta: f32,
        scale: f32,
        device: &Device,
    ) -> Result<Self> {
        let angles = iris::rope_2d_angles(head_dim, height, width, theta, scale);
        Self::from_angles(&angles, height * width, 2 * (head_dim / 4), device)
    }

    /// [`Self::grid_2d`] under `model.rope_aspect` (`square = true`: each axis normalized to
    /// `[0, scale]` on its own — `gen_core_iris::rope_2d_angles_aspect`).
    pub fn grid_2d_aspect(
        head_dim: usize,
        height: usize,
        width: usize,
        theta: f32,
        scale: f32,
        square: bool,
        device: &Device,
    ) -> Result<Self> {
        let angles = iris::rope_2d_angles_aspect(head_dim, height, width, theta, scale, square);
        Self::from_angles(&angles, height * width, 2 * (head_dim / 4), device)
    }

    /// `rope_1d(head_dim, length, theta)`: standard 1-D RoPE over integer positions.
    pub fn line_1d(head_dim: usize, length: usize, theta: f32, device: &Device) -> Result<Self> {
        let angles = iris::rope_1d_angles(head_dim, length, theta);
        Self::from_angles(&angles, length, head_dim / 2, device)
    }

    /// Rotate `x` `[B, N, H, head_dim]` by complex multiplication over **adjacent** channel pairs
    /// (`view_as_complex`), in f32; returns `x`'s dtype.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, h, d) = x.dims4()?;
        let pairs = d / 2;
        let xf = to(x, DType::F32)?.reshape((b, n, h, pairs, 2))?;
        let re = xf.narrow(4, 0, 1)?.squeeze(4)?;
        let im = xf.narrow(4, 1, 1)?.squeeze(4)?;
        let cos = self.cos.reshape((1, n, 1, pairs))?;
        let sin = self.sin.reshape((1, n, 1, pairs))?;
        let out_re = re.broadcast_mul(&cos)?.sub(&im.broadcast_mul(&sin)?)?;
        let out_im = re.broadcast_mul(&sin)?.add(&im.broadcast_mul(&cos)?)?;
        let out = Tensor::stack(&[out_re, out_im], 4)?.reshape((b, n, h, d))?;
        to(&out, x.dtype())
    }
}

/// Grouped-query SDPA over `[B, N, heads, head_dim]` q and `[B, N, kv_heads, head_dim]` k/v, in the
/// compute dtype; returns `[B, N, heads·head_dim]`. `mask` is additive `[B|1, 1, N, N]` f32. The
/// scores run through the shared i32-overflow-safe query-row chunking (a 2048² render's trunk
/// attention is ~11e9 score elements under CFG).
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    compute: DType,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, n, h, d) = q.dims4()?;
    let kv = k.dim(2)?;
    // Operands are rounded to the compute dtype; on the CPU backend (no half GEMM) a half compute
    // dtype then runs the kernel in f32 and rounds its output back, as in `Linear::forward`.
    let kernel = kernel_dtype(q.device(), compute);
    let t = |a: &Tensor| -> Result<Tensor> {
        Ok(to(&to(a, compute)?, kernel)?
            .transpose(1, 2)?
            .contiguous()?)
    };
    let q = t(q)?;
    let k = repeat_kv(&t(k)?, h / kv)?;
    let v = repeat_kv(&t(v)?, h / kv)?;
    let mask = mask.map(|m| to(m, kernel)).transpose()?;
    let scale = 1.0 / (d as f64).sqrt();
    // Under autograd (training) the fused `softmax_last_dim` (no backward) is replaced by the
    // composable softmax over the same axis.
    let differentiable = q.track_op() || k.track_op() || v.track_op() || composable_forced();
    let scope = attention_scope();
    let out = if differentiable {
        // One tape node (sc-25690): the scores, their scaled / masked forms and the softmax's own
        // intermediates are `[B, heads, N, N]` each and none is an operand of anything else. The
        // node wraps whichever kernel the request scope selects, so its chunks (query rows, and
        // batch groups under a bounded scope) are all inside it: q, k and v keep one reader.
        let cancel = scope.as_ref().map(|s| s.cancel.clone());
        let folded = crate::tape::fold3("iris-sdpa", &q, &k, &v, move |q, k, v| {
            let composable = |s: &Tensor| softmax(s, D::Minus1);
            attention_kernel(q, k, v, scale, mask.as_ref(), &composable, scope.as_ref())
                .map_err(candle_gen::candle_core::Error::wrap)
        });
        match folded {
            Ok(out) => out,
            // the fold carries candle errors only: a tripped cancel flag stays a cancellation
            Err(_) if cancel.is_some_and(|c| c.is_cancelled()) => return Err(Error::Canceled),
            Err(e) => return Err(e.into()),
        }
    } else {
        attention_kernel(
            &q,
            &k,
            &v,
            scale,
            mask.as_ref(),
            &softmax_last_dim,
            scope.as_ref(),
        )?
    };
    let out = to(&out, compute)?;
    Ok(out.transpose(1, 2)?.reshape((b, n, h * d))?)
}

/// [`attention`]'s kernel: the i32-overflow guard alone when no request selected the
/// bounded-attention rung (exactly the pre-rung forward), else [`bounded_attention`].
fn attention_kernel(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    softmax: &dyn Fn(&Tensor) -> candle_gen::candle_core::Result<Tensor>,
    scope: Option<&AttentionScope>,
) -> Result<Tensor> {
    match scope {
        None => Ok(candle_gen::sdpa_budgeted_bhsd(
            q,
            k,
            v,
            scale,
            mask,
            softmax,
            candle_gen::ATTN_SCORES_BUDGET,
        )?),
        Some(scope) => bounded_attention(q, k, v, scale, mask, softmax, scope),
    }
}

/// A request's admitted bounded-attention plan (memory-ladder rung 3): the largest scores tensor one
/// attention kernel launch may build, and the request's cancel flag (checked between chunks).
#[derive(Clone)]
pub struct AttentionScope {
    budget: usize,
    cancel: CancelFlag,
}

thread_local! {
    static ATTENTION_SCOPE: RefCell<Option<AttentionScope>> = const { RefCell::new(None) };
}

fn attention_scope() -> Option<AttentionScope> {
    ATTENTION_SCOPE.with(|scope| scope.borrow().clone())
}

/// Run `f` with every [`attention`] on this thread bounded to `budget` score elements per kernel
/// launch (`None` leaves the forward untouched). The budget is capped at the i32-overflow guard
/// ([`candle_gen::ATTN_SCORES_BUDGET`]), which a memory selection may tighten but never loosen.
/// The previous scope is restored on every exit, including an unwinding one.
pub fn with_attention_budget<T>(
    budget: Option<usize>,
    cancel: &CancelFlag,
    f: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<AttentionScope>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ATTENTION_SCOPE.with(|scope| *scope.borrow_mut() = self.0.take());
        }
    }
    let Some(budget) = budget else {
        return f();
    };
    let scope = AttentionScope {
        budget: budget.min(candle_gen::ATTN_SCORES_BUDGET),
        cancel: cancel.clone(),
    };
    let _restore = Restore(ATTENTION_SCOPE.with(|slot| slot.borrow_mut().replace(scope)));
    f()
}

/// How many batch rows one bounded kernel launch may take: the most whose **single query row**
/// (`rows · heads · keys` score elements) fits `budget`, never fewer than one. The shared planner
/// chunks the query axis only, and the PiT pixel stage folds every patch into the batch axis (a
/// 1024² CFG pass is 8192 rows of 256 keys), so without this split one query row of that stage
/// alone would exceed the budget the request admitted.
pub fn attention_batch_group(batch: usize, heads: usize, keys: usize, budget: usize) -> usize {
    (budget / (heads * keys).max(1)).clamp(1, batch.max(1))
}

/// [`attention`]'s kernel under an [`AttentionScope`]: batch groups of
/// [`attention_batch_group`] rows, each chunked over its query rows by the shared rung-3 planner
/// ([`candle_gen::sdpa_planned_bhsd`]). Rows and query rows are independent, so the result equals
/// the single pass to GEMM-order rounding.
fn bounded_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    softmax: &dyn Fn(&Tensor) -> candle_gen::candle_core::Result<Tensor>,
    scope: &AttentionScope,
) -> Result<Tensor> {
    let (b, h, _, _) = q.dims4()?;
    let sk = k.dim(2)?;
    let plan = AttentionPlan::budgeted(candle_gen::attention::attention_budget_from_usize(
        scope.budget,
    ))
    .with_cancel(&scope.cancel);
    let group = attention_batch_group(b, h, sk, scope.budget);
    if group >= b {
        return candle_gen::sdpa_planned_bhsd(q, k, v, scale, mask, softmax, plan);
    }
    let mut parts = Vec::with_capacity(b.div_ceil(group));
    let mut start = 0;
    while start < b {
        if scope.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let len = group.min(b - start);
        // A per-row mask follows its rows; a broadcast `[1, …]` mask serves every group.
        let mask = match mask {
            Some(m) if m.dim(0)? == b => Some(m.narrow(0, start, len)?),
            other => other.cloned(),
        };
        parts.push(candle_gen::sdpa_planned_bhsd(
            &q.narrow(0, start, len)?,
            &k.narrow(0, start, len)?,
            &v.narrow(0, start, len)?,
            scale,
            mask.as_ref(),
            softmax,
            plan,
        )?);
        start += len;
    }
    Ok(Tensor::cat(&parts, 0)?)
}

/// `TransformerTextEmbedder`'s key mask: real keys OR the diagonal, additive `[B, 1, T, T]` f32.
pub fn key_padding_mask(mask: &[Vec<i32>], tokens: usize, device: &Device) -> Result<Tensor> {
    let data = iris::key_padding_additive(mask, tokens);
    Ok(Tensor::from_vec(
        data,
        (mask.len(), 1, tokens, tokens),
        device,
    )?)
}

/// Shape check with a named error (checkpoint/config disagreement surfaces at load, by key).
/// [`expect_shape`] over an `[out, in]` already in hand (a projection's
/// [`Linear::logical_shape`]).
pub fn expect_dims(name: &str, have: (usize, usize), want: (usize, usize)) -> Result<()> {
    if have != want {
        return Err(Error::Msg(format!(
            "iris: {name} has shape {have:?}, the config implies {want:?}"
        )));
    }
    Ok(())
}

pub fn expect_shape(name: &str, a: &Tensor, want: &[usize]) -> Result<()> {
    if a.dims() != want {
        return Err(Error::Msg(format!(
            "iris: {name} has shape {:?}, the config implies {want:?}",
            a.dims()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_follows_torch() {
        let dev = Device::Cpu;
        let a = Tensor::new(&[1f32, 2.0], &dev).unwrap();
        let b = a.to_dtype(DType::BF16).unwrap();
        assert_eq!(add(&a, &b).unwrap().dtype(), DType::F32);
        assert_eq!(add(&b, &b).unwrap().dtype(), DType::BF16);
        assert_eq!(cat(&[&b, &a], 0).unwrap().dtype(), DType::F32);
    }

    #[test]
    fn gqa_repeat_is_interleaved_per_kv_head() {
        let dev = Device::Cpu;
        // kv heads hold 0 and 1; with rep 2 the four query heads read 0,0,1,1.
        let k = Tensor::new(&[0f32, 1.0], &dev)
            .unwrap()
            .reshape((1, 2, 1, 1))
            .unwrap();
        let r = repeat_kv(&k, 2).unwrap().flatten_all().unwrap();
        assert_eq!(r.to_vec1::<f32>().unwrap(), [0.0, 0.0, 1.0, 1.0]);
    }

    /// Deterministic, non-symmetric values in `[-1, 1)`.
    fn values(shape: (usize, usize), seed: u32) -> Tensor {
        let n = shape.0 * shape.1;
        let data: Vec<f32> = (0..n as u32)
            .map(|i| (((i * 7919 + seed * 104_729) % 1000) as f32) / 500.0 - 1.0)
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    /// The largest absolute difference between `r(x)` (installed at f32) and `x · δᵀ`.
    fn residual_error(r: &Residual, delta: &Tensor, rows: usize) -> f32 {
        let (_, in_f) = delta.dims2().unwrap();
        let x = values((rows, in_f), 11);
        let got = r
            .clone()
            .prepare(DType::F32, &Device::Cpu)
            .unwrap()
            .forward(&x)
            .unwrap();
        let want = x.matmul(&delta.t().unwrap()).unwrap();
        (got - want)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn lokr_runs_the_structured_kronecker_product() {
        use candle_gen::train::lora::reconstruct_lokr_delta;
        // w1 [2, 3] ⊗ w2_a·w2_b [4, 5] → [8, 15].
        let (w1, w2_a, w2_b) = (values((2, 3), 1), values((4, 2), 2), values((2, 5), 3));
        let lokr = |shape| {
            crate::adapters::lokr_residual(
                Some(&w1),
                None,
                None,
                None,
                Some(&w2_a),
                Some(&w2_b),
                4.0,
                2.0,
                0.5,
                shape,
            )
            .unwrap()
        };
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            None,
            Some(&w2_a),
            Some(&w2_b),
            4.0,
            2.0,
            0.5,
            (8, 15),
        )
        .unwrap();
        let r = lokr((8, 15));
        assert!(matches!(r, Residual::Kron(_)), "{r:?}");
        assert_eq!(r.shape().unwrap(), ((8, 15), true));
        let merged = max_diff(&r.delta().unwrap(), &delta);
        assert!(
            merged < 1e-6,
            "merged view off the Kronecker delta by {merged}"
        );
        let err = residual_error(&r, &delta, 6);
        assert!(err < 1e-5, "structured residual off x·δᵀ by {err}");
        // Factors that do not decompose the projection fall back to the reconstructed delta.
        let r = lokr((4, 30));
        assert!(matches!(r, Residual::Delta(_)), "{r:?}");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED nn::tests::lokr_runs_the_structured_kronecker_product");
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn loha_runs_its_low_rank_factorisation_below_the_break_even_rank() {
        use candle_gen::train::merge::ThirdPartyLoha;
        let loha = |rank: usize, out: usize, inp: usize| ThirdPartyLoha {
            w1_a: Some(values((out, rank), 4)),
            w1_b: Some(values((rank, inp), 5)),
            w2_a: Some(values((out, rank), 6)),
            w2_b: Some(values((rank, inp), 7)),
            alpha: Some(3.0),
        };
        // r² = 4: 4·(24 + 16) = 160 < 24·16 = 384 → factorised.
        let g = loha(2, 24, 16);
        let r = crate::adapters::loha_residual(&g, (24, 16), 0.75).unwrap();
        assert!(matches!(r, Residual::LowRank { .. }), "{r:?}");
        assert_eq!(r.shape().unwrap(), ((24, 16), true));
        let delta = g.delta((24, 16), 0.75).unwrap();
        let merged = max_diff(&r.delta().unwrap(), &delta);
        assert!(
            merged < 1e-5,
            "merged view off the Hadamard delta by {merged}"
        );
        let err = residual_error(&r, &delta, 5);
        assert!(err < 1e-4, "factorised residual off x·δᵀ by {err}");
        // r² = 16: 16·(24 + 16) = 640 ≥ 384 → the dense delta is cheaper.
        let g = loha(4, 24, 16);
        let r = crate::adapters::loha_residual(&g, (24, 16), 0.75).unwrap();
        assert!(matches!(r, Residual::Delta(_)), "{r:?}");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED nn::tests::loha_runs_its_low_rank_factorisation_below_the_break_even_rank");
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let dev = Device::Cpu;
        let rope = Rope::line_1d(4, 1, 10_000.0, &dev).unwrap();
        let x = Tensor::new(&[1f32, 2.0, 3.0, 4.0], &dev)
            .unwrap()
            .reshape((1, 1, 1, 4))
            .unwrap();
        let y = rope.apply(&x).unwrap().flatten_all().unwrap();
        assert_eq!(y.to_vec1::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    }

    fn ramp(shape: (usize, usize, usize, usize), phase: f32) -> Tensor {
        let n = shape.0 * shape.1 * shape.2 * shape.3;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.37 + phase).sin()) * 0.5)
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    /// sc-25690, rung 3: under an admitted budget no kernel launch builds a scores tensor larger
    /// than the budget — including when ONE query row of the whole batch already exceeds it, the
    /// PiT pixel stage's shape (every patch is a batch row) — and the result is the single pass.
    ///
    /// The softmax closure is handed the real scores tensor of each launch, so its element count
    /// is observed, not derived.
    ///
    /// *Mutation that reds this:* `let group = b;` in `bounded_attention` (drop the batch split):
    /// the 24-element budget then launches 4·2·8 = 64-element rows.
    #[test]
    fn a_bounded_attention_never_launches_scores_over_its_budget() {
        let (b, h, s, d) = (4, 2, 8, 4);
        let q = ramp((b, h, s, d), 0.1);
        let k = ramp((b, h, s, d), 0.7);
        let v = ramp((b, h, s, d), 1.3);
        let scale = 1.0 / (d as f64).sqrt();
        let single = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            None,
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )
        .unwrap();
        // 512 = the whole scores tensor (no chunking), 128 = one batch row, 24 = less than one
        // query row of the batch (4·2·8 = 64), 16 = exactly one query row of one batch row.
        for budget in [512usize, 128, 24, 16] {
            let largest = std::cell::Cell::new(0usize);
            let launches = std::cell::Cell::new(0usize);
            let recording = |scores: &Tensor| {
                largest.set(largest.get().max(scores.elem_count()));
                launches.set(launches.get() + 1);
                softmax_last_dim(scores)
            };
            let scope = AttentionScope {
                budget,
                cancel: CancelFlag::new(),
            };
            let out = bounded_attention(&q, &k, &v, scale, None, &recording, &scope).unwrap();
            assert!(
                largest.get() <= budget,
                "budget {budget}: a launch built {} score elements",
                largest.get()
            );
            assert_eq!(launches.get() > 1, budget < 512, "budget {budget}");
            let diff = (&out - &single)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                diff < 1e-5,
                "budget {budget}: off the single pass by {diff}"
            );
        }
        assert_eq!(attention_batch_group(4, 2, 8, 24), 1);
        assert_eq!(attention_batch_group(4, 2, 8, 40), 2);
        assert_eq!(attention_batch_group(4, 2, 8, 512), 4);
        // Never zero rows, even below one query row of one batch row.
        assert_eq!(attention_batch_group(4, 2, 8, 3), 1);
    }

    /// A per-row mask follows its rows through the batch split; a tripped cancel flag is the typed
    /// cancellation, checked between launches.
    #[test]
    fn a_bounded_attention_narrows_a_per_row_mask_and_honours_cancel() {
        let (b, h, s, d) = (3, 2, 4, 4);
        let q = ramp((b, h, s, d), 0.2);
        let k = ramp((b, h, s, d), 0.9);
        let v = ramp((b, h, s, d), 1.7);
        // Row 1 masks its last two keys; rows 0 and 2 mask nothing.
        let mut mask = vec![0f32; b * s * s];
        for query in 0..s {
            for key in 2..s {
                mask[(s + query) * s + key] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(mask, (b, 1, s, s), &Device::Cpu).unwrap();
        let scale = 0.5;
        let single = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            Some(&mask),
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )
        .unwrap();
        let scope = AttentionScope {
            budget: 8,
            cancel: CancelFlag::new(),
        };
        let out =
            bounded_attention(&q, &k, &v, scale, Some(&mask), &softmax_last_dim, &scope).unwrap();
        let diff = (&out - &single)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-5, "masked rows off the single pass by {diff}");

        scope.cancel.cancel();
        assert!(matches!(
            bounded_attention(&q, &k, &v, scale, Some(&mask), &softmax_last_dim, &scope),
            Err(Error::Canceled)
        ));
    }

    /// The scope bounds only the closure it wraps, tightens but never loosens the i32-overflow
    /// guard, and `None` installs nothing.
    #[test]
    fn the_attention_scope_is_restored_on_exit_and_capped_at_the_overflow_guard() {
        let cancel = CancelFlag::new();
        assert!(attention_scope().is_none());
        with_attention_budget(None, &cancel, || assert!(attention_scope().is_none()));
        with_attention_budget(Some(64), &cancel, || {
            assert_eq!(attention_scope().unwrap().budget, 64);
            with_attention_budget(Some(usize::MAX), &cancel, || {
                assert_eq!(
                    attention_scope().unwrap().budget,
                    candle_gen::ATTN_SCORES_BUDGET
                );
            });
            assert_eq!(
                attention_scope().unwrap().budget,
                64,
                "the outer scope returns"
            );
        });
        assert!(attention_scope().is_none());
    }
}
