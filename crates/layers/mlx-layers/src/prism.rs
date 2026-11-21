//! Prism ternary affine operators for Ternary Bonsai checkpoints (`SW-6957` / `SW-F335`).
//!
//! The packed MLX artifact stores 16 two-bit affine codes per `u32`, with one scale and bias for
//! each 128 input values. Bonsai folds a normalized blockwise Walsh-Hadamard rotation into every
//! packed matrix: linear activations apply `signs` then `H`, while embedding rows apply the inverse
//! `H` then `signs`. These operators keep the packed weights resident and use MLX's native
//! two-bit matmul/dequantize kernels; they never materialize a full dense model. Each rotation —
//! cast, sign, blockwise Hadamard, cast — is one fused Metal kernel per projection input
//! (sc-24444), bit-identical to the op chain it replaces.

use std::cell::Cell;
use std::thread::LocalKey;

use core_llm::defaults::MLX;
use core_llm::switch::ProcessSwitch;
use mlx_rs::fast::MetalKernel;
use mlx_rs::ops::{dequantize, multiply, quantized_matmul};
use mlx_rs::{Array, Dtype, Stream};

use crate::llm_projection::{Error, Result};

/// Environment variable of [`FUSED_ROTATION`].
pub const FUSED_ROTATION_ENV: &str = "MLX_LLM_FUSED_ROTATION";

/// A set variable means **on** unless it is one of the off words.
pub fn env_value_enables(v: &str) -> bool {
    !matches!(v, "0" | "off" | "false" | "no")
}

/// One MLX optimization switch: a thread-scoped override ([`scoped`](Self::scoped)) over the
/// process switch ([`process`](Self::process): runtime override, environment, defaults table).
/// The thread layer lets a test (or a bench comparing two paths) flip the optimization for its
/// own generation without changing it under every other thread of the process.
pub struct MlxSwitch {
    process: ProcessSwitch,
    thread: &'static LocalKey<Cell<Option<bool>>>,
}

impl MlxSwitch {
    /// Construct an MLX process-and-thread switch.
    pub const fn new(
        env: &'static str,
        default: bool,
        thread: &'static LocalKey<Cell<Option<bool>>>,
    ) -> Self {
        Self {
            process: ProcessSwitch::new(env, default, env_value_enables),
            thread,
        }
    }

    /// Whether the optimization is on for the current thread: its [`scoped`](Self::scoped)
    /// override if one is active, else the process switch.
    pub fn enabled(&self) -> bool {
        self.thread
            .with(Cell::get)
            .unwrap_or_else(|| self.process.enabled())
    }

    /// The process-wide layer: [`ProcessSwitch::set`] / [`ProcessSwitch::guard`] change it for
    /// every thread; its environment variable is [`ProcessSwitch::env`].
    pub fn process(&self) -> &ProcessSwitch {
        &self.process
    }

    /// Run `f` with the optimization forced `enabled` on this thread only, restoring the previous
    /// thread setting afterwards (also on unwind).
    pub fn scoped<R>(&self, enabled: bool, f: impl FnOnce() -> R) -> R {
        struct Restore(&'static LocalKey<Cell<Option<bool>>>, Option<bool>);
        impl Drop for Restore {
            fn drop(&mut self) {
                self.0.with(|c| c.set(self.1));
            }
        }
        let _restore = Restore(self.thread, self.thread.with(|c| c.replace(Some(enabled))));
        f()
    }
}

thread_local! {
    static FUSED_ROTATION_THREAD: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Whether the Prism block-Hadamard rotation dispatches the fused Metal kernel (sc-24444).
pub static FUSED_ROTATION: MlxSwitch = MlxSwitch::new(
    FUSED_ROTATION_ENV,
    MLX.fused_rotation,
    &FUSED_ROTATION_THREAD,
);

/// Fused-versus-reference route tally (epic sc-24432 E3): the MLX primitives that have a fused
/// Metal route — the Prism Hadamard rotation ([`super`]) and the Gated DeltaNet recurrence
/// (`gated_delta`) — count every run by the route that served it, on this thread, so a
/// request reports `fused` / `reference` / `mixed` / `none` from what actually ran
/// ([`DecodeReport::fused_primitives`](core_llm::DecodeReport::fused_primitives)) — the same
/// shared [`fused::FusedTally`] Candle reports through (E8).
///
/// A route is decided when the op is built (MLX is lazy), on the thread building the request's
/// graph, which is the thread the provider and the engine run on. The tally is monotone; a request
/// reports [`fused::FusedTally::since`] its start.
pub mod fused {
    use std::cell::Cell;

    pub use core_llm::FusedTally;

    /// Why a fused route did not run: the op was built for a CPU stream (a custom Metal kernel runs
    /// only on the GPU).
    pub const REASON_CPU_STREAM: &str = "cpu_stream";
    /// Why a fused route did not run: the input's shape or dtype is outside the kernel's (a Prism
    /// rotation block that is not a power of two up to the fused maximum, an unsupported dtype, an
    /// empty input).
    pub const REASON_SHAPE: &str = "shape";
    /// Why a fused route did not run: a test forced the reference route (the parity oracle).
    pub const REASON_FORCED_REFERENCE: &str = "forced_reference";
    /// Why a fused route did not run: its process switch ([`super::FUSED_ROTATION`]) — the MLX row of the
    /// defaults table, the environment, or a runtime override — turned the kernel off (the label
    /// Candle uses for its own switch).
    pub const REASON_DISABLED: &str = "disabled";

    thread_local! {
        static TALLY: Cell<FusedTally> = const {
            Cell::new(FusedTally { fused: 0, reference: 0, reference_reason: None })
        };
    }

    /// This thread's monotone tally.
    pub fn fused_tally() -> FusedTally {
        TALLY.with(Cell::get)
    }

    /// Record one run served by a fused kernel.
    #[inline]
    pub fn note_fused() {
        TALLY.with(|c| {
            let mut t = c.get();
            t.fused = t.fused.wrapping_add(1);
            c.set(t);
        });
    }

    /// Record one run served by the reference route, with why.
    #[inline]
    pub fn note_reference(reason: &'static str) {
        TALLY.with(|c| {
            let mut t = c.get();
            t.reference = t.reference.wrapping_add(1);
            t.reference_reason = Some(reason);
            c.set(t);
        });
    }
}

/// Whether `stream` is a GPU stream — the only place a custom Metal kernel can run. Every fused
/// kernel resolves [`Stream::task_local_or_default`](mlx_rs::Stream::task_local_or_default) once,
/// gates on this, and dispatches on that same stream: the task-local stream need not be on the
/// process default device.
pub fn stream_is_gpu(stream: &mlx_rs::Stream) -> bool {
    // SAFETY: `dev` is created and freed here; `stream` outlives both calls.
    unsafe {
        let mut dev = mlx_sys::mlx_device_new();
        let mut ty: mlx_sys::mlx_device_type = mlx_sys::mlx_device_type__MLX_CPU;
        let ok = mlx_sys::mlx_stream_get_device(&mut dev, stream.as_ptr()) == 0
            && mlx_sys::mlx_device_get_type(&mut ty, dev) == 0;
        mlx_sys::mlx_device_free(dev);
        ok && ty == mlx_sys::mlx_device_type__MLX_GPU
    }
}

#[doc(hidden)]
pub const GROUP_SIZE: i32 = 128;
#[doc(hidden)]
pub const BITS: i32 = 2;

fn validate_parts(
    label: &str,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    signs: &Array,
    block: i32,
) -> Result<(i32, i32)> {
    let ws = weight.shape();
    let ss = scales.shape();
    let bs = biases.shape();
    if weight.dtype() != Dtype::Uint32 {
        return Err(Error::Config(format!(
            "Prism packed `{label}` weight must be U32, got {:?}",
            weight.dtype()
        )));
    }
    if ws.len() != 2 || ss.len() != 2 || bs != ss {
        return Err(Error::Config(format!(
            "Prism packed `{label}` has invalid part shapes: weight {ws:?}, scales {ss:?}, biases {bs:?}"
        )));
    }
    if !matches!(
        scales.dtype(),
        Dtype::Float16 | Dtype::Float32 | Dtype::Bfloat16
    ) || biases.dtype() != scales.dtype()
    {
        return Err(Error::Config(format!(
            "Prism packed `{label}` affine parameters must share an F16/F32/BF16 dtype"
        )));
    }
    let rows = ss[0];
    let width = ss[1]
        .checked_mul(GROUP_SIZE)
        .ok_or_else(|| Error::Config(format!("Prism packed `{label}` width overflow")))?;
    if rows <= 0
        || width <= 0
        || ws[0] != rows
        || ws[1] != width / 16
        || block <= 0
        || width % block != 0
        || signs.shape() != [width]
        || signs.dtype() != Dtype::Float32
    {
        return Err(Error::Config(format!(
            "Prism packed `{label}` geometry mismatch: weight {ws:?}, scales {ss:?}, signs {:?}, block {block}",
            signs.shape()
        )));
    }
    let sign_values = signs.as_slice::<f32>();
    if sign_values
        .iter()
        .any(|&value| value != -1.0 && value != 1.0)
    {
        return Err(Error::Config(format!(
            "Prism packed `{label}` signs must contain only -1 or +1"
        )));
    }
    let scale_values = scales.as_dtype(Dtype::Float32)?;
    let bias_values = biases.as_dtype(Dtype::Float32)?;
    if scale_values
        .as_slice::<f32>()
        .iter()
        .chain(bias_values.as_slice::<f32>())
        .any(|value| !value.is_finite())
    {
        return Err(Error::Config(format!(
            "Prism packed `{label}` affine parameters must be finite"
        )));
    }
    Ok((rows, width))
}

/// The unfused reference chain: cast to F32, apply `signs` (forward), the normalized blockwise
/// Hadamard transform, `signs` (inverse), and cast back. [`block_hadamard`] must reproduce this
/// bit for bit; it is kept as the fallback for shapes the fused kernel does not cover and as the
/// oracle its tests compare against.
#[doc(hidden)]
pub fn block_hadamard_unfused(
    x: &Array,
    signs: &Array,
    block: i32,
    inverse: bool,
    scale: f32,
) -> Result<Array> {
    let shape = x.shape().to_vec();
    let dtype = x.dtype();
    let mut transformed = x.as_dtype(Dtype::Float32)?;
    if !inverse {
        transformed = multiply(&transformed, signs)?;
    }
    transformed = transformed
        .reshape(&[-1, block])?
        .hadamard_transform(Some(scale))?
        .reshape(&shape)?;
    if inverse {
        transformed = multiply(&transformed, signs)?;
    }
    Ok(transformed.as_dtype(dtype)?)
}

/// The normalization MLX's `hadamard_transform` is given for one `block`.
#[doc(hidden)]
pub fn hadamard_scale(block: i32) -> f32 {
    1.0 / (block as f32).sqrt()
}

/// Largest power-of-two block MLX transforms in one threadgroup pass (`hadamard_n`). Above it MLX
/// splits the transform into two strided passes with a different summation order, so the fused
/// kernel defers to the unfused chain there.
const FUSED_MAX_BLOCK: i32 = 8192;

/// Helpers for [`FUSED_ROTATION_SOURCE`]. `radix_func` is MLX v0.32's thread-local radix-`R`
/// Walsh-Hadamard butterfly (`mlx/backend/metal/kernels/hadamard.h`) verbatim: same stage order,
/// same `a + b` / `a - b` pairs, so every output is the same F32 summation tree.
const FUSED_ROTATION_HEADER: &str = r#"
template <short R>
METAL_FUNC void prism_radix_func(thread float* x) {
  constexpr short logR = __builtin_ctz(R);
  short h = 1;
  for (short s = 0; s < logR; s++) {
    for (short i = 0; i < R / 2; i++) {
      short k = i & (h - 1);
      short j = ((i - k) << 1) + k;
      float a = x[j];
      float b = x[j + h];
      x[j] = a + b;
      x[j + h] = a - b;
    }
    h <<= 1;
  }
}
"#;

/// One fused kernel for `cast -> sign -> blockwise Hadamard -> (sign) -> cast`. The body is MLX
/// v0.32's contiguous `hadamard_n<float, N, R, RW>` with the input cast and the forward sign folded
/// into its device read, and the scale, the inverse sign and the output cast folded into its device
/// write. Sign products are exact (`±1`), the butterfly stages are MLX's, and the only rounding
/// steps are the F32 butterfly adds, the F32 scale product and the final cast — the unfused chain's
/// exact rounding points.
const FUSED_ROTATION_SOURCE: &str = r#"
  constexpr short num_threads = N / R;
  constexpr short logN = __builtin_ctz(N);
  constexpr short logR = __builtin_ctz(R);
  constexpr short num_steps = logN / logR;
  constexpr short logFinal = logN % logR;
  constexpr short final_radix = 1 << (logFinal);

  int batch_idx = int(thread_position_in_grid.y) * N;
  short i = short(thread_position_in_grid.x);

  threadgroup float buf[N];

  for (short j = 0; j < R / RW; j++) {
    short index = j * RW * num_threads + i * RW;
    for (short r = 0; r < RW; r++) {
      int at = batch_idx + index + r;
      float v = static_cast<float>(x[at]);
      if (!INVERSE) {
        v = v * signs[at % W];
      }
      buf[index + r] = v;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float values[R];
  short h = 1;

  for (short s = 0; s < num_steps; s++) {
    short k = i & (h - 1);
    short j = ((i - k) << logR) + k;
    for (short r = 0; r < R; r++) {
      values[r] = buf[j + h * r];
    }
    prism_radix_func<R>(values);
    for (short r = 0; r < R; r++) {
      buf[j + h * r] = values[r];
    }
    h <<= logR;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (final_radix > 1) {
    for (int t = 0; t < R / final_radix; t++) {
      short index = i + t * num_threads;
      short k = index & (h - 1);
      short j = ((index - k) << logFinal) + k;
      for (short r = 0; r < final_radix; r++) {
        values[r] = buf[j + h * r];
      }
      prism_radix_func<final_radix>(values);
      for (short r = 0; r < final_radix; r++) {
        buf[j + h * r] = values[r];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (short j = 0; j < R / RW; j++) {
    short index = j * RW * num_threads + i * RW;
    for (short r = 0; r < RW; r++) {
      int at = batch_idx + index + r;
      float y = buf[index + r] * scale[0];
      if (INVERSE) {
        y = y * signs[at % W];
      }
      out[at] = static_cast<T>(y);
    }
  }
"#;

thread_local! {
    /// The fused rotation kernel handle. MLX caches the compiled library by name and template
    /// arguments; the handle itself is only the (thread-affine) C config.
    static FUSED_ROTATION_KERNEL: std::cell::OnceCell<std::result::Result<MetalKernel, String>> =
        const { std::cell::OnceCell::new() };
}

/// Which rotation implementation [`block_hadamard`] ran — recorded when [`recording_rotation_routes`]
/// is active so a test can assert the route, not just the numbers (the fused kernel and the unfused
/// chain are bit-identical by construction, so numbers alone cannot tell them apart).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationRoute {
    /// One fused Metal kernel dispatch.
    Fused,
    /// One run of the unfused op chain.
    Unfused,
}

thread_local! {
    static ROTATION_ROUTES: std::cell::RefCell<Option<Vec<RotationRoute>>> =
        const { std::cell::RefCell::new(None) };
    /// Route [`block_hadamard`] through the unfused chain (the oracle) on this thread.
    #[doc(hidden)]
    pub static FORCE_UNFUSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn record(route: RotationRoute) {
    ROTATION_ROUTES.with(|r| {
        if let Some(routes) = r.borrow_mut().as_mut() {
            routes.push(route);
        }
    });
}

/// Run `f`, returning every [`RotationRoute`] [`block_hadamard`] recorded on this thread.
#[doc(hidden)]
pub fn recording_rotation_routes<R>(f: impl FnOnce() -> R) -> (R, Vec<RotationRoute>) {
    let previous = ROTATION_ROUTES.with(|r| r.replace(Some(Vec::new())));
    let out = f();
    let routes = ROTATION_ROUTES
        .with(|r| r.replace(previous))
        .unwrap_or_default();
    (out, routes)
}

/// Run `f` with the fused rotation disabled on this thread, restoring it afterwards.
#[doc(hidden)]
pub fn with_unfused_rotation<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_UNFUSED.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(FORCE_UNFUSED.with(|flag| flag.replace(true)));
    f()
}

/// Whether [`block_hadamard`] dispatches the fused kernel for this input on `stream` (the stream
/// the dispatch will run on — a custom Metal kernel runs only on a GPU stream), or why not (the
/// reason the request's fused-primitive report names, E3).
#[doc(hidden)]
pub fn fused_rotation_applies(
    x: &Array,
    block: i32,
    stream: &Stream,
) -> std::result::Result<(), &'static str> {
    let shape_ok = (2..=FUSED_MAX_BLOCK).contains(&block)
        && (block as u32).is_power_of_two()
        && matches!(x.dtype(), Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32)
        && x.size() > 0;
    if FORCE_UNFUSED.with(std::cell::Cell::get) {
        return Err(fused::REASON_FORCED_REFERENCE);
    }
    if !shape_ok {
        Err(fused::REASON_SHAPE)
    } else if !stream_is_gpu(stream) {
        Err(fused::REASON_CPU_STREAM)
    } else if !FUSED_ROTATION.enabled() {
        Err(fused::REASON_DISABLED)
    } else {
        Ok(())
    }
}

#[doc(hidden)]
pub fn fused_block_hadamard(
    x: &Array,
    signs: &Array,
    scale: &Array,
    block: i32,
    width: i32,
    inverse: bool,
    stream: &Stream,
) -> Result<Array> {
    let rows = i32::try_from(x.size() / block as usize)
        .map_err(|_| Error::Msg("Prism fused rotation has too many rows".into()))?;
    // MLX's `hadamard_mn_contiguous` choices for a single power-of-two pass.
    let radix = block.min(16);
    let read_width = if block == 2 { 2 } else { 4 };
    let threads = block / radix;
    FUSED_ROTATION_KERNEL.with(|cell| {
        let kernel = cell
            .get_or_init(|| {
                MetalKernel::with_options(
                    "sceneworks_prism_fused_rotation",
                    &["x", "signs", "scale"],
                    &["out"],
                    FUSED_ROTATION_SOURCE,
                    FUSED_ROTATION_HEADER,
                    true,
                    false,
                )
                .map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| Error::Msg(format!("Prism fused rotation kernel: {e}")))?;
        let mut outputs = kernel
            .apply()
            .input(x)
            .input(signs)
            .input(scale)
            .output_shape(x.shape().to_vec(), x.dtype())
            .grid(threads, rows, 1)
            .thread_group(threads, 1, 1)
            .template_arg("T", x.dtype())
            .template_arg("N", block)
            .template_arg("R", radix)
            .template_arg("RW", read_width)
            .template_arg("W", width)
            .template_arg("INVERSE", inverse)
            .run_device(stream)?;
        outputs
            .pop()
            .ok_or_else(|| Error::Msg("Prism fused rotation returned no output".into()))
    })
}

/// Rotate the final axis: forward applies `signs` then the normalized blockwise Hadamard, inverse
/// applies the Hadamard then `signs`. One fused Metal kernel on the GPU for a power-of-two block
/// up to [`FUSED_MAX_BLOCK`], bit-identical to [`block_hadamard_unfused`]; the unfused chain
/// otherwise.
#[doc(hidden)]
pub fn block_hadamard(
    x: &Array,
    signs: &Array,
    scale: &Array,
    block: i32,
    inverse: bool,
) -> Result<Array> {
    let width = *x
        .shape()
        .last()
        .ok_or_else(|| Error::Config("Prism Hadamard input must have a final axis".into()))?;
    if block <= 0 || width % block != 0 || signs.shape() != [width] {
        return Err(Error::Config(format!(
            "Prism Hadamard width {width} is incompatible with block {block} and signs {:?}",
            signs.shape()
        )));
    }
    // Resolve the stream once: the device check and the kernel dispatch must see the same one
    // (the task-local stream when set, which need not be on the process default device).
    let stream = Stream::task_local_or_default();
    match fused_rotation_applies(x, block, &stream) {
        Ok(()) => {
            record(RotationRoute::Fused);
            fused::note_fused();
            fused_block_hadamard(x, signs, scale, block, width, inverse, &stream)
        }
        Err(reason) => {
            record(RotationRoute::Unfused);
            fused::note_reference(reason);
            block_hadamard_unfused(x, signs, block, inverse, hadamard_scale(block))
        }
    }
}

/// A packed Prism affine matrix with its required forward activation rotation.
#[derive(Debug)]
pub struct PrismLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    signs: Array,
    /// `[1]` F32 Hadamard normalization, resident so a decode step allocates nothing for it.
    scale: Array,
    block: i32,
}

impl PrismLinear {
    /// The arrays this matrix holds, for load-time materialization (sc-24446).
    pub fn push_arrays(&self, out: &mut Vec<Array>) {
        out.extend([
            self.weight.clone(),
            self.scales.clone(),
            self.biases.clone(),
            self.signs.clone(),
            self.scale.clone(),
        ]);
    }

    /// Validate and retain a packed `[out, in]` matrix without dequantizing it.
    pub fn new(
        label: &str,
        weight: Array,
        scales: Array,
        biases: Array,
        signs: Array,
        block: i32,
    ) -> Result<Self> {
        validate_parts(label, &weight, &scales, &biases, &signs, block)?;
        Ok(Self {
            weight,
            scales,
            biases,
            signs,
            scale: Array::from_slice(&[hadamard_scale(block)], &[1]),
            block,
        })
    }

    /// Apply `signs`, normalized blockwise Hadamard, then packed affine matmul.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let rotated = block_hadamard(x, &self.signs, &self.scale, self.block, false)?;
        Ok(quantized_matmul(
            &rotated,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            GROUP_SIZE,
            BITS,
        )?)
    }
}

/// Packed Prism token embeddings with the inverse rotation applied after row dequantization.
#[derive(Debug)]
pub struct PrismEmbedding {
    weight: Array,
    scales: Array,
    biases: Array,
    signs: Array,
    /// `[1]` F32 Hadamard normalization.
    scale: Array,
    block: i32,
    width: i32,
}

impl PrismEmbedding {
    /// The arrays this matrix holds, for load-time materialization (sc-24446).
    pub fn push_arrays(&self, out: &mut Vec<Array>) {
        out.extend([
            self.weight.clone(),
            self.scales.clone(),
            self.biases.clone(),
            self.signs.clone(),
            self.scale.clone(),
        ]);
    }

    /// Validate and retain a packed `[vocab, hidden]` embedding table.
    pub fn new(
        label: &str,
        weight: Array,
        scales: Array,
        biases: Array,
        signs: Array,
        block: i32,
    ) -> Result<Self> {
        let (_, width) = validate_parts(label, &weight, &scales, &biases, &signs, block)?;
        Ok(Self {
            weight,
            scales,
            biases,
            signs,
            scale: Array::from_slice(&[hadamard_scale(block)], &[1]),
            block,
            width,
        })
    }

    /// Gather only requested packed rows, dequantize them, then apply `H` and `signs`.
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        let id_shape = ids.shape();
        if id_shape.len() != 2 {
            return Err(Error::Msg(format!(
                "Prism embedding expects [batch, sequence] ids, got {id_shape:?}"
            )));
        }
        let flat = ids.reshape(&[-1])?;
        let weight = self.weight.take_axis(&flat, 0)?;
        let scales = self.scales.take_axis(&flat, 0)?;
        let biases = self.biases.take_axis(&flat, 0)?;
        let rows = dequantize(&weight, &scales, Some(&biases), GROUP_SIZE, BITS)?;
        let rows = rows.reshape(&[id_shape[0], id_shape[1], self.width])?;
        block_hadamard(&rows, &self.signs, &self.scale, self.block, true)
    }
}

#[cfg(test)]
mod tests {
    use super::fused::{fused_tally, note_fused, note_reference, REASON_CPU_STREAM};
    use super::*;

    #[test]
    fn the_tally_counts_each_route_on_this_thread() {
        let start = fused_tally();
        note_fused();
        note_reference(REASON_CPU_STREAM);
        let since = fused_tally().since(&start);
        assert_eq!((since.fused, since.reference), (1, 1));
        assert_eq!(since.label(), "mixed");
        assert_eq!(since.reference_reason, Some(REASON_CPU_STREAM));
        std::thread::spawn(note_fused).join().unwrap();
        assert_eq!(fused_tally().since(&start).fused, 1);
    }

    #[test]
    fn fused_rotation_switch_defaults_to_mlx_row_and_supports_scoped_overrides() {
        assert_eq!(FUSED_ROTATION.process().env(), FUSED_ROTATION_ENV);
        if std::env::var_os(FUSED_ROTATION_ENV).is_none()
            && FUSED_ROTATION.process().explicit().is_none()
        {
            assert_eq!(FUSED_ROTATION.enabled(), MLX.fused_rotation);
        }
        let before = FUSED_ROTATION.enabled();
        FUSED_ROTATION.scoped(!before, || {
            assert_eq!(FUSED_ROTATION.enabled(), !before);
            let other =
                std::thread::scope(|s| s.spawn(|| FUSED_ROTATION.enabled()).join().unwrap());
            assert_eq!(other, before);
        });
        assert_eq!(FUSED_ROTATION.enabled(), before);
        assert!(env_value_enables("1") && env_value_enables("on"));
        assert!(!env_value_enables("0") && !env_value_enables("off"));
    }

    #[test]
    fn hadamard_scale_is_power_of_two_reciprocal_sqrt() {
        assert_eq!(hadamard_scale(64), 0.125);
        assert_eq!(hadamard_scale(256), 0.0625);
        assert_eq!(hadamard_scale(1024), 0.03125);
    }
}
