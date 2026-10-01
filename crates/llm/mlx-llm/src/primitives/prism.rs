//! Prism ternary affine operators for Ternary Bonsai checkpoints.
//!
//! The packed MLX artifact stores 16 two-bit affine codes per `u32`, with one scale and bias for
//! each 128 input values. Bonsai folds a normalized blockwise Walsh-Hadamard rotation into every
//! packed matrix: linear activations apply `signs` then `H`, while embedding rows apply the inverse
//! `H` then `signs`. These operators keep the packed weights resident and use MLX's native
//! two-bit matmul/dequantize kernels; they never materialize a full dense model. Each rotation —
//! cast, sign, blockwise Hadamard, cast — is one fused Metal kernel per projection input
//! (sc-24444), bit-identical to the op chain it replaces.

use mlx_rs::fast::MetalKernel;
use mlx_rs::ops::{dequantize, multiply, quantized_matmul};
use mlx_rs::{Array, Dtype, Stream};

use crate::error::{Error, Result};
use crate::primitives::stream_is_gpu;

const GROUP_SIZE: i32 = 128;
const BITS: i32 = 2;

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
fn block_hadamard_unfused(
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
fn hadamard_scale(block: i32) -> f32 {
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
    static FUSED_ROTATION: std::cell::OnceCell<std::result::Result<MetalKernel, String>> =
        const { std::cell::OnceCell::new() };
}

/// Which rotation implementation [`block_hadamard`] ran — recorded (test builds only), so a test
/// can assert the route, not just the numbers (the fused kernel and the unfused chain are
/// bit-identical by construction, so numbers alone cannot tell them apart).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RotationRoute {
    /// One fused Metal kernel dispatch.
    Fused,
    /// One run of the unfused op chain.
    Unfused,
}

#[cfg(test)]
thread_local! {
    static ROTATION_ROUTES: std::cell::RefCell<Option<Vec<RotationRoute>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn record(route: RotationRoute) {
    ROTATION_ROUTES.with(|r| {
        if let Some(routes) = r.borrow_mut().as_mut() {
            routes.push(route);
        }
    });
}

/// Run `f`, returning every [`RotationRoute`] [`block_hadamard`] recorded on this thread.
#[cfg(test)]
pub(crate) fn recording_rotation_routes<R>(f: impl FnOnce() -> R) -> (R, Vec<RotationRoute>) {
    let previous = ROTATION_ROUTES.with(|r| r.replace(Some(Vec::new())));
    let out = f();
    let routes = ROTATION_ROUTES
        .with(|r| r.replace(previous))
        .unwrap_or_default();
    (out, routes)
}

/// Whether [`block_hadamard`] dispatches the fused kernel for this input on `stream` (the stream
/// the dispatch will run on — a custom Metal kernel runs only on a GPU stream), or why not (the
/// reason the request's fused-primitive report names, E3).
fn fused_rotation_applies(
    x: &Array,
    block: i32,
    stream: &Stream,
) -> std::result::Result<(), &'static str> {
    let shape_ok = (2..=FUSED_MAX_BLOCK).contains(&block)
        && (block as u32).is_power_of_two()
        && matches!(x.dtype(), Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32)
        && x.size() > 0;
    #[cfg(test)]
    if tests::FORCE_UNFUSED.with(std::cell::Cell::get) {
        return Err(super::fused::REASON_FORCED_REFERENCE);
    }
    if !shape_ok {
        Err(super::fused::REASON_SHAPE)
    } else if !stream_is_gpu(stream) {
        Err(super::fused::REASON_CPU_STREAM)
    } else if !crate::switches::FUSED_ROTATION.enabled() {
        Err(super::fused::REASON_DISABLED)
    } else {
        Ok(())
    }
}

fn fused_block_hadamard(
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
    FUSED_ROTATION.with(|cell| {
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
fn block_hadamard(
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
            #[cfg(test)]
            record(RotationRoute::Fused);
            super::fused::note_fused();
            fused_block_hadamard(x, signs, scale, block, width, inverse, &stream)
        }
        Err(reason) => {
            #[cfg(test)]
            record(RotationRoute::Unfused);
            super::fused::note_reference(reason);
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
pub(crate) mod tests {
    use super::*;
    use crate::primitives::nn::linear;

    thread_local! {
        /// Route [`block_hadamard`] through the unfused chain (the oracle) on this thread.
        pub(crate) static FORCE_UNFUSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Run `f` with the fused rotation disabled on this thread, restoring it afterwards.
    pub(crate) fn with_unfused_rotation<R>(f: impl FnOnce() -> R) -> R {
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                FORCE_UNFUSED.with(|flag| flag.set(self.0));
            }
        }
        let _restore = Restore(FORCE_UNFUSED.with(|flag| flag.replace(true)));
        f()
    }

    fn bits(a: &Array) -> Vec<u32> {
        // Widening to F32 is exact for F16/BF16 and keeps the sign of zero and NaN payload class.
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    fn random_signs(width: i32, seed: u64) -> Array {
        let key = mlx_rs::random::key(seed).unwrap();
        let coin = mlx_rs::random::bernoulli(None, &[width][..], Some(&key)).unwrap();
        mlx_rs::ops::r#where(&coin, Array::from_f32(1.0), Array::from_f32(-1.0)).unwrap()
    }

    /// AC1: the fused kernel is bit-identical to the unfused cast -> sign -> Hadamard -> cast chain
    /// on random inputs at Ternary Bonsai 2 27B's projection input widths (hidden 5120, attention
    /// output 6144, MLP 17408, block 1024), for decode and prefill row counts, both directions and
    /// every activation dtype — plus smaller power-of-two blocks covering each final-radix branch.
    #[test]
    fn fused_rotation_is_bit_identical_to_the_unfused_chain() {
        mlx_rs::with_new_default_stream(Stream::gpu(), fused_rotation_matches_at_every_case);
    }

    fn fused_rotation_matches_at_every_case() {
        let gpu = Stream::task_local_or_default();
        let mut cases = Vec::new();
        for width in [5120, 6144, 17408] {
            cases.push((width, 1024));
        }
        for block in [2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 4096, 8192] {
            cases.push((block * 2, block));
        }
        let mut seed = 0u64;
        for (width, block) in cases {
            for rows in [1, 3, 17] {
                for dtype in [Dtype::Bfloat16, Dtype::Float16, Dtype::Float32] {
                    seed += 1;
                    let key = mlx_rs::random::key(seed).unwrap();
                    let x = mlx_rs::random::normal::<f32>(
                        &[1, rows, width][..],
                        None,
                        None,
                        Some(&key),
                    )
                    .unwrap()
                    .multiply(Array::from_f32(3.0))
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                    let signs = random_signs(width, seed + 10_000);
                    let scale = Array::from_slice(&[hadamard_scale(block)], &[1]);
                    for inverse in [false, true] {
                        assert!(fused_rotation_applies(&x, block, &gpu).is_ok());
                        let fused =
                            fused_block_hadamard(&x, &signs, &scale, block, width, inverse, &gpu)
                                .unwrap();
                        let reference = block_hadamard_unfused(
                            &x,
                            &signs,
                            block,
                            inverse,
                            hadamard_scale(block),
                        )
                        .unwrap();
                        assert_eq!(fused.dtype(), dtype);
                        assert_eq!(fused.shape(), x.shape());
                        assert!(
                            bits(&fused) == bits(&reference),
                            "width {width} block {block} rows {rows} {dtype:?} inverse {inverse}"
                        );
                    }
                }
            }
        }
    }

    /// Signed zeros, exact powers of two and extreme finite values keep their exact bits too.
    #[test]
    fn fused_rotation_matches_on_edge_values() {
        mlx_rs::with_new_default_stream(Stream::gpu(), fused_rotation_matches_edge_values);
    }

    fn fused_rotation_matches_edge_values() {
        let gpu = Stream::task_local_or_default();
        let width = 1024;
        let mut values = vec![0.0f32; width as usize];
        for (i, v) in values.iter_mut().enumerate() {
            *v = match i % 8 {
                0 => 0.0,
                1 => -0.0,
                2 => 65504.0,
                3 => -65504.0,
                4 => f32::MIN_POSITIVE,
                5 => 1.0e-8,
                6 => (i as f32).exp2().min(1.0e30),
                _ => -(i as f32) / 7.0,
            };
        }
        let x = Array::from_slice(&values, &[1, width]);
        let signs = random_signs(width, 77);
        let scale = Array::from_slice(&[hadamard_scale(width)], &[1]);
        for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
            let x = x.as_dtype(dtype).unwrap();
            for inverse in [false, true] {
                let fused =
                    fused_block_hadamard(&x, &signs, &scale, width, width, inverse, &gpu).unwrap();
                let reference =
                    block_hadamard_unfused(&x, &signs, width, inverse, hadamard_scale(width))
                        .unwrap();
                assert!(bits(&fused) == bits(&reference), "{dtype:?} {inverse}");
            }
        }
    }

    /// A block MLX splits into two passes (> 8192) stays on the unfused chain, as does a forced
    /// oracle run or a CPU stream; the dispatcher never hands the kernel a shape it was not proven
    /// on, nor a stream it cannot run on.
    #[test]
    fn rotation_dispatch_uses_the_kernel_only_where_it_is_proven() {
        let gpu = Stream::gpu();
        let x = Array::zeros::<f32>(&[1, 16384]).unwrap();
        use crate::primitives::fused::{REASON_CPU_STREAM, REASON_FORCED_REFERENCE, REASON_SHAPE};
        assert_eq!(fused_rotation_applies(&x, 16384, &gpu), Err(REASON_SHAPE));
        assert_eq!(fused_rotation_applies(&x, 1536, &gpu), Err(REASON_SHAPE));
        assert_eq!(fused_rotation_applies(&x, 1024, &gpu), Ok(()));
        assert_eq!(
            fused_rotation_applies(&x, 1024, &Stream::cpu()),
            Err(REASON_CPU_STREAM)
        );
        with_unfused_rotation(|| {
            assert_eq!(
                fused_rotation_applies(&x, 1024, &gpu),
                Err(REASON_FORCED_REFERENCE)
            )
        });
        assert_eq!(fused_rotation_applies(&x, 1024, &gpu), Ok(()));
        // E3/E5: the process switch turned off is named as the reference route's reason.
        crate::switches::FUSED_ROTATION.scoped(false, || {
            assert_eq!(
                fused_rotation_applies(&x, 1024, &gpu),
                Err(crate::primitives::fused::REASON_DISABLED)
            )
        });
        let ints = Array::zeros::<i32>(&[1, 1024]).unwrap();
        assert_eq!(fused_rotation_applies(&ints, 1024, &gpu), Err(REASON_SHAPE));
    }

    /// E3: every rotation counts in the request's fused-primitive report by the route that ran —
    /// the kernel as `fused`, the unfused chain as `reference` with why (a forced oracle run, a
    /// CPU stream).
    #[test]
    fn every_rotation_route_is_tallied_for_the_request_report() {
        use crate::primitives::fused::{fused_tally, REASON_CPU_STREAM, REASON_FORCED_REFERENCE};
        let signs = random_signs(1024, 41);
        let scale = Array::from_slice(&[hadamard_scale(1024)], &[1]);
        let x = Array::zeros::<f32>(&[1, 2, 1024]).unwrap();
        let rotate = || block_hadamard(&x, &signs, &scale, 1024, false).unwrap();
        let ran = |f: &dyn Fn()| {
            let start = fused_tally();
            f();
            fused_tally().since(&start)
        };
        let fused = ran(&|| {
            mlx_rs::with_new_default_stream(Stream::gpu(), || {
                rotate();
            })
        });
        assert_eq!(
            (fused.fused, fused.reference, fused.label()),
            (1, 0, "fused")
        );
        let forced = ran(&|| {
            mlx_rs::with_new_default_stream(Stream::gpu(), || {
                with_unfused_rotation(|| {
                    rotate();
                })
            })
        });
        assert_eq!(forced.label(), "reference");
        assert_eq!(forced.reference_reason, Some(REASON_FORCED_REFERENCE));
        let cpu = ran(&|| {
            crate::primitives::kv_cache::testing::on_cpu(|| {
                rotate();
            })
        });
        assert_eq!((cpu.fused, cpu.reference), (0, 1));
        assert_eq!(cpu.reference_reason, Some(REASON_CPU_STREAM));
    }

    /// The route follows the stream ops are issued on, never the process default device: a
    /// task-local GPU stream inside a task-local CPU scope runs the fused kernel on that stream,
    /// bit-identical to the unfused chain, and once it closes the enclosing CPU scope takes the
    /// unfused chain again. Both scopes are task-local (switching the process default device races
    /// every parallel test in this binary — sc-24439).
    #[test]
    fn fused_rotation_runs_on_the_task_local_gpu_stream_inside_a_cpu_scope() {
        let width = 5120;
        let signs = random_signs(width, 32);
        let scale = Array::from_slice(&[hadamard_scale(1024)], &[1]);
        let input = |seed| {
            let key = mlx_rs::random::key(seed).unwrap();
            mlx_rs::random::normal::<f32>(&[1, 3, width][..], None, None, Some(&key))
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        crate::primitives::kv_cache::testing::on_cpu(|| {
            mlx_rs::with_new_default_stream(Stream::gpu(), || {
                let x = input(31);
                for inverse in [false, true] {
                    let (fused, routes) = recording_rotation_routes(|| {
                        block_hadamard(&x, &signs, &scale, 1024, inverse).unwrap()
                    });
                    assert_eq!(routes, [RotationRoute::Fused], "inverse {inverse}");
                    let reference =
                        block_hadamard_unfused(&x, &signs, 1024, inverse, hadamard_scale(1024))
                            .unwrap();
                    assert!(bits(&fused) == bits(&reference), "inverse {inverse}");
                }
            });
            let x = input(33);
            let (_, routes) = recording_rotation_routes(|| {
                block_hadamard(&x, &signs, &scale, 1024, false).unwrap()
            });
            assert_eq!(
                routes,
                [RotationRoute::Unfused],
                "the enclosing CPU scope is restored"
            );
        });
        // The kernel dispatches on the stream it is handed, not the default one: MLX refuses a
        // custom Metal kernel on a CPU stream.
        let x = input(34);
        let on_cpu = fused_block_hadamard(&x, &signs, &scale, 1024, width, false, &Stream::cpu())
            .and_then(|y| Ok(y.eval()?));
        assert!(
            on_cpu.is_err(),
            "the kernel ignored the stream it was handed"
        );
    }

    /// sc-24446 (E5): with the fused rotation switched off (`MLX_LLM_FUSED_ROTATION`; here its
    /// thread-scoped layer) a GPU stream runs the unfused chain, bit-identical to the kernel.
    #[test]
    fn the_fused_rotation_switch_off_runs_the_unfused_chain_on_the_gpu() {
        let width = 2048;
        let signs = random_signs(width, 32);
        let scale = Array::from_slice(&[hadamard_scale(1024)], &[1]);
        mlx_rs::with_new_default_stream(Stream::gpu(), || {
            let key = mlx_rs::random::key(35).unwrap();
            let x = mlx_rs::random::normal::<f32>(&[1, 2, width][..], None, None, Some(&key))
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let rotate = || block_hadamard(&x, &signs, &scale, 1024, false).unwrap();
            let (fused, on) = recording_rotation_routes(rotate);
            assert_eq!(on, [RotationRoute::Fused]);
            let (unfused, off) =
                crate::switches::FUSED_ROTATION.scoped(false, || recording_rotation_routes(rotate));
            assert_eq!(off, [RotationRoute::Unfused], "the switch is off");
            assert!(bits(&fused) == bits(&unfused));
        });
    }

    #[test]
    fn packed_linear_and_inverse_embedding_match_dense_operator_oracles() {
        let width = 128i32;
        let rows = 2i32;
        let signs = Array::from_slice(
            &(0..width)
                .map(|i| if i % 3 == 0 { -1.0f32 } else { 1.0 })
                .collect::<Vec<_>>(),
            &[width],
        );
        let mut words = vec![0u32; (rows * width / 16) as usize];
        for row in 0..rows as usize {
            for col in 0..width as usize {
                let code = ((row + col) % 3) as u32;
                words[row * (width as usize / 16) + col / 16] |= code << (2 * (col % 16));
            }
        }
        let scales = Array::from_slice(&[0.25f32, 0.5], &[rows, 1]);
        let biases = Array::from_slice(&[-0.25f32, -0.5], &[rows, 1]);
        let weight = Array::from_slice(&words, &[rows, width / 16]);
        let dense_rotated = dequantize(&weight, &scales, Some(&biases), GROUP_SIZE, BITS).unwrap();
        let x = Array::from_slice(
            &(0..width)
                .map(|i| (i as f32 - 50.0) / 64.0)
                .collect::<Vec<_>>(),
            &[1, width],
        );
        let rotated =
            block_hadamard_unfused(&x, &signs, width, false, hadamard_scale(width)).unwrap();
        let expected = linear(&rotated, &dense_rotated, None).unwrap();
        let packed = PrismLinear::new(
            "test",
            weight.clone(),
            scales.clone(),
            biases.clone(),
            signs.clone(),
            width,
        )
        .unwrap()
        .forward(&x)
        .unwrap();
        assert!(packed
            .all_close(&expected, 1e-4, 1e-4, None)
            .unwrap()
            .item::<bool>());

        let ids = Array::from_slice(&[1i32, 0], &[1, 2]);
        let embedding =
            PrismEmbedding::new("embedding", weight, scales, biases, signs.clone(), width)
                .unwrap()
                .forward(&ids)
                .unwrap();
        let gathered = dense_rotated
            .take_axis(Array::from_slice(&[1i32, 0], &[2]), 0)
            .unwrap()
            .reshape(&[1, 2, width])
            .unwrap();
        let expected =
            block_hadamard_unfused(&gathered, &signs, width, true, hadamard_scale(width)).unwrap();
        assert!(embedding
            .all_close(&expected, 1e-5, 1e-5, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn malformed_packed_metadata_fails_closed() {
        let weight = Array::from_slice(&[0u32; 8], &[1, 8]);
        let scales = Array::from_slice(&[1.0f32], &[1, 1]);
        let biases = Array::from_slice(&[-1.0f32], &[1, 1]);
        let bad_signs = Array::from_slice(&vec![1.0f32; 127], &[127]);
        let err = PrismLinear::new("bad", weight, scales, biases, bad_signs, 128).unwrap_err();
        assert!(err.to_string().contains("geometry mismatch"), "{err}");
    }
}
