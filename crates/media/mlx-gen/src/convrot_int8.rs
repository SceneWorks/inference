//! Descriptor-driven symmetric INT8 projections, not MLX's affine Q8 format (sc-26683).
//! Only activations and the tiny rotation matrix are floating; stored weights stay I8.

use crate::{weights::Weights, Result};
use gen_core::convrot_int8::{ConvRotInt8, MAX_DESCRIPTOR_BYTES};
use mlx_rs::{fast::MetalKernel, ops::matmul, transforms::eval, Array, Dtype};
use std::cell::RefCell;

const QUANT: &str = r#"
uint row = thread_position_in_grid.x;
if (row >= M) return;
float peak = 0.0f;
for (uint j=0; j<K; ++j) { float v=abs(float(x[row*K+j])); if (!isnan(v)) peak=max(peak,v); }
peak = min(peak, DTYPE==1 ? 3.38953139e38f : DTYPE==2 ? 65504.0f : 3.402823466e38f);
// Kitchen rounds scale and quotient to the activation dtype before nearbyintf.
float scale = max(peak * (1.0f / 127.0f), 1.0e-30f);
s[row] = scale;
float divisor = float(T(scale));
for (uint j=0; j<K; ++j) {
    float z = float(T(float(x[row*K+j]) / divisor));
    q[row*K+j] = char(isnan(z) ? -128.0f : clamp(rint(z), -128.0f, 127.0f));
}
"#;
const PROJECT: &str = r#"
uint lane = thread_position_in_threadgroup.x;
uint out = thread_position_in_grid.y;
uint row = thread_position_in_grid.z;
int acc = 0;
for (uint j=lane; j<K; j+=32) acc += int(q[row*K+j]) * int(w[out*K+j]);
acc = simd_sum(acc);
if (lane == 0) {
    float value = (float(acc) * sx[row]) * sw[out];
    y[row*N+out] = T(value + b[out]);
}
"#;

thread_local! {
    // MetalKernel is thread-affine; never place one in a Send provider/model base.
    // These handles capture no arrays and share MLX's existing compiled-pipeline cache.
    static KERNELS: RefCell<Option<(MetalKernel, MetalKernel)>> = const { RefCell::new(None) };
}

fn kernels<T>(f: impl FnOnce(&MetalKernel, &MetalKernel) -> Result<T>) -> Result<T> {
    crate::nn::prepare_retained_compilation_thread();
    KERNELS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.is_none() {
            *cache = Some((
                MetalKernel::new("sc_convrot_int8_quant_v1", &["x"], &["q", "s"], QUANT)?,
                MetalKernel::new(
                    "sc_convrot_int8_project_v1",
                    &["q", "w", "sx", "sw", "b"],
                    &["y"],
                    PROJECT,
                )?,
            ));
        }
        let (quant, project) = cache.as_ref().expect("initialized above");
        f(quant, project)
    })
}

/// Compact native integer base, shareable by adapter-bearing projections.
#[derive(Clone)]
pub struct ConvRotLinear {
    pub(crate) descriptor: ConvRotInt8,
    codes: Array,
    scales: Array,
    pub(crate) bias: Option<Array>,
    rotation: Array,
}

impl ConvRotLinear {
    /// Validated geometry cannot be mutated independently of the resident buffers.
    pub fn descriptor(&self) -> ConvRotInt8 {
        self.descriptor
    }

    /// Admit only a complete authoritative descriptor plus I8/F32 operands. No dequantization.
    pub fn from_weights(w: &Weights, base: &str, bias: bool) -> Result<Option<Self>> {
        let dk = format!("{base}.comfy_quant");
        let sk = format!("{base}.weight_scale");
        let codes = w.require(&format!("{base}.weight"))?;
        if w.get(&dk).is_none() {
            if w.get(&sk).is_some() || codes.dtype() == Dtype::Int8 {
                return Err(format!(
                    "{base}: INT8 requires an authoritative .comfy_quant descriptor"
                )
                .into());
            }
            return Ok(None);
        }
        if w.get(&format!("{base}.scales")).is_some() || w.get(&format!("{base}.biases")).is_some()
        {
            return Err(format!("{base}: contradictory affine and ConvRot operands").into());
        }
        let payload = w.require(&dk)?;
        if payload.dtype() != Dtype::Uint8
            || payload.ndim() != 1
            || payload.size() > MAX_DESCRIPTOR_BYTES
        {
            return Err(format!("{base}: .comfy_quant must be a bounded U8 vector").into());
        }
        let scales = w.require(&sk)?;
        if codes.dtype() != Dtype::Int8 || scales.dtype() != Dtype::Float32 {
            return Err(format!(
                "{base}: ConvRot requires I8 codes and F32 scales without casting"
            )
            .into());
        }
        eval([payload, scales])?;
        let shape: Vec<usize> = codes.shape().iter().map(|&v| v as usize).collect();
        let scale_shape: Vec<usize> = scales.shape().iter().map(|&v| v as usize).collect();
        let descriptor = ConvRotInt8::parse(payload.as_slice::<u8>(), &shape, &scale_shape)
            .map_err(|e| crate::Error::Msg(format!("{base}: {e}")))?;
        descriptor
            .validate_scales(scales.as_slice::<f32>())
            .map_err(crate::Error::Msg)?;
        let bias = if bias {
            Some(w.require(&format!("{base}.bias"))?.clone())
        } else {
            None
        };
        if bias.as_ref().is_some_and(|b| {
            b.shape() != [descriptor.out_features as i32]
                || !matches!(b.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        }) {
            return Err(format!("{base}: invalid ConvRot bias").into());
        }
        let g = descriptor.rotation_group;
        let rotation = w.convrot_rotation(g, || {
            let h4 = [
                [1f32, 1., 1., -1.],
                [1., 1., -1., 1.],
                [1., -1., 1., 1.],
                [-1., 1., 1., 1.],
            ];
            let values: Vec<f32> = (0..g * g)
                .map(|i| {
                    let (mut r, mut c, mut s) = (i / g, i % g, 1f32);
                    while r != 0 || c != 0 {
                        s *= h4[r % 4][c % 4];
                        r /= 4;
                        c /= 4;
                    }
                    s / (g as f32).sqrt()
                })
                .collect();
            Array::from_slice(&values, &[g as i32, g as i32])
        });
        Ok(Some(Self {
            descriptor,
            codes: codes.clone(),
            scales: scales.clone(),
            bias,
            rotation,
        }))
    }

    /// Native signed integer dot, FP32 row-scale epilogue, one output rounding after bias.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let d = self.descriptor;
        if x.shape().last() != Some(&(d.in_features as i32))
            || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err("ConvRot INT8 activation shape/dtype mismatch".into());
        }
        let k = d.in_features as i32;
        let n = d.out_features as i32;
        let g = d.rotation_group as i32;
        let rows = x.size() / d.in_features;
        if rows == 0
            || rows > i32::MAX as usize
            || x.size() > u32::MAX as usize
            || rows
                .checked_mul(d.out_features)
                .is_none_or(|s| s > u32::MAX as usize)
        {
            return Err("ConvRot INT8 activation exceeds native index geometry".into());
        }
        let m = rows as i32;
        // Rotation in the source dtype, as Kitchen's _rotate_activation, NOT a permanent float weight.
        let rotated =
            matmul(&x.reshape(&[-1, g])?, &self.rotation.as_dtype(x.dtype())?)?.reshape(&[m, k])?;
        kernels(|quant, project| {
            let mut q = quant
                .apply()
                .inputs([&rotated])
                .output_shape([m, k], Dtype::Int8)
                .output_shape([m], Dtype::Float32)
                .template_arg("T", x.dtype())
                .template_arg("M", m)
                .template_arg("K", k)
                .template_arg(
                    "DTYPE",
                    match x.dtype() {
                        Dtype::Bfloat16 => 1i32,
                        Dtype::Float16 => 2,
                        _ => 0,
                    },
                )
                .grid(m, 1, 1)
                .thread_group(m.min(256), 1, 1)
                .run()?
                .into_iter();
            let codes = q.next().ok_or("ConvRot quantizer returned no codes")?;
            let scales = q.next().ok_or("ConvRot quantizer returned no scales")?;
            let bias = match &self.bias {
                Some(b) => b.as_dtype(x.dtype())?.as_dtype(Dtype::Float32)?,
                None => mlx_rs::ops::zeros::<f32>(&[n])?,
            };
            let mut out = project
                .apply()
                .inputs([&codes, &self.codes, &scales, &self.scales, &bias])
                .output_shape([m, n], x.dtype())
                .template_arg("T", x.dtype())
                .template_arg("K", k)
                .template_arg("N", n)
                .grid(32, n, m)
                .thread_group(32, 1, 1)
                .run()?;
            let mut shape = x.shape().to_vec();
            *shape.last_mut().unwrap() = n;
            Ok(out.remove(0).reshape(&shape)?)
        })
    }

    /// Retained native arrays only; no dense weight set can be materialized through this API.
    pub fn materialize(&self) -> Result<()> {
        eval(
            [&self.codes, &self.scales, &self.rotation]
                .into_iter()
                .chain(self.bias.iter()),
        )?;
        Ok(())
    }
    /// Projection-owned bytes; the two immutable rotation constants are shared per load.
    pub fn nbytes(&self) -> usize {
        self.descriptor.resident_bytes() + self.bias.as_ref().map_or(0, Array::nbytes)
    }
}
