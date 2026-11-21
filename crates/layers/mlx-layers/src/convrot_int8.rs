//! Descriptor-driven symmetric INT8 projections, not MLX's affine Q8 format (sc-26683).
//! Only activations and the tiny rotation matrix are floating; stored weights stay I8.

use crate::Result;
use gen_core::convrot_int8::{ConvRotInt8, MAX_DESCRIPTOR_BYTES};
#[cfg(any(test, feature = "test-support"))]
use mlx_rs::fast::MetalKernel;
use mlx_rs::{ops::matmul, transforms::eval, Array, Dtype};
use std::{cell::RefCell, collections::HashMap};

/// The original serial kernels (sc-26683), kept ONLY as the parity reference for the tiled path in
/// [`crate::int8_gemm`]: one row per thread for the quantizer and one 32-lane SIMD group per output
/// element for the dot product. Correct, but every activation row re-reads the whole weight, which
/// made H3 take ~47 minutes per denoise step on Metal (sc-26691). Never on a production route.
#[cfg(any(test, feature = "test-support"))]
const REFERENCE_QUANT: &str = r#"
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
/// See [`REFERENCE_QUANT`]. Shared verbatim (same name and source) with the W4A8 reference.
#[cfg(any(test, feature = "test-support"))]
pub const REFERENCE_PROJECT: &str = r#"
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

#[cfg(any(test, feature = "test-support"))]
thread_local! {
    // MetalKernel is thread-affine; never place one in a Send provider/model base.
    static REFERENCE_KERNELS: RefCell<Option<(MetalKernel, MetalKernel)>> = const { RefCell::new(None) };
}

#[cfg(any(test, feature = "test-support"))]
fn reference_kernels<T>(f: impl FnOnce(&MetalKernel, &MetalKernel) -> Result<T>) -> Result<T> {
    mlx_rs::transforms::compile::prepare_retained_compilation_thread();
    REFERENCE_KERNELS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.is_none() {
            *cache = Some((
                MetalKernel::new(
                    "sc_convrot_int8_quant_v1",
                    &["x"],
                    &["q", "s"],
                    REFERENCE_QUANT,
                )?,
                MetalKernel::new(
                    "sc_convrot_int8_project_v1",
                    &["q", "w", "sx", "sw", "b"],
                    &["y"],
                    REFERENCE_PROJECT,
                )?,
            ));
        }
        let (quant, project) = cache.as_ref().expect("initialized above");
        f(quant, project)
    })
}

/// Source operands and load-scoped rotation cache. Both engines mark operands through their
/// normal access-tracking weight map so block eviction and materialization keep their semantics.
pub trait MlxWeightSource {
    fn require(&self, key: &str) -> Result<&Array> {
        self.get(key)
            .ok_or_else(|| crate::Error::Msg(format!("missing tensor {key}")))
    }
    fn planned_fp8(&self, _base: &str) -> Option<&crate::fp8::Fp8Linear> {
        None
    }
    fn convrot_rotation(&self, group: usize, build: impl FnOnce() -> Array) -> Array {
        self.rotation_cache()
            .borrow_mut()
            .entry(group)
            .or_insert_with(build)
            .clone()
    }
    fn get(&self, key: &str) -> Option<&Array>;
    fn rotation_cache(&self) -> &RefCell<HashMap<usize, Array>>;
    fn planned_geometry(&self, _base: &str) -> Option<ConvRotInt8> {
        None
    }
}

pub fn descriptor(blob: &Array, base: &str) -> Result<gen_core::comfy_quant::ComfyQuantDescriptor> {
    if blob.dtype() != Dtype::Uint8 || blob.ndim() != 1 || blob.size() > MAX_DESCRIPTOR_BYTES {
        return Err(format!("{base}: .comfy_quant must be a bounded U8 vector").into());
    }
    blob.eval()?;
    gen_core::comfy_quant::parse_comfy_quant_descriptor(blob.as_slice::<u8>())
        .map_err(|e| crate::Error::Msg(format!("{base}: {e}")))
}

/// Restore a complete declaration for consumers that rename keys through a raw MLX map.
pub fn descriptor_array(group: usize) -> Array {
    let json = format!(
        r#"{{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":{group},"per_row":true}}"#
    );
    Array::from_slice(json.as_bytes(), &[json.len() as i32])
}

/// Resolve file-level declarations against the complete shard-set key inventory before a
/// provider can rename its keys or drop file metadata. Per-layer payloads remain authoritative.
pub fn attach_metadata(
    tensors: &mut HashMap<String, Array>,
    declarations: &[String],
) -> Result<()> {
    use gen_core::comfy_quant::{parse_comfy_quant_descriptor, parse_quantization_metadata};
    use std::collections::BTreeMap;
    for declaration in declarations {
        let table = parse_quantization_metadata(declaration)
            .map_err(|e| crate::Error::Msg(e.to_string()))?;
        let names: BTreeMap<_, _> = tensors.keys().map(|k| (k.as_str(), ())).collect();
        let prefix = gen_core::checkpoint_codec::resolve_metadata_prefix(&names, &table)
            .map_err(|e| crate::Error::Msg(e.to_string()))?;
        for (name, entry) in table {
            let base = format!("{prefix}{name}");
            let key = format!("{base}.comfy_quant");
            if let Some(blob) = tensors.get(&key) {
                if blob.dtype() != Dtype::Uint8
                    || blob.ndim() != 1
                    || blob.size() > MAX_DESCRIPTOR_BYTES
                {
                    return Err(format!("{base}: .comfy_quant must be a bounded U8 vector").into());
                }
                blob.eval()?;
                let parsed = parse_comfy_quant_descriptor(blob.as_slice::<u8>())
                    .map_err(|e| crate::Error::Msg(format!("{base}: {e}")))?;
                let conflict = entry.disagreement_with(&parsed);
                if !conflict.is_empty() {
                    return Err(
                        format!("{base}: contradictory descriptor fields {conflict:?}").into(),
                    );
                }
            } else if entry.convrot == Some(true)
                || matches!(
                    entry.format,
                    gen_core::comfy_quant::ComfyQuantFormat::Float8E4M3Fn
                        | gen_core::comfy_quant::ComfyQuantFormat::Float8E5M2
                        | gen_core::comfy_quant::ComfyQuantFormat::Mxfp8
                )
            {
                let d = entry
                    .into_complete()
                    .map_err(|e| crate::Error::Msg(format!("{base}: {e}")))?;
                let blob = if let Some(group) = d.rotation_group {
                    descriptor_array(group)
                } else {
                    crate::fp8::descriptor_array(
                        d.format,
                        d.full_precision_matrix_mult,
                        d.orig_shape.as_deref(),
                    )
                };
                tensors.insert(key, blob);
            }
        }
    }
    Ok(())
}

/// Compact native integer base, shareable by adapter-bearing projections.
#[derive(Clone, Debug)]
pub struct ConvRotLinear {
    descriptor: ConvRotInt8,
    codes: Array,
    scales: Array,
    bias: Option<Array>,
    rotation: Array,
}

impl ConvRotLinear {
    /// Slice output rows without changing signed codes, bias, row scales or rotation.
    pub fn slice_rows(&self, lo: i32, hi: i32) -> Result<Self> {
        use mlx_rs::ops::indexing::IndexOp;
        if lo < 0 || hi <= lo || hi as usize > self.descriptor.out_features {
            return Err("ConvRot row slice is outside the stored projection".into());
        }
        let n = (hi - lo) as usize;
        let layout = ConvRotInt8::from_geometry(
            self.descriptor.rotation_group,
            &[n, self.descriptor.in_features],
            &[n, 1],
        )
        .map_err(crate::Error::Msg)?;
        Ok(Self {
            descriptor: layout,
            codes: self.codes.index((lo..hi, ..)),
            scales: self.scales.index((lo..hi, ..)),
            bias: self.bias.as_ref().map(|b| b.index(lo..hi)),
            rotation: self.rotation.clone(),
        })
    }

    pub fn bias(&self) -> Option<&Array> {
        self.bias.as_ref()
    }

    pub fn set_bias(&mut self, bias: Array) -> Result<()> {
        if bias.shape() != [self.descriptor.out_features as i32]
            || !matches!(
                bias.dtype(),
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            )
        {
            return Err("ConvRot bias does not match the output width".into());
        }
        self.bias = Some(bias);
        Ok(())
    }

    pub fn scales(&self) -> &Array {
        &self.scales
    }

    pub fn rotation(&self) -> &Array {
        &self.rotation
    }

    pub fn push_arrays(&self, out: &mut Vec<Array>) {
        out.extend([
            self.codes.clone(),
            self.scales.clone(),
            self.rotation.clone(),
        ]);
        out.extend(self.bias.iter().cloned());
    }

    /// Validated geometry cannot be mutated independently of the resident buffers.
    pub fn descriptor(&self) -> ConvRotInt8 {
        self.descriptor
    }

    /// Admit only a complete authoritative descriptor plus I8/F32 operands. No dequantization.
    pub fn from_weights(
        w: &(impl MlxWeightSource + ?Sized),
        base: &str,
        bias: bool,
    ) -> Result<Option<Self>> {
        Self::from_key(w, base, &format!("{base}.weight"), bias)
    }

    /// Bind the physical weight key for architectures with nonstandard names.
    pub fn from_key(
        w: &(impl MlxWeightSource + ?Sized),
        base: &str,
        weight_key: &str,
        bias: bool,
    ) -> Result<Option<Self>> {
        let dk = format!("{base}.comfy_quant");
        let sk = format!("{base}.weight_scale");
        let require = |key: &str| {
            w.get(key)
                .ok_or_else(|| crate::Error::Msg(format!("missing tensor {key}")))
        };
        let codes = require(weight_key)?;
        let planned = w.planned_geometry(base);
        let declaration = if planned.is_none() {
            if let Some(payload) = w.get(&dk) {
                if payload.dtype() != Dtype::Uint8
                    || payload.ndim() != 1
                    || payload.size() > MAX_DESCRIPTOR_BYTES
                {
                    return Err(format!("{base}: .comfy_quant must be a bounded U8 vector").into());
                }
                payload.eval()?;
                let descriptor =
                    gen_core::comfy_quant::parse_comfy_quant_descriptor(payload.as_slice::<u8>())
                        .map_err(|e| crate::Error::Msg(format!("{base}: {e}")))?;
                if descriptor.format != gen_core::comfy_quant::ComfyQuantFormat::ConvRotInt8 {
                    return Ok(None);
                }
                descriptor.rotation_group
            } else {
                if codes.dtype() == Dtype::Int8 {
                    return Err(format!(
                        "{base}: INT8 requires an authoritative .comfy_quant descriptor"
                    )
                    .into());
                }
                return Ok(None);
            }
        } else {
            planned.map(|g| g.rotation_group)
        };
        let group = declaration
            .ok_or_else(|| crate::Error::Msg(format!("{base}: missing ConvRot group")))?;
        if w.get(&format!("{base}.scales")).is_some() || w.get(&format!("{base}.biases")).is_some()
        {
            return Err(format!("{base}: contradictory affine and ConvRot operands").into());
        }
        let scales = require(&sk)?;
        if codes.dtype() != Dtype::Int8 || scales.dtype() != Dtype::Float32 {
            return Err(
                format!("{base}: ConvRot requires original I8 codes and F32 scales").into(),
            );
        }
        scales.eval()?;
        let shape: Vec<usize> = codes.shape().iter().map(|&v| v as usize).collect();
        let scale_shape: Vec<usize> = scales.shape().iter().map(|&v| v as usize).collect();
        let descriptor =
            ConvRotInt8::from_geometry(group, &shape, &scale_shape).map_err(crate::Error::Msg)?;
        if planned.is_some_and(|d| d != descriptor) {
            return Err(format!("{base}: ConvRot operands changed since planning").into());
        }
        descriptor
            .validate_scales(scales.as_slice::<f32>())
            .map_err(crate::Error::Msg)?;
        let bias = if bias {
            Some(require(&format!("{base}.bias"))?.clone())
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
        let rotation = w
            .rotation_cache()
            .borrow_mut()
            .entry(g)
            .or_insert_with(|| {
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
            })
            .clone();
        Ok(Some(Self {
            descriptor,
            codes: codes.clone(),
            scales: scales.clone(),
            bias,
            rotation,
        }))
    }

    /// Validated activation geometry plus the source-dtype rotation (Kitchen's
    /// `_rotate_activation`, NOT a permanent float weight), shared by [`Self::forward`] and the
    /// test-only reference.
    fn rotate(&self, x: &Array) -> Result<(Array, usize)> {
        let d = self.descriptor;
        if x.shape().last() != Some(&(d.in_features as i32))
            || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err("ConvRot INT8 activation shape/dtype mismatch".into());
        }
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
        let (m, k, g) = (rows as i32, d.in_features as i32, d.rotation_group as i32);
        let rotated =
            matmul(&x.reshape(&[-1, g])?, &self.rotation.as_dtype(x.dtype())?)?.reshape(&[m, k])?;
        Ok((rotated, rows))
    }

    /// The F32 epilogue bias: the stored bias rounded to the activation dtype first, as the S4
    /// ConvRot authority adds it.
    fn epilogue_bias(&self, x: &Array, bias: Option<&Array>) -> Result<Array> {
        Ok(match bias {
            Some(b) => b.as_dtype(x.dtype())?.as_dtype(Dtype::Float32)?,
            None => mlx_rs::ops::zeros::<f32>(&[self.descriptor.out_features as i32])?,
        })
    }

    fn output_shape(&self, x: &Array) -> Vec<i32> {
        let mut shape = x.shape().to_vec();
        *shape.last_mut().unwrap() = self.descriptor.out_features as i32;
        shape
    }

    /// Native signed integer dot, FP32 row-scale epilogue, one output rounding after bias.
    ///
    /// The dot is [`crate::int8_gemm`]'s tiled kernel: exact int32 accumulation through windowed
    /// FP32 `simdgroup_matrix` products, bit-identical to the serial reference (that module states
    /// the exactness bound).
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_with_bias(x, self.bias.as_ref())
    }

    /// Bind an architecture's external bias in the same F32 epilogue as a stored projection bias.
    pub fn forward_with_bias(&self, x: &Array, bias: Option<&Array>) -> Result<Array> {
        if bias.is_some_and(|b| {
            b.shape() != [self.descriptor.out_features as i32]
                || !matches!(b.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        }) {
            return Err("ConvRot bias requires one floating value per output row".into());
        }
        let d = self.descriptor;
        let (rotated, rows) = self.rotate(x)?;
        let (codes, scales) = crate::int8_gemm::quantize_rows(
            &rotated,
            rows,
            d.in_features,
            x.dtype(),
            crate::int8_gemm::Numerator::ConvRot,
        )?;
        let bias = self.epilogue_bias(x, bias)?;
        let y = crate::int8_gemm::project(
            &codes,
            &scales,
            crate::int8_gemm::Weight::ConvRot { codes: &self.codes },
            &self.scales,
            &bias,
            rows,
            d.out_features,
            d.in_features,
            x.dtype(),
        )?;
        Ok(y.reshape(&self.output_shape(x))?)
    }

    /// The original serial kernels (sc-26683): the parity reference for [`Self::forward`]. Test
    /// support only; it is orders of magnitude slower and never on a production route.
    #[cfg(any(test, feature = "test-support"))]
    pub fn forward_reference(&self, x: &Array) -> Result<Array> {
        let d = self.descriptor;
        let (rotated, rows) = self.rotate(x)?;
        let (m, k, n) = (rows as i32, d.in_features as i32, d.out_features as i32);
        let bias = self.epilogue_bias(x, self.bias.as_ref())?;
        reference_kernels(|quant, project| {
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
            Ok(out.remove(0).reshape(&self.output_shape(x))?)
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
