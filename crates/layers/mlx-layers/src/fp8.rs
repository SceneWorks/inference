//! Projection over original Comfy E4M3FN, E5M2 or MXFP8 bytes. Scalar activation
//! scales are fixed by the checkpoint. MXFP8 uses one E8M0 scale per 32 input channels.
use crate::{Error, Result};
use gen_core::comfy_quant::{validate_mxfp8_geometry, ComfyQuantFormat};
use mlx_rs::{fast::MetalKernel, ops::indexing::IndexOp, Array, Dtype};
use std::cell::RefCell;

const HEADER: &str = include_str!("fp8.metal");

const ACTIVATION: &str = r#"
const uint block = threadgroup_position_in_grid.x;
const uint row = threadgroup_position_in_grid.y;
const uint col = block * 32u + thread_index_in_simdgroup;
const float xval = col < uint(K) ? float(x[row * uint(K) + col]) : 0.0f;
float rounded;
if (MX) {
    float required = max(simd_max(abs(xval)) / 448.0f, 0x1p-127f);
    int exponent;
    float mantissa = frexp(required, exponent);
    exponent = clamp(exponent - int(mantissa == 0.5f), -127, 127);
    rounded = ldexp(sc_round_fp8(ldexp(xval, -exponent), false), exponent);
} else {
    rounded = sc_round_fp8(xval / sx[0], E5);
}
if (col < uint(K)) q[row * uint(K) + col] = rounded;
"#;

// Four SIMD groups compute a 16-by-16 tile using 8-by-8 fragments. The tail
// is zero-padded in threadgroup storage only. No dense weight graph is built.
const PROJECT: &str = r#"
threadgroup float act[16 * 32], weight[16 * 32], tile[16 * 16];
uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
uint row0 = threadgroup_position_in_grid.y * 16u;
uint col0 = threadgroup_position_in_grid.x * 16u;
simdgroup_float8x8 sum = make_filled_simdgroup_matrix<float,8,8>(0.0f);
for (uint block = 0; block < (uint(K) + 31u)/32u; ++block) {
    for (uint i = tid; i < 512u; i += 128u) {
        uint r = i/32u, k = block*32u + i%32u;
        uint ar = row0+r, wr = col0+r;
        act[i] = ar < uint(M) && k < uint(K) ? q[ar*uint(K)+k] : 0.0f;
        float value = 0.0f;
        if (wr < uint(N) && k < uint(K)) {
            value = sc_decode_fp8(w[wr*uint(STRIDE)+k], E5);
            if (MX) {
                uint sr = wr + uint(SCALE_ROW);
                uint atom = (sr/128u)*uint(SCALE_TILES) + block/4u;
                uint index = atom*512u + (sr%32u)*16u + (sr%128u)/32u*4u + block%4u;
                uint e = uint(sw[index]);
                value = e == 255u ? NAN : ldexp(value, int(e)-127);
            }
        }
        weight[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 32u; k+=8u) {
        simdgroup_float8x8 a,b;
        simdgroup_load(a, act+(sg/2u)*8u*32u+k,32);
        simdgroup_load(b, weight+(sg%2u)*8u*32u+k,32,ulong2(0,0),true);
        simdgroup_multiply_accumulate(sum,a,b,sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(sum, tile+(sg/2u)*8u*16u+(sg%2u)*8u,16);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i=tid; i < 256u; i+=128u) {
    uint row=row0+i/16u, col=col0+i%16u;
    if (row < uint(M) && col < uint(N))
        y[row*uint(N)+col] = bfloat(MX ? tile[i] : (tile[i]*sx[0])*float(sw[0]));
}
"#;

thread_local! {
    static KERNELS: RefCell<Option<(MetalKernel,MetalKernel)>> = const {RefCell::new(None)};
}

#[derive(Debug, Clone)]
pub struct Fp8Linear {
    pub format: ComfyQuantFormat,
    codes: Array,
    scales: Array,
    input_scale: Array,
    bias: Option<Array>,
    shape: [i32; 2],
    scale_row: i32,
}

impl Fp8Linear {
    pub fn new(
        format: ComfyQuantFormat,
        codes: Array,
        scale: Option<Array>,
        input_scale: Option<Array>,
        logical: &[usize],
        bias: Option<Array>,
    ) -> Result<Self> {
        if codes.dtype() != Dtype::Uint8
            || codes.ndim() != 2
            || logical.len() != 2
            || logical.contains(&0)
        {
            return Err(
                "native FP8 requires a rank-two byte matrix and positive logical dimensions".into(),
            );
        }
        let stored: Vec<_> = codes.shape().iter().map(|&v| v as usize).collect();
        let scales = match format {
            ComfyQuantFormat::Mxfp8 => {
                let scale = scale.ok_or("MXFP8 requires E8M0 scale atoms")?;
                let grid: Vec<_> = scale.shape().iter().map(|&v| v as usize).collect();
                validate_mxfp8_geometry(&stored, &grid, Some(logical))
                    .map_err(|e| Error::Msg(e.to_string()))?;
                if scale.dtype() != Dtype::Uint8 || input_scale.is_some() {
                    return Err(
                        "MXFP8 requires E8M0 scales and dynamic block activation scaling".into(),
                    );
                }
                scale
            }
            ComfyQuantFormat::Float8E4M3Fn | ComfyQuantFormat::Float8E5M2 => {
                if logical != stored {
                    return Err("scalar FP8 logical and stored dimensions differ".into());
                }
                let scales = scale.unwrap_or_else(unit_scale);
                scalar(&scales, false)?;
                scales
            }
            _ => return Err("unsupported FP8 matrix encoding".into()),
        };
        let sx = input_scale.unwrap_or_else(unit_scale);
        scalar(&sx, true)?;
        let mut this = Self {
            format,
            codes,
            scales,
            input_scale: sx,
            bias: None,
            shape: [logical[0] as i32, logical[1] as i32],
            scale_row: 0,
        };
        if let Some(bias) = bias {
            this.set_bias(bias)?;
        }
        Ok(this)
    }

    pub fn shape(&self) -> [i32; 2] {
        self.shape
    }
    pub fn scales(&self) -> &Array {
        &self.scales
    }
    pub fn input_scale(&self) -> &Array {
        &self.input_scale
    }
    pub fn codes(&self) -> &Array {
        &self.codes
    }
    pub fn bias(&self) -> Option<&Array> {
        self.bias.as_ref()
    }
    pub fn set_bias(&mut self, b: Array) -> Result<()> {
        if b.shape() != [self.shape[0]]
            || !matches!(b.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err("native FP8 bias must be a floating output vector".into());
        }
        self.bias = Some(b);
        Ok(())
    }
    pub fn push_arrays(&self, out: &mut Vec<Array>) {
        out.extend([
            self.codes.clone(),
            self.scales.clone(),
            self.input_scale.clone(),
        ]);
        out.extend(self.bias.iter().cloned());
    }
    pub fn materialize(&self) -> Result<()> {
        let mut a = Vec::new();
        self.push_arrays(&mut a);
        mlx_rs::transforms::eval(a.iter())?;
        Ok(())
    }
    pub fn slice_rows(&self, lo: i32, hi: i32) -> Result<Self> {
        if lo < 0 || hi <= lo || hi > self.shape[0] {
            return Err("invalid native FP8 row slice".into());
        }
        let mut sliced = self.clone();
        sliced.codes = self.codes.index((lo..hi, ..));
        sliced.bias = self.bias.as_ref().map(|b| b.index(lo..hi));
        sliced.shape[0] = hi - lo;
        sliced.scale_row += lo;
        Ok(sliced)
    }
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_with_bias(x, self.bias.as_ref())
    }
    pub fn forward_with_bias(&self, x: &Array, bias: Option<&Array>) -> Result<Array> {
        let [n, k] = self.shape;
        if x.shape().last() != Some(&k)
            || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        {
            return Err("native FP8 activation width/dtype mismatch".into());
        }
        let m = i32::try_from(x.size() / (k as usize))
            .map_err(|_| Error::Msg("native FP8 row count exceeds Metal indexing".into()))?;
        if m == 0
            || x.size() > u32::MAX as usize
            || self.codes.size() > u32::MAX as usize
            || (u64::from(m as u32) * n as u64) > u32::MAX as u64
        {
            return Err("native FP8 matrix exceeds Metal index geometry".into());
        }
        let mx = i32::from(self.format == ComfyQuantFormat::Mxfp8);
        let e5 = i32::from(self.format == ComfyQuantFormat::Float8E5M2);
        let xb = (k - 1) / 32 + 1;
        let ng = ((n - 1) / 16 + 1)
            .checked_mul(128)
            .ok_or("FP8 grid overflow")?;
        let mg = (m - 1) / 16 + 1;
        mlx_rs::transforms::compile::prepare_retained_compilation_thread();
        let y = KERNELS.with(|cell| -> Result<Array> {
            let mut cache = cell.borrow_mut();
            if cache.is_none() {
                *cache = Some((
                    MetalKernel::with_options(
                        "sc_fp8_activation_v1",
                        &["x", "sx"],
                        &["q"],
                        ACTIVATION,
                        HEADER,
                        true,
                        false,
                    )?,
                    MetalKernel::with_options(
                        "sc_fp8_project_v1",
                        &["q", "w", "sw", "sx"],
                        &["y"],
                        PROJECT,
                        HEADER,
                        true,
                        false,
                    )?,
                ));
            }
            let (quant, project) = cache.as_ref().expect("initialized above");
            let q = quant
                .apply()
                .inputs([x, &self.input_scale])
                .output_shape([m, k], Dtype::Float32)
                .template_arg("K", k)
                .template_arg("MX", mx)
                .template_arg("E5", e5)
                .grid(
                    xb.checked_mul(32).ok_or("FP8 activation grid overflow")?,
                    m,
                    1,
                )
                .thread_group(32, 1, 1)
                .run()?
                .remove(0);
            let tiles = if mx != 0 {
                self.scales.shape()[1] / 4
            } else {
                0
            };
            Ok(project
                .apply()
                .inputs([&q, &self.codes, &self.scales, &self.input_scale])
                .output_shape([m, n], Dtype::Bfloat16)
                .template_arg("M", m)
                .template_arg("N", n)
                .template_arg("K", k)
                .template_arg("STRIDE", self.codes.shape()[1])
                .template_arg("SCALE_TILES", tiles)
                .template_arg("SCALE_ROW", self.scale_row)
                .template_arg("MX", mx)
                .template_arg("E5", e5)
                .grid(ng, mg, 1)
                .thread_group(128, 1, 1)
                .run()?
                .remove(0))
        })?;
        let mut result_shape = x.shape().to_vec();
        *result_shape.last_mut().expect("validated rank") = n;
        let y = y.reshape(&result_shape)?;
        Ok(match bias {
            Some(b) => mlx_rs::ops::add(&y, b.as_dtype(y.dtype())?)?,
            None => y,
        })
    }
}

fn scalar(a: &Array, positive: bool) -> Result<()> {
    if a.dtype() != Dtype::Float32 || a.size() != 1 {
        return Err("FP8 scale must be one F32 value".into());
    }
    a.eval()?;
    let value = a.as_slice::<f32>()[0];
    if !value.is_finite() || (positive && value <= 0.) {
        return Err("invalid FP8 scale".into());
    }
    Ok(())
}

pub fn descriptor_array(format: ComfyQuantFormat, full: bool, shape: Option<&[usize]>) -> Array {
    let shape = shape
        .map(|s| format!(r#","orig_shape":{s:?}"#))
        .unwrap_or_default();
    let desc = format!(
        r#"{{"format":"{}","full_precision_matrix_mult":{}{shape}}}"#,
        format.as_str(),
        full
    );
    Array::from_slice(desc.as_bytes(), &[desc.len() as i32])
}

/// Convert only the weight rows required by an explicitly dense operator.
pub fn decoded_codes(codes: &Array, format: ComfyQuantFormat) -> Result<Array> {
    if codes.dtype() != Dtype::Uint8 {
        return Err("FP8 source must retain its original bytes".into());
    }
    if format == ComfyQuantFormat::Float8E5M2 {
        Ok(mlx_rs::ops::multiply(
            codes.as_dtype(Dtype::Uint16)?,
            Array::from_slice(&[256_u16], &[]),
        )?
        .view_dtype(Dtype::Float16)?
        .as_dtype(Dtype::Float32)?)
    } else {
        Ok(codes.from_fp8(Dtype::Float32)?)
    }
}

impl Fp8Linear {
    pub fn dense_weight(&self) -> Result<Array> {
        let decoded = decoded_codes(&self.codes, self.format)?;
        let values = if self.format == ComfyQuantFormat::Mxfp8 {
            let rows = self.codes.shape()[0];
            let columns = self.codes.shape()[1];
            let block_count = columns / 32;
            let scale_shape = self.scales.shape();
            let rt = scale_shape[0] / 128;
            let ct = scale_shape[1] / 4;
            let grid = self
                .scales
                .reshape(&[rt * ct, 32, 4, 4])?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[rt, ct, 128, 4])?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(scale_shape)?;
            let grid = grid
                .index((self.scale_row..self.scale_row + rows, ..block_count))
                .as_dtype(Dtype::Float32)?;
            let scales = mlx_rs::ops::power(
                Array::from_f32(2.0),
                mlx_rs::ops::subtract(&grid, Array::from_f32(127.0))?,
            )?;
            let scales = mlx_rs::ops::r#where(
                mlx_rs::ops::eq(&grid, Array::from_f32(255.))?,
                Array::from_f32(f32::NAN),
                scales,
            )?;
            mlx_rs::ops::multiply(
                decoded.reshape(&[rows, block_count, 32])?,
                scales.reshape(&[rows, block_count, 1])?,
            )?
            .reshape(&[rows, columns])?
        } else {
            mlx_rs::ops::multiply(decoded, &self.scales)?
        };
        Ok(values
            .index((..self.shape[0], ..self.shape[1]))
            .as_dtype(Dtype::Bfloat16)?)
    }
}

/// Reconcile original file dtype with MLX's U8 storage view before any provider renames its
/// keys. Plain FP8 casts have a unit scale; 1D norms, biases and true convolution filters use
/// their dense operator contract. A described projection must have a genuine matching file dtype.
pub fn prepare_sources(
    tensors: &mut std::collections::HashMap<String, Array>,
    paths: &[std::path::PathBuf],
    tables: &[String],
) -> Result<()> {
    use gen_core::weightsmeta::Dtype as Stored;
    use ComfyQuantFormat::{Float8E4M3Fn, Float8E5M2, Mxfp8};
    crate::convrot_int8::attach_metadata(tensors, tables)?;
    if !tensors.values().any(|v| v.dtype() == Dtype::Uint8) {
        return Ok(());
    }
    for path in paths {
        let headers = gen_core::safetensors_path_tensor_headers(path)
            .map_err(|e| Error::Msg(e.to_string()))?;
        for h in headers {
            let Some(array) = tensors.get(&h.name).cloned() else {
                continue;
            };
            let format = match h.dtype {
                Stored::F8_E4M3 => Float8E4M3Fn,
                Stored::F8_E5M2 => Float8E5M2,
                _ => continue,
            };
            // The low-precision scale of another codec is never a scalar FP8 weight.
            if h.name.ends_with(".weight_scale") || h.name.ends_with(".input_scale") {
                continue;
            }
            let Some(base) = h
                .name
                .strip_suffix(".weight")
                .filter(|_| h.shape.len() == 2)
            else {
                tensors.insert(
                    h.name,
                    decoded_codes(&array, format)?.as_dtype(Dtype::Bfloat16)?,
                );
                continue;
            };
            let dk = format!("{base}.comfy_quant");
            let sk = format!("{base}.weight_scale");
            let ik = format!("{base}.input_scale");
            if let Some(blob) = tensors.get(&dk) {
                let desc = crate::convrot_int8::descriptor(blob, base)?;
                if desc.format != format && !(desc.format == Mxfp8 && format == Float8E4M3Fn) {
                    return Err(format!("{base}: FP8 dtype contradicts descriptor").into());
                }
                if desc.full_precision_matrix_mult {
                    let sw = tensors
                        .get(&sk)
                        .cloned()
                        .ok_or_else(|| Error::Msg(format!("{base}: missing FP8 scale")))?;
                    let shape = desc.orig_shape.as_deref().unwrap_or(&h.shape);
                    let p = Fp8Linear::new(
                        desc.format,
                        array,
                        Some(sw),
                        tensors.get(&ik).cloned(),
                        shape,
                        None,
                    )?;
                    tensors.insert(h.name, p.dense_weight()?);
                    tensors.remove(&dk);
                    tensors.remove(&sk);
                    tensors.remove(&ik);
                }
            } else {
                if tensors.contains_key(&sk) || tensors.contains_key(&ik) {
                    return Err(
                        format!("{base}: FP8 scale requires an authoritative descriptor").into(),
                    );
                }
                tensors.insert(dk, descriptor_array(format, false, None));
                tensors.insert(sk, unit_scale());
            }
        }
    }
    Ok(())
}

pub(crate) fn unit_scale() -> Array {
    thread_local! {static UNIT: Array=Array::from_f32(1.0);}
    UNIT.with(Clone::clone)
}
