//! Native MLX projection layers shared by media and language models.

pub mod convrot_int8;
pub mod fp8;
pub mod int8_gemm;
pub mod source;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),
    #[error("{0}")]
    Msg(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Self::Msg(value.to_owned())
    }
}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Self::Msg(value)
    }
}

pub use convrot_int8::MlxWeightSource;
pub use gen_core::comfy_quant::ComfyQuantFormat;

#[derive(Clone, Debug)]
pub enum NativeProjection {
    ConvRot(convrot_int8::ConvRotLinear),
    Fp8(fp8::Fp8Linear),
}

impl NativeProjection {
    pub fn from_weights(
        w: &(impl MlxWeightSource + ?Sized),
        base: &str,
        bias: bool,
    ) -> Result<Option<Self>> {
        Self::from_key(w, base, &format!("{base}.weight"), bias)
    }
    pub fn from_key(
        w: &(impl MlxWeightSource + ?Sized),
        base: &str,
        key: &str,
        bias: bool,
    ) -> Result<Option<Self>> {
        let mut native = Self::bind_key(w, base, key, bias)?;
        if let (Some(bound), Some(observation)) = (native.as_mut(), w.observation(key)) {
            match bound {
                Self::Fp8(p) => p.observe(observation)?,
                Self::ConvRot(p) => p.observe(observation)?,
            }
        }
        Ok(native)
    }
    fn bind_key(
        w: &(impl MlxWeightSource + ?Sized),
        base: &str,
        key: &str,
        bias: bool,
    ) -> Result<Option<Self>> {
        if let Some(p) = w.planned_fp8(base) {
            let mut p = p.clone();
            w.require(key)?;
            w.get(&format!("{base}.weight_scale"));
            w.get(&format!("{base}.input_scale"));
            if bias {
                p.set_bias(w.require(&format!("{base}.bias"))?.clone())?;
            }
            return Ok(Some(Self::Fp8(p)));
        }
        if let Some(blob) = w.get(&format!("{base}.comfy_quant")) {
            let desc = convrot_int8::descriptor(blob, base)?;
            use gen_core::comfy_quant::ComfyQuantFormat::*;
            if matches!(desc.format, Float8E4M3Fn | Float8E5M2 | Fp8Rowwise | Mxfp8) {
                if desc.full_precision_matrix_mult {
                    return Err(format!("{base}: full precision FP8 requires the dense decode selected by the checkpoint").into());
                }
                let codes = w.require(key)?.clone();
                let shape: Vec<_> = codes.shape().iter().map(|&n| n as usize).collect();
                let logical = if desc.format == Mxfp8 {
                    desc.orig_shape
                        .as_deref()
                        .ok_or("MXFP8 requires architecture-declared logical dimensions")?
                } else {
                    &shape
                };
                let scale = w.require(&format!("{base}.weight_scale"))?.clone();
                let sx = w.get(&format!("{base}.input_scale")).cloned();
                let bias = bias
                    .then(|| w.require(&format!("{base}.bias")).cloned())
                    .transpose()?;
                return Ok(Some(Self::Fp8(fp8::Fp8Linear::new(
                    desc.format,
                    codes,
                    Some(scale),
                    sx,
                    logical,
                    bias,
                )?)));
            }
            if desc.format != ConvRotInt8 {
                return Ok(None);
            }
        }
        Ok(convrot_int8::ConvRotLinear::from_key(w, base, key, bias)?.map(Self::ConvRot))
    }
    pub fn shape(&self) -> [i32; 2] {
        match self {
            Self::ConvRot(p) => [
                p.descriptor().out_features as i32,
                p.descriptor().in_features as i32,
            ],
            Self::Fp8(p) => p.shape(),
        }
    }
    pub fn rows(&self, ids: &mlx_rs::Array) -> Result<mlx_rs::Array> {
        self.rows_as(ids, mlx_rs::Dtype::Bfloat16)
    }
    pub fn rows_as(&self, ids: &mlx_rs::Array, dtype: mlx_rs::Dtype) -> Result<mlx_rs::Array> {
        match self {
            Self::ConvRot(p) => p.rows_as(ids, dtype),
            Self::Fp8(p) => p.rows_as(ids, dtype),
        }
    }
    pub fn materialize(&self) -> Result<()> {
        match self {
            Self::ConvRot(p) => p.materialize(),
            Self::Fp8(p) => p.materialize(),
        }
    }
    pub fn operands(&self) -> Vec<(&'static str, &mlx_rs::Array)> {
        match self {
            Self::ConvRot(p) => vec![
                ("weight", p.codes()),
                ("weight_scale", p.scales()),
                ("rotation", p.rotation()),
            ],
            Self::Fp8(p) => vec![
                ("weight", p.codes()),
                ("weight_scale", p.scales()),
                ("input_scale", p.input_scale()),
            ],
        }
    }
    pub fn set_bias(&mut self, b: mlx_rs::Array) -> Result<()> {
        match self {
            Self::ConvRot(p) => p.set_bias(b),
            Self::Fp8(p) => p.set_bias(b),
        }
    }
    pub fn slice_rows(&self, lo: i32, hi: i32) -> Result<Self> {
        Ok(match self {
            Self::ConvRot(p) => Self::ConvRot(p.slice_rows(lo, hi)?),
            Self::Fp8(p) => Self::Fp8(p.slice_rows(lo, hi)?),
        })
    }
    pub fn push_arrays(&self, out: &mut Vec<mlx_rs::Array>) {
        match self {
            Self::ConvRot(p) => p.push_arrays(out),
            Self::Fp8(p) => p.push_arrays(out),
        }
    }
    pub fn forward_with_bias(
        &self,
        x: &mlx_rs::Array,
        b: Option<&mlx_rs::Array>,
    ) -> Result<mlx_rs::Array> {
        match self {
            Self::ConvRot(p) => p.forward_with_bias(x, b),
            Self::Fp8(p) => p.forward_with_bias(x, b),
        }
    }
    pub fn forward(&self, x: &mlx_rs::Array) -> Result<mlx_rs::Array> {
        match self {
            Self::ConvRot(p) => p.forward(x),
            Self::Fp8(p) => p.forward(x),
        }
    }
}
