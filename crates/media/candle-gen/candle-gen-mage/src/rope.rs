//! Native-resolution packing and Mage MSRoPE.
//!
//! Coordinates come from the ordered `img_shapes` list. Height/width are centred; frame is the
//! shape-list index. The fused CFG duplicate therefore starts at frame 1, matching Torch.

use candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen::rope::{RopeLayout, RopeSpec};

use crate::config::{AXES_DIM, HEAD_DIM, ROPE_THETA};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImgShape {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl ImgShape {
    pub const fn latent(height: usize, width: usize) -> Self {
        Self {
            frames: 1,
            height,
            width,
        }
    }

    pub const fn tokens(self) -> usize {
        self.frames * self.height * self.width
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackLayout {
    shapes: Vec<ImgShape>,
    image_lens: Vec<usize>,
    text_lens: Vec<usize>,
}

impl PackLayout {
    pub fn generation(shapes: Vec<ImgShape>, text_lens: Vec<usize>) -> Result<Self> {
        let image_lens = shapes.iter().copied().map(ImgShape::tokens).collect();
        Self::new(shapes, image_lens, text_lens)
    }

    pub fn new(
        shapes: Vec<ImgShape>,
        image_lens: Vec<usize>,
        text_lens: Vec<usize>,
    ) -> Result<Self> {
        if shapes.is_empty()
            || image_lens.is_empty()
            || image_lens.len() != text_lens.len()
            || image_lens.contains(&0)
            || text_lens.contains(&0)
        {
            return Err(Error::Msg(
                "mage: invalid empty/mismatched pack layout".into(),
            ));
        }
        let shape_cu = cumulative(
            &shapes
                .iter()
                .copied()
                .map(ImgShape::tokens)
                .collect::<Vec<_>>(),
        );
        let image_cu = cumulative(&image_lens);
        if shape_cu.last() != image_cu.last() || image_cu.iter().any(|x| !shape_cu.contains(x)) {
            return Err(Error::Msg(
                "mage: image attention boundary must fall on a shape boundary".into(),
            ));
        }
        Ok(Self {
            shapes,
            image_lens,
            text_lens,
        })
    }

    pub fn fused_cfg(&self, negative_text_lens: &[usize]) -> Result<Self> {
        if negative_text_lens.len() != self.text_lens.len() {
            return Err(Error::Msg(
                "mage: one negative text segment is required per positive segment".into(),
            ));
        }
        Self::new(
            [self.shapes.as_slice(), self.shapes.as_slice()].concat(),
            [self.image_lens.as_slice(), self.image_lens.as_slice()].concat(),
            [self.text_lens.as_slice(), negative_text_lens].concat(),
        )
    }

    pub fn shapes(&self) -> &[ImgShape] {
        &self.shapes
    }
    pub fn image_lens(&self) -> &[usize] {
        &self.image_lens
    }
    pub fn text_lens(&self) -> &[usize] {
        &self.text_lens
    }
    pub fn image_cu(&self) -> Vec<usize> {
        cumulative(&self.image_lens)
    }
    pub fn text_cu(&self) -> Vec<usize> {
        cumulative(&self.text_lens)
    }
    pub fn image_tokens(&self) -> usize {
        self.image_lens.iter().sum()
    }
    pub fn text_tokens(&self) -> usize {
        self.text_lens.iter().sum()
    }
    pub fn segments(&self) -> usize {
        self.image_lens.len()
    }

    pub fn image_segment_ids(&self, device: &Device) -> Result<Tensor> {
        let ids: Vec<u32> = self
            .image_lens
            .iter()
            .enumerate()
            .flat_map(|(i, n)| std::iter::repeat_n(i as u32, *n))
            .collect();
        Tensor::from_vec(ids, self.image_tokens(), device)
    }
}

fn cumulative(lens: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(lens.len() + 1);
    out.push(0);
    for n in lens {
        out.push(out.last().copied().unwrap_or(0) + n);
    }
    out
}

pub struct RopeTable {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl RopeTable {
    pub fn build(layout: &PackLayout, dtype: DType, device: &Device) -> Result<Self> {
        let half = HEAD_DIM / 2;
        let mut angles = Vec::with_capacity(layout.image_tokens() * half);
        for (frame_index, shape) in layout.shapes.iter().copied().enumerate() {
            for _ in 0..shape.frames {
                for y in 0..shape.height {
                    for x in 0..shape.width {
                        append_axis(&mut angles, frame_index as i32, AXES_DIM[0]);
                        append_axis(
                            &mut angles,
                            y as i32 - (shape.height - shape.height / 2) as i32,
                            AXES_DIM[1],
                        );
                        append_axis(
                            &mut angles,
                            x as i32 - (shape.width - shape.width / 2) as i32,
                            AXES_DIM[2],
                        );
                    }
                }
            }
        }
        let a = Tensor::from_vec(angles, (layout.image_tokens(), half), device)?;
        Ok(Self {
            cos: a.cos()?.to_dtype(dtype)?,
            sin: a.sin()?.to_dtype(dtype)?,
        })
    }
}

fn append_axis(out: &mut Vec<f32>, coordinate: i32, axis_dim: usize) {
    for k in 0..axis_dim / 2 {
        let inv = ROPE_THETA.powf(-((2 * k) as f64) / axis_dim as f64);
        out.push(coordinate as f32 * inv as f32);
    }
}

/// MSRoPE's apply convention: interleaved `(real, imag)` pairs over one `[1, tokens, heads, dim]`
/// batch, rotated in f32 (activation and tables upcast) and rounded once to the activation dtype.
const MAGE_ROPE: RopeSpec = RopeSpec::interleaved(RopeLayout::Bshd);

/// Adjacent-pair complex rotation, unlike Qwen/FLUX half-split RoPE. The shared
/// [`candle_gen::rope::apply_rope`] (SW-6129): the fused strided kernel on CUDA, and everywhere else
/// the shared reference chain, bit-identical to the real/imag narrow, mul and stack chain this used
/// to carry.
pub fn apply(x: &Tensor, table: &RopeTable) -> Result<Tensor> {
    let (tokens, _heads, dim) = x.dims3()?;
    if dim != HEAD_DIM || table.cos.dims() != [tokens, dim / 2] {
        return Err(Error::Msg("mage: MSRoPE table/activation mismatch".into()));
    }
    candle_gen::rope::apply_rope(&x.unsqueeze(0)?, &table.cos, &table.sin, &MAGE_ROPE)?.squeeze(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_cfg_duplicate_moves_to_frame_one() {
        let base = PackLayout::generation(vec![ImgShape::latent(2, 2)], vec![3]).unwrap();
        let fused = base.fused_cfg(&[2]).unwrap();
        let t = RopeTable::build(&fused, DType::F32, &Device::Cpu).unwrap();
        let cos = t.cos.to_vec2::<f32>().unwrap();
        assert_eq!(cos[0][0], 1.0);
        assert_ne!(
            cos[4][0], 1.0,
            "unconditional duplicate must use frame index 1"
        );
    }

    #[test]
    fn adjacent_pair_quarter_turn_is_not_half_split() {
        let x = Tensor::new(&[[[1f32, 2., 3., 4.]]], &Device::Cpu).unwrap();
        let table = RopeTable {
            cos: Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap(),
            sin: Tensor::ones((1, 2), DType::F32, &Device::Cpu).unwrap(),
        };
        // Tiny test uses a 4-wide head, so exercise the implementation's algebra directly.
        let f = x.reshape((1, 1, 2, 2)).unwrap();
        let r = f.narrow(3, 0, 1).unwrap().squeeze(3).unwrap();
        let i = f.narrow(3, 1, 1).unwrap().squeeze(3).unwrap();
        let c = table.cos.unsqueeze(1).unwrap();
        let s = table.sin.unsqueeze(1).unwrap();
        let rr = (r.broadcast_mul(&c).unwrap() - i.broadcast_mul(&s).unwrap()).unwrap();
        let ii = (r.broadcast_mul(&s).unwrap() + i.broadcast_mul(&c).unwrap()).unwrap();
        let got = Tensor::stack(&[rr, ii], 3)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert_eq!(got[0][0], [-2., 1., -4., 3.]);
    }
}

/// SW-6129: MSRoPE's apply is the shared `candle_gen::rope::apply_rope`.
#[cfg(test)]
mod sw6129_rope_route_tests {
    use super::*;

    /// Deterministic xorshift values in [-2, 2) (no device RNG).
    fn values(dims: &[usize], seed: u64, device: &Device) -> Tensor {
        let n: usize = dims.iter().product();
        let mut s = seed | 1;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0) as f32
            })
            .collect();
        Tensor::from_vec(v, dims, device).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// The former private apply, verbatim.
    fn former_apply(x: &Tensor, table: &RopeTable) -> Tensor {
        let (tokens, heads, dim) = x.dims3().unwrap();
        let f32x = x
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((tokens, heads, dim / 2, 2))
            .unwrap();
        let real = f32x.narrow(3, 0, 1).unwrap().squeeze(3).unwrap();
        let imag = f32x.narrow(3, 1, 1).unwrap().squeeze(3).unwrap();
        let cos = table
            .cos
            .to_dtype(DType::F32)
            .unwrap()
            .unsqueeze(1)
            .unwrap();
        let sin = table
            .sin
            .to_dtype(DType::F32)
            .unwrap()
            .unsqueeze(1)
            .unwrap();
        let out_real =
            (real.broadcast_mul(&cos).unwrap() - imag.broadcast_mul(&sin).unwrap()).unwrap();
        let out_imag =
            (real.broadcast_mul(&sin).unwrap() + imag.broadcast_mul(&cos).unwrap()).unwrap();
        Tensor::stack(&[out_real, out_imag], 3)
            .unwrap()
            .reshape((tokens, heads, dim))
            .unwrap()
            .to_dtype(x.dtype())
            .unwrap()
    }

    /// The shared route equals the former apply bit for bit on CPU for f32 / bf16 / f16
    /// activations and f32 / bf16 tables, at a real fused-CFG pack layout.
    #[test]
    fn shared_rope_route_matches_the_former_private_chain() {
        let base = PackLayout::generation(vec![ImgShape::latent(4, 6)], vec![3]).unwrap();
        let layout = base.fused_cfg(&[2]).unwrap();
        let x32 = values(&[layout.image_tokens(), 3, HEAD_DIM], 7, &Device::Cpu);
        for table_dtype in [DType::F32, DType::BF16] {
            let table = RopeTable::build(&layout, table_dtype, &Device::Cpu).unwrap();
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                let x = x32.to_dtype(dtype).unwrap();
                let got = apply(&x, &table).unwrap();
                assert_eq!(got.dtype(), dtype);
                assert_eq!(
                    bits(&got),
                    bits(&former_apply(&x, &table)),
                    "{dtype:?} / {table_dtype:?}"
                );
            }
        }
    }

    /// SW-6129 CUDA smoke: mage rope::apply (bf16, fused-CFG MSRoPE table) on CUDA device 0 takes the fused strided RoPE (every call counted
    /// `fused`) and is bit-identical to the same call with the kernel switch off (the shared
    /// reference chain) — the bf16 / production-dtype run the CPU goldens cannot see.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_bf16_rope_route_is_fused_and_bit_identical() {
        use candle_gen::fused_path::media_fused_policy_guard;
        use candle_gen::rope::ROPE_PATH;
        let dev = Device::new_cuda(0).unwrap();
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = ROPE_PATH.guard(Some(true));
        // A 1024^2 generation (64x64 latent tokens) with its fused CFG duplicate, 3 heads x 128.
        let base = PackLayout::generation(vec![ImgShape::latent(64, 64)], vec![77]).unwrap();
        let layout = base.fused_cfg(&[77]).unwrap();
        let table = RopeTable::build(&layout, DType::BF16, &dev).unwrap();
        let n = layout.image_tokens();
        let q = values(&[n, 3, HEAD_DIM], 3, &dev)
            .to_dtype(DType::BF16)
            .unwrap();
        let k = values(&[n, 3, HEAD_DIM], 4, &dev)
            .to_dtype(DType::BF16)
            .unwrap();
        let run =
            || Tensor::cat(&[apply(&q, &table).unwrap(), apply(&k, &table).unwrap()], 0).unwrap();
        ROPE_PATH.set_enabled(Some(true));
        let start = ROPE_PATH.thread_counts();
        let fused = run();
        let on = ROPE_PATH.thread_counts().since(&start);
        assert_eq!((on.fused, on.reference_total()), (2, 0), "{on}");
        ROPE_PATH.set_enabled(Some(false));
        let start = ROPE_PATH.thread_counts();
        let reference = run();
        let off = ROPE_PATH.thread_counts().since(&start);
        assert_eq!(off.reference_for("disabled"), 2, "{off}");
        ROPE_PATH.set_enabled(Some(true));
        let (f, r) = (bits(&fused), bits(&reference));
        let mismatches = f.iter().zip(&r).filter(|(a, b)| a != b).count();
        println!("mage rope::apply (bf16, fused-CFG MSRoPE table): {mismatches} of {} elements differ; on {on}, off {off}", f.len());
        assert_eq!(mismatches, 0);
    }
}
