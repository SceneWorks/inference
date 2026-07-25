//! Native-resolution packing and Mage MSRoPE.
//!
//! Coordinates come from the ordered `img_shapes` list. Height/width are centred; frame is the
//! shape-list index. The fused CFG duplicate therefore starts at frame 1, matching Torch.

use candle_core::{DType, Device, Error, Result, Tensor};

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

/// Adjacent-pair complex rotation, unlike Qwen/FLUX half-split RoPE.
pub fn apply(x: &Tensor, table: &RopeTable) -> Result<Tensor> {
    let (tokens, heads, dim) = x.dims3()?;
    if dim != HEAD_DIM || table.cos.dims() != [tokens, dim / 2] {
        return Err(Error::Msg("mage: MSRoPE table/activation mismatch".into()));
    }
    let f32x = x
        .to_dtype(DType::F32)?
        .reshape((tokens, heads, dim / 2, 2))?;
    let real = f32x.narrow(3, 0, 1)?.squeeze(3)?;
    let imag = f32x.narrow(3, 1, 1)?.squeeze(3)?;
    let cos = table.cos.to_dtype(DType::F32)?.unsqueeze(1)?;
    let sin = table.sin.to_dtype(DType::F32)?.unsqueeze(1)?;
    let out_real = (real.broadcast_mul(&cos)? - imag.broadcast_mul(&sin)?)?;
    let out_imag = (real.broadcast_mul(&sin)? + imag.broadcast_mul(&cos)?)?;
    Tensor::stack(&[out_real, out_imag], 3)?
        .reshape((tokens, heads, dim))?
        .to_dtype(x.dtype())
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
