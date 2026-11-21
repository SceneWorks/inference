//! Head-major coefficient planes for a predictor which multiplies unquantized features by
//! BF16-rounded weights. Packed FP8 planes retain the published E4M3 codes and F32 head scales.
use std::borrow::Cow;
#[derive(Clone)]
pub enum HeadwiseWeights {
    Bf16(Vec<u16>),
    Fp8 { codes: Vec<u8>, scales: Vec<f32> },
}
impl From<Vec<u16>> for HeadwiseWeights {
    fn from(v: Vec<u16>) -> Self {
        Self::Bf16(v)
    }
}
impl FromIterator<u16> for HeadwiseWeights {
    fn from_iter<T: IntoIterator<Item = u16>>(v: T) -> Self {
        Self::Bf16(v.into_iter().collect())
    }
}
impl HeadwiseWeights {
    pub fn from_fp8(codes: &[u8], scales: Vec<f32>, width: usize) -> crate::Result<Self> {
        if width == 0 || scales.is_empty() || scales.len().checked_mul(width) != Some(codes.len()) {
            return Err("invalid headwise FP8 shape".into());
        }
        for (row, &s) in codes.chunks_exact(width).zip(&scales) {
            if !s.is_finite() || s <= 0. {
                return Err("invalid headwise FP8 scale".into());
            }
            // Validate only code values that actually occur, including overflow on BF16 rounding.
            let mut used = [false; 256];
            for &byte in row {
                used[byte as usize] = true;
            }
            for (code, &present) in used.iter().enumerate() {
                if present && !coefficient(code as u8, s).is_finite() {
                    return Err("non-finite projection coefficient".into());
                }
            }
        }
        Ok(Self::Fp8 {
            codes: codes.to_vec(),
            scales,
        })
    }
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Bf16(v) => v.len() * 2,
            Self::Fp8 { codes, scales } => codes.len() + 4 * scales.len(),
        }
    }
    pub fn heads<'a>(&'a self, ids: &'a [i32], width: usize) -> HeadwiseProjection<'a> {
        HeadwiseProjection {
            plane: self,
            ids,
            width,
        }
    }
}
fn coefficient(code: u8, scale: f32) -> f32 {
    f32::from_bits(
        u32::from(crate::numeric::bf16_bits(
            crate::fp8_e4m3fn_to_f32(code) * scale,
        )) << 16,
    )
}
pub struct HeadwiseProjection<'a> {
    plane: &'a HeadwiseWeights,
    ids: &'a [i32],
    width: usize,
}
impl HeadwiseProjection<'_> {
    /// A contiguous head batch borrows original payload and calibration. An arbitrary gather
    /// keeps both in the same source representation and follows the requested head order.
    pub fn packed(&self) -> Option<(Cow<'_, [u8]>, Cow<'_, [f32]>)> {
        let HeadwiseWeights::Fp8 { codes, scales } = self.plane else {
            return None;
        };
        Some((
            gather(codes, self.ids, self.width),
            gather(scales, self.ids, 1),
        ))
    }
    pub fn dense(&self) -> Vec<f32> {
        match self.plane {
            HeadwiseWeights::Bf16(values) => gather(values, self.ids, self.width)
                .iter()
                .map(|&v| f32::from_bits(u32::from(v) << 16))
                .collect(),
            HeadwiseWeights::Fp8 { codes, scales } => {
                let mut values = Vec::with_capacity(self.width * self.ids.len());
                for &id in self.ids {
                    let h = id as usize;
                    let table: [f32; 256] =
                        std::array::from_fn(|i| coefficient(i as u8, scales[h]));
                    values.extend(
                        codes[h * self.width..(h + 1) * self.width]
                            .iter()
                            .map(|&b| table[b as usize]),
                    );
                }
                values
            }
        }
    }
}
fn gather<'a, T: Clone>(values: &'a [T], ids: &[i32], width: usize) -> Cow<'a, [T]> {
    let Some(&first) = ids.first() else {
        return Cow::Borrowed(&[]);
    };
    if ids
        .iter()
        .enumerate()
        .all(|(i, &h)| h as usize == first as usize + i)
    {
        let start = first as usize * width;
        return Cow::Borrowed(&values[start..start + ids.len() * width]);
    }
    let mut out = Vec::with_capacity(ids.len() * width);
    for &h in ids {
        let start = h as usize * width;
        out.extend_from_slice(&values[start..start + width]);
    }
    Cow::Owned(out)
}
