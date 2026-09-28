//! Tensor-neutral Prism/Bonsai packed-weight and Hadamard metadata contract.
//!
//! This module deliberately borrows packed bytes. Backends may decode one row/tile at a time into
//! their own device format; it never expands a checkpoint or depends on MLX/Candle.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

pub const PRISM_GROUP_SIZE: usize = 128;
pub const PRISM_PQ2_0_GGML_TYPE: u32 = 142;
pub const PRISM_PTQ1_0_GGML_TYPE: u32 = 143;
const PQ2_BLOCK_BYTES: usize = 34;
const PTQ1_BLOCK_BYTES: usize = 28;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrismPackedKind {
    Pq2_0,
    Ptq1_0,
}
impl PrismPackedKind {
    pub fn from_ggml_type(value: u32) -> Result<Self, PrismError> {
        match value {
            PRISM_PQ2_0_GGML_TYPE => Ok(Self::Pq2_0),
            PRISM_PTQ1_0_GGML_TYPE => Ok(Self::Ptq1_0),
            other => Err(PrismError::UnsupportedGgmlType(other)),
        }
    }
    pub const fn ggml_type(self) -> u32 {
        match self {
            Self::Pq2_0 => PRISM_PQ2_0_GGML_TYPE,
            Self::Ptq1_0 => PRISM_PTQ1_0_GGML_TYPE,
        }
    }
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Pq2_0 => PQ2_BLOCK_BYTES,
            Self::Ptq1_0 => PTQ1_BLOCK_BYTES,
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum PrismError {
    #[error("unsupported Prism GGML tensor type {0}")]
    UnsupportedGgmlType(u32),
    #[error(
        "Prism matrix must have exactly two positive GGUF dimensions [input, output], got {0:?}"
    )]
    BadDimensions(Vec<usize>),
    #[error("Prism input width {0} is not divisible by group size {PRISM_GROUP_SIZE}")]
    BadInputWidth(usize),
    #[error("Prism packed bytes: expected {expected}, got {actual}")]
    BadPackedLength { expected: usize, actual: usize },
    #[error("Prism row {row} is outside {rows} rows")]
    BadRow { row: usize, rows: usize },
    #[error("Prism destination: expected {expected} floats, got {actual}")]
    BadDestination { expected: usize, actual: usize },
    #[error("Prism quantization scale is non-finite")]
    NonFiniteScale,
    #[error("Prism Hadamard metadata: {0}")]
    BadHadamardMetadata(String),
    #[error("Prism GDN geometry: {0}")]
    BadGdnGeometry(String),
}

/// Borrowed GGUF tensor data. GGUF dimensions are `[input_width, output_rows]`.
#[derive(Clone, Copy, Debug)]
pub struct PrismPackedMatrixRef<'a> {
    kind: PrismPackedKind,
    input_width: usize,
    output_rows: usize,
    data: &'a [u8],
}
impl<'a> PrismPackedMatrixRef<'a> {
    pub fn from_gguf(
        kind: PrismPackedKind,
        dimensions: &[usize],
        data: &'a [u8],
    ) -> Result<Self, PrismError> {
        if dimensions.len() != 2 || dimensions.contains(&0) {
            return Err(PrismError::BadDimensions(dimensions.to_vec()));
        }
        let (input_width, output_rows) = (dimensions[0], dimensions[1]);
        if input_width % PRISM_GROUP_SIZE != 0 {
            return Err(PrismError::BadInputWidth(input_width));
        }
        let blocks = input_width
            .checked_mul(output_rows)
            .and_then(|n| n.checked_div(PRISM_GROUP_SIZE))
            .ok_or_else(|| PrismError::BadDimensions(dimensions.to_vec()))?;
        let expected = blocks
            .checked_mul(kind.block_bytes())
            .ok_or_else(|| PrismError::BadDimensions(dimensions.to_vec()))?;
        if data.len() != expected {
            return Err(PrismError::BadPackedLength {
                expected,
                actual: data.len(),
            });
        }
        // Validate every compressed block at admission time. CPU decoding also checks this scale,
        // but device kernels consume the retained bytes directly, so a successful load must carry
        // the same finite-scale guarantee on every backend.
        for block in data.chunks_exact(kind.block_bytes()) {
            let scale_at = match kind {
                PrismPackedKind::Pq2_0 => 0,
                PrismPackedKind::Ptq1_0 => 26,
            };
            if !f16(block[scale_at], block[scale_at + 1]).is_finite() {
                return Err(PrismError::NonFiniteScale);
            }
        }
        Ok(Self {
            kind,
            input_width,
            output_rows,
            data,
        })
    }
    pub const fn kind(self) -> PrismPackedKind {
        self.kind
    }
    pub const fn input_width(self) -> usize {
        self.input_width
    }
    pub const fn output_rows(self) -> usize {
        self.output_rows
    }
    /// CPU reference dequantization for a single GGUF row. This is intentionally a diagnostic
    /// path; loaders should use [`Self::transcode_affine_row_into`] to retain 2-bit storage.
    pub fn decode_row_into(&self, row: usize, out: &mut [f32]) -> Result<(), PrismError> {
        if row >= self.output_rows {
            return Err(PrismError::BadRow {
                row,
                rows: self.output_rows,
            });
        }
        if out.len() != self.input_width {
            return Err(PrismError::BadDestination {
                expected: self.input_width,
                actual: out.len(),
            });
        }
        let blocks_per_row = self.input_width / PRISM_GROUP_SIZE;
        let begin = row * blocks_per_row * self.kind.block_bytes();
        for (i, block) in self.data[begin..begin + blocks_per_row * self.kind.block_bytes()]
            .chunks_exact(self.kind.block_bytes())
            .enumerate()
        {
            decode_block_into(
                self.kind,
                block,
                &mut out[i * PRISM_GROUP_SIZE..(i + 1) * PRISM_GROUP_SIZE],
            )?;
        }
        Ok(())
    }
    /// Transcode one row to the universal affine 2-bit representation: eight little-endian
    /// `u32` words (16 lanes per word) plus one scale and bias for each 128-value group.
    /// The caller owns the destination buffers, so this never expands an entire checkpoint.
    pub fn transcode_affine_row_into(
        &self,
        row: usize,
        code_words: &mut [u32],
        scales: &mut [f32],
        biases: &mut [f32],
    ) -> Result<(), PrismError> {
        if row >= self.output_rows {
            return Err(PrismError::BadRow {
                row,
                rows: self.output_rows,
            });
        }
        let blocks_per_row = self.input_width / PRISM_GROUP_SIZE;
        if code_words.len() != blocks_per_row * PRISM_AFFINE_WORDS_PER_BLOCK {
            return Err(PrismError::BadDestination {
                expected: blocks_per_row * PRISM_AFFINE_WORDS_PER_BLOCK,
                actual: code_words.len(),
            });
        }
        if scales.len() != blocks_per_row || biases.len() != blocks_per_row {
            return Err(PrismError::BadDestination {
                expected: blocks_per_row,
                actual: scales.len().min(biases.len()),
            });
        }
        let begin = row * blocks_per_row * self.kind.block_bytes();
        for (i, block) in self.data[begin..begin + blocks_per_row * self.kind.block_bytes()]
            .chunks_exact(self.kind.block_bytes())
            .enumerate()
        {
            let (scale, bias) = transcode_block_to_affine(
                self.kind,
                block,
                &mut code_words
                    [i * PRISM_AFFINE_WORDS_PER_BLOCK..(i + 1) * PRISM_AFFINE_WORDS_PER_BLOCK],
            )?;
            scales[i] = scale;
            biases[i] = bias;
        }
        Ok(())
    }
}

/// A 128-value Prism group becomes eight little-endian words with sixteen 2-bit lanes each.
pub const PRISM_AFFINE_WORDS_PER_BLOCK: usize = PRISM_GROUP_SIZE / 16;

/// Decode a native PQ2/PTQ1 block into a backend-neutral affine 2-bit block.
/// Returns `(scale, bias)`, where values are `code * scale + bias` and `bias == -scale`.
pub fn transcode_block_to_affine(
    kind: PrismPackedKind,
    raw: &[u8],
    code_words: &mut [u32],
) -> Result<(f32, f32), PrismError> {
    if raw.len() != kind.block_bytes() {
        return Err(PrismError::BadPackedLength {
            expected: kind.block_bytes(),
            actual: raw.len(),
        });
    }
    if code_words.len() != PRISM_AFFINE_WORDS_PER_BLOCK {
        return Err(PrismError::BadDestination {
            expected: PRISM_AFFINE_WORDS_PER_BLOCK,
            actual: code_words.len(),
        });
    }
    let scale_at = match kind {
        PrismPackedKind::Pq2_0 => 0,
        PrismPackedKind::Ptq1_0 => 26,
    };
    let scale = f16(raw[scale_at], raw[scale_at + 1]);
    if !scale.is_finite() {
        return Err(PrismError::NonFiniteScale);
    }
    code_words.fill(0);
    match kind {
        PrismPackedKind::Pq2_0 => {
            for (word, bytes) in code_words.iter_mut().zip(raw[2..].chunks_exact(4)) {
                *word = u32::from_le_bytes(bytes.try_into().expect("exact four-byte chunks"));
            }
        }
        PrismPackedKind::Ptq1_0 => {
            const POW3: [u16; 5] = [1, 3, 9, 27, 81];
            let mut lane = 0;
            for &(lo, hi, ntrits) in &[(0, 16, 5), (16, 24, 5), (24, 26, 4)] {
                for &power in POW3.iter().take(ntrits) {
                    for &byte in &raw[lo..hi] {
                        let code = ((((byte as u16 * power) & 255) * 3) >> 8) as u32;
                        code_words[lane / 16] |= code << (2 * (lane % 16));
                        lane += 1;
                    }
                }
            }
            debug_assert_eq!(lane, PRISM_GROUP_SIZE);
        }
    }
    Ok((scale, -scale))
}

pub fn decode_block_into(
    kind: PrismPackedKind,
    raw: &[u8],
    out: &mut [f32],
) -> Result<(), PrismError> {
    if out.len() != PRISM_GROUP_SIZE {
        return Err(PrismError::BadDestination {
            expected: PRISM_GROUP_SIZE,
            actual: out.len(),
        });
    }
    let mut codes = [0; PRISM_AFFINE_WORDS_PER_BLOCK];
    let (scale, bias) = transcode_block_to_affine(kind, raw, &mut codes)?;
    for (word_index, word) in codes.into_iter().enumerate() {
        for lane in 0..16 {
            out[word_index * 16 + lane] = ((word >> (2 * lane) & 3) as f32) * scale + bias;
        }
    }
    Ok(())
}
fn f16(lo: u8, hi: u8) -> f32 {
    let b = u16::from_le_bytes([lo, hi]);
    let s = ((b >> 15) as i32) * -2 + 1;
    let e = ((b >> 10) & 0x1f) as i32;
    let m = (b & 0x03ff) as u32;
    match e {
        0 => s as f32 * (m as f32) * 2f32.powi(-24),
        31 => {
            if m == 0 {
                s as f32 * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => s as f32 * (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GdnLayout {
    pub head_dim: usize,
    pub groups: usize,
    pub repetitions: usize,
}
impl GdnLayout {
    pub fn from_ssm_out(
        input_width: usize,
        time_step_rank: usize,
        group_count: usize,
    ) -> Result<Self, PrismError> {
        if time_step_rank == 0
            || group_count == 0
            || !time_step_rank.is_multiple_of(group_count)
            || !input_width.is_multiple_of(time_step_rank)
        {
            return Err(PrismError::BadGdnGeometry(format!(
                "input={input_width}, rank={time_step_rank}, groups={group_count}"
            )));
        }
        Ok(Self {
            head_dim: input_width / time_step_rank,
            groups: group_count,
            repetitions: time_step_rank / group_count,
        })
    }
    pub const fn width(self) -> usize {
        self.head_dim * self.groups * self.repetitions
    }
}
/// `[head_dim, groups, repetitions] -> [head_dim, repetitions, groups]` on one feature axis.
pub fn gdn_reorder_last_axis_in_place(
    values: &mut [f32],
    layout: GdnLayout,
) -> Result<(), PrismError> {
    if values.len() != layout.width() {
        return Err(PrismError::BadDestination {
            expected: layout.width(),
            actual: values.len(),
        });
    }
    let old = values.to_vec();
    for hd in 0..layout.head_dim {
        for group in 0..layout.groups {
            for rep in 0..layout.repetitions {
                let src = (hd * layout.groups + group) * layout.repetitions + rep;
                let dst = (hd * layout.repetitions + rep) * layout.groups + group;
                values[dst] = old[src];
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct PrismHadamardMetadata {
    pub block_size: usize,
    pub signs_by_width: BTreeMap<usize, Vec<i8>>,
    pub forward_weight_names: BTreeSet<String>,
    pub inverse_weight_names: BTreeSet<String>,
    pub gdn_v_grouped: bool,
}
impl PrismHadamardMetadata {
    pub fn validate(&self) -> Result<(), PrismError> {
        if self.block_size == 0 || !self.block_size.is_power_of_two() {
            return Err(PrismError::BadHadamardMetadata(
                "block_size must be a power of two".into(),
            ));
        }
        if self.forward_weight_names.iter().any(|n| n.is_empty())
            || self.inverse_weight_names.iter().any(|n| n.is_empty())
            || !self
                .forward_weight_names
                .is_disjoint(&self.inverse_weight_names)
        {
            return Err(PrismError::BadHadamardMetadata(
                "weight names must be nonempty and forward/inverse-disjoint".into(),
            ));
        }
        for (&width, signs) in &self.signs_by_width {
            if width == 0
                || width % self.block_size != 0
                || signs.len() != width
                || signs.iter().any(|&s| s != -1 && s != 1)
            {
                return Err(PrismError::BadHadamardMetadata(format!(
                    "invalid explicit sign vector for width {width}"
                )));
            }
        }
        Ok(())
    }
    pub fn role(&self, name: &str) -> PrismTransformRole {
        if self.inverse_weight_names.contains(name) {
            PrismTransformRole::Inverse
        } else if self.forward_weight_names.contains(name) {
            PrismTransformRole::Forward
        } else {
            PrismTransformRole::None
        }
    }
    pub fn signs(&self, input_width: usize) -> Result<&[i8], PrismError> {
        self.signs_by_width
            .get(&input_width)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                PrismError::BadHadamardMetadata(format!(
                    "missing signs for input width {input_width}"
                ))
            })
    }
    /// Validate a named packed tensor against its input width and return its transform contract.
    pub fn classify_weight(
        &self,
        name: &str,
        input_width: usize,
    ) -> Result<PrismWeightTransform<'_>, PrismError> {
        self.validate()?;
        let role = self.role(name);
        let signs = match role {
            PrismTransformRole::None => None,
            PrismTransformRole::Forward | PrismTransformRole::Inverse => {
                Some(self.signs(input_width)?)
            }
        };
        Ok(PrismWeightTransform {
            role,
            signs,
            gdn_v_grouped: self.gdn_v_grouped
                && role == PrismTransformRole::Forward
                && is_gdn_ssm_out_weight(name),
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrismWeightTransform<'a> {
    pub role: PrismTransformRole,
    pub signs: Option<&'a [i8]>,
    pub gdn_v_grouped: bool,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrismTransformRole {
    None,
    Forward,
    Inverse,
}
/// Sign flip then normalized blockwise FWHT. Call GDN reordering first only for `ssm_out`.
pub fn apply_hadamard_forward_in_place(
    values: &mut [f32],
    signs: &[i8],
    block_size: usize,
    gdn: Option<GdnLayout>,
) -> Result<(), PrismError> {
    if let Some(layout) = gdn {
        gdn_reorder_last_axis_in_place(values, layout)?;
    }
    apply_signs(values, signs)?;
    normalized_fwht_in_place(values, block_size)
}
/// Inverse embedding path: normalized FWHT then the same sign flip (both operations are involutions).
pub fn apply_hadamard_inverse_in_place(
    values: &mut [f32],
    signs: &[i8],
    block_size: usize,
) -> Result<(), PrismError> {
    normalized_fwht_in_place(values, block_size)?;
    apply_signs(values, signs)
}
fn apply_signs(values: &mut [f32], signs: &[i8]) -> Result<(), PrismError> {
    if values.len() != signs.len() || signs.iter().any(|&s| s != -1 && s != 1) {
        return Err(PrismError::BadHadamardMetadata(
            "sign shape or domain invalid".into(),
        ));
    }
    for (x, &s) in values.iter_mut().zip(signs) {
        *x *= s as f32;
    }
    Ok(())
}
pub fn normalized_fwht_in_place(values: &mut [f32], block_size: usize) -> Result<(), PrismError> {
    if block_size == 0 || !block_size.is_power_of_two() || !values.len().is_multiple_of(block_size)
    {
        return Err(PrismError::BadHadamardMetadata(
            "FWHT length must be a multiple of a power-of-two block".into(),
        ));
    }
    let scale = (block_size as f32).sqrt().recip();
    for block in values.chunks_exact_mut(block_size) {
        let mut step = 1;
        while step < block_size {
            for base in (0..block_size).step_by(step * 2) {
                for i in 0..step {
                    let (a, b) = (block[base + i], block[base + step + i]);
                    block[base + i] = a + b;
                    block[base + step + i] = a - b;
                }
            }
            step *= 2;
        }
        for v in block {
            *v *= scale;
        }
    }
    Ok(())
}

pub fn is_gdn_ssm_out_weight(name: &str) -> bool {
    name.contains(".ssm_out.")
}
/// GGUF names are already canonical for the Prism architecture; this prevents backend-specific aliases.
pub fn gguf_weight_name(name: &str) -> Result<&str, PrismError> {
    if name.is_empty() || name.starts_with('.') || name.contains("..") {
        Err(PrismError::BadHadamardMetadata(format!(
            "invalid GGUF weight name {name:?}"
        )))
    } else {
        Ok(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pq2_decodes_lanes_and_scale() {
        let mut b = [0u8; 34];
        b[..2].copy_from_slice(&[0, 60]);
        b[2] = 0b11_10_01_00;
        let mut out = [0.; 128];
        decode_block_into(PrismPackedKind::Pq2_0, &b, &mut out).unwrap();
        assert_eq!(&out[..4], &[-1., 0., 1., 2.]);
    }
    #[test]
    fn ptq_matches_independent_trit_formula() {
        let mut b = [0u8; 28];
        b[26..].copy_from_slice(&[0, 60]);
        for x in &mut b[..26] {
            *x = 255
        }
        let mut out = [0.; 128];
        decode_block_into(PrismPackedKind::Ptq1_0, &b, &mut out).unwrap();
        assert!(out.iter().all(|&x| x == 1.));
    }
    #[test]
    fn affine_transcode_retains_two_bit_words() {
        let mut pq = [0u8; 34];
        pq[..2].copy_from_slice(&[0, 60]);
        pq[2..6].copy_from_slice(&0xE4_E4_E4_E4u32.to_le_bytes());
        let mut words = [0; PRISM_AFFINE_WORDS_PER_BLOCK];
        assert_eq!(
            transcode_block_to_affine(PrismPackedKind::Pq2_0, &pq, &mut words).unwrap(),
            (1., -1.)
        );
        assert_eq!(words[0], 0xE4_E4_E4_E4);

        let mut ptq = [255u8; 28];
        ptq[26..].copy_from_slice(&[0, 60]);
        let mut ptq_words = [0; PRISM_AFFINE_WORDS_PER_BLOCK];
        transcode_block_to_affine(PrismPackedKind::Ptq1_0, &ptq, &mut ptq_words).unwrap();
        assert!(ptq_words.iter().all(|&word| word == 0xAAAA_AAAA));
    }
    #[test]
    fn malformed_codec_rejected() {
        assert!(PrismPackedKind::from_ggml_type(141).is_err());
        assert!(PrismPackedMatrixRef::from_gguf(PrismPackedKind::Pq2_0, &[127, 1], &[]).is_err());
        assert!(
            PrismPackedMatrixRef::from_gguf(PrismPackedKind::Pq2_0, &[128, 1], &[0; 33]).is_err()
        );
        let mut nonfinite = [0u8; 34];
        nonfinite[..2].copy_from_slice(&[0, 124]);
        assert!(matches!(
            PrismPackedMatrixRef::from_gguf(PrismPackedKind::Pq2_0, &[128, 1], &nonfinite),
            Err(PrismError::NonFiniteScale)
        ));
        assert_eq!(
            decode_block_into(PrismPackedKind::Pq2_0, &nonfinite, &mut [0.; 128]),
            Err(PrismError::NonFiniteScale)
        );
        let mut ptq_nonfinite = [0u8; 28];
        ptq_nonfinite[26..].copy_from_slice(&[0, 124]);
        assert!(matches!(
            PrismPackedMatrixRef::from_gguf(PrismPackedKind::Ptq1_0, &[128, 1], &ptq_nonfinite,),
            Err(PrismError::NonFiniteScale)
        ));
    }
    #[test]
    fn fwht_and_gdn_order() {
        let mut v = vec![1., 0., 0., 0.];
        normalized_fwht_in_place(&mut v, 4).unwrap();
        assert_eq!(v, vec![0.5, 0.5, 0.5, 0.5]);
        let mut x = (0..8).map(|n| n as f32).collect::<Vec<_>>();
        gdn_reorder_last_axis_in_place(
            &mut x,
            GdnLayout {
                head_dim: 1,
                groups: 2,
                repetitions: 4,
            },
        )
        .unwrap();
        assert_eq!(x, vec![0., 4., 1., 5., 2., 6., 3., 7.]);
    }
    #[test]
    fn metadata_and_inverse_are_checked() {
        let mut signs = BTreeMap::new();
        signs.insert(4, vec![1, -1, 1, -1]);
        let m = PrismHadamardMetadata {
            block_size: 4,
            signs_by_width: signs,
            forward_weight_names: ["blk.0.ssm_out.weight".into()].into_iter().collect(),
            inverse_weight_names: ["token_embd.weight".into()].into_iter().collect(),
            gdn_v_grouped: true,
        };
        m.validate().unwrap();
        assert_eq!(m.role("token_embd.weight"), PrismTransformRole::Inverse);
        let transform = m.classify_weight("blk.0.ssm_out.weight", 4).unwrap();
        assert_eq!(transform.role, PrismTransformRole::Forward);
        assert!(transform.gdn_v_grouped);
        assert_eq!(transform.signs.unwrap(), &[1, -1, 1, -1]);
        let mut x = vec![1., 2., 3., 4.];
        apply_hadamard_inverse_in_place(&mut x, m.signs(4).unwrap(), 4).unwrap();
        assert_eq!(x, vec![5., 1., -2., 0.]);
    }
}
