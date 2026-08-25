//! NVFP4 weight container + offline packer + CPU dequant reference (sc-11040, epic 11037).
//!
//! NVFP4 is the FP4 storage/compute format for consumer Blackwell (sm_120). A weight is stored as a
//! **two-level** quantization:
//!
//! 1. **4-bit E2M1 elements** ([`E2M1_LUT`]) packed two-per-byte over **16-element blocks**
//!    ([`NVFP4_BLOCK`]) along the contraction (K / `in_features`) axis.
//! 2. One **FP8 E4M3** *micro-scale* per 16-element block (the finer block + real FP8 scale is what
//!    buys NVFP4 its near-FP8 accuracy vs MXFP4's block-32 power-of-two `E8M0` scale).
//! 3. One **FP32 per-tensor** scale ([`Nvfp4Tensor::global_scale`]), the second level, consumed by the
//!    sc-11039 cuBLASLt GEMM via `alpha` (the master-gate spike found `D_SCALE_POINTER` is
//!    `NOT_SUPPORTED` for a bf16 output, so the per-tensor scale folds into `alpha`).
//!
//! Effective footprint: 4 bits/element (E2M1) + 8 bits per 16-element block (E4M3) = **4.5 effective
//! bits/weight**, plus a negligible 32 bits/tensor.
//!
//! Dequant of element `(r, c)` in block `b = c / 16`:
//!
//! ```text
//! value = E2M1_LUT[nibble] * e4m3(block_scale[r, b]) * global_scale
//! ```
//!
//! # Canonical byte / scale layout (the sc-11039 cuBLASLt GEMM reads this)
//!
//! This is the **canonical, cuBLASLt-consumable** layout for a `[rows = out_features, cols =
//! in_features]` weight (the master-gate spike sc-11038 confirmed cuBLASLt-primary on sm_120; the
//! hand-rolled MMQ warp-MMA `ue4m3` fallback is not the shipping path). sc-11039 relies on the exact
//! layout documented here.
//!
//! ## Packed E2M1 nibbles ([`Nvfp4Tensor::packed`])
//!
//! Row-major `[rows, cols_padded / 2]` bytes. `cols_padded = round_up(cols, 16)` — K is padded up to a
//! multiple of the 16-element block (cuBLASLt requires K a multiple of the block size; see *Padding*).
//! Two E2M1 codes per byte, **little-endian nibble order**: logical column `2j` is the **low** nibble
//! of byte `j`, column `2j+1` the **high** nibble (`packed[j] = (code[2j+1] << 4) | code[2j]`). This
//! matches the vendored MXFP4 kernel unpack convention (`mmq_common.cuh`: `byte & 0x0f` is element 0,
//! `byte >> 4` is element 1) and `CUDA_R_4F_E2M1`.
//!
//! ## UE4M3 block scales ([`Nvfp4Tensor::scales`]) — padded 128×4 swizzle
//!
//! One FP8 E4M3 byte per (row, 16-block). Emitted in the **CUTLASS `Sm1xxBlockScaledConfig`
//! scale-factor atom** layout (`getScaleTensorSize`) so cuBLASLt's `VEC16` `UE4M3` block-scale mode
//! consumes them directly. The logical scale grid is `[rows, n_blocks]` with `n_blocks = cols_padded /
//! 16`; it is tiled by a **128 (rows) × 4 (block-cols) atom** and padded to that atom:
//!
//! - `sf_rows = round_up(rows, 128)`, `sf_cols = round_up(n_blocks, 4)`; total `scales.len() = sf_rows
//!   * sf_cols` bytes (== CUTLASS `getScaleTensorSize`).
//! - **Intra-atom** offset for a row `mr ∈ [0,128)`, block-col `kc ∈ [0,4)` (the CUTLASS SF atom
//!   `((32,4),4):((16,4),1)`): `intra = (mr % 32) * 16 + (mr / 32) * 4 + kc`  — a bijection onto
//!   `[0,512)`.
//! - **Atom tiling** over the `(sf_rows/128, sf_cols/4)` atom grid is **column-major** (m-atom
//!   fastest, matching CuTe `blocked_product` default `LayoutLeft`): `atom_index = m_atom + (sf_rows /
//!   128) * k_atom`, and the byte lands at `atom_index * 512 + intra`.
//! - See [`Nvfp4Tensor::scale_offset`]. Padded rows/blocks (rows ≥ `rows`, block-cols ≥ `n_blocks`)
//!   hold E4M3 `0x00` and are never read for a valid output.
//!
//! sc-11039 note: the intra-atom swizzle is the fixed, load-bearing part; the **atom-tiling order**
//! (column-major here) is the one degree of freedom — confirm it against the live cuBLASLt matrix-scale
//! descriptor and flip [`Nvfp4Tensor::scale_offset`] if the runtime wants row-major atom order.
//!
//! ## FP32 per-tensor scale ([`Nvfp4Tensor::global_scale`])
//!
//! Stored as a plain `f32`. cuBLASLt consumes it via `alpha` (for the weight operand) — not through
//! `D_SCALE_POINTER`.
//!
//! # Padding policy
//!
//! - **K not a multiple of 16** (`cols % 16 != 0`): K is padded up to `cols_padded = round_up(cols,
//!   16)`. Padded columns hold E2M1 `0` and do not participate in any block's amax (they are pure
//!   zeros), so they contribute nothing to the dequantized real columns. [`Nvfp4Tensor::dequantize`]
//!   returns the **logical** `[rows, cols]` (padding dropped).
//! - **Scale-tensor 128×4 padding**: rows padded to 128 and block-cols to 4 as above; padded scale
//!   bytes are `0x00`.
//! - **Rows (M)**: the packed nibble data itself is *not* row-padded (cuBLASLt accepts an arbitrary M
//!   for the operand); only the scale tensor is padded to the 128-row atom.
//!
//! # Relation to the MMQ `block_nvfp4` kernel struct (fallback only)
//!
//! The vendored MMQ struct `block_nvfp4 { uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; uint8_t qs[QK_NVFP4/2]; }`
//! (`mmq_common.cuh:143`, `QK_NVFP4 = 64`, `QK_NVFP4_SUB = 16`) groups **four** 16-element sub-blocks
//! into a 64-element super-block: 4 UE4M3 sub-scales `d[4]` + 32 nibble bytes `qs[32]`. That is the
//! same 16-element micro-scale granularity as this container, re-tiled into 64-wide super-blocks. The
//! MMQ path also uses a *bespoke* `ggml_cuda_ue4m3_to_fp32` decode (bias 8, `/2`, `0x7F`→0) that
//! differs from the standard OCP E4M3 this container emits for cuBLASLt; reconciling the two is
//! sc-11039's concern **iff** the MMQ fallback is ever revived (the spike deprecated it).

use candle_core::{DType, Device, Result, Tensor};

/// E2M1 elements per FP8 micro-scale block (the NVFP4 block size along K).
pub const NVFP4_BLOCK: usize = 16;

/// Max magnitude representable in E2M1 (`0b0111` = 6.0).
pub const E2M1_MAX: f32 = 6.0;

/// Max finite magnitude representable in OCP FP8 E4M3 (`S.1111.110` = 448.0).
pub const E4M3_MAX: f32 = 448.0;

/// CUTLASS `Sm1xxBlockScaledConfig` scale-factor atom: 128 rows.
pub const SF_ATOM_ROWS: usize = 128;

/// CUTLASS `Sm1xxBlockScaledConfig` scale-factor atom: 4 block-columns.
pub const SF_ATOM_COLS: usize = 4;

/// E2M1 (FP4) code → value. Codes `0..8` are `+`, `8..16` are `-` (sign in bit 3). Identical to the
/// vendored MXFP4 `FP4_LUT` (candle-gen-lens `text_encoder.rs`) — the same E2M1 grid.
pub const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// The 8 non-negative E2M1 magnitudes (codes `0..8`), for nearest-magnitude quantization.
const E2M1_MAGS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

#[inline]
fn round_up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

/// The **one** non-finite refusal message this container emits, shared by every f32 input gate so a
/// NaN/inf is named the same way wherever it enters: [`Nvfp4Tensor::from_kitchen_parts`]'s per-tensor
/// scale and [`Nvfp4Tensor::pack_from_slice`]'s weight elements.
///
/// Both entry points must refuse rather than absorb: `e2m1_from_f32`/`e4m3_from_f32` map NaN to code
/// `0`, and `f32::max` (the amax fold) *drops* NaN operands entirely, so a NaN weight would pack as a
/// silent zero and an `inf` weight would drive `global_scale` to `inf` and zero the **whole tensor**.
fn non_finite_msg(what: &str, value: f32) -> String {
    format!("nvfp4 {what} must be finite, got {value}")
}

/// Decode one OCP FP8 **E4M3** byte to f32 (bias 7; subnormals at `E==0`; `S.1111.111` is NaN, max
/// finite 448 at `S.1111.110`). Block scales are always non-negative, so only codes `0x00..=0x7E`
/// occur in [`Nvfp4Tensor::scales`], but this decodes the full 8-bit space.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as i32;
    if exp == 0 {
        // Subnormal: man * 2^-6 / 8 = man * 2^-9.
        sign * (man as f32) * 2f32.powi(-9)
    } else if exp == 0x0F && man == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + man as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

/// Encode a **non-negative** scale to the nearest OCP FP8 E4M3 code (round-to-nearest, ties-to-even),
/// clamped to `[0, 448]`. Non-positive / NaN inputs map to `0x00`. Scans the 127 positive finite codes
/// (`0x00..=0x7E`, skipping the `0x7F` NaN) — the packer is offline, one call per 16-element block.
pub fn e4m3_from_f32(v: f32) -> u8 {
    if v.is_nan() || v <= 0.0 {
        return 0;
    }
    let v = v.min(E4M3_MAX);
    let (mut best, mut best_d) = (0u8, f32::INFINITY);
    for code in 0u8..=0x7E {
        let rv = e4m3_to_f32(code);
        let d = (rv - v).abs();
        // `<` keeps the first (smaller) code on a tie; codes are monotonic in value for the positive
        // range, so the two tie candidates are adjacent and the even-code (even mantissa LSB) wins.
        if d < best_d || (d == best_d && code.is_multiple_of(2)) {
            best_d = d;
            best = code;
        }
    }
    best
}

/// Encode a signed value to the nearest E2M1 (FP4) code (round-to-nearest-even over the 8 magnitudes,
/// sign in bit 3), saturating at ±6. Returns a nibble in `0..16`.
pub fn e2m1_from_f32(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    let sign: u8 = if v.is_sign_negative() { 0x08 } else { 0x00 };
    let mag = v.abs().min(E2M1_MAX);
    let (mut best, mut best_d) = (0usize, f32::INFINITY);
    for (i, &m) in E2M1_MAGS.iter().enumerate() {
        let d = (m - mag).abs();
        // Ties-to-even: prefer the even-index magnitude (mantissa LSB 0).
        if d < best_d || (d == best_d && i.is_multiple_of(2)) {
            best_d = d;
            best = i;
        }
    }
    sign | best as u8
}

/// A weight packed in the canonical NVFP4 container (see the [module docs](self) for the exact byte /
/// scale layout the sc-11039 cuBLASLt GEMM consumes).
#[derive(Clone, Debug)]
pub struct Nvfp4Tensor {
    /// Logical row count (`out_features`).
    pub rows: usize,
    /// Logical column count (`in_features`), **before** the K→multiple-of-16 padding.
    pub cols: usize,
    /// `cols` padded up to a multiple of [`NVFP4_BLOCK`] (16).
    pub cols_padded: usize,
    /// Row-major `[rows, cols_padded / 2]` E2M1 nibble bytes (two codes/byte, low = even column).
    pub packed: Vec<u8>,
    /// Padded, 128×4-swizzled UE4M3 block scales — `sf_rows * sf_cols` bytes. See the module docs and
    /// [`Self::scale_offset`].
    pub scales: Vec<u8>,
    /// Scale-tensor rows: `round_up(rows, 128)`.
    pub sf_rows: usize,
    /// Scale-tensor block-columns: `round_up(cols_padded / 16, 4)`.
    pub sf_cols: usize,
    /// The FP32 second-level per-tensor scale (consumed by cuBLASLt via `alpha`).
    pub global_scale: f32,
}

impl Nvfp4Tensor {
    /// Build the canonical Candle NVFP4 container from a ComfyUI Kitchen serialization.
    ///
    /// Two lossless representation conversions happen here; no weight or scale is requantized, and
    /// the FP32 second-level scale has the same numeric convention in both.
    ///
    /// 1. **Nibble order.** Kitchen's `hi_first=true` stores the even K element in the high nibble
    ///    (`comfy.float.stochastic_float_to_fp4_e2m1`: `packed = (even << 4) | odd`), while
    ///    [`Nvfp4Tensor`] stores it in the low nibble. Every byte is rotated.
    ///
    /// 2. **Block-scale atom order** (sc-20641). Both layouts are the cuBLAS 128×4 scale-factor
    ///    swizzle with the *same* intra-atom slot, but they walk the atom grid differently:
    ///    ComfyUI's `comfy.float.to_blocked` permutes `(R/128, 128, B/4, 4) → (R/128, B/4, 128, 4)`,
    ///    i.e. **row-major** over atoms, while [`Self::scale_offset_for`] walks it **column-major**
    ///    (CuTe `blocked_product` `LayoutLeft`). The two coincide only when the weight has a single
    ///    atom in one dimension — which is why this constructor's original single-atom test could
    ///    not see the difference, and why every real DiT projection was affected (a `[6144, 6144]`
    ///    layer has 48 row atoms × 48 block atoms). Scales are therefore **permuted** into the
    ///    container's own convention — read through [`gen_core::blocked_scale_index`], written
    ///    through [`Self::scale_offset_for`] — rather than copied verbatim.
    ///
    ///    Going through both index functions keeps this correct if the atom order the [module
    ///    docs](self) flag as "the one degree of freedom" is later flipped to match live cuBLASLt:
    ///    the permutation collapses to the identity on its own, with no edit here.
    ///
    /// `rows` and `cols` are the logical (already 16-aligned) weight dimensions. The constructor is
    /// intentionally strict because imported checkpoints are untrusted input and a malformed scale
    /// surface would otherwise reach cuBLASLt as a device buffer with the wrong layout.
    pub fn from_kitchen_parts(
        packed_hi_first: &[u8],
        blocked_scales: &[u8],
        global_scale: f32,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 || !rows.is_multiple_of(16) || !cols.is_multiple_of(NVFP4_BLOCK) {
            return Err(candle_core::Error::Msg(format!(
                "nvfp4 Kitchen weight shape must be non-zero and 16-aligned, got [{rows}, {cols}]"
            )));
        }
        let packed_len = rows
            .checked_mul(cols / 2)
            .ok_or_else(|| candle_core::Error::Msg("nvfp4 Kitchen weight size overflow".into()))?;
        if packed_hi_first.len() != packed_len {
            return Err(candle_core::Error::Msg(format!(
                "nvfp4 Kitchen packed weight has {} bytes, expected {packed_len} for [{rows}, {cols}]",
                packed_hi_first.len()
            )));
        }
        let scale_len = Self::scale_tensor_len(rows, cols);
        if blocked_scales.len() != scale_len {
            return Err(candle_core::Error::Msg(format!(
                "nvfp4 Kitchen block scales have {} bytes, expected {scale_len} for [{rows}, {cols}]",
                blocked_scales.len()
            )));
        }
        if !global_scale.is_finite() {
            return Err(candle_core::Error::Msg(non_finite_msg(
                "Kitchen global scale",
                global_scale,
            )));
        }
        if global_scale < 0.0 {
            return Err(candle_core::Error::Msg(format!(
                "nvfp4 Kitchen global scale must be non-negative, got {global_scale}"
            )));
        }

        let packed = packed_hi_first
            .iter()
            .map(|byte| byte.rotate_left(4))
            .collect();
        let cols_padded = cols;
        let blocks = cols / NVFP4_BLOCK;
        let sf_rows = round_up(rows, SF_ATOM_ROWS);
        let sf_cols = round_up(blocks, SF_ATOM_COLS);
        // Permute the atom grid from Kitchen's `to_blocked` order into this container's — see the
        // doc comment. Padded slots keep the 0x00 they are initialized with; both layouts pad the
        // same `sf_rows * sf_cols` surface, so nothing is lost or invented.
        let mut scales = vec![0u8; sf_rows * sf_cols];
        for r in 0..rows {
            for blk in 0..blocks {
                scales[Self::scale_offset_for(r, blk, sf_rows)] =
                    blocked_scales[gen_core::blocked_scale_index(rows, blocks, r, blk)];
            }
        }
        Ok(Self {
            rows,
            cols,
            cols_padded,
            packed,
            scales,
            sf_rows,
            sf_cols,
            global_scale,
        })
    }

    /// A contiguous **row slice** of this container, without dequantizing anything (sc-21547).
    ///
    /// This is what a fused-QKV checkpoint needs: one stored `[3·d, d]` NVFP4 matrix becomes three
    /// `[d, d]` operands, each still packed. The nibble payload is row-major, so its slice is a
    /// contiguous byte range; the block scales are **re-emitted** through
    /// [`Self::scale_offset_for`] against the slice's own `sf_rows` rather than copied, because the
    /// 128×4 swizzle interleaves rows within an atom and the destination has a different atom grid.
    ///
    /// `start` and `len` must both be multiples of [`SF_ATOM_ROWS`]. That is not a convenience
    /// restriction: a slice that cut a scale-factor atom in half would leave the source's scale
    /// bytes split across two padded destination tensors whose sizes no longer sum to the source's,
    /// so the residency the plan priced and the residency the reader measures would disagree by the
    /// padding. A tile-aligned slice takes whole atoms and the sizes partition exactly. The plan
    /// compiler refuses a misaligned slice before any payload is read
    /// (`LogicalTransformError::PackedSliceAlignment`); this is the container's own restatement of
    /// the same rule, because the container is public and its callers are not all the reader.
    pub fn slice_rows(&self, start: usize, len: usize) -> Result<Self> {
        if len == 0 || start.saturating_add(len) > self.rows {
            return Err(candle_core::Error::Msg(format!(
                "nvfp4 row slice [{start}, {start}+{len}) is out of bounds for a {}-row tensor",
                self.rows
            )));
        }
        if !start.is_multiple_of(SF_ATOM_ROWS) || !len.is_multiple_of(SF_ATOM_ROWS) {
            return Err(candle_core::Error::Msg(format!(
                "nvfp4 row slice [{start}, {start}+{len}) must start and end on a multiple of \
                 {SF_ATOM_ROWS} (the cuBLAS block-scale row tile) so it takes whole scale-factor \
                 atoms"
            )));
        }
        let row_bytes = self.cols_padded / 2;
        let packed = self.packed[start * row_bytes..(start + len) * row_bytes].to_vec();
        let blocks = self.blocks_per_row();
        let sf_rows = round_up(len, SF_ATOM_ROWS);
        let mut scales = vec![0u8; sf_rows * self.sf_cols];
        for row in 0..len {
            for blk in 0..blocks {
                scales[Self::scale_offset_for(row, blk, sf_rows)] =
                    self.scales[self.scale_offset(start + row, blk)];
            }
        }
        Ok(Self {
            rows: len,
            cols: self.cols,
            cols_padded: self.cols_padded,
            packed,
            scales,
            sf_rows,
            sf_cols: self.sf_cols,
            global_scale: self.global_scale,
        })
    }

    /// Number of 16-element blocks per row (`cols_padded / 16`).
    #[inline]
    pub fn blocks_per_row(&self) -> usize {
        self.cols_padded / NVFP4_BLOCK
    }

    /// Byte offset of the UE4M3 scale for logical row `r` and 16-block `blk` within [`Self::scales`],
    /// per the CUTLASS `Sm1xxBlockScaledConfig` 128×4 atom (intra-atom `((32,4),4):((16,4),1)`,
    /// column-major atom tiling). See the module docs.
    #[inline]
    pub fn scale_offset(&self, r: usize, blk: usize) -> usize {
        Self::scale_offset_for(r, blk, self.sf_rows)
    }

    /// [`Self::scale_offset`] without needing an instance (used by the packer before the struct
    /// exists). `sf_rows` must be the padded row count (`round_up(rows, 128)`).
    #[inline]
    pub fn scale_offset_for(r: usize, blk: usize, sf_rows: usize) -> usize {
        let m_atom = r / SF_ATOM_ROWS;
        let k_atom = blk / SF_ATOM_COLS;
        let num_m_atoms = sf_rows / SF_ATOM_ROWS;
        // Column-major over the atom grid (m-atom fastest — CuTe `blocked_product` LayoutLeft).
        let atom_index = m_atom + num_m_atoms * k_atom;
        let mr = r % SF_ATOM_ROWS;
        let kc = blk % SF_ATOM_COLS;
        let intra = (mr % 32) * 16 + (mr / 32) * 4 + kc;
        atom_index * (SF_ATOM_ROWS * SF_ATOM_COLS) + intra
    }

    /// The padded scale-tensor byte length for a `[rows, cols]` weight — CUTLASS `getScaleTensorSize`:
    /// `round_up(rows, 128) * round_up(ceil(cols / 16), 4)`.
    pub fn scale_tensor_len(rows: usize, cols: usize) -> usize {
        let n_blocks = round_up(cols, NVFP4_BLOCK) / NVFP4_BLOCK;
        round_up(rows, SF_ATOM_ROWS) * round_up(n_blocks, SF_ATOM_COLS)
    }

    /// Offline pack a dense `[rows, cols]` weight tensor (bf16 or f32, any device) to NVFP4. The tensor
    /// is materialized to CPU f32; `bf16` inputs are upcast exactly.
    pub fn pack(weight: &Tensor) -> Result<Self> {
        let (rows, cols) = weight.dims2()?;
        let data = weight
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Self::pack_from_slice(&data, rows, cols)
    }

    /// Offline pack a row-major `[rows, cols]` f32 slice to NVFP4. Panics if `data.len() != rows *
    /// cols`. See the [module docs](self) for the numeric recipe:
    ///
    /// - per-tensor amax → `global_scale = amax / (6 * 448)` (maps the largest block scale to E4M3 448);
    /// - per-block amax → UE4M3 `block_scale ≈ amax_blk / (6 * global_scale)`;
    /// - per-element E2M1 code = round(value / (e4m3(block_scale) * global_scale)).
    ///
    /// **Refuses a non-finite input** naming the offending element, the same gate (and the same
    /// `non_finite_msg` wording) [`Self::from_kitchen_parts`] applies to its per-tensor scale. Without it the
    /// packer absorbs the bad value instead of reporting it: the `f32::max` amax fold silently drops a
    /// NaN (which then packs as E2M1 code `0`), and a single `inf` makes `global_scale` infinite, which
    /// zeroes **every** element of the tensor. A checkpoint with one poisoned weight would otherwise
    /// import as a plausible-looking all-zero layer.
    pub fn pack_from_slice(data: &[f32], rows: usize, cols: usize) -> Result<Self> {
        assert_eq!(data.len(), rows * cols, "data length must be rows * cols");
        if let Some(i) = data.iter().position(|x| !x.is_finite()) {
            return Err(candle_core::Error::Msg(non_finite_msg(
                &format!("weight element {i} (row {}, col {})", i / cols, i % cols),
                data[i],
            )));
        }
        let cols_padded = round_up(cols, NVFP4_BLOCK);
        let n_blocks = cols_padded / NVFP4_BLOCK;
        let sf_rows = round_up(rows, SF_ATOM_ROWS);
        let sf_cols = round_up(n_blocks, SF_ATOM_COLS);

        // Per-tensor amax over the real elements.
        let amax = data.iter().fold(0f32, |m, &x| m.max(x.abs()));
        // All-zero tensor: keep a finite non-zero scale so nothing divides by zero (every code is 0
        // regardless). Otherwise map the largest possible block scale to E4M3's 448.
        let global_scale = if amax > 0.0 {
            amax / (E2M1_MAX * E4M3_MAX)
        } else {
            1.0
        };

        let row_bytes = cols_padded / 2;
        let mut packed = vec![0u8; rows * row_bytes];
        let mut scales = vec![0u8; sf_rows * sf_cols];

        for r in 0..rows {
            let row = &data[r * cols..r * cols + cols];
            for blk in 0..n_blocks {
                let c0 = blk * NVFP4_BLOCK;
                let c1 = (c0 + NVFP4_BLOCK).min(cols); // real columns in this block
                                                       // Block amax over the real (non-padding) columns.
                let a_blk = if c0 < cols {
                    row[c0..c1].iter().fold(0f32, |m, &x| m.max(x.abs()))
                } else {
                    0.0
                };
                // UE4M3 block scale that maps a_blk → 6.0, expressed relative to the per-tensor scale.
                let sf_real = if global_scale > 0.0 {
                    a_blk / (E2M1_MAX * global_scale)
                } else {
                    0.0
                };
                let sf_byte = e4m3_from_f32(sf_real);
                scales[Self::scale_offset_for(r, blk, sf_rows)] = sf_byte;

                // Effective per-element dequant scale for this block.
                let elem_scale = e4m3_to_f32(sf_byte) * global_scale;
                for j in 0..NVFP4_BLOCK {
                    let c = c0 + j;
                    let v = if c < cols { row[c] } else { 0.0 };
                    let code = if elem_scale > 0.0 {
                        e2m1_from_f32(v / elem_scale)
                    } else {
                        0
                    };
                    let byte_idx = r * row_bytes + c / 2;
                    if c.is_multiple_of(2) {
                        packed[byte_idx] = (packed[byte_idx] & 0xF0) | code;
                    } else {
                        packed[byte_idx] = (packed[byte_idx] & 0x0F) | (code << 4);
                    }
                }
            }
        }

        Ok(Self {
            rows,
            cols,
            cols_padded,
            packed,
            scales,
            sf_rows,
            sf_cols,
            global_scale,
        })
    }

    /// CPU dequant reference → a row-major `[rows, cols]` f32 `Vec` (the **logical** shape; the
    /// K-padding is dropped). `value = E2M1_LUT[nibble] * e4m3(block_scale) * global_scale`. This is the
    /// reference the round-trip test and the non-Blackwell fallback use.
    pub fn dequantize_to_vec(&self) -> Vec<f32> {
        let row_bytes = self.cols_padded / 2;
        let mut out = vec![0f32; self.rows * self.cols];
        for r in 0..self.rows {
            for blk in 0..self.blocks_per_row() {
                let sf_byte = self.scales[self.scale_offset(r, blk)];
                let elem_scale = e4m3_to_f32(sf_byte) * self.global_scale;
                let c0 = blk * NVFP4_BLOCK;
                for j in 0..NVFP4_BLOCK {
                    let c = c0 + j;
                    if c >= self.cols {
                        break; // padding column — not part of the logical tensor
                    }
                    let byte = self.packed[r * row_bytes + c / 2];
                    let code = if c.is_multiple_of(2) {
                        byte & 0x0F
                    } else {
                        byte >> 4
                    };
                    out[r * self.cols + c] = E2M1_LUT[code as usize] * elem_scale;
                }
            }
        }
        out
    }

    /// [`Self::dequantize_to_vec`] as a `[rows, cols]` CPU f32 [`Tensor`].
    pub fn dequantize(&self) -> Result<Tensor> {
        Tensor::from_vec(
            self.dequantize_to_vec(),
            (self.rows, self.cols),
            &Device::Cpu,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random f32 in roughly `[-1, 1)` (xorshift → unit) — no `rand` dep.
    fn prng(seed: &mut u64) -> f32 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn rel_rms(a: &[f32], b: &[f32]) -> f32 {
        let mut num = 0f64;
        let mut den = 0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            num += ((*x - *y) as f64).powi(2);
            den += (*x as f64).powi(2);
        }
        (num / (den + 1e-30)).sqrt() as f32
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    // ---- E4M3 / E2M1 primitive round-trips ---------------------------------------------------

    #[test]
    fn e4m3_known_values() {
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0); // exp=7 (bias) man=0 → 1.0
        assert_eq!(e4m3_to_f32(0x7E), 448.0); // max finite
        assert!(e4m3_to_f32(0x7F).is_nan()); // reserved NaN
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9)); // min subnormal
                                                      // Encode picks the nearest representable code.
        assert_eq!(e4m3_from_f32(1.0), 0x38);
        assert_eq!(e4m3_from_f32(448.0), 0x7E);
        assert_eq!(e4m3_from_f32(1e9), 0x7E); // saturates, never NaN
        assert_eq!(e4m3_from_f32(0.0), 0x00);
        assert_eq!(e4m3_from_f32(-3.0), 0x00); // negatives → 0 (scales are non-negative)
    }

    #[test]
    fn e2m1_grid_and_ties() {
        // Every grid magnitude encodes to itself.
        for (i, &m) in E2M1_MAGS.iter().enumerate() {
            assert_eq!(e2m1_from_f32(m), i as u8);
            assert_eq!(e2m1_from_f32(-m) & 0x07, i as u8);
        }
        assert_eq!(E2M1_LUT[e2m1_from_f32(6.0) as usize], 6.0);
        assert_eq!(E2M1_LUT[e2m1_from_f32(100.0) as usize], 6.0); // saturates
                                                                  // Sign preserved.
        assert_eq!(e2m1_from_f32(-1.0) & 0x08, 0x08);
        // Ties-to-even at 2.5 (between 2.0@idx4 and 3.0@idx5) → even index 4 → 2.0.
        assert_eq!(E2M1_LUT[e2m1_from_f32(2.5) as usize], 2.0);
    }

    /// sc-21547. A fused checkpoint stores one matrix where the model wants several, so the
    /// container must be sliceable **without dequantizing**: each slice keeps its own nibbles and
    /// its own re-emitted block scales, and the slices' padded scale surfaces partition the
    /// source's rather than duplicating it. A boundary that cut a scale-factor atom would break
    /// that partition, so it refuses.
    ///
    /// `cols = 128` on purpose (sc-21547 review): that is `n_blocks = 8` → `sf_cols = 8` → **two**
    /// k-atom columns, so [`Nvfp4Tensor::scale_offset_for`]'s `atom_index = m_atom + num_m_atoms *
    /// k_atom` genuinely changes when `num_m_atoms` shrinks from the source's to the slice's, and
    /// the scale bytes must be re-emitted rather than memcpy'd. At one k-atom column (`cols <= 64`)
    /// the remap is the identity and a naive `to_vec()` of the byte range would pass this test
    /// without ever exercising the permutation. The 384-row case adds a **middle** slice, whose
    /// source `m_atom` is neither 0 nor the last.
    #[test]
    fn a_row_slice_is_lossless_and_refuses_a_boundary_that_cuts_a_scale_atom() -> Result<()> {
        let values = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|index| ((index % 97) as f32 - 48.0) / 16.0)
                .collect()
        };
        let (rows, cols) = (256_usize, 128_usize);
        let data = values(rows, cols);
        let tensor = Nvfp4Tensor::pack_from_slice(&data, rows, cols)?;
        assert_eq!(
            tensor.sf_cols, 8,
            "the fixture must span more than one k-atom column, or the atom-index remap below is \
             the identity and a memcpy would pass"
        );
        let whole = tensor.dequantize_to_vec();

        let top = tensor.slice_rows(0, 128)?;
        let bottom = tensor.slice_rows(128, 128)?;
        assert_eq!((top.rows, top.cols), (128, cols));
        assert_eq!(top.global_scale, tensor.global_scale);
        assert_eq!(
            top.scales.len() + bottom.scales.len(),
            tensor.scales.len(),
            "tile-aligned slices take whole atoms, so their padded scale surfaces partition the \
             source's"
        );
        assert_eq!(
            top.packed.len() + bottom.packed.len(),
            tensor.packed.len(),
            "and the nibble payload partitions with them"
        );
        assert_eq!(top.dequantize_to_vec(), whole[..128 * cols]);
        assert_eq!(bottom.dequantize_to_vec(), whole[128 * cols..]);

        // The atom-index remap, stated as bytes: the source has two m-atoms, each slice one, so the
        // second k-atom column moves from source atom 2/3 to slice atom 1. A memcpy of the byte
        // range `[m_atom * 512 .. )` would put those bytes at the wrong offset.
        for (slice, source_start) in [(&top, 0_usize), (&bottom, 128_usize)] {
            for row in [0_usize, 1, 33, 127] {
                for blk in [0_usize, 3, 4, 7] {
                    assert_eq!(
                        slice.scales[Nvfp4Tensor::scale_offset_for(row, blk, 128)],
                        tensor.scales[tensor.scale_offset(source_start + row, blk)],
                        "slice from row {source_start}: scale for (row {row}, block {blk}) must be \
                         re-emitted at the SLICE's atom index, not copied at the source's"
                    );
                }
            }
        }

        // A middle slice of a 3-m-atom tensor: source m-atom 1, neither first nor last.
        let (mid_rows, mid_cols) = (384_usize, 128_usize);
        let mid_data = values(mid_rows, mid_cols);
        let mid_tensor = Nvfp4Tensor::pack_from_slice(&mid_data, mid_rows, mid_cols)?;
        let mid_whole = mid_tensor.dequantize_to_vec();
        let middle = mid_tensor.slice_rows(128, 128)?;
        assert_eq!((middle.rows, middle.sf_rows), (128, 128));
        assert_eq!(
            middle.dequantize_to_vec(),
            mid_whole[128 * mid_cols..256 * mid_cols],
            "the middle slice dequantizes to exactly its rows of the source"
        );
        for row in [0_usize, 1, 33, 127] {
            for blk in [0_usize, 3, 4, 7] {
                assert_eq!(
                    middle.scales[Nvfp4Tensor::scale_offset_for(row, blk, 128)],
                    mid_tensor.scales[mid_tensor.scale_offset(128 + row, blk)],
                    "middle slice: scale for (row {row}, block {blk}) must be re-emitted at the \
                     slice's atom index"
                );
            }
        }

        for (start, len) in [(0, 64), (64, 128), (0, 0)] {
            let error = tensor
                .slice_rows(start, len)
                .expect_err("a misaligned or empty slice must refuse")
                .to_string();
            assert!(error.contains("nvfp4 row slice"), "{error}");
        }
        assert!(tensor.slice_rows(128, 256).is_err(), "out of bounds");
        Ok(())
    }

    #[test]
    fn kitchen_parts_swap_only_the_nibble_order() -> Result<()> {
        let (rows, cols) = (128, 64);
        let mut packed = vec![0u8; rows * cols / 2];
        // Kitchen hi-first: even element code 1 in the high nibble, odd code 2 in the low nibble.
        packed[0] = 0x12;
        let mut scales = vec![0u8; Nvfp4Tensor::scale_tensor_len(rows, cols)];
        scales[Nvfp4Tensor::scale_offset_for(0, 0, rows)] = 0x38; // E4M3 1.0

        let tensor = Nvfp4Tensor::from_kitchen_parts(&packed, &scales, 2.0, rows, cols)?;
        assert_eq!(tensor.packed[0], 0x21);
        assert_eq!(
            tensor.scales, scales,
            "a single-atom weight's blocked scales are byte-exact: the atom-order permutation is \
             the identity when there is only one atom"
        );
        let dense = tensor.dequantize_to_vec();
        assert_eq!(dense[0], 1.0); // E2M1(1) 0.5 * scale 1.0 * global 2.0
        assert_eq!(dense[1], 2.0); // E2M1(2) 1.0 * scale 1.0 * global 2.0
        Ok(())
    }

    /// sc-20641. The case the single-atom test above structurally cannot reach: a weight with more
    /// than one scale atom in **both** dimensions, where ComfyUI's row-major `to_blocked` atom walk
    /// and this container's column-major [`Nvfp4Tensor::scale_offset_for`] disagree.
    ///
    /// Every real DiT projection is in this regime (`[6144, 6144]` → 48 × 48 atoms), so a verbatim
    /// copy of Kitchen's scale buffer would hand cuBLASLt — and the dequant fallback — the right
    /// bytes in the wrong slots. The guard is on the decoded value of a specific element whose
    /// block scale moves under the permutation, not on the buffer as a whole.
    #[test]
    fn kitchen_parts_permute_block_scales_across_a_multi_atom_grid() -> Result<()> {
        // 256 rows = 2 row atoms; 128 cols = 8 blocks = 2 block atoms.
        let (rows, cols) = (256, 128);
        let blocks = cols / NVFP4_BLOCK;
        assert!(
            rows / SF_ATOM_ROWS > 1 && blocks / SF_ATOM_COLS > 1,
            "the fixture must have >1 atom in both dimensions or it proves nothing"
        );

        // Kitchen buffer: give each (row, block) a distinct, recoverable E4M3 code so a wrong slot
        // is a wrong *value*, not just a wrong address. Codes 0x30..0x40 are 0.5..1.875.
        let code_for = |r: usize, blk: usize| -> u8 { 0x30 + ((r / 32 + blk * 3) % 16) as u8 };
        let mut kitchen_scales = vec![0u8; Nvfp4Tensor::scale_tensor_len(rows, cols)];
        for r in 0..rows {
            for blk in 0..blocks {
                kitchen_scales[gen_core::blocked_scale_index(rows, blocks, r, blk)] =
                    code_for(r, blk);
            }
        }
        // Every element code 2 (E2M1 1.0), so the decoded value *is* the block scale × global.
        let packed = vec![0x22_u8; rows * cols / 2];
        let global = 4.0_f32;

        let tensor = Nvfp4Tensor::from_kitchen_parts(&packed, &kitchen_scales, global, rows, cols)?;

        // The permutation is NOT the identity here — this is the assertion the old fixture could
        // not make.
        assert_ne!(
            tensor.scales, kitchen_scales,
            "a multi-atom grid must be permuted, not copied"
        );
        // Each scale landed in the container's own slot...
        for r in 0..rows {
            for blk in 0..blocks {
                assert_eq!(
                    tensor.scales[Nvfp4Tensor::scale_offset_for(r, blk, tensor.sf_rows)],
                    code_for(r, blk),
                    "scale for (row {r}, block {blk})"
                );
            }
        }
        // ...and the container's own dequant therefore recovers the value ComfyUI meant, element by
        // element. Column 100 sits in block 6, whose scale moves atoms under the permutation.
        let dense = tensor.dequantize_to_vec();
        for (r, c) in [(0_usize, 0_usize), (5, 100), (200, 100), (255, 127)] {
            let expected = e4m3_to_f32(code_for(r, c / NVFP4_BLOCK)) * global;
            assert_eq!(dense[r * cols + c], expected, "element ({r}, {c})");
        }
        Ok(())
    }

    // ---- container shape / size (the ~4.5-bit + padded scale-tensor expectation) --------------

    #[test]
    fn packed_size_matches_4p5_bits() {
        let (rows, cols) = (256, 512); // cols already a multiple of 16
        let mut seed = 0x1234_5678u64;
        let data: Vec<f32> = (0..rows * cols).map(|_| prng(&mut seed)).collect();
        let t = Nvfp4Tensor::pack_from_slice(&data, rows, cols).unwrap();

        assert_eq!(t.cols_padded, cols);
        // Nibble bytes: rows * cols/2.
        assert_eq!(t.packed.len(), rows * cols / 2);
        // Scale tensor: round_up(256,128)=256 rows × round_up(512/16=32, 4)=32 cols.
        assert_eq!(t.sf_rows, 256);
        assert_eq!(t.sf_cols, 32);
        assert_eq!(t.scales.len(), 256 * 32);
        assert_eq!(t.scales.len(), Nvfp4Tensor::scale_tensor_len(rows, cols));

        // Effective bits/weight on the block-16 basis: 4 (E2M1) + 8/16 (E4M3) = 4.5.
        let nibble_bits = (t.packed.len() * 8) as f64;
        let scale_bits = (rows * (cols / NVFP4_BLOCK) * 8) as f64; // logical (unpadded) scale count
        let eff = (nibble_bits + scale_bits) / (rows * cols) as f64;
        assert!((eff - 4.5).abs() < 1e-9, "effective bits/weight = {eff}");
    }

    // ---- round-trip within NVFP4 quantization error -------------------------------------------

    #[test]
    fn roundtrip_within_tolerance_with_outliers_and_zeros() {
        let (rows, cols) = (128, 256);
        let mut seed = 0xC0FF_EE00u64;
        let mut data: Vec<f32> = (0..rows * cols)
            .map(|_| prng(&mut seed) * 0.5) // bulk ~ N(0, ~0.3)
            .collect();
        // Inject a few massive-activation-style outliers (the sc-7702 / spike outlier class).
        for k in 0..8 {
            data[k * 37 % (rows * cols)] = if k % 2 == 0 { 50.0 } else { -40.0 };
        }
        // Force one entire near-zero block (row 3, block 1) → must dequant to ~0, not NaN.
        for c in 16..32 {
            data[3 * cols + c] = 1e-6 * prng(&mut seed);
        }

        let t = Nvfp4Tensor::pack_from_slice(&data, rows, cols).unwrap();
        let back = t.dequantize_to_vec();
        assert!(
            back.iter().all(|v| v.is_finite()),
            "no NaN/Inf from dequant"
        );

        let rr = rel_rms(&data, &back);
        assert!(rr < 0.12, "rel-RMS {rr} exceeds NVFP4 tolerance");

        // The near-zero block reconstructs near zero (the rel-RMS bound already covers the outliers,
        // which land within one E2M1 step of their block scale rather than being silently zeroed).
        let zblk = &back[3 * cols + 16..3 * cols + 32];
        assert!(
            zblk.iter().all(|v| v.abs() < 1e-4),
            "near-zero block did not collapse to ~0"
        );
    }

    #[test]
    fn roundtrip_bf16_tensor_path() -> Result<()> {
        let (rows, cols) = (64, 128);
        let mut seed = 0xABCD_1234u64;
        let data: Vec<f32> = (0..rows * cols).map(|_| prng(&mut seed)).collect();
        let w =
            Tensor::from_vec(data.clone(), (rows, cols), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let t = Nvfp4Tensor::pack(&w)?;
        let back = t.dequantize()?;
        assert_eq!(back.dims(), &[rows, cols]);
        // Compare against the bf16-rounded reference (bf16 is the packer's actual input).
        let bf16_ref = w.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let back_v = back.flatten_all()?.to_vec1::<f32>()?;
        let rr = rel_rms(&bf16_ref, &back_v);
        assert!(rr < 0.12, "bf16 round-trip rel-RMS {rr}");
        // Max-abs bound: inputs ∈ [-1,1) → per-block scale ≤ ~1/6, largest E2M1 step ≤ 2·scale, so a
        // single-element error stays well under 0.2 (no outliers in this fixture).
        let ma = max_abs_err(&bf16_ref, &back_v);
        assert!(ma < 0.2, "bf16 round-trip max-abs {ma}");
        Ok(())
    }

    // ---- padding policy: non-multiple-of-16 K -------------------------------------------------

    #[test]
    fn non_multiple_of_16_k_padding() {
        let (rows, cols) = (32, 40); // 40 is not a multiple of 16 → pad to 48
        let mut seed = 0x5EED_5EEDu64;
        let data: Vec<f32> = (0..rows * cols).map(|_| prng(&mut seed)).collect();
        let t = Nvfp4Tensor::pack_from_slice(&data, rows, cols).unwrap();

        assert_eq!(t.cols_padded, 48);
        assert_eq!(t.blocks_per_row(), 3);
        assert_eq!(t.packed.len(), rows * 48 / 2);

        // Dequant returns the logical [32, 40] and round-trips the real columns.
        let back = t.dequantize_to_vec();
        assert_eq!(back.len(), rows * cols);
        assert!(back.iter().all(|v| v.is_finite()));
        let rr = rel_rms(&data, &back);
        assert!(rr < 0.12, "padded-K round-trip rel-RMS {rr}");
    }

    // ---- padding policy: 128×4 scale swizzle --------------------------------------------------

    #[test]
    fn scale_swizzle_padding_and_bijection() {
        // rows=64 (< 128 → pad to 128), cols=32 → 2 blocks (< 4 → pad to 4). One 128×4 atom = 512 B.
        let (rows, cols) = (64, 32);
        let mut seed = 0x0BAD_F00Du64;
        let data: Vec<f32> = (0..rows * cols).map(|_| prng(&mut seed)).collect();
        let t = Nvfp4Tensor::pack_from_slice(&data, rows, cols).unwrap();

        assert_eq!(t.sf_rows, 128);
        assert_eq!(t.sf_cols, 4);
        assert_eq!(t.scales.len(), 512);
        assert_eq!(t.scales.len(), Nvfp4Tensor::scale_tensor_len(rows, cols));

        // The swizzle is a bijection over the full padded (row, block) grid onto [0, 512).
        let mut seen = vec![false; 512];
        for r in 0..t.sf_rows {
            for blk in 0..t.sf_cols {
                let off = Nvfp4Tensor::scale_offset_for(r, blk, t.sf_rows);
                assert!(off < 512, "offset {off} out of atom");
                assert!(
                    !seen[off],
                    "collision at offset {off} for (r={r}, blk={blk})"
                );
                seen[off] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "swizzle did not cover every byte");

        // Padded scale bytes (rows >= 64, or block-cols >= 2) are untouched zeros.
        for r in 64..128 {
            for blk in 0..4 {
                assert_eq!(t.scales[Nvfp4Tensor::scale_offset_for(r, blk, 128)], 0);
            }
        }
        for r in 0..64 {
            for blk in 2..4 {
                assert_eq!(t.scales[Nvfp4Tensor::scale_offset_for(r, blk, 128)], 0);
            }
        }

        // And the real data still round-trips.
        let rr = rel_rms(&data, &t.dequantize_to_vec());
        assert!(rr < 0.12, "swizzle round-trip rel-RMS {rr}");
    }

    // ---- scale swizzle: ORDER, pinned to the external producer --------------------------------

    /// sc-20651 review (9). `scale_swizzle_padding_and_bijection` above proves the swizzle is a
    /// **bijection**; it cannot see the **order**. Neither can any round-trip test in this crate: the
    /// packer writes through [`Nvfp4Tensor::scale_offset_for`] and the dequant reads back through the
    /// same function, so *any* bijective permutation of the offset — a transpose of the intra-atom
    /// slot, a transpose of the atom grid — cancels exactly and is invisible end to end. The order is
    /// only load-bearing at the two seams where a **foreign** producer/consumer is on the other side:
    /// ComfyUI Kitchen's `comfy_kitchen.float_utils.to_blocked` buffer on the way in, and cuBLASLt's
    /// `VEC16` `UE4M3` block-scale descriptor on the way out.
    ///
    /// So this pins the offset against the external producer instead of against itself.
    /// [`gen_core::blocked_scale_index`] *is* `to_blocked`, transliterated from its definition and
    /// itself pinned in `gen-core` against a matrix computed by running `to_blocked` in a real ComfyUI
    /// venv (`mxfp8_swizzle_index_matches_comfy_kitchen_to_blocked`). The two layouts are the same
    /// cuBLAS 128×4 swizzle and differ in exactly one documented way — Kitchen walks the atom grid
    /// row-major, this container walks it column-major (CuTe `LayoutLeft`) — so:
    ///
    /// - the **intra-atom slot** (`offset % 512`) must be *equal* to Kitchen's, and
    /// - the **atom index** (`offset / 512`) must be Kitchen's atom index *transposed*.
    ///
    /// Both halves are stated in terms of `blocked_scale_index`, never by re-spelling
    /// `scale_offset_for`'s own arithmetic, so a permutation applied to the production function has
    /// nothing to cancel against.
    ///
    /// Live cuBLASLt confirmation of the atom order (the second seam) is still the terminal CUDA-box
    /// gate for `nvfp4_native=true`; this is the host-side half.
    #[test]
    fn scale_offset_order_is_pinned_to_the_external_to_blocked_producer() {
        const ATOM: usize = SF_ATOM_ROWS * SF_ATOM_COLS; // 512 bytes

        // A real-DiT-shaped multi-atom grid: 256 rows = 2 row atoms, 8 blocks = 2 block atoms. With a
        // single atom in either dimension the two atom walks coincide and this proves nothing.
        let (rows, blocks) = (256_usize, 8_usize);
        let sf_rows = round_up(rows, SF_ATOM_ROWS);
        let (num_m_atoms, num_k_atoms) = (sf_rows / SF_ATOM_ROWS, blocks / SF_ATOM_COLS);
        assert!(
            num_m_atoms > 1 && num_k_atoms > 1,
            "the fixture must have >1 atom in both dimensions or the atom-order half is vacuous"
        );

        for r in 0..rows {
            for blk in 0..blocks {
                let ours = Nvfp4Tensor::scale_offset_for(r, blk, sf_rows);
                let kitchen = gen_core::blocked_scale_index(rows, blocks, r, blk);

                // (i) Intra-atom slot: identical to `to_blocked`'s. This is the half the module docs
                // call "the fixed, load-bearing part" — a transpose of `intra` breaks it here.
                assert_eq!(
                    ours % ATOM,
                    kitchen % ATOM,
                    "intra-atom slot for (row {r}, block {blk}) must match comfy `to_blocked`"
                );

                // (ii) Atom index: Kitchen is row-major over the atom grid, this container is
                // column-major, so ours must be the transpose of Kitchen's — not merely different.
                let kitchen_atom = kitchen / ATOM;
                let transposed =
                    (kitchen_atom % num_k_atoms) * num_m_atoms + (kitchen_atom / num_k_atoms);
                assert_eq!(
                    ours / ATOM,
                    transposed,
                    "atom index for (row {r}, block {blk}) must be `to_blocked`'s atom transposed"
                );
            }
        }

        // Two concrete bytes of the documented CUTLASS layout, so a reader can check the claim by
        // hand and so a mutation that happens to preserve the relations above still trips here.
        // Intra-atom: row 1 → slot 16 (`(1 % 32) * 16`); row 32 → slot 4 (`(32 / 32) * 4`). A
        // transpose of the intra formula swaps exactly these two.
        assert_eq!(Nvfp4Tensor::scale_offset_for(1, 0, 128), 16);
        assert_eq!(Nvfp4Tensor::scale_offset_for(32, 0, 128), 4);
        // Atom order, on the 2×2 grid: (row 0, block 4) is atom 2 column-major → byte 1024, where
        // `to_blocked` puts it at 512 (gen-core pins that literal). The disagreement is the whole
        // reason `from_kitchen_parts` permutes rather than copies.
        assert_eq!(Nvfp4Tensor::scale_offset_for(0, 4, 256), 1024);
        assert_eq!(gen_core::blocked_scale_index(256, 8, 0, 4), 512);
    }

    // ---- non-finite inputs are refused, not absorbed -------------------------------------------

    /// sc-20651 review (minor). A NaN weight would be dropped by the `f32::max` amax fold and then
    /// pack as E2M1 code `0`; a single `+inf` would make `global_scale` infinite and zero **every**
    /// element. Both are silent — the container comes back well-formed and wrong, which for an
    /// imported checkpoint is an all-zero layer that renders black rather than a load error.
    #[test]
    fn pack_from_slice_refuses_non_finite_inputs() {
        let (rows, cols) = (2_usize, 16_usize);

        for (label, bad, at) in [("NaN", f32::NAN, 3_usize), ("inf", f32::INFINITY, 17)] {
            let mut data = vec![0.25_f32; rows * cols];
            data[at] = bad;
            let err = Nvfp4Tensor::pack_from_slice(&data, rows, cols)
                .expect_err("a non-finite weight element must be refused");
            let msg = err.to_string();
            // The refusal names the offending element and its value — not a bare "invalid input".
            assert!(
                msg.contains(&format!("weight element {at}")),
                "{label}: refusal must name the element index, got {msg}"
            );
            assert!(
                msg.contains(&format!("row {}, col {}", at / cols, at % cols)),
                "{label}: refusal must name the (row, col), got {msg}"
            );
            assert!(
                msg.contains("must be finite"),
                "{label}: refusal must name the violated property, got {msg}"
            );
        }

        // -inf too, and the all-finite fixture still packs (the gate is not a blanket refusal).
        let mut data = vec![0.25_f32; rows * cols];
        data[0] = f32::NEG_INFINITY;
        assert!(Nvfp4Tensor::pack_from_slice(&data, rows, cols).is_err());
        data[0] = 0.25;
        assert!(Nvfp4Tensor::pack_from_slice(&data, rows, cols).is_ok());
    }

    /// The Kitchen constructor's per-tensor-scale gate shares [`non_finite_msg`] with the packer, so
    /// both name a NaN/inf the same way. Non-negativity stays a separate, separately-worded refusal.
    #[test]
    fn kitchen_parts_refuses_a_non_finite_or_negative_global_scale() {
        let (rows, cols) = (128_usize, 64_usize);
        let packed = vec![0u8; rows * cols / 2];
        let scales = vec![0u8; Nvfp4Tensor::scale_tensor_len(rows, cols)];

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let msg = Nvfp4Tensor::from_kitchen_parts(&packed, &scales, bad, rows, cols)
                .expect_err("non-finite global scale must be refused")
                .to_string();
            assert!(msg.contains("must be finite"), "got {msg}");
        }
        let msg = Nvfp4Tensor::from_kitchen_parts(&packed, &scales, -1.0, rows, cols)
            .expect_err("negative global scale must be refused")
            .to_string();
        assert!(msg.contains("must be non-negative"), "got {msg}");
    }

    // ---- nibble packing convention ------------------------------------------------------------

    #[test]
    fn nibble_low_high_order_matches_mxfp4() {
        // Column 0 → low nibble of byte 0, column 1 → high nibble of byte 0.
        let (rows, cols) = (1, 16);
        // Values chosen so codes differ between even/odd columns.
        let mut data = vec![0f32; cols];
        data[0] = 6.0; // will map to the block-max magnitude
        data[1] = -6.0;
        let t = Nvfp4Tensor::pack_from_slice(&data, rows, cols).unwrap();
        let byte0 = t.packed[0];
        let (low, high) = (byte0 & 0x0F, byte0 >> 4);
        assert_eq!(low & 0x08, 0x00, "col 0 (+6) low nibble positive");
        assert_eq!(high & 0x08, 0x08, "col 1 (-6) high nibble negative");
    }
}
