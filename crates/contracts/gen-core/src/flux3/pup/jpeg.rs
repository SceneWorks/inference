//! The **pixel effect** of `image.save(buffer, format="JPEG", quality=92)` followed by
//! `Image.open(buffer).convert("RGB")` in Pillow — i.e. libjpeg-turbo's baseline encoder and
//! decoder at their defaults (ISLOW integer DCTs, 4:2:0 chroma, the Annex K tables scaled for
//! quality 92, "fancy" triangle chroma upsampling) — reproduced with libjpeg's own fixed-point
//! arithmetic so the upsampler sees the same reference bytes upstream's VLM saw.
//!
//! Nothing is entropy-coded: the quantized coefficients go straight from the forward transform
//! to the inverse one, which is exactly what a JPEG file carries (Huffman coding is lossless), so
//! the output is the decoder's output, bit for bit. The sources are `jccolor.c`,
//! `jcsample.c` (`h2v2_downsample`), `jfdctint.c`, `jcdctmgr.c` (`quantize`),
//! `jddctmgr.c` / `jidctint.c`, `jdsample.c` (`h2v2_fancy_upsample`) and `jdcolor.c`; the
//! libjpeg-turbo SIMD paths are specified to match them.
//!
//! The reference policy only ever produces sides that are multiples of 32, so the iMCU padding
//! the codec applies to other sizes is not reproduced: any other size is refused.

use crate::{Error, Result};

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;
const FIX_0_298631336: i64 = 2446;
const FIX_0_390180644: i64 = 3196;
const FIX_0_541196100: i64 = 4433;
const FIX_0_765366865: i64 = 6270;
const FIX_0_899976223: i64 = 7373;
const FIX_1_175875602: i64 = 9633;
const FIX_1_501321110: i64 = 12299;
const FIX_1_847759065: i64 = 15137;
const FIX_1_961570560: i64 = 16069;
const FIX_2_053119869: i64 = 16819;
const FIX_2_562915447: i64 = 20995;
const FIX_3_072711026: i64 = 25172;

/// Annex K.1 luminance quantization table, natural order.
const STD_LUMINANCE: [i32; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
/// Annex K.2 chrominance quantization table, natural order.
const STD_CHROMINANCE: [i32; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// `jpeg_quality_scaling` + `jpeg_add_quant_table(…, force_baseline = TRUE)`.
fn scaled_table(basic: &[i32; 64], quality: u8) -> [i32; 64] {
    let quality = i32::from(quality).clamp(1, 100);
    let scale = if quality < 50 {
        5000 / quality
    } else {
        200 - quality * 2
    };
    let mut table = [0; 64];
    for (slot, &base) in table.iter_mut().zip(basic) {
        *slot = ((base * scale + 50) / 100).clamp(1, 255);
    }
    table
}

#[inline]
fn descale(x: i64, n: i32) -> i64 {
    (x + (1i64 << (n - 1))) >> n
}

/// `jpeg_fdct_islow` over one 8×8 block of level-shifted samples, in place.
fn fdct_islow(data: &mut [i64; 64]) {
    // Pass 1: rows.
    for row in data.chunks_exact_mut(8) {
        let tmp0 = row[0] + row[7];
        let tmp7 = row[0] - row[7];
        let tmp1 = row[1] + row[6];
        let tmp6 = row[1] - row[6];
        let tmp2 = row[2] + row[5];
        let tmp5 = row[2] - row[5];
        let tmp3 = row[3] + row[4];
        let tmp4 = row[3] - row[4];
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        row[0] = (tmp10 + tmp11) << PASS1_BITS;
        row[4] = (tmp10 - tmp11) << PASS1_BITS;
        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        row[2] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS - PASS1_BITS);
        row[6] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS - PASS1_BITS);
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;
        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = z1 * (-FIX_0_899976223);
        let z2 = z2 * (-FIX_2_562915447);
        let z3 = z3 * (-FIX_1_961570560) + z5;
        let z4 = z4 * (-FIX_0_390180644) + z5;
        row[7] = descale(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        row[5] = descale(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        row[3] = descale(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        row[1] = descale(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);
    }
    // Pass 2: columns.
    for c in 0..8 {
        let d = |r: usize| data[r * 8 + c];
        let tmp0 = d(0) + d(7);
        let tmp7 = d(0) - d(7);
        let tmp1 = d(1) + d(6);
        let tmp6 = d(1) - d(6);
        let tmp2 = d(2) + d(5);
        let tmp5 = d(2) - d(5);
        let tmp3 = d(3) + d(4);
        let tmp4 = d(3) - d(4);
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        data[c] = descale(tmp10 + tmp11, PASS1_BITS);
        data[4 * 8 + c] = descale(tmp10 - tmp11, PASS1_BITS);
        let z1 = (tmp12 + tmp13) * FIX_0_541196100;
        data[2 * 8 + c] = descale(z1 + tmp13 * FIX_0_765366865, CONST_BITS + PASS1_BITS);
        data[6 * 8 + c] = descale(z1 + tmp12 * (-FIX_1_847759065), CONST_BITS + PASS1_BITS);
        let z1 = tmp4 + tmp7;
        let z2 = tmp5 + tmp6;
        let z3 = tmp4 + tmp6;
        let z4 = tmp5 + tmp7;
        let z5 = (z3 + z4) * FIX_1_175875602;
        let tmp4 = tmp4 * FIX_0_298631336;
        let tmp5 = tmp5 * FIX_2_053119869;
        let tmp6 = tmp6 * FIX_3_072711026;
        let tmp7 = tmp7 * FIX_1_501321110;
        let z1 = z1 * (-FIX_0_899976223);
        let z2 = z2 * (-FIX_2_562915447);
        let z3 = z3 * (-FIX_1_961570560) + z5;
        let z4 = z4 * (-FIX_0_390180644) + z5;
        data[7 * 8 + c] = descale(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS);
        data[5 * 8 + c] = descale(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS);
        data[3 * 8 + c] = descale(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS);
        data[8 + c] = descale(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS);
    }
}

/// `jcdctmgr.c quantize`: the forward DCT output is scaled by 8, so the divisor is `8·q`;
/// round-half-away-from-zero division.
fn quantize(coefs: &mut [i64; 64], table: &[i32; 64]) {
    for (c, &q) in coefs.iter_mut().zip(table) {
        let qval = i64::from(q) << 3;
        let temp = *c;
        *c = if temp < 0 {
            -((-temp + (qval >> 1)) / qval)
        } else {
            (temp + (qval >> 1)) / qval
        };
    }
}

/// `jpeg_idct_islow` over one dequantized 8×8 block, producing 8-bit samples (the range-limit
/// table of libjpeg adds the level shift and clamps).
fn idct_islow(coefs: &[i64; 64], table: &[i32; 64], out: &mut [u8; 64]) {
    let mut ws = [0i64; 64];
    // Pass 1: columns.
    for c in 0..8 {
        let deq = |r: usize| coefs[r * 8 + c] * i64::from(table[r * 8 + c]);
        let z2 = deq(2);
        let z3 = deq(6);
        let z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 + z3 * (-FIX_1_847759065);
        let tmp3 = z1 + z2 * FIX_0_765366865;
        let z2 = deq(0);
        let z3 = deq(4);
        let tmp0 = (z2 + z3) << CONST_BITS;
        let tmp1 = (z2 - z3) << CONST_BITS;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        let tmp0 = deq(7);
        let tmp1 = deq(5);
        let tmp2 = deq(3);
        let tmp3 = deq(1);
        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * FIX_1_175875602;
        let tmp0 = tmp0 * FIX_0_298631336;
        let tmp1 = tmp1 * FIX_2_053119869;
        let tmp2 = tmp2 * FIX_3_072711026;
        let tmp3 = tmp3 * FIX_1_501321110;
        let z1 = z1 * (-FIX_0_899976223);
        let z2 = z2 * (-FIX_2_562915447);
        let z3 = z3 * (-FIX_1_961570560) + z5;
        let z4 = z4 * (-FIX_0_390180644) + z5;
        let tmp0 = tmp0 + z1 + z3;
        let tmp1 = tmp1 + z2 + z4;
        let tmp2 = tmp2 + z2 + z3;
        let tmp3 = tmp3 + z1 + z4;
        let n = CONST_BITS - PASS1_BITS;
        ws[c] = descale(tmp10 + tmp3, n);
        ws[7 * 8 + c] = descale(tmp10 - tmp3, n);
        ws[8 + c] = descale(tmp11 + tmp2, n);
        ws[6 * 8 + c] = descale(tmp11 - tmp2, n);
        ws[2 * 8 + c] = descale(tmp12 + tmp1, n);
        ws[5 * 8 + c] = descale(tmp12 - tmp1, n);
        ws[3 * 8 + c] = descale(tmp13 + tmp0, n);
        ws[4 * 8 + c] = descale(tmp13 - tmp0, n);
    }
    // Pass 2: rows.
    let limit = |v: i64| -> u8 { (v + 128).clamp(0, 255) as u8 };
    for r in 0..8 {
        let w = &ws[r * 8..r * 8 + 8];
        let z2 = w[2];
        let z3 = w[6];
        let z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 + z3 * (-FIX_1_847759065);
        let tmp3 = z1 + z2 * FIX_0_765366865;
        let tmp0 = (w[0] + w[4]) << CONST_BITS;
        let tmp1 = (w[0] - w[4]) << CONST_BITS;
        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;
        let tmp0 = w[7];
        let tmp1 = w[5];
        let tmp2 = w[3];
        let tmp3 = w[1];
        let z1 = tmp0 + tmp3;
        let z2 = tmp1 + tmp2;
        let z3 = tmp0 + tmp2;
        let z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * FIX_1_175875602;
        let tmp0 = tmp0 * FIX_0_298631336;
        let tmp1 = tmp1 * FIX_2_053119869;
        let tmp2 = tmp2 * FIX_3_072711026;
        let tmp3 = tmp3 * FIX_1_501321110;
        let z1 = z1 * (-FIX_0_899976223);
        let z2 = z2 * (-FIX_2_562915447);
        let z3 = z3 * (-FIX_1_961570560) + z5;
        let z4 = z4 * (-FIX_0_390180644) + z5;
        let tmp0 = tmp0 + z1 + z3;
        let tmp1 = tmp1 + z2 + z4;
        let tmp2 = tmp2 + z2 + z3;
        let tmp3 = tmp3 + z1 + z4;
        let n = CONST_BITS + PASS1_BITS + 3;
        let o = &mut out[r * 8..r * 8 + 8];
        o[0] = limit(descale(tmp10 + tmp3, n));
        o[7] = limit(descale(tmp10 - tmp3, n));
        o[1] = limit(descale(tmp11 + tmp2, n));
        o[6] = limit(descale(tmp11 - tmp2, n));
        o[2] = limit(descale(tmp12 + tmp1, n));
        o[5] = limit(descale(tmp12 - tmp1, n));
        o[3] = limit(descale(tmp13 + tmp0, n));
        o[4] = limit(descale(tmp13 - tmp0, n));
    }
}

/// Forward DCT + quantize + dequantize + inverse DCT over a whole plane (block by block).
fn codec_plane(plane: &[u8], width: usize, height: usize, table: &[i32; 64]) -> Vec<u8> {
    let mut out = vec![0u8; plane.len()];
    let mut block = [0i64; 64];
    let mut samples = [0u8; 64];
    for by in 0..height / 8 {
        for bx in 0..width / 8 {
            for r in 0..8 {
                for c in 0..8 {
                    block[r * 8 + c] = i64::from(plane[(by * 8 + r) * width + bx * 8 + c]) - 128;
                }
            }
            fdct_islow(&mut block);
            quantize(&mut block, table);
            idct_islow(&block, table, &mut samples);
            for r in 0..8 {
                let row = (by * 8 + r) * width + bx * 8;
                out[row..row + 8].copy_from_slice(&samples[r * 8..r * 8 + 8]);
            }
        }
    }
    out
}

/// `jcsample.c h2v2_downsample`: 2×2 box average with the alternating `1, 2` rounding bias.
fn h2v2_downsample(plane: &[u8], width: usize, height: usize) -> Vec<u8> {
    let (ow, oh) = (width / 2, height / 2);
    let mut out = Vec::with_capacity(ow * oh);
    for oy in 0..oh {
        let row0 = &plane[(2 * oy) * width..(2 * oy + 1) * width];
        let row1 = &plane[(2 * oy + 1) * width..(2 * oy + 2) * width];
        let mut bias = 1u32;
        for ox in 0..ow {
            let sum = u32::from(row0[2 * ox])
                + u32::from(row0[2 * ox + 1])
                + u32::from(row1[2 * ox])
                + u32::from(row1[2 * ox + 1]);
            out.push(((sum + bias) >> 2) as u8);
            bias ^= 3;
        }
    }
    out
}

/// `jdsample.c h2v2_fancy_upsample`: the triangle filter (3/4 · nearer + 1/4 · further) in both
/// directions, with libjpeg's `+8` / `+7` alternating rounding and replicated edge context rows.
fn h2v2_fancy_upsample(plane: &[u8], width: usize, height: usize) -> Vec<u8> {
    let (ow, oh) = (width * 2, height * 2);
    let mut out = vec![0u8; ow * oh];
    for inrow in 0..height {
        for v in 0..2 {
            let near = &plane[inrow * width..(inrow + 1) * width];
            let far_row = if v == 0 {
                inrow.saturating_sub(1)
            } else {
                (inrow + 1).min(height - 1)
            };
            let far = &plane[far_row * width..(far_row + 1) * width];
            let outrow = 2 * inrow + v;
            let o = &mut out[outrow * ow..(outrow + 1) * ow];
            let colsum = |c: usize| u32::from(near[c]) * 3 + u32::from(far[c]);
            if width == 1 {
                let s = colsum(0);
                o[0] = ((s * 4 + 8) >> 4) as u8;
                o[1] = ((s * 4 + 7) >> 4) as u8;
                continue;
            }
            let mut this = colsum(0);
            let mut next = colsum(1);
            o[0] = ((this * 4 + 8) >> 4) as u8;
            o[1] = ((this * 3 + next + 7) >> 4) as u8;
            let mut last = this;
            this = next;
            for col in 1..width - 1 {
                next = colsum(col + 1);
                o[2 * col] = ((this * 3 + last + 8) >> 4) as u8;
                o[2 * col + 1] = ((this * 3 + next + 7) >> 4) as u8;
                last = this;
                this = next;
            }
            o[2 * (width - 1)] = ((this * 3 + last + 8) >> 4) as u8;
            o[2 * (width - 1) + 1] = ((this * 4 + 7) >> 4) as u8;
        }
    }
    out
}

const SCALEBITS: i32 = 16;
const ONE_HALF: i64 = 1 << (SCALEBITS - 1);
const CBCR_OFFSET: i64 = 128 << SCALEBITS;

/// `jccolor.c rgb_ycc_convert` fixed-point constants (`FIX(x) = x · 65536 + 0.5`).
const FIX_0_29900: i64 = 19595;
const FIX_0_58700: i64 = 38470;
const FIX_0_11400: i64 = 7471;
const FIX_0_16874: i64 = 11059;
const FIX_0_33126: i64 = 21709;
const FIX_0_50000: i64 = 32768;
const FIX_0_41869: i64 = 27439;
const FIX_0_08131: i64 = 5329;
/// `jdcolor.c build_ycc_rgb_table` constants.
const FIX_1_40200: i64 = 91881;
const FIX_1_77200: i64 = 116130;
const FIX_0_71414: i64 = 46802;
const FIX_0_34414: i64 = 22554;

fn rgb_to_ycc(rgb: &[u8]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let n = rgb.len() / 3;
    let (mut y, mut cb, mut cr) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for px in rgb.chunks_exact(3) {
        let (r, g, b) = (i64::from(px[0]), i64::from(px[1]), i64::from(px[2]));
        y.push(
            ((FIX_0_29900 * r + FIX_0_58700 * g + FIX_0_11400 * b + ONE_HALF) >> SCALEBITS) as u8,
        );
        cb.push(
            ((-FIX_0_16874 * r - FIX_0_33126 * g + FIX_0_50000 * b + CBCR_OFFSET + ONE_HALF - 1)
                >> SCALEBITS) as u8,
        );
        cr.push(
            ((FIX_0_50000 * r - FIX_0_41869 * g - FIX_0_08131 * b + CBCR_OFFSET + ONE_HALF - 1)
                >> SCALEBITS) as u8,
        );
    }
    (y, cb, cr)
}

fn ycc_to_rgb(y: &[u8], cb: &[u8], cr: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(y.len() * 3);
    let clamp = |v: i64| v.clamp(0, 255) as u8;
    for i in 0..y.len() {
        let yy = i64::from(y[i]);
        let xb = i64::from(cb[i]) - 128;
        let xr = i64::from(cr[i]) - 128;
        let cr_r = (FIX_1_40200 * xr + ONE_HALF) >> SCALEBITS;
        let cb_b = (FIX_1_77200 * xb + ONE_HALF) >> SCALEBITS;
        let g_off = ((-FIX_0_34414 * xb + ONE_HALF) + (-FIX_0_71414 * xr)) >> SCALEBITS;
        out.push(clamp(yy + cr_r));
        out.push(clamp(yy + g_off));
        out.push(clamp(yy + cb_b));
    }
    out
}

/// The pixels Pillow decodes back from its own quality-`quality` baseline JPEG of `rgb`
/// (`width × height × 3`, sides multiples of 16).
pub fn round_trip(rgb: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || w % 16 != 0 || h % 16 != 0 {
        return Err(Error::Msg(format!(
            "flux3 upsampler: JPEG round trip needs sides that are multiples of 16, got {width}x{height}"
        )));
    }
    if rgb.len() != w * h * 3 {
        return Err(Error::Msg(format!(
            "flux3 upsampler: JPEG round trip buffer of {} bytes is not {width}x{height}x3",
            rgb.len()
        )));
    }
    let luma = scaled_table(&STD_LUMINANCE, quality);
    let chroma = scaled_table(&STD_CHROMINANCE, quality);
    let (y, cb, cr) = rgb_to_ycc(rgb);
    let y = codec_plane(&y, w, h, &luma);
    let cb = codec_plane(&h2v2_downsample(&cb, w, h), w / 2, h / 2, &chroma);
    let cr = codec_plane(&h2v2_downsample(&cr, w, h), w / 2, h / 2, &chroma);
    let cb = h2v2_fancy_upsample(&cb, w / 2, h / 2);
    let cr = h2v2_fancy_upsample(&cr, w / 2, h / 2);
    Ok(ycc_to_rgb(&y, &cb, &cr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_92_scales_the_annex_k_tables_like_libjpeg() {
        let luma = scaled_table(&STD_LUMINANCE, 92);
        // scale = 200 - 184 = 16: 16 -> (256 + 50) / 100 = 3, 11 -> 2, 121 -> 19.
        assert_eq!(luma[0], 3);
        assert_eq!(luma[1], 2);
        assert_eq!(luma[53], 19);
        assert_eq!(scaled_table(&STD_CHROMINANCE, 92)[0], 3);
        assert_eq!(scaled_table(&STD_LUMINANCE, 100)[0], 1);
        assert_eq!(scaled_table(&STD_LUMINANCE, 1)[63], 255, "baseline clamp");
    }

    #[test]
    fn a_flat_block_survives_the_transform_pair_exactly() {
        let rgb = vec![200u8; 16 * 16 * 3];
        let out = round_trip(&rgb, 16, 16, 92).unwrap();
        assert!(
            out.iter().all(|&v| (199..=201).contains(&v)),
            "{:?}",
            &out[..6]
        );
        assert!(round_trip(&rgb, 12, 16, 92).is_err());
        assert!(round_trip(&rgb[..10], 16, 16, 92).is_err());
    }

    #[test]
    fn fancy_upsampling_reproduces_the_triangle_weights() {
        // One 2x1 chroma row: [0, 16] -> [0, 4, 12, 16] horizontally (3/4, 1/4 with +8/+7).
        let up = h2v2_fancy_upsample(&[0, 16], 2, 1);
        assert_eq!(up, vec![0, 4, 12, 16, 0, 4, 12, 16]);
        assert_eq!(
            h2v2_downsample(&[1, 2, 3, 4], 2, 2),
            vec![(1 + 2 + 3 + 4 + 1) >> 2]
        );
    }

    #[test]
    fn colour_conversion_matches_libjpeg_tables() {
        let (y, cb, cr) = rgb_to_ycc(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]);
        assert_eq!((y[0], cb[0], cr[0]), (76, 85, 255));
        assert_eq!((y[1], cb[1], cr[1]), (150, 44, 21));
        assert_eq!((y[2], cb[2], cr[2]), (29, 255, 107));
        assert_eq!((y[3], cb[3], cr[3]), (128, 128, 128));
        assert_eq!(ycc_to_rgb(&[128], &[128], &[128]), vec![128, 128, 128]);
        // libjpeg's own round trip of pure red lands on 254 (the familiar YCbCr loss).
        assert_eq!(ycc_to_rgb(&[76], &[85], &[255]), vec![254, 0, 0]);
    }
}
