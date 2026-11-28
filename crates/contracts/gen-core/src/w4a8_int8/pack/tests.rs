//! Packer / CPU dequant reference goldens. Every expected value here is derived from the format
//! definition in the parent module's docs (nibble order, E4M3FN encoding, grid rounding, the
//! regular Hadamard), never from a measured device value.

use super::*;
use crate::w4a8_int8::{dequant_packed_to_int8, regular_hadamard, W4A8DescriptorError, W4A8Int8};

/// Deterministic standard-normal samples (64-bit LCG + Box-Muller) — no RNG dependency.
fn gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut uniform = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
        })
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

#[test]
fn e4m3_encode_inverts_every_finite_code_and_rounds_ties_to_even() {
    use crate::comfy_quant::fp8_e4m3fn_to_f32 as dec;
    for code in (0u8..=0x7E).chain(0x80..=0xFE) {
        assert_eq!(f32_to_e4m3fn(dec(code)), code, "code {code:#04x}");
    }
    // Midpoints between adjacent positive codes resolve to the even (mantissa LSB 0) code.
    for code in 0u8..0x7E {
        let mid = (dec(code) + dec(code + 1)) / 2.0;
        let even = if code % 2 == 0 { code } else { code + 1 };
        assert_eq!(f32_to_e4m3fn(mid), even, "midpoint above {code:#04x}");
    }
    assert_eq!(f32_to_e4m3fn(448.0), 0x7E);
    assert_eq!(f32_to_e4m3fn(1.0e6), 0x7E, "saturates, never NaN");
    assert_eq!(f32_to_e4m3fn(-1.0e6), 0xFE);
    assert_eq!(f32_to_e4m3fn(1.0), 0x38);
    assert_eq!(f32_to_e4m3fn(129.503), 0x70, "129.5 rounds to 128");
    assert_eq!(f32_to_e4m3fn(f32::NAN), 0x7F);
}

#[test]
fn nibble_golden_even_column_is_the_low_nibble() {
    // Format definition: byte = code[2j] | code[2j+1] << 4.
    let codes = [0x1u8, 0xA, 0xF, 0x0, 0x7, 0x8];
    assert_eq!(pack_nibbles(&codes), vec![0xA1, 0x0F, 0x87]);
    assert_eq!(unpack_nibbles(&[0xA1, 0x0F, 0x87]), codes.to_vec());
}

/// The rotated-basis construction behind the packer golden: row `r`, group `g`, column `j` holds
/// code `(j + g + r) % 16` — so every group contains code 15, whose level `0.980671` is the group's
/// max magnitude — scaled by one per-row float scale.
fn golden_codes(n: usize, k: usize) -> Vec<u8> {
    (0..n * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            ((c % GROUP_SIZE + c / GROUP_SIZE + r) % 16) as u8
        })
        .collect()
}

fn golden_weight(n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let scales: Vec<f32> = (0..n).map(|r| 0.01 * (r as f32 + 1.0)).collect();
    let mut w: Vec<f32> = golden_codes(n, k)
        .iter()
        .enumerate()
        .map(|(i, &q)| DEFAULT_CODEBOOK[q as usize] * scales[i / k])
        .collect();
    // Back to the original basis (R is an involution): the packer's own rotation recovers it.
    rotate_rows(&mut w, k, CONVROT_GROUP);
    (w, scales)
}

/// Golden from the format definition. Each rotated group is exactly `cb[q] · S`, so:
/// * the codes are `q` (nearest level, preserved by ALS and by the int8 grid);
/// * `s_channel = cb[15] · S / 127`;
/// * `s_rel = e4m3(S / s_channel) = e4m3(127 / 0.980671 = 129.50) = 0x70` (128);
/// * the stored byte is `q[2j] | q[2j+1] << 4`;
/// * the decoded int8 is `round_ties_even(cb[q] · 128)`.
#[test]
fn packer_golden_from_the_format_definition() {
    let (n, k) = (3, 512);
    let (w, scales) = golden_weight(n, k);
    let p = quantize_weight(&w, n, k).unwrap();
    let codes = golden_codes(n, k);
    let want_bytes: Vec<u8> = codes.chunks_exact(2).map(|q| q[0] | (q[1] << 4)).collect();
    assert_eq!(p.packed, want_bytes, "packed nibbles");
    assert!(p.s_rel.iter().all(|&b| b == 0x70), "s_rel = e4m3(128)");
    for (r, &s) in p.s_channel.iter().enumerate() {
        let want = DEFAULT_CODEBOOK[15] * scales[r] / 127.0;
        assert!(
            (s - want).abs() <= want * 1e-5,
            "s_channel[{r}] {s} vs {want}"
        );
    }
    let want_int8: Vec<i8> = codes
        .iter()
        .map(|&q| (DEFAULT_CODEBOOK[q as usize] * 128.0).round_ties_even() as i8)
        .collect();
    assert_eq!(p.dequant_int8(), want_int8);
    assert_eq!(p.codebook, DEFAULT_CODEBOOK);
}

/// Mutation check for the golden above: swapping the nibble order (odd column low) produces bytes
/// the golden rejects, and those bytes decode to a different int8 weight. If the packer (or the
/// decoder) ever swapped the order, `packer_golden_from_the_format_definition` would fail.
#[test]
fn swapped_nibble_order_fails_the_golden() {
    let (n, k) = (3, 512);
    let (w, _) = golden_weight(n, k);
    let p = quantize_weight(&w, n, k).unwrap();
    let swapped: Vec<u8> = golden_codes(n, k)
        .chunks_exact(2)
        .map(|q| q[1] | (q[0] << 4))
        .collect();
    assert_ne!(p.packed, swapped, "the golden discriminates nibble order");
    let mutated = W4A8Packed {
        packed: swapped,
        ..p.clone()
    };
    assert_ne!(mutated.dequant_int8(), p.dequant_int8());
    assert!(cosine(&decode_dense(&mutated), &w) < 0.9);
}

#[test]
fn rotation_is_the_regular_hadamard_and_an_involution() {
    let k = 256;
    let x = gaussian(2 * k, 7);
    let mut y = x.clone();
    rotate_rows(&mut y, k, CONVROT_GROUP);
    let h = regular_hadamard(k);
    for r in 0..2 {
        for j in 0..k {
            let want: f64 = (0..k)
                .map(|i| x[r * k + i] as f64 * h[i * k + j] as f64)
                .sum();
            assert!((y[r * k + j] as f64 - want).abs() < 1e-5, "[{r},{j}]");
        }
    }
    rotate_rows(&mut y, k, CONVROT_GROUP);
    for (a, b) in y.iter().zip(&x) {
        assert!((a - b).abs() < 1e-5);
    }
}

/// The CPU dequant reference is the int8 grid times `s_channel`, un-rotated — and on a Gaussian
/// DiT-like weight it reconstructs the original at the 4-bit error level.
#[test]
fn decode_dense_is_the_definition_and_reconstructs_the_weight() {
    let (n, k) = (32, 1024);
    let w = gaussian(n * k, 11);
    let p = quantize_weight(&w, n, k).unwrap();
    let mut want: Vec<f32> = p
        .dequant_int8()
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f32 * p.s_channel[i / k])
        .collect();
    rotate_rows(&mut want, k, CONVROT_GROUP);
    assert_eq!(decode_dense(&p), want);
    let cos = cosine(&decode_dense(&p), &w);
    assert!(cos > 0.995, "4-bit reconstruction cosine {cos}");
}

#[test]
fn zero_rows_pack_to_a_valid_zero_projection() {
    let (n, k) = (2, 256);
    let p = quantize_weight(&vec![0.0; n * k], n, k).unwrap();
    assert!(decode_dense(&p).iter().all(|&v| v == 0.0));
    p.descriptor().validate_s_rel(&p.s_rel).unwrap();
    p.descriptor().validate_s_channel(&p.s_channel).unwrap();
}

#[test]
fn checkpoint_tensor_set_round_trips_byte_for_byte() {
    let (n, k) = (48, 768);
    let p = quantize_weight(&gaussian(n * k, 3), n, k).unwrap();
    let tensors = p.checkpoint_tensors("blocks.0.attn.to_q");
    let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "blocks.0.attn.to_q.comfy_quant",
            "blocks.0.attn.to_q.weight",
            "blocks.0.attn.to_q.weight_codebook",
            "blocks.0.attn.to_q.weight_s_channel",
            "blocks.0.attn.to_q.weight_s_rel",
        ]
    );
    let bytes = serialize_checkpoint(&tensors).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    for t in &tensors {
        let view = st.tensor(&t.name).unwrap();
        assert_eq!(view.dtype(), t.dtype, "{}", t.name);
        assert_eq!(view.shape(), t.shape.as_slice(), "{}", t.name);
        assert_eq!(view.data(), t.data.as_slice(), "{}", t.name);
    }
    let back = W4A8Packed::from_safetensors(&st, "blocks.0.attn.to_q")
        .unwrap()
        .expect("a W4A8 projection");
    assert_eq!(back, p);
    let again = back.checkpoint_tensors("blocks.0.attn.to_q");
    let bytes2 = serialize_checkpoint(&again).unwrap();
    assert_eq!(bytes2, bytes, "re-serialized file is byte-identical");
    assert!(W4A8Packed::from_safetensors(&st, "blocks.0.attn.to_k")
        .unwrap()
        .is_none());
}

/// A `.comfy_quant` descriptor naming W4A8 parses, prices its geometry, and decodes (through the
/// descriptor's own host decoder) to the same dense weight as the CPU reference.
#[test]
fn descriptor_parses_prices_and_decodes_to_the_cpu_reference() {
    let (n, k) = (16, 512);
    let p = quantize_weight(&gaussian(n * k, 5), n, k).unwrap();
    let d =
        W4A8Int8::parse_typed(DESCRIPTOR_PAYLOAD, &[n, k / 2], &[n, k / 16], &[n], &[16]).unwrap();
    assert_eq!(d, p.descriptor());
    assert_eq!(
        d.resident_bytes(),
        p.packed.len() + p.s_rel.len() + 4 * p.s_channel.len() + 64
    );
    let signed: Vec<i8> = p.packed.iter().map(|&b| b as i8).collect();
    let mut grid = Vec::new();
    dequant_packed_to_int8(&signed, &p.s_rel, &p.codebook, n, k, &mut grid).unwrap();
    let mut dense: Vec<f32> = grid
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f32 * p.s_channel[i / k])
        .collect();
    rotate_rows(&mut dense, d.in_features, d.rotation_group);
    assert_eq!(dense, decode_dense(&p));
}

#[test]
fn malformed_group_size_refuses_with_a_typed_error() {
    let bad = br#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":8}"#;
    assert_eq!(
        W4A8Int8::parse_typed(bad, &[16, 128], &[16, 32], &[16], &[16]),
        Err(W4A8DescriptorError::GroupSize { declared: 8 })
    );
    let bad = br#"{"convrot_groupsize":64,"format":"asym_w4a8_int8","group_size":16}"#;
    assert_eq!(
        W4A8Int8::parse_typed(bad, &[16, 128], &[16, 16], &[16], &[16]),
        Err(W4A8DescriptorError::RotationGroup { declared: 64 })
    );
    // The String entry point keeps the exact historical message.
    let bad = br#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":8}"#;
    assert_eq!(
        W4A8Int8::parse(bad, &[16, 128], &[16, 32], &[16], &[16]).unwrap_err(),
        "W4A8 group_size must be 16, got 8"
    );
}

#[test]
fn packer_refuses_what_the_format_cannot_hold() {
    assert_eq!(
        quantize_weight(&[0.0; 2 * 128], 2, 128),
        Err(W4A8PackError::InFeaturesNotRotationMultiple { in_features: 128 })
    );
    assert!(matches!(
        quantize_weight(&[0.0; 10], 2, 256),
        Err(W4A8PackError::Shape { .. })
    ));
    let mut w = vec![0.0; 256];
    w[3] = f32::NAN;
    assert_eq!(quantize_weight(&w, 1, 256), Err(W4A8PackError::NonFinite));
    let mut book = DEFAULT_CODEBOOK;
    book.swap(0, 1);
    assert_eq!(
        quantize_weight_with_codebook(&[0.0; 256], 1, 256, &book),
        Err(W4A8PackError::Codebook)
    );
}

#[test]
fn nearest_level_breaks_ties_to_the_lower_index() {
    let levels = [-1.0f32, 0.0, 1.0];
    assert_eq!(
        nearest_level(&levels, 0.5),
        1,
        "tie between 0 and 1 -> lower"
    );
    assert_eq!(
        nearest_level(&levels, -0.5),
        0,
        "tie between -1 and 0 -> lower"
    );
    assert_eq!(nearest_level(&levels, 0.51), 2);
    assert_eq!(nearest_level(&levels, 9.0), 2);
    assert_eq!(nearest_level(&levels, -9.0), 0);
}
