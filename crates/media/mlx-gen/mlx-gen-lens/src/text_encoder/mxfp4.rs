//! MXFP4 dequantization for the gpt-oss MoE experts (the only MXFP4 tensors in the checkpoint).
//!
//! Faithful port of `transformers.integrations.mxfp4.convert_moe_packed_tensors`: each `uint8` byte
//! packs two FP4 (e2m1) nibbles (low then high) looked up in `FP4_VALUES`, and every block of 32
//! values (16 bytes) shares one `e8m0` scale (`exponent = scale − 127`, applied via `·2^exponent`).
//! The dequantized rows are then transposed (`[E, out, G·32]` → `[E, G·32, out]`) to land in the
//! `[E, in, out]` layout the eager `GptOssExperts` uses (`x · gate_up_proj[e]`).
//!
//! Done host-side (bit-exact, simple): mlx-rs exposes no bitwise/shift ops to vectorize the nibble
//! split, and the per-layer cost is small. The memory-efficient Q4/Q8 **re-quant** (keep ~12 GB
//! across all 24 layers) is sc-3172; this is the plain dequant the MoE forward consumes.

use mlx_rs::{Array, Dtype};

use mlx_gen::{Error, Result};

/// FP4 (e2m1) value lookup, indexed by the 4-bit nibble (sign bit = MSB).
const FP4_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Dequantize MXFP4 `blocks` `[E, R, G, 16]` (`uint8`) + `scales` `[E, R, G]` (`uint8`, e8m0) to a
/// dense `[E, G·32, R]` array at `dtype` — the `GptOssExperts` parameter layout
/// (`gate_up_proj`: `[E, hidden, 2·inter]`; `down_proj`: `[E, inter, hidden]`).
pub fn dequantize_mxfp4(blocks: &Array, scales: &Array, dtype: Dtype) -> Result<Array> {
    let bsh = blocks.shape();
    if bsh.len() != 4 || bsh[3] != 16 {
        return Err(Error::Msg(format!(
            "mxfp4 blocks must be [E, R, G, 16], got {bsh:?}"
        )));
    }
    let (e, r, g) = (bsh[0] as usize, bsh[1] as usize, bsh[2] as usize);
    let in_c = g * 32; // dequantized contraction length

    let blk = blocks.as_dtype(Dtype::Uint8)?;
    let blk = blk.as_slice::<u8>(); // [E*R*G*16]
    let scl = scales.as_dtype(Dtype::Uint8)?;
    let scl = scl.as_slice::<u8>(); // [E*R*G]

    // Build [E, R, in_c] (= [E, R, G·32]) then transpose the last two axes below.
    let mut out = vec![0f32; e * r * in_c];
    for (block_i, (chunk, &scale)) in blk.chunks_exact(16).zip(scl.iter()).enumerate() {
        let row = block_i / g; // flattened (e, r) index
        let gi = block_i % g;
        let mul = 2f32.powi(scale as i32 - 127);
        let obase = row * in_c + gi * 32;
        for (bi, &byte) in chunk.iter().enumerate() {
            out[obase + bi * 2] = FP4_VALUES[(byte & 0x0F) as usize] * mul;
            out[obase + bi * 2 + 1] = FP4_VALUES[(byte >> 4) as usize] * mul;
        }
    }

    let dense = Array::from_slice(&out, &[e as i32, r as i32, in_c as i32]);
    drop(out);
    let result = dense.transpose_axes(&[0, 2, 1])?.as_dtype(dtype)?;
    // Materialize this projection before the caller loads the next one. Otherwise the lazy
    // bf16 cast retains its full f32 source across all 24 encoder layers (~71 GiB of staging
    // for the production configuration), even after the source weight map is drained.
    drop(dense);
    result.eval()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequantized_bf16_does_not_retain_full_f32_staging_buffer() {
        // A small stand-in for the 48 expert projections loaded by the dense encoder.
        let blocks = Array::from_slice(&vec![0x22u8; 8 * 256 * 8 * 16], &[8, 256, 8, 16]);
        let scales = Array::from_slice(&vec![127u8; 8 * 256 * 8], &[8, 256, 8]);
        blocks.eval().unwrap();
        scales.eval().unwrap();
        let before = mlx_rs::memory::get_active_memory();
        let outputs: Vec<_> = (0..8)
            .map(|_| dequantize_mxfp4(&blocks, &scales, Dtype::Bfloat16).unwrap())
            .collect();
        let output = &outputs[0];
        let retained = mlx_rs::memory::get_active_memory().saturating_sub(before);
        let output_bytes: usize = outputs.iter().map(Array::nbytes).sum();
        // A completed Metal submission can retain its last input until the command buffer is
        // retired. Permit one f32 projection, but never one per resident bf16 projection.
        assert!(
            retained <= output_bytes + 2 * output.nbytes() + 64 * 1024,
            "dequantization retained {retained} bytes for {output_bytes} bytes of bf16 outputs; f32 staging must not accumulate across projections"
        );
        let values = output.as_dtype(Dtype::Float32).unwrap();
        assert_eq!(values.shape(), &[8, 256, 256]);
        assert!(values.as_slice::<f32>().iter().all(|&v| v == 1.0));
    }
}
