// Fused on-device NVFP4 activation quantizer (sc-12078, epic 11037).
//
// Replaces the ~40-op unfused candle chain in `CublasLt::quantize_nvfp4_activation` with two device
// passes over the activation: (1) per-16-block amax + per-tensor amax; (2) E4M3 block scale + E2M1
// element codes + nibble pack + swizzle into cuBLASLt's row-major 128x4 UE4M3 SF-atom layout. It
// produces byte-identical `DevNvfp4` bytes to the CPU packer (`Nvfp4Tensor::pack_from_slice`) — the
// E2M1/E4M3 rounding below mirrors `e2m1_from_f32` / `e4m3_from_f32` (OCP, round-to-nearest-even).
//
// Compiled at runtime via nvrtc (no fp8/fp4 headers, no build.rs); one thread per (row, 16-block).

// ln2 / (1/ln2) as f32 — matching candle's `e4m3_round_tensor` / `e4m3_byte_from_decoded`, which
// compute log2 as `ln(x) * (1/ln2)` and 2^y as `exp(y * ln2)`. Using logf/expf with these exact
// constants (not log2f/exp2f) makes the floor(log2) land on the SAME integer candle's tensor ops do,
// so the E4M3 grid rounding is bit-identical to the proven on-device path (sc-12078).
#define NVFP4_LN2      0.6931471805599453f
#define NVFP4_INV_LN2  1.4426950408889634f

extern "C" {

// ---- OCP E2M1 (FP4): nearest of {0,.5,1,1.5,2,3,4,6}, ties-to-even, sign in bit 3, saturate +/-6 ----
// Mirrors `e2m1_from_f32` (nvfp4.rs): prefers the even-index magnitude on a tie. Returns a nibble 0..15.
__device__ __forceinline__ unsigned int e2m1_code(float v) {
    if (isnan(v)) return 0u;
    unsigned int sign = signbit(v) ? 0x8u : 0x0u; // sign in bit 3, matches is_sign_negative()
    float mag = fminf(fabsf(v), 6.0f);
    unsigned int idx;
    // Boundaries are the exact midpoints; `<=`/`<` chosen so a tie lands on the even index.
    if      (mag <= 0.25f) idx = 0; // |0.0
    else if (mag <  0.75f) idx = 1; //  0.5
    else if (mag <= 1.25f) idx = 2; //  1.0
    else if (mag <  1.75f) idx = 3; //  1.5
    else if (mag <= 2.5f)  idx = 4; //  2.0  (tie @2.5 -> even idx4)
    else if (mag <  3.5f)  idx = 5; //  3.0
    else if (mag <= 5.0f)  idx = 6; //  4.0  (tie @5.0 -> even idx6)
    else                   idx = 7; //  6.0
    return sign | idx;
}

// ---- OCP E4M3: round a non-negative value onto the E4M3 grid, returning the DECODED on-grid value.
// Mirrors the candle `e4m3_round_tensor` arithmetic (which matches the CPU `e4m3_from_f32` scan to
// rel-RMS 0.000000): single ULP formula covering the normal grid (ULP 2^(e-3)) and, via e clamped to
// -6, the subnormal grid (ULP 2^-9). Saturates at 448.
__device__ __forceinline__ float e4m3_round(float v) {
    v = fminf(fmaxf(v, 0.0f), 448.0f);
    float vc = fminf(fmaxf(v, 0.015625f /* 2^-6 */), 448.0f);
    float e = floorf(logf(vc) * NVFP4_INV_LN2); // floor(log2 vc) in [-6, 8], candle's ln-based log2
    float ulp = expf((e - 3.0f) * NVFP4_LN2);   // 2^(e-3)
    float q = roundf(v / ulp) * ulp;
    return fminf(fmaxf(q, 0.0f), 448.0f);
}

// ---- The OCP E4M3 byte (0..=254) whose decode equals the on-grid value `q` from e4m3_round.
// Mirrors `e4m3_byte_from_decoded` (cublaslt.rs): zero, subnormal (round(q*512)), or normal
// ((e+7)<<3 | mant-8, with mantissa overflow folding into the next exponent). Clamped to 254 (0x7F=NaN).
__device__ __forceinline__ unsigned int e4m3_byte_from_decoded(float q) {
    if (q < (1.0f / 512.0f) * 0.5f) return 0u;      // below half the min subnormal -> 0
    if (q < 0.015625f) {                             // subnormal
        int b = (int)roundf(q * 512.0f);
        return (unsigned int)(b < 0 ? 0 : (b > 254 ? 254 : b));
    }
    float e = floorf(logf(q) * NVFP4_INV_LN2);       // floor(log2 q), candle's ln-based log2
    float pow_e = expf(e * NVFP4_LN2);               // 2^e
    int mant = (int)roundf(q / pow_e * 8.0f);        // 8..16 (16 folds into next exponent)
    int byte = (int)e * 8 + 48 + mant;              // (e+7)*8 + (mant-8)
    if (byte < 0) byte = 0;
    if (byte > 254) byte = 254;
    return (unsigned int)byte;
}

// ---- Pass 1: per-16-block amax over the real (non-padding) columns + per-tensor amax. ------------
// One thread per (row, 16-block). `g_amax` must be pre-zeroed (0.0f); the atomicMax orders
// non-negative floats by their bit pattern (sign bit is 0, so int order == float order).
__global__ void nvfp4_block_amax_f32(
    const float* __restrict__ x,        // [m, k] row-major
    float* __restrict__ block_amax,     // [m * n_blocks]
    int* __restrict__ g_amax,           // [1], pre-zeroed; reinterpreted float bits
    int m, int k, int n_blocks)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)m * n_blocks;
    if (idx >= total) return;
    int r = (int)(idx / n_blocks);
    int blk = (int)(idx % n_blocks);
    int c0 = blk * 16;
    const float* row = x + (long)r * k;
    float a = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        int c = c0 + j;
        if (c < k) a = fmaxf(a, fabsf(row[c]));
    }
    block_amax[idx] = a;
    atomicMax(g_amax, __float_as_int(a));
}

// ---- Pass 2: E4M3 block scale + E2M1 codes + nibble pack + swizzled scale scatter. --------------
// One thread per (row, 16-block): writes its 8 packed nibble bytes + 1 swizzled scale byte. `scales`
// must be pre-zeroed so padding atoms (rows>=m or blocks>=n_blocks, up to the 128x4 atom) stay 0.
__global__ void nvfp4_pack_f32(
    const float* __restrict__ x,          // [m, k] row-major
    const float* __restrict__ block_amax, // [m * n_blocks]
    unsigned char* __restrict__ packed,   // [m * (cols_padded/2)]
    unsigned char* __restrict__ scales,   // [sf_rows * sf_cols], pre-zeroed
    int m, int k, int n_blocks,
    int cols_padded, int sf_cols,
    float global_scale)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)m * n_blocks;
    if (idx >= total) return;
    int r = (int)(idx / n_blocks);
    int blk = (int)(idx % n_blocks);

    float a_blk = block_amax[idx];
    // sf_real = a_blk / (6 * global_scale), but computed as a reciprocal-MULTIPLY to bit-match candle's
    // `block_amax.affine(1.0/(E2M1_MAX*global_scale))` (the reciprocal is formed in f64, then f32) — a
    // divide here differs in the last bit and flips E4M3 grid cells for blocks near a boundary (sc-12078).
    float inv6gs = (float)(1.0 / (double)(6.0f * global_scale));
    float sf_real = (global_scale > 0.0f) ? (a_blk * inv6gs) : 0.0f;
    float q = e4m3_round(sf_real);                  // decoded on-grid E4M3 block scale
    unsigned int sf_byte = e4m3_byte_from_decoded(q);

    // Swizzled scale byte offset — row-major atom tiling, matching cublaslt_scale_layout /
    // nvfp4_act_scale_gather_index: intra = (mr%32)*16 + (mr/32)*4 + kc; atom = k_atom + num_k_atoms*m_atom.
    int m_atom = r / 128;
    int mr = r % 128;
    int num_k_atoms = sf_cols / 4;
    int k_atom = blk / 4;
    int kc = blk % 4;
    long atom_index = (long)k_atom + (long)num_k_atoms * m_atom;
    int intra = (mr % 32) * 16 + (mr / 32) * 4 + kc;
    scales[atom_index * 512 + intra] = (unsigned char)sf_byte;

    float elem_scale = q * global_scale;
    int c0 = blk * 16;
    int row_bytes = cols_padded / 2;
    const float* row = x + (long)r * k;
    unsigned char* prow = packed + (long)r * row_bytes;
    #pragma unroll
    for (int j = 0; j < 16; j += 2) {
        int cl = c0 + j;      // even column -> low nibble
        int ch = c0 + j + 1;  // odd column  -> high nibble
        float vl = (cl < k) ? row[cl] : 0.0f;
        float vh = (ch < k) ? row[ch] : 0.0f;
        unsigned int code_l = (elem_scale > 0.0f) ? e2m1_code(vl / elem_scale) : 0u;
        unsigned int code_h = (elem_scale > 0.0f) ? e2m1_code(vh / elem_scale) : 0u;
        prow[cl / 2] = (unsigned char)((code_h << 4) | code_l);
    }
}

} // extern "C"
