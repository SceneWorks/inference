// SW-6142 (SW-F287 M4) - fused video-VAE conv input: `pad(silu(norm(x)))`, written straight into
// the channels-last (NDHWC) padded buffer the following Conv3d reads.
//
// Written from the in-tree reference op chains (`mlx_gen::vae_norm_silu_pad::reference_conv_input`,
// the Wan z16 `rms_norm_channels` and LTX `pixel_norm` paths); no third-party kernel source is
// ported. The SiLU below repeats the two statements of MLX's `Sigmoid` unary functor
// (mlx v0.32.0 `backend/metal/kernels/unary_ops.h`) and the following multiply, which is also
// exactly what the patched eager `silu_exact` kernel computes.
//
// Exactness: the channel reduction (sum of squares, clamp/eps, sqrt) is NOT done here - the caller
// computes the per-voxel denominator with the reference MLX ops, so it is bit-identical by
// construction. This kernel performs only the per-element tail, one IEEE operation per statement,
// in the reference order (divide, [x sqrt(C), x gamma], sigmoid, multiply), with FP contraction
// off so no multiply/add pair is fused into an FMA the op chain would round twice.
//
// Precise math, explicitly: MLX builds `mlx.metallib` with `-fno-fast-math`, so the reference
// chain's `Divide` (`x / y`) is the correctly rounded divide and `Sigmoid`'s `metal::exp` resolves
// to `metal::precise::exp`. This file is JIT-compiled through `metal_kernel`, whose
// `MTLCompileOptions` set only the math mode, leaving `mathFloatingPointFunctions` at its `Fast`
// default, so a bare `exp` (and `/`) here can land a few ULP away from the metallib result. Every
// divide and the exp below therefore name `metal::precise::` directly, which pins them to the
// metallib's functions whatever the JIT defaults are.
//
// MLX `metal_kernel` contract: the first part of this file (up to the marker line) is the kernel
// `header`; the rest is the kernel body. Inputs: `inp` (x, any float dtype, arbitrary strides),
// `denom` (f32 [B,1,T,H,W]), `gamma` (C elements, any float dtype; a dummy when !AFFINE),
// `cache` (f32 activated prefix frames [B,C,TC,H,W]; a dummy when !HAS_CACHE), `dims` (int32[16]).
// Outputs: `out` (f32 [B,TP,HP,WP,C], contiguous) and `tail` (f32 [B,C,NT,H,W]; a dummy when
// !HAS_TAIL). Template bools: AFFINE, HAS_CACHE, HAS_TAIL, REPLICATE.
// dims: 0 B, 1 C, 2 T, 3 H, 4 W, 5 TC, 6 FRONT, 7 BACK, 8 PH, 9 PW, 10 NT, 11 TP, 12 HP, 13 WP,
//       14 bits of the f32 sqrt(C) scale, 15 total output elements.

#pragma clang fp contract(off)

inline float sceneworks_vae_silu(float n) {
  const float y = metal::precise::divide(1.0f, 1.0f + metal::precise::exp(metal::abs(n)));
  const float s = (n < 0) ? y : 1 - y;
  return n * s;
}

// ---- sceneworks kernel body ----
  const uint gid = thread_position_in_grid.x;
  const uint total = uint(dims[15]);
  if (gid >= total) {
    return;
  }
  const int C = dims[1];
  const int T = dims[2];
  const int H = dims[3];
  const int W = dims[4];
  const int TC = dims[5];
  const int FRONT = dims[6];
  const int PH = dims[8];
  const int PW = dims[9];
  const int NT = dims[10];
  const int TP = dims[11];
  const int HP = dims[12];
  const int WP = dims[13];
  const float scale = as_type<float>(dims[14]);

  uint r = gid;
  const int c = int(r % uint(C));
  r /= uint(C);
  const int wp = int(r % uint(WP));
  r /= uint(WP);
  const int hp = int(r % uint(HP));
  r /= uint(HP);
  const int tp = int(r % uint(TP));
  const int b = int(r / uint(TP));
  const int h = hp - PH;
  const int w = wp - PW;

  float value = 0.0f;
  if (h >= 0 && h < H && w >= 0 && w < W) {
    int t = -1;
    if (REPLICATE) {
      t = metal::clamp(tp - FRONT, 0, T - 1);
    } else if (tp >= FRONT) {
      const int k = tp - FRONT;
      if (HAS_CACHE && k < TC) {
        const int64_t at = int64_t(b) * cache_strides[0] + int64_t(c) * cache_strides[1] +
            int64_t(k) * cache_strides[2] + int64_t(h) * cache_strides[3] +
            int64_t(w) * cache_strides[4];
        value = static_cast<float>(cache[at]);
      } else {
        const int frame = HAS_CACHE ? k - TC : k;
        if (frame < T) {
          t = frame;
        }
      }
    }
    if (t >= 0) {
      const int64_t xat = int64_t(b) * inp_strides[0] + int64_t(c) * inp_strides[1] +
          int64_t(t) * inp_strides[2] + int64_t(h) * inp_strides[3] +
          int64_t(w) * inp_strides[4];
      const int64_t dat = int64_t(b) * denom_strides[0] + int64_t(t) * denom_strides[2] +
          int64_t(h) * denom_strides[3] + int64_t(w) * denom_strides[4];
      float n = metal::precise::divide(static_cast<float>(inp[xat]), static_cast<float>(denom[dat]));
      if (AFFINE) {
        n = n * scale;
        n = n * static_cast<float>(gamma[int64_t(c) * gamma_strides[0]]);
      }
      value = sceneworks_vae_silu(n);
      if (HAS_TAIL && t >= T - NT) {
        const int64_t tat =
            (((int64_t(b) * C + c) * NT + (t - (T - NT))) * H + h) * int64_t(W) + w;
        tail[tat] = value;
      }
    }
  }
  out[gid] = value;
