//! Qwen-Image 3-axis (frame/height/width) RoPE. Port of the fork's `QwenEmbedRopeMLX`
//! (`theta=10000`, `axes_dim=[16,56,56]`, `scale_rope=True`). Produces **interleaved** complex
//! cos/sin tables (`head_dim/2 = 64` pairs) for the image and text streams.
//!
//! Frequencies per axis: `omega_d[k] = theta^(-2k/d)`, `k in 0..d/2`. The 64 pair-frequencies are
//! `[omega16 (8), omega56 (28), omega56 (28)]`. For the image stream the frame axis sits at
//! position 0 (no rotation), while height/width use **centered** positions
//! `[-(N - N/2), …, -1, 0, …, N/2 - 1]` (`scale_rope`). The text stream applies a single scalar
//! position `max(H/2, W/2) + t` across all 64 frequencies.

use mlx_rs::Array;

use mlx_gen::Result;

pub struct QwenRope3d {
    theta: f32,
    axes_dim: [i32; 3],
}

impl QwenRope3d {
    pub fn new(theta: f32, axes_dim: [i32; 3]) -> Self {
        Self { theta, axes_dim }
    }

    /// The Qwen-Image default: θ=10000, axes `[16, 56, 56]` (Σ/2 = 64 = head_dim/2).
    pub fn qwen_image() -> Self {
        Self::new(10000.0, [16, 56, 56])
    }

    fn omega(&self, dim: i32) -> Vec<f32> {
        (0..dim / 2)
            .map(|k| 1.0 / self.theta.powf((2 * k) as f32 / dim as f32))
            .collect()
    }

    /// Single-image RoPE (T2I). `(img_cos, img_sin, txt_cos, txt_sin)`: image tables
    /// `(latent_h·latent_w, 64)`, text `(txt_seq, 64)`. Equivalent to [`forward_multi`] with one shape.
    pub fn forward(
        &self,
        latent_h: usize,
        latent_w: usize,
        txt_seq: usize,
    ) -> Result<(Array, Array, Array, Array)> {
        self.forward_multi(&[(latent_h, latent_w)], txt_seq)
    }

    /// Multi-image RoPE (Qwen-Image-Edit dual-latent): one `(latent_h, latent_w)` per concatenated
    /// image in sequence order — the noise latents first (image index 0), then each reference
    /// (index 1, 2, …). Port of the fork's `QwenEmbedRopeMLX(video_fhw=img_shapes)`: image `idx`
    /// drives the **frame axis** position (so the reference's frame freqs differ from the noise's),
    /// while height/width stay per-image **centered** (`scale_rope`). The text base is
    /// `max_i(max(h_i/2, w_i/2))`. Image tables are `(Σ h_i·w_i, 64)`.
    pub fn forward_multi(
        &self,
        shapes: &[(usize, usize)],
        txt_seq: usize,
    ) -> Result<(Array, Array, Array, Array)> {
        let (o0, o1, o2) = (
            self.omega(self.axes_dim[0]),
            self.omega(self.axes_dim[1]),
            self.omega(self.axes_dim[2]),
        );
        let half: usize = o0.len() + o1.len() + o2.len(); // 8 + 28 + 28 = 64

        let total_seq: usize = shapes.iter().map(|(h, w)| h * w).sum();
        let mut img_cos = vec![0f32; total_seq * half];
        let mut img_sin = vec![0f32; total_seq * half];
        let mut off = 0usize; // running row offset into the concatenated sequence
        let mut txt_base = 0i32;
        for (idx, &(latent_h, latent_w)) in shapes.iter().enumerate() {
            let h_off = (latent_h - latent_h / 2) as i32;
            let w_off = (latent_w - latent_w / 2) as i32;
            for h in 0..latent_h {
                let hp = h as i32 - h_off;
                for w in 0..latent_w {
                    let wp = w as i32 - w_off;
                    let row = (off + h * latent_w + w) * half;
                    let mut j = 0;
                    // frame axis: position = image index (0 for noise, 1 for the first reference, …)
                    for &f in &o0 {
                        let a = idx as f32 * f;
                        img_cos[row + j] = a.cos();
                        img_sin[row + j] = a.sin();
                        j += 1;
                    }
                    for &f in &o1 {
                        let a = hp as f32 * f;
                        img_cos[row + j] = a.cos();
                        img_sin[row + j] = a.sin();
                        j += 1;
                    }
                    for &f in &o2 {
                        let a = wp as f32 * f;
                        img_cos[row + j] = a.cos();
                        img_sin[row + j] = a.sin();
                        j += 1;
                    }
                }
            }
            off += latent_h * latent_w;
            txt_base = txt_base.max((latent_h / 2).max(latent_w / 2) as i32);
        }

        // --- text stream: scalar position max_i(max(H_i/2, W_i/2)) + t across all 64 frequencies ---
        let all_omega: Vec<f32> = o0.iter().chain(&o1).chain(&o2).copied().collect();
        let mut txt_cos = vec![0f32; txt_seq * half];
        let mut txt_sin = vec![0f32; txt_seq * half];
        for t in 0..txt_seq {
            let p = (txt_base + t as i32) as f32;
            let row = t * half;
            for (j, &f) in all_omega.iter().enumerate() {
                let a = p * f;
                txt_cos[row + j] = a.cos();
                txt_sin[row + j] = a.sin();
            }
        }

        let h = half as i32;
        Ok((
            Array::from_slice(&img_cos, &[total_seq as i32, h]),
            Array::from_slice(&img_sin, &[total_seq as i32, h]),
            Array::from_slice(&txt_cos, &[txt_seq as i32, h]),
            Array::from_slice(&txt_sin, &[txt_seq as i32, h]),
        ))
    }
}
