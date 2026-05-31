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

    /// `(img_cos, img_sin, txt_cos, txt_sin)`:
    /// image tables are `(latent_h·latent_w, 64)`, text tables `(txt_seq, 64)`.
    pub fn forward(
        &self,
        latent_h: usize,
        latent_w: usize,
        txt_seq: usize,
    ) -> Result<(Array, Array, Array, Array)> {
        let (o0, o1, o2) = (
            self.omega(self.axes_dim[0]),
            self.omega(self.axes_dim[1]),
            self.omega(self.axes_dim[2]),
        );
        let half: usize = o0.len() + o1.len() + o2.len(); // 8 + 28 + 28 = 64

        // --- image stream: frame at pos 0, height/width centered (scale_rope) ---
        let img_seq = latent_h * latent_w;
        let mut img_cos = vec![0f32; img_seq * half];
        let mut img_sin = vec![0f32; img_seq * half];
        let h_off = (latent_h - latent_h / 2) as i32;
        let w_off = (latent_w - latent_w / 2) as i32;
        for h in 0..latent_h {
            let hp = h as i32 - h_off;
            for w in 0..latent_w {
                let wp = w as i32 - w_off;
                let row = (h * latent_w + w) * half;
                let mut j = 0;
                // frame axis: position 0 → cos 1, sin 0
                for _ in &o0 {
                    img_cos[row + j] = 1.0;
                    img_sin[row + j] = 0.0;
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

        // --- text stream: scalar position max(H/2, W/2) + t across all 64 frequencies ---
        let base = (latent_h / 2).max(latent_w / 2) as i32;
        let all_omega: Vec<f32> = o0.iter().chain(&o1).chain(&o2).copied().collect();
        let mut txt_cos = vec![0f32; txt_seq * half];
        let mut txt_sin = vec![0f32; txt_seq * half];
        for t in 0..txt_seq {
            let p = (base + t as i32) as f32;
            let row = t * half;
            for (j, &f) in all_omega.iter().enumerate() {
                let a = p * f;
                txt_cos[row + j] = a.cos();
                txt_sin[row + j] = a.sin();
            }
        }

        let h = half as i32;
        Ok((
            Array::from_slice(&img_cos, &[img_seq as i32, h]),
            Array::from_slice(&img_sin, &[img_seq as i32, h]),
            Array::from_slice(&txt_cos, &[txt_seq as i32, h]),
            Array::from_slice(&txt_sin, &[txt_seq as i32, h]),
        ))
    }
}
