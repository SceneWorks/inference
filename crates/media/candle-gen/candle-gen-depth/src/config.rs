//! Depth Anything V2 configuration — mirrors the HF `transformers` `DepthAnythingConfig` (+ its
//! DINOv2 `backbone_config`) for the `depth-anything/Depth-Anything-V2-Small-hf` checkpoint
//! (epic 8236, sc-8413). The candle twin of `mlx-gen-depth`'s `config.rs`.
//!
//! Only the **Small** (ViT-S/14) variant is wired as the default — the preprocessing tier favors
//! speed/size (the standard DA-V2 ControlNet-preprocessor choice). The Base (ViT-B) and Large
//! (ViT-L) checkpoints share the identical module graph and differ only in these scalars, so they
//! plug in by swapping the config (see [`DepthAnythingConfig::small`]).

/// DINOv2 ViT backbone + DPT neck/head hyperparameters. Defaults are the shipped
/// `depth-anything/Depth-Anything-V2-Small-hf` values.
///
/// The channel/intermediate scalars (`num_channels`, `mlp_ratio`, `neck_hidden_sizes`,
/// `fusion_hidden_size`, `head_hidden_size`) don't drive any runtime tensor shape on their own —
/// every affected shape rides the loaded checkpoint — but `DepthAnythingV2::from_weights` asserts
/// them against the checkpoint at load time (F-160, sc-11244), so a Base/Large plug-in that edits
/// them without swapping weights fails loudly instead of silently changing nothing.
#[derive(Clone, Debug)]
pub struct DepthAnythingConfig {
    // --- backbone (DINOv2 ViT) ---
    /// Backbone embedding dim (384 for ViT-S).
    pub hidden_size: usize,
    /// Number of transformer layers (12).
    pub num_hidden_layers: usize,
    /// Attention heads (6); `head_dim = hidden_size / num_attention_heads` (64).
    pub num_attention_heads: usize,
    /// FFN expansion ratio (4 ⇒ intermediate = 1536).
    pub mlp_ratio: usize,
    /// Input channels (3).
    pub num_channels: usize,
    /// Default inference image size (518) → `image_size / patch_size` token grid (37).
    pub image_size: usize,
    /// Patch / conv-stem stride (14).
    pub patch_size: usize,
    /// LayerNorm epsilon (1e-6, the DINOv2 default).
    pub layer_norm_eps: f64,
    /// 1-based backbone layer indices whose **output** hidden states feed the neck
    /// (`out_indices` = [3, 6, 9, 12]). The reassemble stage consumes these four.
    pub out_indices: [usize; 4],

    // --- neck (DPT reassemble + fusion) ---
    /// Per-stage reassemble output channels (`neck_hidden_sizes` = [48, 96, 192, 384]).
    pub neck_hidden_sizes: [usize; 4],
    /// Per-stage spatial resize factors over the backbone token grid
    /// (`reassemble_factors` = [4.0, 2.0, 1.0, 0.5]): >1 → transposed-conv upsample,
    /// ==1 → identity, <1 → strided-conv downsample.
    pub reassemble_factors: [f32; 4],
    /// Channel dim every neck `conv` projects into and the fusion stage runs at
    /// (`fusion_hidden_size` = 64).
    pub fusion_hidden_size: usize,

    // --- head ---
    /// Penultimate head conv channel dim (`head_hidden_size` = 32).
    pub head_hidden_size: usize,
}

impl Default for DepthAnythingConfig {
    fn default() -> Self {
        Self::small()
    }
}

impl DepthAnythingConfig {
    /// The shipped `depth-anything/Depth-Anything-V2-Small-hf` (ViT-S/14) configuration.
    pub fn small() -> Self {
        Self {
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 6,
            mlp_ratio: 4,
            num_channels: 3,
            image_size: 518,
            patch_size: 14,
            layer_norm_eps: 1e-6,
            out_indices: [3, 6, 9, 12],
            neck_hidden_sizes: [48, 96, 192, 384],
            reassemble_factors: [4.0, 2.0, 1.0, 0.5],
            fusion_hidden_size: 64,
            head_hidden_size: 32,
        }
    }

    /// `depth-anything/Depth-Anything-V2-Base-hf` (ViT-B/14): same graph, wider (the published
    /// `config.json`; identical to the MLX twin's).
    pub fn base() -> Self {
        Self {
            hidden_size: 768,
            num_attention_heads: 12,
            neck_hidden_sizes: [96, 192, 384, 768],
            fusion_hidden_size: 128,
            ..Self::small()
        }
    }

    /// `depth-anything/Depth-Anything-V2-Large-hf` (ViT-L/14): 24 layers, captures layers
    /// `[5, 12, 18, 24]`.
    pub fn large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            out_indices: [5, 12, 18, 24],
            neck_hidden_sizes: [256, 512, 1024, 1024],
            fusion_hidden_size: 256,
            ..Self::small()
        }
    }

    /// The config for a [`candle_gen::gen_core::train::DepthModelSize`].
    pub fn for_size(size: candle_gen::gen_core::train::DepthModelSize) -> Self {
        use candle_gen::gen_core::train::DepthModelSize;
        match size {
            DepthModelSize::Small => Self::small(),
            DepthModelSize::Base => Self::base(),
            DepthModelSize::Large => Self::large(),
        }
    }

    /// Exact parameter count of the module graph `DepthAnythingV2::from_weights` loads (backbone +
    /// neck + head) — for the trainer memory estimate (epic 2123 E7).
    pub fn param_count(&self) -> u64 {
        let h = self.hidden_size as u64;
        let inter = self.intermediate_size() as u64;
        let p = self.patch_size as u64;
        let tokens = (self.grid() as u64).pow(2) + 1;
        let mut n = h * 3 * p * p + h + h + tokens * h; // patch embed, cls, pos
        let layer = 4 * h + 4 * (h * h + h) + 2 * h + (inter * h + inter) + (h * inter + h);
        n += layer * self.num_hidden_layers as u64 + 2 * h;
        let fh = self.fusion_hidden_size as u64;
        for i in 0..4 {
            let nh = self.neck_hidden_sizes[i] as u64;
            n += nh * h + nh;
            let f = self.reassemble_factors[i];
            if f > 1.0 {
                let k = f as u64;
                n += nh * nh * k * k + nh;
            } else if f < 1.0 {
                n += nh * nh * 9 + nh;
            }
            n += fh * nh * 9;
            n += 4 * (fh * fh * 9 + fh) + fh * fh + fh;
        }
        let half = fh / 2;
        let hh = self.head_hidden_size as u64;
        n += half * fh * 9 + half + hh * half * 9 + hh + hh + 1;
        n
    }

    /// Conservative training working set of one differentiable forward + backward at the native
    /// square [`image_size`](Self::image_size), in bytes (f32, x2 for cotangents) — the same
    /// accounting as the MLX twin; not a measured value.
    pub fn training_working_set_bytes(&self) -> u64 {
        let tokens = (self.grid() as u64).pow(2) + 1;
        let h = self.hidden_size as u64;
        let heads = self.num_attention_heads as u64;
        let per_layer = 2 * heads * tokens * tokens + (12 + 2 * self.mlp_ratio as u64) * tokens * h;
        let backbone = per_layer * self.num_hidden_layers as u64;
        let g = self.grid() as u64;
        let fh = self.fusion_hidden_size as u64;
        let fusion: u64 = (0..4).map(|k| 12 * fh * (g * (2 << k)).pow(2)).sum();
        let full = (self.image_size as u64).pow(2);
        let head = 6 * (fh / 2 + self.head_hidden_size as u64) * full;
        (backbone + fusion + head) * 4 * 2
    }

    /// The aspect-preserving model input size for an `h x w` image (upstream
    /// `DifferentiableDepthEncoder._aspect_preserving_hw`): long side -> [`image_size`](Self::image_size),
    /// short side scaled and rounded to a multiple of the patch size (at least one patch).
    pub fn input_hw(&self, h: usize, w: usize) -> (usize, usize) {
        let (s, p) = (self.image_size, self.patch_size);
        let short = |short: usize, long: usize| -> usize {
            let scaled = (short as f64 * s as f64 / long as f64 / p as f64).round() as usize * p;
            scaled.max(p)
        };
        if h >= w {
            (s, short(w, h))
        } else {
            (short(h, w), s)
        }
    }

    /// `head_dim = hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// FFN intermediate dim (`hidden_size * mlp_ratio`).
    pub fn intermediate_size(&self) -> usize {
        self.hidden_size * self.mlp_ratio
    }

    /// Token grid side for the configured image size (`image_size / patch_size` = 37 default).
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Zero-based backbone layer indices whose output the neck consumes (`out_indices` are 1-based;
    /// the captured hidden is the *output* of that layer).
    pub fn capture_layers(&self) -> [usize; 4] {
        [
            self.out_indices[0] - 1,
            self.out_indices[1] - 1,
            self.out_indices[2] - 1,
            self.out_indices[3] - 1,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Aspect-preserving model input: long side 518, short side a multiple of 14 (same sizes as
    /// the MLX twin). Mutation: return `(s, s)` => red.
    #[test]
    fn input_hw_preserves_aspect_on_the_patch_grid() {
        let c = DepthAnythingConfig::small();
        assert_eq!(c.input_hw(1024, 1024), (518, 518));
        assert_eq!(c.input_hw(768, 1024), (392, 518));
        assert_eq!(c.input_hw(1024, 576), (518, 294));
        assert_eq!(c.input_hw(10, 1000), (14, 518));
        let (h, w) = c.input_hw(832, 1216);
        assert_eq!((h % 14, w), (0, 518));
    }

    #[test]
    fn sizes_grow_and_large_captures_its_published_layers() {
        let (s, b, l) = (
            DepthAnythingConfig::small(),
            DepthAnythingConfig::base(),
            DepthAnythingConfig::large(),
        );
        assert!(s.param_count() < b.param_count() && b.param_count() < l.param_count());
        assert!(s.training_working_set_bytes() < l.training_working_set_bytes());
        assert_eq!(l.capture_layers(), [4, 11, 17, 23]);
        // ~24.8M (S) / ~335M (L) published parameter counts.
        assert!(
            (24_000_000..26_000_000).contains(&s.param_count()),
            "{}",
            s.param_count()
        );
        assert!(
            (330_000_000..340_000_000).contains(&l.param_count()),
            "{}",
            l.param_count()
        );
    }

    #[test]
    fn small_geometry_is_the_shipped_vits() {
        let c = DepthAnythingConfig::small();
        assert_eq!(c.head_dim(), 64, "384 / 6 = 64");
        assert_eq!(c.intermediate_size(), 1536, "384 * 4 = 1536");
        assert_eq!(c.grid(), 37, "518 / 14 = 37");
        assert_eq!(
            c.grid() * c.grid() + 1,
            1370,
            "37² + 1 = 1370 (the pos-embed length)"
        );
        assert_eq!(
            c.capture_layers(),
            [2, 5, 8, 11],
            "1-based out_indices [3,6,9,12] → zero-based [2,5,8,11]"
        );
    }
}
