//! SAM2.1 image-encoder configs — the four official Hiera sizes (tiny / small / base-plus / large).
//!
//! Mirrors `mlx_sam/config.py` (`avbiswas/sam2-mlx`) field-for-field; the production person-track
//! segmenter defaults to **large** to match the PyTorch quality baseline (spike sc-3635), with the
//! smaller sizes available as the speed/quality dial.

/// Hiera hierarchical-ViT trunk hyperparameters (`mlx_sam.config.HieraConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct HieraConfig {
    pub image_size: i32,
    pub embed_dim: i32,
    pub num_heads: i32,
    /// Blocks per stage (4 stages). `depth = sum(stages)`.
    pub stages: Vec<i32>,
    /// Block indices that run *global* (non-windowed) attention.
    pub global_att_blocks: Vec<i32>,
    /// Per-stage attention window size.
    pub window_spec: Vec<i32>,
    /// Number of leading stage transitions that apply a `q_stride` query pool.
    pub q_pool: i32,
    /// Query-pool stride (square; SAM2 uses 2).
    pub q_stride: i32,
    pub mlp_ratio: f32,
    pub dim_mul: f32,
    pub head_mul: f32,
    /// Side length of the (square) learned absolute position-embedding grid.
    pub pos_embed_hw: i32,
}

/// FPN-neck hyperparameters (`mlx_sam.config.FpnConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct FpnConfig {
    pub d_model: i32,
    /// Trunk output channels per level, coarse→fine (matches the conv order).
    pub backbone_channel_list: Vec<i32>,
    /// FPN levels that receive a top-down (nearest-2×) merge.
    pub fpn_top_down_levels: Vec<i32>,
    /// Number of coarsest FPN levels dropped from the returned features.
    pub scalp: i32,
}

/// Full SAM2.1 image-encoder config (trunk + neck).
#[derive(Clone, Debug, PartialEq)]
pub struct Sam2ImageEncoderConfig {
    pub hiera: HieraConfig,
    pub fpn: FpnConfig,
}

/// The four official SAM2.1 Hiera checkpoint sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sam2ModelSize {
    Tiny,
    Small,
    BasePlus,
    Large,
}

impl Sam2ImageEncoderConfig {
    /// `facebook/sam2.1-hiera-small` (the `mlx_sam` default config).
    pub fn small() -> Self {
        Self {
            hiera: HieraConfig {
                image_size: 1024,
                embed_dim: 96,
                num_heads: 1,
                stages: vec![1, 2, 11, 2],
                global_att_blocks: vec![7, 10, 13],
                window_spec: vec![8, 4, 14, 7],
                q_pool: 3,
                q_stride: 2,
                mlp_ratio: 4.0,
                dim_mul: 2.0,
                head_mul: 2.0,
                pos_embed_hw: 256,
            },
            fpn: FpnConfig {
                d_model: 256,
                backbone_channel_list: vec![768, 384, 192, 96],
                fpn_top_down_levels: vec![2, 3],
                scalp: 1,
            },
        }
    }

    /// `facebook/sam2.1-hiera-tiny`.
    pub fn tiny() -> Self {
        let mut c = Self::small();
        c.hiera.stages = vec![1, 2, 7, 2];
        c.hiera.global_att_blocks = vec![5, 7, 9];
        c
    }

    /// `facebook/sam2.1-hiera-base-plus`.
    pub fn base_plus() -> Self {
        Self {
            hiera: HieraConfig {
                image_size: 1024,
                embed_dim: 112,
                num_heads: 2,
                stages: vec![2, 3, 16, 3],
                global_att_blocks: vec![12, 16, 20],
                window_spec: vec![8, 4, 14, 7],
                q_pool: 3,
                q_stride: 2,
                mlp_ratio: 4.0,
                dim_mul: 2.0,
                head_mul: 2.0,
                pos_embed_hw: 256,
            },
            fpn: FpnConfig {
                d_model: 256,
                backbone_channel_list: vec![896, 448, 224, 112],
                fpn_top_down_levels: vec![2, 3],
                scalp: 1,
            },
        }
    }

    /// `facebook/sam2.1-hiera-large` — the production person-track default.
    pub fn large() -> Self {
        Self {
            hiera: HieraConfig {
                image_size: 1024,
                embed_dim: 144,
                num_heads: 2,
                stages: vec![2, 6, 36, 4],
                global_att_blocks: vec![23, 33, 43],
                window_spec: vec![8, 4, 16, 8],
                q_pool: 3,
                q_stride: 2,
                mlp_ratio: 4.0,
                dim_mul: 2.0,
                head_mul: 2.0,
                pos_embed_hw: 256,
            },
            fpn: FpnConfig {
                d_model: 256,
                backbone_channel_list: vec![1152, 576, 288, 144],
                fpn_top_down_levels: vec![2, 3],
                scalp: 1,
            },
        }
    }

    /// Config for a named size.
    pub fn for_size(size: Sam2ModelSize) -> Self {
        match size {
            Sam2ModelSize::Tiny => Self::tiny(),
            Sam2ModelSize::Small => Self::small(),
            Sam2ModelSize::BasePlus => Self::base_plus(),
            Sam2ModelSize::Large => Self::large(),
        }
    }
}

impl Sam2ModelSize {
    /// Infer the model size from a checkpoint name / id (mirrors `mlx_sam.config.model_config_for_name`).
    pub fn from_name(name: &str) -> Option<Self> {
        let n = name.to_lowercase();
        if n.contains("hiera_tiny") || n.contains("hiera-tiny") {
            Some(Self::Tiny)
        } else if n.contains("hiera_small") || n.contains("hiera-small") {
            Some(Self::Small)
        } else if n.contains("hiera_base_plus")
            || n.contains("hiera-base-plus")
            || n.contains("hiera_b+")
        {
            Some(Self::BasePlus)
        } else if n.contains("hiera_large") || n.contains("hiera-large") {
            Some(Self::Large)
        } else {
            None
        }
    }
}
