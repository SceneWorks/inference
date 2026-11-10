//! MiniMax-H3-**Image** weight tiers: names, published artifact layout, and which text encoder each
//! tier reads (SW-7093).
//!
//! Both backends read the same MLX-affine packed artifacts, so the layout is declared once here
//! rather than twice. This module is pure naming: it touches no file, and it neither packs nor
//! loads a tensor.
//!
//! # Layout
//!
//! The DiT is the only component the image repository tiers. Its VAE and tokenizer are
//! tier-agnostic and resolve against the upstream `MiniMaxAI/MiniMax-H3-Image` snapshot root, so a
//! packed install holds one packed DiT and no second copy of anything else.
//!
//! | artifact | where |
//! |---|---|
//! | packed DiT | [`IMAGE_TIER_REPO`] `{q8,q4}/transformer_diffusers/` |
//! | per-step heads | [`IMAGE_TIER_REPO`] `{q8,q4}/pdd_fused_heads.safetensors`, dense (F32), copied byte for byte |
//! | text encoder | **not a new copy**: MiniMax-H3's own tier, see [`ImageTier::text_encoder`] |
//! | `bf16` DiT, VAE, tokenizer | upstream `MiniMaxAI/MiniMax-H3-Image` |
//!
//! The caller stages the tier's directories as the `transformer_diffusers` and `text_encoder`
//! components of the `LoadSpec`; neither backend builds these paths itself, and `spec.quantize`
//! only asserts which tier was staged (the on-disk `quantization` marker is authoritative).

/// The new repository the packed image DiT tiers are published to.
pub const IMAGE_TIER_REPO: &str = "SceneWorks/minimax-h3-image-mlx";
/// MiniMax-H3's existing MLX tier repository; the text encoder of every packed image tier.
pub const H3_TIER_REPO: &str = "SceneWorks/minimax-h3-mlx";
/// Upstream MiniMax-H3, the dense text encoder of the `bf16` tier.
pub const H3_UPSTREAM_REPO: &str = "MiniMaxAI/MiniMax-H3";
/// The component directory of the image DiT, in every tier.
pub const DIT_DIR: &str = "transformer_diffusers";
/// The text encoder component directory, in every tier.
pub const TEXT_ENCODER_DIR: &str = "text_encoder";
/// The per-step output heads file published beside the DiT. Dense in every tier.
pub const PDD_FUSED_HEADS_FILE: &str = "pdd_fused_heads.safetensors";

/// A published MiniMax-H3-Image weight tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageTier {
    /// The upstream dense checkpoint.
    Bf16,
    /// 8-bit MLX affine pack.
    Q8,
    /// 4-bit MLX affine pack.
    Q4,
}

/// A directory inside a hosted repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoDir {
    /// `owner/name` of the hosted repository.
    pub repo: &'static str,
    /// Directory inside it.
    pub dir: String,
}

impl ImageTier {
    /// Every tier, in quality order.
    pub const ALL: [ImageTier; 3] = [ImageTier::Bf16, ImageTier::Q8, ImageTier::Q4];

    /// The tier's published directory name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    /// Packed bit width, `None` for the dense tier.
    pub fn bits(self) -> Option<i32> {
        match self {
            Self::Bf16 => None,
            Self::Q8 => Some(8),
            Self::Q4 => Some(4),
        }
    }

    /// The tier a packed bit width names.
    pub fn from_bits(bits: i32) -> Option<Self> {
        match bits {
            8 => Some(Self::Q8),
            4 => Some(Self::Q4),
            _ => None,
        }
    }

    /// Where this tier's DiT directory is published. The dense tier is the upstream snapshot's own
    /// `transformer_diffusers/`; a packed tier is `{tier}/transformer_diffusers/` in
    /// [`IMAGE_TIER_REPO`].
    pub fn dit(self) -> RepoDir {
        match self {
            Self::Bf16 => RepoDir {
                repo: "MiniMaxAI/MiniMax-H3-Image",
                dir: DIT_DIR.to_owned(),
            },
            packed => RepoDir {
                repo: IMAGE_TIER_REPO,
                dir: format!("{}/{DIT_DIR}", packed.as_str()),
            },
        }
    }

    /// The per-step heads file of a packed tier, `None` for the dense tier (whose heads are inside
    /// the DiT shards and the snapshot's own `pdd_fused_heads.safetensors`).
    pub fn pdd_fused_heads(self) -> Option<RepoDir> {
        self.bits().map(|_| RepoDir {
            repo: IMAGE_TIER_REPO,
            dir: format!("{}/{PDD_FUSED_HEADS_FILE}", self.as_str()),
        })
    }

    /// The text encoder this tier reads: **MiniMax-H3's existing artifact** at the same tier,
    /// never a copy in the image repository. Packed tiers use `{tier}/text_encoder/` in
    /// [`H3_TIER_REPO`]; the dense tier uses upstream MiniMax-H3's `text_encoder/`.
    pub fn text_encoder(self) -> RepoDir {
        match self {
            Self::Bf16 => RepoDir {
                repo: H3_UPSTREAM_REPO,
                dir: TEXT_ENCODER_DIR.to_owned(),
            },
            packed => RepoDir {
                repo: H3_TIER_REPO,
                dir: format!("{}/{TEXT_ENCODER_DIR}", packed.as_str()),
            },
        }
    }
}

/// Group size the tiers pack at, the codebase-wide MLX affine default.
pub const GROUP_SIZE: usize = 64;

/// Which bit width, if any, MiniMax-H3's policy packs a DiT linear at, from its `base` (the tensor
/// key minus `.weight`). `None` means dense **in every tier**.
///
/// This is the policy `mlx_gen_minimax_h3::convert` applies, stated once for converters that cannot
/// link MLX (the Candle lane's converter uses it; a test in the MLX crate asserts the two agree on
/// every published image-DiT tensor name).
///
/// * dense: the float32 I/O heads (`proj_in`, `proj_out`, `audio_proj_in`, `audio_proj_out`, both
///   `time_embedder` layers), `context_embedder`, `norm_out.linear`, **every `pdd_proj_out.{i}`
///   per-step head**, and every norm;
/// * packed at `adaln_bits`: the per-block `adaln_proj.linear`;
/// * packed at `tier_bits`: every other linear (block and token-refiner attention and feed-forward).
///
/// The caller still applies the shape guard (2-D, input width a multiple of [`GROUP_SIZE`]).
pub fn dit_pack_bits(base: &str, tier_bits: i32, adaln_bits: i32) -> Option<i32> {
    const DENSE: &[&str] = &[
        "proj_in",
        "proj_out",
        "audio_proj_in",
        "audio_proj_out",
        "time_embedder.linear_1",
        "time_embedder.linear_2",
        "context_embedder",
        "norm_out.linear",
    ];
    const DENSE_NORM_SUFFIXES: &[&str] = &[
        ".norm1",
        ".norm2",
        ".norm_q",
        ".norm_k",
        ".norm",
        ".final_norm",
    ];
    if DENSE.contains(&base)
        || base.starts_with("pdd_proj_out.")
        || DENSE_NORM_SUFFIXES.iter().any(|s| base.ends_with(s))
    {
        None
    } else if base.ends_with(".adaln_proj.linear") {
        Some(adaln_bits)
    } else {
        Some(tier_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_policy_keeps_heads_and_io_dense_and_packs_the_blocks() {
        for dense in [
            "proj_in",
            "context_embedder",
            "time_embedder.linear_1",
            "norm_out.linear",
            "pdd_proj_out.0",
            "pdd_proj_out.5",
            "transformer_blocks.0.norm1",
            "transformer_blocks.0.attn.norm_q",
            "token_refiner.final_norm",
        ] {
            assert_eq!(dit_pack_bits(dense, 4, 4), None, "{dense}");
        }
        assert_eq!(
            dit_pack_bits("transformer_blocks.3.attn.to_q", 4, 8),
            Some(4)
        );
        assert_eq!(
            dit_pack_bits("transformer_blocks.3.ff.net.2", 8, 8),
            Some(8)
        );
        assert_eq!(
            dit_pack_bits("transformer_blocks.3.adaln_proj.linear", 4, 8),
            Some(8)
        );
        assert_eq!(
            dit_pack_bits("token_refiner.refiner_blocks.1.ff.net.0.proj", 4, 4),
            Some(4)
        );
    }

    #[test]
    fn tiers_name_and_round_trip_their_bit_widths() {
        let names: Vec<_> = ImageTier::ALL.iter().map(|t| t.as_str()).collect();
        assert_eq!(names, ["bf16", "q8", "q4"]);
        for t in ImageTier::ALL {
            assert_eq!(t.bits().and_then(ImageTier::from_bits).unwrap_or(t), t);
        }
        assert_eq!(ImageTier::from_bits(6), None);
    }

    #[test]
    fn packed_dits_live_in_the_image_repo_and_the_dense_one_upstream() {
        assert_eq!(
            ImageTier::Q8.dit(),
            RepoDir {
                repo: "SceneWorks/minimax-h3-image-mlx",
                dir: "q8/transformer_diffusers".into()
            }
        );
        assert_eq!(ImageTier::Q4.dit().dir, "q4/transformer_diffusers");
        assert_eq!(ImageTier::Bf16.dit().repo, "MiniMaxAI/MiniMax-H3-Image");
        assert_eq!(ImageTier::Bf16.dit().dir, "transformer_diffusers");
    }

    #[test]
    fn the_pdd_heads_file_ships_beside_each_packed_dit_only() {
        assert_eq!(
            ImageTier::Q4.pdd_fused_heads().unwrap().dir,
            "q4/pdd_fused_heads.safetensors"
        );
        assert!(ImageTier::Q8.pdd_fused_heads().is_some());
        assert!(ImageTier::Bf16.pdd_fused_heads().is_none());
    }

    /// AC2: every tier's text encoder is H3's existing artifact, and no tier points at the image
    /// repository for it.
    #[test]
    fn the_text_encoder_is_h3s_existing_artifact_at_every_tier() {
        assert_eq!(
            ImageTier::Q8.text_encoder(),
            RepoDir {
                repo: "SceneWorks/minimax-h3-mlx",
                dir: "q8/text_encoder".into()
            }
        );
        assert_eq!(
            ImageTier::Q4.text_encoder(),
            RepoDir {
                repo: "SceneWorks/minimax-h3-mlx",
                dir: "q4/text_encoder".into()
            }
        );
        assert_eq!(
            ImageTier::Bf16.text_encoder(),
            RepoDir {
                repo: "MiniMaxAI/MiniMax-H3",
                dir: "text_encoder".into()
            }
        );
        for t in ImageTier::ALL {
            assert_ne!(t.text_encoder().repo, IMAGE_TIER_REPO, "{t:?}");
        }
    }
}
