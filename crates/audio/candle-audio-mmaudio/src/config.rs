//! Synchformer / MotionFormer visual-encoder hyperparameters.
//!
//! Every value is transcribed verbatim from MMAudio's pinned `divided_224_16x4.yaml` (the config
//! MMAudio instantiates the `vfeat_extractor` with) and `mmaudio/model/sequence_config.py` (the
//! segmentation scheme). See the crate-root doc for the full source cross-reference.

/// Spatial crop / input resolution (px). `DATA.TRAIN_CROP_SIZE`.
pub const IMG_SIZE: usize = 224;
/// Spatial patch size (px). `VIT.PATCH_SIZE`.
pub const PATCH_SIZE: usize = 16;
/// Temporal patch size (frames per patch along time). `VIT.PATCH_SIZE_TEMP` (`z_block_size`).
pub const PATCH_SIZE_TEMP: usize = 2;
/// Input RGB channels. `VIT.CHANNELS`.
pub const IN_CHANS: usize = 3;
/// Transformer hidden width. `VIT.EMBED_DIM`.
pub const EMBED_DIM: usize = 768;
/// Number of divided space-time blocks. `VIT.DEPTH`.
pub const DEPTH: usize = 12;
/// Attention heads (backbone and aggregation). `VIT.NUM_HEADS`.
pub const NUM_HEADS: usize = 12;
/// Per-head dimension (`EMBED_DIM / NUM_HEADS`).
pub const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS;
/// MLP expansion ratio. `VIT.MLP_RATIO` → hidden = 3072.
pub const MLP_RATIO: usize = 4;
/// MLP hidden width.
pub const MLP_HIDDEN: usize = EMBED_DIM * MLP_RATIO;
/// Number of RGB frames per segment. `DATA.NUM_FRAMES`.
pub const NUM_FRAMES: usize = 16;
/// Temporal token resolution after 3D patching (`NUM_FRAMES / PATCH_SIZE_TEMP`). `VIT.TEMPORAL_RESOLUTION`.
pub const TEMPORAL_RESOLUTION: usize = NUM_FRAMES / PATCH_SIZE_TEMP;
/// Spatial patches per frame (`(IMG_SIZE / PATCH_SIZE)^2` = 14×14 = 196). = `patch_embed.num_patches`.
pub const NUM_SPATIAL_PATCHES: usize = (IMG_SIZE / PATCH_SIZE) * (IMG_SIZE / PATCH_SIZE);
/// Patches along one spatial axis (14).
pub const GRID: usize = IMG_SIZE / PATCH_SIZE;
/// LayerNorm epsilon used throughout the backbone and aggregation (`layer_norm_eps=1e-6`).
pub const LN_EPS: f64 = 1e-6;

/// Per-channel normalization mean. `DATA.MEAN` = 0.5 (maps `[0,1]` → `[-1,1]` with STD 0.5).
pub const NORM_MEAN: f32 = 0.5;
/// Per-channel normalization std. `DATA.STD` = 0.5.
pub const NORM_STD: f32 = 0.5;

/// Frame sampling rate Synchformer's data pipeline / MMAudio operate at (Hz).
///
/// **Resolved to 25, not 24.** MMAudio `sequence_config.py` sets `sync_frame_rate = 25`; the
/// Synchformer repo's data reencode targets `vfps=25` (`h264_..._25fps_...`). The arithmetic
/// clincher: one segment is exactly `NUM_FRAMES / SYNC_FRAME_RATE = 16/25 = 0.64 s`; 0.64 s × 24 =
/// 15.36 frames (non-integer), so 25 fps is the only rate consistent with a 16-frame segment. The
/// "24 fps" in the paper is prose approximation, not what the code uses.
pub const SYNC_FRAME_RATE: usize = 25;

/// Frames advanced between consecutive segments (`sync_step_size`). = 8 → 50% segment overlap.
pub const SYNC_STEP_SIZE: usize = 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_dims_match_reference() {
        assert_eq!(NUM_SPATIAL_PATCHES, 196, "14x14 spatial patches");
        assert_eq!(TEMPORAL_RESOLUTION, 8, "16 frames / z=2");
        assert_eq!(HEAD_DIM, 64, "768 / 12 heads");
        assert_eq!(MLP_HIDDEN, 3072, "768 * 4");
        assert_eq!(GRID, 14);
    }

    #[test]
    fn segment_is_exactly_0_64s_at_25fps() {
        // The 24-vs-25 fps resolver: only 25 fps makes a 16-frame segment an integer duration.
        assert_eq!(SYNC_FRAME_RATE, 25);
        let dur = NUM_FRAMES as f64 / SYNC_FRAME_RATE as f64;
        assert!((dur - 0.64).abs() < 1e-9, "segment duration must be 0.64s");
        // 24 fps would be non-integer frames for 0.64s.
        assert!(
            (0.64 * 24.0_f64).fract() > 1e-6,
            "24 fps is inconsistent with 16-frame segment"
        );
    }
}
