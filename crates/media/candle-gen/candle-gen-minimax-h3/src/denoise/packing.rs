//! The **packed sequence**: row order, the three index tensors, the per-row modality tags and row
//! classes, and the scatter/gather between per-modality row blocks and the one sequence.
//!
//! [`crate::dit::positions`] owns the *coordinate conventions*; this module owns the *assembly*
//! (`MiniMaxH3PrepareLayoutStep.build_packed_sequence`).
//!
//! # The row order
//!
//! ```text
//! [ text | keyframe conditions | target audio | target video ]
//!   ^0     ^condition_start      ^audio_start   ^video_start
//! ```
//!
//! The three index tensors are **not** three contiguous slices of that:
//!
//! | tensor | rows |
//! |---|---|
//! | `text_indices` | `0 .. num_text_tokens` |
//! | `audio_indices` | `audio_start .. video_start` — contiguous |
//! | `video_indices` | `condition_start .. audio_start` **then** `video_start .. seq_len` — it **skips the audio block** |
//!
//! That last one is the trap. `video_indices` is the conditioning rows *followed by* the target
//! rows, so "the generated video rows" is `video_indices[num_condition_video_rows..]`, and the
//! scheduler writes back only to that tail — which is how the `fl2va` anchors survive the whole
//! loop untouched. Treating `video_indices` as one contiguous range silently denoises the anchors.
//!
//! # Row classes
//!
//! Each row sits at one of **four** timesteps within a single forward pass
//! (`build_row_timesteps`). Read from the reference rather than restated:
//!
//! ```python
//! row_timesteps = torch.full((sequence_length,), video_timestep)          # <- the DEFAULT
//! row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
//! row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
//! row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
//! ```
//!
//! | class | rows | timestep |
//! |---|---|---|
//! | [`RowClass::Video`] | target video **and text** | the video schedule's `t` |
//! | [`RowClass::Audio`] | target audio | the audio schedule's `t` |
//! | [`RowClass::ConditionVideo`] | `fl2va` keyframe anchors | `max(video_t, 0.999)` |
//! | [`RowClass::ConditionAudio`] | `ref2va` reference soundtrack | `1.0` |
//!
//! **Text rows take the VIDEO timestep, not `1.0`.** They are simply never overwritten by any of
//! the three assignments, so they keep the default fill. The `1.0` class is real but belongs to
//! `ref2va` reference-audio rows, which `t2va` and `fl2va` have none of.
//!
//! # Why all four classes are always declared
//!
//! [`crate::dit::adaln::TimestepSchedule::new`] rejects a schedule whose steps declare different
//! class counts, because a dropped class shifts every later class index. So [`PackedLayout`] emits
//! all four at every step even when no row belongs to two of them; the unused timesteps cost one
//! extra block of the modulation table (~10 MB against the 26.02 GB the eviction saves) and buy a
//! class order that is stable across presentations.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::{CandleError, Result};

use crate::denoise::geometry::JointGeometry;
use crate::denoise::schedule::{KEYFRAME_NOISE_AUG, REFERENCE_AUDIO_TIMESTEP};
use crate::dit::config::MODALITY_NUM;
use crate::dit::positions::{
    audio_position_ids, frame_grid, keyframe_position_ids, text_position_ids, to_tensor,
    video_position_ids, KeyframeAnchor,
};

/// `MINIMAX_H3_VIDEO_TAG`.
pub const VIDEO_TAG: u32 = 0;
/// `MINIMAX_H3_TEXT_TAG`.
pub const TEXT_TAG: u32 = 1;
/// `MINIMAX_H3_AUDIO_TAG`.
pub const AUDIO_TAG: u32 = 2;

/// Row classes every step of a [`crate::dit::adaln::TimestepSchedule`] declares, in this order.
pub const NUM_ROW_CLASSES: usize = 4;

/// Which of the four per-step timesteps a row sits at. The discriminants are the class indices
/// [`crate::dit::adaln::TimestepSchedule`] is built with, and the order is load-bearing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RowClass {
    /// Target video rows **and text rows** — the video schedule's `t`. See the module docs: text
    /// inherits the default fill, it is not pinned to 1.0.
    Video = 0,
    /// Target audio rows — the audio schedule's `t`.
    Audio = 1,
    /// `fl2va` keyframe conditioning rows — `max(video_t, 0.999)`.
    ConditionVideo = 2,
    /// `ref2va` reference-soundtrack rows — `1.0`.
    ConditionAudio = 3,
}

/// One assembled packed sequence's structure — everything about the rows, and nothing about their
/// contents.
///
/// Built once per request and reused at every step: none of it depends on the timestep.
#[derive(Debug, Clone)]
pub struct PackedLayout {
    geometry: JointGeometry,
    num_text_tokens: usize,
    audio_channels: usize,
    rows_per_frame: usize,
    num_condition_video_rows: usize,
    num_condition_audio_rows: usize,
    seq_len: usize,
    position_rows: Vec<[f64; 3]>,
    position_ids: Tensor,
    token_tags: Vec<u32>,
    row_classes: Vec<u32>,
    video_indices: Vec<u32>,
    audio_indices: Vec<u32>,
    text_indices: Vec<u32>,
    /// `inverse[dest_row] = source_row` over `concat(text, video, audio)`. See [`Self::scatter`].
    inverse: Tensor,
}

impl PackedLayout {
    /// Assemble the layout for a `t2va` / `fl2va` request.
    ///
    /// `text_token_tags` is one tag per text row — all [`TEXT_TAG`] for a plain prompt, but a
    /// keyframe's vision-block rows are tagged [`VIDEO_TAG`] by the conditioner, so it is supplied
    /// rather than assumed. `keyframe_anchors` is empty for `t2va`.
    pub fn build(
        geometry: JointGeometry,
        patch_size: [usize; 3],
        text_token_tags: &[u32],
        audio_channels: usize,
        keyframe_anchors: &[KeyframeAnchor],
        device: &Device,
    ) -> Result<Self> {
        geometry.validate()?;
        let (patch_h, patch_w) = (patch_size[1], patch_size[2]);
        if patch_h == 0 || patch_w == 0 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 packing: patch {patch_size:?} must be positive"
            )));
        }
        if text_token_tags.is_empty() {
            return Err(CandleError::Msg(
                "minimax-h3 packing: the packed sequence always carries at least one text row"
                    .into(),
            ));
        }
        if let Some(bad) = text_token_tags
            .iter()
            .find(|&&t| t as usize >= MODALITY_NUM)
        {
            return Err(CandleError::Msg(format!(
                "minimax-h3 packing: text token tag {bad} is outside [0, {MODALITY_NUM})"
            )));
        }
        if audio_channels == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 packing: audio_channels must be positive, got 0".into(),
            ));
        }

        let num_text_tokens = text_token_tags.len();
        let rows_per_frame = (geometry.latent_height / patch_h) * (geometry.latent_width / patch_w);
        if rows_per_frame == 0 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 packing: a {}x{} latent at patch {patch_h}x{patch_w} has no rows",
                geometry.latent_height, geometry.latent_width
            )));
        }
        let num_condition_video_rows = keyframe_anchors.len() * rows_per_frame;
        // `t2va` / `fl2va` carry no reference soundtrack; `ref2va` (sc-17157) is the only
        // presentation with leading audio conditioning rows.
        let num_condition_audio_rows = 0usize;
        let num_audio_rows = geometry.num_audio_latents * audio_channels;
        let num_video_rows = geometry.num_latent_frames * rows_per_frame;
        let seq_len = num_text_tokens + num_condition_video_rows + num_audio_rows + num_video_rows;

        let condition_start = num_text_tokens;
        let audio_start = condition_start + num_condition_video_rows;
        let video_start = audio_start + num_audio_rows;

        // --- positions, in packed order -------------------------------------------------------
        let (frame_rows, width_grid) = frame_grid(
            geometry.latent_height,
            geometry.latent_width,
            patch_h,
            patch_w,
        )?;
        let mut rows = text_position_ids(num_text_tokens)?;
        for &anchor in keyframe_anchors {
            rows.extend(keyframe_position_ids(
                anchor,
                num_text_tokens,
                geometry.num_latent_frames,
                &frame_rows,
            )?);
        }
        rows.extend(audio_position_ids(
            num_text_tokens,
            geometry.num_audio_latents,
            audio_channels,
            &width_grid,
        )?);
        rows.extend(video_position_ids(
            num_text_tokens,
            geometry.num_latent_frames,
            &frame_rows,
        )?);
        if rows.len() != seq_len {
            return Err(CandleError::Msg(format!(
                "minimax-h3 packing: assembled {} position rows for a {seq_len}-row sequence",
                rows.len()
            )));
        }
        let position_ids = to_tensor(&rows, device)?;

        // --- indices ---------------------------------------------------------------------------
        // video_indices SKIPS the audio block: conditioning rows, then the target rows.
        let video_indices: Vec<u32> = (condition_start..audio_start)
            .chain(video_start..seq_len)
            .map(|i| i as u32)
            .collect();
        let audio_indices: Vec<u32> = (audio_start..video_start).map(|i| i as u32).collect();
        let text_indices: Vec<u32> = (0..num_text_tokens).map(|i| i as u32).collect();

        // --- tags and row classes ---------------------------------------------------------------
        let mut tags = vec![0u32; seq_len];
        let mut classes = vec![0u32; seq_len];
        for (row, &tag) in text_indices.iter().zip(text_token_tags) {
            tags[*row as usize] = tag;
            // Text rows are never reassigned by `build_row_timesteps`: they keep the video default.
            classes[*row as usize] = RowClass::Video as u32;
        }
        for (i, &row) in audio_indices.iter().enumerate() {
            tags[row as usize] = AUDIO_TAG;
            classes[row as usize] = if i < num_condition_audio_rows {
                RowClass::ConditionAudio as u32
            } else {
                RowClass::Audio as u32
            };
        }
        for (i, &row) in video_indices.iter().enumerate() {
            tags[row as usize] = VIDEO_TAG;
            classes[row as usize] = if i < num_condition_video_rows {
                RowClass::ConditionVideo as u32
            } else {
                RowClass::Video as u32
            };
        }

        // --- the scatter permutation -------------------------------------------------------------
        // `concat(text, video, audio)` is the order per-modality row blocks arrive in; `inverse`
        // maps a destination row back to its source row, so one gather performs the scatter.
        let mut inverse = vec![u32::MAX; seq_len];
        for (source, &dest) in text_indices
            .iter()
            .chain(&video_indices)
            .chain(&audio_indices)
            .enumerate()
        {
            if inverse[dest as usize] != u32::MAX {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 packing: row {dest} is claimed by two index tensors"
                )));
            }
            inverse[dest as usize] = source as u32;
        }
        if inverse.contains(&u32::MAX) {
            return Err(CandleError::Msg(
                "minimax-h3 packing: the three index tensors do not cover every row".into(),
            ));
        }
        let inverse = Tensor::from_vec(inverse, (seq_len,), device)?;

        Ok(Self {
            geometry,
            num_text_tokens,
            audio_channels,
            rows_per_frame,
            num_condition_video_rows,
            num_condition_audio_rows,
            seq_len,
            position_rows: rows,
            position_ids,
            token_tags: tags,
            row_classes: classes,
            video_indices,
            audio_indices,
            text_indices,
            inverse,
        })
    }

    /// Rows in the packed sequence.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// The geometry this layout was built for.
    pub fn geometry(&self) -> &JointGeometry {
        &self.geometry
    }

    /// Text rows.
    pub fn num_text_tokens(&self) -> usize {
        self.num_text_tokens
    }

    /// Patched rows one latent frame contributes.
    pub fn rows_per_frame(&self) -> usize {
        self.rows_per_frame
    }

    /// Channels the soundtrack is packed channel-major over.
    pub fn audio_channels(&self) -> usize {
        self.audio_channels
    }

    /// Leading rows of [`Self::video_indices`] that are `fl2va` conditioning anchors, **not**
    /// denoised.
    pub fn num_condition_video_rows(&self) -> usize {
        self.num_condition_video_rows
    }

    /// Leading rows of [`Self::audio_indices`] that are `ref2va` reference audio — always 0 for
    /// `t2va` / `fl2va`.
    pub fn num_condition_audio_rows(&self) -> usize {
        self.num_condition_audio_rows
    }

    /// `[seq_len, 3]` float32 MM-RoPE coordinates.
    pub fn position_ids(&self) -> &Tensor {
        &self.position_ids
    }

    /// The same coordinates as host-side rows, for an exact comparison against a reference grid.
    pub fn position_rows(&self) -> &[[f64; 3]] {
        &self.position_rows
    }

    /// `[seq_len]` per-row modality tags.
    pub fn token_tags(&self) -> &[u32] {
        &self.token_tags
    }

    /// `[seq_len]` per-row [`RowClass`] — the class index
    /// [`crate::dit::adaln::TimestepSchedule::adaln_indices`] consumes.
    pub fn row_classes(&self) -> &[u32] {
        &self.row_classes
    }

    /// Rows the video stream owns: conditioning anchors first, then the target rows.
    pub fn video_indices(&self) -> &[u32] {
        &self.video_indices
    }

    /// Rows the audio stream owns.
    pub fn audio_indices(&self) -> &[u32] {
        &self.audio_indices
    }

    /// Rows the text condition owns.
    pub fn text_indices(&self) -> &[u32] {
        &self.text_indices
    }

    /// The four per-step timesteps, in [`RowClass`] order, for a given pair of schedule timesteps.
    ///
    /// This is `build_row_timesteps`' class table: video `t`, audio `t`,
    /// `max(video_t, KEYFRAME_NOISE_AUG)`, and `REFERENCE_AUDIO_TIMESTEP`.
    pub fn row_timesteps(video_t: f32, audio_t: f32) -> [f32; NUM_ROW_CLASSES] {
        [
            video_t,
            audio_t,
            video_t.max(KEYFRAME_NOISE_AUG),
            REFERENCE_AUDIO_TIMESTEP,
        ]
    }

    /// Scatter per-modality row blocks into one `[1, seq_len, features]` packed sequence.
    ///
    /// `text` is `[1, num_text_tokens, F]`, `video` is `[1, video_indices.len(), F]` (conditioning
    /// rows first) and `audio` is `[1, audio_indices.len(), F]`. Performed as a single gather
    /// through the layout's own inverse permutation, so a wrong index tensor produces a wrong
    /// sequence rather than a coincidentally-right concatenation.
    pub fn scatter(&self, text: &Tensor, video: &Tensor, audio: &Tensor) -> Result<Tensor> {
        let features = self.check_block(text, self.text_indices.len(), "text")?;
        if self.check_block(video, self.video_indices.len(), "video")? != features
            || self.check_block(audio, self.audio_indices.len(), "audio")? != features
        {
            return Err(CandleError::Msg(
                "minimax-h3 packing: the three row blocks must share a feature width".into(),
            ));
        }
        let sources = Tensor::cat(&[text, video, audio], 1)?.contiguous()?;
        Ok(sources.index_select(&self.inverse, 1)?)
    }

    /// Gather one modality's rows back out of a packed sequence.
    ///
    /// The inverse of [`Self::scatter`] for one block; `[1, seq_len, F]` → `[1, n, F]`.
    pub fn gather(&self, packed: &Tensor, rows: &[u32]) -> Result<Tensor> {
        let s = packed.dims();
        if s.len() != 3 || s[0] != 1 || s[1] != self.seq_len {
            return Err(CandleError::Msg(format!(
                "minimax-h3 packing: expected a [1, {}, F] packed sequence, got {s:?}",
                self.seq_len
            )));
        }
        gather_rows(packed, rows, self.seq_len)
    }

    /// The generated (non-conditioning) tail of a video row block — the rows the scheduler writes.
    pub fn generated_video_rows(&self) -> &[u32] {
        &self.video_indices[self.num_condition_video_rows..]
    }

    /// The generated tail of an audio row block.
    pub fn generated_audio_rows(&self) -> &[u32] {
        &self.audio_indices[self.num_condition_audio_rows..]
    }

    fn check_block(&self, x: &Tensor, rows: usize, what: &str) -> Result<usize> {
        let s = x.dims();
        if s.len() != 3 || s[0] != 1 || s[1] != rows {
            return Err(CandleError::Msg(format!(
                "minimax-h3 packing: the {what} block must be [1, {rows}, F], got {s:?}"
            )));
        }
        Ok(s[2])
    }
}

/// Gather rows out of a `[1, seq_len, F]` sequence, bounds-checked on the host.
///
/// The host check is what makes a wrong index a legible error naming the sequence rather than an
/// opaque kernel failure — the same reason [`crate::dit::block`] checks its AdaLN indices.
pub(crate) fn gather_rows(packed: &Tensor, rows: &[u32], seq_len: usize) -> Result<Tensor> {
    if rows.is_empty() {
        return Err(CandleError::Msg(
            "minimax-h3 packing: cannot gather zero rows".into(),
        ));
    }
    if let Some(bad) = rows.iter().find(|&&r| r as usize >= seq_len) {
        return Err(CandleError::Msg(format!(
            "minimax-h3 packing: row {bad} is outside [0, {seq_len})"
        )));
    }
    let idx =
        Tensor::from_vec(rows.to_vec(), (rows.len(),), packed.device())?.to_dtype(DType::U32)?;
    Ok(packed.index_select(&idx, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::denoise::schedule::{DenoiseModality, SigmaSchedule};

    /// The **shortest legal render** at a small canvas: 124 frames ⇒ 37 latent frames and 207
    /// stereo audio latents, on a 4×6 latent at patch `[1, 2, 2]`.
    ///
    /// Deliberately a real geometry rather than a shrunken one. 37 latent frames walks the
    /// `ROPE_FRAMES_PER_LATENT` cycle past index 5, which
    /// [`crate::dit::positions::ROPE_FRAMES_PER_LATENT`] warns is "past every tiny fixture" — and
    /// 207 ≠ 37·k, so the AV residual is exercised too. It is still only 641 rows.
    const TEST_FRAMES: usize = 124;

    fn dev() -> Device {
        Device::Cpu
    }

    fn layout(anchors: &[KeyframeAnchor]) -> PackedLayout {
        let geometry = JointGeometry::new(TEST_FRAMES, 4, 6).unwrap();
        PackedLayout::build(geometry, [1, 2, 2], &[TEXT_TAG; 5], 2, anchors, &dev()).unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn the_row_order_is_text_conditions_audio_video() {
        let l = layout(&[]);
        // 5 text + 0 conditions + 207·2 audio + 37·(2·3) video = 641.
        assert_eq!(l.geometry().num_latent_frames, 37);
        assert_eq!(l.geometry().num_audio_latents, 207);
        assert_eq!(l.rows_per_frame(), 6);
        assert_eq!(l.seq_len(), 5 + 414 + 222);
        assert_eq!(l.text_indices(), (0..5u32).collect::<Vec<_>>());
        assert_eq!(l.audio_indices(), (5..419u32).collect::<Vec<_>>());
        assert_eq!(l.video_indices(), (419..641u32).collect::<Vec<_>>());
        assert_eq!(l.num_condition_video_rows(), 0);

        let k = layout(&[KeyframeAnchor::First, KeyframeAnchor::Last]);
        assert_eq!(k.seq_len(), 641 + 12);
        assert_eq!(k.num_condition_video_rows(), 12);
        // ...and the conditioning rows sit BETWEEN the text and the audio.
        assert_eq!(k.audio_indices(), (17..431u32).collect::<Vec<_>>());
    }

    /// **`video_indices` skips the audio block.** The trap: it is the conditioning rows followed by
    /// the target rows, not one contiguous range.
    #[test]
    fn video_indices_are_discontiguous_across_the_audio_block() {
        let k = layout(&[KeyframeAnchor::First, KeyframeAnchor::Last]);
        let v = k.video_indices();
        assert_eq!(v.len(), 12 + 222);
        assert_eq!(&v[..12], &(5..17u32).collect::<Vec<_>>()[..], "the anchors");
        assert_eq!(
            &v[12..],
            &(431..653u32).collect::<Vec<_>>()[..],
            "the targets"
        );
        assert!(
            v[11] + 1 != v[12],
            "the audio block sits between the two runs"
        );
        // The scheduler writes only the tail — which is how the anchors survive the loop.
        assert_eq!(
            k.generated_video_rows(),
            &(431..653u32).collect::<Vec<_>>()[..]
        );
        assert_eq!(k.generated_audio_rows(), k.audio_indices());
    }

    /// **Text rows carry the VIDEO timestep**, not a clean 1.0 — they are never reassigned by
    /// `build_row_timesteps`.
    #[test]
    fn text_rows_carry_the_video_timestep() {
        let k = layout(&[KeyframeAnchor::First]);
        let classes = k.row_classes();
        for &row in k.text_indices() {
            assert_eq!(
                classes[row as usize],
                RowClass::Video as u32,
                "text row {row} must sit at the video timestep, not 1.0"
            );
        }
        // ...and the 1.0 class exists but is unused here, because `fl2va` has no reference audio.
        assert_eq!(k.num_condition_audio_rows(), 0);
        assert!(
            !classes.contains(&(RowClass::ConditionAudio as u32)),
            "no fl2va row sits at t = 1.0"
        );

        let t = PackedLayout::row_timesteps(0.4, 0.7);
        assert_eq!(t.len(), NUM_ROW_CLASSES);
        assert_eq!(t[RowClass::Video as usize], 0.4);
        assert_eq!(t[RowClass::Audio as usize], 0.7);
        assert_eq!(t[RowClass::ConditionVideo as usize], KEYFRAME_NOISE_AUG);
        assert_eq!(t[RowClass::ConditionAudio as usize], 1.0);
        // Once the video schedule passes 0.999 the anchors follow it.
        assert_eq!(PackedLayout::row_timesteps(0.9995, 0.7)[2], 0.9995);
    }

    /// Tags and classes are assigned per row, and the conditioning anchors are tagged **video**
    /// while sitting in their own class.
    #[test]
    fn tags_and_classes_are_assigned_per_row() {
        let k = layout(&[KeyframeAnchor::First]);
        let tags = k.token_tags();
        let classes = k.row_classes();
        for &row in k.text_indices() {
            assert_eq!(tags[row as usize], TEXT_TAG);
        }
        for &row in k.audio_indices() {
            assert_eq!(tags[row as usize], AUDIO_TAG);
            assert_eq!(classes[row as usize], RowClass::Audio as u32);
        }
        for (i, &row) in k.video_indices().iter().enumerate() {
            assert_eq!(tags[row as usize], VIDEO_TAG, "anchors are tagged video");
            let want = if i < k.num_condition_video_rows() {
                RowClass::ConditionVideo
            } else {
                RowClass::Video
            };
            assert_eq!(classes[row as usize], want as u32);
        }
        assert!(tags.iter().all(|&t| (t as usize) < MODALITY_NUM));
    }

    /// The scatter round-trips through the index tensors, and its result is the packed row order.
    #[test]
    fn scatter_and_gather_round_trip() {
        let k = layout(&[KeyframeAnchor::First]);
        let f = 2usize;
        let block = |n: usize, base: f32| {
            let v: Vec<f32> = (0..n * f).map(|i| base + i as f32).collect();
            Tensor::from_vec(v, (1, n, f), &dev()).unwrap()
        };
        let text = block(k.text_indices().len(), 1000.0);
        let video = block(k.video_indices().len(), 2000.0);
        let audio = block(k.audio_indices().len(), 3000.0);

        let packed = k.scatter(&text, &video, &audio).unwrap();
        assert_eq!(packed.dims(), &[1, k.seq_len(), f]);

        for (rows, want) in [
            (k.text_indices(), &text),
            (k.video_indices(), &video),
            (k.audio_indices(), &audio),
        ] {
            let got = k.gather(&packed, rows).unwrap();
            assert_eq!(flat(&got), flat(want));
        }

        // The packed order really is [text | conditions | audio | targets]: row 0 is text and the
        // first audio row lands after the conditioning block.
        let f_flat = flat(&packed);
        assert_eq!(f_flat[0], 1000.0);
        let audio_start = k.audio_indices()[0] as usize * f;
        assert_eq!(f_flat[audio_start], 3000.0);

        assert!(
            k.scatter(&video, &video, &audio).is_err(),
            "wrong block size"
        );
        assert!(
            k.gather(&text, k.text_indices()).is_err(),
            "a block that is not the packed sequence is rejected"
        );
        for bad in [
            vec![k.seq_len() as u32],
            vec![0, k.seq_len() as u32 + 7],
            vec![],
        ] {
            assert!(
                k.gather(&packed, &bad).is_err(),
                "{bad:?} must be rejected rather than reading adjacent memory"
            );
        }
    }

    /// The four row classes drive a `TimestepSchedule` whose per-row lookups resolve to the right
    /// timesteps, with no out-of-bounds gather.
    #[test]
    fn the_layout_drives_an_adaln_schedule() {
        use crate::dit::adaln::TimestepSchedule;

        let k = layout(&[KeyframeAnchor::First]);
        let video = SigmaSchedule::new(DenoiseModality::Video, 4).unwrap();
        let audio = SigmaSchedule::new(DenoiseModality::Audio, 4).unwrap();
        let steps: Vec<Vec<f32>> = (0..video.num_evals())
            .map(|i| {
                PackedLayout::row_timesteps(video.timesteps()[i], audio.timesteps()[i]).to_vec()
            })
            .collect();
        let schedule = TimestepSchedule::new(steps).unwrap();
        assert_eq!(schedule.num_row_classes(), NUM_ROW_CLASSES);

        for step in 0..schedule.num_steps() {
            let idx = schedule
                .adaln_indices(step, k.row_classes(), k.token_tags(), &dev())
                .unwrap();
            assert_eq!(idx.dims(), &[k.seq_len()]);
            let v = idx.to_vec1::<u32>().unwrap();
            assert!(*v.iter().max().unwrap() < schedule.modulation_rows() as u32);
            // Every text row addresses the same table block as a target video row: same class,
            // different tag.
            let text_row = k.text_indices()[0] as usize;
            let target_row = *k.generated_video_rows().first().unwrap() as usize;
            assert_eq!(
                v[text_row] - TEXT_TAG,
                v[target_row] - VIDEO_TAG,
                "text and target video share a timestep block"
            );
        }
    }

    #[test]
    fn malformed_layouts_are_rejected() {
        let g = JointGeometry::new(TEST_FRAMES, 4, 6).unwrap();
        let d = dev();
        assert!(
            PackedLayout::build(g, [1, 2, 2], &[], 2, &[], &d).is_err(),
            "no text"
        );
        assert!(
            PackedLayout::build(g, [1, 2, 2], &[9], 2, &[], &d).is_err(),
            "tag out of range"
        );
        assert!(
            PackedLayout::build(g, [1, 2, 2], &[TEXT_TAG], 0, &[], &d).is_err(),
            "no audio channels"
        );
        assert!(
            PackedLayout::build(g, [1, 5, 5], &[TEXT_TAG], 2, &[], &d).is_err(),
            "latent extent is not a patch multiple"
        );
        assert!(
            PackedLayout::build(g, [1, 0, 2], &[TEXT_TAG], 2, &[], &d).is_err(),
            "zero patch"
        );
        // A geometry whose latent-frame count has drifted off the audio clock never reaches the
        // row assembly at all.
        let drifted = JointGeometry {
            num_latent_frames: 31,
            ..g
        };
        assert!(
            PackedLayout::build(drifted, [1, 2, 2], &[TEXT_TAG], 2, &[], &d).is_err(),
            "the packing must not assemble a time-misaligned request"
        );
    }
}
