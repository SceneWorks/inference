//! Packed native-resolution geometry + `MageFlowEmbedRope` (the 3-axis "msrope") — **owned by
//! sc-14040**.
//!
//! Port of `_vendor/mage_flow/models/modules/mage_layers.py:105-210`. The pinned facts (all
//! verified in `_vendor/MAGE_FLOW_GAPS.md` GAP 3, several of which correct the epic description):
//!
//! - **`theta = 10000` is hardcoded in code**, not read from the config's `"theta"` key
//!   ([`crate::config::ROPE_THETA`]); the axis split is `axes_dim = [16, 56, 56]` =
//!   `(frame, height, width)`, half-dims `[8, 28, 28]`, asserted to sum to `head_dim`.
//! - There is **no `axes_lens` field**. A fixed [`crate::config::ROPE_TABLE_LEN`]-entry positive
//!   table plus a mirrored negative table (`index.flip(0) * -1 - 1`) is precomputed per axis.
//! - `scale_rope = true`: height/width are **centred** —
//!   `cat(neg[-(L - L/2):], pos[:L/2])`, i.e. indices `-(L - L/2) … L/2 - 1` (`:194-203`).
//!   The frame axis is *not* centred: it is the segment's index in `img_shapes` (`:171`, `:192`).
//! - Coordinates come from **`img_shapes`, never `img_ids`**. `img_ids` is computed by the
//!   reference pipeline but never reaches the model — it is vestigial, and is not ported.
//!
//! **Trap — `batch_cfg` shifts the unconditional branch's frame index.** `_build_pack_ctx`
//! concatenates the *segment list* (`pipeline.py:167`), so the duplicated uncond half rotates at
//! frame index **1**, not 0 (measured: the cond half is exact identity, the uncond half differs by
//! max_abs 0.9589, confined entirely to the frame lanes). [`PackLayout::fused_cfg`] reproduces the
//! shift bit-for-bit — the parity goldens were dumped through the fused path, and
//! `tools/verify_mage_flow_golden.py` asserts it there.
//!
//! It shifts the **table**; it does not change the **render**. See [`PackLayout::fused_cfg`] for
//! the argument (RoPE is relative, and the duplication offsets every shape in the second copy
//! equally) and `tests/dit_real_weights.rs` for the measurement. That corrects the inference drawn
//! in the sc-14036 write-up; the reference's docstring at `pipeline.py:136-140` is right about the
//! output even though the tables differ.
//!
//! ## Table-free equivalence
//!
//! The reference materialises a `[4096, 64]` complex table per sign and *indexes* it. This port
//! computes `angle = coord · theta^(-2k/d)` for each token's `(frame, y, x)` coordinate directly —
//! the identical arithmetic on the identical f32 inputs, since the table rows are exactly those
//! products. The table's other role is a **capacity bound**, which [`MsRope::forward`] enforces
//! explicitly rather than reproducing as an out-of-range slice deep inside a `view`.

use mlx_rs::Array;

use mlx_gen::{nn, Error, Result};

use crate::config::{MageFlowConfig, ROPE_TABLE_LEN, ROPE_THETA, SCALE_ROPE};

/// One packed image segment's `(frames, height, width)` — an `img_shapes` entry
/// (`mage_flow.py:311`, `:356`). `patch_size == 1`, so `height`/`width` are **latent** cells
/// (pixels / 16) and the segment contributes `frames · height · width` tokens.
///
/// The segment's **position in the `img_shapes` list is its msrope frame index**
/// (`mage_layers.py:171`, `:192`) — which is exactly why the fused-CFG duplicate rotates at 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImgShape {
    pub frames: i32,
    pub height: i32,
    pub width: i32,
}

impl ImgShape {
    /// A single-frame latent grid — every generation segment
    /// (`img_shapes.append([(1, h, w)])`, `mage_flow.py:310`).
    pub fn latent(height: i32, width: i32) -> Self {
        Self {
            frames: 1,
            height,
            width,
        }
    }

    pub fn new(frames: i32, height: i32, width: i32) -> Self {
        Self {
            frames,
            height,
            width,
        }
    }

    /// Token count contributed by this segment, `frames · height · width`
    /// (`mage_layers.py:188`).
    pub fn tokens(&self) -> i32 {
        self.frames * self.height * self.width
    }

    fn validate(&self) -> Result<()> {
        if self.frames <= 0 || self.height <= 0 || self.width <= 0 {
            return Err(Error::Msg(format!(
                "mage_flow: img_shape {self:?} must have positive frames/height/width"
            )));
        }
        Ok(())
    }
}

/// The varlen packing of one NR-MMDiT forward: the msrope segment list (`img_shapes`) plus the
/// attention segmentation of the image and text streams (`img_cu_seqlens` / `txt_cu_seqlens`),
/// which isolate samples by cumulative offsets rather than by a block-diagonal mask.
///
/// **`img_shapes` entries and attention segments are NOT one-to-one.** They coincide for
/// generation — one latent grid per sample (`pipeline.py:310-321`) — but the edit path packs
/// `[noisy_target, ref₁, …, ref_N]` into a **single** attention segment of length `Lt + N·Lr`
/// while appending `N + 1` entries to `img_shapes` (`pipeline.py:517-519`). That is what gives the
/// frame axis its job: within one attention window the target rotates at frame 0 and ref *j* at
/// frame *j*, so RoPE's relative-position arithmetic separates them.
///
/// The two are therefore stored separately and cross-checked: every attention boundary must fall
/// on an `img_shapes` boundary, and the totals must agree. The reference threads the two lists
/// independently all the way from the pipeline with nothing checking that they describe the same
/// packing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackLayout {
    img_shapes: Vec<ImgShape>,
    img_lens: Vec<i32>,
    txt_lens: Vec<i32>,
}

impl PackLayout {
    /// The general form: an msrope segment list plus the attention segmentation of both streams.
    ///
    /// `img_lens` and `txt_lens` must have one entry per sample; `img_lens` must partition the
    /// `img_shapes` token stream on shape boundaries.
    pub fn new(img_shapes: Vec<ImgShape>, img_lens: Vec<i32>, txt_lens: Vec<i32>) -> Result<Self> {
        if img_shapes.is_empty() {
            return Err(Error::Msg(
                "mage_flow: a pack needs at least one image shape".into(),
            ));
        }
        if img_lens.len() != txt_lens.len() {
            return Err(Error::Msg(format!(
                "mage_flow: pack has {} image attention segments but {} text segments",
                img_lens.len(),
                txt_lens.len()
            )));
        }
        if img_lens.is_empty() {
            return Err(Error::Msg(
                "mage_flow: a pack needs at least one attention segment".into(),
            ));
        }
        for shape in &img_shapes {
            shape.validate()?;
        }
        for (what, lens) in [("image", &img_lens), ("text", &txt_lens)] {
            if lens.iter().any(|&len| len <= 0) {
                return Err(Error::Msg(format!(
                    "mage_flow: {what} attention segment lengths must be > 0 (got {lens:?})"
                )));
            }
        }
        // Every attention boundary must land on a shape boundary, or a latent grid would be split
        // across two attention windows and its msrope coordinates would describe neither.
        let shape_bounds = cumulative(&img_shapes.iter().map(ImgShape::tokens).collect::<Vec<_>>());
        let seg_bounds = cumulative(&img_lens);
        if seg_bounds.last() != shape_bounds.last() {
            return Err(Error::Msg(format!(
                "mage_flow: img_shapes carry {} tokens but the attention segments carry {}",
                shape_bounds.last().copied().unwrap_or(0),
                seg_bounds.last().copied().unwrap_or(0)
            )));
        }
        if let Some(bad) = seg_bounds.iter().find(|b| !shape_bounds.contains(b)) {
            return Err(Error::Msg(format!(
                "mage_flow: attention boundary {bad} does not fall on an img_shapes boundary \
                 ({shape_bounds:?}) — a latent grid would straddle two attention windows"
            )));
        }
        Ok(Self {
            img_shapes,
            img_lens,
            txt_lens,
        })
    }

    /// The **generation** packing: one latent grid per sample, so each `img_shapes` entry is its
    /// own attention segment (`pipeline.py:310-321`).
    pub fn generation(img_shapes: Vec<ImgShape>, txt_lens: Vec<i32>) -> Result<Self> {
        let img_lens = img_shapes.iter().map(ImgShape::tokens).collect();
        Self::new(img_shapes, img_lens, txt_lens)
    }

    /// The **fused classifier-free-guidance** layout: this (conditional) pack duplicated, with the
    /// negative text segments appended — a faithful port of `_build_pack_ctx`'s `batch_cfg` branch
    /// (`pipeline.py:161-173`), where `d_img_shapes = [img_shapes[0] + img_shapes[0]]` and
    /// `d_img_cu = cu(img_lens + img_lens)`.
    ///
    /// **The msrope frame index of the duplicate is shifted**, because the *shape list* is what
    /// gets concatenated: with one generation sample the copy lands at list position 1 and
    /// `_compute_video_freqs` rotates it at frame **1** rather than 0 (`mage_layers.py:171`,
    /// `:192`). That shift is real in the table and is reproduced here bit-for-bit.
    ///
    /// It is, however, **inert in the output**, and the port does not depend on which way that
    /// goes: RoPE encodes *relative* position, `q'ᵢ·k'ⱼ = qᵢᵀR(pⱼ − pᵢ)kⱼ`, and the duplication
    /// offsets every shape in the second copy by the same amount — so within each attention
    /// segment the frame differences, and therefore the attention scores, are unchanged. The
    /// reference's docstring at `pipeline.py:136-140` ("numerically identical to two separate
    /// forwards") is right about the render even though the tables differ; measured on the real
    /// checkpoint the two rotations move `dit_out` by mean-relative 1.1e-2, *below* the model's own
    /// bf16 sensitivity of 2.8e-2. See `tests/dit_real_weights.rs` for the measurement.
    ///
    /// The frame axis is not decorative — it is load-bearing exactly where a single attention
    /// segment spans several shapes, i.e. the edit path's `[target, ref₁, …]`.
    pub fn fused_cfg(&self, neg_txt_lens: &[i32]) -> Result<Self> {
        if neg_txt_lens.len() != self.txt_lens.len() {
            return Err(Error::Msg(format!(
                "mage_flow: fused CFG needs one negative text segment per positive one \
                 ({} vs {})",
                neg_txt_lens.len(),
                self.txt_lens.len()
            )));
        }
        Self::new(
            [self.img_shapes.as_slice(), self.img_shapes.as_slice()].concat(),
            [self.img_lens.as_slice(), self.img_lens.as_slice()].concat(),
            [self.txt_lens.as_slice(), neg_txt_lens].concat(),
        )
    }

    /// Attention segments in this pack (one per sample, or per CFG branch when fused).
    pub fn segments(&self) -> usize {
        self.img_lens.len()
    }

    /// msrope segments — **not** the same as [`PackLayout::segments`] on the edit path.
    pub fn img_shapes(&self) -> &[ImgShape] {
        &self.img_shapes
    }

    pub fn txt_lens(&self) -> &[i32] {
        &self.txt_lens
    }

    /// Per-attention-segment image token counts.
    pub fn img_lens(&self) -> &[i32] {
        &self.img_lens
    }

    /// `img_cu_seqlens` — `[0, l0, l0+l1, …]`, length `segments() + 1`.
    pub fn img_cu(&self) -> Vec<i32> {
        cumulative(&self.img_lens)
    }

    /// `txt_cu_seqlens` — `[0, t0, t0+t1, …]`, length `segments() + 1`.
    pub fn txt_cu(&self) -> Vec<i32> {
        cumulative(&self.txt_lens)
    }

    /// Total packed image tokens.
    pub fn img_tokens(&self) -> i32 {
        self.img_lens.iter().sum()
    }

    /// Total packed text tokens.
    pub fn txt_tokens(&self) -> i32 {
        self.txt_lens.iter().sum()
    }
}

fn cumulative(lens: &[i32]) -> Vec<i32> {
    let mut cu = Vec::with_capacity(lens.len() + 1);
    let mut acc = 0;
    cu.push(0);
    for &len in lens {
        acc += len;
        cu.push(acc);
    }
    cu
}

/// The msrope frequencies for one pack: `(cos, sin)`, each `[img_tokens, head_dim / 2]`.
///
/// The reference carries the same table as a single complex64 tensor (`torch.polar(ones, freqs)`)
/// and multiplies with `view_as_complex`. Splitting it into real/imaginary halves is the same
/// arithmetic without a complex dtype MLX does not carry — the golden dumper splits it the same
/// way, and for the same reason (`tools/dump_mage_flow_golden.py`, `_split_complex`).
#[derive(Debug, Clone)]
pub struct RopeTable {
    pub cos: Array,
    pub sin: Array,
}

/// `MageFlowEmbedRope(theta, axes_dim, scale_rope)` — weightless; fully determined by the config.
#[derive(Debug, Clone)]
pub struct MsRope {
    axes_dim: Vec<i32>,
    theta: f32,
    scale_rope: bool,
    table_len: i32,
}

impl MsRope {
    /// `scale_rope` and `table_len` are parameters (rather than hardcoded to [`SCALE_ROPE`] /
    /// [`ROPE_TABLE_LEN`]) because the reference implements both coordinate conventions
    /// (`mage_layers.py:193-206`) and the parity suite uses the un-centred one as a discrimination
    /// probe. Production builds go through [`MsRope::from_config`].
    pub fn new(axes_dim: &[i32], theta: f32, scale_rope: bool, table_len: i32) -> Result<Self> {
        if axes_dim.len() != 3 {
            return Err(Error::Msg(format!(
                "mage_flow: msrope needs 3 axes (frame, height, width); got {axes_dim:?}"
            )));
        }
        for &dim in axes_dim {
            if dim <= 0 || dim % 2 != 0 {
                return Err(Error::Msg(format!(
                    "mage_flow: msrope axis dim must be positive and even; got {axes_dim:?}"
                )));
            }
        }
        if table_len <= 0 {
            return Err(Error::Msg(format!(
                "mage_flow: msrope table length must be > 0 (got {table_len})"
            )));
        }
        Ok(Self {
            axes_dim: axes_dim.to_vec(),
            theta,
            scale_rope,
            table_len,
        })
    }

    /// The production embedder: `axes_dim` from the checkpoint config; `theta`, `scale_rope` and
    /// the table length from this crate's constants — because the reference hardcodes all three in
    /// code and ignores the published `"theta"` key (`mage_flow.py:72`).
    pub fn from_config(cfg: &MageFlowConfig) -> Result<Self> {
        Self::new(&cfg.axes_dim, ROPE_THETA, SCALE_ROPE, ROPE_TABLE_LEN)
    }

    pub fn axes_dim(&self) -> &[i32] {
        &self.axes_dim
    }

    pub fn theta(&self) -> f32 {
        self.theta
    }

    pub fn scale_rope(&self) -> bool {
        self.scale_rope
    }

    pub fn table_len(&self) -> i32 {
        self.table_len
    }

    /// Rotation width, `Σ axes_dim / 2` — the `head_dim / 2` complex lanes per token.
    pub fn half_dim(&self) -> i32 {
        self.axes_dim.iter().sum::<i32>() / 2
    }

    /// Build the `(cos, sin)` table for a whole pack, segments concatenated in order.
    pub fn forward(&self, shapes: &[ImgShape]) -> Result<RopeTable> {
        if shapes.is_empty() {
            return Err(Error::Msg(
                "mage_flow: msrope needs at least one image segment".into(),
            ));
        }
        let total: i64 = shapes.iter().map(|s| s.tokens() as i64).sum();
        if total > i32::MAX as i64 {
            return Err(Error::Msg(format!(
                "mage_flow: packed image sequence of {total} tokens exceeds the i32 index range"
            )));
        }
        let mut ids = Vec::with_capacity(total as usize * 3);
        for (idx, shape) in shapes.iter().enumerate() {
            shape.validate()?;
            let idx = i32::try_from(idx)
                .map_err(|_| Error::Msg("mage_flow: too many packed image segments".to_string()))?;
            self.check_bounds(idx, shape)?;
            // Centred height/width when `scale_rope`: row `y` carries coordinate `y - (L - L/2)`,
            // exactly `cat(neg[-(L - L/2):], pos[:L/2])` (`mage_layers.py:194-203`).
            let (h_off, w_off) = if self.scale_rope {
                (
                    shape.height - shape.height / 2,
                    shape.width - shape.width / 2,
                )
            } else {
                (0, 0)
            };
            for f in 0..shape.frames {
                // `freqs_pos[0][idx : idx + frame]` — the frame axis starts at the segment's own
                // position in `img_shapes`, NOT at 0 (`mage_layers.py:192`).
                let frame_pos = (idx + f) as f32;
                for y in 0..shape.height {
                    for x in 0..shape.width {
                        ids.push(frame_pos);
                        ids.push((y - h_off) as f32);
                        ids.push((x - w_off) as f32);
                    }
                }
            }
        }
        let ids = Array::from_slice(&ids, &[total as i32, 3]);
        let (cos, sin) = nn::rope_sincos_from_ids(&ids, &self.axes_dim, self.theta)?;
        Ok(RopeTable { cos, sin })
    }

    /// The reference's table size is a real capacity limit: an out-of-range frame index or an
    /// oversized spatial extent yields a short slice and then a `view` shape error inside
    /// `_compute_video_freqs`. Surfaced here as a typed error naming the offending axis instead.
    fn check_bounds(&self, idx: i32, shape: &ImgShape) -> Result<()> {
        if idx + shape.frames > self.table_len {
            return Err(Error::Msg(format!(
                "mage_flow: msrope frame index {} exceeds the {}-entry table (segment {idx}, \
                 {} frames)",
                idx + shape.frames - 1,
                self.table_len,
                shape.frames
            )));
        }
        for (axis, extent) in [("height", shape.height), ("width", shape.width)] {
            // `neg[-(L - L/2):]` and `pos[:L/2]` each index at most `table_len` rows.
            let need = if self.scale_rope {
                (extent - extent / 2).max(extent / 2)
            } else {
                extent
            };
            if need > self.table_len {
                return Err(Error::Msg(format!(
                    "mage_flow: msrope {axis} extent {extent} needs {need} table entries but the \
                     table holds {}",
                    self.table_len
                )));
            }
        }
        Ok(())
    }
}

/// Everything a block needs to know about the pack, built once per forward: the layout, the msrope
/// table, and the per-token segment indices that expand `[segments, dim]` adaLN modulation to
/// `[tokens, dim]` (the reference's `repeat_interleave`, `mage_layers.py:566-568`).
///
/// Built once by [`crate::transformer::MageTransformer::forward`] and shared by all twelve blocks
/// and the output head, so the host-side index build and the trig table are paid once, not 12×.
#[derive(Debug, Clone)]
pub struct PackContext {
    layout: PackLayout,
    rope: RopeTable,
    img_segment_ids: Array,
    txt_segment_ids: Array,
    img_cu: Vec<i32>,
    txt_cu: Vec<i32>,
}

impl PackContext {
    pub fn new(layout: PackLayout, rope: &MsRope) -> Result<Self> {
        let table = rope.forward(layout.img_shapes())?;
        Self::with_rope_table(layout, table)
    }

    /// Build a context around an **externally supplied** rope table — the seam the block-level
    /// parity test uses to feed the golden's own `image_rotary_emb` straight in, isolating the
    /// block from the msrope port.
    pub fn with_rope_table(layout: PackLayout, rope: RopeTable) -> Result<Self> {
        let img_tokens = layout.img_tokens();
        let rope_rows = rope.cos.shape()[0];
        if rope_rows != img_tokens {
            return Err(Error::Msg(format!(
                "mage_flow: rope table has {rope_rows} rows but the pack carries {img_tokens} \
                 image tokens"
            )));
        }
        let img_segment_ids = segment_ids(layout.img_lens());
        let txt_segment_ids = segment_ids(layout.txt_lens());
        let img_cu = layout.img_cu();
        let txt_cu = layout.txt_cu();
        Ok(Self {
            layout,
            rope,
            img_segment_ids,
            txt_segment_ids,
            img_cu,
            txt_cu,
        })
    }

    pub fn layout(&self) -> &PackLayout {
        &self.layout
    }

    pub fn rope(&self) -> &RopeTable {
        &self.rope
    }

    pub fn segments(&self) -> usize {
        self.layout.segments()
    }

    /// `[img_tokens]` int32 — the segment each image token belongs to.
    pub fn img_segment_ids(&self) -> &Array {
        &self.img_segment_ids
    }

    /// `[txt_tokens]` int32 — the segment each text token belongs to.
    pub fn txt_segment_ids(&self) -> &Array {
        &self.txt_segment_ids
    }

    /// Interior image segment boundaries suitable for `split_sections` (`cu[1..segments]`).
    pub fn img_split_points(&self) -> &[i32] {
        &self.img_cu[1..self.img_cu.len() - 1]
    }

    /// Interior text segment boundaries suitable for `split_sections` (`cu[1..segments]`).
    pub fn txt_split_points(&self) -> &[i32] {
        &self.txt_cu[1..self.txt_cu.len() - 1]
    }

    pub fn img_cu(&self) -> &[i32] {
        &self.img_cu
    }

    pub fn txt_cu(&self) -> &[i32] {
        &self.txt_cu
    }
}

fn segment_ids(lens: &[i32]) -> Array {
    let total: i32 = lens.iter().sum();
    let mut ids = Vec::with_capacity(total as usize);
    for (segment, &len) in lens.iter().enumerate() {
        ids.extend(std::iter::repeat_n(segment as i32, len as usize));
    }
    Array::from_slice(&ids, &[total])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_cfg_puts_the_uncond_duplicate_at_frame_index_one() {
        // `pipeline.py:167` — the segment LIST is concatenated, so the uncond copy is segment 1.
        let cond = PackLayout::generation(vec![ImgShape::latent(16, 16)], vec![20]).unwrap();
        let layout = cond.fused_cfg(&[6]).unwrap();
        assert_eq!(layout.segments(), 2);
        assert_eq!(layout.img_cu(), vec![0, 256, 512]);
        assert_eq!(layout.txt_cu(), vec![0, 20, 26]);

        let rope = MsRope::new(&[16, 56, 56], 10_000.0, true, 4096).unwrap();
        let table = rope.forward(layout.img_shapes()).unwrap();
        let cos = table.cos.as_slice::<f32>();
        let sin = table.sin.as_slice::<f32>();
        let half = 64usize;
        // Frame lanes are the first axes_dim[0]/2 = 8 columns. Segment 0 rotates at frame 0 ⇒
        // exactly the identity (1, 0).
        for k in 0..8 {
            assert_eq!(cos[k], 1.0, "cond frame lane {k} must be cos(0)");
            assert_eq!(sin[k], 0.0, "cond frame lane {k} must be sin(0)");
        }
        // Segment 1 rotates at frame 1 ⇒ e^{i·1·10000^(−2k/16)}.
        let row = 256 * half;
        for k in 0..8 {
            let w = 1.0f32 / 10_000f32.powf((2 * k) as f32 / 16.0);
            assert!((cos[row + k] - w.cos()).abs() < 1e-6, "uncond cos lane {k}");
            assert!((sin[row + k] - w.sin()).abs() < 1e-6, "uncond sin lane {k}");
        }
        // ...and the spatial lanes are untouched by the duplication.
        for k in 8..half {
            assert_eq!(cos[k], cos[row + k], "spatial lane {k} must not move");
            assert_eq!(sin[k], sin[row + k], "spatial lane {k} must not move");
        }
    }

    #[test]
    fn scale_rope_centres_the_spatial_axes() {
        let rope = MsRope::new(&[16, 56, 56], 10_000.0, true, 4096).unwrap();
        let table = rope.forward(&[ImgShape::latent(4, 4)]).unwrap();
        let sin = table.sin.as_slice::<f32>();
        let half = 64usize;
        // h = 4 ⇒ offset 4 − 2 = 2 ⇒ row 0 carries coordinate −2, row 2 carries 0. Height lanes
        // start at column 8 and the first of them has ω = 1.
        assert!((sin[8] - (-2.0f32).sin()).abs() < 1e-6);
        assert!(sin[half * 8 + 8].abs() < 1e-6, "latent row 2 is the origin");
        // Un-centred is a *different* table — the discrimination this flag exists for.
        let plain = MsRope::new(&[16, 56, 56], 10_000.0, false, 4096).unwrap();
        let plain = plain.forward(&[ImgShape::latent(4, 4)]).unwrap();
        assert!(plain.sin.as_slice::<f32>()[8].abs() < 1e-9, "row 0 is 0");
    }

    #[test]
    fn odd_extents_centre_the_way_the_reference_slices() {
        // L = 5 ⇒ L/2 = 2, L − L/2 = 3 ⇒ coordinates −3 … 1.
        let rope = MsRope::new(&[16, 56, 56], 10_000.0, true, 4096).unwrap();
        let table = rope.forward(&[ImgShape::new(1, 1, 5)]).unwrap();
        let sin = table.sin.as_slice::<f32>();
        let half = 64usize;
        let width_lane = 8 + 28; // frame(8) + height(28)
        for (x, want) in [-3.0f32, -2.0, -1.0, 0.0, 1.0].into_iter().enumerate() {
            assert!(
                (sin[x * half + width_lane] - want.sin()).abs() < 1e-6,
                "width column {x}"
            );
        }
    }

    #[test]
    fn out_of_range_geometry_is_a_typed_error_not_a_panic() {
        let rope = MsRope::new(&[16, 56, 56], 10_000.0, true, 8).unwrap();
        // 8-entry table: 9 segments overflow the frame axis...
        let many: Vec<ImgShape> = (0..9).map(|_| ImgShape::latent(2, 2)).collect();
        assert!(rope.forward(&many).is_err());
        // ...and a 17-tall latent needs 9 negative rows.
        assert!(rope.forward(&[ImgShape::latent(17, 2)]).is_err());
        // 16 is the largest that fits (8 negative + 8 positive).
        assert!(rope.forward(&[ImgShape::latent(16, 2)]).is_ok());
    }

    #[test]
    fn generation_derives_one_attention_segment_per_shape() {
        let layout = PackLayout::generation(
            vec![ImgShape::latent(2, 3), ImgShape::latent(4, 4)],
            vec![7, 5],
        )
        .unwrap();
        assert_eq!(layout.img_lens(), &[6, 16]);
        assert_eq!(layout.img_cu(), vec![0, 6, 22]);
        assert_eq!(layout.txt_cu(), vec![0, 7, 12]);
        assert_eq!(layout.img_tokens(), 22);
        assert_eq!(layout.txt_tokens(), 12);
        assert!(PackLayout::generation(vec![ImgShape::latent(2, 2)], vec![]).is_err());
        assert!(PackLayout::generation(vec![], vec![]).is_err());
        assert!(PackLayout::generation(vec![ImgShape::latent(2, 2)], vec![0]).is_err());
    }

    /// The edit packing: `[target, ref]` is ONE attention segment carrying TWO msrope shapes
    /// (`pipeline.py:517-519`, `:531`), so the frame axis separates them inside one window.
    #[test]
    fn one_attention_segment_may_span_several_msrope_shapes() {
        let target = ImgShape::latent(4, 4);
        let layout = PackLayout::new(vec![target, target], vec![32], vec![9]).unwrap();
        assert_eq!(layout.segments(), 1, "one sample ⇒ one attention window");
        assert_eq!(layout.img_shapes().len(), 2, "target + one reference");
        assert_eq!(layout.img_cu(), vec![0, 32]);

        let rope = MsRope::new(&[16, 56, 56], 10_000.0, true, 4096).unwrap();
        let table = rope.forward(layout.img_shapes()).unwrap();
        let (cos, half) = (table.cos.as_slice::<f32>(), 64usize);
        // The reference tokens (rows 16..) rotate at frame 1 while the target rotates at frame 0 —
        // INSIDE one attention window, so the difference is not a common offset and does not cancel.
        assert_eq!(cos[0], 1.0);
        assert!((cos[16 * half] - 1.0f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn an_attention_boundary_off_a_shape_boundary_is_rejected() {
        let shape = ImgShape::latent(4, 4); // 16 tokens
                                            // 24 + 8 splits the second grid in half: its msrope coordinates would describe neither
                                            // window, so this must not load.
        assert!(PackLayout::new(vec![shape, shape], vec![24, 8], vec![3, 3]).is_err());
        assert!(PackLayout::new(vec![shape, shape], vec![16, 16], vec![3, 3]).is_ok());
        // Totals must agree too.
        assert!(PackLayout::new(vec![shape], vec![32], vec![3]).is_err());
    }

    /// Fused CFG offsets every shape in the second copy by the SAME amount, so the frame
    /// differences inside each attention window are unchanged — which is why the shift is inert in
    /// exact arithmetic even though the table moves.
    #[test]
    fn fused_cfg_offsets_the_second_copy_uniformly() {
        let target = ImgShape::latent(2, 2);
        let cond = PackLayout::new(vec![target, target], vec![8], vec![4]).unwrap();
        let fused = cond.fused_cfg(&[2]).unwrap();
        assert_eq!(fused.segments(), 2);
        assert_eq!(fused.img_shapes().len(), 4);
        assert_eq!(fused.img_cu(), vec![0, 8, 16]);
        assert_eq!(fused.txt_cu(), vec![0, 4, 6]);
        // Shapes 0,1 → frames 0,1 (cond window); shapes 2,3 → frames 2,3 (uncond window).
        // Within each window the frame DIFFERENCE is 1 either way.
        assert!(cond.fused_cfg(&[2, 2]).is_err());
    }

    #[test]
    fn segment_ids_expand_like_repeat_interleave() {
        let layout = PackLayout::generation(
            vec![ImgShape::latent(1, 2), ImgShape::latent(1, 3)],
            vec![2, 1],
        )
        .unwrap();
        let rope = MsRope::new(&[16, 56, 56], 10_000.0, true, 4096).unwrap();
        let ctx = PackContext::new(layout, &rope).unwrap();
        assert_eq!(ctx.img_segment_ids().as_slice::<i32>(), &[0, 0, 1, 1, 1]);
        assert_eq!(ctx.txt_segment_ids().as_slice::<i32>(), &[0, 0, 1]);
        assert_eq!(ctx.img_cu(), &[0, 2, 5]);
        assert_eq!(ctx.txt_cu(), &[0, 2, 3]);
        assert_eq!(ctx.img_split_points(), &[2]);
        assert_eq!(ctx.txt_split_points(), &[2]);
    }

    #[test]
    fn rope_table_row_count_must_match_the_pack() {
        let layout = PackLayout::generation(vec![ImgShape::latent(2, 2)], vec![3]).unwrap();
        let rope = MsRope::new(&[16, 56, 56], 10_000.0, true, 4096).unwrap();
        let wrong = rope.forward(&[ImgShape::latent(4, 4)]).unwrap();
        assert!(PackContext::with_rope_table(layout, wrong).is_err());
    }

    #[test]
    fn from_config_uses_the_code_hardcoded_theta_not_the_json_key() {
        let rope = MsRope::from_config(&MageFlowConfig::mage_flow()).unwrap();
        assert_eq!(rope.axes_dim(), &[16, 56, 56]);
        assert_eq!(rope.half_dim(), 64);
        assert_eq!(rope.theta(), ROPE_THETA);
        assert!(rope.scale_rope());
        assert_eq!(rope.table_len(), ROPE_TABLE_LEN);
    }
}
