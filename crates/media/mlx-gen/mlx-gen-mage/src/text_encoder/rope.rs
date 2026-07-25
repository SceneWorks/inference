//! Qwen3-VL **interleaved M-RoPE** — the 3-axis `(t, h, w)` rotary embedding the LM applies to
//! every decoder layer's q/k.
//!
//! Port of `transformers`' `Qwen3VLTextRotaryEmbedding` (`rope_scaling.mrope_interleaved: true`,
//! `mrope_section` [`QwenVlTextConfig::mrope_section`](crate::config::QwenVlTextConfig) = `[24, 20,
//! 20]`), driven from the position ids the reference builds in
//! `_vendor/mage_flow/models/modules/text_encoder.py:500-508`.
//!
//! ## The interleave rule
//!
//! Frequencies are **not** laid out in three contiguous chunks `[TTT…HHH…WWW]`. For each of the
//! `head_dim / 2` frequency indices `j`, the axis is chosen by `j % 3`, bounded by the section
//! width times three:
//!
//! - `j < 3·section[1]` and `j % 3 == 1` → the **height** position,
//! - `j < 3·section[2]` and `j % 3 == 2` → the **width** position,
//! - otherwise → the **temporal/text** position.
//!
//! With `[24, 20, 20]` over 64 frequencies that is `H` at `j ∈ {1, 4, …, 58}` (20 slots), `W` at
//! `j ∈ {2, 5, …, 59}` (20 slots) and `T` everywhere else — `{0, 3, …, 57}` plus the tail
//! `{60, 61, 62, 63}`, 24 slots. The angles are then written to **both halves** of the head dim
//! (`emb = cat(freqs, freqs)`), which is what makes the half-split
//! [`apply_text_rope`](mlx_gen::nn::apply_text_rope) apply the same angle to each rotated pair.
//!
//! ## Why the generation path still looks one-dimensional
//!
//! The reference's [`TextEncoder`](crate::text_encoder::MageTextEncoder) counterpart always builds
//! its own **flat** position ids — `torch.arange(length)` per packed segment
//! (`text_encoder.py:501-504`) — and the patched model expands that single row across all three
//! axes (`text_encoder.py:243-246`). So for Mage-Flow's conditioning encode, *both* the generation
//! and the edit path feed `t == h == w` and the interleave collapses to plain 1-D RoPE. That is a
//! property of the caller, **not** of the module: the axis split is real and is what the vision
//! path (sc-14048) will vary. [`MRopePositions::text`] builds the degenerate form;
//! `positions_are_equal_across_axes` in the tests pins the collapse, and
//! `interleave_axis_matches_the_reference_slices` pins the split itself so the two claims cannot
//! quietly become one.

use mlx_rs::{Array, Dtype};

use mlx_gen::{Error, Result};

/// Per-token 3-axis M-RoPE positions. One entry per token in the (packed) sequence.
///
/// For a text-only encode all three vectors are equal — see the module docs. `sc-14048` produces a
/// genuinely 3-D layout by giving an image block its merged `(row, col)` grid on `h`/`w` while `t`
/// holds the block's start offset.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MRopePositions {
    /// Temporal / text axis.
    pub t: Vec<i32>,
    /// Height axis.
    pub h: Vec<i32>,
    /// Width axis.
    pub w: Vec<i32>,
}

impl MRopePositions {
    /// Build positions with genuinely independent temporal, height, and width axes.
    pub fn from_axes(t: Vec<i32>, h: Vec<i32>, w: Vec<i32>) -> Result<Self> {
        if t.len() != h.len() || t.len() != w.len() {
            return Err(Error::Msg(format!(
                "mage_flow text M-RoPE axes have different lengths: t={} h={} w={}",
                t.len(),
                h.len(),
                w.len()
            )));
        }
        Ok(Self { t, h, w })
    }

    /// The text-only layout for ONE segment of `len` tokens: `0 … len-1` on all three axes.
    ///
    /// This is `torch.arange(length)` from `text_encoder.py:503`, expanded across the three axes by
    /// `text_encoder.py:244`. Packed encodes call this per segment, so **each segment restarts at
    /// 0** rather than continuing the packed offset — that restart is what makes a packed forward
    /// equal a per-segment one.
    pub fn text(len: usize) -> Self {
        let ramp: Vec<i32> = (0..len as i32).collect();
        Self {
            t: ramp.clone(),
            h: ramp.clone(),
            w: ramp,
        }
    }

    /// Token count. All three axes are the same length by construction.
    pub fn len(&self) -> usize {
        self.t.len()
    }

    /// `true` when there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    pub fn axes(&self) -> (&[i32], &[i32], &[i32]) {
        (&self.t, &self.h, &self.w)
    }
}

/// Which axis frequency index `j` reads: `0` = temporal/text, `1` = height, `2` = width.
///
/// The executable form of `apply_interleaved_mrope`'s two `slice(offset, 3·section[dim], 3)`
/// overwrites. Exposed (crate-internal) so the unit test can pin the mapping index-by-index
/// instead of only observing its effect through a cosine.
pub(crate) fn interleaved_axis(j: usize, section: [i32; 3]) -> usize {
    if j < (section[1] * 3) as usize && j % 3 == 1 {
        1
    } else if j < (section[2] * 3) as usize && j % 3 == 2 {
        2
    } else {
        0
    }
}

/// Build the interleaved-M-RoPE `cos`/`sin` pair, each `[1, seq, head_dim]` in `dtype`.
///
/// `angle(i, j) = pos_axis(j)[i] · theta^(-2j / head_dim)`, written to slots `j` and `half + j`
/// (`emb = cat(freqs, freqs)`), then `cos`/`sin`. Computed on the host in `f64` for the inverse
/// frequencies (`theta` is 5e6, so `theta^(-2j/128)` underflows f32 precision well before it
/// underflows range) and `f32` for the angles, matching the reference's `torch.autocast(enabled=
/// False)` float32 block before its cast down to the activation dtype.
pub fn mrope_cos_sin(
    pos: &MRopePositions,
    head_dim: i32,
    theta: f64,
    section: [i32; 3],
    dtype: Dtype,
) -> Result<(Array, Array)> {
    let s = pos.len();
    let hd = head_dim as usize;
    let half = hd / 2;

    let inv: Vec<f32> = (0..half)
        .map(|j| theta.powf(-(2.0 * j as f64) / head_dim as f64) as f32)
        .collect();

    let mut emb = vec![0f32; s * hd];
    for i in 0..s {
        for (j, &inv_freq) in inv.iter().enumerate() {
            let p = match interleaved_axis(j, section) {
                1 => pos.h[i],
                2 => pos.w[i],
                _ => pos.t[i],
            };
            let angle = p as f32 * inv_freq;
            emb[i * hd + j] = angle;
            emb[i * hd + half + j] = angle;
        }
    }

    let arr = Array::from_slice(&emb, &[1, s as i32, head_dim]);
    Ok((arr.cos()?.as_dtype(dtype)?, arr.sin()?.as_dtype(dtype)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{QwenVlTextConfig, TE_ROPE_THETA};

    /// The interleave is `j % 3` gated by `3 · section[dim]`, NOT three contiguous chunks. Pinned
    /// index-by-index against `apply_interleaved_mrope`'s `slice(offset, 3·section[dim], 3)`
    /// semantics, and the per-axis slot counts must reproduce `mrope_section` exactly.
    #[test]
    fn interleave_axis_matches_the_reference_slices() {
        let section = QwenVlTextConfig::mage_flow().mrope_section;
        assert_eq!(section, [24, 20, 20]);
        let half = (QwenVlTextConfig::mage_flow().head_dim / 2) as usize;
        assert_eq!(half, 64);

        // The reference's two overwrites, transcribed as sets.
        let h_slots: Vec<usize> = (1..(section[1] * 3) as usize).step_by(3).collect();
        let w_slots: Vec<usize> = (2..(section[2] * 3) as usize).step_by(3).collect();
        assert_eq!(h_slots.len(), section[1] as usize);
        assert_eq!(w_slots.len(), section[2] as usize);
        assert_eq!(*h_slots.last().unwrap(), 58);
        assert_eq!(*w_slots.last().unwrap(), 59);

        let mut counts = [0usize; 3];
        for j in 0..half {
            let axis = interleaved_axis(j, section);
            let want = if h_slots.contains(&j) {
                1
            } else if w_slots.contains(&j) {
                2
            } else {
                0
            };
            assert_eq!(axis, want, "frequency {j} routed to the wrong M-RoPE axis");
            counts[axis] += 1;
        }
        assert_eq!(
            counts,
            [
                section[0] as usize,
                section[1] as usize,
                section[2] as usize
            ],
            "per-axis slot counts must equal mrope_section"
        );
        // The tail past 3·20 is temporal — that is where the 24-4=20 vs 24 split comes from.
        for j in 60..half {
            assert_eq!(interleaved_axis(j, section), 0);
        }
    }

    /// A 3-D layout genuinely reaches different frequencies per axis: perturbing ONLY the height
    /// positions must change exactly the height slots and nothing else. Without this, an
    /// implementation that ignored `h`/`w` entirely would still pass every text-only check.
    #[test]
    fn height_positions_move_only_the_height_slots() {
        let cfg = QwenVlTextConfig::mage_flow();
        let base = MRopePositions {
            t: vec![0, 1],
            h: vec![0, 1],
            w: vec![0, 1],
        };
        let bumped = MRopePositions {
            h: vec![7, 9],
            ..base.clone()
        };
        let (c0, _) = mrope_cos_sin(
            &base,
            cfg.head_dim,
            TE_ROPE_THETA,
            cfg.mrope_section,
            Dtype::Float32,
        )
        .unwrap();
        let (c1, _) = mrope_cos_sin(
            &bumped,
            cfg.head_dim,
            TE_ROPE_THETA,
            cfg.mrope_section,
            Dtype::Float32,
        )
        .unwrap();
        let (a, b) = (c0.as_slice::<f32>(), c1.as_slice::<f32>());
        let hd = cfg.head_dim as usize;
        let half = hd / 2;
        let mut moved = 0usize;
        for token in 0..2usize {
            for j in 0..half {
                let differs = (a[token * hd + j] - b[token * hd + j]).abs() > 0.0;
                let is_h = interleaved_axis(j, cfg.mrope_section) == 1;
                // Token 0 at position 0 vs 7 differs; the check is that ONLY h slots may differ.
                assert!(!differs || is_h, "non-height slot {j} moved with h");
                // Both halves must move together (emb = cat(f, f)).
                assert_eq!(
                    differs,
                    (a[token * hd + half + j] - b[token * hd + half + j]).abs() > 0.0,
                    "the two head-dim halves disagree at slot {j}"
                );
                if differs {
                    moved += 1;
                }
            }
        }
        assert!(
            moved > 0,
            "changing h changed nothing — the axis is ignored"
        );
    }

    /// Text-only positions are equal on all three axes (`text_encoder.py:503` + `:244`), so the
    /// interleave collapses to plain 1-D RoPE. Pinned against `mlx_gen::nn::TextRope`, the shared
    /// half-split table every other Qwen3 encoder in this workspace uses.
    #[test]
    fn text_positions_collapse_to_plain_1d_rope() {
        let cfg = QwenVlTextConfig::mage_flow();
        let pos = MRopePositions::text(5);
        assert_eq!(pos.t, pos.h);
        assert_eq!(pos.t, pos.w);

        let (cos, sin) = mrope_cos_sin(
            &pos,
            cfg.head_dim,
            TE_ROPE_THETA,
            cfg.mrope_section,
            Dtype::Float32,
        )
        .unwrap();
        let rope = mlx_gen::nn::TextRope::new(cfg.head_dim, TE_ROPE_THETA as f32);
        let (rcos, rsin) = rope.forward(5).unwrap();
        for (mine, theirs) in [(&cos, &rcos), (&sin, &rsin)] {
            assert_eq!(mine.shape(), theirs.shape());
            let (a, b) = (mine.as_slice::<f32>(), theirs.as_slice::<f32>());
            let max = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                max < 2e-5,
                "text-only M-RoPE diverges from 1-D RoPE: {max:e}"
            );
        }
    }
}
