//! Video-VAE decode **tiling** — the family-agnostic geometry layer shared by the LTX and Wan VAEs.
//!
//! Decoding a large/long latent in one pass is memory-bound; tiling splits it into overlapping
//! spatial/temporal tiles, decodes each independently, and trapezoidally blends the results. This
//! module is the **pure** half — tiling presets, the per-axis interval split, the 1-D blend mask,
//! and the full [`TilePlan`] for a latent. The Array blend loop (slice each tile, decode, weight,
//! pad-and-accumulate, normalize) lives in each crate's `vae.rs` so it can reach that VAE's decoder;
//! the reference allocates full-size `output`+`weights` accumulators and processes one tile at a
//! time, so the pad-and-accumulate form keeps the same bounded peak memory.
//!
//! Port of the `mlx_video` reference `models/ltx/video_vae/tiling.py` (the shared primitives) plus
//! `models/wan/tiling.py`'s `causal_temporal` generalization. The per-VAE upsample factors and the
//! causal-vs-non-causal temporal mapping are carried by [`VaeTiling`]:
//!  - **LTX** ([`VaeTiling::LTX`]): spatial ×32 (8× learned × 4× unpatchify), temporal ×8, **causal**
//!    (`out_f = 1 + (f−1)·8`).
//!  - **Wan 2.1** ([`VaeTiling::WAN`]): spatial ×8, temporal ×4, **non-causal** (`out_f = f·4`) — the
//!    temporal axis tiles exactly like a spatial axis.

/// The historical MLX over-bound write threshold, in elements: `i32::MAX`.
///
/// **This is operation- and runtime-specific, not a universal "any tensor over 2^31 is wrong" law**
/// (sc-12438 corrected the earlier universal framing). It stood in for several distinct 0.31.2 failures
/// sharing the 2^31-element (`i32`) threshold — but **sc-12748: on this repo's current pin (MLX 0.32.0
/// via `pmetal-mlx-rs eb76c4ba` + the sc-12746 pad/concat copy-gate patch) every operation the
/// tiled/untiled video decode actually uses above the bound is now int64-safe.** Re-probed on this
/// exact pin on 2026-08-11 at `128×D×480×848` (`D=41` →
/// 2,136,145,920 = 0.995×; `D=42` → 2,188,247,040 = 1.019×) with position-dependent data compared only
/// on sub-bound slices (`mlx-gen/tests/mlx_write_bound_probe.rs`):
///
/// | operation | on 0.31.2 | on this pin (0.32.0 fork `eb76c4ba` + patch) |
/// |---|---|---|
/// | `conv3d` 8→128, output over the bound | **wrong** from ~1.007× (per-thread output offset in the Metal steel-conv kernel overflows `int32`) | **EXACT** — MLX PR #3524 promoted those offsets to `size_t` (probe-verified, `first_bad=-1`) |
/// | `pad` to a full output over the bound (the tiled accumulator's placement op) | **wrong** from ~1.003× | **EXACT** — the sc-12746 `pad-copy-int64.patch` gates the copy on the addressable span (probe-verified) |
/// | `concatenate` over the bound (same `copy_gpu_inplace` path) | **wrong** | **EXACT** — same copy-gate patch (probe-verified) |
/// | reshape at over-bound size | **overflow** (flat `i32` shape-product) | **mixed** — a single `reshape(-1)` dimension still raises, while a multi-dimensional reshape whose individual dimensions fit `i32` is exact above the bound; `contiguous` uses that verified multi-dimensional path |
/// | `from_slice` (host `Vec` → over-bound `Array`) | **overflow** — mlx-rs asserts `len == shape.product::<i32>()` (MLX #3327) | **still i32-capped** — a fork-side mlx-rs bug, unfixed; the one residual (the decode paths read back via `as_slice`, never `from_slice` the full output) |
/// | reading back (`as_slice`) an over-bound array | correct | correct |
///
/// Consequences for the guard sites that reference this bound (all relaxed in **sc-12748**):
///  - **Decoder-stage write** ([`VaeTiling::writable_frame_cap`], used by [`budgeted_plan`]): the widest
///    full-resolution tensor a *single* decode/tile writes. Its failure was the `conv3d` row, now
///    **EXACT** on this pin (#3524), and the video decoders whose full-res write it bounds are conv-only
///    at that stage (LTX/Mochi are attention-free; Wan's VAE attention runs at latent resolution). So the
///    convolution-corruption justification is **retired**; the cap is kept as belt-and-suspenders
///    defense-in-depth (a cheap tiling trigger that yields correct output regardless, guarding against a
///    future MLX regression re-breaking conv or a future full-res op the probes don't cover).
///  - **Assembled-output write** (`mlx_gen::vae_tiling::check_output_writable`): the full
///    `output`/`weights` accumulators a tiled decode builds by `pad`-and-add and reads back via
///    `contiguous`'s multi-dimensional `reshape`+`as_slice` path. Every operation in that path is now
///    int64-safe (rows above), so the sc-12438
///    **refusal is lifted** — an over-bound assembled output now RENDERS. A single-dimension flatten
///    remains rejected but is not used by this path. `check_output_writable` is kept as a narrow
///    backstop for the remaining possible host-materialization hazard, a `from_slice` the decode never
///    does — a retained latent tripwire with **no production caller** today (sc-12926), ready to wire
///    if future code from_slices an output-scale host buffer.
///  - **Mochi's decode guard** (`mlx-gen-mochi`'s `decode_body`): its over-bound write is `block_out`'s
///    conv3d, now EXACT — the refusal is lifted; chunking is retained for **peak memory** only.
///
/// **Why it hid (0.31.2).** Reductions have tiny outputs and stay correct, so a checksum noticed nothing;
/// and any verification computed *over* an oversized tensor was itself an oversized write, so it silently
/// reported agreement. Detecting it needed **position-dependent** data and a comparison that never wrote
/// past the bound — the method the probes and the sc-12748 renders honor.
///
/// First found via Mochi's AsymmVAE decode (sc-12291), whose full-resolution `block_out` was exact
/// through 31 frames at 848×480 and returned ±2.67 — where a valid video is ~[-1, 1] — from 37 on (on
/// 0.31.2). This bound is per-backend: the CUDA/`candle-gen` path uses its own tensor library and does
/// **not** share the MLX `i32` write limit (only the MLX crates ever gated on it).
pub const MAX_WRITABLE_ELEMS: i64 = i32::MAX as i64;

/// A VAE's tiling parameters: the decoder's spatial/temporal upsample factors, whether its temporal
/// decode is causal (`out_f = 1 + (f−1)·scale`) or non-causal (`out_f = f·scale`), and the channel
/// width of its widest full-resolution stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaeTiling {
    pub spatial_scale: i32,
    pub temporal_scale: i32,
    pub causal_temporal: bool,
    /// Channels in the widest stage that runs at **full output resolution** — the width that sizes the
    /// largest tensor the decoder writes (`full_res_channels × out_f × out_h × out_w`). Keeps that
    /// write under [`MAX_WRITABLE_ELEMS`]; see [`VaeTiling::writable_frame_cap`].
    ///
    /// This is a *correctness* input, not a memory estimate — the memory estimator stays with each
    /// crate's `peak_cost`. Err high if unsure: over-stating it costs an unnecessary tile, understating
    /// it admits a silently-wrong decode.
    pub full_res_channels: i32,
}

impl VaeTiling {
    /// LTX-2 video VAE: spatial ×32 (8× upsample × 4× unpatchify), temporal ×8, causal.
    ///
    /// `full_res_channels: 8` — its 128-channel stage runs at H/4 and W/4 (the ×4 unpatchify is last),
    /// so per *output* voxel the decoder writes 128/16 = 8 channels' worth.
    pub const LTX: Self = Self {
        spatial_scale: 32,
        temporal_scale: 8,
        causal_temporal: true,
        full_res_channels: 8,
    };
    /// Wan 2.1 z16 VAE: spatial ×8, temporal ×4, non-causal (`T → T·4`).
    ///
    /// `full_res_channels: 96` — `dim × 1` at full resolution (the final res-blocks + `head_conv`
    /// input), before `head_conv` drops 96 → 3. The last `UpsampleBlock` does `upsample_nearest`
    /// (input-width) then `Conv2d(C→C/2)`, which *structurally* looks like a 192-ch full-res transient —
    /// but sc-12438 **measured the real z16 decode on the bf16 weights** and that transient does NOT
    /// materialize as a single over-bound write: a single-pass decode is exact vs the tiled reference
    /// through 92 frames at 512² (`96·85·512² = 2.1e9`, the 96-ch cap), diverging only into a
    /// decode-time *error* far past it — never the silent conv-style corruption a 192 write would show.
    /// So 96 (the materialized res-block width) is validated as the effective bound;
    /// `mlx-gen-wan/tests/vae16_write_bound_real.rs` is that evidence.
    pub const WAN: Self = Self {
        spatial_scale: 8,
        temporal_scale: 4,
        causal_temporal: false,
        full_res_channels: 96,
    };
    /// Wan 2.2 z48 `vae22` VAE: spatial ×16 (8× conv upsample × 2× unpatchify), temporal ×4,
    /// **causal** (`out_f = 1 + (f−1)·4` — the decoder runs `first_chunk=True`, so the leading
    /// temporal-padding frames are trimmed). The 5B's TI2V-5B VAE (sc-2680).
    ///
    /// `full_res_channels: 64` — 256 channels at H/2 (`dec_dim = 256`, ×2 unpatchify last) = 256/4.
    /// (sc-12438 kept this: the same MLX fusion that makes the z16 upsample transient non-materializing
    /// applies here, so the post-reduction materialized stage is the effective bound; validated by the
    /// z16 evidence above and the family shared decode structure.)
    pub const WAN22: Self = Self {
        spatial_scale: 16,
        temporal_scale: 4,
        causal_temporal: true,
        full_res_channels: 64,
    };
    /// Qwen-Image `AutoencoderKLQwenImage` VAE (sc-11747): a still-image VAE — spatial ×8, and the
    /// temporal axis is a **singleton** (T=1), so `temporal_scale: 1` non-causal makes the temporal
    /// axis a no-op (`out_f = f = 1`) while the spatial ×8 upsample drives the H/W tiling. Used by the
    /// Krea 2 pose-control decode (and reusable by the Qwen txt2img/edit lanes) to bound the
    /// end-of-generation decode spike on a 32 GB Mac.
    ///
    /// `full_res_channels: 96` — at T=1 and its 2048² cap this cannot approach
    /// [`MAX_WRITABLE_ELEMS`] (96 × 2048² = 4.0e8, 0.19×), so the cap never binds here. (sc-12438: the
    /// last `Resample3d` upsample looks like a 192-ch full-res transient, but per the measured z16
    /// evidence that MLX-fused transient does not materialize as a single over-bound write; the
    /// materialized 96-ch stage is the effective bound. Inert regardless at the shipped ≤2048² sizes.)
    pub const QWEN_IMAGE: Self = Self {
        spatial_scale: 8,
        temporal_scale: 1,
        causal_temporal: false,
        full_res_channels: 96,
    };

    /// The most **output frames** this VAE can decode in one pass at `out_h × out_w` while keeping its
    /// widest full-resolution write under [`MAX_WRITABLE_ELEMS`]. 0 when a single frame already
    /// exceeds it (only reachable at resolutions far beyond any shipped bucket).
    ///
    /// On MLX 0.31.2 exceeding this returned **wrong pixels, silently** (the widest write is a conv3d).
    /// sc-12748 recorded that on the current pin (0.32.0, #3524) that conv is int64-safe (see
    /// [`MAX_WRITABLE_ELEMS`]), and concluded this is no longer a correctness bound — retained as a
    /// defense-in-depth tiling trigger and a conservative single-pass ceiling.
    ///
    /// ⚠️ **That conclusion is contradicted by observation and by this module's own
    /// [`budgeted_plan`] doc, which still calls it "the correctness bound that memory cannot see".**
    /// During sc-8446 an 84-output-frame single-pass z16 decode at 832×480 — 1.5× over this cap —
    /// produced a washed-out result that diverges from every tiled decode of the same latents **at
    /// frame 0** (saturation 0.068 vs 0.329; the two agree at 0.293 when the same comparison is run at
    /// 36 frames, under the cap). A correct decode's first frame cannot depend on the clip length, so
    /// this is the silently-wrong-wide-write signature, not a tiling artifact.
    ///
    /// ⚠️ Scope of that evidence: it is **one over-cap and one under-cap point**. It establishes a
    /// *length-dependent* correctness failure; it does **not** establish that the threshold is exactly
    /// this cap. A bisection belongs in the story before this doc is rewritten rather than annotated.
    /// Tracked as **sc-15402** (which also links the prior sc-12438 / sc-12748 work); until it is
    /// settled, treat exceeding this cap as unsafe.
    pub fn writable_frame_cap(&self, out_h: i32, out_w: i32) -> i64 {
        let per_frame = self.full_res_channels as i64 * out_h as i64 * out_w as i64;
        if per_frame <= 0 {
            return i64::MAX;
        }
        MAX_WRITABLE_ELEMS / per_frame
    }
}

/// A backend-neutral, composable conservative VAE decode-phase memory profile.
///
/// `resident_decoder_bytes_included` is the portion of `working_set_bytes` substitutable by decoder
/// resident bytes already charged in a model contract. It is part of, not additional to, the working
/// set, and the checked constructor enforces `included <= working_set`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VideoDecodeMemoryProfile {
    /// Conservative decode-phase working set, including any resident decoder bytes identified below.
    working_set_bytes: u64,
    /// Portion of `working_set_bytes` substitutable by decoder bytes already present in a contract.
    resident_decoder_bytes_included: u64,
}

impl VideoDecodeMemoryProfile {
    /// Build a valid profile, returning `None` when the included resident portion exceeds the total.
    pub fn new(working_set_bytes: u64, resident_decoder_bytes_included: u64) -> Option<Self> {
        (resident_decoder_bytes_included <= working_set_bytes).then_some(Self {
            working_set_bytes,
            resident_decoder_bytes_included,
        })
    }

    /// Conservative decode-phase working set in bytes.
    pub const fn working_set_bytes(self) -> u64 {
        self.working_set_bytes
    }

    /// Portion of the working set substitutable by decoder bytes already charged in a contract.
    pub const fn resident_decoder_bytes_included(self) -> u64 {
        self.resident_decoder_bytes_included
    }

    /// Bytes to add to a composition that already includes `contract_decoder_bytes`.
    ///
    /// If the contract decoder is below the included floor, the uncovered portion remains. If it is
    /// above the floor, the contract decoder is charged once and only non-resident decode work is
    /// added. Arithmetic overflow returns `None`.
    pub fn incremental_above_contract_decoder_bytes(
        self,
        contract_decoder_bytes: u64,
    ) -> Option<u64> {
        self.working_set_bytes
            .checked_sub(contract_decoder_bytes.min(self.resident_decoder_bytes_included))
    }

    /// Add this profile to an already-accounted contract composition with checked arithmetic.
    /// Returns `None` when the declared decoder component exceeds the composition that contains it.
    pub fn checked_composed_peak(
        self,
        composition_bytes: u64,
        contract_decoder_bytes: u64,
    ) -> Option<u64> {
        if contract_decoder_bytes > composition_bytes {
            return None;
        }
        composition_bytes
            .checked_add(self.incremental_above_contract_decoder_bytes(contract_decoder_bytes)?)
    }
}

/// Per-frame spatial tiling (tile + overlap in **output pixels**).
#[derive(Clone, Copy, Debug)]
pub struct SpatialTiling {
    pub tile_px: i32,
    pub overlap_px: i32,
}

/// Temporal tiling (tile + overlap in **output frames**).
#[derive(Clone, Copy, Debug)]
pub struct TemporalTiling {
    pub tile_frames: i32,
    pub overlap_frames: i32,
}

/// Which axes to tile. `None` on either axis disables tiling there. Tile/overlap sizes are in
/// **output** units (pixels / frames) and convert to latent units by the VAE's scale.
#[derive(Clone, Copy, Debug, Default)]
pub struct TilingConfig {
    pub spatial: Option<SpatialTiling>,
    pub temporal: Option<TemporalTiling>,
}

impl TilingConfig {
    /// Reference default: 512 px / 64 px spatial, 64 / 24 frame temporal.
    pub fn default_preset() -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px: 512,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 64,
                overlap_frames: 24,
            }),
        }
    }

    /// Aggressive (smaller tiles, lowest memory): 256/64 px, 32/8 frame.
    pub fn aggressive() -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px: 256,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 32,
                overlap_frames: 8,
            }),
        }
    }

    /// Conservative (larger tiles, faster, less saving): 768/64 px, 96/24 frame.
    pub fn conservative() -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px: 768,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 96,
                overlap_frames: 24,
            }),
        }
    }

    pub fn spatial_only(tile_px: i32, overlap_px: i32) -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px,
                overlap_px,
            }),
            temporal: None,
        }
    }

    pub fn temporal_only(tile_frames: i32, overlap_frames: i32) -> Self {
        Self {
            spatial: None,
            temporal: Some(TemporalTiling {
                tile_frames,
                overlap_frames,
            }),
        }
    }

    /// Auto-select a config from **output** dimensions (reference `TilingConfig.auto`), or `None`
    /// when no tiling is needed. Thresholds (spatial > 512 px, temporal > 65 frames) are in output
    /// units, so this is VAE-scale-independent.
    pub fn auto(height: i32, width: i32, num_frames: i32) -> Option<Self> {
        let needs_spatial = height > 512 || width > 512;
        let needs_temporal = num_frames > 65;
        if !needs_spatial && !needs_temporal {
            return None;
        }
        let est_gb = (3.0 * num_frames as f64 * height as f64 * width as f64 * 4.0)
            / (1024.0 * 1024.0 * 1024.0);
        if est_gb > 2.0 || ((height as i64) * (width as i64) > 768 * 1024 && num_frames > 100) {
            return Some(Self::aggressive());
        }
        let spatial = needs_spatial.then(|| {
            // F-057: this `auto` heuristic is mlx-gen-original — the frozen reference `TilingConfig`
            // has no `auto` (it ships fixed `vae_encode_tile_size = 512`), so there is no parity
            // constraint to preserve. The old `>1024 → 384 / >768 → 512 / else → 384` was non-monotone
            // (a 700 px output got SMALLER tiles than a 1000 px one — a transposed threshold): larger
            // outputs must not get larger tiles. Monotone now: bound memory by shrinking the tile once
            // the output exceeds 1024 px, otherwise use the reference's 512.
            let max_dim = height.max(width);
            let tile_px = if max_dim > 1024 { 384 } else { 512 };
            SpatialTiling {
                tile_px,
                overlap_px: 64,
            }
        });
        let temporal = needs_temporal.then(|| {
            let (tile_frames, overlap_frames) = if num_frames > 200 {
                (32, 8)
            } else if num_frames > 100 {
                (48, 16)
            } else {
                (64, 24)
            };
            TemporalTiling {
                tile_frames,
                overlap_frames,
            }
        });
        Some(Self { spatial, temporal })
    }

    /// Whether tiling actually fires for a latent `[_, _, f, h, w]` under VAE `vae` (i.e. some axis
    /// exceeds its latent-space tile size).
    pub fn needs_tiling(&self, vae: VaeTiling, f: i32, h: i32, w: i32) -> bool {
        let s = self.spatial.is_some_and(|s| {
            let t = s.tile_px / vae.spatial_scale;
            h > t || w > t
        });
        let t = self
            .temporal
            .is_some_and(|tc| f > tc.tile_frames / vae.temporal_scale);
        s || t
    }

    /// Build the [`TilePlan`] for a latent of shape `[_, _, f, h, w]` under VAE `vae`.
    pub fn plan(&self, vae: VaeTiling, f: i32, h: i32, w: i32) -> TilePlan {
        let (t_tile, t_over) = match self.temporal {
            Some(tc) => (
                tc.tile_frames / vae.temporal_scale,
                tc.overlap_frames / vae.temporal_scale,
            ),
            None => (f, 0),
        };
        let (s_tile, s_over) = match self.spatial {
            Some(sc) => (
                sc.tile_px / vae.spatial_scale,
                sc.overlap_px / vae.spatial_scale,
            ),
            None => (h.max(w), 0),
        };
        TilePlan {
            t: temporal_tiles(t_tile, t_over, f, vae.temporal_scale, vae.causal_temporal),
            h: spatial_tiles(s_tile, s_over, h, vae.spatial_scale),
            w: spatial_tiles(s_tile, s_over, w, vae.spatial_scale),
            out_f: if vae.causal_temporal {
                1 + (f - 1) * vae.temporal_scale
            } else {
                f * vae.temporal_scale
            },
            out_h: h * vae.spatial_scale,
            out_w: w * vae.spatial_scale,
        }
    }
}

/// One tile along one axis: latent `[start, end)`, the output `[out_start, out_stop)` it maps to,
/// and the 1-D blend `mask` (length `out_stop − out_start`).
#[derive(Clone, Debug)]
pub struct AxisTile {
    pub start: i32,
    pub end: i32,
    pub out_start: i32,
    pub out_stop: i32,
    pub mask: Vec<f32>,
}

/// `compute_trapezoidal_mask_1d`: ones with a left fade-in (`ramp_left`) and right fade-out
/// (`ramp_right`). `left_from_0` chooses the linspace convention (temporal causal tiles fade from 0).
pub fn trapezoidal_mask(
    length: i32,
    ramp_left: i32,
    ramp_right: i32,
    left_from_0: bool,
) -> Vec<f32> {
    // Internal tiling invariant (tile extents are always positive); clamp keeps release builds safe
    // without an abort, debug_assert documents the contract (F-020/L-A).
    debug_assert!(length > 0, "mask length must be positive");
    let length = length.max(1) as usize;
    let ramp_left = ramp_left.clamp(0, length as i32) as usize;
    let ramp_right = ramp_right.clamp(0, length as i32) as usize;
    let mut mask = vec![1.0f32; length];

    if ramp_left > 0 {
        let interval = if left_from_0 {
            ramp_left + 1
        } else {
            ramp_left + 2
        };
        // linspace(0, 1, interval), drop last; if !left_from_0 also drop first.
        let full: Vec<f32> = (0..interval)
            .map(|i| i as f32 / (interval as f32 - 1.0))
            .collect();
        let fade_in: &[f32] = if left_from_0 {
            &full[..interval - 1]
        } else {
            &full[1..interval - 1]
        };
        for i in 0..ramp_left.min(fade_in.len()) {
            mask[i] *= fade_in[i];
        }
    }

    if ramp_right > 0 {
        // fade_out = linspace(1, 0, ramp_right+2)[1:-1] = (ramp_right+1-i)/(ramp_right+1), i=1..ramp_right
        for i in 0..ramp_right {
            let v = (ramp_right as f32 + 1.0 - (i as f32 + 1.0)) / (ramp_right as f32 + 1.0);
            mask[length - ramp_right + i] *= v;
        }
    }

    for v in &mut mask {
        *v = v.clamp(0.0, 1.0);
    }
    mask
}

/// Raw per-axis interval split (`split_in_spatial`): `(starts, ends, left_ramps, right_ramps)`.
fn split_spatial(size: i32, overlap: i32, dim: i32) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    // Guard degenerate configs (F-005): a caller-supplied tile ≤ overlap (reachable via
    // `TilingConfig::spatial_only`/`temporal_only`), or a tile floored to 0 by latent downscaling,
    // would divide by zero — or wrap `amount` to a huge `usize` (capacity panic) — below. Clamp to a
    // tile ≥ 1 and an overlap in `0..size`. For every valid config (`overlap < size`) this is a no-op.
    let size = size.max(1);
    let overlap = overlap.clamp(0, size - 1);
    if dim <= size {
        return (vec![0], vec![dim], vec![0], vec![0]);
    }
    let amount = (dim + size - 2 * overlap - 1) / (size - overlap);
    let starts: Vec<i32> = (0..amount).map(|i| i * (size - overlap)).collect();
    let mut ends: Vec<i32> = starts.iter().map(|s| s + size).collect();
    *ends.last_mut().unwrap() = dim;
    let mut left = vec![overlap; amount as usize];
    left[0] = 0;
    let mut right = vec![overlap; amount as usize];
    *right.last_mut().unwrap() = 0;
    (starts, ends, left, right)
}

/// `split_in_temporal`: spatial split, then `starts[1:] -= 1`, `left_ramps[1:] += 1` (causal).
fn split_temporal(size: i32, overlap: i32, dim: i32) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut starts, ends, mut left, right) = split_spatial(size, overlap, dim);
    for i in 1..starts.len() {
        starts[i] -= 1;
        left[i] += 1;
    }
    (starts, ends, left, right)
}

/// Build the spatial-axis tiles (`map_spatial_slice`: out = latent·scale, mask `left_from_0=false`).
fn spatial_tiles(tile_latent: i32, overlap_latent: i32, dim: i32, scale: i32) -> Vec<AxisTile> {
    let (starts, ends, left, right) = split_spatial(tile_latent, overlap_latent, dim);
    starts
        .iter()
        .enumerate()
        .map(|(i, &begin)| {
            let end = ends[i];
            let out_start = begin * scale;
            let out_stop = end * scale;
            let mask = trapezoidal_mask(
                out_stop - out_start,
                left[i] * scale,
                right[i] * scale,
                false,
            );
            AxisTile {
                start: begin,
                end,
                out_start,
                out_stop,
                mask,
            }
        })
        .collect()
}

/// Build the temporal-axis tiles. **Causal** (`out = 1+(latent−1)·scale`, `map_temporal_slice`,
/// `left_from_0`) for LTX; **non-causal** temporal tiles exactly like a spatial axis (`out =
/// latent·scale`) for Wan — the reference's `causal_temporal=False` path.
fn temporal_tiles(
    tile_latent: i32,
    overlap_latent: i32,
    dim: i32,
    scale: i32,
    causal: bool,
) -> Vec<AxisTile> {
    if !causal {
        return spatial_tiles(tile_latent, overlap_latent, dim, scale);
    }
    let (starts, ends, left, right) = split_temporal(tile_latent, overlap_latent, dim);
    starts
        .iter()
        .enumerate()
        .map(|(i, &begin)| {
            let end = ends[i];
            let out_start = begin * scale;
            let out_stop = 1 + (end - 1) * scale;
            let left_scaled = if left[i] > 0 {
                1 + (left[i] - 1) * scale
            } else {
                0
            };
            let mask = trapezoidal_mask(out_stop - out_start, left_scaled, right[i] * scale, true);
            AxisTile {
                start: begin,
                end,
                out_start,
                out_stop,
                mask,
            }
        })
        .collect()
}

/// The full tiling plan for a latent `[_, _, f, h, w]`: per-axis tile lists + the output dims.
pub struct TilePlan {
    pub t: Vec<AxisTile>,
    pub h: Vec<AxisTile>,
    pub w: Vec<AxisTile>,
    pub out_f: i32,
    pub out_h: i32,
    pub out_w: i32,
}

// --- Memory-budgeted tile selection (sc-6894) -----------------------------------------------------
//
// The geometry above answers "given a `TilingConfig`, what tiles?". This section answers the policy
// question one level up: "given a memory budget, *which* `TilingConfig`?". It is the backend- and
// VAE-neutral core of the budgeted selector first written for Wan's z48 vae22 decode (sc-4998) and
// lifted here so every video VAE (LTX, Wan z16/z48) and **both** backends (mlx-gen on Metal,
// candle-gen on CUDA) share one selector. The per-VAE/per-backend peak-cost constants and the budget
// source (e.g. the MLX memory limit) stay in the caller — this layer holds **zero** such knowledge,
// so it keeps gen-core's zero-tensor-dep / Linux-buildable invariant.

/// The **minimum temporal receptive field**, in *latent* frames, a temporal decode tile must span
/// (sc-15325).
///
/// ## Why this exists
///
/// Every video VAE decoder is a stack of temporal convolutions. A temporal tile is decoded
/// *independently* and then trapezoidally blended, so the decoder sees only the latent frames inside
/// the tile: a tile that is short in **latent** frames starves those convolutions of context, and the
/// blend cannot recover what was never computed. The failure is not a seam — it is per-tile
/// content-level corruption whose period is the tile *stride* (violet snow, rainbow chromatic
/// fringing, blown highlights).
///
/// ## What it is calibrated from
///
/// Measured on the real z16 Wan VAE (`temporal_scale = 4`) at 832×480 / 36 output frames, decoding the
/// **same** Krea Realtime latents single-pass and tiled (`mlx-gen-krea-realtime`'s
/// `decode_tiling_sweep_against_single_pass`). Mean |Δ| against the single-pass reference, in /255,
/// with highlight-clipping mean/worst:
///
/// | latent tile / overlap | mean abs err | clipping mean / worst |
/// |---|---|---|
/// | single-pass (reference) | 0 | 0.08% / 0.25% |
/// | 2 / 0 | 18.5 | 9.7% / 26.6% |
/// | 2 / 1 | 17.1 | 5.2% / 14.7% |
/// | 4 / 1 | 7.5 | 1.8% / 12.9% |
/// | 4 / 2 | 6.4 | 1.4% / 7.0% |
/// | **8 / 2** | **2.5** | **0.08% / 0.25%** |
/// | 8 / 4 | 2.0 | 0.14% / 1.1% |
///
/// Latent tile size is the dominant term (−67…−70 % per doubling at matched overlap); overlap is a
/// real but secondary one. **8 latent frames is where both metrics reach the single-pass floor** — the
/// clipping metric is indistinguishable from the reference there, which is why the floor is 8 and not
/// 4 (4 still blows out 12.9 % of a worst frame).
///
/// ## How it is enforced — and what is actually doing the work
///
/// [`budgeted_plan`] refuses any temporal candidate spanning fewer than this many latent frames, so a
/// too-short tile is never *selectable*.
///
/// ⚠️ **For the Wan z16/z48 grids this floor is, today, a no-op.** Floor-on vs floor-off selects an
/// identical tile in every cell of a 6-bucket × 4-frame-count × 5-budget sweep: the shipped z16
/// candidate grid bottoms out at a latent-8 tile anyway, and the largest-volume search reaches for it
/// first. **The change that actually fixed sc-15325 is that `mlx-gen-krea-realtime` and
/// `mlx-gen-scail2` now call this selector at all** — both previously sized a temporal-only window
/// from their own local px·frame budget and never saw these candidates. The floor is *prospective
/// insurance*: it makes the starved window unreachable by a future budget tweak, a new candidate
/// entry, or a cheaper cost model. Do not "simplify" the engine routing back to a locally-computed
/// window on the belief that the floor is what protects the picture — it is not; the routing is.
///
/// Where the floor is not a no-op is LTX, whose grid does carry latent-3 and latent-6 entries (below),
/// and any future family whose grid does the same.
///
/// ## ⚠️ The z16 collapse is not known to be universal — LTX was cleared EMPIRICALLY, not explained
///
/// This constant is deliberately VAE-neutral, but the evidence behind it is z16's. **LTX was measured
/// separately (`mlx-gen-ltx`'s `ltx_tiled_decode_tracks_single_pass`) and did not reproduce the
/// defect** at the bucket that was measured: at a latent tile of **3** it is 1.73/255 from single-pass
/// with **0.00 %** highlight clipping, where z16 at latent 2 is 24.4/255 with 30.8 % worst-frame
/// clipping.
///
/// **The mechanism is unexplained, and the obvious candidate explanation is wrong.** It is tempting to
/// credit LTX's *causal* temporal tiling (each tile is handed a preceding context frame). That cannot
/// be the reason: `causal_temporal` is also `true` for [`VaeTiling::WAN22`] (z48, shipping), so a
/// causal-VAE explanation would predict z48 is equally immune — which nobody has measured. Worse,
/// z16's `4/2` row (6.4/255) has the *same effective context* as an LTX interior tile (4 latent
/// frames, 1 overlap + 1 causal), yet LTX reads 1.73/255 there: a **3.7× gap at matched context** that
/// no context-counting argument explains. Something else — decoder depth, channel width, the ×8 vs ×4
/// temporal scale, the latent distribution itself — is responsible, and it has not been identified.
///
/// The verdict is also narrower than "LTX is fine". It was cleared at **one** bucket (640×384 × 89
/// frames, q8, a smooth source) which **never tiles in production**: a single pass there peaks
/// ~10 GiB, so this selector returns `Ok(None)` at any realistic budget and the test has to force the
/// config by hand. That source's channel amplitudes (0.70/0.70/0.60 on [−1, 1]) also never approach
/// the 0.9608 clip threshold, so LTX's "0.00 % clipping" is partly *structural* rather than purely a
/// property of the decode. Read the LTX result as: **cleared empirically at one low-dynamic-range
/// bucket, mechanism unknown — and post-fix it cannot reach the starved candidates anyway.**
///
/// The floor is applied to LTX on a **cost/benefit** argument rather than a correctness one: latent
/// 3 → 8 is 6.6× less error (1.73 → 0.26 /255) for 17 % more decode peak (7.34 → 8.57 GiB). Do not
/// read a red `budgeted_never_selects_a_starved_temporal_tile` on a *new* VAE family as proof that
/// family is corrupt — measure it the way LTX was measured before concluding anything, and if a family
/// turns out to tolerate short tiles *and* the memory matters there, that is a legitimate reason to
/// make the floor per-VAE. Equally, do not treat "LTX was fine" as evidence that some *other* causal
/// VAE is safe: that is exactly the inference shown above not to hold.
///
/// ## Why this is affordable — the spatial axis is nearly free, the temporal axis is not
///
/// A latent-8 temporal tile costs 76 GiB full-frame at 832×480, which is why the earlier analysis
/// treated it as a memory trade. It is not: the receptive-field problem is **temporal**, and the
/// memory can be bought back **spatially** at almost no quality cost. Same VAE, same clip, same latent
/// tile 8 / overlap 4, varying only the spatial tile — measured on **real generated Krea latents** by
/// `mlx-gen-krea-realtime`'s `decode_tiling_sweep_against_single_pass` (832×480, 36 output frames):
///
/// | spatial tile | latent px per tile | mean abs err vs single-pass | per-**column** mean abs err | clipping mean / worst | MLX active peak |
/// |---|---|---|---|---|---|
/// | none (full frame) | 104 × 60 | 1.954 /255 | 1.629 … 2.440 | 0.14 % / 1.05 % | 75.77 GiB |
/// | 448 px | 56 | 2.021 | 1.696 … 2.505 | 0.14 % / 1.04 % | 38.57 GiB |
/// | 320 px | 40 | 2.054 | 1.719 … 2.531 | 0.13 % / 1.04 % | 20.08 GiB |
/// | 256 px | 32 | 2.075 | 1.735 … 2.579 | 0.13 % / 1.04 % | 12.99 GiB |
/// | 192 px | 24 | 2.121 | 1.769 … 2.630 | 0.13 % / 1.02 % | 7.49 GiB |
///
/// **A 10.1× memory reduction costs 0.17/255**, and highlight clipping is flat-to-improving on every
/// row. The two axes are not comparable: halving the latent *temporal* tile costs 67-70 % more error,
/// while shrinking the spatial tile 4.3× costs 8 %. At ×8 spatial scale even the smallest 192 px
/// candidate is 24 latent px per tile with an 8-latent-px overlap — an order of magnitude more context
/// per axis than the 2 latent frames the pre-fix temporal window allowed.
///
/// The **per-column** column is what rules out a seam, and it is why this table is measured on real
/// latents rather than on a band-limited synthetic source. (It previously cited a smooth-sinusoid
/// stimulus, which structurally cannot show either a spatial seam or a starved spatial receptive
/// field — it had essentially no energy above DC. The conclusion survived the change of stimulus; the
/// evidence had to.) A whole-frame mean averages a tile boundary away; the per-column mean does not.
/// Across the whole sweep the per-column error just *shifts* — the floor rises 1.629 → 1.769 as the
/// ceiling rises 2.440 → 2.630 — with **no spike at the tile stride**: the 192 px row's worst column
/// is 1.24× its own clip mean. Spatial tiling costs a little error everywhere, not a lot in a few
/// columns.
pub const MIN_TEMPORAL_TILE_LATENT_FRAMES: i32 = 8;

/// The minimum temporal **overlap**, in *latent* frames, stamped onto a selected temporal tile
/// (sc-15325). Overlap is the secondary term above, and it is **free in peak**: the tiled decode
/// `eval`s per tile, so the peak is one tile's transient plus the output accumulators and the overlap
/// enters neither — a shorter stride is more passes (wall time), not a bigger graph.
///
/// ## ⚠️ This raise is a separate change from the tile-size fix — here is its bound
///
/// sc-15325's defect was tile *size*. Raising the overlap is a distinct, **beneficial but
/// independently-motivated** change. [`temporal_overlap_for`] raises the shared quality-oriented
/// policy's candidates to half a latent tile:
/// `(96, 24) → (96, 48)` (stride 72 → 48), `(64, 24) → (64, 32)`, `(48, 16) → (48, 24)`,
/// `(32, 8) → (32, 16)`.
///
///  * **Memory and plan selection are provably unchanged.** Overlap enters neither `peak_cost` (whose
///    tile term is `tile_f · tile_h · tile_w`) nor [`budgeted_plan`]'s selection key (`voxels`, the
///    same product). It is stamped onto the winner *after* the search, so it cannot change which
///    candidate wins or what that candidate peaks at.
///  * **Wall time only.** More overlap is a shorter stride, i.e. more decode passes over the same
///    output. Worst case is **1.5× the temporal decode passes** (a stride cut from 72 → 48 on a clip
///    long enough to be dominated by interior tiles); at a realistic 81-frame clip it is **~1.25×**.
///    That is a fraction of one decode, not of a generation.
///
/// On **accuracy** the raise is unambiguously positive: at latent tile 8, going 8/2 → 8/4 improves
/// mean |Δ| against single-pass from **2.51 → 1.95 /255** on real Krea latents (and 2.956 → 2.221 on
/// a higher-detail source). On **highlight clipping the direction is source-dependent**, which
/// resolves the "clipping got worse at tile 8" note in the sc-15325 story: on a high-dynamic-range
/// source it improves (0.76 %/2.09 % → 0.61 %/1.52 %), while on the smoother real-latent clip it moves
/// slightly the other way (0.08 %/0.25 % → 0.14 %/1.05 %, against a 0.08 %/0.25 % single-pass
/// reference). The story's note is therefore **reproducible but not general**, and either way the
/// clipping stays within ~1 % of the single-pass floor — two orders of magnitude below the
/// 9.7 %/26.6 % the starved latent-2 window produced. Tile *size* is what governs clipping; the
/// overlap barely moves it in either direction.
///
/// sc-15445 then measured that bound on both real Wan VAEs at the 640×384×81 and 832×480×121
/// shipping points. The half-tile policy cost **31.8–48.5 %** on z16 and **21.1–39.3 %** on z48 for
/// only **0.30–0.60/255** and **0.10–0.15/255** less error respectively, with unchanged clipping and
/// peak. That is material wall time for a marginal Wan gain, so the Wan product selectors now choose
/// [`TemporalOverlapPolicy::Candidate`]. Krea Realtime and SCAIL-2 deliberately retain
/// [`TemporalOverlapPolicy::HalfTile`]: their sc-15325 correction is quality-oriented, and they share
/// the z16 VAE while entering through a separate selector. LTX also retains half-tile overlap under
/// its separately measured cost/benefit argument above. The full A/B, hashes, and environment are in
/// `docs/migration/SC_15445_WAN_OVERLAP_AB.md`.
pub const MIN_TEMPORAL_TILE_LATENT_OVERLAP: i32 = 2;

/// The smallest **output**-frame temporal tile that satisfies [`MIN_TEMPORAL_TILE_LATENT_FRAMES`] for
/// `vae` (`8 · temporal_scale`: 32 output frames for the ×4 Wan VAEs, 64 for the ×8 LTX VAE).
pub fn min_temporal_tile_frames(vae: VaeTiling) -> i32 {
    MIN_TEMPORAL_TILE_LATENT_FRAMES * vae.temporal_scale.max(1)
}

/// The temporal overlap (output frames) to stamp on a `tile_frames` temporal tile: the candidate's own
/// `candidate_overlap`, raised so the *latent* overlap is at least
/// [`MIN_TEMPORAL_TILE_LATENT_OVERLAP`] and at least half the latent tile, then clamped below the tile
/// (`split_spatial` clamps the latent overlap to `tile − 1`, so anything larger is inert).
///
/// Public so a consumer can reason about — and a test can gate — the policy without re-deriving the
/// arithmetic.
pub fn temporal_overlap_for(vae: VaeTiling, tile_frames: i32, candidate_overlap: i32) -> i32 {
    temporal_overlap_for_policy(
        TemporalOverlapPolicy::HalfTile,
        vae,
        tile_frames,
        candidate_overlap,
    )
}

/// How a selected temporal candidate's overlap is finalized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemporalOverlapPolicy {
    /// Preserve the VAE/provider candidate grid's measured overlap, enforcing only the shared
    /// two-latent-frame floor. This is the Wan product policy after sc-15445.
    Candidate,
    /// Raise overlap to at least half the latent tile. This remains the quality-oriented policy for
    /// Krea Realtime, SCAIL-2, and LTX.
    HalfTile,
}

fn temporal_overlap_for_policy(
    policy: TemporalOverlapPolicy,
    vae: VaeTiling,
    tile_frames: i32,
    candidate_overlap: i32,
) -> i32 {
    let scale = vae.temporal_scale.max(1);
    let tile_latent = (tile_frames / scale).max(1);
    let policy_floor = match policy {
        TemporalOverlapPolicy::Candidate => MIN_TEMPORAL_TILE_LATENT_OVERLAP,
        TemporalOverlapPolicy::HalfTile => (tile_latent / 2).max(MIN_TEMPORAL_TILE_LATENT_OVERLAP),
    };
    let want_latent = policy_floor
        .max(candidate_overlap / scale)
        // `split_spatial` clamps the latent overlap to `tile − 1`; emitting more would be inert, and
        // silently-inert config is how the original defect hid (`overlap = 4` at a latent tile of 2
        // read as generous while planning identically to `overlap = 8`).
        .min((tile_latent - 1).max(0));
    want_latent * scale
}

/// A candidate tile-size grid for [`budgeted_plan`], in **output** units. Each VAE supplies its own —
/// the sweet-spot tile sizes differ by decoder architecture (channel widths, resblock depth).
#[derive(Clone, Copy, Debug)]
pub struct TileCandidates<'a> {
    /// Candidate spatial tile sizes (output px). Order is irrelevant — the selector keeps the
    /// largest-volume tile that fits, regardless of position.
    pub spatial_px: &'a [i32],
    /// Spatial overlap (output px) stamped onto whichever spatial tile is chosen.
    pub spatial_overlap_px: i32,
    /// Candidate temporal tiles `(tile_frames, overlap_frames)` in output frames.
    pub temporal: &'a [(i32, i32)],
    /// Whether the selected temporal candidate keeps its own overlap or is raised to half a tile.
    pub temporal_overlap_policy: TemporalOverlapPolicy,
}

/// Why [`budgeted_plan`] could not fit a decode within the safe budget even with tiling. Carries the
/// numbers; the caller formats a model-specific message (gen-core stays free of model/backend wording
/// and units the caller knows better — e.g. "wan z48 vae22 decode: …").
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TilingBudgetError {
    /// The full-output accumulators alone (the assembled video the decode must hold) exceed the safe
    /// budget — no tiling can help, since every plan pays this floor.
    AccumulatorsExceedBudget { projected_gib: f64, safe_gib: f64 },
    /// Even the smallest candidate tile peaks over the safe budget.
    SmallestTileExceedsBudget { projected_gib: f64, safe_gib: f64 },
}

impl core::fmt::Display for TilingBudgetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AccumulatorsExceedBudget {
                projected_gib,
                safe_gib,
            } => write!(
                f,
                "video VAE decode: the output buffers alone need ~{projected_gib:.0} GB, over the \
                 ~{safe_gib:.0} GB safe budget; reduce the resolution or frame count"
            ),
            Self::SmallestTileExceedsBudget {
                projected_gib,
                safe_gib,
            } => write!(
                f,
                "video VAE decode: peaks at ~{projected_gib:.0} GB even with the smallest tile, over \
                 the ~{safe_gib:.0} GB safe budget; reduce the resolution or frame count"
            ),
        }
    }
}

impl std::error::Error for TilingBudgetError {}

/// Pick the **memory-budgeted** tiling for a video VAE decode (the neutral core of sc-4998). Given the
/// decoded **output** dims, a safe peak-GiB ceiling, a candidate tile grid, and a per-VAE `peak_cost`
/// estimator, returns:
///   • `Ok(None)`    — a single-pass decode already fits `safe_gib` (small/short video); the caller's
///                     existing single-pass `decode` runs, so single-pass is reached **only** when safe.
///   • `Ok(Some(c))` — tiling is required; `c` is the **largest** tile whose estimated peak ≤
///                     `safe_gib` (largest ⇒ fewest tiles ⇒ least overlap-recompute ⇒ fastest within
///                     budget).
///   • `Err(..)`     — infeasible even tiled: a **catchable** signal so the caller errors *before* the
///                     decode rather than letting the OS hard-kill the process (SIGKILL) or the GPU
///                     command buffer abort mid-decode.
///
/// `peak_cost(out_f, out_h, out_w, tile_f, tile_h, tile_w)` returns the estimated concurrent GPU peak
/// in GiB for a decode whose largest tile spans `tile_*` output voxels while assembling `out_*`. The
/// single-pass case is `tile_* == out_*`; a **zero tile** `(out_f, out_h, out_w, 0, 0, 0)` must yield
/// the accumulator-only floor (the unavoidable cost of holding the assembled output). The estimator
/// owns every model/dtype constant, so this selector carries none.
///
/// `vae` supplies the **correctness** bound that memory cannot see: past
/// [`VaeTiling::writable_frame_cap`] a single pass does not OOM, it returns wrong pixels silently
/// (sc-12349). Without it this selector had an inverted safety property — the tiling decision was
/// purely a memory test while the write bound is fixed, so **a bigger machine was more likely to pick
/// the silently-wrong single pass**. Both bounds now apply; neither substitutes for the other.
pub fn budgeted_plan(
    vae: VaeTiling,
    out_height: i32,
    out_width: i32,
    out_frames: i32,
    safe_gib: f64,
    candidates: TileCandidates<'_>,
    peak_cost: impl Fn(i64, i64, i64, i64, i64, i64) -> f64,
) -> Result<Option<TilingConfig>, TilingBudgetError> {
    let (h, w, f) = (out_height as i64, out_width as i64, out_frames as i64);

    // 0. The write bound: how many output frames one pass may span before the decoder's widest
    //    full-resolution write exceeds what MLX can address. Independent of memory, and of the machine.
    let frame_cap = vae.writable_frame_cap(out_height, out_width);
    if frame_cap == 0 {
        return Err(TilingBudgetError::AccumulatorsExceedBudget {
            projected_gib: peak_cost(f, h, w, 0, 0, 0),
            safe_gib,
        });
    }

    // 1. Single-pass (the whole output as one tile) fits the budget AND stays writable → no tiling.
    let single = peak_cost(f, h, w, f, h, w);
    if single <= safe_gib && f <= frame_cap {
        return Ok(None);
    }

    // 2. The full-output accumulators are unavoidable (they hold the assembled video); if they alone
    //    blow the budget no tiling can help — fail catchably rather than OOM mid-decode.
    let accum = peak_cost(f, h, w, 0, 0, 0);
    if accum >= safe_gib {
        return Err(TilingBudgetError::AccumulatorsExceedBudget {
            projected_gib: accum,
            safe_gib,
        });
    }

    // 3. Search candidate tiles; among those that fit, keep the one with the **largest** output
    //    volume (fewest tiles → least overlap recompute). Candidate axes include the full dimension
    //    (= "don't tile this axis"), so a spatial-only or temporal-only plan can win.
    let max_sp = h.max(w) as i32;
    let mut spatial: Vec<i32> = candidates
        .spatial_px
        .iter()
        .copied()
        .filter(|&s| s < max_sp)
        .collect();
    spatial.push(max_sp); // full spatial extent = no spatial tiling
                          // sc-15325: a temporal tile shorter than `MIN_TEMPORAL_TILE_LATENT_FRAMES` latent frames is not
                          // selectable at any budget — it starves the decoder's temporal convolutions and corrupts the
                          // *content* of every tile, which no amount of blending recovers (see the constant's measurements).
                          // Memory pressure therefore has to be relieved on the spatial axis (or by refusing the decode),
                          // never by shortening the temporal receptive field. The full-extent entry below is exempt: it is
                          // "do not tile temporally at all", which gives the decoder the whole sequence.
    let min_tile_frames = min_temporal_tile_frames(vae);
    let mut temporal: Vec<(i32, i32)> = candidates
        .temporal
        .iter()
        .copied()
        .filter(|&(t, _)| (t as i64) < f && t >= min_tile_frames)
        .map(|(t, o)| {
            (
                t,
                temporal_overlap_for_policy(candidates.temporal_overlap_policy, vae, t, o),
            )
        })
        .collect();
    temporal.push((f as i32, 0)); // full temporal extent = no temporal tiling

    let mut best: Option<(i64, i32, i32, i32)> = None; // (tile_voxels, s, t, t_overlap)
    let mut min_peak = single; // finite floor for the "smallest tile" error if nothing fits
    for &s in &spatial {
        let tile_h = (s as i64).min(h);
        let tile_w = (s as i64).min(w);
        for &(t, t_over) in &temporal {
            let tile_f = (t as i64).min(f);
            // Skip the single-pass cell (handled in step 1; it does not fit here by construction).
            if tile_h == h && tile_w == w && tile_f == f {
                continue;
            }
            // A tile must be writable as well as affordable: the decoder materializes it at full
            // resolution and `full_res_channels` wide, and past MAX_WRITABLE_ELEMS that write is
            // silently wrong rather than merely expensive. Fitting the budget is not enough.
            if vae.full_res_channels as i64 * tile_f * tile_h * tile_w > MAX_WRITABLE_ELEMS {
                continue;
            }
            let peak = peak_cost(f, h, w, tile_f, tile_h, tile_w);
            min_peak = min_peak.min(peak);
            if peak > safe_gib {
                continue;
            }
            let voxels = tile_f * tile_h * tile_w;
            if best.is_none_or(|(bv, ..)| voxels > bv) {
                best = Some((voxels, s, t, t_over));
            }
        }
    }

    let Some((_, s, t, t_over)) = best else {
        return Err(TilingBudgetError::SmallestTileExceedsBudget {
            projected_gib: min_peak,
            safe_gib,
        });
    };

    // Only tile an axis whose chosen tile is actually smaller than the axis.
    let spatial = ((s as i64) < max_sp as i64).then_some(SpatialTiling {
        tile_px: s,
        overlap_px: candidates.spatial_overlap_px,
    });
    let temporal = ((t as i64) < f).then_some(TemporalTiling {
        tile_frames: t,
        overlap_frames: t_over,
    });
    Ok(Some(TilingConfig { spatial, temporal }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_memory_profile_composes_contract_decoder_exactly_once() {
        assert_eq!(VideoDecodeMemoryProfile::new(399, 400), None);

        let profile = VideoDecodeMemoryProfile::new(1_000, 400).unwrap();
        assert_eq!(
            profile.incremental_above_contract_decoder_bytes(0),
            Some(1_000)
        );
        assert_eq!(
            profile.incremental_above_contract_decoder_bytes(250),
            Some(750)
        );
        assert_eq!(profile.checked_composed_peak(1_250, 250), Some(2_000));
        assert_eq!(
            profile.incremental_above_contract_decoder_bytes(400),
            Some(600)
        );
        assert_eq!(
            profile.incremental_above_contract_decoder_bytes(600),
            Some(600)
        );
        assert_eq!(profile.checked_composed_peak(1_600, 600), Some(2_200));
        assert_eq!(profile.checked_composed_peak(399, 400), None);
        assert_eq!(profile.checked_composed_peak(u64::MAX, 600), None);
    }

    #[test]
    fn trapezoid_no_ramp_is_all_ones() {
        assert_eq!(trapezoidal_mask(4, 0, 0, false), vec![1.0; 4]);
    }

    #[test]
    fn trapezoid_right_fade_out() {
        // ramp_right=2: last two = (3-1)/3, (3-2)/3 = 2/3, 1/3.
        let m = trapezoidal_mask(5, 0, 2, false);
        assert_eq!(m[0], 1.0);
        assert_eq!(m[2], 1.0);
        assert!((m[3] - 2.0 / 3.0).abs() < 1e-6);
        assert!((m[4] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn trapezoid_left_from_0_fade_in() {
        // ramp_left=2, left_from_0: linspace(0,1,3)[:-1] = [0, 0.5].
        let m = trapezoidal_mask(5, 2, 0, true);
        assert!((m[0] - 0.0).abs() < 1e-6);
        assert!((m[1] - 0.5).abs() < 1e-6);
        assert_eq!(m[2], 1.0);
    }

    #[test]
    fn spatial_split_three_tiles() {
        // tile=2, overlap=1, dim=4 → amount=(4+2-2-1)/1=3.
        let (starts, ends, left, right) = split_spatial(2, 1, 4);
        assert_eq!(starts, vec![0, 1, 2]);
        assert_eq!(ends, vec![2, 3, 4]);
        assert_eq!(left, vec![0, 1, 1]);
        assert_eq!(right, vec![1, 1, 0]);
    }

    #[test]
    fn temporal_split_causal_adjust() {
        // tile=2, overlap=1, dim=3 → spatial(2,1,3): amount=(3+2-2-1)/1=2, starts=[0,1].
        // temporal: starts[1]-=1 → [0,0], left[1]+=1.
        let (starts, _ends, left, _right) = split_temporal(2, 1, 3);
        assert_eq!(starts, vec![0, 0]);
        assert_eq!(left, vec![0, 2]);
    }

    #[test]
    fn needs_tiling_thresholds_ltx() {
        // LTX spatial_scale 32: tile_px 64 → 2 latent.
        let cfg = TilingConfig::spatial_only(64, 32);
        assert!(cfg.needs_tiling(VaeTiling::LTX, 1, 4, 4)); // h=4 > 2
        assert!(!cfg.needs_tiling(VaeTiling::LTX, 10, 2, 2)); // h=w=2 not > 2
        let tc = TilingConfig::temporal_only(16, 8); // temporal_scale 8: 16 → 2 latent
        assert!(tc.needs_tiling(VaeTiling::LTX, 3, 2, 2)); // f=3 > 2
        assert!(!tc.needs_tiling(VaeTiling::LTX, 2, 99, 99)); // f=2 not > 2
    }

    #[test]
    fn needs_tiling_thresholds_wan() {
        // Wan spatial_scale 8: tile_px 64 → 8 latent; temporal_scale 4: 16 frames → 4 latent.
        let cfg = TilingConfig::spatial_only(64, 32);
        assert!(cfg.needs_tiling(VaeTiling::WAN, 1, 9, 4)); // h=9 > 8
        assert!(!cfg.needs_tiling(VaeTiling::WAN, 10, 8, 8)); // h=w=8 not > 8
        let tc = TilingConfig::temporal_only(16, 8);
        assert!(tc.needs_tiling(VaeTiling::WAN, 5, 2, 2)); // f=5 > 4
        assert!(!tc.needs_tiling(VaeTiling::WAN, 4, 99, 99)); // f=4 not > 4
    }

    /// LTX (causal) temporal mapping: `out_f = 1 + (f−1)·8`, first tile starts at 0.
    #[test]
    fn plan_ltx_causal_temporal_output_dims() {
        let cfg = TilingConfig::temporal_only(16, 8); // tile=2, overlap=1 latent
        let plan = cfg.plan(VaeTiling::LTX, 3, 2, 2);
        assert_eq!(plan.out_f, 1 + (3 - 1) * 8); // 17
        assert_eq!(plan.out_h, 2 * 32);
        assert_eq!(plan.out_w, 2 * 32);
        assert_eq!(plan.t[0].out_start, 0);
    }

    /// Wan (non-causal) temporal mapping: `out_f = f·4`, temporal tiles behave like spatial.
    #[test]
    fn plan_wan_noncausal_temporal_output_dims() {
        let cfg = TilingConfig::temporal_only(16, 8); // temporal_scale 4: tile=4, overlap=2 latent
        let plan = cfg.plan(VaeTiling::WAN, 6, 2, 2);
        assert_eq!(plan.out_f, 6 * 4); // 24, NOT 1+(6-1)*4
        assert_eq!(plan.out_h, 2 * 8);
        assert_eq!(plan.out_w, 2 * 8);
        // Non-causal: the first temporal tile starts at 0 and maps out_start = 0.
        assert_eq!(plan.t[0].out_start, 0);
        assert_eq!(plan.t.last().unwrap().out_stop, 24);
    }

    /// Coverage invariant: the summed blend weight is strictly positive at **every** output position
    /// on each axis (no zero-weight gaps → the final divide is well-defined). Checked for both VAEs.
    #[test]
    fn plan_covers_every_output_position() {
        // Includes the causal z48 vae22 (WAN22): its temporal `out_stop = 1+(end−1)·scale` mapping
        // and per-tile left-ramp adjustment must still cover every output frame with no zero-weight
        // gap when combined with spatial tiling (sc-5690 — the combined-plan blend relies on this).
        for (vae, f, h, w) in [
            (VaeTiling::WAN, 9, 9, 13),
            (VaeTiling::WAN22, 9, 9, 13),
            (VaeTiling::LTX, 5, 5, 5),
        ] {
            let cfg = TilingConfig {
                spatial: Some(SpatialTiling {
                    tile_px: 4 * vae.spatial_scale,
                    overlap_px: 2 * vae.spatial_scale,
                }),
                temporal: Some(TemporalTiling {
                    tile_frames: 3 * vae.temporal_scale,
                    overlap_frames: vae.temporal_scale,
                }),
            };
            let plan = cfg.plan(vae, f, h, w);
            for (axis, tiles, out) in [
                ("t", &plan.t, plan.out_f),
                ("h", &plan.h, plan.out_h),
                ("w", &plan.w, plan.out_w),
            ] {
                let mut weight = vec![0f32; out as usize];
                for tile in tiles {
                    for (i, &m) in tile.mask.iter().enumerate() {
                        weight[tile.out_start as usize + i] += m;
                    }
                }
                assert!(
                    weight.iter().all(|&v| v > 1e-6),
                    "{vae:?} axis {axis}: zero-weight output position (gap in tiling)"
                );
            }
        }
    }

    /// F-005: degenerate tile/overlap configs (tile == overlap, overlap > tile, and a tile floored to
    /// 0 by latent downscaling) must not panic — they clamp to a valid split instead of dividing by
    /// zero or wrapping `amount` to a huge length.
    #[test]
    fn split_spatial_survives_degenerate_overlap() {
        // tile == overlap (would divide by zero), overlap > tile (would wrap), tile == 0 (floored).
        for (size, overlap) in [(8, 8), (8, 16), (0, 0), (0, 4)] {
            let (starts, ends, left, right) = split_spatial(size, overlap, 64);
            assert!(
                !starts.is_empty(),
                "size={size} overlap={overlap}: no tiles"
            );
            assert_eq!(starts.len(), ends.len());
            assert_eq!(left.len(), right.len());
            assert_eq!(*ends.last().unwrap(), 64, "last tile must reach dim");
        }
    }

    /// The crash is reachable through the public `plan` via `spatial_only`/`temporal_only` with a tile
    /// ≤ overlap; it must produce a valid, gap-free plan rather than panicking.
    #[test]
    fn plan_survives_tile_equal_overlap() {
        let cfg = TilingConfig::spatial_only(64, 64); // tile_px == overlap_px
        let plan = cfg.plan(VaeTiling::WAN, 1, 16, 16);
        for (tiles, out) in [(&plan.h, plan.out_h), (&plan.w, plan.out_w)] {
            let mut weight = vec![0f32; out as usize];
            for tile in tiles {
                for (i, &m) in tile.mask.iter().enumerate() {
                    weight[tile.out_start as usize + i] += m;
                }
            }
            assert!(
                weight.iter().all(|&v| v > 1e-6),
                "tile==overlap plan left a zero-weight gap"
            );
        }
    }

    // --- budgeted_plan (sc-6894) ------------------------------------------------------------------

    // Synthetic linear peak model shaped like a real VAE's: `accum`·out_voxels (the output buffers,
    // paid by every plan) + `tile`·tile_voxels (the per-tile decoder working set), both GiB/voxel.
    fn lin_cost(accum: f64, tile: f64) -> impl Fn(i64, i64, i64, i64, i64, i64) -> f64 {
        move |of, oh, ow, tf, th, tw| accum * (of * oh * ow) as f64 + tile * (tf * th * tw) as f64
    }

    const T_SPATIAL: [i32; 3] = [256, 192, 128];
    const T_TEMPORAL: [(i32, i32); 2] = [(32, 8), (16, 4)];
    fn t_cands() -> TileCandidates<'static> {
        TileCandidates {
            spatial_px: &T_SPATIAL,
            spatial_overlap_px: 64,
            temporal: &T_TEMPORAL,
            temporal_overlap_policy: TemporalOverlapPolicy::HalfTile,
        }
    }

    /// Re-derive the chosen plan's peak the way the selector sizes its largest tile.
    fn chosen_peak(
        cfg: &TilingConfig,
        h: i64,
        w: i64,
        f: i64,
        cost: &impl Fn(i64, i64, i64, i64, i64, i64) -> f64,
    ) -> f64 {
        let tile_h = cfg.spatial.map(|s| (s.tile_px as i64).min(h)).unwrap_or(h);
        let tile_w = cfg.spatial.map(|s| (s.tile_px as i64).min(w)).unwrap_or(w);
        let tile_f = cfg
            .temporal
            .map(|t| (t.tile_frames as i64).min(f))
            .unwrap_or(f);
        cost(f, h, w, tile_f, tile_h, tile_w)
    }

    #[test]
    fn budgeted_single_pass_when_it_fits() {
        // A generous budget → the whole decode fits in one pass, no tiling.
        let cost = lin_cost(4e-8, 4e-6);
        let plan = budgeted_plan(VaeTiling::WAN, 512, 512, 64, 1_000.0, t_cands(), &cost).unwrap();
        assert!(
            plan.is_none(),
            "should not tile under a huge budget: {plan:?}"
        );
    }

    /// `writable_frame_cap` is the correctness bound: how many output frames fit under
    /// [`MAX_WRITABLE_ELEMS`] at a given resolution, given the VAE's full-res width.
    #[test]
    fn writable_frame_cap_tracks_full_res_width() {
        // Wan z16 at 720p: 96 ch × 1280 × 720 = 88,473,600 per frame → 24 frames fit under 2^31.
        assert_eq!(VaeTiling::WAN.writable_frame_cap(720, 1280), 24);
        // LTX is 12× narrower per output voxel (8 ch), so it reaches far more frames at the same size.
        assert_eq!(VaeTiling::LTX.writable_frame_cap(720, 1280), 291);
        // A still-image VAE at its 2048² cap is nowhere near the bound.
        assert!(VaeTiling::QWEN_IMAGE.writable_frame_cap(2048, 2048) > 1);
        // The cap must scale with the declared width: doubling channels halves the frames.
        let narrow = VaeTiling {
            full_res_channels: 48,
            ..VaeTiling::WAN
        };
        assert_eq!(
            narrow.writable_frame_cap(720, 1280),
            2 * VaeTiling::WAN.writable_frame_cap(720, 1280),
            "the cap must be driven by full_res_channels, not hardcoded"
        );
    }

    /// **The inversion this fixed** (sc-12349): the tiling decision used to be a pure memory test, so a
    /// machine with a big enough budget would choose the single pass at a geometry where the 0.31.2
    /// decode was silently WRONG. sc-12748: on 0.32.0 that single pass is now *correct* (conv is
    /// int64-safe), so this is defense-in-depth rather than a correctness necessity — but the selector
    /// still tiles past the cap (a cheap, always-correct default), which this pins.
    ///
    /// ## sc-15325 — a DELIBERATE loosening of what this asserts
    ///
    /// This test used to assert "the selector must tile the **TEMPORAL** axis past the cap". That is
    /// no longer the right property and the change is intentional, not incidental. 720p's write cap is
    /// 24 output frames = **6 latent frames**, which is *below*
    /// [`MIN_TEMPORAL_TILE_LATENT_FRAMES`]: after sc-15325 there is no temporal tile that both
    /// satisfies the receptive-field floor and relieves the write bound, so the only correct relief is
    /// spatial. Asserting the axis would have forced a choice between the write bound and the
    /// receptive field, which is a false trade — the bound is on the tile's *volume*
    /// (`full_res_channels · tile_f · tile_h · tile_w`), and either axis satisfies it.
    ///
    /// So the assertion is now on the property that actually matters — **the selected tile is
    /// writable** — and is deliberately silent on which axis paid for it. It is strictly stronger in
    /// the sense that it checks the tile the caller will really materialise rather than a proxy for
    /// it; it is weaker only about an axis whose choice is now a memory/quality decision, not a
    /// correctness one.
    #[test]
    fn budgeted_tiles_past_the_write_bound_even_with_an_infinite_budget() {
        // Free memory: without the write bound this returns Ok(None) — a single pass — always.
        let free = |_: i64, _: i64, _: i64, _: i64, _: i64, _: i64| 0.0;
        let (h, w) = (720i32, 1280i32);
        let cap = VaeTiling::WAN.writable_frame_cap(h, w); // 24

        // At the cap: writable, budget is free → single pass is correct and allowed.
        let at = budgeted_plan(
            VaeTiling::WAN,
            h,
            w,
            cap as i32,
            f64::INFINITY,
            t_cands(),
            free,
        )
        .unwrap();
        assert!(
            at.is_none(),
            "at the write bound a single pass is fine: {at:?}"
        );

        // One frame past it: memory still says "free", but the decode would write past what MLX can
        // address, so the selector MUST tile rather than return None.
        let past = budgeted_plan(
            VaeTiling::WAN,
            h,
            w,
            cap as i32 + 1,
            f64::INFINITY,
            t_cands(),
            free,
        )
        .unwrap();
        let cfg = past.expect(
            "past the write bound the selector must tile even on an unlimited budget — returning \
             None here is the silently-wrong single pass this bound exists to prevent",
        );
        // sc-15325: which AXIS relieves the bound is no longer fixed. The write bound is a per-tile
        // limit on `full_res_channels · tile_f · tile_h · tile_w`, so shrinking the tile spatially
        // satisfies it exactly as well as shrinking it temporally — and it is now the *only* way when
        // the cap is below `MIN_TEMPORAL_TILE_LATENT_FRAMES` worth of frames (720p's cap is 24 output
        // frames = 6 latent, under the 8-latent-frame receptive-field floor). Pin the property that
        // matters — the selected tile is writable — rather than the axis it came from.
        let tile_f = cfg
            .temporal
            .map(|t| (t.tile_frames as i64).min(cap + 1))
            .unwrap_or(cap + 1);
        let tile_h = cfg
            .spatial
            .map(|s| (s.tile_px as i64).min(h as i64))
            .unwrap_or(h as i64);
        let tile_w = cfg
            .spatial
            .map(|s| (s.tile_px as i64).min(w as i64))
            .unwrap_or(w as i64);
        assert!(
            VaeTiling::WAN.full_res_channels as i64 * tile_f * tile_h * tile_w
                <= MAX_WRITABLE_ELEMS,
            "the chosen tile must itself be writable: {tile_f}x{tile_h}x{tile_w} at 96 ch is over \
             the {MAX_WRITABLE_ELEMS}-element bound"
        );
    }

    // --- sc-15325: the temporal receptive-field floor ---------------------------------------------

    /// **The regression guard for sc-15325 at the shared seam** — at a geometry where the *pre-fix*
    /// selector genuinely reaches the starved candidate.
    ///
    /// ⚠️ **Vacuity is the failure mode this guard has already had once.** Its first draft swept
    /// 8–80 GiB at 512²×64, where the pre-fix largest-volume search picks `(32, …)` at *every* budget:
    /// `(16, 4)` was never reachable, and `(32, 8)`'s latent overlap of 2 already satisfied the blend
    /// floor. It therefore passed verbatim on pre-fix production code (reverting both changed lines in
    /// [`budgeted_plan`] — the `t >= min_tile_frames` filter and the [`temporal_overlap_for`] map —
    /// left the whole `tiling::` suite green). The budgets below are chosen so the pre-fix answer
    /// differs in **every** row, and the post-fix answer is *pinned* rather than merely constrained,
    /// so a revert cannot be green.
    ///
    /// With `lin_cost(4e-8, 4e-6)` at 512×512×64 the accumulator floor is 0.671 GiB and a tile costs
    /// 4e-6 GiB per output voxel, which makes the three regimes exactly reachable:
    ///
    /// | safe GiB | pre-fix choice | post-fix choice (asserted) |
    /// |---|---|---|
    /// | 2.0 | spatial 128 + temporal `(16, 4)` → **latent 4** | `Err(SmallestTileExceedsBudget)` — refuse rather than starve |
    /// | 2.9 | spatial 128 + temporal `(32, 8)` → latent 8 / **overlap 2** | spatial 128 + temporal `(32, 16)` → **overlap 4** |
    /// | 5.0 | spatial 256 + temporal `(16, 4)` → **latent 4** | **spatial-only** 128 px — no temporal tiling at all |
    ///
    /// Row 3 is the shape of the whole fix: memory pressure that used to be relieved by shortening the
    /// temporal receptive field is now relieved on the spatial axis instead.
    #[test]
    fn budgeted_never_selects_a_starved_temporal_tile() {
        let cost = lin_cost(4e-8, 4e-6);
        let plan = |safe: f64| budgeted_plan(VaeTiling::WAN, 512, 512, 64, safe, t_cands(), &cost);

        // Row 1 — only a starved tile would have fit. Refusing is the correct answer; the caller
        // surfaces it as a catchable over-budget error instead of silently corrupting the picture.
        let tight = plan(2.0);
        assert!(
            matches!(
                tight,
                Err(TilingBudgetError::SmallestTileExceedsBudget { .. })
            ),
            "at 2.0 GiB the only affordable temporal tile is latent 4; the selector must refuse, not \
             starve the decoder — got {tight:?}"
        );

        // Row 2 — the tile is fine but the candidate's own overlap is latent 2. The policy raises it
        // to half the latent tile (4 latent = 16 output frames). Pre-fix this stayed at 8.
        let raised = plan(2.9).unwrap().expect("2.9 GiB must tile");
        let rt = raised.temporal.expect("2.9 GiB must tile temporally");
        assert_eq!(
            (rt.tile_frames, rt.overlap_frames),
            (32, 16),
            "the selected temporal tile must carry the RAISED overlap (latent 8 / 4), not the \
             candidate grid's latent-2 value: {raised:?}"
        );
        assert_eq!(raised.spatial.map(|s| s.tile_px), Some(128));

        // Row 3 — the pre-fix selector bought volume by shortening time (spatial 256 + latent-4
        // temporal tile). The fix inverts that: the whole sequence stays intact and the memory comes
        // off the spatial axis.
        let relieved = plan(5.0).unwrap().expect("5.0 GiB must tile");
        assert!(
            relieved.temporal.is_none(),
            "at 5.0 GiB a full-sequence, spatially-relieved plan fits — taking a latent-4 temporal \
             tile for more tile volume is exactly the sc-15325 defect: {relieved:?}"
        );
        assert_eq!(relieved.spatial.map(|s| s.tile_px), Some(128));

        // ...and the invariant itself, across the whole feasible range rather than at three points.
        for safe in [2.0f64, 2.9, 3.5, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0] {
            let Ok(Some(cfg)) = plan(safe) else {
                continue; // infeasible or single-pass — neither can starve a tile.
            };
            let Some(t) = cfg.temporal else { continue };
            let lat = t.tile_frames / VaeTiling::WAN.temporal_scale;
            assert!(
                lat >= MIN_TEMPORAL_TILE_LATENT_FRAMES,
                "safe={safe}: selected a {lat}-latent-frame temporal tile ({} output frames), under \
                 the sc-15325 receptive-field floor",
                t.tile_frames
            );
            let lat_over = (t.overlap_frames / VaeTiling::WAN.temporal_scale).min(lat - 1);
            assert!(
                lat_over >= MIN_TEMPORAL_TILE_LATENT_OVERLAP,
                "safe={safe}: latent overlap {lat_over} is under the blend floor"
            );
        }
    }

    /// The floor is expressed in **latent** frames, so it scales with the VAE — an LTX `(24, 8)`
    /// candidate (24 output frames ÷ temporal_scale 8 = 3 latent) is starved even though it is three
    /// times as many *output* frames as a Wan `(8, 2)` window. Reading the grid in output units is
    /// exactly the mistake that left LTX uncleared, so pin the unit.
    #[test]
    fn the_floor_is_in_latent_frames_not_output_frames() {
        assert_eq!(min_temporal_tile_frames(VaeTiling::WAN), 32);
        assert_eq!(min_temporal_tile_frames(VaeTiling::WAN22), 32);
        assert_eq!(min_temporal_tile_frames(VaeTiling::LTX), 64);

        // The real LTX grid, filtered through the floor: 24 and 48 output frames are latent 3 and 6.
        const LTX_GRID: [(i32, i32); 4] = [(96, 24), (64, 16), (48, 16), (24, 8)];
        let kept: Vec<i32> = LTX_GRID
            .iter()
            .map(|&(t, _)| t)
            .filter(|&t| t >= min_temporal_tile_frames(VaeTiling::LTX))
            .collect();
        assert_eq!(
            kept,
            vec![96, 64],
            "the floor must remove LTX's latent-3 (24f) and latent-6 (48f) temporal candidates"
        );
    }

    /// Overlap scales with the tile and is clamped below it. `split_spatial` clamps the *latent*
    /// overlap to `tile − 1`, so an overlap at or above the tile is inert — the policy must not emit
    /// one and then rely on the clamp.
    #[test]
    fn temporal_overlap_scales_with_the_tile_and_stays_below_it() {
        // Wan ×4: a 32-frame tile is latent 8 → half is latent 4 → 16 output frames.
        assert_eq!(temporal_overlap_for(VaeTiling::WAN, 32, 8), 16);
        // A candidate that already asks for more than half the tile keeps its own, larger value
        // (96 output frames is latent 24; a 64-frame overlap is latent 16 > 12).
        assert_eq!(temporal_overlap_for(VaeTiling::WAN, 96, 64), 64);
        // ...but a candidate asking for less is raised to half the latent tile (24/2 = 12 → 48).
        assert_eq!(temporal_overlap_for(VaeTiling::WAN, 96, 24), 48);
        assert_eq!(temporal_overlap_for(VaeTiling::WAN, 96, 40), 48);
        // LTX ×8: a 64-frame tile is latent 8 → half is latent 4 → 32 output frames.
        assert_eq!(temporal_overlap_for(VaeTiling::LTX, 64, 16), 32);
        // Never at or above the tile: at latent tile 1 the only legal latent overlap is 0.
        assert_eq!(temporal_overlap_for(VaeTiling::WAN, 4, 8), 0);
    }

    /// sc-15445: the policy split must remain explicit. Wan's product candidates keep their measured
    /// overlap, while quality-oriented consumers keep the half-tile raise. Every row discriminates
    /// against deleting the policy branch or swapping either variant.
    #[test]
    fn wan_candidate_and_quality_overlap_policies_discriminate_every_shipping_row() {
        for (tile, candidate, half) in [(96, 24, 48), (64, 24, 32), (48, 16, 24), (32, 8, 16)] {
            assert_eq!(
                temporal_overlap_for_policy(
                    TemporalOverlapPolicy::Candidate,
                    VaeTiling::WAN,
                    tile,
                    candidate,
                ),
                candidate,
            );
            assert_eq!(
                temporal_overlap_for_policy(
                    TemporalOverlapPolicy::HalfTile,
                    VaeTiling::WAN,
                    tile,
                    candidate,
                ),
                half,
            );
        }

        // The exact two A/B geometries: overlap alone raises temporal/all VAE tile calls by 25 %.
        for (vae, latent, h, w, tile, candidate, expected) in [
            (VaeTiling::WAN, 21, 48, 80, 32, 8, (4, 5, 60, 75)),
            (VaeTiling::WAN, 31, 60, 104, 48, 16, (4, 5, 96, 120)),
            (VaeTiling::WAN22, 21, 24, 40, 32, 8, (4, 5, 60, 75)),
            (VaeTiling::WAN22, 31, 30, 52, 48, 16, (4, 5, 96, 120)),
        ] {
            let iterations = |policy| {
                let cfg = TilingConfig {
                    spatial: Some(SpatialTiling {
                        tile_px: 192,
                        overlap_px: 64,
                    }),
                    temporal: Some(TemporalTiling {
                        tile_frames: tile,
                        overlap_frames: temporal_overlap_for_policy(policy, vae, tile, candidate),
                    }),
                };
                let plan = cfg.plan(vae, latent, h, w);
                (plan.t.len(), plan.t.len() * plan.h.len() * plan.w.len())
            };
            let candidate_iters = iterations(TemporalOverlapPolicy::Candidate);
            let half_iters = iterations(TemporalOverlapPolicy::HalfTile);
            assert_eq!(
                (
                    candidate_iters.0,
                    half_iters.0,
                    candidate_iters.1,
                    half_iters.1
                ),
                expected,
            );
        }
    }

    /// The floor must not make a previously-feasible decode impossible at a realistic budget: the
    /// savings move to the spatial axis. This is the affordability claim sc-15325's operating point
    /// rests on, checked against the real z16 cost coefficients (64 B/out-voxel accumulators,
    /// 6500 B/tile-out-voxel working set, fit from `vae16_decode_sweep.rs`).
    #[test]
    fn the_floor_stays_feasible_by_tiling_spatially() {
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        let z16 = |of: i64, oh: i64, ow: i64, tf: i64, th: i64, tw: i64| {
            (64.0 * (of * oh * ow) as f64 + 6500.0 * (tf * th * tw) as f64) / GIB
        };
        const SPATIAL: [i32; 8] = [768, 640, 512, 448, 384, 320, 256, 192];
        const TEMPORAL: [(i32, i32); 4] = [(96, 24), (64, 24), (48, 16), (32, 8)];
        let cands = TileCandidates {
            spatial_px: &SPATIAL,
            spatial_overlap_px: 64,
            temporal: &TEMPORAL,
            temporal_overlap_policy: TemporalOverlapPolicy::HalfTile,
        };
        // Every bucket the pre-fix krea/scail2 window collapsed at, plus the one it did not, at a
        // 12 GiB budget — well under the ~21 GiB the old fixed 8-frame full-frame window actually cost.
        for (w, h) in [
            (640, 384),
            (512, 512),
            (768, 512),
            (832, 480),
            (1280, 720),
            (512, 384),
        ] {
            let cfg = budgeted_plan(VaeTiling::WAN, h, w, 81, 12.0, cands, z16)
                .unwrap_or_else(|e| panic!("{w}x{h}x81 became infeasible at 12 GiB: {e}"))
                .unwrap_or_else(|| panic!("{w}x{h}x81 unexpectedly fits a 12 GiB single pass"));
            let lat = cfg.temporal.map(|t| t.tile_frames / 4).unwrap_or(i32::MAX); // no temporal tiling = the whole sequence
            assert!(
                lat >= MIN_TEMPORAL_TILE_LATENT_FRAMES,
                "{w}x{h}: {lat} latent frames at a 12 GiB budget"
            );
            assert!(
                cfg.spatial.is_some() || cfg.temporal.is_some(),
                "{w}x{h}: an empty plan is not a plan"
            );
        }
    }

    #[test]
    fn budgeted_tiles_and_stays_under_budget() {
        // Single-pass blows the budget; the selector must return a tile whose recomputed peak is both
        // ≤ the safe budget and strictly below the single-pass peak.
        let cost = lin_cost(4e-8, 4e-6);
        let (h, w, f) = (512, 512, 64);
        let single = cost(f, h, w, f, h, w);
        let safe = 20.0;
        assert!(
            single > safe,
            "test precondition: single-pass must exceed budget"
        );
        let cfg = budgeted_plan(
            VaeTiling::WAN,
            h as i32,
            w as i32,
            f as i32,
            safe,
            t_cands(),
            &cost,
        )
        .unwrap()
        .expect("must tile when single-pass is over budget");
        let peak = chosen_peak(&cfg, h, w, f, &cost);
        assert!(peak <= safe, "chosen peak {peak:.2} over safe {safe}");
        assert!(
            peak < single,
            "tiling must lower the peak ({peak:.2} vs {single:.2})"
        );
    }

    #[test]
    fn budgeted_errors_when_accumulators_alone_exceed_budget() {
        // Absurd per-output-voxel accum cost: even a zero tile (the unavoidable output buffers) blows
        // the budget, so no tiling can help → AccumulatorsExceedBudget.
        let cost = lin_cost(1.0, 1e-3);
        let err = budgeted_plan(VaeTiling::WAN, 512, 512, 64, 5.0, t_cands(), &cost).unwrap_err();
        assert!(
            matches!(err, TilingBudgetError::AccumulatorsExceedBudget { .. }),
            "expected AccumulatorsExceedBudget, got {err:?}"
        );
    }

    #[test]
    fn budgeted_errors_when_even_smallest_tile_exceeds_budget() {
        // Tiny accumulators (output fits) but an enormous per-tile-voxel cost: every candidate tile,
        // even the smallest, peaks over budget → SmallestTileExceedsBudget (catchable, not OOM).
        let cost = lin_cost(1e-9, 1e-3);
        let err = budgeted_plan(VaeTiling::WAN, 512, 512, 64, 5.0, t_cands(), &cost).unwrap_err();
        match err {
            TilingBudgetError::SmallestTileExceedsBudget {
                projected_gib,
                safe_gib,
            } => {
                assert_eq!(safe_gib, 5.0);
                assert!(projected_gib.is_finite() && projected_gib > 5.0);
            }
            other => panic!("expected SmallestTileExceedsBudget, got {other:?}"),
        }
    }

    #[test]
    fn budgeted_picks_temporal_only_when_spatial_already_fits() {
        // Output is small spatially (every candidate ≥ the full spatial extent, so spatial can't tile)
        // but long in frames → the winning plan tiles only the temporal axis.
        let cost = lin_cost(4e-8, 4e-6);
        let cfg = budgeted_plan(VaeTiling::WAN, 128, 128, 200, 8.0, t_cands(), &cost)
            .unwrap()
            .expect("a 200-frame clip must tile");
        assert!(
            cfg.spatial.is_none(),
            "spatial should stay un-tiled: {cfg:?}"
        );
        assert!(cfg.temporal.is_some(), "temporal axis must tile: {cfg:?}");
        let peak = chosen_peak(&cfg, 128, 128, 200, &cost);
        assert!(peak <= 8.0, "temporal-only peak {peak:.2} over budget");
    }

    /// sc-12438 (Req 3): a **single-pass** decode that satisfies the decoder-stage write bound
    /// ([`VaeTiling::writable_frame_cap`], sized by `full_res_channels`) also satisfies the assembled
    /// **RGB output** bound (3 channels) — because every VAE's widest full-resolution stage is ≥ 3
    /// channels, so the RGB output is never the binding write. This is the proof that the single-pass
    /// entry points which bypass [`budgeted_plan`] (the Wan training preview and Qwen native decode,
    /// both T=1) cannot silently corrupt: at any frame count within the decoder-stage cap, the output is
    /// under [`MAX_WRITABLE_ELEMS`]. Mutation-check: a `full_res_channels < 3` would fail the second
    /// assertion (`3·cap·h·w = 3·MAX/full_res > MAX`).
    #[test]
    fn single_pass_decoder_bound_implies_output_bound() {
        const RGB: i64 = 3;
        for vae in [
            VaeTiling::LTX,
            VaeTiling::WAN,
            VaeTiling::WAN22,
            VaeTiling::QWEN_IMAGE,
        ] {
            assert!(
                vae.full_res_channels as i64 >= RGB,
                "{vae:?}: full_res_channels {} < {RGB} RGB output channels — a single pass could pass \
                 the decoder-stage bound while writing an over-bound RGB output",
                vae.full_res_channels
            );
            let (h, w) = (1280i64, 1280i64);
            let cap = vae.writable_frame_cap(h as i32, w as i32);
            if cap > 0 && cap < i64::MAX {
                let rgb_output = RGB * cap * h * w;
                assert!(
                    rgb_output <= MAX_WRITABLE_ELEMS,
                    "{vae:?}: at the decoder frame cap {cap}, the RGB output {rgb_output} exceeds the \
                     {MAX_WRITABLE_ELEMS} bound"
                );
            }
        }
    }
}
