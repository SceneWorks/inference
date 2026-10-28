//! Native H3 VAE working-storage envelopes, not measurements or resident component weights.
//!
//! Source twins: mlx-gen-minimax-h3 and candle-gen-minimax-h3 `vae_encoder.rs`, `vae.rs`,
//! `decoder.rs`, `blocks.rs`, `audio_vae_encoder.rs`, `audio_vae.rs`, and `alias_free.rs`.
//! CNN convolutions have front-only temporal padding and reflecting spatial padding; norm folds
//! time into batch, NOT channels. The decoder is dense ViT attention, not window attention.
//! Sequential operations/layers/tiles/references take a maximum of transient storage. Retained
//! moments, condition rows and temporal/stitch outputs are additional live allocations. An MLX
//! graph retaining unevaluated nodes is not evidence that all evaluated intermediates coexist.
//! No allocator/RSS observation, currency gate, model loading, or unrelated-family calibration.

use crate::veda::workspace::{h3_visual_grids_with_patch, Request};
use crate::{Conditioning, Error, Result};

/// Facts copied from the selected native VAE configurations and materialized video dtype.
/// Audio encode is F32; decode follows its selected native materialized dtype (MLX F32,
/// Candle self.dtype). Neither is inferred from the DiT quantization tier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3ConditioningMemoryFacts {
    pub lazy_encode: bool,
    pub unified_memory: bool,
    pub video_element_bytes: u64,
    /// Largest selected native cast/fusion temporary that may first materialize INSIDE forward.
    /// Endpoint component residency is counted elsewhere; this overlaps one active serial op,
    /// not merely a constructor-only maximum and not every layer/reference at once.
    pub video_forward_extra_bytes: u64,
    pub audio_encode_forward_extra_bytes: u64,
    pub audio_decode_forward_extra_bytes: u64,
    pub latent_channels: u64,
    pub pixel_channels: u64,
    pub encoder_channels: Vec<u64>,
    pub encoder_spatial_strides: Vec<u64>,
    pub encoder_temporal_strides: Vec<u64>,
    pub encoder_layers_per_block: u64,
    pub clip_length: u64,
    pub token_drop: u64,
    pub spatial_ratio: u64,
    pub temporal_ratio: u64,
    pub decoder_layers: u64,
    pub decoder_heads: u64,
    pub decoder_head_dim: u64,
    pub decoder_ffn_mult: u64,
    pub decoder_suffix_tokens: u64,
    pub tile_edge: u64,
    pub tile_overlap: u64,
    pub audio_decode_element_bytes: u64,
    pub audio_sample_rate: u64,
    pub audio_output_channels: u64,
    pub audio_latent_channels: u64,
    pub audio_encoder_dim: u64,
    pub audio_encoder_rates: Vec<u64>,
    pub audio_attention_heads: u64,
    pub audio_decoder_dim: u64,
    pub audio_upsample_rates: Vec<u64>,
    pub audio_upsample_kernels: Vec<u64>,
    pub audio_resblock_kernels: Vec<u64>,
    pub audio_resblock_dilations: Vec<Vec<u64>>,
}

impl Default for H3ConditioningMemoryFacts {
    /// Published native configuration, useful for CPU fixtures. Providers must overwrite these
    /// from the selected component configs/dtype, not use this as a missing-profile fallback.
    fn default() -> Self {
        Self {
            lazy_encode: true,
            unified_memory: true,
            video_element_bytes: 4,
            video_forward_extra_bytes: 0,
            audio_encode_forward_extra_bytes: 0,
            audio_decode_forward_extra_bytes: 0,
            latent_channels: 24,
            pixel_channels: 3,
            encoder_channels: vec![128, 256, 256, 512, 512, 1024],
            encoder_spatial_strides: vec![2, 2, 2, 2, 1, 1],
            encoder_temporal_strides: vec![1, 2, 2, 1, 1, 1],
            encoder_layers_per_block: 2,
            clip_length: 17,
            token_drop: 3,
            spatial_ratio: 16,
            temporal_ratio: 4,
            decoder_layers: 36,
            decoder_heads: 32,
            decoder_head_dim: 64,
            decoder_ffn_mult: 4,
            decoder_suffix_tokens: 5,
            tile_edge: 256,
            tile_overlap: 64,
            audio_decode_element_bytes: 4,
            audio_sample_rate: 32_000,
            audio_output_channels: 2,
            audio_latent_channels: 32,
            audio_encoder_dim: 64,
            audio_encoder_rates: vec![2, 4, 4, 5, 5],
            audio_attention_heads: 8,
            audio_decoder_dim: 1024,
            audio_upsample_rates: vec![5, 5, 2, 2, 2, 2, 2],
            audio_upsample_kernels: vec![9, 9, 4, 4, 4, 4, 4],
            audio_resblock_kernels: vec![3, 7, 11],
            audio_resblock_dilations: vec![vec![1, 3, 5]; 3],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct H3ConditioningWorkspace {
    /// VAE/audio-reference encode substage only; TE/vision has its existing separate estimator.
    pub conditioning_bytes: u64,
    pub video_decode_bytes: u64,
    pub audio_decode_bytes: u64,
    /// Convenience maximum WITHOUT resident weights. Consumers with different resident component
    /// partitions must use the two decode fields separately before taking their maximum.
    pub decode_bytes: u64,
    /// Distinct delivered RGB/PCM host allocations, not process RSS. Metadata, already included
    /// once in accounted stage bytes on unified memory; excluded from CUDA device stage bytes.
    pub host_output_bytes: u64,
}

fn bad() -> Error {
    Error::Msg("H3 conditioning workspace: invalid or overflowing shape/config".into())
}
fn mul(v: &[u64]) -> Result<u64> {
    v.iter()
        .try_fold(1u64, |a, b| a.checked_mul(*b).ok_or_else(bad))
}
fn add(v: &[u64]) -> Result<u64> {
    v.iter()
        .try_fold(0u64, |a, b| a.checked_add(*b).ok_or_else(bad))
}
fn positive(v: &[u64]) -> Result<()> {
    if v.contains(&0) {
        Err(bad())
    } else {
        Ok(())
    }
}

impl H3ConditioningMemoryFacts {
    fn validate(&self) -> Result<()> {
        positive(&[
            self.video_element_bytes,
            self.latent_channels,
            self.pixel_channels,
            self.encoder_layers_per_block,
            self.clip_length,
            self.spatial_ratio,
            self.temporal_ratio,
            self.decoder_heads,
            self.decoder_head_dim,
            self.decoder_ffn_mult,
            self.tile_edge,
            self.audio_sample_rate,
            self.audio_output_channels,
            self.audio_latent_channels,
            self.audio_encoder_dim,
            self.audio_attention_heads,
            self.audio_decoder_dim,
        ])?;
        if ![2, 4].contains(&self.audio_decode_element_bytes)
            || ![2, 4].contains(&self.video_element_bytes)
            || self.encoder_channels.is_empty()
            || self.encoder_channels.len() != self.encoder_spatial_strides.len()
            || self.encoder_channels.len() != self.encoder_temporal_strides.len()
            || self.audio_encoder_rates.is_empty()
            || self.audio_upsample_rates.is_empty()
            || self.audio_upsample_rates.len() != self.audio_upsample_kernels.len()
            || self.audio_resblock_kernels.is_empty()
            || self.audio_resblock_kernels.len() != self.audio_resblock_dilations.len()
            || self.tile_overlap >= self.tile_edge
            || !self.tile_edge.is_multiple_of(self.spatial_ratio)
            || !self.tile_overlap.is_multiple_of(self.spatial_ratio)
            || self.token_drop >= self.clip_length.div_ceil(self.temporal_ratio)
        {
            return Err(bad());
        }
        positive(&self.encoder_channels)?;
        positive(&self.encoder_spatial_strides)?;
        positive(&self.encoder_temporal_strides)?;
        positive(&self.audio_encoder_rates)?;
        positive(&self.audio_upsample_rates)?;
        positive(&self.audio_upsample_kernels)?;
        positive(&self.audio_resblock_kernels)?;
        for d in &self.audio_resblock_dilations {
            positive(d)?;
            if d.is_empty() {
                return Err(bad());
            }
        }
        if mul(&self.encoder_spatial_strides)? != self.spatial_ratio
            || mul(&self.encoder_temporal_strides)? != self.temporal_ratio
            || mul(&self.audio_encoder_rates)? != mul(&self.audio_upsample_rates)?
        {
            return Err(bad());
        }
        Ok(())
    }
}

/// Counts and maximum lengths from native `spatial_tiling::TilePlan::split`. Slack distribution
/// moves tile STARTS only; for a multi-tile axis every tile retains the configured full length.
fn tile_axis(length: u64, f: &H3ConditioningMemoryFacts) -> Result<(u64, u64)> {
    positive(&[length])?;
    if length <= f.tile_edge {
        return Ok((1, length));
    }
    Ok((
        (length - f.tile_overlap).div_ceil(f.tile_edge - f.tile_overlap),
        f.tile_edge,
    ))
}

/// Conservatively lowered 3x3x3 convolution: padded input copies, output/tap accumulation,
/// contiguous layout copies and one im2col workspace. Candle lowers temporal taps serially into
/// conv2d and chunks im2col at 128M elements (one oversized frame is the minimum launch). MLX
/// uses native conv3d; the full 27-tap lowering bounds its source-shaped scratch without assuming
/// a particular optimized Metal kernel. No resident kernel/weight bytes are charged here.
fn cnn_conv(t: u64, h: u64, w: u64, ci: u64, co: u64, e: u64, lazy: bool) -> Result<u64> {
    let input = mul(&[t, h, w, ci])?;
    let padded = mul(&[add(&[t, 2])?, add(&[h, 2])?, add(&[w, 2])?, ci])?;
    let output = mul(&[t, h, w, co])?;
    let frame_cols = mul(&[h, w, ci, 9])?;
    let cols = if lazy {
        mul(&[input, 27])?
    } else {
        mul(&[t, frame_cols])?.min((128 << 20).max(frame_cols))
    };
    mul(&[
        add(&[
            mul(&[input, 2])?,
            mul(&[padded, 3])?,
            mul(&[output, 5])?,
            cols,
            // Candle conv3d_ncthw narrows one temporal weight tap then makes wk contiguous.
            // This is a temporary copied slice, not another resident component weight set.
            if lazy { 0 } else { mul(&[ci, co, 9])? },
        ])?,
        e,
    ])
}

/// One normalized image/clip, with already authoritative pixel geometry. Single frame bypasses
/// temporal clip padding. Longer clips repeat the last frame to `clip_length`, not to a guessed
/// ceil(frames/4) tensor. Working storage takes MAX over sequential CNN blocks, not depth SUM.
fn video_encode(frames: u64, h: u64, w: u64, f: &H3ConditioningMemoryFacts) -> Result<(u64, u64)> {
    positive(&[frames, h, w])?;
    let (nr, th) = tile_axis(h, f)?;
    let (nc, tw) = tile_axis(w, f)?;
    let clips = if frames == 1 {
        1
    } else {
        frames.div_ceil(f.clip_length)
    };
    let ct = if frames == 1 { 1 } else { f.clip_length };
    let padded_frames = mul(&[clips, ct])?;
    // Incoming F32 pixels + video-dtype cast and repeated-frame concatenation coexist.
    let inputs = mul(&[
        padded_frames,
        h,
        w,
        f.pixel_channels,
        4 + 2 * f.video_element_bytes,
    ])?;
    let (mut t, mut y, mut x, mut c) = (ct, th, tw, f.encoder_channels[0]);
    let mut active = cnn_conv(
        t,
        y,
        x,
        f.pixel_channels,
        c,
        f.video_element_bytes,
        f.lazy_encode,
    )?;
    for (i, &out) in f.encoder_channels.iter().enumerate() {
        // Norm: F32 cast/contiguous, centred, square, normalized, affine multiply/add,
        // contiguous return; two scalar reductions are bounded by another full F32 array.
        // Residual, video-dtype cast, SiLU and convolution buffers coexist conservatively.
        let norm = mul(&[t, y, x, c.max(out), 9 * 4 + 4 * f.video_element_bytes])?;
        active = active.max(add(&[
            norm,
            // Both conv2 of the width-changing residual and subsequent residual blocks
            // consume `out` channels, even when the first conv1 consumes fewer channels.
            cnn_conv(
                t,
                y,
                x,
                c.max(out),
                out,
                f.video_element_bytes,
                f.lazy_encode,
            )?,
        ])?);
        c = out;
        if f.encoder_spatial_strides[i] > 1 || f.encoder_temporal_strides[i] > 1 {
            active = active.max(cnn_conv(
                t,
                add(&[y, 1])?,
                add(&[x, 1])?,
                c,
                c,
                f.video_element_bytes,
                f.lazy_encode,
            )?);
            t = t.div_ceil(f.encoder_temporal_strides[i]);
            y = y.div_ceil(f.encoder_spatial_strides[i]);
            x = x.div_ceil(f.encoder_spatial_strides[i]);
        }
    }
    active = active.max(cnn_conv(
        t,
        y,
        x,
        c,
        2 * f.latent_channels,
        f.video_element_bytes,
        f.lazy_encode,
    )?);
    // All tile moments are allowed to remain reachable until stitching/temporal concatenation.
    // This overbounds BoundedStitch's strip retention, without assuming eval at encode tile exit.
    let moments = mul(&[
        clips,
        nr,
        nc,
        t,
        y,
        x,
        2,
        f.latent_channels,
        f.video_element_bytes,
    ])?;
    // Quant-conv/moments, stitched concat, posterior mean/logvar/std, seeded noise, normalized
    // condition rows and packed rows. Eight full posterior-sized buffers cover these copies;
    // token-drop/slicing can share a larger allocation, so retain pre-drop size intentionally.
    let retained = mul(&[moments, 8])?;
    Ok((
        add(&[inputs, active, retained, f.video_forward_extra_bytes])?,
        add(&[inputs, retained])?,
    ))
}

/// Alias-free Activation1d's actual storage forms at NLC [batch,length,channels]: replicate pad
/// (5 on each side for 12 taps/2x), zero insertion, full filtered output, trimmed 2x signal,
/// SnakeBeta's multiply/sin/square/divide/add, low-pass replicate pad/filter, output. Bounds the
/// CUDA native transpose-convolution too (which does not need the explicit zero insertion).
fn alias_free(length: u64, channels: u64, batch: u64, e: u64) -> Result<u64> {
    let padded = add(&[length, 10])?;
    let up = mul(&[padded, 2])?;
    let full = add(&[up, 11])?;
    let signal = mul(&[length, 2])?;
    let elements = add(&[
        length,
        padded,
        up,
        full,
        // Conservative 12-tap lowered filter scratch, reused between up/down filters.
        mul(&[full, 12])?,
        mul(&[signal, 7])?,
        add(&[signal, 11])?,
        length,
    ])?;
    mul(&[elements, channels, batch, e])
}

fn audio_encode(samples: u64, channels: u64, f: &H3ConditioningMemoryFacts) -> Result<(u64, u64)> {
    positive(&[samples, channels])?;
    let hop = mul(&f.audio_encoder_rates)?;
    let tokens = samples.div_ceil(hop);
    let padded = mul(&[tokens, hop])?;
    let inputs = mul(&[padded, channels, 4, 3])?; // deinterleave, pad, contiguous input
    let (mut length, mut c) = (padded, f.audio_encoder_dim);
    let mut peak = mul(&[length, channels, c, 4, 9])?; // k7 input conv + output
    for &rate in &f.audio_encoder_rates {
        // Three serial residual units (k7 dilations 1/3/9); Snake's elementary operations,
        // padded convolution input, im2col k7, both branch outputs and residual sum.
        peak = peak.max(mul(&[add(&[length, 54])?, channels, c, 4, 20])?);
        let next_c = mul(&[c, 2])?;
        peak = peak.max(mul(&[
            length,
            channels,
            next_c,
            4,
            add(&[mul(&[rate, 2])?, 5])?,
        ])?);
        length = length.div_ceil(rate);
        c = next_c;
    }
    // Causal pre_block: normalized input branches, Q/K/V + head layouts/attended output,
    // score/scale/softmax F32 buffers; channels are BATCH items, not concatenated samples.
    let attention = add(&[
        mul(&[channels, tokens, c, 4, 12])?,
        mul(&[channels, f.audio_attention_heads, tokens, tokens, 4, 3])?,
        mul(&[channels, tokens, f.audio_latent_channels, 4, 20])?,
        // One Candle attention projection's contiguous transposed weight scratch.
        if f.lazy_encode { 0 } else { mul(&[c, c, 4])? },
    ])?;
    peak = peak.max(attention);
    let retained = mul(&[channels, tokens, f.audio_latent_channels, 4, 8])?;
    Ok((
        add(&[inputs, peak, retained, f.audio_encode_forward_extra_bytes])?,
        retained,
    ))
}

#[cfg(test)]
#[cfg(test)]
fn video_decode(target: [u64; 3], frames: u64, f: &H3ConditioningMemoryFacts) -> Result<u64> {
    video_decode_with_patch(target, frames, f, [1, 2, 2])
}

fn video_decode_with_patch(
    target: [u64; 3],
    frames: u64,
    f: &H3ConditioningMemoryFacts,
    patch: [usize; 3],
) -> Result<u64> {
    let [lt, ph, pw] = target;
    let h = mul(&[ph, 16, patch[1] as u64])?;
    let w = mul(&[pw, 16, patch[2] as u64])?;
    let (_, th) = tile_axis(h, f)?;
    let (cols, tw) = tile_axis(w, f)?;
    let chunk = f.clip_length.div_ceil(f.temporal_ratio);
    let overlap = (chunk - f.token_drop % chunk) % chunk;
    let tile_t = add(&[chunk, overlap])?;
    let seq = add(&[
        mul(&[tile_t, th / f.spatial_ratio, tw / f.spatial_ratio])?,
        f.decoder_suffix_tokens,
    ])?;
    let dim = mul(&[f.decoder_heads, f.decoder_head_dim])?;
    // Hidden/residual/norm/scale, QKV, normalized+rotated QK, contiguous head layouts,
    // attention output/projection: <=24 token-width arrays. Gated FFN has [2*inner] w1,
    // SiLU + product + split copies (<=6*inner). RoPE tables/ids use F32 regardless of dtype.
    let block = if f.decoder_layers == 0 {
        0
    } else {
        add(&[
            mul(&[
                seq,
                dim,
                f.video_element_bytes,
                add(&[24, mul(&[6, f.decoder_ffn_mult])?])?,
            ])?,
            mul(&[f.decoder_heads, seq, seq, 4, 3])?,
        ])?
    };
    // Only one tile's transformer stack runs at once: MLX explicitly evals every decoded tile;
    // Candle eager forward overwrites hidden each block. Depth selects presence, not a 36x sum.
    let rotary = mul(&[seq, add(&[mul(&[f.decoder_head_dim, 2])?, 3])?, 4])?;
    let tile_pixels = mul(&[
        tile_t,
        f.temporal_ratio,
        th,
        tw,
        f.pixel_channels,
        f.video_element_bytes,
    ])?;
    let output_projection = mul(&[
        seq,
        f.pixel_channels,
        f.temporal_ratio,
        f.spatial_ratio,
        f.spatial_ratio,
        f.video_element_bytes,
    ])?;
    // Spatial stitch holds row strips, then decode_temporal retains previous clips/overlap and
    // concatenates. Use padded pixel-frame envelope rather than dropping cropped backing storage.
    let output = mul(&[
        add(&[frames, mul(&[f.clip_length, 2])?])?,
        h,
        w,
        f.pixel_channels,
        f.video_element_bytes,
    ])?;
    let strips = mul(&[cols, tile_pixels, 4])?;
    let latent = mul(&[
        add(&[lt, chunk])?,
        ph,
        patch[1] as u64,
        pw,
        patch[2] as u64,
        f.latent_channels,
        f.video_element_bytes,
        4,
    ])?;
    let converted = mul(&[frames, h, w, f.pixel_channels, 4 * 4 + 2])?;
    // Candle nn::linear explicitly materializes w.t().contiguous() before matmul.
    // One projection's transient transformed weight coexists with the original resident weight.
    // MLX's transpose is a view; no extra resident weight is invented for that path.
    let patch_features = mul(&[
        f.pixel_channels,
        f.temporal_ratio,
        f.spatial_ratio,
        f.spatial_ratio,
    ])?;
    let projection_columns = patch_features
        .max(f.latent_channels)
        .max(if f.decoder_layers > 0 {
            mul(&[2, f.decoder_ffn_mult, dim])?
        } else {
            0
        });
    let transformed_weight = if f.lazy_encode {
        0
    } else {
        mul(&[dim, projection_columns, f.video_element_bytes])?
    };
    add(&[
        f.video_forward_extra_bytes,
        transformed_weight,
        block,
        rotary,
        output_projection,
        tile_pixels,
        strips,
        mul(&[output, 4])?,
        latent,
        converted,
    ])
}

fn audio_decode(frames: u64, f: &H3ConditioningMemoryFacts) -> Result<(u64, u64)> {
    let hop = mul(&f.audio_upsample_rates)?;
    // JointGeometry uses nearest-integer audio tokens (not ceil). For positive fractions,
    // round-half-up is exact integer arithmetic; shipped lattice has no half ties.
    let denominator = mul(&[24, hop])?;
    let tokens = add(&[mul(&[frames, f.audio_sample_rate])?, denominator / 2])? / denominator;
    let e = f.audio_decode_element_bytes;
    let batch = f.audio_output_channels; // decode_stereo folds channels into batch
    let trunk = mul(&[
        f.audio_encoder_dim,
        1u64.checked_shl(f.audio_encoder_rates.len() as u32)
            .ok_or_else(bad)?,
    ])?;
    let mut peak = mul(&[tokens, batch, trunk, e, 3])?;
    let (mut length, mut c) = (tokens, f.audio_decoder_dim);
    for (i, &rate) in f.audio_upsample_rates.iter().enumerate() {
        let kernel = f.audio_upsample_kernels[i];
        let up_length = mul(&[length, rate])?;
        // MLX explicit zero-insert transpose conv, full conv and crop; c is the INPUT width.
        peak = peak.max(mul(&[
            add(&[up_length, kernel])?,
            batch,
            c,
            e,
            add(&[kernel, 4])?,
        ])?);
        c /= 2;
        positive(&[c])?;
        length = up_length;
        let mut stage = alias_free(length, c, batch, e)?;
        for (k, dilations) in f
            .audio_resblock_kernels
            .iter()
            .zip(&f.audio_resblock_dilations)
        {
            for d in dilations {
                let padded = add(&[length, mul(&[k - 1, *d])?])?;
                let conv = mul(&[padded, c, batch, e, add(&[*k, 4])?])?;
                stage = stage.max(add(&[alias_free(length, c, batch, e)?, conv])?);
            }
        }
        // Serial AMP blocks retain x + running acc + y + next add/divide result, not all layers.
        peak = peak.max(add(&[stage, mul(&[length, c, batch, e, 5])?])?);
    }
    peak = peak.max(alias_free(length, c, batch, e)?);
    let pcm = mul(&[length, batch, 4])?;
    // denormalized latent, mono-folded decoder, unclamped/clamped waveform and interleave copies.
    Ok((
        add(&[
            f.audio_decode_forward_extra_bytes,
            peak,
            mul(&[pcm, 4])?,
            mul(&[tokens, batch, f.audio_latent_channels, e, 3])?,
            if f.lazy_encode {
                0
            } else {
                mul(&[trunk, f.audio_latent_channels, e])?
            },
        ])?,
        pcm,
    ))
}

/// Price the exact admitted candidate request. Geometry remains owned by the existing H3 packed
/// row helper: short-edge rounding, keyframe canvas, video fit and temporal reference envelopes
/// must not diverge here. The helper's grids are DiT patches (32px); VAE latents are 16px for the
/// shipped config. Clip reference duration is deliberately its existing full-request envelope.
pub fn workspace(
    req: &Request<'_>,
    f: &H3ConditioningMemoryFacts,
) -> Result<H3ConditioningWorkspace> {
    workspace_with_patch(req, f, [1, 2, 2])
}

/// Selected DiT patches affect packed row counts, not the VAE's physical image dimensions.
pub fn workspace_with_patch(
    req: &Request<'_>,
    f: &H3ConditioningMemoryFacts,
    patch: [usize; 3],
) -> Result<H3ConditioningWorkspace> {
    f.validate()?;
    let (target, refs) = h3_visual_grids_with_patch(*req, patch)?;
    let mut transient = 0;
    let mut retained = 0;
    for grid in refs {
        let frames = if grid[0] == 1 {
            1
        } else {
            u64::from(req.frames)
        };
        let (peak, keep) = video_encode(
            frames,
            mul(&[grid[1], 16, patch[1] as u64])?,
            mul(&[grid[2], 16, patch[2] as u64])?,
            f,
        )?;
        transient = transient.max(peak - keep);
        retained = add(&[retained, keep])?;
    }
    for c in req.conditioning {
        let audio = match c {
            Conditioning::ReferenceAudio { audio, .. } => Some(audio),
            Conditioning::ReferenceVideo { audio, .. } => audio.as_ref(),
            _ => None,
        };
        if let Some(a) = audio {
            positive(&[a.sample_rate as u64, a.channels as u64])?;
            // AudioTrack carries interleaved samples. Native demands the VAE sample rate; it
            // does not resample silently. Each channel becomes a distinct encoder batch item.
            if a.sample_rate as u64 != f.audio_sample_rate
                || a.samples.len() % a.channels as usize != 0
            {
                return Err(bad());
            }
            let (peak, keep) = audio_encode(
                (a.samples.len() / a.channels as usize) as u64,
                a.channels as u64,
                f,
            )?;
            transient = transient.max(peak - keep);
            retained = add(&[retained, keep])?;
        }
    }
    let encode = add(&[transient, retained])?;
    let video = video_decode_with_patch(target, req.frames as u64, f, patch)?;
    let (audio, pcm) = audio_decode(req.frames as u64, f)?;
    let rgb = mul(&[
        req.frames as u64,
        target[1],
        16,
        patch[1] as u64,
        target[2],
        16,
        patch[2] as u64,
        f.pixel_channels,
    ])?;
    // fit_audio_to_video pads/trims to round(duration*sample_rate), which can be a
    // few hundred samples longer than the independently rounded latent-token decode.
    let delivered_samples = add(&[mul(&[req.frames as u64, f.audio_sample_rate])?, 12])? / 24;
    let host_pcm = pcm.max(mul(&[delivered_samples, f.audio_output_channels, 4])?);
    let host = add(&[rgb, host_pcm])?;
    let video = add(&[video, if f.unified_memory { rgb } else { 0 }])?;
    let audio = add(&[audio, if f.unified_memory { host } else { 0 }])?;
    Ok(H3ConditioningWorkspace {
        conditioning_bytes: encode,
        video_decode_bytes: video,
        audio_decode_bytes: audio,
        decode_bytes: video.max(audio),
        host_output_bytes: host,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_patch_changes_rows_not_vae_pixels_or_generated_stereo() {
        let mut req = crate::GenerationRequest {
            width: 1344,
            height: 768,
            frames: Some(124),
            ..Default::default()
        };
        req.conditioning.push(Conditioning::Reference {
            image: crate::Image {
                width: 320,
                height: 576,
                pixels: vec![],
            },
            strength: None,
        });
        let facts = H3ConditioningMemoryFacts::default();
        let stock = workspace(&Request::from(&req), &facts).unwrap();
        let unit = workspace_with_patch(&Request::from(&req), &facts, [1, 1, 1]).unwrap();
        assert_eq!(
            stock, unit,
            "VAE work sees identical physical resolution on either DiT patch"
        );
    }

    #[test]
    fn actual_config_and_tile_temporal_shapes() {
        let f = H3ConditioningMemoryFacts::default();
        f.validate().unwrap();
        assert_eq!(tile_axis(768, &f).unwrap(), (4, 256));
        assert_eq!(tile_axis(1344, &f).unwrap(), (7, 256));
        assert_eq!(tile_axis(128, &f).unwrap(), (1, 128));
        let chunk = f.clip_length.div_ceil(f.temporal_ratio);
        assert_eq!(chunk + (chunk - f.token_drop % chunk) % chunk, 7);
        assert_eq!(mul(&f.audio_encoder_rates).unwrap(), 800);
    }
    #[test]
    fn keyframe_and_video_encode_are_positive_and_geometry_sensitive() {
        let f = H3ConditioningMemoryFacts::default();
        let image = video_encode(1, 256, 256, &f).unwrap();
        let bigger = video_encode(1, 768, 1344, &f).unwrap();
        let clip = video_encode(124, 768, 1344, &f).unwrap();
        assert!(image.0 > image.1 && image.1 > 0);
        assert!(bigger.0 > image.0 && bigger.1 > image.1);
        assert!(clip.0 > bigger.0 && clip.1 > bigger.1);
    }
    #[test]
    fn serial_refs_max_work_plus_retained_outputs_not_sum_of_work() {
        let f = H3ConditioningMemoryFacts::default();
        let (a, ka) = video_encode(1, 256, 256, &f).unwrap();
        let (b, kb) = video_encode(1, 768, 1344, &f).unwrap();
        let together = (a - ka).max(b - kb) + ka + kb;
        assert!(together > b);
        assert!(together < a + b);
        assert_eq!(together - b, ka);
    }
    #[test]
    fn audio_channels_are_f32_batch_and_not_longer_mono() {
        let f = H3ConditioningMemoryFacts::default();
        let mono = audio_encode(32_000, 1, &f).unwrap();
        let stereo = audio_encode(32_000, 2, &f).unwrap();
        assert_eq!(stereo, (mono.0 * 2, mono.1 * 2));
        let mut bf16 = f.clone();
        bf16.video_element_bytes = 2;
        assert_eq!(audio_encode(32_000, 2, &bf16).unwrap(), stereo);
        assert_eq!(
            audio_decode(124, &bf16).unwrap(),
            audio_decode(124, &f).unwrap()
        );
        assert_eq!(stereo.1, 2 * 40 * 32 * 4 * 8);
        assert!(audio_encode(64_000, 2, &f).unwrap().0 > stereo.0);
    }
    #[test]
    fn decoder_present_without_refs_and_uses_full_chunk_not_ceil_frames() {
        let f = H3ConditioningMemoryFacts::default();
        let small = video_decode([37, 10, 16], 124, &f).unwrap();
        let large = video_decode([37, 24, 42], 124, &f).unwrap();
        assert!(small > 0 && large > small);
        assert!(video_decode([102, 24, 42], 345, &f).unwrap() > large);
        let mut half = f.clone();
        half.video_element_bytes = 2;
        assert!(large > video_decode([37, 24, 42], 124, &half).unwrap());
        // Scores/RoPE stay F32, so halving all allocation bytes would undercharge.
        assert!(video_decode([37, 24, 42], 124, &half).unwrap() * 2 > large);
    }
    #[test]
    fn decoder_depth_is_not_a_blind_layer_sum() {
        let f = H3ConditioningMemoryFacts::default();
        let mut deeper = f.clone();
        deeper.decoder_layers *= 2;
        assert_eq!(
            video_decode([37, 24, 42], 124, &f).unwrap(),
            video_decode([37, 24, 42], 124, &deeper).unwrap()
        );
        deeper.decoder_layers = 0;
        assert!(
            video_decode([37, 24, 42], 124, &deeper).unwrap()
                < video_decode([37, 24, 42], 124, &f).unwrap()
        );
    }
    fn request(conditioning: &[Conditioning]) -> Request<'_> {
        Request {
            veda: None,
            width: 1344,
            height: 768,
            frames: 124,
            count: 1,
            prompt: "fixture",
            conditioning,
            reference_image_short_edge: Some(256),
        }
    }
    #[test]
    fn production_no_refs_has_zero_encode_and_both_decoders() {
        let f = H3ConditioningMemoryFacts::default();
        let bound = workspace(&request(&[]), &f).unwrap();
        assert_eq!(bound.conditioning_bytes, 0);
        assert!(bound.video_decode_bytes > 0 && bound.audio_decode_bytes > 0);
        assert_eq!(
            bound.decode_bytes,
            bound.video_decode_bytes.max(bound.audio_decode_bytes)
        );
    }
    #[test]
    fn production_refs_use_admitted_canvas_shortedge_and_serial_retention() {
        let f = H3ConditioningMemoryFacts::default();
        let image = crate::Image {
            width: 320,
            height: 320,
            pixels: vec![],
        };
        let refs = [Conditioning::Reference {
            image: image.clone(),
            strength: None,
        }];
        let one = workspace(&request(&refs), &f).unwrap();
        assert!(one.conditioning_bytes > 0);
        let mut large = request(&refs);
        large.reference_image_short_edge = Some(512);
        assert!(workspace(&large, &f).unwrap().conditioning_bytes > one.conditioning_bytes);
        let refs2 = [Conditioning::MultiReference {
            images: vec![image.clone(), image],
        }];
        let two = workspace(&request(&refs2), &f).unwrap();
        assert!(two.conditioning_bytes > one.conditioning_bytes);
        assert!(two.conditioning_bytes < one.conditioning_bytes * 2);
        assert_eq!(two.video_decode_bytes, one.video_decode_bytes);
    }
    #[test]
    fn production_first_last_and_ordered_image_video_audio_refs() {
        let f = H3ConditioningMemoryFacts::default();
        let image = crate::Image {
            width: 320,
            height: 320,
            pixels: vec![],
        };
        let anchors = [
            Conditioning::Keyframe {
                image: image.clone(),
                frame_idx: 0,
                strength: 1.0,
            },
            Conditioning::Keyframe {
                image: image.clone(),
                frame_idx: 123,
                strength: 1.0,
            },
        ];
        let first = workspace(&request(&anchors[..1]), &f).unwrap();
        let last = workspace(&request(&anchors[1..]), &f).unwrap();
        assert_eq!(first, last);
        assert!(
            workspace(&request(&anchors), &f)
                .unwrap()
                .conditioning_bytes
                > first.conditioning_bytes
        );
        let audio = crate::AudioTrack {
            samples: vec![0.0; 64_000],
            sample_rate: 32_000,
            channels: 2,
            stems: vec![],
        };
        let refs = [
            Conditioning::Reference {
                image: image.clone(),
                strength: None,
            },
            Conditioning::ReferenceVideo {
                frames: vec![image],
                fps: 24.0,
                audio: Some(audio.clone()),
            },
            Conditioning::ReferenceAudio {
                audio,
                strength: None,
            },
        ];
        let all = workspace(&request(&refs), &f).unwrap();
        let reversed = [refs[2].clone(), refs[1].clone(), refs[0].clone()];
        assert_eq!(all, workspace(&request(&reversed), &f).unwrap());
        // Ordering is semantic at runtime, not a reason to drop any carrier from memory pricing.
        assert!(
            all.conditioning_bytes
                > workspace(&request(&refs[..1]), &f)
                    .unwrap()
                    .conditioning_bytes
        );
        assert!(
            workspace(&request(&refs[2..]), &f)
                .unwrap()
                .conditioning_bytes
                > 0
        );
    }
    #[test]
    fn production_host_allocations_only_enter_unified_stage_budget() {
        let mut f = H3ConditioningMemoryFacts::default();
        let unified = workspace(&request(&[]), &f).unwrap();
        f.unified_memory = false;
        let device = workspace(&request(&[]), &f).unwrap();
        let rgb = 124 * 1344 * 768 * 3;
        assert_eq!(unified.video_decode_bytes - device.video_decode_bytes, rgb);
        assert_eq!(
            unified.audio_decode_bytes - device.audio_decode_bytes,
            unified.host_output_bytes
        );
        assert_eq!(device.host_output_bytes, unified.host_output_bytes);
    }
    #[test]
    fn selected_audio_decode_dtype_does_not_change_f32_encoder() {
        let f = H3ConditioningMemoryFacts::default();
        let mut bf16 = f.clone();
        bf16.audio_decode_element_bytes = 2;
        assert!(audio_decode(124, &bf16).unwrap().0 < audio_decode(124, &f).unwrap().0);
        assert_eq!(
            audio_encode(32_000, 2, &bf16).unwrap(),
            audio_encode(32_000, 2, &f).unwrap()
        );
    }
    #[test]
    fn forward_cast_fusion_storage_overlaps_decoder_work_not_load_only_max() {
        let base = H3ConditioningMemoryFacts::default();
        let original = workspace(&request(&[]), &base).unwrap();
        let mut derived = base.clone();
        derived.video_forward_extra_bytes = 13 << 20;
        derived.audio_encode_forward_extra_bytes = 17 << 20;
        derived.audio_decode_forward_extra_bytes = 19 << 20;
        let with_extra = workspace(&request(&[]), &derived).unwrap();
        assert_eq!(with_extra.conditioning_bytes, 0);
        assert_eq!(
            with_extra.video_decode_bytes - original.video_decode_bytes,
            derived.video_forward_extra_bytes
        );
        assert_eq!(
            with_extra.audio_decode_bytes - original.audio_decode_bytes,
            derived.audio_decode_forward_extra_bytes
        );
        assert_eq!(with_extra.host_output_bytes, original.host_output_bytes);
    }
    #[test]
    fn forward_cast_fusion_storage_is_one_serial_encode_extra_not_per_reference_sum() {
        let base = H3ConditioningMemoryFacts::default();
        let mut derived = base.clone();
        derived.video_forward_extra_bytes = 64 << 20;
        derived.audio_encode_forward_extra_bytes = 128 << 20;
        let image = crate::Image {
            width: 320,
            height: 320,
            pixels: vec![],
        };
        let refs = [Conditioning::MultiReference {
            images: vec![image.clone(), image],
        }];
        let original = workspace(&request(&refs), &base).unwrap();
        let with_extra = workspace(&request(&refs), &derived).unwrap();
        assert_eq!(
            with_extra.conditioning_bytes - original.conditioning_bytes,
            derived.video_forward_extra_bytes
        );
        let audio = [Conditioning::ReferenceAudio {
            audio: crate::AudioTrack {
                samples: vec![0.0; 64_000],
                sample_rate: 32_000,
                channels: 2,
                stems: vec![],
            },
            strength: None,
        }];
        let original_audio = workspace(&request(&audio), &base).unwrap();
        let with_audio_extra = workspace(&request(&audio), &derived).unwrap();
        assert_eq!(
            with_audio_extra.conditioning_bytes - original_audio.conditioning_bytes,
            derived.audio_encode_forward_extra_bytes
        );
        let mixed = [refs[0].clone(), audio[0].clone()];
        let visual = video_encode(1, 256, 256, &base).unwrap();
        let wave = audio_encode(32_000, 2, &base).unwrap();
        let expected = (visual.0 - visual.1 + derived.video_forward_extra_bytes)
            .max(wave.0 - wave.1 + derived.audio_encode_forward_extra_bytes)
            + 2 * visual.1
            + wave.1;
        assert_eq!(
            workspace(&request(&mixed), &derived)
                .unwrap()
                .conditioning_bytes,
            expected
        );
    }
    #[test]
    fn invalid_and_overflow_shapes_are_refused_without_allocations() {
        let mut f = H3ConditioningMemoryFacts::default();
        f.encoder_temporal_strides[1] = 0;
        assert!(f.validate().is_err());
        f = H3ConditioningMemoryFacts::default();
        f.tile_overlap = f.tile_edge;
        assert!(f.validate().is_err());
        assert!(video_encode(
            u64::MAX,
            u64::MAX,
            1344,
            &H3ConditioningMemoryFacts::default()
        )
        .is_err());
        assert!(audio_encode(u64::MAX, 2, &H3ConditioningMemoryFacts::default()).is_err());
    }
}
