//! The assembled Synchformer visual encoder (MMAudio's `vfeat_extractor`, a MotionFormer).
//!
//! Forward: `(S, C=3, T=16, H=224, W=224)` segments → 3D patch embed → prepend CLS → separate
//! spatio-temporal positional embeddings → 12 divided space-time blocks → final LayerNorm → drop
//! CLS → restore `(S, D, t=8, h=14, w=14)` → spatial aggregation (CLS-pool the 14×14 grid per
//! frame) → **`(S, t=8, 768)`** per-segment sync features. Temporal aggregation is `Identity` in
//! MMAudio's config, so the 8 temporal tokens per segment are retained (not collapsed).

use candle_audio::candle_core::{DType, Device, Result as CResult, Tensor};
use candle_nn::{layer_norm, LayerNorm, Module, VarBuilder};

use crate::agg::SpatialAggLayer;
use crate::blocks::{DividedSpaceTimeBlock, PatchEmbed3d};
use crate::config;

/// The MotionFormer visual feature extractor, weights already resolved.
pub struct SynchformerVisualEncoder {
    patch_embed_3d: PatchEmbed3d,
    cls_token: Tensor,  // (1, 1, D)
    pos_embed: Tensor,  // (1, 197, D)  — spatial (CLS + 196)
    temp_embed: Tensor, // (1, 8, D)   — temporal
    blocks: Vec<DividedSpaceTimeBlock>,
    norm: LayerNorm,
    spatial_attn_agg: SpatialAggLayer,
    device: Device,
}

impl SynchformerVisualEncoder {
    /// Load the encoder from a `VarBuilder` rooted at the `vfeat_extractor.` sub-tree of the
    /// `synchformer_state_dict.pth` checkpoint.
    pub fn load(vb: VarBuilder, device: Device) -> CResult<Self> {
        let d = config::EMBED_DIM;
        let patch_embed_3d = PatchEmbed3d::load(vb.pp("patch_embed_3d"))?;
        let cls_token = vb.get((1, 1, d), "cls_token")?;
        // pos_embed covers CLS + 196 spatial patches.
        let pos_embed = vb.get((1, config::NUM_SPATIAL_PATCHES + 1, d), "pos_embed")?;
        let temp_embed = vb.get((1, config::TEMPORAL_RESOLUTION, d), "temp_embed")?;
        let mut blocks = Vec::with_capacity(config::DEPTH);
        for i in 0..config::DEPTH {
            blocks.push(DividedSpaceTimeBlock::load(vb.pp("blocks").pp(i))?);
        }
        let norm = layer_norm(d, config::LN_EPS, vb.pp("norm"))?;
        let spatial_attn_agg = SpatialAggLayer::load(vb.pp("spatial_attn_agg"))?;
        Ok(Self {
            patch_embed_3d,
            cls_token,
            pos_embed,
            temp_embed,
            blocks,
            norm,
            spatial_attn_agg,
            device,
        })
    }

    /// The compute device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Add the separate spatio-temporal positional embeddings (`POS_EMBED == "separate"`):
    /// `total = tile(spatial, T) + repeat_interleave(temporal, 196)`, CLS pos prepended.
    fn add_pos_embed(&self, x: &Tensor) -> CResult<Tensor> {
        let d = config::EMBED_DIM;
        let t = config::TEMPORAL_RESOLUTION;
        let n = config::NUM_SPATIAL_PATCHES;
        let cls_embed = self.pos_embed.narrow(1, 0, 1)?; // (1,1,D)
        let spatial = self.pos_embed.narrow(1, 1, n)?; // (1,196,D)
                                                       // tile_pos: repeat the 196 spatial embeds T times → [s0..s195]×T.
        let tile_pos = spatial
            .unsqueeze(1)? // (1,1,196,D)
            .broadcast_as((1, t, n, d))?
            .contiguous()?
            .reshape((1, t * n, d))?;
        // tile_temp: repeat_interleave each temporal embed 196 times → [t0×196, t1×196, ...].
        let tile_temp = self
            .temp_embed
            .unsqueeze(2)? // (1,8,1,D)
            .broadcast_as((1, t, n, d))?
            .contiguous()?
            .reshape((1, t * n, d))?;
        let total = (tile_pos + tile_temp)?;
        let total = Tensor::cat(&[&cls_embed, &total], 1)?; // (1, 1+T·196, D)
        x.broadcast_add(&total)
    }

    /// `(BS, C, T, H, W)` → `(BS, 1+F·196, D)` backbone features (CLS retained).
    fn forward_features(&self, x: &Tensor) -> CResult<Tensor> {
        let tokens = self.patch_embed_3d.forward(x)?; // (BS, 1568, D)
        let bs = tokens.dim(0)?;
        let cls = self.cls_token.broadcast_as((bs, 1, config::EMBED_DIM))?;
        let x = Tensor::cat(&[&cls, &tokens], 1)?; // (BS, 1569, D)
        let mut x = self.add_pos_embed(&x)?;
        for blk in &self.blocks {
            x = blk.forward(&x, config::TEMPORAL_RESOLUTION, config::NUM_SPATIAL_PATCHES)?;
        }
        Ok(x)
    }

    /// Encode `(S, C=3, T=16, H=224, W=224)` segments → `(S, t=8, D=768)` sync features.
    pub fn encode(&self, segments: &Tensor) -> CResult<Tensor> {
        let (s, _c, _t, _h, _w) = segments.dims5()?;
        let x = self.forward_features(segments)?; // (S, 1569, D)
                                                  // Drop CLS, final norm.
        let x = x.narrow(1, 1, x.dim(1)? - 1)?; // (S, 1568, D)
        let x = self.norm.forward(&x)?;
        // restore_spatio_temp_dims: (S, 1568, D) → (S, D, t, h, w).
        let d = config::EMBED_DIM;
        let (t, grid) = (config::TEMPORAL_RESOLUTION, config::GRID);
        let x = x.transpose(1, 2)?.contiguous()?; // (S, D, 1568)
        let x = x.reshape((s, d, t, grid, grid))?; // (S, D, t, h, w)
                                                   // spatial_attn_agg: 'BS D t h w -> (BS t) (h w) D'.
        let x = x
            .permute((0, 2, 3, 4, 1))? // (S, t, h, w, D)
            .contiguous()?
            .reshape((s * t, grid * grid, d))?; // (S·t, 196, D)
        let x = self.spatial_attn_agg.forward(&x)?; // (S·t, D)
                                                    // reshape back to (S, t, D); temp_attn_agg = Identity.
        x.reshape((s, t, d))
    }

    /// Cast a preprocessed segment tensor to the encoder's compute dtype/device (f32).
    pub fn prepare_input(&self, segments: &Tensor) -> CResult<Tensor> {
        segments.to_device(&self.device)?.to_dtype(DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::Shape;
    use candle_nn::var_builder::SimpleBackend;
    use candle_nn::VarBuilder;

    /// A deterministic pseudo-random weight backend: every requested `(shape, name)` yields a small
    /// N(0, ~0.02)-scale tensor seeded by a hash of `name`, so two builds produce **identical**
    /// weights (determinism) while the weights are non-degenerate (unlike the zeros default), which
    /// is what lets the structural forward test prove the graph is input-sensitive.
    struct DetRandBackend;

    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn seed_of(name: &str) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for b in name.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    impl SimpleBackend for DetRandBackend {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: candle_nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> CResult<Tensor> {
            let n = s.elem_count();
            let mut state = seed_of(name);
            let mut data = Vec::with_capacity(n);
            for _ in 0..n {
                state = splitmix64(state);
                // map to [-0.03, 0.03]
                let u = (state >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
                data.push(((u - 0.5) * 0.06) as f32);
            }
            Tensor::from_vec(data, s, dev)?.to_dtype(dtype)
        }
        fn get_unchecked(&self, _name: &str, _dtype: DType, _dev: &Device) -> CResult<Tensor> {
            candle_audio::candle_core::bail!("unchecked get not supported in DetRandBackend")
        }
        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    fn det_encoder() -> SynchformerVisualEncoder {
        let dev = Device::Cpu;
        let vb = VarBuilder::from_backend(Box::new(DetRandBackend), DType::F32, dev.clone());
        SynchformerVisualEncoder::load(vb.pp("vfeat_extractor"), dev).expect("build det encoder")
    }

    /// One synthetic segment tensor `(S, C, T, H, W)` whose pixel content is a deterministic
    /// function of `fill` — different `fill` ⇒ genuinely different frame content.
    fn segments(s: usize, fill: f32, dev: &Device) -> Tensor {
        let (c, t, h, w) = (
            config::IN_CHANS,
            config::NUM_FRAMES,
            config::IMG_SIZE,
            config::IMG_SIZE,
        );
        let n = s * c * t * h * w;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            // a smooth spatially/temporally varying pattern scaled by `fill`
            let v = ((i as f32) * 0.0001 + fill).sin() * 0.5;
            data.push(v);
        }
        Tensor::from_vec(data, (s, c, t, h, w), dev).unwrap()
    }

    #[test]
    fn forward_shape_finite_and_deterministic() {
        let enc = det_encoder();
        let dev = Device::Cpu;
        let x = segments(2, 0.3, &dev);
        let out1 = enc.encode(&x).expect("encode");
        assert_eq!(
            out1.dims(),
            &[2, config::TEMPORAL_RESOLUTION, config::EMBED_DIM],
            "output must be (S, 8, 768)"
        );
        // finite
        let flat = out1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "features must be finite"
        );
        // deterministic: same weights + same input ⇒ byte-identical
        let out2 = enc.encode(&x).expect("encode again");
        let flat2 = out2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat, flat2, "encoder must be deterministic run-to-run");
    }

    #[test]
    fn features_are_frame_varying() {
        let enc = det_encoder();
        let dev = Device::Cpu;
        let a = enc
            .encode(&segments(1, 0.1, &dev))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = enc
            .encode(&segments(1, 2.0, &dev))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Different frame content must yield materially different features (not a constant map).
        let max_abs_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs_diff > 1e-4,
            "different frames must produce different features (max|Δ|={max_abs_diff})"
        );
        // And the features themselves must not be constant across the feature dim.
        let var = {
            let mean = a.iter().sum::<f32>() / a.len() as f32;
            a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / a.len() as f32
        };
        assert!(
            var > 1e-8,
            "features must not be a constant vector (var={var})"
        );
    }
}
