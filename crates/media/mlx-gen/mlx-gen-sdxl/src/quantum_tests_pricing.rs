//! Epic 16774: header-only iOS pricing (`crate::ios_pricing`) IS the guards' formulas, and the
//! levers that make every fully optimized SDXL-family configuration fit are priced, proven,
//! output-preserving and mutation-sensitive. Tiny random-weight models on MLX's CPU device inside
//! a child process that makes any Metal use fatal (the parent module's `cpu_child`).

use super::*;
use crate::ios_pricing::{ios_quantum_phase, Architecture, Inputs, Priced, Regime, Route};
use mlx_gen::gen_core::CfgBatching;
use mlx_gen::ios_pricing::IosPhase;
use std::path::PathBuf;

/// Latent edge of the tiny fixture's requests (two U-Net downsamples need a multiple of 4).
const LATENT: i32 = 8;
/// Text tokens per CLIP window of the tiny encoders (their position tables hold 8).
const TOKENS: i32 = 8;
/// The tiny VAE's spatial scale (two blocks, one downsample).
const VAE_SCALE: i32 = 2;
/// The tiny ControlNet conditioning embedding's stride (three stride-2 convolutions).
const CONTROL_SCALE: i32 = 8;

// ---------------------------------------------------------------------------------------------
// Every quantum label attributes to exactly one phase.
// ---------------------------------------------------------------------------------------------

/// Every `&str` label constant `quantum.rs` declares, read off its source so a label added later
/// is covered without editing this list.
fn label_constants() -> Vec<(String, String)> {
    include_str!("quantum.rs")
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("pub(crate) const ")?;
            let (name, value) = rest.split_once(": &str = \"")?;
            Some((name.to_owned(), value.strip_suffix("\";")?.to_owned()))
        })
        .collect()
}

#[test]
fn every_quantum_label_maps_to_its_phase() {
    let labels = label_constants();
    assert!(labels.len() >= 45, "read every label constant: {labels:?}");
    for (name, value) in &labels {
        assert!(
            ios_quantum_phase(value).is_some(),
            "{name} = {value:?} attributes to no phase"
        );
    }
    // Shared VAE code runs in both the encode and the decode: each pass checks its own labels.
    for (label, phase) in [
        (VAE_RESNET, IosPhase::Decode),
        (VAE_ATTENTION, IosPhase::Decode),
        (VAE_TILE, IosPhase::Decode),
        (VAE_TILED_STAGE, IosPhase::Decode),
        (VAE_ENCODE_RESNET, IosPhase::Encode),
        (VAE_ENCODE_ATTENTION, IosPhase::Encode),
        (VAE_ENCODE_TILE, IosPhase::Encode),
        (VAE_ENCODE_TILED_STAGE, IosPhase::Encode),
        (CONTROL_COND_CONV, IosPhase::Encode),
        (TRANSFORMER_BLOCK, IosPhase::Denoise),
        (VISION_LAYER, IosPhase::Conditioning),
        (UNET_LOAD, IosPhase::Load),
    ] {
        assert_eq!(ios_quantum_phase(label), Some(phase), "{label}");
    }
    assert_eq!(ios_quantum_phase(TILED_CORE), None);
}

// ---------------------------------------------------------------------------------------------
// The tiny on-disk snapshot.
// ---------------------------------------------------------------------------------------------

/// A tiny SDXL-family snapshot on disk: both CLIP towers, a U-Net, a staged fp16-fix-style VAE, a
/// ControlNet and an IP-Adapter, every weight stored f32 (the loads cast to their compute dtypes).
struct Snapshot {
    _dir: tempfile::TempDir,
    root: PathBuf,
    bits: Option<i32>,
    /// True only on the separate groupable-TE fixture; legacy TE8/32 fixtures stay unchanged.
    text_quantize: bool,
    context: i32,
    unet: UNetConfig,
    te1: ClipTextConfig,
    te2: ClipTextConfig,
    vae: crate::config::VaeConfig,
    vision: crate::vision_encoder::VisionConfig,
    resampler: crate::ip_adapter::ResamplerConfig,
}

impl Snapshot {
    fn new(transformer_layers: i32) -> Self {
        Self::build(transformer_layers, None)
    }

    /// Legacy U-Net/control packed fixture: the U-Net is pre-quantized (u32 codes with bf16
    /// scales/biases), dense leaves at f16, ControlNet packed at load. Its narrow CLIP towers
    /// are deliberately NOT a full production q-tier text-packing proof; keep that binding intact.
    fn packed(bits: i32) -> Self {
        Self::build(1, Some(bits))
    }

    /// Separate actual-config CLIP64+CLIP64 fixture (U-Net context128, pooled16), with either
    /// dense text files packed at load or already-packed text files. No real SDXL config injection.
    fn groupable_text(bits: Option<i32>, stored_packed: bool) -> Self {
        Self::build_with_text(1, bits, true, stored_packed)
    }

    fn build(transformer_layers: i32, bits: Option<i32>) -> Self {
        Self::build_with_text(transformer_layers, bits, false, false)
    }

    fn build_with_text(
        transformer_layers: i32,
        bits: Option<i32>,
        text_quantize: bool,
        stored_packed: bool,
    ) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let tiny = crate::test_support::tiny_control_unet_config();
        let (width, context, pooled) = match bits {
            None => (tiny.block_out_channels.clone(), 16, 16),
            Some(_) => (vec![64, 64, 128], 64, 16),
        };
        let context = if text_quantize { 128 } else { context };
        let unet = UNetConfig {
            transformer_layers_per_block: vec![transformer_layers; 3],
            block_out_channels: width,
            cross_attention_dim: vec![context; 3],
            projection_class_embeddings_input_dim: Some(pooled + 6 * 8),
            ..tiny
        };
        // Conditioning: the two towers' hidden widths concatenate to the U-Net's context; the
        // projected pooled output is its pooled input. Stored as the published snapshots store
        // them: the CLIP towers, U-Net, ControlNet and IP-Adapter at f16 (no load-time cast), the
        // fp16-fix VAE at f32 (cast to f16 at load).
        let (w1, te1) = tiny_clip(context / 2, 2, None, 300);
        let (w2, te2) = tiny_clip(context / 2, 2, Some(pooled), 400);
        let text_weights = |weights: Weights, cfg: &ClipTextConfig| {
            let weights = half(weights);
            match bits.filter(|_| stored_packed) {
                Some(bits) => {
                    let paths = crate::text_encoder::quantizable_text_paths(cfg);
                    let packed = mlx_gen::quant::quantize_map(
                        weights.into_tensors(),
                        bits,
                        crate::quant::GROUP_SIZE,
                        |name| paths.iter().any(|(target, _)| target == name),
                    )
                    .unwrap();
                    let mut weights = Weights::from_map(packed);
                    weights.cast_unpacked_floats(crate::model::DTYPE).unwrap();
                    weights
                }
                None => weights,
            }
        };
        save(
            &text_weights(w1, &te1),
            &root.join("text_encoder/model.safetensors"),
        );
        save(
            &text_weights(w2, &te2),
            &root.join("text_encoder_2/model.safetensors"),
        );
        let unet_w = half(crate::test_support::tiny_unet_weights_for(&unet, 7).unwrap());
        let unet_w = match bits {
            None => unet_w,
            Some(bits) => {
                let packed = mlx_gen::quant::quantize_map(
                    unet_w.into_tensors(),
                    bits,
                    crate::quant::GROUP_SIZE,
                    |_| true,
                )
                .unwrap();
                let mut w = Weights::from_map(packed);
                w.cast_unpacked_floats(Dtype::Float16).unwrap();
                w
            }
        };
        save(
            &unet_w,
            &root.join("unet/diffusion_pytorch_model.safetensors"),
        );
        if let Some(bits) = bits {
            // The published packed tiers' marker (`convert.rs`): what makes the load leave the
            // pre-quantized blocks lazy, so the contract declares rung 4 on it.
            std::fs::write(
                root.join("unet/config.json"),
                format!(
                    r#"{{"quantization": {{"bits": {bits}, "group_size": {}}}}}"#,
                    crate::quant::GROUP_SIZE
                ),
            )
            .unwrap();
        }
        let (vw, vae) = crate::vae::tests::fixture();
        save(
            &vw,
            &root.join("vae_fp16_fix/diffusion_pytorch_model.safetensors"),
        );
        save(
            &half(crate::test_support::tiny_controlnet_weights(&unet, 5).unwrap()),
            &root.join("controlnet/diffusion_pytorch_model.safetensors"),
        );
        let (vision, resampler, encoder_w, adapter_w) = tiny_ip_weights(&unet);
        save(
            &half(encoder_w),
            &root
                .join("ip")
                .join(crate::loader::IP_ADAPTER_IMAGE_ENCODER_FILE),
        );
        save(
            &half(adapter_w),
            &root
                .join("ip")
                .join(crate::loader::IP_ADAPTER_WEIGHT_FILES[1]),
        );
        Self {
            _dir: dir,
            root,
            bits,
            text_quantize,
            context,
            unet,
            te1,
            te2,
            vae,
            vision,
            resampler,
        }
    }

    fn arch(&self) -> Architecture {
        Architecture {
            unet: self.unet.clone(),
            vae: self.vae.clone(),
            te1: self.te1.clone(),
            te2: self.te2.clone(),
            vision: self.vision.clone(),
            resampler: self.resampler.clone(),
            text_tokens: TOKENS,
            context_dim: self.context,
            latent_scale: VAE_SCALE,
            control_scale: CONTROL_SCALE,
        }
    }

    fn file(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }

    /// The pricing inputs of a load of this snapshot (dense, streamable: the U-Net stays lazy).
    fn inputs(&self, route: Route) -> Inputs {
        Inputs {
            fused: None,
            arch: self.arch(),
            te1: self.file("text_encoder/model.safetensors"),
            te2: self.file("text_encoder_2/model.safetensors"),
            unet: self.file("unet/diffusion_pytorch_model.safetensors"),
            vae: (
                self.file("vae_fp16_fix/diffusion_pytorch_model.safetensors"),
                crate::loader::VAE_FP16_FIX_DTYPE,
            ),
            controls: if route == Route::Control {
                vec![self.file("controlnet/diffusion_pytorch_model.safetensors")]
            } else {
                Vec::new()
            },
            ip: (route == Route::IpAdapter).then(|| {
                (
                    self.root
                        .join("ip")
                        .join(crate::loader::IP_ADAPTER_IMAGE_ENCODER_FILE),
                    self.root
                        .join("ip")
                        .join(crate::loader::IP_ADAPTER_WEIGHT_FILES[1]),
                )
            }),
            quant_bits: self.bits,
            text_quant_bits: self.bits.filter(|_| self.text_quantize),
            streamable: true,
        }
    }
}

/// `w` with every tensor at f16, evaluated.
fn half(w: Weights) -> Weights {
    let keys: Vec<String> = w.keys().map(str::to_owned).collect();
    let map: HashMap<String, Array> = keys
        .into_iter()
        .map(|k| {
            let a = w.get(&k).unwrap().as_dtype(Dtype::Float16).unwrap();
            a.eval().unwrap();
            (k, a)
        })
        .collect();
    Weights::from_map(map)
}

/// A tiny ViT image tower and an IP-Adapter-plus Resampler whose output is the tiny U-Net's 16-wide
/// context, plus one decoupled K/V pair per cross-attention block in the install order
/// (down → up → mid) — `(tower config, Resampler config, tower weights, IP weights)`.
fn tiny_ip_weights(
    unet: &UNetConfig,
) -> (
    crate::vision_encoder::VisionConfig,
    crate::ip_adapter::ResamplerConfig,
    Weights,
    Weights,
) {
    // A tower far wider than the rest of the tiny bundle, as ViT-H is beside a streamed SDXL U-Net
    // (the lever's price must see whether it was released).
    const HIDDEN: i32 = 256;
    let vision = crate::vision_encoder::VisionConfig {
        hidden: HIDDEN,
        num_layers: 4,
        num_heads: 4,
        patch: 4,
        image_size: 16,
        num_channels: 3,
        quick_gelu: false,
    };
    let mut g = Gen::new(500);
    g.put(
        "vision_model.embeddings.patch_embedding.weight".into(),
        &[HIDDEN, 3, 4, 4],
        0.2,
    );
    g.put(
        "vision_model.embeddings.class_embedding".into(),
        &[HIDDEN],
        0.1,
    );
    g.put(
        "vision_model.embeddings.position_embedding.weight".into(),
        &[17, HIDDEN],
        0.1,
    );
    g.ln("vision_model.pre_layrnorm", HIDDEN);
    g.clip_layers("vision_model", 4, HIDDEN, 4 * HIDDEN);
    let encoder = g.weights();
    let resampler = crate::ip_adapter::ResamplerConfig {
        dim: 32,
        depth: 2,
        heads: 2,
        dim_head: 16,
        num_queries: 4,
        embed_dim: HIDDEN,
        output_dim: unet.cross_attention_dim[0],
    };
    let mut g = Gen::new(600);
    g.put("image_proj.latents".into(), &[1, 4, 32], 0.5);
    g.linear("image_proj.proj_in", 32, HIDDEN, true);
    g.linear("image_proj.proj_out", resampler.output_dim, 32, true);
    g.ln("image_proj.norm_out", resampler.output_dim);
    for i in 0..2 {
        let a = format!("image_proj.layers.{i}.0");
        g.ln(&format!("{a}.norm1"), 32);
        g.ln(&format!("{a}.norm2"), 32);
        g.linear(&format!("{a}.to_q"), 32, 32, false);
        g.linear(&format!("{a}.to_kv"), 64, 32, false);
        g.linear(&format!("{a}.to_out"), 32, 32, false);
        let f = format!("image_proj.layers.{i}.1");
        g.ln(&format!("{f}.0"), 32);
        g.linear(&format!("{f}.1"), 48, 32, false);
        g.linear(&format!("{f}.3"), 32, 48, false);
    }
    // One pair per cross-attention block, in `install_ip_adapter`'s walk order.
    let boc = &unet.block_out_channels;
    let n = boc.len();
    let mut dims = Vec::new();
    for (i, width) in boc.iter().enumerate() {
        if unet.down_block_types[i].contains("CrossAttn") {
            for _ in 0..unet.layers_per_block[i] * unet.transformer_layers_per_block[i] {
                dims.push(*width);
            }
        }
    }
    for k in 0..n {
        let ci = n - 1 - k;
        if unet.up_block_types[ci].contains("CrossAttn") {
            for _ in 0..(unet.layers_per_block[ci] + 1) * unet.transformer_layers_per_block[ci] {
                dims.push(boc[ci]);
            }
        }
    }
    for _ in 0..unet.transformer_layers_per_block[n - 1] {
        dims.push(boc[n - 1]);
    }
    for (i, dim) in dims.iter().enumerate() {
        let id = 2 * i + 1;
        g.linear(
            &format!("ip_adapter.{id}.to_k_ip"),
            *dim,
            resampler.output_dim,
            false,
        );
        g.linear(
            &format!("ip_adapter.{id}.to_v_ip"),
            *dim,
            resampler.output_dim,
            false,
        );
    }
    (vision, resampler, encoder, g.weights())
}

/// The components a staged request loads, through the production loaders (each checking its
/// header floor), in `model::load_heavy`'s order, armed for rung 4 like an iOS load.
struct Loaded {
    te1: crate::text_encoder::ClipTextEncoder,
    te2: crate::text_encoder::ClipTextEncoder,
    unet: UNet2DConditionModel,
    vae: crate::vae::Autoencoder,
    controls: Vec<crate::unet::ControlNet>,
    ip: Option<crate::ip_adapter::IpImageEncoder>,
}

fn load(s: &Snapshot, route: Route) -> Result<Loaded> {
    let files = Files::of(s);
    let (te1, te2) = files.text()?;
    let (unet, controls, vae, ip) = files.heavy(route)?;
    Ok(Loaded {
        te1,
        te2,
        unet,
        vae,
        controls,
        ip,
    })
}

/// The owned load facts of a [`Snapshot`] (its loaders outlive no borrow, so a `Residency` can
/// hold them).
#[derive(Clone)]
struct Files {
    root: PathBuf,
    bits: Option<i32>,
    text_quantize: bool,
    /// Independent frozen old pair-pack expression, used only for unchanged RGB proofs.
    text_finalize_reference: bool,
    /// Exercise actual fused component-map build order (not the outer full-file split/guard).
    text_ldm: bool,
    streamed: bool,
    unet: UNetConfig,
    te1: ClipTextConfig,
    te2: ClipTextConfig,
    vae: crate::config::VaeConfig,
    vision: crate::vision_encoder::VisionConfig,
    resampler: crate::ip_adapter::ResamplerConfig,
}

/// `(U-Net, ControlNet branches, VAE, IP image encoder)` of one heavy load.
type Heavy = (
    UNet2DConditionModel,
    Vec<crate::unet::ControlNet>,
    crate::vae::Autoencoder,
    Option<crate::ip_adapter::IpImageEncoder>,
);

impl Files {
    fn of(s: &Snapshot) -> Self {
        Self {
            root: s.root.clone(),
            bits: s.bits,
            text_quantize: s.text_quantize,
            text_finalize_reference: false,
            text_ldm: false,
            streamed: true,
            unet: s.unet.clone(),
            te1: s.te1.clone(),
            te2: s.te2.clone(),
            vae: s.vae.clone(),
            vision: s.vision.clone(),
            resampler: s.resampler.clone(),
        }
    }

    /// Both CLIP towers through the production loader.
    fn text(
        &self,
    ) -> Result<(
        crate::text_encoder::ClipTextEncoder,
        crate::text_encoder::ClipTextEncoder,
    )> {
        if self.text_finalize_reference {
            let (mut te1, mut te2) = if self.text_ldm {
                (
                    crate::loader::load_clip_from_weights(
                        Weights::from_file(self.root.join("text_encoder/model.safetensors"))?,
                        &self.te1,
                        crate::model::DTYPE,
                    )?,
                    crate::loader::load_clip_from_weights(
                        Weights::from_file(self.root.join("text_encoder_2/model.safetensors"))?,
                        &self.te2,
                        crate::model::DTYPE,
                    )?,
                )
            } else {
                self.text_unfinalized()?
            };
            if let Some(bits) = self.bits.filter(|_| self.text_quantize) {
                // Frozen pre-refactor runtime expression, not the new pair helper.
                te1.quantize(bits)?;
                te2.quantize(bits)?;
            }
            if self.text_ldm {
                te1.materialize_weights()?;
                te2.materialize_weights()?;
            }
            return Ok((te1, te2));
        }
        let quant = self.bits.filter(|_| self.text_quantize).map(|bits| {
            if bits == 4 {
                mlx_gen::gen_core::Quant::Q4
            } else {
                mlx_gen::gen_core::Quant::Q8
            }
        });
        if self.text_ldm {
            crate::model::build_ldm_text_using(
                Weights::from_file(self.root.join("text_encoder/model.safetensors"))?,
                Weights::from_file(self.root.join("text_encoder_2/model.safetensors"))?,
                quant,
                |weights, dtype| crate::loader::load_clip_from_weights(weights, &self.te1, dtype),
                |weights, dtype| crate::loader::load_clip_from_weights(weights, &self.te2, dtype),
                crate::text_encoder::ClipTextEncoder::materialize_weights,
            )
        } else {
            crate::model::load_text_encoders_using(
                &self.root,
                quant,
                |root, dtype| {
                    crate::loader::load_clip_dtype(root, "text_encoder", &self.te1, dtype)
                },
                |root, dtype| {
                    crate::loader::load_clip_dtype(root, "text_encoder_2", &self.te2, dtype)
                },
            )
        }
    }

    fn text_unfinalized(
        &self,
    ) -> Result<(
        crate::text_encoder::ClipTextEncoder,
        crate::text_encoder::ClipTextEncoder,
    )> {
        let dtype = crate::model::DTYPE;
        Ok((
            crate::loader::load_clip_dtype(&self.root, "text_encoder", &self.te1, dtype)?,
            crate::loader::load_clip_dtype(&self.root, "text_encoder_2", &self.te2, dtype)?,
        ))
    }

    /// `route`'s heavy bundle through the production loaders, in `model::load_heavy`'s order.
    /// `streamed` controls both lazy materialization and stream arming; false owns all weights.
    fn heavy(&self, route: Route) -> Result<Heavy> {
        let dtype = crate::model::DTYPE;
        let mut unet = crate::loader::load_unet_with_config_streamed(
            &self.root,
            dtype,
            &self.unet,
            self.streamed,
        )?;
        let vae = crate::loader::load_vae_fp16_fix_with(
            &mlx_gen::WeightsSource::Dir(self.root.join("vae_fp16_fix")),
            &self.vae,
        )?;
        let mut controls = Vec::new();
        if route == Route::Control {
            controls.push(crate::loader::load_controlnet_with(
                &mlx_gen::WeightsSource::Dir(self.root.join("controlnet")),
                dtype,
                &self.unet,
            )?);
        }
        let mut ip = None;
        if route == Route::IpAdapter {
            let (encoder, pairs) = crate::loader::load_ip_adapter_with(
                &self.root.join("ip"),
                dtype,
                &self.vision,
                &self.resampler,
            )?;
            unet.install_ip_adapter(pairs)?;
            ip = Some(encoder);
        }
        if let Some(bits) = self.bits {
            crate::quant::guarded_load_quantize(|| -> Result<()> {
                unet.quantize(bits)?;
                for control in &mut controls {
                    control.quantize(bits)?;
                }
                Ok(())
            })?;
        }
        if self.streamed {
            unet.arm_block_streams(
                &mlx_gen::WeightsSource::File(crate::loader::resolve_unet_weight_file(
                    &self.root, dtype,
                )?),
                dtype,
                self.bits,
            );
        }
        Ok((unet, controls, vae, ip))
    }
}

fn ids() -> Array {
    let n = TOKENS;
    let mut ids: Vec<i32> = (0..2 * n).map(|i| 1 + (i * 7) % 30).collect();
    ids[(n - 1) as usize] = 39;
    ids[(2 * n - 1) as usize] = 39;
    Array::from_slice(&ids, &[2, n])
}

fn image(w: i32, h: i32, seed: i32) -> mlx_gen::media::Image {
    mlx_gen::media::Image {
        width: w as u32,
        height: h as u32,
        pixels: (0..w * h * 3)
            .map(|i| ((i * 37 + seed) % 251) as u8)
            .collect(),
    }
}

/// Run one staged request's execution after its loads — the production pipeline functions in the
/// production order: the CFG text encode, the route's encode, a two-step guided ancestral denoise
/// at sequential CFG, the rung-2 tiled decode and the pixel conversion.
fn run_request(l: &Loaded, route: Route, tiling: &mlx_gen::tiling::TilingConfig) -> Result<()> {
    let (pixels_w, pixels_h) = (LATENT * VAE_SCALE, LATENT * VAE_SCALE);
    let (cond, pooled) = crate::pipeline::encode_conditioning(&l.te1, &l.te2, &ids())?;
    let ip_tokens = match &l.ip {
        Some(encoder) => {
            let tokens = encoder.tokens(&image(20, 18, 1))?;
            let zeros = encoder.zeros_like_tokens(tokens.dtype())?;
            Some(mlx_rs::ops::concatenate_axis(&[&tokens, &zeros], 0)?)
        }
        None => None,
    };
    let euler = crate::sampler::EulerSampler::new_with_dtype(
        &crate::config::DiffusionConfig::sdxl_base(),
        true,
        crate::model::DTYPE,
    )?;
    let mut blend = None;
    let latents = if route.encodes_init_for_test() {
        let init = crate::pipeline::encode_init_latents_tiled(
            &l.vae,
            &image(pixels_w, pixels_h, 3),
            pixels_w as u32,
            pixels_h as u32,
            Some(tiling),
            None,
        )?;
        let noise = rnd(init.shape(), 5, 1.0);
        let x_t = euler.add_noise_with(&init, &noise, 800.0)?;
        if route == Route::Mask {
            // `preprocess_mask` downsamples by SDXL's 8 (the tiny VAE's latent is the same 8x8).
            let edge = LATENT * 8;
            let mask =
                crate::inpaint::preprocess_mask(&image(edge, edge, 9), edge as u32, edge as u32)?;
            let t_prev = euler
                .timesteps(2, 800.0)?
                .into_iter()
                .map(|(_, t)| t)
                .collect();
            blend = Some(crate::inpaint::InpaintBlend::new(
                &euler, mask, init, noise, t_prev,
            ));
        }
        x_t
    } else {
        euler.sample_prior(&[1, LATENT, LATENT, 4])?
    };
    let sampler = crate::sampler::AncestralEuler::new(&euler, 2, 800.0)?;
    let plan = crate::plan::SdxlForwardPlan::UNBOUNDED.with_cfg_batching(CfgBatching::Sequential);
    let d = crate::pipeline::Denoiser::with_plan(&l.unet, &sampler, plan);
    let ids = crate::pipeline::text_time_ids(2);
    let cancel = mlx_gen::CancelFlag::default();
    let latents = match route {
        Route::Control => {
            let control = crate::pipeline::preprocess_control_image(
                &image(LATENT * CONTROL_SCALE, LATENT * CONTROL_SCALE, 7),
                (LATENT * CONTROL_SCALE) as u32,
                (LATENT * CONTROL_SCALE) as u32,
            )?;
            let control = mlx_rs::ops::concatenate_axis(&[&control, &control], 0)?;
            let cc = crate::pipeline::ControlContext {
                controlnet: &l.controls[0],
                cond_embed: l.controls[0].embed_cond(&control)?,
                scale: 1.0,
            };
            crate::pipeline::denoise_control(
                &d,
                latents,
                &cond,
                &pooled,
                &ids,
                5.0,
                &cancel,
                &mut |_| {},
                &cc,
            )?
        }
        Route::IpAdapter => crate::pipeline::denoise_ip(
            &d,
            latents,
            &cond,
            &pooled,
            &ids,
            5.0,
            &cancel,
            &mut |_| {},
            ip_tokens.as_ref().expect("IP tokens"),
            0.6,
        )?,
        Route::Mask => crate::pipeline::denoise_inpaint(
            &d,
            latents,
            &cond,
            &pooled,
            &ids,
            5.0,
            &cancel,
            &mut |_| {},
            blend.as_ref().expect("inpaint blend"),
        )?,
        _ => {
            crate::pipeline::denoise(&d, latents, &cond, &pooled, &ids, 5.0, &cancel, &mut |_| {})?
        }
    };
    crate::pipeline::decode_image_tiled(&l.vae, &latents, None, Some(tiling), None)?;
    Ok(())
}

impl Route {
    fn encodes_init_for_test(self) -> bool {
        matches!(self, Route::Reference | Route::Mask)
    }
}

/// Explicit request semantics for header-only/tiny diagnostics; image carriers are symbolic here.
/// The execution fixtures separately provision their real random RGB inputs.
pub(super) fn generation_request(
    route: Route,
    width: u32,
    height: u32,
    memory: &mlx_gen::gen_core::GenerationMemory,
) -> mlx_gen::GenerationRequest {
    use mlx_gen::Conditioning;
    let reference = || Conditioning::Reference {
        image: if route == Route::IpAdapter {
            image(20, 18, 1)
        } else {
            mlx_gen::Image::default()
        },
        strength: None,
    };
    let conditioning = match route {
        Route::Text => vec![],
        Route::Reference | Route::IpAdapter => vec![reference()],
        Route::Mask => vec![
            reference(),
            Conditioning::Mask {
                image: mlx_gen::Image::default(),
            },
        ],
        Route::Control => vec![Conditioning::Control {
            image: mlx_gen::Image::default(),
            kind: mlx_gen::gen_core::ControlKind::Canny,
            scale: None,
        }],
    };
    mlx_gen::GenerationRequest {
        width,
        height,
        count: 1,
        steps: Some(2),
        guidance: Some(5.0),
        strength: route.encodes_init_for_test().then_some(0.8),
        memory: Some(*memory),
        conditioning,
        ..Default::default()
    }
}

/// The regime the tiny request runs: staged, sequential CFG, rung 2 at an 8-pixel tile edge (so the
/// tiny decode and init encode both tile), one 8-token CLIP window.
fn regime(route: Route) -> Regime {
    let memory = mlx_gen::gen_core::GenerationMemory {
        cfg_batching: Some(CfgBatching::Sequential),
        ..Default::default()
    };
    let request = generation_request(
        route,
        (LATENT * VAE_SCALE) as u32,
        (LATENT * VAE_SCALE) as u32,
        &memory,
    );
    Regime {
        route,
        sampling: crate::sampling::ResolvedSampling::resolve(&request).unwrap(),
        count: request.count,
        has_init: route.encodes_init_for_test(),
        width: LATENT * VAE_SCALE,
        height: LATENT * VAE_SCALE,
        staged: true,
        cfg_batching: CfgBatching::Sequential,
        tiling: Some(mlx_gen::tiling::TilingConfig::spatial_only(8, 0)),
        context_tokens: TOKENS,
        ip_scale: crate::model::ip_reference_scale(None),
        reference_image: crate::sampling::SamplingWorkFacts::from_request(&request).reference_image,
    }
}

/// The largest checked scratch per phase (and, for the loads, per label) of an observed run.
fn observed_by_phase(events: &[Ev]) -> (HashMap<IosPhase, u64>, HashMap<String, u64>) {
    let (mut phases, mut loads) = (HashMap::new(), HashMap::new());
    for e in events {
        if let Ev::Checked(label, scratch) = e {
            let phase = ios_quantum_phase(label)
                .unwrap_or_else(|| panic!("checked label {label:?} attributes to no phase"));
            let slot = phases.entry(phase).or_insert(0u64);
            *slot = (*slot).max(*scratch);
            if phase == IosPhase::Load {
                let slot = loads.entry(label.clone()).or_insert(0u64);
                *slot = (*slot).max(*scratch);
            }
        }
    }
    (phases, loads)
}

/// THE proof that pricing is the guards' formulas: for txt2img, img2img and inpaint (an init-image
/// encode), ControlNet (a conditioning-embedding encode) and IP-Adapter on the tiny snapshot, run the real
/// loaders and pipeline under the CPU diagnostic observer and require, per phase, the largest
/// observed checked scratch to EQUAL `ios_pricing`'s priced scratch for that phase (every load
/// floor equal by label), and every observed label to be priced.
#[test]
fn pricing_is_the_guards_formulas() {
    if !cpu_child("pricing::pricing_is_the_guards_formulas") {
        return;
    }
    let snap = Snapshot::new(2);
    for route in [
        Route::Text,
        Route::Reference,
        Route::Mask,
        Route::Control,
        Route::IpAdapter,
    ] {
        assert_pricing_is_observed(&snap, route);
    }
}

/// Run `route` on `snap` through the real loaders and pipeline under the CPU diagnostic observer
/// and require pricing to equal the guards: per phase the largest checked scratch, per load label
/// its floor, and every checked label priced.
fn assert_pricing_is_observed(snap: &Snapshot, route: Route) {
    let regime = regime(route);
    let (result, events) = observe(PLENTY, || {
        let loaded = load(snap, route)?;
        run_request(&loaded, route, regime.tiling.as_ref().unwrap())
    });
    result.unwrap();
    // The load floors are checks before a construction (no synchronized quantum); the rest of
    // the request is a well-ordered quantum sequence.
    let execution: Vec<Ev> = events
        .iter()
        .filter(|e| !matches!(e, Ev::Checked(l, _) if ios_quantum_phase(l) == Some(IosPhase::Load)))
        .cloned()
        .collect();
    assert_well_ordered(&execution);
    let (observed, observed_loads) = observed_by_phase(&events);
    let (pricing, items) = crate::quantum::with_backend(Backend::Cpu, || {
        Priced::build(&snap.inputs(route))
            .and_then(|priced| priced.price(&regime))
            .unwrap()
    });
    for phase in [
        IosPhase::Conditioning,
        IosPhase::Encode,
        IosPhase::Denoise,
        IosPhase::Decode,
    ] {
        let priced = pricing.phase(phase).map(|p| p.scratch_bytes);
        if observed.get(&phase).copied() != priced {
            // Name the drifting quantum: each label's largest observed vs priced scratch.
            let mut labels: Vec<&str> = items
                .iter()
                .filter(|q| q.phase == phase)
                .map(|q| q.label)
                .collect();
            labels.sort_unstable();
            labels.dedup();
            let seen_seq: Vec<(String, u64)> = events
                .iter()
                .filter_map(|e| match e {
                    Ev::Checked(l, v) if ios_quantum_phase(l) == Some(phase) => {
                        Some((l.clone(), *v))
                    }
                    _ => None,
                })
                .collect();
            let want_seq: Vec<(&str, u64)> = items
                .iter()
                .filter(|q| q.phase == phase)
                .map(|q| (q.label, q.scratch))
                .collect();
            for (i, w) in want_seq.iter().enumerate() {
                eprintln!("SEQ {i}: priced {w:?} observed {:?}", seen_seq.get(i));
            }
            for label in labels {
                let seen = checked(&events, label).into_iter().max();
                let want = items
                    .iter()
                    .filter(|q| q.label == label)
                    .map(|q| q.scratch)
                    .max();
                eprintln!("{route:?} {phase:?} {label}: observed {seen:?} priced {want:?}");
            }
        }
        assert_eq!(
            observed.get(&phase).copied(),
            priced,
            "{route:?} {phase:?}: largest observed check vs priced (binding {:?})",
            pricing.phase(phase).map(|p| &p.binding)
        );
    }
    for (label, scratch) in &observed_loads {
        let priced = items
            .iter()
            .filter(|q| q.label == label.as_str())
            .map(|q| q.scratch)
            .max();
        assert_eq!(priced, Some(*scratch), "{route:?} load {label}");
    }
    // Pricing attributes every quantum to the phase its label names (what a device capture reads).
    for q in &items {
        assert_eq!(
            ios_quantum_phase(q.label),
            Some(q.phase),
            "{route:?}: {} priced in {:?}",
            q.label,
            q.phase
        );
    }
    // Nothing the run checked is unpriced.
    for e in &events {
        if let Ev::Checked(label, _) = e {
            assert!(
                items.iter().any(|q| q.label == label.as_str()),
                "{route:?}: checked {label} is not priced"
            );
        }
    }
}

/// [`pricing_is_the_guards_formulas`] on a q4-packed tier: the packed projections' bf16 scales
/// promote the U-Net's activations (and from the first step's output on, the latent) to f32, and
/// the load-time-quantized ControlNet's to f32 — every dtype the guards read, priced header-only.
#[test]
fn packed_pricing_is_the_guards_formulas() {
    if !cpu_child("pricing::packed_pricing_is_the_guards_formulas") {
        return;
    }
    let snap = Snapshot::packed(4);
    for route in [Route::Text, Route::Reference, Route::Control] {
        assert_pricing_is_observed(&snap, route);
    }
}

/// Exercise the actual text/IP final-quantum call sites, not just the shared norm helper.
#[test]
fn final_norm_guards_cover_norm_pool_and_projection() {
    if !cpu_child("pricing::final_norm_guards_cover_norm_pool_and_projection") {
        return;
    }
    for (tier, snap) in [
        ("bf16", Snapshot::new(2)),
        ("q8", Snapshot::packed(8)),
        ("q4", Snapshot::packed(4)),
    ] {
        let loaded = load(&snap, Route::IpAdapter).unwrap();
        let tiling = mlx_gen::tiling::TilingConfig::spatial_only(8, 0);
        with_cpu_diagnostic_headroom(PLENTY, || run_request(&loaded, Route::IpAdapter, &tiling))
            .unwrap();
        let counts = Rc::new(RefCell::new(HashMap::<String, usize>::new()));
        let sink = counts.clone();
        let mut open = None;
        with_cpu_diagnostic_headroom(PLENTY, || {
            with_cpu_diagnostic_observer(
                move |event| match event {
                    QuantumEvent::Checked { label, scratch }
                        if [CLIP_POOLED, RESAMPLER_PROJ_OUT].contains(&label.as_str()) =>
                    {
                        assert!(open.is_none(), "final quanta cannot overlap");
                        open = Some((
                            label.clone(),
                            *scratch,
                            mlx_rs::memory::get_active_memory() as u64,
                        ));
                        mlx_rs::memory::reset_peak_memory();
                    }
                    QuantumEvent::Synchronized { label, .. }
                        if [CLIP_POOLED, RESAMPLER_PROJ_OUT].contains(&label.as_str()) =>
                    {
                        let (checked, scratch, base) =
                            open.take().expect("final quantum was checked");
                        assert_eq!(&checked, label);
                        let growth =
                            (mlx_rs::memory::get_peak_memory() as u64).saturating_sub(base);
                        eprintln!(
                            "FINAL NORM GUARD {tier} {label}: checked {scratch} measured {growth}"
                        );
                        assert_tracks_measured(label, scratch, growth, 1.5);
                        *sink.borrow_mut().entry(label.clone()).or_default() += 1;
                    }
                    _ => {}
                },
                || run_request(&loaded, Route::IpAdapter, &tiling),
            )
        })
        .unwrap();
        assert_eq!(
            counts.borrow().get(CLIP_POOLED),
            Some(&2),
            "both CLIP encoders executed"
        );
        assert_eq!(
            counts.borrow().get(RESAMPLER_PROJ_OUT),
            Some(&1),
            "IP projection/norm executed"
        );
    }
}

/// The sampler's wider img2img state is NOT its f16 forward input. Dense text ends f16, dense
/// img2img ends f32; packed eps promotes both to f32. Distinct decode casts are retained only
/// between the first decoder check and VAE return, and never charged twice as resident + scratch.
#[test]
fn decode_resident_separates_sampler_state_forward_cast_and_compute_alias() {
    if !cpu_child("pricing::decode_resident_separates_sampler_state_forward_cast_and_compute_alias")
    {
        return;
    }
    for (tier, snap) in [
        ("bf16", Snapshot::new(2)),
        ("q8", Snapshot::packed(8)),
        ("q4", Snapshot::packed(4)),
    ] {
        crate::quantum::with_backend(Backend::Cpu, || {
            let items = |route| {
                let r = regime(route);
                Priced::build(&snap.inputs(route))
                    .unwrap()
                    .price(&r)
                    .unwrap()
                    .1
            };
            let text = items(Route::Text);
            let reference = items(Route::Reference);
            let resident = |items: &[crate::ios_pricing::PricedQuantum], label| {
                items.iter().find(|q| q.label == label).unwrap().resident
            };
            let elements = (LATENT * LATENT * 4) as u64;
            let caller_delta = if tier == "bf16" {
                elements * (4 - 2)
            } else {
                0
            };
            let cast_delta = caller_delta;
            for (label, delta) in [
                (VAE_DECODE_IN, caller_delta),
                (VAE_RESNET, caller_delta + cast_delta),
                (DECODED_TO_IMAGE, caller_delta),
            ] {
                assert_eq!(resident(&reference, label) - resident(&text, label), delta,
                    "{tier} {label}: sampler caller and distinct compute copy have different lifetimes");
            }
        });
    }
}

// ---------------------------------------------------------------------------------------------
// Lever (b): rung 4 on the iOS load — every block read inside its own priced quantum, the load
// floor counting only the resident remainder.
// ---------------------------------------------------------------------------------------------

/// A streamed transformer block's quantum (the block read from the snapshot inside it, then its
/// forward) is priced at its activations plus `SdxlBlockStream::streamed_bytes`, and that price
/// tracks the stalled measured peak (>= measured, <= 1.5x) at an f32 block (256 wide, a 1024-wide
/// GEGLU) over 64 tokens, weight-dominated as every production SDXL block is.
#[test]
fn streamed_block_quantum_tracks_its_measured_peak() {
    if !cpu_child("pricing::streamed_block_quantum_tracks_its_measured_peak") {
        return;
    }
    let (dim, ctx, ff, heads) = (256, 64, 1024, 4);
    let mut g = Gen::new(9100);
    g.ln("t.norm", dim);
    g.linear("t.proj_in", dim, dim, true);
    g.linear("t.proj_out", dim, dim, true);
    let b = "t.transformer_blocks.0";
    for n in ["norm1", "norm2", "norm3"] {
        g.ln(&format!("{b}.{n}"), dim);
    }
    for (attn, kv) in [("attn1", dim), ("attn2", ctx)] {
        g.linear(&format!("{b}.{attn}.to_q"), dim, dim, false);
        g.linear(&format!("{b}.{attn}.to_k"), dim, kv, false);
        g.linear(&format!("{b}.{attn}.to_v"), dim, kv, false);
        g.linear(&format!("{b}.{attn}.to_out.0"), dim, dim, true);
    }
    g.linear(&format!("{b}.ff.net.0.proj"), 2 * ff, dim, true);
    g.linear(&format!("{b}.ff.net.2"), dim, ff, true);
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("stack.safetensors");
    save(&g.weights(), &file);
    let mut stack = crate::unet::Transformer2D::from_weights(
        &Weights::from_file(&file).unwrap(),
        "t",
        dim,
        heads,
        1,
    )
    .unwrap();
    stack.arm_block_stream(
        mlx_gen::WeightsSource::File(file.clone()),
        "t",
        Dtype::Float32,
        None,
    );
    let memory = rnd(&[1, 7, ctx], 2, 1.0);
    let plan = crate::plan::SdxlForwardPlan::UNBOUNDED;
    let streamed = stack.stream_bytes_for_test(0).unwrap();
    // Weight-dominated, as every production SDXL block is (a 1280-wide block's ~96 MB of weights
    // against its ~10 MB of 1024² activations): the bound is tight.
    let y = rnd(&[1, 64, dim], 1, 1.0);
    mlx_rs::transforms::eval([&y, &memory]).unwrap();
    let price = stack.block_price(0, &y, &memory, None, plan).unwrap();
    super::proofs::prove(TRANSFORMER_BLOCK, price, || {
        Ok(vec![stack.streamed_block_for_test(0, &y, &memory, plan)?])
    });
    assert!(streamed > price / 2 && streamed < price);
    // Activation-dominated: the block's FFN weights are read after its attention's scores are
    // released (each sub-layer is materialized before the next is built), so the sum of the two
    // is a looser bound — but never below the measurement.
    let y = rnd(&[1, 256, dim], 3, 1.0);
    y.eval().unwrap();
    let price = stack.block_price(0, &y, &memory, None, plan).unwrap();
    let measured = with_cpu_diagnostic_headroom(PLENTY, || {
        mlx_gen::ios_quantum::measured_peak_bytes(|| {
            Ok(vec![stack.streamed_block_for_test(0, &y, &memory, plan)?])
        })
    })
    .unwrap();
    eprintln!("PROOF {TRANSFORMER_BLOCK} (256 tokens): formula {price} B, measured {measured} B");
    assert!(price >= measured, "{price} < {measured}");
}

/// A streamed U-Net load's floor counts only the tensors the load materializes (the resident
/// remainder; the blocks are read by their quanta) and tracks the measured peak of that load.
#[test]
fn streamed_unet_load_floor_tracks_its_measured_peak() {
    if !cpu_child("pricing::streamed_unet_load_floor_tracks_its_measured_peak") {
        return;
    }
    let snap = Snapshot::new(6);
    let dtype = crate::model::DTYPE;
    let file = crate::loader::resolve_unet_weight_file(&snap.root, dtype).unwrap();
    let files = [file];
    let (_, streamed) =
        crate::loader::unet_load_parts(&files, dtype, true, true, &snap.unet).unwrap();
    let (_, whole) =
        crate::loader::unet_load_parts(&files, dtype, true, false, &snap.unet).unwrap();
    let (_, events) = observe(PLENTY, || {
        crate::loader::load_unet_with_config_streamed(&snap.root, dtype, &snap.unet, true)
    });
    assert_eq!(checked(&events, UNET_LOAD), vec![streamed]);
    super::proofs::prove(UNET_LOAD, streamed, || {
        let unet =
            crate::loader::load_unet_with_config_streamed(&snap.root, dtype, &snap.unet, true)?;
        drop(unet);
        Ok(Vec::new())
    });
    assert!(streamed < whole, "{streamed} vs {whole}");
}

/// Parity: holding a cast load's sources changes no value — the fp16-fix VAE loaded on the iOS
/// path (sources held) decodes bit-identically to the host load (no hold).
#[test]
fn held_cast_sources_load_bit_identical_weights() {
    if !cpu_child("pricing::held_cast_sources_load_bit_identical_weights") {
        return;
    }
    let snap = Snapshot::new(1);
    let source = mlx_gen::WeightsSource::Dir(snap.root.join("vae_fp16_fix"));
    let host = crate::loader::load_vae_fp16_fix_with(&source, &snap.vae).unwrap();
    let held = with_cpu_diagnostic_headroom(PLENTY, || {
        crate::loader::load_vae_fp16_fix_with(&source, &snap.vae)
    })
    .unwrap();
    let z = rnd(&[1, LATENT, LATENT, 4], 31, 1.0);
    z.eval().unwrap();
    assert_eq!(
        bits(&host.decode(&z).unwrap()),
        bits(&held.decode(&z).unwrap())
    );
}

// ---------------------------------------------------------------------------------------------
// Lever (d): a staged IP-Adapter request sheds the image encoder once its image prompt is encoded.
// ---------------------------------------------------------------------------------------------

/// The heavy bundle of a staged IP-Adapter request, loaded through the production loaders.
fn ip_bundle(s: &Snapshot) -> crate::model::SdxlHeavyOwned {
    let l = load(s, Route::IpAdapter).unwrap();
    crate::model::SdxlHeavyOwned::for_test(l.unet, l.vae, l.ip)
}

/// Output parity: the shed path's image-prompt tokens are bit-identical to the resident path's,
/// so the denoise that consumes them is too; the shed path releases the tower (and a resident
/// bundle keeps it for the next request).
#[test]
fn staged_ip_request_sheds_the_image_encoder_with_identical_tokens() {
    if !cpu_child("pricing::staged_ip_request_sheds_the_image_encoder_with_identical_tokens") {
        return;
    }
    let snap = Snapshot::new(1);
    let prompt = image(20, 18, 1);
    let resident = ip_bundle(&snap);
    let kept = resident.ip_tokens_for_test(&prompt, true, false).unwrap();
    assert!(resident.has_ip_adapter_for_test());
    let staged = ip_bundle(&snap);
    let shed = staged
        .ip_tokens_for_test(&prompt, true, crate::model::sheds_ip_image_encoder(true))
        .unwrap();
    assert_eq!(bits(&kept), bits(&shed));
    assert!(
        !staged.has_ip_adapter_for_test(),
        "a staged request must release the image encoder once its tokens exist"
    );
    assert!(!crate::model::sheds_ip_image_encoder(false));
}

/// The lever's price: the staged IP denoise's priced resident bytes (U-Net remainder + VAE + the
/// installed K/V pairs, no image tower) track the measured bytes the bundle still holds after the
/// shed (>= measured, <= 1.5x) — the tower here is most of the bundle, so pricing it resident (or
/// not shedding it) fails the bound.
#[test]
fn staged_ip_denoise_resident_tracks_the_measured_bundle() {
    if !cpu_child("pricing::staged_ip_denoise_resident_tracks_the_measured_bundle") {
        return;
    }
    let snap = Snapshot::new(1);
    let (_, items) = crate::quantum::with_backend(Backend::Cpu, || {
        Priced::build(&snap.inputs(Route::IpAdapter))
            .and_then(|priced| priced.price(&regime(Route::IpAdapter)))
            .unwrap()
    });
    let resident = items
        .iter()
        .find(|q| q.phase == IosPhase::Denoise)
        .map(|q| q.resident)
        .unwrap();
    let prompt = image(20, 18, 1);
    let run = || {
        with_cpu_diagnostic_headroom(PLENTY, || {
            let bundle = ip_bundle(&snap);
            let tokens = bundle.ip_tokens_for_test(
                &prompt,
                true,
                crate::model::sheds_ip_image_encoder(true),
            )?;
            drop(tokens);
            Ok(bundle)
        })
        .unwrap()
    };
    // One warm-up request first: the compiled GELU's process-wide constant cache is not the bundle.
    drop(run());
    mlx_rs::memory::clear_cache();
    let before = mlx_rs::memory::get_active_memory() as u64;
    let bundle = run();
    mlx_rs::memory::clear_cache();
    let measured = (mlx_rs::memory::get_active_memory() as u64).saturating_sub(before);
    eprintln!("PROOF staged IP denoise resident: priced {resident} B, measured {measured} B");
    assert_tracks_measured("staged IP denoise resident", resident, measured, 1.5);
    drop(bundle);
}

/// A CAST load (the fp16-fix VAE: f32 on disk, f16 resident): the floor the guard checks tracks the
/// measured load peak (the read sources and their copies live together), and the load leaves
/// exactly the copies resident — the sources are held until the load materializes, so no copy is
/// donated (and keeps) a wider source buffer.
#[test]
fn cast_load_floor_and_resident_track_their_measurements() {
    if !cpu_child("pricing::cast_load_floor_and_resident_track_their_measurements") {
        return;
    }
    let snap = Snapshot::new(1);
    let file = snap
        .root
        .join("vae_fp16_fix/diffusion_pytorch_model.safetensors");
    let (resident, floor) =
        crate::loader::cast_load_parts(&[file], crate::loader::VAE_FP16_FIX_DTYPE, true, |_| true)
            .unwrap();
    let source = mlx_gen::WeightsSource::Dir(snap.root.join("vae_fp16_fix"));
    let load = || crate::loader::load_vae_fp16_fix_with(&source, &snap.vae);
    let (_, events) = observe(PLENTY, load);
    assert_eq!(checked(&events, VAE_FP16_FIX_LOAD), vec![floor]);
    super::proofs::prove(VAE_FP16_FIX_LOAD, floor, || {
        load()?;
        Ok(Vec::new())
    });
    mlx_rs::memory::clear_cache();
    let before = mlx_rs::memory::get_active_memory() as u64;
    let vae = with_cpu_diagnostic_headroom(PLENTY, load).unwrap();
    mlx_rs::memory::clear_cache();
    let held = (mlx_rs::memory::get_active_memory() as u64).saturating_sub(before);
    eprintln!("PROOF {VAE_FP16_FIX_LOAD} resident: priced {resident} B, measured {held} B");
    assert_tracks_measured("fp16-fix VAE resident", resident, held, 1.5);
    assert_eq!(
        resident, held,
        "the load leaves exactly the f16 copies resident"
    );
    drop(vae);
}

#[path = "quantum_tests_additivity.rs"]
mod additivity;

/// Nonzero production consumer discriminator. Native HashMap order and actual Array values are
/// independent witnesses; no selected-world metadata is used to choose the native winning alias.
/// This is source-logical/guard parity, NOT a physical allocator, RSS, device or RGB certificate.
#[test]
fn adapter_world_prefix_forward_and_conv_consumer() {
    let Some(mut completed) =
        completed_cpu_child("pricing::adapter_world_prefix_forward_and_conv_consumer")
    else {
        return;
    };
    {
        use crate::adapters::{AdapterInstallStorage as S, AdapterWork};
        use mlx_gen::adapters::{AdaptableHost, Adapter};
        use mlx_gen::array::scalar;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let mut mask = 0u8;
        let snap = Snapshot::packed(4);
        let inputs = snap.inputs(Route::Text);
        let headers =
            mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&inputs.unet).unwrap();
        let target = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
        let projection = crate::quant::linear_from_headers(&headers, target).unwrap();
        let (out, input) = (projection.shape[0], projection.shape[1]);
        let tensor = |shape: &[i32], value: f32, dtype: Dtype| {
            let count = shape.iter().map(|d| *d as usize).product::<usize>();
            let array = Array::from_slice(&vec![value; count], shape)
                .as_dtype(dtype)
                .unwrap();
            array.eval().unwrap();
            array
        };
        let down = format!("lora_unet_{}.lora_down.weight", target.replace('.', "_"));
        let up_a = format!("lora_unet_{}.lora_up.weight", target.replace('.', "_"));
        let up_b = format!("base_model.model.unet.{target}.lora_B.default.weight");
        let alias_file = snap.file("nonzero_alias.safetensors");
        save(
            &Weights::from_map(HashMap::from([
                (down, tensor(&[2, input], 0.125, Dtype::Float16)),
                (up_a.clone(), tensor(&[out, 2], 0.25, Dtype::Float16)),
                (up_b.clone(), tensor(&[out, 2], 0.375, Dtype::Bfloat16)),
            ])),
            &alias_file,
        );
        let conv_file = snap.file("broadcast_conv.safetensors");
        let shortcut = headers
            .iter()
            .find_map(|header| {
                header
                    .name
                    .strip_suffix(".conv_shortcut.weight")
                    .map(|prefix| format!("{prefix}.conv_shortcut"))
            })
            .expect("fixture has an actual dense shortcut");
        save(
            &Weights::from_map(HashMap::from([
                (
                    "lora_unet_conv_in.lora_down.weight".into(),
                    tensor(&[2, 1, 1, 1], 0.25, Dtype::Float16),
                ),
                (
                    "lora_unet_conv_in.lora_up.weight".into(),
                    tensor(&[1, 2, 1, 1], 0.5, Dtype::Float16),
                ),
                (
                    format!("lora_unet_{}.lora_down.weight", shortcut.replace('.', "_")),
                    tensor(&[2, 1, 1, 1], 0.25, Dtype::Float16),
                ),
                (
                    format!("lora_unet_{}.lora_up.weight", shortcut.replace('.', "_")),
                    tensor(&[1, 2, 1, 1], 0.5, Dtype::Float16),
                ),
            ])),
            &conv_file,
        );
        let mut spec =
            mlx_gen::gen_core::LoadSpec::new(mlx_gen::WeightsSource::Dir(snap.root.clone()));
        spec.adapters = vec![
            AdapterSpec::new(alias_file.clone(), 1.0, AdapterKind::Lora),
            AdapterSpec::new(conv_file.clone(), 1.0, AdapterKind::Lora),
        ];
        let mut native = crate::loader::load_unet_with_config_streamed(
            &snap.root,
            crate::model::DTYPE,
            &snap.unet,
            false,
        )
        .unwrap();
        let parts: Vec<_> = target.split('.').collect();
        let probe = tensor(&[1, input], 1.0, Dtype::Float32);
        let baseline = native
            .adaptable_mut(&parts)
            .unwrap()
            .forward(&probe)
            .unwrap();
        baseline.eval().unwrap();
        let original_conv = native.conv_in_weight().clone();
        let shortcut_parts: Vec<_> = shortcut.split('.').collect();
        let original_shortcut = native
            .adaptable_mut(&shortcut_parts)
            .unwrap()
            .dense_weight()
            .unwrap()
            .0
            .clone();
        let ((report, ()), actual) = crate::adapters::with_adapter_work(|| {
            let report = crate::adapters::apply_sdxl_adapters_with(
                &mut native,
                &spec.adapters,
                crate::adapters::configured_lora_coverage(),
            )
            .unwrap();
            (report, ())
        });
        assert_eq!(
            report.merged, 3,
            "nonzero fixture reached packed LoRA, dense conv and dense shortcut installs"
        );
        let raw = actual
            .iter()
            .find_map(|event| match event {
                AdapterWork::FileMaterialized(payloads) => Some(payloads),
                _ => None,
            })
            .unwrap();
        let winner = raw
            .iter()
            .rfind(|(key, _)| key == &up_a || key == &up_b)
            .unwrap();
        let coefficient = if winner.0 == up_a { 0.25 } else { 0.375 };
        let linear = native.adaptable_mut(&parts).unwrap();
        let Adapter::Lora { a, b, scale } = &linear.adapters()[0] else {
            panic!("actual packed LoRA residual absent");
        };
        assert_eq!(*scale, 1.0);
        assert_eq!(
            b.dtype(),
            winner.1.dtype,
            "actual unsorted raw last-role winner"
        );
        assert!(a
            .as_dtype(Dtype::Float32)
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|v| *v == 0.125));
        assert!(b
            .as_dtype(Dtype::Float32)
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|v| *v == coefficient));
        let residual = input as f32 * 0.125 * 2.0 * coefficient;
        assert!(residual > 0.0);
        let expected_linear = mlx_rs::ops::add(&baseline, scalar(residual)).unwrap();
        let actual_linear = linear.forward(&probe).unwrap();
        assert_eq!(
            actual_linear.try_as_slice::<f32>().unwrap(),
            expected_linear.try_as_slice::<f32>().unwrap(),
            "independent known nonzero factor product, not a copied header producer"
        );
        mask |= 1;
        eprintln!("SDXL_B_MECHANISM 0 raw_alias_order_and_nonzero_native_values");

        let expected_conv = mlx_rs::ops::add(
            &original_conv,
            scalar(0.25f32).as_dtype(original_conv.dtype()).unwrap(),
        )
        .unwrap();
        let flattened = |array: &Array| {
            array
                .reshape(&[-1])
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .try_as_slice::<f32>()
                .unwrap()
                .to_vec()
        };
        assert_eq!(
            flattened(native.conv_in_weight()),
            flattened(&expected_conv),
            "native NCHW delta -> OHWI + broadcast sum uses the FULL base geometry"
        );
        let installed_conv = actual
            .iter()
            .find_map(|event| match event {
                AdapterWork::Install(work) if work.storage == S::DenseConv => Some(work),
                _ => None,
            })
            .unwrap();
        let native_bytes = native.conv_in_weight().size() as u64
            * crate::quantum::element_bytes(native.conv_in_weight());
        assert!(
            native_bytes > 4,
            "broadcast result is genuinely larger than the scalar-shaped delta"
        );
        assert!(
            installed_conv.scratch >= mlx_gen::ios_quantum::allocated_bytes(native_bytes),
            "native conv guard underprices the broadcast-sized installed output"
        );
        let expected_shortcut = mlx_rs::ops::add(
            &original_shortcut,
            scalar(0.25f32).as_dtype(original_shortcut.dtype()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            flattened(
                native
                    .adaptable_mut(&shortcut_parts)
                    .unwrap()
                    .dense_weight()
                    .unwrap()
                    .0
            ),
            flattened(&expected_shortcut),
            "independent native dense-shortcut broadcast result"
        );
        let installed_shortcut = actual
            .iter()
            .find_map(|event| match event {
                AdapterWork::Install(work) if work.target == shortcut => Some(work),
                _ => None,
            })
            .unwrap();
        assert!(
            installed_shortcut.scratch
                >= mlx_gen::ios_quantum::allocated_bytes(
                    original_shortcut.size() as u64
                        * crate::quantum::element_bytes(&original_shortcut)
                ),
            "native shortcut guard underprices its full broadcast-sized sum"
        );
        let mut worlds = 0;
        let mut matched_native = false;
        crate::adapters::visit_configured_adapter_outcomes(&spec, &headers, &snap.unet, |world| {
            assert!(world.refusal.is_none());
            worlds += 1;
            let installs: Vec<_> = world
                .work
                .iter()
                .filter_map(|event| match event {
                    AdapterWork::Install(work) => Some(work),
                    _ => None,
                })
                .collect();
            assert_eq!(installs.len(), 3);
            let conv = installs
                .iter()
                .find(|work| work.storage == S::DenseConv)
                .unwrap();
            assert_eq!(conv.installed[0].shape, native.conv_in_weight().shape());
            assert_eq!(conv.installed[0].dtype, native.conv_in_weight().dtype());
            matched_native |= installs[0].installed[1].dtype == winner.1.dtype;
            let ledger = crate::ios_pricing::AdapterLoadLedger::from_work_checked(world.work, 137)?;
            assert_eq!(ledger.residuals.len(), 1);
            assert!(ledger.preparation_upper > 0);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            worlds, 2,
            "both complete alias worlds, not a refused or joined dtype"
        );
        assert!(matched_native);
        mask |= 2;
        eprintln!("SDXL_B_MECHANISM 1 whole_selected_world_conv_and_residual_ancestry");

        let mut priced = Priced::build_configured(&inputs, &spec).unwrap();
        let pricing = priced
            .price_configured(&spec, &inputs.unet, inputs.quant_bits, &regime(Route::Text))
            .unwrap();
        let x = tensor(&[1, LATENT, LATENT, 4], 0.125, crate::model::DTYPE);
        let context = tensor(&[1, TOKENS, snap.context], 0.0625, crate::model::DTYPE);
        let pooled = tensor(&[1, 16], 0.125, crate::model::DTYPE);
        let time_ids = crate::pipeline::text_time_ids(1);
        let (result, events) = observe(PLENTY, || -> Result<Vec<f32>> {
            // Actual ordered base-only pack retains the already installed nonzero residual.
            crate::quant::guarded_load_quantize(|| native.quantize(4))?;
            let output = native.forward(&x, 500.0, &context, &pooled, &time_ids)?;
            Ok(flattened(&output))
        });
        let values = result.unwrap();
        assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
        for event in events {
            if let Ev::Checked(label, scratch) = event {
                assert!(
                    pricing
                        .windows
                        .iter()
                        .any(|window| window.label == label && window.scratch_bytes >= scratch),
                    "whole-world production price lost actual guarded {label}: {scratch}"
                );
            }
        }
        assert!(pricing.phase(IosPhase::Denoise).is_some());
        mask |= 4;
        eprintln!("SDXL_B_MECHANISM 2 actual_ordered_pack_and_native_whole_forward");

        // ALL alpha operands are parsed, even one later overwritten under the same resolved role.
        let malformed = snap.file("overwritten_invalid_alpha.safetensors");
        save(
            &Weights::from_map(HashMap::from([
                (
                    format!("lora_unet_{}.alpha", target.replace('.', "_")),
                    tensor(&[2], 1.0, Dtype::Float16),
                ),
                (
                    format!("base_model.model.unet.{target}.alpha"),
                    tensor(&[1], 2.0, Dtype::Float32),
                ),
            ])),
            &malformed,
        );
        spec.adapters
            .push(AdapterSpec::new(malformed, 1.0, AdapterKind::Lora));
        let expected_raw: u64 = spec
            .adapters
            .iter()
            .map(|adapter| {
                mlx_gen::ios_quantum::load_floor_bytes(std::slice::from_ref(&adapter.path), |_| {
                    true
                })
                .unwrap()
            })
            .sum();
        let mut prefixes = 0;
        crate::adapters::visit_configured_adapter_outcomes(&spec, &headers, &snap.unet, |world| {
            assert!(world.refusal.as_ref().unwrap().contains("alpha"));
            prefixes += 1;
            assert_eq!(
                world
                    .work
                    .iter()
                    .filter(|event| matches!(event, AdapterWork::Install(_)))
                    .count(),
                3,
                "refusal fabricated an install or lost an earlier file's success"
            );
            assert_eq!(
                world
                    .work
                    .iter()
                    .filter(|event| matches!(event, AdapterWork::FileMaterialized(_)))
                    .count(),
                3
            );
            let ledger = crate::ios_pricing::AdapterLoadLedger::from_work_checked(world.work, 137)?;
            assert_eq!(ledger.raw_parent_upper, expected_raw);
            assert_eq!(ledger.residuals.len(), 1);
            assert!(ledger
                .quanta
                .iter()
                .any(|q| q.resident > 137 + expected_raw));
            Ok(())
        })
        .unwrap();
        assert_eq!(prefixes, 2);
        let mut refused_native = crate::loader::load_unet_with_config_streamed(
            &snap.root,
            crate::model::DTYPE,
            &snap.unet,
            false,
        )
        .unwrap();
        let (refusal, prefix) = crate::adapters::with_adapter_work(|| {
            crate::adapters::apply_sdxl_adapters_with(
                &mut refused_native,
                &spec.adapters,
                crate::adapters::configured_lora_coverage(),
            )
        });
        assert!(
            refusal.is_err(),
            "actual parser must refuse the invalid overwritten alpha"
        );
        assert_eq!(
            prefix
                .iter()
                .filter(|event| matches!(event, AdapterWork::Install(_)))
                .count(),
            3
        );
        assert_eq!(
            prefix
                .iter()
                .filter(|event| matches!(event, AdapterWork::FileMaterialized(_)))
                .count(),
            3
        );
        mask |= 8;
        eprintln!("SDXL_B_MECHANISM 3 actual_overwritten_alpha_refusal_preserves_prior_files");

        // Independent actual Arrays vs approved resnet metadata extraction, at MIXED precast widths.
        let conv1 = tensor(&[64, 3, 3, 32], 0.25, Dtype::Float16);
        let conv2 = tensor(&[64, 3, 3, 64], 0.25, Dtype::Float32);
        for backend in [crate::quantum::Backend::Cpu, crate::quantum::Backend::Metal] {
            let actual =
                crate::quantum::resnet_bytes(&[1, 7, 9, 16], 2, 4, &conv1, &conv2, backend);
            let independent = crate::quantum::resnet_bytes_from_metadata(
                &[1, 7, 9, 16],
                2,
                4,
                mlx_gen::ios_quantum::ConvOperandMetadata {
                    shape: &[64, 3, 3, 32],
                    element_bytes: 2,
                },
                mlx_gen::ios_quantum::ConvOperandMetadata {
                    shape: &[64, 3, 3, 64],
                    element_bytes: 4,
                },
                backend,
            );
            assert_eq!(
                actual, independent,
                "precast element widths, concat and shortcut geometry"
            );
        }
        mask |= 16;
        eprintln!("SDXL_B_MECHANISM 4 independent_mixed_width_resnet_array_metadata");
        assert_eq!(
            mask, 0b1_1111,
            "all ordered B mechanisms reached the normal body end"
        );
    }
    completed.body_completed = true;
}

/// Actual direct broadcast installs whose output exceeds the ENTIRE legacy guard, including its
/// unchanged page slack. Geometry comes from that source contract, never a host-sized golden.
/// Native allocator coordinates below are separate from logical extents and process/physical RSS.
#[test]
fn broadcast_install_cost_exceeds_legacy_budget() {
    let Some(mut completed) =
        completed_cpu_child("pricing::broadcast_install_cost_exceeds_legacy_budget")
    else {
        return;
    };
    {
        use mlx_gen::adapters::{AdaptableConv2d, AdaptableLinear};
        let float_width = crate::quantum::dtype_bytes(Dtype::Float32);
        // Both source F32 paths: one delta-width buffer plus one merged-base-width buffer;
        // already materialized base => unread0, F32 factors => small0. The helper includes its
        // complete canonical PAGE_BYTES/page_slack contract. This is GEOMETRY, not the oracle.
        let legacy = crate::quantum::adapter_install_bytes(1, &[float_width, float_width], 0, 0);
        let elements = legacy
            .checked_div(float_width)
            .unwrap()
            .checked_add(1)
            .unwrap();
        let rows = i32::try_from(elements).expect("bounded source-derived broadcast rows");
        let len = usize::try_from(elements).unwrap();
        let mut mask = 0u8;
        let mut cases = 0usize;
        for (index, convolution) in [(0, true), (1, false)] {
            let kind = if convolution { "conv" } else { "shortcut" };
            assert_eq!(
                unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
                0,
                "direct broadcast pre-input CPU synchronization"
            );
            let mut initial_active = 0usize;
            assert_eq!(
                unsafe { mlx_sys::mlx_get_active_memory(&mut initial_active) },
                0,
                "direct broadcast initial native allocator counter"
            );
            let shape = if convolution {
                vec![rows, 1, 1, 1]
            } else {
                vec![rows, 1]
            };
            let base = Array::from_slice(&vec![0.0f32; len], &shape);
            base.eval().unwrap();
            let retained_old = base.clone();
            let down = Array::from_slice(&[0.5f32], &[1, 1, 1, 1]);
            let up = Array::from_slice(&[0.5f32], &[1, 1, 1, 1]);
            down.eval().unwrap();
            up.eval().unwrap();
            let delta = mlx_gen::adapters::conv_lora_delta(&down, &up, 1.0, 1.0, 1.0).unwrap();
            let delta = if convolution {
                delta
            } else {
                delta.reshape(&[1, 1]).unwrap()
            };
            delta.eval().unwrap();
            assert!(mlx_gen::array::is_materialized(&retained_old));
            assert!(mlx_gen::array::is_materialized(&delta));
            assert_eq!(delta.dtype(), Dtype::Float32);
            assert_eq!(delta.try_as_slice::<f32>().unwrap(), &[0.25]);
            assert_eq!(
                unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
                0,
                "materialized direct broadcast input CPU synchronization"
            );
            let old_bytes = (retained_old.size() as u64)
                .checked_mul(crate::quantum::element_bytes(&retained_old))
                .unwrap();
            let mut before = 0usize;
            assert_eq!(
                unsafe { mlx_sys::mlx_get_active_memory(&mut before) },
                0,
                "direct broadcast native input-baseline counter"
            );
            // Establish this CPU allocator coordinate with an ACTUAL newly allocated retained base.
            // If this domain does not charge that owner, fail: no zero/NoEvidence/skip substitution.
            assert!((before as u64) >= (initial_active as u64).checked_add(old_bytes).unwrap(),
                "{kind}: native allocator coordinate did not account the newly materialized retained CPU base");
            assert_eq!(
                unsafe { mlx_sys::mlx_reset_peak_memory() },
                0,
                "direct broadcast interval native peak reset"
            );
            let (result, events) = if convolution {
                let mut native = AdaptableConv2d::new(base, None);
                observe(PLENTY, || {
                    crate::adapters::install_conv_delta_at(
                        &mut native,
                        &delta,
                        up.dtype(),
                        0,
                        "direct_broadcast.conv",
                    )?;
                    Ok(native.weight().clone())
                })
            } else {
                let mut native = AdaptableLinear::dense(base, None);
                observe(PLENTY, || {
                    crate::adapters::install_delta_at(
                        &mut native,
                        delta.clone(),
                        &[float_width],
                        0,
                        "direct_broadcast.shortcut",
                        mlx_gen::ios_quantum::allocated_bytes(float_width),
                    )?;
                    Ok(native
                        .dense_weight()
                        .expect("actual dense shortcut after merge")
                        .0
                        .clone())
                })
            };
            let output = result.unwrap();
            output.eval().unwrap();
            assert_eq!(
                unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
                0,
                "direct broadcast installed output CPU synchronization"
            );
            let mut peak = 0usize;
            assert_eq!(
                unsafe { mlx_sys::mlx_get_peak_memory(&mut peak) },
                0,
                "direct broadcast actual native interval peak"
            );
            // Readbacks/reference checks follow the peak query, so they cannot supply its peak.
            let guarded = checked(&events, ADAPTER_MERGE);
            assert_eq!(
                guarded.len(),
                1,
                "{kind}: exactly one actual Checked merge guard"
            );
            assert_eq!(
                synced(&events, ADAPTER_MERGE),
                1,
                "{kind}: exactly one actual finish"
            );
            assert!(
                matches!(events.as_slice(), [Ev::Checked(label, _), Ev::Synced(end)]
                if label == ADAPTER_MERGE && end == ADAPTER_MERGE),
                "{kind}: actual guard/finish order"
            );
            let guard = guarded[0];
            let output_bytes = (output.size() as u64)
                .checked_mul(crate::quantum::element_bytes(&output))
                .unwrap();
            assert_eq!(output.dtype(), Dtype::Float32);
            assert_eq!(output.shape(), shape.as_slice());
            assert!(
                output_bytes > legacy,
                "{kind}: source geometry must exceed the ENTIRE old budget"
            );
            // PRIMARY independent contribution oracle: actual native output size/itemwidth,
            // NOT AdapterInstallWork, adapter_install_bytes or allocated_bytes as an oracle.
            assert!(guard >= output_bytes,
                "{kind}: actual direct install guard underprices its native broadcast output: guard={guard}, native_output={output_bytes}");
            assert!(
                retained_old
                    .try_as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .all(|v| *v == 0.0),
                "{kind}: retained materialized original base must remain unchanged (no donation)"
            );
            assert!(
                output
                    .try_as_slice::<f32>()
                    .unwrap()
                    .iter()
                    .all(|v| *v == 0.25),
                "{kind}: actual broadcast install must produce the independent exact coefficient"
            );
            // Stronger SEPARATE native-allocator interval checks, never process/RSS/physical claims.
            // No upper subtraction, fitted offset, stalled kernel, cache clearing or ratio tolerance.
            assert!(
                (peak as u64) >= (before as u64).checked_add(output_bytes).unwrap(),
                "{kind}: actual native allocator peak did not charge the retained broadcast output"
            );
            assert!(
                (peak as u64) <= (before as u64).checked_add(guard).unwrap(),
                "{kind}: actual native allocator interval exceeds the guarded source budget"
            );
            cases += 1;
            mask |= 1 << index;
            if convolution {
                eprintln!("SDXL_B_COST_MECHANISM 0 direct_conv_broadcast_output_and_native_peak");
            } else {
                eprintln!(
                    "SDXL_B_COST_MECHANISM 1 direct_shortcut_broadcast_output_and_native_peak"
                );
            }
        }
        assert_eq!(cases, 2, "both direct native install mechanisms completed");
        assert_eq!(
            mask, 0b11,
            "both cost mechanisms reached actual normal-body completion"
        );
    }
    completed.body_completed = true;
}

/// A normal accepted file installs a one-column residual on a real packed SDXL projection.
/// This isolates the logical operand contribution of the HOST broadcast add, not an allocator
/// peak: native add may donate an unshared base result. No artificial retention is called public
/// ownership, and neither this extent nor the metadata price establishes physical memory.
#[test]
fn packed_residual_host_add_uses_native_broadcast_extent() {
    let Some(mut completed) =
        completed_cpu_child("pricing::packed_residual_host_add_uses_native_broadcast_extent")
    else {
        return;
    };
    {
        use mlx_gen::adapters::facts::AdapterFacts;
        use mlx_gen::adapters::{AdaptableHost, Adapter};
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let snap = Snapshot::packed(4);
        let target = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k";
        let parts: Vec<_> = target.split('.').collect();
        let file = snap.file("singleton_residual.safetensors");
        let down = Array::from_slice(&[0.125f32; 64], &[1, 64])
            .as_dtype(Dtype::Float16)
            .unwrap();
        let up = Array::from_slice(&[0.25f32], &[1, 1])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        save(
            &Weights::from_map(HashMap::from([
                (
                    format!("lora_unet_{}.lora_down.weight", target.replace('.', "_")),
                    down,
                ),
                (
                    format!("base_model.model.unet.{target}.lora_B.default.weight"),
                    up,
                ),
            ])),
            &file,
        );
        let mut native = crate::loader::load_unet_with_config_streamed(
            &snap.root,
            crate::model::DTYPE,
            &snap.unet,
            false,
        )
        .unwrap();
        let (report, work) = crate::adapters::with_adapter_work(|| {
            crate::adapters::apply_sdxl_adapters_with(
                &mut native,
                &[AdapterSpec::new(file, 1.0, AdapterKind::Lora)],
                crate::adapters::LoraCoverage::Complete,
            )
        });
        assert_eq!(
            report.unwrap().merged,
            1,
            "actual public file/classifier/packed installer accepts singleton B rows"
        );
        let linear = native.adaptable_mut(&parts).unwrap();
        let [Adapter::Lora { a, b, scale }] = linear.adapters() else {
            panic!("actual factor residual missing");
        };
        assert_eq!(*scale, 1.0);
        assert_eq!(a.shape(), &[64, 1]);
        assert_eq!(b.shape(), &[1, 1]);
        let fact = AdapterFacts::of(&linear.adapters()[0]);
        let output_dtype = crate::quantum::linear_output_dtype(linear, Dtype::Float32);
        // Choose rows from the ENTIRE old one-row operand budget plus canonical page slack.
        // The actual multirow old budget is checked independently below; no fitted ratio/offset.
        let one_row =
            crate::adapters::adapter_forward_scratch(&fact, &[1, 64], Dtype::Float32, output_dtype)
                .checked_add(crate::quantum::page_slack(1))
                .unwrap();
        let rows = i32::try_from(one_row.checked_add(1).unwrap()).unwrap();
        let count = usize::try_from(rows).unwrap().checked_mul(64).unwrap();
        let probe = Array::from_slice(&vec![1.0f32; count], &[rows, 64]);
        let base = {
            let mut bare = linear.clone();
            bare.set_adapters(Vec::new());
            bare.forward(&probe).unwrap()
        };
        let residual = linear.adapters()[0].residual(&probe).unwrap();
        let output = linear.forward(&probe).unwrap();
        output.eval().unwrap();
        assert_eq!(
            unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
            0
        );
        assert_eq!(residual.shape(), &[rows, 1]);
        assert_eq!(output.shape(), &[rows, 64]);
        assert_eq!(
            output.dtype(),
            base.dtype(),
            "inference narrowing preserves the actual host dtype"
        );
        let read = |x: &Array| {
            x.as_dtype(Dtype::Float32)
                .unwrap()
                .try_as_slice::<f32>()
                .unwrap()
                .to_vec()
        };
        assert!(
            read(&residual).iter().all(|v| *v == 2.0),
            "independent 64 * .125 * .25 nonzero residual"
        );
        let expected = mlx_rs::ops::add(&base, mlx_gen::array::scalar(2.0f32)).unwrap();
        assert_eq!(
            read(&output),
            read(&expected),
            "actual broadcast sum, not a shape-walker oracle"
        );
        let actual_bytes = (output.size() as u64)
            .checked_mul(crate::quantum::element_bytes(&output))
            .unwrap();
        let entire_legacy = crate::adapters::adapter_forward_scratch(
            &fact,
            probe.shape(),
            probe.dtype(),
            output.dtype(),
        )
        .checked_add(crate::quantum::page_slack(1))
        .unwrap();
        assert!(
            actual_bytes > entire_legacy,
            "actual host add extent must dominate the ENTIRE legacy operand budget and slack"
        );
        let selected = crate::quantum::LinearWork::of(linear);
        assert_eq!(
            selected.output_shape(probe.shape()).unwrap(),
            output.shape()
        );
        assert_eq!(selected.output_dtype(probe.dtype()), output.dtype());
        assert!(selected.operand_bytes(probe.shape(), probe.dtype()) >= actual_bytes,
            "selected operand contribution lost native full host-add work: native_output={actual_bytes}");
        assert!(
            work.iter().any(
                |event| matches!(event, crate::adapters::AdapterWork::Install(w)
            if w.target == target && w.installed[1].shape == b.shape())
            ),
            "independent file producer reached actual installed operands"
        );
        eprintln!(
            "SDXL_B_HOST_ADD_MECHANISM 0 accepted_file_native_broadcast_and_operand_contribution"
        );
    }
    completed.body_completed = true;
}

/// All four classified DOWN/UP winner combinations are supplied as complete native key orders.
/// The two compatible ranks complete BOTH files before the normal U-Net forward; the crossed
/// ranks refuse at the real dense merge after the earlier K install. Native arrays, not another
/// header replay, supply installed geometry, values, packed operands and ordered guard facts.
#[test]
fn correlated_rank_worlds_restore_effects_and_native_prefixes() {
    let Some(mut completed) =
        completed_cpu_child("pricing::correlated_rank_worlds_restore_effects_and_native_prefixes")
    else {
        return;
    };
    {
        use crate::adapters::AdapterWork;
        use mlx_gen::adapters::facts::{AdapterFacts, TensorFacts};
        use mlx_gen::adapters::{AdaptableHost, Adapter};
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let mut mask = 0u8;
        let snap = Snapshot::packed(4);
        let inputs = snap.inputs(Route::Text);
        let q = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
        let k = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k";
        let qp: Vec<_> = q.split('.').collect();
        let kp: Vec<_> = k.split('.').collect();
        let tensor = |shape: &[i32], value: f32, dtype: Dtype| {
            let count = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).unwrap()))
                .unwrap();
            let a = Array::from_slice(&vec![value; count], shape)
                .as_dtype(dtype)
                .unwrap();
            a.eval().unwrap();
            a
        };
        let read = |a: &Array| {
            a.reshape(&[-1])
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .try_as_slice::<f32>()
                .unwrap()
                .to_vec()
        };
        // Only this fixture's checkpoint changes. The original proofs and Snapshot factory stay
        // untouched. File2 restores normal Q geometry before ANY full graph is evaluated.
        let checkpoint = Weights::from_file(&inputs.unet).unwrap();
        // Resolve original producer reads before save overwrites their backing file.
        checkpoint.materialize().unwrap();
        let mut checkpoint = checkpoint.into_tensors();
        checkpoint.insert(format!("{q}.weight"), tensor(&[1, 64], 0.0, Dtype::Float16));
        checkpoint.remove(&format!("{q}.scales"));
        checkpoint.remove(&format!("{q}.biases"));
        checkpoint.remove(&format!("{q}.bias"));
        // K's real source pack starts with one output row. The first installed residual expands
        // the host to64; the second singleton residual must add over that SAME expanded host.
        let first_row = Array::from_slice(&[0i32], &[1]);
        for suffix in ["weight", "scales", "biases"] {
            let key = format!("{k}.{suffix}");
            let row = checkpoint[&key].take_axis(&first_row, 0).unwrap();
            row.eval().unwrap();
            checkpoint.insert(key, row);
        }
        save(&Weights::from_map(checkpoint), &inputs.unet);
        let headers =
            mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&inputs.unet).unwrap();
        let legacy_bundle = Priced::build(&inputs)
            .unwrap()
            .price(&regime(Route::Text))
            .unwrap()
            .0;
        let entire_legacy = legacy_bundle
            .windows
            .iter()
            .map(|w| w.logical_peak_bytes())
            .max()
            .unwrap()
            .checked_add(crate::quantum::page_slack(1))
            .unwrap();
        // DOWN alone's f32 cast exceeds the entire old bundle window, including resident and
        // slack. Power-of-two rank keeps the independent .125/rank * .25 sum exact and nonzero.
        let large_rank = entire_legacy
            .checked_div(64 * 4)
            .unwrap()
            .checked_add(1)
            .unwrap()
            .checked_next_power_of_two()
            .unwrap();
        let large_rank = i32::try_from(large_rank).unwrap();
        assert!(large_rank > 1);
        // TEST RESOURCE PRECONDITIONS, not provider pricing or a physical/RSS/native-counter
        // upper. Snapshot::build_with_text / half, test_support::Gen::{rnd,norm}, tests::Gen,
        // vae::tests::fixture_with, quant::quantize_map and save stage the fixed small setup.
        // Float8/code32 estimate generator/cast/pack/save/read stages; four pages per header
        // estimate layout/graph metadata. These coefficients are planning assumptions, not a law.
        // The already CPU-guarded setup has fixed64/128 UNet, CLIP32, VAE32 and unused ViT256
        // geometry, independent of R; Priced::build/price still constructs its original small
        // lazy shape graph. Do NOT infer a process-memory guarantee from this estimate or sampling.
        let mut float_bytes = 0u64;
        let mut code_bytes = 0u64;
        let mut root_tensors = 0u64;
        for file in [
            inputs.te1.clone(),
            inputs.te2.clone(),
            inputs.unet.clone(),
            inputs.vae.0.clone(),
            snap.file("controlnet/diffusion_pytorch_model.safetensors"),
            snap.root
                .join("ip")
                .join(crate::loader::IP_ADAPTER_IMAGE_ENCODER_FILE),
            snap.root
                .join("ip")
                .join(crate::loader::IP_ADAPTER_WEIGHT_FILES[1]),
        ] {
            let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&file)
                .expect("SDXL_B_RESOURCE_PLAN: small setup header read failed");
            for header in headers {
                use mlx_gen::gen_core::weightsmeta::Dtype as H;
                let total = match header.dtype {
                    H::U32 => &mut code_bytes,
                    H::F16 | H::BF16 | H::F32 => &mut float_bytes,
                    _ => panic!("SDXL_B_RESOURCE_PLAN: unexpected small setup dtype"),
                };
                *total = total
                    .checked_add(header.data_bytes)
                    .expect("SDXL_B_RESOURCE_PLAN: setup byte sum overflow");
                root_tensors = root_tensors
                    .checked_add(1)
                    .expect("SDXL_B_RESOURCE_PLAN: setup tensor count overflow");
            }
        }
        let setup_metadata = root_tensors
            .checked_mul(crate::quantum::PAGE_BYTES)
            .and_then(|n| n.checked_mul(4))
            .expect("SDXL_B_RESOURCE_PLAN: setup metadata estimate overflow");
        let setup_estimate = float_bytes
            .checked_mul(8)
            .and_then(|n| {
                code_bytes
                    .checked_mul(32)
                    .and_then(|codes| n.checked_add(codes))
            })
            .and_then(|n| n.checked_add(setup_metadata))
            .and_then(|n| n.checked_add(8 * 1024 * 1024))
            .expect("SDXL_B_RESOURCE_PLAN: setup estimate overflow");
        let rank = u64::try_from(large_rank).expect("SDXL_B_RESOURCE_PLAN: invalid rank");
        let rank_elements = large_rank
            .checked_mul(64)
            .expect("SDXL_B_RESOURCE_PLAN: native rank tensor element count overflow");
        usize::try_from(rank_elements)
            .ok()
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .expect("SDXL_B_RESOURCE_PLAN: host rank tensor byte count overflow");
        // Whole-test R-sized data production: generation host/native/cast1280R, save staging512R,
        // six actual first-file raw loads1536R, all selected F32 factor casts1536R =4864R;
        // round to5120R plus8MiB for fixed products/readbacks. No release/donation credit.
        let rank_work = rank
            .checked_mul(5120)
            .and_then(|n| n.checked_add(8 * 1024 * 1024))
            .expect("SDXL_B_RESOURCE_PLAN: rank payload work overflow");
        let planned_work = setup_estimate
            .checked_add(rank_work)
            .expect("SDXL_B_RESOURCE_PLAN: combined payload work overflow");
        let large_fmas = rank
            .checked_mul(8192)
            .expect("SDXL_B_RESOURCE_PLAN: large reconstruction work overflow");
        assert!(setup_estimate <= 512 * 1024 * 1024,
            "SDXL_B_RESOURCE_PLAN: fixed setup estimate {setup_estimate} exceeds512MiB; no R-sized allocation permitted");
        assert!(planned_work <= 1024 * 1024 * 1024,
            "SDXL_B_RESOURCE_PLAN: setup+rank payload work {planned_work} exceeds1GiB at R={rank}; no R-sized allocation permitted");
        assert!(large_fmas <= 1 << 30,
            "SDXL_B_RESOURCE_PLAN: large reconstruction work {large_fmas} exceeds2^30 FMAs at R={rank}");
        eprintln!("SDXL_B_RESOURCE_PLAN accepted planning estimates only: setup={setup_estimate} rank_work={rank_work} combined={planned_work} R={rank} rank_elements={rank_elements} large_fmas={large_fmas}; NOT physical/RSS bounds");
        let qa = format!("lora_unet_{}", q.replace('.', "_"));
        let qb = format!("base_model.model.unet.{q}");
        let ka = format!("lora_unet_{}", k.replace('.', "_"));
        let da = format!("{qa}.lora_down.weight");
        let db = format!("{qb}.lora_A.default.weight");
        let ua = format!("{qa}.lora_up.weight");
        let ub = format!("{qb}.lora_B.default.weight");
        let kd = format!("{ka}.lora_down.weight");
        let ku = format!("{ka}.lora_up.weight");
        let alpha = "base_model.model.unet.unresolved.alpha".to_owned();
        let skipped = "text_encoder.skipped.weight".to_owned();
        let first = snap.file("correlated_first.safetensors");
        save(
            &Weights::from_map(HashMap::from([
                (da.clone(), tensor(&[1, 64], 0.125, Dtype::Float16)),
                (
                    db.clone(),
                    tensor(
                        &[large_rank, 64],
                        0.125 / large_rank as f32,
                        Dtype::Bfloat16,
                    ),
                ),
                (ua.clone(), tensor(&[1, 1], 0.25, Dtype::Float16)),
                (ub.clone(), tensor(&[64, large_rank], 0.25, Dtype::Bfloat16)),
                (kd.clone(), tensor(&[1, 64], 0.125, Dtype::Float16)),
                (ku.clone(), tensor(&[64, 1], 0.25, Dtype::Float16)),
                (alpha.clone(), tensor(&[1], 1.0, Dtype::Float16)),
                (skipped.clone(), tensor(&[1], 0.5, Dtype::Float32)),
            ])),
            &first,
        );
        let second = snap.file("correlated_second.safetensors");
        save(
            &Weights::from_map(HashMap::from([
                (da.clone(), tensor(&[1, 64], 0.125, Dtype::Float16)),
                (ua.clone(), tensor(&[64, 1], 0.25, Dtype::Float16)),
                (kd.clone(), tensor(&[2, 64], 0.0625, Dtype::Float16)),
                (ku.clone(), tensor(&[1, 2], 0.5, Dtype::Bfloat16)),
            ])),
            &second,
        );
        let second_order = vec![ku.clone(), da.clone(), kd.clone(), ua.clone()];
        let order = |down_large: bool, up_large: bool| {
            let mut keys = vec![skipped.clone(), kd.clone(), ku.clone(), alpha.clone()];
            keys.extend(if down_large {
                [da.clone(), db.clone()]
            } else {
                [db.clone(), da.clone()]
            });
            keys.extend(if up_large {
                [ua.clone(), ub.clone()]
            } else {
                [ub.clone(), ua.clone()]
            });
            keys
        };
        let mut spec =
            mlx_gen::gen_core::LoadSpec::new(mlx_gen::WeightsSource::Dir(snap.root.clone()));
        spec.adapters = vec![
            AdapterSpec::new(first, 1.0, AdapterKind::Lora),
            AdapterSpec::new(second, 1.0, AdapterKind::Lora),
        ];
        struct NativeCase {
            down: i32,
            up: i32,
            work: Vec<AdapterWork>,
            guards: Vec<Ev>,
            packed: Option<(crate::quantum::LinearWork, crate::quantum::LinearWork)>,
            k_output: Option<(Vec<i32>, Dtype)>,
            packs: Vec<u64>,
            forward: Vec<(String, u64)>,
        }
        let mut native_cases = Vec::new();
        for down_large in [false, true] {
            for up_large in [false, true] {
                let mut native = crate::loader::load_unet_with_config_streamed(
                    &snap.root,
                    crate::model::DTYPE,
                    &snap.unet,
                    false,
                )
                .unwrap();
                let probe = tensor(&[1, 64], 1.0, Dtype::Float32);
                let old_k = native.adaptable_mut(&kp).unwrap().forward(&probe).unwrap();
                old_k.eval().unwrap();
                let (run, guards) = observe(PLENTY, || {
                    Ok(crate::adapters::with_adapter_work(|| {
                        crate::adapters::apply_sdxl_adapters_in_key_order_for_test(
                            &mut native,
                            &spec.adapters,
                            crate::adapters::LoraCoverage::Complete,
                            &[order(down_large, up_large), second_order.clone()],
                        )
                    }))
                });
                let (result, work) = run.unwrap();
                let mut case = NativeCase {
                    down: if down_large { large_rank } else { 1 },
                    up: if up_large { large_rank } else { 1 },
                    work,
                    guards,
                    packed: None,
                    k_output: None,
                    packs: Vec::new(),
                    forward: Vec::new(),
                };
                if down_large != up_large {
                    assert!(
                        result.is_err(),
                        "crossed native factor ranks must refuse in the existing matmul"
                    );
                    assert_eq!(
                        case.work
                            .iter()
                            .filter(|w| matches!(w, AdapterWork::Install(_)))
                            .count(),
                        1,
                        "real BTree K installs before the refused Q; file2 must not run"
                    );
                    assert_eq!(
                        case.work
                            .iter()
                            .filter(|w| matches!(w, AdapterWork::FileMaterialized(_)))
                            .count(),
                        1
                    );
                    assert!(
                        matches!(case.guards.last(), Some(Ev::Checked(label, _)) if label == ADAPTER_MERGE),
                        "native failed Q reaches its original guard without a fabricated finish"
                    );
                } else {
                    assert_eq!(result.unwrap().merged, 4);
                    let q_linear = native.adaptable_mut(&qp).unwrap();
                    let q_weight = q_linear.dense_weight().unwrap().0;
                    assert_eq!(q_weight.shape(), &[64, 64]);
                    assert!(
                        read(q_weight).iter().all(|v| *v == 0.0625),
                        "two independent nonzero .03125 merges"
                    );
                    assert!(read(&q_linear.forward(&probe).unwrap())
                        .iter()
                        .all(|v| *v == 4.0));
                    let k_linear = native.adaptable_mut(&kp).unwrap();
                    assert_eq!(k_linear.adapters().len(), 2);
                    for (index, (rank, coefficient)) in
                        [(1, 0.25), (2, 0.5)].into_iter().enumerate()
                    {
                        let Adapter::Lora { a, b, scale } = &k_linear.adapters()[index] else {
                            panic!("ordered native K residual missing");
                        };
                        assert_eq!(*scale, 1.0);
                        assert_eq!(a.shape(), &[64, rank]);
                        assert_eq!(b.shape(), &[rank, if index == 0 { 64 } else { 1 }]);
                        assert!(read(b).iter().all(|v| *v == coefficient));
                    }
                    let expected_k = mlx_rs::ops::add(
                        mlx_rs::ops::add(&old_k, tensor(&[1, 64], 2.0, Dtype::Float32)).unwrap(),
                        mlx_gen::array::scalar(4.0f32),
                    )
                    .unwrap();
                    let actual_k = k_linear.forward(&probe).unwrap();
                    assert_eq!(
                        actual_k.shape(),
                        &[1, 64],
                        "first residual expands the one-row source pack's output"
                    );
                    assert_eq!(read(&actual_k), read(&expected_k),
                        "native rank1 F16 then rank2 BF16 residuals contribute ordered 2+4 over the expanded host");
                    case.k_output = Some((actual_k.shape().to_vec(), actual_k.dtype()));
                    let residuals_before: Vec<_> =
                        k_linear.adapters().iter().map(AdapterFacts::of).collect();
                    let (packed, pack_events) = observe(PLENTY, || {
                        crate::quant::guarded_load_quantize(|| native.quantize(4))
                    });
                    packed.unwrap();
                    case.packs = checked(&pack_events, LOAD_QUANTIZE);
                    assert_eq!(
                        case.packs.len(),
                        1,
                        "only the changed dense Q is packed; K base and residuals stay untouched"
                    );
                    let q_packed =
                        crate::quantum::LinearWork::of(native.adaptable_mut(&qp).unwrap());
                    let k_packed =
                        crate::quantum::LinearWork::of(native.adaptable_mut(&kp).unwrap());
                    assert_eq!(
                        k_packed.residuals, residuals_before,
                        "actual base-only packing preserves every residual owner/order"
                    );
                    let x = tensor(&[1, LATENT, LATENT, 4], 0.125, crate::model::DTYPE);
                    let context = tensor(&[1, TOKENS, snap.context], 0.0625, crate::model::DTYPE);
                    let pooled = tensor(&[1, 16], 0.125, crate::model::DTYPE);
                    let ids = crate::pipeline::text_time_ids(1);
                    let (output, forward_events) = observe(PLENTY, || {
                        native.forward(&x, 500.0, &context, &pooled, &ids)
                    });
                    let output = output.unwrap();
                    let values = read(&output);
                    assert_eq!(output.shape(), &[1, LATENT, LATENT, 4]);
                    assert!(
                        values.iter().all(|v| v.is_finite()) && values.iter().any(|v| *v != 0.0)
                    );
                    case.forward = forward_events
                        .into_iter()
                        .filter_map(|e| match e {
                            Ev::Checked(label, bytes) => Some((label, bytes)),
                            _ => None,
                        })
                        .collect();
                    assert!(!case.forward.is_empty());
                    case.packed = Some((q_packed, k_packed));
                }
                native_cases.push(case);
            }
        }
        assert_eq!(
            native_cases.len(),
            4,
            "all role choices, not selected convenient native winners"
        );
        mask |= 1;
        eprintln!(
            "SDXL_B_CORRELATED_MECHANISM 0 all_native_rank_orders_nonzero_and_refused_siblings"
        );
        let describe = |facts: &[TensorFacts]| {
            facts
                .iter()
                .map(|a| (a.shape.clone(), a.dtype))
                .collect::<Vec<_>>()
        };
        let installs = |work: &[AdapterWork]| {
            work.iter()
                .filter_map(|w| match w {
                    AdapterWork::Install(w) => {
                        Some((w.target.clone(), w.storage, describe(&w.installed)))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let find_case = |work: &[AdapterWork]| {
            let first_q = work
                .iter()
                .find_map(|w| match w {
                    AdapterWork::GuardAttempt(w) if w.target == q => Some(w),
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                first_q.parents[0].shape,
                &[1, 64],
                "sibling DFS must undo the prior shape-changing Q base"
            );
            let down = first_q.parents[1].shape[0];
            let up = first_q.parents[2].shape[1];
            native_cases
                .iter()
                .position(|c| c.down == down && c.up == up)
                .unwrap()
        };
        let mut seen = [0usize; 4];
        crate::adapters::visit_configured_adapter_outcomes(&spec, &headers, &snap.unet, |world| {
            let index = find_case(world.work);
            seen[index] += 1;
            let native = &native_cases[index];
            assert_eq!(world.refusal.is_none(), native.packed.is_some());
            assert_eq!(installs(world.work), installs(&native.work), "actual installed target/dtype/geometry/order, including refused prefixes");
            let q_attempts: Vec<_> = world.work.iter().filter_map(|w| match w {
                AdapterWork::GuardAttempt(w) if w.target == q => Some(w), _ => None,
            }).collect();
            if native.packed.is_some() {
                assert_eq!(q_attempts.len(), 2);
                let actual_second_q = native.work.iter().filter_map(|w| match w {
                    AdapterWork::Install(w) if w.target == q => Some(w), _ => None,
                }).nth(1).unwrap();
                assert_eq!(describe(&q_attempts[1].parents[..3]), describe(&actual_second_q.parents[..3]),
                    "file2 must consume the SAME world's installed base, not a sibling or dtype maximum");
            }
            Ok(())
        }).unwrap();
        assert_eq!(seen, [1; 4]);
        mask |= 2;
        eprintln!("SDXL_B_CORRELATED_MECHANISM 1 native_world_ancestry_and_changed_target_undo");

        let mut priced = Priced::build_configured(&inputs, &spec).unwrap();
        let before_packs: Vec<_> = priced
            .load_quantize
            .iter()
            .map(|p| (p.before_resident, p.scratch))
            .collect();
        let before_q = describe(&priced.unet_effects.as_ref().unwrap().0[q].weights);
        let mut observed = [0usize; 4];
        let mut native_merge_max = 0;
        let pricing = priced.price_configured_observing(&spec, &inputs.unet, inputs.quant_bits, &regime(Route::Text),
            |world, selected, ledger, candidate| {
                let index = find_case(world.work);
                observed[index] += 1;
                let native = &native_cases[index];
                let (nq, nk) = native.packed.as_ref().expect("only complete worlds reach the real consumer");
                let effects = selected.unet_effects.as_ref().unwrap();
                assert_eq!(describe(&effects.0[q].weights), describe(&nq.weights), "actual Q packed output operands");
                assert_eq!(describe(&effects.0[k].weights), describe(&nk.weights), "source-packed K base unchanged");
                assert_eq!(effects.0[k].residuals, nk.residuals, "actual selected ordered residual effects cannot be omitted");
                let (native_shape, native_dtype) = native.k_output.as_ref().unwrap();
                assert_eq!(effects.0[k].output_shape(&[1, 64]).unwrap(), *native_shape,
                    "ordered effects must carry prior residual expansion, not only original base or last residual width");
                assert_eq!(effects.0[k].output_dtype(Dtype::Float32), *native_dtype);
                assert_eq!(ledger.residuals, nk.residuals, "ledger and installed native operands belong to this same world");
                assert_eq!(selected.load_quantize.iter().map(|p| p.scratch).collect::<Vec<_>>(), native.packs,
                    "metadata-only ordered pack uses the actual post-file2 base");
                for (label, bytes) in &native.forward {
                    assert!(candidate.windows.iter().any(|w| &w.label == label && w.scratch_bytes >= *bytes),
                        "selected WHOLE forward lost independent native guarded {label}");
                }
                let guards = checked(&native.guards, ADAPTER_MERGE);
                let actual_max = guards.into_iter().max().unwrap();
                native_merge_max = native_merge_max.max(actual_max);
                assert!(candidate.windows.iter().any(|w| w.label == ADAPTER_MERGE && w.scratch_bytes >= actual_max));
                assert!(candidate.phase(IosPhase::Denoise).is_some());
                Ok(())
            }).unwrap();
        assert_eq!(
            observed,
            [1, 0, 0, 1],
            "both complete worlds reach effect/ledger/pack/wholeforward before enveloping"
        );
        let small_merge = checked(&native_cases[0].guards, ADAPTER_MERGE)
            .into_iter()
            .max()
            .unwrap();
        assert!(native_merge_max > entire_legacy && native_merge_max > small_merge,
            "large-rank actual merge dominates the ENTIRE original bundle budget and the sibling's guard");
        assert!(
            pricing
                .windows
                .iter()
                .any(|w| w.label == ADAPTER_MERGE && w.scratch_bytes >= native_merge_max),
            "final whole-world envelope must not select the smaller native merge candidate"
        );
        assert_eq!(
            describe(&priced.unet_effects.as_ref().unwrap().0[q].weights),
            before_q,
            "selected effects restored after all worlds"
        );
        assert_eq!(
            priced
                .load_quantize
                .iter()
                .map(|p| (p.before_resident, p.scratch))
                .collect::<Vec<_>>(),
            before_packs,
            "selected pack ledger cannot leak to the next world/request"
        );
        let failed_consumer = priced.price_configured_observing(
            &spec,
            &inputs.unet,
            inputs.quant_bits,
            &regime(Route::Text),
            |world, selected, _, _| {
                let native = &native_cases[find_case(world.work)];
                assert_eq!(
                    describe(&selected.unet_effects.as_ref().unwrap().0[q].weights),
                    describe(&native.packed.as_ref().unwrap().0.weights)
                );
                Err(mlx_gen::Error::Msg(
                    "fixture selected-consumer failure after real world price".into(),
                ))
            },
        );
        assert!(
            matches!(failed_consumer, Err(ref e) if e.to_string().contains("fixture selected-consumer failure"))
        );
        assert_eq!(
            describe(&priced.unet_effects.as_ref().unwrap().0[q].weights),
            before_q,
            "consumer error must restore actual selected effects before propagating"
        );
        assert_eq!(
            priced
                .load_quantize
                .iter()
                .map(|p| (p.before_resident, p.scratch))
                .collect::<Vec<_>>(),
            before_packs,
            "consumer error must restore ordered packs before propagating"
        );
        mask |= 4;
        eprintln!("SDXL_B_CORRELATED_MECHANISM 2 actual_consumer_effects_ordered_pack_forward_and_dominating_envelope");

        let malformed = snap.file("correlated_final_bad_alpha.safetensors");
        let valid_alpha = format!("{qa}.alpha");
        let bad_alpha = format!("{qb}.alpha");
        save(
            &Weights::from_map(HashMap::from([
                (valid_alpha.clone(), tensor(&[1], 1.0, Dtype::Float32)),
                (bad_alpha.clone(), tensor(&[2], 1.0, Dtype::Float16)),
            ])),
            &malformed,
        );
        spec.adapters
            .push(AdapterSpec::new(malformed, 1.0, AdapterKind::Lora));
        let mut refused = [0usize; 4];
        for large in [false, true] {
            let mut native = crate::loader::load_unet_with_config_streamed(
                &snap.root,
                crate::model::DTYPE,
                &snap.unet,
                false,
            )
            .unwrap();
            let (result, actual) = crate::adapters::with_adapter_work(|| {
                crate::adapters::apply_sdxl_adapters_in_key_order_for_test(
                    &mut native,
                    &spec.adapters,
                    crate::adapters::LoraCoverage::Complete,
                    &[
                        order(large, large),
                        second_order.clone(),
                        vec![valid_alpha.clone(), bad_alpha.clone()],
                    ],
                )
            });
            assert!(
                result.is_err(),
                "actual third-file overwritten malformed alpha must refuse"
            );
            assert_eq!(
                actual
                    .iter()
                    .filter(|w| matches!(w, AdapterWork::FileMaterialized(_)))
                    .count(),
                3
            );
            assert_eq!(
                installs(&actual),
                installs(&native_cases[if large { 3 } else { 0 }].work),
                "third-file parser refusal retains both prior files' actual success"
            );
        }
        let expected_raw: u64 = spec
            .adapters
            .iter()
            .map(|a| {
                mlx_gen::ios_quantum::load_floor_bytes(std::slice::from_ref(&a.path), |_| true)
                    .unwrap()
            })
            .sum();
        crate::adapters::visit_configured_adapter_outcomes(&spec, &headers, &snap.unet, |world| {
            let index = find_case(world.work);
            refused[index] += 1;
            assert!(world.refusal.is_some());
            assert_eq!(installs(world.work), installs(&native_cases[index].work));
            let ledger = crate::ios_pricing::AdapterLoadLedger::from_work_checked(world.work, 0)?;
            if native_cases[index].packed.is_some() {
                assert!(world.refusal.as_ref().unwrap().contains("alpha"));
                assert_eq!(
                    ledger.raw_parent_upper, expected_raw,
                    "no file-end release or refusal prefix reset"
                );
                assert_eq!(
                    ledger.residuals,
                    native_cases[index].packed.as_ref().unwrap().1.residuals
                );
            } else {
                assert_eq!(
                    world
                        .work
                        .iter()
                        .filter(|w| matches!(w, AdapterWork::FileMaterialized(_)))
                        .count(),
                    1,
                    "earlier incompatible-rank refusal must not materialize later files"
                );
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(refused, [1; 4]);
        mask |= 8;
        eprintln!(
            "SDXL_B_CORRELATED_MECHANISM 3 native_chronological_refusal_and_prior_file_roots"
        );
        assert_eq!(
            mask, 0b1111,
            "all ordered correlated mechanisms reach the true normal body end"
        );
    }
    completed.body_completed = true;
}

/// Native PEFT constructor order, not a shared header formula as its own oracle. Every row starts
/// with a nonzero real LoRA file so a later constructor refusal must preserve that prior install.
/// Full factors shadow malformed decomposed products; packed FULL-scale finiteness is deliberately
/// NOT the dense positive-metadata-rank policy. No test adds a new accepted-shape restriction.
#[test]
fn peft_lokr_scale_gate_precedes_factor_reconstruction() {
    let Some(mut completed) =
        completed_cpu_child("pricing::peft_lokr_scale_gate_precedes_factor_reconstruction")
    else {
        return;
    };
    {
        use crate::adapters::{AdapterInstallStorage as S, AdapterWork};
        use mlx_gen::adapters::{AdaptableHost, Adapter};
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let target = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";
        let parts: Vec<_> = target.split('.').collect();
        let tensor = |shape: &[i32], value: f32, dtype: Dtype| {
            let count = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).unwrap()))
                .unwrap();
            let a = Array::from_slice(&vec![value; count], shape)
                .as_dtype(dtype)
                .unwrap();
            a.eval().unwrap();
            a
        };
        let read = |a: &Array| {
            a.reshape(&[-1])
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .try_as_slice::<f32>()
                .unwrap()
                .to_vec()
        };
        let mut rows = 0usize;
        let mut mask = 0u8;
        for packed in [true, false] {
            let snap = Snapshot::packed(4);
            let inputs = snap.inputs(Route::Text);
            if !packed {
                let checkpoint = Weights::from_file(&inputs.unet).unwrap();
                checkpoint.materialize().unwrap();
                let mut checkpoint = checkpoint.into_tensors();
                checkpoint.insert(
                    format!("{target}.weight"),
                    tensor(&[64, 64], 0.0, Dtype::Float16),
                );
                checkpoint.remove(&format!("{target}.scales"));
                checkpoint.remove(&format!("{target}.biases"));
                save(&Weights::from_map(checkpoint), &inputs.unet);
            }
            let headers =
                mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&inputs.unet)
                    .unwrap();
            let first = snap.file("peft-prior-nonzero-lora.safetensors");
            let stem = format!("lora_unet_{}", target.replace('.', "_"));
            save(
                &Weights::from_map(HashMap::from([
                    (
                        format!("{stem}.lora_down.weight"),
                        tensor(&[1, 64], 0.125, Dtype::Float16),
                    ),
                    (
                        format!("{stem}.lora_up.weight"),
                        tensor(&[64, 1], 0.25, Dtype::Float16),
                    ),
                ])),
                &first,
            );
            for (name, rank, alpha, strength, full, unmatched) in [
                ("scale-first", "0", "1", 1.0, false, false),
                ("factor-after-valid-scale", "1", "1", 1.0, false, false),
                ("full-over-shadow", "2", "4", 0.5, true, false),
                (
                    "negative-rank-storage-specific",
                    "-2",
                    "4",
                    0.5,
                    true,
                    false,
                ),
                ("zero-full-scale", "2", "0", 1.0, true, false),
                ("unmatched-later-file", "0", "1", 1.0, false, true),
            ] {
                let raw = if unmatched {
                    "not_a_unet_target"
                } else {
                    target
                };
                let mut data = vec![
                    (
                        format!("{raw}.lokr_w1_a"),
                        tensor(&[2, 3], 0.25, Dtype::Float32),
                    ),
                    (
                        format!("{raw}.lokr_w1_b"),
                        tensor(&[5, 2], 0.5, Dtype::Float32),
                    ),
                ];
                if full {
                    data.push((
                        format!("{raw}.lokr_w1"),
                        tensor(&[8, 8], 0.25, Dtype::Bfloat16),
                    ));
                    data.push((
                        format!("{raw}.lokr_w2"),
                        tensor(&[8, 8], 0.5, Dtype::Float16),
                    ));
                } else {
                    data.push((
                        format!("{raw}.lokr_w2"),
                        tensor(&[32, 32], 0.5, Dtype::Float32),
                    ));
                }
                let second = snap.file(&format!("peft-{name}.safetensors"));
                let metadata: HashMap<String, String> =
                    [("networkType", "lokr"), ("rank", rank), ("alpha", alpha)]
                        .into_iter()
                        .map(|(k, v)| (k.into(), v.into()))
                        .collect();
                Array::save_safetensors(
                    data.iter().map(|(k, a)| (k.as_str(), a)),
                    &metadata,
                    &second,
                )
                .unwrap();
                let mut spec = mlx_gen::gen_core::LoadSpec::new(mlx_gen::WeightsSource::Dir(
                    snap.root.clone(),
                ));
                spec.adapters = vec![
                    AdapterSpec::new(first.clone(), 1.0, AdapterKind::Lora),
                    AdapterSpec::new(second, strength, AdapterKind::Lokr),
                ];
                let mut native = crate::loader::load_unet_with_config_streamed(
                    &snap.root,
                    crate::model::DTYPE,
                    &snap.unet,
                    false,
                )
                .unwrap();
                let probe = tensor(&[1, 64], 1.0, Dtype::Float32);
                let baseline = native
                    .adaptable_mut(&parts)
                    .unwrap()
                    .forward(&probe)
                    .unwrap();
                baseline.eval().unwrap();
                let (run, events) = observe(PLENTY, || {
                    Ok(crate::adapters::with_adapter_work(|| {
                        crate::adapters::apply_sdxl_adapters_with(
                            &mut native,
                            &spec.adapters,
                            crate::adapters::LoraCoverage::Complete,
                        )
                    }))
                });
                let (result, actual) = run.unwrap();
                let merge_events: Vec<_> = events
                    .iter()
                    .filter(|event| match event {
                        Ev::Checked(label, _) | Ev::Synced(label) => label == ADAPTER_MERGE,
                    })
                    .cloned()
                    .collect();
                let expected_failure = if name == "scale-first" {
                    Some(if packed {
                        "non-finite"
                    } else {
                        "invalid metadata scale"
                    })
                } else if name == "factor-after-valid-scale" {
                    Some("matmul")
                } else if name == "negative-rank-storage-specific" && !packed {
                    Some("invalid metadata scale")
                } else {
                    None
                };
                let native_error = result.as_ref().err().map(|e| e.to_string().to_lowercase());
                assert_eq!(
                    result.is_err(),
                    expected_failure.is_some(),
                    "{packed}/{name}: actual native acceptance"
                );
                if let Some(family) = expected_failure {
                    assert!(
                        native_error.as_ref().unwrap().contains(family),
                        "{packed}/{name}: native first error family"
                    );
                    assert_eq!(checked(&events, ADAPTER_MERGE).len(), 2);
                    assert_eq!(
                        synced(&events, ADAPTER_MERGE),
                        1,
                        "failed constructor cannot fabricate a finish"
                    );
                    assert!(matches!(merge_events.as_slice(), [Ev::Checked(_, _), Ev::Synced(_), Ev::Checked(_, _)]),
                        "{packed}/{name}: prior finish then failed constructor guard in actual native order");
                } else {
                    assert_eq!(
                        result.as_ref().unwrap().merged,
                        if unmatched { 1 } else { 2 }
                    );
                    assert_eq!(
                        checked(&events, ADAPTER_MERGE).len(),
                        if unmatched { 1 } else { 2 }
                    );
                    assert_eq!(
                        synced(&events, ADAPTER_MERGE),
                        if unmatched { 1 } else { 2 }
                    );
                    assert!(matches!(merge_events.as_slice(), [Ev::Checked(_, _), Ev::Synced(_)]) && unmatched
                        || matches!(merge_events.as_slice(), [Ev::Checked(_, _), Ev::Synced(_), Ev::Checked(_, _), Ev::Synced(_)]) && !unmatched,
                        "{packed}/{name}: each actual successful native constructor finishes in order");
                }
                let describe = |work: &[AdapterWork]| {
                    work.iter()
                        .filter_map(|w| match w {
                            AdapterWork::Install(w) => Some((
                                w.target.clone(),
                                w.storage,
                                w.installed
                                    .iter()
                                    .map(|a| (a.shape.clone(), a.dtype))
                                    .collect::<Vec<_>>(),
                            )),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                };
                let mut outcomes = 0;
                crate::adapters::visit_configured_adapter_outcomes(&spec, &headers, &snap.unet, |world| {
                    outcomes += 1;
                    assert_eq!(world.refusal.is_some(), expected_failure.is_some(), "{packed}/{name}: native/header acceptance");
                    if let Some(family) = expected_failure {
                        assert!(world.refusal.as_ref().unwrap().to_lowercase().contains(family),
                            "{packed}/{name}: header must preserve native scale-before-factor first-error family");
                    }
                    assert_eq!(describe(world.work), describe(&actual), "{packed}/{name}: actual prior-file install prefix/layout");
                    assert_eq!(world.work.iter().filter(|w| matches!(w, AdapterWork::FileMaterialized(_))).count(), 2);
                    assert_eq!(world.work.iter().filter(|w| matches!(w, AdapterWork::GuardAttempt(_))).count(),
                        checked(&events, ADAPTER_MERGE).len(), "{packed}/{name}: actual reached guards");
                    Ok(())
                }).unwrap();
                assert_eq!(outcomes, 1);
                let linear = native.adaptable_mut(&parts).unwrap();
                let prior = mlx_rs::ops::add(&baseline, mlx_gen::array::scalar(2.0f32)).unwrap();
                let delta: f32 = if expected_failure.is_some() || unmatched || alpha == "0" {
                    0.0
                } else if rank == "-2" {
                    -8.0
                } else {
                    8.0
                };
                let expected = if delta == 0.0 {
                    prior
                } else {
                    mlx_rs::ops::add(&prior, mlx_gen::array::scalar(delta)).unwrap()
                };
                assert_eq!(
                    read(&linear.forward(&probe).unwrap()),
                    read(&expected),
                    "{packed}/{name}: independent nonzero prior-file and full-factor value/order"
                );
                if packed && full {
                    let Adapter::LokrStructured { factors } = &linear.adapters()[1] else {
                        panic!("actual structured factors missing");
                    };
                    assert_eq!(factors.w1.shape(), &[8, 8]);
                    assert_eq!(factors.w1.dtype(), Dtype::Float32);
                    assert_eq!(factors.w2.dtype(), Dtype::Float32);
                    assert!(read(&factors.w1).iter().all(|v| *v == 0.25));
                    let coefficient = if alpha == "0" {
                        0.0
                    } else if rank == "-2" {
                        -0.5
                    } else {
                        0.5
                    };
                    assert!(read(&factors.w2).iter().all(|v| *v == coefficient));
                    assert_eq!(linear.adapters()[1].is_disabled(), alpha == "0");
                }
                if full && expected_failure.is_none() && !unmatched {
                    let actual_last = actual
                        .iter()
                        .filter_map(|w| match w {
                            AdapterWork::Install(w) => Some(w),
                            _ => None,
                        })
                        .next_back()
                        .unwrap();
                    assert_eq!(
                        actual_last.storage,
                        if packed {
                            S::StructuredLokr
                        } else {
                            S::DenseLinear
                        }
                    );
                }
                rows += 1;
            }
            mask |= if packed { 1 } else { 2 };
        }
        assert_eq!(rows, 12, "all table branches, not a corpus-count gate");
        assert_eq!(mask, 0b11);
        eprintln!("SDXL_PEFT_GATE_MECHANISM 0 native_storage_specific_scale_before_factor_fullshadow_and_prefix");
    }
    completed.body_completed = true;
}

// Source-proposal TP mechanisms: no native call below is reached before completed_cpu_child.
const TP_PROOF_Q: &str = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q";

/// Native save_safetensors refuses zero-element tensors. Keep its exact nonempty payload writer,
/// then add ONLY the valid zero-byte alpha header entries. The real production loader/parser is
/// not bypassed or patched; a refusal to LOAD this valid file remains a native finding.
fn tp_proof_save_alpha_fixture(
    file: &std::path::Path,
    data: impl IntoIterator<Item = (String, Array)>,
    metadata: &HashMap<String, String>,
) {
    use std::io::Write;
    let data: Vec<_> = data.into_iter().collect();
    let empty: Vec<_> = data.iter().filter(|(_, a)| a.size() == 0).collect();
    Array::save_safetensors(
        data.iter()
            .filter(|(_, a)| a.size() != 0)
            .map(|(key, a)| (key.as_str(), a)),
        metadata,
        file,
    )
    .unwrap();
    if empty.is_empty() {
        return;
    } // Nonempty/large-alpha files use ONLY the original native writer.
    let original = std::fs::read(file).unwrap();
    let header_len =
        usize::try_from(u64::from_le_bytes(original[..8].try_into().unwrap())).unwrap();
    let data_start = 8usize.checked_add(header_len).unwrap();
    assert!(data_start <= original.len());
    let mut header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&original[8..data_start]).unwrap();
    let payload = &original[data_start..];
    for (key, a) in empty {
        assert!(
            key.ends_with(".alpha"),
            "fixture writer only appends empty alpha, never drops a factor"
        );
        assert!(!header.contains_key(key));
        let dtype = match a.dtype() {
            Dtype::Float16 => "F16",
            Dtype::Bfloat16 => "BF16",
            Dtype::Float32 => "F32",
            Dtype::Uint8 => "U8",
            _ => panic!("unexpected empty-alpha fixture dtype"),
        };
        header.insert(
            key.clone(),
            serde_json::json!({"dtype": dtype, "shape": a.shape(),
            "data_offsets": [payload.len(), payload.len()]}),
        );
    }
    let mut encoded = serde_json::to_vec(&header).unwrap();
    while !encoded.len().is_multiple_of(8) {
        encoded.push(b' ');
    }
    let mut output = std::fs::File::create(file).unwrap();
    output
        .write_all(&u64::try_from(encoded.len()).unwrap().to_le_bytes())
        .unwrap();
    output.write_all(&encoded).unwrap();
    output.write_all(payload).unwrap();
    output.sync_all().unwrap();
    // This branch is only the fixed-small EMPTY fixture; never copy a large-alpha file here.
    let verified = std::fs::read(file).unwrap();
    assert_eq!(
        &verified[8 + encoded.len()..],
        payload,
        "native nonempty payload bytes and relative offsets must remain unchanged"
    );
    // Do not let an expected parser `empty` error accidentally mask a native LOADER refusal.
    // This is the original production file/materialization API, not a special loader seam.
    let loaded =
        Weights::from_file(file).expect("native loader rejected valid zero-payload alpha fixture");
    loaded
        .materialize()
        .expect("native materializer rejected valid zero-payload alpha fixture");
    for (key, expected) in data.iter().filter(|(_, a)| a.size() == 0) {
        let actual = loaded
            .require(key)
            .expect("native loader omitted zero-payload alpha tensor");
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.dtype(), expected.dtype());
        assert_eq!(actual.size(), 0);
    }
}

fn tp_proof_tensor(shape: &[i32], value: f32, dtype: Dtype) -> Array {
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).unwrap()))
        .unwrap();
    let value = Array::from_slice(&vec![value; count], shape)
        .as_dtype(dtype)
        .unwrap();
    value.eval().unwrap();
    value
}

fn tp_proof_read(a: &Array) -> Vec<f32> {
    a.reshape(&[-1])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .try_as_slice::<f32>()
        .unwrap()
        .to_vec()
}

fn tp_proof_inputs(snap: &Snapshot, packed: bool) -> Inputs {
    let inputs = snap.inputs(Route::Text);
    if !packed {
        let checkpoint = Weights::from_file(&inputs.unet).unwrap();
        checkpoint.materialize().unwrap();
        let mut checkpoint = checkpoint.into_tensors();
        checkpoint.insert(
            format!("{TP_PROOF_Q}.weight"),
            tp_proof_tensor(&[64, 64], 0.0, Dtype::Float16),
        );
        checkpoint.remove(&format!("{TP_PROOF_Q}.scales"));
        checkpoint.remove(&format!("{TP_PROOF_Q}.biases"));
        checkpoint.remove(&format!("{TP_PROOF_Q}.bias"));
        save(&Weights::from_map(checkpoint), &inputs.unet);
    }
    inputs
}

fn tp_proof_prior(snap: &Snapshot) -> mlx_gen::runtime::AdapterSpec {
    use mlx_gen::runtime::{AdapterKind, AdapterSpec};
    let stem = format!("lora_unet_{}", TP_PROOF_Q.replace('.', "_"));
    let file = snap.file("tp-proof-prior.safetensors");
    save(
        &Weights::from_map(HashMap::from([
            (
                format!("{stem}.lora_down.weight"),
                tp_proof_tensor(&[1, 64], 0.125, Dtype::Float16),
            ),
            (
                format!("{stem}.lora_up.weight"),
                tp_proof_tensor(&[64, 1], 0.25, Dtype::Float16),
            ),
        ])),
        &file,
    );
    AdapterSpec::new(file, 1.0, AdapterKind::Lora)
}

fn tp_proof_spec(
    snap: &Snapshot,
    adapters: Vec<mlx_gen::runtime::AdapterSpec>,
) -> mlx_gen::gen_core::LoadSpec {
    let mut spec = mlx_gen::gen_core::LoadSpec::new(mlx_gen::WeightsSource::Dir(snap.root.clone()));
    spec.adapters = adapters;
    spec
}

/// The actual provider is the oracle: file dispatch, parser, target lookup, guards, installation,
/// failure prefix and the resulting native target forward. No full model/sampler forward is needed.
fn tp_proof_check(
    snap: &Snapshot,
    inputs: &Inputs,
    spec: &mlx_gen::gen_core::LoadSpec,
    increments: &[f32],
    native_error: Option<&str>,
    expected_guards: usize,
) -> Vec<Vec<crate::adapters::AdapterWork>> {
    use crate::adapters::AdapterWork;
    use mlx_gen::adapters::AdaptableHost;
    let parts: Vec<_> = TP_PROOF_Q.split('.').collect();
    let mut native = crate::loader::load_unet_with_config_streamed(
        &snap.root,
        crate::model::DTYPE,
        &snap.unet,
        false,
    )
    .unwrap();
    let probe = tp_proof_tensor(&[1, 64], 1.0, Dtype::Float32);
    let baseline = native
        .adaptable_mut(&parts)
        .unwrap()
        .forward(&probe)
        .unwrap();
    baseline.eval().unwrap();
    let (run, events) = observe(PLENTY, || {
        Ok(crate::adapters::with_adapter_work(|| {
            crate::adapters::apply_sdxl_adapters_with(
                &mut native,
                &spec.adapters,
                crate::adapters::LoraCoverage::Complete,
            )
        }))
    });
    let (result, actual) = run.unwrap();
    assert_eq!(
        result.is_err(),
        native_error.is_some(),
        "actual TP native acceptance"
    );
    if let Some(family) = native_error {
        assert!(
            result
                .as_ref()
                .unwrap_err()
                .to_string()
                .to_lowercase()
                .contains(family),
            "actual TP native first error: {:?}",
            result.as_ref().err()
        );
    } else {
        assert_eq!(result.unwrap().merged, expected_guards);
    }
    assert_eq!(checked(&events, ADAPTER_MERGE).len(), expected_guards);
    assert_eq!(
        synced(&events, ADAPTER_MERGE),
        expected_guards,
        "TP parser/reconstruction errors precede the failing install guard"
    );
    let merge_events: Vec<_> = events
        .iter()
        .filter(|e| match e {
            Ev::Checked(label, _) | Ev::Synced(label) => label == ADAPTER_MERGE,
        })
        .cloned()
        .collect();
    assert_eq!(merge_events.len(), expected_guards * 2);
    for pair in merge_events.chunks_exact(2) {
        assert!(
            matches!(pair, [Ev::Checked(_, _), Ev::Synced(_)]),
            "actual TP completed guards stay in native order"
        );
    }
    let describe = |work: &[AdapterWork]| {
        work.iter()
            .filter_map(|w| match w {
                AdapterWork::Install(w) => Some((
                    w.target.clone(),
                    w.storage,
                    w.installed
                        .iter()
                        .map(|t| (t.shape.clone(), t.dtype))
                        .collect::<Vec<_>>(),
                )),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let headers =
        mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&inputs.unet).unwrap();
    let mut worlds = Vec::new();
    crate::adapters::visit_configured_adapter_outcomes(spec, &headers, &snap.unet, |world| {
        assert_eq!(
            world.refusal.is_some(),
            native_error.is_some(),
            "TP native/header acceptance"
        );
        assert_eq!(
            describe(world.work),
            describe(&actual),
            "header must preserve every actual ordered same-target install and refusal prefix"
        );
        assert_eq!(
            world
                .work
                .iter()
                .filter(|w| matches!(w, AdapterWork::GuardAttempt(_)))
                .count(),
            expected_guards,
            "header cannot fabricate the failed TP install guard"
        );
        assert_eq!(
            world
                .work
                .iter()
                .filter(|w| matches!(w, AdapterWork::FileMaterialized(_)))
                .count(),
            spec.adapters.len(),
            "reached files are materialized before their parser"
        );
        worlds.push(world.work.to_vec());
        Ok(())
    })
    .unwrap();
    assert!(!worlds.is_empty());
    let mut expected = baseline;
    for &delta in increments {
        expected = mlx_rs::ops::add(&expected, mlx_gen::array::scalar(delta)).unwrap();
    }
    let actual = native
        .adaptable_mut(&parts)
        .unwrap()
        .forward(&probe)
        .unwrap();
    assert_eq!(actual.shape(), &[1, 64]);
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(
        tp_proof_read(&actual),
        tp_proof_read(&expected),
        "independent native ordered nonzero contributions and post-error state"
    );
    if native_error.is_none() {
        let native_effect = crate::quantum::LinearWork::of(native.adaptable_mut(&parts).unwrap());
        // Header outputs declare logical operands, never a final backing/stride identity.
        // Project BOTH immutable witnesses; do not clear or fabricate the native facts' strides.
        let logical = |residuals: &[mlx_gen::adapters::facts::AdapterFacts]| {
            residuals
                .iter()
                .map(|r| {
                    use mlx_gen::adapters::facts::AdapterFacts;
                    let kind = match r {
                        AdapterFacts::Lora { .. } => "lora",
                        AdapterFacts::Materialized { .. } => "materialized",
                        AdapterFacts::Structured { .. } => "structured",
                    };
                    (
                        kind,
                        r.tensors()
                            .into_iter()
                            .map(|t| (t.shape.clone(), t.dtype))
                            .collect::<Vec<_>>(),
                        r.disabled(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let native_logical = logical(&native_effect.residuals);
        for (slot, residual) in native_effect.residuals.iter().enumerate() {
            if let mlx_gen::adapters::facts::AdapterFacts::Structured { w1, w2, .. } = residual {
                eprintln!("SDXL_TP_NATIVE_STRUCTURED_LAYOUT slot={slot} w1_shape={:?} w1_dtype={:?} w1_strides={:?} w2_shape={:?} w2_dtype={:?} w2_strides={:?}; observednativeONLYnotheaderlayoutcertificate",
                    w1.shape, w1.dtype, w1.strides, w2.shape, w2.dtype, w2.strides);
            }
        }
        let mut observed = 0usize;
        Priced::build_configured(inputs, spec).unwrap().price_configured_observing(spec, &inputs.unet,
            inputs.quant_bits, &regime(Route::Text), |_, selected, ledger, _| {
                observed += 1;
                let effect = &selected.unet_effects.as_ref().unwrap().0[TP_PROOF_Q];
                assert_eq!(logical(&effect.residuals), native_logical,
                    "selected consumer must keep every actual ordered same-target residual, not dedup by target");
                assert_eq!(logical(&ledger.residuals), native_logical,
                    "ledger and selected effects must preserve the SAME native TP ancestry");
                assert_eq!(effect.output_shape(probe.shape()).unwrap(), actual.shape());
                assert_eq!(effect.output_dtype(probe.dtype()), actual.dtype());
                Ok(())
            }).unwrap();
        assert_eq!(observed, worlds.len());
    }
    worlds
}

#[test]
fn tp_raw_groups_revisit_target_and_preserve_real_prefixes() {
    let Some(mut completed) =
        completed_cpu_child("pricing::tp_raw_groups_revisit_target_and_preserve_real_prefixes")
    else {
        return;
    };
    {
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        for packed in [true, false] {
            let snap = Snapshot::packed(4);
            let inputs = tp_proof_inputs(&snap, packed);
            let prior = tp_proof_prior(&snap);
            let stem = TP_PROOF_Q.replace('.', "_");
            for mode in ["complete", "later-constructor", "later-file-parse"] {
                let mut tensors = HashMap::new();
                for (prefix, two) in [("lora_unet", 0.125), ("lycoris", 0.25)] {
                    let raw = format!("{prefix}_{stem}");
                    tensors.insert(
                        format!("{raw}.lokr_w1"),
                        tp_proof_tensor(&[8, 8], 0.25, Dtype::Bfloat16),
                    );
                    tensors.insert(
                        format!("{raw}.lokr_w2"),
                        tp_proof_tensor(&[8, 8], two, Dtype::Float16),
                    );
                    // With NO rank source both-full forces scale1, even for explicit alpha0.
                    tensors.insert(
                        format!("{raw}.alpha"),
                        tp_proof_tensor(&[1], 0.0, Dtype::Float16),
                    );
                }
                if mode == "later-constructor" {
                    let raw = format!("z_lora_unet_{stem}");
                    tensors.insert(
                        format!("{raw}.lokr_w1_a"),
                        tp_proof_tensor(&[8, 2], 0.25, Dtype::Float16),
                    );
                    tensors.insert(
                        format!("{raw}.lokr_w1_b"),
                        tp_proof_tensor(&[3, 8], 0.5, Dtype::Bfloat16),
                    );
                    tensors.insert(
                        format!("{raw}.lokr_w2"),
                        tp_proof_tensor(&[8, 8], 0.125, Dtype::Float16),
                    );
                }
                let file = snap.file(&format!("tp-ordered-{mode}.safetensors"));
                save(&Weights::from_map(tensors), &file);
                let mut adapters = vec![
                    prior.clone(),
                    AdapterSpec::new(file, 1.0, AdapterKind::Lokr),
                ];
                if mode == "later-file-parse" {
                    let raw = format!("lora_unet_{stem}");
                    let file = snap.file("tp-later-empty-alpha.safetensors");
                    tp_proof_save_alpha_fixture(
                        &file,
                        HashMap::from([
                            (
                                format!("{raw}.lokr_w1"),
                                tp_proof_tensor(&[8, 8], 0.5, Dtype::Float16),
                            ),
                            (
                                format!("{raw}.lokr_w2"),
                                tp_proof_tensor(&[8, 8], 0.5, Dtype::Float16),
                            ),
                            (
                                "unresolved_orphan.alpha".into(),
                                tp_proof_tensor(&[0], 1.0, Dtype::Bfloat16),
                            ),
                        ]),
                        &HashMap::new(),
                    );
                    adapters.push(AdapterSpec::new(file, 1.0, AdapterKind::Lokr));
                }
                let error = match mode {
                    "later-constructor" => Some("matmul"),
                    "later-file-parse" => Some("empty"),
                    _ => None,
                };
                let worlds = tp_proof_check(
                    &snap,
                    &inputs,
                    &tp_proof_spec(&snap, adapters),
                    &[2.0, 2.0, 4.0],
                    error,
                    3,
                );
                assert_eq!(worlds.len(), 1);
            }
        }
        eprintln!(
            "SDXL_TP_MECHANISM 0 actual_raw_group_same_target_order_and_native_refusal_prefixes"
        );
    }
    completed.body_completed = true;
}

/// Planning estimate only. This is the original fixed Snapshot source inventory, not native RSS,
/// allocation independence, or a safe physical limit. All N-sized production follows the checks.
fn tp_proof_setup_estimate(inputs: &Inputs, snap: &Snapshot) -> u64 {
    use mlx_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, Dtype as H};
    let (mut floats, mut codes, mut tensors) = (0u64, 0u64, 0u64);
    for file in [
        inputs.te1.clone(),
        inputs.te2.clone(),
        inputs.unet.clone(),
        inputs.vae.0.clone(),
        snap.file("controlnet/diffusion_pytorch_model.safetensors"),
        snap.root
            .join("ip")
            .join(crate::loader::IP_ADAPTER_IMAGE_ENCODER_FILE),
        snap.root
            .join("ip")
            .join(crate::loader::IP_ADAPTER_WEIGHT_FILES[1]),
    ] {
        for h in safetensors_path_tensor_headers(&file).expect("TP_RESOURCE_PLAN: setup header") {
            let total = match h.dtype {
                H::U32 => &mut codes,
                H::F16 | H::BF16 | H::F32 => &mut floats,
                _ => panic!("TP_RESOURCE_PLAN: unexpected setup dtype"),
            };
            *total = total
                .checked_add(h.data_bytes)
                .expect("TP_RESOURCE_PLAN: setup byte overflow");
            tensors = tensors
                .checked_add(1)
                .expect("TP_RESOURCE_PLAN: tensor count overflow");
        }
    }
    floats
        .checked_mul(8)
        .and_then(|f| codes.checked_mul(32).and_then(|c| f.checked_add(c)))
        .and_then(|v| {
            tensors
                .checked_mul(PAGE_BYTES)
                .and_then(|t| t.checked_mul(4))
                .and_then(|t| v.checked_add(t))
        })
        .and_then(|v| v.checked_add(8 * 1024 * 1024))
        .expect("TP_RESOURCE_PLAN: setup estimate overflow")
}

/// One accepted TP factor graph or LoRA pair; the alpha role is supplied separately so each
/// entrypoint's actual scalar policy, not a generic <=8B fiction, determines the native outcome.
fn tp_proof_alpha_file(
    snap: &Snapshot,
    name: &str,
    family: &str,
    alpha: Array,
) -> (PathBuf, String) {
    let raw = format!("lora_unet_{}", TP_PROOF_Q.replace('.', "_"));
    let mut data = HashMap::new();
    if family == "loha" {
        for (role, shape, value) in [
            ("hada_w1_a", &[64, 1][..], 0.25),
            ("hada_w1_b", &[1, 64][..], 0.5),
            ("hada_w2_a", &[64, 1][..], 0.25),
            ("hada_w2_b", &[1, 64][..], 0.25),
        ] {
            data.insert(
                format!("{raw}.{role}"),
                tp_proof_tensor(shape, value, Dtype::Float16),
            );
        }
    } else if family == "lora" {
        data.insert(
            format!("{raw}.lora_down.weight"),
            tp_proof_tensor(&[1, 64], 0.125, Dtype::Float16),
        );
        data.insert(
            format!("{raw}.lora_up.weight"),
            tp_proof_tensor(&[64, 1], 0.25, Dtype::Float16),
        );
    } else {
        data.insert(
            format!("{raw}.lokr_w1"),
            tp_proof_tensor(&[8, 8], 0.25, Dtype::Bfloat16),
        );
        data.insert(
            format!("{raw}.lokr_w2"),
            tp_proof_tensor(&[8, 8], 0.125, Dtype::Float16),
        );
    }
    let key = if family == "lora" {
        "base_model.model.unet.unresolved_orphan.alpha"
    } else {
        "unresolved_orphan.alpha"
    }
    .to_owned();
    data.insert(key.clone(), alpha);
    let file = snap.file(name);
    tp_proof_save_alpha_fixture(&file, data, &HashMap::new());
    (file, key)
}

#[test]
fn all_alpha_shapes_full_casts_and_entrypoint_policies() {
    let Some(mut completed) =
        completed_cpu_child("pricing::all_alpha_shapes_full_casts_and_entrypoint_policies")
    else {
        return;
    };
    {
        use crate::adapters::AdapterWork;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let snap = Snapshot::packed(4);
        let inputs = tp_proof_inputs(&snap, true);
        let prior = tp_proof_prior(&snap);
        let kinds = [
            ("lokr", Dtype::Float16, AdapterKind::Lokr),
            ("loha", Dtype::Bfloat16, AdapterKind::Lora),
            ("lokr", Dtype::Uint8, AdapterKind::Lokr),
            ("lora", Dtype::Float16, AdapterKind::Lora),
        ];
        // E is the ENTIRE small configured source budget across these exact graph branches.
        // No native allocator/RSS measurement and no subtraction of shared/raw upper terms.
        let mut entire_legacy = 0u64;
        for (i, (family, dtype, kind)) in kinds.iter().enumerate() {
            let (file, _) = tp_proof_alpha_file(
                &snap,
                &format!("alpha-small-{i}.safetensors"),
                family,
                tp_proof_tensor(&[1], 2.0, *dtype),
            );
            let spec = tp_proof_spec(
                &snap,
                vec![prior.clone(), AdapterSpec::new(file, 1.0, *kind)],
            );
            let price = Priced::build_configured(&inputs, &spec)
                .unwrap()
                .price_configured(&spec, &inputs.unet, inputs.quant_bits, &regime(Route::Text))
                .unwrap();
            for window in price.windows {
                entire_legacy = entire_legacy.max(window.logical_peak_bytes());
            }
        }
        let n = entire_legacy
            .checked_add(PAGE_BYTES.checked_mul(2).unwrap())
            .and_then(|v| v.checked_add(1))
            .expect("TP_RESOURCE_PLAN: N overflow");
        let native_n = i32::try_from(n).expect("TP_RESOURCE_PLAN: native N overflow");
        let host_n = usize::try_from(n).expect("TP_RESOURCE_PLAN: host N overflow");
        host_n
            .checked_mul(4)
            .expect("TP_RESOURCE_PLAN: host F32 byte overflow");
        let setup = tp_proof_setup_estimate(&inputs, &snap);
        // Explicit per-row operation trace, with no release/donation credit within the row:
        // host generation4N + native F32 producer4N + source cast<=2N + save staging<=2N
        // + actual file raw<=2N + parser FULL F32 cast4N + independent file raw<=2N
        // + independent FULL F32 cast4N =24N. Large rows' owners end before the next row.
        let alpha_work = n
            .checked_mul(24)
            .and_then(|v| v.checked_add(16 * 1024 * 1024))
            .expect("TP_RESOURCE_PLAN: alpha operation-work overflow");
        let combined = setup
            .checked_add(alpha_work)
            .expect("TP_RESOURCE_PLAN: combined overflow");
        let conversions = n
            .checked_mul(12)
            .expect("TP_RESOURCE_PLAN: conversion work overflow");
        assert!(
            setup <= 512 * 1024 * 1024,
            "TP_RESOURCE_PLAN: setup planning estimate exceeds512MiB"
        );
        assert!(combined <= 1024 * 1024 * 1024,
            "TP_RESOURCE_PLAN: setup+alpha planning estimate exceeds1GiB; no N-sized allocation permitted");
        assert!(
            conversions <= 1 << 30,
            "TP_RESOURCE_PLAN: conversion work exceeds2^30"
        );
        eprintln!("SDXL_TP_ALPHA_RESOURCE_PLAN E={entire_legacy} N={n} setup={setup} alpha_work={alpha_work} combined={combined} conversions={conversions}; estimates NOT physical/RSS upper bounds");
        for (i, (family, dtype, kind)) in kinds.iter().enumerate() {
            // The checked plan above precedes EVERY N-sized Vec, tensor and file write.
            let mut values = vec![2.0f32; host_n];
            values[host_n - 1] = 3.0;
            let alpha = Array::from_slice(&values, &[native_n])
                .as_dtype(*dtype)
                .unwrap();
            alpha.eval().unwrap();
            let (file, key) = tp_proof_alpha_file(
                &snap,
                &format!("alpha-large-{i}.safetensors"),
                family,
                alpha,
            );
            let loaded = Weights::from_file(&file).unwrap();
            let raw = loaded.require(&key).unwrap();
            assert_eq!(raw.shape(), &[native_n]);
            assert_eq!(raw.dtype(), *dtype);
            let converted = raw.as_dtype(Dtype::Float32).unwrap();
            let native_slice = converted.try_as_slice::<f32>().unwrap();
            assert_eq!(native_slice.len(), host_n);
            assert_eq!(native_slice[0], 2.0);
            assert_eq!(
                native_slice[host_n - 1],
                3.0,
                "native FULL conversion/read reaches the last operand"
            );
            let native_bytes = u64::try_from(converted.size())
                .unwrap()
                .checked_mul(4)
                .unwrap();
            let native_raw = loaded
                .keys()
                .try_fold(0u64, |sum, key| {
                    let a = loaded.require(key).unwrap();
                    let width = match a.dtype() {
                        Dtype::Float16 | Dtype::Bfloat16 => 2,
                        Dtype::Float32 => 4,
                        Dtype::Uint8 => 1,
                        _ => panic!("unexpected native alpha-file dtype"),
                    };
                    sum.checked_add(u64::try_from(a.size()).unwrap().checked_mul(width).unwrap())
                })
                .unwrap();
            let headers =
                mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&file).unwrap();
            // The ONLY N-dependent legacy term when BOTH file-load and Parse alpha-cost
            // channels coherently regress to <=8B is the raw root: <=2N, rounded by one page.
            // A SINGLE-site mutant is masked by the other full-cast channel: do NOT claim this
            // bound for that mutant. Independent contribution assertions below cover each site.
            // E includes ALL small graph,
            // source parents, install/forward effects, packs and small cast work, without credit
            // for the replaced small raw root. This positive sum bounds EVERY old whole window.
            let raw_upper = n
                .checked_mul(2)
                .and_then(|v| v.checked_add(PAGE_BYTES - 1))
                .map(|v| v / PAGE_BYTES * PAGE_BYTES)
                .unwrap();
            let entire_old_upper = entire_legacy
                .checked_add(raw_upper)
                .and_then(|v| v.checked_add(PAGE_BYTES))
                .unwrap();
            assert!(native_bytes > entire_old_upper.checked_add(PAGE_BYTES).unwrap(),
                "independent native FULL alpha operand must dominate ENTIRE legacy configured budget plus slack");
            let spec = tp_proof_spec(
                &snap,
                vec![prior.clone(), AdapterSpec::new(file, 1.0, *kind)],
            );
            let is_lora = *family == "lora";
            let increments: &[f32] = if is_lora {
                &[2.0]
            } else if *family == "loha" {
                &[2.0, 0.5]
            } else {
                &[2.0, 2.0]
            };
            let worlds = tp_proof_check(
                &snap,
                &inputs,
                &spec,
                increments,
                is_lora.then_some("reshape"),
                if is_lora { 1 } else { 2 },
            );
            if !is_lora {
                let priced = Priced::build_configured(&inputs, &spec)
                    .unwrap()
                    .price_configured(&spec, &inputs.unet, inputs.quant_bits, &regime(Route::Text))
                    .unwrap();
                let whole = priced
                    .windows
                    .iter()
                    .map(|w| w.logical_peak_bytes())
                    .max()
                    .unwrap();
                assert!(whole >= native_bytes,
                    "whole configured source budget cannot fall below independent native FULL alpha operand");
            }
            // These are COMPONENT assertions, not whole-window underpricing proofs when the
            // other full-cast channel remains intact. Whole-price assertion is deliberately first.
            assert!(crate::adapters::adapter_file_load_bytes(&headers).unwrap()
                >= native_raw.checked_add(native_bytes).unwrap(),
                "file-load channel must include native FULL alpha cast beside its actual raw operands");
            for world in worlds {
                let reads: Vec<_> = world
                    .iter()
                    .filter_map(|w| match w {
                        AdapterWork::AlphaRead { key: k, scratch } if k == &key => Some(*scratch),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    reads.len(),
                    1,
                    "EVERY recognized unresolved/orphan alpha is actually read"
                );
                assert!(reads[0] >= native_bytes,
                    "header FULL alpha cast cannot truncate native N-element operand to first4/8bytes");
            }
        }
        // Native F32 source alpha needs no cast; TP still accepts its non-scalar shape/read.
        let (file, key) = tp_proof_alpha_file(
            &snap,
            "alpha-f32.safetensors",
            "lokr",
            tp_proof_tensor(&[3], 2.0, Dtype::Float32),
        );
        let spec = tp_proof_spec(
            &snap,
            vec![
                prior.clone(),
                AdapterSpec::new(file, 1.0, AdapterKind::Lokr),
            ],
        );
        for world in tp_proof_check(&snap, &inputs, &spec, &[2.0, 2.0], None, 2) {
            assert!(world
                .iter()
                .any(|w| matches!(w, AdapterWork::AlphaRead { key: k, scratch: 0 } if k == &key)));
        }
        // Both scalar aliases are read even though ONE role assignment overwrites the other.
        let flat = format!("lora_unet_{}", TP_PROOF_Q.replace('.', "_"));
        let alpha_keys = [
            format!("{flat}.alpha"),
            format!("base_model.model.unet.{TP_PROOF_Q}.alpha"),
        ];
        for invalid in [None, Some(0usize), Some(1usize)] {
            let mut data = HashMap::from([
                (
                    format!("{flat}.lora_down.weight"),
                    tp_proof_tensor(&[1, 64], 0.125, Dtype::Float16),
                ),
                (
                    format!("{flat}.lora_up.weight"),
                    tp_proof_tensor(&[64, 1], 0.25, Dtype::Float16),
                ),
            ]);
            for (i, key) in alpha_keys.iter().enumerate() {
                let extent = [if invalid == Some(i) { 2 } else { 1 }];
                let shape: &[i32] = if invalid.is_none() && i == 1 {
                    &[]
                } else {
                    &extent
                };
                data.insert(
                    key.clone(),
                    tp_proof_tensor(
                        shape,
                        1.0,
                        if i == 0 {
                            Dtype::Float16
                        } else {
                            Dtype::Bfloat16
                        },
                    ),
                );
            }
            let file = snap.file(&format!("alpha-alias-{invalid:?}.safetensors"));
            save(&Weights::from_map(data), &file);
            let spec = tp_proof_spec(
                &snap,
                vec![
                    prior.clone(),
                    AdapterSpec::new(file, 1.0, AdapterKind::Lora),
                ],
            );
            let increments: &[f32] = if invalid.is_none() {
                &[2.0, 2.0]
            } else {
                &[2.0]
            };
            let worlds = tp_proof_check(
                &snap,
                &inputs,
                &spec,
                increments,
                invalid.map(|_| "reshape"),
                if invalid.is_none() { 2 } else { 1 },
            );
            for world in worlds {
                for key in &alpha_keys {
                    assert!(world.iter().any(|w| matches!(w, AdapterWork::AlphaRead { key: k, scratch } if k == key && *scratch >= 4)),
                        "recognized overwritten scalar alias is not permission to omit its FULL cast");
                }
            }
        }
        // TP empty alpha rejects before any cast, even on an unresolved group; this is a
        // native parse error, not permission to omit a reached file or clear earlier installs.
        for family in ["lokr", "loha"] {
            let (file, key) = tp_proof_alpha_file(
                &snap,
                &format!("alpha-empty-{family}.safetensors"),
                family,
                tp_proof_tensor(&[0], 1.0, Dtype::Bfloat16),
            );
            let kind = if family == "loha" {
                AdapterKind::Lora
            } else {
                AdapterKind::Lokr
            };
            let spec = tp_proof_spec(
                &snap,
                vec![prior.clone(), AdapterSpec::new(file, 1.0, kind)],
            );
            for world in tp_proof_check(&snap, &inputs, &spec, &[2.0], Some("empty"), 1) {
                assert!(
                    !world
                        .iter()
                        .any(|w| matches!(w, AdapterWork::AlphaRead { key: k, .. } if k == &key)),
                    "TP empty-alpha predicate precedes its native F32 cast"
                );
            }
        }
        // PEFT LoKr reads metadata alpha, NOT arbitrary .alpha tensor payloads. An empty tensor
        // here must remain ignored, unlike the actual TP empty-alpha parser error above.
        let file = snap.file("peft-ignored-alpha.safetensors");
        let data = vec![
            (
                format!("{TP_PROOF_Q}.lokr_w1"),
                tp_proof_tensor(&[8, 8], 0.25, Dtype::Float16),
            ),
            (
                format!("{TP_PROOF_Q}.lokr_w2"),
                tp_proof_tensor(&[8, 8], 0.125, Dtype::Bfloat16),
            ),
            (
                format!("{TP_PROOF_Q}.alpha"),
                tp_proof_tensor(&[0], 1.0, Dtype::Float16),
            ),
        ];
        let metadata: HashMap<String, String> =
            [("networkType", "lokr"), ("rank", "1"), ("alpha", "1")]
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect();
        tp_proof_save_alpha_fixture(&file, data, &metadata);
        let spec = tp_proof_spec(
            &snap,
            vec![prior, AdapterSpec::new(file, 1.0, AdapterKind::Lokr)],
        );
        for world in tp_proof_check(&snap, &inputs, &spec, &[2.0, 2.0], None, 2) {
            assert!(
                !world
                    .iter()
                    .any(|w| matches!(w, AdapterWork::AlphaRead { .. })),
                "PEFT metadata alpha is not TP tensor alpha"
            );
        }
        eprintln!(
            "SDXL_TP_MECHANISM 1 all_recognized_alpha_full_native_extent_and_entrypoint_policies"
        );
    }
    completed.body_completed = true;
}

#[test]
fn tp_full_product_tucker_and_batched_native_branches() {
    let Some(mut completed) =
        completed_cpu_child("pricing::tp_full_product_tucker_and_batched_native_branches")
    else {
        return;
    };
    {
        use crate::adapters::AdapterWork;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        for packed in [true, false] {
            let snap = Snapshot::packed(4);
            let inputs = tp_proof_inputs(&snap, packed);
            let prior = tp_proof_prior(&snap);
            let raw = format!("lycoris_{}", TP_PROOF_Q.replace('.', "_"));
            for name in [
                "both-full",
                "full-shadow-product",
                "full-keeps-rank-gate",
                "product-full",
                "full-product",
                "product-product",
                "batched-product",
                "lokr-tucker",
                "full-shadow-tucker",
                "loha-product",
                "loha-both-tucker",
                "loha-one-tucker-ignored",
                "loha-singleton-contract",
                "loha-second-product-error",
            ] {
                let mut data = HashMap::new();
                let mut put = |role: &str, shape: &[i32], value: f32, dtype| {
                    data.insert(
                        format!("{raw}.{role}"),
                        tp_proof_tensor(shape, value, dtype),
                    );
                };
                let (alpha, delta, error) = if name.starts_with("loha") {
                    if name == "loha-both-tucker" || name == "loha-singleton-contract" {
                        let singleton = name == "loha-singleton-contract";
                        let rank = if singleton { 2 } else { 1 };
                        for role in ["hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"] {
                            put(role, &[rank, 64], 0.5, Dtype::Float16);
                        }
                        for role in ["hada_t1", "hada_t2"] {
                            put(role, &[1, rank, 1, 1], 0.5, Dtype::Bfloat16);
                        }
                        (rank as f32, if singleton { 16.0 } else { 1.0 }, None)
                    } else {
                        put("hada_w1_a", &[64, 1], 0.25, Dtype::Float16);
                        put("hada_w1_b", &[1, 64], 0.5, Dtype::Bfloat16);
                        if name == "loha-second-product-error" {
                            put("hada_w2_a", &[64, 2], 0.25, Dtype::Float16);
                            put("hada_w2_b", &[3, 64], 0.25, Dtype::Float16);
                        } else {
                            put("hada_w2_a", &[64, 1], 0.25, Dtype::Float16);
                            put("hada_w2_b", &[1, 64], 0.25, Dtype::Float16);
                        }
                        if name == "loha-one-tucker-ignored" {
                            put("hada_t1", &[2], 0.5, Dtype::Float16);
                        }
                        (
                            1.0,
                            0.5,
                            (name == "loha-second-product-error").then_some("matmul"),
                        )
                    }
                } else {
                    let product1 =
                        matches!(name, "product-full" | "product-product" | "batched-product");
                    let product2 = matches!(name, "full-product" | "product-product");
                    if product1 {
                        let a: &[i32] = if name == "batched-product" {
                            &[1, 8, 2]
                        } else {
                            &[8, 2]
                        };
                        let b: &[i32] = if name == "batched-product" {
                            &[1, 2, 8]
                        } else {
                            &[2, 8]
                        };
                        put("lokr_w1_a", a, 0.25, Dtype::Float16);
                        put("lokr_w1_b", b, 0.5, Dtype::Bfloat16);
                    } else {
                        put("lokr_w1", &[8, 8], 0.25, Dtype::Bfloat16);
                    }
                    if product2 {
                        put("lokr_w2_a", &[8, 2], 0.25, Dtype::Float16);
                        put("lokr_w2_b", &[2, 8], 0.25, Dtype::Float16);
                    } else if name == "lokr-tucker" {
                        put("lokr_t2", &[1, 1, 1, 1], 0.5, Dtype::Bfloat16);
                        put("lokr_w2_a", &[1, 8], 0.5, Dtype::Float16);
                        put("lokr_w2_b", &[1, 8], 0.5, Dtype::Float16);
                    } else {
                        put("lokr_w2", &[8, 8], 0.125, Dtype::Float16);
                    }
                    if matches!(name, "full-shadow-product" | "full-keeps-rank-gate") {
                        put(
                            "lokr_w1_a",
                            if name == "full-keeps-rank-gate" {
                                &[2]
                            } else {
                                &[2, 3]
                            },
                            0.25,
                            Dtype::Float16,
                        );
                        put("lokr_w1_b", &[5, 2], 0.5, Dtype::Float16);
                    }
                    if name == "full-shadow-tucker" {
                        put("lokr_t2", &[1, 1], 0.5, Dtype::Float16);
                    }
                    let alpha = match name {
                        "both-full" => 0.0,
                        "full-shadow-product" => 3.0,
                        "batched-product" => 8.0,
                        "product-full" | "full-product" | "product-product" => 2.0,
                        _ => 1.0,
                    };
                    (
                        alpha,
                        2.0,
                        (name == "full-keeps-rank-gate").then_some("2-d"),
                    )
                };
                put("alpha", &[1], alpha, Dtype::Float16);
                let file = snap.file(&format!("tp-branch-{name}.safetensors"));
                save(&Weights::from_map(data), &file);
                let kind = if name.starts_with("loha") {
                    AdapterKind::Lora
                } else {
                    AdapterKind::Lokr
                };
                let spec = tp_proof_spec(
                    &snap,
                    vec![prior.clone(), AdapterSpec::new(file, 1.0, kind)],
                );
                let ordered = [2.0, delta];
                let increments = if error.is_some() {
                    &ordered[..1]
                } else {
                    &ordered[..]
                };
                let worlds = tp_proof_check(
                    &snap,
                    &inputs,
                    &spec,
                    increments,
                    error,
                    if error.is_some() { 1 } else { 2 },
                );
                assert_eq!(worlds.len(), 1);
                if name == "loha-second-product-error" {
                    // Observe the completed FIRST real product independently; a later native
                    // second-product error does not mean the first logical output never existed.
                    let a = tp_proof_tensor(&[64, 1], 0.25, Dtype::Float16);
                    let b = tp_proof_tensor(&[1, 64], 0.5, Dtype::Bfloat16);
                    let first = mlx_rs::ops::matmul(&a, &b).unwrap();
                    assert_eq!(first.shape(), &[64, 64]);
                    assert!(tp_proof_read(&first).iter().all(|x| *x == 0.125));
                    assert!(worlds[0].iter().any(|w| matches!(w, AdapterWork::Preparation(outputs)
                        if outputs.iter().any(|o| o.shape.as_slice() == first.shape() && o.dtype == first.dtype()))),
                        "header must preserve the first completed native factor before second-product refusal");
                }
            }
        }
        eprintln!(
            "SDXL_TP_MECHANISM 2 actual_full_product_tucker_batched_and_present_rank_gate_branches"
        );
    }
    completed.body_completed = true;
}

/// Native configured file → dense broadcast merge → ACTUAL post-adapter pack, not a fabricated
/// mixed-dtype activation. The old native shape model stays unmodified while selected effects grow.
#[test]
fn selected_projection_operands_use_actual_context_after_native_pack() {
    let Some(mut completed) = completed_cpu_child(
        "pricing::selected_projection_operands_use_actual_context_after_native_pack",
    ) else {
        return;
    };
    {
        use mlx_gen::adapters::AdaptableHost;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec, Quant};
        let target = "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k";
        let value_target = "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v";
        let parts: Vec<_> = target.split('.').collect();
        let value_parts: Vec<_> = value_target.split('.').collect();
        let tensor = |shape: &[i32], value: f32, dtype: Dtype| {
            let count = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).unwrap()))
                .unwrap();
            let a = Array::from_slice(&vec![value; count], shape)
                .as_dtype(dtype)
                .unwrap();
            a.eval().unwrap();
            a
        };
        for route in [Route::Text, Route::IpAdapter] {
            let snap = Snapshot::packed(4);
            let inputs = snap.inputs(route);
            assert_eq!(snap.context, 64);
            let checkpoint = Weights::from_file(&inputs.unet).unwrap();
            checkpoint.materialize().unwrap();
            let mut checkpoint = checkpoint.into_tensors();
            checkpoint.insert(
                format!("{target}.weight"),
                tensor(&[64, 1], 0.0, Dtype::Float16),
            );
            for suffix in ["scales", "biases", "bias"] {
                checkpoint.remove(&format!("{target}.{suffix}"));
            }
            save(&Weights::from_map(checkpoint), &inputs.unet);
            let stem = format!("lora_unet_{}", target.replace('.', "_"));
            let file = snap.file("p5-context-broadcast.safetensors");
            save(
                &Weights::from_map(HashMap::from([
                    (
                        format!("{stem}.lora_down.weight"),
                        tensor(&[1, 64], 0.125, Dtype::Float16),
                    ),
                    (
                        format!("{stem}.lora_up.weight"),
                        tensor(&[64, 1], 0.25, Dtype::Float16),
                    ),
                ])),
                &file,
            );
            let mut spec =
                mlx_gen::gen_core::LoadSpec::new(mlx_gen::WeightsSource::Dir(snap.root.clone()));
            spec.quantize = Some(Quant::Q4);
            spec.adapters = vec![AdapterSpec::new(file, 1.0, AdapterKind::Lora)];
            if route == Route::IpAdapter {
                spec.ip_adapter = Some(mlx_gen::WeightsSource::Dir(snap.root.join("ip")));
            }
            let mut native = crate::loader::load_unet_with_config_streamed(
                &snap.root,
                crate::model::DTYPE,
                &snap.unet,
                false,
            )
            .unwrap();
            assert_eq!(native.adaptable_mut(&parts).unwrap().base_shape(), [64, 1]);
            assert_eq!(
                native.adaptable_mut(&value_parts).unwrap().base_shape(),
                [64, 64],
                "unchanged native V must not borrow the old K's one-column source shape"
            );
            let (applied, _) = observe(PLENTY, || {
                crate::adapters::apply_sdxl_adapters_with(
                    &mut native,
                    &spec.adapters,
                    crate::adapters::LoraCoverage::Complete,
                )
            });
            assert_eq!(applied.unwrap().merged, 1);
            let weight = native
                .adaptable_mut(&parts)
                .unwrap()
                .dense_weight()
                .unwrap()
                .0;
            assert_eq!(weight.shape(), &[64, 64]);
            assert_eq!(weight.dtype(), Dtype::Float16);
            assert!(weight
                .as_dtype(Dtype::Float32)
                .unwrap()
                .try_as_slice::<f32>()
                .unwrap()
                .iter()
                .all(|v| *v == 0.03125));
            // Production model::load installs IP pairs AFTER adapters and BEFORE native quantize.
            let ip_tokens = inputs.ip.as_ref().map(|(_, file)| {
                let weights = crate::loader::lazy_weights(
                    std::slice::from_ref(file),
                    crate::model::DTYPE,
                    false,
                )
                .unwrap();
                native
                    .install_ip_adapter(crate::ip_adapter::load_ip_kv_pairs(&weights).unwrap())
                    .unwrap();
                let resampler = crate::ip_adapter::Resampler::from_weights(
                    &weights,
                    "image_proj",
                    &snap.resampler,
                )
                .unwrap();
                let features = tensor(
                    &[1, 2, snap.resampler.embed_dim],
                    0.125,
                    crate::model::DTYPE,
                );
                let positive = resampler.forward(&features).unwrap();
                crate::ip_adapter::cfg_pair(&positive).unwrap()
            });
            let (packed, _) = observe(PLENTY, || native.quantize(4));
            packed.unwrap();
            let context = tensor(&[2, TOKENS, snap.context], 1.0, crate::model::DTYPE);
            assert_eq!(
                context.dtype(),
                Dtype::Float16,
                "REAL model activation dtype, not a fabricated BF16 input"
            );
            for path in [&parts, &value_parts] {
                let linear = native.adaptable_mut(path).unwrap();
                let (codes, scales, _, _, group, bits) = linear.quantized_params().unwrap();
                assert_eq!(linear.base_shape(), [64, 64]);
                assert_eq!(codes.shape(), &[64, 8]);
                assert_eq!(scales.shape(), &[64, 1]);
                assert_eq!(scales.dtype(), Dtype::Bfloat16);
                assert_eq!((group, bits), (64, 4));
                let output = linear.forward(&context).unwrap();
                output.eval().unwrap();
                assert_eq!(output.shape(), &[2, TOKENS, 64]);
                assert_eq!(
                    output.dtype(),
                    Dtype::Float32,
                    "native q4 BF16 scales promote the REAL F16 context to F32"
                );
                let values = output.try_as_slice::<f32>().unwrap();
                if path == &parts {
                    assert!(
                        values.iter().all(|v| *v == 2.0),
                        "accepted file and native pack preserve nonzero K contribution"
                    );
                } else {
                    assert!(values.iter().any(|v| *v != 0.0));
                }
                let cast = context.as_dtype(output.dtype()).unwrap();
                assert_eq!(cast.shape(), context.shape());
                assert_eq!(
                    cast.try_as_slice::<f32>().unwrap().len(),
                    2 * TOKENS as usize * 64
                );
                // Logical operand extent only; this proves no fresh allocation/donation/RSS bound.
            }
            if let Some(tokens) = &ip_tokens {
                // IP slots are deliberately NOT LoRA adaptable_mut targets. Borrow them from
                // the SAME actual module walk AFTER the real native pack; change no namespace.
                let mut seen = 0usize;
                native
                    .quantize_using(4, &mut |path, linear, _| {
                        if path == "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k_ip"
                            || path
                                == "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v_ip"
                        {
                            seen += 1;
                            let output = linear.forward(tokens)?;
                            output.eval()?;
                            assert_eq!(tokens.shape()[2], 64);
                            assert_eq!(output.shape()[..2], tokens.shape()[..2]);
                            assert!(output
                                .as_dtype(Dtype::Float32)
                                .unwrap()
                                .try_as_slice::<f32>()
                                .unwrap()
                                .iter()
                                .all(|v| v.is_finite()));
                        }
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(seen, 2);
            }
            let mut priced = Priced::build_configured(&inputs, &spec).unwrap();
            let (price, operands) =
                crate::unet::transformer::with_projection_operands_for_test(|| {
                    priced.price_configured(&spec, &inputs.unet, inputs.quant_bits, &regime(route))
                });
            price.unwrap();
            for role in ["attn2.to_k", "attn2.to_v"] {
                let matching: Vec<_> = operands.iter().filter(|o| o.0 == role).collect();
                assert!(
                    !matching.is_empty(),
                    "actual production pricing callers must reach each K/V operand"
                );
                for (_, shape, dtype, _) in matching {
                    assert_eq!(shape, &context.shape().to_vec(),
                        "selected projection operand input MUST match actual context, not borrowed original K width");
                    assert_eq!(*dtype, context.dtype());
                }
            }
            for role in ["attn2.to_k_ip", "attn2.to_v_ip"] {
                let matching: Vec<_> = operands.iter().filter(|o| o.0 == role).collect();
                if let Some(tokens) = &ip_tokens {
                    assert!(
                        !matching.is_empty(),
                        "actual IP pricing caller must reach BOTH token projections"
                    );
                    for (_, shape, dtype, _) in matching {
                        assert_eq!(shape, &tokens.shape().to_vec());
                        assert_eq!(
                            *dtype,
                            tokens.dtype(),
                            "actual native Resampler/CFG dtype, not a guessed token width"
                        );
                    }
                } else {
                    assert!(matching.is_empty());
                }
            }
        }
        eprintln!("SDXL_P5_OPERAND_MECHANISM 0 real_file_postpack_context_width_and_ip_operands");
    }
    completed.body_completed = true;
}

fn rgb_observation_active() -> u64 {
    let mut bytes = 0usize;
    assert_eq!(
        unsafe { mlx_sys::mlx_get_active_memory(&mut bytes) },
        0,
        "RGB observation native active-counter status"
    );
    u64::try_from(bytes).unwrap()
}

fn rgb_observation_sync_cpu() {
    assert_eq!(
        unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
        0,
        "RGB observation checked CPU stream synchronization"
    );
}

/// Resource planning ONLY: no physical/native/RSS bound, runtime admission rule or memory grid.
/// This fixed source inventory is checked BEFORE even the tiny native fixture factory is called.
fn rgb_observation_resource_plan() {
    let channels = 32u64;
    // vae::tests::fixture_with: ten resnets, two mid attentions,28 convs,120 mapped tensors.
    // Upper each conv by32x32x3x3, including smaller in/out and1x1 projections without credit.
    let conv_elements = 28u64
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(channels))
        .and_then(|v| v.checked_mul(9))
        .and_then(|v| v.checked_add(28 * channels))
        .unwrap();
    let norms = (40u64 + 4).checked_mul(channels).unwrap();
    let attentions = 2u64
        .checked_mul(4 * channels * channels + 6 * channels)
        .unwrap();
    let source_bytes = conv_elements
        .checked_add(norms)
        .and_then(|v| v.checked_add(attentions))
        .and_then(|v| v.checked_mul(4))
        .unwrap();
    let max_output_pixels = 8u64.checked_mul(10).unwrap();
    let activation_work = max_output_pixels
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(4 * 48))
        .unwrap();
    // Generator host/native/cast/view/read stages8x plus metadata4pages/root+fixed16MiB;
    // coefficients are planning assumptions, not a backing-independence or process guarantee.
    let planned = source_bytes
        .checked_mul(8)
        .and_then(|v| v.checked_add(120 * 4 * PAGE_BYTES))
        .and_then(|v| v.checked_add(activation_work))
        .and_then(|v| v.checked_add(16 * 1024 * 1024))
        .unwrap();
    let conv_fmas = 28u64
        .checked_mul(max_output_pixels)
        .and_then(|v| v.checked_mul(channels * channels * 9))
        .unwrap();
    let attention_fmas = 2u64
        .checked_mul(2 * 20 * 20 * channels + 4 * 20 * channels * channels)
        .unwrap();
    let fmas = conv_fmas
        .checked_add(attention_fmas)
        .and_then(|v| v.checked_add(20 * 4 * 4))
        .unwrap();
    assert!(planned < 64 * 1024 * 1024,
        "SDXL_RGB_RESOURCE_PLAN: source operation-work estimate exceeds64MiB before fixture/vector/tensor production");
    assert!(
        fmas < 1 << 26,
        "SDXL_RGB_RESOURCE_PLAN: fixed-small operation FMA bound exceeds2^26"
    );
    eprintln!("SDXL_RGB_RESOURCE_PLAN source_bytes={source_bytes} planned_work={planned} fmas={fmas}; planning estimates NOT physical/RSS/nativecounter bounds");
}

/// Counter-only observer: captures no Array, native owner, image or data pointer. These are RAW
/// absolute allocator observations, not an asserted resident partition or a new-allocation lower.
fn rgb_observation_capture<T>(body: impl FnOnce() -> Result<T>) -> (T, Vec<(String, u64, u64)>) {
    let records = Rc::new(RefCell::new(Vec::new()));
    let captured = records.clone();
    let result = mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(PLENTY, || {
        mlx_gen::ios_quantum::with_cpu_diagnostic_observer(
            move |event| {
                let label = match event {
                    mlx_gen::ios_quantum::QuantumEvent::Checked { label, .. } => {
                        format!("checked:{label}")
                    }
                    mlx_gen::ios_quantum::QuantumEvent::Synchronized { label, stream } => {
                        assert_eq!(*stream, mlx_rs::Stream::cpu().get_index().unwrap());
                        format!("synchronized:{label}")
                    }
                };
                captured.borrow_mut().push((
                    label,
                    rgb_observation_active(),
                    mlx_rs::memory::get_peak_memory() as u64,
                ));
            },
            body,
        )
    })
    .unwrap();
    let records = Rc::try_unwrap(records).unwrap().into_inner();
    (result, records)
}

fn rgb_observation_assert_pixel_boundary(events: &[(String, u64, u64)]) {
    let labels: Vec<_> = events
        .iter()
        .filter(|e| e.0.ends_with(DECODED_TO_IMAGE))
        .map(|e| e.0.clone())
        .collect();
    assert_eq!(
        labels,
        [
            format!("checked:{DECODED_TO_IMAGE}"),
            format!("synchronized:{DECODED_TO_IMAGE}")
        ],
        "original RGB guard must check then synchronize its REAL output once"
    );
}

#[test]
fn rgb_realized_f32_literals_preserve_caller_and_observe_absolute_peak() {
    let Some(mut completed) = completed_cpu_child(
        "pricing::rgb_realized_f32_literals_preserve_caller_and_observe_absolute_peak",
    ) else {
        return;
    };
    {
        rgb_observation_resource_plan();
        let values = [
            -1.0f32, -1.0, -1.0, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0,
        ];
        let decoded = Array::from_slice(&values, &[1, 1, 4, 3]);
        decoded.eval().unwrap();
        rgb_observation_sync_cpu();
        // Pointer/content inspection is permitted ONLY in this already-realized F32 case.
        let pointer_before = decoded.try_as_slice::<f32>().unwrap().as_ptr() as usize;
        let active_before = rgb_observation_active();
        mlx_rs::memory::reset_peak_memory();
        let (image, events) =
            rgb_observation_capture(|| crate::pipeline::decoded_to_image(&decoded));
        let absolute_peak = mlx_rs::memory::get_peak_memory() as u64;
        let active_after = rgb_observation_active();
        let observed_increment = absolute_peak.checked_sub(active_before);
        assert_eq!((image.width, image.height), (4, 1));
        // Independent literal expected bytes, NOT a new clamp/round formula as its own oracle.
        assert_eq!(
            image.pixels,
            [0, 0, 0, 64, 64, 64, 191, 191, 191, 255, 255, 255],
            "actual NHWC RGB pixel bytes must retain the original nonzero literal mapping"
        );
        assert_eq!(
            decoded.try_as_slice::<f32>().unwrap(),
            values,
            "borrowed realized decoded caller values must remain unchanged after pixel conversion"
        );
        let pointer_after = decoded.try_as_slice::<f32>().unwrap().as_ptr() as usize;
        assert_eq!(
            pointer_after, pointer_before,
            "already-realized borrowed caller backing stays stable"
        );
        rgb_observation_assert_pixel_boundary(&events);
        eprintln!("SDXL_RGB_OBSERVATION realized_f32 before={active_before} absolute_peak={absolute_peak} after_same_held_owners={active_after} increment={observed_increment:?} pointer_before={pointer_before} pointer_after={pointer_after} events={events:?}; no physical-bound/newallocation certificate");
        drop(decoded);
        rgb_observation_sync_cpu();
        eprintln!(
            "SDXL_RGB_OBSERVATION realized_f32 after_caller_drop={}",
            rgb_observation_active()
        );
        eprintln!("SDXL_RGB_OBSERVATION_MECHANISM 0 literal_pixels_and_realized_caller_unchanged");
    }
    completed.body_completed = true;
}

#[test]
fn rgb_actual_vae_returns_preserve_lazy_boundary_and_observe_absolute_peak() {
    let Some(mut completed) = completed_cpu_child(
        "pricing::rgb_actual_vae_returns_preserve_lazy_boundary_and_observe_absolute_peak",
    ) else {
        return;
    };
    {
        use mlx_gen::LatentDecoder;
        rgb_observation_resource_plan();
        for dtype in [Dtype::Float16, Dtype::Float32] {
            let (mut weights, cfg) = crate::vae::tests::fixture();
            let raw_bytes = weights
                .keys()
                .try_fold(0u64, |sum, key| {
                    let a = weights.require(key).unwrap();
                    assert_eq!(a.dtype(), Dtype::Float32);
                    sum.checked_add(u64::try_from(a.size()).unwrap().checked_mul(4).unwrap())
                })
                .unwrap();
            assert!(raw_bytes <= 1_075_712, "SDXL_RGB_RESOURCE_PLAN: source fixture exceeded prechecked fixed raw-element bound");
            weights.cast_all(dtype).unwrap();
            weights.materialize().unwrap();
            let vae = crate::vae::Autoencoder::from_weights(&weights, &cfg).unwrap();
            drop(weights); // Match consumed load weights; do not add a retained map of native owners.
            let latents = Array::from_slice(&[0.25f32; 4 * 5 * 4], &[1, 4, 5, 4]);
            latents.eval().unwrap();
            rgb_observation_sync_cpu();
            let native = crate::vae::SdxlLatentDecoder::new(&vae);
            // EXACT production decoder seam: NHWC caller→NCHW seam→returnedNCHW→borrowedNHWC.
            let nchw = latents.transpose_axes(&[0, 3, 1, 2]).unwrap();
            let active_before_decode = rgb_observation_active();
            mlx_rs::memory::reset_peak_memory();
            let (returned, decode_events) = rgb_observation_capture(|| native.decode(&nchw));
            let decode_absolute_peak = mlx_rs::memory::get_peak_memory() as u64;
            let active_at_return = rgb_observation_active();
            assert_eq!(
                returned.shape(),
                &[1, 3, 8, 10],
                "real native VAE return relayout preserves non-square RGB axes"
            );
            assert_eq!(returned.dtype(), Dtype::Float32);
            // NO pointer/readback/eval/sync of returned: F16 VAE's F32 cast remains as returned.
            // NO extra retained F16 source clone, and no callback owns an Array.
            let (image, active_before_rgb, rgb_absolute_peak, active_after_rgb, rgb_events) = {
                let decoded_view = returned.transpose_axes(&[0, 2, 3, 1]).unwrap();
                let active_before_rgb = rgb_observation_active();
                mlx_rs::memory::reset_peak_memory();
                let (image, events) =
                    rgb_observation_capture(|| crate::pipeline::decoded_to_image(&decoded_view));
                (
                    image,
                    active_before_rgb,
                    mlx_rs::memory::get_peak_memory() as u64,
                    rgb_observation_active(),
                    events,
                )
            }; // The temporary NHWC view ends as it does at the production call expression.
            let active_after_view_drop = rgb_observation_active();
            assert_eq!(
                (image.width, image.height),
                (10, 8),
                "actual decoder returned RGB axes, not a transposed metadata/golden shape"
            );
            assert_eq!(image.pixels.len(), 10 * 8 * 3);
            assert!(
                image.pixels.iter().any(|v| *v != 0 && *v != 255),
                "native tiny decoder is a nonzero actual RGB producer, not an empty/zero stub"
            );
            rgb_observation_assert_pixel_boundary(&rgb_events);
            let observed_increment = rgb_absolute_peak.checked_sub(active_before_rgb);
            eprintln!("SDXL_RGB_OBSERVATION vae={dtype:?} before_decode={active_before_decode} decode_absolute_peak={decode_absolute_peak} at_return={active_at_return} before_rgb={active_before_rgb} rgb_absolute_peak={rgb_absolute_peak} after_rgb_same_owners={active_after_rgb} after_view_drop={active_after_view_drop} increment={observed_increment:?} decode_events={decode_events:?} rgb_events={rgb_events:?}; observational ONLY/no inferred F16-parent coResidence/donation/newallocation/physical law");
            drop(returned);
            rgb_observation_sync_cpu();
            eprintln!(
                "SDXL_RGB_OBSERVATION vae={dtype:?} after_returned_owner_drop={}",
                rgb_observation_active()
            );
        }
        eprintln!(
            "SDXL_RGB_OBSERVATION_MECHANISM 1 actual_vae_return_and_original_lazy_pixel_boundary"
        );
    }
    completed.body_completed = true;
}

/// HOST-only mechanism regression: no Array/CPU-device/stream/allocator operation. The independent
/// native P5 producer already observes actual post-install q4 codes[64,8]/scales[64,1]/BF16.
#[test]
fn configured_pack_timing_preserves_final_alignment_and_unconfigured_rejection_host() {
    use crate::adapters::{AdapterInstallStorage, AdapterInstallWork};
    use mlx_gen::adapters::{facts::TensorFacts, LinearFacts};
    let tensor = |columns: i32| TensorFacts {
        shape: vec![64, columns],
        dtype: Dtype::Float16,
        strides: Vec::new(),
    };
    let dense = |columns: i32| crate::quantum::LinearWork {
        base: LinearFacts {
            base_shape: vec![64, columns],
            is_quantized: false,
            has_live_adapters: false,
            adapter_count: 0,
            bias_shape: None,
        },
        weights: vec![tensor(columns)],
        bias: None,
        residuals: Vec::new(),
        group_size: 64,
        bits: None,
    };
    // A metadata install declaration, not native state injection or a scratch/physical oracle.
    // Every real native file/value/guard-prefix assertion remains in the separate P5/TP tests.
    let install = |columns: i32| AdapterInstallWork {
        target: "host_projection".into(),
        storage: AdapterInstallStorage::DenseLinear,
        scratch: 4 * 64 * u64::try_from(columns).unwrap() + 128,
        pre_guard: 0,
        parents: vec![tensor(1)],
        installed: vec![tensor(columns)],
        disabled: false,
    };
    let mut configured =
        mlx_gen::gen_core::LoadSpec::new(mlx_gen::WeightsSource::Dir("HOST_metadata_only".into()));
    configured.adapters.push(mlx_gen::runtime::AdapterSpec::new(
        "HOST_adapter_metadata_only".into(),
        1.0,
        mlx_gen::runtime::AdapterKind::Lora,
    ));
    let empty =
        mlx_gen::gen_core::LoadSpec::new(mlx_gen::WeightsSource::Dir("HOST_metadata_only".into()));
    for final_columns in [64, 65] {
        let mut work = dense(1);
        let called = std::cell::Cell::new(false);
        let initial = crate::ios_pricing::initial_heavy_pack(Some(&configured), || {
            called.set(true);
            work.pack(4).map(|_| ())
        });
        assert!(
            initial.is_ok(),
            "configured install must precede validation of its original misaligned base"
        );
        assert!(
            !called.get(),
            "configured original width1 cannot be packed before the real install boundary"
        );
        assert_eq!(work.base.base_shape, [64, 1]);
        assert!(work.bits.is_none());
        work.install(&install(final_columns)).unwrap();
        let result = work.pack(4);
        if final_columns == 64 {
            result.unwrap();
            assert_eq!(work.weights[0].shape, [64, 8]);
            assert_eq!(work.weights[0].dtype, Dtype::Uint32);
            assert_eq!(work.weights[1].shape, [64, 1]);
            assert_eq!(work.weights[1].dtype, Dtype::Bfloat16);
            assert_eq!(work.weights[2].shape, [64, 1]);
            assert_eq!(work.weights[2].dtype, Dtype::Bfloat16);
            assert_eq!(work.bits, Some(4));
        } else {
            assert!(
                result.is_err(),
                "genuinely misaligned FINAL width must still reject after the install"
            );
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("not group-aligned"),
                "final rejection must preserve the actual alignment error family"
            );
            assert!(work.bits.is_none());
        }
    }
    for spec in [None, Some(&empty)] {
        let mut work = dense(1);
        let called = std::cell::Cell::new(false);
        let result = crate::ios_pricing::initial_heavy_pack(spec, || {
            called.set(true);
            work.pack(4).map(|_| ())
        });
        assert!(
            called.get(),
            "unconfigured/default path must keep its original validation boundary"
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not group-aligned"));
    }
    let mut ordinary = dense(64);
    crate::ios_pricing::initial_heavy_pack(Some(&empty), || ordinary.pack(4).map(|_| ())).unwrap();
    assert_eq!(ordinary.weights[0].shape, [64, 8]);
}

// Native-free CLIP component: independent header literals, followed by original-helper native
// producer comparisons. These tests grant no allocator/backing/physical-transfer certificate.
fn clip_literal_headers() -> (
    Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    ClipTextConfig,
) {
    use mlx_gen::gen_core::weightsmeta::{Dtype as Stored, SafetensorsTensorHeader as Header};
    let cfg = ClipTextConfig {
        num_layers: 2,
        model_dims: 64,
        num_heads: 1,
        max_length: 8,
        vocab_size: 40,
        projection_dim: Some(16),
        hidden_act: ClipActivation::Gelu,
    };
    let mut headers = Vec::new();
    let mut put = |name: String, shape: Vec<usize>| {
        let data_bytes = 4 * shape.iter().product::<usize>() as u64;
        headers.push(Header {
            name,
            dtype: Stored::F32,
            shape,
            data_bytes,
        });
    };
    put(
        "text_model.embeddings.token_embedding.weight".into(),
        vec![40, 64],
    );
    put(
        "text_model.embeddings.position_embedding.weight".into(),
        vec![8, 64],
    );
    for index in 0..2 {
        let prefix = format!("text_model.encoder.layers.{index}");
        for norm in ["layer_norm1", "layer_norm2"] {
            for part in ["weight", "bias"] {
                put(format!("{prefix}.{norm}.{part}"), vec![64]);
            }
        }
        // Independent literal constructor inventory, not quantizable_text_paths/LAYER_LINEAR_NAMES.
        for (name, out, input) in [
            ("self_attn.q_proj", 64, 64),
            ("self_attn.k_proj", 64, 64),
            ("self_attn.v_proj", 64, 64),
            ("self_attn.out_proj", 64, 64),
            ("mlp.fc1", 192, 64),
            ("mlp.fc2", 64, 192),
        ] {
            put(format!("{prefix}.{name}.weight"), vec![out, input]);
            put(format!("{prefix}.{name}.bias"), vec![out]);
        }
    }
    put("text_model.final_layer_norm.weight".into(), vec![64]);
    put("text_model.final_layer_norm.bias".into(), vec![64]);
    put("text_projection.weight".into(), vec![16, 64]);
    (headers, cfg)
}

#[test]
fn native_free_clip_headers_preserve_literals_lookup_casts_and_pack_host() {
    use crate::text_encoder::ClipPrice;
    use mlx_gen::gen_core::weightsmeta::{Dtype as Stored, SafetensorsTensorHeader as Header};
    let (headers, cfg) = clip_literal_headers();
    let mut price = ClipPrice::from_headers(&headers, "text_model", &cfg).unwrap();
    let facts = price.projection_facts();
    assert_eq!(facts.len(), 13);
    assert_eq!(facts[0].0, "text_model.encoder.layers.0.self_attn.q_proj");
    assert_eq!(facts[12].0, "text_projection");
    assert_eq!(facts[0].1.weights[0].shape, [64, 64]);
    assert_eq!(facts[4].1.weights[0].shape, [192, 64]);
    assert_eq!(facts[5].1.weights[0].shape, [64, 192]);
    assert!(facts
        .iter()
        .all(|(_, p)| p.weights[0].dtype == Dtype::Float16 && p.bits.is_none()));
    assert_eq!(facts[0].1.bias.as_ref().unwrap().dtype, Dtype::Float16);
    assert!(facts[12].1.bias.is_none());
    assert_eq!(price.pooled_width(), 16);
    crate::quantum::with_backend(Backend::Cpu, || {
        let mut quanta = Vec::new();
        assert_eq!(
            price.price_forward(2, 8, &mut quanta),
            (Dtype::Float16, Dtype::Float16)
        );
        // Independent literals for embedding extent and CPU final norm's competing live sets:
        // 3*(2*8*64*2); max(1024*12,1024*18)+3*16*4 (no affine casts).
        assert_eq!(quanta[0], (CLIP_EMBEDDINGS, 6144));
        assert_eq!(quanta[3], (CLIP_POOLED, 18624));
        assert_eq!(
            quanta.iter().map(|p| p.0).collect::<Vec<_>>(),
            [CLIP_EMBEDDINGS, CLIP_LAYER, CLIP_LAYER, CLIP_POOLED]
        );
    });
    for bits in [4, 8] {
        let mut packed = ClipPrice::from_headers(&headers, "text_model", &cfg).unwrap();
        packed.quantize(bits).unwrap();
        let facts = packed.projection_facts();
        assert_eq!(
            facts[0].1.weights[0].shape,
            [64, if bits == 4 { 8 } else { 16 }]
        );
        assert_eq!(facts[0].1.weights[0].dtype, Dtype::Uint32);
        assert_eq!(facts[0].1.weights[1].shape, [64, 1]);
        assert_eq!(facts[0].1.weights[1].dtype, Dtype::Bfloat16);
        assert_eq!(facts[0].1.weights[2].dtype, Dtype::Bfloat16);
        assert_eq!(facts[0].1.bias.as_ref().unwrap().dtype, Dtype::Bfloat16);
        assert_eq!(
            facts[12].1.weights[0].shape,
            [16, if bits == 4 { 8 } else { 16 }]
        );
        crate::quantum::with_backend(Backend::Cpu, || {
            let mut quanta = Vec::new();
            assert_eq!(
                packed.price_forward(2, 8, &mut quanta),
                (Dtype::Float32, Dtype::Float32)
            );
            // CPU final norm at F32: 1024*12 + 3*16*4 + 2*64*4 affine casts.
            assert_eq!(quanta[3], (CLIP_POOLED, 12992));
        });
    }
    // Literal source q8 codes/scales + ordinary bias. Requested q4 is a constructor/pack no-op,
    // not a public tier-mismatch policy bypass: that policy belongs to the outer loader.
    let mut stored = headers.clone();
    let prefix = "text_model.encoder.layers.0.self_attn.q_proj";
    let weight = stored
        .iter_mut()
        .find(|h| h.name == format!("{prefix}.weight"))
        .unwrap();
    weight.shape = vec![64, 16];
    weight.dtype = Stored::U32;
    weight.data_bytes = 4096;
    for part in ["scales", "biases"] {
        stored.push(Header {
            name: format!("{prefix}.{part}"),
            dtype: Stored::BF16,
            shape: vec![64, 1],
            data_bytes: 128,
        });
    }
    let mut source = ClipPrice::from_headers(&stored, "text_model", &cfg).unwrap();
    source.quantize(4).unwrap();
    let facts = source.projection_facts();
    assert_eq!(facts[0].1.bits, Some(8));
    assert_eq!(facts[0].1.weights[0].shape, [64, 16]);
    assert_eq!(facts[0].1.weights[1].dtype, Dtype::Bfloat16);
    assert_eq!(
        facts[0].1.bias.as_ref().unwrap().dtype,
        Dtype::Float16,
        "stored packed ordinary bias is still a floating dense leaf at load"
    );
    // Actual constructor errors, including priority. Do not replace them with a shape-policy gate.
    let mut missing = headers.clone();
    missing.retain(|h| ![format!("{prefix}.bias"), format!("{prefix}.weight")].contains(&h.name));
    let error = ClipPrice::from_headers(&missing, "text_model", &cfg)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains(&format!("{prefix}.bias")),
        "ordinary bias precedes weight: {error}"
    );
    let mut invalid = stored.clone();
    invalid
        .iter_mut()
        .find(|h| h.name == format!("{prefix}.weight"))
        .unwrap()
        .dtype = Stored::F32;
    invalid.retain(|h| h.name != format!("{prefix}.biases"));
    let error = ClipPrice::from_headers(&invalid, "text_model", &cfg)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("codes must be u32"),
        "packed validator precedes offsets lookup: {error}"
    );
    let mut batched = headers.clone();
    batched
        .iter_mut()
        .find(|h| h.name == format!("{prefix}.weight"))
        .unwrap()
        .shape = vec![1, 64, 64];
    assert!(
        ClipPrice::from_headers(&batched, "text_model", &cfg).is_ok(),
        "dense construction must not acquire the header helper's extra exact-matrix restriction"
    );
    // All source facts remain unchanged; no Array/device/file or native payload factory above.
    price.quantize(4).unwrap();
    assert_eq!(headers, clip_literal_headers().0);
}

#[test]
fn native_free_clip_pair_matches_actual_load_pack_and_forward_cpu() {
    let Some(mut completed) = completed_cpu_child(
        "pricing::native_free_clip_pair_matches_actual_load_pack_and_forward_cpu",
    ) else {
        return;
    };
    {
        use crate::text_encoder::ClipPrice;
        use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
        use mlx_gen::ios_quantum::{set_quantum_listener, QuantumMark};
        use std::sync::{Arc, Mutex};
        struct ListenerReset;
        impl Drop for ListenerReset {
            fn drop(&mut self) {
                assert!(set_quantum_listener(None));
            }
        }
        for (row, (bits, stored, mixed_scales)) in [
            (None, false, false),
            (Some(4), false, false),
            (Some(8), false, false),
            (Some(4), true, false),
            (Some(8), true, false),
            (Some(4), true, true),
        ]
        .into_iter()
        .enumerate()
        {
            let dir = tempfile::tempdir().unwrap();
            let mut configs = Vec::new();
            let mut files = Vec::new();
            let mut initial_owners = Vec::new();
            for encoder in 0..2 {
                let (weights, cfg) = tiny_clip(
                    64,
                    1,
                    if encoder == 1 { Some(16) } else { None },
                    700 + encoder,
                );
                let mut weights = half(weights);
                if stored {
                    let paths = crate::text_encoder::quantizable_text_paths(&cfg);
                    let map = mlx_gen::quant::quantize_map(
                        weights.into_tensors(),
                        bits.unwrap(),
                        64,
                        |name| paths.iter().any(|(path, _)| path == name),
                    )
                    .unwrap();
                    weights = Weights::from_map(map);
                    if mixed_scales {
                        let names: Vec<_> = weights
                            .keys()
                            .filter(|name| name.ends_with(".scales"))
                            .map(str::to_owned)
                            .collect();
                        for name in names {
                            // Actual supported stored F32/F16 scales, not a configured tier guess.
                            let dtype = if name.contains("layers.0.") {
                                Dtype::Float16
                            } else {
                                Dtype::Float32
                            };
                            let array = weights.require(&name).unwrap().as_dtype(dtype).unwrap();
                            weights.insert(name, array);
                        }
                    }
                }
                let file = dir.path().join(if encoder == 0 {
                    "text_encoder/model.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                });
                save(&weights, &file);
                // Independent actual native load/cast owners, logical page-rounded payload only.
                // Drop the map before loading the executable model; no extra owners span its guards.
                let mut loaded = Weights::from_file(&file).unwrap();
                with_cpu_diagnostic_headroom(PLENTY, || {
                    crate::loader::cast_for_load(&mut loaded, crate::model::DTYPE)
                })
                .unwrap();
                initial_owners.push(
                    loaded
                        .keys()
                        .map(|name| {
                            mlx_gen::ios_quantum::allocated_bytes(crate::quantum::bytes(
                                loaded.require(name).unwrap(),
                            ))
                        })
                        .sum::<u64>(),
                );
                drop(loaded);
                drop(weights);
                configs.push(cfg);
                files.push(file);
            }
            let headers = files
                .iter()
                .map(|file| safetensors_path_tensor_headers(file).unwrap())
                .collect::<Vec<_>>();
            let mut metadata = (
                ClipPrice::from_headers(&headers[0], "text_model", &configs[0]).unwrap(),
                ClipPrice::from_headers(&headers[1], "text_model", &configs[1]).unwrap(),
            );
            let pack_facts = [
                crate::text_encoder::text_pack_facts(&headers[0], &configs[0], bits).unwrap(),
                crate::text_encoder::text_pack_facts(&headers[1], &configs[1], bits).unwrap(),
            ];
            if let Some(bits) = bits {
                metadata.0.quantize(bits).unwrap();
                metadata.1.quantize(bits).unwrap();
            }
            let marks = Arc::new(Mutex::new(Vec::<(bool, u64, u64)>::new()));
            let sink = marks.clone();
            assert!(!set_quantum_listener(Some(Box::new(move |mark| {
                if let QuantumMark::Metadata {
                    provider: "sdxl",
                    kind,
                    fields,
                } = mark
                {
                    if ["quantize-linear-before", "quantize-linear-after"].contains(&kind) {
                        let get = |key| {
                            fields
                                .iter()
                                .find(|(name, _)| *name == key)
                                .map_or(0, |(_, v)| *v)
                        };
                        sink.lock().unwrap().push((
                            kind == "quantize-linear-before",
                            get("logical_bytes"),
                            get("budget"),
                        ));
                    }
                }
            }))));
            let reset = ListenerReset;
            let (native, load_events) = observe(PLENTY, || {
                let mut one = crate::loader::load_clip_dtype(
                    dir.path(),
                    "text_encoder",
                    &configs[0],
                    crate::model::DTYPE,
                )?;
                let mut two = crate::loader::load_clip_dtype(
                    dir.path(),
                    "text_encoder_2",
                    &configs[1],
                    crate::model::DTYPE,
                )?;
                if let Some(bits) = bits {
                    crate::text_encoder::quantize_pair(&mut one, &mut two, bits)?;
                }
                Ok((one, two))
            });
            drop(reset);
            let (one, two) = native.unwrap();
            let expected_packs: Vec<_> = pack_facts
                .iter()
                .flat_map(|f| f.points.iter().map(|p| p.scratch))
                .collect();
            let observed_packs: Vec<_> = load_events
                .iter()
                .filter_map(|e| match e {
                    Ev::Checked(label, scratch) if label == LOAD_QUANTIZE => Some(*scratch),
                    _ => None,
                })
                .collect();
            assert_eq!(
                expected_packs, observed_packs,
                "row{row} ordered real pack guards"
            );
            let marks = marks.lock().unwrap();
            let mut cursor = 0;
            for (index, facts) in pack_facts.iter().enumerate() {
                let mut owners = initial_owners[index];
                for point in &facts.points {
                    let before = marks[cursor];
                    let after = marks[cursor + 1];
                    cursor += 2;
                    assert!(before.0 && !after.0, "native pack begin/end order");
                    assert_eq!(
                        point.before_resident, owners,
                        "row{row} encoder{index} actual logical owners"
                    );
                    assert_eq!(point.scratch, before.2);
                    owners = owners
                        .checked_sub(before.1)
                        .unwrap()
                        .checked_add(after.1)
                        .unwrap();
                }
                assert_eq!(
                    facts.retained, owners,
                    "row{row} final declared owners; not backing release"
                );
            }
            assert_eq!(cursor, marks.len());
            drop(marks);
            for (expected, native) in [(&metadata.0, &one), (&metadata.1, &two)] {
                let actual_facts = native.price_facts().projection_facts();
                let expected_facts = expected.projection_facts();
                assert_eq!(expected_facts.len(), actual_facts.len());
                for ((name, want), (actual_name, actual)) in
                    expected_facts.iter().zip(&actual_facts)
                {
                    assert_eq!(name, actual_name);
                    assert_eq!(want.base, actual.base, "row{row} {name} actual native base");
                    assert_eq!(want.bits, actual.bits);
                    assert_eq!(want.group_size, actual.group_size);
                    assert_eq!(
                        want.weights
                            .iter()
                            .map(|t| (&t.shape, t.dtype))
                            .collect::<Vec<_>>(),
                        actual
                            .weights
                            .iter()
                            .map(|t| (&t.shape, t.dtype))
                            .collect::<Vec<_>>(),
                        "row{row} {name} exact native operands, no synthetic stride equality"
                    );
                    assert_eq!(
                        want.bias.as_ref().map(|t| (&t.shape, t.dtype)),
                        actual.bias.as_ref().map(|t| (&t.shape, t.dtype))
                    );
                }
                let mut quanta = Vec::new();
                let dtypes = crate::quantum::with_backend(Backend::Cpu, || {
                    expected.price_forward(2, 8, &mut quanta)
                });
                let ids = clip_ids(8);
                let (output, events) = observe(PLENTY, || native.forward(&ids));
                let output = output.unwrap();
                let observed: Vec<_> = events
                    .iter()
                    .filter_map(|event| match event {
                        Ev::Checked(label, scratch) => Some((label.as_str(), *scratch)),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    quanta, observed,
                    "row{row} every actual CLIP guard, not maxima"
                );
                let penultimate = &output.hidden_states[output.hidden_states.len() - 2];
                assert_eq!(
                    dtypes,
                    (penultimate.dtype(), output.pooled.dtype()),
                    "row{row} actual native output dtypes"
                );
                assert_eq!(expected.pooled_width(), output.pooled.shape()[1]);
                assert_eq!(native.pooled_width(), output.pooled.shape()[1]);
                assert_eq!(
                    super::bits(&output.pooled).len(),
                    (2 * expected.pooled_width()) as usize
                );
            }
            eprintln!("SDXL_NATIVE_FREE_CLIP_ROW {row} bits={bits:?} stored={stored} mixed={mixed_scales}");
        }
        // Actual served Priced construction/conditioning/pooled-width consumer, not only detached
        // metadata tests. Other components still have native graphs; this is not full native-free.
        let snap = Snapshot::groupable_text(Some(4), false);
        let priced = Priced::build(&snap.inputs(Route::Text)).unwrap();
        let (one, two) = Files::of(&snap).text().unwrap();
        let ids = clip_ids(TOKENS);
        let (native, events) = observe(PLENTY, || {
            let one = one.forward(&ids)?;
            let two = two.forward(&ids)?;
            Ok((one, two))
        });
        let (one, two) = native.unwrap();
        let mut quanta = Vec::new();
        let dtypes = crate::quantum::with_backend(Backend::Cpu, || {
            priced.price_text_conditioning(2, &mut quanta)
        });
        let observed: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Ev::Checked(label, scratch) => Some((label.as_str(), *scratch)),
                _ => None,
            })
            .collect();
        assert_eq!(
            quanta, observed,
            "served pair must execute both requested packs"
        );
        assert_eq!(
            dtypes.0,
            Dtype::from_promoting_types(one.hidden_states[1].dtype(), two.hidden_states[1].dtype())
        );
        assert_eq!(dtypes.1, two.pooled.dtype());
        // Its shape is actually consumed inside Priced::price; no fixed TE2 width replacement.
        assert!(priced
            .price(&regime(Route::Text))
            .unwrap()
            .0
            .phase(IosPhase::Conditioning)
            .is_some());
        eprintln!("SDXL_NATIVE_FREE_CLIP_MECHANISM 0 six_native_rows_and_served_pair");
    }
    completed.body_completed = true;
}

// Native-free hierarchy literals are independent of the fixture's native parameter generator.
fn hierarchy_header(
    name: &str,
    shape: &[usize],
    dtype: mlx_gen::gen_core::weightsmeta::Dtype,
) -> mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    let width = match dtype {
        Stored::F32 | Stored::U32 => 4,
        Stored::F16 | Stored::BF16 => 2,
        _ => panic!("literal fixture uses only these source types"),
    };
    mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader {
        name: name.into(),
        dtype,
        shape: shape.to_vec(),
        data_bytes: width * shape.iter().product::<usize>() as u64,
    }
}

#[test]
fn native_free_unet_control_literals_layout_geglu_effects_stream_and_merge_host() {
    use crate::unet::pricing::{add, tensor, HeaderWeights, ResnetFacts, TransformerBlockFacts};
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    let mut headers = Vec::new();
    for (name, shape) in [
        ("r.conv_shortcut.weight", vec![64, 32, 1, 1]),
        ("r.conv_shortcut.bias", vec![64]),
        ("r.time_emb_proj.weight", vec![64, 128]),
        ("r.time_emb_proj.bias", vec![64]),
        ("r.norm1.weight", vec![32]),
        ("r.norm1.bias", vec![32]),
        ("r.conv1.weight", vec![64, 32, 3, 3]),
        ("r.conv1.bias", vec![64]),
        ("r.norm2.weight", vec![64]),
        ("r.norm2.bias", vec![64]),
        ("r.conv2.weight", vec![64, 64, 3, 3]),
        ("r.conv2.bias", vec![64]),
    ] {
        headers.push(hierarchy_header(name, &shape, Stored::F32));
    }
    let weights = HeaderWeights::new(&headers).unwrap();
    let mut resnet = ResnetFacts::from_weights(&weights, "r").unwrap();
    assert_eq!(
        resnet.adaptable_conv_mut(&["conv1"]).unwrap().weight.shape,
        [64, 3, 3, 32]
    );
    assert_eq!(
        resnet.adaptable_conv_mut(&["conv2"]).unwrap().weight.shape,
        [64, 3, 3, 64]
    );
    let shortcut = resnet.adaptable_mut(&["conv_shortcut"]).unwrap();
    assert_eq!(shortcut.weights[0].shape, [64, 32]);
    assert_eq!(shortcut.weights[0].dtype, Dtype::Float16);
    let mut order = Vec::new();
    resnet
        .quantize_using("r", 4, &mut |name, projection, bits| {
            order.push(name.to_owned());
            projection.pack(bits)?;
            Ok(())
        })
        .unwrap();
    assert_eq!(order, ["r.time_emb_proj"]);
    assert!(
        resnet
            .adaptable_mut(&["conv_shortcut"])
            .unwrap()
            .bits
            .is_none(),
        "the actual dense shortcut is not in the pack walk"
    );
    assert_eq!(
        resnet
            .output_shape_with_effects(&[1, 4, 5, 32], None, &crate::quantum::EffectScope::native())
            .unwrap(),
        [1, 4, 5, 64]
    );
    assert_eq!(
        resnet
            .output_dtype_with_effects(
                Dtype::Float16,
                Some(Dtype::Float16),
                &crate::quantum::EffectScope::native()
            )
            .unwrap(),
        Dtype::Float32
    );
    let mut missing = headers.clone();
    missing.retain(|h| h.name != "r.conv_shortcut.bias" && h.name != "r.norm1.weight");
    let error = ResnetFacts::from_weights(&HeaderWeights::new(&missing).unwrap(), "r")
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("r.conv_shortcut.bias"),
        "shortcut reshape/bias precede norm lookup: {error}"
    );

    let mut block = Vec::new();
    for norm in ["norm1", "norm2", "norm3"] {
        for part in ["weight", "bias"] {
            block.push(hierarchy_header(
                &format!("t.{norm}.{part}"),
                &[64],
                Stored::F32,
            ));
        }
    }
    for attention in ["attn1", "attn2"] {
        for role in ["to_q", "to_k", "to_v"] {
            let input = if attention == "attn2" && role != "to_q" {
                128
            } else {
                64
            };
            block.push(hierarchy_header(
                &format!("t.{attention}.{role}.weight"),
                &[64, input],
                Stored::F32,
            ));
        }
        block.push(hierarchy_header(
            &format!("t.{attention}.to_out.0.weight"),
            &[64, 64],
            Stored::F32,
        ));
        block.push(hierarchy_header(
            &format!("t.{attention}.to_out.0.bias"),
            &[64],
            Stored::F32,
        ));
    }
    block.push(hierarchy_header(
        "t.ff.net.0.proj.weight",
        &[384, 64],
        Stored::F32,
    ));
    block.push(hierarchy_header(
        "t.ff.net.0.proj.bias",
        &[384],
        Stored::F32,
    ));
    block.push(hierarchy_header(
        "t.ff.net.2.weight",
        &[64, 192],
        Stored::F32,
    ));
    block.push(hierarchy_header("t.ff.net.2.bias", &[64], Stored::F32));
    let mut model =
        TransformerBlockFacts::from_weights(&HeaderWeights::new(&block).unwrap(), "t", 64, 1)
            .unwrap();
    for role in ["linear1", "linear2"] {
        let projection = model.adaptable_mut(&["ff", role]).unwrap();
        assert_eq!(projection.weights[0].shape, [192, 64]);
        assert_eq!(projection.bias.as_ref().unwrap().shape, [192]);
    }
    let mut order = Vec::new();
    model
        .quantize_using("t", 4, &mut |name, projection, bits| {
            order.push(name.to_owned());
            projection.pack(bits)?;
            Ok(())
        })
        .unwrap();
    assert_eq!(order.len(), 11);
    assert_eq!(
        (&order[0], &order[3], &order[4], &order[7]),
        (
            &"t.attn1.to_q".to_owned(),
            &"t.attn1.to_out.0".to_owned(),
            &"t.attn2.to_q".to_owned(),
            &"t.attn2.to_out.0".to_owned()
        )
    );
    assert_eq!(
        &order[8..],
        ["t.ff.linear1", "t.ff.linear2", "t.ff.linear3"]
    );
    assert_eq!(
        model.adaptable_mut(&["ff", "linear1"]).unwrap().weights[0].shape,
        [192, 8]
    );
    assert_eq!(
        model.adaptable_mut(&["ff", "linear1"]).unwrap().weights[1].dtype,
        Dtype::Bfloat16
    );
    assert_eq!(
        model.adaptable_mut(&["attn2", "to_k"]).unwrap().weights[1].shape,
        [64, 2]
    );
    let mut invalid = block.clone();
    invalid
        .iter_mut()
        .find(|h| h.name == "t.ff.net.0.proj.weight")
        .unwrap()
        .shape[0] = 383;
    invalid.retain(|h| h.name != "t.norm1.weight");
    let error =
        TransformerBlockFacts::from_weights(&HeaderWeights::new(&invalid).unwrap(), "t", 64, 1)
            .err()
            .unwrap()
            .to_string();
    assert!(
        error.contains("odd output dim"),
        "GEGLU selection precedes norms: {error}"
    );

    let left = tensor(&[1, 4, 5, 1], Dtype::Float16).unwrap();
    let right = tensor(&[2, 1, 5, 64], Dtype::Bfloat16).unwrap();
    let sum = add(&left, &right).unwrap();
    assert_eq!(sum.shape, [2, 4, 5, 64]);
    assert_eq!(sum.dtype, Dtype::Float32);
    let hidden = tensor(&[1, 4, 5, 64], Dtype::Float16).unwrap();
    let residual = tensor(&[1, 4, 5, 64], Dtype::Float32).unwrap();
    assert_eq!(
        crate::quantum::control_injection_bytes_metadata(&hidden, &residual),
        10240
    );
    let mut pending = crate::quantum::ControlMergeWork::default();
    pending
        .record_outputs_metadata(std::slice::from_ref(&sum), &sum)
        .unwrap();
    assert_eq!(pending.down, mlx_gen::ios_quantum::allocated_bytes(10240));
    assert_eq!(pending.mid, pending.down);
    // Independent literal dense read+cast+GEGLU selection budget: 98304 raw +49152 cast
    // +49152 half-views +two separate [192] i32 selections below the rounding threshold.
    // Each selection is 768 bytes, not page-rounded. Not physical allocation.
    let stream = [hierarchy_header(
        "a.transformer_blocks.0.ff.net.0.proj.weight",
        &[384, 64],
        Stored::F32,
    )];
    assert_eq!(
        crate::block_stream::streamed_block_bytes_from_headers(
            &stream,
            "a",
            1,
            Dtype::Float16,
            None
        )
        .unwrap(),
        [98_304 + 49_152 + 49_152 + 2 * 768]
    );
    assert!(crate::block_stream::streamed_block_bytes_from_headers(
        &stream,
        "a",
        1,
        Dtype::Float16,
        Some(4)
    )
    .unwrap_err()
    .to_string()
    .contains("not a streamable load"));
}

#[test]
fn native_free_unet_control_matches_actual_constructor_pack_and_guards_cpu() {
    let Some(mut completed) = completed_cpu_child(
        "pricing::native_free_unet_control_matches_actual_constructor_pack_and_guards_cpu",
    ) else {
        return;
    };
    {
        use crate::quantum::LinearWork;
        use crate::unet::pricing::{ip_pairs_from_headers, ControlFacts, UNetFacts};
        use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
        let compare = |name: &str, expected: &LinearWork, actual: &LinearWork| {
            assert_eq!(
                expected.base, actual.base,
                "{name}: native target/base facts"
            );
            assert_eq!(
                expected.bits, actual.bits,
                "{name}: stored/final pack policy"
            );
            assert_eq!(expected.group_size, actual.group_size);
            assert_eq!(
                expected
                    .weights
                    .iter()
                    .map(|t| (&t.shape, t.dtype))
                    .collect::<Vec<_>>(),
                actual
                    .weights
                    .iter()
                    .map(|t| (&t.shape, t.dtype))
                    .collect::<Vec<_>>(),
                "{name}: native operands"
            );
            assert_eq!(
                expected.bias.as_ref().map(|t| (&t.shape, t.dtype)),
                actual.bias.as_ref().map(|t| (&t.shape, t.dtype))
            );
        };
        let mut cases = 0;
        for bits in [None, Some(4), Some(8)] {
            let snap = match bits {
                None => Snapshot::new(2),
                Some(bits) => Snapshot::packed(bits),
            };
            for route in [Route::Text, Route::Control, Route::IpAdapter] {
                let inputs = snap.inputs(route);
                let headers = safetensors_path_tensor_headers(&inputs.unet).unwrap();
                let mut model = UNetFacts::from_headers(&headers, &snap.unet).unwrap();
                if let Some((_, ip_file)) = &inputs.ip {
                    model
                        .install_ip_adapter(
                            ip_pairs_from_headers(
                                &safetensors_path_tensor_headers(ip_file).unwrap(),
                            )
                            .unwrap(),
                        )
                        .unwrap();
                }
                let mut expected = Vec::new();
                model
                    .quantize_using(bits.unwrap_or(4), &mut |path, linear, _| {
                        if let Some(bits) = bits {
                            linear.pack(bits)?;
                        }
                        expected.push((path.to_owned(), linear.clone()));
                        Ok(())
                    })
                    .unwrap();
                let (native, _) = observe(PLENTY, || load(&snap, route));
                let mut loaded = native.unwrap();
                let mut actual = Vec::new();
                loaded
                    .unet
                    .quantize_using(bits.unwrap_or(4), &mut |path, linear, _| {
                        actual.push((path.to_owned(), LinearWork::of(linear)));
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(expected.len(), actual.len());
                for ((name, expected), (actual_name, actual)) in expected.iter().zip(&actual) {
                    assert_eq!(
                        name, actual_name,
                        "UNet actual ordered pack target identity"
                    );
                    compare(name, expected, actual);
                }
                if route == Route::IpAdapter {
                    let ip: Vec<_> = expected
                        .iter()
                        .filter(|(name, _)| name.ends_with("to_k_ip"))
                        .map(|(name, _)| name)
                        .collect();
                    assert!(!ip.is_empty());
                    assert!(ip.first().unwrap().starts_with("down_blocks."));
                    // Installation uses down→up→mid; the actual pack walk independently uses
                    // down→mid→up. Native role/shape comparison above distinguishes both orders.
                }
                if route == Route::Control {
                    let file = &inputs.controls[0];
                    let mut control = ControlFacts::from_headers(
                        &safetensors_path_tensor_headers(file).unwrap(),
                        &snap.unet,
                    )
                    .unwrap();
                    let mut expected = Vec::new();
                    let mut actual = Vec::new();
                    control
                        .quantize_using(bits.unwrap_or(4), &mut |path, linear, _| {
                            if let Some(bits) = bits {
                                linear.pack(bits)?;
                            }
                            expected.push((path.to_owned(), linear.clone()));
                            Ok(())
                        })
                        .unwrap();
                    loaded.controls[0]
                        .quantize_using(bits.unwrap_or(4), &mut |path, linear, _| {
                            actual.push((path.to_owned(), LinearWork::of(linear)));
                            Ok(())
                        })
                        .unwrap();
                    assert_eq!(expected.len(), actual.len());
                    for ((name, expected), (actual_name, actual)) in expected.iter().zip(&actual) {
                        assert_eq!(name, actual_name, "ControlNet actual ordered pack");
                        compare(name, expected, actual);
                    }
                }
                let r = regime(route);
                let (_, items) = crate::quantum::with_backend(Backend::Cpu, || {
                    Priced::build(&inputs).unwrap().price(&r).unwrap()
                });
                let mut expected = Vec::new();
                for item in items
                    .iter()
                    .filter(|q| q.phase == IosPhase::Denoise && q.label != SAMPLER_STEP)
                {
                    expected.push((item.label.to_owned(), item.scratch));
                    if item.label == UNET_CONV_OUT {
                        break;
                    }
                }
                assert_eq!(expected.last().map(|q| q.0.as_str()), Some(UNET_CONV_OUT));
                let tiling = mlx_gen::tiling::TilingConfig::spatial_only(8, 0);
                let (result, events) = observe(PLENTY, || run_request(&loaded, route, &tiling));
                result.unwrap();
                assert_well_ordered(
                    &events
                        .iter()
                        .filter(|e| match e {
                            Ev::Checked(label, _) => {
                                ios_quantum_phase(label) != Some(IosPhase::Load)
                            }
                            _ => true,
                        })
                        .cloned()
                        .collect::<Vec<_>>(),
                );
                let mut actual = Vec::new();
                for event in &events {
                    if let Ev::Checked(label, scratch) = event {
                        if ios_quantum_phase(label) == Some(IosPhase::Denoise)
                            && label != SAMPLER_STEP
                        {
                            actual.push((label.clone(), *scratch));
                            if label == UNET_CONV_OUT {
                                break;
                            }
                        }
                    }
                }
                assert_eq!(
                    expected, actual,
                    "{bits:?}/{route:?}: actual pipeline first-forward ordered guards, not maxima"
                );
                cases += 1;
                eprintln!("SDXL_NATIVE_FREE_HIERARCHY_ROW {cases} bits={bits:?} route={route:?}");
            }
        }
        assert_eq!(cases, 9);
        eprintln!("SDXL_NATIVE_FREE_HIERARCHY_MECHANISM 0 nine_synthetic_native_cases_not_physical_evidence");
    }
    completed.body_completed = true;
}

#[test]
fn native_free_multicontrol_deferred_sums_match_actual_pipeline_cpu() {
    let Some(mut completed) = completed_cpu_child(
        "pricing::native_free_multicontrol_deferred_sums_match_actual_pipeline_cpu",
    ) else {
        return;
    };
    {
        // Private explicit-config fixture: two real loaded branches/contexts. This proves this
        // pipeline/pricing consumer, not catalog admission or a physical allocation theorem.
        let snap = Snapshot::packed(4);
        let mut inputs = snap.inputs(Route::Control);
        inputs.controls.push(inputs.controls[0].clone());
        let (loaded, _) = observe(PLENTY, || {
            let mut loaded = load(&snap, Route::Control)?;
            let mut extra = crate::loader::load_controlnet_with(
                &mlx_gen::WeightsSource::Dir(snap.root.join("controlnet")),
                crate::model::DTYPE,
                &snap.unet,
            )?;
            extra.quantize(4)?;
            loaded.controls.push(extra);
            Ok(loaded)
        });
        let loaded = loaded.unwrap();
        let ((cond, pooled, contexts), _) = {
            let (result, events) = observe(PLENTY, || {
                let (cond, pooled) =
                    crate::pipeline::encode_conditioning(&loaded.te1, &loaded.te2, &ids())?;
                let mut contexts = Vec::new();
                for (index, branch) in loaded.controls.iter().enumerate() {
                    let edge = LATENT * CONTROL_SCALE;
                    let pixels = crate::pipeline::preprocess_control_image(
                        &image(edge, edge, 17 + index as i32),
                        edge as u32,
                        edge as u32,
                    )?;
                    let pair = mlx_rs::ops::concatenate_axis(&[&pixels, &pixels], 0)?;
                    contexts.push(crate::pipeline::ControlContext {
                        controlnet: branch,
                        cond_embed: branch.embed_cond(&pair)?,
                        scale: if index == 0 { 0.25 } else { 0.75 },
                    });
                }
                Ok((cond, pooled, contexts))
            });
            (result.unwrap(), events)
        };
        assert_eq!(contexts.len(), 2);
        let euler = crate::sampler::EulerSampler::new_with_dtype(
            &crate::config::DiffusionConfig::sdxl_base(),
            true,
            crate::model::DTYPE,
        )
        .unwrap();
        let sampler = crate::sampler::AncestralEuler::new(&euler, 2, 800.0).unwrap();
        let plan =
            crate::plan::SdxlForwardPlan::UNBOUNDED.with_cfg_batching(CfgBatching::Sequential);
        let denoiser = crate::pipeline::Denoiser::with_plan(&loaded.unet, &sampler, plan);
        let latents = euler.sample_prior(&[1, LATENT, LATENT, 4]).unwrap();
        let time_ids = crate::pipeline::text_time_ids(2);
        let cancel = mlx_gen::CancelFlag::default();
        let (result, events) = observe(PLENTY, || {
            crate::pipeline::denoise_multi_control(
                &denoiser,
                latents,
                &cond,
                &pooled,
                &time_ids,
                5.0,
                &cancel,
                &mut |_| {},
                &contexts,
            )
        });
        let result = result.unwrap();
        assert_eq!(super::bits(&result).len(), (LATENT * LATENT * 4) as usize);
        let (_, items) = crate::quantum::with_backend(Backend::Cpu, || {
            Priced::build(&inputs)
                .unwrap()
                .price(&regime(Route::Control))
                .unwrap()
        });
        let mut expected = Vec::new();
        for item in items
            .iter()
            .filter(|q| q.phase == IosPhase::Denoise && q.label != SAMPLER_STEP)
        {
            expected.push((item.label.to_owned(), item.scratch));
            if item.label == UNET_CONV_OUT {
                break;
            }
        }
        let mut actual = Vec::new();
        for event in &events {
            if let Ev::Checked(label, scratch) = event {
                if ios_quantum_phase(label) == Some(IosPhase::Denoise) && label != SAMPLER_STEP {
                    actual.push((label.clone(), *scratch));
                    if label == UNET_CONV_OUT {
                        break;
                    }
                }
            }
        }
        assert_eq!(
            expected, actual,
            "two actual pipeline branches, pending sum DAG and boundary casts"
        );
        assert_eq!(actual.iter().filter(|p| p.0 == CONTROL_STEM).count(), 2);
        assert_eq!(
            actual
                .iter()
                .filter(|p| p.0 == CONTROL_DOWN_INJECTION)
                .count(),
            1
        );
        assert_eq!(
            actual
                .iter()
                .filter(|p| p.0 == CONTROL_MID_INJECTION)
                .count(),
            1
        );
        eprintln!(
            "SDXL_NATIVE_FREE_MULTICONTROL_MECHANISM 0 two_real_contexts_quarter_and_three_quarter"
        );
    }
    completed.body_completed = true;
}

fn vae_literal_conv(
    headers: &mut Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
) {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    headers.push(hierarchy_header(
        &format!("{prefix}.weight"),
        &[output, input, kernel, kernel],
        Stored::F32,
    ));
    headers.push(hierarchy_header(
        &format!("{prefix}.bias"),
        &[output],
        Stored::F32,
    ));
}
fn vae_literal_norm(
    headers: &mut Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    prefix: &str,
) {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    for part in ["weight", "bias"] {
        headers.push(hierarchy_header(
            &format!("{prefix}.{part}"),
            &[32],
            Stored::F32,
        ));
    }
}
fn vae_literal_resnet(
    headers: &mut Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    prefix: &str,
) {
    vae_literal_norm(headers, &format!("{prefix}.norm1"));
    vae_literal_norm(headers, &format!("{prefix}.norm2"));
    vae_literal_conv(headers, &format!("{prefix}.conv1"), 32, 32, 3);
    vae_literal_conv(headers, &format!("{prefix}.conv2"), 32, 32, 3);
}
fn vae_literal_mid(
    headers: &mut Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    prefix: &str,
) {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    vae_literal_resnet(headers, &format!("{prefix}.resnets.0"));
    vae_literal_norm(headers, &format!("{prefix}.attentions.0.group_norm"));
    for role in ["to_q", "to_k", "to_v", "to_out.0"] {
        headers.push(hierarchy_header(
            &format!("{prefix}.attentions.0.{role}.weight"),
            &[32, 32],
            Stored::F32,
        ));
        headers.push(hierarchy_header(
            &format!("{prefix}.attentions.0.{role}.bias"),
            &[32],
            Stored::F32,
        ));
    }
    vae_literal_resnet(headers, &format!("{prefix}.resnets.1"));
}
fn vae_literal_headers() -> (
    Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    crate::config::VaeConfig,
) {
    let cfg = crate::config::VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels_out: 8,
        latent_channels_in: 4,
        block_out_channels: vec![32, 32],
        layers_per_block: 1,
        norm_num_groups: 32,
        scaling_factor: 0.13025,
    };
    let mut headers = Vec::new();
    for block in 0..2 {
        vae_literal_resnet(
            &mut headers,
            &format!("encoder.down_blocks.{block}.resnets.0"),
        );
    }
    vae_literal_conv(
        &mut headers,
        "encoder.down_blocks.0.downsamplers.0.conv",
        32,
        32,
        3,
    );
    vae_literal_conv(&mut headers, "encoder.conv_in", 3, 32, 3);
    vae_literal_mid(&mut headers, "encoder.mid_block");
    vae_literal_norm(&mut headers, "encoder.conv_norm_out");
    vae_literal_conv(&mut headers, "encoder.conv_out", 32, 8, 3);
    for block in 0..2 {
        for resnet in 0..2 {
            vae_literal_resnet(
                &mut headers,
                &format!("decoder.up_blocks.{block}.resnets.{resnet}"),
            );
        }
    }
    vae_literal_conv(
        &mut headers,
        "decoder.up_blocks.0.upsamplers.0.conv",
        32,
        32,
        3,
    );
    vae_literal_conv(&mut headers, "decoder.conv_in", 4, 32, 3);
    vae_literal_mid(&mut headers, "decoder.mid_block");
    vae_literal_norm(&mut headers, "decoder.conv_norm_out");
    vae_literal_conv(&mut headers, "decoder.conv_out", 32, 3, 3);
    vae_literal_conv(&mut headers, "quant_conv", 8, 8, 1);
    vae_literal_conv(&mut headers, "post_quant_conv", 4, 4, 1);
    (headers, cfg)
}

#[test]
fn native_free_vae_literal_cast_geometry_halo_and_rgb_host() {
    use crate::unet::pricing::tensor;
    use crate::vae::pricing::AutoencoderFacts;
    let (headers, cfg) = vae_literal_headers();
    for loaded in [Dtype::Float16, Dtype::Float32] {
        let model = AutoencoderFacts::from_headers(&headers, &cfg, loaded).unwrap();
        assert_eq!(model.dtype(), loaded);
        let latents = tensor(&[2, 4, 5, 4], Dtype::Float32).unwrap();
        assert_eq!(
            model.decode_input_retained_bytes(&latents),
            if loaded == Dtype::Float16 { 960 } else { 640 }
        );
        let narrow = tensor(&[2, 4, 5, 4], Dtype::Float16).unwrap();
        assert_eq!(
            model.decode_input_retained_bytes(&narrow),
            320,
            "f32 VAE leaves its narrow caller untouched; f16 identity does not allocate twice"
        );
        for tile in [None, Some(5)] {
            let tiling = tile.map(|edge| mlx_gen::tiling::TilingConfig::spatial_only(edge, 0));
            let mut quanta = Vec::new();
            let decoded = crate::quantum::with_backend(Backend::Cpu, || {
                model.price_decode(&latents, tiling.as_ref(), &mut quanta)
            })
            .unwrap();
            assert_eq!(decoded.shape, [2, 8, 10, 3]);
            assert_eq!(decoded.dtype, Dtype::Float32);
            assert_eq!(
                crate::quantum::decoded_to_image_bytes_from_metadata(decoded.elements()),
                1920 + crate::quantum::PAGE_BYTES,
                "actual B*H*W*C, not pixel-only or single-image count"
            );
            assert_eq!(quanta[0].0, VAE_DECODE_IN);
            if tile.is_some() {
                let attention = quanta.iter().position(|q| q.0 == VAE_ATTENTION).unwrap();
                let first_tile = quanta.iter().position(|q| q.0 == VAE_TILE).unwrap();
                assert!(
                    attention < first_tile,
                    "global mid attention must precede the tiled tail"
                );
                assert!(quanta.iter().any(|q| q.0 == VAE_TILED_STAGE));
            }
        }
        for edge in [64, 5] {
            let mut quanta = Vec::new();
            crate::quantum::with_backend(Backend::Cpu, || {
                model.price_encode(
                    &tensor(&[1, 8, 10, 3], Dtype::Float32).unwrap(),
                    edge,
                    &mut quanta,
                )
            })
            .unwrap();
            assert_eq!(
                quanta[0].0,
                if edge == 64 {
                    VAE_ENCODE_IN
                } else {
                    VAE_ENCODE_TILED_STAGE
                }
            );
            assert_eq!(quanta.last().unwrap().0, VAE_ENCODE_OUT);
        }
    }
    let mut missing = headers.clone();
    missing.retain(|h| {
        h.name != "encoder.down_blocks.0.resnets.0.conv1.weight"
            && h.name != "decoder.conv_in.weight"
    });
    let error = AutoencoderFacts::from_headers(&missing, &cfg, Dtype::Float16)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("encoder.down_blocks.0.resnets.0.conv1.weight"),
        "encoder/down lookup precedes decoder: {error}"
    );
    assert_eq!(crate::vae::max_conv_tile(13, 17, 7), (7, 7));
    // Independent arithmetic of CURRENT5947 source facts. Differences below are logical component
    // correctness, not raw-upper subtraction, fresh backing, residual fitting or physical evidence.
    let tile = [1, 7, 7, 32];
    let weight = [32, 3, 3, 32];
    let raw = |width, normalized| {
        crate::quantum::tile_bytes_from_metadata(
            &tile,
            crate::quantum::TileSource {
                full_width: width,
                normalized,
            },
            2,
            &weight,
            2,
            false,
            Backend::Cpu,
        )
    };
    assert_eq!(
        raw(17, false) - raw(7, false),
        4480,
        "full source row width, not crop width"
    );
    assert_eq!(
        raw(17, true) - raw(17, false),
        28224,
        "global GN/SiLU f32+narrow buffers"
    );
    assert_eq!(
        crate::quantum::tiled_stage_f32_accum_from_metadata(
            &[1, 8, 10, 64],
            &[64, 3, 3, 32],
            18432,
            64,
            2
        ),
        99584,
        "full element counts times4, not already-byte extents"
    );
}

#[test]
fn native_free_vae_encode_decode_tiles_and_rgb_match_actual_cpu() {
    let Some(mut completed) = completed_cpu_child(
        "pricing::native_free_vae_encode_decode_tiles_and_rgb_match_actual_cpu",
    ) else {
        return;
    };
    {
        use crate::vae::{pricing::AutoencoderFacts, Autoencoder};
        use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
        // This comparator targets assembled prospective Root625b +CURRENT native TileSource
        // callers. Own stale Root is not imported or treated as an execution authority.
        let (weights, cfg) = crate::vae::tests::fixture();
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("vae.safetensors");
        save(&weights, &file);
        drop(weights);
        let headers = safetensors_path_tensor_headers(&file).unwrap();
        let check =
            |priced: &crate::quantum::Quanta, events: &[Ev], phase: IosPhase, tiled: bool| {
                let observed: Vec<(String, u64)> = events
                    .iter()
                    .filter_map(|e| match e {
                        Ev::Checked(label, scratch) if ios_quantum_phase(label) == Some(phase) => {
                            Some((label.clone(), *scratch))
                        }
                        _ => None,
                    })
                    .collect();
                let expected: Vec<_> = priced
                    .iter()
                    .map(|(label, scratch)| (label.to_string(), *scratch))
                    .collect();
                if !tiled {
                    assert_eq!(expected, observed, "dense/fallback every ordered guard");
                } else {
                    // Native executes EVERY tile; metadata projects the exact widest crop of each
                    // convolution. Compare complete label maxima plus ordered whole-stage guards;
                    // never pretend one representative tile is a full native window/physical proof.
                    let maxima = |items: &[(String, u64)]| {
                        let mut maxima = std::collections::BTreeMap::new();
                        for (label, scratch) in items {
                            let slot = maxima.entry(label.clone()).or_insert(0u64);
                            *slot = (*slot).max(*scratch);
                        }
                        maxima
                    };
                    assert_eq!(
                        maxima(&expected),
                        maxima(&observed),
                        "all actual tiled guard classes"
                    );
                    let stage_windows = |items: &[(String, u64)]| {
                        let mut windows = Vec::new();
                        let mut current: Option<(String, u64, u64)> = None;
                        for (label, scratch) in items {
                            if [VAE_TILED_STAGE, VAE_ENCODE_TILED_STAGE].contains(&label.as_str()) {
                                if let Some(window) = current.take() {
                                    windows.push(window);
                                }
                                current = Some((label.clone(), *scratch, 0));
                            } else if [VAE_TILE, VAE_ENCODE_TILE].contains(&label.as_str()) {
                                let window = current
                                    .as_mut()
                                    .expect("tile follows its real whole-image stage");
                                window.2 = window.2.max(*scratch);
                            }
                        }
                        if let Some(window) = current {
                            windows.push(window);
                        }
                        windows
                    };
                    assert_eq!(stage_windows(&expected), stage_windows(&observed),
                    "each source-correlated VAE stage and its tile envelope, not a common maximum");
                    for label in [VAE_TILED_STAGE, VAE_ENCODE_TILED_STAGE] {
                        assert_eq!(
                            expected
                                .iter()
                                .filter(|p| p.0 == label)
                                .map(|p| p.1)
                                .collect::<Vec<_>>(),
                            observed
                                .iter()
                                .filter(|p| p.0 == label)
                                .map(|p| p.1)
                                .collect::<Vec<_>>(),
                            "every whole-image/global-stat/halo stage in execution order"
                        );
                    }
                }
            };
        let mut decodes = 0;
        let mut encodes = 0;
        for loaded in [Dtype::Float16, Dtype::Float32] {
            let mut weights = Weights::from_file(&file).unwrap();
            weights.cast_all(loaded).unwrap();
            weights.materialize().unwrap();
            let native = Autoencoder::from_weights(&weights, &cfg).unwrap();
            drop(weights);
            let metadata = AutoencoderFacts::from_headers(&headers, &cfg, loaded).unwrap();
            assert_eq!(native.dtype(), metadata.dtype());
            for input in [Dtype::Float16, Dtype::Float32] {
                let latents = rnd(&[1, 5, 7, 4], 1901, 0.1).as_dtype(input).unwrap();
                assert_eq!(
                    metadata
                        .decode_input_retained_bytes(&crate::quantum::activation_facts(&latents)),
                    native.decode_input_retained_bytes(&latents),
                    "actual input identity versus distinct compute copy"
                );
                for edge in [None, Some(7)] {
                    let tiling =
                        edge.map(|edge| mlx_gen::tiling::TilingConfig::spatial_only(edge, 1));
                    let mut priced = Vec::new();
                    let expected = crate::quantum::with_backend(Backend::Cpu, || {
                        metadata.price_decode(
                            &crate::quantum::activation_facts(&latents),
                            tiling.as_ref(),
                            &mut priced,
                        )
                    })
                    .unwrap();
                    priced.push((
                        DECODED_TO_IMAGE,
                        crate::quantum::decoded_to_image_bytes_from_metadata(expected.elements()),
                    ));
                    let (result, events) = observe(PLENTY, || {
                        let decoded = match &tiling {
                            Some(cfg) => native.decode_tiled(&latents, cfg, None)?,
                            None => native.decode(&latents)?,
                        };
                        // Shape/dtype only, no returned-image pre-eval/readback/pointer or retained
                        // alternate decoded parent before the ORIGINAL NHWC RGB consumer.
                        let shape = decoded.shape().to_vec();
                        let dtype = decoded.dtype();
                        let image = crate::pipeline::decoded_to_image(&decoded)?;
                        Ok((image, shape, dtype))
                    });
                    let (image, shape, dtype) = result.unwrap();
                    assert_eq!(expected.shape, shape);
                    assert_eq!(expected.dtype, dtype);
                    assert_eq!(
                        (image.width, image.height, image.pixels.len()),
                        (14, 10, 14 * 10 * 3)
                    );
                    check(&priced, &events, IosPhase::Decode, edge.is_some());
                    decodes += 1;
                    eprintln!("SDXL_NATIVE_FREE_VAE_DECODE {decodes} loaded={loaded:?} input={input:?} edge={edge:?}");
                }
            }
            let pixels = rnd(&[1, 13, 17, 3], 1902, 0.1);
            for edge in [64, 5] {
                let mut priced = Vec::new();
                crate::quantum::with_backend(Backend::Cpu, || {
                    metadata.price_encode(
                        &crate::quantum::activation_facts(&pixels),
                        edge,
                        &mut priced,
                    )
                })
                .unwrap();
                let (result, events) = observe(PLENTY, || native.encode_tiled(&pixels, edge, None));
                let encoded = result.unwrap();
                assert_eq!(encoded.shape(), [1, 6, 8, 4]);
                assert_eq!(encoded.dtype(), Dtype::Float32);
                assert_eq!(super::bits(&encoded).len(), 6 * 8 * 4);
                check(&priced, &events, IosPhase::Encode, edge == 5);
                encodes += 1;
                eprintln!("SDXL_NATIVE_FREE_VAE_ENCODE {encodes} loaded={loaded:?} edge={edge}");
            }
        }
        assert_eq!((decodes, encodes), (8, 4));
        // Real served Priced init/decode/RGB consumption through existing actual tiny pipeline.
        // Other registered modes/all windows/device admission remain separate SAME-story duties.
        let snapshot = Snapshot::new(2);
        let r = regime(Route::Reference);
        let (_, items) = crate::quantum::with_backend(Backend::Cpu, || {
            Priced::build(&snapshot.inputs(Route::Reference))
                .unwrap()
                .price(&r)
                .unwrap()
        });
        let (result, events) = observe(PLENTY, || {
            let loaded = load(&snapshot, Route::Reference)?;
            run_request(&loaded, Route::Reference, r.tiling.as_ref().unwrap())
        });
        result.unwrap();
        let expected_rgb = items
            .iter()
            .filter(|p| p.label == DECODED_TO_IMAGE)
            .map(|p| p.scratch)
            .max();
        let actual_rgb = events
            .iter()
            .filter_map(|e| match e {
                Ev::Checked(label, scratch) if label == DECODED_TO_IMAGE => Some(*scratch),
                _ => None,
            })
            .max();
        assert!(actual_rgb.is_some());
        assert_eq!(
            expected_rgb, actual_rgb,
            "served RGB input extent must be actual full element count, not a masked phase maximum"
        );
        let expected_decode = items
            .iter()
            .filter(|p| p.phase == IosPhase::Decode)
            .map(|p| (p.label.to_owned(), p.scratch))
            .collect::<Vec<_>>();
        let maxima = |values: &[(String, u64)]| {
            let mut result = std::collections::BTreeMap::new();
            for (label, value) in values {
                let slot = result.entry(label.clone()).or_insert(0u64);
                *slot = (*slot).max(*value);
            }
            result
        };
        let actual_decode = events
            .iter()
            .filter_map(|e| match e {
                Ev::Checked(label, scratch)
                    if ios_quantum_phase(label) == Some(IosPhase::Decode) =>
                {
                    Some((label.clone(), *scratch))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            maxima(&expected_decode),
            maxima(&actual_decode),
            "served VAE/encode-init request decode and RGB consumers"
        );
        eprintln!("SDXL_NATIVE_FREE_VAE_MECHANISM 0 eight_decode_four_encode_and_served_reference_not_physical_evidence");
    }
    completed.body_completed = true;
}

fn ip_literal_headers() -> (
    Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    crate::vision_encoder::VisionConfig,
    crate::ip_adapter::ResamplerConfig,
) {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    let vision = crate::vision_encoder::VisionConfig {
        hidden: 64,
        num_layers: 2,
        num_heads: 1,
        patch: 4,
        image_size: 8,
        num_channels: 3,
        quick_gelu: false,
    };
    let resampler = crate::ip_adapter::ResamplerConfig {
        dim: 32,
        depth: 1,
        heads: 2,
        dim_head: 16,
        num_queries: 4,
        embed_dim: 64,
        output_dim: 48,
    };
    let mut encoder = Vec::new();
    let mut adapter = Vec::new();
    let put = |headers: &mut Vec<_>, name: &str, shape: &[usize]| {
        headers.push(hierarchy_header(name, shape, Stored::F32))
    };
    put(
        &mut encoder,
        "vision_model.embeddings.patch_embedding.weight",
        &[64, 3, 4, 4],
    );
    put(
        &mut encoder,
        "vision_model.embeddings.class_embedding",
        &[64],
    );
    put(
        &mut encoder,
        "vision_model.embeddings.position_embedding.weight",
        &[5, 64],
    );
    for part in ["weight", "bias"] {
        put(
            &mut encoder,
            &format!("vision_model.pre_layrnorm.{part}"),
            &[64],
        );
    }
    for index in 0..2 {
        let p = format!("vision_model.encoder.layers.{index}");
        for norm in ["layer_norm1", "layer_norm2"] {
            for part in ["weight", "bias"] {
                put(&mut encoder, &format!("{p}.{norm}.{part}"), &[64]);
            }
        }
        for (role, output, input) in [
            ("self_attn.q_proj", 64, 64),
            ("self_attn.k_proj", 64, 64),
            ("self_attn.v_proj", 64, 64),
            ("self_attn.out_proj", 64, 64),
            ("mlp.fc1", 192, 64),
            ("mlp.fc2", 64, 192),
        ] {
            put(
                &mut encoder,
                &format!("{p}.{role}.weight"),
                &[output, input],
            );
            put(&mut encoder, &format!("{p}.{role}.bias"), &[output]);
        }
    }
    put(&mut adapter, "image_proj.latents", &[1, 4, 32]);
    for (name, output, input) in [("proj_in", 32, 64), ("proj_out", 48, 32)] {
        put(
            &mut adapter,
            &format!("image_proj.{name}.weight"),
            &[output, input],
        );
        put(&mut adapter, &format!("image_proj.{name}.bias"), &[output]);
    }
    for part in ["weight", "bias"] {
        put(&mut adapter, &format!("image_proj.norm_out.{part}"), &[48]);
    }
    for norm in ["norm1", "norm2"] {
        for part in ["weight", "bias"] {
            put(
                &mut adapter,
                &format!("image_proj.layers.0.0.{norm}.{part}"),
                &[32],
            );
        }
    }
    for (name, output, input) in [("to_q", 32, 32), ("to_kv", 64, 32), ("to_out", 32, 32)] {
        put(
            &mut adapter,
            &format!("image_proj.layers.0.0.{name}.weight"),
            &[output, input],
        );
    }
    for part in ["weight", "bias"] {
        put(
            &mut adapter,
            &format!("image_proj.layers.0.1.0.{part}"),
            &[32],
        );
    }
    put(&mut adapter, "image_proj.layers.0.1.1.weight", &[128, 32]);
    put(&mut adapter, "image_proj.layers.0.1.3.weight", &[32, 128]);
    (encoder, adapter, vision, resampler)
}

#[test]
fn native_free_ip_literal_headers_cfg_and_source_geometry_host() {
    use crate::ip_adapter::pricing::{IpImageEncoderFacts, ResamplerFacts, VisionFacts};
    use crate::unet::pricing::HeaderWeights;
    let (encoder, adapter, vision_cfg, resampler_cfg) = ip_literal_headers();
    let vision =
        VisionFacts::from_weights(&HeaderWeights::new(&encoder).unwrap(), &vision_cfg).unwrap();
    let resampler = ResamplerFacts::from_weights(
        &HeaderWeights::new(&adapter).unwrap(),
        "image_proj",
        &resampler_cfg,
    )
    .unwrap();
    let model = IpImageEncoderFacts::with_image_size(vision, resampler, 8);
    let mut quanta = Vec::new();
    let tokens =
        crate::quantum::with_backend(Backend::Cpu, || model.price_tokens(&mut quanta)).unwrap();
    assert_eq!(tokens.shape, [1, 4, 48]);
    assert_eq!(tokens.dtype, Dtype::Float16);
    assert_eq!(
        quanta.iter().map(|q| q.0).collect::<Vec<_>>(),
        [
            VISION_EMBEDDINGS,
            VISION_LAYER,
            RESAMPLER_PROJ_IN,
            RESAMPLER_LAYER,
            RESAMPLER_PROJ_OUT
        ],
        "penultimate stops before final vision layer; real resampler follows in order"
    );
    let paired = crate::ip_adapter::pricing::cfg_pair(&tokens, true).unwrap();
    assert_eq!(paired.shape, [2, 4, 48]);
    assert_eq!(paired.dtype, Dtype::Float16);
    assert_eq!(
        crate::quantum::ip_cfg_tokens_bytes_metadata(&tokens),
        1920 + 3 * crate::quantum::PAGE_BYTES
    );
    assert_eq!(
        crate::ip_adapter::pricing::cfg_pair(&tokens, false).unwrap(),
        tokens
    );
    let mut missing = encoder.clone();
    missing.retain(|h| {
        h.name != "vision_model.pre_layrnorm.bias"
            && !h.name.contains("encoder.layers.0.layer_norm1.weight")
    });
    let error = VisionFacts::from_weights(&HeaderWeights::new(&missing).unwrap(), &vision_cfg)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("pre_layrnorm.bias"),
        "native pre-LN lookup before layer ctor: {error}"
    );
    let mut wrong = adapter.clone();
    wrong
        .iter_mut()
        .find(|h| h.name == "image_proj.latents")
        .unwrap()
        .shape = vec![1, 3, 32];
    let invalid = ResamplerFacts::from_weights(
        &HeaderWeights::new(&wrong).unwrap(),
        "image_proj",
        &resampler_cfg,
    )
    .unwrap();
    assert!(invalid
        .price_forward(
            &crate::unet::pricing::tensor(&[1, 5, 64], Dtype::Float16).unwrap(),
            &mut Vec::new()
        )
        .unwrap_err()
        .to_string()
        .contains("resampler latents shape"));
    let geometry = crate::ip_adapter::preprocess_geometry(20, 18, 1080, 16).unwrap();
    assert_eq!((geometry.resized_width, geometry.resized_height), (18, 16));
    assert_eq!(geometry.source_buffer_bytes, 1080);
    assert!(
        geometry.host_requested_upper().unwrap() >= 18 * 18 * 3 * 4 + 18 * 16 * 3 * 4,
        "horizontal+resized images and coefficient tables, not crop-only bytes"
    );
    let tall = crate::ip_adapter::preprocess_geometry(1, 1000, 3000, 16).unwrap();
    assert_eq!((tall.resized_width, tall.resized_height), (16, 16000));
    assert!(tall.host_requested_upper().unwrap() > geometry.host_requested_upper().unwrap());
    assert!(crate::ip_adapter::preprocess_geometry(20, 18, 1079, 16)
        .unwrap_err()
        .to_string()
        .contains("image buffer"));
    assert!(crate::ip_adapter::preprocess_geometry(0, 18, 0, 16)
        .unwrap_err()
        .to_string()
        .contains("zero dimension"));
}

#[test]
fn ip_reference_work_sidecar_and_scalar_policy_preserve_actual_context_host() {
    use crate::sampling::{
        ReferenceImageFacts, ResolvedSampling, SamplingRecipe, SamplingWorkFacts,
    };
    use mlx_gen::Conditioning;
    let mut request = generation_request(Route::IpAdapter, 64, 64, &Default::default());
    request.strength = Some(0.4);
    if let Conditioning::Reference { image, strength } = &mut request.conditioning[0] {
        *image = mlx_gen::media::Image {
            width: 1,
            height: 1000,
            pixels: vec![7; 3000],
        };
        *strength = Some(1.7);
    }
    let facts = SamplingWorkFacts::from_request(&request);
    assert_eq!(
        facts.reference_image,
        Some(ReferenceImageFacts {
            width: 1,
            height: 1000,
            buffer_bytes: 3000
        })
    );
    assert_eq!(
        SamplingWorkFacts::from_value(facts.to_value()).unwrap(),
        facts
    );
    facts.validate_actual(&request).unwrap();
    let recipe = SamplingRecipe::from_request(&request).unwrap();
    let allocation = ResolvedSampling::resolve(&request)
        .unwrap()
        .allocation_class();
    assert_eq!(
        ResolvedSampling::resolve(&request)
            .unwrap()
            .effective_ip_scale(),
        1.7,
        "IP scalar is not img2img clamped"
    );
    assert_eq!(crate::model::ip_reference_scale(Some(0.0)), 0.0);
    assert_eq!(crate::model::ip_reference_scale(None), 0.6);
    if let Conditioning::Reference { strength, .. } = &mut request.conditioning[0] {
        *strength = None;
    }
    assert_eq!(
        ResolvedSampling::resolve(&request)
            .unwrap()
            .effective_ip_scale(),
        0.4,
        "missing per-reference override uses actual request strength, not IP default or init clamp"
    );
    if let Conditioning::Reference { strength, .. } = &mut request.conditioning[0] {
        *strength = Some(1.7);
    }
    if let Conditioning::Reference { image, strength } = &mut request.conditioning[0] {
        *image = mlx_gen::media::Image {
            width: 1000,
            height: 1,
            pixels: vec![7; 3000],
        };
        *strength = Some(1.7);
    }
    assert!(
        facts.validate_actual(&request).is_err(),
        "same span/different actual source aspect must not borrow work facts"
    );
    assert_eq!(
        recipe.canonical_token(),
        SamplingRecipe::from_request(&request)
            .unwrap()
            .canonical_token()
    );
    assert_eq!(
        allocation,
        ResolvedSampling::resolve(&request)
            .unwrap()
            .allocation_class(),
        "variable source geometry is exact work/key identity, not a per-aspect structural family"
    );
    if let Conditioning::Reference { image, .. } = &mut request.conditioning[0] {
        image.pixels.truncate(1);
    }
    let malformed = SamplingWorkFacts::from_request(&request);
    assert_eq!(
        malformed.reference_image.unwrap().buffer_bytes,
        1,
        "actual unvalidated len, never width*height*3"
    );
    let controls = mlx_gen::GenerationRequest {
        conditioning: vec![
            Conditioning::Control {
                image: Default::default(),
                kind: mlx_gen::gen_core::ControlKind::Canny,
                scale: Some(0.0),
            },
            Conditioning::Control {
                image: Default::default(),
                kind: mlx_gen::gen_core::ControlKind::Canny,
                scale: Some(0.25),
            },
            Conditioning::Control {
                image: Default::default(),
                kind: mlx_gen::gen_core::ControlKind::Canny,
                scale: None,
            },
        ],
        ..Default::default()
    };
    assert_eq!(
        crate::model::control_inputs(&controls)
            .unwrap()
            .iter()
            .map(|(_, s)| *s)
            .collect::<Vec<_>>(),
        [0.0, 0.25, 1.0]
    );
    for (mask, count, acceleration) in [
        (false, 0, false),
        (true, 0, false),
        (false, 1, false),
        (false, 0, true),
    ] {
        assert_eq!(
            crate::model::uses_ip_reference(true, true, mask, count, acceleration),
            !mask && count == 0 && !acceleration
        );
    }
    assert!(
        crate::model::validate_image_roles("lightning", true, false, true, false)
            .unwrap_err()
            .to_string()
            .contains("txt2img-only")
    );
    assert!(
        crate::model::validate_control_roles("euler_ancestral", Some(7.0), 1, 1, true)
            .unwrap_err()
            .to_string()
            .contains("inpaint Mask")
    );
}

#[test]
fn native_free_ip_tokens_cfg_and_actual_encoder_release_match_cpu() {
    let Some(mut completed) = completed_cpu_child(
        "pricing::native_free_ip_tokens_cfg_and_actual_encoder_release_match_cpu",
    ) else {
        return;
    };
    {
        use crate::ip_adapter::pricing::{IpImageEncoderFacts, ResamplerFacts, VisionFacts};
        use crate::unet::pricing::HeaderWeights;
        use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
        let mut rows = 0;
        for tier in [None, Some(4), Some(8)] {
            let snapshot = match tier {
                None => Snapshot::new(2),
                Some(bits) => Snapshot::packed(bits),
            };
            let inputs = snapshot.inputs(Route::IpAdapter);
            let (encoder_file, adapter_file) = inputs.ip.as_ref().unwrap();
            let encoder_headers = safetensors_path_tensor_headers(encoder_file).unwrap();
            let adapter_headers = safetensors_path_tensor_headers(adapter_file).unwrap();
            let metadata = IpImageEncoderFacts::with_image_size(
                VisionFacts::from_weights(
                    &HeaderWeights::new(&encoder_headers).unwrap(),
                    &snapshot.vision,
                )
                .unwrap(),
                ResamplerFacts::from_weights(
                    &HeaderWeights::new(&adapter_headers).unwrap(),
                    "image_proj",
                    &snapshot.resampler,
                )
                .unwrap(),
                snapshot.vision.image_size as usize,
            );
            let (result, _) = observe(PLENTY, || load(&snapshot, Route::IpAdapter));
            let loaded = result.unwrap();
            let (text, _) = observe(PLENTY, || {
                crate::pipeline::encode_conditioning(&loaded.te1, &loaded.te2, &ids())
            });
            let (conditioning, pooled) = text.unwrap();
            let heavy = crate::model::SdxlHeavyOwned::with_controls_for_test(
                loaded.unet,
                loaded.controls,
                loaded.vae,
                loaded.ip,
            );
            assert!(heavy.has_ip_adapter_for_test());
            let reference = image(20, 18, 1);
            let mut first = None;
            for (cfg, shed) in [(false, false), (true, false), (true, true)] {
                let mut priced = Vec::new();
                let single = crate::quantum::with_backend(Backend::Cpu, || {
                    metadata.price_tokens(&mut priced)
                })
                .unwrap();
                if cfg {
                    priced.push((
                        IP_CFG_TOKENS,
                        crate::quantum::ip_cfg_tokens_bytes_metadata(&single),
                    ));
                }
                let expected = crate::ip_adapter::pricing::cfg_pair(&single, cfg).unwrap();
                let (result, events) =
                    observe(PLENTY, || heavy.ip_tokens_for_test(&reference, cfg, shed));
                let actual = result.unwrap();
                assert_eq!(actual.shape(), expected.shape);
                assert_eq!(actual.dtype(), expected.dtype);
                let checked = events
                    .iter()
                    .filter_map(|e| match e {
                        Ev::Checked(label, scratch) => Some((label.as_str(), *scratch)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(priced,checked,"{tier:?} cfg={cfg} shed={shed}: every actual vision/resampler/CFG guard, not maxima");
                let values = super::bits(&actual);
                let row_len =
                    (snapshot.resampler.num_queries * snapshot.resampler.output_dim) as usize;
                if cfg {
                    assert!(
                        values[row_len..].iter().all(|v| *v == 0),
                        "native CFG uncond is actual zeros"
                    );
                    assert_eq!(first.as_ref().unwrap(), &values[..row_len]);
                } else {
                    first = Some(values.clone());
                }
                assert_eq!(
                    heavy.has_ip_adapter_for_test(),
                    !shed,
                    "materialize-before-drop real encoder+Resampler slot"
                );
                if shed {
                    // Actual installed K/V branch still executes AFTER image encoder/Resampler
                    // release. This is runtime liveness, never a physical backing/release proof.
                    let euler = crate::sampler::EulerSampler::new_with_dtype(
                        &crate::config::DiffusionConfig::sdxl_base(),
                        true,
                        crate::model::DTYPE,
                    )
                    .unwrap();
                    let sampler = crate::sampler::AncestralEuler::new(&euler, 2, 800.0).unwrap();
                    let plan = crate::plan::SdxlForwardPlan::UNBOUNDED
                        .with_cfg_batching(CfgBatching::Sequential);
                    let denoiser = crate::pipeline::Denoiser::with_plan(
                        heavy.parts_for_test().0,
                        &sampler,
                        plan,
                    );
                    let latents = euler.sample_prior(&[1, LATENT, LATENT, 4]).unwrap();
                    let cancel = mlx_gen::CancelFlag::default();
                    let (output, _) = observe(PLENTY, || {
                        crate::pipeline::denoise_ip(
                            &denoiser,
                            latents,
                            &conditioning,
                            &pooled,
                            &crate::pipeline::text_time_ids(2),
                            5.0,
                            &cancel,
                            &mut |_| {},
                            &actual,
                            0.6,
                        )
                    });
                    assert_eq!(
                        super::bits(&output.unwrap()).len(),
                        (LATENT * LATENT * 4) as usize
                    );
                    assert!(!heavy.has_ip_adapter_for_test());
                }
                rows += 1;
            }
            let r = regime(Route::IpAdapter);
            let (pricing, items) = crate::quantum::with_backend(Backend::Cpu, || {
                Priced::build(&inputs).unwrap().price(&r).unwrap()
            });
            let host = items
                .iter()
                .filter(|q| q.label == "SDXL IP preprocess host logical")
                .collect::<Vec<_>>();
            assert_eq!(host.len(), 1);
            assert!(
                pricing.windows.iter().any(|w| w.owner_domain
                    == "sdxl/ip/preprocess-host-requested-logical-v1"
                    && w.class.is_none()),
                "host requested work is explicit, not a native physical certificate"
            );
            let facts = crate::sampling::ReferenceImageFacts::from_image(&reference);
            let geometry = metadata.preprocess_geometry(facts).unwrap();
            assert_eq!(host[0].scratch, geometry.host_requested_upper().unwrap());
            eprintln!(
                "SDXL_NATIVE_FREE_IP_TIER {tier:?} actualencoderreleased_and_installedKVstillruns"
            );
        }
        assert_eq!(rows, 9);
        eprintln!("SDXL_NATIVE_FREE_IP_MECHANISM 0 nine_synthetic_token_cfg_release_rows_not_physical_evidence");
    }
    completed.body_completed = true;
}

fn fused_literal_headers() -> Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader> {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    vec![
        hierarchy_header(
            "model.diffusion_model.time_embed.0.weight",
            &[4, 4],
            Stored::F32,
        ),
        hierarchy_header(
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            &[3, 4],
            Stored::F16,
        ),
        hierarchy_header(
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
            &[6, 4],
            Stored::F32,
        ),
        hierarchy_header(
            "conditioner.embedders.1.model.text_projection",
            &[4, 2],
            Stored::F32,
        ),
        hierarchy_header(
            "first_stage_model.decoder.mid.attn_1.q.weight",
            &[4, 4, 1, 1],
            Stored::F16,
        ),
    ]
}

#[test]
fn fused_full_split_literals_preserve_correlated_overwrite_and_raw_floor_host() {
    use crate::ldm::pricing::{
        order_invariant_projection, split_in_exact_source_order, OrderInvariantProjection,
    };
    use mlx_gen::gen_core::sdxl_ldm::SdxlComponent as C;
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    let mut headers = fused_literal_headers();
    let whole = split_in_exact_source_order(&headers, &[0, 1, 2, 3, 4]).unwrap();
    assert_eq!(whole.unet["time_embedding.linear_1.weight"].shape, [4, 4]);
    assert_eq!(whole.clip_bigg["text_projection.weight"].shape, [2, 4]);
    for role in ["q_proj", "k_proj", "v_proj"] {
        assert_eq!(
            whole.clip_bigg[&format!("text_model.encoder.layers.0.self_attn.{role}.weight")].shape,
            [2, 4]
        );
    }
    assert_eq!(
        whole.vae["decoder.mid_block.attentions.0.to_q.weight"].shape,
        [4, 4]
    );
    // 24 ClipL +48 castQKV +16 castprojection, largestQKV source96+cast48=144.
    assert_eq!(
        crate::ldm::component_load_floor_from_headers(&headers, &[C::ClipL, C::ClipBigG]).unwrap(),
        232
    );
    headers.push(hierarchy_header(
        "conditioner.embedders.1.model.transformer.resblocks.0.attn.q_proj.weight",
        &[3, 4],
        Stored::F32,
    ));
    let a = split_in_exact_source_order(&headers, &[0, 1, 2, 3, 4, 5]).unwrap();
    let b = split_in_exact_source_order(&headers, &[0, 1, 5, 3, 4, 2]).unwrap();
    assert_eq!(
        a.clip_bigg["text_model.encoder.layers.0.self_attn.q_proj.weight"].shape,
        [3, 4]
    );
    assert_eq!(
        b.clip_bigg["text_model.encoder.layers.0.self_attn.q_proj.weight"].shape,
        [2, 4]
    );
    // One split source overwrites Q/K/V jointly. No independent role maximum/world invented.
    for role in ["k_proj", "v_proj"] {
        assert_eq!(
            a.clip_bigg[&format!("text_model.encoder.layers.0.self_attn.{role}.weight")].shape,
            [2, 4]
        );
        assert_eq!(
            b.clip_bigg[&format!("text_model.encoder.layers.0.self_attn.{role}.weight")].shape,
            [2, 4]
        );
    }
    assert!(matches!(order_invariant_projection(&headers).unwrap(),OrderInvariantProjection::NeedsActualOrder{..}),
        "non-equivalent aliases are an unproved source-order dependency, never fake chosen numeric price");
    // Raw source alias is STILL in the actual pre-split floor: +24 cast, largest unchanged144.
    assert_eq!(
        crate::ldm::component_load_floor_from_headers(&headers, &[C::ClipL, C::ClipBigG]).unwrap(),
        256
    );
    let mut malformed = fused_literal_headers();
    malformed[2].shape[0] = 7;
    assert!(split_in_exact_source_order(&malformed, &[0, 1, 2, 3, 4])
        .err()
        .unwrap()
        .to_string()
        .contains("split evenly"));
    malformed[2] = fused_literal_headers()[2].clone();
    malformed[4].name = "ignored_only".into();
    assert!(split_in_exact_source_order(&malformed, &[0, 1, 2, 3, 4])
        .err()
        .unwrap()
        .to_string()
        .contains("missing the VAE"));
}

#[test]
fn fused_split_and_actual_selected_raw_load_guard_match_cpu() {
    let Some(mut completed) =
        completed_cpu_child("pricing::fused_split_and_actual_selected_raw_load_guard_match_cpu")
    else {
        return;
    };
    {
        use crate::ldm::pricing::{order_invariant_projection, OrderInvariantProjection};
        use mlx_gen::gen_core::sdxl_ldm::SdxlComponent as C;
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("fused.safetensors");
        let headers = fused_literal_headers();
        let mut weights = Weights::empty();
        for (index, header) in headers.iter().enumerate() {
            let shape = header.shape.iter().map(|&d| d as i32).collect::<Vec<_>>();
            let value = rnd(&shape, 2200 + index as u64, 0.1)
                .as_dtype(if index == 1 || index == 4 {
                    Dtype::Float16
                } else {
                    Dtype::Float32
                })
                .unwrap();
            weights.insert(header.name.clone(), value);
        }
        save(&weights, &file);
        drop(weights);
        let source =
            mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&file).unwrap();
        let metadata = match order_invariant_projection(&source).unwrap() {
            OrderInvariantProjection::Resolved(v) => v,
            _ => panic!("fixture has pricing-invariant map"),
        };
        let (native, events) = observe(PLENTY, || {
            crate::ldm::before_component_load(
                crate::quantum::LDM_TEXT_LOAD,
                &file,
                &[C::ClipL, C::ClipBigG],
            )?;
            crate::ldm::split_ldm_checkpoint(&file)
        });
        let native = native.unwrap();
        for (expected, actual) in [
            (&metadata.unet, &native.unet),
            (&metadata.clip_l, &native.clip_l),
            (&metadata.clip_bigg, &native.clip_bigg),
            (&metadata.vae, &native.vae),
        ] {
            assert_eq!(expected.len(), actual.keys().count());
            for (name, header) in expected {
                let array = actual.require(name).unwrap();
                assert_eq!(
                    array.shape(),
                    header.shape.iter().map(|&d| d as i32).collect::<Vec<_>>()
                );
                assert_eq!(
                    array.dtype(),
                    crate::quant::header_dtype(header.dtype).unwrap()
                );
                assert_eq!(
                    super::bits(array).len(),
                    header.shape.iter().product::<usize>()
                );
            }
        }
        let observed = events
            .iter()
            .filter_map(|e| match e {
                Ev::Checked(label, bytes) if label == LDM_TEXT_LOAD => Some(*bytes),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            [
                crate::ldm::component_load_floor_from_headers(&source, &[C::ClipL, C::ClipBigG])
                    .unwrap()
            ]
        );
        eprintln!("SDXL_FUSED_FULL_SPLIT_MECHANISM 0 actual_native_parser_remapper_and_raw_floor_not_fullmodel_or_orderwitness");
    }
    completed.body_completed = true;
}

// Header-only connected Fused fixture. No Array factory/model/native loader is used here.
fn fused_skeleton_unet_headers() -> (
    Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    UNetConfig,
) {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    let mut cfg = UNetConfig::sdxl_base();
    cfg.block_out_channels = vec![64];
    cfg.layers_per_block = vec![1];
    cfg.transformer_layers_per_block = vec![1];
    cfg.num_attention_heads = vec![1];
    cfg.cross_attention_dim = vec![128];
    cfg.down_block_types = vec!["DownBlock2D".into()];
    cfg.up_block_types = vec!["UpBlock2D".into()];
    cfg.addition_time_embed_dim = Some(8);
    cfg.projection_class_embeddings_input_dim = Some(64);
    let mut result = Vec::new();
    let put = |out: &mut Vec<_>, name: String, shape: &[usize]| {
        out.push(hierarchy_header(&name, shape, Stored::F32))
    };
    for (name, output, input) in [
        ("time_embedding.linear_1", 256, 64),
        ("time_embedding.linear_2", 256, 256),
        ("add_embedding.linear_1", 256, 64),
        ("add_embedding.linear_2", 256, 256),
    ] {
        put(&mut result, format!("{name}.weight"), &[output, input]);
        put(&mut result, format!("{name}.bias"), &[output]);
    }
    for (prefix, input) in [
        ("down_blocks.0.resnets.0", 64),
        ("mid_block.resnets.0", 64),
        ("mid_block.resnets.1", 64),
        ("up_blocks.0.resnets.0", 128),
        ("up_blocks.0.resnets.1", 128),
    ] {
        for (name, channels) in [("norm1", input), ("norm2", 64)] {
            for part in ["weight", "bias"] {
                put(&mut result, format!("{prefix}.{name}.{part}"), &[channels]);
            }
        }
        for (name, channels) in [("conv1", input), ("conv2", 64)] {
            put(
                &mut result,
                format!("{prefix}.{name}.weight"),
                &[64, channels, 3, 3],
            );
            put(&mut result, format!("{prefix}.{name}.bias"), &[64]);
        }
        put(
            &mut result,
            format!("{prefix}.time_emb_proj.weight"),
            &[64, 256],
        );
        put(&mut result, format!("{prefix}.time_emb_proj.bias"), &[64]);
        if input != 64 {
            put(
                &mut result,
                format!("{prefix}.conv_shortcut.weight"),
                &[64, input, 1, 1],
            );
            put(&mut result, format!("{prefix}.conv_shortcut.bias"), &[64]);
        }
    }
    for (prefix, output, input) in [("conv_in", 64, 4), ("conv_out", 4, 64)] {
        put(
            &mut result,
            format!("{prefix}.weight"),
            &[output, input, 3, 3],
        );
        put(&mut result, format!("{prefix}.bias"), &[output]);
    }
    for part in ["weight", "bias"] {
        put(&mut result, format!("conv_norm_out.{part}"), &[64]);
    }
    let prefix = "mid_block.attentions.0";
    for part in ["weight", "bias"] {
        put(&mut result, format!("{prefix}.norm.{part}"), &[64]);
    }
    for name in ["proj_in", "proj_out"] {
        put(&mut result, format!("{prefix}.{name}.weight"), &[64, 64]);
        put(&mut result, format!("{prefix}.{name}.bias"), &[64]);
    }
    let prefix = "mid_block.attentions.0.transformer_blocks.0";
    for name in ["norm1", "norm2", "norm3"] {
        for part in ["weight", "bias"] {
            put(&mut result, format!("{prefix}.{name}.{part}"), &[64]);
        }
    }
    for attention in ["attn1", "attn2"] {
        for role in ["to_q", "to_k", "to_v"] {
            let input = if attention == "attn2" && role != "to_q" {
                128
            } else {
                64
            };
            put(
                &mut result,
                format!("{prefix}.{attention}.{role}.weight"),
                &[64, input],
            );
        }
        put(
            &mut result,
            format!("{prefix}.{attention}.to_out.0.weight"),
            &[64, 64],
        );
        put(
            &mut result,
            format!("{prefix}.{attention}.to_out.0.bias"),
            &[64],
        );
    }
    for (name, output, input) in [("ff.net.0.proj", 384, 64), ("ff.net.2", 64, 192)] {
        put(
            &mut result,
            format!("{prefix}.{name}.weight"),
            &[output, input],
        );
        put(&mut result, format!("{prefix}.{name}.bias"), &[output]);
    }
    (result, cfg)
}

fn fused_skeleton_headers() -> (
    Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader>,
    Architecture,
) {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    let (unet, cfg) = fused_skeleton_unet_headers();
    let (clip, clip_cfg) = clip_literal_headers();
    let (vae, vae_cfg) = vae_literal_headers();
    let mut arch = Architecture::sdxl();
    arch.unet = cfg;
    arch.vae = vae_cfg;
    arch.te1 = clip_cfg.clone();
    arch.te1.projection_dim = None;
    arch.te2 = clip_cfg;
    arch.text_tokens = 8;
    arch.context_dim = 128;
    arch.latent_scale = 2;
    let mut raw = Vec::new();
    for mut header in unet {
        let mut name = None;
        for (to, from) in [
            ("time_embedding.linear_1.", "time_embed.0."),
            ("time_embedding.linear_2.", "time_embed.2."),
            ("add_embedding.linear_1.", "label_emb.0.0."),
            ("add_embedding.linear_2.", "label_emb.0.2."),
            ("conv_in.", "input_blocks.0.0."),
            ("conv_norm_out.", "out.0."),
            ("conv_out.", "out.2."),
            ("down_blocks.0.resnets.0.", "input_blocks.1.0."),
            ("mid_block.resnets.0.", "middle_block.0."),
            ("mid_block.resnets.1.", "middle_block.2."),
            ("mid_block.attentions.0.", "middle_block.1."),
            ("up_blocks.0.resnets.0.", "output_blocks.0.0."),
            ("up_blocks.0.resnets.1.", "output_blocks.1.0."),
        ] {
            if let Some(rest) = header.name.strip_prefix(to) {
                name = Some(format!("model.diffusion_model.{from}{rest}"));
                break;
            }
        }
        header.name = name.expect("fixture inventory has an actual supported LDM namespace");
        raw.push(header);
    }
    for header in &clip {
        if header.name != "text_projection.weight" {
            let mut one = header.clone();
            one.name = format!("conditioner.embedders.0.transformer.{}", one.name);
            raw.push(one);
        }
        // Exercise the actual correlated source QKV split, rather than three synthetic leaves.
        if header.name.contains(".self_attn.k_proj.") || header.name.contains(".self_attn.v_proj.")
        {
            continue;
        }
        let mut two = header.clone();
        if two.name == "text_projection.weight" {
            two.name = "conditioner.embedders.1.model.text_projection".into();
            two.shape.reverse();
        } else if let Some(rest) = two.name.strip_prefix("text_model.encoder.layers.") {
            let rest = rest
                .replace("layer_norm1", "ln_1")
                .replace("layer_norm2", "ln_2")
                .replace("self_attn.q_proj.weight", "attn.in_proj_weight")
                .replace("self_attn.q_proj.bias", "attn.in_proj_bias")
                .replace("self_attn.", "attn.")
                .replace("mlp.fc1.", "mlp.c_fc.")
                .replace("mlp.fc2.", "mlp.c_proj.");
            two.name = format!("conditioner.embedders.1.model.transformer.resblocks.{rest}");
            if two.name.ends_with("in_proj_weight") || two.name.ends_with("in_proj_bias") {
                two.shape[0] *= 3;
                two.data_bytes *= 3;
            }
        } else {
            two.name = match two.name.as_str() {
                "text_model.embeddings.token_embedding.weight" => {
                    "conditioner.embedders.1.model.token_embedding.weight".into()
                }
                "text_model.embeddings.position_embedding.weight" => {
                    "conditioner.embedders.1.model.positional_embedding".into()
                }
                "text_model.final_layer_norm.weight" => {
                    "conditioner.embedders.1.model.ln_final.weight".into()
                }
                "text_model.final_layer_norm.bias" => {
                    "conditioner.embedders.1.model.ln_final.bias".into()
                }
                _ => panic!("unmapped fixture CLIP key"),
            };
        }
        raw.push(two);
    }
    for mut header in vae {
        let key = header.name.clone();
        let mut name = None;
        for side in ["encoder", "decoder"] {
            for (to, from) in [
                (
                    format!("{side}.conv_norm_out."),
                    format!("{side}.norm_out."),
                ),
                (
                    format!("{side}.mid_block.resnets.0."),
                    format!("{side}.mid.block_1."),
                ),
                (
                    format!("{side}.mid_block.resnets.1."),
                    format!("{side}.mid.block_2."),
                ),
            ] {
                if let Some(rest) = key.strip_prefix(&to) {
                    name = Some(format!("{from}{rest}"));
                }
            }
            if let Some(rest) = key.strip_prefix(&format!("{side}.mid_block.attentions.0.")) {
                let rest = rest
                    .replace("group_norm.", "norm.")
                    .replace("to_q.", "q.")
                    .replace("to_k.", "k.")
                    .replace("to_v.", "v.")
                    .replace("to_out.0.", "proj_out.");
                name = Some(format!("{side}.mid.attn_1.{rest}"));
                if header.shape.len() == 2 {
                    header.shape.extend([1, 1]);
                }
            }
        }
        for block in 0..2 {
            if let Some(rest) = key.strip_prefix(&format!("encoder.down_blocks.{block}.resnets.")) {
                name = Some(format!("encoder.down.{block}.block.{rest}"));
            }
            if let Some(rest) =
                key.strip_prefix(&format!("encoder.down_blocks.{block}.downsamplers.0.conv."))
            {
                name = Some(format!("encoder.down.{block}.downsample.conv.{rest}"));
            }
            if let Some(rest) = key.strip_prefix(&format!("decoder.up_blocks.{block}.resnets.")) {
                name = Some(format!("decoder.up.{}.block.{rest}", 3 - block));
            }
            if let Some(rest) =
                key.strip_prefix(&format!("decoder.up_blocks.{block}.upsamplers.0.conv."))
            {
                name = Some(format!("decoder.up.{}.upsample.conv.{rest}", 3 - block));
            }
        }
        header.name = format!("first_stage_model.{}", name.unwrap_or(key));
        raw.push(header);
    }
    // A real separately named alias with equal target facts; both raw parents must remain priced.
    raw.push(hierarchy_header(
        "conditioner.embedders.1.model.transformer.resblocks.0.attn.q_proj.weight",
        &[64, 64],
        Stored::F32,
    ));
    (raw, arch)
}

fn write_header_only_sparse(
    file: &std::path::Path,
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
) {
    use std::io::Write as _;
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    let mut map = serde_json::Map::new();
    let mut offset = 0u64;
    for header in headers {
        let next = offset.checked_add(header.data_bytes).unwrap();
        assert!(map
            .insert(
                header.name.clone(),
                serde_json::json!({"dtype":format!("{:?}",header.dtype),
            "shape":header.shape,"data_offsets":[offset,next]})
            )
            .is_none());
        offset = next;
    }
    let mut text = serde_json::to_vec(&map).unwrap();
    while !text.len().is_multiple_of(8) {
        text.push(b' ');
    }
    let mut output = std::fs::File::create(file).unwrap();
    output
        .write_all(&(text.len() as u64).to_le_bytes())
        .unwrap();
    output.write_all(&text).unwrap();
    // Sparse span only: no tensor payload is manufactured, read, mapped by MLX or evaluated.
    output.set_len(8 + text.len() as u64 + offset).unwrap();
}

#[test]
fn fused_text_alias_witness_envelope_equals_every_world_host() {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    use mlx_gen::gen_core::{LoadSpec, Quant, WeightsSource};
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("alias.safetensors");
    let (mut raw, arch) = fused_skeleton_headers();
    // Two order-dependent OpenCLIP aliases with different priced facts: a third Q writer (the
    // fixture already carries an equal-facts `attn.q_proj` alias) beside the split in_proj
    // (layer 0), and a direct fc1 writer beside c_fc (layer 1) whose wider output changes the
    // layer quantum and the pack point.
    raw.push(hierarchy_header(
        "conditioner.embedders.1.model.transformer.resblocks.0.self_attn.q_proj.weight",
        &[32, 64],
        Stored::F32,
    ));
    raw.push(hierarchy_header(
        "conditioner.embedders.1.model.transformer.resblocks.1.mlp.fc1.weight",
        &[256, 64],
        Stored::F32,
    ));
    write_header_only_sparse(&file, &raw);
    for (staged, quantized) in [(false, false), (true, true), (false, true)] {
        let mut spec = LoadSpec::new(WeightsSource::File(file.clone()));
        spec.quantize = quantized.then_some(Quant::Q4);
        let mut r = regime(Route::Text);
        r.staged = staged;
        r.tiling = None;
        let mut input = Inputs::from_spec_for_test(&spec, &r).unwrap();
        input.arch = arch.clone();
        let (witness, every, base) = crate::quantum::with_backend(Backend::Cpu, || {
            let witness =
                crate::ios_pricing::alias_envelope_for_test(&mut input, &r, Some(&spec), false)
                    .unwrap();
            let every =
                crate::ios_pricing::alias_envelope_for_test(&mut input, &r, Some(&spec), true)
                    .unwrap();
            let base = Priced::build_configured(&input, &spec)
                .unwrap()
                .price(&r)
                .unwrap()
                .0;
            (witness, every, base)
        });
        let peaks = |pricing: &mlx_gen::ios_pricing::IosPricing| {
            pricing
                .windows
                .iter()
                .map(|w| {
                    (
                        (
                            w.phase,
                            w.class.clone(),
                            w.label.clone(),
                            w.owner_domain.clone(),
                        ),
                        w.logical_peak_bytes(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let phases = |pricing: &mlx_gen::ios_pricing::IosPricing| {
            pricing
                .phases
                .iter()
                .map(|phase| (phase.phase, phase.peak_bytes()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            peaks(&witness),
            peaks(&every),
            "witness worlds reach every window maximum over all realizable worlds staged={staged} quantized={quantized}"
        );
        assert_eq!(phases(&witness), phases(&every));
        assert_eq!(witness.absent_phases, every.absent_phases);
        assert_ne!(
            peaks(&every),
            peaks(&base),
            "the aliases change priced windows; a single world is not the envelope"
        );
    }
}

#[test]
fn fused_file_header_build_and_real_load_recipe_are_connected_host() {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    use mlx_gen::gen_core::{LoadSpec, Quant, WeightsSource};
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("full.safetensors");
    let (raw, arch) = fused_skeleton_headers();
    write_header_only_sparse(&file, &raw);
    let (fix, _) = vae_literal_headers();
    let fix_file = directory.path().join("fix.safetensors");
    write_header_only_sparse(&fix_file, &fix);
    let (vision, mut adapter, vision_cfg, mut resampler_cfg) = ip_literal_headers();
    resampler_cfg.output_dim = 64;
    for header in &mut adapter {
        if header.name == "image_proj.proj_out.weight" {
            header.shape[0] = 64;
        }
        if header.name == "image_proj.proj_out.bias"
            || header.name.starts_with("image_proj.norm_out.")
        {
            header.shape = vec![64];
        }
        header.data_bytes = header.shape.iter().product::<usize>() as u64 * 4;
    }
    for role in ["to_k_ip", "to_v_ip"] {
        adapter.push(hierarchy_header(
            &format!("ip_adapter.1.{role}.weight"),
            &[64, 64],
            Stored::F32,
        ));
    }
    let ip_root = directory.path().join("ip");
    write_header_only_sparse(
        &ip_root.join(crate::loader::IP_ADAPTER_IMAGE_ENCODER_FILE),
        &vision,
    );
    write_header_only_sparse(
        &ip_root.join(crate::loader::IP_ADAPTER_WEIGHT_FILES[0]),
        &adapter,
    );
    for (staged, quantized, external, ip) in [
        (false, false, false, false),
        (true, false, false, false),
        (false, true, true, true),
        (true, true, true, true),
    ] {
        let mut spec = LoadSpec::new(WeightsSource::File(file.clone()));
        spec.quantize = quantized.then_some(Quant::Q4);
        if external {
            spec.components.insert(
                crate::loader::VAE_FP16_FIX_COMPONENT.into(),
                WeightsSource::File(fix_file.clone()),
            );
        }
        if ip {
            spec.ip_adapter = Some(WeightsSource::Dir(ip_root.clone()));
        }
        let mut r = regime(Route::Text);
        r.staged = staged;
        r.tiling = None;
        let mut input = Inputs::from_spec_for_test(&spec, &r).unwrap();
        assert!(input.fused.is_some());
        assert!(!input.streamable);
        assert_eq!(input.te1, file);
        assert_eq!(input.te2, file);
        assert_eq!(input.unet, file);
        input.arch = arch.clone();
        input.arch.vision = vision_cfg.clone();
        input.arch.resampler = resampler_cfg.clone();
        let (priced, pricing, points) = crate::quantum::with_backend(Backend::Cpu, || {
            let priced = Priced::build_configured(&input, &spec).unwrap();
            let (pricing, points) = priced.price(&r).unwrap();
            (priced, pricing, points)
        });
        let loads = points
            .iter()
            .filter(|p| p.phase == IosPhase::Load)
            .collect::<Vec<_>>();
        assert_eq!(loads[0].label, LDM_TEXT_LOAD);
        let heavy = loads
            .iter()
            .position(|p| p.label == LDM_HEAVY_LOAD)
            .unwrap();
        let text_packs = loads[..heavy]
            .iter()
            .filter(|p| p.label == LOAD_QUANTIZE)
            .count();
        assert_eq!(
            text_packs,
            if quantized { 25 } else { 0 },
            "2layers*6projections per tower +TE2projection, both constructors before packs"
        );
        assert!(
            !loads.iter().any(|p| [
                TEXT_ENCODER_1_LOAD,
                TEXT_ENCODER_2_LOAD,
                UNET_LOAD,
                VAE_LOAD
            ]
            .contains(&p.label)),
            "from-weights constructors must not become synthetic Directory guards"
        );
        // Independent raw-source arithmetic: the shared native guard counts all mapped text
        // operands, including the whole QKV parent and the equal alias, at F16 plus largest read.
        let text = raw
            .iter()
            .filter(|h| h.name.starts_with("conditioner.embedders."));
        let (mut sum, mut largest) = (0u64, 0u64);
        for header in text {
            let cast = header.shape.iter().product::<usize>() as u64 * 2;
            sum += cast;
            largest = largest.max(cast + header.data_bytes);
        }
        assert_eq!(loads[0].scratch, sum + largest);
        let text_parents = raw
            .iter()
            .filter(|h| h.name.starts_with("conditioner.embedders."))
            .map(|h| mlx_gen::ios_quantum::allocated_bytes(h.data_bytes))
            .sum::<u64>();
        let heavy_parents = raw
            .iter()
            .filter(|h| {
                h.name.starts_with("model.diffusion_model.")
                    || h.name.starts_with("first_stage_model.")
            })
            .map(|h| mlx_gen::ios_quantum::allocated_bytes(h.data_bytes))
            .sum::<u64>();
        if quantized {
            assert_eq!(
                loads[1].resident,
                text_parents
                    + if staged { 0 } else { heavy_parents }
                    + priced.te_bytes[0].resident
                    + priced.te_bytes[1].resident,
                "first TE pack retains both constructed encoders and warm-only sibling source maps"
            );
        }
        let scalar_width = if quantized { 4 } else { 2 };
        let conditioning = r.conditioning_rows() as u64 * (8 * 128 + 16) * scalar_width;
        assert_eq!(
            loads[heavy].resident,
            if staged {
                conditioning + text_parents
            } else {
                priced.te_pack[0].retained
                    + priced.te_pack[1].retained
                    + text_parents
                    + heavy_parents
            },
            "actual warm versus staged heavy boundary; no native source-parent retirement credit"
        );
        if external {
            let fix = loads
                .iter()
                .position(|p| p.label == VAE_FP16_FIX_LOAD)
                .unwrap();
            let image = loads
                .iter()
                .position(|p| p.label == IP_IMAGE_ENCODER_LOAD)
                .unwrap();
            let adapter = loads
                .iter()
                .position(|p| p.label == IP_ADAPTER_LOAD)
                .unwrap();
            assert!(heavy < fix && fix < image && image < adapter);
            assert!(loads[adapter + 1..]
                .iter()
                .all(|p| p.label == LOAD_QUANTIZE));
            assert_eq!(loads[adapter+1..].len(),24,
                "4time/add +5resnet +2outerproj +8attention +3GEGLU +2installedIP packs, never conv shortcuts");
            assert!(priced.vae_bytes.floor > 0);
        }
        assert!(
            pricing.windows.iter().all(|w| w.class.is_none()),
            "source logical parents never manufacture native backing certificates"
        );
    }
}

#[test]
fn fused_fullsplit_sibling_failure_precedes_missing_clip_constructor_host() {
    use mlx_gen::gen_core::{LoadSpec, WeightsSource};
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("invalid.safetensors");
    let (mut raw, arch) = fused_skeleton_headers();
    raw.retain(|h| {
        h.name != "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight"
    });
    let sibling = raw
        .iter_mut()
        .find(|h| h.name.ends_with("attn.in_proj_weight"))
        .unwrap();
    sibling.shape[0] = 193;
    sibling.data_bytes = sibling.shape.iter().product::<usize>() as u64 * 4;
    write_header_only_sparse(&file, &raw);
    let spec = LoadSpec::new(WeightsSource::File(file.clone()));
    let r = regime(Route::Text);
    let error = Inputs::from_spec_for_test(&spec, &r)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("transform failure"),
        "all source siblings transform before constructors: {error}"
    );
    sibling_failure_removed_for_test(&mut raw);
    write_header_only_sparse(&file, &raw);
    let mut input = Inputs::from_spec_for_test(&spec, &r).unwrap();
    input.arch = arch;
    let error = Priced::build_configured(&input, &spec)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("text_model.embeddings.token_embedding.weight"));
}

fn sibling_failure_removed_for_test(
    raw: &mut [mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
) {
    let sibling = raw
        .iter_mut()
        .find(|h| h.name.ends_with("attn.in_proj_weight"))
        .unwrap();
    sibling.shape[0] = 192;
    sibling.data_bytes = sibling.shape.iter().product::<usize>() as u64 * 4;
}

#[test]
fn fused_both_text_constructors_precede_pack_and_optional_resolution_host() {
    use mlx_gen::gen_core::{LoadSpec, Quant, WeightsSource};
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("chronology.safetensors");
    let (mut raw, arch) = fused_skeleton_headers();
    raw.retain(|h| h.name != "conditioner.embedders.1.model.token_embedding.weight");
    let projection = raw
        .iter_mut()
        .find(|h| {
            h.name
                == "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight"
        })
        .unwrap();
    projection.shape[1] = 65;
    projection.data_bytes = projection.shape.iter().product::<usize>() as u64 * 4;
    write_header_only_sparse(&file, &raw);
    let mut spec = LoadSpec::new(WeightsSource::File(file));
    spec.quantize = Some(Quant::Q4);
    // This invalid optional source must be resolved only at its heavy-load position, not while
    // Inputs resolves the fused source. TE2 constructor must also beat TE1's bad pack alignment.
    spec.ip_adapter = Some(WeightsSource::File(
        directory.path().join("not-an-ip-directory.safetensors"),
    ));
    let r = regime(Route::Text);
    let mut input = Inputs::from_spec_for_test(&spec, &r).unwrap();
    input.arch = arch;
    let error = Priced::build_configured(&input, &spec)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("text_model.embeddings.token_embedding.weight"),
        "TE2 construction before TE1 pack and heavy optional source selection: {error}"
    );
}

#[test]
fn fused_external_fix_replaces_unconstructed_vae_after_unet_host() {
    use mlx_gen::gen_core::{LoadSpec, WeightsSource};
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("vae-missing.safetensors");
    let (mut raw, arch) = fused_skeleton_headers();
    raw.retain(|h| h.name != "first_stage_model.decoder.conv_in.weight");
    write_header_only_sparse(&file, &raw);
    let mut spec = LoadSpec::new(WeightsSource::File(file.clone()));
    let r = regime(Route::Text);
    let mut input = Inputs::from_spec_for_test(&spec, &r).unwrap();
    input.arch = arch.clone();
    let error = Priced::build_configured(&input, &spec)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("decoder.conv_in.weight"));
    let (fix, _) = vae_literal_headers();
    let fix_file = directory.path().join("fp16-fix.safetensors");
    write_header_only_sparse(&fix_file, &fix);
    spec.components.insert(
        crate::loader::VAE_FP16_FIX_COMPONENT.into(),
        WeightsSource::File(fix_file),
    );
    let mut input = Inputs::from_spec_for_test(&spec, &r).unwrap();
    input.arch = arch.clone();
    let priced = Priced::build_configured(&input, &spec).unwrap();
    assert!(
        priced.vae_bytes.floor > 0,
        "only the actual external fix's own load floor"
    );
    // U-Net constructor precedes even an unresolved staged fix selector. The old fused VAE
    // remains structurally nonempty, so the fullsplit's earlier nonempty checks still succeed.
    raw.retain(|h| h.name != "model.diffusion_model.input_blocks.0.0.weight");
    write_header_only_sparse(&file, &raw);
    spec.components.insert(
        crate::loader::VAE_FP16_FIX_COMPONENT.into(),
        WeightsSource::Dir(directory.path().join("missing-fix-snapshot")),
    );
    let mut input = Inputs::from_spec_for_test(&spec, &r).unwrap();
    input.arch = arch;
    let error = Priced::build_configured(&input, &spec)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("conv_in.weight"),
        "UNet constructor beats later fix lookup: {error}"
    );
}
