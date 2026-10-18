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

    /// A `bits`-packed tier, as the published q4/q8 snapshots are: the U-Net pre-quantized (u32
    /// codes with bf16 `.scales`/`.biases`, whose projections promote f16 activations to f32), the
    /// dense leaves at f16; the ControlNet quantized at load. 64-wide so its projections pack.
    fn packed(bits: i32) -> Self {
        Self::build(1, Some(bits))
    }

    fn build(transformer_layers: i32, bits: Option<i32>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let tiny = crate::test_support::tiny_control_unet_config();
        let (width, context, pooled) = match bits {
            None => (tiny.block_out_channels.clone(), 16, 16),
            Some(_) => (vec![64, 64, 128], 64, 16),
        };
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
        save(&half(w1), &root.join("text_encoder/model.safetensors"));
        save(&half(w2), &root.join("text_encoder_2/model.safetensors"));
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
    let dtype = crate::model::DTYPE;
    let te1 = crate::loader::load_clip_dtype(&s.root, "text_encoder", &s.te1, dtype)?;
    let te2 = crate::loader::load_clip_dtype(&s.root, "text_encoder_2", &s.te2, dtype)?;
    let mut unet = crate::loader::load_unet_with_config_streamed(&s.root, dtype, &s.unet, true)?;
    let vae = crate::loader::load_vae_fp16_fix_with(
        &mlx_gen::WeightsSource::Dir(s.root.join("vae_fp16_fix")),
        &s.vae,
    )?;
    let mut controls = Vec::new();
    if route == Route::Control {
        controls.push(crate::loader::load_controlnet_with(
            &mlx_gen::WeightsSource::Dir(s.root.join("controlnet")),
            dtype,
            &s.unet,
        )?);
    }
    let mut ip = None;
    if route == Route::IpAdapter {
        let (encoder, pairs) = crate::loader::load_ip_adapter_with(
            &s.root.join("ip"),
            dtype,
            &s.vision,
            &s.resampler,
        )?;
        unet.install_ip_adapter(pairs)?;
        ip = Some(encoder);
    }
    if let Some(bits) = s.bits {
        unet.quantize(bits)?;
        for control in &mut controls {
            control.quantize(bits)?;
        }
    }
    unet.arm_block_streams(
        &mlx_gen::WeightsSource::File(crate::loader::resolve_unet_weight_file(&s.root, dtype)?),
        dtype,
        s.bits,
    );
    Ok(Loaded {
        te1,
        te2,
        unet,
        vae,
        controls,
        ip,
    })
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

/// The regime the tiny request runs: staged, sequential CFG, rung 2 at an 8-pixel tile edge (so the
/// tiny decode and init encode both tile), one 8-token CLIP window.
fn regime(route: Route) -> Regime {
    Regime {
        route,
        width: LATENT * VAE_SCALE,
        height: LATENT * VAE_SCALE,
        staged: true,
        cfg_batching: CfgBatching::Sequential,
        tiling: Some(mlx_gen::tiling::TilingConfig::spatial_only(8, 0)),
        context_tokens: TOKENS,
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
    mlx_gen::ios_quantum::assert_tracks_measured(
        "staged IP denoise resident",
        resident,
        measured,
        1.5,
    );
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
    mlx_gen::ios_quantum::assert_tracks_measured("fp16-fix VAE resident", resident, held, 1.5);
    assert_eq!(
        resident, held,
        "the load leaves exactly the f16 copies resident"
    );
    drop(vae);
}
