//! Candle LTX-2.3 LoRA training (sc-13867).
//!
//! The recipe matches `mlx-gen-ltx`: single-frame VAE latents and Gemma video features are cached
//! once, the encoder stack is dropped before the DiT loads, and the trainable video-only DiT regresses
//! raw velocity against `noise - clean` at a seeded uniform sigma. LTX is deliberately f32-only;
//! adapter factors, loss reductions, and optimizer state are f32.

use std::path::PathBuf;
use std::sync::Arc;

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor, Var};
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::train::{
    NetworkType, Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest,
};
use candle_gen::gen_core::{self, CancelFlag, Image, LoadSpec, Modality, Progress, WeightsSource};
use candle_gen::train::dataset::{bucket_resolution, load_image_tensor};
use candle_gen::train::flow_match::{
    self, run_flow_match_training, velocity_loss, FlowMatchTrainer, SamplePlan,
};
use candle_gen::train::gradient_checkpoint::checkpointed_backward;
use candle_gen::{CandleError, Result};

use crate::config::{
    AvConfig, ConnectorConfig, GemmaConfig, DEFAULT_FPS, LATENT_CHANNELS, SPATIAL_SCALE,
    STAGE1_SIGMAS, TRAINER_ID,
};
use crate::dit_train::{LtxDiT, LTX_ATTN_TARGETS};
use crate::pipeline::{flatten_latent, frames_to_images, unflatten_latent};
use crate::rope::create_position_grid;
use crate::text_encoder::LtxTextEncoder;
use crate::tier::TierPaths;
use crate::vae::LtxVideoVae;

const LABEL: &str = "ltx_2_3 trainer";
const SAMPLE_PROMPT_CAP: usize = 4;
const TRAIN_TEXT_MAX_LENGTH: usize = 128;
const SAFE_MEMORY_FRACTION: f64 = 0.85;

/// One preview prompt's already-encoded distilled conditioning.
pub struct LtxSampleState {
    contexts: Vec<Tensor>,
    vae: Arc<LtxVideoVae>,
    positions: Tensor,
    latent_edge: usize,
}

/// Lazy trainer: caching loads Gemma + VAE first and drops Gemma before `build_dit`.
pub struct LtxTrainer {
    descriptor: TrainerDescriptor,
    root: PathBuf,
    gemma_override: Option<PathBuf>,
    device: Device,
}

pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: TRAINER_ID,
        family: "ltx",
        backend: "candle",
        modality: Modality::Video,
        supports_lora: true,
        supports_lokr: false,
        supports_control: false,
        supports_full_finetune: false,
    }
}

pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(path) => path.clone(),
        WeightsSource::File(_) => {
            return Err(CandleError::Msg(format!(
                "{LABEL}: expects the split q4/q8 LTX tier directory, not a single checkpoint"
            )))
        }
    };
    let gemma_override = match spec.text_encoder.as_ref() {
        Some(WeightsSource::Dir(path)) => Some(path.clone()),
        Some(WeightsSource::File(_)) => {
            return Err(CandleError::Msg(format!(
                "{LABEL}: text_encoder must be a Gemma snapshot directory"
            )))
        }
        None => None,
    };
    Ok(Box::new(LtxTrainer {
        descriptor: trainer_descriptor(),
        root,
        gemma_override,
        device: candle_gen::default_device()?,
    }))
}

candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

/// LTX always trains f32. Bf16 was measured to decorrelate gradients through the deep distilled DiT.
fn validate_ltx_request(req: &TrainingRequest) -> Result<()> {
    flow_match::validate_flow_match_request(req, LABEL)?;
    if req.config.network_type != NetworkType::Lora {
        return Err(CandleError::Msg(format!(
            "{LABEL}: LoKr training is unsupported; LTX training is LoRA-only"
        )));
    }
    let dtype = req.config.train_dtype.trim();
    if dtype.eq_ignore_ascii_case("bf16") || dtype.eq_ignore_ascii_case("bfloat16") {
        return Err(CandleError::Msg(format!(
            "{LABEL}: bf16 training is rejected; LTX LoRA training requires f32"
        )));
    }
    Ok(())
}

impl Trainer for LtxTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        validate_ltx_request(req).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.validate(req)?;
        run_flow_match_training(self, req, on_progress).map_err(Into::into)
    }
}

/// Dense-backward peak projection inherited from the MLX real-weight sweep (16.9 GiB resident plus
/// 0.0251 GiB/token). This is a conservative fail-fast policy input, not a claim of Candle-calibrated
/// VRAM parity; a dedicated Candle real-weight sweep remains follow-up validation.
pub fn projected_dense_peak_gb(latent_tokens: usize) -> f64 {
    16.9 + 0.0251 * latent_tokens as f64
}

fn preflight_against_budget(resolution: u32, available_bytes: u64) -> Result<()> {
    let edge = bucket_resolution(resolution);
    let latent_edge = edge as usize / SPATIAL_SCALE;
    let projected = projected_dense_peak_gb(latent_edge * latent_edge);
    let available_gb = available_bytes as f64 / 1024f64.powi(3);
    let safe = available_gb * SAFE_MEMORY_FRACTION;
    if projected > safe {
        return Err(CandleError::Msg(format!(
            "{LABEL}: a dense first step at {edge}px projects ~{projected:.1} GiB, exceeding the \
             {safe:.1} GiB safe budget ({available_gb:.1} GiB free x {SAFE_MEMORY_FRACTION:.2}); \
             enable gradient checkpointing or reduce resolution"
        )));
    }
    Ok(())
}

fn available_device_bytes() -> u64 {
    #[cfg(feature = "cuda")]
    {
        use candle_gen::candle_core::cuda_backend::cudarc::driver::result as cuda;
        if let Ok((free, _)) = cuda::mem_get_info() {
            return free as u64;
        }
    }
    // CPU/Metal tests use a conservative 64 GiB logical budget; production candle LTX is CUDA.
    64 * 1024 * 1024 * 1024
}

fn tier(self_: &LtxTrainer) -> Result<TierPaths> {
    TierPaths::detect(&self_.root, self_.gemma_override.as_deref()).ok_or_else(|| {
        CandleError::Msg(format!(
            "{LABEL}: {} is not a split packed LTX tier (missing transformer.safetensors or \
             quantize_config.json)",
            self_.root.display()
        ))
    })
}

fn pad_training_ids(mut ids: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
    ids.truncate(TRAIN_TEXT_MAX_LENGTH);
    let pad = TRAIN_TEXT_MAX_LENGTH - ids.len();
    let mut padded = vec![0u32; pad];
    padded.extend_from_slice(&ids);
    let mut mask = vec![0u32; pad];
    mask.extend(std::iter::repeat_n(1u32, ids.len()));
    (padded, mask)
}

fn tokenize(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    device: &Device,
) -> Result<(Tensor, Vec<u32>)> {
    let enc = tokenizer
        .encode(text, true)
        .map_err(|e| CandleError::Msg(format!("{LABEL}: tokenize: {e}")))?;
    let (padded, mask) = pad_training_ids(enc.get_ids().to_vec());
    Ok((
        Tensor::from_vec(padded, (1, TRAIN_TEXT_MAX_LENGTH), device)?,
        mask,
    ))
}

fn encode_context(
    tokenizer: &tokenizers::Tokenizer,
    encoder: &LtxTextEncoder,
    text: &str,
    device: &Device,
) -> Result<Tensor> {
    let (ids, mask) = tokenize(tokenizer, text, device)?;
    Ok(encoder.encode(&ids, &mask)?.to_dtype(DType::F32)?)
}

/// Seeded sigma for LTX: always uniform and strictly inside `(1e-3, 1-1e-3)`. Timestep type/bias
/// knobs intentionally do not participate in this family recipe.
pub fn sample_ltx_sigma(seed: u64, step: u32) -> f64 {
    let lower = f32::from_bits(1e-3f32.to_bits() + 1);
    let upper = f32::from_bits((1.0f32 - 1e-3).to_bits() - 1);
    flow_match::sample_uniform_range(flow_match::timestep_seed(seed, step), lower, upper) as f64
}

#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &LtxDiT,
    vars: &[Var],
    clean: &Tensor,
    context: &Tensor,
    positions: &Tensor,
    sigma: f64,
    noise: &Tensor,
    mae: bool,
    checkpoint: bool,
) -> Result<(f32, GradStore)> {
    let (x_t, target) = flow_match::build_batch(clean, noise, sigma)?;
    if checkpoint {
        let (hidden, ctx) = dit.forward_pre_main(&x_t, sigma, context, positions)?;
        let mut segments = dit.main_block_segments(&ctx);
        let target = target.clone();
        let ctx_ref = &ctx;
        segments.push(Box::new(move |state: &[Tensor]| {
            let velocity = dit.velocity_out(&state[0], ctx_ref)?;
            Ok(vec![velocity_loss(&velocity, &target, mae)?])
        }));
        checkpointed_backward(&segments, &[hidden.detach()], vars)
    } else {
        let velocity = dit.forward(&x_t, sigma, context, positions)?;
        let loss = velocity_loss(&velocity, &target, mae)?;
        let value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        Ok((value, loss.backward()?))
    }
}

impl FlowMatchTrainer for LtxTrainer {
    type Dit = LtxDiT;
    type Cached = (Tensor, Tensor);
    type Aux = Tensor;
    type SampleState = LtxSampleState;
    const LABEL: &'static str = LABEL;

    fn device(&self) -> &Device {
        &self.device
    }

    fn default_targets(&self) -> &'static [&'static str] {
        &LTX_ATTN_TARGETS
    }

    fn preflight(&self, req: &TrainingRequest) -> Result<()> {
        if !req.config.gradient_checkpointing {
            preflight_against_budget(req.config.resolution, available_device_bytes())?;
        }
        Ok(())
    }

    fn cache(
        &self,
        req: &TrainingRequest,
        device: &Device,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<(Vec<Self::Cached>, Self::Aux, SamplePlan<Self::SampleState>)> {
        let paths = tier(self)?;
        paths.validate_group_size()?;
        let connector = paths.connector_vb(DType::BF16, device)?;
        let connector_root = connector.pp("model.diffusion_model");
        let encoder = LtxTextEncoder::new(
            paths.gemma_vb(DType::BF16, device)?,
            connector_root.clone(),
            connector_root,
            &GemmaConfig::gemma_3_12b(),
            &ConnectorConfig::ltx_2_3(),
        )?;
        let tokenizer = tokenizers::Tokenizer::from_file(paths.tokenizer_path())
            .map_err(|e| CandleError::Msg(format!("{LABEL}: load tokenizer: {e}")))?;
        let vae = LtxVideoVae::new_with_encoder(
            paths.vae_vb(DType::F32, device)?.pp("vae"),
            paths.vae_encoder_vb(DType::F32, device)?.pp("vae"),
            LATENT_CHANNELS,
            4,
        )?;
        let edge = bucket_resolution(req.config.resolution);
        let latent_edge = edge as usize / SPATIAL_SCALE;
        let positions =
            create_position_grid(1, latent_edge, latent_edge, DEFAULT_FPS as f32, device)?;
        let mut cached = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total: req.items.len() as u32,
            });
            let image = load_image_tensor(&item.image_path, edge, device)?;
            let video = image.unsqueeze(2)?;
            let clean = flatten_latent(&vae.encode(&video)?)?.to_dtype(DType::F32)?;
            let context = encode_context(&tokenizer, &encoder, &item.caption, device)?
                .to_dtype(DType::F32)?;
            cached.push((clean, context));
        }

        let sample_plan = if req.config.sample_every > 0 && !req.config.sample_prompts.is_empty() {
            let prompts: Vec<String> = req
                .config
                .sample_prompts
                .iter()
                .take(SAMPLE_PROMPT_CAP)
                .cloned()
                .collect();
            let mut contexts = Vec::with_capacity(prompts.len());
            for prompt in &prompts {
                if req.cancel.is_cancelled() {
                    return Err(CandleError::Canceled);
                }
                contexts.push(encode_context(&tokenizer, &encoder, prompt, device)?);
            }
            SamplePlan {
                prompts,
                state: Some(LtxSampleState {
                    contexts,
                    vae: Arc::new(vae),
                    positions: positions.clone(),
                    latent_edge,
                }),
            }
        } else {
            SamplePlan::disabled()
        };
        // `encoder` and `tokenizer` drop here before the driver builds the 22B DiT.
        Ok((cached, positions, sample_plan))
    }

    fn build_dit(&self, _req: &TrainingRequest, device: &Device) -> Result<LtxDiT> {
        let paths = tier(self)?;
        paths.validate_group_size()?;
        Ok(LtxDiT::new(
            paths
                .dit_vb(DType::F32, device)?
                .pp("model.diffusion_model"),
            &AvConfig::ltx_2_3().video,
        )?)
    }

    fn micro_step(
        &self,
        dit: &LtxDiT,
        vars: &[Var],
        cached: &Self::Cached,
        positions: &Self::Aux,
        cfg: &TrainingConfig,
        step: u32,
        device: &Device,
    ) -> Result<(f32, GradStore)> {
        let (clean, context) = cached;
        let sigma = sample_ltx_sigma(cfg.seed, step);
        let noise =
            flow_match::sample_noise(clean.dims(), flow_match::noise_seed(cfg.seed, step), device)?;
        compute_loss_grads(
            dit,
            vars,
            clean,
            context,
            positions,
            sigma,
            &noise,
            flow_match::is_mae(cfg),
            cfg.gradient_checkpointing,
        )
    }

    fn render_sample(
        &self,
        dit: &LtxDiT,
        state: &LtxSampleState,
        index: usize,
        _cfg: &TrainingConfig,
        seed: u64,
    ) -> Result<Image> {
        let edge = state.latent_edge;
        let latent = crate::pipeline::create_noise(seed, 1, edge, edge, &self.device)?;
        let noise = flatten_latent(&latent)?;
        let cancel = CancelFlag::new();
        let mut progress = |_: Progress| {};
        let context = &state.contexts[index];
        let out = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &STAGE1_SIGMAS,
            noise,
            seed,
            &cancel,
            &mut progress,
            |x, sigma| Ok(dit.forward(x, sigma as f64, context, &state.positions)?),
        )?;
        let latent = unflatten_latent(&out.to_dtype(DType::F32)?, 1, edge, edge)?;
        frames_to_images(&state.vae.decode(&latent)?)?
            .into_iter()
            .next()
            .ok_or_else(|| CandleError::Msg(format!("{LABEL}: preview decode produced no frame")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::runtime::CancelFlag;
    use candle_gen::gen_core::train::TrainingItem;
    use std::path::PathBuf;

    fn request() -> TrainingRequest {
        TrainingRequest {
            items: vec![TrainingItem::captioned(
                PathBuf::from("image.png"),
                "caption".into(),
            )],
            config: TrainingConfig {
                steps: 4,
                train_dtype: "f32".into(),
                timestep_type: "sigmoid".into(),
                timestep_bias: "high_noise".into(),
                ..Default::default()
            },
            output_dir: PathBuf::from("out"),
            file_name: "ltx.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    #[test]
    fn uniform_sigma_is_deterministic_strictly_interior_and_config_invariant() {
        let lower = f32::from_bits(1e-3f32.to_bits() + 1) as f64;
        let upper = f32::from_bits((1.0f32 - 1e-3).to_bits() - 1) as f64;
        let mut endpoint_hits = 0;
        for seed in 0..20_000 {
            let sigma = sample_ltx_sigma(seed, 1);
            assert!(sigma >= lower && sigma <= upper, "{sigma}");
            endpoint_hits += usize::from(sigma == lower || sigma == upper);
        }
        assert!(
            endpoint_hits <= 1,
            "affine sampling must not pile clamped mass onto inward endpoints: {endpoint_hits}"
        );
        for seed in [0, 1, 42, u64::MAX] {
            for step in 1..20 {
                let a = sample_ltx_sigma(seed, step);
                let b = sample_ltx_sigma(seed, step);
                assert_eq!(a, b);
                assert!(a > 1e-3 && a < 1.0 - 1e-3, "{a}");
            }
        }
        // The function accepts no timestep_type/bias: family behavior cannot shift with config.
        let mut a = request();
        let mut b = request();
        b.config.timestep_type = "weighted".into();
        b.config.timestep_bias = "low_noise".into();
        assert_eq!(
            sample_ltx_sigma(a.config.seed, 3),
            sample_ltx_sigma(b.config.seed, 3)
        );
        a.config.timestep_type = "uniform".into();
        assert_eq!(
            sample_ltx_sigma(a.config.seed, 3),
            sample_ltx_sigma(b.config.seed, 3)
        );
    }

    #[test]
    fn training_tokenization_truncates_and_masks_at_128_tokens() {
        let (ids, mask) = pad_training_ids((0..200).collect());
        assert_eq!(ids.len(), TRAIN_TEXT_MAX_LENGTH);
        assert_eq!(mask.len(), TRAIN_TEXT_MAX_LENGTH);
        assert_eq!(ids, (0..128).collect::<Vec<_>>());
        assert!(mask.iter().all(|&value| value == 1));

        let (ids, mask) = pad_training_ids(vec![7, 8, 9]);
        assert_eq!(&ids[125..], &[7, 8, 9]);
        assert!(ids[..125].iter().all(|&value| value == 0));
        assert!(mask[..125].iter().all(|&value| value == 0));
        assert_eq!(&mask[125..], &[1, 1, 1]);
    }

    #[test]
    fn flow_recipe_and_losses_match_reference() {
        let dev = Device::Cpu;
        let clean = Tensor::from_vec(vec![2.0f32, 4.0], (1, 2), &dev).unwrap();
        let noise = Tensor::from_vec(vec![1.0f32, 0.0], (1, 2), &dev).unwrap();
        let (x_t, target) = flow_match::build_batch(&clean, &noise, 0.25).unwrap();
        assert_eq!(x_t.to_vec2::<f32>().unwrap(), vec![vec![1.75, 3.0]]);
        assert_eq!(target.to_vec2::<f32>().unwrap(), vec![vec![-1.0, -4.0]]);
        let prediction = Tensor::zeros((1, 2), DType::BF16, &dev).unwrap();
        let mse = velocity_loss(&prediction, &target, false)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let mae = velocity_loss(&prediction, &target, true)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!((mse - 8.5).abs() < 1e-6);
        assert!((mae - 2.5).abs() < 1e-6);
    }

    #[test]
    fn validation_rejects_bf16_lokr_and_bad_core_configs() {
        let mut req = request();
        validate_ltx_request(&req).unwrap();
        req.config.train_dtype = "bf16".into();
        assert!(validate_ltx_request(&req)
            .unwrap_err()
            .to_string()
            .contains("requires f32"));
        req.config.train_dtype = "f32".into();
        req.config.network_type = NetworkType::Lokr;
        assert!(validate_ltx_request(&req)
            .unwrap_err()
            .to_string()
            .contains("LoRA-only"));
        req.config.network_type = NetworkType::Lora;
        req.config.rank = 0;
        assert!(validate_ltx_request(&req).is_err());
    }

    #[test]
    fn memory_projection_and_guard_are_testable_before_cache() {
        assert!((projected_dense_peak_gb(1024) - 42.6024).abs() < 1e-4);
        preflight_against_budget(512, 64 * 1024 * 1024 * 1024).unwrap();
        let err = preflight_against_budget(2048, 16 * 1024 * 1024 * 1024)
            .unwrap_err()
            .to_string();
        assert!(err.contains("gradient checkpointing"), "{err}");
    }

    #[test]
    fn descriptor_and_default_targets_are_exact() {
        let d = trainer_descriptor();
        assert_eq!(d.id, TRAINER_ID);
        assert_eq!(d.backend, "candle");
        assert!(d.supports_lora);
        assert!(!d.supports_lokr);
        assert_eq!(LTX_ATTN_TARGETS, ["to_q", "to_k", "to_v", "to_out.0"]);
    }
}
