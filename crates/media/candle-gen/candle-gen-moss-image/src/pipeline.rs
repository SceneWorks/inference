//! FP32 author Euler/CFG/SLG and fixed-noise DMD recurrence.
use crate::{
    loader::{load_weights, tensor_f32, Asset, Sources},
    text_encoder::{digest_f32, Conditioning},
    transformer::{check_cancel, install_replacements, Geometry, Transformer},
};
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{
    self, moss_image::*, AdapterApplyReport, GenerationRequest, Image, Progress,
};
use candle_gen::{CandleError, Result};
use candle_gen_flux2::vae::Flux2Vae;
use rand::{rngs::StdRng, SeedableRng};

pub(crate) struct Renderer {
    pub mode: MossImageAcceleration,
    pub transformer: Transformer,
    pub dataspace: Option<(Tensor, Vec<f32>)>,
    pub adapter_reports: Vec<AdapterApplyReport>,
}
impl Renderer {
    pub fn load(sources: &Sources, mode: MossImageAcceleration, device: &Device) -> Result<Self> {
        let assets = sources.assets(mode)?;
        let dtype = if mode == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32) {
            DType::F32
        } else {
            DType::BF16
        };
        let base = load_weights(
            &assets,
            MossImageAssetRole::Transformer,
            device,
            dtype,
            &[""],
        )?;
        let mixed = mode == MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16);
        let adapter = if mixed {
            Some(load_weights(
                &assets,
                MossImageAssetRole::DmdLora,
                device,
                DType::F32,
                &[""],
            )?)
        } else {
            None
        };
        let base = if mixed {
            install_replacements(
                &base,
                &load_weights(
                    &assets,
                    MossImageAssetRole::DmdTime,
                    device,
                    DType::F32,
                    &[""],
                )?,
            )?
        } else {
            base
        };
        let mut transformer =
            Transformer::load(&base, Geometry::from(&sources.config), adapter.as_ref())?;
        let dataspace = if matches!(mode, MossImageAcceleration::Dmd(_)) {
            let w = load_weights(
                &assets,
                MossImageAssetRole::DmdDataspace,
                device,
                DType::F32,
                &[""],
            )?;
            Some((
                w.require("latent_mix.Q")?,
                tensor_f32(&w.require("step_codes")?)?,
            ))
        } else {
            None
        };
        let mut adapter_reports = if mixed {
            vec![AdapterApplyReport {
                adapter_path: assets
                    .iter()
                    .find(|a| a.role == MossImageAssetRole::DmdLora)
                    .expect("validated mixed assets")
                    .pin
                    .loader_path()
                    .into(),
                applied: 252,
                skipped: vec![],
            }]
        } else {
            vec![]
        };
        adapter_reports.extend(crate::adapters::apply(
            &mut transformer,
            sources.variant,
            &sources.spec.adapters,
            &sources.adapter_pins,
        )?);
        sources.ensure_unchanged()?;
        Ok(Self {
            mode,
            transformer,
            dataspace,
            adapter_reports,
        })
    }
}
pub(crate) struct Sampling {
    pub latents: Tensor,
    pub samples: Vec<MossImageSampleReceipt>,
    pub sampling: MossImageSamplingReceipt,
}
pub(crate) fn sample(
    renderer: &Renderer,
    req: &GenerationRequest,
    resolved: &MossImageResolved,
    inputs: MossImageAdvancedInputs<'_>,
    positive: &Conditioning,
    negative: Option<&Conditioning>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Sampling> {
    sample_graph(
        &renderer.transformer,
        renderer.dataspace.as_ref(),
        req,
        resolved,
        inputs,
        positive,
        negative,
        on_progress,
    )
}
/// The same sampling graph is used for generation and in-training previews.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_graph(
    transformer: &Transformer,
    dataspace: Option<&(Tensor, Vec<f32>)>,
    req: &GenerationRequest,
    resolved: &MossImageResolved,
    inputs: MossImageAdvancedInputs<'_>,
    positive: &Conditioning,
    negative: Option<&Conditioning>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Sampling> {
    let device = transformer.device();
    let count = req.count as usize;
    let images = positive.rows.len() * count;
    let h = req.height as usize / 16;
    let w = req.width as usize / 16;
    let channels = transformer.channels();
    let image_len = channels * h * w;
    let mut noises = Vec::with_capacity(images * image_len);
    let mut samples = Vec::with_capacity(images);
    let initial_seed = req.seed.unwrap_or_else(gen_core::default_seed);
    for i in 0..images {
        check_cancel(&req.cancel)?;
        let (seed, noise) = if let Some(n) = inputs.noise {
            (None, n[i * image_len..(i + 1) * image_len].to_vec())
        } else {
            let seed = initial_seed.wrapping_add(i as u64);
            let mut rng = StdRng::seed_from_u64(seed);
            (
                Some(seed),
                candle_gen::seeded_normal_vec(&mut rng, image_len),
            )
        };
        samples.push(MossImageSampleReceipt {
            seed,
            noise_sha256: digest_f32(&noise, None),
        });
        noises.extend(noise);
    }
    let z = Tensor::from_vec(noises, (images, channels, h, w), device)?;
    let (times, sampling) = if let Some((q, codes)) = dataspace {
        let prefix = moss_dmd_step_codes(codes, resolved.steps)?.to_vec();
        let receipt = MossImageSamplingReceipt::Dmd {
            step_codes: prefix.clone(),
            mix_matrix_sha256: digest_f32(&tensor_f32(q)?, None),
        };
        (prefix, receipt)
    } else {
        let sigmas = moss_image_schedule(resolved.steps, resolved.shift, resolved.training_shift)?;
        let times = sigmas[..sigmas.len() - 1]
            .iter()
            .map(|s| s * 1000.0)
            .collect();
        (times, MossImageSamplingReceipt::Euler { sigmas })
    };
    let mut x = if dataspace.is_some() {
        Tensor::zeros(z.shape(), DType::F32, device)?
    } else {
        z.clone()
    };
    for (step, time) in times.iter().enumerate() {
        check_cancel(&req.cancel)?;
        let input = if let Some((q, _)) = dataspace {
            dmd_input(&z, q, &x)?
        } else {
            x.clone()
        };
        let mut velocities = Vec::with_capacity(images);
        for image in 0..images {
            check_cancel(&req.cancel)?;
            let row = image / count;
            let latent = input.narrow(0, image, 1)?;
            let ctx = &positive.rows[row];
            let mask = vec![1; ctx.dim(1)?];
            let cond = transformer
                .forward(&latent, *time, ctx, &mask, &[], &req.cancel)?
                .to_dtype(DType::F32)?;
            let mut v = cond.clone();
            if dataspace.is_none() {
                if resolved.guidance != 1.0 {
                    let neg = &negative.expect("validated CFG conditioning").rows[row];
                    let active = resolved.skipping_active(step, times.len());
                    let skip = if active {
                        resolved.uncond_skip_layers.as_slice()
                    } else {
                        &[]
                    };
                    let uncond = transformer
                        .forward(
                            &latent,
                            *time,
                            neg,
                            &vec![1; neg.dim(1)?],
                            skip,
                            &req.cancel,
                        )?
                        .to_dtype(DType::F32)?;
                    v = uncond.add(&((&cond - &uncond)? * f64::from(resolved.guidance))?)?;
                }
                if let Some(slg) = &resolved.slg {
                    if !slg.layers.is_empty()
                        && slg.scale > 0.0
                        && resolved.skipping_active(step, times.len())
                    {
                        let skipped = transformer
                            .forward(&latent, *time, ctx, &mask, &slg.layers, &req.cancel)?
                            .to_dtype(DType::F32)?;
                        if resolved.guidance == 1.0 {
                            v = skipped.clone();
                        }
                        v = v.add(&((&cond - &skipped)? * f64::from(slg.scale))?)?;
                    }
                }
            }
            velocities.push(v);
        }
        let v = Tensor::cat(&velocities, 0)?;
        x = match &sampling {
            MossImageSamplingReceipt::Euler { sigmas } => {
                x.add(&(v * f64::from(sigmas[step + 1] - sigmas[step]))?)?
            }
            MossImageSamplingReceipt::Dmd { .. } => z.sub(&v)?,
        };
        device.synchronize()?;
        check_cancel(&req.cancel)?;
        on_progress(Progress::Step {
            current: step as u32 + 1,
            total: times.len() as u32,
        });
    }
    check_cancel(&req.cancel)?;
    if tensor_f32(&x)?.iter().any(|v| !v.is_finite()) {
        return Err(CandleError::Msg("MOSS nonfinite sampled latent".into()));
    }
    Ok(Sampling {
        latents: x,
        samples,
        sampling,
    })
}
pub(crate) fn dmd_input(z: &Tensor, q: &Tensor, x0: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = z.dims4()?;
    let flat = x0.reshape((b, c, h * w))?.transpose(1, 2)?;
    let mixed = candle_gen::candle_nn::Linear::new(q.clone(), None).forward(&flat)?;
    Ok(z.add(&mixed.transpose(1, 2)?.reshape((b, c, h, w))?)?)
}
use candle_gen::candle_nn::Module;
pub(crate) fn decode(
    vae: &Flux2Vae,
    latents: &Tensor,
    req: &GenerationRequest,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    check_cancel(&req.cancel)?;
    on_progress(Progress::Decoding);
    let mut images = Vec::new();
    for i in 0..latents.dim(0)? {
        check_cancel(&req.cancel)?;
        let pixels = vae
            .decode_packed(&latents.narrow(0, i, 1)?.to_dtype(DType::F32)?)?
            .clamp(-1.0, 1.0)?;
        let pixels = pixels
            .squeeze(0)?
            .permute((1, 2, 0))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if pixels.iter().any(|v| !v.is_finite()) {
            return Err(CandleError::Msg("MOSS nonfinite decoded pixels".into()));
        }
        let pixels = pixels
            .iter()
            .map(|v| ((v + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8)
            .collect();
        check_cancel(&req.cancel)?;
        images.push(Image {
            width: req.width,
            height: req.height,
            pixels,
        });
    }
    check_cancel(&req.cancel)?;
    Ok(images)
}
pub(crate) fn receipts(assets: &[Asset]) -> gen_core::Result<Vec<MossImageAssetReceipt>> {
    assets.iter().map(Asset::receipt).collect()
}
