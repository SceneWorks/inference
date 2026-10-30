//! Explicit opt-in CPU fixtures using the real Qwen/full-VAE/MOSS trainer and generator.
//! The wrappers below only establish a CPU device/stream scope; they do not implement training,
//! generation, optimizer logic or synthetic outputs. No runtime bundle enables this feature.
use crate::loader::{Asset, Sources};
use gen_core_moss_image::moss_image::*;
use mlx_gen::gen_core::{
    self, AdapterSpec, Generator, LoadSpec, PinnedWeightsFile, Trainer, WeightsSource,
};
use std::path::Path;
pub const HOST_CAP_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub use gen_core_testkit::moss_training_fixture::Fixture as HostFixture;

/// Observe the original Metal contract without selecting or changing a device/stream.
#[cfg(test)]
pub(crate) fn assert_default_metal_profile_device() {
    assert!(
        matches!(
            mlx_rs::Device::try_default()
                .expect("MLX default device metadata")
                .get_type()
                .expect("MLX device type"),
            mlx_rs::DeviceType::Gpu
        ),
        "manual MOSS author proof requires its original default Metal device"
    );
    assert_eq!(
        mlx_rs::Stream::task_local_or_default()
            .get_index()
            .expect("MLX stream metadata"),
        mlx_rs::Stream::gpu()
            .get_index()
            .expect("MLX GPU stream metadata"),
        "manual MOSS author proof requires its original default GPU stream"
    );
    println!("MOSS_METAL_DEFAULT_DEVICE_AND_STREAM=gpu");
}

pub(crate) fn ensure_cpu() -> mlx_gen::Result<()> {
    if !matches!(
        mlx_rs::Device::try_default()?.get_type()?,
        mlx_rs::DeviceType::Cpu
    ) || mlx_rs::Stream::task_local_or_default().get_index()?
        != mlx_rs::Stream::cpu().get_index()?
    {
        return Err(mlx_gen::Error::Unsupported(
            "synthetic full-pipeline fixture requires an explicit CPU device AND CPU stream".into(),
        ));
    }
    Ok(())
}
fn with_cpu<T>(run: impl FnOnce() -> gen_core::Result<T>) -> gen_core::Result<T> {
    struct Restore(mlx_rs::Device);
    impl Drop for Restore {
        fn drop(&mut self) {
            mlx_rs::Device::set_default(&self.0);
        }
    }
    let previous = mlx_rs::Device::try_default().map_err(mlx_gen::Error::from)?;
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let _restore = Restore(previous);
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
        ensure_cpu()?;
        run()
    })
}
#[derive(Clone, Debug)]
pub struct Fixture {
    pub data: HostFixture,
}
impl Fixture {
    pub fn create(path: &Path) -> gen_core::Result<Self> {
        Ok(Self {
            data: HostFixture::create(path)?,
        })
    }
    fn sources(&self) -> gen_core::Result<Sources> {
        self.data.verify()?;
        let mut ordinary = Vec::new();
        for relative in self.data.files.keys() {
            let pin = PinnedWeightsFile::pin(self.data.root.join(relative))?;
            let role = if relative.starts_with("transformer/") {
                MossImageAssetRole::Transformer
            } else if relative.starts_with("text_encoder/") {
                MossImageAssetRole::TextEncoder
            } else if relative.starts_with("tokenizer/") {
                MossImageAssetRole::Tokenizer
            } else {
                MossImageAssetRole::Vae
            };
            ordinary.push(Asset {
                fixture_digest: Some(*pin.content_sha256()),
                pin,
                role,
                repo: "sceneworks/synthetic-test-fixture",
                revision: "moss-full-pipeline-v1",
                relative: relative.clone(),
            });
        }
        Ok(Sources {
            selection: Some(MossImageAcceleration::Off),
            ordinary,
            optional: Vec::new(),
            config: self.data.config.clone(),
        })
    }
    pub fn trainer(&self, variant: MossImageVariant) -> gen_core::Result<Box<dyn Trainer>> {
        let trainer = crate::training::MossImageTrainer::from_sources(variant, self.sources()?)?;
        Ok(Box::new(CpuTrainer(trainer)))
    }
    pub fn trainer_with_probe(
        &self,
        variant: MossImageVariant,
        probe: gen_core_testkit::moss_training_probe::TrainingProbe,
    ) -> gen_core::Result<Box<dyn Trainer>> {
        let mut trainer =
            crate::training::MossImageTrainer::from_sources(variant, self.sources()?)?;
        trainer.set_test_probe(Some(probe));
        Ok(Box::new(CpuTrainer(trainer)))
    }
    pub fn memory_plan(
        &self,
        variant: MossImageVariant,
        req: &gen_core::TrainingRequest,
    ) -> gen_core::Result<serde_json::Value> {
        let mut trainer =
            crate::training::MossImageTrainer::from_sources(variant, self.sources()?)?;
        trainer.set_test_probe(Some(
            gen_core_testkit::moss_training_probe::TrainingProbe::new(
                self.data.evaluation_manifest()?,
            ),
        ));
        with_cpu(|| trainer.test_memory_plan(req).map_err(Into::into))
    }
    /// FP32 witness through the SAME public advanced Generator path, not decoded uint8 pixels.
    pub fn generate_latents(
        &self,
        variant: MossImageVariant,
        adapters: Vec<AdapterSpec>,
        request: &gen_core::GenerationRequest,
    ) -> gen_core::Result<(Vec<f32>, [usize; 4], Vec<gen_core::AdapterApplyReport>)> {
        let spec =
            LoadSpec::new(WeightsSource::Dir(self.data.root.clone())).with_adapters(adapters);
        let sources = self.sources()?;
        with_cpu(|| {
            let model = crate::model::MossImage::from_sources(variant, &spec, sources)?;
            let output = model.generate_moss_advanced(
                request,
                MossImageAdvancedInputs {
                    output: MossImageOutputKind::Latents,
                    ..Default::default()
                },
                &mut |_| {},
            )?;
            let reports = model.adapter_apply_reports();
            match output {
                MossImageAdvancedOutput::Latents { values, shape } => Ok((values, shape, reports)),
                _ => Err(gen_core::Error::Msg(
                    "actual advanced generator did not return latents".into(),
                )),
            }
        })
    }
    pub fn generator(
        &self,
        variant: MossImageVariant,
        adapters: Vec<AdapterSpec>,
    ) -> gen_core::Result<Box<dyn Generator>> {
        let spec =
            LoadSpec::new(WeightsSource::Dir(self.data.root.clone())).with_adapters(adapters);
        let sources = self.sources()?;
        let generator = with_cpu(|| {
            crate::model::MossImage::from_sources(variant, &spec, sources).map_err(Into::into)
        })?;
        Ok(Box::new(CpuGenerator(generator)))
    }
}
struct CpuTrainer(crate::training::MossImageTrainer);
impl Trainer for CpuTrainer {
    fn descriptor(&self) -> &gen_core::TrainerDescriptor {
        self.0.descriptor()
    }
    fn validate(&self, req: &gen_core::TrainingRequest) -> gen_core::Result<()> {
        with_cpu(|| self.0.validate(req))
    }
    fn train(
        &mut self,
        req: &gen_core::TrainingRequest,
        progress: &mut dyn FnMut(gen_core::TrainingProgress),
    ) -> gen_core::Result<gen_core::TrainingOutput> {
        with_cpu(|| self.0.train(req, progress))
    }
}
struct CpuGenerator(crate::model::MossImage);
impl Generator for CpuGenerator {
    fn descriptor(&self) -> &gen_core::ModelDescriptor {
        self.0.descriptor()
    }
    fn validate(&self, req: &gen_core::GenerationRequest) -> gen_core::Result<()> {
        with_cpu(|| self.0.validate(req))
    }
    fn generate(
        &self,
        req: &gen_core::GenerationRequest,
        progress: &mut dyn FnMut(gen_core::Progress),
    ) -> gen_core::Result<gen_core::GenerationOutput> {
        with_cpu(|| self.0.generate(req, progress))
    }
    fn moss_image_execution_report(&self) -> Option<MossImageExecutionReport> {
        self.0.moss_image_execution_report()
    }
    fn adapter_apply_reports(&self) -> Vec<gen_core::AdapterApplyReport> {
        self.0.adapter_apply_reports()
    }
    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.0.memory_strategy_contract()
    }
}
pub fn generation_request() -> gen_core::GenerationRequest {
    gen_core::GenerationRequest {
        prompt: "style landscape blue".into(),
        width: 32,
        height: 32,
        steps: Some(2),
        guidance: Some(1.),
        seed: Some(173),
        moss_image: Some(MossImageOptions {
            max_sequence_length: 32,
            uncond_skip_layers: Some(Vec::new()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run(case: usize) -> gen_core::Result<()> {
        let directory = tempfile::tempdir()?;
        let fixture = Fixture::create(&directory.path().join("source"))?;
        let variant = if case < 2 {
            MossImageVariant::Base
        } else {
            MossImageVariant::Sft
        };
        let probe = gen_core_testkit::moss_training_probe::TrainingProbe::new(
            fixture.data.evaluation_manifest()?,
        )
        .with_artifact_fallback(directory.path().join("observations.json"));
        gen_core_testkit::moss_training_fixture::assert_recoverable_case(
            &fixture.data,
            &directory.path().join("runs"),
            case,
            || fixture.trainer_with_probe(variant, probe.clone()),
            |adapters| fixture.generator(variant, adapters),
        )?;
        let request = generation_request();
        let (before, shape, _) = fixture.generate_latents(variant, Vec::new(), &request)?;
        let kind = if case.is_multiple_of(2) {
            gen_core::AdapterKind::Lora
        } else {
            gen_core::AdapterKind::Lokr
        };
        let saved = directory
            .path()
            .join("runs/interrupted/adapter.safetensors");
        let (after, after_shape, reports) = fixture.generate_latents(
            variant,
            vec![AdapterSpec::new(saved.clone(), 1., kind)],
            &request,
        )?;
        let (disabled, disabled_shape, _) = fixture.generate_latents(
            variant,
            vec![AdapterSpec::new(saved.clone(), 0., kind)],
            &request,
        )?;
        // This is the real pre-update native export, not manufactured zero factors/output.
        let initial = probe.initial_untrained_adapter()?;
        let initial_generated = initial
            .as_ref()
            .map(|snapshot| {
                fixture.generate_latents(
                    variant,
                    vec![AdapterSpec::new(snapshot.path.clone(), 1., kind)],
                    &request,
                )
            })
            .transpose()?;
        let initial_witness = initial_generated.as_ref().map(|(values, shape, reports)| {
            let path = &initial.as_ref().expect("initial export").path;
            (
                shape,
                values.as_slice(),
                reports
                    .iter()
                    .any(|r| r.adapter_path == *path && r.applied > 0 && r.skipped.is_empty()),
            )
        });
        probe.generation_observation(
            &request,
            (&shape, &before),
            (&after_shape, &after),
            (&disabled_shape, &disabled),
            initial_witness,
            (
                &saved,
                reports
                    .iter()
                    .any(|r| r.adapter_path == saved && r.applied > 0 && r.skipped.is_empty()),
            ),
        )?;
        gen_core_testkit::moss_training_probe::assert_complete_case(&probe, case == 3)?;
        assert_eq!(shape, after_shape);
        assert_eq!(before.len(), after.len());
        assert!(before.iter().chain(&after).all(|v| v.is_finite()));
        // Reload/application is required; a tiny native update need not cross an
        // FP32 output boundary. Presence-only precision effects are not learning.
        let (initial_values, initial_shape, initial_reports) = initial_generated
            .as_ref()
            .expect("actual native step-zero adapter generation");
        assert_eq!(&shape, initial_shape);
        assert_eq!(before.len(), initial_values.len());
        assert!(initial_values.iter().all(|v| v.is_finite()));
        let initial_path = &initial.as_ref().expect("actual step-zero export").path;
        assert!(initial_reports
            .iter()
            .any(|r| r.adapter_path == *initial_path && r.applied > 0 && r.skipped.is_empty()));
        assert!(reports
            .iter()
            .any(|r| r.adapter_path == saved && r.applied > 0 && r.skipped.is_empty()));
        assert_eq!(shape, disabled_shape);
        assert_eq!(
            before, disabled,
            "zero-strength adapter must not fake a generation delta"
        );
        Ok(())
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn full_image_caption_base_lora_cached_adamw_resume_reentry() -> gen_core::Result<()> {
        run(0)
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn full_image_caption_base_lokr_uncached_text_adam_resume_reentry() -> gen_core::Result<()> {
        run(1)
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn full_image_caption_sft_lora_uncached_latents_prodigy_resume_reentry() -> gen_core::Result<()>
    {
        run(2)
    }
    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn full_image_caption_sft_lokr_uncached_both_rose_resume_reentry() -> gen_core::Result<()> {
        run(3)
    }
    #[test]
    fn source_sized_request_staging_preserves_public_default_and_refuses_cap(
    ) -> gen_core::Result<()> {
        let root = tempfile::tempdir()?;
        let fixture = Fixture::create(&root.path().join("source"))?;
        with_cpu(|| {
            let sources = fixture.sources()?;
            let staging = sources.load_staging_bytes()?;
            let released = crate::memory_strategy::request_workspace_bytes(32, 32, 1, 32, 1)?;
            let seq = 36_u64;
            let legacy = seq * 4 * (3072 * 12 + 8192 * 4)
                + seq * seq * 24 * 4
                + 4 * 128 * 4 * 5
                + 32 * 4096 * 4
                + 1024 * 96 * 4 * 6
                + 5_362_224_248;
            assert_eq!(released, legacy, "public released-model estimate changed");
            let actual = sources.request_workspace_bytes(32, 32, 1, 32, 1)?;
            assert_eq!(
                actual,
                released - 5_362_224_248 + staging,
                "private fixture source staging ignored"
            );
            assert_eq!(
                actual,
                crate::memory_strategy::request_workspace_bytes_with_staging(
                    32, 32, 1, 32, 1, staging
                )?
            );
            let assets = crate::memory_strategy::facts(&sources)?;
            let cap = sources
                .working_set_budget()?
                .expect("explicit CPU reference ceiling");
            assert!(actual + assets.base_bytes + assets.overlay_bytes < cap);
            assert!(released + assets.base_bytes + assets.overlay_bytes > cap);
            let generator = fixture.generator(MossImageVariant::Base, Vec::new())?;
            let mut request = generation_request();
            request.width = 4096;
            request.height = 4096;
            let mut events = 0;
            let refused = generator.generate(&request, &mut |_| events += 1);
            assert!(
                matches!(refused, Err(gen_core::Error::Unsupported(message)) if message.contains("structural resident peak"))
            );
            assert_eq!(
                events, 0,
                "oversized request executed before unchanged admission guard"
            );
            Ok(())
        })
    }
    #[test]
    fn header_only_demand_profile() -> gen_core::Result<()> {
        let root = tempfile::tempdir()?;
        let fixture = Fixture::create(&root.path().join("source"))?;
        let demand = gen_core_testkit::moss_training_fixture::header_demand_profile(
            &fixture.data,
            &root.path().join("output"),
            |variant, req| fixture.memory_plan(variant, req),
        )?;
        println!("MOSS_FIXTURE_DEMAND={demand}");
        Ok(())
    }

    #[test]
    #[ignore = "manual MOSS CPU model proof; CI lanes=moss-cpu-fixture"]
    fn actual_ragged_token_bound_short_long_mixed_preview_trigger_and_truncation(
    ) -> gen_core::Result<()> {
        with_cpu(|| {
            let root = tempfile::tempdir()?;
            let fixture = Fixture::create(&root.path().join("source"))?;
            gen_core_testkit::moss_training_fixture::header_demand_profile(
                &fixture.data,
                &root.path().join("output"),
                |variant, req| fixture.memory_plan(variant, req),
            )?;
            let sources = fixture.sources()?;
            let encoder =
                crate::text_encoder::Encoder::load(&sources, &gen_core::CancelFlag::new())?;
            gen_core_testkit::moss_training_fixture::assert_ragged_token_bounds(
                &fixture.data,
                &root.path().join("output"),
                |req| {
                    crate::text_encoder::token_bound(&sources, req, HOST_CAP_BYTES)
                        .map_err(Into::into)
                },
                |req| {
                    let prompts =
                        gen_core_moss_image::moss_image_training::recovery::conditioning_prompts(
                            req,
                        );
                    prompts
                        .iter()
                        .map(|prompt| {
                            let hidden = encoder.encode(prompt, 4096, &req.cancel)?.0;
                            u64::try_from(hidden.shape()[1]).map_err(|_| {
                                gen_core::Error::Msg("invalid actual hidden shape".into())
                            })
                        })
                        .collect()
                },
            )
        })
    }
}
