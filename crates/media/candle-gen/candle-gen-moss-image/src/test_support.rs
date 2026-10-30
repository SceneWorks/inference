//! Explicit development-only CPU full-pipeline fixtures. Runtime bundles never enable this module.
//! Factories construct the REAL trainer/generator below their unchanged official asset loaders.
use crate::loader::{Asset, Sources};
use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, AdapterSpec, Generator, LoadSpec, PinnedWeightsFile, Trainer, WeightsSource,
};
use gen_core_moss_image::moss_image::*;
use std::{collections::BTreeMap, path::Path};
pub const HOST_CAP_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub use gen_core_testkit::moss_training_fixture::Fixture as HostFixture;

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
    fn sources(
        &self,
        variant: MossImageVariant,
        adapters: Vec<AdapterSpec>,
    ) -> gen_core::Result<Sources> {
        self.data.verify()?;
        let spec =
            LoadSpec::new(WeightsSource::Dir(self.data.root.clone())).with_adapters(adapters);
        let mut ordinary = Vec::new();
        let mut headers = BTreeMap::new();
        for relative in self.data.files.keys() {
            let pin = PinnedWeightsFile::pin(self.data.root.join(relative))?;
            if relative.ends_with(".safetensors") {
                headers.insert(
                    *pin.content_sha256(),
                    gen_core::weightsmeta::safetensors_path_tensor_headers(pin.loader_path())?,
                );
            }
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
        let tokenizer = ordinary
            .iter()
            .find(|a| a.role == MossImageAssetRole::Tokenizer)
            .ok_or_else(|| gen_core::Error::Msg("fixture tokenizer missing".into()))?
            .clone();
        let text_config = candle_llm::models::qwen35::Qwen35Config::from_json(
            &serde_json::from_slice(&std::fs::read(
                self.data.root.join("text_encoder/config.json"),
            )?)
            .map_err(|e| gen_core::Error::Msg(e.to_string()))?,
        )
        .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
        let adapter_pins = spec
            .adapters
            .iter()
            .map(|a| spec.file_pin_for(&a.path))
            .collect::<gen_core::Result<Vec<_>>>()?;
        Ok(Sources {
            variant,
            adapter_pins,
            selection: Some(MossImageAcceleration::Off),
            ordinary,
            mixed: None,
            full: None,
            config: self.data.config.clone(),
            text_config,
            tokenizer,
            spec,
            headers,
        })
    }
    pub fn trainer(&self, variant: MossImageVariant) -> gen_core::Result<Box<dyn Trainer>> {
        let mut trainer = crate::training::MossImageTrainer::from_sources(
            variant,
            self.sources(variant, Vec::new())?,
            Device::Cpu,
        )?;
        // Explicit CPU oracle ceiling, not a CUDA physical budget or production cap transport.
        trainer.set_memory_budget_override(Some(HOST_CAP_BYTES));
        Ok(Box::new(trainer))
    }
    pub fn trainer_with_probe(
        &self,
        variant: MossImageVariant,
        probe: gen_core_testkit::moss_training_probe::TrainingProbe,
    ) -> gen_core::Result<Box<dyn Trainer>> {
        let mut trainer = crate::training::MossImageTrainer::from_sources(
            variant,
            self.sources(variant, Vec::new())?,
            Device::Cpu,
        )?;
        trainer.set_memory_budget_override(Some(HOST_CAP_BYTES));
        trainer.set_test_probe(Some(probe));
        Ok(Box::new(trainer))
    }
    pub fn memory_plan(
        &self,
        variant: MossImageVariant,
        req: &gen_core::TrainingRequest,
    ) -> gen_core::Result<serde_json::Value> {
        let mut trainer = crate::training::MossImageTrainer::from_sources(
            variant,
            self.sources(variant, Vec::new())?,
            Device::Cpu,
        )?;
        trainer.set_memory_budget_override(Some(HOST_CAP_BYTES));
        trainer.set_test_probe(Some(
            gen_core_testkit::moss_training_probe::TrainingProbe::new(
                self.data.evaluation_manifest()?,
            ),
        ));
        trainer.test_memory_plan(req).map_err(Into::into)
    }
    /// FP32 witness through the SAME public advanced Generator path, not decoded uint8 pixels.
    pub fn generate_latents(
        &self,
        variant: MossImageVariant,
        adapters: Vec<AdapterSpec>,
        request: &gen_core::GenerationRequest,
    ) -> gen_core::Result<(Vec<f32>, [usize; 4], Vec<gen_core::AdapterApplyReport>)> {
        let model = crate::MossImageGenerator::from_sources(
            variant,
            self.sources(variant, adapters)?,
            Device::Cpu,
        )?;
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
    }
    pub fn generator(
        &self,
        variant: MossImageVariant,
        adapters: Vec<AdapterSpec>,
    ) -> gen_core::Result<Box<dyn Generator>> {
        Ok(Box::new(crate::MossImageGenerator::from_sources(
            variant,
            self.sources(variant, adapters)?,
            Device::Cpu,
        )?))
    }
}
/// Existing public generation controls select the small canvas/context; no new tiny request switch.
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
        let root = tempfile::tempdir()?;
        let fixture = Fixture::create(&root.path().join("source"))?;
        gen_core_testkit::moss_training_fixture::header_demand_profile(
            &fixture.data,
            &root.path().join("output"),
            |variant, req| fixture.memory_plan(variant, req),
        )?;
        let sources = fixture.sources(MossImageVariant::Base, Vec::new())?;
        let encoder = sources.text_cancellable(&Device::Cpu, &gen_core::CancelFlag::new())?;
        gen_core_testkit::moss_training_fixture::assert_ragged_token_bounds(
            &fixture.data,
            &root.path().join("output"),
            |req| {
                crate::text_encoder::token_bound(&sources, req, HOST_CAP_BYTES).map_err(Into::into)
            },
            |req| {
                let prompts =
                    gen_core_moss_image::moss_image_training::recovery::conditioning_prompts(req);
                let rows = crate::text_encoder::strings(
                    &sources,
                    &encoder,
                    &prompts,
                    4096,
                    &Device::Cpu,
                    &req.cancel,
                )?
                .rows;
                rows.iter()
                    .map(|row| {
                        row.dim(1)
                            .map(|n| n as u64)
                            .map_err(|e| gen_core::Error::Msg(e.to_string()))
                    })
                    .collect()
            },
        )
    }
}
