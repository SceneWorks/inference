//! # mlx-gen-sdxl
//!
//! The **Stable Diffusion XL** provider crate for [`mlx-gen`](mlx_gen). SDXL is a **U-Net**
//! generator (not a DiT like Z-Image/FLUX/Qwen), brought into Rust from Apple's vendored
//! `mlx-examples/stable_diffusion` path (`_vendor/mlx_sd/`, MIT) plus SceneWorks' LoRA merge — the
//! last Python image-inference path (sc-2400, epic 2337).
//!
//! Depends only on the `mlx-gen` core (nn primitives, adapters, weights, quant, the `Generator`
//! contract, and the registry). Its registration constant is included by this crate's explicit
//! family catalog. The port reuses the core conv primitives already built for
//! the Z-Image VAE (`conv2d`, pytorch-compatible `group_norm`, `silu`, `upsample_nearest`) and the
//! shared `image`/`weights`/`quant`/`adapters` layers; it adds the SDXL-specific surfaces: the
//! `UNet2DConditionModel` (down/mid/up cross-attention blocks + time/`text_time` micro-conditioning
//! embeddings), the dual CLIP-L + OpenCLIP-bigG text encoders and their CLIP-BPE tokenizer, the
//! SDXL VAE, and the discrete Euler / Euler-Ancestral sampler with real classifier-free guidance.
//!
//! Parity target = the vendored fp16 reference (`StableDiffusionXL.generate_latents`), validated
//! stage-by-stage against goldens (see `tools/dump_sdxl_golden.py`).

pub mod adapters;
pub(crate) mod block_stream;
pub mod config;
pub mod convert;
pub mod inpaint;
pub mod ip_adapter;
pub mod ldm;
pub mod loader;
pub mod memory_strategy;
pub mod model;
pub mod pipeline;
pub mod plan;
pub mod preview;
pub mod quant;
pub mod sampler;
pub mod text_encoder;
pub mod tokenizer;
pub mod training;
pub mod unet;
pub mod vae;
pub mod vision_encoder;

pub use adapters::{
    apply_sdxl_adapters, apply_sdxl_adapters_with, lora_delta, LoraCoverage, SdxlLoraReport,
};
pub use config::{
    BetaSchedule, ClipActivation, ClipTextConfig, DiffusionConfig, UNetConfig, VaeConfig,
};
pub use inpaint::{preprocess_mask, InpaintBlend};
pub use ip_adapter::{
    load_ip_kv_pairs, preprocess_clip_image, preprocess_clip_image_sized, IpImageEncoder,
    Resampler, ResamplerConfig,
};
pub use loader::{
    load_controlnet, load_ip_adapter, load_text_encoder_1, load_text_encoder_1_dtype,
    load_text_encoder_2, load_text_encoder_2_dtype, load_tokenizer, load_unet, load_unet_dtype,
    load_unet_kolors_dtype, load_unet_with_config, load_vae, resolve_unet_weight_file,
    resolve_vae_weight_file,
};
pub use model::{
    descriptor, load, load_concrete, load_from_ldm_file, DecodeQualitySample, Sdxl, MODEL_ID,
    PID_BACKBONE, SIZE_MULTIPLE,
};
pub use pipeline::{
    decode_image, decode_image_tiled, decoded_to_image, denoise, denoise_cfgpp,
    denoise_cfgpp_with_preview, denoise_control, denoise_control_with_preview, denoise_curated,
    denoise_curated_with_preview, denoise_inpaint, denoise_inpaint_with_preview, denoise_ip,
    denoise_ip_control, denoise_ip_control_with_preview, denoise_ip_multi_control,
    denoise_ip_multi_control_with_preview, denoise_ip_with_preview, denoise_multi_control,
    denoise_multi_control_with_preview, denoise_with_preview, encode_conditioning,
    encode_init_latents, preprocess_control_image, preprocess_init_image, seeded_prior,
    text_time_ids, ControlContext, Denoiser,
};
pub use plan::{SdxlBlockWindow, SdxlForwardPlan};
pub use sampler::EulerSampler;
pub use text_encoder::{ClipOutput, ClipTextEncoder};
pub use tokenizer::{ClipBpeTokenizer, PAD_ID};
pub use training::family::{train_family, SdxlFamilyHooks, TrainTimestep};
pub use training::{load_trainer, SdxlTrainer};
pub use unet::{ControlNet, ControlResiduals, Transformer2D, UNet2DConditionModel};
pub use vae::{Autoencoder, SdxlLatentDecoder};
pub use vision_encoder::{ClipVisionEncoder, VisionConfig};

/// Shared-optimization toggles whose production call sites this provider can actually execute.
/// Availability never substitutes for the request-local `Applied` receipt required by P6.
pub const BENCHMARK_TOGGLE_CAPABILITIES: &[&str] = &[
    mlx_gen::diagnostics::RETAINED_COMPILATION,
    mlx_gen::diagnostics::GEOMETRY_AWARE_DECODE,
];

// sc-2963 compiled-glue toggle: when on, the UNet's remaining fusable elementwise glue — the **SiLU**
// activations (`x·sigmoid(x)`: ResNet GN→SiLU, the time-embedding MLP, the output head) — runs through
// `mx.compile`, fusing each into one kernel. The GEGLU/erf-GELU activations are already `mx.compile`'d
// in core `nn`, so SiLU is the only chain left.
//
// ⚠️ SDXL is **fp16 and precision-load-bearing** ([[sdxl-fp16-sc2721]]): the reference runs SiLU eager,
// so the fp16 golden matches **eager** SiLU — fusing SiLU is only safe because it is **bit-identical**
// to eager (`tests/compile_parity.rs` proves `max|Δ|=0` in fp16 AND f32). **Enabled by the production
// denoise loop** ([`pipeline::denoise`]); **off by default**.
//
// The toggle + its RAII [`CompileGlueGuard`] are hoisted into core (F-104); re-export core's so the
// request/thread-local setting is shared with the FLUX family.
pub(crate) use mlx_gen::nn::compile_glue;
pub use mlx_gen::nn::{set_compile_glue, CompileGlueGuard};

const SITE_SILU_GLUE: &str = "sdxl::silu_glue";

fn silu_glue_impl(
    x: &mlx_rs::Array,
) -> std::result::Result<mlx_rs::Array, mlx_rs::error::Exception> {
    mlx_rs::ops::multiply(x, &mlx_rs::ops::sigmoid(x)?)
}

thread_local! {
    static RETAINED_SILU_GLUE: std::cell::RefCell<Option<mlx_gen::nn::RetainedUnary>> =
        const { std::cell::RefCell::new(None) };
}

fn retained_silu_glue(
    x: &mlx_rs::Array,
) -> std::result::Result<mlx_rs::Array, mlx_rs::error::Exception> {
    mlx_gen::nn::prepare_retained_compilation_thread();
    RETAINED_SILU_GLUE.with(|slot| {
        slot.borrow_mut()
            .get_or_insert_with(|| {
                mlx_gen::nn::RetainedUnary::new(mlx_rs::transforms::compile::compile_retained(
                    silu_glue_impl,
                    true,
                ))
            })
            .call(SITE_SILU_GLUE, x)
    })
}

/// Exercise this crate's production retained handle once for the release memory audit.
#[doc(hidden)]
pub fn exercise_retained_compile_inventory(input: &mlx_rs::Array) -> mlx_gen::Result<()> {
    let output = retained_silu_glue(input)?;
    output.eval()?;
    drop(output);
    Ok(())
}

/// Add the MLX SDXL generator and trainer to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_activation_memory(model::ACTIVATION_MEMORY_REGISTRATION)
        .register_memory_strategy(model::MEMORY_REGISTRATION)
        .register_memory_behavior(model::MEMORY_BEHAVIOR_REGISTRATION)
        .register_trainer(training::TRAINER_REGISTRATION)
}

/// Build the complete explicit MLX SDXL provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit_generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit_generators, ["sdxl"]);
        assert_eq!(explicit_trainers, ["sdxl"]);
    }

    /// Weights-free behavioral oracle for the shared memory ladder (SC-15525).
    ///
    /// This is the check that makes the declaration non-vacuous without weights: for **every**
    /// declared rung it builds the provider's own representative selection, drives the whole request
    /// scope through it (`configure_request` → phases → `configure_decode` / `configure_attention` /
    /// `materialize_transformer_window` → `finish`), proves the safety check is not blind to an
    /// impossible budget, and — because this contract declares native/PiD decode routes — proves
    /// each route's geometry is accepted on its own route and **rejected on the other**, with the
    /// matching-route controls that keep the rejection non-vacuous.
    #[test]
    fn shared_ladder_registrations_pass_the_weights_free_behavior_oracle() {
        let registry = super::provider_registry().unwrap();
        for shape in [
            mlx_gen::LoadShape::DeferredMaterialization,
            mlx_gen::LoadShape::EagerMaterialization,
        ] {
            let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
                .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
                .with_load_shape(shape);
            gen_core_testkit::memory_strategy::memory_strategy_registry_conformance(
                &registry, &spec,
            );
        }
    }
}

/// SiLU `x·sigmoid(x)` — one fused kernel when the sc-2963 glue toggle is on, else the eager core
/// [`mlx_gen::nn::silu`]. Bit-identical to eager in fp16 AND f32 (proven `max|Δ|=0`,
/// `tests/compile_parity.rs`), so it is golden-safe on the precision-load-bearing fp16 UNet.
pub(crate) fn silu_glue(x: &mlx_rs::Array) -> mlx_gen::Result<mlx_rs::Array> {
    if !compile_glue() {
        mlx_gen::diagnostics::record_fallback(SITE_SILU_GLUE, "compiled_glue_disabled");
        return mlx_gen::nn::silu(x);
    }
    if mlx_gen::nn::retained_compilation_requested() {
        Ok(retained_silu_glue(x)?)
    } else {
        mlx_gen::diagnostics::record_compile(
            SITE_SILU_GLUE,
            mlx_gen::diagnostics::CompileDisposition::OneShot,
        );
        Ok(mlx_rs::transforms::compile::compile(silu_glue_impl, true)(
            x,
        )?)
    }
}

#[cfg(test)]
mod sc2963 {
    use mlx_rs::{random, Array, Dtype};

    fn max_abs(a: &Array, b: &Array) -> f32 {
        let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
        mlx_rs::ops::max(&d, None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>()
    }

    // sc-2963 invariant: the compiled SiLU is **bit-identical** to the eager core SiLU in **fp16**
    // (the precision-load-bearing UNet dtype) AND f32 — `max|Δ|=0`. SDXL's fp16 golden matches the
    // eager (reference) SiLU, so a non-zero gap here would mean enabling compile regresses the golden
    // (it doesn't — unlike the erf-GELU chain, the fused `sigmoid`+`multiply` rounds identically).
    #[test]
    fn compiled_silu_bit_identical_to_eager_fp16_and_f32() {
        let k = random::key(0).unwrap();
        for dt in [Dtype::Float16, Dtype::Float32] {
            let x = random::normal::<f32>(&[4, 64, 64, 320], None, None, Some(&k))
                .unwrap()
                .as_dtype(dt)
                .unwrap();
            super::set_compile_glue(false);
            let eager = super::silu_glue(&x).unwrap();
            super::set_compile_glue(true);
            let compiled = super::silu_glue(&x).unwrap();
            super::set_compile_glue(false);
            assert_eq!(compiled.dtype(), dt, "silu_glue preserves dtype {dt:?}");
            let d = max_abs(&compiled, &eager);
            let rel = mlx_gen::nn::max_rel_diff(&compiled, &eager);
            println!("[sdxl silu {dt:?}] max|Δ|={d:.3e} rel|Δ|={rel:.3e}");
            // sc-12747: under MLX 0.32.0 the compiled SiLU rounds ~1 ULP-f32 differently from eager
            // (0-ULP on the prior 0.31.2 pin); fp16 (the precision-load-bearing UNet dtype whose
            // golden matches eager SiLU) stays bit-identical. f32 takes the shared re-baselined
            // tolerance; fp16 stays exact.
            let tol = if dt == Dtype::Float32 {
                mlx_gen::nn::COMPILED_GLUE_F32_ULP_TOL
            } else {
                0.0
            };
            assert!(
                rel <= tol,
                "SDXL compiled SiLU diverged from eager in {dt:?}: rel|Δ|={rel:e} exceeds {tol:e}"
            );
        }
    }

    #[test]
    fn retained_silu_reuses_across_requests_and_baseline_stays_oneshot() {
        use mlx_gen::diagnostics::{
            self, CompileDisposition, DiagnosticCounter, ToggleDisposition, RETAINED_COMPILATION,
        };

        super::RETAINED_SILU_GLUE.with(|slot| *slot.borrow_mut() = None);
        let x = random::normal::<f32>(&[1, 8, 8, 16], None, None, Some(&random::key(1).unwrap()))
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        super::set_compile_glue(false);
        let eager = super::silu_glue(&x).unwrap();

        super::set_compile_glue(true);
        let scope = diagnostics::begin_request_with_toggles(
            "sdxl-retained-silu",
            "sdxl",
            &[RETAINED_COMPILATION],
        )
        .unwrap();
        for _ in 0..2 {
            assert_eq!(max_abs(&super::silu_glue(&x).unwrap(), &eager), 0.0);
        }
        let report = scope.finish();
        for disposition in [
            CompileDisposition::RetainedMiss,
            CompileDisposition::RetainedHit,
        ] {
            assert!(report.counters.iter().any(|counter| matches!(
                counter,
                DiagnosticCounter::Compile {
                    site: super::SITE_SILU_GLUE,
                    disposition: recorded,
                    count: 1,
                } if *recorded == disposition
            )));
        }
        assert!(report.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Toggle {
                toggle: RETAINED_COMPILATION,
                disposition: ToggleDisposition::Applied,
                count: 2,
            }
        )));

        let next = diagnostics::begin_request_with_toggles(
            "sdxl-retained-silu-next",
            "sdxl",
            &[RETAINED_COMPILATION],
        )
        .unwrap();
        assert_eq!(max_abs(&super::silu_glue(&x).unwrap(), &eager), 0.0);
        let next = next.finish();
        assert!(next.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Compile {
                site: super::SITE_SILU_GLUE,
                disposition: CompileDisposition::RetainedHit,
                count: 1,
            }
        )));

        let baseline = diagnostics::begin_request("sdxl-oneshot-silu", "sdxl").unwrap();
        assert_eq!(max_abs(&super::silu_glue(&x).unwrap(), &eager), 0.0);
        let baseline = baseline.finish();
        assert!(baseline.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Compile {
                site: super::SITE_SILU_GLUE,
                disposition: CompileDisposition::OneShot,
                count: 1,
            }
        )));

        super::set_compile_glue(false);
        super::RETAINED_SILU_GLUE.with(|slot| *slot.borrow_mut() = None);
    }
}
