//! Frozen edit package loading, deterministic reference encoding and request-local sampling.
use crate::{
    loader::{read_weights, Sources},
    pipeline::{check, digest_f32, logical_f32_readback},
    transformer::Transformer,
};
use mlx_gen::gen_core::{
    moss_image::*,
    moss_image_edit::*,
    weightsmeta::{CheckpointMeta, Dtype as StoredDtype},
    PinnedWeightsFile,
};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LoadSpec, Progress, Result, WeightsSource};
use mlx_gen_flux2::vae::{Flux2Vae, Flux2VaeDecoderConfig};
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::{Array, Dtype};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, path::Path};

pub(crate) struct EditAssets {
    config: PinnedWeightsFile,
    weights: PinnedWeightsFile,
}
fn pin(spec: &LoadSpec, path: &Path, digest: &str) -> Result<PinnedWeightsFile> {
    let path = std::path::absolute(path)?;
    let pin = if spec.prepared_file_pins().is_prepared() {
        spec.prepared_file_pins()
            .get(&path)
            .cloned()
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "MOSS edit prepared identity missing {}",
                    path.display()
                ))
            })?
    } else {
        PinnedWeightsFile::pin(&path)?
    };
    if pin.loader_path() != path || hex(pin.content_sha256()) != digest {
        return Err(Error::Unsupported(format!(
            "MOSS edit frozen package hash mismatch: {}",
            path.display()
        )));
    }
    pin.verify_unchanged()?;
    Ok(pin)
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn authenticated_bytes(pin: &PinnedWeightsFile, digest: &str) -> Result<Vec<u8>> {
    pin.read_unchanged(|path| {
        let bytes = std::fs::read(path)?;
        if hex(&Sha256::digest(&bytes)) != digest {
            return Err(mlx_gen::gen_core::Error::Unsupported(
                "consumed MOSS edit bytes changed".into(),
            ));
        }
        Ok(bytes)
    })
    .map_err(Into::into)
}
impl EditAssets {
    pub(crate) fn prepare(spec: &LoadSpec) -> Result<Self> {
        let root = match spec.components.get(EDIT_COMPONENT) {
            Some(WeightsSource::Dir(path)) => path,
            _ => {
                return Err(Error::Unsupported(
                    "MOSS edit requires the exact moss-image-edit package directory".into(),
                ))
            }
        };
        let config = pin(spec, &root.join("adapter_config.json"), EDIT_CONFIG_SHA256)?;
        let weights = pin(
            spec,
            &root.join("adapter_model.safetensors"),
            EDIT_WEIGHTS_SHA256,
        )?;
        // The exact pinned config is the authority, including all 252 module options/targets.
        let value: serde_json::Value =
            serde_json::from_slice(&authenticated_bytes(&config, EDIT_CONFIG_SHA256)?)
                .map_err(|e| Error::Msg(e.to_string()))?;
        validate_edit_config(&value)?;
        Ok(Self { config, weights })
    }
    pub(crate) fn path(&self) -> &Path {
        self.weights.loader_path()
    }
    pub(crate) fn ensure_unchanged(&self) -> Result<()> {
        self.config.verify_unchanged()?;
        self.weights.verify_unchanged()?;
        Ok(())
    }
    pub(crate) fn install(&self, transformer: &mut Transformer, cancel: &CancelFlag) -> Result<()> {
        check(cancel)?;
        let meta =
            CheckpointMeta::from_bytes(authenticated_bytes(&self.weights, EDIT_WEIGHTS_SHA256)?)?;
        let headers = meta
            .keys()
            .map(|name| {
                let t = meta.tensor(name).expect("indexed edit tensor");
                mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader {
                    name: name.into(),
                    shape: t.shape.to_vec(),
                    dtype: t.dtype,
                    data_bytes: t.data.len() as u64,
                }
            })
            .collect::<Vec<_>>();
        validate_edit_adapter_headers(&headers)?;
        let expected = edit_adapter_shapes();
        if meta.keys().count() != expected.len() {
            return Err(Error::Unsupported(
                "MOSS edit requires exactly 504 A/B tensors".into(),
            ));
        }
        let mut arrays = HashMap::new();
        for (name, shape) in &expected {
            check(cancel)?;
            let t = meta
                .tensor(name)
                .ok_or_else(|| Error::MissingTensor(name.clone()))?;
            if t.dtype != StoredDtype::BF16 || t.shape != shape.as_slice() {
                return Err(Error::Unsupported(format!(
                    "MOSS edit tensor shape/dtype mismatch: {name}"
                )));
            }
            let shape: Vec<i32> = shape.iter().map(|v| *v as i32).collect();
            // SAFETY: authenticated owned safetensors validates dtype, shape and payload size;
            // MLX copies the source bytes into its own retained allocation.
            let array =
                unsafe { Array::from_raw_data(t.data.as_ptr().cast(), &shape, Dtype::Bfloat16) };
            arrays.insert(name.clone(), array);
        }
        let weights = Weights::from_map(arrays);
        transformer.install_edit(&weights, 64)?;
        if !weights.unused_keys().is_empty() {
            return Err(Error::Unsupported(
                "unconsumed MOSS edit projections".into(),
            ));
        }
        transformer.materialize_owned(cancel)?;
        self.ensure_unchanged()
    }
}
pub(crate) fn extra_peak(
    inputs: MossImageEditInputs<'_>,
    request: &mlx_gen::GenerationRequest,
) -> Result<u64> {
    let references = MossEditReferenceMemory::checked(inputs)?;
    let reference_tokens = references.tokens;
    let total = references
        .persistent_bytes
        .checked_add(references.encoder_bytes)
        .ok_or_else(|| Error::Unsupported("MOSS edit reference bound overflow".into()))?;
    // Edit-only biased projections cast temporarily, never retaining an FP32 backbone.
    // Bound the text embedder plus one evaluated text block: text*d + 4*d*d + 3*d*f
    // weight elements, plus both latent projections (2*latent*d). This exceeds
    // the complete timestep path (freq*d + 9*d*d).
    // Scratch counts input, dot and biased output for every projection conservatively;
    // text blocks evaluate before the next block, so their cast weights do not overlap.
    const D: u64 = 3072;
    const F: u64 = 8192;
    const TEXT: u64 = 4096;
    const LATENT: u64 = 128;
    let text_tokens = request
        .moss_image
        .as_ref()
        .map_or(4096, |o| o.max_sequence_length) as u64;
    let image_tokens = (request.width as u64 * request.height as u64 / 256).max(reference_tokens);
    let precise_bytes = text_tokens
        .checked_mul(TEXT + 16 * D + 5 * F)
        .and_then(|n| {
            image_tokens
                .checked_mul(3 * D + 3 * LATENT)
                .and_then(|m| n.checked_add(m))
        })
        .and_then(|n| {
            n.checked_add(TEXT * D + 4 * D * D + 3 * D * F + 2 * LATENT * D + 256 + 23 * D)
        })
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Unsupported("MOSS edit precise projection bound overflow".into()))?;
    // The ordinary workspace reserves one full F32 B*B matrix. Extend it to
    // the concatenated reference/body geometry, including all cross-reference terms.
    // Keep the legacy full-matrix reserve, but explicitly price the new bounded
    // source-faithful BF16 attention arrays if they exceed it (not a second full matrix).
    let body_tokens = text_tokens
        .checked_add(request.width as u64 * request.height as u64 / 256)
        .ok_or_else(|| Error::Unsupported("MOSS edit body scratch overflow".into()))?;
    // Ordinary scratch scales with body tokens. Reference blocks run on the whole
    // prefix; reserve the excess linear/FFN activations when R exceeds B.
    let reference_work = reference_tokens
        .saturating_sub(body_tokens)
        .checked_mul(4 * (12 * D + 4 * F))
        .ok_or_else(|| Error::Unsupported("MOSS edit reference scratch overflow".into()))?;
    let old_attention = references
        .attention_elements(body_tokens)?
        .checked_mul(24 * 4)
        .ok_or_else(|| Error::Unsupported("MOSS edit attention scratch overflow".into()))?;
    let combined_keys = body_tokens
        .checked_add(reference_tokens)
        .ok_or_else(|| Error::Unsupported("MOSS edit attention key overflow".into()))?;
    let attention_workspace = [
        (reference_tokens, reference_tokens),
        (body_tokens, combined_keys),
        (text_tokens, text_tokens),
    ]
    .into_iter()
    .try_fold(0, |peak, (rows, keys)| {
        crate::precision::edit_attention_workspace(rows, keys, D, 24)
            .map(|bytes| peak.max(bytes))
            .ok_or_else(|| Error::Unsupported("MOSS edit tiled attention bound overflow".into()))
    })?;
    let ordinary_attention = body_tokens
        .checked_mul(body_tokens)
        .and_then(|n| n.checked_mul(24 * 4))
        .ok_or_else(|| Error::Unsupported("MOSS edit attention scratch overflow".into()))?;
    let attention_bytes = old_attention
        .max(attention_workspace)
        .checked_sub(ordinary_attention)
        .ok_or_else(|| Error::Unsupported("MOSS edit attention scratch overflow".into()))?;

    // Edit SiTU peaks at 18 bytes/FFN channel: borrowed BF16 hidden/gate (4),
    // gated branch (2), rounded division (2), widened unary input/output (8),
    // rounded unary output (2). Existing linear scratch reserves 4*F*4 = 16.
    let situ_extra = body_tokens
        .max(reference_tokens)
        .checked_mul(2 * F)
        .ok_or_else(|| Error::Unsupported("MOSS edit SiTU bound overflow".into()))?;
    total
        .checked_add(situ_extra)
        .and_then(|n| n.checked_add(precise_bytes))
        .and_then(|n| n.checked_add(reference_work))
        .and_then(|n| n.checked_add(attention_bytes))
        .and_then(|n| n.checked_add(2_000_000_000 + 297_271_296 * 2))
        .ok_or_else(|| Error::Unsupported("MOSS edit bound overflow".into()))
}
pub(crate) fn encode_references(
    sources: &Sources,
    inputs: MossImageEditInputs<'_>,
    cancel: &CancelFlag,
) -> Result<(Vec<Array>, Vec<MossImageEditReferenceReceipt>)> {
    check(cancel)?;
    let weights = read_weights(
        sources
            .ordinary
            .iter()
            .filter(|a| a.relative == "vae/diffusion_pytorch_model.safetensors"),
        |k| {
            k.starts_with("encoder.")
                || k.starts_with("decoder.")
                || k.starts_with("quant_conv.")
                || k.starts_with("post_quant_conv.")
                || k == "bn.running_mean"
                || k == "bn.running_var"
        },
        cancel,
    )?;
    let vae = Flux2Vae::from_weights_with_decoder_config(
        &weights,
        &Flux2VaeDecoderConfig {
            block_out_channels: [96, 192, 384, 384],
        },
    )?;
    weights.materialize_accessed()?;
    if !weights.unused_keys().is_empty() {
        return Err(Error::Unsupported(format!(
            "unconsumed MOSS edit VAE weights: {:?}",
            weights.unused_keys()
        )));
    }
    let mut references = Vec::new();
    let mut receipts = Vec::new();
    for original in inputs.references {
        check(cancel)?;
        let image = resize_reference(original, inputs.reference_max_edge)?;
        let values: Vec<f32> = image
            .pixels
            .iter()
            .map(|p| *p as f32 / 127.5 - 1.0)
            .collect();
        let array = Array::from_slice(&values, &[1, image.height as i32, image.width as i32, 3]);
        let latent = reference_latent(&vae, &array, cancel)?;
        let values = logical_f32_readback(&latent)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Msg("nonfinite MOSS reference encoding".into()));
        }
        receipts.push(MossImageEditReferenceReceipt {
            original_dimensions: [original.width, original.height],
            encoded_dimensions: [image.width, image.height],
            pixels_sha256: hex(&Sha256::digest(&original.pixels)),
            resized_pixels_sha256: hex(&Sha256::digest(&image.pixels)),
            latent_sha256: digest_f32(&values),
        });
        references.push(latent);
    }
    check(cancel)?;
    Ok((references, receipts))
}
fn reference_latent(vae: &Flux2Vae, image: &Array, cancel: &CancelFlag) -> Result<Array> {
    check(cancel)?;
    let mean = vae.encode_mean_precise(image)?;
    mlx_rs::transforms::eval([&mean])?;
    check(cancel)?;
    let s = mean.shape();
    let packed = mean
        .transpose_axes(&[0, 3, 1, 2])?
        .reshape(&[1, s[3], s[1] / 2, 2, s[2] / 2, 2])?
        .transpose_axes(&[0, 1, 3, 5, 2, 4])?
        .reshape(&[1, 4 * s[3], s[1] / 2, s[2] / 2])?;
    let latent = vae.bn_normalize_nchw(&packed)?;
    mlx_rs::transforms::eval([&latent])?;
    check(cancel)?;
    Ok(latent)
}
/// Shared production CFG/Euler update. Network predictions may be BF16, but guidance,
/// integration and the returned accumulator always use FP32.
fn cfg_euler_step(
    current: &Array,
    conditional: &Array,
    unconditional: Option<&Array>,
    guidance: f32,
    delta_sigma: f32,
) -> Result<Array> {
    let scalar = |x: f32| Array::from_slice(&[x], &[1]);
    let mut prediction = conditional.as_dtype(Dtype::Float32)?;
    if let Some(unconditional) = unconditional {
        let negative = unconditional.as_dtype(Dtype::Float32)?;
        prediction = add(
            &negative,
            multiply(subtract(prediction, &negative)?, scalar(guidance))?,
        )?;
    }
    Ok(add(
        current.as_dtype(Dtype::Float32)?,
        multiply(prediction, scalar(delta_sigma))?,
    )?)
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static SAMPLING_TRACE: std::cell::RefCell<Option<Vec<[Array; 4]>>> = const { std::cell::RefCell::new(None) };
}

pub(crate) struct EditSampling<'a> {
    pub transformer: &'a Transformer,
    pub resolved: &'a MossImageResolved,
    pub positive: &'a [Array],
    pub negative: &'a [Array],
    pub references: &'a [Array],
    pub use_cache: bool,
}
impl EditSampling<'_> {
    pub(crate) fn run(
        &self,
        noise: &Array,
        cancel: &CancelFlag,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let cache = if self.use_cache {
            Some(
                self.transformer
                    .prepare_edit_cache(self.references, cancel)?,
            )
        } else {
            None
        };
        let sigmas = moss_image_schedule(
            self.resolved.steps,
            self.resolved.shift,
            self.resolved.training_shift,
        )?;
        let mut latents = noise.as_dtype(Dtype::Float32)?;
        for (index, pair) in sigmas.windows(2).enumerate() {
            let mut outputs = Vec::new();
            for (sample, positive) in self.positive.iter().enumerate() {
                check(cancel)?;
                let current = mlx_gen::attention::slice_axis(&latents, 0, sample as i32, 1)?;
                let predict = |context: &Array| -> Result<Array> {
                    Ok(self
                        .transformer
                        .forward_edit(
                            &current,
                            pair[0],
                            crate::transformer::EditConditioning {
                                context,
                                references: self.references,
                                cache: cache.as_ref(),
                            },
                            cancel,
                        )?
                        .as_dtype(Dtype::Float32)?)
                };
                let prediction = predict(positive)?;
                let negative = if self.resolved.guidance != 1.0 {
                    Some(predict(&self.negative[sample])?)
                } else {
                    None
                };
                let next = cfg_euler_step(
                    &current,
                    &prediction,
                    negative.as_ref(),
                    self.resolved.guidance,
                    pair[1] - pair[0],
                )?;
                mlx_rs::transforms::eval([&next])?;
                #[cfg(test)]
                SAMPLING_TRACE.with(|trace| {
                    if let Some(trace) = trace.borrow_mut().as_mut() {
                        trace.push([
                            current.clone(),
                            prediction.clone(),
                            negative.as_ref().unwrap().clone(),
                            next.clone(),
                        ]);
                    }
                });
                outputs.push(next);
            }
            latents = concatenate_axis(&outputs.iter().collect::<Vec<_>>(), 0)?;
            check(cancel)?;
            progress(Progress::Step {
                current: index as u32 + 1,
                total: sigmas.len() as u32 - 1,
            });
        }
        check(cancel)?;
        Ok(latents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    #[test]
    fn frozen_cfg_euler_steps_are_fp32_cpu() {
        const CHILD: &str = "MOSS_EDIT_SAMPLING_CPU_CHILD";
        if std::env::var(CHILD).as_deref() != Ok("1") {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "edit::tests::frozen_cfg_euler_steps_are_fp32_cpu",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "CPU sampling child: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
            let fixture: Value = serde_json::from_str(include_str!(
                "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v2/fixture.json"
            ))
            .unwrap();
            // Source predictions are retained independently of the native network.
            // This tests the production update itself without a BF16 network tolerance.
            const FP32_BOUND: f32 = 1e-6;
            for case in fixture["cases"].as_array().unwrap() {
                let sampling = &case["sampling"];
                let steps = sampling["steps"].as_array().unwrap();
                let sigmas = values(&sampling["sigmas"]["values"]);
                let mut current = array(&steps[0]["input"]);
                for (index, step) in steps.iter().enumerate() {
                    let expected = values(&step["output"]["values"]);
                    current = cfg_euler_step(
                        &current,
                        &array(&step["conditional"]),
                        Some(&array(&step["unconditional"])),
                        sampling["cfg"].as_f64().unwrap() as f32,
                        sigmas[index + 1] - sigmas[index],
                    )
                    .unwrap();
                    // Check dtype before readback, which intentionally accepts other dtypes.
                    assert_eq!(
                        current.dtype(),
                        Dtype::Float32,
                        "step {index} accumulator dtype"
                    );
                    let actual = logical_f32_readback(&current).unwrap();
                    let delta = actual
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        delta <= FP32_BOUND,
                        "step {index}: FP32 error {delta} > {FP32_BOUND}"
                    );
                    // A narrowed accumulator, even widened again before readback, must fail
                    // the very same precision bound at every retained source step.
                    let narrowed = current
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap()
                        .as_dtype(Dtype::Float32)
                        .unwrap();
                    let wrong = logical_f32_readback(&narrowed).unwrap();
                    let wrong_delta = wrong
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f32, f32::max);
                    assert!(
                        wrong_delta > 100.0 * FP32_BOUND,
                        "non-discriminating BF16 control: {wrong_delta}"
                    );
                    eprintln!(
                        "refs={} step={index} fp32_error={delta} bf16_control_error={wrong_delta}",
                        case["reference_count"]
                    );
                }
            }
        });
    }
    fn values(v: &Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }
    fn array(v: &Value) -> Array {
        let shape = v["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        Array::from_slice(&values(&v["values"]), &shape)
    }
    fn compare(actual: &Array, expected: &Value, tolerance: f32) {
        assert_eq!(actual.dtype(), Dtype::Float32);
        assert_eq!(
            actual.shape(),
            expected["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect::<Vec<_>>()
        );
        let actual = logical_f32_readback(actual).unwrap();
        let expected = values(&expected["values"]);
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("reference encoder max_abs={error}, bound={tolerance}");
        assert!(
            error <= tolerance,
            "encoder maximum absolute error {error} > {tolerance}"
        );
    }
    fn encoder_weights(fixture: &Value) -> Weights {
        let mut tensors = HashMap::new();
        for recipe in fixture["weights"].as_array().unwrap() {
            let shape = recipe["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect::<Vec<_>>();
            let count = shape.iter().map(|v| *v as usize).product::<usize>();
            let seed = recipe["seed"].as_u64().unwrap() as u32;
            let data = (0..count as u32)
                .map(|index| {
                    let mut bits = (index + 1) ^ seed;
                    bits ^= bits << 13;
                    bits ^= bits >> 17;
                    bits ^= bits << 5;
                    let signed = (bits & 65535) as i32 - 32768;
                    match recipe["kind"].as_str().unwrap() {
                        "bn_mean" => ((index % 17) as i32 - 8) as f32 / 64.0,
                        "bn_var" => 1.0 + ((index * 7) % 31) as f32 / 32.0,
                        "norm" => 1.0 + signed as f32 / 262144.0,
                        "bias" => signed as f32 / 1048576.0,
                        "weight" => signed as f32 / 8388608.0,
                        "pointwise_identity" => {
                            signed as f32 / 8388608.0
                                + if index / (shape[1] as u32) == index % (shape[1] as u32) {
                                    1.0
                                } else {
                                    0.0
                                }
                        }
                        kind => panic!("unknown encoder fixture recipe {kind}"),
                    }
                })
                .collect::<Vec<_>>();
            let name = recipe["name"].as_str().unwrap();
            assert_eq!(
                data[..data.len().min(4)],
                values(&fixture["weight_prefix_probes"][name])
            );
            tensors.insert(name.into(), Array::from_slice(&data, &shape));
        }
        Weights::from_map(tensors)
    }
    #[test]
    #[ignore = "requires coordinator-admitted Metal ownership; exact VAE activation is Metal-only"]
    fn frozen_reference_encoder_on_metal() {
        assert_eq!(
            std::env::var("MOSS_NATIVE_TEST_ADMITTED").as_deref(),
            Ok("1")
        );
        mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), || {
            let fixture: Value = serde_json::from_str(include_str!(
                "../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_encode/fixture.json"
            ))
            .unwrap();
            let weights = encoder_weights(&fixture);
            let vae = Flux2Vae::from_weights_with_decoder_config(
                &weights,
                &Flux2VaeDecoderConfig {
                    block_out_channels: [32; 4],
                },
            )
            .unwrap();
            let pixels = array(&fixture["pixels_nhwc"]);
            let tolerance = fixture["absolute_tolerance"].as_f64().unwrap() as f32;
            compare(
                &vae.encode_mean_precise(&pixels).unwrap(),
                &fixture["mean_nhwc"],
                tolerance,
            );
            let cancel = CancelFlag::new();
            let latent = reference_latent(&vae, &pixels, &cancel).unwrap();
            compare(&latent, &fixture["normalized_packed_nchw"], tolerance);
            let actual = logical_f32_readback(&latent).unwrap();
            for (name, mutant) in fixture["mutants"].as_object().unwrap() {
                let wrong = values(&mutant["values"]);
                assert_eq!(actual.len(), wrong.len());
                let delta = actual
                    .iter()
                    .zip(wrong)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                eprintln!("reference encoder control {name}: delta={delta}, bound={tolerance}");
                assert!(
                    delta > 100.0 * tolerance,
                    "undiscriminated encoder control {name}"
                );
            }
            cancel.cancel();
            assert!(matches!(
                reference_latent(&vae, &pixels, &cancel),
                Err(Error::Canceled)
            ));
        });
    }
    #[test]
    fn edit_peak_cpu_prices_reference_context_and_cold_rejection() {
        let image = mlx_gen::Image {
            width: 512,
            height: 512,
            pixels: vec![0; 512 * 512 * 3],
        };
        let references = [image];
        let inputs = MossImageEditInputs {
            references: &references,
            ..Default::default()
        };
        let mut request = mlx_gen::GenerationRequest {
            width: 512,
            height: 512,
            ..Default::default()
        };
        let full = extra_peak(inputs, &request).unwrap();
        let smaller = extra_peak(
            MossImageEditInputs {
                reference_max_edge: 256,
                ..inputs
            },
            &request,
        )
        .unwrap();
        assert!(
            full > smaller + 100_000_000,
            "reference encoder/KV must affect admission"
        );
        request.moss_image = Some(Default::default());
        request.moss_image.as_mut().unwrap().max_sequence_length = 256;
        assert!(
            extra_peak(inputs, &request).unwrap() < full,
            "precise projection scratch tracks context"
        );
        let spec = LoadSpec::new(WeightsSource::Dir(std::path::PathBuf::from(
            "unused-cold-snapshot",
        )));
        request.moss_image.as_mut().unwrap().uncond_skip_layers = Some(vec![22]);
        let error = crate::memory_strategy::edit_request_peak(
            mlx_gen::gen_core::moss_image::MossImageVariant::Sft,
            &spec,
            &request,
            inputs,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("full-layer CFG"),
            "typed refusal precedes asset access: {error}"
        );
    }

    #[test]
    fn edit_package_missing_corrupt_and_prepared_identity_reject_before_device() {
        let temp = tempfile::tempdir().unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(temp.path().join("unused-base")));
        assert!(EditAssets::prepare(&spec).is_err());
        spec.components.insert(
            EDIT_COMPONENT.into(),
            WeightsSource::Dir(temp.path().to_path_buf()),
        );
        std::fs::write(temp.path().join("adapter_config.json"), b"{}").unwrap();
        std::fs::write(
            temp.path().join("adapter_model.safetensors"),
            b"not-safetensors",
        )
        .unwrap();
        assert!(EditAssets::prepare(&spec).is_err());
        spec.prepare_with_file_pins([]).unwrap();
        assert!(EditAssets::prepare(&spec).is_err());
    }
}
