//! Real-checkpoint loading from a Krea 2 snapshot (standard diffusers multi-component tree):
//! `text_encoder/` (Qwen3-VL-4B condition encoder), `transformer/` (single-stream DiT), `vae/`
//! (Qwen-Image `AutoencoderKLQwenImage`, loaded via [`crate::vae::load_vae`]). The transformer +
//! text-encoder checkpoints are identity-keyed (diffusers names = the module tree), so
//! [`Weights::from_dir`] drops straight in; the VAE remap lives in `mlx-gen-qwen-image`.

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};
use mlx_gen_boogu::VisionTower;
use mlx_rs::ops::multiply;
use mlx_rs::Dtype;

use crate::config::Krea2Config;
use crate::text_encoder::{krea_vision_config, KreaTeConfig, KreaTextEncoder};
use crate::transformer::Krea2Transformer;

fn prepare_text_weights(mut w: Weights) -> Result<Weights> {
    let packed: std::collections::HashSet<String> = w
        .keys()
        .filter_map(|key| key.strip_suffix(".scales").map(str::to_owned))
        .collect();
    w.cast_matching(mlx_rs::Dtype::Bfloat16, |key| {
        key.starts_with("language_model.")
            && key.ends_with(".weight")
            && !key.contains("norm")
            && !packed.contains(key.strip_suffix(".weight").unwrap_or(key))
    })?;
    w.cast_matching(mlx_rs::Dtype::Float32, |key| {
        key.starts_with("language_model.") && key.ends_with("norm.weight")
    })?;
    Ok(w)
}

/// Load the Qwen3-VL-4B condition encoder from a snapshot's `text_encoder/` dir. The text tower lives
/// under `language_model.*`; the visual tower (`visual.*`) is assembled separately by
/// [`load_vision_tower`] only when image-grounded (edit) encoding is needed.
pub fn load_text_encoder(root: impl AsRef<Path>) -> Result<KreaTextEncoder> {
    let root = root.as_ref();
    let source = crate::model::ENCODER_CONTRACT.source_for_load(
        &mlx_gen::LoadSpec::new(WeightsSource::Dir(root.to_path_buf())),
        root,
    )?;
    source.read_unchanged(|source| load_text_encoder_from_source(root, source))
}

pub(crate) fn load_text_encoder_from_source(
    _model_root: &Path,
    source: &WeightsSource,
) -> Result<KreaTextEncoder> {
    let cfg = KreaTeConfig::qwen3_vl_4b();
    let w = prepare_text_weights(match source {
        WeightsSource::Dir(path) => Weights::from_dir(path)?,
        WeightsSource::File(path) => Weights::from_file(path)?,
    })?;
    let encoder = KreaTextEncoder::from_weights(&w, "language_model", &cfg)?;
    w.materialize_accessed()?;
    Ok(encoder)
}

/// Load the Qwen3-VL-4B **vision tower** from the same `text_encoder/` dir (epic 10871 P2.1, sc-10879):
/// the `visual.*` subtree that text-to-image never assembles. Casts the (small, parity-grade) vision
/// subtree to f32 before building — mirroring boogu's `load_vision_tower` — and feeds the shared
/// [`mlx_gen_boogu::VisionTower`] the Krea-4B [`krea_vision_config`]. Krea keys are `visual.*` (diffusers
/// naming), unlike boogu's `model.visual.*`.
///
/// Krea's converter leaves `visual.*` **dense** (`quantize_map`'s TE predicate covers the language
/// model only), so the group size passed here never selects a packed branch today — it is passed
/// explicitly because the shared tower must not assume any one crate's constant (sc-15154).
pub fn load_vision_tower(root: impl AsRef<Path>) -> Result<VisionTower> {
    let root = root.as_ref();
    let selected = crate::model::ENCODER_CONTRACT
        .validate_source(&WeightsSource::Dir(root.join("text_encoder")))?;
    selected.validate_vision(
        &crate::model::VISION_ENCODER_CONTRACT,
        &crate::model::ENCODER_CONTRACT,
    )?;
    selected.read_unchanged(load_vision_tower_from_source)
}

pub(crate) fn load_vision_tower_from_source(source: &WeightsSource) -> Result<VisionTower> {
    let mut w = match source {
        WeightsSource::Dir(path) => Weights::from_dir(path)?,
        WeightsSource::File(path) => Weights::from_file(path)?,
    };
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with("visual."))
        .map(String::from)
        .collect();
    for k in keys {
        let t = w.require(&k)?.as_dtype(mlx_rs::Dtype::Float32)?;
        w.insert(k, t);
    }
    let tower = VisionTower::from_weights(
        &w,
        krea_vision_config(),
        "visual",
        crate::convert::QUANT_GROUP_SIZE,
    )?;
    w.materialize_accessed()?;
    Ok(tower)
}

/// Load the single-stream DiT from a snapshot's `transformer/` dir: parse + validate the config, load
/// the (identity-keyed diffusers) weights, validate the architecture against the config, then assemble
/// the model. A pre-quantized snapshot loads through the same path (`quant::lin` auto-detects packed
/// keys); a dense bf16 build is quantized later via [`crate::pipeline::KreaPipeline::quantize`].
pub fn load_transformer(root: impl AsRef<Path>) -> Result<Krea2Transformer> {
    load_transformer_with_stream(root, false)
}

/// Load the DiT and, for deferred snapshot loads, retain the re-openable transformer source needed
/// to materialize exact block windows during denoise.
pub(crate) fn load_transformer_with_stream(
    root: impl AsRef<Path>,
    streamable: bool,
) -> Result<Krea2Transformer> {
    let root = root.as_ref();
    let cfg = Krea2Config::from_snapshot(root)?;
    let w = Weights::from_dir(root.join("transformer"))?;
    crate::convert::validate_transformer(&w, &cfg)?;
    let transformer = Krea2Transformer::from_weights(&w, &cfg)?;
    Ok(if streamable {
        transformer.with_block_stream(WeightsSource::Dir(root.join("transformer")))
    } else {
        transformer
    })
}

/// Validate and dequantize the non-rotated ComfyUI int8-tensorwise convention (sc-14023).
///
/// The app detector only has the safetensors header. Here, before any dequantization, every I8
/// projection must carry a real U8 JSON descriptor with `format=int8_tensorwise`, `per_row=true`, and
/// no `convrot` field, plus an F32 `[out]` or `[out,1]` scale (or scalar when `out == 1`). The consumed
/// companions are removed so the existing strict native-key remap still sees exactly the dense DiT
/// surface.
#[cfg(test)]
fn dequant_plain_int8_tensorwise(native: Weights) -> Result<Weights> {
    dequant_plain_int8_tensorwise_with_evaluation(native, true)
}

/// Normalize the native plain-int8 convention, optionally evaluating each reconstructed dense
/// projection immediately. Eager resident loads evaluate here to release the I8 source graph as each
/// projection is rebuilt. Deferred block streams must leave the graph lazy: evaluating every block
/// during each reopen would turn a nominal windowed loader back into full-DiT residency.
fn dequant_plain_int8_tensorwise_with_evaluation(
    mut native: Weights,
    evaluate_dense: bool,
) -> Result<Weights> {
    let int8_weights: Vec<String> = native
        .keys()
        .filter(|key| {
            native
                .get(key)
                .is_some_and(|tensor| tensor.dtype() == Dtype::Int8)
        })
        .map(str::to_owned)
        .collect();
    let descriptors: Vec<String> = native
        .keys()
        .filter(|key| key.ends_with(".comfy_quant"))
        .map(str::to_owned)
        .collect();

    if int8_weights.is_empty() {
        if descriptors.is_empty() {
            return Ok(native);
        }
        return Err(Error::Msg(format!(
            "krea plain int8: found {} `.comfy_quant` descriptor(s) but no I8 weight tensors",
            descriptors.len()
        )));
    }

    for weight_key in int8_weights {
        let Some(base) = weight_key.strip_suffix(".weight") else {
            return Err(Error::Msg(format!(
                "krea plain int8: I8 tensor `{weight_key}` is not a projection `.weight`"
            )));
        };
        let weight = native.require(&weight_key)?;
        let [rows, _cols] = weight.shape() else {
            return Err(Error::Msg(format!(
                "krea plain int8: `{weight_key}` must be rank-2 [out,in], got {:?}",
                weight.shape()
            )));
        };
        let rows = *rows;

        let descriptor_key = format!("{base}.comfy_quant");
        let descriptor = native.require(&descriptor_key).map_err(|_| {
            Error::Msg(format!(
                "krea plain int8: `{weight_key}` is missing `{descriptor_key}`"
            ))
        })?;
        if descriptor.dtype() != Dtype::Uint8 || descriptor.shape().len() != 1 {
            return Err(Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` must be a rank-1 U8 JSON blob"
            )));
        }
        let descriptor_bytes = descriptor.try_as_slice::<u8>().map_err(|error| {
            Error::Msg(format!(
                "krea plain int8: could not read `{descriptor_key}`: {error}"
            ))
        })?;
        let json: serde_json::Value =
            serde_json::from_slice(descriptor_bytes).map_err(|error| {
                Error::Msg(format!(
                    "krea plain int8: `{descriptor_key}` is not valid JSON: {error}"
                ))
            })?;
        if json.get("format").and_then(serde_json::Value::as_str) != Some("int8_tensorwise") {
            return Err(Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` must declare format `int8_tensorwise`"
            )));
        }
        if json.get("per_row").and_then(serde_json::Value::as_bool) != Some(true) {
            return Err(Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` must declare `per_row: true`"
            )));
        }
        if json.get("convrot").is_some() {
            return Err(Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` contains `convrot`; rotated checkpoints are not \
                 the plain MLX int8 format"
            )));
        }

        let scale_key = format!("{base}.weight_scale");
        let scale = native.require(&scale_key).map_err(|_| {
            Error::Msg(format!(
                "krea plain int8: `{weight_key}` is missing `{scale_key}`"
            ))
        })?;
        if scale.dtype() != Dtype::Float32 {
            return Err(Error::Msg(format!(
                "krea plain int8: `{scale_key}` must be F32, got {:?}",
                scale.dtype()
            )));
        }
        let scalar_single_row = rows == 1 && scale.shape().is_empty();
        if !scalar_single_row && scale.shape() != [rows] && scale.shape() != [rows, 1] {
            return Err(Error::Msg(format!(
                "krea plain int8: `{scale_key}` must be [{rows}] or [{rows},1]{}; got {:?}",
                if rows == 1 { " or scalar" } else { "" },
                scale.shape()
            )));
        }

        let codes = native
            .remove(&weight_key)
            .ok_or_else(|| Error::MissingTensor(weight_key.clone()))?
            .as_dtype(Dtype::Float32)?;
        let scale = native
            .remove(&scale_key)
            .ok_or_else(|| Error::MissingTensor(scale_key.clone()))?;
        let scale = match scale.shape() {
            [] if rows == 1 => scale.reshape(&[1, 1])?,
            [_] => scale.reshape(&[rows, 1])?,
            _ => scale,
        };
        let dense = multiply(&codes, &scale)?.as_dtype(Dtype::Bfloat16)?;
        if evaluate_dense {
            // MLX is lazy: materialize projection-by-projection so the eager dense model does not
            // retain a graph edge to every removed I8 code/scale buffer (which would keep both the
            // 13.5 GB source and the BF16 reconstruction alive for the whole resident load).
            dense.eval()?;
        }
        native.insert(weight_key, dense);
        native.remove(&descriptor_key);
    }

    if let Some(orphan) = native.keys().find(|key| key.ends_with(".comfy_quant")) {
        return Err(Error::Msg(format!(
            "krea plain int8: `{orphan}` does not describe an I8 projection weight"
        )));
    }
    Ok(native)
}

/// Load a community single-file Krea 2 DiT through the shared native→diffusers remap.
///
/// Dense bf16 files pass through unchanged. Plain int8-per-row files are descriptor-validated and
/// dequantized first as `codes.i8 * weight_scale` with no rotation. The remapped set then receives the
/// same architecture coverage/shape validation and transformer assembly as the published snapshot.
/// `cfg` comes from the resident base snapshot because the single file has no `config.json`.
pub(crate) fn normalized_native_weights(dit_file: &Path) -> Result<Weights> {
    normalized_native_weights_with_materializer(dit_file, |weights| {
        #[cfg(test)]
        run_native_materialize_test_hook(NativeMaterializeTestStage::Before, weights)?;
        weights.materialize()?;
        #[cfg(test)]
        run_native_materialize_test_hook(NativeMaterializeTestStage::After, weights)?;
        Ok(())
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeMaterializeTestStage {
    Before,
    After,
}

#[cfg(test)]
type NativeMaterializeTestHook = Box<dyn FnMut(NativeMaterializeTestStage, &Weights) -> Result<()>>;

#[cfg(test)]
thread_local! {
    /// Deterministic barrier inside the real eager native-file entrypoint. Production builds contain
    /// no hook; the ignored Metal regression uses it to replace the selected path between the first
    /// evaluated provider array and the production `Weights::materialize` completion boundary.
    static NATIVE_MATERIALIZE_TEST_HOOK: std::cell::RefCell<Option<NativeMaterializeTestHook>> =
        std::cell::RefCell::new(None);
}

#[cfg(test)]
fn run_native_materialize_test_hook(
    stage: NativeMaterializeTestStage,
    weights: &Weights,
) -> Result<()> {
    NATIVE_MATERIALIZE_TEST_HOOK.with(|slot| match slot.borrow_mut().as_mut() {
        Some(hook) => hook(stage, weights),
        None => Ok(()),
    })
}

/// Normalize the native namespace without evaluating the complete checkpoint. The bounded block
/// loader consumes only its accessed window under a separate pin guard; eagerly evaluating here
/// would retain the full DiT and invalidate that implementation's residency bound.
pub(crate) fn normalized_native_weights_lazy(dit_file: &Path) -> Result<Weights> {
    normalized_native_weights_with_options(dit_file, false, |_| Ok(()))
}

fn normalized_native_weights_with_materializer(
    dit_file: &Path,
    materialize: impl FnOnce(&Weights) -> Result<()>,
) -> Result<Weights> {
    normalized_native_weights_with_options(dit_file, true, materialize)
}

fn normalized_native_weights_with_options(
    dit_file: &Path,
    evaluate_plain_int8: bool,
    materialize: impl FnOnce(&Weights) -> Result<()>,
) -> Result<Weights> {
    let native = dequant_plain_int8_tensorwise_with_evaluation(
        Weights::from_file(dit_file)?,
        evaluate_plain_int8,
    )?;
    let mut remapped = crate::native_remap::remap_native_dit_to_diffusers(native)?;
    crate::native_remap::normalize_modulation_tables(&mut remapped)?;
    // `Weights::from_file` and every MLX cast/remap above are lazy. Force the final normalized map
    // while the caller's `PinnedWeightsFile::read_unchanged` guard still spans this function.
    materialize(&remapped)?;
    Ok(remapped)
}

pub fn load_transformer_from_native_file(
    dit_file: impl AsRef<Path>,
    cfg: &Krea2Config,
) -> Result<Krea2Transformer> {
    load_transformer_from_native_file_with_stream(dit_file, cfg, false)
}

/// Native-file loader with a retained, lstat-pinned source for bounded block residency.
pub(crate) fn load_transformer_from_native_file_with_stream(
    dit_file: impl AsRef<Path>,
    cfg: &Krea2Config,
    streamable: bool,
) -> Result<Krea2Transformer> {
    let pinned = mlx_gen::PinnedWeightsFile::pin(dit_file.as_ref())?;
    load_transformer_from_pinned_native_file_with_stream(&pinned, cfg, streamable)
}

/// Native-file loader over a caller-owned pin. Lazy/sequential generators create this pin once and
/// reuse it for every materialization, so a path replacement can never become the new base between
/// requests. Validation, model assembly, and the optional block stream are all tied to that pin.
pub(crate) fn load_transformer_from_pinned_native_file_with_stream(
    pinned: &mlx_gen::PinnedWeightsFile,
    cfg: &Krea2Config,
    streamable: bool,
) -> Result<Krea2Transformer> {
    pinned.read_unchanged(|path| {
        if streamable {
            let remapped = normalized_native_weights_lazy(path)?;
            // Validation reads representative shapes only. Keep that access bookkeeping on a clone
            // so the pin-bound evaluation below contains exactly the static model tensors retained by
            // the deferred constructor, not a representative transformer block.
            crate::convert::validate_transformer(&remapped.clone(), cfg)?;
            let transformer = Krea2Transformer::from_weights_deferred(&remapped, cfg)?;
            // The retained static arrays are also lazy file-backed graphs. Consume their exact read
            // set before the immutable-file post-check; block payloads are consumed later under each
            // KreaBlockStream reopen's own pin guard.
            remapped.materialize_accessed()?;
            Ok(transformer.with_native_block_stream(pinned.clone()))
        } else {
            let remapped = normalized_native_weights(path)?;
            crate::convert::validate_transformer(&remapped, cfg)?;
            Krea2Transformer::from_weights(&remapped, cfg)
        }
    })
}

/// Build, mutate (adapter fold + quantization), and materialize an imported DiT while one immutable
/// source pin spans the whole operation.  Normalization stays lazy and the final transformer walks
/// its retained representation projection-by-projection, so Q4/Q8 loading never first evaluates the
/// complete dense DiT.
pub(crate) fn load_transformer_from_pinned_native_file_bounded(
    pinned: &mlx_gen::PinnedWeightsFile,
    cfg: &Krea2Config,
    prepare: impl FnOnce(&mut Krea2Transformer) -> Result<()>,
) -> Result<Krea2Transformer> {
    pinned.read_unchanged(|path| {
        let remapped = normalized_native_weights_lazy(path)?;
        crate::convert::validate_transformer(&remapped.clone(), cfg)?;
        let mut transformer = Krea2Transformer::from_weights(&remapped, cfg)?;
        // `from_weights` clones the lazy Array handles it retains. Drop the normalized source map
        // before the projection walk: otherwise that map keeps an extra reference to every dense
        // file-backed source array and evaluated inputs can accumulate to the full dense DiT while
        // the packed transformer is being materialized.
        drop(remapped);
        prepare(&mut transformer)?;
        transformer.materialize_weights()?;
        Ok(transformer)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};

    fn plain_int8_weights(descriptor: &str, scale: Array) -> Weights {
        plain_int8_weights_with_shape(descriptor, &[1_i8, -2, 3, -4, 5, -6], &[2, 3], scale)
    }

    fn plain_int8_weights_with_shape(
        descriptor: &str,
        codes: &[i8],
        shape: &[i32],
        scale: Array,
    ) -> Weights {
        let mut weights = Weights::empty();
        weights.insert(
            "model.diffusion_model.blocks.0.attn.wq.weight",
            Array::from_slice(codes, shape),
        );
        weights.insert("model.diffusion_model.blocks.0.attn.wq.weight_scale", scale);
        weights.insert(
            "model.diffusion_model.blocks.0.attn.wq.comfy_quant",
            Array::from_slice(descriptor.as_bytes(), &[descriptor.len() as i32]),
        );
        weights
    }

    #[test]
    fn deferred_native_normalization_skips_the_full_map_materializer() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("deferred-native.safetensors");
        let mut header = br#"{"model.diffusion_model.first.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"model.diffusion_model.first.bias":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header.len() + 4);
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&[0, 0, 0, 0]);
        std::fs::write(&source, file).unwrap();

        let materialized = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&materialized);
        NATIVE_MATERIALIZE_TEST_HOOK.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move |_, _| {
                observed.store(true, Ordering::SeqCst);
                Ok(())
            }));
        });
        let normalized = normalized_native_weights_lazy(&source).unwrap();
        NATIVE_MATERIALIZE_TEST_HOOK.with(|slot| *slot.borrow_mut() = None);

        assert!(normalized.get("img_in.weight").is_some());
        assert!(normalized.get("img_in.bias").is_some());
        assert!(
            !materialized.load(Ordering::SeqCst),
            "the deferred File normalizer must never call the eager full-map materialization seam"
        );
    }

    /// MLX keeps safetensors arrays and the remap/cast graph lazy. This regression places a real
    /// atomic path replacement between two provider-array evaluations and proves the Krea native
    /// normalization does not return to `PinnedWeightsFile` for its post-check until the final map is
    /// materialized. Non-macOS runners still compile the test, but cannot execute MLX's Metal runtime.
    #[test]
    #[ignore = "requires an accessible Apple Metal device; run explicitly on a physical macOS GPU host"]
    fn native_file_pin_postchecks_after_final_mlx_evaluation() {
        let dir = tempfile::tempdir().expect("temp dir");
        let source = dir.path().join("krea-native.safetensors");
        let replacement = dir.path().join("krea-native.replacement.safetensors");
        let elements = 64 * 1024;

        let original_weight = Array::from_slice(&vec![0.25_f32; elements], &[elements as i32]);
        let original_bias = Array::from_slice(&vec![0.75_f32; elements], &[elements as i32]);
        Array::save_safetensors(
            vec![
                ("model.diffusion_model.first.weight", &original_weight),
                ("model.diffusion_model.first.bias", &original_bias),
            ],
            None,
            &source,
        )
        .expect("write original native checkpoint");
        let replacement_weight = Array::from_slice(&vec![-0.25_f32; elements], &[elements as i32]);
        let replacement_bias = Array::from_slice(&vec![-0.75_f32; elements], &[elements as i32]);
        Array::save_safetensors(
            vec![
                ("model.diffusion_model.first.weight", &replacement_weight),
                ("model.diffusion_model.first.bias", &replacement_bias),
            ],
            None,
            &replacement,
        )
        .expect("write replacement native checkpoint");

        let first_evaluated = Arc::new(Barrier::new(2));
        let replacement_done = Arc::new(Barrier::new(2));
        let final_evaluated = Arc::new(AtomicBool::new(false));

        let writer_source = source.clone();
        let writer_replacement = replacement.clone();
        let writer_first = Arc::clone(&first_evaluated);
        let writer_done = Arc::clone(&replacement_done);
        let writer = std::thread::spawn(move || {
            writer_first.wait();
            let swapped = std::fs::rename(writer_replacement, writer_source);
            // Release the materialize hook whatever the swap did. The hook is parked on this
            // barrier on the main thread, so a writer that panics — or returns — ahead of it
            // strands the hook there and hangs the whole test binary; the outcome is asserted on
            // `join` instead, so a failed swap reads as a red test naming the error.
            writer_done.wait();
            swapped
        });

        let evaluated = Arc::clone(&final_evaluated);
        NATIVE_MATERIALIZE_TEST_HOOK.with(|slot| {
            *slot.borrow_mut() = Some(Box::new(move |stage, weights| {
                match stage {
                    NativeMaterializeTestStage::Before => {
                        let first = weights.require("img_in.weight")?;
                        first.eval()?;
                        assert!(first.as_slice::<f32>().iter().all(|value| *value == 0.25));
                        first_evaluated.wait();
                        replacement_done.wait();
                    }
                    NativeMaterializeTestStage::After => {
                        assert_eq!(
                            weights.require("img_in.bias")?.as_slice::<f32>().len(),
                            elements
                        );
                        evaluated.store(true, Ordering::SeqCst);
                    }
                }
                Ok(())
            }));
        });
        // Call the public production loader, not the guard or normalizer primitive. Validation/model
        // assembly may fail for this intentionally small checkpoint, but the enclosing pin's mutation
        // post-check must run first and win as the stronger provenance diagnosis.
        let result = load_transformer_from_native_file(&source, &Krea2Config::turbo());
        NATIVE_MATERIALIZE_TEST_HOOK.with(|slot| *slot.borrow_mut() = None);
        // Diagnose the swap before the load result: the `Ok` arm below panics, and a panic there
        // would otherwise mask a writer that never managed to replace the checkpoint at all.
        writer
            .join()
            .expect("replacement writer")
            .expect("atomically replace the native checkpoint during lazy evaluation");
        let error = match result {
            Ok(_) => {
                panic!("replacement during MLX evaluation must invalidate the native file pin")
            }
            Err(error) => error.to_string(),
        };

        assert!(
            final_evaluated.load(Ordering::SeqCst),
            "the final MLX array map must be evaluated before the post-check rejects the mutation"
        );
        assert!(error.contains("changed after load"), "unexpected: {error}");
    }

    #[test]
    fn plain_int8_dequants_per_row_without_rotation() {
        let weights = plain_int8_weights(
            r#"{"format":"int8_tensorwise","per_row":true}"#,
            Array::from_slice(&[0.5_f32, 2.0], &[2, 1]),
        );
        let dequant = dequant_plain_int8_tensorwise(weights).unwrap();
        assert!(dequant
            .get("model.diffusion_model.blocks.0.attn.wq.weight_scale")
            .is_none());
        assert!(dequant
            .get("model.diffusion_model.blocks.0.attn.wq.comfy_quant")
            .is_none());
        let got = dequant
            .require("model.diffusion_model.blocks.0.attn.wq.weight")
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        assert_eq!(got.as_slice::<f32>(), &[0.5, -1.0, 1.5, -8.0, 10.0, -12.0]);
    }

    #[test]
    fn plain_int8_accepts_scalar_scale_for_single_row() {
        let weights = plain_int8_weights_with_shape(
            r#"{"format":"int8_tensorwise","per_row":true}"#,
            &[1_i8, -2, 3],
            &[1, 3],
            Array::from_slice(&[0.5_f32], &[]),
        );
        let dequant = dequant_plain_int8_tensorwise(weights).unwrap();
        let got = dequant
            .require("model.diffusion_model.blocks.0.attn.wq.weight")
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        assert_eq!(got.as_slice::<f32>(), &[0.5, -1.0, 1.5]);
    }

    #[test]
    fn plain_int8_rejects_convrot_or_wrong_descriptor() {
        for (descriptor, expected) in [
            (
                r#"{"format":"int8_tensorwise","per_row":true,"convrot":true}"#,
                "convrot",
            ),
            (r#"{"format":"mxfp4","per_row":true}"#, "int8_tensorwise"),
            (r#"{"format":"int8_tensorwise","per_row":false}"#, "per_row"),
        ] {
            let error = match dequant_plain_int8_tensorwise(plain_int8_weights(
                descriptor,
                Array::from_slice(&[0.5_f32, 2.0], &[2]),
            )) {
                Ok(_) => panic!("invalid descriptor must fail"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn plain_int8_rejects_non_per_row_scale_shape() {
        for scale in [
            Array::from_slice(&[0.5_f32], &[1]),
            Array::from_slice(&[0.5_f32], &[]),
        ] {
            let error = match dequant_plain_int8_tensorwise(plain_int8_weights(
                r#"{"format":"int8_tensorwise","per_row":true}"#,
                scale,
            )) {
                Ok(_) => panic!("wrong scale shape must fail"),
                Err(error) => error.to_string(),
            };
            assert!(
                error.contains("weight_scale") && error.contains("[2]"),
                "{error}"
            );
        }
    }
}
