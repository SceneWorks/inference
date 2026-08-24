//! Real-checkpoint loading from a Krea 2 snapshot (standard diffusers multi-component tree):
//! `text_encoder/` (Qwen3-VL-4B condition encoder), `transformer/` (single-stream DiT), `vae/`
//! (Qwen-Image `AutoencoderKLQwenImage`, loaded via [`crate::vae::load_vae`]). The transformer +
//! text-encoder checkpoints are identity-keyed (diffusers names = the module tree), so
//! [`Weights::from_dir`] drops straight in; the VAE remap lives in `mlx-gen-qwen-image`.

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};
use mlx_gen_boogu::VisionTower;

use crate::config::Krea2Config;
use crate::native_remap::DeclaredLogicalShapes;
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

/// Load a community single-file Krea 2 DiT through the mapped logical-weight reader and the
/// engine's registered codec table (sc-20634 dense; sc-20385 descriptor codecs).
///
/// Dense bf16/f32 tensors pass through byte-identical; plain int8-per-row, scalar fp8
/// (`float8_e4m3fn`/`float8_e5m2` descriptors or the plain fp8 cast) and MXFP8 layers dequantize
/// per layer through their codec rows. The remapped set then receives the same architecture
/// coverage/shape validation and transformer assembly as the published snapshot. `cfg` comes from
/// the resident base snapshot because the single file has no `config.json`.
///
/// `shapes` carries that same architecture into the plan compiler (sc-20644): MXFP8 storage is
/// 32-padded and records no true shape, so the adapter must declare one for the layer to unpad to
/// the geometry the DiT then validates. Callers holding a `Krea2Config` pass
/// [`DeclaredLogicalShapes::FromConfig`]; a caller with none passes
/// [`DeclaredLogicalShapes::NotInScope`] and gets the fail-closed padded-shape behaviour.
pub(crate) fn normalized_native_weights(
    dit_file: &Path,
    shapes: DeclaredLogicalShapes<'_>,
) -> Result<Weights> {
    normalized_native_weights_with_receipt(dit_file, shapes).map(|(weights, _)| weights)
}

/// [`normalized_native_weights`], additionally returning the logical-weight receipt of the read.
/// Since sc-20385 every native single-file convention reads through the codec seam, so a receipt is
/// always returned.
///
/// Callers that want the receipt take it from here rather than from the process-global slot: the
/// global's `reset → load → read` window is not atomic, so two observers in one test binary can
/// clobber each other's observation.
pub(crate) fn normalized_native_weights_with_receipt(
    dit_file: &Path,
    shapes: DeclaredLogicalShapes<'_>,
) -> Result<(Weights, mlx_gen::gen_core::LogicalWeightReceipt)> {
    normalized_native_weights_with_materializer(dit_file, shapes, |weights| {
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
pub(crate) fn normalized_native_weights_lazy(
    dit_file: &Path,
    shapes: DeclaredLogicalShapes<'_>,
) -> Result<Weights> {
    normalized_native_weights_lazy_with_receipt(dit_file, shapes).map(|(weights, _)| weights)
}

/// [`normalized_native_weights_lazy`], additionally returning the receipt of the read.
pub(crate) fn normalized_native_weights_lazy_with_receipt(
    dit_file: &Path,
    shapes: DeclaredLogicalShapes<'_>,
) -> Result<(Weights, mlx_gen::gen_core::LogicalWeightReceipt)> {
    normalized_native_weights_with_options(dit_file, shapes, false, |_| Ok(()))
}

fn normalized_native_weights_with_materializer(
    dit_file: &Path,
    shapes: DeclaredLogicalShapes<'_>,
    materialize: impl FnOnce(&Weights) -> Result<()>,
) -> Result<(Weights, mlx_gen::gen_core::LogicalWeightReceipt)> {
    normalized_native_weights_with_options(dit_file, shapes, true, materialize)
}

/// The receipt of the most recent native-file read that went through the mapped logical-weight
/// reader on this process (sc-20634). Diagnostics only: the loader's correctness does not depend
/// on it, but SceneWorks' parity checks read it to prove a dense bf16 file was consumed through
/// the codec seam and what the codec left resident.
///
/// `reset → load → read` across a process-global slot is not atomic. Anything that observes this
/// slot — rather than taking the receipt straight off
/// [`normalized_native_weights_with_receipt`] / [`normalized_native_weights_lazy_with_receipt`],
/// which is what every in-crate test does — must hold [`RECEIPT_LOCK`] across the whole window, or
/// a concurrently running observer's reset can clear the observation. That invariant is pinned by
/// `every_process_global_receipt_observation_is_serialized`.
static LAST_NATIVE_FILE_RECEIPT: std::sync::Mutex<Option<mlx_gen::gen_core::LogicalWeightReceipt>> =
    std::sync::Mutex::new(None);

/// Serializes the `reset → load → read` window of every observer of the process-global receipt
/// slot. See [`LAST_NATIVE_FILE_RECEIPT`].
#[cfg(test)]
pub(crate) static RECEIPT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn last_native_file_receipt() -> Option<mlx_gen::gen_core::LogicalWeightReceipt> {
    LAST_NATIVE_FILE_RECEIPT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

pub fn reset_native_file_receipt() {
    *LAST_NATIVE_FILE_RECEIPT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

fn record_native_file_receipt(receipt: mlx_gen::gen_core::LogicalWeightReceipt) {
    *LAST_NATIVE_FILE_RECEIPT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(receipt);
}

fn normalized_native_weights_with_options(
    dit_file: &Path,
    shapes: DeclaredLogicalShapes<'_>,
    eager: bool,
    materialize: impl FnOnce(&Weights) -> Result<()>,
) -> Result<(Weights, mlx_gen::gen_core::LogicalWeightReceipt)> {
    // One route for every native single-file convention (sc-20634 dense; sc-20385 descriptor
    // codecs): the adapter's key mapping + the registered codec table. The plan is compiled from
    // the header plus the `.comfy_quant` descriptor payloads, so an unmapped key, a collision, a
    // stored format without a codec (packed u8, nvfp4, …), a malformed descriptor, or a missing /
    // mis-shaped scale companion refuses here, per layer, before any MLX array is created. Plain
    // int8-per-row, scalar fp8 (E4M3/E5M2, described or the plain cast), and MXFP8 all decode
    // through their registered codec rows — the former bespoke int8 arm's math lives in the
    // engine's `int8-per-row-v1` implementation now.
    let plan = mlx_gen::logical_weights::plan_logical_weights(
        dit_file,
        &crate::native_remap::KreaNativeToDiffusersMapping::new(shapes),
    )?;
    let mut materialize = Some(materialize);
    let mut eager_materializer = |weights: &mut Weights| -> Result<()> {
        crate::native_remap::normalize_modulation_tables(weights)?;
        match materialize.take() {
            Some(materialize) => materialize(weights),
            None => Err(Error::Msg(
                "krea native-file materializer invoked twice".to_owned(),
            )),
        }
    };
    let mode = if eager {
        mlx_gen::logical_weights::LogicalReadMode::Eager(&mut eager_materializer)
    } else {
        mlx_gen::logical_weights::LogicalReadMode::Deferred
    };
    let mlx_gen::logical_weights::LogicalWeights {
        mut weights,
        receipt,
    } = mlx_gen::logical_weights::read_logical_weights(dit_file, &plan, mode)?;
    if receipt.materialization == mlx_gen::gen_core::LogicalReadMaterialization::Deferred {
        // The deferred reader leaves the closure unused; the shape normalization is still part of
        // the canonical logical form, so apply it here (lazily, under the caller's pin guard).
        crate::native_remap::normalize_modulation_tables(&mut weights)?;
    }
    record_native_file_receipt(receipt.clone());
    Ok((weights, receipt))
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
            let remapped =
                normalized_native_weights_lazy(path, DeclaredLogicalShapes::FromConfig(cfg))?;
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
            let remapped = normalized_native_weights(path, DeclaredLogicalShapes::FromConfig(cfg))?;
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
        let remapped =
            normalized_native_weights_lazy(path, DeclaredLogicalShapes::FromConfig(cfg))?;
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
    use mlx_rs::{Array, Dtype};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};

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
        let normalized =
            normalized_native_weights_lazy(&source, DeclaredLogicalShapes::NotInScope).unwrap();
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

        // Dense bf16, the one encoding the baseline codec table serves (sc-20634).
        let original_weight = Array::from_slice(&vec![0.25_f32; elements], &[elements as i32])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let original_bias = Array::from_slice(&vec![0.75_f32; elements], &[elements as i32])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        Array::save_safetensors(
            vec![
                ("model.diffusion_model.first.weight", &original_weight),
                ("model.diffusion_model.first.bias", &original_bias),
            ],
            None,
            &source,
        )
        .expect("write original native checkpoint");
        let replacement_weight = Array::from_slice(&vec![-0.25_f32; elements], &[elements as i32])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let replacement_bias = Array::from_slice(&vec![-0.75_f32; elements], &[elements as i32])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
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
                        let first = mlx_gen::weights::to_f32(weights.require("img_in.weight")?)?;
                        first.eval()?;
                        assert!(first.as_slice::<f32>().iter().all(|value| *value == 0.25));
                        first_evaluated.wait();
                        replacement_done.wait();
                    }
                    NativeMaterializeTestStage::After => {
                        assert_eq!(
                            mlx_gen::weights::to_f32(weights.require("img_in.bias")?)?
                                .as_slice::<f32>()
                                .len(),
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

    fn write_native_safetensors(path: &Path, tensors: &[(&str, &str, &[usize], Vec<u8>)]) {
        let mut header_entries = Vec::new();
        let mut body = Vec::new();
        for (name, dtype, shape, payload) in tensors {
            let start = body.len();
            body.extend_from_slice(payload);
            let end = body.len();
            header_entries.push(format!(
                "{:?}:{{\"dtype\":{:?},\"shape\":{:?},\"data_offsets\":[{start},{end}]}}",
                name, dtype, shape
            ));
        }
        let mut header = format!("{{{}}}", header_entries.join(",")).into_bytes();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = Vec::with_capacity(8 + header.len() + body.len());
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&body);
        std::fs::write(path, file).unwrap();
    }

    fn bf16_le(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn native_fixture_dir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix(&format!("krea-native-{}-", std::process::id()))
            .tempdir()
            .unwrap()
    }

    /// sc-20634: a dense bf16 native file is consumed through the mapped logical-weight reader and
    /// the registered dense-bf16 codec — keys arrive canonical, the adapter's modulation reshape is
    /// part of the logical form, and the receipt reports what the codec left resident, measured
    /// from the arrays (dense bf16 ⇒ exactly the source bytes).
    #[test]
    fn dense_native_file_reads_through_the_logical_codec_seam_and_reports_residency() {
        let dir = native_fixture_dir();
        let source = dir.path().join("dense-native.safetensors");
        write_native_safetensors(
            &source,
            &[
                (
                    "model.diffusion_model.first.weight",
                    "BF16",
                    &[2],
                    bf16_le(&[0.25, -2.0]),
                ),
                (
                    "model.diffusion_model.first.bias",
                    "BF16",
                    &[2],
                    bf16_le(&[1.0, 0.5]),
                ),
                (
                    "model.diffusion_model.blocks.0.mod.lin",
                    "BF16",
                    &[12],
                    bf16_le(&[1.0; 12]),
                ),
            ],
        );
        let (normalized, receipt) =
            normalized_native_weights_with_receipt(&source, DeclaredLogicalShapes::NotInScope)
                .unwrap();
        let mut keys: Vec<&str> = normalized.keys().collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "img_in.bias",
                "img_in.weight",
                "transformer_blocks.0.scale_shift_table"
            ]
        );
        assert_eq!(
            normalized
                .require("transformer_blocks.0.scale_shift_table")
                .unwrap()
                .shape(),
            [6, 2],
            "the adapter-owned flat→[6, hidden] reshape is applied inside the logical form"
        );
        let weight =
            mlx_gen::weights::to_f32(normalized.require("img_in.weight").unwrap()).unwrap();
        assert_eq!(weight.as_slice::<f32>(), [0.25, -2.0]);

        assert_eq!(
            receipt.mapping_id,
            crate::native_remap::KreaNativeToDiffusersMapping::MAPPING_ID
        );
        assert_eq!(receipt.tensor_count, 3);
        assert_eq!(receipt.source_bytes, 4 + 4 + 24);
        assert_eq!(
            receipt.materialization,
            mlx_gen::gen_core::LogicalReadMaterialization::Materialized
        );
        assert_eq!(receipt.residency.len(), 1);
        assert_eq!(receipt.residency[0].codec_id, "dense-bf16-v1");
        assert_eq!(receipt.residency[0].tensor_count, 3);
        let measured: usize = normalized
            .keys()
            .map(|key| normalized.get(key).unwrap().nbytes())
            .sum();
        assert_eq!(receipt.residency[0].resident_bytes, measured as u64);
        assert_eq!(receipt.resident_bytes(), 32);

        // The deferred (block-stream / bounded) entry keeps payloads lazy and says so.
        let (lazy, receipt) =
            normalized_native_weights_lazy_with_receipt(&source, DeclaredLogicalShapes::NotInScope)
                .unwrap();
        assert_eq!(
            lazy.require("transformer_blocks.0.scale_shift_table")
                .unwrap()
                .shape(),
            [6, 2]
        );
        assert_eq!(
            receipt.materialization,
            mlx_gen::gen_core::LogicalReadMaterialization::Deferred
        );
        assert!(receipt.residency.is_empty());
    }

    /// sc-20385: the native single-file route dispatches EVERY ComfyUI convention per layer
    /// through the registered codec table — plain fp8 casts decode at unit scale, descriptor
    /// int8-per-row decodes through `int8-per-row-v1` (the former bespoke arm's math), a foreign
    /// key still refuses, and a stored format without a codec (packed u8) refuses from the header
    /// naming the tensor.
    #[test]
    fn native_file_conventions_dispatch_per_layer_through_the_codec_seam() {
        let dir = native_fixture_dir();

        // Plain fp8 cast (the KreaMania V1/V2 fp8 shape: every tensor F8_E4M3, no companions).
        let fp8 = dir.path().join("fp8-native.safetensors");
        write_native_safetensors(
            &fp8,
            &[
                (
                    "model.diffusion_model.first.weight",
                    "F8_E4M3",
                    &[2, 1],
                    vec![0x38, 0x40], // 1.0, 2.0
                ),
                (
                    "model.diffusion_model.first.bias",
                    "F8_E4M3",
                    &[2],
                    vec![0xB8, 0x48], // -1.0, 4.0
                ),
            ],
        );
        let (weights, receipt) =
            normalized_native_weights_with_receipt(&fp8, DeclaredLogicalShapes::NotInScope)
                .unwrap();
        let weight = mlx_gen::weights::to_f32(weights.require("img_in.weight").unwrap()).unwrap();
        assert_eq!(weight.as_slice::<f32>(), [1.0, 2.0]);
        let bias = mlx_gen::weights::to_f32(weights.require("img_in.bias").unwrap()).unwrap();
        assert_eq!(bias.as_slice::<f32>(), [-1.0, 4.0]);
        assert_eq!(receipt.residency.len(), 1);
        assert_eq!(receipt.residency[0].codec_id, "fp8-e4m3-scalar-v1");
        // fp8 → bf16 residency: twice the stored bytes, measured from the arrays.
        assert_eq!(receipt.resident_bytes(), 2 * receipt.source_bytes);

        // Foreign key: refuses naming the tensor.
        let foreign = dir.path().join("foreign-native.safetensors");
        write_native_safetensors(
            &foreign,
            &[(
                "model.diffusion_model.unknown.weight",
                "BF16",
                &[1],
                bf16_le(&[0.25]),
            )],
        );
        let error = normalized_native_weights(&foreign, DeclaredLogicalShapes::NotInScope)
            .err()
            .expect("foreign key must refuse")
            .to_string();
        assert!(
            error.contains("\"model.diffusion_model.unknown.weight\"")
                && error.contains("no canonical logical key"),
            "{error}"
        );

        // A stored format without a codec (packed u8 nibbles) refuses from the header.
        let packed = dir.path().join("packed-native.safetensors");
        write_native_safetensors(
            &packed,
            &[
                (
                    "model.diffusion_model.first.weight",
                    "BF16",
                    &[1],
                    bf16_le(&[0.25]),
                ),
                (
                    "model.diffusion_model.first.bias",
                    "U8",
                    &[2],
                    vec![0x12, 0x34],
                ),
            ],
        );
        let error = normalized_native_weights(&packed, DeclaredLogicalShapes::NotInScope)
            .err()
            .expect("packed u8 must refuse")
            .to_string();
        assert!(
            error.contains("\"model.diffusion_model.first.bias\"")
                && error.contains("no checkpoint codec is registered"),
            "{error}"
        );

        // Plain int8 descriptor convention: now the registry's `int8-per-row-v1` row, with a
        // receipt — the same dequant values as the former bespoke arm.
        let int8 = dir.path().join("int8-native.safetensors");
        let descriptor = br#"{"format":"int8_tensorwise","per_row":true}"#;
        write_native_safetensors(
            &int8,
            &[
                (
                    "model.diffusion_model.blocks.0.attn.wq.weight",
                    "I8",
                    &[1, 2],
                    vec![2, 0xFE],
                ),
                (
                    "model.diffusion_model.blocks.0.attn.wq.weight_scale",
                    "F32",
                    &[1],
                    0.5_f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.diffusion_model.blocks.0.attn.wq.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let (dequantized, receipt) =
            normalized_native_weights_with_receipt(&int8, DeclaredLogicalShapes::NotInScope)
                .unwrap();
        let projection = mlx_gen::weights::to_f32(
            dequantized
                .require("transformer_blocks.0.attn.to_q.weight")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(projection.as_slice::<f32>(), [1.0, -1.0]);
        assert!(
            dequantized
                .get("transformer_blocks.0.attn.to_q.weight_scale")
                .is_none(),
            "the scale companion is consumed, not remapped"
        );
        assert_eq!(receipt.residency.len(), 1);
        assert_eq!(receipt.residency[0].codec_id, "int8-per-row-v1");
    }

    /// sc-20385: int8 per-row descriptor defects refuse per layer with the exact defect — the
    /// refusal set the former bespoke arm enforced, now from the shared plan compiler.
    #[test]
    fn plain_int8_descriptor_defects_refuse_by_layer_via_the_plan() {
        let dir = native_fixture_dir();
        let write_int8 = |name: &str,
                          descriptor: &[u8],
                          scale_shape: &[usize],
                          scale_payload: Vec<u8>|
         -> std::path::PathBuf {
            let path = dir.path().join(name);
            write_native_safetensors(
                &path,
                &[
                    (
                        "model.diffusion_model.blocks.0.attn.wq.weight",
                        "I8",
                        &[2, 3],
                        vec![1, 0xFE, 3, 0xFC, 5, 0xFA],
                    ),
                    (
                        "model.diffusion_model.blocks.0.attn.wq.weight_scale",
                        "F32",
                        scale_shape,
                        scale_payload,
                    ),
                    (
                        "model.diffusion_model.blocks.0.attn.wq.comfy_quant",
                        "U8",
                        &[descriptor.len()],
                        descriptor.to_vec(),
                    ),
                ],
            );
            path
        };
        let two_scales: Vec<u8> = [0.5_f32, 2.0]
            .iter()
            .flat_map(|scale| scale.to_le_bytes())
            .collect();

        // The happy path first, so the refusals below are about the defects, not the fixture.
        let good = write_int8(
            "good.safetensors",
            br#"{"format":"int8_tensorwise","per_row":true}"#,
            &[2, 1],
            two_scales.clone(),
        );
        let (weights, _) =
            normalized_native_weights_with_receipt(&good, DeclaredLogicalShapes::NotInScope)
                .unwrap();
        let got = mlx_gen::weights::to_f32(
            weights
                .require("transformer_blocks.0.attn.to_q.weight")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(got.as_slice::<f32>(), [0.5, -1.0, 1.5, -8.0, 10.0, -12.0]);

        for (name, descriptor, scale_shape, scale_payload, expected) in [
            (
                "convrot.safetensors",
                br#"{"format":"int8_tensorwise","per_row":true,"convrot":true}"#.as_slice(),
                [2_usize, 1].as_slice(),
                two_scales.clone(),
                "convrot",
            ),
            (
                "wrongformat.safetensors",
                br#"{"format":"mxfp4","per_row":true}"#.as_slice(),
                [2, 1].as_slice(),
                two_scales.clone(),
                "mxfp4",
            ),
            (
                "notperrow.safetensors",
                br#"{"format":"int8_tensorwise","per_row":false}"#.as_slice(),
                [2, 1].as_slice(),
                two_scales.clone(),
                "per_row",
            ),
            (
                "badscale.safetensors",
                br#"{"format":"int8_tensorwise","per_row":true}"#.as_slice(),
                [1].as_slice(),
                0.5_f32.to_le_bytes().to_vec(),
                "weight_scale",
            ),
        ] {
            let path = write_int8(name, descriptor, scale_shape, scale_payload);
            let error = normalized_native_weights(&path, DeclaredLogicalShapes::NotInScope)
                .err()
                .unwrap_or_else(|| panic!("{name} must refuse"))
                .to_string();
            assert!(
                error.contains(expected) && error.contains("blocks.0.attn.wq"),
                "{name}: {error}"
            );
        }
    }

    /// Single-row int8 keeps accepting the scalar scale form the converter writes for `out == 1`.
    #[test]
    fn plain_int8_accepts_scalar_scale_for_single_row() {
        let dir = native_fixture_dir();
        let path = dir.path().join("single-row.safetensors");
        let descriptor = br#"{"format":"int8_tensorwise","per_row":true}"#;
        write_native_safetensors(
            &path,
            &[
                (
                    "model.diffusion_model.blocks.0.attn.wq.weight",
                    "I8",
                    &[1, 3],
                    vec![1, 0xFE, 3],
                ),
                (
                    "model.diffusion_model.blocks.0.attn.wq.weight_scale",
                    "F32",
                    &[],
                    0.5_f32.to_le_bytes().to_vec(),
                ),
                (
                    "model.diffusion_model.blocks.0.attn.wq.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );
        let (weights, _) =
            normalized_native_weights_with_receipt(&path, DeclaredLogicalShapes::NotInScope)
                .unwrap();
        let got = mlx_gen::weights::to_f32(
            weights
                .require("transformer_blocks.0.attn.to_q.weight")
                .unwrap(),
        )
        .unwrap();
        assert_eq!(got.as_slice::<f32>(), [0.5, -1.0, 1.5]);
    }

    /// A deliberately non-32-aligned architecture: hidden 40 (2 heads × 20), so a single-stream
    /// `to_q` is logically `[40, 40]` and MXFP8 stores it 32-padded at `[64, 64]`. Padding is
    /// therefore observable — an undeclared plan lands on 64s, a declared one on 40s.
    fn unaligned_config() -> Krea2Config {
        let cfg = Krea2Config {
            in_channels: 12,
            patch_size: 2,
            hidden_size: 40,
            num_attention_heads: 2,
            num_kv_heads: 1,
            attention_head_dim: 20,
            num_layers: 1,
            intermediate_size: 44,
            norm_eps: 1e-5,
            axes_dims_rope: [4, 8, 8],
            rope_theta: 1000.0,
            timestep_embed_dim: 12,
            num_text_layers: 3,
            num_layerwise_text_blocks: 1,
            num_refiner_text_blocks: 1,
            text_hidden_dim: 40,
            text_intermediate_size: 44,
            text_num_attention_heads: 2,
            text_num_kv_heads: 1,
        };
        cfg.validate().expect("the test architecture is coherent");
        cfg
    }

    /// **sc-20644 — a real MXFP8 Krea DiT plans and decodes at the LOGICAL (unpadded) shape when a
    /// config is in scope, and at the stored padded shape when none is.**
    ///
    /// A synthetic MXFP8 layer: a `[64, 64]` E4M3 payload (the 32-padded storage of a logical
    /// `[40, 40]` `to_q`) plus its swizzled E8M0 `weight_scale` companion and the `mxfp8` descriptor.
    /// The padded rows/columns carry poison (E4M3 448 under a NaN exponent), so a wrong unpad or a
    /// missing declaration cannot pass by accident.
    #[test]
    fn mxfp8_plans_at_the_declared_logical_shape_and_at_stored_padding_without_a_config() {
        let cfg = unaligned_config();
        let logical = [cfg.q_dim(), cfg.hidden_size]; // [40, 40] — the `to_q` the DiT builds.
        let stored = [64_usize, 64]; // 32-padded on both axes.
        let scale_shape = mlx_gen::gen_core::mxfp8_scale_shape(stored);

        let mut values = vec![0x7E_u8; stored[0] * stored[1]]; // 448 poison in the padding
        for row in 0..logical[0] {
            for col in 0..logical[1] {
                values[row * stored[1] + col] = 0x38 + ((row + col) % 4) as u8;
            }
        }
        let mut scales = vec![0xFF_u8; scale_shape[0] * scale_shape[1]]; // NaN exponent poison
        for row in 0..logical[0] {
            for block in 0..logical[1].div_ceil(32) {
                scales[mlx_gen::gen_core::mxfp8_swizzled_scale_index(stored, row, block)] =
                    126 + ((row + block) % 3) as u8;
            }
        }

        let dir = native_fixture_dir();
        let path = dir.path().join("mxfp8-native.safetensors");
        let descriptor = br#"{"format": "mxfp8"}"#;
        write_native_safetensors(
            &path,
            &[
                (
                    "model.diffusion_model.blocks.0.attn.wq.weight",
                    "F8_E4M3",
                    &stored,
                    values,
                ),
                (
                    "model.diffusion_model.blocks.0.attn.wq.weight_scale",
                    "U8",
                    &scale_shape,
                    scales,
                ),
                (
                    "model.diffusion_model.blocks.0.attn.wq.comfy_quant",
                    "U8",
                    &[descriptor.len()],
                    descriptor.to_vec(),
                ),
            ],
        );

        // With the architecture config in scope the plan unpads to the DiT's real geometry.
        let declared = mlx_gen::logical_weights::plan_logical_weights(
            &path,
            &crate::native_remap::KreaNativeToDiffusersMapping::for_config(&cfg),
        )
        .expect("the mxfp8 layer plans through the Krea native mapping");
        assert_eq!(declared.tensors.len(), 1);
        assert_eq!(
            declared.tensors[0].shape,
            logical.to_vec(),
            "a declared logical shape must unpad the 32-padded storage"
        );
        // Dense residency is priced from the LOGICAL element count, not the padded one.
        assert_eq!(
            declared.resident_bytes(),
            (logical[0] * logical[1] * 2) as u64
        );

        // ...and the decoded array really is that shape, so the DiT's own architecture validation
        // (which is what refused a genuine MXFP8 Krea before sc-20644) now accepts the tensor.
        let (weights, _) =
            normalized_native_weights_with_receipt(&path, DeclaredLogicalShapes::FromConfig(&cfg))
                .expect("the mxfp8 layer decodes through the codec seam");
        assert_eq!(
            weights
                .require("transformer_blocks.0.attn.to_q.weight")
                .unwrap()
                .shape(),
            &[logical[0] as i32, logical[1] as i32],
        );

        // With no config in scope the pre-sc-20644 behaviour is unchanged: the stored padded shape.
        let undeclared = mlx_gen::logical_weights::plan_logical_weights(
            &path,
            &crate::native_remap::KreaNativeToDiffusersMapping::without_config(),
        )
        .expect("an undeclared mxfp8 layer still plans");
        assert_eq!(
            undeclared.tensors[0].shape,
            stored.to_vec(),
            "with no declaration the plan can only use the stored padded shape"
        );
        // Which is exactly the geometry the DiT refuses — the fail-closed backstop is intact.
        assert_ne!(undeclared.tensors[0].shape, logical.to_vec());
    }

    /// sc-20634 review: `LAST_NATIVE_FILE_RECEIPT` is process-global and its
    /// `reset → load → read` window is not atomic, so two tests observing it concurrently in this
    /// binary can clear or overwrite each other's observation. Every in-crate test now takes the
    /// receipt straight off the read (`*_with_receipt`); anything that still touches the global
    /// must hold `RECEIPT_LOCK` across the whole window. Pin that so a future test cannot
    /// reintroduce the race by reaching for `reset_native_file_receipt()` on its own.
    #[test]
    fn every_process_global_receipt_observation_is_serialized() {
        for (file, source) in [
            ("loader.rs", include_str!("loader.rs")),
            ("model.rs", include_str!("model.rs")),
        ] {
            let body = source
                .split_once("mod tests {")
                .map_or(source, |(_, tests)| tests);
            let resets = body.matches("reset_native_file_receipt()").count();
            let reads = body.matches("last_native_file_receipt()").count();
            let guards = body.matches("RECEIPT_LOCK").count();
            assert!(
                guards >= resets.max(reads),
                "{file}: {resets} reset(s) and {reads} read(s) of the process-global receipt but \
                 only {guards} RECEIPT_LOCK acquisition(s) — every observer must hold the lock \
                 across its reset → load → read window, or take the receipt from \
                 `normalized_native_weights_with_receipt` instead"
            );
        }
    }

    /// Real-weight check of the dense walking-skeleton input (sc-20634): the community
    /// `kreamania_variant5` DiT plans entirely onto the dense-bf16 codec through the Krea native
    /// mapping (430 tensors, every byte of the data region accounted for) and its deferred read
    /// goes through the codec seam with a recorded receipt. Header-only plus a lazy open: no GPU
    /// render. `KREA_NATIVE_DIT` points at the file.
    #[test]
    #[ignore = "needs real weights: set KREA_NATIVE_DIT to a dense bf16 ComfyUI Krea 2 DiT"]
    fn variant5_plans_as_dense_bf16_through_the_codec_seam() {
        let Some(dit) = std::env::var_os("KREA_NATIVE_DIT") else {
            panic!("set KREA_NATIVE_DIT to a dense bf16 ComfyUI Krea 2 single-file DiT");
        };
        let dit = std::path::PathBuf::from(dit);
        assert!(dit.is_file(), "{} is not a file", dit.display());
        let plan = mlx_gen::logical_weights::plan_logical_weights(
            &dit,
            &crate::native_remap::KreaNativeToDiffusersMapping::without_config(),
        )
        .expect("variant5 plans through the Krea native mapping");
        // variant5 is 415 bf16 + 15 f32 tensors (in/out projections and biases): both dense rows.
        assert_eq!(plan.codec_ids(), ["dense-bf16-v1", "dense-f32-v1"]);
        let headers = mlx_gen::gen_core::safetensors_path_tensor_headers(&dit).unwrap();
        assert_eq!(plan.tensor_count(), headers.len());
        let declared: u64 = headers.iter().map(|header| header.data_bytes).sum();
        assert_eq!(plan.source_bytes, declared);
        let mut file = std::fs::File::open(&dit).unwrap();
        let mut prefix = [0_u8; 8];
        std::io::Read::read_exact(&mut file, &mut prefix).unwrap();
        let header_len = u64::from_le_bytes(prefix);
        let file_len = std::fs::metadata(&dit).unwrap().len();
        assert_eq!(
            plan.source_bytes,
            file_len - 8 - header_len,
            "every byte of the data region is planned"
        );
        assert!(plan
            .logical_keys()
            .any(|key| key == "transformer_blocks.0.attn.to_q.weight"));

        let (lazy, receipt) =
            normalized_native_weights_lazy_with_receipt(&dit, DeclaredLogicalShapes::NotInScope)
                .expect("deferred read through the seam");
        assert!(lazy.get("img_in.weight").is_some());
        assert_eq!(
            receipt.mapping_id,
            crate::native_remap::KreaNativeToDiffusersMapping::MAPPING_ID
        );
        assert_eq!(receipt.tensor_count, headers.len());
        assert_eq!(receipt.source_bytes, declared);
        assert_eq!(
            receipt.materialization,
            mlx_gen::gen_core::LogicalReadMaterialization::Deferred
        );
        eprintln!(
            "RESULT variant5 tensors={} source_bytes={} codecs={:?}",
            receipt.tensor_count,
            receipt.source_bytes,
            plan.codec_ids()
        );
    }

    /// Real-weight check of the KreaMania fp8 shape (sc-20385): the plain `fp8_e4m3fn` cast of the
    /// community Krea 2 DiT — every fp8 tensor `F8_E4M3`, no scale companions, no descriptors —
    /// plans onto the scalar fp8 codec at **unit scale** through the Krea native mapping and its
    /// deferred read goes through the codec seam with a recorded receipt.
    /// Header plus a lazy open: no GPU render. `KREA_NATIVE_DIT_FP8` points at the file.
    ///
    /// The real artifact for this shape is `kreamania_variant7_fp8` (430 tensors: 265 `F8_E4M3`
    /// + 158 BF16 + 7 F32, no companions, no `__metadata__`).
    ///
    /// **This is NOT what the published V1/V2 fp8 artifacts are.** An earlier revision of this
    /// comment claimed they were; reading their headers disproved it. `kreamania_variant1_fp8` and
    /// `kreamania_variant2_fp8` are descriptor-gated **scaled** fp8: 942 tensors carrying 256
    /// `F8_E4M3` weights, 256 F32 `.weight_scale` companions with real per-tensor values (not 1.0),
    /// and 256 `.comfy_quant` descriptors — 160 `{"format": "float8_e4m3fn"}` plus 96
    /// `{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}` — with the other 174
    /// tensors left BF16. They exercise the companion-scale and forced-dense paths this test does
    /// not; `kreamania_scaled_fp8_plans_with_companion_scales_and_fpmm` covers them.
    #[test]
    #[ignore = "needs real weights: set KREA_NATIVE_DIT_FP8 to a plain fp8_e4m3fn Krea 2 DiT"]
    fn kreamania_fp8_cast_plans_as_scalar_fp8_through_the_codec_seam() {
        let Some(dit) = std::env::var_os("KREA_NATIVE_DIT_FP8") else {
            panic!("set KREA_NATIVE_DIT_FP8 to a plain fp8_e4m3fn Krea 2 single-file DiT");
        };
        let dit = std::path::PathBuf::from(dit);
        assert!(dit.is_file(), "{} is not a file", dit.display());
        let plan = mlx_gen::logical_weights::plan_logical_weights(
            &dit,
            &crate::native_remap::KreaNativeToDiffusersMapping::without_config(),
        )
        .expect("the fp8 cast plans through the Krea native mapping");
        // ComfyUI's UNETLoader cast converts the big linear weights and leaves norms/biases alone,
        // so a real plain-cast checkpoint is fp8 MIXED with dense rows — it is not fp8 throughout.
        // (An earlier revision asserted a single codec id and `2 * declared` residency here; that
        // was true of the locally produced surrogate, which cast every tensor, and false of every
        // real artifact.)
        assert!(
            plan.codec_ids().contains(&"fp8-e4m3-scalar-v1"),
            "{:?}",
            plan.codec_ids()
        );
        assert!(plan.companions.is_empty(), "a plain cast has no companions");
        let headers = mlx_gen::gen_core::safetensors_path_tensor_headers(&dit).unwrap();
        assert_eq!(plan.tensor_count(), headers.len());
        let declared: u64 = headers.iter().map(|header| header.data_bytes).sum();
        assert_eq!(plan.source_bytes, declared);

        // Every fp8 layer is at UNIT scale — the defining property of the plain cast, and what
        // separates it from the published V1/V2 scaled artifacts.
        let fp8: Vec<_> = plan
            .tensors
            .iter()
            .filter(|tensor| tensor.codec_id == "fp8-e4m3-scalar-v1")
            .collect();
        assert!(!fp8.is_empty(), "the fixture must carry fp8 layers");
        assert!(
            fp8.iter().all(|tensor| matches!(
                &tensor.codec,
                mlx_gen::gen_core::TensorCodecSpec::ScalarFp8 {
                    scale: mlx_gen::gen_core::ScalarScaleSource::Unit,
                    ..
                }
            )),
            "a plain cast carries no scale companion, so every layer plans at unit scale"
        );
        // Residency: fp8 layers double (→ bf16), dense rows keep their stored bytes. Summing the
        // two is the real relationship the blanket `2 * declared` was standing in for.
        let fp8_source: u64 = fp8.iter().map(|tensor| tensor.source_bytes).sum();
        let dense_source: u64 = plan
            .tensors
            .iter()
            .filter(|tensor| tensor.codec_id != "fp8-e4m3-scalar-v1")
            .map(|tensor| tensor.source_bytes)
            .sum();
        assert_eq!(
            plan.resident_bytes(),
            2 * fp8_source + dense_source,
            "fp8 → bf16 doubles; dense rows are byte-preserving"
        );
        let (lazy, receipt) =
            normalized_native_weights_lazy_with_receipt(&dit, DeclaredLogicalShapes::NotInScope)
                .expect("deferred read through the seam");
        assert!(lazy.get("img_in.weight").is_some());
        assert_eq!(receipt.tensor_count, headers.len());
        assert_eq!(
            receipt.materialization,
            mlx_gen::gen_core::LogicalReadMaterialization::Deferred
        );
        eprintln!(
            "RESULT kreamania-fp8 tensors={} source_bytes={} resident_bytes={}",
            plan.tensor_count(),
            plan.source_bytes,
            plan.resident_bytes()
        );
    }

    /// Real-weight check of the int8-per-row regression (sc-20385): the community
    /// `kreamania_variant4` (264 I8 projections + per-row scales + descriptors, mixed with dense
    /// f32/bf16 tensors) plans through the registry's `int8-per-row-v1` row — the former bespoke
    /// arm — with every companion consumed and every byte of the data region accounted for.
    /// `KREA_NATIVE_DIT_INT8` points at the file.
    #[test]
    #[ignore = "needs real weights: set KREA_NATIVE_DIT_INT8 to the int8-per-row kreamania_variant4"]
    fn kreamania_int8_plans_through_the_registry_codec() {
        let Some(dit) = std::env::var_os("KREA_NATIVE_DIT_INT8") else {
            panic!("set KREA_NATIVE_DIT_INT8 to the int8-per-row kreamania_variant4");
        };
        let dit = std::path::PathBuf::from(dit);
        assert!(dit.is_file(), "{} is not a file", dit.display());
        let plan = mlx_gen::logical_weights::plan_logical_weights(
            &dit,
            &crate::native_remap::KreaNativeToDiffusersMapping::without_config(),
        )
        .expect("variant4 plans through the registry int8 codec");
        assert!(plan.codec_ids().contains(&"int8-per-row-v1"));
        let int8_layers = plan
            .tensors
            .iter()
            .filter(|tensor| tensor.codec_id == "int8-per-row-v1")
            .count();
        assert_eq!(int8_layers, 264, "variant4 carries 264 int8 projections");
        assert_eq!(
            plan.companions.len(),
            264 * 2,
            "each projection consumes its weight_scale and comfy_quant companions"
        );
        let headers = mlx_gen::gen_core::safetensors_path_tensor_headers(&dit).unwrap();
        let declared: u64 = headers.iter().map(|header| header.data_bytes).sum();
        assert_eq!(plan.source_bytes, declared, "every byte is accounted for");
        let (lazy, receipt) =
            normalized_native_weights_lazy_with_receipt(&dit, DeclaredLogicalShapes::NotInScope)
                .expect("deferred read through the seam");
        assert!(lazy.get("transformer_blocks.0.attn.to_q.weight").is_some());
        assert!(
            lazy.keys()
                .all(|key| !key.ends_with(".comfy_quant") && !key.ends_with(".weight_scale")),
            "companions are consumed, never remapped"
        );
        assert_eq!(
            receipt.materialization,
            mlx_gen::gen_core::LogicalReadMaterialization::Deferred
        );
        eprintln!(
            "RESULT kreamania-int8 tensors={} int8_layers={int8_layers} source_bytes={} \
             resident_bytes={}",
            plan.tensor_count(),
            plan.source_bytes,
            plan.resident_bytes()
        );
    }

    /// Real-weight check of the **published KreaMania V1/V2 fp8** artifacts (sc-20385 AC3): these
    /// are descriptor-gated **scaled** fp8, not the plain cast — 942 tensors carrying 256
    /// `F8_E4M3` weights, 256 F32 `.weight_scale` companions with real per-tensor values, and 256
    /// `.comfy_quant` descriptors, of which **96 set `full_precision_matrix_mult`**, with the
    /// remaining 174 tensors left BF16.
    ///
    /// So one file exercises three things the plain cast cannot: `ScalarScaleSource::Companion`
    /// (never `Unit`), the `full_precision_matrix_mult` forced-dense rule, and mixed per-layer
    /// dispatch against dense bf16 — on the real published artifact.
    /// `KREA_NATIVE_DIT_FP8_SCALED` points at the file.
    #[test]
    #[ignore = "needs real weights: set KREA_NATIVE_DIT_FP8_SCALED to a published KreaMania V1/V2 fp8"]
    fn kreamania_scaled_fp8_plans_with_companion_scales_and_fpmm() {
        let Some(dit) = std::env::var_os("KREA_NATIVE_DIT_FP8_SCALED") else {
            panic!("set KREA_NATIVE_DIT_FP8_SCALED to a published KreaMania V1/V2 fp8 DiT");
        };
        let dit = std::path::PathBuf::from(dit);
        assert!(dit.is_file(), "{} is not a file", dit.display());
        let plan = mlx_gen::logical_weights::plan_logical_weights(
            &dit,
            &crate::native_remap::KreaNativeToDiffusersMapping::without_config(),
        )
        .expect("the published scaled fp8 artifact plans through the Krea native mapping");

        assert_eq!(plan.tensor_count(), 430);
        assert_eq!(plan.codec_ids(), ["dense-bf16-v1", "fp8-e4m3-scalar-v1"]);
        let fp8: Vec<_> = plan
            .tensors
            .iter()
            .filter(|tensor| tensor.codec_id == "fp8-e4m3-scalar-v1")
            .collect();
        assert_eq!(fp8.len(), 256, "256 scaled fp8 projections");

        // Every fp8 layer takes its scale from a companion — NOT the plain cast's unit scale.
        // This is the assertion the plain-cast test cannot make.
        assert!(
            fp8.iter().all(|tensor| matches!(
                &tensor.codec,
                mlx_gen::gen_core::TensorCodecSpec::ScalarFp8 {
                    scale: mlx_gen::gen_core::ScalarScaleSource::Companion { .. },
                    ..
                }
            )),
            "a published V1/V2 layer must never plan at unit scale"
        );
        let fpmm = fp8
            .iter()
            .filter(|tensor| tensor.codec.full_precision_matrix_mult())
            .count();
        assert_eq!(fpmm, 96, "96 layers declare full_precision_matrix_mult");
        // NOTE: asserting these layers plan *dense* would be vacuous here — MLX plans under
        // `DenseResidencyPolicy`, so every layer is dense whether or not the flag is honored
        // (verified: deleting the fpmm-forces-dense rule in the compiler leaves this test green).
        // The flag's residency consequence is tested where a packed policy exists, in gen-core's
        // `packed_policy_prices_stored_bytes_plus_scales_and_honors_full_precision_layers`. What
        // this assertion is worth on the real artifact is that the descriptor parse recovered the
        // flag from 96 of 256 real `.comfy_quant` payloads at all.

        // 256 weight_scale + 256 comfy_quant, every one consumed; every data byte accounted for.
        assert_eq!(plan.companions.len(), 256 * 2);
        let headers = mlx_gen::gen_core::safetensors_path_tensor_headers(&dit).unwrap();
        let declared: u64 = headers.iter().map(|header| header.data_bytes).sum();
        assert_eq!(plan.source_bytes, declared, "every byte is accounted for");

        let (lazy, receipt) =
            normalized_native_weights_lazy_with_receipt(&dit, DeclaredLogicalShapes::NotInScope)
                .expect("deferred read through the seam");
        assert!(lazy.get("transformer_blocks.0.attn.to_q.weight").is_some());
        assert!(
            lazy.keys()
                .all(|key| !key.ends_with(".comfy_quant") && !key.ends_with(".weight_scale")),
            "companions are consumed, never remapped"
        );
        assert_eq!(
            receipt.materialization,
            mlx_gen::gen_core::LogicalReadMaterialization::Deferred
        );
        eprintln!(
            "RESULT kreamania-scaled-fp8 tensors={} fp8={} fpmm={fpmm} source_bytes={} \
             resident_bytes={}",
            plan.tensor_count(),
            fp8.len(),
            plan.source_bytes,
            plan.resident_bytes()
        );
    }

    /// Real-weight check of the **NVFP4 fail-closed** path (sc-20385 review): the community
    /// `kreamania_variant7` is a genuine ComfyUI NVFP4 checkpoint — 224 `U8` packed weights
    /// `[6144, 3072]`, 224 `F8_E4M3` block scales `[6144, 384]`, 224 F32 `weight_scale_2`, with the
    /// quantization declared in the **file header** (`__metadata__._quantization_metadata`, every
    /// layer `{"format": "nvfp4"}`) rather than as `.comfy_quant` tensors. Both of those are out of
    /// scope for this story, and the plan must refuse **by name** rather than silently plan the
    /// `U8` payloads as dense weights or decode the `F8_E4M3` scales at unit scale.
    ///
    /// The refusal is the AC3 evidence for this artifact — an unsupported real checkpoint that
    /// fails closed is as good as a supported one that renders. **sc-20641 owns flipping this from
    /// a refusal to a supported codec**; when it lands, this test's expectation changes with it.
    /// `KREA_NATIVE_DIT_NVFP4` points at the file.
    #[test]
    #[ignore = "needs real weights: set KREA_NATIVE_DIT_NVFP4 to the nvfp4 kreamania_variant7"]
    fn kreamania_nvfp4_refuses_by_name_until_sc_20641() {
        let Some(dit) = std::env::var_os("KREA_NATIVE_DIT_NVFP4") else {
            panic!("set KREA_NATIVE_DIT_NVFP4 to the nvfp4 kreamania_variant7");
        };
        let dit = std::path::PathBuf::from(dit);
        assert!(dit.is_file(), "{} is not a file", dit.display());

        // The fixture really is the NVFP4 shape, not merely some file that happens to refuse.
        let headers = mlx_gen::gen_core::safetensors_path_tensor_headers(&dit).unwrap();
        let scale_2 = headers
            .iter()
            .filter(|header| header.name.ends_with(".weight_scale_2"))
            .count();
        let block_scales = headers
            .iter()
            .filter(|header| {
                header.name.ends_with(".weight_scale")
                    && header.dtype == mlx_gen::gen_core::weightsmeta::Dtype::F8_E4M3
            })
            .count();
        assert_eq!(scale_2, 224, "nvfp4 second-level scales");
        assert_eq!(block_scales, 224, "E4M3 block scales");
        assert!(
            headers
                .iter()
                .all(|header| !header.name.ends_with(".comfy_quant")),
            "this convention declares quantization in the file header, not in .comfy_quant tensors"
        );

        let error = mlx_gen::logical_weights::plan_logical_weights(
            &dit,
            &crate::native_remap::KreaNativeToDiffusersMapping::without_config(),
        )
        .expect_err("an nvfp4 checkpoint has no registered codec and must refuse");
        let error = error.to_string();
        // Names the exact tensor, the exact format, and the story that will support it.
        assert!(
            error.contains("weight_scale_2")
                && error.contains("nvfp4")
                && error.contains("sc-20641"),
            "{error}"
        );
        assert!(
            error.contains(".weight_scale_2\""),
            "the refusal must name the offending tensor, not just the format: {error}"
        );
        eprintln!("RESULT kreamania-nvfp4 refused: {error}");
    }
}
