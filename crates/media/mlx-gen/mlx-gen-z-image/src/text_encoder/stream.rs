//! Rung 4's **text-encoder** stream (SC-15794): materialize `model.layers.<i>` one window at a time
//! instead of holding all 36 resident.
//!
//! This is the same rung and the same driver the DiT stream uses
//! ([`mlx_gen::block_residency::run_windowed`]) applied to a different transformer — which is the
//! whole point of the component scope. Nothing here re-implements the window lifecycle.
//!
//! ## Why the encoder is worth streaming
//!
//! Rung 4 cut the denoise far enough that **conditioning became the binding phase** on the larger
//! tiers. Measured on z_image_turbo at 1024², Apple/Metal, real weights (SC-15794), against a decode
//! floor of ~4.365 GiB:
//!
//! | tier | encoder weights | conditioning, resident | one full window | window = 1 |
//! |---|---:|---:|---:|---:|
//! | bf16 | 7.440 GiB | 8.489 GiB (**bound the request**) | 2.718 GiB | 1.455 GiB |
//! | q4 | 2.132 GiB | 3.669 GiB | 2.398 GiB | 0.413 GiB |
//!
//! Weights, not activations, dominate the conditioning peak — the activation share is 13.7% on bf16
//! and 42% on q4 — so a window has real work to do.
//!
//! **Two separate effects, and it matters which is which.** Most of the bf16 drop is *streaming at
//! all*: even one full-width window drains the view per layer (see `materialize`) instead of holding
//! all 36 layers' tensors alive the way `TextEncoder::from_weights` does, which is 8.489 → 2.718 GiB
//! on its own. Tightening the window buys the rest. Before this change bf16 conditioning bound the
//! request above an 8 GB budget; it is now decode-bound.
//!
//! `text_encoder_window_real_weights` owns the ladder above and
//! `component_scope_real_weights` the end-to-end request-peak effect.
//!
//! ## It is a cheaper trade than the DiT stream
//!
//! The DiT re-materializes every block **once per denoise step**; the encoder runs **once per
//! generation**. Same mechanism, and the re-materialization is paid a single time — which is why the
//! encoder tolerates a tighter window than the DiT does.
//!
//! ## Z-Image is not where this matters most
//!
//! Rung 1 (staged residency) bounds the staged peak to `max(text_encoder, everything_else)`. For a
//! model whose encoder **is** the largest component, staging buys nothing — the encoder is already the
//! maximum — so an encoder window is the only lever that touches the binding component at all.
//!
//! Measured TE share of total on-disk weights across the locally cached catalog (SC-15794):
//!
//! | family | total | text encoder | share |
//! |---|---:|---:|---:|
//! | Mage-Flow components | 17.7 GiB | 16.7 GiB | **94.5%** |
//! | lens | 17.7 GiB | 13.4 GiB | **75.6%** |
//! | Kolors | 16.6 GiB | 11.6 GiB | **70.1%** |
//! | flux2-klein-9b | 47.6 GiB | 30.5 GiB | **64.1%** |
//! | Sana 1600M / Sana Sprint | 9.0 GiB | 4.9 GiB | **54.1%** |
//! | Mage-Flow (model repos) | 16.3 GiB | 8.3 GiB | **50.8%** |
//!
//! Z-Image is *not* in that list — its encoder is a minority of the model, and the win here came from
//! conditioning happening to bind the request after rung 4 cut the denoise. The families above are
//! where this rung should pay most, and each needs its own measurement before any of it is assumed:
//! share-of-disk predicts nothing about the *resident* peak (lens-turbo's mxfp4-on-disk gpt-oss
//! encoder expands to bf16 on load, which is where 17.24 GiB of its calibrated headroom goes).
//!
//! Caveat on the table: a multi-tier snapshot directory sums every tier, so these are whole-repo
//! ratios rather than per-tier ones. They rank the candidates; they are not the per-tier figure a
//! calibration record needs.

use std::ops::Range;

use mlx_rs::Array;

use mlx_gen::block_residency::BlockPlan;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result, WeightsSource};

use super::{EncoderLayer, ZTextEncoderConfig};

/// A re-openable source for the encoder's `model.layers.<i>` stack.
///
/// `source` must be re-openable — a snapshot directory or a single `.safetensors` file. An in-memory
/// / ComfyUI-normalized `Weights` has none, so those loads simply do not construct a stream and the
/// contract declares the text-encoder scope unavailable for them, exactly as the DiT stream does.
pub(crate) struct TextEncoderBlockStream {
    validated: mlx_gen::gen_core::ValidatedEncoderSource,
    /// Key prefix for the stack, so block `n` is `{base}.{n}` (`"model.layers"` on a real snapshot).
    base: String,
    cfg: ZTextEncoderConfig,
    /// The load-time quantization the resident encoder received, replayed per materialized block so a
    /// streamed layer is byte-identical to its resident twin. `None` on a dense/bf16 tier; on a
    /// pre-quantized (packed) snapshot this is recorded but is the same documented no-op the resident
    /// path performed.
    quant_bits: Option<i32>,
}

impl TextEncoderBlockStream {
    pub(crate) fn new_validated(
        validated: mlx_gen::gen_core::ValidatedEncoderSource,
        prefix: &str,
        cfg: ZTextEncoderConfig,
    ) -> Self {
        Self {
            validated,
            base: super::join(prefix, "layers"),
            cfg,
            quant_bits: None,
        }
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.cfg.n_layers
    }

    pub(crate) fn set_quant_bits(&mut self, bits: i32) {
        self.quant_bits = Some(bits);
    }

    /// Open a fresh lazy view of the encoder's weights. Called once per window by `run_windowed`.
    pub(crate) fn open(&self) -> Result<Weights> {
        let open = |source: &WeightsSource| match source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        };
        self.validated.read_unchanged(open)
    }

    /// Recheck the retained receipt after a lazy window graph has actually been evaluated. Opening
    /// a `Weights` view is not the payload read on MLX, so the guard must also span the later eval.
    fn ensure_source_unchanged(&self) -> Result<()> {
        self.validated.read_unchanged::<(), Error>(|_| Ok(()))
    }

    /// Materialize encoder layer `index` out of `view`, quantized exactly like its resident twin, then
    /// drain the view of precisely the tensors that layer read. Also returns the stored bytes of
    /// exactly those tensors (dtype x shape off the lazy handles; the read happens inside the layer's
    /// iOS quantum, which is priced with them) plus the packed copies of a replayed load-time
    /// quantization.
    pub(crate) fn materialize_counted(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<(EncoderLayer, u64)> {
        if index >= self.cfg.n_layers {
            return Err(Error::Msg(format!(
                "z-image text-encoder stream: layer {index} is out of range for a {}-layer encoder",
                self.cfg.n_layers
            )));
        }
        materialize_layer_counted(
            view,
            &format!("{}.{index}", self.base),
            &self.cfg,
            self.quant_bits,
        )
    }
}

/// Materialize the encoder layer at `prefix` out of `view`, quantized at `quant_bits` (the replayed
/// load-time quantization), with [`layer_counted`]'s count.
fn materialize_layer_counted(
    view: &mut Weights,
    prefix: &str,
    cfg: &ZTextEncoderConfig,
    quant_bits: Option<i32>,
) -> Result<(EncoderLayer, u64)> {
    let (mut layer, extra) = layer_counted(view, prefix, cfg, quant_bits)?;
    if let Some(bits) = quant_bits {
        layer.quantize(bits)?;
    }
    Ok((layer, extra))
}

/// Build the encoder layer at `prefix` from `view` (lazy handles: nothing is read), drain the view of
/// exactly the tensors it read, and return what the layer's iOS quantum adds for having been
/// materialized there: the stored bytes of those tensors plus, replaying `quant_bits`, the packed
/// copies the quantization builds. Does not quantize. Shared with the header-only iOS pricing
/// (`crate::ios_pricing`), which runs it over shape-only handles.
pub(crate) fn layer_counted(
    view: &mut Weights,
    prefix: &str,
    cfg: &ZTextEncoderConfig,
    quant_bits: Option<i32>,
) -> Result<(EncoderLayer, u64)> {
    let layer = EncoderLayer::from_weights(
        view,
        prefix,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.rms_norm_eps,
    )?;
    // LOAD-BEARING (SC-15750, re-established for this stack by SC-15794): the view keeps its own
    // refcounted handle to every tensor the constructor cloned. Draining exactly the accessed keys
    // is what makes the window's drop a real release rather than a no-op that still produces a
    // correct conditioning tensor. The DiT twin measured 8.0 MiB vs 238.4 MiB on this distinction.
    // Match the logical capacity domain of the resident inventory and replayed packed outputs.
    // Already-packed sources need this too: a production MLP scale/offset has 778240 stored
    // bytes but 786432 page-rounded capacity. This is not a native backing or release witness.
    let mut extra = view
        .accessed_entries()
        .iter()
        .fold(0u64, |total, (_, array)| {
            total.saturating_add(mlx_gen::ios_quantum::allocated_bytes(array.nbytes() as u64))
        });
    view.remove_accessed();
    if let Some(bits) = quant_bits {
        extra = extra.saturating_add(layer.quantize_bytes(bits));
    }
    Ok((layer, extra))
}

/// The encoder's carried state across a window boundary.
///
/// The resident forward returns `all_hidden_states[-2]` — the **second-to-last** layer output — so it
/// tracks two arrays, not one. Both must cross the window boundary, and both must be materialized
/// before the window's weights are released: `second_to_last` is an unevaluated graph referencing an
/// earlier window's weights, so dropping the window would otherwise free nothing (the exact trap the
/// materialize hook exists for).
pub(crate) struct EncoderCarry {
    pub(crate) prev: Array,
    pub(crate) second_to_last: Array,
}

/// Run the encoder's layer stack in windows, materializing each window's weights and releasing them
/// before advancing. Returns `all_hidden_states[-2]`, identical to the resident loop.
pub(crate) fn run_windowed_layers(
    stream: &TextEncoderBlockStream,
    plan: &BlockPlan,
    cancel: &CancelFlag,
    embed: Array,
    cos: &Array,
    sin: &Array,
    mask: &Array,
) -> Result<Array> {
    if plan.n_blocks() != stream.n_blocks() {
        return Err(Error::Msg(format!(
            "z-image text-encoder stream: block plan covers {} layers but the encoder has {}",
            plan.n_blocks(),
            stream.n_blocks()
        )));
    }
    let init = EncoderCarry {
        second_to_last: embed.clone(),
        prev: embed,
    };
    let out = mlx_gen::block_residency::run_windowed(
        plan,
        cancel,
        init,
        || stream.open(),
        |carry: EncoderCarry, view: &mut Weights, range: Range<usize>| {
            let EncoderCarry {
                mut prev,
                mut second_to_last,
            } = carry;
            for i in range {
                let (layer, read_bytes) = stream.materialize_counted(view, i)?;
                let cur = super::encoder::layer_quantum(&layer, &prev, cos, sin, mask, read_bytes)?;
                // Identical bookkeeping to the resident loop: after this step the hidden-state list
                // would end [..., prev, cur], so the second-to-last is the old `prev`.
                second_to_last = std::mem::replace(&mut prev, cur);
            }
            Ok(EncoderCarry {
                prev,
                second_to_last,
            })
        },
        |carry: &EncoderCarry| {
            mlx_rs::transforms::eval([&carry.prev, &carry.second_to_last])?;
            stream.ensure_source_unchanged()
        },
    )?;
    Ok(out.second_to_last)
}

#[cfg(all(test, target_os = "macos"))]
mod capacity_proofs {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::Write;

    const CHILD: &str = "ZIMAGE_TE_SOURCE_CAPACITY_CPU_CHILD";
    const DONE: &str = "ZIMAGE_TE_SOURCE_CAPACITY_CPU_DONE";
    const FQN: &str = "text_encoder::stream::capacity_proofs::packed_local_layer_source_capacity_matches_full_logical_quantum";

    struct Completion;
    impl Drop for Completion {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                assert_eq!(
                    unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
                    0,
                    "TE source-capacity CPU synchronization after fixture owners dropped"
                );
                std::fs::write(std::env::var(DONE).expect("fresh completion path"), b"ok")
                    .expect("write TE source-capacity completion");
            }
        }
    }

    #[test]
    fn source_capacity_rounds_only_above_one_page() {
        let page = mlx_gen::ios_quantum::PAGE_BYTES;
        for bytes in [0, 1, page - 1, page, page * 2] {
            assert_eq!(mlx_gen::ios_quantum::allocated_bytes(bytes), bytes);
        }
        assert_eq!(mlx_gen::ios_quantum::allocated_bytes(page + 1), page * 2);
        assert_eq!(
            mlx_gen::ios_quantum::allocated_bytes(page * 2 + 1),
            page * 3
        );
    }

    /// One real packed local-layer source factory, with only a header and a sparse payload extent.
    /// Nothing evaluates the payload. Dimensions come from the production config/group, not a
    /// corpus population golden. The oracle is a SOURCE-KEY capacity sum, not native alias identity.
    fn packed_source(
        root: &std::path::Path,
        cfg: &ZTextEncoderConfig,
        bits: i32,
    ) -> (std::path::PathBuf, BTreeMap<String, u64>, BTreeSet<String>) {
        let mut header = serde_json::Map::new();
        let mut selected = BTreeMap::new();
        let mut offset = 0u64;
        let mut tensor = |name: String, shape: Vec<i32>, dtype: &str, width: u64, used: bool| {
            let bytes = shape.iter().fold(width, |n, &dim| n * dim as u64);
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]
                }),
            );
            offset += bytes;
            if used {
                selected.insert(name, bytes);
            }
        };
        for (path, out, input) in [
            (
                "self_attn.q_proj",
                cfg.n_heads * cfg.head_dim,
                cfg.hidden_size,
            ),
            (
                "self_attn.k_proj",
                cfg.n_kv_heads * cfg.head_dim,
                cfg.hidden_size,
            ),
            (
                "self_attn.v_proj",
                cfg.n_kv_heads * cfg.head_dim,
                cfg.hidden_size,
            ),
            (
                "self_attn.o_proj",
                cfg.hidden_size,
                cfg.n_heads * cfg.head_dim,
            ),
            ("mlp.gate_proj", cfg.intermediate_size, cfg.hidden_size),
            ("mlp.up_proj", cfg.intermediate_size, cfg.hidden_size),
            ("mlp.down_proj", cfg.hidden_size, cfg.intermediate_size),
        ] {
            let base = format!("model.layers.0.{path}");
            tensor(
                format!("{base}.weight"),
                vec![out, input * bits / 32],
                "U32",
                4,
                true,
            );
            for suffix in ["scales", "biases"] {
                tensor(
                    format!("{base}.{suffix}"),
                    vec![out, input / crate::quant::GROUP_SIZE],
                    "BF16",
                    2,
                    true,
                );
            }
        }
        for (path, size) in [
            ("input_layernorm", cfg.hidden_size),
            ("post_attention_layernorm", cfg.hidden_size),
            ("self_attn.q_norm", cfg.head_dim),
            ("self_attn.k_norm", cfg.head_dim),
        ] {
            tensor(
                format!("model.layers.0.{path}.weight"),
                vec![size],
                "BF16",
                2,
                true,
            );
        }
        let untouched = BTreeSet::from([
            "model.layers.0.mlp.gate_proj.bias".to_owned(), // actual bias-free constructor ignores it
            "model.layers.0.unused".to_owned(),
            "model.layers.1.unused".to_owned(),
        ]);
        for key in &untouched {
            tensor(key.clone(), vec![1], "BF16", 2, false);
        }
        let mut bytes = serde_json::to_vec(&header).unwrap();
        while !bytes.len().is_multiple_of(8) {
            bytes.push(b' ');
        }
        let path = root.join(format!("packed-q{bits}.safetensors"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(8 + bytes.len() as u64 + offset).unwrap();
        (path, selected, untouched)
    }

    #[test]
    fn packed_local_layer_source_capacity_matches_full_logical_quantum() {
        // Declared FIRST and armed LAST: the marker follows all native owners and fixture scopes.
        let mut completion: Option<Completion> = None;
        if std::env::var_os(CHILD).is_none() {
            let dir = tempfile::tempdir().unwrap();
            let marker = dir.path().join("done");
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FQN, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .env(DONE, &marker)
                .status()
                .unwrap();
            assert!(status.success(), "TE source-capacity CPU child failed");
            assert_eq!(
                std::fs::read(marker).expect("fresh normal-completion marker"),
                b"ok"
            );
            return;
        }
        // BEFORE any Array/model/file factory; no inherited native device or stream is trusted.
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        assert_eq!(
            mlx_rs::Device::try_default().unwrap(),
            mlx_rs::Device::cpu()
        );
        assert_eq!(
            unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
            0
        );
        assert_eq!(
            mlx_rs::Stream::new().get_index().unwrap(),
            mlx_rs::Stream::cpu().get_index().unwrap()
        );
        eprintln!("ZIMAGE_CPU_GUARDS_READY_V1 device=cpu native_stream=cpu fatal_metal_trap=1");
        let mut completed = 0u8;
        {
            let dir = tempfile::tempdir().unwrap();
            let cfg = ZTextEncoderConfig::z_image();
            let page = mlx_gen::ios_quantum::PAGE_BYTES;
            for (tier, stored_bits) in [4, 8].into_iter().enumerate() {
                let (path, selected, untouched) = packed_source(dir.path(), &cfg, stored_bits);
                let headers =
                    mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&path).unwrap();
                let selected_headers = headers
                    .iter()
                    .filter(|h| selected.contains_key(&h.name))
                    .collect::<Vec<_>>();
                assert_eq!(
                    selected_headers
                        .iter()
                        .map(|h| h.name.clone())
                        .collect::<BTreeSet<_>>(),
                    selected.keys().cloned().collect::<BTreeSet<_>>()
                );
                let raw = selected_headers.iter().map(|h| h.data_bytes).sum::<u64>();
                let capacity = selected_headers
                    .iter()
                    .map(|h| {
                        let bytes = h.data_bytes;
                        if bytes > page {
                            bytes.div_ceil(page) * page
                        } else {
                            bytes
                        }
                    })
                    .sum::<u64>();
                assert!(
                    capacity > raw,
                    "derived packed geometry must discriminate source capacity"
                );
                for (replay, quant_bits) in [None, Some(4), Some(8)].into_iter().enumerate() {
                    let mut view = Weights::from_file(&path).unwrap();
                    // Re-reading ONE key must not count it twice. This is not arbitrary alias-key
                    // deduplication: distinct keys sharing an ArrayDesc remain a different domain.
                    let key = selected.keys().next().unwrap();
                    view.require(key).unwrap();
                    view.require(key).unwrap();
                    let (layer, read) =
                        materialize_layer_counted(&mut view, "model.layers.0", &cfg, quant_bits)
                            .unwrap();
                    assert_eq!(
                        view.keys().map(str::to_owned).collect::<BTreeSet<_>>(),
                        untouched,
                        "TE-CAPACITY-EXACT-DRAIN: only actual constructor keys leave the view"
                    );
                    assert!(view.accessed_entries().is_empty());
                    // This PRIVATE layer API no-op does not authorize a production Q4->Q8 load:
                    // needs_load_time_quant's stored/requested mismatch refusal is unchanged.
                    assert_eq!(layer.quantize_bytes(quant_bits.unwrap_or(stored_bits)), 0);
                    let scratch = layer.quantum_scratch(
                        1,
                        crate::loader::MAX_LENGTH as i32,
                        4,
                        mlx_gen::ios_quantum::Backend::Metal,
                    );
                    let total = scratch + read;
                    assert_eq!(total, scratch + capacity,
                        "TE-CAPACITY-FULL-QUANTUM: SAME layer scratch plus independent source-key capacity");
                    // Fresh exact child, with NO admitted/calibrated envelope ever opened. Assert
                    // the actual requirement's full scratch AND unchanged reserve, not a delta.
                    let old_total = scratch + raw;
                    let reserve = mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES;
                    mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(old_total + reserve, || {
                        assert!(mlx_gen::ios_quantum::take_refusal().is_none());
                        mlx_gen::ios_quantum::before(crate::quantum::TE_LAYER, old_total)?;
                        assert!(mlx_gen::ios_quantum::before(crate::quantum::TE_LAYER, total).is_err(),
                            "TE-CAPACITY-FULL-BUDGET: corrected logical total refuses before graph construction");
                        let refusal = mlx_gen::ios_quantum::take_refusal().expect("actual logical guard refusal");
                        assert_eq!((refusal.scratch_bytes, refusal.reserve_bytes), (total, reserve));
                        Ok(())
                    }).unwrap();
                    let index = tier * 3 + replay;
                    completed |= 1 << index;
                    eprintln!("ZIMAGE_TE_SOURCE_CAPACITY_MECHANISM_V1 index={index}");
                }
            }
        }
        assert_eq!(
            completed, 63,
            "all packed source-capacity mechanisms completed"
        );
        eprintln!(
            "ZIMAGE_TE_SOURCE_CAPACITY_BODY_COMPLETED_V1 mechanisms=6 fullmask=63 payload_eval=0"
        );
        let _ = completion.replace(Completion);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validated_stream_rechecks_source_before_each_deferred_open() {
        let fixture = tempfile::tempdir().unwrap();
        let contract = crate::bounded_encoder_contract();
        gen_core_testkit::write_encoder_contract_fixture(fixture.path(), contract).unwrap();
        let source = contract
            .validate_source(&WeightsSource::Dir(fixture.path().to_path_buf()))
            .unwrap();
        let stream =
            TextEncoderBlockStream::new_validated(source, "model", ZTextEncoderConfig::z_image());

        std::fs::write(fixture.path().join("config.json"), b"{}\n").unwrap();
        let error = stream
            .open()
            .err()
            .expect("a deferred reopen must reject a post-validation replacement")
            .to_string();
        assert!(error.contains("changed after load"), "{error}");
    }

    #[test]
    fn validated_stream_rechecks_source_after_a_lazy_view_was_opened() {
        let fixture = tempfile::tempdir().unwrap();
        let contract = crate::bounded_encoder_contract();
        gen_core_testkit::write_encoder_contract_fixture(fixture.path(), contract).unwrap();
        let source = contract
            .validate_source(&WeightsSource::Dir(fixture.path().to_path_buf()))
            .unwrap();
        let stream =
            TextEncoderBlockStream::new_validated(source, "model", ZTextEncoderConfig::z_image());

        let _lazy_view = stream.open().unwrap();
        std::fs::write(fixture.path().join("config.json"), b"{}\n").unwrap();
        let error = stream
            .ensure_source_unchanged()
            .expect_err("payload evaluation must retain the source guard after lazy open")
            .to_string();
        assert!(error.contains("changed after load"), "{error}");
    }
}
