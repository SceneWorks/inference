//! Ladder rung 4 — **bounded transformer residency** for the SDXL U-Net (SC-15525, closing SC-16355).
//!
//! The window lifecycle is NOT implemented here: it is
//! [`mlx_gen::block_residency::run_windowed`] (SC-15750), the shared primitive Z-Image, Anima, Qwen,
//! Mage-Flow and FLUX also drive. This module is only the family-side half — "how do I rebuild SDXL
//! sub-stack *S*'s block *n* from the snapshot, carrying the same tier and the same installed state
//! the resident block carries" — which is the part that genuinely differs per family.
//!
//! ## What makes SDXL different from every ladder shipped before it
//!
//! Every prior MLX rung-4 adopter is a **DiT with one flat block list**, so its stream is one object
//! addressing `blocks.{n}` and its plan is one [`BlockPlan`](mlx_gen::block_residency::BlockPlan).
//! SDXL is the first **U-Net**: the windowable blocks are 70 `TransformerBlock`s distributed across
//! **eleven** independent `Transformer2D` sub-stacks at two depths —
//!
//! | sub-stack | key prefix | count | depth |
//! |---|---|---:|---:|
//! | `down_blocks.1.attentions.{0,1}` | 2 | 2 | 2 |
//! | `down_blocks.2.attentions.{0,1}` | 2 | 2 | 10 |
//! | `mid_block.attentions.0` | 1 | 1 | 10 |
//! | `up_blocks.0.attentions.{0,1,2}` | 3 | 3 | 10 |
//! | `up_blocks.1.attentions.{0,1,2}` | 3 | 3 | 2 |
//!
//! — six 10-deep and **five** 2-deep. (SC-16355's description said "four 2-deep"; the up path has
//! `layers_per_block + 1 = 3` attentions, not 2, so it is five. `up_blocks.2` / `down_blocks.0` are
//! plain `UpBlock2D`/`DownBlock2D` and carry no attention at all, which is why the config's
//! `transformer_layers_per_block[0] = 1` is never read.)
//!
//! So there is **one stream per sub-stack**, each with its own prefix, its own depth and its own
//! plan, and one U-Net forward drives `run_windowed` eleven times rather than once. That is the
//! whole of SC-16355's "a per-`Transformer2D` block loader, not a per-model one".
//!
//! The conv/resnet trunk, the up/down samplers, the `norm`/`proj_in`/`proj_out` of each sub-stack,
//! and the down→up skip stack are the **resident remainder**: this rung does not bound them.
//!
//! ## Why re-reading per window is nearly free
//!
//! [`Weights::from_file`] goes through `Array::load_safetensors_with_metadata`, which is **lazy per
//! tensor**: the handles exist, the bytes do not, until something is evaluated. Re-opening the
//! U-Net's `diffusion_pytorch_model{,.fp16}.safetensors` once per window costs a JSON header parse,
//! not the tier. Only the tensors a window's blocks actually read are materialized, and
//! [`Weights::remove_accessed`] then drops the view's own reference to exactly those.
//!
//! `remove_accessed` is load-bearing rather than decorative: `Array` is refcounted, and
//! `TransformerBlock::from_weights` **clones** out of the view (it is the same constructor the
//! resident path uses). Draining after each block is what turns the window's drop into a real
//! release. The drain is *exact*, so a constructor read that never happened stays visible instead of
//! being wiped along with the rest.
//!
//! ## Replay: the two kinds of installed state, and the one that cannot be replayed
//!
//! A block rebuilt from the snapshot carries only what is *on disk*. Two things the resident block
//! holds are not:
//!
//! 1. **IP-Adapter decoupled K/V projections** (sc-3059). These are always forward-time installed
//!    state (`AttentionMHA::to_k_ip` / `to_v_ip`), so they are captured per block and re-installed
//!    verbatim. This matters more here than the count suggests: `install_ip_adapter` consumes one
//!    pair per cross-attention in **down → up → mid** order from a single iterator, so the mapping
//!    from pair index to block is positional and unreconstructable from a key path. Capturing the
//!    built `AdaptableLinear` off the resident block sidesteps re-deriving it entirely — and it is
//!    captured *after* `unet.quantize`, so a packed tier's IP projections replay packed.
//!    A streamed block that loses its pair would produce a **plausible wrong image** with no error,
//!    which is why `SdxlBlockStream::materialize` fails loudly instead
//!    (`ip_expected` / `has_ip`).
//! 2. **LoRA / LoKr adapters** — and these split. On a *packed* (pre-quantized Q4/Q8) snapshot
//!    `apply_sdxl_adapters_with` installs them as forward-time residuals (`AdaptableLinear::push`),
//!    which capture and replay exactly like Z-Image's. On a *dense* snapshot it calls
//!    `merge_dense_delta`, folding the delta into the base weight — and **that is gone from the
//!    snapshot**. There is no residual to capture, and a re-read block would silently be the
//!    un-adapted model.
//!
//!    This module therefore captures residuals when they exist, and `model::load_heavy` refuses to
//!    arm streaming at all for an adapter load that merged (see
//!    `memory_strategy::adapters_are_replayable`). Refusing is the only honest option: the
//!    alternative is a render that quietly drops the LoRA the user asked for.

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};
use mlx_rs::Dtype;

use crate::unet::transformer::TransformerBlock;

/// The forward-time residual adapters installed on ONE block, addressed by the block-local dotted
/// path (`"attn1.to_q"`, `"attn2.to_out.0"`, `"ff.linear1"`, …).
///
/// Captured from the resident blocks after `apply_sdxl_adapters_with` has run, so the streamed block
/// replays exactly what the resident block ended up holding rather than a second derivation of which
/// adapter should have landed where. [`Adapter`] holds refcounted `Array` factors, so a capture is a
/// handle copy — a rank-16 LoRA's factors, not an `[out, in]` delta.
#[derive(Clone, Default)]
pub(crate) struct BlockAdapters {
    per_path: Vec<(String, Vec<Adapter>)>,
}

impl BlockAdapters {
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.per_path.is_empty()
    }

    fn install(&self, block: &mut TransformerBlock) -> Result<()> {
        for (path, adapters) in &self.per_path {
            let segs: Vec<&str> = path.split('.').collect();
            let target = block.adaptable_mut(&segs).ok_or_else(|| {
                Error::Msg(format!(
                    "sdxl block stream: adapter target `{path}` is absent from a materialized \
                     block — the streamed block tree does not match the resident one"
                ))
            })?;
            target.set_adapters(adapters.clone());
        }
        Ok(())
    }
}

/// The IP-Adapter decoupled K/V projections captured off ONE resident block.
#[derive(Clone)]
struct BlockIp {
    k: AdaptableLinear,
    v: AdaptableLinear,
}

/// Everything needed to rebuild one `Transformer2D`'s blocks from the U-Net snapshot, on demand.
///
/// Held by `Transformer2D` — one per sub-stack — and consumed by its
/// windowed forward. Cheap to clone: a path, a prefix, three scalars and refcounted handles.
#[derive(Clone)]
pub(crate) struct SdxlBlockStream {
    /// The re-openable U-Net checkpoint. Resolved to the exact file the resident stack was built
    /// from (`unet/diffusion_pytorch_model{,.fp16}.safetensors`) rather than the caller's raw
    /// `LoadSpec::weights`, which is the snapshot ROOT and would re-read every component.
    source: WeightsSource,
    source_inventory: Option<mlx_gen::gen_core::SourceInventory>,
    /// This sub-stack's `attentions.{i}` prefix, already absolute
    /// (`down_blocks.2.attentions.0`, `mid_block.attentions.0`, …), so block `n` is
    /// `{base}.transformer_blocks.{n}`.
    prefix: String,
    model_dims: i32,
    num_heads: i32,
    n_blocks: usize,
    /// The compute dtype the resident stack was cast to at load (`loader::cast_for_load`). Every
    /// re-read window applies the same packed-aware cast, so a producer's stored-f32 dense leaves
    /// (norms, biases) come back at the resident width instead of promoting the window to f32.
    dtype: Dtype,
    /// The load-time quantization the resident stack received, replayed per materialized block so a
    /// streamed block is byte-identical to its resident twin. `None` for a dense/fp16 tier. On a
    /// pre-quantized (packed) snapshot this is recorded but is a documented no-op — the same no-op
    /// the resident path performed.
    quant_bits: Option<i32>,
    /// Per-block captured residual adapters, indexed by block. Empty when none is installed.
    adapters: Vec<BlockAdapters>,
    /// Per-block captured IP-Adapter projections, indexed by block. `None` when this block's
    /// cross-attention carries no IP pair.
    ip: Vec<Option<BlockIp>>,
    /// Whether the resident stack carried IP projections at all. A streamed block that comes back
    /// without one when this is set is a hard error, not a silent quality regression.
    ip_expected: bool,
    /// Per-block streamed weight bytes ([`Self::streamed_bytes`]), read from the source's
    /// safetensors headers once, on first use.
    block_bytes: std::sync::Arc<std::sync::OnceLock<Vec<u64>>>,
}

impl SdxlBlockStream {
    /// Declare a streamable sub-stack. `source` must be re-openable — the resolved U-Net weight
    /// file. An in-memory / LDM-split `Weights` has no re-openable source, so those loads simply do
    /// not construct one and rung 4 is declared unavailable for them.
    pub(crate) fn new(
        source: WeightsSource,
        prefix: &str,
        model_dims: i32,
        num_heads: i32,
        n_blocks: usize,
        dtype: Dtype,
        quant_bits: Option<i32>,
    ) -> Self {
        Self {
            source,
            source_inventory: None,
            prefix: prefix.to_owned(),
            model_dims,
            num_heads,
            n_blocks,
            dtype,
            quant_bits,
            adapters: vec![BlockAdapters::default(); n_blocks],
            ip: vec![None; n_blocks],
            ip_expected: false,
            block_bytes: std::sync::Arc::default(),
        }
    }

    /// The bytes block `index` brings into the quantum that materializes it from the snapshot (its
    /// iOS price beside the block's activations — `Transformer2D::block_price`), from the source's
    /// safetensors headers: every stored tensor of the block as read, plus the copy
    /// [`crate::loader::cast_for_load`] makes of each tensor it casts (the iOS rule: every differing
    /// tensor of a dense snapshot; on a packed one the dense float leaves, never the codes or their
    /// `.scales`/`.biases`), plus the GEGLU `ff.net.0.proj` value/gate halves, row slices that live
    /// beside the full stored projection they are sliced from, and the two `[hidden]` i32 row
    /// selections that slice them — each one page-rounded allocation. The captured IP projections and
    /// adapters are resident, not read. A load-time quantization replay over a dense snapshot never
    /// arms (`memory_strategy::streamable`), so it is refused here rather than priced.
    pub(crate) fn streamed_bytes(&self, index: usize) -> Result<u64> {
        if self.block_bytes.get().is_none() {
            let bytes = self.read_block_bytes()?;
            let _ = self.block_bytes.set(bytes);
        }
        self.block_bytes
            .get()
            .and_then(|bytes| bytes.get(index).copied())
            .ok_or_else(|| {
                Error::Msg(format!(
                    "sdxl block stream `{}`: block {index} is out of range for a {}-block \
                     sub-stack",
                    self.prefix, self.n_blocks
                ))
            })
    }

    fn read_block_bytes(&self) -> Result<Vec<u64>> {
        use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
        let files: Vec<std::path::PathBuf> = match &self.source {
            WeightsSource::File(file) => vec![file.clone()],
            WeightsSource::Dir(dir) => {
                let mut files: Vec<_> = std::fs::read_dir(dir)
                    .map_err(|e| Error::Msg(format!("sdxl block stream: {e}")))?
                    .filter_map(|entry| entry.ok().map(|e| e.path()))
                    .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
                    .collect();
                files.sort();
                files
            }
        };
        let (target, width) = stream_target(self.dtype)?;
        let mut headers = Vec::new();
        for file in &files {
            headers.extend(
                safetensors_path_tensor_headers(file).map_err(|e| Error::Msg(e.to_string()))?,
            );
        }
        streamed_block_bytes_for_target(
            &headers,
            &self.prefix,
            self.n_blocks,
            self.quant_bits,
            target,
            width,
        )
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    #[cfg(test)]
    pub(crate) fn has_adapters(&self) -> bool {
        self.adapters.iter().any(|a| !a.is_empty())
    }

    #[cfg(test)]
    pub(crate) fn expects_ip(&self) -> bool {
        self.ip_expected
    }

    /// Capture everything a materialized block must reproduce off the **resident** blocks: the
    /// forward-time residual adapters and the IP-Adapter K/V projections.
    ///
    /// Called once, after the resident stack has been built, IP-installed, adapted and quantized, so
    /// the two paths cannot diverge on *which* state, *which* target, or *what order*.
    pub(crate) fn capture_from(&mut self, blocks: &mut [TransformerBlock]) {
        self.adapters = blocks
            .iter_mut()
            .map(|block| {
                let paths = block.adaptable_paths();
                let mut per_path = Vec::new();
                for path in paths {
                    let segs: Vec<&str> = path.split('.').collect();
                    // SC-18319 — a capture walks EVERY projection, so it resolves through the PROBE
                    // half and takes the `&mut` only where something is actually installed.
                    if block
                        .adaptable_facts(&segs)
                        .is_some_and(|f| f.adapter_count > 0)
                    {
                        let adapters = block
                            .adaptable_mut(&segs)
                            .expect("resolved through the probe above")
                            .adapters()
                            .to_vec();
                        per_path.push((path.clone(), adapters));
                    }
                }
                BlockAdapters { per_path }
            })
            .collect();
        self.ip = blocks
            .iter()
            .map(|block| block.ip_projections().map(|(k, v)| BlockIp { k, v }))
            .collect();
        self.ip_expected = self.ip.iter().any(Option::is_some);
    }

    /// Open a fresh lazy view of the U-Net checkpoint, cast exactly as the resident load cast it.
    /// Called once per window by [`run_windowed`](mlx_gen::block_residency::run_windowed).
    ///
    /// The cast is the resident path's own rule ([`crate::loader::cast_for_load`]: packs untouched,
    /// every dense leaf to `dtype`), applied to the whole view once per window rather than per
    /// block. It is lazy — dtype and key reads only — so only the tensors a block actually reads are
    /// ever converted. Without it a streamed block keeps the file's stored width: a packed snapshot
    /// whose producer wrote its norms/biases at f32 would hand the window f32 dense leaves while the
    /// resident twin holds f16, breaking byte-identity and promoting the window's activations.
    pub(crate) fn bind_source_inventory(&mut self, guard: mlx_gen::gen_core::SourceInventory) {
        self.source_inventory = Some(guard);
    }

    pub(crate) fn open(&self) -> Result<Weights> {
        if let Some(guard) = &self.source_inventory {
            guard.verify_unchanged()?;
        }
        let mut view = match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir)?,
            WeightsSource::File(file) => Weights::from_file(file)?,
        };
        crate::loader::cast_for_load(&mut view, self.dtype)?;
        if let Some(guard) = &self.source_inventory {
            guard.verify_unchanged()?;
        }
        Ok(view)
    }

    /// Materialize block `index` of this sub-stack out of `view` — quantized, adapted and
    /// IP-installed exactly like its resident twin — then drain the view of precisely the tensors
    /// that block read.
    pub(crate) fn materialize(&self, view: &mut Weights, index: usize) -> Result<TransformerBlock> {
        if let Some(guard) = &self.source_inventory {
            guard.verify_unchanged()?;
        }
        if index >= self.n_blocks {
            return Err(Error::Msg(format!(
                "sdxl block stream `{}`: block {index} is out of range for a {}-block sub-stack",
                self.prefix, self.n_blocks
            )));
        }
        let prefix = format!("{}.transformer_blocks.{index}", self.prefix);
        let mut block =
            TransformerBlock::from_weights(view, &prefix, self.model_dims, self.num_heads)?;
        // LOAD-BEARING (SC-15750): the view keeps its own refcounted handle to every tensor the
        // constructor cloned. Draining exactly the accessed keys is what makes the window's drop a
        // real release rather than a no-op that still produces correct images.
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            block.quantize(bits)?;
        }
        // IP replay BEFORE the adapter replay, mirroring `model::load_heavy`'s order (install_ip →
        // adapters → quantize is the load order; the captured artifacts are post-everything, so
        // only their relative install order here has to be stable and total).
        match (&self.ip[index], self.ip_expected) {
            (Some(ip), _) => block.set_ip_projections(ip.k.clone(), ip.v.clone()),
            // The resident stack HAD IP projections and this block's capture is empty. That is not a
            // recoverable state: the streamed block would run its cross-attention without the image
            // prompt while its neighbours ran with it — a plausible, wrong image and no error
            // anywhere. Refuse loudly.
            (None, true) => {
                return Err(Error::Msg(format!(
                    "sdxl block stream `{}`: block {index} has no captured IP-Adapter K/V pair, but \
                     this sub-stack was armed with IP projections installed. A streamed block \
                     without its pair would silently drop the image prompt for these layers.",
                    self.prefix
                )));
            }
            (None, false) => {}
        }
        if !block.has_ip() && self.ip_expected {
            return Err(Error::Msg(format!(
                "sdxl block stream `{}`: block {index} came back without IP-Adapter projections \
                 after replay",
                self.prefix
            )));
        }
        self.adapters[index].install(&mut block)?;
        if let Some(guard) = &self.source_inventory {
            // Lazy file arrays must be consumed while the retained identity is guarded, not
            // returned to a later forward that might otherwise reopen a replaced source.
            block.materialize_weights()?;
            guard.verify_unchanged()?;
        }
        Ok(block)
    }
}

// Shared pure header budget. Native reader validation stays before its header reads; the metadata
// hierarchy uses the same selector and arithmetic without constructing a block stream/model.
fn stream_target(dtype: Dtype) -> Result<(mlx_gen::gen_core::weightsmeta::Dtype, u64)> {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    Ok(match dtype {
        Dtype::Float16 => (Stored::F16, 2),
        Dtype::Bfloat16 => (Stored::BF16, 2),
        Dtype::Float32 => (Stored::F32, 4),
        other => {
            return Err(Error::Msg(format!(
                "sdxl block stream: unsupported compute dtype {other:?}"
            )))
        }
    })
}

pub(crate) fn streamed_block_bytes_from_headers(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    prefix: &str,
    n_blocks: usize,
    dtype: Dtype,
    quant_bits: Option<i32>,
) -> Result<Vec<u64>> {
    let (target, width) = stream_target(dtype)?;
    streamed_block_bytes_for_target(headers, prefix, n_blocks, quant_bits, target, width)
}

fn streamed_block_bytes_for_target(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    prefix: &str,
    n_blocks: usize,
    quant_bits: Option<i32>,
    target: mlx_gen::gen_core::weightsmeta::Dtype,
    width: u64,
) -> Result<Vec<u64>> {
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    let packed = headers.iter().any(|h| h.name.ends_with(".scales"));
    if quant_bits.is_some() && !packed {
        return Err(Error::Unsupported(format!(
            "sdxl block stream `{}`: a load-time quantization replay over a dense snapshot is \
                 not a streamable load and has no iOS price",
            prefix
        )));
    }
    let stem = format!("{}.transformer_blocks.", prefix);
    let mut bytes = vec![0u64; n_blocks];
    for header in headers {
        let Some(rest) = header.name.strip_prefix(&stem) else {
            continue;
        };
        let Some((index, leaf)) = rest.split_once('.') else {
            continue;
        };
        let Ok(index) = index.parse::<usize>() else {
            continue;
        };
        let Some(slot) = bytes.get_mut(index) else {
            continue;
        };
        let floating = matches!(
            header.dtype,
            Stored::F16 | Stored::BF16 | Stored::F32 | Stored::F64
        );
        let companion = header.name.ends_with(".scales") || header.name.ends_with(".biases");
        let casts = header.dtype != target && (!packed || (floating && !companion));
        let elements = header
            .shape
            .iter()
            .fold(1u64, |a, &d| a.saturating_mul(d as u64));
        let page = mlx_gen::ios_quantum::allocated_bytes;
        let resident = page(if casts {
            elements.saturating_mul(width)
        } else {
            header.data_bytes
        });
        let cast_copy = if casts { resident } else { 0 };
        // The value/gate halves (two row slices together the projection's size) and the two
        // `[hidden]` i32 row selections that slice them.
        let geglu = if leaf.starts_with("ff.net.0.proj.") {
            let half = header.shape.first().copied().unwrap_or(0) as u64 / 2;
            let selections =
                if leaf == "ff.net.0.proj.scales" || (leaf == "ff.net.0.proj.weight" && !packed) {
                    2 * page(half.saturating_mul(4))
                } else {
                    0
                };
            resident.saturating_add(selections)
        } else {
            0
        };
        *slot = slot
            .saturating_add(page(header.data_bytes))
            .saturating_add(cast_copy)
            .saturating_add(geglu);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    const DIMS: i32 = 64;
    const HEADS: i32 = 2;

    /// A tiny two-block SDXL `Transformer2D` sub-stack written to a temp `.safetensors`, so the
    /// stream can be exercised without the 6.6 GB snapshot. Widths are group-64-divisible so the
    /// quantization replay test can pack them.
    fn fixture(tmp: &tempfile::TempDir, tag: &str, n_blocks: usize) -> std::path::PathBuf {
        let dir = tmp.path().join(format!("sdxl-block-stream-{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut named: Vec<(String, Array)> = Vec::new();
        let mut push = |key: String, shape: Vec<i32>| {
            let n: i32 = shape.iter().product();
            let values: Vec<f32> = (0..n).map(|k| ((k % 97) as f32) * 0.01).collect();
            named.push((key, Array::from_slice(&values, &shape)));
        };
        for i in 0..n_blocks {
            let b = format!("attentions.0.transformer_blocks.{i}");
            for norm in ["norm1", "norm2", "norm3"] {
                push(format!("{b}.{norm}.weight"), vec![DIMS]);
                push(format!("{b}.{norm}.bias"), vec![DIMS]);
            }
            for attn in ["attn1", "attn2"] {
                for leaf in ["to_q", "to_k", "to_v"] {
                    push(format!("{b}.{attn}.{leaf}.weight"), vec![DIMS, DIMS]);
                }
                push(format!("{b}.{attn}.to_out.0.weight"), vec![DIMS, DIMS]);
                push(format!("{b}.{attn}.to_out.0.bias"), vec![DIMS]);
            }
            // GEGLU: one `[2*hidden, D]` linear, row-split into value/gate halves.
            push(format!("{b}.ff.net.0.proj.weight"), vec![2 * DIMS, DIMS]);
            push(format!("{b}.ff.net.0.proj.bias"), vec![2 * DIMS]);
            push(format!("{b}.ff.net.2.weight"), vec![DIMS, DIMS]);
            push(format!("{b}.ff.net.2.bias"), vec![DIMS]);
        }
        let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        path
    }

    fn stream(path: &std::path::Path, n: usize) -> SdxlBlockStream {
        SdxlBlockStream::new(
            WeightsSource::File(path.to_path_buf()),
            "attentions.0",
            DIMS,
            HEADS,
            n,
            Dtype::Float32,
            None,
        )
    }

    #[test]
    fn stream_reopen_rejects_retained_inventory_change_before_loading() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("unet.safetensors");
        std::fs::write(&file, b"not tensor data").unwrap();
        let spec = mlx_gen::LoadSpec::new(WeightsSource::Dir(tmp.path().to_owned()));
        let guard = mlx_gen::gen_core::SourceInventory::capture(&spec).unwrap();
        let mut stream = stream(&file, 1);
        stream.bind_source_inventory(guard);
        std::fs::write(tmp.path().join("added.safetensors"), b"new").unwrap();
        let error = stream.open().err().unwrap().to_string();
        assert!(
            error.contains("source inventory changed"),
            "must reject at reopen, not parse tensors: {error}"
        );
    }

    fn resident(stream: &SdxlBlockStream, n: usize) -> Vec<TransformerBlock> {
        let view = stream.open().unwrap();
        (0..n)
            .map(|i| {
                TransformerBlock::from_weights(
                    &view,
                    &format!("attentions.0.transformer_blocks.{i}"),
                    DIMS,
                    HEADS,
                )
                .unwrap()
            })
            .collect()
    }

    /// The drain must remove EXACTLY the block's own tensors — not a whole prefix.
    ///
    /// The distinction is load-bearing. A `remove_prefix` would drain the same bytes here but would
    /// also hide a constructor that silently stopped reading a tensor: the block would be built from
    /// a stale default and the wipe would erase the evidence. Exact draining leaves an un-read key
    /// visible, which is what the second half asserts.
    #[test]
    fn block_stream_drains_exactly_what_the_block_read() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "drain", 2);
        let s = stream(&path, 2);
        let mut view = s.open().unwrap();
        let before = view.len();
        s.materialize(&mut view, 0).unwrap();
        let after = view.len();
        assert!(
            after < before,
            "materializing block 0 must drain its tensors"
        );
        assert!(view
            .get("attentions.0.transformer_blocks.1.attn1.to_q.weight")
            .is_some());
        assert!(view
            .get("attentions.0.transformer_blocks.0.attn1.to_q.weight")
            .is_none());
        assert_eq!(
            view.keys()
                .filter(|k| k.starts_with("attentions.0.transformer_blocks.0."))
                .count(),
            0,
            "every block-0 tensor the constructor read must have been drained"
        );
        assert_eq!(before - after, before / 2);

        // Mutation half: a tensor the constructor never reads is NOT drained, so a constructor that
        // quietly drops a read stays observable instead of being wiped along with the rest.
        let mut view = s.open().unwrap();
        view.insert(
            "attentions.0.transformer_blocks.0.unread_by_the_block",
            Array::from_f32(1.0),
        );
        s.materialize(&mut view, 0).unwrap();
        assert!(
            view.get("attentions.0.transformer_blocks.0.unread_by_the_block")
                .is_some(),
            "an un-read key must survive an exact drain"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A fresh view per window is what keeps the bound: two windows must not share a map that
    /// accumulates. Re-opening returns the full key set every time.
    #[test]
    fn each_window_opens_an_independent_view() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "reopen", 2);
        let s = stream(&path, 2);
        let mut first = s.open().unwrap();
        let full = first.len();
        s.materialize(&mut first, 0).unwrap();
        assert!(first.len() < full);
        let second = s.open().unwrap();
        assert_eq!(second.len(), full, "a new window must see the whole stack");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn out_of_range_blocks_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "range", 2);
        let s = stream(&path, 2);
        let mut view = s.open().unwrap();
        let err = match s.materialize(&mut view, 2) {
            Ok(_) => panic!("block 2 of a 2-block sub-stack must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("out of range"), "{err}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// A materialized block replays the recorded quantization, and matches a RESIDENT block
    /// quantized the same way — the property that matters, since "is quantized" alone would pass on
    /// a different tier or group size.
    #[test]
    fn a_materialized_block_replays_the_recorded_quantization() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "quant", 1);
        let mut s = stream(&path, 1);

        let mut view = s.open().unwrap();
        let mut dense = s.materialize(&mut view, 0).unwrap();
        assert!(!dense
            .adaptable_mut(&["attn1", "to_q"])
            .unwrap()
            .is_quantized());

        s.quant_bits = Some(4);
        let mut view = s.open().unwrap();
        let mut packed = s.materialize(&mut view, 0).unwrap();
        for path_segs in [["attn1", "to_q"], ["attn2", "to_v"], ["ff", "linear3"]] {
            assert!(
                packed.adaptable_mut(&path_segs).unwrap().is_quantized(),
                "{path_segs:?} must be quantized by the replay"
            );
        }

        let mut twin = resident(&s, 1);
        twin[0].quantize(4).unwrap();
        let (rw, rs, rb, rg, rbits) = twin[0]
            .adaptable_mut(&["attn1", "to_q"])
            .unwrap()
            .quantized_params()
            .map(|(w, s, b, _bias, g, bits)| (w.clone(), s.clone(), b.clone(), g, bits))
            .expect("resident is quantized");
        let (pw, ps, pb, pg, pbits) = packed
            .adaptable_mut(&["attn1", "to_q"])
            .unwrap()
            .quantized_params()
            .map(|(w, s, b, _bias, g, bits)| (w.clone(), s.clone(), b.clone(), g, bits))
            .expect("streamed is quantized");
        assert_eq!((rg, rbits), (pg, pbits), "group size / bits must match");
        for (a, b, label) in [
            (&rw, &pw, "codes"),
            (&rs, &ps, "scales"),
            (&rb, &pb, "biases"),
        ] {
            assert!(
                mlx_rs::ops::all_close(a, b, None, None, None)
                    .unwrap()
                    .item::<bool>(),
                "{label} differ between the resident and streamed block"
            );
        }
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// sc-26268 review: a **packed** snapshot whose producer wrote its dense leaves (norms, linear
    /// biases) at f32 — the shape of the published `realvisxl`/`illustrious-xl` q4/q8 U-Nets. On iOS
    /// the resident load casts those leaves to the compute dtype (`loader::cast_for_load`); on the
    /// host it leaves them stored (sc-26270 E2: host bits unchanged). Either way a streamed block
    /// must come back with its resident twin's dtypes and run the same forward, bit for bit.
    #[test]
    fn a_streamed_packed_block_casts_its_dense_leaves_like_the_resident_block() {
        let _cpu = crate::test_device::cpu_only();
        let tmp = tempfile::tempdir().unwrap();
        let dense = fixture(&tmp, "packed-src", 1);
        // Pack every 2-D linear weight (u32 codes + f16 scales/biases, the converter's layout) and
        // keep every dense leaf at f32 — exactly what the published packed snapshots carry.
        let src = Weights::from_file(&dense).unwrap();
        let mut named: Vec<(String, Array)> = Vec::new();
        for key in src.keys().map(str::to_owned).collect::<Vec<_>>() {
            let a = src.require(&key).unwrap().clone();
            if key.ends_with(".weight") && a.ndim() == 2 {
                let (codes, scales, biases) = mlx_rs::ops::quantize(&a, 64, 4).unwrap();
                let base = key.trim_end_matches(".weight");
                named.push((key.clone(), codes));
                named.push((
                    format!("{base}.scales"),
                    scales.as_dtype(Dtype::Float16).unwrap(),
                ));
                named.push((
                    format!("{base}.biases"),
                    biases.as_dtype(Dtype::Float16).unwrap(),
                ));
            } else {
                assert_eq!(
                    a.dtype(),
                    Dtype::Float32,
                    "{key}: the fixture's dense leaves are f32"
                );
                named.push((key, a));
            }
        }
        let dir = tmp.path().join("sdxl-block-stream-packed");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();

        // The production load rule at f16, on the host (packed leaves untouched — sc-26270 E2) and
        // under the iOS CPU fixture (dense leaves cast to the compute dtype).
        let check = |ios: bool| -> mlx_gen::Result<()> {
            // Resident twin.
            let mut w = Weights::from_file(&path).unwrap();
            crate::loader::cast_for_load(&mut w, Dtype::Float16).unwrap();
            let mut resident = TransformerBlock::from_weights(
                &w,
                "attentions.0.transformer_blocks.0",
                DIMS,
                HEADS,
            )
            .unwrap();

            // Streamed block: armed at the same compute dtype.
            let s = SdxlBlockStream::new(
                WeightsSource::File(path.clone()),
                "attentions.0",
                DIMS,
                HEADS,
                1,
                Dtype::Float16,
                None,
            );
            let mut view = s.open().unwrap();
            let mut streamed = s.materialize(&mut view, 0).unwrap();
            assert!(
                streamed
                    .adaptable_mut(&["attn1", "to_q"])
                    .unwrap()
                    .is_quantized(),
                "the fixture must load packed"
            );

            let resident_dtypes = resident.dense_leaf_dtypes();
            assert!(
                resident_dtypes.len() > 6,
                "the probe must cover the linear biases, not just the norms"
            );
            let leaf = if ios { Dtype::Float16 } else { Dtype::Float32 };
            assert!(
                resident_dtypes.iter().all(|(_, d)| *d == leaf),
                "ios={ios}: resident dense leaves must be {leaf:?}: {resident_dtypes:?}"
            );
            assert_eq!(
                streamed.dense_leaf_dtypes(),
                resident_dtypes,
                "a streamed block's dense leaves must carry its resident twin's dtypes"
            );

            let ramp = |shape: &[i32], phase: f32| {
                let n = shape.iter().product::<i32>();
                let v: Vec<f32> = (0..n).map(|i| ((i as f32 + phase) * 0.37).sin()).collect();
                Array::from_slice(&v, shape)
                    .as_dtype(Dtype::Float16)
                    .unwrap()
            };
            let x = ramp(&[1, 16, DIMS], 0.0);
            let memory = ramp(&[1, 7, DIMS], 3.0);
            let plan = crate::plan::SdxlForwardPlan::default();
            let r = resident.forward_ip(&x, &memory, None, plan).unwrap();
            let t = streamed.forward_ip(&x, &memory, None, plan).unwrap();
            // iOS: the f16 leaves keep the window at f16 (on the host the stored leaves decide it,
            // exactly as every release before sc-26268).
            if ios {
                assert_eq!(
                    r.dtype(),
                    Dtype::Float16,
                    "the iOS resident window stays at f16"
                );
            }
            assert_eq!(
                t.dtype(),
                r.dtype(),
                "the streamed window must match its twin"
            );
            assert!(
                mlx_rs::ops::array_eq(&r, &t, false).unwrap().item::<bool>(),
                "a streamed block must run bit-identically to its resident twin"
            );
            Ok(())
        };
        check(false).unwrap();
        mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1 << 34, || check(true)).unwrap();
    }

    /// Captured residual adapters travel onto a materialized block, per block — a LoRA installed on
    /// block 1 must NOT appear on block 0.
    #[test]
    fn captured_adapters_are_reinstalled_per_block() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "adapters", 2);
        let mut s = stream(&path, 2);
        let mut twin = resident(&s, 2);
        let a = Array::from_slice(&vec![0.5f32; (DIMS * 2) as usize], &[DIMS, 2]);
        let b = Array::from_slice(&vec![0.25f32; (2 * DIMS) as usize], &[2, DIMS]);
        twin[1]
            .adaptable_mut(&["attn1", "to_q"])
            .unwrap()
            .push(Adapter::Lora { a, b, scale: 1.0 });
        s.capture_from(&mut twin);
        assert!(s.has_adapters());

        let mut view = s.open().unwrap();
        let mut b0 = s.materialize(&mut view, 0).unwrap();
        let mut view = s.open().unwrap();
        let mut b1 = s.materialize(&mut view, 1).unwrap();
        assert!(b0
            .adaptable_mut(&["attn1", "to_q"])
            .unwrap()
            .adapters()
            .is_empty());
        assert_eq!(
            b1.adaptable_mut(&["attn1", "to_q"])
                .unwrap()
                .adapters()
                .len(),
            1
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// **The silent-IP-drop guard.**
    ///
    /// A streamed block rebuilt from the snapshot has no IP-Adapter K/V pair — the snapshot does not
    /// carry one. If capture/replay is skipped for a block while its neighbours keep theirs, the
    /// render silently drops the image prompt for those layers: a plausible wrong image, no compile
    /// error, and no memory assertion that would notice. This proves the stream refuses instead.
    #[test]
    fn a_streamed_block_missing_its_ip_pair_errors_loudly() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "ip", 2);
        let mut s = stream(&path, 2);
        let mut twin = resident(&s, 2);
        // Install an IP pair on block 0 ONLY, exactly the partial state the guard exists for.
        let k = Array::from_slice(&vec![0.1f32; (DIMS * DIMS) as usize], &[DIMS, DIMS]);
        let v = Array::from_slice(&vec![0.2f32; (DIMS * DIMS) as usize], &[DIMS, DIMS]);
        twin[0].set_ip_projections(
            AdaptableLinear::dense(k, None),
            AdaptableLinear::dense(v, None),
        );
        s.capture_from(&mut twin);
        assert!(
            s.expects_ip(),
            "the stream must record that IP was installed"
        );

        // Block 0 replays its pair.
        let mut view = s.open().unwrap();
        let b0 = s.materialize(&mut view, 0).unwrap();
        assert!(b0.has_ip(), "block 0's captured pair must be replayed");

        // Block 1 has none, and the stream refuses rather than rendering without it.
        let mut view = s.open().unwrap();
        let err = match s.materialize(&mut view, 1) {
            Ok(_) => panic!("a streamed block with no captured IP pair must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("IP-Adapter") && err.contains("silently drop"),
            "the refusal must name the hazard, got: {err}"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// The block-local adapter path list must exactly match what `adaptable_mut` resolves.
    ///
    /// A hand-written enumeration next to a `match` is a classic drift hazard, and here the drift is
    /// SILENT in the direction that matters: a path dropped from the list means rung 4 stops
    /// replaying that adapter, and the streamed render quietly loses part of the LoRA. Both
    /// directions are checked.
    #[test]
    fn block_adapter_paths_all_resolve() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "paths", 1);
        let s = stream(&path, 1);
        let mut twin = resident(&s, 1);
        let listed = twin[0].adaptable_paths();
        assert_eq!(
            listed.len(),
            crate::unet::transformer::BLOCK_ADAPTER_PATHS.len()
        );
        for p in &listed {
            let segs: Vec<&str> = p.split('.').collect();
            assert!(
                twin[0].adaptable_mut(&segs).is_some(),
                "listed adapter path `{p}` does not resolve"
            );
        }
        // And a path that is NOT on the list must not resolve either — otherwise the list is
        // incomplete and a capture would miss a real target.
        for absent in ["attn1.to_out", "ff.net.0.proj", "norm1", "attn3.to_q"] {
            let segs: Vec<&str> = absent.split('.').collect();
            assert!(
                twin[0].adaptable_mut(&segs).is_none(),
                "`{absent}` resolves but is absent from BLOCK_ADAPTER_PATHS"
            );
        }
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// With no IP installed anywhere, materialization is unaffected — the guard must not fire on the
    /// ordinary no-IP path (the mutation-discriminating other half of the test above).
    #[test]
    fn no_ip_installed_means_no_ip_guard() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "noip", 2);
        let mut s = stream(&path, 2);
        let mut twin = resident(&s, 2);
        s.capture_from(&mut twin);
        assert!(!s.expects_ip());
        let mut view = s.open().unwrap();
        for i in 0..2 {
            let block = s.materialize(&mut view, i).unwrap();
            assert!(!block.has_ip());
        }
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
