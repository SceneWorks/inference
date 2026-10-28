//! Ladder rung 4 — **bounded transformer residency** for the MLX Z-Image DiT (SC-15754).
//!
//! The window lifecycle itself is NOT implemented here: it is
//! [`mlx_gen::block_residency::run_windowed`] (SC-15750), the shared primitive that also serves
//! Mage-Flow / Qwen / FLUX. This module is only the family-side half — "how do I rebuild Z-Image
//! block *n* from the snapshot, with the same quantization and the same adapters the resident block
//! carries" — which is the part that genuinely differs per family.
//!
//! ## Why re-reading per window is nearly free
//!
//! [`Array::load_safetensors`](mlx_rs::Array::load_safetensors_with_metadata) is **lazy per tensor**:
//! SC-15744 measured 1073 handles at 0.0 MiB until something is evaluated. So opening the snapshot
//! once per window costs a JSON header parse, not 3.4 GB. Only the tensors a window's blocks actually
//! read are materialized, and [`Weights::remove_accessed`] then drops the view's own reference to
//! exactly those, so the window's residency is the blocks themselves and nothing else.
//!
//! `remove_accessed` is deliberate rather than decorative. SC-15750's primitive documents that `apply`
//! must take tensors OUT of the view instead of cloning them, because `Array` is refcounted and a
//! clone left in the map keeps the materialized buffer alive. The Z-Image block constructors take
//! `&Weights` and clone (they are the same constructors the resident path uses, and forking them would
//! be a second way to build a block — precisely the divergence hazard this epic exists to remove), so
//! the drain runs immediately after each block instead: `from_weights` records every key it read
//! through `require`/`get`, and `remove_accessed` removes exactly that set. A constructor read that
//! never happened therefore cannot be drained, which is what makes
//! [`block_stream_drains_exactly_what_the_block_read`] a real check rather than a tautology.
//!
//! ## The deferred-materialization precondition
//!
//! Streaming bounds residency only if the full `layers` stack remains lazy. The load-time
//! [`mlx_gen::LoadShape::DeferredMaterialization`] flag selects that shape independently from
//! [`mlx_gen::OffloadPolicy`]: Resident keeps the generator and non-windowed state warm, while
//! Sequential also releases completed phases. A rung-4 request on an eager load is rejected before
//! execution, preventing a false-green calibration over already-materialized blocks.

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};

/// The adapters installed on ONE block, addressed by the block-local dotted path
/// (`"attention.to_q"`, `"feed_forward.w1"`, `"adaLN_modulation.0"`, …).
///
/// Captured from the resident blocks after [`crate::adapters::apply_z_image_adapters`] has run, and
/// re-installed verbatim on each materialized block. Z-Image installs every adapter as a *forward-time
/// residual* (`apply_adapters_strict` only ever `push`es an [`Adapter`]; it never folds a delta into a
/// base the way the diff-patch path does), so replaying the captured list reproduces the resident
/// block exactly rather than approximating it. [`Adapter`] holds refcounted `Array` factors, so the
/// capture is a handle copy — a rank-16 LoRA's factors, not a `[out, in]` delta.
#[derive(Clone, Default)]
pub(crate) struct BlockAdapters {
    per_path: Vec<(String, Vec<Adapter>)>,
}

impl BlockAdapters {
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.per_path.is_empty()
    }

    /// Re-install the captured adapters on a freshly materialized block.
    fn install(&self, block: &mut ZImageTransformerBlock) -> Result<()> {
        for (path, adapters) in &self.per_path {
            let segs: Vec<&str> = path.split('.').collect();
            let target = block.adaptable_mut(&segs).ok_or_else(|| {
                Error::Msg(format!(
                    "z-image block stream: adapter target `{path}` is absent from a materialized \
                     block — the streamed block tree does not match the resident one"
                ))
            })?;
            target.set_adapters(adapters.clone());
        }
        Ok(())
    }
}

/// Everything needed to rebuild one Z-Image transformer block stack from its snapshot, on demand.
///
/// Held by [`ZImageTransformer`](crate::ZImageTransformer) (and by the ControlNet variant for its own
/// `control_layers`), and consumed by the windowed forward. Cheap to clone: a path, a few scalars, and
/// refcounted adapter factors.
#[derive(Clone)]
pub(crate) struct ZImageBlockStream {
    source: WeightsSource,
    /// Key prefix for the stack (`"layers"`, `"control_layers"`), already joined with any checkpoint
    /// prefix, so block `n` is `{base}.{n}`.
    base: String,
    cfg: ZImageBlockConfig,
    n_blocks: usize,
    /// The load-time quantization the resident stack received, replayed per materialized block so the
    /// streamed weights are byte-identical to the resident ones. `None` for a dense/bf16 tier. On a
    /// pre-quantized (packed) snapshot this is recorded but is a documented no-op — the same no-op the
    /// resident path performed.
    quant_bits: Option<i32>,
    /// Per-block captured adapters, indexed by block. Empty when no adapter is installed.
    adapters: Vec<BlockAdapters>,
    /// Whether the ControlNet block variant is being streamed (it carries the extra `before_proj` /
    /// `after_proj` projections). Base stacks are `false`.
    control: bool,
}

impl ZImageBlockStream {
    /// Declare a streamable stack. `source` must be re-openable — a snapshot directory or a single
    /// `.safetensors` file. An in-memory / ComfyUI-normalized `Weights` has no re-openable source, so
    /// those loads simply do not construct one and rung 4 is declared unavailable for them.
    pub(crate) fn new(
        source: WeightsSource,
        base: impl Into<String>,
        cfg: ZImageBlockConfig,
        n_blocks: usize,
        control: bool,
    ) -> Self {
        Self {
            source,
            base: base.into(),
            cfg,
            n_blocks,
            quant_bits: None,
            adapters: vec![BlockAdapters::default(); n_blocks],
            control,
        }
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// The block shape this stream materializes at — so a test can build the resident twin from the
    /// same derivation rather than a second spelling of it.
    #[cfg(test)]
    pub(crate) fn cfg_for_test(&self) -> &ZImageBlockConfig {
        &self.cfg
    }

    #[cfg(test)]
    pub(crate) fn has_adapters(&self) -> bool {
        self.adapters.iter().any(|a| !a.is_empty())
    }

    /// Record the load-time quantization so a materialized block reproduces it.
    pub(crate) fn set_quant_bits(&mut self, bits: i32) {
        self.quant_bits = Some(bits);
    }

    /// Capture the adapters currently installed on `blocks` so materialized blocks carry them too.
    ///
    /// Called after the resident stack has been adapted, so the two paths cannot diverge on *which*
    /// adapters, *which* targets, or *what order* — the streamed block replays exactly what the
    /// resident block ended up holding.
    pub(crate) fn capture_adapters(&mut self, blocks: &mut [ZImageTransformerBlock]) {
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
    }

    /// Open a fresh lazy view of the stack's weights. Called once per window by
    /// [`run_windowed`](mlx_gen::block_residency::run_windowed).
    pub(crate) fn open(&self) -> Result<Weights> {
        match &self.source {
            WeightsSource::Dir(dir) => Weights::from_dir(dir),
            WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    fn check(&self, index: usize, control: bool) -> Result<String> {
        if index >= self.n_blocks {
            return Err(Error::Msg(format!(
                "z-image block stream: block {index} is out of range for a {}-block stack",
                self.n_blocks
            )));
        }
        if control != self.control {
            return Err(Error::Msg(format!(
                "z-image block stream `{}` is a {} stack; it cannot materialize a {} block",
                self.base,
                if self.control { "control" } else { "base" },
                if control { "control" } else { "base" },
            )));
        }
        Ok(format!("{}.{index}", self.base))
    }

    /// Materialize base block `index` out of `view`, quantized and adapted exactly like its resident
    /// twin, then drain the view of precisely the tensors that block read.
    #[cfg(test)]
    pub(crate) fn materialize(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<ZImageTransformerBlock> {
        Ok(self.materialize_counted(view, index)?.0)
    }

    /// [`Self::materialize`], also returning what the block's iOS quantum adds for having been
    /// materialized here: the stored bytes of exactly the tensors it read (dtype x shape off the
    /// lazy handles, never read here — the read happens inside the quantum) plus, when the stack
    /// replays a load-time quantization over dense weights, the packed copies it builds.
    pub(crate) fn materialize_counted(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<(ZImageTransformerBlock, u64)> {
        let prefix = self.check(index, false)?;
        let (mut block, extra) = base_block_counted(view, &prefix, self.cfg, self.quant_bits)?;
        if let Some(bits) = self.quant_bits {
            block.quantize(bits)?;
        }
        if let Some(adapters) = self.adapters.get(index) {
            adapters.install(&mut block)?;
        }
        Ok((block, extra))
    }

    /// Materialize ControlNet block `index` — the base submodules plus the `after_proj` hint
    /// projection and, on block 0, the `before_proj` seed.
    ///
    /// No adapter replay: [`AdaptableHost for ZImageControlTransformer`](crate::ZImageControlTransformer)
    /// routes every adapter path to the composed **base** DiT, because the fork trains LoRA/LoKr on the
    /// base transformer and never on the ControlNet branch. So a control block has no adapters to
    /// carry, and `capture_adapters` is never called on a control stream.
    pub(crate) fn materialize_control(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<(crate::control_transformer_block::ZImageControlBlock, u64)> {
        let prefix = self.check(index, true)?;
        let (mut block, extra) =
            control_block_counted(view, &prefix, self.cfg, index == 0, self.quant_bits)?;
        if let Some(bits) = self.quant_bits {
            block.quantize(bits)?;
        }
        Ok((block, extra))
    }
}

/// Build base block `prefix` from `view` (lazy handles: nothing is read) and drain the view of
/// exactly the tensors it read, returning what the block's iOS quantum adds for having been
/// materialized there: their stored bytes plus, replaying `quant_bits`, the packed copies the
/// quantization builds. Does not quantize. Shared with the header-only iOS pricing
/// (`crate::ios_pricing`).
pub(crate) fn base_block_counted(
    view: &mut Weights,
    prefix: &str,
    cfg: ZImageBlockConfig,
    quant_bits: Option<i32>,
) -> Result<(ZImageTransformerBlock, u64)> {
    let block = ZImageTransformerBlock::from_weights(view, prefix, cfg)?;
    let mut extra = stored_bytes(view);
    // LOAD-BEARING (SC-15750): the view keeps its own refcounted handle to every tensor the
    // constructor cloned. Draining exactly the accessed keys is what makes the window's drop a
    // real release rather than a no-op that still produces correct images.
    view.remove_accessed();
    if let Some(bits) = quant_bits {
        extra = extra.saturating_add(
            block
                .projections()
                .iter()
                .map(|lin| crate::quantum::quantize_bytes(lin, bits))
                .sum::<u64>(),
        );
    }
    Ok((block, extra))
}

/// [`base_block_counted`] for a ControlNet block (block 0 carries `before_proj`): the stored bytes
/// the quantum reads, plus the packed copies of a load-time quantization replayed over the
/// (dense-only published) ControlNet weights. Does not quantize.
pub(crate) fn control_block_counted(
    view: &mut Weights,
    prefix: &str,
    cfg: ZImageBlockConfig,
    has_before_proj: bool,
    quant_bits: Option<i32>,
) -> Result<(crate::control_transformer_block::ZImageControlBlock, u64)> {
    let block = crate::control_transformer_block::ZImageControlBlock::from_weights(
        view,
        prefix,
        cfg,
        has_before_proj,
    )?;
    let mut extra = stored_bytes(view);
    view.remove_accessed();
    if let Some(bits) = quant_bits {
        extra = extra.saturating_add(block.quantize_bytes(bits));
    }
    Ok((block, extra))
}

/// Logical capacity of the source keys a constructor read from `view`, matching the resident
/// inventory and declared packed outputs. Hint/output page slack remains a separate scratch term.
/// This source-key sum is not native alias deduplication, a backing charge, or release evidence.
fn stored_bytes(view: &Weights) -> u64 {
    view.accessed_entries()
        .iter()
        .fold(0u64, |total, (_, array)| {
            total.saturating_add(mlx_gen::ios_quantum::allocated_bytes(array.nbytes() as u64))
        })
}

/// The plan + cancellation a windowed forward runs under. A thin pair so the forward signatures do not
/// grow two more positional arguments, and so `Copy` keeps the per-step closures cheap.
#[derive(Clone, Copy)]
pub(crate) struct BlockWindow<'a> {
    pub(crate) plan: mlx_gen::block_residency::BlockPlan,
    pub(crate) cancel: &'a mlx_gen::CancelFlag,
}

#[cfg(all(test, target_os = "macos"))]
mod capacity_proofs {
    use super::*;
    use mlx_gen::attention::AttentionPlan;
    use mlx_rs::Array;
    use std::collections::{BTreeMap, BTreeSet};
    use std::io::Write;

    const CHILD: &str = "ZIMAGE_BLOCK_SOURCE_CAPACITY_CPU_CHILD";
    const DONE: &str = "ZIMAGE_BLOCK_SOURCE_CAPACITY_CPU_DONE";
    const FQN: &str = "block_stream::capacity_proofs::block_file_epoch_source_capacity_matches_full_logical_quanta";

    struct Completion;
    impl Drop for Completion {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                assert_eq!(
                    unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
                    0,
                    "block source-capacity CPU synchronization after fixture owners dropped"
                );
                std::fs::write(std::env::var(DONE).expect("fresh completion path"), b"ok")
                    .expect("write block source-capacity completion");
            }
        }
    }

    #[derive(Clone, Copy)]
    enum Role {
        Base,
        ControlFirst,
        ControlLater,
    }
    impl Role {
        fn prefix(self) -> &'static str {
            match self {
                Self::Base => "layers.0",
                Self::ControlFirst => "control_layers.0",
                Self::ControlLater => "control_layers.1",
            }
        }
        fn before(self) -> bool {
            matches!(self, Self::ControlFirst)
        }
        fn label(self) -> &'static str {
            match self {
                Self::Base => crate::quantum::DIT_BLOCK,
                _ => crate::quantum::CONTROL_BLOCK,
            }
        }
    }

    enum Block {
        Base(Box<ZImageTransformerBlock>),
        Control(Box<crate::control_transformer_block::ZImageControlBlock>),
    }
    impl Block {
        fn counted(
            view: &mut Weights,
            role: Role,
            cfg: ZImageBlockConfig,
            bits: Option<i32>,
        ) -> Result<(Self, u64)> {
            match role {
                Role::Base => base_block_counted(view, role.prefix(), cfg, bits)
                    .map(|(block, read)| (Self::Base(Box::new(block)), read)),
                _ => control_block_counted(view, role.prefix(), cfg, role.before(), bits)
                    .map(|(block, read)| (Self::Control(Box::new(block)), read)),
            }
        }
        fn from_headers(view: &Weights, role: Role, cfg: ZImageBlockConfig) -> Result<Self> {
            match role {
                Role::Base => ZImageTransformerBlock::from_weights(view, role.prefix(), cfg)
                    .map(|block| Self::Base(Box::new(block))),
                _ => crate::control_transformer_block::ZImageControlBlock::from_weights(
                    view,
                    role.prefix(),
                    cfg,
                    role.before(),
                )
                .map(|block| Self::Control(Box::new(block))),
            }
        }
        fn scratch(&self, x: &Array, freqs: &Array) -> u64 {
            let plan = AttentionPlan::UNBOUNDED;
            let backend = mlx_gen::ios_quantum::Backend::Metal;
            match self {
                // Include the actual hinted-base caller's allowance ONCE. Its output slack is
                // unchanged and cannot be repurposed as a source-key capacity correction.
                Self::Base(block) => block
                    .quantum_scratch(x, freqs, plan, backend)
                    .saturating_add(crate::control_transformer::add_hint_bytes(x)),
                // This already includes the real 1/3-output Control page_slack term.
                Self::Control(block) => block.quantum_scratch(x, x, freqs, plan, backend),
            }
        }
    }

    struct Tensor {
        shape: Vec<usize>,
        dtype: &'static str,
        width: u64,
        selected: bool,
    }
    type Headers = BTreeMap<String, Tensor>;

    fn tensor(
        headers: &mut Headers,
        key: String,
        shape: Vec<usize>,
        dtype: &'static str,
        width: u64,
        selected: bool,
    ) {
        assert!(headers
            .insert(
                key,
                Tensor {
                    shape,
                    dtype,
                    width,
                    selected
                }
            )
            .is_none());
    }

    /// A small, valid constructor geometry, not production catalog dimensions. Each file contains
    /// ONE local block, including ignored keys; no N-block payload or unaccessed shell is evaluated.
    fn headers(role: Role, cfg: ZImageBlockConfig, stored: Option<i32>) -> (Headers, Vec<String>) {
        let d = cfg.dim as usize;
        let prefix = role.prefix();
        let mut out = Headers::new();
        let mut projections = Vec::new();
        let mut shapes = vec![
            ("attention.to_q", d, d, false),
            ("attention.to_k", d, d, false),
            ("attention.to_v", d, d, false),
            ("attention.to_out.0", d, d, false),
            ("feed_forward.w1", d, d, false),
            ("feed_forward.w2", d, d, false),
            ("feed_forward.w3", d, d, false),
            ("adaLN_modulation.0", 4 * d, d, true),
        ];
        if !matches!(role, Role::Base) {
            shapes.push(("after_proj", d, d, true));
        }
        if role.before() {
            shapes.push(("before_proj", d, d, true));
        }
        for (path, rows, cols, bias) in shapes {
            let base = format!("{prefix}.{path}");
            projections.push(base.clone());
            if let Some(bits) = stored {
                tensor(
                    &mut out,
                    format!("{base}.weight"),
                    vec![rows, cols * bits as usize / 32],
                    "U32",
                    4,
                    true,
                );
                for suffix in ["scales", "biases"] {
                    tensor(
                        &mut out,
                        format!("{base}.{suffix}"),
                        vec![rows, cols / crate::quant::GROUP_SIZE as usize],
                        "BF16",
                        2,
                        true,
                    );
                }
            } else {
                tensor(
                    &mut out,
                    format!("{base}.weight"),
                    vec![rows, cols],
                    "BF16",
                    2,
                    true,
                );
            }
            if bias {
                tensor(
                    &mut out,
                    format!("{base}.bias"),
                    vec![rows],
                    "BF16",
                    2,
                    true,
                );
            }
        }
        for (path, width) in [
            ("attention.norm_q.weight", d / cfg.n_heads as usize),
            ("attention.norm_k.weight", d / cfg.n_heads as usize),
            ("attention_norm1.weight", d),
            ("attention_norm2.weight", d),
            ("ffn_norm1.weight", d),
            ("ffn_norm2.weight", d),
        ] {
            tensor(
                &mut out,
                format!("{prefix}.{path}"),
                vec![width],
                "BF16",
                2,
                true,
            );
        }
        for key in ["attention.to_q.bias", "feed_forward.w1.bias", "unused"] {
            tensor(
                &mut out,
                format!("{prefix}.{key}"),
                vec![d],
                "BF16",
                2,
                false,
            );
        }
        tensor(
            &mut out,
            "unrelated.weight".into(),
            vec![1],
            "BF16",
            2,
            false,
        );
        if matches!(role, Role::ControlLater) {
            // Presence in the dictionary does not turn a later block into the block-0 constructor.
            tensor(
                &mut out,
                format!("{prefix}.before_proj.weight"),
                vec![d, d],
                "BF16",
                2,
                false,
            );
            tensor(
                &mut out,
                format!("{prefix}.before_proj.bias"),
                vec![d],
                "BF16",
                2,
                false,
            );
        }
        (out, projections)
    }

    fn sparse_file(root: &std::path::Path, name: &str, headers: &Headers) -> std::path::PathBuf {
        let mut json = serde_json::Map::new();
        let mut offset = 0u64;
        for (key, t) in headers {
            let bytes = t
                .shape
                .iter()
                .try_fold(t.width, |n, &dim| n.checked_mul(dim as u64))
                .unwrap();
            json.insert(
                key.clone(),
                serde_json::json!({"dtype": t.dtype, "shape": t.shape,
                "data_offsets": [offset, offset + bytes]}),
            );
            offset += bytes;
        }
        let mut bytes = serde_json::to_vec(&json).unwrap();
        while !bytes.len().is_multiple_of(8) {
            bytes.push(b' ');
        }
        let path = root.join(format!("{name}.safetensors"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(8 + bytes.len() as u64 + offset).unwrap();
        path
    }

    fn capacity(bytes: u64) -> u64 {
        let page = mlx_gen::ios_quantum::PAGE_BYTES;
        if bytes > page {
            bytes.div_ceil(page) * page
        } else {
            bytes
        }
    }

    /// Independent DECLARED future pack capacity, never an observed allocation. BF16 sources and
    /// biases need no dtype casts. Packed sources are the private quantizer's no-op at any replay
    /// argument; this is NOT acceptance of a production stored/requested tier mismatch.
    fn declared_pack(
        parsed: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
        projections: &[String],
        stored: Option<i32>,
        replay: Option<i32>,
    ) -> u64 {
        let Some(bits) = replay.filter(|_| stored.is_none()) else {
            return 0;
        };
        projections
            .iter()
            .map(|base| {
                let weight = parsed
                    .iter()
                    .find(|h| h.name == format!("{base}.weight"))
                    .unwrap();
                let (rows, cols) = (weight.shape[0] as u64, weight.shape[1] as u64);
                capacity(rows * cols * bits as u64 / 8)
                    + 2 * capacity(rows * (cols / crate::quant::GROUP_SIZE as u64) * 2)
            })
            .sum()
    }

    #[test]
    fn block_file_epoch_source_capacity_matches_full_logical_quanta() {
        let mut completion: Option<Completion> = None; // FIRST None; armed at the true final tail.
        if std::env::var_os(CHILD).is_none() {
            let dir = tempfile::tempdir().unwrap();
            let marker = dir.path().join("done");
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", FQN, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .env(DONE, &marker)
                .status()
                .unwrap();
            assert!(status.success(), "block source-capacity CPU child failed");
            assert_eq!(
                std::fs::read(marker)
                    .expect("fresh normal-completion marker")
                    .as_slice(),
                b"ok"
            );
            return;
        }
        // BEFORE any source file, Array, or block constructor. No inherited device/stream trusted.
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
        let formats = [None, Some(4), Some(8)];
        let roles = [Role::Base, Role::ControlFirst, Role::ControlLater];
        let replays = [None, Some(4), Some(8)];
        let cases = formats.len() * roles.len() * replays.len();
        let expected_mask = (1u64 << cases) - 1;
        let mut completed = 0u64;
        {
            let dir = tempfile::tempdir().unwrap();
            let cfg = ZImageBlockConfig {
                dim: 192,
                n_heads: 3,
                norm_eps: 1e-5,
            };
            let mut inputs = Headers::new();
            tensor(
                &mut inputs,
                "x".into(),
                vec![1, 32, cfg.dim as usize],
                "F32",
                4,
                true,
            );
            tensor(
                &mut inputs,
                "freqs".into(),
                vec![32, (cfg.dim / cfg.n_heads) as usize],
                "F32",
                4,
                true,
            );
            let inputs =
                Weights::from_file(sparse_file(dir.path(), "scratch-input-metadata", &inputs))
                    .unwrap();
            let (x, freqs) = (
                inputs.require("x").unwrap(),
                inputs.require("freqs").unwrap(),
            );
            for (format_index, stored) in formats.into_iter().enumerate() {
                for (role_index, role) in roles.into_iter().enumerate() {
                    let (source_headers, projections) = headers(role, cfg, stored);
                    let path = sparse_file(
                        dir.path(),
                        &format!("source-{format_index}-{role_index}"),
                        &source_headers,
                    );
                    let parsed =
                        mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&path)
                            .unwrap();
                    let selected = source_headers
                        .iter()
                        .filter(|(_, t)| t.selected)
                        .map(|(k, _)| k.clone())
                        .collect::<BTreeSet<_>>();
                    let untouched = source_headers
                        .iter()
                        .filter(|(_, t)| !t.selected)
                        .map(|(k, _)| k.clone())
                        .collect::<BTreeSet<_>>();
                    let selected_headers = parsed
                        .iter()
                        .filter(|h| selected.contains(&h.name))
                        .collect::<Vec<_>>();
                    assert_eq!(
                        selected_headers
                            .iter()
                            .map(|h| h.name.clone())
                            .collect::<BTreeSet<_>>(),
                        selected
                    );
                    let source_raw = selected_headers.iter().map(|h| h.data_bytes).sum::<u64>();
                    let source_capacity = selected_headers
                        .iter()
                        .map(|h| capacity(h.data_bytes))
                        .sum::<u64>();
                    assert!(
                        source_capacity > source_raw,
                        "derived file geometry must discriminate source capacity"
                    );
                    for (replay_index, replay) in replays.into_iter().enumerate() {
                        let index = (format_index * roles.len() + role_index) * replays.len()
                            + replay_index;
                        let mut view = Weights::from_file(&path).unwrap(); // FRESH file-factory epoch
                        let key = format!("{}.attention_norm1.weight", role.prefix());
                        view.require(&key).unwrap();
                        view.require(&key).unwrap(); // repeated SAME KEY, not arbitrary ArrayDesc aliases
                        let (module, read) = Block::counted(&mut view, role, cfg, replay).unwrap();
                        assert_eq!(
                            view.keys().map(str::to_owned).collect::<BTreeSet<_>>(),
                            untouched,
                            "BLOCK-CAPACITY-EXACT-DRAIN: preserve every unaccessed source key"
                        );
                        assert!(view.accessed_entries().is_empty());
                        let packing = declared_pack(&parsed, &projections, stored, replay);
                        let scratch = if stored.is_none() && replay.is_some() {
                            // A future EFFECTIVE metadata proxy for the actual forward's dtype/shape,
                            // not a second loaded production module or evaluated quantization graph.
                            let (effective, _) = headers(role, cfg, replay);
                            let path =
                                sparse_file(dir.path(), &format!("effective-{index}"), &effective);
                            let effective = Weights::from_file(path).unwrap();
                            Block::from_headers(&effective, role, cfg)
                                .unwrap()
                                .scratch(x, freqs)
                        } else {
                            module.scratch(x, freqs)
                        };
                        let total = scratch + read;
                        assert_eq!(total, scratch + source_capacity + packing,
                            "BLOCK-CAPACITY-FULL-QUANTUM: SAME block scratch including hint/output slack plus independent source-key capacity and declared packing");
                        // Fresh exact child never opens an admitted/calibrated envelope. Check the
                        // existing full requirement/reserve, not native counters or component delta.
                        let old_total = scratch + source_raw + packing;
                        let reserve = mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES;
                        mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(old_total + reserve, || {
                            assert!(mlx_gen::ios_quantum::take_refusal().is_none());
                            mlx_gen::ios_quantum::before(role.label(), old_total)?;
                            assert!(mlx_gen::ios_quantum::before(role.label(), total).is_err(),
                                "BLOCK-CAPACITY-FULL-BUDGET: corrected logical total refuses before construction");
                            let refusal = mlx_gen::ios_quantum::take_refusal().expect("actual logical guard refusal");
                            assert_eq!((refusal.scratch_bytes, refusal.reserve_bytes), (total, reserve));
                            Ok(())
                        }).unwrap();
                        completed |= 1u64 << index;
                        eprintln!("ZIMAGE_BLOCK_SOURCE_CAPACITY_MECHANISM_V1 index={index} role={} stored={stored:?} replay={replay:?}", role.prefix());
                    }
                }
            }
        } // ALL model/source/input/effective-metadata owners and file scopes dropped before arm.
        assert_eq!(
            completed, expected_mask,
            "all declared block source-capacity cases completed"
        );
        eprintln!("ZIMAGE_BLOCK_SOURCE_CAPACITY_BODY_COMPLETED_V1 cases={cases} fullmask={expected_mask} payload_eval=0");
        let _ = completion.replace(Completion);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    fn cfg() -> ZImageBlockConfig {
        ZImageBlockConfig {
            dim: 8,
            n_heads: 2,
            norm_eps: 1e-5,
        }
    }

    /// A two-block synthetic stack written to a temp `.safetensors`, so the stream can be exercised
    /// without a real snapshot.
    fn fixture(tmp: &tempfile::TempDir, tag: &str, n_blocks: usize) -> std::path::PathBuf {
        fixture_at(tmp, tag, n_blocks, cfg().dim)
    }

    /// [`fixture`] at an explicit width, so the quantization test can use a group-64-divisible one.
    fn fixture_at(
        tmp: &tempfile::TempDir,
        tag: &str,
        n_blocks: usize,
        dim: i32,
    ) -> std::path::PathBuf {
        let dir = tmp.path().join(format!("zimage-block-stream-{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let mut named: Vec<(String, Array)> = Vec::new();
        for i in 0..n_blocks {
            for (leaf, shape) in [
                ("attention.to_q.weight", vec![dim, dim]),
                ("attention.to_k.weight", vec![dim, dim]),
                ("attention.to_v.weight", vec![dim, dim]),
                ("attention.to_out.0.weight", vec![dim, dim]),
                ("attention.norm_q.weight", vec![dim / 2]),
                ("attention.norm_k.weight", vec![dim / 2]),
                ("feed_forward.w1.weight", vec![dim, dim]),
                ("feed_forward.w2.weight", vec![dim, dim]),
                ("feed_forward.w3.weight", vec![dim, dim]),
                ("attention_norm1.weight", vec![dim]),
                ("attention_norm2.weight", vec![dim]),
                ("ffn_norm1.weight", vec![dim]),
                ("ffn_norm2.weight", vec![dim]),
                ("adaLN_modulation.0.weight", vec![4 * dim, dim]),
            ] {
                let n: i32 = shape.iter().product();
                let values: Vec<f32> = (0..n).map(|k| (k as f32 + i as f32) * 0.01).collect();
                named.push((
                    format!("layers.{i}.{leaf}"),
                    Array::from_slice(&values, &shape),
                ));
            }
        }
        let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, &path).unwrap();
        path
    }

    fn stream(path: &std::path::Path, n: usize) -> ZImageBlockStream {
        ZImageBlockStream::new(
            WeightsSource::File(path.to_path_buf()),
            "layers",
            cfg(),
            n,
            false,
        )
    }

    /// The drain must remove EXACTLY the block's own tensors — not a whole prefix.
    ///
    /// The distinction is load-bearing. `remove_prefix("layers.0.")` would drain the same bytes here
    /// but would also hide a constructor that silently stopped reading a tensor: the block would be
    /// built from a stale default and the wipe would erase the evidence. Exact draining leaves an
    /// un-read key visible, which is what the second half asserts — and it is not hypothetical, it is
    /// how the first draft of this fixture was caught spelling `q_norm` where the block reads
    /// `norm_q`.
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
        // Block 1's tensors are untouched, and block 0's are gone.
        assert!(view.get("layers.1.attention.to_q.weight").is_some());
        assert!(view.get("layers.0.attention.to_q.weight").is_none());
        // Nothing of block 0 survived: the fixture holds the block's whole read set and no more.
        assert_eq!(
            view.keys().filter(|k| k.starts_with("layers.0.")).count(),
            0,
            "every block-0 tensor the constructor read must have been drained"
        );
        assert_eq!(before - after, before / 2);

        // Mutation half: a tensor the constructor never reads is NOT drained, so a constructor that
        // quietly drops a read stays observable instead of being wiped along with the rest.
        let mut view = s.open().unwrap();
        view.insert("layers.0.unread_by_the_block", Array::from_f32(1.0));
        s.materialize(&mut view, 0).unwrap();
        assert!(
            view.get("layers.0.unread_by_the_block").is_some(),
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

    /// **The quantization replay, which had no coverage anywhere.**
    ///
    /// `set_quant_bits` → `block.quantize(bits)` is what makes a streamed block byte-identical to its
    /// resident twin on a **dense snapshot loaded with `spec.quantize`** — a real configuration
    /// (`mlx_gen::quant::needs_load_time_quant` returning `true` is exactly it). Nothing reached it:
    /// the DiT tests never quantize, and the real-weights suite runs against a *pre-quantized tier
    /// dir* where `quantize()` is a documented no-op. So the replay was correct-by-construction and
    /// unverified, while a doc comment claimed otherwise.
    ///
    /// It is tested here rather than at the DiT level because the whole-transformer `quantize()` also
    /// touches the timestep embedder, whose tiny fixture in-features are not divisible by the group
    /// size — the mechanism under test is per-block, and this is where it lives.
    #[test]
    fn a_materialized_block_replays_the_recorded_quantization() {
        let tmp = tempfile::tempdir().unwrap();
        // Group-64 packing needs in-features divisible by 64, so this fixture is 64-wide.
        let path = fixture_at(&tmp, "quant", 1, 64);
        let mut s = ZImageBlockStream::new(
            WeightsSource::File(path.clone()),
            "layers",
            ZImageBlockConfig {
                dim: 64,
                n_heads: 2,
                norm_eps: 1e-5,
            },
            1,
            false,
        );

        // No bits recorded → the materialized block stays dense, like its unquantized twin.
        let mut view = s.open().unwrap();
        let mut dense = s.materialize(&mut view, 0).unwrap();
        assert!(!dense
            .adaptable_mut(&["attention", "to_q"])
            .unwrap()
            .is_quantized());

        // Bits recorded (what `ZImageTransformer::quantize` does) → the block comes back packed.
        s.set_quant_bits(4);
        let mut view = s.open().unwrap();
        let mut packed = s.materialize(&mut view, 0).unwrap();
        for path in ["attention.to_q", "feed_forward.w1", "adaLN_modulation.0"] {
            let segs: Vec<&str> = path.split('.').collect();
            assert!(
                packed.adaptable_mut(&segs).unwrap().is_quantized(),
                "{path} must be quantized by the replay"
            );
        }

        // And it matches a RESIDENT block quantized the same way — the property that matters, since
        // "is quantized" alone would pass on a different tier or group size.
        let view = s.open().unwrap();
        let mut resident =
            ZImageTransformerBlock::from_weights(&view, "layers.0", *s.cfg_for_test()).unwrap();
        resident.quantize(4).unwrap();
        let (rw, rs, rb, _, rg, rbits) = resident
            .adaptable_mut(&["attention", "to_q"])
            .unwrap()
            .quantized_params()
            .map(|(w, s, b, bias, g, bits)| {
                (w.clone(), s.clone(), b.clone(), bias.cloned(), g, bits)
            })
            .expect("resident is quantized");
        let (pw, ps, pb, _, pg, pbits) = packed
            .adaptable_mut(&["attention", "to_q"])
            .unwrap()
            .quantized_params()
            .map(|(w, s, b, bias, g, bits)| {
                (w.clone(), s.clone(), b.clone(), bias.cloned(), g, bits)
            })
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

    #[test]
    fn out_of_range_blocks_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "range", 2);
        let s = stream(&path, 2);
        let mut view = s.open().unwrap();
        let err = match s.materialize(&mut view, 2) {
            Ok(_) => panic!("block 2 of a 2-block stack must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("out of range"), "{err}");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// Captured adapters travel onto a materialized block. The capture is by block-local path, so a
    /// LoRA installed on block 1 must NOT appear on block 0.
    #[test]
    fn captured_adapters_are_reinstalled_per_block() {
        let tmp = tempfile::tempdir().unwrap();
        let path = fixture(&tmp, "adapters", 2);
        let mut s = stream(&path, 2);

        // Build the two resident blocks and install a LoRA on block 1 only.
        let view = s.open().unwrap();
        let mut resident: Vec<ZImageTransformerBlock> = (0..2)
            .map(|i| {
                ZImageTransformerBlock::from_weights(&view, &format!("layers.{i}"), cfg()).unwrap()
            })
            .collect();
        let a = Array::from_slice(&vec![0.5f32; (cfg().dim * 2) as usize], &[cfg().dim, 2]);
        let b = Array::from_slice(&vec![0.25f32; (2 * cfg().dim) as usize], &[2, cfg().dim]);
        resident[1]
            .adaptable_mut(&["attention", "to_q"])
            .unwrap()
            .push(Adapter::Lora {
                a: a.clone(),
                b: b.clone(),
                scale: 1.0,
            });
        s.capture_adapters(&mut resident);
        assert!(s.has_adapters());

        let mut view = s.open().unwrap();
        let b0 = s.materialize(&mut view, 0).unwrap();
        let mut view = s.open().unwrap();
        let mut b1 = s.materialize(&mut view, 1).unwrap();
        let mut b0 = b0;
        assert!(b0
            .adaptable_mut(&["attention", "to_q"])
            .unwrap()
            .adapters()
            .is_empty());
        assert_eq!(
            b1.adaptable_mut(&["attention", "to_q"])
                .unwrap()
                .adapters()
                .len(),
            1
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
