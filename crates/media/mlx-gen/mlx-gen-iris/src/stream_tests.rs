//! Rung 4 (bounded transformer residency) and rung 3's per-row PiT stage on the committed
//! miniature exports, in a CPU-only child with Metal trapped:
//!
//! * **parity** — a streamed backbone / tower / route reproduces its resident twin bit for bit
//!   (same constructors, same casts, same graph), at every window, dense and packed; the per-row
//!   PiT stage matches the batched one within the bounded-attention tolerance;
//! * **residency** — a streamed load holds only its skeleton, and each window is released before
//!   the next one is read (the active memory at every window boundary stays flat);
//! * **pricing** — [`crate::ios_pricing::ios_pricing`] states, per phase, exactly the largest
//!   scratch the route's guards check on a real run of every regime.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::rc::Rc;

use mlx_gen::gen_core::iris::{IrisConfig, TEXT_ENCODER_COMPONENT};
use mlx_gen::gen_core::{
    GenerationMemory, LoadShape, MemoryNumericTier, MemorySelection, MemoryStrategy,
    TransformerComponent,
};
use mlx_gen::ios_pricing::{IosPhase, IosPricingRequest};
use mlx_gen::ios_quantum::{
    with_cpu_diagnostic_headroom, with_cpu_diagnostic_observer, QuantumEvent,
};
use mlx_gen::{
    AttentionPlan, GenerationOutput, GenerationRequest, Image, InputSizing, LoadSpec, Precision,
    Quant, TargetSize, Transform, TransformRequest, WeightsSource,
};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use crate::dit::{Execution, Stack, TextBatch};
use crate::packed::ConvertKind;
use crate::test_metal_guard as metal_guard;

pub(crate) const AMPLE: u64 = 1 << 34;

pub(crate) fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub(crate) fn backbone_dir() -> PathBuf {
    fixtures().join("tiny-snapshot/iris")
}

fn text_encoder_dir() -> PathBuf {
    fixtures().join("tiny-snapshot/text_encoder")
}

pub(crate) fn in_cpu_child(test: &str, body: fn()) {
    const CHILD: &str = "IRIS_STREAM_CPU_CHILD";
    const DONE: &str = "IRIS_STREAM_CPU_CHILD_DONE";
    if std::env::var(CHILD).as_deref() == Ok(test) {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert_eq!(
            mlx_rs::Device::try_default().expect("CPU default-device query failed"),
            mlx_rs::Device::cpu()
        );
        assert!(metal_guard::forbid_metal());
        assert!(metal_guard::metal_forbidden());
        // Select the native CPU stream before constructing any fixture, not just in scopes.
        assert_eq!(
            unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
            0
        );
        assert_eq!(
            mlx_rs::Stream::new().get_index().unwrap(),
            mlx_rs::Stream::cpu().get_index().unwrap()
        );
        body();
        std::fs::write(std::env::var(DONE).unwrap(), b"ok").unwrap();
        return;
    }
    let done = tempfile::tempdir().unwrap();
    let marker = done.path().join("done");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture", "--test-threads", "1"])
        .env(CHILD, test)
        .env(DONE, &marker)
        .status()
        .unwrap();
    assert!(status.success(), "{test} failed in its CPU child");
    assert!(marker.is_file(), "{test}: the CPU child ran no body");
}

fn host(a: &Array) -> Vec<f32> {
    let n: i32 = a.shape().iter().product();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    a.eval().unwrap();
    a.as_slice::<f32>().to_vec()
}

pub(crate) fn rand(shape: &[i32], seed: u64) -> Array {
    random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap())).unwrap()
}

/// `max |a − b| ≤ tol · max(1, max |b|)` — the crate's parity bound (`tests/common::assert_close`).
fn assert_close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: shape");
    let peak = want.iter().fold(0f32, |a, b| a.max(b.abs()));
    let max_abs = got
        .iter()
        .zip(want)
        .fold(0f32, |a, (g, w)| a.max((g - w).abs()));
    let bound = tol * peak.max(1.0);
    eprintln!("{name}: max|Δ|={max_abs:.3e} bound={bound:.3e}");
    assert!(max_abs <= bound, "{name}: max|Δ| {max_abs} > {bound}");
}

/// A batch-2 forward over a 16x24 image.
struct Case {
    x: Array,
    t: Array,
    states: Array,
    mask: Vec<Vec<i32>>,
}

impl Case {
    fn new(config: &IrisConfig) -> Self {
        let m = &config.model;
        let layers = config.text_encoder.hidden_layers.len() as i32;
        let mut mask = vec![vec![1; m.text_len]; 2];
        mask[1][m.text_len - 3..].fill(0);
        Self {
            x: rand(&[2, 3, 16, 24], 7),
            t: Array::from_slice(&[500.0f32, 250.0], &[2]),
            states: rand(&[2, m.text_len as i32, layers, m.text_dim as i32], 8),
            mask,
        }
    }

    fn forward(&self, dit: &crate::IrisDiT, exec: Execution<'_>) -> Vec<f32> {
        host(
            &dit.forward_exec(
                &self.x,
                &self.t,
                &TextBatch {
                    states: &self.states,
                    mask: &self.mask,
                },
                exec,
            )
            .unwrap(),
        )
    }
}

fn exec(window: Option<usize>, pixel_rows: bool) -> Execution<'static> {
    Execution {
        plan: AttentionPlan::UNBOUNDED,
        window,
        pixel_rows,
    }
}

// ---- parity ---------------------------------------------------------------------------------

fn backbone_parity_body() {
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let case = Case::new(&config);
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let resident = crate::model::load_backbone(&backbone_dir(), &config, dtype).unwrap();
        let want = case.forward(&resident, exec(None, false));
        let streamed =
            crate::model::load_backbone_streamed(&backbone_dir(), &config, dtype, None, None)
                .unwrap();
        assert!(streamed.is_streamed() && !resident.is_streamed());
        for window in [Some(1), Some(2), None] {
            assert_eq!(
                case.forward(&streamed, exec(window, false)),
                want,
                "{dtype:?} window {window:?}: a streamed backbone must reproduce the resident one"
            );
        }
        // Rung 3's per-row PiT stage: the rows never attend to each other, so the stage is the
        // batched one up to matmul blocking — on both backbones alike.
        let tol = if dtype == Dtype::Float32 { 2e-4 } else { 0.03 };
        let rows = case.forward(&resident, exec(None, true));
        assert_close(&format!("per-row PiT {dtype:?}"), &rows, &want, tol);
        assert_eq!(case.forward(&streamed, exec(Some(1), true)), rows);
        // A streamed backbone has no resident stacks for a training hook to run.
        let refused = streamed.forward_hooked(
            &case.x,
            &case.t,
            &TextBatch {
                states: &case.states,
                mask: &case.mask,
            },
            &mut crate::dit::NoHooks,
        );
        assert!(matches!(refused, Err(mlx_gen::Error::Unsupported(_))));
    }
    // A packed export streams at its own tier, bit-identical to its eager load; the dense release
    // asked for at a packed tier is refused rather than re-quantized every window.
    let root = tempfile::tempdir().unwrap();
    for quant in [Quant::Q4, Quant::Q8] {
        let packed = crate::packed_tests::convert(
            ConvertKind::Export { depth: false },
            &backbone_dir(),
            root.path(),
            quant,
        );
        let pconfig = IrisConfig::from_dir(&packed).unwrap();
        let (eager, _) = crate::model::load_backbone_quantized(
            &packed,
            &pconfig,
            Dtype::Bfloat16,
            &[],
            mlx_gen::gen_core::iris::IrisTask::Generation,
            crate::model::MODEL_ID,
            Some(quant),
        )
        .unwrap();
        let streamed = crate::model::load_backbone_streamed(
            &packed,
            &pconfig,
            Dtype::Bfloat16,
            Some(quant),
            None,
        )
        .unwrap();
        assert_eq!(
            case.forward(&streamed, exec(Some(1), false)),
            case.forward(&eager, exec(None, false)),
            "{quant:?}: a streamed packed backbone must reproduce its eager load"
        );
        match crate::model::load_backbone_streamed(
            &backbone_dir(),
            &config,
            Dtype::Bfloat16,
            Some(quant),
            None,
        ) {
            Err(mlx_gen::Error::Unsupported(m)) => assert!(m.contains("re-quantize"), "{m}"),
            other => panic!("expected a typed refusal, got {:?}", other.err()),
        }
    }
}

#[test]
fn a_streamed_backbone_reproduces_the_resident_one() {
    in_cpu_child(
        "stream_tests::a_streamed_backbone_reproduces_the_resident_one",
        backbone_parity_body,
    );
}

/// The miniature tower split into two readable shards (layers on both), with an index — the
/// release's sharded layout, so a window can span shards.
fn two_shard_text_encoder(root: &std::path::Path, source: &std::path::Path) -> PathBuf {
    let dir = root.join("two-shard-text-encoder");
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_owned();
        if name != "model.safetensors" {
            std::fs::copy(&path, dir.join(name)).unwrap();
        }
    }
    let tensors = Array::load_safetensors(source.join("model.safetensors")).unwrap();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let (mut first, mut second, mut map) = (Vec::new(), Vec::new(), serde_json::Map::new());
    for (i, name) in names.into_iter().enumerate() {
        let shard = if i % 2 == 0 {
            first.push((name.as_str(), &tensors[name]));
            "model-00001-of-00002.safetensors"
        } else {
            second.push((name.as_str(), &tensors[name]));
            "model-00002-of-00002.safetensors"
        };
        map.insert(name.clone(), shard.into());
    }
    for (part, file) in [
        (first, "model-00001-of-00002.safetensors"),
        (second, "model-00002-of-00002.safetensors"),
    ] {
        Array::save_safetensors(
            part,
            None::<&std::collections::HashMap<String, String>>,
            dir.join(file),
        )
        .unwrap();
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::json!({ "metadata": {}, "weight_map": map }).to_string(),
    )
    .unwrap();
    dir
}

fn tower_parity_body() {
    use crate::text_encoder::IrisTextEncoder;
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let cancel = mlx_gen::CancelFlag::default();
    let sharded = two_shard_text_encoder(root.path(), &text_encoder_dir());
    let aligned = crate::packed_tests::aligned_tiny_text_encoder(root.path());
    let mut towers = vec![
        (text_encoder_dir(), None, text_encoder_dir()),
        (sharded, None, text_encoder_dir()),
    ];
    for quant in [Quant::Q4, Quant::Q8] {
        let packed =
            crate::packed_tests::convert(ConvertKind::TextEncoder, &aligned, root.path(), quant);
        towers.push((packed.clone(), Some(quant), packed));
    }
    for (dir, quant, reference) in towers {
        let resident =
            IrisTextEncoder::load_quantized(&reference, &config.text_encoder, quant).unwrap();
        let streamed = IrisTextEncoder::load_streamed(&dir, &config.text_encoder, quant).unwrap();
        assert!(streamed.is_streamed() && !resident.is_streamed());
        assert_eq!(streamed.is_quantized(), quant.is_some());
        for prompt in ["a red fox in the snow", ""] {
            let window = resident.window(prompt).unwrap();
            let want = resident.encode_window(&window).unwrap();
            for layers in [Some(1), Some(5), None] {
                let got = streamed
                    .encode_window_with(&window, layers, &cancel)
                    .unwrap();
                assert_eq!(
                    host(&got.states),
                    host(&want.states),
                    "{} {quant:?} window {layers:?}: the streamed tower must reproduce the \
                     resident one",
                    dir.display()
                );
                assert_eq!(got.mask, want.mask);
            }
        }
    }
    // The dense release asked for at a packed tier is refused rather than re-quantized per window.
    match IrisTextEncoder::load_streamed(&text_encoder_dir(), &config.text_encoder, Some(Quant::Q4))
    {
        Err(mlx_gen::Error::Unsupported(m)) => assert!(m.contains("re-quantize"), "{m}"),
        other => panic!(
            "expected a typed refusal, got {:?}",
            other.err().map(|e| e.to_string())
        ),
    }
}

#[test]
fn a_streamed_text_tower_reproduces_the_resident_one() {
    in_cpu_child(
        "stream_tests::a_streamed_text_tower_reproduces_the_resident_one",
        tower_parity_body,
    );
}

pub(crate) fn generation_spec(shape: LoadShape) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(backbone_dir())).with_load_shape(shape);
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(text_encoder_dir()),
    );
    spec
}

/// The contract's selection of `strategy` for `spec` — its cheapest declared one whose component
/// scope is `component` (any when `None`) — and the `GenerationMemory` it writes on a request.
fn selection(
    provider: &str,
    spec: &LoadSpec,
    strategy: MemoryStrategy,
    component: Option<TransformerComponent>,
) -> (MemorySelection, GenerationMemory) {
    let contract = crate::memory_strategy::memory_strategy_contract(provider, spec).unwrap();
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let space = contract.selection_space(strategy, tier, false).unwrap();
    let chosen = *space
        .iter()
        .rev()
        .find(|s| component.is_none() || s.parameters.transformer_window_component == component)
        .unwrap();
    let memory = contract.generation_memory(&chosen).unwrap_or_else(|| {
        assert_eq!(
            chosen.strategy,
            MemoryStrategy::Resident,
            "optimized selection lost its execution controls"
        );
        assert_eq!(
            contract.resident_request_memory,
            mlx_gen::gen_core::ResidentRequestMemory::PreserveLoadDefaults
        );
        GenerationMemory::default()
    });
    (chosen, memory)
}

fn prompt_batch_request(memory: GenerationMemory) -> GenerationRequest {
    GenerationRequest {
        prompt: String::new(),
        prompt_batch: vec!["a red fox in the snow".into(), "golden hour".into()],
        width: 32,
        height: 16,
        steps: Some(2),
        seed: Some(42),
        memory: (memory != GenerationMemory::default()).then_some(memory),
        ..Default::default()
    }
}

fn pixels(output: GenerationOutput) -> Vec<Vec<u8>> {
    match output {
        GenerationOutput::Images(images) => images.into_iter().map(|i| i.pixels).collect(),
        _ => panic!("expected images"),
    }
}

fn generation_parity_body() {
    let root = tempfile::tempdir().unwrap();
    for (tier, deferred_spec) in crate::anchor_tests::generation_tiers(root.path()) {
        eprintln!("GENERATION PARITY {tier}");
        let eager_spec = deferred_spec
            .clone()
            .with_load_shape(LoadShape::EagerMaterialization);
        let eager = crate::model::load(&eager_spec).unwrap();
        let deferred = crate::model::load(&deferred_spec).unwrap();
        let run = |g: &dyn mlx_gen::Generator, memory| {
            pixels(
                g.generate(&prompt_batch_request(memory), &mut |_| {})
                    .unwrap(),
            )
        };
        // The Mac default (no memory block) on a deferred load: whole-stack windows, the resident
        // image bit for bit.
        let default = run(eager.as_ref(), GenerationMemory::default());
        assert_eq!(run(deferred.as_ref(), GenerationMemory::default()), default);
        // Rung 3 (staged, chunked attention, per-row PiT stage) on the eager load is the reference of
        // every rung-4 scope on the deferred one.
        let (_, rung3) = selection(
            crate::model::MODEL_ID,
            &eager_spec,
            MemoryStrategy::BoundedAttention,
            None,
        );
        assert!(rung3.chunk_attention && rung3.stage_residency);
        let reference = run(eager.as_ref(), rung3);
        for component in [
            TransformerComponent::Dit,
            TransformerComponent::TextEncoder,
            TransformerComponent::Both,
        ] {
            let (_, rung4) = selection(
                crate::model::MODEL_ID,
                &deferred_spec,
                MemoryStrategy::BoundedTransformerResidency,
                Some(component),
            );
            assert!(rung4.stream_transformer_blocks && rung4.chunk_attention);
            assert_eq!(rung4.transformer_window_size, Some(1));
            assert_eq!(
                run(deferred.as_ref(), rung4),
                reference,
                "{component:?}: rung 4 must reproduce rung 3's image"
            );
        }
        // The eager generator refuses a rung-4 request by name rather than ignoring it.
        let (_, rung4) = selection(
            crate::model::MODEL_ID,
            &deferred_spec,
            MemoryStrategy::BoundedTransformerResidency,
            None,
        );
        assert!(eager
            .generate(&prompt_batch_request(rung4), &mut |_| {})
            .is_err());
    }
}

#[test]
fn a_deferred_generation_reproduces_the_eager_one() {
    in_cpu_child(
        "stream_tests::a_deferred_generation_reproduces_the_eager_one",
        generation_parity_body,
    );
}

pub(crate) fn restore_request() -> TransformRequest {
    TransformRequest {
        image: Image {
            width: 16,
            height: 12,
            pixels: (0..16 * 12 * 3).map(|i| (i * 53 % 256) as u8).collect(),
        },
        target: TargetSize::Scale(4.0),
        input_sizing: InputSizing::Original,
        ..Default::default()
    }
}

fn downstream_parity_body() {
    use crate::restoration::IrisRestorer;
    let root = tempfile::tempdir().unwrap();
    let upscaler = fixtures().join("tiny-snapshot/upscaler");
    for (tier, spec) in crate::anchor_tests::downstream_tiers(root.path(), &upscaler, false) {
        let eager = IrisRestorer::load(
            &spec
                .clone()
                .with_load_shape(LoadShape::EagerMaterialization),
        )
        .unwrap();
        let deferred = IrisRestorer::load(&spec).unwrap();
        let req = restore_request();
        assert!(eager.plan(&req).unwrap().forward_count() > 1);
        assert_eq!(
            deferred.apply(&req, &mut |_| {}).unwrap().pixels,
            eager.apply(&req, &mut |_| {}).unwrap().pixels,
            "{tier}: a deferred restorer must reproduce the eager one"
        );
    }
    let depth = fixtures().join("tiny-depth");
    for (tier, spec) in crate::anchor_tests::downstream_tiers(root.path(), &depth, true) {
        let eager = crate::depth::load_depth(
            &spec
                .clone()
                .with_load_shape(LoadShape::EagerMaterialization),
        )
        .unwrap();
        let deferred = crate::depth::load_depth(&spec).unwrap();
        let rgb = rand(&[1, 3, 16, 24], 11);
        assert_eq!(
            host(&deferred.forward(&rgb).unwrap()),
            host(&eager.forward(&rgb).unwrap()),
            "{tier}: a deferred depth model must reproduce the eager one"
        );
    }
}

#[test]
fn deferred_depth_and_restoration_reproduce_the_eager_ones() {
    in_cpu_child(
        "stream_tests::deferred_depth_and_restoration_reproduce_the_eager_ones",
        downstream_parity_body,
    );
}

// ---- residency ------------------------------------------------------------------------------

fn active() -> u64 {
    mlx_rs::memory::get_active_memory() as u64
}

fn residency_body() {
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let case = Case::new(&config);
    // The inputs are materialized before anything is measured: a lazy input would land inside the
    // first window and read as growth.
    eval([&case.x, &case.t, &case.states]).unwrap();
    mlx_rs::memory::set_cache_limit(0);
    mlx_rs::memory::clear_cache();
    let base = active();
    let resident = crate::model::load_backbone(&backbone_dir(), &config, Dtype::Float32).unwrap();
    let resident_bytes = active() - base;
    drop(resident);
    mlx_rs::memory::clear_cache();
    let base = active();
    let streamed =
        crate::model::load_backbone_streamed(&backbone_dir(), &config, Dtype::Float32, None, None)
            .unwrap();
    let skeleton = active() - base;
    // What is active is the skeleton the backbone holds (each buffer page-rounded at most).
    let arrays = streamed.arrays().len() as u64;
    assert!(
        skeleton >= streamed.resident_weight_bytes()
            && skeleton <= streamed.resident_weight_bytes() + arrays * crate::retention::PAGE_BYTES,
        "skeleton {skeleton} B vs held {} B",
        streamed.resident_weight_bytes()
    );
    let stream = streamed.block_stream().unwrap();
    let stacks: u64 = Stack::ALL
        .into_iter()
        .flat_map(|stack| (0..stream.len(stack)).map(move |i| (stack, i)))
        .map(|(stack, i)| stream.read(stack, i).held)
        .sum();
    eprintln!("resident {resident_bytes} B, skeleton {skeleton} B, stacks {stacks} B");
    assert!(
        skeleton + stacks / 2 < resident_bytes,
        "the streamed load must hold its skeleton, not its stacks"
    );
    // Window by window, the active memory at each window boundary only moves by the carried
    // state: the previous window's blocks were released before the next was read. A window that
    // leaked (its graph left lazy, or its view or blocks kept) would add at least the stack's
    // smallest block at the boundary after it.
    let m = &config.model;
    let (b, n, t, d) = (
        2u64,
        (16 / m.patch_size * 24 / m.patch_size) as u64,
        m.text_len as u64,
        m.hidden_size as u64,
    );
    let layers = config.text_encoder.hidden_layers.len() as u64;
    let carry = |stack: Stack| -> i64 {
        let a = crate::retention::allocated;
        (match stack {
            Stack::Layer => a(4 * b * t * layers * m.text_dim as u64),
            Stack::Refiner => a(4 * b * t * d),
            Stack::Trunk => a(4 * b * n * d) + a(4 * b * t * d),
            Stack::Pixel => {
                a(4 * b * n * (m.patch_size * m.patch_size * m.pixel.hidden_size) as u64)
            }
        }) as i64
    };
    let seen: Rc<RefCell<BTreeMap<Stack, Vec<u64>>>> = Rc::default();
    let sink = seen.clone();
    crate::dit::WINDOW_HOOK.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move |stack, _| {
            sink.borrow_mut().entry(stack).or_default().push(active());
        }))
    });
    let out = case.forward(&streamed, exec(Some(1), false));
    crate::dit::WINDOW_HOOK.with(|hook| hook.borrow_mut().take());
    assert!(!out.is_empty());
    let seen = seen.take();
    assert_eq!(seen.len(), 4, "every stack streamed");
    for (stack, boundaries) in &seen {
        assert_eq!(boundaries.len(), stream.len(*stack), "{stack:?}");
        let smallest = (0..stream.len(*stack))
            .map(|i| stream.read(*stack, i).held)
            .min()
            .unwrap() as i64;
        for (k, pair) in boundaries.windows(2).enumerate() {
            let grew = pair[1] as i64 - pair[0] as i64;
            // Into the second window the carry replaces the stack's input; after it, carry for
            // carry.
            let allowed = if k == 0 { carry(*stack) } else { 0 };
            assert!(
                grew < allowed + smallest,
                "{stack:?}: active memory grew {grew} B into window {} ({boundaries:?}, carry \
                 {allowed} B, smallest block {smallest} B) — a window was not released",
                k + 1
            );
        }
    }

    // The tower: its embedding (and final norm) resident, every layer window released.
    use crate::text_encoder::IrisTextEncoder;
    mlx_rs::memory::clear_cache();
    let base = active();
    let te =
        IrisTextEncoder::load_streamed(&text_encoder_dir(), &config.text_encoder, None).unwrap();
    assert!(active() - base <= te.resident_weight_bytes() + 2 * crate::retention::PAGE_BYTES);
    let tower = te.streamed().unwrap();
    let smallest = (0..tower.config().num_layers)
        .map(|i| tower.read_price(i).held)
        .min()
        .unwrap();
    let seen: Rc<RefCell<Vec<u64>>> = Rc::default();
    let sink = seen.clone();
    crate::text_encoder::TOWER_WINDOW_HOOK.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move |_| sink.borrow_mut().push(active())))
    });
    let window = te.window("a red fox in the snow").unwrap();
    te.encode_window_with(&window, Some(1), &mlx_gen::CancelFlag::default())
        .unwrap();
    crate::text_encoder::TOWER_WINDOW_HOOK.with(|hook| hook.borrow_mut().take());
    let seen = seen.take();
    let deepest = *config.text_encoder.hidden_layers.last().unwrap();
    assert_eq!(seen.len(), deepest);
    // The picked states the walk keeps grow by one state each pick; the layer windows do not.
    let state = 2 * window.input_ids.len() as u64 * tower.config().hidden_size as u64;
    let picks = config.text_encoder.hidden_layers.len() as u64;
    let spread = seen.iter().max().unwrap() - seen.iter().min().unwrap();
    assert!(
        spread < smallest + picks * crate::retention::allocated(state),
        "the tower's active memory grew {spread} B across its windows ({seen:?})"
    );
}

#[test]
fn a_streamed_load_holds_its_skeleton_and_releases_every_window() {
    in_cpu_child(
        "stream_tests::a_streamed_load_holds_its_skeleton_and_releases_every_window",
        residency_body,
    );
}

// ---- pricing --------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum PricingBoundary {
    Checked(String, u64),
    Synchronized(String),
    BackboneWindow(Stack, usize),
    TowerWindow(usize),
}

fn raw_load(label: &str) -> bool {
    matches!(
        label,
        crate::model::BACKBONE_LOAD_QUANTUM
            | crate::text_encoder::quanta::LOAD
            | crate::depth::quanta::LOAD
            | crate::restoration::quanta::LOAD
    )
}

/// sc-26270 — streamed block constructor targets: check every reached boundary, not merely
/// the largest check of a label. Existing window-open hooks independently witness each read.
fn assert_boundary_protocol(events: &[PricingBoundary]) {
    use crate::text_encoder::quanta as te_q;
    let checked = |event: Option<&PricingBoundary>, label: &str| matches!(event, Some(PricingBoundary::Checked(found, _)) if found == label);
    let synchronized = |event: Option<&PricingBoundary>, label: &str| matches!(event, Some(PricingBoundary::Synchronized(found)) if found == label);
    let mut raw = 0;
    let mut joint = 0;
    let mut backbone_raw = 0;
    let mut export_raw = 0;
    let mut export_targets = 0;
    let mut pending = false;
    let mut construction_count = 0;
    let mut evaluation_count = 0;
    let mut backbone_windows = 0;
    let mut tower_windows = 0;
    let mut backbone_reads = 0;
    let mut tower_reads = 0;
    let mut backbone_finishes = 0;
    let mut tower_finishes = 0;
    for (index, event) in events.iter().enumerate() {
        match event {
            PricingBoundary::BackboneWindow(stack, first) => {
                backbone_windows += 1;
                assert!(
                    checked(events.get(index + 1), crate::dit::quanta::BLOCK_READ),
                    "{stack:?} window {first} must check before its read: {events:?}"
                );
                assert!(
                    synchronized(events.get(index + 2), crate::dit::quanta::BLOCK_READ),
                    "{stack:?} window {first} must finish that read before its blocks run"
                );
            }
            PricingBoundary::TowerWindow(first) => {
                tower_windows += 1;
                assert!(
                    checked(events.get(index + 1), te_q::LAYER_READ),
                    "tower window {first} must check before its read: {events:?}"
                );
                assert!(
                    synchronized(events.get(index + 2), te_q::LAYER_READ),
                    "tower window {first} must finish its read before running a layer"
                );
            }
            PricingBoundary::Checked(label, scratch) => {
                if raw_load(label) {
                    raw += 1;
                    backbone_raw += usize::from(label == crate::model::BACKBONE_LOAD_QUANTUM);
                    export_raw += usize::from(
                        label == crate::depth::quanta::LOAD
                            || label == crate::restoration::quanta::LOAD,
                    );
                    assert!(
                        checked(events.get(index + 1), crate::stream::LOAD_TARGET_QUANTUM),
                        "raw {label} ({scratch}) must be followed by its real joint check"
                    );
                } else if label == crate::stream::LOAD_TARGET_QUANTUM {
                    joint += 1;
                    assert!(
                        matches!(index.checked_sub(1).and_then(|i| events.get(i)),
                        Some(PricingBoundary::Checked(previous, _)) if raw_load(previous)),
                        "joint check must belong to the immediately preceding raw source"
                    );
                } else if label == crate::nn::PENDING_CONSTRUCTION_QUANTUM {
                    assert!(
                        !pending,
                        "a repeated build lost its preceding evaluation check"
                    );
                    pending = true;
                    construction_count += 1;
                } else if label == crate::nn::ADAPTED_BASE_EVAL_QUANTUM {
                    assert!(
                        pending,
                        "evaluation must follow an actual construction boundary"
                    );
                    pending = false;
                    evaluation_count += 1;
                } else if label == crate::stream::EXPORT_TARGET_QUANTUM {
                    export_targets += 1;
                } else if label == crate::dit::quanta::BLOCK_READ {
                    backbone_reads += 1;
                    assert!(
                        matches!(
                            index.checked_sub(1).and_then(|i| events.get(i)),
                            Some(PricingBoundary::BackboneWindow(_, _))
                        ),
                        "every backbone read check must identify its actual window"
                    );
                } else if label == te_q::LAYER_READ {
                    tower_reads += 1;
                    assert!(
                        matches!(
                            index.checked_sub(1).and_then(|i| events.get(i)),
                            Some(PricingBoundary::TowerWindow(_))
                        ),
                        "every tower read check must identify its actual window"
                    );
                }
            }
            PricingBoundary::Synchronized(label) => {
                if label == crate::dit::quanta::BLOCK_READ {
                    backbone_finishes += 1;
                    assert!(checked(
                        index.checked_sub(1).and_then(|i| events.get(i)),
                        label
                    ));
                } else if label == te_q::LAYER_READ {
                    tower_finishes += 1;
                    assert!(checked(
                        index.checked_sub(1).and_then(|i| events.get(i)),
                        label
                    ));
                }
            }
        }
    }
    assert_eq!(
        raw, joint,
        "no raw/joint occurrence may be lost in a per-label maximum"
    );
    assert_eq!(
        export_raw, export_targets,
        "each export owns one late upload boundary"
    );
    assert!(
        !pending,
        "the last build must also reach its evaluation check"
    );
    assert_eq!(construction_count, evaluation_count);
    if construction_count != 0 {
        // Each observed body uses one specification, including any staged reloads of it.
        assert_eq!(
            construction_count, backbone_raw,
            "every repeated adapter/dense-quant backbone load must reach both pending checks"
        );
    }
    assert_eq!(
        (backbone_windows, backbone_windows),
        (backbone_reads, backbone_finishes)
    );
    assert_eq!(
        (tower_windows, tower_windows),
        (tower_reads, tower_finishes)
    );
}

#[test]
fn ordered_pricing_protocol_rejects_a_missing_repeated_boundary() {
    let read = crate::dit::quanta::BLOCK_READ;
    let valid = vec![
        PricingBoundary::BackboneWindow(Stack::Trunk, 0),
        PricingBoundary::Checked(read.into(), 64),
        PricingBoundary::Synchronized(read.into()),
        PricingBoundary::BackboneWindow(Stack::Trunk, 1),
        PricingBoundary::Checked(read.into(), 64),
        PricingBoundary::Synchronized(read.into()),
    ];
    assert_boundary_protocol(&valid);
    // The per-label maximum remains 64 in both mutations. The independent operation hooks
    // still demand TWO ordered check/finish pairs, so neither omission can disappear in a map.
    let mut missing_check = valid.clone();
    missing_check.remove(4);
    assert!(std::panic::catch_unwind(|| assert_boundary_protocol(&missing_check)).is_err());
    let mut missing_pair = valid;
    missing_pair.truncate(4);
    assert!(std::panic::catch_unwind(|| assert_boundary_protocol(&missing_pair)).is_err());
}

struct RestorePricingHooks {
    backbone: Option<crate::dit::WindowHook>,
    tower: Option<crate::text_encoder::TowerWindowHook>,
}

impl Drop for RestorePricingHooks {
    fn drop(&mut self) {
        crate::dit::WINDOW_HOOK.with(|hook| *hook.borrow_mut() = self.backbone.take());
        crate::text_encoder::TOWER_WINDOW_HOOK.with(|hook| *hook.borrow_mut() = self.tower.take());
    }
}

/// Collect the original largest-per-label prices AND an independently count-bearing ordered
/// protocol. Hooks only record scalar window identifiers; they add no native query or evaluation.
fn observed_phases(body: impl FnOnce() -> mlx_gen::Result<()>) -> BTreeMap<String, u64> {
    let seen: Rc<RefCell<BTreeMap<String, u64>>> = Rc::default();
    let boundaries: Rc<RefCell<Vec<PricingBoundary>>> = Rc::default();
    let windows = boundaries.clone();
    let backbone = crate::dit::WINDOW_HOOK.with(|hook| {
        hook.borrow_mut().replace(Box::new(move |stack, first| {
            windows
                .borrow_mut()
                .push(PricingBoundary::BackboneWindow(stack, first))
        }))
    });
    let windows = boundaries.clone();
    let tower = crate::text_encoder::TOWER_WINDOW_HOOK.with(|hook| {
        hook.borrow_mut().replace(Box::new(move |first| {
            windows
                .borrow_mut()
                .push(PricingBoundary::TowerWindow(first))
        }))
    });
    let _restore = RestorePricingHooks { backbone, tower };
    let sink = seen.clone();
    let events = boundaries.clone();
    with_cpu_diagnostic_headroom(AMPLE, || {
        with_cpu_diagnostic_observer(
            move |event| match event {
                QuantumEvent::Checked { label, scratch } => {
                    let mut seen = sink.borrow_mut();
                    let entry = seen.entry(label.clone()).or_insert(0);
                    *entry = (*entry).max(*scratch);
                    events
                        .borrow_mut()
                        .push(PricingBoundary::Checked(label.clone(), *scratch));
                }
                QuantumEvent::Synchronized { label, .. } => {
                    events
                        .borrow_mut()
                        .push(PricingBoundary::Synchronized(label.clone()));
                }
            },
            body,
        )
    })
    .unwrap();
    assert_boundary_protocol(&boundaries.borrow());
    seen.take()
}

/// Assert the pricing prices exactly the quanta a run checks — every label it observed, at that
/// label's largest observed scratch, and no label the run never checked — and so states each
/// phase's largest check as the phase's `scratch_bytes`.
fn assert_priced(
    name: &str,
    observed: &BTreeMap<String, u64>,
    priced: &crate::ios_pricing::Priced,
) {
    let mut by_label: BTreeMap<String, u64> = BTreeMap::new();
    for &(_, label, scratch) in &priced.quanta {
        let entry = by_label.entry(label.to_owned()).or_insert(0);
        *entry = (*entry).max(scratch);
    }
    eprintln!("{name}: observed {observed:?}");
    assert_eq!(
        by_label.keys().collect::<Vec<_>>(),
        observed.keys().collect::<Vec<_>>(),
        "{name}: the priced quanta are not the checked ones"
    );
    for (label, scratch) in observed {
        assert_eq!(
            by_label[label], *scratch,
            "{name}: {label} priced {} but checked at {scratch}",
            by_label[label]
        );
    }
    let mut per_phase: BTreeMap<IosPhase, u64> = BTreeMap::new();
    for (label, scratch) in observed {
        let phase = crate::ios_pricing::ios_quantum_phase(label).unwrap();
        let entry = per_phase.entry(phase).or_insert(0);
        *entry = (*entry).max(*scratch);
    }
    assert_eq!(
        priced
            .pricing
            .phases
            .iter()
            .map(|p| (p.phase, p.scratch_bytes))
            .collect::<BTreeMap<_, _>>(),
        per_phase,
        "{name}: each phase's scratch_bytes is its largest check"
    );
}

fn generation_pricing_case(
    name: &str,
    spec: LoadSpec,
    strategy: MemoryStrategy,
    component: Option<TransformerComponent>,
    prompts: &[&str],
) {
    let (chosen, memory) = selection(crate::model::MODEL_ID, &spec, strategy, component);
    let request = GenerationRequest {
        prompt: if prompts.len() == 1 {
            prompts[0].into()
        } else {
            String::new()
        },
        prompt_batch: if prompts.len() == 1 {
            Vec::new()
        } else {
            prompts.iter().map(|p| p.to_string()).collect()
        },
        width: 32,
        height: 16,
        // Three steps: the second-order update (the priced step) runs from the second step on,
        // and DPM-Solver++ 2M lowers the final step's order.
        steps: Some(3),
        seed: Some(42),
        memory: (memory != GenerationMemory::default()).then_some(memory),
        ..Default::default()
    };
    // The longest encoded window of the request (its prompts and the CFG negative).
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let te_dir = match &spec.components[TEXT_ENCODER_COMPONENT] {
        WeightsSource::Dir(dir) => dir.clone(),
        WeightsSource::File(_) => unreachable!(),
    };
    let te = if spec.load_shape == LoadShape::DeferredMaterialization {
        crate::text_encoder::IrisTextEncoder::load_streamed(
            &te_dir,
            &config.text_encoder,
            spec.quantize,
        )
    } else {
        crate::text_encoder::IrisTextEncoder::load(&te_dir, &config.text_encoder)
    }
    .unwrap();
    let token_window = prompts
        .iter()
        .chain(std::iter::once(&""))
        .map(|p| te.window(p).unwrap().input_ids.len() as u32)
        .max()
        .unwrap();
    drop(te);
    let observed = observed_phases(|| {
        let generator = crate::model::load(&spec)?;
        generator
            .generate(&request, &mut |_| {})
            .map_err(|e| mlx_gen::Error::Msg(e.to_string()))?;
        Ok(())
    });
    let mode = mlx_gen::gen_core::MemoryMode::TextToImage;
    let pricing = crate::ios_pricing::price(
        &IosPricingRequest {
            provider: crate::model::MODEL_ID,
            spec: &spec,
            mode: &mode,
            geometry: mlx_gen::gen_core::MemoryGeometry {
                width: 32,
                height: 16,
                batch: prompts.len() as u32,
                frames: 1,
                reference_count: 0,
            },
            selection: &chosen,
            execution: &memory,
            generation: Some(&request),
            transform: None,
            depth: None,
            token_window,
        },
        crate::ios_pricing::Pricing::default(),
    )
    .unwrap();
    assert_priced(name, &observed, &pricing);
}

fn pricing_body() {
    let batch = ["a red fox in the snow", "golden hour"];
    let single = ["a lighthouse on a cliff at dusk"];
    let eager = generation_spec(LoadShape::EagerMaterialization);
    let deferred = generation_spec(LoadShape::DeferredMaterialization);
    let mut fp32 = generation_spec(LoadShape::EagerMaterialization);
    fp32.precision = Precision::Fp32;
    generation_pricing_case(
        "generation eager resident (fp32)",
        fp32,
        MemoryStrategy::Resident,
        None,
        &batch,
    );
    generation_pricing_case(
        "generation eager rung 3",
        eager.clone(),
        MemoryStrategy::BoundedAttention,
        None,
        &batch,
    );
    generation_pricing_case(
        "generation eager rung 3 (text)",
        eager,
        MemoryStrategy::BoundedAttention,
        None,
        &single,
    );
    for component in [
        TransformerComponent::Dit,
        TransformerComponent::TextEncoder,
        TransformerComponent::Both,
    ] {
        generation_pricing_case(
            &format!("generation deferred rung 4 {component:?}"),
            deferred.clone(),
            MemoryStrategy::BoundedTransformerResidency,
            Some(component),
            &batch,
        );
    }
    generation_pricing_case(
        "generation deferred rung 4 Both (text)",
        deferred.clone(),
        MemoryStrategy::BoundedTransformerResidency,
        Some(TransformerComponent::Both),
        &single,
    );
    generation_pricing_case(
        "generation deferred resident",
        deferred,
        MemoryStrategy::Resident,
        None,
        &batch,
    );
    // The packed tiers: the Mac's eager load of the dense release quantized at load, and the iOS
    // deferred load of the packed exports (backbone and tower).
    let root = tempfile::tempdir().unwrap();
    let aligned = crate::packed_tests::aligned_tiny_text_encoder(root.path());
    for quant in [Quant::Q4, Quant::Q8] {
        let mut at_load = generation_spec(LoadShape::EagerMaterialization).with_quant(quant);
        at_load.components.insert(
            TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(aligned.clone()),
        );
        generation_pricing_case(
            &format!("generation eager rung 3 {quant:?} (quantized at load)"),
            at_load,
            MemoryStrategy::BoundedAttention,
            None,
            &batch,
        );
        let backbone = crate::packed_tests::convert(
            ConvertKind::Export { depth: false },
            &backbone_dir(),
            root.path(),
            quant,
        );
        let tower =
            crate::packed_tests::convert(ConvertKind::TextEncoder, &aligned, root.path(), quant);
        let mut packed = LoadSpec::new(WeightsSource::Dir(backbone))
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_quant(quant);
        packed
            .components
            .insert(TEXT_ENCODER_COMPONENT.into(), WeightsSource::Dir(tower));
        generation_pricing_case(
            &format!("generation deferred rung 4 Both {quant:?} (packed)"),
            packed,
            MemoryStrategy::BoundedTransformerResidency,
            Some(TransformerComponent::Both),
            &batch,
        );
    }

    let upscaler_q4 = crate::packed_tests::convert(
        ConvertKind::Export { depth: false },
        &fixtures().join("tiny-snapshot/upscaler"),
        root.path(),
        Quant::Q4,
    );
    let depth_q4 = crate::packed_tests::convert(
        ConvertKind::Export { depth: true },
        &fixtures().join("tiny-depth"),
        root.path(),
        Quant::Q4,
    );
    for (shape, quant) in [
        (LoadShape::EagerMaterialization, None),
        (LoadShape::DeferredMaterialization, None),
        (LoadShape::DeferredMaterialization, Some(Quant::Q4)),
    ] {
        let strategy = if shape == LoadShape::DeferredMaterialization {
            MemoryStrategy::BoundedTransformerResidency
        } else {
            MemoryStrategy::Resident
        };
        let (upscaler, depth_dir) = match quant {
            Some(_) => (upscaler_q4.clone(), depth_q4.clone()),
            None => (
                fixtures().join("tiny-snapshot/upscaler"),
                fixtures().join("tiny-depth"),
            ),
        };
        let with_tier = |spec: LoadSpec| match quant {
            Some(quant) => spec.with_quant(quant),
            None => spec,
        };
        // Restoration, at the size its plan runs.
        let spec = with_tier(LoadSpec::new(WeightsSource::Dir(upscaler)).with_load_shape(shape));
        let req = restore_request();
        let mut geometry = None;
        let observed = observed_phases(|| {
            let restorer = crate::restoration::IrisRestorer::load(&spec)?;
            let plan = restorer.plan(&req)?;
            assert!(!plan.enlarged && plan.color_fix);
            geometry = Some(mlx_gen::gen_core::MemoryGeometry {
                width: req.image.width,
                height: req.image.height,
                batch: 1,
                frames: 1,
                reference_count: 0,
            });
            restorer
                .apply(&req, &mut |_| {})
                .map_err(|e| mlx_gen::Error::Msg(e.to_string()))?;
            Ok(())
        });
        let (chosen, memory) = selection(crate::restoration::MODEL_ID, &spec, strategy, None);
        let mode =
            crate::memory_strategy::memory_mode(mlx_gen::gen_core::iris::IrisTask::Restoration);
        let pricing = crate::ios_pricing::price(
            &IosPricingRequest {
                provider: crate::restoration::MODEL_ID,
                spec: &spec,
                mode: &mode,
                geometry: geometry.unwrap(),
                selection: &chosen,
                execution: &memory,
                generation: None,
                transform: Some(&req),
                depth: None,
                token_window: 0,
            },
            crate::ios_pricing::Pricing::default(),
        )
        .unwrap();
        assert_priced(
            &format!("restoration {shape:?} {quant:?}"),
            &observed,
            &pricing,
        );

        // Depth, at its model grid.
        let spec = with_tier(LoadSpec::new(WeightsSource::Dir(depth_dir)).with_load_shape(shape));
        let actual_depth = mlx_gen::gen_core::iris::depth::DepthRequest {
            image: Image {
                width: 24,
                height: 16,
                pixels: (0..24 * 16 * 3).map(|i| (i * 31 % 256) as u8).collect(),
            },
            resolution: mlx_gen::gen_core::iris::depth::DepthResolution::Native,
            ..Default::default()
        };
        let observed = observed_phases(|| {
            let depth = crate::depth::load_depth(&spec)?;
            mlx_gen::gen_core::iris::depth::IrisDepthEstimator::estimate(
                &depth,
                &actual_depth,
                &mut |_| {},
            )
            .map_err(mlx_gen::Error::from)?;
            Ok(())
        });
        let (chosen, memory) = selection(crate::depth::DEPTH_MODEL_ID, &spec, strategy, None);
        let mode = crate::memory_strategy::memory_mode(mlx_gen::gen_core::iris::IrisTask::Depth);
        let pricing = crate::ios_pricing::price(
            &IosPricingRequest {
                provider: crate::depth::DEPTH_MODEL_ID,
                spec: &spec,
                mode: &mode,
                geometry: mlx_gen::gen_core::MemoryGeometry {
                    width: 24,
                    height: 16,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                selection: &chosen,
                execution: &memory,
                generation: None,
                transform: None,
                depth: Some(&actual_depth),
                token_window: 0,
            },
            crate::ios_pricing::Pricing::default(),
        )
        .unwrap();
        assert_priced(&format!("depth {shape:?} {quant:?}"), &observed, &pricing);
    }
}

#[test]
fn the_pricing_states_the_largest_check_of_every_phase() {
    in_cpu_child(
        "stream_tests::the_pricing_states_the_largest_check_of_every_phase",
        pricing_body,
    );
}
