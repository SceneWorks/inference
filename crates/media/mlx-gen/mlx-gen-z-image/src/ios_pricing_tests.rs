//! Proofs of the header-only iOS pricing ([`crate::ios_pricing`]) against the production quanta.
//!
//! A tiny snapshot with the production component structure (text encoder, DiT, VAE and a
//! Fun-Controlnet-Union checkpoint, at tiny widths, random bf16 weights) is written as safetensors;
//! the pricing reads ITS headers exactly as the device reads a real snapshot's. The production code
//! paths then run on it under the CPU diagnostic fixture (`with_cpu_diagnostic_headroom` +
//! `with_cpu_diagnostic_observer`) in the order the staged generators run them — the loaders, the
//! prompt encode, the img2img / control encodes, the denoise batch and the decode batch — and every
//! observed check is compared with the pricing: per label and per phase, the largest observed
//! scratch EQUALS the largest priced scratch, and each phase's binding quantum is one the run checked
//! at exactly its priced scratch. Each test re-executes itself in a child pinned to the CPU with
//! Metal made fatal.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::{
    GenerationMemory, LoadShape, LoadSpec, OffloadPolicy, Quant, TransformerComponent,
    WeightsSource,
};
use mlx_gen::ios_pricing::IosPhase;
use mlx_gen::ios_quantum::{
    with_cpu_diagnostic_headroom, with_cpu_diagnostic_observer, Backend, QuantumEvent,
};
use mlx_gen::{CancelFlag, FlowMatchEuler};
use mlx_rs::{random, Array, Dtype};

use crate::ios_pricing::{fold, ios_quantum_phase, Component, Facts, Mode, Model, Priced, Route};
use crate::text_encoder::ZTextEncoderConfig;
use crate::transformer::ZImageTransformerConfig;

const CHILD: &str = "ZIMAGE_IOS_PRICING_CPU_CHILD";

const DONE: &str = "ZIMAGE_IOS_PRICING_CPU_CHILD_DONE";

// Canonical CPU-child completion guard, armed only at a target's true final statement.
struct CpuChildCompletion;
impl Drop for CpuChildCompletion {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let sync = unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) };
            assert_eq!(sync, 0, "native CPU synchronization after endpoint fixture owners dropped failed");
            std::fs::write(
                std::env::var(DONE).expect("CPU child completion path"),
                b"ok",
            )
            .expect("write CPU child completion marker");
        }
    }
}

fn legacy_final_arm_target(name: &str) -> bool {
    matches!(name,
        "text_pricing_equals_the_observed_quanta"
            | "reference_pricing_equals_the_observed_quanta"
            | "untiled_reference_pricing_equals_the_observed_quanta"
            | "control_pricing_equals_the_observed_quanta"
            | "control_reference_pricing_equals_the_observed_quanta"
            | "tile_extent_is_the_largest_tile_the_tiler_cuts"
            | "a_deferred_control_load_packs_only_its_resident_blocks"
    )
}

/// Far above every tiny quantum's scratch + the reserve: every guard passes.
pub(crate) const AMPLE: u64 = 1 << 34;
/// The fixture image edge: above the iOS bounded-encode tile (256), so the encode tiles, and above
/// the decode tile (256), so the decode tiles.
pub(crate) const EDGE: u32 = 288;
/// An edge within both tiles: the whole-image encode and the untiled decode.
const SMALL_EDGE: u32 = 128;
/// The fixture's padded token window and valid caption tokens (a padded caption: 20 → 32).
const WINDOW: i32 = 32;
const VALID: i32 = 20;

fn in_cpu_child(name: &str) -> bool {
    if std::env::var_os(CHILD).is_none() {
        let done = tempfile::tempdir().unwrap();
        let marker = done.path().join("done");
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("ios_pricing_tests::{name}"),
                "--nocapture",
                "--test-threads=1",
                "--include-ignored",
            ])
            .env(CHILD, "1")
            .env(DONE, &marker)
            .status()
            .unwrap();
        assert!(status.success(), "CPU child for {name} failed");
        assert!(
            marker.is_file(),
            "CPU child for {name} ran no completed body"
        );
        if legacy_final_arm_target(name) {
            assert_eq!(
                std::fs::read(&marker).expect("CPU child normal completion marker").as_slice(),
                b"ok",
                "CPU child for {name} did not write its current normal completion marker"
            );
        }
        return false;
    }
    if legacy_final_arm_target(name) {
        let marker = PathBuf::from(std::env::var(DONE).expect("CPU child requires parent DONE path"));
        assert!(marker.is_absolute(), "CPU child DONE path must be the absolute parent path");
        assert!(!marker.exists(), "CPU child requires a fresh parent completion path");
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert_eq!(
        mlx_rs::Device::try_default().expect("CPU default-device query failed"),
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
    assert!(crate::test_metal_guard::forbid_metal());
    assert!(crate::test_metal_guard::metal_forbidden());
    eprintln!("ZIMAGE_CPU_GUARDS_READY_V1 device=cpu native_stream=cpu fatal_metal_trap=1");
    true
}

fn rand(shape: &[i32], seed: u64, scale: f32) -> Array {
    let a = random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap()))
        .unwrap()
        .multiply(Array::from_f32(scale))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    a.eval().unwrap();
    a
}

/// A tensor writer: random bf16 at a fan-in scale (norm weights near one).
struct Tensors {
    seed: u64,
    out: Vec<(String, Array)>,
}

impl Tensors {
    fn new(seed: u64) -> Self {
        Self {
            seed,
            out: Vec::new(),
        }
    }
    fn put(&mut self, name: impl Into<String>, shape: &[i32]) {
        let name = name.into();
        self.seed += 1;
        let a = if name.contains("norm") && shape.len() == 1 {
            mlx_rs::ops::add(rand(shape, self.seed, 0.05), Array::from_f32(1.0))
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        } else {
            let fan_in = shape[1..].iter().product::<i32>().max(1);
            rand(shape, self.seed, (1.0 / fan_in as f32).sqrt())
        };
        self.out.push((name, a));
    }
    fn save(&self, path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        Array::save_safetensors(
            self.out
                .iter()
                .map(|(k, a)| (k.as_str(), a))
                .collect::<Vec<_>>(),
            None,
            path,
        )
        .unwrap();
    }
}

const DIM: i32 = 64;
const FF: i32 = 128;

/// The tiny DiT (production structure; widths quantizable at group 64).
pub(crate) fn dit_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 16,
        dim: DIM,
        n_layers: 4,
        n_refiner_layers: 2,
        n_heads: 4,
        norm_eps: 1e-5,
        cap_feat_dim: 64,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![8, 4, 4],
        axes_lens: vec![128, 64, 64],
        frequency_embedding_size: 256,
    }
}

/// One DiT block's tensors under `p` (`adaln`: the adaLN modulation of a noise-refiner / unified /
/// control block).
fn block(t: &mut Tensors, p: &str, adaln: bool) {
    for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
        t.put(format!("{p}.attention.{proj}.weight"), &[DIM, DIM]);
    }
    for norm in ["norm_q", "norm_k"] {
        t.put(format!("{p}.attention.{norm}.weight"), &[DIM / 4]);
    }
    for norm in [
        "attention_norm1",
        "attention_norm2",
        "ffn_norm1",
        "ffn_norm2",
    ] {
        t.put(format!("{p}.{norm}.weight"), &[DIM]);
    }
    t.put(format!("{p}.feed_forward.w1.weight"), &[FF, DIM]);
    t.put(format!("{p}.feed_forward.w2.weight"), &[DIM, FF]);
    t.put(format!("{p}.feed_forward.w3.weight"), &[FF, DIM]);
    if adaln {
        t.put(format!("{p}.adaLN_modulation.0.weight"), &[4 * DIM, DIM]);
        t.put(format!("{p}.adaLN_modulation.0.bias"), &[4 * DIM]);
    }
}

/// A conv weight (NCHW, as a diffusers checkpoint stores it) and its bias.
fn conv(t: &mut Tensors, p: &str, o: i32, i: i32, k: i32) {
    t.put(format!("{p}.weight"), &[o, i, k, k]);
    t.put(format!("{p}.bias"), &[o]);
}

fn norm(t: &mut Tensors, p: &str, c: i32) {
    t.put(format!("{p}.weight"), &[c]);
    t.put(format!("{p}.bias"), &[c]);
}

fn resnet(t: &mut Tensors, p: &str, i: i32, o: i32) {
    norm(t, &format!("{p}.norm1"), i);
    conv(t, &format!("{p}.conv1"), o, i, 3);
    norm(t, &format!("{p}.norm2"), o);
    conv(t, &format!("{p}.conv2"), o, o, 3);
    if i != o {
        conv(t, &format!("{p}.conv_shortcut"), o, i, 1);
    }
}

fn mid(t: &mut Tensors, p: &str, c: i32) {
    resnet(t, &format!("{p}.resnets.0"), c, c);
    norm(t, &format!("{p}.attentions.0.group_norm"), c);
    for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
        t.put(format!("{p}.attentions.0.{proj}.weight"), &[c, c]);
        t.put(format!("{p}.attentions.0.{proj}.bias"), &[c]);
    }
    resnet(t, &format!("{p}.resnets.1"), c, c);
}

/// The tiny snapshot: `text_encoder/` (the bounded Qwen3 contract, 3 layers), `transformer/`,
/// `vae/` (the production AutoencoderKL structure at channels 32/32/64/64) and a ControlNet file.
pub(crate) struct Snapshot {
    _dir: tempfile::TempDir,
    pub(crate) root: PathBuf,
    pub(crate) control: PathBuf,
    te_cfg: ZTextEncoderConfig,
    te_contract: mlx_gen::gen_core::EncoderContract,
}

pub(crate) fn snapshot() -> Snapshot {
    snapshot_with_control_storage(Dtype::Bfloat16, Dtype::Bfloat16)
}

fn snapshot_with_control_storage(control_weights: Dtype, control_bias: Dtype) -> Snapshot {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("snapshot");
    // Text encoder: the bounded contract's exact tensor surface, random bf16 payload.
    let te_contract = mlx_gen::gen_core::EncoderContract {
        num_hidden_layers: 3,
        loaded_hidden_layers: 3,
        selected_hidden_layers: &[2],
        ..crate::bounded_encoder_contract()
    };
    let te_dir = root.join("text_encoder");
    gen_core_testkit::write_encoder_contract_fixture(&te_dir, te_contract).unwrap();
    let file = te_dir.join("model.safetensors");
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&file).unwrap();
    let mut te = Tensors::new(100);
    for h in &headers {
        let shape: Vec<i32> = h.shape.iter().map(|&d| d as i32).collect();
        te.put(h.name.clone(), &shape);
    }
    std::fs::remove_file(&file).unwrap();
    te.save(&file);
    let te_cfg = ZTextEncoderConfig {
        vocab_size: te_contract.vocab_size as i32,
        hidden_size: te_contract.hidden_size as i32,
        n_layers: te_contract.num_hidden_layers,
        n_heads: te_contract.num_attention_heads as i32,
        n_kv_heads: te_contract.num_key_value_heads as i32,
        head_dim: te_contract.head_dim as i32,
        intermediate_size: te_contract.intermediate_size as i32,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
    };
    assert_eq!(te_cfg.hidden_size, dit_cfg().cap_feat_dim);

    // DiT (diffusers names; `remap_transformer_keys` aliases the t-embedder and final layer).
    let cfg = dit_cfg();
    let patch = cfg.patch_size * cfg.patch_size * cfg.in_channels;
    let mut t = Tensors::new(1000);
    t.put("all_x_embedder.2-1.weight", &[DIM, patch]);
    t.put("all_x_embedder.2-1.bias", &[DIM]);
    t.put("t_embedder.mlp.0.weight", &[1024, 256]);
    t.put("t_embedder.mlp.0.bias", &[1024]);
    t.put("t_embedder.mlp.2.weight", &[DIM, 1024]);
    t.put("t_embedder.mlp.2.bias", &[DIM]);
    t.put("cap_embedder.0.weight", &[cfg.cap_feat_dim]);
    t.put("cap_embedder.1.weight", &[DIM, cfg.cap_feat_dim]);
    t.put("cap_embedder.1.bias", &[DIM]);
    t.put("x_pad_token", &[1, DIM]);
    t.put("cap_pad_token", &[1, DIM]);
    t.put("all_final_layer.2-1.linear.weight", &[patch, DIM]);
    t.put("all_final_layer.2-1.linear.bias", &[patch]);
    t.put("all_final_layer.2-1.adaLN_modulation.1.weight", &[DIM, DIM]);
    t.put("all_final_layer.2-1.adaLN_modulation.1.bias", &[DIM]);
    for i in 0..cfg.n_refiner_layers {
        block(&mut t, &format!("noise_refiner.{i}"), true);
        block(&mut t, &format!("context_refiner.{i}"), false);
    }
    for i in 0..cfg.n_layers {
        block(&mut t, &format!("layers.{i}"), true);
    }
    t.save(&root.join("transformer/model.safetensors"));

    // ControlNet: the patch embedder, 15 main blocks and the 2-block refiner (block 0 of each with
    // `before_proj`).
    let mut c = Tensors::new(5000);
    let cpatch = cfg.patch_size * cfg.patch_size * crate::CONTROL_IN_DIM;
    c.put("control_all_x_embedder.2-1.weight", &[DIM, cpatch]);
    c.put("control_all_x_embedder.2-1.bias", &[DIM]);
    for (stack, n) in [("control_layers", 15), ("control_noise_refiner", 2)] {
        for i in 0..n {
            let p = format!("{stack}.{i}");
            block(&mut c, &p, true);
            c.put(format!("{p}.after_proj.weight"), &[DIM, DIM]);
            c.put(format!("{p}.after_proj.bias"), &[DIM]);
            if i == 0 {
                c.put(format!("{p}.before_proj.weight"), &[DIM, DIM]);
                c.put(format!("{p}.before_proj.bias"), &[DIM]);
            }
        }
    }
    // Synthetic native conversion proof only: choose actual quantizer leaves by the SAME source
    // target set, not by rank-2 guessing. Dense patch embedder and all norms remain untouched.
    let control_storage = crate::ios_pricing::control_quantization_targets()
        .into_iter()
        .flat_map(|(path, _)| {
            [
                (format!("{path}.weight"), control_weights),
                (format!("{path}.bias"), control_bias),
            ]
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    for (name, array) in &mut c.out {
        if let Some(&dtype) = control_storage.get(name) {
            if array.dtype() != dtype {
                *array = array.as_dtype(dtype).unwrap();
                array.eval().unwrap();
            }
        }
    }
    let control = dir.path().join("control.safetensors");
    c.save(&control);

    // VAE: the production structure (4 down blocks of 2 resnets, 4 up blocks of 3, x8) at tiny
    // channels.
    let channels = [32, 32, 64, 64];
    let mut v = Tensors::new(9000);
    conv(&mut v, "encoder.conv_in", channels[0], 3, 3);
    let mut prev = channels[0];
    for (i, &out) in channels.iter().enumerate() {
        for r in 0..2 {
            resnet(
                &mut v,
                &format!("encoder.down_blocks.{i}.resnets.{r}"),
                if r == 0 { prev } else { out },
                out,
            );
        }
        if i < 3 {
            conv(
                &mut v,
                &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                out,
                out,
                3,
            );
        }
        prev = out;
    }
    mid(&mut v, "encoder.mid_block", prev);
    norm(&mut v, "encoder.conv_norm_out", prev);
    conv(&mut v, "encoder.conv_out", 32, prev, 3);
    conv(&mut v, "decoder.conv_in", prev, 16, 3);
    mid(&mut v, "decoder.mid_block", prev);
    for (i, &out) in channels.iter().rev().enumerate() {
        for r in 0..3 {
            resnet(
                &mut v,
                &format!("decoder.up_blocks.{i}.resnets.{r}"),
                if r == 0 { prev } else { out },
                out,
            );
        }
        if i < 3 {
            conv(
                &mut v,
                &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                out,
                out,
                3,
            );
        }
        prev = out;
    }
    norm(&mut v, "decoder.conv_norm_out", prev);
    conv(&mut v, "decoder.conv_out", 3, prev, 3);
    v.save(&root.join("vae/model.safetensors"));

    Snapshot {
        _dir: dir,
        root,
        control,
        te_cfg,
        te_contract,
    }
}

/// A non-evaluating witness: inspecting a lazy placeholder must fail, never materialize it.
fn storage_witness(array: &Array) -> usize {
    let mut available = false;
    assert_eq!(
        unsafe { mlx_sys::_mlx_array_is_available(&mut available, array.as_ptr()) },
        0
    );
    assert!(
        available,
        "endpoint owns a lazy parameter placeholder, not resident storage"
    );
    let address = unsafe { mlx_sys::mlx_array_data_uint8(array.as_ptr()) } as usize;
    assert_ne!(address, 0);
    address
}

/// The actual capture endpoint load form, not just its selection name.
pub(crate) fn endpoint_spec(root: &Path, bits: Option<i32>, resident: bool) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_owned()))
        .with_offload_policy(if resident {
            OffloadPolicy::Resident
        } else {
            OffloadPolicy::Sequential
        })
        .with_load_shape(if resident {
            LoadShape::EagerMaterialization
        } else {
            LoadShape::DeferredMaterialization
        });
    spec.quantize = bits.map(|b| match b {
        4 => Quant::Q4,
        8 => Quant::Q8,
        _ => panic!("unsupported fixture tier {b}"),
    });
    spec
}

impl Snapshot {
    pub(crate) fn facts_for_spec(&self, control: bool, spec: &LoadSpec) -> Facts {
        let bits = spec.quantize.map(|q| match q {
            Quant::Q4 => 4,
            Quant::Q8 => 8,
            _ => panic!("unsupported fixture tier {q:?}"),
        });
        let mut facts = self.facts(control, bits);
        facts.streamable = spec.load_shape == LoadShape::DeferredMaterialization;
        let WeightsSource::Dir(root) = &spec.weights else {
            panic!("fixture needs a directory")
        };
        facts.dit = Component::read(&WeightsSource::Dir(root.join("transformer"))).unwrap();
        facts
    }

    pub(crate) fn facts(&self, control: bool, quant_bits: Option<i32>) -> Facts {
        Facts {
            te_cfg: self.te_cfg,
            dit_cfg: dit_cfg(),
            te: Component::read(&WeightsSource::Dir(self.root.join("text_encoder"))).unwrap(),
            dit: Component::read(&WeightsSource::Dir(self.root.join("transformer"))).unwrap(),
            vae: Component::read(&WeightsSource::Dir(self.root.join("vae"))).unwrap(),
            control: control
                .then(|| Component::read(&WeightsSource::File(self.control.clone())).unwrap()),
            window_tokens: WINDOW as u32,
            caption_tokens: VALID as u32,
            quant_bits,
            streamable: true,
            load_components: None,
            quant_conversions: std::array::from_fn(|_| Vec::new()),
            te_quant_bits: None,
        }
    }
}

/// The fully optimized execution (staged, the narrowest decode tile, chunked attention, rung 4
/// over both stacks) the proof runs.
fn execution() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        tile_vae_decode: true,
        chunk_attention: true,
        stream_transformer_blocks: true,
        decode_tile_edge: Some(256),
        decode_overlap: Some(64),
        attention_chunk_size: Some(crate::memory_strategy::ATTENTION_CHUNK_SIZE),
        transformer_window_size: Some(1),
        transformer_window_component: Some(TransformerComponent::Both),
        ..Default::default()
    }
}

pub(crate) fn image(edge: u32, seed: u32) -> mlx_gen::Image {
    mlx_gen::Image {
        width: edge,
        height: edge,
        pixels: (0..edge * edge * 3)
            .map(|i| ((i * 37 + seed) % 251) as u8)
            .collect(),
    }
}

/// Actual private fixture program points, not planned price windows or Data/backing ownership.
/// The booleans report only Rust model-handle presence; absence is never a physical release claim.
fn fixture_phase(phase: IosPhase, te: bool, dit: bool, vae: bool) {
    let phase = match phase {
        IosPhase::Load => 0, IosPhase::Conditioning => 1, IosPhase::Encode => 2,
        IosPhase::Denoise => 3, IosPhase::Decode => 4,
    };
    mlx_gen::ios_quantum::diagnostic_metadata("z_image_fixture", "phase", &[
        ("schema", 1), ("phase", phase),
        ("te_rust_handle_present", u64::from(te)),
        ("dit_rust_handle_present", u64::from(dit)),
        ("vae_rust_handle_present", u64::from(vae)),
    ]);
}

/// Run one request's production phases on the snapshot, in the staged generators' order
/// (`model::ZImageTurbo::generate_impl` / `model_control::ZImageTurboControl::generate_impl`
/// over `Residency::run_staged_request_scoped`), returning every checked `(label, scratch)`.
fn run(snap: &Snapshot, spec: &LoadSpec, route: &Route) -> Vec<(String, u64)> {
    let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let sink = events.clone();
    with_cpu_diagnostic_headroom(AMPLE, || {
        with_cpu_diagnostic_observer(
            move |e| {
                if let QuantumEvent::Checked { label, scratch } = e {
                    sink.borrow_mut().push((label.clone(), *scratch));
                }
            },
            || run_request(snap, spec, route, &image(route.width, 1)),
        )
    })
    .unwrap();
    let events = events.borrow().clone();
    let at = |label: &str| {
        events
            .iter()
            .position(|(l, _)| l == label)
            .unwrap_or_else(|| panic!("missing actual lifecycle quantum {label}"))
    };
    for label in [
        crate::quantum::TE_LOAD,
        crate::quantum::DIT_LOAD,
        crate::quantum::VAE_LOAD,
    ] {
        assert_eq!(
            events.iter().filter(|(l, _)| l == label).count(),
            1,
            "component reload: {label}"
        );
    }
    assert!(at(crate::quantum::TE_LOAD) < at(crate::quantum::TE_EMBED));
    if spec.load_shape == LoadShape::EagerMaterialization {
        assert!(at(crate::quantum::DIT_LOAD) < at(crate::quantum::TE_EMBED));
        assert!(at(crate::quantum::VAE_LOAD) < at(crate::quantum::TE_EMBED));
        if route.mode.control() {
            assert!(at(crate::quantum::CONTROL_LOAD) < at(crate::quantum::TE_EMBED));
        }
    } else {
        assert!(at(crate::quantum::TE_EMBED) < at(crate::quantum::DIT_LOAD));
    }
    events
}

/// One request's production phases on the snapshot at `root` (the dense snapshot, or a packed copy
/// whose DiT is stored at `quant_bits`), fresh loads included, in the generators' order: under
/// `stage_residency` (`Residency::run_staged_request_scoped`) the text encoder loads, encodes and
/// drops before the heavy components load, and the decode runs after the DiT is shed; without it
/// every component loads up front and stays resident to the end. A quantized load replays the
/// generators' load-time `quantize` (`model::load_heavy` / `model_control::load_control_heavy`: a
/// no-op on the packed base, packing the dense ControlNet). `control_image` is the control route's
/// hint image (the reference init is `image(edge, 2)`).
pub(crate) fn run_request(
    snap: &Snapshot,
    spec: &LoadSpec,
    route: &Route,
    control_image: &mlx_gen::Image,
) -> mlx_gen::Result<()> {
    let WeightsSource::Dir(root) = &spec.weights else {
        panic!("fixture needs a directory")
    };
    let streamable = spec.load_shape == LoadShape::DeferredMaterialization;
    let quant_bits = spec.quantize.map(|q| match q {
        Quant::Q4 => 4,
        Quant::Q8 => 8,
        _ => panic!("unsupported fixture tier {q:?}"),
    });
    if spec.offload_policy == OffloadPolicy::Resident && !streamable {
        assert!(
            !route.execution.stage_residency,
            "literal resident cannot stage components"
        );
        assert!(
            !route.execution.stream_transformer_blocks,
            "literal resident cannot run layer windows"
        );
    }
    let id = crate::model::MODEL_ID;
    let cfg = dit_cfg();
    let mode = route.mode;
    let edge = route.width;
    let req = route.request();
    let staged = route.execution.stage_residency;
    let cancel = CancelFlag::default();

    let source = snap
        .te_contract
        .validate_source(&WeightsSource::Dir(snap.root.join("text_encoder")))
        .unwrap();
    // Actual boundary after source validation and BEFORE the first fixture model is loaded.
    // No pricing replay or allocator lower/cancellation claim is attached to these source facts.
    mlx_gen::ios_quantum::diagnostic_metadata("z_image_fixture", "before_model_load", &[
        ("schema", 1), ("width", u64::from(route.width)), ("height", u64::from(route.height)),
        ("serial_count", u64::from(route.count)), ("control", u64::from(mode.control())),
        ("reference", u64::from(mode.img2img())), ("streamable", u64::from(streamable)),
        ("staged", u64::from(staged)), ("requested_bits", quant_bits.unwrap_or(0) as u64),
        ("te_rust_handle_present", 0), ("dit_rust_handle_present", 0), ("vae_rust_handle_present", 0),
    ]);
    let te = if streamable {
        crate::loader::load_validated_text_encoder_streamable_with(&source, &snap.te_cfg)?
    } else {
        crate::loader::load_validated_text_encoder_with(&source, &snap.te_cfg)?
    };
    assert_eq!(
        te.is_streamable(),
        streamable,
        "TE ownership must match the actual load form"
    );
    let te_storage = if streamable {
        None
    } else {
        let witnesses = te.endpoint_storage_witnesses();
        assert_eq!(witnesses.len(), snap.te_cfg.n_layers);
        Some(
            witnesses
                .into_iter()
                .map(storage_witness)
                .collect::<Vec<_>>(),
        )
    };
    enum Heavy {
        Base(Box<crate::ZImageTransformer>),
        Control(Box<crate::ZImageControlTransformer>),
    }
    impl Heavy {
        fn storage(&self) -> Vec<usize> {
            let (base, control) = match self {
                Self::Base(dit) => (dit.as_ref(), None),
                Self::Control(dit) => (&dit.base, Some(&dit.control_layers)),
            };
            let mut addresses = vec![storage_witness(&base.cap_norm_w)];
            addresses.extend(
                base.layers
                    .iter()
                    .map(|b| storage_witness(b.endpoint_storage_witness())),
            );
            if let Some(blocks) = control {
                assert!(!blocks.is_empty());
                addresses.extend(
                    blocks
                        .iter()
                        .map(|b| storage_witness(b.base.endpoint_storage_witness())),
                );
            }
            addresses
        }
    }
    let load_heavy = || -> mlx_gen::Result<(Heavy, crate::vae::Vae)> {
        let dit = if mode.control() {
            let mut dit = crate::loader::load_control_transformer_with_stream_cfg(
                root,
                &WeightsSource::File(snap.control.clone()),
                cfg.clone(),
                streamable,
                streamable,
            )?;
            assert_eq!(
                dit.can_stream_blocks(),
                streamable,
                "ControlNet stream ownership"
            );
            if let Some(bits) = quant_bits {
                dit.quantize(bits)?;
            }
            Heavy::Control(Box::new(dit))
        } else {
            let mut dit = crate::loader::load_transformer_with_stream_cfg(
                root,
                cfg.clone(),
                streamable,
                streamable,
            )?;
            assert_eq!(dit.can_stream_blocks(), streamable, "DiT stream ownership");
            assert_eq!(
                dit.layers.len(),
                cfg.n_layers,
                "eager/stream-armed model inventory"
            );
            if let Some(bits) = quant_bits {
                dit.quantize(bits)?;
            }
            Heavy::Base(Box::new(dit))
        };
        let mut vae = crate::loader::load_vae(root)?;
        if let Some(bits) = quant_bits {
            vae.quantize(bits)?;
        }
        Ok((dit, vae))
    };
    // Unstaged: the heavy components load beside the encoder, before conditioning.
    let mut heavy = if staged { None } else { Some(load_heavy()?) };
    let heavy_storage = (!streamable).then(|| {
        heavy
            .as_ref()
            .expect("eager heavy must already be owned")
            .0
            .storage()
    });

    // Phase A: the streamable text encoder's prompt encode (then, staged, its drop).
    let ids: Vec<i32> = (0..WINDOW).map(|i| 3 + i * 7).collect();
    let mask: Vec<i32> = (0..WINDOW).map(|i| i32::from(i < VALID)).collect();
    let window = crate::pipeline::EncoderWindow::resolve(&req, te.is_streamable(), id)?;
    fixture_phase(IosPhase::Conditioning, true, heavy.is_some(), heavy.is_some());
    let cap = crate::pipeline::encode_tokens(
        &te,
        &Array::from_slice(&ids, &[1, WINDOW]),
        &Array::from_slice(&mask, &[1, WINDOW]),
        VALID,
        window,
    )?;
    let cap = if mode.img2img() {
        cap
    } else {
        crate::pipeline::caption_cast_quantum(&cap)?
    };
    cap.eval()?;
    let te = if staged {
        drop(te);
        None
    } else {
        Some(te)
    };

    // Phase B: the heavy components, the encodes, the denoise batch.
    if heavy.is_none() {
        fixture_phase(IosPhase::Load, te.is_some(), false, false);
        heavy = Some(load_heavy()?);
    }
    let (dit, vae) = heavy.take().unwrap();
    let block_window = crate::pipeline::resolve_block_window(&req, streamable, id)?;
    if !streamable {
        assert!(
            block_window.is_none(),
            "eager DiT must use its owned layers, not a full-cover window"
        );
        assert!(te.is_some(), "resident TE must survive conditioning");
    }
    let budget = crate::pipeline::attention_budget(&req);
    let steps = 3;
    let start = if mode.img2img() { 1 } else { 0 };
    let scheduler = FlowMatchEuler::for_static_shift(steps, 3.0);
    if mode.control() || mode.img2img() {
        fixture_phase(IosPhase::Encode, te.is_some(), true, true);
    }
    let cc = match &dit {
        Heavy::Control(_) => Some(crate::pipeline::encode_control_context(
            &vae,
            control_image,
            edge,
            edge,
        )?),
        Heavy::Base(_) => None,
    };
    let clean = if mode.img2img() {
        Some(crate::pipeline::encode_init_latents(
            &vae,
            &image(edge, 2),
            edge,
            edge,
        )?)
    } else {
        None
    };
    fixture_phase(IosPhase::Denoise, te.is_some(), true, true);
    let latents = crate::pipeline::denoise_batch(
        &scheduler,
        clean.as_ref(),
        start,
        7,
        &req,
        &mut |_| {},
        |latents, seed, op| match &dit {
            Heavy::Control(dit) => crate::pipeline::denoise_control_with_progress(
                dit,
                &scheduler,
                None,
                seed,
                latents,
                &cap,
                cc.as_ref().unwrap(),
                0.8,
                start,
                budget,
                block_window,
                &cancel,
                op,
            ),
            Heavy::Base(dit) => crate::pipeline::denoise_with_progress(
                dit,
                &scheduler,
                None,
                seed,
                latents,
                &cap,
                start,
                budget,
                block_window,
                &cancel,
                op,
            ),
        },
    )?;
    // The denoise phase's locals (the control context, the img2img init latents) end with it.
    drop((cc, clean));
    // Staged: the DiT (and ControlNet) shed before the decode.
    let dit = if staged {
        drop(dit);
        None
    } else {
        Some(dit)
    };

    // Phase C: eager production ownership retains the actual TE and heavy objects, not copies.
    if !streamable && !staged {
        assert!(
            te.is_some() && dit.is_some(),
            "resident components must survive through decode"
        );
        assert_eq!(
            te.as_ref()
                .unwrap()
                .endpoint_storage_witnesses()
                .into_iter()
                .map(storage_witness)
                .collect::<Vec<_>>(),
            te_storage.unwrap(),
            "the same materialized TE layer storage must survive to decode"
        );
        assert_eq!(
            dit.as_ref().unwrap().storage(),
            heavy_storage.unwrap(),
            "the same materialized DiT/ControlNet layer storage must survive to decode"
        );
    }
    fixture_phase(IosPhase::Decode, te.is_some(), dit.is_some(), true);
    let tiling = crate::pipeline::decode_tiling(&req, staged);
    crate::pipeline::decode_batch(&vae, tiling.as_ref(), latents, &cancel, &mut |_| {})?;
    drop((te, dit, vae));
    Ok(())
}

fn largest_by_label<'a>(quanta: impl Iterator<Item = (&'a str, u64)>) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for (label, scratch) in quanta {
        let e = out.entry(label.to_string()).or_insert(0);
        *e = (*e).max(scratch);
    }
    out
}

fn largest_by_phase(by_label: &BTreeMap<String, u64>) -> BTreeMap<IosPhase, (u64, String)> {
    let mut out: BTreeMap<IosPhase, (u64, String)> = BTreeMap::new();
    for (label, &scratch) in by_label {
        let phase = ios_quantum_phase(label).unwrap_or_else(|| panic!("unmapped label {label}"));
        let e = out.entry(phase).or_insert((0, String::new()));
        if scratch > e.0 {
            *e = (scratch, label.clone());
        }
    }
    out
}

/// The proof for one mode: every observed label is priced, every priced label observed, each at the
/// same largest scratch; per phase the largest observed scratch equals the largest priced one (same
/// label); and the folded pricing's binding quantum per phase was checked at its priced scratch.
fn assert_priced_exactly(mode: Mode, edge: u32) {
    let snap = snapshot();
    for resident in [false, true] {
        let spec = endpoint_spec(&snap.root, None, resident);
        let route = Route {
            mode,
            width: edge,
            height: edge,
            count: 1,
            execution: if resident {
                GenerationMemory::default()
            } else {
                execution()
            },
        };
        let observed = run(&snap, &spec, &route);
        let priced: Vec<Priced> = Model::new(snap.facts_for_spec(mode.control(), &spec))
            .unwrap()
            .quanta(&route, Backend::Cpu)
            .unwrap();
        let seen = largest_by_label(observed.iter().map(|(l, s)| (l.as_str(), *s)));
        let want = largest_by_label(priced.iter().map(|p| (p.label, p.scratch)));
        eprintln!(
            "{mode:?}: {} checks over {} labels: {seen:?}",
            observed.len(),
            seen.len()
        );
        assert_eq!(seen, want, "{mode:?}: observed vs priced, per label");
        assert_eq!(
            largest_by_phase(&seen),
            largest_by_phase(&want),
            "{mode:?}: largest scratch per phase"
        );
        let pricing = fold(&priced);
        for p in &pricing.phases {
            assert!(
                observed
                    .iter()
                    .any(|(l, s)| *l == p.binding && *s == p.scratch_bytes),
                "{mode:?} {:?}: binding {} at {} B was never checked at that scratch",
                p.phase,
                p.binding,
                p.scratch_bytes
            );
        }
        let phases: Vec<IosPhase> = pricing.phases.iter().map(|p| p.phase).collect();
        let mut expected = vec![
            IosPhase::Load,
            IosPhase::Conditioning,
            IosPhase::Denoise,
            IosPhase::Decode,
        ];
        if mode != Mode::Text {
            expected.insert(2, IosPhase::Encode);
        }
        assert_eq!(phases, expected, "{mode:?}: the phases the request runs");
    }
}

#[test]
fn text_pricing_equals_the_observed_quanta() {
    if !in_cpu_child("text_pricing_equals_the_observed_quanta") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    assert_priced_exactly(Mode::Text, EDGE);
    let _ = parent_completion.replace(CpuChildCompletion);
}

#[test]
fn reference_pricing_equals_the_observed_quanta() {
    if !in_cpu_child("reference_pricing_equals_the_observed_quanta") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    assert_priced_exactly(Mode::Reference, EDGE);
    let _ = parent_completion.replace(CpuChildCompletion);
}

/// The whole-image encode and the untiled decode (an image within both tiles).
#[test]
fn untiled_reference_pricing_equals_the_observed_quanta() {
    if !in_cpu_child("untiled_reference_pricing_equals_the_observed_quanta") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    assert_priced_exactly(Mode::Reference, SMALL_EDGE);
    let _ = parent_completion.replace(CpuChildCompletion);
}

#[test]
fn control_pricing_equals_the_observed_quanta() {
    if !in_cpu_child("control_pricing_equals_the_observed_quanta") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    assert_priced_exactly(Mode::Control, EDGE);
    let _ = parent_completion.replace(CpuChildCompletion);
}

#[test]
fn control_reference_pricing_equals_the_observed_quanta() {
    if !in_cpu_child("control_reference_pricing_equals_the_observed_quanta") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    assert_priced_exactly(Mode::ControlReference, EDGE);
    let _ = parent_completion.replace(CpuChildCompletion);
}

/// Actual tiny native quantization + the production Control block guard, not a metadata-only
/// success and not an allocator/process certificate. No full image, sampler or measured grid runs.
#[test]
fn control_postpack_and_replayed_metadata_match_actual_native_quanta() {
    if !in_cpu_child("control_postpack_and_replayed_metadata_match_actual_native_quanta") {
        return;
    }
    // Bound first and armed only after the actual body/scopes/CPU synchronization complete.
    let mut parent_completion: Option<CpuChildCompletion> = None;
    let normal_end = std::cell::Cell::new(false);
    let completed_mechanisms = std::cell::Cell::new(0_u8);
    with_cpu_diagnostic_headroom(AMPLE, || {
        let _scope = mlx_gen::ios_quantum::Scope::try_new().unwrap();
        let _glue = crate::CompileGlueGuard::enable();
        for bits in [4, 8] {
            for streamed in [false, true] {
                let snap = snapshot_with_control_storage(Dtype::Float16, Dtype::Float32);
                let mut facts = snap.facts(true, Some(bits));
                facts.streamable = streamed;
                let model = Model::new(facts).unwrap();
                let route = Route {
                    mode: Mode::Control,
                    width: 16,
                    height: 16,
                    count: 1,
                    execution: if streamed {
                        execution()
                    } else {
                        GenerationMemory::default()
                    },
                };
                let priced = model.quanta(&route, Backend::Cpu).unwrap();
                let expected_pack = priced
                    .iter()
                    .filter(|p| p.label == crate::quantum::CONTROL_QUANTIZE)
                    .map(|p| p.scratch)
                    .collect::<Vec<_>>();
                let forward = priced
                    .iter()
                    .filter(|p| p.label == crate::quantum::CONTROL_BLOCK)
                    .map(|p| p.scratch)
                    .collect::<Vec<_>>();
                let expected_forward = vec![
                    forward[0],
                    forward[crate::control_transformer::CONTROL_REFINER_PLACES.len()],
                    forward[crate::control_transformer::CONTROL_REFINER_PLACES.len()],
                ];
                let mut actual = crate::loader::load_control_transformer_with_stream_cfg(
                    &snap.root,
                    &WeightsSource::File(snap.control.clone()),
                    dit_cfg(),
                    streamed,
                    streamed,
                )
                .unwrap();
                let pack = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
                let sink = pack.clone();
                with_cpu_diagnostic_observer(
                    move |event| {
                        if let QuantumEvent::Checked { label, scratch } = event {
                            if label == crate::quantum::CONTROL_QUANTIZE {
                                sink.borrow_mut().push(*scratch);
                            }
                        }
                    },
                    || actual.quantize(bits),
                )
                .unwrap();
                let is_packed = |block: &crate::control_transformer_block::ZImageControlBlock| {
                    block
                        .projections()
                        .iter()
                        .all(|linear| linear.is_quantized())
                };
                // Deferred resident shells are NOT packed by load. A local streamed forward uses a
                // separately reconstructed/packed graph, even on the second callback.
                let shell_packed = is_packed(&actual.control_layers[0]);
                let refiner_packed = is_packed(&actual.control_noise_refiner()[0]);
                let raw = Component::read(&WeightsSource::File(snap.control.clone())).unwrap();
                let projected_tensors = crate::ios_pricing::control_quantization_targets()
                    .into_iter()
                    .flat_map(|(path, _)| {
                        ["weight", "bias", "scales", "biases"]
                            .map(|suffix| format!("{path}.{suffix}"))
                    })
                    .collect::<std::collections::BTreeSet<_>>();
                let mut native_logical_resident = raw
                    .headers
                    .iter()
                    .filter(|h| h.name.starts_with("control_all_x_embedder."))
                    .map(|h| mlx_gen::ios_quantum::allocated_bytes(h.data_bytes))
                    .sum::<u64>();
                for (stack, blocks) in [
                    (
                        "control_layers",
                        if streamed {
                            &[][..]
                        } else {
                            actual.control_layers.as_slice()
                        },
                    ),
                    ("control_noise_refiner", actual.control_noise_refiner()),
                ] {
                    for (index, block) in blocks.iter().enumerate() {
                        let prefix = format!("{stack}.{index}.");
                        native_logical_resident += raw
                            .headers
                            .iter()
                            .filter(|h| {
                                h.name.starts_with(&prefix) && !projected_tensors.contains(&h.name)
                            })
                            .map(|h| mlx_gen::ios_quantum::allocated_bytes(h.data_bytes))
                            .sum::<u64>();
                        native_logical_resident += block
                            .weight_arrays()
                            .iter()
                            .map(|a| mlx_gen::ios_quantum::allocated_bytes(a.nbytes() as u64))
                            .sum::<u64>();
                    }
                }
                let plan = mlx_gen::attention::AttentionPlan::budgeted(
                    crate::pipeline::attention_budget_for(Some(route.execution)),
                );
                let make = |shape: &[i32]| {
                    let array = Array::ones::<f32>(shape).unwrap();
                    array.eval().unwrap();
                    array
                };
                let time = make(&[1, DIM]);
                let c32 = make(&[1, 32, DIM]);
                let f32 = make(&[32, DIM / 4 / 2, 2]);
                let c64 = make(&[1, 64, DIM]);
                let f64 = make(&[64, DIM / 4 / 2, 2]);
                let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
                let sink = seen.clone();
                let mut replay_packed = Vec::new();
                with_cpu_diagnostic_observer(
                    move |event| {
                        if let QuantumEvent::Checked { label, scratch } = event {
                            if label == crate::quantum::CONTROL_BLOCK {
                                sink.borrow_mut().push(*scratch);
                            }
                        }
                    },
                    || -> mlx_gen::Result<()> {
                        let _ = crate::control_transformer::control_block_quantum(
                            &actual.control_noise_refiner()[0],
                            0,
                            c32.clone(),
                            &c32,
                            &f32,
                            &time,
                            plan,
                            0,
                        )?;
                        if streamed {
                            let mut stream = crate::block_stream::ZImageBlockStream::new(
                                WeightsSource::File(snap.control.clone()),
                                "control_layers",
                                dit_cfg().block_cfg(),
                                crate::control_transformer::CONTROL_LAYERS_PLACES.len(),
                                true,
                            );
                            stream.set_quant_bits(bits);
                            for _ in 0..2 {
                                let mut view = stream.open()?;
                                let (block, read) = stream.materialize_control(&mut view, 0)?;
                                replay_packed.push(is_packed(&block));
                                let _ = crate::control_transformer::control_block_quantum(
                                    &block,
                                    0,
                                    c64.clone(),
                                    &c64,
                                    &f64,
                                    &time,
                                    plan,
                                    read,
                                )?;
                                // ONE local block/view drops before the next replay; no Vec of weights.
                                drop((block, view));
                            }
                        } else {
                            for _ in 0..2 {
                                replay_packed.push(is_packed(&actual.control_layers[0]));
                                let _ = crate::control_transformer::control_block_quantum(
                                    &actual.control_layers[0],
                                    0,
                                    c64.clone(),
                                    &c64,
                                    &f64,
                                    &time,
                                    plan,
                                    0,
                                )?;
                            }
                        }
                        Ok(())
                    },
                )
                .unwrap();
                // Scope/model/native roots stay alive through the assertion. These are logical
                // representation/guard bytes, NOT unique backing or last-shared-owner measurements.
                assert_eq!(
                    (
                        model.control_forward_is_packed(),
                        shell_packed,
                        refiner_packed,
                        replay_packed,
                        pack.borrow().clone(),
                        seen.borrow().clone(),
                        model.control_resident()
                    ),
                    (
                        Some((true, true)),
                        !streamed,
                        true,
                        vec![true, true],
                        expected_pack,
                        expected_forward,
                        native_logical_resident
                    ),
                );
                let mechanism = 2 * u8::from(bits == 8) + u8::from(streamed);
                completed_mechanisms.set(completed_mechanisms.get() | (1 << mechanism));
                eprintln!("ZIMAGE_CONTROL_MECHANISM_COMPLETED_V1 bits={bits} streamed={streamed}");
            }
        }
        // Final statement of the ACTUAL diagnostic body: a skipped helper/early body return
        // cannot arm this. Closure-local scopes/fixtures/native models drop as the body returns.
        normal_end.set(true);
        Ok(())
    })
    .unwrap();
    assert!(
        normal_end.get(),
        "actual Control diagnostic body did not reach its final statement"
    );
    assert_eq!(
        completed_mechanisms.get(),
        0b1111,
        "missing Control quantization mechanism"
    );
    let sync = unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) };
    assert_eq!(
        sync, 0,
        "native CPU synchronization after fixture scope drop failed"
    );
    // This is NORMAL TEST BODY completion only, never last native shared-owner/process release.
    eprintln!(
        "ZIMAGE_CONTROL_NATIVE_BODY_COMPLETED_V1 mechanisms=4 fixture_owners_dropped=1 cpu_sync=0"
    );
    let _ = parent_completion.replace(CpuChildCompletion);
}

/// Every quantum label constant of the crate (`quantum.rs`) and the shared labels a Z-Image request
/// emits map to a phase, and a label never names two phases (the VAE's encoder and decoder carry
/// distinct labels).
#[test]
fn every_quantum_label_maps_to_one_phase() {
    let source = include_str!("quantum.rs");
    let mut labels = Vec::new();
    for line in source.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("pub(crate) const ") {
            if let Some((_, value)) = rest.split_once(": &str = \"") {
                labels.push(value.trim_end_matches("\";").to_string());
            }
        }
    }
    assert!(labels.len() >= 30, "parsed {} labels", labels.len());
    labels.push(mlx_gen::sampler::SAMPLER_STEP_QUANTUM.to_string());
    labels.push(mlx_gen::adapters::loader::ADAPTER_LOAD_QUANTUM.to_string());
    for label in &labels {
        assert!(
            ios_quantum_phase(label).is_some(),
            "label {label:?} maps to no phase"
        );
    }
    let unique: std::collections::BTreeSet<_> = labels.iter().collect();
    assert_eq!(unique.len(), labels.len(), "a label string is shared");
    for (encode, decode) in [(crate::quantum::ENCODE_LABELS, crate::quantum::DECODE_LABELS)] {
        for (e, d) in [
            (encode.resnet, decode.resnet),
            (encode.attention, decode.attention),
            (encode.tile, decode.tile),
            (encode.stage, decode.stage),
        ] {
            assert_eq!(ios_quantum_phase(e), Some(IosPhase::Encode), "{e}");
            assert_eq!(ios_quantum_phase(d), Some(IosPhase::Decode), "{d}");
        }
    }
}

/// [`crate::ios_pricing::tile_extent`] against the tiles `mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc`
/// actually cuts (its preprocess hook sees each halo-expanded tile).
#[test]
fn tile_extent_is_the_largest_tile_the_tiler_cuts() {
    if !in_cpu_child("tile_extent_is_the_largest_tile_the_tiler_cuts") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    let weight = rand(&[2, 3, 3, 2], 1, 0.1);
    for (h, w, edge) in [
        (16, 16, 16),
        (17, 16, 16),
        (30, 9, 8),
        (64, 33, 12),
        (5, 40, 3),
    ] {
        let x = rand(&[1, h, w, 2], 2, 1.0);
        let largest = std::cell::Cell::new((0, 0));
        mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc(&x, &weight, None, edge, None, |tile| {
            let s = tile.shape();
            let (a, b) = largest.get();
            largest.set((a.max(s[1]), b.max(s[2])));
            Ok(tile.clone())
        })
        .unwrap();
        assert_eq!(
            crate::ios_pricing::tile_extent(h, w, edge),
            largest.get(),
            "{h}x{w} at {edge}"
        );
    }
    let _ = parent_completion.replace(CpuChildCompletion);
}

/// The tiny DiT stored PACKED at `bits` (a published Q4/Q8 turnkey's form: every Linear as its
/// `weight` codes + `scales` + `biases`, norms / pad tokens / biases dense) under `root`.
pub(crate) fn packed_transformer(snap: &Snapshot, root: &Path, bits: i32) {
    let dense =
        mlx_gen::weights::Weights::from_file(snap.root.join("transformer/model.safetensors"))
            .unwrap();
    let mut out = Vec::new();
    let mut keys: Vec<String> = dense.keys().map(String::from).collect();
    keys.sort();
    for key in keys {
        let a = dense.require(&key).unwrap().clone();
        match key.strip_suffix(".weight") {
            Some(base) if a.ndim() == 2 => {
                let (wq, scales, biases) = mlx_rs::ops::quantize(&a, 64, bits).unwrap();
                out.push((format!("{base}.weight"), wq));
                out.push((format!("{base}.scales"), scales));
                out.push((format!("{base}.biases"), biases));
            }
            _ => out.push((key, a)),
        }
    }
    let path = root.join("transformer/model.safetensors");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    Array::save_safetensors(
        out.iter().map(|(k, a)| (k.as_str(), a)).collect::<Vec<_>>(),
        None,
        &path,
    )
    .unwrap();
}

/// The deferred ControlNet lever: a deferred-materialization load of a quantized control route
/// (packed base, dense ControlNet) packs only its resident ControlNet blocks (the 2-block refiner) —
/// its 15 streamed main blocks stay lazy and are packed per window by the stream. Check each load's
/// positive MLX active-memory counter delta against `dit_resident + control_resident`: the formula
/// must not underprice that delta; the ratio (including the 1.5x reference) is diagnostic only.
/// This counter is not a Darwin physical-footprint or last-native-owner release witness.
/// Parity: the deferred load's windowed forward equals the eager load's resident forward bit for
/// bit, and a non-windowed forward on the deferred load is refused rather than run dense.
#[test]
fn a_deferred_control_load_packs_only_its_resident_blocks() {
    if !in_cpu_child("a_deferred_control_load_packs_only_its_resident_blocks") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    let snap = snapshot();
    let bits = 4;
    let packed = tempfile::tempdir().unwrap();
    packed_transformer(&snap, packed.path(), bits);
    let control = WeightsSource::File(snap.control.clone());
    let load = |streamed: bool| -> (crate::ZImageControlTransformer, u64) {
        mlx_rs::memory::clear_cache();
        let before = mlx_rs::memory::get_active_memory() as u64;
        let mut t = crate::loader::load_control_transformer_with_stream_cfg(
            packed.path(),
            &control,
            dit_cfg(),
            true,
            streamed,
        )
        .unwrap();
        // Through the iOS quantum path (one bounded quantum per packed block).
        with_cpu_diagnostic_headroom(AMPLE, || t.quantize(bits)).unwrap();
        mlx_rs::memory::clear_cache();
        let after = mlx_rs::memory::get_active_memory() as u64;
        (t, after.saturating_sub(before))
    };
    let resident = |streamable: bool| {
        let mut facts = snap.facts(true, Some(bits));
        facts.dit =
            Component::read(&WeightsSource::Dir(packed.path().join("transformer"))).unwrap();
        facts.streamable = streamable;
        let model = Model::new(facts).unwrap();
        model.dit_resident() + model.control_resident()
    };
    let (eager, eager_bytes) = load(false);
    let (deferred, deferred_bytes) = load(true);
    let (eager_formula, deferred_formula) = (resident(false), resident(true));
    eprintln!(
        "control residency: eager measured {eager_bytes} B priced {eager_formula} B; deferred \
         measured {deferred_bytes} B priced {deferred_formula} B"
    );
    for (label, formula, measured) in [
        ("eager control load residency", eager_formula, eager_bytes),
        ("deferred control load residency", deferred_formula, deferred_bytes),
    ] {
        assert!(
            measured > 0,
            "ZIMAGE_CONTROL_ACTIVE_COUNTER_PRIMITIVE_VALIDITY_V1 {label}: positive MLX active-memory delta required, observed {measured} B"
        );
        assert!(
            formula >= measured,
            "ZIMAGE_CONTROL_RESIDENT_FORMULA_BELOW_ACTIVE_DELTA_V1 {label}: formula {formula} B is BELOW the measured MLX active-memory delta {measured} B"
        );
        let ratio = if measured == 0 {
            None
        } else {
            Some(formula as f64 / measured as f64)
        };
        eprintln!(
            "ZIMAGE_CONTROL_RESIDENT_RATIO_REPORT_V1 {label}: formula={formula} active_delta={measured} ratio={ratio:?}; 1.5x is report-only"
        );
    }

    // Parity: the same control forward, resident vs windowed.
    let w = 16;
    let x = rand(&[16, 1, w, w], 7, 1.0)
        .as_dtype(Dtype::Float32)
        .unwrap();
    let cap = rand(&[VALID, dit_cfg().cap_feat_dim], 8, 1.0)
        .as_dtype(Dtype::Float32)
        .unwrap();
    let cc = rand(&[crate::CONTROL_IN_DIM, 1, w, w], 9, 1.0)
        .as_dtype(Dtype::Float32)
        .unwrap();
    let plan = mlx_gen::attention::AttentionPlan::UNBOUNDED;
    let cancel = CancelFlag::default();
    let resident_out = eager
        .forward_budgeted(&x, 0.5, &cap, Some(&cc), 0.8, plan)
        .unwrap();
    let windowed_out = deferred
        .forward_windowed(&x, 0.5, &cap, Some(&cc), 0.8, plan, 1, &cancel)
        .unwrap();
    let bits_of = |a: &Array| {
        let f = a.as_dtype(Dtype::Float32).unwrap();
        f.eval().unwrap();
        f.as_slice::<f32>()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits_of(&resident_out), bits_of(&windowed_out));
    assert!(deferred
        .forward_budgeted(&x, 0.5, &cap, Some(&cc), 0.8, plan)
        .is_err());
    let _ = parent_completion.replace(CpuChildCompletion);
}

/// Actual native leaf construction/packing, with an independently retained raw dictionary. This
/// does not turn raw ignored headers into module parameters or claim last native owner release.
#[test]
fn constructor_bias_ownership_matches_actual_native_packing() {
    if !in_cpu_child("constructor_bias_ownership_matches_actual_native_packing") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    let normal_end = std::cell::Cell::new(false);
    let completed = std::cell::Cell::new(0_u8);
    with_cpu_diagnostic_headroom(AMPLE, || {
        let _scope = mlx_gen::ios_quantum::Scope::try_new()?;
        for bits in [4, 8] {
            for (path, has_bias, expected_bias) in [
                (
                    "layers.0.attention.to_q",
                    crate::quant::DIT_ATTENTION_HAS_BIAS,
                    false,
                ),
                ("decoder.mid_block.attentions.0.to_q", true, true),
            ] {
                // Real evaluated native arrays, not a shape-carrier model or fabricated backing.
                let weight = Array::ones::<f32>(&[4, 64])?.as_dtype(Dtype::Float16)?;
                let bias = Array::ones::<f32>(&[4])?;
                mlx_rs::transforms::eval([&weight, &bias])?;
                let raw = mlx_gen::weights::Weights::from_map(
                    [
                        (format!("{path}.weight"), weight),
                        (format!("{path}.bias"), bias),
                    ]
                    .into_iter()
                    .collect(),
                );
                let mut linear = crate::quant::lin(&raw, path, has_bias)?;
                raw.materialize_accessed()?;
                let dense_bias = linear.bias().map(|b| b.dtype());
                linear.quantize(bits, None)?;
                linear.materialize_weights()?;
                let (codes, scales, offsets, _, group, packed_bits) =
                    linear.quantized_params().expect("actual packed native leaf");
                assert_eq!(
                    (
                        dense_bias,
                        linear.bias().map(|b| b.dtype()),
                        raw.require(&format!("{path}.bias"))?.dtype(),
                        raw.require(&format!("{path}.weight"))?.dtype(),
                        codes.shape().to_vec(),
                        scales.dtype(),
                        offsets.dtype(),
                        group,
                        packed_bits,
                    ),
                    (
                        expected_bias.then_some(Dtype::Float32),
                        expected_bias.then_some(Dtype::Bfloat16),
                        Dtype::Float32,
                        Dtype::Float16,
                        vec![4, 64 * bits / 32],
                        Dtype::Bfloat16,
                        Dtype::Bfloat16,
                        64,
                        bits,
                    ),
                    "actual constructor and raw dictionary differ for {path} Q{bits}",
                );
                let mechanism = 2 * u8::from(bits == 8) + u8::from(expected_bias);
                completed.set(completed.get() | (1 << mechanism));
                eprintln!(
                    "ZIMAGE_BIAS_NATIVE_MECHANISM_V1 bits={bits} module_bias={expected_bias}"
                );
            }
        }
        normal_end.set(true);
        Ok(())
    })
    .unwrap();
    assert!(normal_end.get(), "native bias diagnostic body did not complete");
    assert_eq!(completed.get(), 0b1111, "missing native bias mechanism");
    let sync = unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) };
    assert_eq!(sync, 0, "native bias CPU synchronization after scope drop failed");
    eprintln!(
        "ZIMAGE_BIAS_NATIVE_BODY_COMPLETED_V1 mechanisms=4 fixture_owners_dropped=1 cpu_sync=0"
    );
    let _ = parent_completion.replace(CpuChildCompletion);
}

/// The actual ordinary and streamed encoder loops execute/finish ALL configured layers, even
/// though the returned hidden state is second-to-last. Only the first entry carries an alias pair.
#[test]
fn encoder_carry_matches_ordinary_and_streamed_run_finish_chronology() {
    if !in_cpu_child("encoder_carry_matches_ordinary_and_streamed_run_finish_chronology") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    let normal_end = std::cell::Cell::new(false);
    let completed = std::cell::Cell::new(0_u8);
    with_cpu_diagnostic_headroom(AMPLE, || {
        let _scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let snap = snapshot();
        let selected = snap.te_contract.validate_source(&WeightsSource::Dir(
            snap.root.join("text_encoder"),
        ))?;
        let ids = Array::from_slice(
            &(0..WINDOW).map(|i| 3 + i * 7).collect::<Vec<_>>(),
            &[1, WINDOW],
        );
        let mask = Array::from_slice(&vec![1_i32; WINDOW as usize], &[1, WINDOW]);
        let mut first_output = None;
        for streamed in [false, true] {
            let encoder = if streamed {
                crate::loader::load_validated_text_encoder_streamable_with(&selected, &snap.te_cfg)?
            } else {
                selected.read_unchanged(|source| {
                    let WeightsSource::Dir(dir) = source else {
                        unreachable!("fixture is a validated directory")
                    };
                    let raw = mlx_gen::weights::Weights::from_dir(dir)?;
                    let encoder = crate::text_encoder::TextEncoder::from_weights(
                        &raw,
                        "model",
                        &snap.te_cfg,
                    )?;
                    raw.materialize_accessed()?;
                    Ok::<_, mlx_gen::Error>(encoder)
                })?
            };
            let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
            let sink = events.clone();
            let output = with_cpu_diagnostic_observer(
                move |event| match event {
                    QuantumEvent::Checked { label, .. } if label == crate::quantum::TE_LAYER => {
                        sink.borrow_mut().push(false);
                    }
                    QuantumEvent::Synchronized { label, .. }
                        if label == crate::quantum::TE_LAYER =>
                    {
                        sink.borrow_mut().push(true);
                    }
                    _ => {}
                },
                || {
                    if streamed {
                        encoder.forward_windowed(&ids, &mask, 1, &CancelFlag::default())
                    } else {
                        encoder.forward(&ids, &mask)
                    }
                },
            )?;
            output.eval()?;
            let values = output.as_slice::<f32>().to_vec();
            if let Some(first) = &first_output {
                assert_eq!(&values, first, "ordinary/streamed returned hidden state");
            } else {
                first_output = Some(values);
            }
            let mut facts = snap.facts(false, None);
            facts.streamable = streamed;
            let model = Model::new(facts)?;
            let route = Route {
                mode: Mode::Text,
                width: 16,
                height: 16,
                count: 1,
                execution: if streamed { execution() } else { GenerationMemory::default() },
            };
            let priced = model.quanta(&route, Backend::Cpu)?;
            let layers = priced.iter().filter(|p| p.label == crate::quantum::TE_LAYER);
            let carries = layers.map(|p| p.window.held_bytes).collect::<Vec<_>>();
            let ab = crate::quantum::ab;
            let hidden = ab(4, &[1, WINDOW as u64, snap.te_cfg.hidden_size as u64]);
            let tables_and_mask = 2 * ab(4, &[1, WINDOW as u64, snap.te_cfg.head_dim as u64])
                + ab(4, &[1, 1, WINDOW as u64, WINDOW as u64]);
            assert_eq!(
                (
                    events.borrow().clone(),
                    carries,
                    output.shape().to_vec(),
                    output.dtype(),
                ),
                (
                    vec![false, true, false, true, false, true],
                    vec![
                        hidden + tables_and_mask,
                        2 * hidden + tables_and_mask,
                        2 * hidden + tables_and_mask,
                    ],
                    vec![1, WINDOW, snap.te_cfg.hidden_size],
                    Dtype::Float32,
                ),
                "all 3 actual run+finish pairs and source carry entries, streamed={streamed}",
            );
            completed.set(completed.get() | (1 << u8::from(streamed)));
            eprintln!("ZIMAGE_ENCODER_CARRY_NATIVE_MECHANISM_V1 streamed={streamed} layers=3");
        }
        normal_end.set(true);
        Ok(())
    })
    .unwrap();
    assert!(normal_end.get(), "native encoder diagnostic body did not complete");
    assert_eq!(completed.get(), 0b11, "missing native encoder mechanism");
    let sync = unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) };
    assert_eq!(sync, 0, "native encoder CPU synchronization after scope drop failed");
    eprintln!(
        "ZIMAGE_ENCODER_CARRY_NATIVE_BODY_COMPLETED_V1 mechanisms=2 fixture_owners_dropped=1 cpu_sync=0"
    );
    let _ = parent_completion.replace(CpuChildCompletion);
}

#[test]
fn dit_alias_suffixes_match_actual_native_constructor_and_quantizer() {
    if !in_cpu_child("dit_alias_suffixes_match_actual_native_constructor_and_quantizer") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    let normal_end = std::cell::Cell::new(false);
    let completed = std::cell::Cell::new(0_u16);
    with_cpu_diagnostic_headroom(AMPLE, || {
        let _scope = mlx_gen::ios_quantum::Scope::try_new()?;
        for bits in [4, 8] {
            for (index, (source_weight, source_bias, packed, source_scales)) in [
                (true, false, false, false),
                (true, true, false, false),
                (false, true, false, false),
                (true, false, true, false),
                (true, true, true, true),
            ].into_iter().enumerate() {
                let dense = Array::ones::<f32>(&[4, 64])?.as_dtype(Dtype::Float16)?;
                let seed = mlx_gen::weights::Weights::from_map(
                    [("seed.weight".into(), dense.clone())].into_iter().collect(),
                );
                let mut packed_seed = crate::quant::lin(&seed, "seed", false)?;
                packed_seed.quantize(bits, None)?;
                packed_seed.materialize_weights()?;
                let (codes, scales, offsets, _, _, _) = packed_seed.quantized_params().unwrap();
                let from = "t_embedder.mlp.0";
                let to = "t_embedder.linear1";
                let mut tensors = std::collections::HashMap::new();
                tensors.insert(format!("{to}.weight"), if source_weight {
                    Array::ones::<f32>(&[8, 64])?
                } else { dense.clone() });
                let target_bias = Array::ones::<f32>(&[4])?;
                tensors.insert(format!("{to}.bias"), if source_bias {
                    target_bias.as_dtype(Dtype::Float16)?
                } else { target_bias });
                if source_weight {
                    tensors.insert(format!("{from}.weight"), if packed { codes.clone() } else { dense });
                }
                if source_bias {
                    tensors.insert(format!("{from}.bias"), Array::ones::<f32>(&[4])?);
                }
                if packed {
                    tensors.insert(format!("{to}.scales"), scales.clone());
                    tensors.insert(format!("{to}.biases"), offsets.clone());
                    if source_scales {
                        tensors.insert(format!("{from}.scales"), scales.clone());
                    }
                }
                // Independent raw dictionary remains deliberately owned through the assertion.
                // Cloning these Array handles does not make another parameter payload.
                let raw = mlx_gen::weights::Weights::from_map(tensors.clone());
                let mut view = mlx_gen::weights::Weights::from_map(tensors);
                crate::loader::remap_transformer_keys(&mut view);
                let mut actual = crate::quant::lin(&view, to, true)?;
                view.materialize_accessed()?;
                let before_bias = actual.bias().map(|b| b.dtype());
                actual.quantize(bits, None)?;
                actual.materialize_weights()?;
                let (wq, _, _, _, _, actual_bits) = actual.quantized_params().unwrap();
                assert_eq!(
                    (
                        before_bias,
                        actual.bias().map(|b| b.dtype()),
                        actual.base_shape(),
                        wq.shape().to_vec(),
                        actual_bits,
                        raw.require(&format!("{to}.weight"))?.shape().to_vec(),
                        raw.require(&format!("{to}.bias"))?.dtype(),
                    ),
                    (
                        Some(Dtype::Float32),
                        Some(if packed { Dtype::Float32 } else { Dtype::Bfloat16 }),
                        vec![4, 64], vec![4, 64 * bits / 32], bits,
                        vec![if source_weight { 8 } else { 4 }, 64],
                        if source_bias { Dtype::Float16 } else { Dtype::Float32 },
                    ),
                    "actual per-suffix alias case={index} Q{bits}",
                );
                let mechanism = 5 * usize::from(bits == 8) + index;
                completed.set(completed.get() | (1 << mechanism));
                eprintln!("ZIMAGE_DIT_ALIAS_NATIVE_MECHANISM_V1 bits={bits} case={index}");
            }
        }
        normal_end.set(true);
        Ok(())
    }).unwrap();
    assert!(normal_end.get(), "actual alias diagnostic body incomplete");
    assert_eq!(completed.get(), 0b11_1111_1111, "missing actual alias case");
    let sync = unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) };
    assert_eq!(sync, 0, "alias CPU synchronization after scope drop failed");
    eprintln!("ZIMAGE_DIT_ALIAS_NATIVE_BODY_COMPLETED_V1 mechanisms=10 fixture_owners_dropped=1 cpu_sync=0");
    let _ = parent_completion.replace(CpuChildCompletion);
}

#[test]
fn dit_constructor_inventory_matches_actual_accesses_and_retains_deferred_shells() {
    if !in_cpu_child("dit_constructor_inventory_matches_actual_accesses_and_retains_deferred_shells") {
        return;
    }
    let mut parent_completion: Option<CpuChildCompletion> = None;
    let normal_end = std::cell::Cell::new(false);
    let completed = std::cell::Cell::new(0_u8);
    with_cpu_diagnostic_headroom(AMPLE, || {
        let _scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let snap = snapshot();
        let dir = snap.root.join("transformer");
        let raw = mlx_gen::weights::Weights::from_dir(&dir)?;
        // Fixture preparation only: finish the tiny original file reads before rewriting THIS
        // fixture file. Never eager-read inside a production block stream or change its loader.
        raw.materialize()?;
        let mut tensors = raw.keys().map(|k| (k.to_owned(), raw.get(k).unwrap().clone()))
            .collect::<std::collections::HashMap<_, _>>();
        tensors.insert("t_embedder.linear1.weight".into(), Array::ones::<f32>(&[8, 64])?);
        tensors.insert("unconsumed_matrix.weight".into(), Array::ones::<f32>(&[2, 64])?);
        mlx_rs::transforms::eval(tensors.values())?;
        Array::save_safetensors(tensors.iter().map(|(k, a)| (k.as_str(), a)).collect::<Vec<_>>(), None, dir.join("model.safetensors"))?;
        for streamed in [false, true] {
            let actual = crate::loader::load_transformer_with_stream_cfg(
                &snap.root, dit_cfg(), streamed, streamed,
            )?;
            let mut source = mlx_gen::weights::Weights::from_dir(&dir)?;
            crate::loader::remap_transformer_keys(&mut source);
            let native_constructor = crate::transformer::ZImageTransformer::from_weights(
                &source, "", dit_cfg(),
            )?;
            let accessed = source.accessed_entries();
            let actual_keys = accessed.iter().map(|(key, _)| key.clone())
                .collect::<std::collections::BTreeSet<_>>();
            // This independent oracle reads actual native constructor arrays, not the new origin
            // helper. Aliases only expose the one canonical constructor key for each projection.
            let actual_payload = accessed.iter().filter(|(key, _)| !streamed || !key.starts_with("layers."))
                .map(|(_, array)| mlx_gen::ios_quantum::allocated_bytes(array.nbytes() as u64))
                .sum::<u64>() + native_constructor.rope.resident_bytes();
            let mut facts = snap.facts(false, None);
            facts.streamable = streamed;
            let model = Model::new(facts)?;
            let origins = model.dit_constructor_origins();
            assert_eq!(
                (
                    origins.keys().cloned().collect::<std::collections::BTreeSet<_>>(),
                    origins["t_embedder.linear1.weight"].as_str(),
                    origins.contains_key("unconsumed_matrix.weight"),
                    origins.keys().filter(|key| key.starts_with("layers.")).count(),
                    actual.layers.len(),
                    model.dit_resident(),
                ),
                (
                    actual_keys,
                    "t_embedder.mlp.0.weight", false,
                    accessed.iter().filter(|(key, _)| key.starts_with("layers.")).count(),
                    dit_cfg().n_layers,
                    actual_payload,
                ),
                "actual constructor membership, aliases and deferred shell domain streamed={streamed}",
            );
            // This proves retained shell MODULES, not evaluated N-layer payloads, descriptor heap
            // size, alias disjointness, last native owner, foreign tracing, or physical release.
            completed.set(completed.get() | (1 << u8::from(streamed)));
            eprintln!("ZIMAGE_DIT_INVENTORY_NATIVE_MECHANISM_V1 streamed={streamed} shells_retained=1");
        }
        normal_end.set(true);
        Ok(())
    }).unwrap();
    assert!(normal_end.get(), "actual constructor inventory diagnostic body incomplete");
    assert_eq!(completed.get(), 0b11, "missing constructor inventory mechanism");
    let sync = unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) };
    assert_eq!(sync, 0, "inventory CPU synchronization after scope drop failed");
    eprintln!("ZIMAGE_DIT_INVENTORY_NATIVE_BODY_COMPLETED_V1 mechanisms=2 fixture_owners_dropped=1 cpu_sync=0");
    let _ = parent_completion.replace(CpuChildCompletion);
}

// Source-only private execution sidecar. This include creates NO public provider export or
// dependency on the catalog; catalog unit tests include the same engine/core-only source.
#[cfg(target_os = "macos")]
pub(crate) mod original_execution_sidecar {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/common/original_z_execution_sidecar.rs"));
}

#[cfg(target_os = "macos")]
impl Snapshot {
    /// Describe the actual manual fixture caller, not Route::request's default DTO. This method
    /// neither loads models nor changes run_request. A caller constructing snapshot()/tiers()
    /// MUST have the strict native CPU guard first; the metadata-only unit below uses a proven
    /// native-free Snapshot field fixture instead. Host image bytes may be copied for description;
    /// no Array reference is acquired, cloned, evaluated, or retained.
    pub(crate) fn original_execution_sidecar(
        &self,
        spec: &LoadSpec,
        route: &Route,
        hint: &mlx_gen::Image,
        kind: mlx_gen::gen_core::ControlKind,
        position: original_execution_sidecar::Position,
    ) -> mlx_gen::Result<original_execution_sidecar::Projection> {
        use original_execution_sidecar::{Inputs, ManualMode};
        let mode = match route.mode {
            Mode::Text => ManualMode::Text,
            Mode::Reference => ManualMode::Reference,
            Mode::Control => ManualMode::Control,
            Mode::ControlReference => ManualMode::ControlReference,
        };
        // Only the memory field of this DTO is consumed by the SAME original window selectors.
        // None of its default sampling/conditioning values supplies execution-sidecar facts.
        let window_request = route.request();
        let streamable = spec.load_shape == LoadShape::DeferredMaterialization;
        let te_window = crate::pipeline::resolve_encoder_window(
            &window_request, streamable, crate::model::MODEL_ID)?;
        let dit_window = crate::pipeline::resolve_block_window(
            &window_request, streamable, crate::model::MODEL_ID)?;
        let cfg = dit_cfg();
        original_execution_sidecar::project(Inputs {
            provider: if route.mode.control() { crate::model_control::MODEL_ID } else { crate::model::MODEL_ID },
            mode, position, original_spec: spec,
            actual_text_source: &self.root.join("text_encoder"),
            actual_control_source: &self.control,
            width: route.width, height: route.height, count: route.count, execution: route.execution,
            hint: hint.clone(),
            init: route.mode.img2img().then(|| image(route.width, 2)),
            control_kind: kind,
            original_factory_source: include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ios_pricing_tests.rs")),
            actual_vocab_size: self.te_cfg.vocab_size,
            configurations: [format!("{:?}", self.te_cfg), format!("{cfg:?}"),
                format!("{:?}", crate::vae::VaeEncoderConfig::default_z_image()),
                format!("{:?}", crate::vae::VaeDecoderConfig::default_z_image())],
            te_explicit_window: te_window,
            te_full_cover_fallback: (streamable && te_window.is_none()).then_some(self.te_cfg.n_layers),
            dit_selector_window: dit_window,
            actual_te_layers: self.te_cfg.n_layers, actual_base_blocks: cfg.n_layers,
            actual_control_blocks: route.mode.control().then_some(crate::control_transformer::CONTROL_LAYERS_PLACES.len()),
        })
    }
}

#[cfg(target_os = "macos")]
#[test]
fn original_sidecar_describes_manual_inputs_without_default_dto_facts() {
    use mlx_gen::gen_core::ControlKind;
    use original_execution_sidecar::Position;
    // Native-free PRIVATE wrapper input: only a TempDir, paths, scalar configs and an encoder
    // contract. This DOES NOT call snapshot(), tiers(), Tensors, Model/view, Array or a device.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("metadata-only");
    let te_contract = mlx_gen::gen_core::EncoderContract {
        num_hidden_layers: 3, loaded_hidden_layers: 3, selected_hidden_layers: &[2],
        ..crate::bounded_encoder_contract()
    };
    gen_core_testkit::write_encoder_contract_fixture(&root.join("text_encoder"), te_contract).unwrap();
    let te_cfg = ZTextEncoderConfig {
        vocab_size: te_contract.vocab_size as i32, hidden_size: te_contract.hidden_size as i32,
        n_layers: te_contract.num_hidden_layers, n_heads: te_contract.num_attention_heads as i32,
        n_kv_heads: te_contract.num_key_value_heads as i32, head_dim: te_contract.head_dim as i32,
        intermediate_size: te_contract.intermediate_size as i32, rope_theta: 1_000_000.0, rms_norm_eps: 1e-6,
    };
    let control = dir.path().join("actual-control.safetensors");
    let snap = Snapshot { _dir: dir, root, control, te_cfg, te_contract };
    let tier_root = snap.root.join("q8-literal-source");
    let spec = endpoint_spec(&tier_root, Some(8), false);
    let route = Route { mode: Mode::Control, width: EDGE, height: EDGE, count: 1,
        execution: GenerationMemory::default() };
    let hint = image(EDGE, 1);
    let actual = snap.original_execution_sidecar(&spec, &route, &hint, ControlKind::Pose, Position::Intermediate).unwrap();
    let wrong = route.request();
    assert!(spec.control.is_none());
    assert!(actual.described_spec.control.is_some());
    assert_eq!(actual.provider, crate::model_control::MODEL_ID);
    assert!(wrong.conditioning.is_empty());
    assert_eq!(actual.request.memory_reference_count(), 1);
    assert_eq!(wrong.steps, None);
    assert_eq!(actual.request.steps, Some(3));
    assert_eq!(actual.request.seed, Some(7));
    assert_eq!(actual.request.scheduler_shift, None);
    assert_eq!(actual.program["manual_flow"]["shift_bits"], 3.0_f32.to_bits());
    let controls = crate::sampling::resolve(&actual.request).unwrap();
    assert_eq!(controls.steps(), 3);
    assert_eq!(controls.solver(), mlx_gen::Solver::Euler);
    crate::ios_pricing::validate_pricing_roles(crate::model_control::MODEL_ID, 1, 0, 0, 1).unwrap();
    assert_eq!(actual.program["actual_sources"]["control"], snap.control.to_str().unwrap());
    assert_eq!(actual.program["actual_sources"]["text_encoder"], snap.root.join("text_encoder").to_str().unwrap());
    assert_eq!(actual.program["actual_sources"]["vae"], tier_root.join("vae").to_str().unwrap());
    assert_eq!(actual.program["position"], "intermediate_deferred_full_cover_unstaged");
    assert_eq!(actual.program["manual_flow"]["start"], 0);
    assert_eq!(actual.program["manual_flow"]["control_scale_bits"], 0.8_f32.to_bits());
    assert_eq!(actual.program["windows"]["TE_full_cover_fallback"], te_cfg.n_layers);
    assert_eq!(actual.program["windows"]["DIT_selector"], crate::transformer::ZImageTransformerConfig::turbo().n_layers);
    assert_eq!(actual.program["actual_load_form"]["TE_load_quant"], serde_json::Value::Null);
    let tokens = &actual.program["tokens"];
    assert_eq!(tokens["window"], WINDOW);
    assert_eq!(tokens["valid_tokens"], VALID);
    assert_eq!(tokens["shape"], serde_json::json!([1, WINDOW]));
    assert_eq!(tokens["layout"], "row-major contiguous");
    assert_eq!(tokens["input_ids"]["dtype"], "i32");
    assert_eq!(tokens["attention_mask"]["dtype"], "i32");
    assert_eq!(tokens["vocabulary"]["size"], te_cfg.vocab_size);
    let ids = (0..WINDOW).map(|i| 3 + i * 7).collect::<Vec<i32>>();
    let mask = (0..WINDOW).map(|i| i32::from(i < VALID)).collect::<Vec<i32>>();
    assert_eq!(tokens["input_ids"]["values"], serde_json::json!(ids));
    assert_eq!(tokens["attention_mask"]["values"], serde_json::json!(mask));
    assert_eq!(tokens["input_ids"]["values_sha256"], original_execution_sidecar::source_i32_values_hash(&ids));
    assert_eq!(tokens["attention_mask"]["values_sha256"], original_execution_sidecar::source_i32_values_hash(&mask));
    assert_eq!(tokens["padding"]["masked_tail_count"], WINDOW - VALID);
    assert_eq!(tokens["padding"]["pad_token_id_applied"], serde_json::Value::Null);
    let reference = Route { mode: Mode::Reference, ..route };
    let reference = snap.original_execution_sidecar(&spec, &reference, &hint, ControlKind::Pose, Position::Intermediate).unwrap();
    assert_eq!(reference.program["manual_flow"]["start"], 1);
    assert_eq!(reference.request.strength, None);
    assert_eq!(crate::pipeline::init_time_step(3, reference.request.strength), 0,
        "manual start is an additional input, never an invented public strength");
    // Position names are descriptions, not self-certifying endpoint grants; actual endpoint
    // selection/flags/family/work bindings are tested through the catalog's real helpers.
    let _ = [Position::ResidentEndpoint, Position::OptimizedEndpoint];
}
