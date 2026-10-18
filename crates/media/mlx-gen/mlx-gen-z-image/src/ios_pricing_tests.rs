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

use mlx_gen::gen_core::{GenerationMemory, TransformerComponent, WeightsSource};
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
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("ios_pricing_tests::{name}"),
                "--nocapture",
                "--test-threads=1",
                "--include-ignored",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success(), "CPU child for {name} failed");
        return false;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(crate::test_metal_guard::forbid_metal());
    assert!(crate::test_metal_guard::metal_forbidden());
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

impl Snapshot {
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

/// Run one request's production phases on the snapshot, in the staged generators' order
/// (`model::ZImageTurbo::generate_impl` / `model_control::ZImageTurboControl::generate_impl`
/// over `Residency::run_staged_request_scoped`), returning every checked `(label, scratch)`.
fn run(snap: &Snapshot, route: &Route) -> Vec<(String, u64)> {
    let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let sink = events.clone();
    with_cpu_diagnostic_headroom(AMPLE, || {
        with_cpu_diagnostic_observer(
            move |e| {
                if let QuantumEvent::Checked { label, scratch } = e {
                    sink.borrow_mut().push((label.clone(), *scratch));
                }
            },
            || run_request(snap, &snap.root, route, None, &image(route.width, 1)),
        )
    })
    .unwrap();
    let events = events.borrow().clone();
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
    root: &Path,
    route: &Route,
    quant_bits: Option<i32>,
    control_image: &mlx_gen::Image,
) -> mlx_gen::Result<()> {
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
    let te = crate::loader::load_validated_text_encoder_streamable_with(&source, &snap.te_cfg)?;
    enum Heavy {
        Base(Box<crate::ZImageTransformer>),
        Control(Box<crate::ZImageControlTransformer>),
    }
    let load_heavy = || -> mlx_gen::Result<(Heavy, crate::vae::Vae)> {
        let dit = if mode.control() {
            let mut dit = crate::loader::load_control_transformer_with_stream_cfg(
                root,
                &WeightsSource::File(snap.control.clone()),
                cfg.clone(),
                true,
                true,
            )?;
            if let Some(bits) = quant_bits {
                dit.quantize(bits)?;
            }
            Heavy::Control(Box::new(dit))
        } else {
            let mut dit =
                crate::loader::load_transformer_with_stream_cfg(root, cfg.clone(), true, true)?;
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

    // Phase A: the streamable text encoder's prompt encode (then, staged, its drop).
    let ids: Vec<i32> = (0..WINDOW).map(|i| 3 + i * 7).collect();
    let mask: Vec<i32> = (0..WINDOW).map(|i| i32::from(i < VALID)).collect();
    let window = crate::pipeline::EncoderWindow::resolve(&req, true, id)?;
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
        heavy = Some(load_heavy()?);
    }
    let (dit, vae) = heavy.take().unwrap();
    let block_window = crate::pipeline::resolve_block_window(&req, true, id)?;
    let budget = crate::pipeline::attention_budget(&req);
    let steps = 3;
    let start = if mode.img2img() { 1 } else { 0 };
    let scheduler = FlowMatchEuler::for_static_shift(steps, 3.0);
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

    // Phase C: the (tiled) decode.
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
    let route = Route {
        mode,
        width: edge,
        height: edge,
        count: 1,
        execution: execution(),
    };
    let observed = run(&snap, &route);
    let priced: Vec<Priced> = Model::new(snap.facts(mode.control(), None))
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

#[test]
fn text_pricing_equals_the_observed_quanta() {
    if !in_cpu_child("text_pricing_equals_the_observed_quanta") {
        return;
    }
    assert_priced_exactly(Mode::Text, EDGE);
}

#[test]
fn reference_pricing_equals_the_observed_quanta() {
    if !in_cpu_child("reference_pricing_equals_the_observed_quanta") {
        return;
    }
    assert_priced_exactly(Mode::Reference, EDGE);
}

/// The whole-image encode and the untiled decode (an image within both tiles).
#[test]
fn untiled_reference_pricing_equals_the_observed_quanta() {
    if !in_cpu_child("untiled_reference_pricing_equals_the_observed_quanta") {
        return;
    }
    assert_priced_exactly(Mode::Reference, SMALL_EDGE);
}

#[test]
fn control_pricing_equals_the_observed_quanta() {
    if !in_cpu_child("control_pricing_equals_the_observed_quanta") {
        return;
    }
    assert_priced_exactly(Mode::Control, EDGE);
}

#[test]
fn control_reference_pricing_equals_the_observed_quanta() {
    if !in_cpu_child("control_reference_pricing_equals_the_observed_quanta") {
        return;
    }
    assert_priced_exactly(Mode::ControlReference, EDGE);
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
/// its 15 streamed main blocks stay lazy and are packed per window by the stream. Proof: each load's
/// measured resident bytes (MLX active memory across load + quantize) track the pricing's resident
/// formula (`dit_floor + control_resident`) within 1.5x, for the deferred and the eager load.
/// Parity: the deferred load's windowed forward equals the eager load's resident forward bit for
/// bit, and a non-windowed forward on the deferred load is refused rather than run dense.
#[test]
fn a_deferred_control_load_packs_only_its_resident_blocks() {
    if !in_cpu_child("a_deferred_control_load_packs_only_its_resident_blocks") {
        return;
    }
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
    mlx_gen::ios_quantum::assert_tracks_measured(
        "eager control load residency",
        eager_formula,
        eager_bytes,
        1.5,
    );
    mlx_gen::ios_quantum::assert_tracks_measured(
        "deferred control load residency",
        deferred_formula,
        deferred_bytes,
        1.5,
    );

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
}
