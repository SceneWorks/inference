//! iOS bounded-quantum guards on the Z-Image inference path (epic 16774, sc-26433), exercised on
//! tiny CPU models through the diagnostic fixture (`mlx_gen::ios_quantum::with_cpu_diagnostic_headroom`).
//! Every test re-executes itself in a child process pinned to the CPU device with Metal made fatal,
//! so nothing here can touch the GPU.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, WeightsSource};
use mlx_rs::{random, Array, Dtype};

use crate::control_transformer::ZImageControlTransformer;
use crate::text_encoder::{TextEncoder, ZTextEncoderConfig};
use crate::transformer::{ZImageTransformer, ZImageTransformerConfig};
use crate::vae::{Vae, VaeDecoderConfig, VaeEncoderConfig};

const CHILD: &str = "ZIMAGE_IOS_QUANTUM_CPU_CHILD";

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.safetensors"))
}

/// Re-run `name` in a child pinned to the CPU with Metal made fatal. Returns `true` inside the
/// child (the caller runs its body), `false` in the parent after the child succeeded.
fn in_cpu_child(name: &str) -> bool {
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("quantum_tests::{name}"),
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

fn rand(shape: &[i32], seed: u64, dtype: Dtype) -> Array {
    let a = random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap()))
        .unwrap()
        .as_dtype(dtype)
        .unwrap();
    a.eval().unwrap();
    a
}

// ---- Tiny modules ----------------------------------------------------------------------------

/// The DiT of `tests/fixtures/denoise_loop.safetensors` (prefix `w`).
fn dit_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 4,
        dim: 96,
        n_layers: 2,
        n_refiner_layers: 1,
        n_heads: 4,
        norm_eps: 1e-5,
        cap_feat_dim: 32,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![8, 8, 8],
        axes_lens: vec![64, 64, 64],
        frequency_embedding_size: 256,
    }
}

/// The control DiT of `tests/fixtures/z_control_transformer.safetensors` (prefix `w`).
fn control_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 4,
        dim: 64,
        n_layers: 4,
        n_refiner_layers: 2,
        n_heads: 4,
        norm_eps: 1e-5,
        cap_feat_dim: 32,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![8, 4, 4],
        axes_lens: vec![64, 64, 64],
        frequency_embedding_size: 256,
    }
}

/// A streamable DiT (rung 4 reads its `w.layers.*` back from the fixture file).
fn dit() -> ZImageTransformer {
    let path = fixture("denoise_loop");
    let w = Weights::from_file(&path).unwrap();
    ZImageTransformer::from_weights(&w, "w", dit_cfg())
        .unwrap()
        .with_block_stream(WeightsSource::File(path), "w")
}

/// A streamable control DiT over its fixture.
fn control_dit() -> (ZImageControlTransformer, Weights) {
    let path = fixture("z_control_transformer");
    let w = Weights::from_file(&path).unwrap();
    let base = ZImageTransformer::from_weights(&w, "w", control_cfg())
        .unwrap()
        .with_block_stream(WeightsSource::File(path.clone()), "w");
    let model = ZImageControlTransformer::from_weights(base, &w, "w")
        .unwrap()
        .with_control_block_stream(WeightsSource::File(path), "w");
    (model, w)
}

/// The resident tiny text encoder of `tests/fixtures/text_encoder.safetensors`.
fn text_encoder() -> (TextEncoder, Weights) {
    let w = Weights::from_file(fixture("text_encoder")).unwrap();
    let p: Vec<i32> = w
        .metadata("cfg")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let cfg = ZTextEncoderConfig {
        vocab_size: p[0],
        hidden_size: p[1],
        n_layers: p[2] as usize,
        n_heads: p[3],
        n_kv_heads: p[4],
        head_dim: p[5],
        intermediate_size: p[6],
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
    };
    (TextEncoder::from_weights(&w, "", &cfg).unwrap(), w)
}

/// A three-layer validated (streamable) text encoder over random F16 weights at the bounded
/// contract's geometry, written to `dir`.
fn streamable_text_encoder(dir: &std::path::Path) -> TextEncoder {
    let contract = mlx_gen::gen_core::EncoderContract {
        num_hidden_layers: 3,
        loaded_hidden_layers: 3,
        selected_hidden_layers: &[2],
        ..crate::bounded_encoder_contract()
    };
    let te = dir.join("text_encoder");
    gen_core_testkit::write_encoder_contract_fixture(&te, contract).unwrap();
    // Replace the sparse zero payload with seeded random weights of the identical header.
    let file = te.join("model.safetensors");
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&file).unwrap();
    let arrays: Vec<(String, Array)> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let shape: Vec<i32> = h.shape.iter().map(|&d| d as i32).collect();
            let scale = if h.name.ends_with("norm.weight") {
                1.0
            } else {
                0.05
            };
            let a = mlx_rs::ops::multiply(
                rand(&shape, 1000 + i as u64, Dtype::Float32),
                Array::from_f32(scale),
            )
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
            (h.name.clone(), a)
        })
        .collect();
    std::fs::remove_file(&file).unwrap();
    Array::save_safetensors(
        arrays
            .iter()
            .map(|(k, a)| (k.as_str(), a))
            .collect::<Vec<_>>(),
        None,
        &file,
    )
    .unwrap();
    let source = contract
        .validate_source(&WeightsSource::Dir(te.clone()))
        .unwrap();
    let w = Weights::from_dir(&te).unwrap();
    TextEncoder::from_validated_streamable_source(
        &w,
        source,
        "model",
        &ZTextEncoderConfig {
            vocab_size: contract.vocab_size as i32,
            hidden_size: contract.hidden_size as i32,
            n_layers: contract.num_hidden_layers,
            n_heads: contract.num_attention_heads as i32,
            n_kv_heads: contract.num_key_value_heads as i32,
            head_dim: contract.head_dim as i32,
            intermediate_size: contract.intermediate_size as i32,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        },
    )
    .unwrap()
}

fn vae_decoder_cfg() -> VaeDecoderConfig {
    VaeDecoderConfig {
        up_blocks: vec![(3, true), (3, false)],
    }
}

/// A fixture weight writer `(weights, key, shape, scale)`.
type Put<'a> = &'a mut dyn FnMut(&mut Weights, String, &[i32], f32);

/// The tiny VAE decoder fixture plus a seeded random two-block encoder (32 → 64 channels) that
/// downsamples by 2 into the decoder's 16 latent channels.
pub(crate) fn vae() -> Vae {
    let mut w = Weights::from_file(fixture("vae_decoder")).unwrap();
    let mut seed = 2000u64;
    let mut put = |w: &mut Weights, key: String, shape: &[i32], scale: f32| {
        seed += 1;
        let a = mlx_rs::ops::multiply(rand(shape, seed, Dtype::Float32), Array::from_f32(scale))
            .unwrap();
        a.eval().unwrap();
        w.insert(key, a);
    };
    let conv = |w: &mut Weights, put: Put<'_>, p: &str, o: i32, i: i32, k: i32| {
        put(
            w,
            format!("{p}.weight"),
            &[o, k, k, i],
            (1.0 / (i * k * k) as f32).sqrt(),
        );
        put(w, format!("{p}.bias"), &[o], 0.02);
    };
    let norm = |w: &mut Weights, p: &str, c: i32| {
        w.insert(format!("{p}.weight"), Array::ones::<f32>(&[c]).unwrap());
        w.insert(format!("{p}.bias"), Array::zeros::<f32>(&[c]).unwrap());
    };
    let resnet = |w: &mut Weights, put: Put<'_>, p: &str, i: i32, o: i32| {
        norm(w, &format!("{p}.norm1"), i);
        conv(w, put, &format!("{p}.conv1"), o, i, 3);
        norm(w, &format!("{p}.norm2"), o);
        conv(w, put, &format!("{p}.conv2"), o, o, 3);
        if i != o {
            conv(w, put, &format!("{p}.conv_shortcut"), o, i, 1);
        }
    };
    conv(&mut w, &mut put, "encoder.conv_in.conv", 32, 3, 3);
    resnet(&mut w, &mut put, "encoder.down_blocks.0.resnets.0", 32, 32);
    conv(
        &mut w,
        &mut put,
        "encoder.down_blocks.0.downsamplers.0.conv",
        32,
        32,
        3,
    );
    resnet(&mut w, &mut put, "encoder.down_blocks.1.resnets.0", 32, 64);
    resnet(&mut w, &mut put, "encoder.mid_block.resnets.0", 64, 64);
    norm(&mut w, "encoder.mid_block.attentions.0.group_norm", 64);
    for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
        put(
            &mut w,
            format!("encoder.mid_block.attentions.0.{proj}.weight"),
            &[64, 64],
            0.125,
        );
        put(
            &mut w,
            format!("encoder.mid_block.attentions.0.{proj}.bias"),
            &[64],
            0.02,
        );
    }
    resnet(&mut w, &mut put, "encoder.mid_block.resnets.1", 64, 64);
    norm(&mut w, "encoder.conv_norm_out.norm", 64);
    conv(&mut w, &mut put, "encoder.conv_out.conv", 32, 64, 3);
    Vae::from_weights(&w, "", &vae_decoder_cfg())
        .unwrap()
        .with_encoder(
            &w,
            "encoder",
            &VaeEncoderConfig {
                down_blocks: vec![(1, true), (1, false)],
            },
        )
        .unwrap()
}

// ---- Host anchor -------------------------------------------------------------------------------

/// A machine-tolerant fingerprint of one output: shape, element count, f64 sum and sum of squares,
/// and eight evenly spaced values. CPU BLAS ordering may differ in the last bits across Apple
/// Silicon generations, so the anchor compares at a stated relative tolerance rather than a hash
/// (no machine-dependent golden); in-process byte identity is asserted separately.
#[derive(Debug)]
struct Fingerprint {
    shape: Vec<i32>,
    sum: f64,
    sum_sq: f64,
    samples: [f32; 8],
}

fn fingerprint(a: &Array) -> Fingerprint {
    let f = a.as_dtype(Dtype::Float32).unwrap();
    f.eval().unwrap();
    let v = f.as_slice::<f32>();
    let mut samples = [0f32; 8];
    for (i, s) in samples.iter_mut().enumerate() {
        *s = v[(i * (v.len() - 1)) / 7];
    }
    Fingerprint {
        shape: a.shape().to_vec(),
        sum: v.iter().map(|&x| x as f64).sum(),
        sum_sq: v.iter().map(|&x| (x as f64) * (x as f64)).sum(),
        samples,
    }
}

/// Every host path the iOS guards wrap, run on the tiny fixtures with no fixture headroom (the guards
/// are pass-throughs). Each output's fingerprint is anchored to values recorded at the base revision
/// (`e467ffabe`, before any guard existed): `ZIMAGE_PRINT_ANCHOR=1` prints the table.
fn host_paths() -> Vec<(&'static str, Array)> {
    host_paths_result().unwrap()
}

/// [`host_paths`], failing (never panicking) at a refused quantum.
fn host_paths_result() -> mlx_gen::Result<Vec<(&'static str, Array)>> {
    let mut out = Vec::new();
    let cancel = CancelFlag::default();
    // Text encoder: resident, and the rung-4 windowed stream (window 1).
    let (te, w) = text_encoder();
    let ids = w.require("input_ids").unwrap();
    let mask = w.require("attention_mask").unwrap();
    out.push(("te_resident", te.forward(ids, mask)?));
    let dir = tempfile::tempdir().unwrap();
    let streamed = streamable_text_encoder(dir.path());
    let ids = Array::from_slice(&[151_644i32, 7, 9, 11, 151_645, 151_643], &[1, 6]);
    let mask = Array::from_slice(&[1i32, 1, 1, 1, 1, 0], &[1, 6]);
    out.push((
        "te_windowed",
        streamed.forward_windowed(&ids, &mask, 1, &cancel)?,
    ));
    out.push(("te_streamed_unscoped", streamed.forward(&ids, &mask)?));

    // DiT: unbounded, chunked attention, and the rung-4 window (f32 and bf16 latents).
    let model = dit();
    let w = Weights::from_file(fixture("denoise_loop")).unwrap();
    let x = w.require("init").unwrap();
    let cap = w.require("cap_feats").unwrap();
    out.push((
        "dit_unbounded",
        model.forward_budgeted(x, 0.6, cap, AttentionPlan::UNBOUNDED)?,
    ));
    let chunked = AttentionPlan::budgeted(AttentionBudget::from_score_elements(4 * 16, true));
    out.push(("dit_chunked", model.forward_budgeted(x, 0.6, cap, chunked)?));
    out.push((
        "dit_windowed",
        model.forward_windowed(x, 0.6, cap, AttentionPlan::UNBOUNDED, 1, &cancel)?,
    ));
    let xb = x.as_dtype(Dtype::Bfloat16).unwrap();
    let capb = cap.as_dtype(Dtype::Bfloat16).unwrap();
    out.push((
        "dit_bf16_windowed",
        model.forward_windowed(&xb, 0.6, &capb, chunked, 1, &cancel)?,
    ));

    // The denoise loop (flow sampler) over the tiny DiT.
    let sigmas = w.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scheduler = mlx_gen::FlowMatchEuler { sigmas };
    out.push((
        "denoise_loop",
        crate::pipeline::denoise_with_progress(
            &model,
            &scheduler,
            None,
            0,
            x.clone(),
            cap,
            0,
            AttentionBudget::UNBOUNDED,
            Some(1),
            &cancel,
            &mut |_| {},
        )?,
    ));

    // Control DiT: resident and windowed.
    let (control, cw) = control_dit();
    let cx = cw.require("in.x").unwrap();
    let ccap = cw.require("in.cap_feats").unwrap();
    let cc = cw.require("in.control_context").unwrap();
    out.push((
        "control_resident",
        control.forward_budgeted(cx, 0.7, ccap, Some(cc), 0.8, AttentionPlan::UNBOUNDED)?,
    ));
    out.push((
        "control_windowed",
        control.forward_windowed(cx, 0.7, ccap, Some(cc), 0.8, chunked, 1, &cancel)?,
    ));

    // VAE: dense decode, layer-tiled decode at two edges, encode; the img2img / control encodes.
    let vae = vae();
    let latent = Weights::from_file(fixture("vae_decoder"))
        .unwrap()
        .require("in.latent")
        .unwrap()
        .clone();
    let sh = latent.shape().to_vec();
    let latent5 = latent.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap();
    out.push(("vae_decode", vae.decode(&latent5)?));
    for (name, edge) in [("vae_decode_tiled_8", 8), ("vae_decode_tiled_12", 12)] {
        out.push((
            name,
            vae.decode_tiled(
                &latent5,
                &mlx_gen::tiling::TilingConfig::spatial_only(edge, 0),
                Some(&cancel),
            )?,
        ));
    }
    let image = rand(&[1, 3, 16, 16], 77, Dtype::Float32);
    out.push(("vae_encode", vae.encode(&image)?));
    let rgb = mlx_gen::Image {
        width: 16,
        height: 16,
        pixels: (0..16 * 16 * 3).map(|i| ((i * 37) % 251) as u8).collect(),
    };
    out.push((
        "init_latents",
        crate::pipeline::encode_init_latents(&vae, &rgb, 16, 16)?,
    ));
    out.push((
        "control_context",
        crate::pipeline::encode_control_context(&vae, &rgb, 16, 16)?,
    ));
    Ok(out)
}

/// `(name, shape, sum, sum_sq, samples)` recorded at the base revision (`e467ffabe`).
type Anchor = (&'static str, &'static [i32], f64, f64, [f32; 8]);
#[rustfmt::skip]
const HOST_ANCHOR: &[Anchor] = &[
    ("te_resident", &[1, 6, 64], 0.06828193180263042, 30.04791015325288, [-0.05576579, 0.12642717, -0.23280004, 0.09486472, -0.358954, 0.19556129, 0.084167026, -0.0913527]),
    ("te_windowed", &[1, 6, 64], 1.1831357972696424, 14.691652266526521, [0.09214767, -0.06120155, 0.07878521, -0.074746944, -0.010057155, 0.0345654, -0.055111848, 0.052309874]),
    ("te_streamed_unscoped", &[1, 6, 64], 1.1831357972696424, 14.691652266526521, [0.09214767, -0.06120155, 0.07878521, -0.074746944, -0.010057155, 0.0345654, -0.055111848, 0.052309874]),
    ("dit_unbounded", &[4, 1, 4, 4], -5.986734326928854, 17.249452246356995, [-1.1888548, 0.19170101, -0.63116735, 0.21690644, 0.13489853, -0.051616065, 0.21040832, -0.55439603]),
    ("dit_chunked", &[4, 1, 4, 4], -5.986734326928854, 17.249452246356995, [-1.1888548, 0.19170101, -0.63116735, 0.21690644, 0.13489853, -0.051616065, 0.21040832, -0.55439603]),
    ("dit_windowed", &[4, 1, 4, 4], -5.986734326928854, 17.249452246356995, [-1.1888548, 0.19170101, -0.63116735, 0.21690644, 0.13489853, -0.051616065, 0.21040832, -0.55439603]),
    ("dit_bf16_windowed", &[4, 1, 4, 4], -5.968599426560104, 17.24840434691701, [-1.1897036, 0.19164684, -0.63189054, 0.2152679, 0.13458577, -0.0515379, 0.21002091, -0.55428]),
    ("denoise_loop", &[4, 1, 4, 4], -1.8559004664421082, 62.104969476491746, [0.53971255, 0.62446153, 0.18356782, -0.53829026, 1.0851169, 0.74978226, 0.8789936, -1.5826539]),
    ("control_resident", &[4, 1, 4, 4], -1.2369640469551086, 22.701355291235437, [0.80415463, 0.33662274, 0.057968166, -0.76531816, 0.924518, -0.50644517, -0.0771547, -0.686625]),
    ("control_windowed", &[4, 1, 4, 4], -1.2369640469551086, 22.701355291235437, [0.80415463, 0.33662274, 0.057968166, -0.76531816, 0.924518, -0.50644517, -0.0771547, -0.686625]),
    ("vae_decode", &[1, 3, 1, 16, 16], -60.77404884574935, 77.55777182942765, [0.04651965, -0.14706917, -0.23672934, 0.034378037, -0.05924949, -0.18061781, -0.22273679, -0.043577056]),
    ("vae_decode_tiled_8", &[1, 3, 1, 16, 16], -60.77404470817419, 77.5577672675349, [0.046519533, -0.14706944, -0.23672938, 0.034377992, -0.059249472, -0.18061781, -0.22273721, -0.04357715]),
    ("vae_decode_tiled_12", &[1, 3, 1, 16, 16], -60.77404470258625, 77.55776726807217, [0.046519533, -0.14706944, -0.23672938, 0.034377992, -0.059249472, -0.18061781, -0.22273721, -0.043577142]),
    ("vae_encode", &[1, 16, 8, 8], -43.880394232983235, 45.23534858501691, [-0.098774016, 0.0026902782, -0.19412242, -0.03593259, -0.08291191, -0.030655669, -0.32192057, -0.04601451]),
    ("init_latents", &[16, 1, 8, 8], -56.3256067293114, 46.07315649845846, [-0.21028519, -0.27972057, -0.8311409, 0.3332775, -0.07439374, -0.2910177, -0.45073307, 0.069349185]),
    ("control_context", &[33, 1, 8, 8], -56.3256067293114, 46.07315649845846, [-0.21028519, -0.66728574, -0.34802523, 0.025742859, 0.0, 0.0, 0.0, 0.0]),
];

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1.0)
}

#[test]
fn host_paths_match_the_base_revision_anchor() {
    if !in_cpu_child("host_paths_match_the_base_revision_anchor") {
        return;
    }
    let paths = host_paths();
    if std::env::var_os("ZIMAGE_PRINT_ANCHOR").is_some() {
        for (name, a) in &paths {
            let f = fingerprint(a);
            println!(
                "    (\"{name}\", &{:?}, {:?}, {:?}, {:?}),",
                f.shape, f.sum, f.sum_sq, f.samples
            );
        }
        return;
    }
    assert_eq!(paths.len(), HOST_ANCHOR.len(), "anchor table out of date");
    for ((name, a), (anchor, shape, sum, sum_sq, samples)) in paths.iter().zip(HOST_ANCHOR) {
        assert_eq!(name, anchor);
        let f = fingerprint(a);
        assert_eq!(f.shape, *shape, "{name}: shape");
        assert!(close(f.sum, *sum, 1e-4), "{name}: sum {} vs {sum}", f.sum);
        assert!(
            close(f.sum_sq, *sum_sq, 1e-4),
            "{name}: sum_sq {} vs {sum_sq}",
            f.sum_sq
        );
        for (got, want) in f.samples.iter().zip(samples) {
            assert!(
                close(*got as f64, *want as f64, 1e-3),
                "{name}: sample {got} vs {want}"
            );
        }
    }
}

// ---- Guard behaviour -----------------------------------------------------------------------------

use mlx_gen::ios_quantum::{
    take_refusal, with_cpu_diagnostic_headroom, with_cpu_diagnostic_observer, QuantumEvent,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Far above every tiny fixture's scratch + the 512 MiB reserve: every guard passes.
const AMPLE: u64 = 1 << 34;

/// Run `body` under the CPU fixture at `headroom`, recording every quantum event.
fn observed<T>(
    headroom: u64,
    body: impl FnOnce() -> mlx_gen::Result<T>,
) -> (mlx_gen::Result<T>, Vec<QuantumEvent>) {
    let events = Rc::new(RefCell::new(Vec::new()));
    let sink = events.clone();
    let result = with_cpu_diagnostic_headroom(headroom, || {
        with_cpu_diagnostic_observer(move |e| sink.borrow_mut().push(e.clone()), body)
    });
    let events = events.borrow().clone();
    (result, events)
}

fn checks(events: &[QuantumEvent]) -> Vec<(String, u64)> {
    events
        .iter()
        .filter_map(|e| match e {
            QuantumEvent::Checked { label, scratch } => Some((label.clone(), *scratch)),
            QuantumEvent::Synchronized { .. } => None,
        })
        .collect()
}

/// Every synchronized quantum was checked first, and no check is left unsynchronized except the
/// pre-load floors and the stage checks of a tiled layer (whose tiles synchronize as "tiled conv
/// core" and whose stage synchronizes once at its end).
fn assert_checked_before_synchronized(events: &[QuantumEvent]) {
    let mut pending: Vec<String> = Vec::new();
    for e in events {
        match e {
            QuantumEvent::Checked { label, .. } => pending.push(label.clone()),
            QuantumEvent::Synchronized { label, .. } => {
                // A tile's core synchronizes as "tiled conv core": it answers its side's tile check.
                let core = label == "tiled conv core";
                let at = pending
                    .iter()
                    .rposition(|l| {
                        if core {
                            l == crate::quantum::VAE_DECODE_TILE
                                || l == crate::quantum::VAE_ENCODE_TILE
                        } else {
                            l == label
                        }
                    })
                    .unwrap_or_else(|| panic!("{label} synchronized without a check: {events:?}"));
                pending.remove(at);
            }
        }
    }
    for label in &pending {
        assert!(
            [
                crate::quantum::TE_LOAD,
                crate::quantum::DIT_LOAD,
                crate::quantum::VAE_LOAD,
                crate::quantum::CONTROL_LOAD,
            ]
            .contains(&label.as_str()),
            "{label} was checked but never synchronized"
        );
    }
}

fn bits(a: &Array) -> Vec<u32> {
    let f = a.as_dtype(Dtype::Float32).unwrap();
    f.eval().unwrap();
    f.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
}

/// Every host path run under bounded quanta: every quantum is checked before it synchronizes, the
/// expected labels all appear, and every output is BYTE-IDENTICAL to the unguarded host run. The
/// guards and barriers only split the lazy graph at quantum boundaries; the fixture's VAE attention
/// fits one chunk, so no path here changes its arithmetic (the chunked iOS attention is held to its
/// tolerance separately, [`chunked_vae_attention_matches_the_unchunked_attention`]).
#[test]
fn guarded_paths_check_every_quantum_and_stay_byte_identical() {
    if !in_cpu_child("guarded_paths_check_every_quantum_and_stay_byte_identical") {
        return;
    }
    let host = host_paths();
    let (guarded, events) = observed(AMPLE, || Ok(host_paths()));
    let guarded = guarded.unwrap();
    assert_checked_before_synchronized(&events);
    for ((name, a), (_, b)) in host.iter().zip(&guarded) {
        assert_eq!(
            bits(a),
            bits(b),
            "{name}: guarded output differs from the host"
        );
    }
    let labels: std::collections::BTreeSet<String> =
        checks(&events).into_iter().map(|(l, _)| l).collect();
    for label in [
        crate::quantum::TE_EMBED,
        crate::quantum::TE_LAYER,
        crate::quantum::DIT_PREPARE,
        crate::quantum::DIT_EMBED,
        crate::quantum::DIT_CAPTION,
        crate::quantum::DIT_BLOCK,
        crate::quantum::DIT_CONTEXT_BLOCK,
        crate::quantum::DIT_UNIFY,
        crate::quantum::DIT_OUTPUT,
        crate::quantum::CONTROL_EMBED,
        crate::quantum::CONTROL_BLOCK,
        crate::quantum::CONTROL_CONTEXT,
        crate::quantum::VAE_DECODE_IN,
        crate::quantum::VAE_ENCODE_IN,
        crate::quantum::VAE_DECODE_RESNET,
        crate::quantum::VAE_ENCODE_RESNET,
        crate::quantum::VAE_DECODE_ATTENTION,
        crate::quantum::VAE_ENCODE_ATTENTION,
        crate::quantum::VAE_UPSAMPLE,
        crate::quantum::VAE_DOWNSAMPLE,
        crate::quantum::VAE_DECODE_OUT,
        crate::quantum::VAE_ENCODE_OUT,
        crate::quantum::VAE_DECODE_TILE,
        crate::quantum::VAE_DECODE_TILED_STAGE,
        crate::quantum::INIT_IMAGE,
        mlx_gen::sampler::SAMPLER_STEP_QUANTUM,
    ] {
        assert!(
            labels.contains(label),
            "no {label} quantum was checked: {labels:?}"
        );
    }
    // One checked quantum per block / layer: the tiny DiT's 1+1 refiners and 2 main layers, run by
    // 5 forwards (unbounded, chunked, 2 windowed, and the 3-step loop's first-order steps).
    let count = |label: &str| checks(&events).iter().filter(|(l, _)| l == label).count();
    // The tiny DiT runs 1 noise refiner + 2 main blocks per forward, and 1 context block.
    assert!(count(crate::quantum::DIT_BLOCK) >= 3 * count(crate::quantum::DIT_CONTEXT_BLOCK));
}

/// The iOS VAE attention's query-chunked plan against the unchunked attention: a different graph
/// (each chunk's scores and softmax run at their own shape), so it is held to the cross-graph parity
/// standard — 1e-5 of `max(1, peak)`, SANA's tiled-encode standard — rather than bit for bit. A
/// chunk that drops or repeats rows, or misses keys, moves the output by O(1e-1).
#[test]
fn chunked_vae_attention_matches_the_unchunked_attention() {
    if !in_cpu_child("chunked_vae_attention_matches_the_unchunked_attention") {
        return;
    }
    let (c, side) = (64, 8);
    let n = side * side;
    let mut w = Weights::from_map(Default::default());
    w.insert(
        "a.group_norm.weight".to_string(),
        rand(&[c], 1, Dtype::Float32),
    );
    w.insert(
        "a.group_norm.bias".to_string(),
        rand(&[c], 2, Dtype::Float32),
    );
    for (i, name) in ["to_q", "to_k", "to_v", "to_out.0"].iter().enumerate() {
        let weight = mlx_rs::ops::multiply(
            rand(&[c, c], 10 + i as u64, Dtype::Float32),
            Array::from_f32(0.125),
        )
        .unwrap();
        w.insert(format!("a.{name}.weight"), weight);
        w.insert(
            format!("a.{name}.bias"),
            rand(&[c], 20 + i as u64, Dtype::Float32),
        );
    }
    let attention = crate::vae::VaeAttention::from_weights(&w, "a").unwrap();
    let x = rand(&[1, c, side, side], 30, Dtype::Float32);
    let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(n as u64 * 4, true));
    let rows = plan.budget.query_block(1, 1, n, n);
    let chunks = (n + rows - 1) / rows.max(1);
    assert!(
        rows > 0 && chunks > 1,
        "the plan must chunk: {rows} of {n} rows"
    );
    let whole = attention.forward(&x).unwrap();
    let chunked = attention.forward_planned(&x, plan).unwrap();
    assert_eq!(chunked.shape(), whole.shape());
    let (got, want) = (bits(&chunked), bits(&whole));
    let peak = want
        .iter()
        .fold(0.0f32, |m, &v| m.max(f32::from_bits(v).abs()));
    let delta = got.iter().zip(&want).fold(0.0f32, |m, (&g, &v)| {
        m.max((f32::from_bits(g) - f32::from_bits(v)).abs())
    });
    eprintln!("chunked VAE attention: {chunks} chunks, max delta {delta:.3e} (peak {peak:.3e})");
    assert!(
        delta <= 1e-5 * peak.max(1.0),
        "chunked attention moved {delta:.3e} from the unchunked attention (peak {peak:.3e})"
    );
}

/// For every quantum label the host paths check, a run whose FIRST check of that label sees live
/// headroom one byte short of its price (every other check sees ample headroom) refuses EXACTLY that
/// quantum before it is constructed: the run fails, the typed refusal names it and its price, and no
/// later quantum is checked or synchronized — the checks before it are exactly the ample run's.
#[test]
fn each_quantum_refuses_before_construction_when_headroom_is_short() {
    if !in_cpu_child("each_quantum_refuses_before_construction_when_headroom_is_short") {
        return;
    }
    let (_, events) = observed(AMPLE, || Ok(host_paths()));
    let all = checks(&events);
    let mut seen = std::collections::BTreeSet::new();
    for (i, (label, scratch)) in all.iter().enumerate() {
        if !seen.insert(label.clone()) {
            continue;
        }
        let _ = take_refusal();
        let (result, events) = observed(AMPLE, || {
            mlx_gen::ios_quantum::with_cpu_diagnostic_short_at(label, host_paths_result)
        });
        assert!(
            result.is_err(),
            "{label}: the run must fail at its refused quantum"
        );
        let record = take_refusal().unwrap_or_else(|| panic!("{label}: no refusal recorded"));
        assert_eq!(&record.label, label);
        assert_eq!(record.scratch_bytes, *scratch, "{label}");
        assert_eq!(
            checks(&events),
            all[..i].to_vec(),
            "{label}: a quantum after the refused one was checked, or an earlier one changed"
        );
    }
    eprintln!("refused each of: {seen:?}");
    assert!(seen.len() >= 24, "{seen:?}");
}

/// A genuine tiny safetensors file holding `tensors` (`(name, dtype, shape, bytes)`); payload zeros.
fn write_headers(path: &std::path::Path, tensors: &[(&str, &str, &[usize], u64)]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut offset = 0u64;
    let mut header = String::from("{");
    for (i, (name, dtype, shape, bytes)) in tensors.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{offset},{}]}}",
            offset + bytes
        ));
        offset += bytes;
    }
    header.push('}');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.resize(file.len() + offset as usize, 0);
    std::fs::write(path, file).unwrap();
}

/// Run `body` with the first check of `label` short; assert it is refused, typed, at `price`.
fn assert_refused_at<T>(
    label: &str,
    price: u64,
    body: impl FnOnce() -> mlx_gen::Result<T>,
) -> Vec<QuantumEvent> {
    let _ = take_refusal();
    let (result, events) = observed(AMPLE, || {
        mlx_gen::ios_quantum::with_cpu_diagnostic_short_at(label, body)
    });
    assert!(result.is_err(), "{label}: must be refused");
    let record = take_refusal().unwrap_or_else(|| panic!("{label}: no refusal recorded"));
    assert_eq!(record.label, label);
    assert_eq!(record.scratch_bytes, price, "{label}");
    events
}

/// The bounded encode (sc-26433): the iOS img2img / control encode of an image larger than
/// `IOS_ENCODE_TILE_PX` runs layer-tiled — `conv_in`, the resnets and the stride-2 downsample each
/// one tiled stage whose tiles are each checked before their graph — and matches the whole-image
/// encode within the cross-graph parity standard (1e-5 of `max(1, peak)`: the tiled convolutions
/// are different graphs). Off device `Vae::encode` stays the whole-image encode; a smaller image
/// takes it on device too. A tile short of headroom refuses before its graph.
#[test]
fn bounded_encode_matches_the_whole_image_encode_and_checks_every_tile() {
    if !in_cpu_child("bounded_encode_matches_the_whole_image_encode_and_checks_every_tile") {
        return;
    }
    let vae = vae();
    let image = rand(&[1, 3, 40, 40], 77, Dtype::Float32);
    let whole = vae.encode(&image).unwrap();
    let tiled = vae.encode_tiled(&image, 16, None).unwrap();
    assert_eq!(tiled.shape(), whole.shape());
    let (got, want) = (bits(&tiled), bits(&whole));
    let peak = want
        .iter()
        .fold(0.0f32, |m, &v| m.max(f32::from_bits(v).abs()));
    let delta = got.iter().zip(&want).fold(0.0f32, |m, (&g, &v)| {
        m.max((f32::from_bits(g) - f32::from_bits(v)).abs())
    });
    eprintln!("bounded encode: max delta {delta:.3e} (peak {peak:.3e})");
    assert!(
        delta <= 1e-5 * peak.max(1.0),
        "the bounded encode moved {delta:.3e} from the whole-image encode (peak {peak:.3e})"
    );
    // Guarded: the same graph, bit for bit; every layer one stage (conv_in, the two resnets, the
    // downsample) and every stage's tiles checked inside it, before their graphs.
    let (guarded, events) = observed(AMPLE, || vae.encode_tiled(&image, 16, None));
    assert_eq!(bits(&guarded.unwrap()), got);
    assert_checked_before_synchronized(&events);
    let all = checks(&events);
    let stages: Vec<usize> = all
        .iter()
        .enumerate()
        .filter(|(_, (l, _))| l == crate::quantum::VAE_ENCODE_TILED_STAGE)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        stages.len(),
        4,
        "conv_in, resnet, downsample, resnet: {all:?}"
    );
    for (n, &at) in stages.iter().enumerate() {
        let end = stages.get(n + 1).copied().unwrap_or(all.len());
        assert!(
            all[at + 1..end]
                .iter()
                .filter(|(l, _)| l == crate::quantum::VAE_ENCODE_TILE)
                .count()
                > 1,
            "stage {n} checks each of its tiles: {all:?}"
        );
    }
    let (conv_in, downs, _, _) = vae.encoder().unwrap().parts();
    assert_eq!(
        all[stages[0]].1,
        mlx_gen::ios_quantum::tiled_stage_bytes(&[1, 40, 40, 3], 3, 32, 4, false),
        "conv_in's stage price"
    );
    let ds = downs[0].parts().1.unwrap();
    assert_eq!(
        all[stages[2]].1,
        mlx_gen::ios_quantum::downsample_stage_bytes(&[1, 20, 20, 32], 4),
        "the downsample's stage price"
    );
    assert!(
        all[stages[2] + 1].1
            == mlx_gen::ios_quantum::downsample_tile_bytes(
                &[1, 17, 17, 32],
                4,
                ds.weight(),
                mlx_gen::ios_quantum::Backend::Cpu
            ),
        "the downsample's first tile price: {all:?}"
    );
    let _ = conv_in;
    // Routing: on device an image larger than the iOS tile edge takes the bounded encode; off
    // device, or for an image within one tile, the whole-image encode (no tiled stage).
    let edge = crate::vae::decoder::IOS_ENCODE_TILE_PX + 8;
    let big = rand(&[1, 3, edge, edge], 78, Dtype::Float32);
    let (routed, events) = observed(AMPLE, || vae.encode(&big));
    assert!(checks(&events)
        .iter()
        .any(|(l, _)| l == crate::quantum::VAE_ENCODE_TILED_STAGE));
    assert_eq!(
        bits(&routed.unwrap()),
        bits(
            &vae.encode_tiled(&big, crate::vae::decoder::IOS_ENCODE_TILE_PX, None)
                .unwrap()
        )
    );
    let (_, events) = observed(AMPLE, || vae.encode(&image));
    assert!(!checks(&events)
        .iter()
        .any(|(l, _)| l == crate::quantum::VAE_ENCODE_TILED_STAGE));
    // A tile short of headroom refuses before its graph: nothing after it is checked.
    let events = assert_refused_at(
        crate::quantum::VAE_ENCODE_TILE,
        all[stages[0] + 1].1,
        || vae.encode_tiled(&image, 16, None),
    );
    assert_eq!(checks(&events), all[..=stages[0]].to_vec());
}

/// The pre-load floors: each is checked against the header bytes of exactly what its load reads
/// (a rung-4 load: everything but the streamed block bodies) BEFORE any tensor is read — the fake
/// components here are not even loadable, so a load that read first would fail differently.
#[test]
fn loads_are_refused_on_their_header_floor_before_any_read() {
    if !in_cpu_child("loads_are_refused_on_their_header_floor_before_any_read") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write_headers(
        &root.join("transformer/a.safetensors"),
        &[
            ("x_embedder.weight", "BF16", &[8, 8], 128),
            ("layers.0.attention.to_q.weight", "BF16", &[16, 16], 512),
            ("layers.1.attention.to_q.weight", "BF16", &[16, 16], 512),
        ],
    );
    write_headers(
        &root.join("vae/a.safetensors"),
        &[
            ("decoder.conv_in.weight", "BF16", &[4, 4], 32),
            ("encoder.x", "F32", &[4], 16),
        ],
    );
    write_headers(
        &root.join("control.safetensors"),
        &[
            ("control_all_x_embedder.2-1.weight", "BF16", &[8, 8], 128),
            ("control_layers.0.after_proj.weight", "BF16", &[16, 16], 512),
        ],
    );
    let streamed = assert_refused_at(crate::quantum::DIT_LOAD, 128, || {
        crate::loader::load_transformer_with_stream(root, true, true)
    });
    assert!(checks(&streamed).is_empty());
    assert_refused_at(crate::quantum::DIT_LOAD, 128 + 1024, || {
        crate::loader::load_transformer_with_stream(root, true, false)
    });
    assert_refused_at(crate::quantum::VAE_LOAD, 48, || {
        crate::loader::load_vae(root)
    });
    let control = WeightsSource::File(root.join("control.safetensors"));
    assert_refused_at(crate::quantum::CONTROL_LOAD, 128, || {
        crate::loader::before_control_load(&control, true)
    });
    assert_refused_at(crate::quantum::CONTROL_LOAD, 640, || {
        crate::loader::before_control_load(&control, false)
    });
    // The text encoder: a resident load reads every tensor; a streamable one only the embedding.
    let te_dir = root.join("te");
    let contract = crate::bounded_encoder_contract();
    gen_core_testkit::write_encoder_contract_fixture(&te_dir.join("text_encoder"), contract)
        .unwrap();
    let file = te_dir.join("text_encoder/model.safetensors");
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&file).unwrap();
    let all: u64 = headers.iter().map(|h| h.data_bytes).sum();
    let embed: u64 = headers
        .iter()
        .filter(|h| !h.name.starts_with("model.layers."))
        .map(|h| h.data_bytes)
        .sum();
    assert!(embed < all);
    let source = WeightsSource::Dir(te_dir.join("text_encoder"));
    assert_refused_at(crate::quantum::TE_LOAD, all, || {
        crate::loader::load_text_encoder_from_source(&source)
    });
    let validated = contract.validate_source(&source).unwrap();
    assert_refused_at(crate::quantum::TE_LOAD, embed, || {
        crate::loader::load_validated_text_encoder_streamable(&validated)
    });
}

/// The quanta outside the host paths' forwards: the ControlNet's load-time packing, the latents,
/// the decoded-image conversion and the caption features — each refused before it is built.
#[test]
fn pipeline_quanta_refuse_before_construction() {
    if !in_cpu_child("pipeline_quanta_refuse_before_construction") {
        return;
    }
    // ControlNet packing: refused, and the block stays dense (nothing was built).
    let w = Weights::from_file(fixture("z_control_transformer")).unwrap();
    let mut block = crate::control_transformer_block::ZImageControlBlock::from_weights(
        &w,
        "w.control_layers.0",
        control_cfg().block_cfg(),
        true,
    )
    .unwrap();
    let price = block.quantize_bytes(4);
    assert!(price > 0);
    assert_refused_at(crate::quantum::CONTROL_QUANTIZE, price, || {
        block.quantize_quantum(4)
    });
    assert!(
        block.weight_arrays().len() < 4 * 10,
        "the block must still be dense"
    );
    // Latents: refused before the noise is drawn or the denoise runs.
    let req = mlx_gen::GenerationRequest {
        prompt: "p".into(),
        width: 64,
        height: 64,
        count: 1,
        ..Default::default()
    };
    let scheduler = mlx_gen::FlowMatchEuler::for_static_shift(4, 3.0);
    let denoised = std::cell::Cell::new(false);
    assert_refused_at(
        crate::quantum::LATENTS,
        crate::quantum::latents_bytes(16 * 8 * 8, false),
        || {
            crate::pipeline::denoise_batch(&scheduler, None, 0, 1, &req, &mut |_| {}, |x, _, _| {
                denoised.set(true);
                Ok(x)
            })
        },
    );
    assert!(!denoised.get());
    // The decoded-image conversion.
    let decoded = rand(&[1, 3, 1, 16, 16], 3, Dtype::Float32);
    struct Fixed(Array);
    impl mlx_gen::LatentDecoder for Fixed {
        fn input_latent_space(&self) -> Option<&mlx_gen::gen_core::LatentSpace> {
            Some(&mlx_gen::gen_core::FLUX1_LATENT_SPACE)
        }
        fn decode(&self, _: &Array) -> mlx_gen::Result<Array> {
            Ok(self.0.clone())
        }
    }
    let fixed = Fixed(decoded.clone());
    assert_refused_at(
        crate::quantum::DECODED_TO_IMAGE,
        crate::quantum::zimage_pixel_conversion_bytes(decoded.size() as u64, 4),
        || {
            crate::pipeline::decode_batch(
                &fixed,
                None,
                vec![rand(&[16, 1, 2, 2], 4, Dtype::Float32)],
                &CancelFlag::default(),
                &mut |_| {},
            )
        },
    );
    // The caption features' gather and cast.
    let enc = rand(&[1, 8, 64], 5, Dtype::Float32);
    let built = std::cell::Cell::new(false);
    assert_refused_at(
        crate::quantum::CAPTION_FEATURES,
        crate::quantum::ab(4, &[5]) + crate::quantum::ab(4, &[5, 64]),
        || {
            crate::pipeline::caption_quantum(&enc, 5, |e| {
                built.set(true);
                crate::pipeline::slice_valid(e, 5)
            })
        },
    );
    assert!(!built.get());
    let cap = rand(&[5, 64], 6, Dtype::Float32);
    assert_refused_at(
        crate::quantum::CAPTION_FEATURES,
        crate::quantum::ab(2, &[cap.size() as u64]),
        || crate::pipeline::caption_cast_quantum(&cap),
    );
}

/// Streamed (rung-4) blocks and layers are priced with what they read inside their quantum: a
/// windowed forward's block checks carry the stored bytes of exactly the tensors each block reads
/// on top of the resident price.
#[test]
fn streamed_quanta_are_priced_with_the_bytes_they_read() {
    if !in_cpu_child("streamed_quanta_are_priced_with_the_bytes_they_read") {
        return;
    }
    let model = dit();
    let w = Weights::from_file(fixture("denoise_loop")).unwrap();
    let x = w.require("init").unwrap();
    let cap = w.require("cap_feats").unwrap();
    let cancel = CancelFlag::default();
    let (resident, r_events) = observed(AMPLE, || {
        model.forward_budgeted(x, 0.6, cap, AttentionPlan::UNBOUNDED)
    });
    let (windowed, w_events) = observed(AMPLE, || {
        model.forward_windowed(x, 0.6, cap, AttentionPlan::UNBOUNDED, 1, &cancel)
    });
    // The same graph over the same weight values (only where the blocks' arrays come from
    // differs), so this one holds bit for bit on any host.
    assert_eq!(bits(&resident.unwrap()), bits(&windowed.unwrap()));
    let blocks = |events: &[QuantumEvent]| -> Vec<u64> {
        checks(events)
            .into_iter()
            .filter(|(l, _)| l == crate::quantum::DIT_BLOCK)
            .map(|(_, s)| s)
            .collect()
    };
    let (r, wn) = (blocks(&r_events), blocks(&w_events));
    assert_eq!(r.len(), wn.len());
    // Block 0 is the (resident) noise refiner; the two main blocks are streamed.
    assert_eq!(r[0], wn[0]);
    let headers =
        mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(fixture("denoise_loop"))
            .unwrap();
    for (i, n) in [(1usize, 0), (2, 1)] {
        let read: u64 = headers
            .iter()
            .filter(|h| h.name.starts_with(&format!("w.layers.{n}.")))
            .map(|h| h.data_bytes)
            .sum();
        assert!(read > 0);
        assert_eq!(wn[i], r[i] + read, "main block {n}");
    }
    // The text encoder's streamed layers likewise.
    let dir = tempfile::tempdir().unwrap();
    let te = streamable_text_encoder(dir.path());
    let ids = Array::from_slice(&[151_644i32, 7, 9, 11], &[1, 4]);
    let mask = Array::from_slice(&[1i32, 1, 1, 1], &[1, 4]);
    let (_, events) = observed(AMPLE, || te.forward_windowed(&ids, &mask, 1, &cancel));
    let layers: Vec<u64> = checks(&events)
        .into_iter()
        .filter(|(l, _)| l == crate::quantum::TE_LAYER)
        .map(|(_, s)| s)
        .collect();
    assert_eq!(layers.len(), 3);
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(
        dir.path().join("text_encoder/model.safetensors"),
    )
    .unwrap();
    // The same encoder built resident: its layer checks are the windowed ones minus the read.
    let w = Weights::from_dir(dir.path().join("text_encoder")).unwrap();
    let resident = TextEncoder::from_weights(
        &w,
        "model",
        &ZTextEncoderConfig {
            vocab_size: 151_936,
            hidden_size: 64,
            n_layers: 3,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 32,
            intermediate_size: 128,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        },
    )
    .unwrap();
    let (_, events) = observed(AMPLE, || resident.forward(&ids, &mask));
    let resident_layers: Vec<u64> = checks(&events)
        .into_iter()
        .filter(|(l, _)| l == crate::quantum::TE_LAYER)
        .map(|(_, s)| s)
        .collect();
    assert_eq!(resident_layers.len(), 3);
    for (i, (streamed, resident)) in layers.iter().zip(&resident_layers).enumerate() {
        let read: u64 = headers
            .iter()
            .filter(|h| h.name.starts_with(&format!("model.layers.{i}.")))
            .map(|h| h.data_bytes)
            .sum();
        assert!(read > 0);
        assert_eq!(*streamed, resident + read, "layer {i}");
    }
}

/// The ControlNet route's main blocks are priced with the hint they add inside their quantum (the
/// scaled f32 hint and the sum), resident and streamed alike, and its streamed control blocks with
/// the bytes they read: the control forward's block checks exceed the base forward's by exactly the
/// hint at every injection place and by nothing elsewhere.
#[test]
fn control_quanta_are_priced_with_their_hints_and_reads() {
    if !in_cpu_child("control_quanta_are_priced_with_their_hints_and_reads") {
        return;
    }
    let (control, w) = control_dit();
    let x = w.require("in.x").unwrap();
    let cap = w.require("in.cap_feats").unwrap();
    let cc = w.require("in.control_context").unwrap();
    let cancel = CancelFlag::default();
    let scratches = |events: &[QuantumEvent], label: &str| -> Vec<u64> {
        checks(events)
            .into_iter()
            .filter(|(l, _)| l == label)
            .map(|(_, s)| s)
            .collect()
    };
    let plan = AttentionPlan::UNBOUNDED;
    for windowed in [false, true] {
        let run = |context: Option<&Array>| {
            observed(AMPLE, || {
                if windowed {
                    control.forward_windowed(x, 0.7, cap, context, 0.8, plan, 1, &cancel)
                } else {
                    control.forward_budgeted(x, 0.7, cap, context, 0.8, plan)
                }
            })
        };
        let (_, base) = run(None);
        let (_, ctl) = run(Some(cc));
        let (base, ctl) = (
            scratches(&base, crate::quantum::DIT_BLOCK),
            scratches(&ctl, crate::quantum::DIT_BLOCK),
        );
        // 2 noise-refiner blocks then 4 main blocks; hints at refiner 0, 1 and main 0, 2.
        assert_eq!(base.len(), 6);
        assert_eq!(ctl.len(), 6);
        let hint = |rows: u64| {
            let units = if crate::compile_glue() { 1 } else { 2 };
            units * crate::quantum::tb(4, &[1, rows, 64]) + mlx_gen::ios_quantum::page_slack(units)
        };
        let (n, l) = (32, 64);
        for (i, rows, hinted) in [
            (0usize, n, true),
            (1, n, true),
            (2, l, true),
            (3, l, false),
            (4, l, true),
            (5, l, false),
        ] {
            let want = if hinted { hint(rows) } else { 0 };
            assert_eq!(ctl[i] - base[i], want, "windowed={windowed} block {i}");
        }
    }
    // Streamed control blocks carry their stored bytes on top of the resident price.
    let (_, resident) = observed(AMPLE, || {
        control.forward_budgeted(x, 0.7, cap, Some(cc), 0.8, plan)
    });
    let (_, streamed) = observed(AMPLE, || {
        control.forward_windowed(x, 0.7, cap, Some(cc), 0.8, plan, 1, &cancel)
    });
    let (r, s) = (
        scratches(&resident, crate::quantum::CONTROL_BLOCK),
        scratches(&streamed, crate::quantum::CONTROL_BLOCK),
    );
    assert_eq!(r.len(), 17);
    assert_eq!(s.len(), 17);
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(fixture(
        "z_control_transformer",
    ))
    .unwrap();
    for i in 0..15 {
        let read: u64 = headers
            .iter()
            .filter(|h| h.name.starts_with(&format!("w.control_layers.{i}.")))
            .map(|h| h.data_bytes)
            .sum();
        // The refiner's 2 blocks run first (resident in both); then the 15 main control blocks.
        assert_eq!(s[2 + i], r[2 + i] + read, "control block {i}");
    }
}
