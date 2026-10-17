//! iOS bounded-quantum guards on the Z-Image inference path (epic 16774, sc-26433), exercised on
//! tiny CPU models through the diagnostic fixture (`mlx_gen::ios_quantum::with_cpu_diagnostic_headroom`).
//! Every test re-executes itself in a child process pinned to the CPU device with Metal made fatal,
//! so nothing here can touch the GPU.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Result, WeightsSource};
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
                &rand(&shape, 1000 + i as u64, Dtype::Float32),
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
        arrays.iter().map(|(k, a)| (k.as_str(), a)).collect::<Vec<_>>(),
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
            n_layers: contract.num_hidden_layers as usize,
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

/// The tiny VAE decoder fixture plus a seeded random two-block encoder (32 → 64 channels) that
/// downsamples by 2 into the decoder's 16 latent channels.
fn vae() -> Vae {
    let mut w = Weights::from_file(fixture("vae_decoder")).unwrap();
    let mut seed = 2000u64;
    let mut put = |w: &mut Weights, key: String, shape: &[i32], scale: f32| {
        seed += 1;
        let a = mlx_rs::ops::multiply(&rand(shape, seed, Dtype::Float32), Array::from_f32(scale))
            .unwrap();
        a.eval().unwrap();
        w.insert(key, a);
    };
    let conv = |w: &mut Weights, put: &mut dyn FnMut(&mut Weights, String, &[i32], f32), p: &str, o: i32, i: i32, k: i32| {
        put(w, format!("{p}.weight"), &[o, k, k, i], (1.0 / (i * k * k) as f32).sqrt());
        put(w, format!("{p}.bias"), &[o], 0.02);
    };
    let norm = |w: &mut Weights, p: &str, c: i32| {
        w.insert(format!("{p}.weight"), Array::ones::<f32>(&[c]).unwrap());
        w.insert(format!("{p}.bias"), Array::zeros::<f32>(&[c]).unwrap());
    };
    let resnet = |w: &mut Weights, put: &mut dyn FnMut(&mut Weights, String, &[i32], f32), p: &str, i: i32, o: i32| {
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
    conv(&mut w, &mut put, "encoder.down_blocks.0.downsamplers.0.conv", 32, 32, 3);
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
    let mut out = Vec::new();
    let cancel = CancelFlag::default();
    // Text encoder: resident, and the rung-4 windowed stream (window 1).
    let (te, w) = text_encoder();
    let ids = w.require("input_ids").unwrap();
    let mask = w.require("attention_mask").unwrap();
    out.push(("te_resident", te.forward(ids, mask).unwrap()));
    let dir = tempfile::tempdir().unwrap();
    let streamed = streamable_text_encoder(dir.path());
    let ids = Array::from_slice(&[151_644i32, 7, 9, 11, 151_645, 151_643], &[1, 6]);
    let mask = Array::from_slice(&[1i32, 1, 1, 1, 1, 0], &[1, 6]);
    out.push((
        "te_windowed",
        streamed.forward_windowed(&ids, &mask, 1, &cancel).unwrap(),
    ));
    out.push(("te_streamed_unscoped", streamed.forward(&ids, &mask).unwrap()));

    // DiT: unbounded, chunked attention, and the rung-4 window (f32 and bf16 latents).
    let model = dit();
    let w = Weights::from_file(fixture("denoise_loop")).unwrap();
    let x = w.require("init").unwrap();
    let cap = w.require("cap_feats").unwrap();
    out.push((
        "dit_unbounded",
        model
            .forward_budgeted(x, 0.6, cap, AttentionPlan::UNBOUNDED)
            .unwrap(),
    ));
    let chunked = AttentionPlan::budgeted(AttentionBudget::from_score_elements(4 * 16, true));
    out.push((
        "dit_chunked",
        model.forward_budgeted(x, 0.6, cap, chunked).unwrap(),
    ));
    out.push((
        "dit_windowed",
        model
            .forward_windowed(x, 0.6, cap, AttentionPlan::UNBOUNDED, 1, &cancel)
            .unwrap(),
    ));
    let xb = x.as_dtype(Dtype::Bfloat16).unwrap();
    let capb = cap.as_dtype(Dtype::Bfloat16).unwrap();
    out.push((
        "dit_bf16_windowed",
        model
            .forward_windowed(&xb, 0.6, &capb, chunked, 1, &cancel)
            .unwrap(),
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
        )
        .unwrap(),
    ));

    // Control DiT: resident and windowed.
    let (control, cw) = control_dit();
    let cx = cw.require("in.x").unwrap();
    let ccap = cw.require("in.cap_feats").unwrap();
    let cc = cw.require("in.control_context").unwrap();
    out.push((
        "control_resident",
        control
            .forward_budgeted(cx, 0.7, ccap, Some(cc), 0.8, AttentionPlan::UNBOUNDED)
            .unwrap(),
    ));
    out.push((
        "control_windowed",
        control
            .forward_windowed(cx, 0.7, ccap, Some(cc), 0.8, chunked, 1, &cancel)
            .unwrap(),
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
    out.push(("vae_decode", vae.decode(&latent5).unwrap()));
    for (name, edge) in [("vae_decode_tiled_8", 8), ("vae_decode_tiled_12", 12)] {
        out.push((
            name,
            vae.decode_tiled(
                &latent5,
                &mlx_gen::tiling::TilingConfig::spatial_only(edge, 0),
                Some(&cancel),
            )
            .unwrap(),
        ));
    }
    let image = rand(&[1, 3, 16, 16], 77, Dtype::Float32);
    out.push(("vae_encode", vae.encode(&image).unwrap()));
    let rgb = mlx_gen::Image {
        width: 16,
        height: 16,
        pixels: (0..16 * 16 * 3).map(|i| ((i * 37) % 251) as u8).collect(),
    };
    out.push((
        "init_latents",
        crate::pipeline::encode_init_latents(&vae, &rgb, 16, 16).unwrap(),
    ));
    out.push((
        "control_context",
        crate::pipeline::encode_control_context(&vae, &rgb, 16, 16).unwrap(),
    ));
    out
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
