#![cfg(test)]

use super::*;
use mlx_gen::train::perceptual::{AuxLoss, AuxLossSchedule};
use mlx_gen::train::taehv::{synthetic_taehv_weights, TaehvDecoder};
use mlx_gen_depth::anchor::{synthetic_weights, tiny_config, DepthAnchorLoss};
use mlx_gen_depth::DepthAnythingV2;

struct DeviceGuard(mlx_rs::Device);
impl Drop for DeviceGuard {
    fn drop(&mut self) {
        mlx_rs::Device::set_default(&self.0);
    }
}
fn cpu_only() -> DeviceGuard {
    let guard = DeviceGuard(mlx_rs::Device::try_default().expect("default device"));
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    guard
}

fn schedule() -> AuxLossSchedule {
    AuxLossSchedule {
        weight: 0.1,
        t_min: 0.0,
        t_max: 1.0,
        every_n: 2,
    }
}

fn tiny_decoder() -> TaehvDecoder {
    let tae = TaehvConfig {
        latent_channels: 4,
        channels: [8, 6, 4, 4],
        ..TaehvConfig::taeltx2_3()
    };
    TaehvDecoder::from_weights(&synthetic_taehv_weights(&tae, 11).unwrap(), tae).unwrap()
}

fn path() -> PerceptualPath {
    let da2 = tiny_config();
    let depth = DepthAnchorLoss::new(
        DepthAnythingV2::from_weights(&synthetic_weights(&da2, 12).unwrap(), da2).unwrap(),
    );
    PerceptualPath::new(
        Some(Box::new(Ltx25DepthDecoder(tiny_decoder()))),
        vec![AuxLoss {
            schedule: schedule(),
            loss: Box::new(depth),
        }],
    )
    .unwrap()
}

fn rnd(shape: &[i32], seed: u64) -> Array {
    let a = random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap())).unwrap();
    eval([&a]).unwrap();
    a
}

struct Fixture {
    dit: AvDiT,
    targets: Vec<LtxLoraTarget>,
    params: LoraParams,
    batch: Ltx25PreparedBatch,
    clean_video: Array,
    depth: Ltx25DepthFrames,
}

/// `video_plan` decides which video tokens generate; `with_audio` adds the generated audio
/// stream (the joint AV forward) or leaves the video-only forward.
fn fixture(video_plan: Ltx25ModalityPlan, with_audio: bool) -> Fixture {
    let mut cfg = crate::transformer::rung4_block_window_tests::tiny_cfg();
    cfg.ff_bias = false;
    let weights =
        Weights::from_map(crate::transformer::rung4_block_window_tests::tiny_weight_map(&cfg));
    let mut dit = AvDiT::from_weights(&weights, &cfg, Precision::quant_f32(8, 64)).unwrap();
    let suffixes = ["to_q", "to_k", "to_v", "to_out.0"].map(str::to_string);
    let (targets, params) = build_targets_25(&mut dit, cfg.num_layers, &suffixes, 2, 17).unwrap();
    let (f, h, w) = (2usize, 2usize, 2usize);
    let tokens = (f * h * w) as i32;
    let clean_video = rnd(&[1, tokens, 4], 31);
    let video_plan = build_ltx25_token_plan(
        video_plan,
        Ltx25TokenGeometry {
            frames: f,
            height: h,
            width: w,
            spatial_scale: 32,
        },
        Ltx25ConditionInputs::default(),
    )
    .unwrap();
    let video = prepare_ltx25_modality(
        &clean_video,
        &rnd(&[1, tokens, 4], 32),
        &rnd(&[1, 3, 24], 33),
        &create_position_grid(1, f, h, w),
        None,
        None,
        0.4,
        &video_plan,
    )
    .unwrap();
    let audio = with_audio.then(|| {
        let audio_plan = build_ltx25_token_plan(
            GENERATED,
            Ltx25TokenGeometry {
                frames: 4,
                height: 1,
                width: 1,
                spatial_scale: 1,
            },
            Ltx25ConditionInputs::default(),
        )
        .unwrap();
        prepare_ltx25_modality(
            &rnd(&[1, 4, 4], 34),
            &rnd(&[1, 4, 4], 35),
            &rnd(&[1, 3, 8], 36),
            &create_audio_position_grid(1, 4),
            None,
            None,
            0.4,
            &audio_plan,
        )
        .unwrap()
    });
    let frames = select_depth_frames(
        &video_plan.loss_mask[..video_plan.target_tokens],
        f,
        h * w,
        2,
    );
    Fixture {
        dit,
        targets,
        params,
        batch: Ltx25PreparedBatch {
            video: Some(video),
            audio,
        },
        clean_video,
        depth: Ltx25DepthFrames {
            grid: [4, f as i32, h as i32, w as i32],
            frames,
        },
    }
}

fn step(f: &mut Fixture, aux: Option<Ltx25AuxStep<'_>>) -> (StepLosses, LoraParams) {
    let (l, g) = compute_ltx25_step_loss_grads(
        &mut f.dit,
        &f.params,
        &f.targets,
        2.0,
        2.0,
        &f.batch,
        false,
        Dtype::Float32,
        aux,
    )
    .unwrap();
    eval(g.values()).unwrap();
    (l, g)
}

fn bits(a: &Array) -> Vec<u32> {
    a.as_slice::<f32>().iter().map(|x| x.to_bits()).collect()
}

/// AC (a)+(b) on the joint AV forward (T2V) and the video-only forward (inpainting-shaped): a
/// depth step (key 2) computes no diffusion term, total IS the depth term, the zero-init LoRA-B
/// factors get a nonzero finite gradient through the decoded video x0; a diffusion step (key 1)
/// carries no depth term. Mutation: force `diffusion_on = true` ⇒ red.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn depth_step_trains_the_lora_through_the_video_stream_only() {
    let _cpu = cpu_only();
    for with_audio in [true, false] {
        let mut f = fixture(GENERATED, with_audio);
        let mut p = path();
        p.ensure_reference(0, &f.depth.nchw(&f.clean_video).unwrap())
            .unwrap();
        let depth = f.depth.clone();
        let plan = p.plan(1, 0, 0.4).unwrap();
        let (diff, _) = step(
            &mut f,
            Some(Ltx25AuxStep {
                path: &p,
                plan: &plan,
                image: 0,
                depth: &depth,
            }),
        );
        assert_eq!(diff.aux, None, "audio={with_audio}");
        assert_eq!(Some(diff.total), diff.diffusion);
        let plan = p.plan(2, 0, 0.4).unwrap();
        assert!(!plan.diffusion);
        let (d, g) = step(
            &mut f,
            Some(Ltx25AuxStep {
                path: &p,
                plan: &plan,
                image: 0,
                depth: &depth,
            }),
        );
        assert_eq!(d.diffusion, None, "audio={with_audio}");
        let a = d.aux.expect("depth term");
        assert!(a > 0.0 && a.is_finite(), "audio={with_audio}: {a}");
        assert_eq!(d.total, a);
        let gb: f32 = g
            .iter()
            .filter(|(k, _)| k.ends_with(".lora_b"))
            .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
            .sum();
        assert!(
            gb > 0.0 && gb.is_finite(),
            "audio={with_audio}: LoRA-B grad {gb}"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::ltx25_depth_anchoring_tests::depth_step_trains_the_lora_through_the_video_stream_only");
}

/// AC (c): depth off ⇒ bit-identical to the pre-epic-2123 LTX-2.5 step (its closure reproduced),
/// and a diffusion-only planned step equals it too. Mutation: scale the joint loss (×1.0001) ⇒
/// red.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn depth_off_is_bit_identical_to_the_legacy_av_step() {
    let _cpu = cpu_only();
    let mut f = fixture(GENERATED, true);
    let (off, g_off) = step(&mut f, None);
    assert_eq!(off.aux, None);
    let batch = f.batch.clone();
    let dit = &mut f.dit;
    let targets = &f.targets;
    let legacy = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        install_train_lora(dit, &p, targets, 2.0, 2.0, None)?;
        let (video, audio) = (batch.video.as_ref().unwrap(), batch.audio.as_ref().unwrap());
        let (vp, ap) = dit
            .forward(
                &video.noisy,
                &video.timestep,
                &video.context,
                None,
                &video.positions,
                &audio.noisy,
                &audio.timestep,
                &audio.context,
                None,
                &audio.positions,
                None,
                None,
            )
            .map_err(|e| Exception::custom(e.to_string()))?;
        Ok(vec![add(
            &masked_modality_loss(&vp, video, false)?,
            &masked_modality_loss(&ap, audio, false)?,
        )?])
    };
    let (val, g_legacy) = keyed_value_and_grad(legacy)(f.params.clone(), 0).unwrap();
    eval(g_legacy.values()).unwrap();
    assert_eq!(off.total, val[0].item::<f32>());
    for (k, v) in &g_legacy {
        assert_eq!(bits(v), bits(&g_off[k]), "{k}");
    }
    let mut p = path();
    p.ensure_reference(0, &f.depth.nchw(&f.clean_video).unwrap())
        .unwrap();
    let depth = f.depth.clone();
    let plan = p.plan(1, 0, 0.4).unwrap();
    let (on, g_on) = step(
        &mut f,
        Some(Ltx25AuxStep {
            path: &p,
            plan: &plan,
            image: 0,
            depth: &depth,
        }),
    );
    assert_eq!(on, off);
    for (k, v) in &g_off {
        assert_eq!(bits(v), bits(&g_on[k]), "{k}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::ltx25_depth_anchoring_tests::depth_off_is_bit_identical_to_the_legacy_av_step");
}

/// The depth frames skip conditioning frames (I2V first frame) and pick evenly spaced generated
/// frames; `nchw` takes exactly those frames from the patchified tokens. Mutations: drop the
/// candidate filter ⇒ frame 0 is picked ⇒ red; take frames along the wrong axis ⇒ red.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn depth_frames_follow_the_generated_frames_and_the_token_layout() {
    let _cpu = cpu_only();
    // 4 frames × 2 tokens; frame 0 conditioning.
    let mask = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    assert_eq!(select_depth_frames(&mask, 4, 2, 2), vec![1, 3]);
    assert_eq!(select_depth_frames(&mask, 4, 2, 1), vec![2]);
    assert_eq!(select_depth_frames(&mask, 4, 2, 9), vec![1, 2, 3]);
    assert!(select_depth_frames(&[0.0; 8], 4, 2, 2).is_empty());
    // A first-frame condition that always fires: frame 0 of the 2-frame clip is conditioning.
    const FIRST_ALWAYS: &[Ltx25ConditionSpec] = &[Ltx25ConditionSpec::simple(
        Ltx25ConditionKind::FirstFrame,
        1.0,
    )];
    let i2v = fixture(generated(FIRST_ALWAYS), false);
    assert_eq!(i2v.depth.frames, vec![1]);

    let grid = rnd(&[1, 3, 4, 2, 2], 40);
    let tokens = crate::conditioning::patchify_grid(&grid).unwrap();
    use mlx_rs::ops::indexing::IndexOp;
    // (selected frames, the grid frame each decode-batch row must hold): each selected
    // `k > 0` preceded by `k − 1`, frame 0 alone.
    for (selected, decoded) in [(vec![1, 3], vec![0, 1, 2, 3]), (vec![0, 3], vec![0, 2, 3])] {
        let sel = Ltx25DepthFrames {
            grid: [3, 4, 2, 2],
            frames: selected.clone(),
        };
        assert_eq!(sel.decode_frames().unwrap(), decoded);
        let out = sel.nchw(&tokens).unwrap();
        assert_eq!(
            out.shape(),
            &[decoded.len() as i32, 3, 2, 2],
            "{selected:?}"
        );
        for (i, &fr) in decoded.iter().enumerate() {
            // `as_slice` reads the physical buffer: compare row-major copies.
            let want =
                mlx_gen::array::contiguous(&grid.index((.., .., fr)).reshape(&[3, 2, 2]).unwrap())
                    .unwrap();
            let got = mlx_gen::array::contiguous(&out.index(i as i32).reshape(&[3, 2, 2]).unwrap())
                .unwrap();
            eval([&want, &got]).unwrap();
            assert_eq!(
                bits(&want),
                bits(&got),
                "{selected:?}: row {i} = frame {fr}"
            );
        }
    }
    let bad = Ltx25DepthFrames {
        grid: [3, 4, 2, 2],
        frames: vec![2, 0],
    };
    assert!(bad.decode_frames().is_err());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::ltx25_depth_anchoring_tests::depth_frames_follow_the_generated_frames_and_the_token_layout");
}

/// `Ltx25DepthDecoder` decodes one pixel frame per selected frame: frame 0 exactly as a lone
/// `T = 1` clip, a frame `k > 0` exactly as the last output frame of the reference
/// `decode_video` of the clip `[k − 1, k]` (≠ decoding `k` alone, the out-of-distribution
/// path it replaces), and the gradient reaches both the selected frame and its predecessor.
/// Mutations: decode every row as an independent `T = 1` frame ⇒ red; drop the lone-frame-0
/// branch (treat the batch as pairs only) ⇒ red.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn depth_decoder_decodes_each_frame_with_its_predecessor() {
    let _cpu = cpu_only();
    use mlx_rs::ops::indexing::IndexOp;
    let dec = Ltx25DepthDecoder(tiny_decoder());
    let grid = rnd(&[1, 4, 3, 2, 2], 41);
    let tokens = crate::conditioning::patchify_grid(&grid).unwrap();
    let sel = Ltx25DepthFrames {
        grid: [4, 3, 2, 2],
        frames: vec![0, 2],
    };
    let input = sel.nchw(&tokens).unwrap();
    let frame = |k: i32| grid.index((.., .., k)).reshape(&[1, 4, 2, 2]).unwrap();
    let diff = |a: &Array, b: &Array| {
        let d = subtract(a, b).unwrap().abs().unwrap().max(None).unwrap();
        d.item::<f32>()
    };
    let px = dec.decode(&input).unwrap();
    let up = 2 * dec.0.config().spatial_upscale();
    assert_eq!(
        px.shape(),
        &[2, up, up, 3],
        "one pixel frame per selected frame"
    );
    let f0 = dec.0.decode_frames(&frame(0)).unwrap();
    assert!(diff(&px.index(..1), &f0) < 1e-6);
    let clip = concatenate_axis(&[&frame(1), &frame(2)], 0)
        .unwrap()
        .reshape(&[1, 2, 4, 2, 2])
        .unwrap();
    let video = dec.0.decode_video(&clip).unwrap();
    let t = video.shape()[1];
    let last = video.index((.., t - 1)).reshape(&[1, up, up, 3]).unwrap();
    let k2 = px.index(1..);
    assert!(diff(&k2, &last) < 1e-5);
    let alone = dec.0.decode_frames(&frame(2)).unwrap();
    assert!(
        diff(&k2, &alone) > 1e-3,
        "the predecessor must reach frame 2's decode"
    );
    let g = mlx_rs::transforms::grad(|z: &Array| -> MlxResult<Array> {
        dec.decode(z)
            .map_err(|e| Exception::custom(e.to_string()))?
            .sum(None)
    })(&input)
    .unwrap();
    for row in 0..3 {
        let m = g.index(row).abs().unwrap().sum(None).unwrap().item::<f32>();
        assert!(m > 0.0 && m.is_finite(), "row {row}: {m}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::ltx25_depth_anchoring_tests::depth_decoder_decodes_each_frame_with_its_predecessor");
}

/// The depth memory estimate's video grid comes from each bundle's safetensors HEADER alone:
/// bundles whose tensor data is absent (header only, data truncated) still yield the max
/// `(F, H, W)` over items. Mutation: open the bundle with `Weights::from_file` (loads the
/// tensors) ⇒ the truncated bundles fail to load ⇒ red.
#[test]
fn video_grids_read_only_the_bundle_header() {
    let tmp = tempfile::tempdir().unwrap();
    let bundle = |name: &str, shape: [i32; 5]| {
        let n: i32 = shape.iter().product();
        let header = serde_json::json!({
            "__metadata__": { "videoShape": serde_json::to_string(&shape).unwrap() },
            "video_latents": {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [0, n * 4],
            },
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        // No tensor bytes: only a header-only reader can read this file.
        let path = tmp.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        let mut item = super::validate_request_tests::request(1).items.remove(0);
        item.model_options.insert(
            "ltxPreparedBundlePath".into(),
            serde_json::json!(path.to_str().unwrap()),
        );
        item
    };
    let items = [
        bundle("a.safetensors", [1, 4, 3, 5, 7]),
        bundle("b.safetensors", [1, 4, 9, 2, 8]),
    ];
    assert_eq!(ltx25_video_grids(&items).unwrap(), (9, 5, 8));
}

/// AC (d), E7: the LTX-2.5 depth footprint grows with the decoded frames and with the Large
/// DA2, sizes each selected frame's decode as a `T = 2` clip (its predecessor rides along:
/// exactly one extra clip-frame of TAELTX2.3 working set per selected frame over the `T = 1`
/// sizing), and the guard refuses at a synthetic budget between base and base+aux. Mutations:
/// drop `+ extra_gb` from the guard ⇒ red; size the LTX-2.5 decode as `T = 1` ⇒ red.
#[test]
fn memory_estimate_grows_with_frames_and_the_guard_counts_it() {
    let mut on = TrainingConfig::default();
    on.depth_anchoring.schedule = schedule();
    let geom = mlx_gen_perceptual::AuxGeometry {
        height: 512,
        width: 512,
        frames: 2,
        entries: 2,
    };
    let tae = TaehvConfig::taeltx2_3();
    let extra =
        tae.clip_training_working_set_bytes(512, 512, 2) - tae.training_working_set_bytes(512, 512);
    assert!(extra > 0);
    assert_eq!(
        mlx_gen_perceptual::perceptual_footprint(&on, &ltx25_decoder(), geom)
            - mlx_gen_perceptual::perceptual_footprint(&on, &ltx_decoder(), geom),
        2 * extra
    );
    let one = ltx25_perceptual_footprint_gb(&on, 16, 16, 1, 2);
    let two = ltx25_perceptual_footprint_gb(&on, 16, 16, 2, 2);
    assert!(one > 0.0 && two > one, "{one} {two}");
    on.depth_anchoring.model_size = gen_core::train::DepthModelSize::Large;
    let large = ltx25_perceptual_footprint_gb(&on, 16, 16, 2, 2);
    assert!(large - two > 1.0, "{two} {large}");
    assert_eq!(
        ltx25_perceptual_footprint_gb(&TrainingConfig::default(), 16, 16, 2, 2),
        0.0
    );
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let between = ((checkpointed_baseline_gb() + large / 2.0) / 0.85 * GIB) as usize;
    assert!(ltx25_aux_memory_guard(&on, 0.0, between).is_ok());
    let err = ltx25_aux_memory_guard(&on, large, between)
        .unwrap_err()
        .to_string();
    // The refusal names the enabled aux losses (the shared E7 guard).
    assert!(err.contains("[depth]"), "{err}");
}

/// AC (e): the LTX-2.5 descriptor declares depth anchoring; exactly the workflows with no
/// generated video (V2A's frozen video, and the five audio-only workflows) are refused with a
/// typed `Unsupported` naming the workflow — from the shared refusal and the weights-free
/// preflight — and every other workflow passes the refusal; a malformed frame knob is an error.
/// Mutations: refuse nothing (declare-only) ⇒ red; refuse on `video.is_some()` instead of
/// generated ⇒ V2A passes ⇒ red.
#[test]
fn video_less_workflows_are_refused_by_name() {
    assert!(trainer_descriptor_25().techniques.depth_anchoring);
    let refused = [
        "v2a_lora",
        "t2a_lora",
        "audio_extend_lora",
        "audio_inpainting_lora",
        "audio_suffix_lora",
        "a2a_ic_lora",
    ];
    for workflow in LTX25_WORKFLOWS {
        let mut req = super::validate_request_tests::request(1);
        req.config
            .model_options
            .insert("ltxWorkflow".into(), serde_json::json!(workflow.id()));
        req.config.depth_anchoring.schedule = schedule();
        req.config.depth_anchoring.model_dir = Some("/m/da2".into());
        req.config.perceptual_decoder_dir = Some("/m/taehv".into());
        let result = refuse_ltx25_depth_anchoring(MODEL_25_ID, &req);
        if refused.contains(&workflow.id()) {
            match result {
                Err(gen_core::Error::Unsupported(m)) => {
                    assert!(m.contains(workflow.id()) && m.contains("no video"), "{m}")
                }
                other => panic!("{}: expected Unsupported, got {other:?}", workflow.id()),
            }
            match validate_ltx25_training_request(&req) {
                Err(mlx_gen::Error::Unsupported(m)) => {
                    assert!(m.contains(workflow.id()), "{m}")
                }
                other => panic!("{}: preflight {other:?}", workflow.id()),
            }
        } else {
            assert!(result.is_ok(), "{}: {result:?}", workflow.id());
        }
        // sc-24833: the VAE anchor decodes the same frames — refused on the same workflows.
        req.config.depth_anchoring.schedule = AuxLossSchedule::OFF;
        req.config.vae_anchor.schedule = schedule();
        req.config.vae_anchor.model_dir = Some("/m/flux2-vae".into());
        assert_eq!(
            refuse_ltx25_depth_anchoring(MODEL_25_ID, &req).is_err(),
            refused.contains(&workflow.id()),
            "{}: VAE anchor",
            workflow.id()
        );
        // Every aux loss off: never refused.
        req.config.vae_anchor.schedule = AuxLossSchedule::OFF;
        assert!(refuse_ltx25_depth_anchoring(MODEL_25_ID, &req).is_ok());
        // sc-24831: the identity / face-landmark losses decode the same stream ⇒ the same
        // refusal, naming the loss. Mutation: gate on depth alone ⇒ red.
        req.config.identity_loss.schedule = schedule();
        req.config.face_landmark_loss.schedule = schedule();
        let face = refuse_ltx25_depth_anchoring(MODEL_25_ID, &req);
        if refused.contains(&workflow.id()) {
            match face {
                Err(gen_core::Error::Unsupported(m)) => {
                    assert!(m.contains("identity") && m.contains("face-landmark"), "{m}")
                }
                other => panic!("{}: expected Unsupported, got {other:?}", workflow.id()),
            }
        } else {
            assert!(face.is_ok(), "{}: {face:?}", workflow.id());
        }
        req.config.identity_loss.schedule = AuxLossSchedule::OFF;
        req.config.face_landmark_loss.schedule = AuxLossSchedule::OFF;
        // sc-24832: a body loss decodes the same video stream, so it is refused alike.
        // Mutation: check only depth in `refuse_ltx25_depth_anchoring` ⇒ red.
        req.config.body_losses.normal = schedule();
        assert_eq!(
            refuse_ltx25_depth_anchoring(MODEL_25_ID, &req).is_err(),
            refused.contains(&workflow.id()),
            "{} (body loss)",
            workflow.id()
        );
        req.config.body_losses.normal = AuxLossSchedule::OFF;
    }
    let mut req = super::validate_request_tests::request(1);
    req.config
        .model_options
        .insert("ltxWorkflow".into(), serde_json::json!("t2v_lora"));
    req.config.depth_anchoring.schedule = schedule();
    req.config
        .model_options
        .insert(DEPTH_ANCHORING_FRAMES_KEY.into(), serde_json::json!(0));
    assert!(refuse_ltx25_depth_anchoring(MODEL_25_ID, &req).is_err());
    req.config
        .model_options
        .insert(DEPTH_ANCHORING_FRAMES_KEY.into(), serde_json::json!(3));
    assert_eq!(depth_anchoring_frames(&req.config).unwrap(), 3);
    assert_eq!(
        depth_anchoring_frames(&TrainingConfig::default()).unwrap(),
        DEFAULT_DEPTH_ANCHORING_FRAMES
    );
}
