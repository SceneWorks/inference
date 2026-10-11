use super::*;
use crate::runtime::CancelFlag;
use serde_json::json;

fn tiny_config() -> IrisConfig {
    IrisConfig::parse(
        "model:\n  dual_depth: 2\n  hidden_size: 64\n  depth: 3\n  num_heads: 4\n  num_kv_heads: 2\n  \
         patch_size: 4\n  text_dim: 32\n  text_len: 10\n  text_lap_num_heads: 4\n  pixel:\n    \
         depth: 2\n    hidden_size: 8\n    attn_hidden_size: 32\n    num_heads: 2\n\
         text_encoder:\n  dim: 32\n  max_length: 10\n",
    )
    .unwrap()
}

fn base_cfg() -> TrainingConfig {
    TrainingConfig {
        resolution: 16,
        steps: 4,
        full_finetune: true,
        loss_type: "l1".into(),
        ..Default::default()
    }
}

fn item(dir: &Path, i: usize, w: u32, h: u32, depth: impl Fn(u32, u32) -> f32) -> TrainingItem {
    let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([x as u8, y as u8, 7]));
    let rgb = dir.join(format!("rgb{i}.png"));
    img.save(&rgb).unwrap();
    let path = dir.join(format!("depth{i}.npy"));
    write_npy_f32(&path, h as usize, w as usize, |y, x| {
        depth(x as u32, y as u32)
    });
    let mut it = TrainingItem::captioned(rgb, String::new());
    it.model_options.insert(
        ITEM_DEPTH_KEY.into(),
        json!({"path": path.to_str().unwrap()}),
    );
    it
}

fn write_npy_f32(path: &Path, h: usize, w: usize, f: impl Fn(usize, usize) -> f32) {
    let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({h}, {w}), }}");
    let mut header = header.into_bytes();
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&header);
    for y in 0..h {
        for x in 0..w {
            bytes.extend_from_slice(&f(y, x).to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

fn request(items: Vec<TrainingItem>, config: TrainingConfig) -> TrainingRequest {
    TrainingRequest {
        items,
        config,
        output_dir: "/out".into(),
        file_name: "depth.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: CancelFlag::default(),
    }
}

fn one_row(target: Vec<f32>, valid: Vec<f32>, h: usize, w: usize) -> DepthBatch {
    let n = valid.iter().filter(|&&v| v > 0.0).count();
    DepthBatch {
        batch: 1,
        height: h,
        width: w,
        rgb: vec![0.0; 3 * h * w],
        target,
        valid,
        valid_count: vec![n],
    }
}

/// Eq. 5 by hand on a 2×2 image: value 1+0+1+2 = 4, forward-difference pairs |0−1| twice in x and
/// |0−2| twice in y = 6, so (4 + 5·6) / 4 = 8.5; with the bottom-right pixel masked, its value term
/// and both pairs touching it drop: (2 + 5·3) / 3.
#[test]
fn eq5_matches_the_hand_derived_values() {
    let plan = DepthLossPlan::default();
    let pred = [1.0f32; 4];
    let full = one_row(vec![0.0, 1.0, 2.0, 3.0], vec![1.0; 4], 2, 2);
    assert!((depth_loss_host(&pred, &full, &plan) - 8.5).abs() < 1e-12);
    let masked = one_row(vec![0.0, 1.0, 2.0, 0.0], vec![1.0, 1.0, 1.0, 0.0], 2, 2);
    assert!((depth_loss_host(&pred, &masked, &plan) - 17.0 / 3.0).abs() < 1e-6);
    // The gradient term alone (λ only) and the value term alone (λ = 0).
    let value_only = DepthLossPlan {
        gradient_weight: 0.0,
        ..plan
    };
    assert!((depth_loss_host(&pred, &full, &value_only) - 1.0).abs() < 1e-12);
}

/// The analytic subgradient agrees with central differences away from the kinks.
#[test]
fn loss_gradient_matches_finite_differences() {
    let (h, w) = (3, 4);
    let mut rng = HostRng::new(5);
    let target: Vec<f32> = (0..h * w).map(|_| rng.normal() as f32).collect();
    let pred: Vec<f32> = (0..h * w).map(|_| rng.normal() as f32).collect();
    let mut valid = vec![1.0f32; h * w];
    valid[5] = 0.0;
    valid[10] = 0.0;
    for reduction in [LossReduction::ValidPixels, LossReduction::PerImage] {
        let plan = DepthLossPlan {
            reduction,
            ..DepthLossPlan::default()
        };
        let mut batch = one_row(target.clone(), valid.clone(), h, w);
        // a second, all-invalid row: weight 0 under both reductions
        batch.batch = 2;
        batch.rgb.extend(vec![0.0; 3 * h * w]);
        batch.target.extend(vec![0.0; h * w]);
        batch.valid.extend(vec![0.0; h * w]);
        batch.valid_count.push(0);
        let mut p2 = pred.clone();
        p2.extend(vec![0.3; h * w]);
        let g = depth_loss_grad_host(&p2, &batch, &plan);
        for i in 0..p2.len() {
            let eps = 1e-3f32;
            let mut a = p2.clone();
            a[i] += eps;
            let mut b = p2.clone();
            b[i] -= eps;
            let fd = (depth_loss_host(&a, &batch, &plan) - depth_loss_host(&b, &batch, &plan))
                / (2.0 * eps as f64);
            assert!(
                (fd - g[i]).abs() < 1e-3,
                "{reduction:?} [{i}] fd {fd} vs {}",
                g[i]
            );
        }
        assert!(g[h * w..].iter().all(|&v| v == 0.0));
    }
}

#[test]
fn per_image_reduction_averages_rows_with_support() {
    let batch = DepthBatch {
        batch: 3,
        height: 1,
        width: 1,
        rgb: vec![0.0; 9],
        target: vec![0.0; 3],
        valid: vec![1.0, 0.0, 1.0],
        valid_count: vec![4, 0, 2],
    };
    assert_eq!(
        batch.row_weights(LossReduction::PerImage),
        [0.125, 0.0, 0.25]
    );
    assert_eq!(
        batch.row_weights(LossReduction::ValidPixels),
        [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
    );
}

/// 2nd / 98th percentiles of the valid log depth map to −1 / +1 (numpy `linear`), invalid pixels
/// are excluded from the percentiles and zeroed, and the map is not clamped unless asked.
#[test]
fn normalization_maps_the_valid_percentiles() {
    let depth: Vec<f32> = (1..=101).map(|i| (i as f32 / 10.0).exp()).collect();
    let mut valid = vec![true; 101];
    let mut with_hole = depth.clone();
    with_hole.push(1e9);
    valid.push(false);
    let plan = DepthLossPlan::default();
    let t = normalize_log_depth(&with_hole, &valid, &plan, 2).unwrap();
    // logs are 0.1 … 10.1; p2 = 0.3, p98 = 9.9
    let expect = |l: f64| 2.0 * (l - 0.3) / 9.6 - 1.0;
    assert!((t[2] as f64 - expect(0.3)).abs() < 1e-5, "{}", t[2]);
    assert!((t[98] as f64 - expect(9.9)).abs() < 1e-5, "{}", t[98]);
    assert!((t[0] as f64 - expect(0.1)).abs() < 1e-5 && t[0] < -1.0);
    assert_eq!(t[101], 0.0);
    let clipped = normalize_log_depth(
        &with_hole,
        &valid,
        &DepthLossPlan {
            clip_target: true,
            ..plan
        },
        2,
    )
    .unwrap();
    assert_eq!(clipped[0], -1.0);
    // degenerate: a constant plane, or fewer than min_valid pixels
    assert!(normalize_log_depth(&[2.0; 9], &[true; 9], &plan, 2).is_none());
    assert!(normalize_log_depth(&depth, &[true; 101], &plan, 200).is_none());
}

#[test]
fn npy_reader_reads_the_supported_layouts() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("d.npy");
    write_npy_f32(&p, 2, 3, |y, x| (y * 10 + x) as f32);
    let plane = read_plane(&p).unwrap();
    assert_eq!((plane.width, plane.height), (3, 2));
    assert_eq!(plane.values, [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
    // a fortran-order array is refused, not transposed silently
    let bytes = std::fs::read(&p).unwrap();
    let text = String::from_utf8_lossy(&bytes).replace("False", "True ");
    std::fs::write(&p, text.as_bytes()).unwrap();
    assert!(read_plane(&p).is_err());
    // 16-bit PNG depth
    let png = dir.path().join("d.png");
    image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::from_fn(2, 2, |x, y| {
        image::Luma([(1000 + x + 2 * y) as u16])
    })
    .save(&png)
    .unwrap();
    assert_eq!(
        read_plane(&png).unwrap().values,
        [1000.0, 1001.0, 1002.0, 1003.0]
    );
}

/// Validity: the mask, a positive finite depth, the min / far planes; misaligned planes refuse.
#[test]
fn sample_validity_and_alignment() {
    let dir = tempfile::tempdir().unwrap();
    let mut it = item(dir.path(), 0, 4, 2, |x, _| x as f32 * 30.0);
    let mask = dir.path().join("m.npy");
    write_npy_f32(&mask, 2, 4, |y, _| if y == 0 { 1.0 } else { 0.0 });
    it.model_options.insert(
        ITEM_DEPTH_KEY.into(),
        json!({"path": dir.path().join("depth0.npy").to_str().unwrap(),
               "mask_path": mask.to_str().unwrap(), "scale": 0.5, "max_depth": 40.0}),
    );
    let s = load_depth_sample(&it).unwrap();
    // depth·scale = 0, 15, 30, 45: 0 invalid (log), 45 beyond the far plane, row 1 masked.
    assert_eq!(
        s.valid,
        [false, true, true, false, false, false, false, false]
    );
    assert_eq!(s.depth[1], 15.0);
    let bad = dir.path().join("bad.npy");
    write_npy_f32(&bad, 3, 4, |_, _| 1.0);
    it.model_options.insert(
        ITEM_DEPTH_KEY.into(),
        json!({"path": bad.to_str().unwrap()}),
    );
    let err = load_depth_sample(&it).unwrap_err().to_string();
    assert!(err.contains("aligned"), "{err}");
}

/// The paired transform keeps RGB, depth and mask aligned through resize, crop and flip: the image
/// encodes `(x, y)` in R/G and the depth encodes the same coordinates.
#[test]
fn paired_transforms_keep_rgb_depth_and_mask_aligned() {
    let (w, h) = (12usize, 8usize);
    let sample = DepthSample {
        width: w,
        height: h,
        rgb: (0..h)
            .flat_map(|y| (0..w).flat_map(move |x| [x as u8 * 20, y as u8 * 30, 0]))
            .collect(),
        depth: (0..h)
            .flat_map(|y| (0..w).map(move |x| 1.0 + (x + 100 * y) as f32))
            .collect(),
        valid: (0..h * w).map(|i| i % 5 != 0).collect(),
    };
    let decode = |row: &DepthRow, o: usize| {
        let x = ((row.rgb_chw[o] + 1.0) * 127.5).round() as usize / 20;
        let y = ((row.rgb_chw[row.depth.len() + o] + 1.0) * 127.5).round() as usize / 30;
        (x, y)
    };
    // native resolution (no resample), crop + flip
    let aug = DepthAugmentPlan {
        height: 4,
        width: 8,
        resize: ResizeMode::None,
        crop: CropMode::Random,
        hflip_prob: 0.5,
    };
    for draw in [
        RowDraw {
            top: 1,
            left: 3,
            flip: true,
        },
        RowDraw {
            top: 4,
            left: 0,
            flip: false,
        },
    ] {
        let row = transform_sample(&sample, &aug, draw).unwrap();
        for o in 0..32 {
            let (x, y) = decode(&row, o);
            let src = y * w + x;
            assert_eq!(row.depth[o], sample.depth[src], "pixel {o}");
            assert_eq!(row.valid[o], sample.valid[src], "pixel {o}");
            let (ox, oy) = (o % 8, o / 8);
            let want_x = draw.left + if draw.flip { 7 - ox } else { ox };
            assert_eq!((x, y), (want_x, draw.top + oy));
        }
    }
    // a 2× upscale cover (nearest depth / mask; the RGB of the same nearest source pixel is not
    // exact under bicubic, so alignment is checked on depth ↔ mask ↔ geometry)
    let aug = DepthAugmentPlan {
        height: 16,
        width: 16,
        resize: ResizeMode::Cover,
        crop: CropMode::Center,
        hflip_prob: 0.0,
    };
    assert_eq!(cover_size(8, 12, 16, 16), (16, 24));
    assert!(covered_size(
        8,
        12,
        &DepthAugmentPlan {
            resize: ResizeMode::None,
            ..aug
        }
    )
    .is_err());
    let row = transform_sample(
        &sample,
        &aug,
        RowDraw {
            top: 0,
            left: 4,
            flip: false,
        },
    )
    .unwrap();
    for o in 0..256 {
        let (ox, oy) = (o % 16, o / 16);
        let src = (oy / 2) * w + (ox + 4) / 2;
        assert_eq!(row.depth[o], sample.depth[src]);
        assert_eq!(row.valid[o], sample.valid[src]);
    }
}

#[test]
fn positional_draws_reproduce() {
    let aug = DepthAugmentPlan {
        height: 4,
        width: 4,
        resize: ResizeMode::Cover,
        crop: CropMode::Random,
        hflip_prob: 0.5,
    };
    let draws = |seed| {
        let mut rng = HostRng::new(batch_seed(seed, 1, 3));
        (0..6)
            .map(|_| draw_row(&mut rng, &aug, 9, 7))
            .collect::<Vec<_>>()
    };
    assert_eq!(draws(4), draws(4));
    assert_ne!(draws(4), draws(5));
    assert!(draws(4).iter().all(|d| d.top <= 5 && d.left <= 3));
}

#[test]
fn plan_defaults_follow_the_paper_and_refuse_what_they_cannot_honour() {
    let cfg = tiny_config();
    let plan = DepthTrainPlan::resolve(&request(vec![], base_cfg()), &cfg).unwrap();
    assert_eq!(plan.optimizer.betas, (0.9, 0.999));
    assert_eq!(plan.gradient_clip, 1.0);
    assert_eq!(plan.ema, None);
    assert_eq!(plan.export_weights, WeightsSelect::Raw);
    assert_eq!(plan.loss, DepthLossPlan::default());
    assert_eq!((plan.augment.height, plan.augment.width), (16, 16));
    assert_eq!(plan.augment.hflip_prob, 0.5);
    assert_eq!(plan.invalid, InvalidPolicy::Reject);
    let mut paper = base_cfg();
    apply_paper_recipe(&mut paper);
    let p = DepthTrainPlan::resolve(&request(vec![], paper), &cfg).unwrap();
    assert_eq!(
        (p.optimizer.lr, p.optimizer.weight_decay),
        (3e-5f32 as f64, 0.01f32 as f64)
    );
    assert_eq!((p.batch_size, p.max_steps, p.warmup_steps), (32, 10_000, 0));

    let refused = |f: &dyn Fn(&mut TrainingConfig)| {
        let mut c = base_cfg();
        f(&mut c);
        DepthTrainPlan::resolve(&request(vec![], c), &cfg).unwrap_err()
    };
    assert!(matches!(
        refused(&|c| c.full_finetune = false),
        Error::Unsupported(_)
    ));
    assert!(matches!(
        refused(&|c| c.loss_type = "mse".into()),
        Error::Unsupported(_)
    ));
    assert!(matches!(
        refused(&|c| c.timestep_type = "uniform".into()),
        Error::Unsupported(_)
    ));
    assert!(matches!(
        refused(&|c| c.sample_prompts = vec!["x".into()]),
        Error::Unsupported(_)
    ));
    assert!(matches!(
        refused(&|c| {
            c.model_options
                .insert(OPTIONS_KEY.into(), json!({"text_dropout": 0.1}));
        }),
        Error::Unsupported(_)
    ));
    refused(&|c| c.resolution = 18);
    refused(&|c| {
        c.model_options
            .insert(OPTIONS_KEY.into(), json!({"depth_size": [16, 18]}));
    });
    refused(&|c| {
        c.model_options
            .insert(OPTIONS_KEY.into(), json!({"export_weights": "ema"}));
    });
    let mut c = base_cfg();
    c.model_options.insert(
        OPTIONS_KEY.into(),
        json!({"depth_size": [8, 16], "crop": "random", "loss_reduction": "per_image",
               "gradient_weight": 2.0, "invalid_examples": "skip", "reducer_init": "channel_mean"}),
    );
    let p = DepthTrainPlan::resolve(&request(vec![], c), &cfg).unwrap();
    assert_eq!(
        (p.augment.height, p.augment.width, p.augment.crop),
        (8, 16, CropMode::Random)
    );
    assert_eq!(p.loss.reduction, LossReduction::PerImage);
    assert_eq!(p.loss.gradient_weight, 2.0);
    assert_eq!(p.invalid, InvalidPolicy::Skip);
    assert_eq!(p.reducer_init, ReducerInit::ChannelMean);
    assert_ne!(p.data_identity(), plan.data_identity());
}

#[test]
fn items_are_validated_by_name() {
    let dir = tempfile::tempdir().unwrap();
    let ok = item(dir.path(), 0, 4, 4, |x, y| 1.0 + (x + y) as f32);
    validate_depth_items(&request(vec![ok.clone()], base_cfg())).unwrap();
    let mut captioned = ok.clone();
    captioned.caption = "a room".into();
    assert!(matches!(
        validate_depth_items(&request(vec![captioned], base_cfg())),
        Err(Error::Unsupported(_))
    ));
    let mut none = ok.clone();
    none.model_options.clear();
    assert!(validate_depth_items(&request(vec![none], base_cfg())).is_err());
    let mut unknown = ok;
    unknown.model_options.insert(
        ITEM_DEPTH_KEY.into(),
        json!({"path": "x.npy", "units": "cm"}),
    );
    assert!(validate_depth_items(&request(vec![unknown], base_cfg())).is_err());
}

/// The explicit invalid-example policy: `reject` refuses the run naming the item, `skip` drops it,
/// an all-invalid dataset is refused either way.
#[test]
fn invalid_examples_are_rejected_or_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let good = item(dir.path(), 0, 4, 4, |x, y| 1.0 + (x + 4 * y) as f32);
    let constant = item(dir.path(), 1, 4, 4, |_, _| 3.0);
    let empty = item(dir.path(), 2, 4, 4, |_, _| 0.0);
    let cfg = tiny_config();
    let plan = |policy: &str| {
        let mut c = base_cfg();
        c.model_options
            .insert(OPTIONS_KEY.into(), json!({"invalid_examples": policy}));
        DepthTrainPlan::resolve(&request(vec![], c), &cfg).unwrap()
    };
    let req = request(
        vec![good.clone(), constant.clone(), empty.clone()],
        base_cfg(),
    );
    let err = depth_dataset_pass(&req, &plan("reject"), |_| {})
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("rgb1.png") && err.contains("degenerate"),
        "{err}"
    );
    let mut seen = Vec::new();
    let kept = depth_dataset_pass(&req, &plan("skip"), |i| seen.push(i)).unwrap();
    assert_eq!(kept, [0]);
    assert_eq!(seen, [0, 1, 2]);
    let bad = request(vec![constant, empty], base_cfg());
    assert!(depth_dataset_pass(&bad, &plan("skip"), |_| {}).is_err());
    let cancelled = request(vec![good], base_cfg());
    cancelled.cancel.cancel();
    assert!(matches!(
        depth_dataset_pass(&cancelled, &plan("reject"), |_| {}),
        Err(Error::Canceled)
    ));
}

#[test]
fn batches_are_deterministic_and_normalized() {
    let dir = tempfile::tempdir().unwrap();
    let items: Vec<_> = (0..3)
        .map(|i| {
            item(dir.path(), i, 8, 4, move |x, y| {
                1.0 + (x + 8 * y + i as u32) as f32
            })
        })
        .collect();
    let mut c = base_cfg();
    c.resolution = 4;
    c.batch_size = 2;
    c.model_options
        .insert(OPTIONS_KEY.into(), json!({"crop": "random"}));
    let req = request(items, c);
    let plan = DepthTrainPlan::resolve(&req, &tiny_config()).unwrap();
    let walk = DataWalk::new(3, 2).unwrap();
    let a = assemble_batch(&req, &plan, &[0, 1, 2], &walk, 1, 0).unwrap();
    let b = assemble_batch(&req, &plan, &[0, 1, 2], &walk, 1, 0).unwrap();
    assert_eq!(a, b);
    assert_eq!((a.batch, a.height, a.width), (2, 4, 4));
    assert_eq!(a.valid_count, [16, 16]);
    for row in 0..2 {
        let t = &a.target[row * 16..(row + 1) * 16];
        let lo = t.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(lo < -0.9 && hi > 0.9, "row {row}: [{lo}, {hi}]");
    }
    let last = assemble_batch(&req, &plan, &[0, 1, 2], &walk, 1, 1).unwrap();
    assert_eq!(last.batch, 1);
}

#[test]
fn export_config_keeps_the_inference_sections_and_names_the_task() {
    let source =
        "model:\n  patch_size: 4\n  pixel:\n    depth: 2\ntext_encoder:\n  hidden_layers:\n  \
                  - 2\n  - 5\nflow:\n  shift: 4.0\ntrain:\n  lr: 1\ntask:\n  name: restoration\n";
    let out = depth_export_config_yaml(source).unwrap();
    assert_eq!(
        out,
        "model:\n  patch_size: 4\n  pixel:\n    depth: 2\ntext_encoder:\n  hidden_layers:\n  - 2\n  \
         - 5\nflow:\n  shift: 4.0\ntask:\n  name: depth\n"
    );
}

#[test]
fn depth_shapes_widen_the_input_projections_only() {
    let cfg = tiny_config();
    let base = super::super::backbone_tensor_shapes(&cfg.model);
    let depth = depth_tensor_shapes(&cfg.model);
    assert_eq!(depth.len(), base.len() + 2);
    let get =
        |v: &[(String, Vec<usize>)], k: &str| v.iter().find(|(n, _)| n == k).unwrap().1.clone();
    assert_eq!(get(&depth, "s_embedder.proj.weight"), [64, 4 * 4 * 4]);
    assert_eq!(get(&depth, "pixel_embedder.proj.weight"), [8, 4]);
    assert_eq!(get(&depth, REDUCER_WEIGHT), [1, 3, 1, 1]);
    let differing: Vec<_> = base
        .iter()
        .filter(|(k, s)| get(&depth, k) != *s)
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(
        differing,
        ["s_embedder.proj.weight", "pixel_embedder.proj.weight"]
    );
}

#[test]
fn reducer_init_is_seeded_and_bounded() {
    let (w, b) = reducer_init_values(ReducerInit::TorchDefault, 3, 7);
    assert_eq!((w.len(), b.len()), (3, 1));
    let bound = 1.0 / 3f32.sqrt();
    assert!(w.iter().chain(&b).all(|v| v.abs() <= bound));
    assert_eq!(
        reducer_init_values(ReducerInit::TorchDefault, 3, 7),
        (w.clone(), b)
    );
    assert_ne!(reducer_init_values(ReducerInit::TorchDefault, 3, 8).0, w);
    assert_eq!(
        reducer_init_values(ReducerInit::ChannelMean, 3, 7),
        (vec![1.0 / 3.0; 3], vec![0.0])
    );
}

#[test]
fn depth_resume_checks_task_state_base_and_data() {
    let cfg = tiny_config();
    let plan = DepthTrainPlan::resolve(&request(vec![], base_cfg()), &cfg).unwrap();
    let saved = CheckpointState {
        step: 2,
        epoch: 1,
        batches_consumed: 2,
        scheduler_step: 2,
        lr: 1e-4,
        nan_count: 0,
        last_loss: 0.1,
        artifact: "full".into(),
        optimizer: "adamw".into(),
        ema_decay: None,
        seed: 0,
        backend: "mlx".into(),
        state_identity: plan.state_identity(),
        data_identity: plan.data_identity(),
        dataset_fingerprint: "ds".into(),
        base_identity: "base".into(),
        task: IrisTask::Depth,
    };
    check_depth_resume(&saved, &plan, "ds", "base").unwrap();
    assert!(check_depth_resume(&saved, &plan, "other", "base").is_err());
    assert!(check_depth_resume(&saved, &plan, "ds", "other").is_err());
    let mut gen = saved.clone();
    gen.task = IrisTask::Generation;
    assert!(check_depth_resume(&gen, &plan, "ds", "base").is_err());
    let mut np = plan.clone();
    np.resume_data_policy = ResumeDataPolicy::NewPhase;
    check_depth_resume(&saved, &np, "other", "base").unwrap();
    // the state.json round trip keeps the task and names the depth trainer
    let json = saved.to_json();
    assert_eq!(json["trainer"], DEPTH_TRAINER_ID);
    assert_eq!(CheckpointState::from_json(&json).unwrap(), saved);
}
