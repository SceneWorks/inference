#![cfg(test)]

use super::{
    encode_bucket_latents, flatten_latent, guard_latent_edge, largest_bucket, latent_edge_of,
    projected_dense_peak_gb, step_entry, still_subject_weight, trainer_descriptor,
    trainer_descriptor_25, PreparedSubjectMask,
};
use crate::positions::create_position_grid;
use mlx_gen::gen_core::{self, BucketSchedule, ResolutionBucket, TrainingConfig};
use mlx_gen::train::dataset::bucket_edges;
use mlx_rs::Array;

fn rb(resolution: u32, repeats: u32) -> ResolutionBucket {
    ResolutionBucket {
        resolution,
        repeats,
    }
}

fn edges_of(buckets: &[ResolutionBucket]) -> Vec<u32> {
    bucket_edges(&TrainingConfig {
        resolution_buckets: buckets.to_vec(),
        ..TrainingConfig::default()
    })
}

/// sc-2127: 2.3 declares buckets; 2.5 (prepared latent packs) does not.
#[test]
fn only_the_2_3_descriptor_declares_resolution_buckets() {
    assert!(trainer_descriptor().techniques.resolution_buckets);
    assert!(!trainer_descriptor_25().techniques.resolution_buckets);
}

/// sc-24828 × sc-2127: with subject-masked loss on and two buckets, each bucket's cached
/// weight is the item's mask resampled onto **that bucket's** latent grid (centre-square crop
/// of a non-square image) and patchified like it — the patchified clean latent's exact
/// `(1, S, C)` shape — with the masked-out (right-half-of-crop) tokens zero and the subject
/// tokens at the subject weight.
///
/// *Mutation that reds this:* `encode_bucket_latents` resampling the mask once at the first
/// bucket's latent shape and reusing it for every bucket.
#[test]
fn subject_mask_weight_is_resampled_per_bucket() {
    let tmp = tempfile::tempdir().unwrap();
    // 96×64 image: the centre square is x ∈ [16, 80); the subject is x < 48 — exactly the
    // crop's left half.
    let image_path = tmp.path().join("a.png");
    image::RgbImage::from_pixel(96, 64, image::Rgb([90, 120, 150]))
        .save(&image_path)
        .unwrap();
    let mask_path = tmp.path().join("a.mask.png");
    image::GrayImage::from_fn(96, 64, |x, _| image::Luma([if x < 48 { 255 } else { 0 }]))
        .save(&mask_path)
        .unwrap();
    let mut item = mlx_gen::TrainingItem::captioned(image_path, "a".into());
    item.subject_mask_path = Some(mask_path);
    let cfg = gen_core::SubjectMaskLoss {
        background_weight: 0.0,
        subject_weight: 1.0,
    };
    let mask = PreparedSubjectMask::load_if_enabled("ltx", &item, Some(&cfg)).unwrap();
    assert!(mask.is_some());
    let edges = [256u32, 512];
    let channels = 8usize;
    let entries = encode_bucket_latents(
        &edges,
        |edge| {
            let le = (edge / 32) as i32;
            Ok(mlx_rs::ops::zeros::<f32>(&[1, channels as i32, 1, le, le])?)
        },
        |shape| still_subject_weight("ltx", mask.as_ref(), shape),
    )
    .unwrap();
    assert_eq!(entries.len(), edges.len());
    for (b, (clean, weight)) in entries.iter().enumerate() {
        let weight = weight.as_ref().expect("mask loss on ⇒ a weight");
        assert_eq!(weight.shape(), clean.shape(), "bucket {b} weight shape");
        let le = (edges[b] / 32) as usize;
        assert_eq!(clean.shape()[1] as usize, le * le, "bucket {b} tokens");
        let flat = weight.reshape(&[-1]).unwrap();
        for (token, chans) in flat.as_slice::<f32>().chunks(channels).enumerate() {
            let want = if token % le < le / 2 { 1.0 } else { 0.0 };
            assert!(
                chans.iter().all(|&v| (v - want).abs() < 1e-5),
                "bucket {b} token {token}: {chans:?} want {want}"
            );
        }
    }
}

/// sc-2127 / epic 2123 E7: the guard sizes for the LARGEST bucket — [512, 1024] gives exactly
/// the 1024-only latent edge (and projection), more than the 512-only one.
#[test]
fn guard_sizes_for_the_largest_bucket() {
    let both = guard_latent_edge(&edges_of(&[rb(512, 1), rb(1024, 1)]));
    let big = guard_latent_edge(&edges_of(&[rb(1024, 1)]));
    let small = guard_latent_edge(&edges_of(&[rb(512, 1)]));
    assert_eq!(both, big);
    assert_eq!(both, 32);
    let p = |le: usize| projected_dense_peak_gb((le * le) as f64);
    assert_eq!(p(both), p(big));
    assert!(p(both) > p(small));
    // Buckets off: the legacy single edge.
    assert_eq!(
        guard_latent_edge(&bucket_edges(&TrainingConfig::default())),
        latent_edge_of(bucket_edges(&TrainingConfig::default())[0])
    );
}

/// sc-2127: one bucket walks the pre-bucket `(step-1) % n` order on bucket 0.
#[test]
fn one_bucket_step_entry_is_the_pre_bucket_order() {
    let n = 5usize;
    let schedule = BucketSchedule::new(n, &[rb(768, 1)], 3);
    for step in 1..=60u32 {
        assert_eq!(step_entry(&schedule, step), (((step - 1) as usize) % n, 0));
    }
}

/// sc-2127: over a multi-bucket schedule every step's RoPE grid (picked by `step_entry`'s bucket)
/// matches the token count of the cached latent it trains on — tiny synthetic latents, no
/// weights. Also checks the 4:1 per-item mix and that previews take the largest grid.
#[test]
fn per_step_positions_match_the_cached_latent() {
    let edges = edges_of(&[rb(64, 4), rb(128, 1)]);
    let latent_edges: Vec<usize> = edges.iter().map(|&e| latent_edge_of(e)).collect();
    assert_eq!(latent_edges, vec![2, 4]);
    assert_eq!(largest_bucket(&latent_edges), 1);
    let n_items = 2usize;
    // Item-major cache of flattened (1, S, 128) latents.
    let mut cache = Vec::new();
    for _ in 0..n_items {
        for &le in &latent_edges {
            let latent = Array::zeros::<f32>(&[1, 128, 1, le as i32, le as i32]).unwrap();
            cache.push(flatten_latent(&latent).unwrap());
        }
    }
    let positions: Vec<Array> = latent_edges
        .iter()
        .map(|&le| create_position_grid(1, 1, le, le))
        .collect();
    let schedule = BucketSchedule::new(n_items, &[rb(64, 4), rb(128, 1)], 9);
    let mut counts = vec![[0u32; 2]; n_items];
    for step in 1..=(2 * schedule.epoch_len()) as u32 {
        let (entry, bucket) = step_entry(&schedule, step);
        let tokens = cache[entry].shape()[1];
        // Position grid is `[B, 3, S, 2]`.
        assert_eq!(positions[bucket].shape()[2], tokens, "step {step}");
        counts[entry / 2][bucket] += 1;
    }
    for c in counts {
        assert_eq!(c, [8, 2]);
    }
}

/// sc-2127: the 2.5 weights-free preflight refuses buckets with the typed `Unsupported` (the
/// same floor the trainer's `validate` applies), before touching any prepared pack.
#[test]
fn ltx25_preflight_refuses_resolution_buckets() {
    let mut req = super::validate_request_tests::request(1);
    req.config.resolution_buckets = vec![rb(512, 1), rb(1024, 1)];
    match super::validate_ltx25_training_request(&req) {
        Err(mlx_gen::Error::Unsupported(message)) => {
            assert!(message.contains("resolution_buckets"), "{message}")
        }
        other => panic!("expected a typed Unsupported, got {other:?}"),
    }
}
