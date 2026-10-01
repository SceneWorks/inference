//! SC-20686 synthetic Wan VAE builders and the encode footprint harness. Uses only the VAEs' public
//! constructors and `encode`, so the same harness measures any revision of the encoders.

use std::collections::HashMap;

use mlx_gen::weights::Weights;
use mlx_rs::{Array, Dtype};

use crate::vae::WanVae;
use crate::vae22::Wan22Vae;

fn draw(shape: &[i32], dtype: Dtype, key: &Array) -> Array {
    let value = mlx_rs::random::normal::<f32>(shape, None, Some(0.2), Some(key))
        .unwrap()
        .as_dtype(dtype)
        .unwrap();
    mlx_rs::transforms::eval([&value]).unwrap();
    value
}

/// A z16 Wan 2.1 VAE (decoder + encoder) with the production topology at base width `dim` (96 in
/// production) and seeded random f32 weights, each materialized as it is made.
pub(crate) fn synthetic_z16(dim: i32) -> WanVae {
    const DIM_MULT: [i32; 4] = [1, 2, 4, 4];
    const Z: i32 = 16;
    let key = mlx_rs::random::key(20686).unwrap();
    let mut tensors = HashMap::new();
    let mut put = |name: String, shape: &[i32]| {
        tensors.insert(name, draw(shape, Dtype::Float32, &key));
    };
    fn conv(put: &mut dyn FnMut(String, &[i32]), p: &str, o: i32, i: i32, kt: i32, k: i32) {
        put(format!("{p}.weight"), &[o, kt, k, k, i]);
        put(format!("{p}.bias"), &[o]);
    }
    fn res(put: &mut dyn FnMut(String, &[i32]), p: &str, i: i32, o: i32) {
        put(format!("{p}.residual.0.gamma"), &[i]);
        conv(put, &format!("{p}.residual.2"), o, i, 3, 3);
        put(format!("{p}.residual.3.gamma"), &[o]);
        conv(put, &format!("{p}.residual.6"), o, o, 3, 3);
        if i != o {
            conv(put, &format!("{p}.shortcut"), o, i, 1, 1);
        }
    }
    fn attention(put: &mut dyn FnMut(String, &[i32]), p: &str, c: i32) {
        put(format!("{p}.norm.gamma"), &[c]);
        put(format!("{p}.to_qkv.weight"), &[3 * c, 1, 1, c]);
        put(format!("{p}.to_qkv.bias"), &[3 * c]);
        put(format!("{p}.proj.weight"), &[c, 1, 1, c]);
        put(format!("{p}.proj.bias"), &[c]);
    }
    let top = dim * DIM_MULT[3];
    // Decoder.
    conv(&mut put, "conv2", Z, Z, 1, 1);
    conv(&mut put, "decoder.conv1", top, Z, 3, 3);
    res(&mut put, "decoder.middle.0", top, top);
    attention(&mut put, "decoder.middle.1", top);
    res(&mut put, "decoder.middle.2", top, top);
    let (mut input, mut index) = (top, 0);
    for stage in 0..DIM_MULT.len() {
        let output = dim * DIM_MULT[DIM_MULT.len() - 1 - stage];
        for block in 0..3 {
            let block_input = if block == 0 { input } else { output };
            res(
                &mut put,
                &format!("decoder.upsamples.{index}"),
                block_input,
                output,
            );
            index += 1;
        }
        input = output;
        if stage < 3 {
            let p = format!("decoder.upsamples.{index}");
            if stage < 2 {
                conv(
                    &mut put,
                    &format!("{p}.time_conv"),
                    2 * output,
                    output,
                    3,
                    1,
                );
            }
            put(
                format!("{p}.resample.1.weight"),
                &[output / 2, 3, 3, output],
            );
            put(format!("{p}.resample.1.bias"), &[output / 2]);
            input = output / 2;
            index += 1;
        }
    }
    put("decoder.head.0.gamma".into(), &[dim]);
    conv(&mut put, "decoder.head.2", 3, dim, 3, 3);
    // Encoder (dims [d, d, 2d, 4d, 4d]) + the post-encoder pointwise conv.
    conv(&mut put, "encoder.conv1", dim, 3, 3, 3);
    let dims = [
        dim,
        dim * DIM_MULT[0],
        dim * DIM_MULT[1],
        dim * DIM_MULT[2],
        top,
    ];
    let mut index = 0;
    for stage in 0..DIM_MULT.len() {
        let (input, output) = (dims[stage], dims[stage + 1]);
        for block in 0..2 {
            let block_input = if block == 0 { input } else { output };
            res(
                &mut put,
                &format!("encoder.downsamples.{index}"),
                block_input,
                output,
            );
            index += 1;
        }
        if stage < 3 {
            let p = format!("encoder.downsamples.{index}");
            if stage > 0 {
                conv(&mut put, &format!("{p}.time_conv"), output, output, 3, 1);
            }
            put(format!("{p}.resample.1.weight"), &[output, 3, 3, output]);
            put(format!("{p}.resample.1.bias"), &[output]);
            index += 1;
        }
    }
    res(&mut put, "encoder.middle.0", top, top);
    attention(&mut put, "encoder.middle.1", top);
    res(&mut put, "encoder.middle.2", top, top);
    put("encoder.head.0.gamma".into(), &[top]);
    conv(&mut put, "encoder.head.2", 2 * Z, top, 3, 3);
    conv(&mut put, "conv1", 2 * Z, 2 * Z, 1, 1);
    WanVae::from_weights(&Weights::from_map(tensors)).unwrap()
}

/// The production z48 VAE schema with seeded random f32 *encoder* weights; the decoder (unused by an
/// encode) is shape-valid zero-stride broadcasts, so it costs no memory.
fn synthetic_z48_production_encoder() -> Wan22Vae {
    let schema = crate::vae22::production_weight_schema();
    let key = mlx_rs::random::key(20686).unwrap();
    let mut tensors = HashMap::new();
    for spec in &schema.decoder {
        let shape: Vec<i32> = spec.shape.iter().map(|&d| d as i32).collect();
        let value = mlx_rs::ops::broadcast_to(Array::from_slice(&[0.01f32], &[1]), &shape).unwrap();
        tensors.insert(spec.name.clone(), value.as_dtype(Dtype::Bfloat16).unwrap());
    }
    for spec in &schema.encoder {
        let shape: Vec<i32> = spec.shape.iter().map(|&d| d as i32).collect();
        tensors.insert(spec.name.clone(), draw(&shape, Dtype::Float32, &key));
    }
    Wan22Vae::from_weights(&Weights::from_map(tensors)).unwrap()
}

/// `(live peak, sampled live + cache peak, cache left after)` above the pre-run baseline, in bytes.
fn measure(run: impl FnOnce()) -> (u64, u64, u64) {
    mlx_rs::memory::clear_cache();
    let base = mlx_rs::memory::get_active_memory() as u64;
    mlx_rs::memory::reset_peak_memory();
    let probe = mlx_gen::memory_probe::AllocatorProbe::start(std::time::Duration::from_millis(1));
    run();
    let report = probe.finish();
    let live = (mlx_rs::memory::get_peak_memory() as u64).saturating_sub(base);
    let footprint = report
        .sampled_footprint_peak_bytes
        .max(live + base)
        .saturating_sub(base);
    (live, footprint, mlx_rs::memory::get_cache_memory() as u64)
}

fn report(label: &str, voxels: f64, (live, footprint, cache): (u64, u64, u64)) {
    eprintln!(
        "{label}: {voxels} input voxels; live {:.0} B/voxel, live+cache {:.0} B/voxel, cache left \
         {cache} B",
        live as f64 / voxels,
        footprint as f64 / voxels
    );
}

/// Measurement harness (sc-20686), not a gate: single-pass encode of a `[1,3,5,64,64]` clip (20,480
/// input voxels) through the production-width z16 encoder (dim 96, f32, synthetic weights, ~0.5 GB of
/// decoder + encoder weights) and the production-width z48 encoder (f32 encoder weights ~0.5 GB,
/// zero-cost decoder). Activations at this clip are well under 0.5 GB; total < 1.5 GB. Run alone:
/// `cargo test -p mlx-gen-wan --lib encode_footprint_harness -- --ignored --test-threads=1
/// --nocapture`.
#[test]
#[ignore = "measurement harness; production-width weights; run alone on request"]
fn encode_footprint_harness() {
    let (t, h, w) = (5, 64, 64);
    let voxels = f64::from(t * h * w);
    let key = mlx_rs::random::key(7).unwrap();
    {
        let vae = synthetic_z16(96);
        let video =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, 3, t, h, w], Some(&key)).unwrap();
        mlx_rs::transforms::eval([&video]).unwrap();
        report(
            "z16 f32 encode",
            voxels,
            measure(|| {
                let z = vae.encode(&video).unwrap();
                mlx_rs::transforms::eval([&z]).unwrap();
            }),
        );
    }
    mlx_rs::memory::clear_cache();
    {
        let vae = synthetic_z48_production_encoder();
        let video =
            mlx_rs::random::uniform::<f32, f32>(-1.0, 1.0, &[1, t, h, w, 3], Some(&key)).unwrap();
        mlx_rs::transforms::eval([&video]).unwrap();
        report(
            "z48 f32 encode",
            voxels,
            measure(|| {
                let z = vae.encode(&video).unwrap();
                mlx_rs::transforms::eval([&z]).unwrap();
            }),
        );
    }
}
