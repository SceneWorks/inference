//! Reachable-sizes PREDICTION for the iOS phone (epic 16774, sc-26433): per tier and square edge,
//! the resident bytes of each Z-Image Turbo phase plus its largest quantum under the campaign's
//! rung-4 selection, against the entitled launch headroom less the engineering reserve. Derived on
//! the host from the real snapshot's TENSOR HEADERS — every module below is built over lazy
//! safetensors handles and every activation is an unevaluated array of the request's shape, so no
//! payload is read and nothing runs — and priced by the SAME counted functions the device guards
//! with (`Backend::Metal`). Predictions, never evidence: the device campaign measures.
//!
//! `ZIMAGE_TIER_ROOT=<SceneWorks/z-image-turbo-mlx snapshot> cargo test -p mlx-gen-z-image --lib
//! reachable_sizes -- --ignored --nocapture`.
//!
//! The selection is the campaign's rung-4 representative: staged residency (the text encoder is
//! dropped before the DiT and VAE load), the DiT's 30 main blocks streamed one at a time (their
//! bytes read inside each block's quantum), chunked attention (fused on the device, so only the
//! chunk outputs and their concatenation are added), the decode tiled at the 512 px default edge
//! (an image no larger than the tile decodes untiled), a window-filling 512-token prompt. The text
//! encoder stays resident (the representative window component is the DiT).

use std::path::{Path, PathBuf};

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::ios_quantum::{Backend, PROCESS_RESERVE_BYTES};
use mlx_gen::weights::Weights;
use mlx_rs::{Array, Dtype};

/// The entitled launch headroom of the target phone (MiB): iPhone 16 Pro Max, three launches.
const ENTITLED_MIB: u64 = 6142;
const MIB: u64 = 1024 * 1024;
const METAL: Backend = Backend::Metal;

fn lazy(shape: &[i32], dtype: Dtype) -> Array {
    // Never evaluated: only its shape and dtype are read by the prices.
    Array::zeros::<f32>(shape).unwrap().as_dtype(dtype).unwrap()
}

fn bytes_where(dir: &Path, keep: impl Fn(&str) -> bool) -> u64 {
    let mut total = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            for h in mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&path).unwrap()
            {
                if keep(&h.name) {
                    total += h.data_bytes;
                }
            }
        }
    }
    total
}

/// One phase: what stays resident, and the largest quantum (label, scratch) inside it.
struct Phase {
    name: &'static str,
    resident: u64,
    quantum: (&'static str, u64),
}

impl Phase {
    fn peak(&self) -> u64 {
        self.resident + self.quantum.1 + PROCESS_RESERVE_BYTES
    }
}

fn largest(quanta: Vec<(&'static str, u64)>) -> (&'static str, u64) {
    quanta.into_iter().max_by_key(|q| q.1).unwrap()
}

fn predict(root: &Path, edge: u32, tile: u32) -> Vec<Phase> {
    let _glue = crate::CompileGlueGuard::enable();
    let te_dir = root.join("text_encoder");
    let dit_dir = root.join("transformer");
    let vae_dir = root.join("vae");
    let (s, c0) = (512, 512);
    // Conditioning: the whole encoder resident; its embed, one layer, the caption gather and cast.
    let te_w = Weights::from_dir(&te_dir).unwrap();
    let te = crate::text_encoder::TextEncoder::from_weights(
        &te_w,
        "model",
        &crate::text_encoder::ZTextEncoderConfig::z_image(),
    )
    .unwrap();
    let layer =
        crate::text_encoder::EncoderLayer::from_weights(&te_w, "model.layers.0", 32, 8, 128, 1e-6)
            .unwrap();
    let ab = crate::quantum::ab;
    let conditioning = Phase {
        name: "conditioning",
        resident: bytes_where(&te_dir, |_| true),
        quantum: largest(vec![
            ("Qwen3 embed", te.embed_bytes(1, s)),
            ("Qwen3 layer", layer.quantum_scratch(1, s, 4, METAL)),
            ("caption features", ab(4, &[c0]) + ab(4, &[c0, 2560])),
        ]),
    };
    // Denoise: the DiT's non-block tensors and the VAE resident, the caption features; one
    // streamed block (+ its stored bytes) at a time.
    let mut dit_w = Weights::from_dir(&dit_dir).unwrap();
    crate::loader::remap_transformer_keys(&mut dit_w);
    let dit = crate::transformer::ZImageTransformer::from_weights(
        &dit_w,
        "",
        crate::transformer::ZImageTransformerConfig::turbo(),
    )
    .unwrap();
    let lat = (edge / 8) as i32;
    let x = lazy(&[16, 1, lat, lat], Dtype::Bfloat16);
    let n = (lat / 2) * (lat / 2);
    let cap = lazy(&[c0 as i32, 2560], Dtype::Float32);
    let l = n + c0 as i32;
    let unified = lazy(&[1, l, 3840], Dtype::Float32);
    let freqs = lazy(&[l, 64, 2], Dtype::Float32);
    let x_stream = lazy(&[1, n, 3840], Dtype::Float32);
    let x_freqs = lazy(&[n, 64, 2], Dtype::Float32);
    let cap_stream = lazy(&[1, c0 as i32, 3840], Dtype::Float32);
    let cap_freqs = lazy(&[c0 as i32, 64, 2], Dtype::Float32);
    let plan = AttentionPlan::budgeted(AttentionBudget::CONSTRAINED);
    let block_read = bytes_where(&dit_dir, |k| k.starts_with("layers.0."));
    let latent_elements = 16 * u64::from(edge / 8) * u64::from(edge / 8);
    let denoise = Phase {
        name: "denoise",
        resident: bytes_where(&dit_dir, |k| !k.starts_with("layers."))
            + bytes_where(&vae_dir, |_| true)
            + ab(2, &[c0, 2560])
            + ab(2, &[latent_elements]),
        quantum: largest(vec![
            ("DiT prepare", dit.prepare_bytes((16, 1, lat, lat), &cap)),
            ("DiT image embed", dit.embed_bytes(&x, 0, METAL)),
            ("DiT caption embed", dit.caption_bytes(&cap, METAL)),
            (
                "DiT refiner block",
                dit.noise_refiner[0].quantum_scratch(&x_stream, &x_freqs, plan, METAL),
            ),
            (
                "DiT context block",
                dit.context_refiner[0].quantum_scratch(&cap_stream, &cap_freqs, plan, METAL),
            ),
            (
                "DiT streamed block",
                dit.layers[0].quantum_scratch(&unified, &freqs, plan, METAL) + block_read,
            ),
            ("DiT output", dit.output_bytes(&unified, METAL)),
            (
                "sampler step",
                mlx_gen::ios_quantum::sampler_step_bytes(latent_elements),
            ),
        ]),
    };
    // Decode: the DiT shed; the VAE and the latents resident.
    let mut vae_w = Weights::from_dir(&vae_dir).unwrap();
    crate::loader::remap_vae_decoder(&mut vae_w).unwrap();
    let decoder = crate::vae::Decoder::from_weights(
        &vae_w,
        "",
        &crate::vae::VaeDecoderConfig::default_z_image(),
    )
    .unwrap();
    let (conv_in, mid, ups, conv_out) = decoder.parts();
    let tiled = edge > tile;
    let mut quanta = Vec::new();
    let ch = |w: &Array| mlx_gen::ios_quantum::conv_channels(w);
    let mid_ch = ch(conv_in.weight()).1;
    let latent = lazy(&[1, mid_ch, lat, lat], Dtype::Float32);
    let call = mlx_gen::ios_quantum::ConvCall::new(&[1, lat, lat, 16], 4, conv_in.weight(), 1, 1);
    quanta.push((
        "VAE decode conv_in",
        mlx_gen::ios_quantum::single_conv_bytes(&call, 0, 0, METAL) + 2 * ab(4, &[latent_elements]),
    ));
    let (r0, attn, _) = mid.parts();
    quanta.push(("VAE mid resnet", r0.quantum_scratch(&latent, METAL)));
    let rows =
        crate::vae::VaeAttention::bounded_plan()
            .budget
            .query_block(1, 1, lat * lat, lat * lat);
    let _ = attn;
    quanta.push((
        "VAE mid attention",
        mlx_gen::ios_quantum::vae_attention_bytes(1, lat * lat, mid_ch, rows, 4, METAL),
    ));
    let mut res = lat;
    let mut remaining = 8;
    let edge_at = |remaining: i32| ((tile as i32 + remaining - 1) / remaining).max(3);
    for up in ups {
        let (resnets, upsampler) = up.parts();
        for r in resnets {
            let (c1, c2) = r.conv_weights();
            let (cin, cout) = ch(c1);
            if tiled {
                let t = edge_at(remaining).min(res);
                let stage = mlx_gen::ios_quantum::tiled_stage_bytes(
                    &[1, res, res, cin],
                    cin,
                    cout,
                    4,
                    true,
                );
                let tile_bytes =
                    mlx_gen::ios_quantum::tile_bytes(&[1, t, t, cin], 4, c1, false, METAL).max(
                        mlx_gen::ios_quantum::tile_bytes(&[1, t, t, cout], 4, c2, false, METAL),
                    );
                quanta.push(("VAE tiled resnet stage + tile", stage + tile_bytes));
            } else {
                let x = lazy(&[1, cin, res, res], Dtype::Float32);
                quanta.push(("VAE resnet", r.quantum_scratch(&x, METAL)));
            }
        }
        if let Some(u) = upsampler {
            let w = u.weight();
            let (cin, cout) = ch(w);
            let next = res * 2;
            remaining /= 2;
            if tiled {
                let t = edge_at(remaining).min(next);
                let stage = mlx_gen::ios_quantum::tiled_stage_bytes(
                    &[1, next, next, cin],
                    cin,
                    cout,
                    4,
                    false,
                );
                quanta.push((
                    "VAE tiled upsample stage + tile",
                    stage + mlx_gen::ios_quantum::tile_bytes(&[1, t, t, cin], 4, w, false, METAL),
                ));
            } else {
                let call = mlx_gen::ios_quantum::ConvCall::new(&[1, next, next, cin], 4, w, 1, 1);
                quanta.push((
                    "VAE upsample",
                    mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, METAL),
                ));
            }
            res = next;
        }
    }
    let (cin, cout) = ch(conv_out.weight());
    if tiled {
        quanta.push((
            "VAE tiled conv_out stage + tile",
            mlx_gen::ios_quantum::tiled_stage_bytes(&[1, res, res, cin], cin, cout, 4, false)
                + mlx_gen::ios_quantum::tile_bytes(
                    &[1, tile as i32, tile as i32, cin],
                    4,
                    conv_out.weight(),
                    false,
                    METAL,
                ),
        ));
    } else {
        let call =
            mlx_gen::ios_quantum::ConvCall::new(&[1, res, res, cin], 4, conv_out.weight(), 1, 1);
        quanta.push((
            "VAE conv_out",
            mlx_gen::ios_quantum::norm_conv_bytes(&call, METAL),
        ));
    }
    let pixels = 3 * u64::from(edge) * u64::from(edge);
    quanta.push((
        "decoded to image",
        crate::quantum::zimage_pixel_conversion_bytes(pixels, 4),
    ));
    let decode = Phase {
        name: "decode",
        resident: bytes_where(&vae_dir, |_| true) + ab(2, &[latent_elements]),
        quantum: largest(quanta),
    };
    // The same conditioning under the rung-4 text-encoder window (component `TextEncoder` / `Both`):
    // the embedding resident, each layer read inside its own quantum.
    let te_windowed = Phase {
        name: "conditioning (TE window)",
        resident: bytes_where(&te_dir, |k| !k.starts_with("model.layers.")),
        quantum: (
            "Qwen3 streamed layer",
            layer.quantum_scratch(1, s, 4, METAL)
                + bytes_where(&te_dir, |k| k.starts_with("model.layers.0.")),
        ),
    };
    vec![conditioning, denoise, decode, te_windowed]
}

#[test]
#[ignore = "reads a provisioned Z-Image Turbo snapshot's headers (ZIMAGE_TIER_ROOT)"]
fn reachable_sizes() {
    let root = PathBuf::from(std::env::var_os("ZIMAGE_TIER_ROOT").expect("ZIMAGE_TIER_ROOT"));
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let limit = ENTITLED_MIB * MIB;
    println!(
        "| tier | edge | conditioning | denoise | decode (tile {}) | binding quantum | fits {} MiB | largest fitting decode tile |",
        crate::memory_strategy::DECODE_TILE_EDGES[0],
        ENTITLED_MIB
    );
    println!("|---|---:|---:|---:|---:|---|---|---|");
    for tier in ["q4", "q8", "bf16"] {
        let dir = root.join(tier);
        if !dir.is_dir() {
            continue;
        }
        for edge in [512u32, 768, 1024] {
            // The campaign's representative bounded decode: the first published native edge.
            let phases = predict(&dir, edge, crate::memory_strategy::DECODE_TILE_EDGES[0]);
            let worst = phases[..3].iter().max_by_key(|p| p.peak()).unwrap();
            let mib = |p: &Phase| p.peak().div_ceil(MIB);
            // The largest published decode edge at which every phase fits (with the text-encoder
            // window when the resident encoder alone does not), if any.
            let best = crate::memory_strategy::DECODE_TILE_EDGES
                .iter()
                .find_map(|&tile| {
                    let phases = predict(&dir, edge, tile);
                    let peak = |te: usize| {
                        phases[te]
                            .peak()
                            .max(phases[1].peak())
                            .max(phases[2].peak())
                    };
                    if peak(0) <= limit {
                        Some(format!("tile {tile}: {} MiB", peak(0).div_ceil(MIB)))
                    } else {
                        (peak(3) <= limit).then(|| {
                            format!("tile {tile} + TE window: {} MiB", peak(3).div_ceil(MIB))
                        })
                    }
                });
            println!(
                "| {tier} | {edge} | {} | {} | {} | {} {} ({} MiB resident + {} MiB) | {} | {} |",
                mib(&phases[0]),
                mib(&phases[1]),
                mib(&phases[2]),
                worst.name,
                worst.quantum.0,
                worst.resident.div_ceil(MIB),
                worst.quantum.1.div_ceil(MIB),
                if worst.peak() <= limit { "yes" } else { "NO" },
                best.unwrap_or_else(|| "none".into())
            );
        }
    }
    println!(
        "(peak = resident + largest quantum + the {} MiB reserve; entitled launch headroom {} MiB)",
        PROCESS_RESERVE_BYTES / MIB,
        ENTITLED_MIB
    );
}
