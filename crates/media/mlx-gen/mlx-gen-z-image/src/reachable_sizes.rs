//! Reachable-sizes PREDICTION for the iOS phone (epic 16774, sc-26433): per mode, tier and square
//! edge, the peak of every Z-Image Turbo phase — its resident bytes plus its largest quantum (the
//! quantum's live input plus its counted scratch) plus the reserve — under each rung-4 selection the
//! campaign enumerates, against the entitled launch headroom. Derived on the host from the real
//! snapshot's TENSOR HEADERS — every module below is built over lazy safetensors handles and every
//! activation is an unevaluated array of the request's shape, so no payload is read and nothing
//! runs — and priced by the SAME counted functions the device guards with (`Backend::Metal`).
//! Predictions, never evidence: the device campaign measures.
//!
//! `ZIMAGE_TIER_ROOT=<SceneWorks/z-image-turbo-mlx snapshot> cargo test -p mlx-gen-z-image --lib
//! reachable_sizes -- --ignored --nocapture`.
//!
//! The selection space is the campaign's rung-4 one (`MemoryProviderContract::selection_space`),
//! walked quality first exactly as the capture ladder walks it: unstaged before staged (staged
//! residency drops the text encoder before the DiT and VAE load and the DiT before the decode;
//! unstaged keeps every component loaded throughout), the window scope `Dit` (DiT blocks streamed,
//! text encoder resident) before `TextEncoder` (encoder layers streamed, DiT resident) before
//! `Both`, and the decode tile from the first published native edge (768; an image no larger than
//! the tile decodes untiled) down to 256. Every cell runs chunked attention (fused on the device)
//! and a window-filling 512-token prompt.
//!
//! The modes: `text`; `reference` (img2img: the init image VAE-encoded inside the denoise phase,
//! with the denoise phase's components resident — on the device the bounded encode, every
//! full-resolution layer tiled at 256 input pixels — and f32 caption features); `control` (the Fun-Controlnet-Union route: the control image encoded the same way into
//! the 33-channel control context, the control branch's resident embedder and 2-block refiner, its
//! 15 main blocks streamed with the DiT's (each block's dense read plus its packing at the tier),
//! and its 17 hints live through the base stack); `control_reference` (both encodes). The
//! ControlNet checkpoint is not provisioned on this host: its bytes are derived from the published
//! architecture — each control block is a DiT block (the bf16 tier's `layers.0` header bytes: the
//! checkpoint is published dense) plus its `[3840, 3840]` `after_proj` (and, on block 0,
//! `before_proj`), and the `[3840, 132]` patch embedder.

use std::path::{Path, PathBuf};

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::gen_core::TransformerComponent;
use mlx_gen::ios_quantum::{Backend, ConvCall, PROCESS_RESERVE_BYTES};
use mlx_gen::weights::Weights;
use mlx_rs::{Array, Dtype};

use crate::quantum::ab;

/// The entitled launch headroom of the target phone (MiB): iPhone 16 Pro Max, three launches.
const ENTITLED_MIB: u64 = 6142;
const MIB: u64 = 1024 * 1024;
const METAL: Backend = Backend::Metal;
/// The window-filling prompt's token count.
const TOKENS: i32 = 512;
const CAP_DIM: u64 = 2560;
const DIM: i32 = 3840;
const LATENT_CHANNELS: u64 = 16;

fn lazy(shape: &[i32], dtype: Dtype) -> Array {
    // Never evaluated: only its shape and dtype are read by the prices.
    Array::zeros::<f32>(shape).unwrap().as_dtype(dtype).unwrap()
}

fn f32_bytes(shape: &[i32]) -> u64 {
    ab(
        4,
        &shape
            .iter()
            .map(|&d| crate::quantum::u(d))
            .collect::<Vec<_>>(),
    )
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Text,
    Reference,
    Control,
    ControlReference,
}

impl Mode {
    const ALL: [Mode; 4] = [
        Mode::Text,
        Mode::Reference,
        Mode::Control,
        Mode::ControlReference,
    ];
    fn name(self) -> &'static str {
        match self {
            Mode::Text => "text",
            Mode::Reference => "reference",
            Mode::Control => "control",
            Mode::ControlReference => "control_reference",
        }
    }
    fn control(self) -> bool {
        matches!(self, Mode::Control | Mode::ControlReference)
    }
    fn img2img(self) -> bool {
        matches!(self, Mode::Reference | Mode::ControlReference)
    }
    fn encodes(self) -> bool {
        self != Mode::Text
    }
}

/// One rung-4 selection: staging, window scope, decode tile.
#[derive(Clone, Copy, Debug)]
struct Selection {
    staged: bool,
    scope: TransformerComponent,
    tile: u32,
}

impl Selection {
    fn token(self) -> String {
        format!(
            "{}t{}_o64_c67108864_w1_{}",
            if self.staged { "s_" } else { "" },
            self.tile,
            match self.scope {
                TransformerComponent::Dit => "dit",
                TransformerComponent::TextEncoder => "te",
                TransformerComponent::Both => "both",
            }
        )
    }
}

/// One phase: what stays resident, and the largest quantum (label, live input + scratch) inside it.
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

/// The header-derived bytes and the lazily built modules of one tier.
struct Tier {
    te_dir: PathBuf,
    dit_dir: PathBuf,
    vae_dir: PathBuf,
    te: crate::text_encoder::TextEncoder,
    te_layer: crate::text_encoder::EncoderLayer,
    dit: crate::transformer::ZImageTransformer,
    vae: crate::vae::Vae,
    /// One dense (bf16) control block, block 0's (with `before_proj`).
    control_block: crate::control_transformer_block::ZImageControlBlock,
    /// Packing bits of the tier (`None`: dense bf16).
    bits: Option<i32>,
}

/// A dense `[out, in]` projection plus bias at `bits` (`None`: bf16).
fn projection_bytes(out: u64, inp: u64, bits: Option<i32>) -> u64 {
    let bias = 2 * out;
    match bits {
        None => 2 * out * inp + bias,
        Some(b) => out * inp * b as u64 / 8 + 2 * 2 * (out * inp / 64) + bias,
    }
}

impl Tier {
    fn load(root: &Path, dense_root: &Path, bits: Option<i32>) -> Self {
        let te_dir = root.join("text_encoder");
        let dit_dir = root.join("transformer");
        let vae_dir = root.join("vae");
        let te_w = Weights::from_dir(&te_dir).unwrap();
        let te = crate::text_encoder::TextEncoder::from_weights(
            &te_w,
            "model",
            &crate::text_encoder::ZTextEncoderConfig::z_image(),
        )
        .unwrap();
        let te_layer = crate::text_encoder::EncoderLayer::from_weights(
            &te_w,
            "model.layers.0",
            32,
            8,
            128,
            1e-6,
        )
        .unwrap();
        let mut dit_w = Weights::from_dir(&dit_dir).unwrap();
        crate::loader::remap_transformer_keys(&mut dit_w);
        let dit = crate::transformer::ZImageTransformer::from_weights(
            &dit_w,
            "",
            crate::transformer::ZImageTransformerConfig::turbo(),
        )
        .unwrap();
        let mut vae_w = Weights::from_dir(&vae_dir).unwrap();
        let vae = crate::loader::build_vae(&mut vae_w).unwrap();
        // The control block: the dense DiT block's tensors plus the control projections.
        let mut dense_w = Weights::from_dir(dense_root.join("transformer")).unwrap();
        crate::loader::remap_transformer_keys(&mut dense_w);
        for name in ["after_proj", "before_proj"] {
            dense_w.insert(
                format!("layers.0.{name}.weight"),
                lazy(&[DIM, DIM], Dtype::Bfloat16),
            );
            dense_w.insert(
                format!("layers.0.{name}.bias"),
                lazy(&[DIM], Dtype::Bfloat16),
            );
        }
        let control_block = crate::control_transformer_block::ZImageControlBlock::from_weights(
            &dense_w,
            "layers.0",
            dit.cfg.block_cfg(),
            true,
        )
        .unwrap();
        Self {
            te_dir,
            dit_dir,
            vae_dir,
            te,
            te_layer,
            dit,
            vae,
            control_block,
            bits,
        }
    }

    fn te_bytes(&self, streamed: bool) -> u64 {
        bytes_where(&self.te_dir, |k| {
            !streamed || !k.starts_with("model.layers.")
        })
    }
    fn dit_bytes(&self, streamed: bool) -> u64 {
        bytes_where(&self.dit_dir, |k| !streamed || !k.starts_with("layers."))
    }
    fn vae_bytes(&self) -> u64 {
        bytes_where(&self.vae_dir, |_| true)
    }
    /// A DiT block's stored bytes at this tier (a streamed block's read).
    fn dit_block_read(&self) -> u64 {
        bytes_where(&self.dit_dir, |k| k.starts_with("layers.0."))
    }
    /// A control block at this tier once packed: the DiT block at the tier plus its projections.
    fn control_block_packed(&self, with_before: bool) -> u64 {
        let proj = projection_bytes(DIM as u64, DIM as u64, self.bits);
        self.dit_block_read() + proj * (1 + u64::from(with_before))
    }
    /// A control block's dense checkpoint read (block 0: with `before_proj`).
    fn control_block_read(&self, dense: &Tier) -> u64 {
        dense.dit_block_read() + 2 * projection_bytes(DIM as u64, DIM as u64, None)
    }
    /// The control branch resident outside its 15 main blocks: the dense patch embedder and the
    /// 2-block refiner packed at the tier.
    fn control_resident(&self) -> u64 {
        projection_bytes(DIM as u64, 132, None)
            + self.control_block_packed(true)
            + self.control_block_packed(false)
    }
}

/// The VAE encode of one full-resolution image as the device runs it — the bounded encode at
/// [`crate::vae::decoder::IOS_ENCODE_TILE_PX`] (every full-resolution layer one tiled stage plus its
/// largest tile; an image within one tile encodes whole) — as quanta with their live inputs; the
/// image itself is resident through the encode.
fn encode_quanta(vae: &crate::vae::Vae, edge: i32) -> Vec<(&'static str, u64)> {
    let encoder = vae
        .encoder()
        .expect("the snapshot's VAE carries its encoder");
    let (conv_in, downs, mid, conv_out) = encoder.parts();
    let ch = |w: &Array| mlx_gen::ios_quantum::conv_channels(w);
    let tile_px = crate::vae::decoder::IOS_ENCODE_TILE_PX;
    let tiled = edge > tile_px;
    let mut quanta = vec![("init image upload", ab(4, &[edge as u64, edge as u64, 3]))];
    let image = [1, 3, edge, edge];
    let (_, c_in) = ch(conv_in.weight());
    quanta.push((
        "VAE encode conv_in",
        f32_bytes(&image)
            + if tiled {
                mlx_gen::ios_quantum::tiled_stage_bytes(&[1, edge, edge, 3], 3, c_in, 4, false)
                    + mlx_gen::ios_quantum::tile_bytes(
                        &[1, tile_px, tile_px, 3],
                        4,
                        conv_in.weight(),
                        false,
                        METAL,
                    )
            } else {
                let call = ConvCall::new(&[1, edge, edge, 3], 4, conv_in.weight(), 1, 1);
                mlx_gen::ios_quantum::single_conv_bytes(&call, 0, 0, METAL)
            },
    ));
    let mut res = edge;
    let mut c = c_in;
    let mut scale = 1;
    for down in downs {
        let (resnets, downsampler) = down.parts();
        let t = ((tile_px + scale - 1) / scale).max(3).min(res);
        for r in resnets {
            let (c1, c2) = r.conv_weights();
            let cout = ch(c2).1;
            let input = f32_bytes(&[1, c, res, res]);
            quanta.push((
                "VAE encode resnet",
                input
                    + if tiled {
                        mlx_gen::ios_quantum::tiled_stage_bytes(&[1, res, res, c], c, cout, 4, true)
                            + mlx_gen::ios_quantum::tile_bytes(&[1, t, t, c], 4, c1, false, METAL)
                                .max(mlx_gen::ios_quantum::tile_bytes(
                                    &[1, t, t, cout],
                                    4,
                                    c2,
                                    false,
                                    METAL,
                                ))
                    } else {
                        r.quantum_scratch(&lazy(&[1, c, res, res], Dtype::Float32), METAL)
                    },
            ));
            c = cout;
        }
        if let Some(ds) = downsampler {
            scale *= 2;
            let core = ((tile_px + scale - 1) / scale).max(1);
            let out = res / 2;
            quanta.push((
                "VAE encode downsample",
                f32_bytes(&[1, c, res, res])
                    + if tiled {
                        mlx_gen::ios_quantum::downsample_stage_bytes(&[1, out, out, c], 4)
                            + mlx_gen::ios_quantum::downsample_tile_bytes(
                                &[1, 2 * core + 1, 2 * core + 1, c],
                                4,
                                ds.weight(),
                                METAL,
                            )
                    } else {
                        let call = ConvCall::new(&[1, res + 1, res + 1, c], 4, ds.weight(), 2, 0);
                        mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, METAL)
                    },
            ));
            res = out;
        }
    }
    let (r0, _, r1) = mid.parts();
    for r in [r0, r1] {
        let x = lazy(&[1, c, res, res], Dtype::Float32);
        quanta.push((
            "VAE encode mid resnet",
            f32_bytes(&[1, c, res, res]) + r.quantum_scratch(&x, METAL),
        ));
    }
    let rows =
        crate::vae::VaeAttention::bounded_plan()
            .budget
            .query_block(1, 1, res * res, res * res);
    quanta.push((
        "VAE encode mid attention",
        f32_bytes(&[1, c, res, res])
            + mlx_gen::ios_quantum::vae_attention_bytes(1, res * res, c, rows, 4, METAL),
    ));
    let call = ConvCall::new(&[1, res, res, c], 4, conv_out.weight(), 1, 1);
    quanta.push((
        "VAE encode head",
        f32_bytes(&[1, c, res, res])
            + mlx_gen::ios_quantum::norm_conv_bytes(&call, METAL)
            + crate::vae::Vae::latent_mean_bytes(&lazy(&image, Dtype::Float32)),
    ));
    quanta
}

/// Every phase of one cell.
fn predict(t: &Tier, dense: &Tier, mode: Mode, edge: u32, sel: Selection) -> Vec<Phase> {
    let _glue = crate::CompileGlueGuard::enable();
    let te_streamed = sel.scope != TransformerComponent::Dit;
    let dit_streamed = sel.scope != TransformerComponent::TextEncoder;
    let te_resident = t.te_bytes(te_streamed);
    let dit_resident = t.dit_bytes(dit_streamed);
    let vae = t.vae_bytes();
    let control_layers = crate::control_transformer::CONTROL_LAYERS_PLACES.len() as u64;
    let control_resident = if mode.control() {
        t.control_resident()
            + if dit_streamed {
                0
            } else {
                // Not windowed: the 15 main blocks resident, packed.
                t.control_block_packed(false) * control_layers
            }
    } else {
        0
    };
    let heavy = dit_resident + vae + control_resident;
    let (s, c0) = (TOKENS, TOKENS as u64);
    let lat = (edge / 8) as i32;
    let latent_elements = LATENT_CHANNELS * u64::from(edge / 8) * u64::from(edge / 8);
    // The f32 caption features of img2img / the bf16 cast of text-to-image.
    let cap_bytes = ab(if mode.img2img() { 4 } else { 2 }, &[c0, CAP_DIM]);

    // Conditioning: the encoder (or its non-layer tensors, each layer read in its quantum).
    let hidden = f32_bytes(&[1, s, 2560]);
    let mut quanta = vec![
        ("Qwen3 embed", t.te.embed_bytes(1, s)),
        (
            "Qwen3 layer",
            hidden
                + t.te_layer.quantum_scratch(1, s, 4, METAL)
                + if te_streamed {
                    bytes_where(&t.te_dir, |k| k.starts_with("model.layers.0."))
                } else {
                    0
                },
        ),
        (
            "caption features",
            ab(4, &[c0]) + ab(4, &[c0, CAP_DIM]) + hidden,
        ),
    ];
    if !mode.img2img() {
        quanta.push((
            "caption cast",
            ab(4, &[c0, CAP_DIM]) + ab(2, &[c0, CAP_DIM]),
        ));
    }
    let conditioning = Phase {
        name: "conditioning",
        resident: te_resident + if sel.staged { 0 } else { heavy },
        quantum: largest(quanta),
    };
    // The conditioning output and, unstaged, the encoder stay resident from here on.
    let carried = cap_bytes + if sel.staged { 0 } else { te_resident };

    let mut phases = vec![conditioning];
    // Encode (img2img init / control context): inside the denoise phase, its components resident.
    if mode.encodes() {
        let mut quanta = encode_quanta(&t.vae, edge as i32);
        if mode.control() {
            let latents = [16, 1, lat, lat];
            let plane = |c: i32| f32_bytes(&[c, 1, lat, lat]);
            quanta.push((
                "control context",
                f32_bytes(&latents) + plane(1) + plane(16) + plane(33),
            ));
        }
        let encoded = if mode == Mode::ControlReference {
            // The first encode's output stays while the second runs.
            f32_bytes(&[1, 16, lat, lat])
        } else {
            0
        };
        phases.push(Phase {
            name: "encode",
            resident: heavy + carried + f32_bytes(&[1, 3, edge as i32, edge as i32]) + encoded,
            quantum: largest(quanta),
        });
    }

    // Denoise.
    let x = lazy(&[16, 1, lat, lat], Dtype::Bfloat16);
    let n = (lat / 2) * (lat / 2);
    let cap = lazy(&[TOKENS, CAP_DIM as i32], Dtype::Float32);
    let l = n + TOKENS;
    let unified = lazy(&[1, l, DIM], Dtype::Float32);
    let freqs = lazy(&[l, 64, 2], Dtype::Float32);
    let x_stream = lazy(&[1, n, DIM], Dtype::Float32);
    let x_freqs = lazy(&[n, 64, 2], Dtype::Float32);
    let cap_stream = lazy(&[1, TOKENS, DIM], Dtype::Float32);
    let cap_freqs = lazy(&[TOKENS, 64, 2], Dtype::Float32);
    let plan = AttentionPlan::budgeted(AttentionBudget::CONSTRAINED);
    let read = if dit_streamed { t.dit_block_read() } else { 0 };
    let unified_bytes = f32_bytes(&[1, l, DIM]);
    let hint_add = if mode.control() {
        crate::control_transformer::add_hint_bytes(&unified)
    } else {
        0
    };
    let mut quanta = vec![
        ("DiT prepare", t.dit.prepare_bytes((16, 1, lat, lat), &cap)),
        ("DiT image embed", t.dit.embed_bytes(&x, 0, METAL)),
        ("DiT caption embed", t.dit.caption_bytes(&cap, METAL)),
        (
            "DiT refiner block",
            f32_bytes(&[1, n, DIM])
                + t.dit.noise_refiner[0].quantum_scratch(&x_stream, &x_freqs, plan, METAL)
                + if mode.control() {
                    crate::control_transformer::add_hint_bytes(&x_stream)
                } else {
                    0
                },
        ),
        (
            "DiT context block",
            f32_bytes(&[1, TOKENS, DIM])
                + t.dit.context_refiner[0].quantum_scratch(&cap_stream, &cap_freqs, plan, METAL),
        ),
        (
            "DiT block",
            unified_bytes
                + t.dit.layers[0].quantum_scratch(&unified, &freqs, plan, METAL)
                + read
                + hint_add,
        ),
        (
            "DiT output",
            unified_bytes + t.dit.output_bytes(&unified, METAL),
        ),
        (
            "sampler step",
            mlx_gen::ios_quantum::sampler_step_bytes(latent_elements),
        ),
    ];
    let mut denoise_resident =
        heavy + carried + ab(2, &[latent_elements]) + f32_bytes(&[1, TOKENS, DIM]);
    if mode.img2img() {
        denoise_resident += f32_bytes(&[16, 1, lat, lat]);
    }
    if mode.control() {
        // The control context, its cached embedding, the refiner's 2 hints and threaded state, and
        // the 15 main hints (live through the base stack).
        denoise_resident += f32_bytes(&[33, 1, lat, lat])
            + 4 * f32_bytes(&[1, n, DIM])
            + control_layers * unified_bytes;
        let control_read = if dit_streamed {
            t.control_block_read(dense) + t.bits.map_or(0, |b| t.control_block.quantize_bytes(b))
        } else {
            0
        };
        quanta.push((
            "control block",
            2 * unified_bytes
                + t.control_block
                    .quantum_scratch(&unified, &unified, &freqs, plan, METAL)
                + control_read,
        ));
    }
    phases.push(Phase {
        name: "denoise",
        resident: denoise_resident,
        quantum: largest(quanta),
    });

    // Decode: staged, the DiT (and the control branch) shed.
    let decoder = t.vae.decoder();
    let (conv_in, mid, ups, conv_out) = decoder.parts();
    let tile = sel.tile;
    let tiled = edge > tile;
    let mut quanta = Vec::new();
    let ch = |w: &Array| mlx_gen::ios_quantum::conv_channels(w);
    let mid_ch = ch(conv_in.weight()).1;
    let latent = lazy(&[1, mid_ch, lat, lat], Dtype::Float32);
    let call = ConvCall::new(&[1, lat, lat, 16], 4, conv_in.weight(), 1, 1);
    quanta.push((
        "VAE decode conv_in",
        f32_bytes(&[1, 16, lat, lat])
            + mlx_gen::ios_quantum::single_conv_bytes(&call, 0, 0, METAL)
            + 2 * ab(4, &[latent_elements]),
    ));
    let (r0, _attn, _) = mid.parts();
    quanta.push((
        "VAE mid resnet",
        f32_bytes(&[1, mid_ch, lat, lat]) + r0.quantum_scratch(&latent, METAL),
    ));
    let rows =
        crate::vae::VaeAttention::bounded_plan()
            .budget
            .query_block(1, 1, lat * lat, lat * lat);
    quanta.push((
        "VAE mid attention",
        f32_bytes(&[1, mid_ch, lat, lat])
            + mlx_gen::ios_quantum::vae_attention_bytes(1, lat * lat, mid_ch, rows, 4, METAL),
    ));
    let mut res = lat;
    let mut remaining = 8;
    let edge_at = |remaining: i32| ((tile as i32 + remaining - 1) / remaining).max(3);
    for up in ups {
        let (resnets, upsampler) = up.parts();
        for r in resnets {
            let (c1, c2) = r.conv_weights();
            let (cin, cout) = ch(c1);
            let input = f32_bytes(&[1, cin, res, res]);
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
                quanta.push(("VAE tiled resnet stage + tile", input + stage + tile_bytes));
            } else {
                let x = lazy(&[1, cin, res, res], Dtype::Float32);
                quanta.push(("VAE resnet", input + r.quantum_scratch(&x, METAL)));
            }
        }
        if let Some(u) = upsampler {
            let w = u.weight();
            let (cin, cout) = ch(w);
            let input = f32_bytes(&[1, cin, res, res]);
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
                    input
                        + stage
                        + mlx_gen::ios_quantum::tile_bytes(&[1, t, t, cin], 4, w, false, METAL),
                ));
            } else {
                let call = ConvCall::new(&[1, next, next, cin], 4, w, 1, 1);
                quanta.push((
                    "VAE upsample",
                    input + mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, METAL),
                ));
            }
            res = next;
        }
    }
    let (cin, cout) = ch(conv_out.weight());
    let input = f32_bytes(&[1, cin, res, res]);
    if tiled {
        quanta.push((
            "VAE tiled conv_out stage + tile",
            input
                + mlx_gen::ios_quantum::tiled_stage_bytes(&[1, res, res, cin], cin, cout, 4, false)
                + mlx_gen::ios_quantum::tile_bytes(
                    &[1, tile as i32, tile as i32, cin],
                    4,
                    conv_out.weight(),
                    false,
                    METAL,
                ),
        ));
    } else {
        let call = ConvCall::new(&[1, res, res, cin], 4, conv_out.weight(), 1, 1);
        quanta.push((
            "VAE conv_out",
            input + mlx_gen::ios_quantum::norm_conv_bytes(&call, METAL),
        ));
    }
    let pixels = 3 * u64::from(edge) * u64::from(edge);
    quanta.push((
        "decoded to image",
        ab(4, &[pixels]) + crate::quantum::zimage_pixel_conversion_bytes(pixels, 4),
    ));
    phases.push(Phase {
        name: "decode",
        resident: vae
            + ab(2, &[latent_elements])
            + if sel.staged { 0 } else { heavy - vae + carried },
        quantum: largest(quanta),
    });
    phases
}

/// The campaign's rung-4 selections, quality first (as `selection_space` orders them).
fn selections() -> Vec<Selection> {
    let mut out = Vec::new();
    for staged in [false, true] {
        for scope in [
            TransformerComponent::Dit,
            TransformerComponent::TextEncoder,
            TransformerComponent::Both,
        ] {
            for &tile in crate::memory_strategy::DECODE_TILE_EDGES {
                out.push(Selection {
                    staged,
                    scope,
                    tile,
                });
            }
        }
    }
    out
}

#[test]
#[ignore = "reads a provisioned Z-Image Turbo snapshot's headers (ZIMAGE_TIER_ROOT)"]
fn reachable_sizes() {
    let root = PathBuf::from(std::env::var_os("ZIMAGE_TIER_ROOT").expect("ZIMAGE_TIER_ROOT"));
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let limit = ENTITLED_MIB * MIB;
    let dense = Tier::load(&root.join("bf16"), &root.join("bf16"), None);
    let tiers = [("q4", Some(4)), ("q8", Some(8)), ("bf16", None)];
    let loaded: Vec<_> = tiers
        .iter()
        .map(|&(name, bits)| (name, Tier::load(&root.join(name), &root.join("bf16"), bits)))
        .collect();
    let mib = |b: u64| b.div_ceil(MIB);
    println!(
        "| mode | tier | edge | first selection ({}) peak | binding | first fitting selection (quality first) | its peak | its binding phase / quantum |",
        selections()[0].token()
    );
    println!("|---|---|---:|---:|---|---|---:|---|");
    for mode in Mode::ALL {
        for (name, tier) in &loaded {
            for edge in [512u32, 768, 1024] {
                let worst = |phases: &[Phase]| {
                    let p = phases.iter().max_by_key(|p| p.peak()).unwrap();
                    (p.peak(), format!("{} {}", p.name, p.quantum.0))
                };
                let first = predict(tier, &dense, mode, edge, selections()[0]);
                let (first_peak, first_binding) = worst(&first);
                let fitting = selections().into_iter().find_map(|sel| {
                    let phases = predict(tier, &dense, mode, edge, sel);
                    let (peak, binding) = worst(&phases);
                    (peak <= limit).then(|| (sel, peak, binding))
                });
                let (fit, fit_peak, fit_binding) = match fitting {
                    Some((sel, peak, binding)) => (sel.token(), mib(peak).to_string(), binding),
                    None => ("none".into(), "-".into(), "-".into()),
                };
                println!(
                    "| {} | {name} | {edge} | {} | {first_binding} | {fit} | {fit_peak} | {fit_binding} |",
                    mode.name(),
                    mib(first_peak),
                );
            }
        }
    }
    // Every selection of the q4 1024 text cell, in the ladder's order (the walk a capture takes).
    let (_, q4) = &loaded[0];
    println!("\nq4 / 1024 / text, every rung-4 selection in ladder order:");
    for sel in selections() {
        let phases = predict(q4, &dense, Mode::Text, 1024, sel);
        let p = phases.iter().max_by_key(|p| p.peak()).unwrap();
        println!(
            "  {:<32} {:>6} MiB  {}  ({} {})",
            sel.token(),
            mib(p.peak()),
            if p.peak() <= limit { "fits" } else { "refused" },
            p.name,
            p.quantum.0
        );
    }
    println!(
        "(MiB; peak = resident + the largest quantum's live input and scratch + the {} MiB reserve; entitled launch headroom {} MiB)",
        PROCESS_RESERVE_BYTES / MIB,
        ENTITLED_MIB
    );
}
