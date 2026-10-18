use super::*;
use mlx_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, Dtype as Stored};

const MIB: u64 = 1 << 20;
const ENTITLED: u64 = 6142 * MIB;
const HALF: u64 = 2;

/// The caller-provisioned snapshot for `name`, from `SDXL_PREDICTION_SNAPSHOTS` (a directory
/// holding one sub-directory per route — `sdxl`, `illustrious_xl_v1`, … each with its `bf16` /
/// `q8` / `q4` tier directories — plus `vae_fp16_fix` and `controlnet_tile`). Inference never
/// derives a cache path.
fn snapshot(name: &str) -> Option<std::path::PathBuf> {
    let root = std::path::PathBuf::from(std::env::var_os("SDXL_PREDICTION_SNAPSHOTS")?);
    Some(root.join(name)).filter(|p| p.is_dir())
}

/// Resident bytes of `file` after the iOS load: every float tensor not at `target` is cast to
/// it (`width` bytes) — on a packed file only the dense leaves (never codes or `.scales` /
/// `.biases`) — else its stored bytes.
fn resident(file: &std::path::Path, target: Stored, width: u64) -> u64 {
    let headers = safetensors_path_tensor_headers(file).unwrap();
    let packed = headers.iter().any(|h| h.name.ends_with(".scales"));
    headers
        .iter()
        .map(|h| {
            let floating = matches!(
                h.dtype,
                Stored::F16 | Stored::BF16 | Stored::F32 | Stored::F64
            );
            let companion = h.name.ends_with(".scales") || h.name.ends_with(".biases");
            let casts = h.dtype != target && (!packed || (floating && !companion));
            if casts {
                h.shape.iter().product::<usize>() as u64 * width
            } else {
                h.data_bytes
            }
        })
        .sum()
}

/// A dense f16 ControlNet's resident bytes after `quantize(bits)` (its attention/FFN/embedding
/// linears pack at `bits` + one bf16 scale and bias per 64-group; convs and norms stay f16).
fn quantized_controlnet(file: &std::path::Path, bits: Option<u64>) -> u64 {
    safetensors_path_tensor_headers(file)
        .unwrap()
        .iter()
        .map(|h| {
            let elements = h.shape.iter().product::<usize>() as u64;
            let linear = h.shape.len() == 2
                && h.name.ends_with(".weight")
                && !h.name.contains("conv")
                && !h.name.contains("norm");
            match bits {
                Some(b) if linear => elements * b / 8 + elements / 64 * 2 * 2,
                _ => elements * HALF,
            }
        })
        .sum()
}

/// The largest tensor of `file` as stored (a load's read transient; `before_cast_for_load`'s
/// scratch beyond the resident bytes for an uncast file).
fn largest(file: &std::path::Path) -> u64 {
    safetensors_path_tensor_headers(file)
        .unwrap()
        .iter()
        .map(|h| h.data_bytes)
        .max()
        .unwrap_or(0)
}

fn w(o: i32, k: i32, c: i32) -> Array {
    Array::zeros::<f32>(&[o, k, k, c])
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap()
}

/// The activation width a component file's projections run its f16 inputs at: f32 when its
/// packed projections carry scales of another dtype (MLX promotes the quantized matmul to
/// `result_type(x, scales)`), else f16.
fn activation_width(file: &std::path::Path) -> u64 {
    let promotes = safetensors_path_tensor_headers(file)
        .unwrap()
        .iter()
        .any(|h| h.name.ends_with(".scales") && h.dtype != Stored::F16);
    if promotes {
        4
    } else {
        HALF
    }
}

fn worst(quanta: &[(String, u64)]) -> (String, u64) {
    quanta.iter().max_by_key(|q| q.1).cloned().unwrap()
}

/// Every counted quantum of one SDXL U-Net forward at latent edge `l` (sequential CFG: one
/// latent row, the pair's text memory), `ip` IP tokens, its activations at `act` bytes from the
/// first resnet's time-embedding sum on (the stem and that resnet's first half run the f16
/// latent).
fn unet_quanta(l: i32, ip: i32, act: u64) -> Vec<(String, u64)> {
    let m = Backend::Metal;
    let mut q = Vec::new();
    let x = Array::zeros::<f32>(&[1, l, l, 4])
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let ctx = Array::zeros::<f32>(&[2, 77, 2048])
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    q.push((
        UNET_STEM.into(),
        crate::unet::stem_bytes(&x, &ctx, &w(320, 3, 4), None, 0, m),
    ));
    let (l1, l2) = (l / 2, l / 4);
    // (grid, pre-concat channels, conv1 in, out)
    let resnets = [
        (l, 320, 320, 320),
        (l1, 320, 320, 640),
        (l1, 640, 640, 640),
        (l2, 640, 640, 1280),
        (l2, 1280, 1280, 1280),
        (l2, 1280, 2560, 1280),
        (l2, 1280, 1920, 1280),
        (l1, 1280, 1920, 640),
        (l1, 640, 1280, 640),
        (l1, 640, 960, 640),
        (l, 640, 960, 320),
        (l, 320, 640, 320),
    ];
    for (i, (h, cx, cin, cout)) in resnets.into_iter().enumerate() {
        let act_in = if i == 0 { HALF } else { act };
        q.push((
            format!("{UNET_RESNET} {cin}->{cout} @{h}"),
            resnet_bytes(
                &[1, h, h, cx],
                act_in,
                act,
                &w(cout, 3, cin),
                &w(cout, 3, cout),
                m,
            ),
        ));
    }
    for (h, c) in [(l, 320), (l1, 640)] {
        let call = ConvCall::new(&[1, h, h, c], act, &w(c, 3, c), 2, 1);
        q.push((
            format!("{UNET_DOWNSAMPLE} @{h}"),
            single_conv_bytes(&call, 0, 0, m),
        ));
    }
    for (h, c) in [(l2, 1280), (l1, 640)] {
        let call = ConvCall::new(&[1, 2 * h, 2 * h, c], act, &w(c, 3, c), 1, 1);
        q.push((
            format!("{UNET_UPSAMPLE} @{h}"),
            single_conv_bytes(&call, 1, 0, m),
        ));
    }
    for (h, dim, heads) in [(l1, 640u64, 10), (l2, 1280, 20)] {
        let tokens = h * h;
        q.push((
            format!("{TRANSFORMER_BLOCK} {dim} @{h}"),
            transformer_block_bytes(
                1,
                2,
                tokens,
                dim,
                heads,
                64,
                77 + ip,
                4 * dim,
                tokens,
                ip > 0,
                act,
                m,
            ),
        ));
        let plane = |c: u64| (tokens as u64) * c * act;
        q.push((
            format!("{TRANSFORMER_PROJ_IN} @{h}"),
            plane(dim) * (GROUP_NORM_BUFFERS + 1) + page_slack(GROUP_NORM_BUFFERS + 1),
        ));
    }
    let call = ConvCall::new(&[1, l, l, 320], act, &w(4, 3, 320), 1, 1);
    q.push((UNET_CONV_OUT.into(), norm_conv_bytes(&call, m)));
    q
}

/// The ControlNet branch's own quanta at latent edge `l`: its conditioning embedding at full
/// resolution (once per request) and the zero-convs; its encoder copy is the U-Net's down/mid.
fn controlnet_quanta(l: i32) -> Vec<(String, u64)> {
    let m = Backend::Metal;
    let mut q = Vec::new();
    let e = 8 * l;
    let convs = [
        (e, 3, 16, 1),
        (e, 16, 16, 1),
        (e, 16, 32, 2),
        (e / 2, 32, 32, 1),
        (e / 2, 32, 96, 2),
        (e / 4, 96, 96, 1),
        (e / 4, 96, 256, 2),
        (e / 8, 256, 320, 1),
    ];
    for (h, c, o, s) in convs {
        let call = ConvCall::new(&[1, h, h, c], HALF, &w(o, 3, c), s, 1);
        q.push((
            format!("{CONTROL_COND_CONV} {c}->{o} @{h}"),
            single_conv_bytes(&call, 0, 1, m),
        ));
    }
    let call = ConvCall::new(&[1, l, l, 320], HALF, &w(320, 1, 320), 1, 0);
    q.push((CONTROL_ZERO_CONV.into(), single_conv_bytes(&call, 0, 0, m)));
    q
}

/// The fp16-fix VAE decode at latent edge `l`: the dense head (conv_in, mid resnets, the
/// chunked mid attention) and either the dense tail or the layer-wise tiled tail at output
/// tile edge `tile` (each stage priced with its widest tile), and the pixel conversion.
fn decode_quanta(l: i32, tile: Option<i32>) -> Vec<(String, u64)> {
    let m = Backend::Metal;
    let mut q = Vec::new();
    let call = ConvCall::new(&[1, l, l, 4], HALF, &w(512, 3, 4), 1, 1);
    q.push((VAE_DECODE_IN.into(), single_conv_bytes(&call, 2, 0, m)));
    q.push((
        format!("{VAE_RESNET} mid @{l}"),
        resnet_bytes(
            &[1, l, l, 512],
            HALF,
            HALF,
            &w(512, 3, 512),
            &w(512, 3, 512),
            m,
        ),
    ));
    let n = l * l;
    let rows = mlx_gen::attention::AttentionBudget::from_score_elements(
        VAE_ATTENTION_SCORE_ELEMENTS,
        true,
    )
    .query_block(1, 1, n, n);
    q.push((
        format!("{VAE_ATTENTION} ({rows}-row chunks)"),
        vae_attention_bytes(1, n, 512, rows, HALF, m),
    ));
    // (input grid, cin, cout, upsample after)
    let blocks = [
        (l, 512, 512, true),
        (2 * l, 512, 512, true),
        (4 * l, 512, 256, true),
        (8 * l, 256, 128, false),
    ];
    let bias = |c: i32| {
        Array::zeros::<f32>(&[c])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap()
    };
    for (i, (h, cin, cout, up)) in blocks.into_iter().enumerate() {
        let scale = 8 / (1 << i);
        for j in 0..3 {
            let ci = if j == 0 { cin } else { cout };
            let label = format!("{VAE_RESNET} {ci}->{cout} @{h}");
            q.push((
                label,
                match tile {
                    None => resnet_bytes(
                        &[1, h, h, ci],
                        HALF,
                        HALF,
                        &w(cout, 3, ci),
                        &w(cout, 3, cout),
                        m,
                    ),
                    Some(t) => {
                        let e = ((t + scale - 1) / scale).max(3);
                        tiled_stage_bytes(&[1, h, h, cout], ci, cout, HALF, true)
                            + tile_bytes(
                                &[1, e.min(h) + 2, e.min(h) + 2, ci],
                                TileSource {
                                    full_width: h,
                                    normalized: true,
                                },
                                HALF,
                                &w(cout, 3, ci),
                                false,
                                m,
                            )
                    }
                },
            ));
        }
        if up {
            let label = format!("{VAE_UPSAMPLE} @{}", 2 * h);
            q.push((
                label,
                match tile {
                    None => single_conv_bytes(
                        &ConvCall::new(&[1, 2 * h, 2 * h, cout], HALF, &w(cout, 3, cout), 1, 1),
                        1,
                        0,
                        m,
                    ),
                    Some(t) => {
                        let e = ((t + scale / 2 - 1) / (scale / 2)).max(3);
                        tiled_stage_f32_accum(
                            &[1, 2 * h, 2 * h, cout],
                            &w(cout, 3, cout),
                            &bias(cout),
                            HALF,
                        ) + tile_bytes(
                            &[1, e.min(2 * h) + 2, e.min(2 * h) + 2, cout],
                            TileSource {
                                full_width: 2 * h,
                                normalized: false,
                            },
                            HALF,
                            &w(cout, 3, cout),
                            true,
                            m,
                        )
                    }
                },
            ));
        }
    }
    let e8 = 8 * l;
    q.push((
        VAE_DECODE_OUT.into(),
        match tile {
            None => norm_conv_bytes(
                &ConvCall::new(&[1, e8, e8, 128], HALF, &w(3, 3, 128), 1, 1),
                m,
            ),
            Some(t) => {
                tiled_stage_bytes(&[1, e8, e8, 3], 128, 3, HALF, false)
                    + tile_bytes(
                        &[1, t.min(e8) + 2, t.min(e8) + 2, 128],
                        TileSource {
                            full_width: e8,
                            normalized: true,
                        },
                        HALF,
                        &w(3, 3, 128),
                        false,
                        m,
                    )
            }
        },
    ));
    let decoded = Array::zeros::<f32>(&[1, e8, e8, 3]).unwrap();
    q.push((DECODED_TO_IMAGE.into(), decoded_to_image_bytes(&decoded)));
    q
}

/// The bounded (tiled) fp16-fix VAE encode at image edge `8l` with `tile` input pixels: each
/// full-resolution layer a tiled stage priced with its widest tile, the latent-resolution mid
/// block dense with the chunked attention, the output head.
fn encode_tiled_quanta(l: i32, tile: i32) -> Vec<(String, u64)> {
    let m = Backend::Metal;
    let e = 8 * l;
    let crop = |edge: i32, h: i32| edge.min(h) + 2;
    let mut q = vec![(
        format!("{VAE_TILED_STAGE} conv_in @{e}"),
        tiled_stage_bytes(&[1, e, e, 128], 3, 128, HALF, false)
            + tile_bytes(
                &[1, crop(tile, e), crop(tile, e), 3],
                TileSource {
                    full_width: e,
                    normalized: false,
                },
                HALF,
                &w(128, 3, 3),
                false,
                m,
            ),
    )];
    let mut scale = 1;
    for (h, cin, cout) in [
        (e, 128, 128),
        (e / 2, 128, 256),
        (e / 4, 256, 512),
        (e / 8, 512, 512),
    ] {
        let edge = ((tile + scale - 1) / scale).max(3);
        for ci in [cin, cout] {
            q.push((
                format!("{VAE_TILED_STAGE} resnet {ci}->{cout} @{h}"),
                tiled_stage_bytes(&[1, h, h, cout], ci, cout, HALF, true)
                    + tile_bytes(
                        &[1, crop(edge, h), crop(edge, h), ci],
                        TileSource {
                            full_width: h,
                            normalized: true,
                        },
                        HALF,
                        &w(cout, 3, ci),
                        false,
                        m,
                    ),
            ));
        }
        if h > e / 8 {
            scale *= 2;
            let core = ((tile + scale - 1) / scale).max(1).min(h / 2);
            q.push((
                format!("{VAE_TILED_STAGE} downsample @{h}"),
                downsample_stage_bytes(&[1, h / 2, h / 2, cout], HALF)
                    + downsample_tile_bytes(
                        &[1, 2 * core + 1, 2 * core + 1, cout],
                        HALF,
                        &w(cout, 3, cout),
                        m,
                    ),
            ));
        }
    }
    q.push((
        format!("{VAE_RESNET} mid @{l}"),
        resnet_bytes(
            &[1, l, l, 512],
            HALF,
            HALF,
            &w(512, 3, 512),
            &w(512, 3, 512),
            m,
        ),
    ));
    let n = l * l;
    let rows = mlx_gen::attention::AttentionBudget::from_score_elements(
        VAE_ATTENTION_SCORE_ELEMENTS,
        true,
    )
    .query_block(1, 1, n, n);
    q.push((
        VAE_ATTENTION.into(),
        vae_attention_bytes(1, n, 512, rows, HALF, m),
    ));
    let call = ConvCall::new(&[1, l, l, 512], HALF, &w(8, 3, 512), 1, 1);
    q.push((
        VAE_ENCODE_OUT.into(),
        norm_conv_bytes(&call, m) + 4 * call.out_bytes() * 2,
    ));
    q
}

// ---------------------------------------------------------------------------------------------
// The BEFORE model: the pre-epic-16774 prediction (the U-Net, ControlNet and IP-Adapter wholly
// resident beside the VAE through the heavy phase; rung 4 not modelled), kept verbatim so the
// table shows what each lever changed.
// ---------------------------------------------------------------------------------------------

/// The old model's per-phase peaks (resident + largest quantum, bytes, no reserve) for one cell.
fn before_cell(
    dir: &std::path::Path,
    fix: &std::path::Path,
    cn: &std::path::Path,
    bits: Option<u64>,
    mode: &str,
    edge: i32,
) -> [Option<u64>; 5] {
    let file = |sub: &str, stem: &str| {
        let fp16 = dir.join(sub).join(format!("{stem}.fp16.safetensors"));
        if fp16.exists() {
            fp16
        } else {
            dir.join(sub).join(format!("{stem}.safetensors"))
        }
    };
    let vae = resident(fix, Stored::F16, HALF);
    let te = resident(&file("text_encoder", "model"), Stored::F16, HALF)
        + resident(&file("text_encoder_2", "model"), Stored::F16, HALF);
    let unet_file = file("unet", "diffusion_pytorch_model");
    let unet = resident(&unet_file, Stored::F16, HALF);
    let control = quantized_controlnet(cn, bits);
    let ip_pairs =
        341_000_000u64 * bits.map_or(16, |b| b) / 8 + bits.map_or(0, |_| 341_000_000 / 64 * 4);
    let ip_resident = (632_000_000 + 83_000_000) * HALF + ip_pairs;
    let te_act = activation_width(&file("text_encoder", "model"))
        .max(activation_width(&file("text_encoder_2", "model")));
    let act = activation_width(&unet_file);
    let l = edge / 8;
    let peak = |resident: u64, q: (String, u64)| resident + q.1;
    let text = worst(&[
        (
            CLIP_LAYER.to_string(),
            clip_layer_bytes(2, 77, 768, 3072, 12, 64, true, te_act, Backend::Metal),
        ),
        (
            CLIP_LAYER.to_string(),
            clip_layer_bytes(2, 77, 1280, 5120, 20, 64, true, te_act, Backend::Metal),
        ),
    ]);
    let mut load = (te + largest(&file("text_encoder_2", "model"))).max(unet + largest(&unet_file));
    if mode == "control" {
        load = load.max(unet + vae + resident(cn, Stored::F16, HALF) + largest(cn));
    }
    let mut conditioning = peak(te, text);
    if mode == "ip_adapter" {
        let vision = clip_layer_bytes(1, 257, 1280, 5120, 16, 80, false, HALF, Backend::Metal);
        conditioning = conditioning.max(unet + vae + ip_resident + vision);
    }
    let encode = matches!(mode, "reference" | "mask").then(|| {
        peak(
            unet + vae,
            worst(&encode_tiled_quanta(l, crate::vae::IOS_ENCODE_TILE_PX)),
        )
    });
    let denoise = match mode {
        "control" => {
            let mut cq = unet_quanta(l, 0, act);
            cq.extend(controlnet_quanta(l));
            peak(unet + vae + control, worst(&cq))
        }
        "ip_adapter" => peak(unet + vae + ip_resident, worst(&unet_quanta(l, 16, act))),
        _ => peak(unet + vae, worst(&unet_quanta(l, 0, act))),
    };
    let decode = peak(vae, worst(&decode_quanta(l, Some(256))));
    [
        Some(load),
        Some(conditioning),
        encode,
        Some(denoise),
        Some(decode),
    ]
}

// ---------------------------------------------------------------------------------------------
// The AFTER pricing: `crate::ios_pricing` over the real headers at the fully optimized selection.
// ---------------------------------------------------------------------------------------------

/// Write a header-only (sparse) safetensors file: the real JSON header, a file of the payload's
/// length that is never written. Pricing reads headers only.
fn write_header_only(path: &std::path::Path, tensors: &[(String, Vec<usize>)]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<usize>() * 2;
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F16", "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        offset += bytes;
    }
    let mut json = serde_json::Value::Object(header).to_string();
    while !json.len().is_multiple_of(8) {
        json.push(' ');
    }
    let mut data = (json.len() as u64).to_le_bytes().to_vec();
    data.extend(json.as_bytes());
    std::fs::write(path, &data).unwrap();
    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_len((data.len() + offset) as u64).unwrap();
}

/// The IP-Adapter-plus SDXL (ViT-H/14) snapshot's tensors, from the published architecture — the
/// weights are not on this host, so its sizes are config-derived (f16): the 32-layer ViT-H image
/// encoder, the 4-layer Resampler (16 queries, 2048 out) and one decoupled K/V pair per SDXL
/// cross-attention block in `install_ip_adapter`'s walk order (down 1, down 2, up 0, up 1, mid).
fn write_ip_adapter(dir: &std::path::Path) {
    let h = 1280usize;
    let mut enc: Vec<(String, Vec<usize>)> = vec![
        ("vision_model.embeddings.class_embedding".into(), vec![h]),
        (
            "vision_model.embeddings.patch_embedding.weight".into(),
            vec![h, 3, 14, 14],
        ),
        (
            "vision_model.embeddings.position_embedding.weight".into(),
            vec![257, h],
        ),
        ("vision_model.pre_layrnorm.weight".into(), vec![h]),
        ("vision_model.pre_layrnorm.bias".into(), vec![h]),
        ("vision_model.post_layernorm.weight".into(), vec![h]),
        ("vision_model.post_layernorm.bias".into(), vec![h]),
        ("visual_projection.weight".into(), vec![1024, h]),
    ];
    for i in 0..32 {
        let l = format!("vision_model.encoder.layers.{i}");
        for n in ["layer_norm1", "layer_norm2"] {
            enc.push((format!("{l}.{n}.weight"), vec![h]));
            enc.push((format!("{l}.{n}.bias"), vec![h]));
        }
        for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            enc.push((format!("{l}.self_attn.{p}.weight"), vec![h, h]));
            enc.push((format!("{l}.self_attn.{p}.bias"), vec![h]));
        }
        enc.push((format!("{l}.mlp.fc1.weight"), vec![4 * h, h]));
        enc.push((format!("{l}.mlp.fc1.bias"), vec![4 * h]));
        enc.push((format!("{l}.mlp.fc2.weight"), vec![h, 4 * h]));
        enc.push((format!("{l}.mlp.fc2.bias"), vec![h]));
    }
    write_header_only(
        &dir.join(crate::loader::IP_ADAPTER_IMAGE_ENCODER_FILE),
        &enc,
    );
    let mut ip: Vec<(String, Vec<usize>)> = vec![
        ("image_proj.latents".into(), vec![1, 16, h]),
        ("image_proj.proj_in.weight".into(), vec![h, h]),
        ("image_proj.proj_in.bias".into(), vec![h]),
        ("image_proj.proj_out.weight".into(), vec![2048, h]),
        ("image_proj.proj_out.bias".into(), vec![2048]),
        ("image_proj.norm_out.weight".into(), vec![2048]),
        ("image_proj.norm_out.bias".into(), vec![2048]),
    ];
    for i in 0..4 {
        let a = format!("image_proj.layers.{i}.0");
        for n in ["norm1", "norm2"] {
            ip.push((format!("{a}.{n}.weight"), vec![h]));
            ip.push((format!("{a}.{n}.bias"), vec![h]));
        }
        ip.push((format!("{a}.to_q.weight"), vec![h, h]));
        ip.push((format!("{a}.to_kv.weight"), vec![2 * h, h]));
        ip.push((format!("{a}.to_out.weight"), vec![h, h]));
        let f = format!("image_proj.layers.{i}.1");
        ip.push((format!("{f}.0.weight"), vec![h]));
        ip.push((format!("{f}.0.bias"), vec![h]));
        ip.push((format!("{f}.1.weight"), vec![4 * h, h]));
        ip.push((format!("{f}.3.weight"), vec![h, 4 * h]));
    }
    let dims = std::iter::repeat_n(640usize, 4)
        .chain(std::iter::repeat_n(1280, 20))
        .chain(std::iter::repeat_n(1280, 30))
        .chain(std::iter::repeat_n(640, 6))
        .chain(std::iter::repeat_n(1280, 10));
    for (i, dim) in dims.enumerate() {
        let id = 2 * i + 1;
        ip.push((format!("ip_adapter.{id}.to_k_ip.weight"), vec![dim, 2048]));
        ip.push((format!("ip_adapter.{id}.to_v_ip.weight"), vec![dim, 2048]));
    }
    write_header_only(&dir.join(crate::loader::IP_ADAPTER_WEIGHT_FILES[1]), &ip);
}

/// The iOS load spec of `route` at `tier` (`mlx-gen-ios-catalog::sdxl::sdxl_load_spec`'s
/// fields: the sequential offload policy, deferred materialization, the staged fp16-fix VAE), with
/// the mode's overlay.
fn ios_spec(
    route: &str,
    dir: &std::path::Path,
    bits: Option<u64>,
    fix: &std::path::Path,
    overlay: (&str, &std::path::Path, &std::path::Path),
) -> mlx_gen::LoadSpec {
    let mut spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(dir.to_path_buf()))
        .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
        .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
    spec.resolved_route = Some(route.into());
    spec.quantize = match bits {
        Some(4) => Some(mlx_gen::Quant::Q4),
        Some(8) => Some(mlx_gen::Quant::Q8),
        _ => None,
    };
    spec.components.insert(
        crate::loader::VAE_FP16_FIX_COMPONENT.into(),
        mlx_gen::WeightsSource::Dir(fix.to_path_buf()),
    );
    let (mode, cn, ip) = overlay;
    if mode == "control" {
        spec.control = Some(mlx_gen::WeightsSource::File(cn.to_path_buf()));
    }
    if mode == "ip_adapter" {
        spec.ip_adapter = Some(mlx_gen::WeightsSource::Dir(ip.to_path_buf()));
    }
    spec
}

/// `(the fully optimized selection's GenerationMemory and label, the rung-0 resident one)`: the
/// deepest `Implemented` rung of `spec`'s contract at the LAST (cheapest) selection of its
/// quality-first `selection_space` — what the campaign captures — and the Resident selection; each
/// with the catalog's sequential CFG.
fn selections(
    spec: &mlx_gen::LoadSpec,
) -> (
    (
        mlx_gen::gen_core::MemorySelection,
        mlx_gen::gen_core::GenerationMemory,
    ),
    (
        mlx_gen::gen_core::MemorySelection,
        mlx_gen::gen_core::GenerationMemory,
    ),
) {
    use mlx_gen::gen_core::{MemoryNumericTier, MemoryStrategy, MemoryStrategySupport};
    let contract = crate::memory_strategy::memory_strategy_contract("sdxl", spec).unwrap();
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let deepest = [
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::StagedResidency,
    ]
    .into_iter()
    .find(|s| {
        contract
            .capability(*s)
            .is_some_and(|c| matches!(c.support, MemoryStrategySupport::Implemented))
    })
    .unwrap_or(MemoryStrategy::Resident);
    let pick = |strategy| {
        let selection = contract
            .selection_space(strategy, tier, false)
            .unwrap()
            .pop()
            .unwrap();
        let mut memory = contract.generation_memory(&selection).unwrap_or_default();
        memory.cfg_batching = Some(mlx_gen::gen_core::CfgBatching::Sequential);
        (selection, memory)
    };
    (pick(deepest), pick(MemoryStrategy::Resident))
}

fn mib(bytes: u64) -> String {
    format!("{:.0}", bytes as f64 / MIB as f64)
}

/// Every SDXL-family route x tier x iOS mode x size: the per-phase peak (resident + largest
/// counted quantum + the 512 MiB reserve, MiB) of (A) the fully optimized selection under the
/// pre-epic model, (B) the same selection priced by `crate::ios_pricing` (the guards' formulas, the
/// levers in place), the binding quantum and the lever that made a previously refused cell fit, and
/// (C) the rung-0 resident configuration. Asserts every (B) cell fits the 6142 MiB entitled launch
/// headroom.
#[test]
#[ignore = "prediction table from on-disk safetensors headers; run with --ignored --nocapture"]
fn reachable_sizes_prediction() {
    if !cpu_child("reachable::reachable_sizes_prediction") {
        return;
    }
    use mlx_gen::ios_pricing::{IosPhase, IosPricingRequest};
    let (Some(fix_dir), Some(cn_dir)) = (snapshot("vae_fp16_fix"), snapshot("controlnet_tile"))
    else {
        println!("SDXL_PREDICTION_SNAPSHOTS lacks vae_fp16_fix / controlnet_tile");
        return;
    };
    let fix = fix_dir.join("diffusion_pytorch_model.safetensors");
    let cn = cn_dir.join("diffusion_pytorch_model.safetensors");
    let ip_tmp = tempfile::tempdir().unwrap();
    write_ip_adapter(ip_tmp.path());
    let routes = [
        "sdxl",
        "illustrious_xl_v1",
        "illustrious_xl_v2",
        "realvisxl",
        "realvisxl_lightning",
    ];
    let modes = [
        ("text", mlx_gen::gen_core::MemoryMode::TextToImage, 0),
        ("reference", mlx_gen::gen_core::MemoryMode::ImageToImage, 1),
        ("mask", mlx_gen::gen_core::MemoryMode::Edit, 1),
        (
            "control",
            mlx_gen::gen_core::MemoryMode::Other("control".into()),
            0,
        ),
        (
            "ip_adapter",
            mlx_gen::gen_core::MemoryMode::Other("ip_adapter".into()),
            1,
        ),
    ];
    println!("| route | tier | mode | edge | selection | load A→B | conditioning A→B | encode A→B | denoise A→B | decode A→B | A | B | B binding (phase: quantum) | lever | rung-0 C |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut refused = Vec::new();
    for route in routes {
        let Some(route_dir) = snapshot(route) else {
            println!("<!-- {route}: not provisioned -->");
            continue;
        };
        for (tier, bits) in [("bf16", None), ("q8", Some(8u64)), ("q4", Some(4))] {
            let dir = route_dir.join(tier);
            if !dir.is_dir() {
                println!("<!-- {route} {tier}: not provisioned -->");
                continue;
            }
            for (mode_name, mode, references) in &modes {
                let spec = ios_spec(route, &dir, bits, &fix_dir, (mode_name, &cn, ip_tmp.path()));
                let ((optimized, memory), (resident_sel, resident_memory)) = selections(&spec);
                let streamable = crate::memory_strategy::streamable(&spec);
                for edge in [512u32, 768, 1024] {
                    let geometry = mlx_gen::gen_core::MemoryGeometry {
                        width: edge,
                        height: edge,
                        batch: 1,
                        frames: 1,
                        reference_count: *references,
                    };
                    let price = |selection, execution| {
                        crate::ios_pricing::ios_pricing(&IosPricingRequest {
                            provider: "sdxl",
                            spec: &spec,
                            mode,
                            geometry,
                            selection,
                            execution,
                            token_window: 77,
                        })
                        .unwrap()
                    };
                    let after = price(&optimized, &memory);
                    let rung0 = price(&resident_sel, &resident_memory);
                    let before = before_cell(&dir, &fix, &cn, bits, mode_name, edge as i32);
                    let reserve = PROCESS_RESERVE_BYTES;
                    let a_peak = before.iter().flatten().max().unwrap() + reserve;
                    let b_peak = after.peak_bytes() + reserve;
                    let c_peak = rung0.peak_bytes() + reserve;
                    let phases = [
                        IosPhase::Load,
                        IosPhase::Conditioning,
                        IosPhase::Encode,
                        IosPhase::Denoise,
                        IosPhase::Decode,
                    ];
                    let cells: Vec<String> = phases
                        .iter()
                        .zip(before)
                        .map(|(phase, a)| {
                            let b = after.phase(*phase).map(|p| p.peak_bytes() + reserve);
                            match (a, b) {
                                (None, None) => "—".into(),
                                (a, b) => format!(
                                    "{}→{}",
                                    a.map_or("—".into(), |a| mib(a + reserve)),
                                    b.map_or("—".into(), mib)
                                ),
                            }
                        })
                        .collect();
                    let binding = after
                        .phases
                        .iter()
                        .max_by_key(|p| p.peak_bytes())
                        .map(|p| format!("{}: {}", p.phase.as_key(), p.binding))
                        .unwrap();
                    let lever = if a_peak <= ENTITLED {
                        "—".to_string()
                    } else if b_peak > ENTITLED {
                        "STILL REFUSED".to_string()
                    } else {
                        let mut levers = Vec::new();
                        if streamable {
                            levers.push("b");
                        }
                        if *mode_name == "ip_adapter" && memory.stage_residency {
                            levers.push("d");
                        }
                        levers.join("+")
                    };
                    if b_peak > ENTITLED {
                        refused.push(format!("{route} {tier} {mode_name} {edge}"));
                    }
                    let selection = format!(
                        "{:?} w{:?} t{:?} s{}",
                        optimized.strategy,
                        memory.transformer_window_size,
                        memory.decode_tile_edge,
                        u8::from(memory.stage_residency)
                    );
                    let verdict = |peak: u64| {
                        format!(
                            "{} {}",
                            mib(peak),
                            if peak <= ENTITLED { "fits" } else { "REFUSED" }
                        )
                    };
                    println!(
                        "| {route} | {tier} | {mode_name} | {edge} | {selection} | {} | {} | {} | {} | {} | {} | {} | {binding} | {lever} | {} |",
                        cells[0], cells[1], cells[2], cells[3], cells[4],
                        verdict(a_peak), verdict(b_peak), verdict(c_peak)
                    );
                }
            }
        }
    }
    assert!(
        refused.is_empty(),
        "fully optimized cells that do not fit: {refused:?}"
    );
}
