//! Reachable-sizes PREDICTION for the iOS phone (epic 16774): per route, tier and edge, the
//! resident bytes of each phase, the largest quantum scratch in it under the chosen ladder rung,
//! and whether `resident + scratch + reserve` fits the entitled launch headroom. Derived on the
//! host from the real snapshot's configs and tensor HEADERS (no payload is read, nothing runs) and
//! the same counted formulas the device guards with. These are predictions, never evidence.
//!
//! `cargo test -p mlx-gen-iris --lib reachable_sizes -- --ignored --nocapture`, with
//! `IRIS_WEIGHTS_DIR` / `IRIS_TEXT_ENCODER_DIR` or the Hugging Face cache snapshots.
//!
//! The rung: generation runs staged residency (the text tower is dropped before the backbone
//! loads) with the attention unbounded — the iOS GPU's fused attention streams its scores, so a
//! query-chunked plan only adds the chunk outputs and their concatenation. Every tier computes at
//! bf16 (the packed tiers' scales are bf16; the bf16 tier is the dense release cast at load).

use std::path::{Path, PathBuf};

use mlx_gen::gen_core::iris::restoration::{plan, Dims, RestorationOptions, TileGeometry};
use mlx_gen::gen_core::iris::{IrisConfig, PROMPT_PREFIX, PROMPT_SUFFIX};
use mlx_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::{Precision, Quant};

use crate::dit::{EmbedGeometry, IrisDiT, PixelBlockGeometry, TextBlockGeometry, TrunkGeometry};

/// The entitled launch headroom of the target phone (MiB).
const ENTITLED_MIB: u64 = 6142;
const MIB: u64 = 1024 * 1024;
const CW: u64 = 2;

/// The caller-provisioned snapshot for `repo` (inference never derives a cache path).
fn snapshot(env: &str, _repo: &str) -> Option<PathBuf> {
    std::env::var_os(env).map(PathBuf::from)
}

fn header(path: &Path, name: &str) -> Vec<usize> {
    mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)
        .unwrap()
        .into_iter()
        .find(|h| h.name == name)
        .unwrap_or_else(|| panic!("{} has no {name}", path.display()))
        .shape
}

/// Backbone widths read from the config and the weight file's headers.
struct Backbone {
    cfg: IrisConfig,
    file: PathBuf,
    prefix: &'static str,
}

impl Backbone {
    fn rows(&self, name: &str) -> u64 {
        header(&self.file, &format!("{}{name}", self.prefix))[0] as u64
    }

    /// Zero columns a `tier` load pads the `[out, in]` projection `name`'s input by.
    fn pad(&self, name: &str, tier: Option<Quant>) -> u64 {
        let input = header(&self.file, &format!("{}{name}", self.prefix))[1];
        match crate::nn::ProjectionLayout::of(
            input,
            crate::nn::BackboneQuant::for_load(tier).unwrap(),
        ) {
            crate::nn::ProjectionLayout::Quantized { padded_in, .. } => (padded_in - input) as u64,
            crate::nn::ProjectionLayout::Dense => 0,
        }
    }

    /// The largest denoise-phase quantum of one forward over an `h × w` batch of `b` at `tier`,
    /// priced for the device (fused norm and attention kernels).
    fn forward_peak(&self, b: u64, h: u64, w: u64, tier: Option<Quant>) -> (u64, &'static str) {
        let m = &self.cfg.model;
        let p = m.patch_size as u64;
        let n = (h / p) * (w / p);
        let t = m.text_len as u64;
        let tokens = n + t;
        let hidden = m.hidden_size as u64;
        let heads = m.num_heads as u64;
        let kv_heads = m.num_kv_heads.unwrap_or(m.num_heads) as u64;
        let layers = self.cfg.text_encoder.hidden_layers.len() as u64;
        let text_dim = m.text_dim as u64;
        let last = m.depth - 1;
        let trunk = |ff: u64, text: bool, down_pad: u64| TrunkGeometry {
            compute_bytes: CW,
            batch: b,
            img_tokens: n,
            txt_tokens: t,
            dim: hidden,
            heads,
            kv_heads,
            ff,
            query_rows: tokens,
            eval_per_chunk: false,
            split_qkv: kv_heads != heads,
            text,
            down_pad,
            img_stream_bytes: 4,
        };
        let dual = TrunkGeometry {
            img_stream_bytes: CW,
            ..trunk(
                self.rows("blocks.0.mlp_x.w1.weight"),
                true,
                self.pad("blocks.0.mlp_x.w2.weight", tier),
            )
        };
        let single = trunk(
            self.rows(&format!("blocks.{last}.mlp.w1.weight")),
            true,
            self.pad(&format!("blocks.{last}.mlp.w2.weight"), tier),
        );
        let text_block = |refiner: bool| {
            let (batch, tokens, dim, heads, ff, pad) = if refiner {
                (
                    b,
                    t,
                    hidden,
                    heads,
                    self.rows("y_embedder.refiner.blocks.0.mlp.w1.weight"),
                    self.pad("y_embedder.refiner.blocks.0.mlp.w2.weight", tier),
                )
            } else {
                (
                    b * t,
                    layers,
                    text_dim,
                    m.text_lap_num_heads as u64,
                    self.rows("y_embedder.layer_blocks.0.mlp.0.weight"),
                    self.pad("y_embedder.layer_blocks.0.mlp.2.weight", tier),
                )
            };
            IrisDiT::text_block_bytes(
                TextBlockGeometry {
                    compute_bytes: CW,
                    batch,
                    tokens,
                    dim,
                    heads,
                    ff,
                    query_rows: tokens,
                    eval_per_chunk: false,
                    refiner,
                    down_pad: pad,
                    stream_bytes: if refiner { CW } else { 4 },
                },
                false,
            )
        };
        let mut quanta: Vec<(u64, &'static str)> = vec![
            (
                IrisDiT::dual_block_bytes(&dual, false)
                    + IrisDiT::trunk_entry_bytes(b, n, t, hidden / heads, hidden),
                "first (dual) trunk block + RoPE/positional entry",
            ),
            (
                IrisDiT::single_block_bytes(&single, false),
                "single trunk block",
            ),
            (
                IrisDiT::pixel_block_bytes(
                    PixelBlockGeometry {
                        compute_bytes: CW,
                        batch: b,
                        patches: n,
                        patch_pixels: p * p,
                        pixel_width: m.pixel.hidden_size as u64,
                        compact_width: m.pixel.attn_hidden_size as u64,
                        modulation_width: self.rows("pixel_blocks.0.adaln.weight"),
                        condition_width: hidden,
                        fc1_width: self.rows("pixel_blocks.0.mlp.fc1.weight"),
                        heads: m.pixel.num_heads as u64,
                        query_rows: n,
                        eval_per_chunk: false,
                    },
                    false,
                ),
                "pixel block",
            ),
            (
                IrisDiT::embed_bytes(EmbedGeometry {
                    compute_bytes: CW,
                    batch: b,
                    patches: n,
                    patch_in: header(
                        &self.file,
                        &format!("{}s_embedder.proj.weight", self.prefix),
                    )[1] as u64,
                    hidden,
                    sinusoid: header(
                        &self.file,
                        &format!("{}t_embedder.mlp.0.weight", self.prefix),
                    )[1] as u64,
                    t_hidden: self.rows("t_embedder.mlp.0.weight"),
                    t_out: self.rows("t_embedder.mlp.2.weight"),
                    cores: 2 * self.rows("modulation_cores.adaln_img.weight"),
                }) + mlx_gen::ios_quantum::sampler_step_bytes(b * m.in_channels as u64 * h * w),
                "embed + sampler allowance",
            ),
            (text_block(false), "text adapter layer block"),
            (text_block(true), "text refiner block"),
            (
                IrisDiT::text_pool_bytes(CW, b, t, layers, text_dim, hidden),
                "text pool",
            ),
            (
                IrisDiT::text_output_bytes(CW, b, t, hidden, false),
                "text output",
            ),
            (
                IrisDiT::pixel_embed_bytes(
                    CW,
                    b,
                    h,
                    w,
                    p,
                    hidden,
                    m.pixel.hidden_size as u64,
                    header(
                        &self.file,
                        &format!("{}pixel_embedder.proj.weight", self.prefix),
                    )[1] as u64,
                ),
                "pixel embed",
            ),
            (
                IrisDiT::output_bytes(
                    CW,
                    b,
                    n,
                    p * p,
                    m.pixel.hidden_size as u64,
                    m.in_channels as u64,
                    false,
                ),
                "output",
            ),
        ];
        quanta.sort_by_key(|q| std::cmp::Reverse(q.0));
        quanta[0]
    }

    fn resident(&self, tier: Option<Quant>) -> u64 {
        crate::memory_strategy::backbone_bytes(
            &self.file,
            Precision::Bf16,
            crate::nn::BackboneQuant::for_load(tier).unwrap(),
        )
        .unwrap()
    }
}

fn row(route: &str, tier: &str, edge: u64, phase: &str, resident: u64, scratch: u64, what: &str) {
    let need = resident + scratch + mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES;
    let fits = need <= ENTITLED_MIB * MIB;
    println!(
        "| {route} | {tier} | {edge} | {phase} | {:.0} | {:.0} ({what}) | {:.0} | {} |",
        resident as f64 / MIB as f64,
        scratch as f64 / MIB as f64,
        need as f64 / MIB as f64,
        if fits { "fits" } else { "NO" }
    );
}

#[test]
#[ignore = "prints the iOS reachable-sizes prediction from the real snapshot's headers"]
fn print_the_reachable_sizes_prediction() {
    let (Some(root), Some(te_dir)) = (
        snapshot("IRIS_WEIGHTS_DIR", "models--speridlabs--iris-3b"),
        snapshot(
            "IRIS_TEXT_ENCODER_DIR",
            "models--Qwen--Qwen3-VL-4B-Instruct",
        ),
    ) else {
        println!("no Iris snapshot on disk: nothing to predict");
        return;
    };
    let generation = Backbone {
        cfg: IrisConfig::from_dir(&root).unwrap(),
        file: root.join("model.safetensors"),
        prefix: "",
    };
    let restore = Backbone {
        cfg: IrisConfig::from_dir(&root.join("upscaler")).unwrap(),
        file: root.join("upscaler/model.safetensors"),
        prefix: "",
    };
    let depth = Backbone {
        cfg: IrisConfig::from_dir(&root.join("depth")).unwrap(),
        file: root.join("depth/model.safetensors"),
        prefix: "pixel.",
    };
    let te_cfg = mlx_llm::ModelConfig::from_dir(&te_dir).unwrap();
    let tokenizer = TextTokenizer::from_file(
        te_dir.join("tokenizer.json"),
        TokenizerConfig {
            max_length: usize::MAX,
            pad_token_id: 0,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .unwrap();
    let prefix = tokenizer.encode_ids(PROMPT_PREFIX, false).unwrap().len() as i32;
    let max_length = generation.cfg.text_encoder.max_length as i32;
    // The full window: the prefix, a caption at the budget, the suffix.
    let window = prefix + max_length;
    let deepest = *generation.cfg.text_encoder.hidden_layers.last().unwrap();
    let layers = generation.cfg.text_encoder.hidden_layers.len() as u64;
    let dim = generation.cfg.text_encoder.dim as u64;
    let conditioning = mlx_gen::ios_quantum::tensor_bytes(4, &[max_length as u64, layers, dim]);
    let _ = PROMPT_SUFFIX;
    println!(
        "| route | tier | edge | phase | resident MiB | largest quantum MiB | + {} MiB reserve = MiB | ≤ {ENTITLED_MIB} MiB |",
        mlx_gen::ios_quantum::PROCESS_RESERVE_BYTES / MIB
    );
    println!("|---|---|---|---|---:|---|---:|---|");
    for (tier, label) in [
        (Some(Quant::Q4), "q4"),
        (Some(Quant::Q8), "q8"),
        (None, "bf16"),
    ] {
        let te = crate::memory_strategy::text_encoder_bytes(&te_dir, tier).unwrap();
        let bb = generation.resident(tier);
        for (mode, prompts) in [("text", 1u64), ("prompt_batch 2", 2)] {
            let route = format!("iris_3b {mode}");
            let b = 2 * prompts; // CFG doubles every solve
            for edge in [512u64, 768, 1024] {
                // Text phase: the tower alone (staged), each prompt and the CFG null encoded in
                // turn; the conditionings already returned stay resident.
                let (layer, held) =
                    crate::text_encoder::decoder_layer_parts(&te_cfg, window, deepest);
                let cond = crate::text_encoder::conditioning_bytes(
                    layers,
                    max_length as u64,
                    dim,
                    max_length,
                );
                row(
                    &route,
                    label,
                    edge,
                    "text",
                    te + prompts * conditioning,
                    (layer + held).max(cond),
                    "deepest decoder layer + held states/KV",
                );
                // Denoise: the backbone, every conditioning and their CFG batch.
                let resident = bb + (prompts + 1) * conditioning + b * conditioning;
                let (scratch, what) = generation.forward_peak(b, edge, edge, tier);
                row(&route, label, edge, "denoise", resident, scratch, what);
                let image = crate::model::image_scratch(prompts * 3 * edge * edge);
                row(&route, label, edge, "decode", bb, image, "RGB8 conversion");
            }
        }
        // Restoration: the upscaler export; edge = the OUTPUT edge at the default 4x.
        let rb = restore.resident(tier) + conditioning;
        for edge in [512u64, 768, 1024] {
            let source = Dims::new((edge / 4) as u32, (edge / 4) as u32);
            let p = plan(
                source,
                &RestorationOptions {
                    scale: 4.0,
                    ..Default::default()
                },
                TileGeometry::RELEASE,
                crate::restoration::MODEL_ID,
            )
            .unwrap();
            let (th, tw) = if p.single_pass() {
                (p.padded.height as u64, p.padded.width as u64)
            } else {
                (p.tile as u64, p.tile as u64)
            };
            let (forward, what) = restore.forward_peak(1, th, tw, tier);
            let host = crate::restoration::host_plane_bytes(&p);
            row(
                "iris_3b_restore",
                label,
                edge,
                "tile",
                rb + host.tile_phase,
                forward + crate::restoration::tile_scratch(3 * th * tw),
                what,
            );
            row(
                "iris_3b_restore",
                label,
                edge,
                "post-tile",
                rb + host.live_at_last_tile,
                host.post_tile_new(),
                "host planes",
            );
        }
        // Depth: the depth export at the request's model resolution.
        let db = depth.resident(tier) + conditioning;
        for edge in [512u64, 768, 1024] {
            let (forward, what) = depth.forward_peak(1, edge, edge, tier);
            let input = crate::depth::depth_input_scratch(1, edge as i32, edge as i32);
            let reducer = crate::depth::depth_reducer_bytes(3 * edge * edge, 3, CW);
            row(
                "iris_3b_depth",
                label,
                edge,
                "forward",
                db,
                forward.max(input).max(reducer),
                what,
            );
        }
    }
}
