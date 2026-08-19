//! sc-18775 — **the LTX-2.5 `q4`/`q8`/`bf16` tiers built from the real 73 GB component set**, then
//! walked and driven through the shipped MLX ports.
//!
//! `src/tiers/tests.rs` proves the converter's rules on a miniature bundle. That is a claim about
//! the code. This is the claim about the **weights**: the tiers build from `Lightricks/LTX-2.5`,
//! every component loads through the sc-18757 split resolver, and each one that has an MLX port runs
//! a forward at every tier.
//!
//! # What is and is not covered here
//!
//! A **full render comparison per tier is not yet possible** and is not skipped scope: no pipeline
//! selects LTX-2.5 until the engine descriptors land (sc-18778) and the end-to-end render gate
//! (sc-18791). That is the epic's own sequencing. What is achievable now — and is required before a
//! tier may be rehosted — is per-tier load verification of every component plus component-level
//! forward spot checks, which is exactly what these tests do.
//!
//! # Running
//!
//! ```text
//! LTX25_BUNDLE_DIR=/path/to/Lightricks--LTX-2.5/snapshots/<rev> \
//! LTX25_TIER_DIR=/path/to/scratch/ltx25-tiers \
//!   cargo test -p mlx-gen-ltx --release --test ltx_2_5_tiers_real_weights -- --ignored --nocapture
//! ```
//!
//! `LTX25_BUNDLE_DIR` is the upstream snapshot root (scanned recursively and classified by each
//! file's own metadata — never by name). `LTX25_TIER_DIR` is where the tiers are written and is
//! **required**: three tiers are ~135 GB and must not land in a temp dir by accident. Paths are
//! supplied by the caller; this crate never names a model cache or derives one.
//!
//! # Cost
//!
//! The `bf16` tier alone reads and rewrites ~73 GB. Run it alone, and expect tens of minutes.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use mlx_rs::ops::{abs, max as max_op, mean, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::ltx_checkpoint::{
    discover_split_bundle, LtxBundle, LtxCheckpointLayout, LtxComponent,
};
use mlx_gen::weights::Weights;
use mlx_gen_ltx::audio_vae::AudioDecoder;
use mlx_gen_ltx::config::{AudioVaeConfig, LtxConfig, LtxVaeConfig, SplitModel, VocoderConfig};
use mlx_gen_ltx::connector::Connector;
use mlx_gen_ltx::tiers::{convert_2_5_tiers, DenseReason, LtxTier, DEFAULT_GROUP_SIZE};
use mlx_gen_ltx::transformer::{Precision, VideoBlock};
use mlx_gen_ltx::upsampler::LatentUpsampler;
use mlx_gen_ltx::vae::LtxVideoVae;
use mlx_gen_ltx::vocoder::LtxVocoder;

// =================================================================================================
// Environment + one-shot tier build
// =================================================================================================

fn bundle_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("LTX25_BUNDLE_DIR")?);
    dir.is_dir().then_some(dir)
}

fn tier_root() -> Option<PathBuf> {
    Some(PathBuf::from(std::env::var_os("LTX25_TIER_DIR")?))
}

/// Resolve the upstream bundle and build (or reuse) all three tiers under `LTX25_TIER_DIR`.
///
/// Reuse is keyed on every tier's manifest existing: rebuilding 135 GB to run a second assertion
/// would make this suite unusable. The build itself is the thing under test the first time.
fn tiers() -> Option<PathBuf> {
    let Some(src) = bundle_dir() else {
        eprintln!("skip: set LTX25_BUNDLE_DIR to the Lightricks/LTX-2.5 snapshot root");
        return None;
    };
    let Some(out) = tier_root() else {
        eprintln!("skip: set LTX25_TIER_DIR to a writable directory with ~135 GB free");
        return None;
    };
    let built = LtxTier::ALL
        .iter()
        .all(|t| out.join(t.id()).join("split_model.json").is_file());
    if built {
        eprintln!("[tiers] reusing {}", out.display());
        return Some(out);
    }

    let bundle = discover_split_bundle(&src).expect("resolve the LTX-2.5 bundle");
    assert_eq!(bundle.layout(), LtxCheckpointLayout::Split);
    assert_eq!(bundle.model_version(), Some("2.5.0"));

    let t = Instant::now();
    let reports = convert_2_5_tiers(&bundle, &out, LtxTier::ALL, DEFAULT_GROUP_SIZE)
        .expect("build the LTX-2.5 tiers");
    eprintln!(
        "[tiers] built {} tiers in {:.1} min",
        reports.len(),
        t.elapsed().as_secs_f64() / 60.0
    );
    for report in &reports {
        eprintln!(
            "[tier {}] {} bytes total ({:.2} GiB), {} quantized Linears",
            report.tier,
            report.bytes,
            gib(report.bytes as usize),
            report.quantized_linears()
        );
        for c in &report.components {
            eprintln!(
                "    {:<24} {:>14} B  tensors {:>5}  quantized {:>5}  dense-float {:>5}{}",
                c.name,
                c.bytes,
                c.tensors,
                c.quantized_linears,
                c.dense_float_tensors,
                match c.dense_reason {
                    Some(r) => format!("  [{}]", r.id()),
                    None => String::new(),
                }
            );
        }
    }
    Some(out)
}

fn gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn max_abs(a: &Array) -> f32 {
    max_op(abs(a).unwrap(), None).unwrap().item::<f32>()
}

fn psnr_db(got: &Array, want: &Array) -> f32 {
    let diff = subtract(got, want).unwrap();
    let mse = mean(multiply(&diff, &diff).unwrap(), None)
        .unwrap()
        .item::<f32>();
    20.0 * (2.0f32).log10() - 10.0 * mse.max(1e-20).log10()
}

/// Relative deviation of `got` from `want`, scaled by `want`'s own magnitude.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = max_abs(&subtract(got, want).unwrap());
    diff / max_abs(want).max(1e-12)
}

/// The same band-limited clip `ltx_2_5_vae_real_weights.rs` uses: white noise has nothing for a
/// 32×-spatial / 8×-temporal autoencoder to reconstruct.
fn synthetic_clip(frames: i32, h: i32, w: i32) -> Array {
    let mut data = Vec::with_capacity((3 * frames * h * w) as usize);
    for c in 0..3 {
        for f in 0..frames {
            let t = f as f32 / frames.max(2) as f32;
            for y in 0..h {
                let v = y as f32 / h as f32;
                for x in 0..w {
                    let u = x as f32 / w as f32;
                    let value =
                        0.55 * ((6.0 * u + 1.7 * t + c as f32).sin()) * ((4.0 * v - 2.3 * t).cos())
                            + 0.25 * (2.0 * v - 1.0)
                            + 0.10 * ((13.0 * u * v + 3.0 * t).sin());
                    data.push(value.clamp(-1.0, 1.0));
                }
            }
        }
    }
    Array::from_slice(&data, &[1, 3, frames, h, w])
}

// =================================================================================================
// Header measurement (never a hash — `save_file` orders `__metadata__` nondeterministically)
// =================================================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
struct TensorHeader {
    dtype: String,
    shape: Vec<i64>,
}

struct FileHeader {
    metadata: BTreeMap<String, String>,
    tensors: BTreeMap<String, TensorHeader>,
    bytes: u64,
}

impl FileHeader {
    fn read(path: &Path) -> FileHeader {
        let mut f =
            std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let bytes = f.metadata().unwrap().len();
        let mut len = [0u8; 8];
        f.read_exact(&mut len).unwrap();
        let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
        f.read_exact(&mut buf).unwrap();
        let json: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&buf).unwrap();
        let mut metadata = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        for (key, value) in json {
            if key == "__metadata__" {
                for (k, v) in value.as_object().unwrap() {
                    metadata.insert(k.clone(), v.as_str().unwrap().to_string());
                }
                continue;
            }
            tensors.insert(
                key,
                TensorHeader {
                    dtype: value["dtype"].as_str().unwrap().to_string(),
                    shape: value["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| d.as_i64().unwrap())
                        .collect(),
                },
            );
        }
        FileHeader {
            metadata,
            tensors,
            bytes,
        }
    }
}

fn component_header(root: &Path, tier: LtxTier, component: &str) -> FileHeader {
    FileHeader::read(
        &root
            .join(tier.id())
            .join(format!("{component}.safetensors")),
    )
}

fn manifest(root: &Path, tier: LtxTier) -> serde_json::Value {
    let path = root.join(tier.id()).join("split_model.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

/// Reload a tier directory as a bundle through the split resolver, with the SceneWorks component
/// names mapped onto their [`LtxComponent`] slots.
///
/// The tier tree splits the upstream conv-VAE file into two halves and lifts the connector out of
/// the transformer, so the slots that carry one file each are provisioned explicitly; the rest of
/// the tree's classification is exercised by [`discover_split_bundle`] in
/// `a_tier_tree_resolves_through_the_split_resolver`.
fn tier_bundle(root: &Path, tier: LtxTier) -> LtxBundle {
    let dir = root.join(tier.id());
    let mut builder = mlx_gen::gen_core::ltx_checkpoint::LtxBundleBuilder::new()
        .with_component(
            LtxComponent::Transformer,
            dir.join("transformer.safetensors"),
        )
        .with_component(
            LtxComponent::TextEncoder,
            dir.join("text_encoder.safetensors"),
        )
        .with_component(
            LtxComponent::ConvVideoVae,
            dir.join("vae_decoder.safetensors"),
        )
        .with_component(LtxComponent::AudioVae, dir.join("audio_vae.safetensors"));
    for (component, name) in [
        (LtxComponent::SpatialUpsampler, "spatial_upsampler"),
        (LtxComponent::TemporalUpsampler, "temporal_upsampler"),
        (LtxComponent::DurationHead, "duration_head"),
    ] {
        let path = dir.join(format!("{name}.safetensors"));
        if path.is_file() {
            builder = builder.with_component(component, path);
        }
    }
    builder.build().expect("the tier tree must resolve")
}

// =================================================================================================
// The tests
// =================================================================================================

/// **The whole-pipeline-contract assertion on real weights.**
///
/// Walks every component of every tier and asserts, from the produced files, that each quantizable
/// segment is packed at the tier's bit-width and that every component that is *not* declares a
/// reason. Also records the exact on-disk bytes per tier — the numbers sc-18781's manifest
/// `estimatedSizeBytes` / `footprint.diskSizeBytes` must carry, which is why they are measured here
/// rather than estimated.
#[test]
#[ignore = "sc-18775: builds ~135 GB of tiers from the gated Lightricks/LTX-2.5 component set"]
fn every_tier_is_quantized_end_to_end_and_its_size_is_measured() {
    let Some(root) = tiers() else {
        return;
    };

    // Counts derived from the checkpoint's own declared geometry, never pinned literals.
    let cfg = LtxConfig::from_model_dir(&root.join(LtxTier::Bf16.id())).expect("tier config");
    let dit_linears = (cfg.num_layers as usize) * (6 * 4 + 4);
    let connector_linears = (cfg.connector_num_layers as usize) * 6 * 2;

    let mut totals: Vec<(LtxTier, u64)> = Vec::new();
    for tier in LtxTier::ALL {
        let m = manifest(&root, *tier);
        assert_eq!(m["model_version"], "2.5.0");
        assert_eq!(m["tier"], tier.id());
        assert_eq!(m["quantized"], tier.bits().is_some());
        assert_eq!(m["quantization_group_size"], DEFAULT_GROUP_SIZE);

        let mut tier_bytes = 0u64;
        for entry in m["component_detail"].as_array().unwrap() {
            let name = entry["name"].as_str().unwrap();
            let header = component_header(&root, *tier, name);
            tier_bytes += header.bytes;
            assert_eq!(
                header.bytes,
                entry["bytes"].as_u64().unwrap(),
                "{tier}/{name}: the manifest's byte count must be the file's actual size"
            );

            let packed: Vec<&String> = header
                .tensors
                .keys()
                .filter(|k| k.ends_with(".scales"))
                .collect();
            assert_eq!(
                packed.len() as u64,
                entry["quantized_linears"].as_u64().unwrap(),
                "{tier}/{name}: the manifest's quantized count must match the file"
            );

            if packed.is_empty() {
                let reason = entry.get("dense_reason").and_then(|v| v.as_str());
                match tier.bits() {
                    None => assert_eq!(reason, Some("dense-tier"), "{tier}/{name}"),
                    Some(_) => assert!(
                        matches!(reason, Some("no-linear-weights") | Some("no-mlx-port")),
                        "{tier}/{name}: a component dense inside a quantized tier must declare \
                         why; got {reason:?}"
                    ),
                }
            }

            // Every packed weight really is packed, and every unpacked float really is bf16 — the
            // "nothing silently stayed at a different precision" claim, measured.
            for key in header.tensors.keys() {
                let h = &header.tensors[key];
                if key.ends_with(".scales") || key.ends_with(".biases") {
                    assert_eq!(h.dtype, "BF16", "{tier}/{name}/{key}");
                    continue;
                }
                let is_packed = key
                    .strip_suffix(".weight")
                    .is_some_and(|b| header.tensors.contains_key(&format!("{b}.scales")));
                if is_packed {
                    assert_eq!(h.dtype, "U32", "{tier}/{name}/{key} must be packed");
                } else {
                    assert!(
                        h.dtype == "BF16" || h.dtype == "U8",
                        "{tier}/{name}/{key} is {} — an unpacked tensor must be bf16 (or a \
                         `U8` packed HF asset)",
                        h.dtype
                    );
                }
            }
        }

        // The three quantizing components, by name, at the geometry the config declares.
        let expected = |n: usize| if tier.bits().is_some() { n } else { 0 };
        for (name, want) in [
            ("transformer", expected(dit_linears)),
            ("connector", expected(connector_linears)),
        ] {
            let header = component_header(&root, *tier, name);
            let packed = header
                .tensors
                .keys()
                .filter(|k| k.ends_with(".scales"))
                .count();
            assert_eq!(packed, want, "{tier}/{name}: packed Linear count");
        }
        // The text encoder's set is per-layer and per-layer-type, so derive it from the file:
        // every `{q,k,v,o}_proj` / `{gate,up,down}_proj` present must be packed, and the embedding
        // table must not be.
        let te = component_header(&root, *tier, "text_encoder");
        let mut projections = 0usize;
        for key in te.tensors.keys() {
            let is_proj = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            .iter()
            .any(|p| key.ends_with(&format!(".{p}.weight")))
                || key.ends_with("_aggregate_embed.weight");
            if !is_proj {
                continue;
            }
            projections += 1;
            let base = key.strip_suffix(".weight").unwrap();
            assert_eq!(
                te.tensors.contains_key(&format!("{base}.scales")),
                tier.bits().is_some(),
                "{tier}/text_encoder/{key}: packed state must follow the tier"
            );
        }
        assert!(
            projections > 300,
            "{tier}: the real Gemma 4 encoder has 328 decoder projections plus the two aggregate \
             embeds; found {projections}"
        );
        assert_eq!(
            te.tensors["model.embed_tokens.weight"].dtype, "BF16",
            "{tier}: the embedding table is an exempt lookup"
        );

        totals.push((*tier, tier_bytes));
        eprintln!(
            "[size] {tier}: {tier_bytes} bytes ({:.2} GiB)",
            gib(tier_bytes as usize)
        );
    }

    let q4 = totals[0].1;
    let q8 = totals[1].1;
    let bf16 = totals[2].1;
    assert!(
        q4 < q8 && q8 < bf16,
        "tier sizes must be strictly ordered: q4 {q4} < q8 {q8} < bf16 {bf16}"
    );
}

/// A tier tree resolves as an LTX-2.5 split bundle, its Gemma version assertion passes, and its
/// sidecars parse through the shipped config readers.
#[test]
#[ignore = "sc-18775: needs the built tiers (LTX25_TIER_DIR)"]
fn a_tier_tree_resolves_through_the_split_resolver() {
    let Some(root) = tiers() else {
        return;
    };
    for tier in LtxTier::ALL {
        let dir = root.join(tier.id());
        assert_eq!(
            mlx_gen_ltx::declared_layout(&dir).unwrap(),
            LtxCheckpointLayout::Split,
            "{tier}: the manifest declares 2.5, so the tree keeps the split layout"
        );

        // Directory-scan discovery must not be ambiguous: the split halves and the connector
        // deliberately carry no config section of their own.
        let scanned = discover_split_bundle(&dir)
            .unwrap_or_else(|e| panic!("{tier}: the tier tree must resolve unambiguously: {e}"));
        for component in [
            LtxComponent::Transformer,
            LtxComponent::TextEncoder,
            LtxComponent::AudioVae,
            LtxComponent::DurationHead,
        ] {
            assert!(
                scanned.require(component).is_ok(),
                "{tier}: {} must resolve from the scan",
                component.id()
            );
        }

        let bundle = tier_bundle(&root, *tier);
        assert!(
            matches!(
                mlx_gen_ltx::assert_gemma_version(&bundle).unwrap(),
                mlx_gen::gen_core::ltx_checkpoint::GemmaVersionCheck::Matched(_)
            ),
            "{tier}: the packed text encoder must satisfy the transformer's gemma_source_checkpoint"
        );

        let split = SplitModel::from_model_dir(&dir).unwrap();
        assert_eq!(split.quantized, tier.bits().is_some());
        assert_eq!(split.group, DEFAULT_GROUP_SIZE);
        if let Some(bits) = tier.bits() {
            assert_eq!(split.bits, bits);
        }
        let cfg = LtxConfig::from_model_dir(&dir).unwrap();
        assert_eq!(cfg.num_layers, 48, "{tier}: the LTX-2.5 DiT has 48 blocks");
        assert_eq!(cfg.connector_num_layers, 8);
        let vae = LtxVaeConfig::from_model_dir(&dir).unwrap();
        assert_eq!(vae.latent_channels, 128);
        assert_eq!(vae.patch_size, 4);
        AudioVaeConfig::from_model_dir(&dir).unwrap();
        VocoderConfig::from_model_dir(&dir).unwrap();
    }
}

/// Every component with an MLX port loads **and runs a forward** at every tier.
///
/// Per tier, one component at a time, dropped before the next: the q4 transformer alone is ~11 GB
/// and the bf16 one ~38 GB. The DiT is exercised as a single block rather than all 48 — the load
/// path, the quantized-Linear binding and the block math are what a tier can break, and holding a
/// whole bf16 DiT resident to prove it would cost 38 GB for no additional signal.
#[test]
#[ignore = "sc-18775: needs the built tiers (LTX25_TIER_DIR) + a GPU"]
fn every_component_loads_and_forwards_at_every_tier() {
    let Some(root) = tiers() else {
        return;
    };

    for tier in LtxTier::ALL {
        let dir = root.join(tier.id());
        let cfg = LtxConfig::from_model_dir(&dir).unwrap();
        let split = SplitModel::from_model_dir(&dir).unwrap();
        // bf16 activations x whatever the checkpoint stores — the production path.
        let prec = Precision::quant_bf16(split.bits, split.group);

        // ---- one DiT block -------------------------------------------------------------------
        {
            let w = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
            reset_peak_memory();
            let t = Instant::now();
            let block = VideoBlock::load(&w, "transformer_blocks.0", &cfg, prec)
                .unwrap_or_else(|e| panic!("{tier}: load DiT block 0: {e}"));
            let load_s = t.elapsed().as_secs_f64();
            drop(w);
            clear_cache();

            let dim = cfg.num_attention_heads * cfg.attention_head_dim;
            let (b, s, ctx) = (1, 64, 32);
            let x = Array::ones::<f32>(&[b, s, dim])
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let context = Array::ones::<f32>(&[b, ctx, cfg.cross_attention_dim])
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let timesteps = Array::ones::<f32>(&[b, 1, dim * 9])
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let head_half = cfg.attention_head_dim / 2;
            let cos = Array::ones::<f32>(&[b, cfg.num_attention_heads, s, head_half]).unwrap();
            let sin = Array::zeros::<f32>(&[b, cfg.num_attention_heads, s, head_half]).unwrap();
            let t = Instant::now();
            let out = block
                .forward(&x, &timesteps, None, &context, None, &cos, &sin)
                .unwrap_or_else(|e| panic!("{tier}: DiT block forward: {e}"));
            out.eval().unwrap();
            eprintln!(
                "[{tier}] DiT block 0: load {load_s:.1}s forward {:.2}s peak {:.2} GiB out {:?}",
                t.elapsed().as_secs_f64(),
                gib(get_peak_memory()),
                out.shape()
            );
            assert_eq!(out.shape(), x.shape());
            let m = max_abs(&out);
            assert!(
                m.is_finite() && m > 1e-4,
                "{tier}: the DiT block produced a degenerate output (max|v| = {m})"
            );
            drop(block);
            clear_cache();
        }

        // ---- both embeddings connectors ---------------------------------------------------------
        {
            let w = Weights::from_file(dir.join("connector.safetensors")).expect("connector");
            reset_peak_memory();
            let video = Connector::from_weights(&w, "video_embeddings_connector.", &cfg, prec)
                .unwrap_or_else(|e| panic!("{tier}: load the video connector: {e}"));
            let audio = Connector::from_weights_dims(
                &w,
                "audio_embeddings_connector.",
                cfg.connector_num_layers,
                cfg.audio_connector_num_attention_heads,
                cfg.audio_connector_attention_head_dim,
                cfg.positional_embedding_theta,
                cfg.connector_positional_embedding_max_pos,
                cfg.connector_ff_bias,
                prec,
            )
            .unwrap_or_else(|e| panic!("{tier}: load the audio connector: {e}"));
            drop(w);
            clear_cache();

            // The connector replaces left-padding with `connector_num_learnable_registers`
            // registers, so the sequence must be a multiple of that count and at least as long.
            let seq = cfg.connector_num_learnable_registers.max(128);
            let mask01 = Array::ones::<f32>(&[1, seq]).unwrap();
            for (label, conn, dim) in [
                (
                    "video",
                    &video,
                    cfg.connector_num_attention_heads * cfg.connector_attention_head_dim,
                ),
                (
                    "audio",
                    &audio,
                    cfg.audio_connector_num_attention_heads
                        * cfg.audio_connector_attention_head_dim,
                ),
            ] {
                let features = Array::ones::<f32>(&[1, seq, dim]).unwrap();
                let t = Instant::now();
                let out = conn
                    .forward(&features, &mask01)
                    .unwrap_or_else(|e| panic!("{tier}: {label} connector forward: {e}"));
                out.eval().unwrap();
                assert_eq!(out.shape(), &[1, seq, dim]);
                let m = max_abs(&out);
                eprintln!(
                    "[{tier}] {label} connector: forward {:.2}s max|v| {m:.4e} peak {:.2} GiB",
                    t.elapsed().as_secs_f64(),
                    gib(get_peak_memory())
                );
                assert!(
                    m.is_finite() && m > 1e-4,
                    "{tier}: the {label} connector produced a degenerate output"
                );
            }
            drop(video);
            drop(audio);
            clear_cache();
        }

        // ---- conv video VAE round trip ----------------------------------------------------------
        {
            let vae_cfg = LtxVaeConfig::from_model_dir(&dir).unwrap();
            let dec = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae_decoder");
            let enc = Weights::from_file(dir.join("vae_encoder.safetensors")).expect("vae_encoder");
            let vae = LtxVideoVae::from_weights(&dec, Some(&enc), &vae_cfg)
                .unwrap_or_else(|e| panic!("{tier}: build LtxVideoVae: {e}"));
            drop(dec);
            drop(enc);
            clear_cache();

            let clip = synthetic_clip(25, 512, 768);
            clip.eval().unwrap();
            let t = Instant::now();
            let latent = vae.encode(&clip).expect("encode");
            latent.eval().unwrap();
            let round = vae.decode(&latent).expect("decode");
            round.eval().unwrap();
            let psnr = psnr_db(&round, &clip);
            eprintln!(
                "[{tier}] conv VAE 768x512x25: round trip {:.1}s PSNR {psnr:.2} dB",
                t.elapsed().as_secs_f64()
            );
            // The VAE is dense in every tier (no quantized convolution exists), so this number must
            // NOT move with the tier — it is the control proving the exemption is real.
            assert!(
                psnr > 45.0,
                "{tier}: conv VAE round-trip PSNR {psnr:.2} dB is too low"
            );
            drop(vae);
            clear_cache();
        }

        // ---- audio VAE + vocoder -----------------------------------------------------------------
        {
            let audio_cfg = AudioVaeConfig::from_model_dir(&dir).unwrap();
            let voc_cfg = VocoderConfig::from_model_dir(&dir).unwrap();
            let audio_w = Weights::from_file(dir.join("audio_vae.safetensors")).unwrap();
            let voc_w = Weights::from_file(dir.join("vocoder.safetensors")).unwrap();
            let decoder = AudioDecoder::from_weights(&audio_w, &audio_cfg)
                .unwrap_or_else(|e| panic!("{tier}: build AudioDecoder: {e}"));
            let vocoder = LtxVocoder::from_weights(&voc_w, &voc_cfg)
                .unwrap_or_else(|e| panic!("{tier}: build LtxVocoder: {e}"));
            drop(audio_w);
            drop(voc_w);
            clear_cache();

            let key = mlx_rs::random::key(18775).unwrap();
            let latent = mlx_rs::random::normal::<f32>(
                &[1, audio_cfg.z_channels, 93, 16],
                None,
                None,
                Some(&key),
            )
            .unwrap();
            let mel = decoder.decode(&latent).expect("audio decode");
            mel.eval().unwrap();
            let wav = vocoder.forward(&mel).expect("vocoder");
            wav.eval().unwrap();
            let amp = max_abs(&wav);
            eprintln!(
                "[{tier}] audio: mel {:?} -> wav {:?} max|v| {amp:.4}",
                mel.shape(),
                wav.shape()
            );
            assert!(
                amp.is_finite() && amp > 1e-4 && amp <= 1.5,
                "{tier}: the vocoder waveform is not a sane audio signal (max|v| = {amp})"
            );
            drop(decoder);
            drop(vocoder);
            clear_cache();
        }

        // ---- latent upsamplers --------------------------------------------------------------------
        for name in ["spatial_upsampler", "temporal_upsampler"] {
            let path = dir.join(format!("{name}.safetensors"));
            if !path.is_file() {
                continue;
            }
            let w = Weights::from_file(&path).unwrap();
            let up = LatentUpsampler::from_weights(&w)
                .unwrap_or_else(|e| panic!("{tier}: build {name}: {e}"));
            drop(w);
            clear_cache();
            let latent = Array::ones::<f32>(&[1, 128, 2, 8, 8]).unwrap();
            let out = up
                .forward(&latent)
                .unwrap_or_else(|e| panic!("{tier}: {name} forward: {e}"));
            out.eval().unwrap();
            eprintln!("[{tier}] {name}: {:?} -> {:?}", latent.shape(), out.shape());
            assert!(
                max_abs(&out).is_finite(),
                "{tier}: {name} produced non-finite latents"
            );
            drop(up);
            clear_cache();
        }
    }
}

/// The quantized tiers must actually *change* the DiT's numbers relative to bf16 — and by an amount
/// that orders q8 tighter than q4.
///
/// A tier that loaded its packed weights but silently dequantized to the same values (or a bf16 tier
/// that was quietly quantized) would pass every count-based assertion above. This is the executed
/// control: the same block, the same input, three precisions, one measurement.
#[test]
#[ignore = "sc-18775: needs the built tiers (LTX25_TIER_DIR) + a GPU"]
fn quantization_moves_the_dit_block_and_q8_is_tighter_than_q4() {
    let Some(root) = tiers() else {
        return;
    };
    let bf16_dir = root.join(LtxTier::Bf16.id());
    let cfg = LtxConfig::from_model_dir(&bf16_dir).unwrap();
    let dim = cfg.num_attention_heads * cfg.attention_head_dim;
    let (b, s, ctx) = (1, 64, 32);

    let key = mlx_rs::random::key(18775).unwrap();
    let x = mlx_rs::random::normal::<f32>(&[b, s, dim], None, None, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let context =
        mlx_rs::random::normal::<f32>(&[b, ctx, cfg.cross_attention_dim], None, None, Some(&key))
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
    let timesteps = Array::ones::<f32>(&[b, 1, dim * 9])
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let head_half = cfg.attention_head_dim / 2;
    let cos = Array::ones::<f32>(&[b, cfg.num_attention_heads, s, head_half]).unwrap();
    let sin = Array::zeros::<f32>(&[b, cfg.num_attention_heads, s, head_half]).unwrap();

    let mut outputs: Vec<(LtxTier, Array)> = Vec::new();
    for tier in LtxTier::ALL {
        let dir = root.join(tier.id());
        let split = SplitModel::from_model_dir(&dir).unwrap();
        let w = Weights::from_file(dir.join("transformer.safetensors")).unwrap();
        let block = VideoBlock::load(
            &w,
            "transformer_blocks.0",
            &cfg,
            Precision::quant_bf16(split.bits, split.group),
        )
        .unwrap();
        drop(w);
        clear_cache();
        let out = block
            .forward(&x, &timesteps, None, &context, None, &cos, &sin)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        out.eval().unwrap();
        outputs.push((*tier, out));
        drop(block);
        clear_cache();
    }

    let reference = &outputs
        .iter()
        .find(|(t, _)| *t == LtxTier::Bf16)
        .expect("a bf16 tier")
        .1;
    let mut deltas: BTreeMap<&'static str, f32> = BTreeMap::new();
    for (tier, out) in &outputs {
        let rel = peak_rel(out, reference);
        eprintln!("[dit block] {tier} vs bf16: peak_rel {rel:.4e}");
        deltas.insert(tier.id(), rel);
    }
    assert_eq!(
        deltas["bf16"], 0.0,
        "bf16 is the reference and must match itself"
    );
    assert!(
        deltas["q8"] > 0.0 && deltas["q4"] > 0.0,
        "a quantized tier must actually differ from bf16 — equality here means the packed weights \
         were never bound: q8 {:.3e}, q4 {:.3e}",
        deltas["q8"],
        deltas["q4"]
    );
    assert!(
        deltas["q8"] < deltas["q4"],
        "8-bit must be closer to bf16 than 4-bit: q8 {:.3e} vs q4 {:.3e}",
        deltas["q8"],
        deltas["q4"]
    );
}

/// The 48 per-layer `layer_scalar` buffers and the missing `v_proj` on the eight `full_attention`
/// layers survive the repack at every tier — the two shapes the Gemma 4 loader would silently
/// mis-bind (upstream measured 11 tower tensors left at random init from exactly this class of
/// mistake).
#[test]
#[ignore = "sc-18775: needs the built tiers (LTX25_TIER_DIR)"]
fn the_gemma_4_checkpoint_shapes_survive_at_every_tier() {
    let Some(root) = tiers() else {
        return;
    };
    let source = component_header(&root, LtxTier::Bf16, "text_encoder");
    let scalars = source
        .tensors
        .keys()
        .filter(|k| k.ends_with(".layer_scalar"))
        .count();
    assert_eq!(
        scalars, 48,
        "the shipped Gemma 4 encoder carries one trained `layer_scalar` per layer"
    );
    let v_projs = source
        .tensors
        .keys()
        .filter(|k| k.ends_with(".self_attn.v_proj.weight"))
        .count();
    assert_eq!(
        v_projs, 40,
        "eight of the 48 layers are `full_attention` with `attention_k_eq_v` and ship no v_proj"
    );

    for tier in LtxTier::ALL {
        let te = component_header(&root, *tier, "text_encoder");
        for key in source
            .tensors
            .keys()
            .filter(|k| k.ends_with(".layer_scalar"))
        {
            assert_eq!(
                te.tensors.get(key),
                source.tensors.get(key),
                "{tier}: {key} must survive verbatim"
            );
        }
        assert_eq!(
            te.tensors
                .keys()
                .filter(|k| k.ends_with(".self_attn.v_proj.weight"))
                .count(),
            v_projs,
            "{tier}: the v_proj population must not change"
        );
        // The packed HF assets pass through untouched — a cast here corrupts the tokenizer.
        let assets = mlx_gen::gen_core::gemma_assets::GemmaAssets::from_single_file(
            root.join(tier.id()).join("text_encoder.safetensors"),
        )
        .unwrap_or_else(|e| panic!("{tier}: unpack the tier's Gemma assets: {e}"));
        assert_eq!(
            assets.tokenizer_json().len(),
            32_169_626,
            "{tier}: the packed tokenizer.json must survive byte-for-byte"
        );
        // ...and it still builds a working tokenizer.
        let tok = mlx_gen::gen_core::gemma_assets::LtxGemmaTokenizer::from_assets(&assets)
            .unwrap_or_else(|e| panic!("{tier}: build the tokenizer from the tier: {e}"));
        let out = tok
            .encode("a cinematic shot of a harbour at dusk", 128)
            .unwrap();
        assert_eq!(out.ids.len(), 128);
        assert!(out.mask.iter().any(|m| *m == 1));

        // The `quantization` block that binds the packed projections is present iff the tier packs.
        let gemma: serde_json::Value = serde_json::from_str(&te.metadata["gemma_config"]).unwrap();
        match tier.bits() {
            Some(bits) => assert_eq!(gemma["quantization"]["bits"], bits, "{tier}"),
            None => assert!(gemma.get("quantization").is_none(), "{tier}"),
        }
    }
    let _ = DenseReason::NoLinearWeights;
}
