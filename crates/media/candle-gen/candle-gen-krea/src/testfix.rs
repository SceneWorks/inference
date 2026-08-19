//! Shared **test-only** tiny-DiT fixture (extracted from `training.rs` for the sc-8460 control-branch
//! tests): the smallest valid Krea DiT config, a random serialized `.safetensors` of it, and a matching
//! `(x0, cap, noise)` batch.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::seeded_normal_vec;
use rand::rngs::StdRng;

use crate::config::Krea2Config;
use crate::loader::Weights;
use crate::train_dit::KreaTrainDit;
use crate::transformer::Krea2Transformer;

/// The smallest valid Krea DiT config: 1 single-stream block, 1 layerwise + 1 refiner text block,
/// head_dim 16 (= sum [4,6,6]), hidden 32, GQA 2/1.
pub(crate) fn tiny_cfg() -> Krea2Config {
    Krea2Config {
        in_channels: 16,
        patch_size: 2,
        hidden_size: 32,
        num_attention_heads: 2,
        num_kv_heads: 1,
        attention_head_dim: 16,
        num_layers: 1,
        intermediate_size: 16,
        norm_eps: 1e-5,
        axes_dims_rope: [4, 6, 6],
        rope_theta: 1000.0,
        timestep_embed_dim: 8,
        num_text_layers: 2,
        num_layerwise_text_blocks: 1,
        num_refiner_text_blocks: 1,
        text_hidden_dim: 32,
        text_intermediate_size: 16,
        text_num_attention_heads: 2,
        text_num_kv_heads: 2,
    }
}

/// Unseeded `N(0, 0.05)` CPU draw — the fixture default for tests that only assert structural or
/// identity properties, where run-to-run weight variance is harmless.
pub(crate) fn rnd(shape: &[usize]) -> Tensor {
    Tensor::randn(0f32, 0.05f32, shape, &Device::Cpu).unwrap()
}

/// Deterministic `N(mean, std)` CPU draw from a seeded `rng`. candle's CPU backend refuses
/// `Device::set_seed` (its `randn` pulls the process-global `rand::rng()`), so a *reproducible*
/// fixture must draw through [`seeded_normal_vec`] — the crate's launch-portable seeded-noise
/// primitive — and build the tensor from the drawn values. Same seed + same call order ⇒ identical
/// tensors every run and every platform (sc-10794).
pub(crate) fn randn_seeded(rng: &mut StdRng, mean: f32, std: f32, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = seeded_normal_vec(rng, n)
        .into_iter()
        .map(|z| mean + std * z)
        .collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

/// A draw function over a tensor shape — the seam that lets [`build_tiny_map`] serve both the
/// unseeded (`|s| rnd(s)`) and seeded (`|s| randn_seeded(rng, ..)`) fixtures from one builder.
type Draw<'a> = dyn FnMut(&[usize]) -> Tensor + 'a;

pub(crate) fn lin(
    draw: &mut Draw,
    t: &mut HashMap<String, Tensor>,
    name: &str,
    out: usize,
    inn: usize,
    bias: bool,
) {
    t.insert(format!("{name}.weight"), draw(&[out, inn]));
    if bias {
        t.insert(format!("{name}.bias"), draw(&[out]));
    }
}

/// Push one gated-attention + SwiGLU block's tensors under `prefix` (shared shape between the text
/// fusion and single-stream blocks, parameterized by widths).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_ffn(
    draw: &mut Draw,
    t: &mut HashMap<String, Tensor>,
    prefix: &str,
    hidden: usize,
    heads: usize,
    kv: usize,
    hd: usize,
    inter: usize,
) {
    t.insert(format!("{prefix}.norm1.weight"), draw(&[hidden]));
    t.insert(format!("{prefix}.norm2.weight"), draw(&[hidden]));
    lin(
        draw,
        t,
        &format!("{prefix}.attn.to_q"),
        heads * hd,
        hidden,
        false,
    );
    lin(
        draw,
        t,
        &format!("{prefix}.attn.to_k"),
        kv * hd,
        hidden,
        false,
    );
    lin(
        draw,
        t,
        &format!("{prefix}.attn.to_v"),
        kv * hd,
        hidden,
        false,
    );
    lin(
        draw,
        t,
        &format!("{prefix}.attn.to_gate"),
        hidden,
        hidden,
        false,
    );
    lin(
        draw,
        t,
        &format!("{prefix}.attn.to_out.0"),
        hidden,
        hidden,
        false,
    );
    t.insert(format!("{prefix}.attn.norm_q.weight"), draw(&[hd]));
    t.insert(format!("{prefix}.attn.norm_k.weight"), draw(&[hd]));
    lin(draw, t, &format!("{prefix}.ff.gate"), inter, hidden, false);
    lin(draw, t, &format!("{prefix}.ff.up"), inter, hidden, false);
    lin(draw, t, &format!("{prefix}.ff.down"), hidden, inter, false);
}

/// Build the tiny-Krea transformer tensor map for `num_layers` single-stream blocks, drawing every
/// weight from `draw`. Split out so the unseeded (`tiny_dit*`) and seeded (`tiny_dit_seeded`)
/// fixtures share one construction — the only difference is the draw source.
fn build_tiny_map(draw: &mut Draw, num_layers: usize) -> (HashMap<String, Tensor>, Krea2Config) {
    let mut c = tiny_cfg();
    c.num_layers = num_layers;
    let (hidden, heads, kv, hd) = (
        c.hidden_size,
        c.num_attention_heads,
        c.num_kv_heads,
        c.attention_head_dim,
    );
    let (th, theads, tkv) = (
        c.text_hidden_dim,
        c.text_num_attention_heads,
        c.text_num_kv_heads,
    );
    let mut t: HashMap<String, Tensor> = HashMap::new();

    lin(draw, &mut t, "img_in", hidden, c.in_channels, true);
    lin(
        draw,
        &mut t,
        "time_embed.linear_1",
        hidden,
        c.timestep_embed_dim,
        true,
    );
    lin(draw, &mut t, "time_embed.linear_2", hidden, hidden, true);
    lin(draw, &mut t, "time_mod_proj", 6 * hidden, hidden, true);
    t.insert("txt_in.norm.weight".into(), draw(&[th]));
    lin(draw, &mut t, "txt_in.linear_1", hidden, th, true);
    lin(draw, &mut t, "txt_in.linear_2", hidden, hidden, true);
    for i in 0..c.num_layerwise_text_blocks {
        attn_ffn(
            draw,
            &mut t,
            &format!("text_fusion.layerwise_blocks.{i}"),
            th,
            theads,
            tkv,
            hd,
            c.text_intermediate_size,
        );
    }
    for i in 0..c.num_refiner_text_blocks {
        attn_ffn(
            draw,
            &mut t,
            &format!("text_fusion.refiner_blocks.{i}"),
            th,
            theads,
            tkv,
            hd,
            c.text_intermediate_size,
        );
    }
    lin(
        draw,
        &mut t,
        "text_fusion.projector",
        1,
        c.num_text_layers,
        false,
    );
    for i in 0..c.num_layers {
        let p = format!("transformer_blocks.{i}");
        t.insert(format!("{p}.scale_shift_table"), draw(&[6, hidden]));
        attn_ffn(draw, &mut t, &p, hidden, heads, kv, hd, c.intermediate_size);
    }
    t.insert("final_layer.scale_shift_table".into(), draw(&[2, hidden]));
    t.insert("final_layer.norm.weight".into(), draw(&[hidden]));
    lin(
        draw,
        &mut t,
        "final_layer.linear",
        c.in_channels,
        hidden,
        true,
    );

    (t, c)
}

/// Serialize the real tiny DiT through the production native-mmdit key dialect and write the
/// snapshot config consumed by [`crate::pipeline::load_dit_base`]. The returned root deliberately
/// contains no diffusers `model.safetensors`: a successful load with the returned pin therefore proves
/// the native-file branch was used rather than silently falling back to snapshot weights.
pub(crate) fn tiny_native_transformer_fixture(
    tmp: &tempfile::TempDir,
) -> (PathBuf, PathBuf, Krea2Config) {
    static N: AtomicUsize = AtomicUsize::new(0);
    let fixture = N.fetch_add(1, Ordering::Relaxed);
    let root = tmp.path().join(format!("krea_tiny_native_{fixture}"));
    let transformer = root.join("transformer");
    std::fs::create_dir_all(&transformer).unwrap();

    let (mut diffusers, mut cfg) = build_tiny_map(&mut |shape| rnd(shape), 1);
    // Match the production Krea latent surface (`LATENT_CHANNELS == 16`, patch size 2) so the
    // fixture can traverse the real render driver's `init_noise` path rather than injecting a
    // test-only latent. Other tiny fixtures retain four latent channels for cheaper unit tests.
    cfg.in_channels = 16 * cfg.patch_size * cfg.patch_size;
    diffusers.insert(
        "img_in.weight".into(),
        rnd(&[cfg.hidden_size, cfg.in_channels]),
    );
    diffusers.insert(
        "final_layer.linear.weight".into(),
        rnd(&[cfg.in_channels, cfg.hidden_size]),
    );
    diffusers.insert("final_layer.linear.bias".into(), rnd(&[cfg.in_channels]));
    let native: HashMap<String, Tensor> = diffusers
        .into_iter()
        .map(|(key, tensor)| {
            let native = crate::loader::convrot_diffusers_to_native(&key)
                .unwrap_or_else(|| panic!("tiny fixture key has no native mapping: {key}"));
            (format!("model.diffusion_model.{native}"), tensor)
        })
        .collect();
    let native_path = root.join("tiny-native.safetensors");
    candle_gen::candle_core::safetensors::save(&native, &native_path).unwrap();

    let config = serde_json::json!({
        "in_channels": cfg.in_channels,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "attention_head_dim": cfg.attention_head_dim,
        "num_layers": cfg.num_layers,
        "intermediate_size": cfg.intermediate_size,
        "norm_eps": cfg.norm_eps,
        "axes_dims_rope": cfg.axes_dims_rope,
        "rope_theta": cfg.rope_theta,
        "timestep_embed_dim": cfg.timestep_embed_dim,
        "num_text_layers": cfg.num_text_layers,
        "num_layerwise_text_blocks": cfg.num_layerwise_text_blocks,
        "num_refiner_text_blocks": cfg.num_refiner_text_blocks,
        "text_hidden_dim": cfg.text_hidden_dim,
        "text_intermediate_size": cfg.text_intermediate_size,
        "text_num_attention_heads": cfg.text_num_attention_heads,
        "text_num_key_value_heads": cfg.text_num_kv_heads,
    });
    std::fs::write(
        transformer.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();

    (root, native_path, cfg)
}

/// Serialize the real tiny DiT as an ordinary **diffusers snapshot** (`transformer/model.safetensors`
/// + `config.json`) — the BUILT-IN load route twin of [`tiny_native_transformer_fixture`], for tests
/// that must prove the snapshot branch of a driver (sc-20418). Same production latent surface
/// (`LATENT_CHANNELS == 16`, patch 2) so `init_noise` traverses the real path.
pub(crate) fn tiny_snapshot_transformer_fixture(tmp: &tempfile::TempDir) -> (PathBuf, Krea2Config) {
    static N: AtomicUsize = AtomicUsize::new(0);
    let fixture = N.fetch_add(1, Ordering::Relaxed);
    let root = tmp.path().join(format!("krea_tiny_snapshot_{fixture}"));
    let transformer = root.join("transformer");
    std::fs::create_dir_all(&transformer).unwrap();

    let (mut diffusers, mut cfg) = build_tiny_map(&mut |shape| rnd(shape), 1);
    cfg.in_channels = 16 * cfg.patch_size * cfg.patch_size;
    diffusers.insert(
        "img_in.weight".into(),
        rnd(&[cfg.hidden_size, cfg.in_channels]),
    );
    diffusers.insert(
        "final_layer.linear.weight".into(),
        rnd(&[cfg.in_channels, cfg.hidden_size]),
    );
    diffusers.insert("final_layer.linear.bias".into(), rnd(&[cfg.in_channels]));
    candle_gen::candle_core::safetensors::save(&diffusers, &transformer.join("model.safetensors"))
        .unwrap();

    let config = serde_json::json!({
        "in_channels": cfg.in_channels,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "attention_head_dim": cfg.attention_head_dim,
        "num_layers": cfg.num_layers,
        "intermediate_size": cfg.intermediate_size,
        "norm_eps": cfg.norm_eps,
        "axes_dims_rope": cfg.axes_dims_rope,
        "rope_theta": cfg.rope_theta,
        "timestep_embed_dim": cfg.timestep_embed_dim,
        "num_text_layers": cfg.num_text_layers,
        "num_layerwise_text_blocks": cfg.num_layerwise_text_blocks,
        "num_refiner_text_blocks": cfg.num_refiner_text_blocks,
        "text_hidden_dim": cfg.text_hidden_dim,
        "text_intermediate_size": cfg.text_intermediate_size,
        "text_num_attention_heads": cfg.text_num_attention_heads,
        "text_num_key_value_heads": cfg.text_num_kv_heads,
    });
    std::fs::write(
        transformer.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();

    (root, cfg)
}

/// Serialize a built tensor map to a `.safetensors` inside `tmp` and load it as a [`KreaTrainDit`].
/// Returns `(dit, path)`; `tmp` owns the file and removes it on drop.
fn serialize_and_load(
    tmp: &tempfile::TempDir,
    t: &HashMap<String, Tensor>,
    c: &Krea2Config,
) -> (KreaTrainDit, PathBuf) {
    static N: AtomicUsize = AtomicUsize::new(0);
    let path = tmp.path().join(format!(
        "krea_tiny_{}.safetensors",
        N.fetch_add(1, Ordering::Relaxed)
    ));
    candle_gen::candle_core::safetensors::save(t, &path).unwrap();
    let w = Weights::from_file(&path, &Device::Cpu, DType::F32).unwrap();
    let dit = KreaTrainDit::load(&w, c).unwrap();
    (dit, path)
}

/// Serialize a tiny Krea transformer to a temp `.safetensors` and load it as a [`KreaTrainDit`].
/// Returns `(dit, cfg, path)`; `tmp` owns the file. Unseeded weights.
pub(crate) fn tiny_dit(tmp: &tempfile::TempDir) -> (KreaTrainDit, Krea2Config, PathBuf) {
    tiny_dit_layers(tmp, 1)
}

/// Serialize a tiny Krea transformer to a temp `.safetensors` and load it as the txt2img inference
/// [`Krea2Transformer`] (vs [`tiny_dit`]'s trainable [`KreaTrainDit`]). Returns `(dit, cfg)`;
/// `tmp` owns the serialized file. Unseeded weights, **dense F32 tier** so every projection —
/// including the front-end globals like `time_mod_proj` — is an adapter-capable `AdaptLinear` (its
/// `QLinear::as_adapt_mut` returns `Some`), which is what the additive-surface coverage test asserts.
pub(crate) fn tiny_transformer(tmp: &tempfile::TempDir) -> (Krea2Transformer, Krea2Config) {
    static N: AtomicUsize = AtomicUsize::new(0);
    let (t, c) = build_tiny_map(&mut |s| rnd(s), 1);
    let path = tmp.path().join(format!(
        "krea_tiny_xf_{}.safetensors",
        N.fetch_add(1, Ordering::Relaxed)
    ));
    candle_gen::candle_core::safetensors::save(&t, &path).unwrap();
    let w = Weights::from_file(&path, &Device::Cpu, DType::F32).unwrap();
    let dit = Krea2Transformer::load(&w, &c).unwrap();
    (dit, c)
}

/// A resident and a **block-streamed** [`Krea2Transformer`] over the SAME serialized weights, at a
/// configurable trunk depth. Returns `(resident, streamed, cfg)`; `tmp` must outlive them, because
/// `load_block_streamed` reads its blocks from that file lazily.
///
/// One file, two residency policies, is the whole point: it makes "identical to the resident path"
/// (SC-15792's parity criterion) an assertion about the *schedule* rather than about two weight
/// draws that happen to agree. Deep enough by default for a ragged tail — `BlockPlan` at window 4
/// over 6 blocks leaves a 2-block last window, which is where a driver that mis-clamps silently
/// drops layers.
///
/// CPU/F32 on purpose: the rung's schedule, its release ordering and its cancellation contract are
/// backend-neutral, so pinning them here keeps them covered on every CI lane instead of only on the
/// Windows CUDA compile check. The CUDA-specific claims — peak by window, the pool's driver/reserved
/// split — are what `rung4_block_window_real_weights.rs` measures, and they cannot be faked here.
pub(crate) fn tiny_transformer_streamed_pair(
    tmp: &tempfile::TempDir,
    num_layers: usize,
) -> (Krea2Transformer, Krea2Transformer, Krea2Config) {
    use std::sync::Arc;

    static N: AtomicUsize = AtomicUsize::new(0);
    let (t, c) = build_tiny_map(&mut |s| rnd(s), num_layers);
    let path = tmp.path().join(format!(
        "krea_tiny_stream_{}.safetensors",
        N.fetch_add(1, Ordering::Relaxed)
    ));
    candle_gen::candle_core::safetensors::save(&t, &path).unwrap();

    let resident = {
        let w = Weights::from_file(&path, &Device::Cpu, DType::F32).unwrap();
        Krea2Transformer::load(&w, &c).unwrap()
    };
    let streamed = {
        let w = Weights::from_file(&path, &Device::Cpu, DType::F32).unwrap();
        Krea2Transformer::load_block_streamed(Arc::new(w), &c).unwrap()
    };
    (resident, streamed, c)
}

/// [`tiny_dit`] with a configurable single-stream depth (the control-branch inject-offset tests
/// need ≥ 2 main blocks). Unseeded weights.
pub(crate) fn tiny_dit_layers(
    tmp: &tempfile::TempDir,
    num_layers: usize,
) -> (KreaTrainDit, Krea2Config, PathBuf) {
    let (t, c) = build_tiny_map(&mut |s| rnd(s), num_layers);
    let (dit, path) = serialize_and_load(tmp, &t, &c);
    (dit, c, path)
}

/// Deterministic single-layer [`tiny_dit`] whose weights are drawn entirely from `rng` — same seed ⇒
/// identical base weights every run and platform. The descent-margin tests need a reproducible base
/// so a marginal loss trajectory can't flip its sign on an unlucky draw (sc-10794).
pub(crate) fn tiny_dit_seeded(
    tmp: &tempfile::TempDir,
    rng: &mut StdRng,
) -> (KreaTrainDit, Krea2Config, PathBuf) {
    let (t, c) = build_tiny_map(&mut |s| randn_seeded(rng, 0.0, 0.05, s), 1);
    let (dit, path) = serialize_and_load(tmp, &t, &c);
    (dit, c, path)
}

/// `(x0, cap, noise)` for the tiny DiT: a `[1, latent_ch, 4, 4]` latent + matching noise, and a
/// `[3, num_text_layers, text_hidden]` caption stack. Unseeded.
pub(crate) fn tiny_batch(c: &Krea2Config) -> (Tensor, Tensor, Tensor) {
    let latent_ch = c.in_channels / (c.patch_size * c.patch_size);
    let x0 = rnd(&[1, latent_ch, 4, 4]);
    let cap = rnd(&[3, c.num_text_layers, c.text_hidden_dim]);
    let noise = rnd(&[1, latent_ch, 4, 4]);
    (x0, cap, noise)
}

/// Deterministic [`tiny_batch`] drawn from `rng` (see [`tiny_dit_seeded`]).
pub(crate) fn tiny_batch_seeded(c: &Krea2Config, rng: &mut StdRng) -> (Tensor, Tensor, Tensor) {
    let latent_ch = c.in_channels / (c.patch_size * c.patch_size);
    let x0 = randn_seeded(rng, 0.0, 0.05, &[1, latent_ch, 4, 4]);
    let cap = randn_seeded(rng, 0.0, 0.05, &[3, c.num_text_layers, c.text_hidden_dim]);
    let noise = randn_seeded(rng, 0.0, 0.05, &[1, latent_ch, 4, 4]);
    (x0, cap, noise)
}
