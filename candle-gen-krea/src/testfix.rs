//! Shared **test-only** tiny-DiT fixture (extracted from `training.rs` for the sc-8460 control-branch
//! tests): the smallest valid Krea DiT config, a random serialized `.safetensors` of it, and a matching
//! `(x0, cap, noise)` batch.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use candle_gen::candle_core::{DType, Device, Tensor};

use crate::config::Krea2Config;
use crate::loader::Weights;
use crate::train_dit::KreaTrainDit;

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

pub(crate) fn rnd(shape: &[usize]) -> Tensor {
    Tensor::randn(0f32, 0.05f32, shape, &Device::Cpu).unwrap()
}

pub(crate) fn lin(t: &mut HashMap<String, Tensor>, name: &str, out: usize, inn: usize, bias: bool) {
    t.insert(format!("{name}.weight"), rnd(&[out, inn]));
    if bias {
        t.insert(format!("{name}.bias"), rnd(&[out]));
    }
}

/// Push one gated-attention + SwiGLU block's tensors under `prefix` (shared shape between the text
/// fusion and single-stream blocks, parameterized by widths).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_ffn(
    t: &mut HashMap<String, Tensor>,
    prefix: &str,
    hidden: usize,
    heads: usize,
    kv: usize,
    hd: usize,
    inter: usize,
) {
    t.insert(format!("{prefix}.norm1.weight"), rnd(&[hidden]));
    t.insert(format!("{prefix}.norm2.weight"), rnd(&[hidden]));
    lin(t, &format!("{prefix}.attn.to_q"), heads * hd, hidden, false);
    lin(t, &format!("{prefix}.attn.to_k"), kv * hd, hidden, false);
    lin(t, &format!("{prefix}.attn.to_v"), kv * hd, hidden, false);
    lin(t, &format!("{prefix}.attn.to_gate"), hidden, hidden, false);
    lin(t, &format!("{prefix}.attn.to_out.0"), hidden, hidden, false);
    t.insert(format!("{prefix}.attn.norm_q.weight"), rnd(&[hd]));
    t.insert(format!("{prefix}.attn.norm_k.weight"), rnd(&[hd]));
    lin(t, &format!("{prefix}.ff.gate"), inter, hidden, false);
    lin(t, &format!("{prefix}.ff.up"), inter, hidden, false);
    lin(t, &format!("{prefix}.ff.down"), hidden, inter, false);
}

/// Serialize a tiny Krea transformer to a temp `.safetensors` and load it as a [`KreaTrainDit`].
/// Returns `(dit, cfg, temp_path)` — the caller drops the file when done.
pub(crate) fn tiny_dit() -> (KreaTrainDit, Krea2Config, PathBuf) {
    let c = tiny_cfg();
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

    lin(&mut t, "img_in", hidden, c.in_channels, true);
    lin(
        &mut t,
        "time_embed.linear_1",
        hidden,
        c.timestep_embed_dim,
        true,
    );
    lin(&mut t, "time_embed.linear_2", hidden, hidden, true);
    lin(&mut t, "time_mod_proj", 6 * hidden, hidden, true);
    t.insert("txt_in.norm.weight".into(), rnd(&[th]));
    lin(&mut t, "txt_in.linear_1", hidden, th, true);
    lin(&mut t, "txt_in.linear_2", hidden, hidden, true);
    for i in 0..c.num_layerwise_text_blocks {
        attn_ffn(
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
            &mut t,
            &format!("text_fusion.refiner_blocks.{i}"),
            th,
            theads,
            tkv,
            hd,
            c.text_intermediate_size,
        );
    }
    lin(&mut t, "text_fusion.projector", 1, c.num_text_layers, false);
    for i in 0..c.num_layers {
        let p = format!("transformer_blocks.{i}");
        t.insert(format!("{p}.scale_shift_table"), rnd(&[6, hidden]));
        attn_ffn(&mut t, &p, hidden, heads, kv, hd, c.intermediate_size);
    }
    t.insert("final_layer.scale_shift_table".into(), rnd(&[2, hidden]));
    t.insert("final_layer.norm.weight".into(), rnd(&[hidden]));
    lin(&mut t, "final_layer.linear", c.in_channels, hidden, true);

    static N: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "krea_tiny_{}_{}.safetensors",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    candle_gen::candle_core::safetensors::save(&t, &path).unwrap();
    let w = Weights::from_file(&path, &Device::Cpu, DType::F32).unwrap();
    let dit = KreaTrainDit::load(&w, &c).unwrap();
    (dit, c, path)
}

/// `(x0, cap, noise)` for the tiny DiT: a `[1, latent_ch, 4, 4]` latent + matching noise, and a
/// `[3, num_text_layers, text_hidden]` caption stack.
pub(crate) fn tiny_batch(c: &Krea2Config) -> (Tensor, Tensor, Tensor) {
    let latent_ch = c.in_channels / (c.patch_size * c.patch_size);
    let x0 = rnd(&[1, latent_ch, 4, 4]);
    let cap = rnd(&[3, c.num_text_layers, c.text_hidden_dim]);
    let noise = rnd(&[1, latent_ch, 4, 4]);
    (x0, cap, noise)
}
