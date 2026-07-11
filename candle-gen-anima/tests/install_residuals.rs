//! `install_anima_residuals` at the model seam (sc-10640, sc-10713) — builds a **real** (tiny)
//! `CosmosDiT` + `AnimaTextConditioner` from synthetic dense weights and drives the forward-time
//! additive-residual installer end to end: the visit routes both a DiT target and a conditioner target,
//! the residual actually shifts the built model's forward, a scale-0 residual is a mutation no-op, a
//! **LoKr installs as a structured Kronecker residual** and shifts the forward, a **LoHa is rejected**
//! with the sc-10713 message, and an off-surface target hard-errors (the sc-10274 strict guard).
//!
//! The install path is base-agnostic — it pushes residuals via `AdaptLinear::push_lora` /
//! `push_lokr_structured`, which never read the base weight — so a **dense** synthetic host exercises the
//! routing/strict/reject logic faithfully (no group-64 packed fixture needed). The packed-survives-load
//! property (`.scales` intact, no dense weight materialized) and the LoKr↔folded-delta parity are proven
//! at the linear granularity in the crate's `adapt` unit tests against a real MLX-packed `QLinear`.

use std::collections::{HashMap, HashSet};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen_anima::adapt::AdaptLinear;
use candle_gen_anima::adapters::install_anima_residuals;
use candle_gen_anima::conditioner::AnimaTextConditioner;
use candle_gen_anima::config::{ConditionerConfig, DitConfig};
use candle_gen_anima::transformer::CosmosDiT;

/// A tiny Cosmos DiT config — same structure as `DitConfig::anima`, small dims (hidden = 12).
fn dit_cfg() -> DitConfig {
    DitConfig {
        in_channels: 16,
        out_channels: 16,
        num_attention_heads: 2,
        attention_head_dim: 6, // hidden 12
        num_layers: 2,
        mlp_ratio: 4.0,
        text_embed_dim: 8,
        adaln_lora_dim: 8,
        max_size: (128, 120, 120),
        patch_size: (1, 2, 2),
        rope_scale: (1.0, 4.0, 4.0),
        concat_padding_mask: true,
    }
}

/// A tiny conditioner config — same structure as `ConditionerConfig::anima`, model_dim 8, 1 layer.
fn cond_cfg() -> ConditionerConfig {
    ConditionerConfig {
        source_dim: 8,
        target_dim: 8,
        model_dim: 8,
        num_layers: 1,
        num_attention_heads: 2, // head_dim 4
        mlp_ratio: 4.0,
        target_vocab_size: 16,
        min_sequence_length: 8,
        rope_theta: 10000.0,
        norm_eps: 1e-6,
    }
}

/// Deterministic LCG in `[-1, 1)` seeded by the key — so both DiT copies get identical weights.
fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed & 0x7fff_ffff;
    (0..n)
        .map(|_| {
            s = (s.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fff_ffff;
            (s as f64 / 2147483647.0 * 2.0 - 1.0) as f32
        })
        .collect()
}

fn key_seed(key: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in key.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Insert a synthetic tensor for `key`: norms near 1.0, other weights at 0.3·lcg (keeps the two-layer
/// gated forward finite while a modulation change still moves the output).
fn put(map: &mut HashMap<String, Tensor>, key: &str, shape: &[usize]) {
    let n: usize = shape.iter().product();
    let raw = lcg(n, key_seed(key));
    let data: Vec<f32> = if key.ends_with("norm.weight") {
        raw.iter().map(|&v| 1.0 + 0.1 * v).collect()
    } else {
        raw.iter().map(|&v| 0.3 * v).collect()
    };
    map.insert(
        key.to_string(),
        Tensor::from_vec(data, shape.to_vec(), &Device::Cpu).unwrap(),
    );
}

/// The full dense DiT weight map (Cosmos `net.` names), derived from `cfg`.
fn dit_map(cfg: &DitConfig) -> HashMap<String, Tensor> {
    let h = cfg.hidden_size();
    let hd = cfg.attention_head_dim;
    let lora = cfg.adaln_lora_dim;
    let ctx = cfg.text_embed_dim;
    let ff = (cfg.mlp_ratio * h as f32) as usize;
    let (pt, ph, pw) = cfg.patch_size;
    let patch_in = cfg.patch_in_channels() * pt * ph * pw;
    let proj_out = ph * pw * pt * cfg.out_channels;
    let mut w = HashMap::new();
    put(&mut w, "net.x_embedder.proj.1.weight", &[h, patch_in]);
    put(&mut w, "net.t_embedder.1.linear_1.weight", &[3 * h, h]);
    put(&mut w, "net.t_embedder.1.linear_2.weight", &[3 * h, 3 * h]);
    put(&mut w, "net.t_embedding_norm.weight", &[h]);
    for i in 0..cfg.num_layers {
        let b = format!("net.blocks.{i}");
        for m in [
            "adaln_modulation_self_attn",
            "adaln_modulation_cross_attn",
            "adaln_modulation_mlp",
        ] {
            put(&mut w, &format!("{b}.{m}.1.weight"), &[lora, h]);
            put(&mut w, &format!("{b}.{m}.2.weight"), &[3 * h, lora]);
        }
        for (attn, kv_in) in [("self_attn", h), ("cross_attn", ctx)] {
            put(&mut w, &format!("{b}.{attn}.q_proj.weight"), &[h, h]);
            put(&mut w, &format!("{b}.{attn}.k_proj.weight"), &[h, kv_in]);
            put(&mut w, &format!("{b}.{attn}.v_proj.weight"), &[h, kv_in]);
            put(&mut w, &format!("{b}.{attn}.output_proj.weight"), &[h, h]);
            put(&mut w, &format!("{b}.{attn}.q_norm.weight"), &[hd]);
            put(&mut w, &format!("{b}.{attn}.k_norm.weight"), &[hd]);
        }
        put(&mut w, &format!("{b}.mlp.layer1.weight"), &[ff, h]);
        put(&mut w, &format!("{b}.mlp.layer2.weight"), &[h, ff]);
    }
    put(
        &mut w,
        "net.final_layer.adaln_modulation.1.weight",
        &[lora, h],
    );
    put(
        &mut w,
        "net.final_layer.adaln_modulation.2.weight",
        &[2 * h, lora],
    );
    put(&mut w, "net.final_layer.linear.weight", &[proj_out, h]);
    w
}

/// The full dense conditioner weight map (`llm_adapter.` names), derived from `cfg`.
fn cond_map(cfg: &ConditionerConfig) -> HashMap<String, Tensor> {
    let d = cfg.model_dim;
    let hd = cfg.head_dim();
    let ff = (cfg.mlp_ratio * d as f32) as usize;
    let mut w = HashMap::new();
    put(
        &mut w,
        "llm_adapter.embed.weight",
        &[cfg.target_vocab_size, d],
    );
    for i in 0..cfg.num_layers {
        let b = format!("llm_adapter.blocks.{i}");
        put(&mut w, &format!("{b}.norm_self_attn.weight"), &[d]);
        put(&mut w, &format!("{b}.norm_cross_attn.weight"), &[d]);
        put(&mut w, &format!("{b}.norm_mlp.weight"), &[d]);
        for attn in ["self_attn", "cross_attn"] {
            for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                put(&mut w, &format!("{b}.{attn}.{p}.weight"), &[d, d]);
            }
            put(&mut w, &format!("{b}.{attn}.q_norm.weight"), &[hd]);
            put(&mut w, &format!("{b}.{attn}.k_norm.weight"), &[hd]);
        }
        put(&mut w, &format!("{b}.mlp.0.weight"), &[ff, d]);
        put(&mut w, &format!("{b}.mlp.0.bias"), &[ff]);
        put(&mut w, &format!("{b}.mlp.2.weight"), &[d, ff]);
        put(&mut w, &format!("{b}.mlp.2.bias"), &[d]);
    }
    put(&mut w, "llm_adapter.out_proj.weight", &[cfg.target_dim, d]);
    put(&mut w, "llm_adapter.out_proj.bias", &[cfg.target_dim]);
    put(&mut w, "llm_adapter.norm.weight", &[cfg.target_dim]);
    w
}

fn build_dit() -> CosmosDiT {
    let vb = VarBuilder::from_tensors(dit_map(&dit_cfg()), DType::F32, &Device::Cpu);
    CosmosDiT::new(&vb.pp("net"), dit_cfg()).expect("synthetic DiT loads")
}

fn build_cond() -> AnimaTextConditioner {
    let vb = VarBuilder::from_tensors(cond_map(&cond_cfg()), DType::F32, &Device::Cpu);
    AnimaTextConditioner::new(&vb.pp("llm_adapter"), cond_cfg())
        .expect("synthetic conditioner loads")
}

/// Write a LoRA safetensors targeting `(path, out, in)` triples at `rank`, ComfyUI `diffusion_model.`
/// namespace, factors at 0.1·lcg so the residual is a real (finite) shift.
fn write_lora(
    dir: &std::path::Path,
    targets: &[(&str, usize, usize)],
    rank: usize,
) -> std::path::PathBuf {
    let mut m = HashMap::new();
    for (i, (path, out, inp)) in targets.iter().enumerate() {
        let a = lcg(rank * inp, 1000 + i as u64)
            .iter()
            .map(|v| 0.1 * v)
            .collect::<Vec<_>>();
        let b = lcg(out * rank, 2000 + i as u64)
            .iter()
            .map(|v| 0.1 * v)
            .collect::<Vec<_>>();
        m.insert(
            format!("diffusion_model.{path}.lora_A.weight"),
            Tensor::from_vec(a, (rank, *inp), &Device::Cpu).unwrap(),
        );
        m.insert(
            format!("diffusion_model.{path}.lora_B.weight"),
            Tensor::from_vec(b, (*out, rank), &Device::Cpu).unwrap(),
        );
    }
    let path = dir.join("lora.safetensors");
    candle_gen::candle_core::safetensors::save(&m, &path).unwrap();
    path
}

/// A non-square latent `[1,16,1,4,6]` + encoder `[1,5,8]` + `sigma=0.7` DiT forward (the parity synth).
fn dit_forward(dit: &CosmosDiT) -> Tensor {
    let latent = Tensor::from_vec(lcg(16 * 4 * 6, 1), (1, 16, 1, 4, 6), &Device::Cpu).unwrap();
    let encoder = Tensor::from_vec(lcg(5 * 8, 2), (1, 5, 8), &Device::Cpu).unwrap();
    let sigma = Tensor::from_vec(vec![0.7f32], (1,), &Device::Cpu).unwrap();
    dit.forward(&latent, &sigma, &encoder, DType::F32)
        .expect("forward")
}

/// A conditioner forward: Qwen3-side `source` `[1,5,8]` + T5 query ids `[1,6]` → `[1,512-or-St,8]`. Also
/// exercises the `AdaptLinear`-swapped conditioner forward (only the ignored real-weights stage does).
fn cond_forward(cond: &AnimaTextConditioner) -> Tensor {
    let source = Tensor::from_vec(lcg(5 * 8, 3), (1, 5, 8), &Device::Cpu).unwrap();
    let ids: Vec<u32> = (0..6u32).map(|i| i % 16).collect(); // ids < vocab 16
    let target = Tensor::from_vec(ids, (1, 6), &Device::Cpu).unwrap();
    cond.forward(&source, &target, DType::F32)
        .expect("conditioner forward")
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// The DiT and conditioner targets a real Anima LoRA hits, at this tiny config's dims: a DiT
/// self-attn q (in/out = hidden = 12) and a conditioner self-attn q (in/out = model_dim = 8).
const DIT_TARGET: (&str, usize, usize) = ("blocks.0.self_attn.q_proj", 12, 12);
const COND_TARGET: (&str, usize, usize) = ("llm_adapter.blocks.0.self_attn.q_proj", 8, 8);

/// install routes BOTH a DiT target and a conditioner target (the visit reaches both hosts), the
/// residual shifts the built DiT's forward, and a **scale-0** residual is an exact no-op (mutation).
#[test]
fn install_routes_dit_and_conditioner_and_residual_shifts_forward() {
    let dir = std::env::temp_dir().join(format!("anima_install_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let baseline = dit_forward(&build_dit());
    let lora = write_lora(&dir, &[DIT_TARGET, COND_TARGET], 2);

    // scale 1.0: both targets route, and the DiT forward moves.
    let mut dit = build_dit();
    let mut cond = build_cond();
    let spec = AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora);
    let report = install_anima_residuals(&mut dit, &mut cond, std::slice::from_ref(&spec)).unwrap();
    assert_eq!(
        report.merged, 2,
        "both the DiT and conditioner target must route (visit reaches both hosts)"
    );
    let shifted = dit_forward(&dit);
    assert!(
        max_abs_diff(&shifted, &baseline) > 1e-4,
        "the installed residual must move the DiT forward"
    );
    // The conditioner residual is live in the (AdaptLinear-swapped) conditioner forward too — its output
    // moves vs a fresh un-adapted conditioner. Guards the 60-conditioner-target class (a trained Anima
    // LoRA has non-zero conditioner deltas), which a DiT-only install would silently miss.
    let cond_baseline = cond_forward(&build_cond());
    assert!(
        max_abs_diff(&cond_forward(&cond), &cond_baseline) > 1e-4,
        "the installed conditioner residual must move the conditioner forward"
    );

    // scale 0.0: routes the same two targets, but the forward is byte-identical to baseline (mutation:
    // the residual is real, and the scale gates it — a broken residual would move even at scale 0).
    let mut dit0 = build_dit();
    let mut cond0 = build_cond();
    let spec0 = AdapterSpec::new(lora, 0.0, AdapterKind::Lora);
    let r0 = install_anima_residuals(&mut dit0, &mut cond0, std::slice::from_ref(&spec0)).unwrap();
    assert_eq!(r0.merged, 2);
    assert_eq!(
        max_abs_diff(&dit_forward(&dit0), &baseline),
        0.0,
        "a scale-0 residual must be an exact no-op"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A LoKr (detected by `lokr_` keys even under an `AdapterKind::Lora` spec) installs as a **structured
/// Kronecker residual** and shifts the DiT forward — no `[out,in]` materialized (sc-10713). Targets the
/// DiT q_proj `[out,in] = [12,12]`, factored as `w1 [2,2] ⊗ w2 [6,6]` (a·b = 12, c·d = 12).
#[test]
fn lokr_installs_as_structured_residual_and_shifts_forward() {
    let dir = std::env::temp_dir().join(format!("anima_install_lokr_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let baseline = dit_forward(&build_dit());

    // Full-factor LoKr on the DiT q_proj: w1 [2,2], w2 [6,6] ⇒ kron = [12,12]. Non-zero factors so the
    // residual is real. No metadata ⇒ rank = alpha = 1 ⇒ full scale = 1.0.
    let mut m = HashMap::new();
    let scaled =
        |n: usize, seed: u64| -> Vec<f32> { lcg(n, seed).iter().map(|&v| 0.3f32 * v).collect() };
    m.insert(
        "diffusion_model.blocks.0.self_attn.q_proj.lokr_w1".to_string(),
        Tensor::from_vec(scaled(4, 5), (2, 2), &Device::Cpu).unwrap(),
    );
    m.insert(
        "diffusion_model.blocks.0.self_attn.q_proj.lokr_w2".to_string(),
        Tensor::from_vec(scaled(36, 6), (6, 6), &Device::Cpu).unwrap(),
    );
    let path = dir.join("lokr.safetensors");
    candle_gen::candle_core::safetensors::save(&m, &path).unwrap();

    let mut dit = build_dit();
    let mut cond = build_cond();
    // `AdapterKind::Lora` spec — the `lokr_` keys drive the LoKr dispatch (keys win).
    let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora);
    let report = install_anima_residuals(&mut dit, &mut cond, std::slice::from_ref(&spec))
        .expect("a linear LoKr must install as a structured residual, not error");
    assert_eq!(
        report.merged, 1,
        "the LoKr target must route as a structured residual"
    );
    assert!(
        max_abs_diff(&dit_forward(&dit), &baseline) > 1e-4,
        "the structured LoKr residual must move the DiT forward"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A LoHa (Hadamard, `hada_` keys) is rejected with the actionable sc-10713 message — its Hadamard
/// product has no allocation-free structured form, so it cannot ride unmerged on a packed tier.
#[test]
fn loha_on_the_residual_path_is_a_clear_error() {
    let dir = std::env::temp_dir().join(format!("anima_install_loha_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // A file carrying explicit LoHa factor keys (`hada_*`) → rejected.
    let mut m = HashMap::new();
    for (k, r, c) in [
        ("hada_w1_a", 12usize, 2usize),
        ("hada_w1_b", 2, 12),
        ("hada_w2_a", 12, 2),
        ("hada_w2_b", 2, 12),
    ] {
        m.insert(
            format!("diffusion_model.blocks.0.self_attn.q_proj.{k}"),
            Tensor::from_vec(lcg(r * c, 7), (r, c), &Device::Cpu).unwrap(),
        );
    }
    let path = dir.join("loha.safetensors");
    candle_gen::candle_core::safetensors::save(&m, &path).unwrap();

    let mut dit = build_dit();
    let mut cond = build_cond();
    let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora);
    let err = install_anima_residuals(&mut dit, &mut cond, std::slice::from_ref(&spec))
        .expect_err("LoHa on the packed residual path must error");
    let msg = err.to_string();
    assert!(
        msg.contains("LoHa") && msg.contains("sc-10713"),
        "reject must name LoHa and the follow-up story: {msg}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The visit emits **exactly** the canonical adaptable surface — the routing backstop. The expected set
/// is built independently from the sc-10640 spec, so a typo'd or missing visit-path string (which would
/// make a real Anima LoRA target hit the `did not route` hard error at load) fails HERE instead. Covers
/// every projection class, not just `q_proj`.
#[test]
fn visit_emits_exactly_the_canonical_adaptable_surface() {
    let mut dit = build_dit();
    let mut cond = build_cond();

    let mut emitted: HashSet<String> = HashSet::new();
    {
        let emitted = &mut emitted;
        let mut collect = |path: &str, _lin: &mut AdaptLinear| -> candle_gen::Result<()> {
            assert!(
                emitted.insert(path.to_string()),
                "duplicate visit path: {path}"
            );
            Ok(())
        };
        dit.visit_adaptable_mut(&mut collect).unwrap();
        cond.visit_adaptable_mut(&mut collect).unwrap();
    }

    let mut expected: HashSet<String> = HashSet::new();
    for g in [
        "x_embedder.proj.1",
        "t_embedder.1.linear_1",
        "t_embedder.1.linear_2",
        "final_layer.adaln_modulation.1",
        "final_layer.adaln_modulation.2",
        "final_layer.linear",
    ] {
        expected.insert(g.to_string());
    }
    for i in 0..dit_cfg().num_layers {
        for attn in ["self_attn", "cross_attn"] {
            for p in ["q_proj", "k_proj", "v_proj", "output_proj"] {
                expected.insert(format!("blocks.{i}.{attn}.{p}"));
            }
        }
        for m in [
            "adaln_modulation_self_attn",
            "adaln_modulation_cross_attn",
            "adaln_modulation_mlp",
        ] {
            expected.insert(format!("blocks.{i}.{m}.1"));
            expected.insert(format!("blocks.{i}.{m}.2"));
        }
        expected.insert(format!("blocks.{i}.mlp.layer1"));
        expected.insert(format!("blocks.{i}.mlp.layer2"));
    }
    for i in 0..cond_cfg().num_layers {
        for attn in ["self_attn", "cross_attn"] {
            for p in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                expected.insert(format!("llm_adapter.blocks.{i}.{attn}.{p}"));
            }
        }
        expected.insert(format!("llm_adapter.blocks.{i}.mlp.0"));
        expected.insert(format!("llm_adapter.blocks.{i}.mlp.2"));
    }
    expected.insert("llm_adapter.out_proj".to_string());

    assert_eq!(
        emitted, expected,
        "the visit must emit EXACTLY the canonical adaptable surface — a missing/typo'd path would make \
         a real Anima LoRA target hard-error at load (sc-10274)"
    );
}

/// A LoRA target that matches no DiT/conditioner projection is a hard error (no silent partial residual
/// — sc-10274), never a quiet drop.
#[test]
fn off_surface_target_is_a_hard_error() {
    let dir = std::env::temp_dir().join(format!("anima_install_unrouted_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // A bogus path present in no host surface.
    let lora = write_lora(&dir, &[("blocks.0.self_attn.does_not_exist", 12, 12)], 2);

    let mut dit = build_dit();
    let mut cond = build_cond();
    let spec = AdapterSpec::new(lora, 1.0, AdapterKind::Lora);
    let err = install_anima_residuals(&mut dit, &mut cond, std::slice::from_ref(&spec))
        .expect_err("an off-surface target must not be silently dropped");
    assert!(
        err.to_string().contains("did not route"),
        "expected an unrouted-target error, got: {err}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
