//! The frozen training adapter on a synthetic two-block DiT (the committed `dit_block.safetensors`
//! fixture, CPU, float32): installed through the inference key mapping on exactly the projections
//! a Fizgig-shaped kohya file names, carried by every block residency the trainer has, frozen
//! beside the trainable factors, and never written into the saved adapter.
//!
//! Each test builds complete synthetic DiTs and runs their learned projections, so it is a model
//! proof: ignored by ordinary CI, registered in `scripts/ci/model-inference-proofs.json` and run
//! only by the explicit model-proof workflow (`--features model-inference-proof`, which observes
//! the learned output's device and prints the late BODY marker).

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::train::TrainingConfig;
use candle_gen::train::flow_match::save_adapter;
use candle_gen::Weights;

use super::training_adapter::install_training_adapter;
use super::{install_adapters, provenance};
use crate::dit::{MiniMaxH3Dit, MiniMaxH3DitConfig};

/// The fixture's geometry: hidden 64, four 24-wide heads (inner 96), SwiGLU width 32, two blocks.
fn config() -> MiniMaxH3DitConfig {
    MiniMaxH3DitConfig {
        num_attention_heads: 4,
        attention_head_dim: 24,
        hidden_size: 64,
        num_layers: 2,
        num_refiner_layers: 2,
        ffn_dim: 32,
        text_dim: 40,
        freq_dim: 16,
        time_embed_hidden_dim: 64,
        time_embed_dim: 48,
        rope_freq_dim: 2,
        ..Default::default()
    }
}

const HIDDEN: usize = 64;
const INNER: usize = 96;
const FFN: usize = 32;
/// The adapter's rank: not a multiple of three, so the fused `qkv_proj` splits shared-`A` (the
/// published rank-16 file's case).
const RANK: usize = 2;
const ALPHA: f32 = 1.0;

fn dit() -> MiniMaxH3Dit {
    let map = candle_gen::candle_core::safetensors::load(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/dit_block.safetensors"
        ),
        &Device::Cpu,
    )
    .unwrap();
    MiniMaxH3Dit::from_weights(&Weights::from_map(map), &config(), &Device::Cpu, DType::F32)
        .unwrap()
}

fn seeded(shape: &[usize], seed: u32) -> Tensor {
    let n: usize = shape.iter().product();
    let values: Vec<f32> = (0..n as u32)
        .map(|i| ((i.wrapping_mul(2_654_435_761) ^ seed) % 997) as f32 / 997.0 - 0.5)
        .collect();
    Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
}

/// Every `(block, kohya module) → (lora_down, lora_up)` an adapter file holds.
type Factors = HashMap<(usize, &'static str), (Tensor, Tensor)>;

/// The four kohya modules the shipped Fizgig adapter names per block, with their `(in, out)`.
const MODULES: [(&str, usize, usize); 4] = [
    ("attn_qkv_proj", HIDDEN, 3 * INNER),
    ("attn_out_proj", INNER, HIDDEN),
    ("mlp_fc1", HIDDEN, 2 * FFN),
    ("mlp_fc2", FFN, HIDDEN),
];

/// A Fizgig-shaped adapter (`lora_unet_blocks_{i}_<module>.{lora_down,lora_up}.weight` + `.alpha`)
/// over `blocks`; returns the file and every `(block, module) → (down, up)`.
fn write_adapter(dir: &Path, blocks: &[usize]) -> (PathBuf, Factors) {
    let mut tensors = HashMap::new();
    let mut factors = HashMap::new();
    for &block in blocks {
        for (i, (module, inp, out)) in MODULES.iter().enumerate() {
            let seed = (block * 16 + i) as u32;
            let down = (seeded(&[RANK, *inp], seed) * 0.5).unwrap();
            let up = (seeded(&[*out, RANK], seed + 101) * 0.5).unwrap();
            let stem = format!("lora_unet_blocks_{block}_{module}");
            tensors.insert(format!("{stem}.lora_down.weight"), down.clone());
            tensors.insert(format!("{stem}.lora_up.weight"), up.clone());
            tensors.insert(
                format!("{stem}.alpha"),
                Tensor::new(ALPHA, &Device::Cpu).unwrap(),
            );
            factors.insert((block, *module), (down, up));
        }
    }
    let file = dir.join("fizgig_shaped_training_adapter.safetensors");
    candle_gen::candle_core::safetensors::save(&tensors, &file).unwrap();
    (file, factors)
}

/// The residual a `(down, up)` pair adds at strength `s`: `s · alpha/rank · x · downᵀ · upᵀ`.
fn residual(x: &Tensor, down: &Tensor, up: &Tensor, s: f32) -> Tensor {
    let fold = f64::from(s * ALPHA / RANK as f32);
    (x.matmul(&down.t().unwrap())
        .unwrap()
        .matmul(&up.t().unwrap())
        .unwrap()
        * fold)
        .unwrap()
}

/// `up`'s rows narrowed to `[start, start + len)`.
fn rows(up: &Tensor, start: usize, len: usize) -> Tensor {
    up.narrow(0, start, len).unwrap()
}

/// The delta each DiT leaf of `block` receives from the kohya modules, per the inference mapping:
/// `qkv_proj` → `to_{q,k,v}` (shared `A`, `B` split in thirds), `out_proj` → `to_out.0`, `fc1` →
/// `ff.net.0.proj` with its SwiGLU halves swapped, `fc2` → `ff.net.2`.
fn expected(factors: &Factors, block: usize, leaf: &str, x: &Tensor, s: f32) -> Tensor {
    let get = |m: &str| factors[&(block, MODULES.iter().find(|e| e.0 == m).unwrap().0)].clone();
    match leaf {
        "attn.to_q" | "attn.to_k" | "attn.to_v" => {
            let (down, up) = get("attn_qkv_proj");
            let third = ["attn.to_q", "attn.to_k", "attn.to_v"]
                .iter()
                .position(|l| *l == leaf)
                .unwrap();
            residual(x, &down, &rows(&up, third * INNER, INNER), s)
        }
        "attn.to_out.0" => {
            let (down, up) = get("attn_out_proj");
            residual(x, &down, &up, s)
        }
        "ff.net.0.proj" => {
            let (down, up) = get("mlp_fc1");
            let swapped = Tensor::cat(&[rows(&up, FFN, FFN), rows(&up, 0, FFN)], 0).unwrap();
            residual(x, &down, &swapped, s)
        }
        "ff.net.2" => {
            let (down, up) = get("mlp_fc2");
            residual(x, &down, &up, s)
        }
        other => panic!("{other}"),
    }
}

const LEAVES: [(&str, usize); 6] = [
    ("attn.to_q", HIDDEN),
    ("attn.to_k", HIDDEN),
    ("attn.to_v", HIDDEN),
    ("attn.to_out.0", INNER),
    ("ff.net.0.proj", HIDDEN),
    ("ff.net.2", FFN),
];

fn segs(block: usize, leaf: &str) -> Vec<String> {
    format!("transformer_blocks.{block}.{leaf}")
        .split('.')
        .map(str::to_owned)
        .collect()
}

fn leaf_forward(dit: &mut MiniMaxH3Dit, block: usize, leaf: &str, x: &Tensor) -> Tensor {
    let segs = segs(block, leaf);
    let segs: Vec<&str> = segs.iter().map(String::as_str).collect();
    __model_proof_output!(dit.adaptable_mut(&segs).unwrap().forward(x)).unwrap()
}

fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// (a) Through the inference key mapping, every projection the Fizgig-shaped file names moves by
/// exactly its mapped delta at the configured strength — the fused `qkv_proj` split, the SwiGLU
/// swap — and nothing else; (b) a streamed window copy and the resident promotion carry it.
/// Mutations: skip the install ⇒ every delta assert fails; strength 2 ⇒ the delta assert fails;
/// a window copy dropping its residuals ⇒ the residency asserts fail.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn the_adapter_adds_its_mapped_delta_in_every_residency() {
    let tmp = tempfile::tempdir().unwrap();
    let (file, factors) = write_adapter(tmp.path(), &[0, 1]);
    for strength in [1.0f32, 0.5] {
        let mut bare = dit();
        let mut adapted = dit();
        let applied = install_training_adapter(&mut adapted, &file, strength).unwrap();
        assert_eq!(applied, 2 * LEAVES.len());
        for block in 0..2 {
            for (leaf, inp) in LEAVES {
                let x = seeded(&[5, inp], 7 + block as u32);
                let got = (leaf_forward(&mut adapted, block, leaf, &x)
                    - leaf_forward(&mut bare, block, leaf, &x))
                .unwrap();
                let want = expected(&factors, block, leaf, &x, strength);
                assert!(
                    max_abs(&got, &want) < 1e-5,
                    "{strength} block {block} {leaf}: {}",
                    max_abs(&got, &want)
                );
            }
        }
        // Untargeted modules stay bit-identical (the refiner's projections here).
        let x = seeded(&[3, HIDDEN], 3);
        let path = ["token_refiner", "refiner_blocks", "0", "attn", "to_q"];
        assert_eq!(
            max_abs(
                &__model_proof_output!(adapted.adaptable_mut(&path).unwrap().forward(&x)).unwrap(),
                &__model_proof_output!(bare.adaptable_mut(&path).unwrap().forward(&x)).unwrap()
            ),
            0.0
        );
    }

    // The trainer's residencies: the host blocks it installs on, a streamed window copy of each,
    // and the resident promotion — every one computes base + adapter.
    let mut adapted = dit();
    install_training_adapter(&mut adapted, &file, 1.0).unwrap();
    let x = seeded(&[4, HIDDEN], 11);
    let want: Vec<Tensor> = (0..2)
        .map(|block| leaf_forward(&mut adapted, block, "attn.to_q", &x))
        .collect();
    adapted.prepare_streamed(&Device::Cpu, 1).unwrap();
    for (block, want) in want.iter().enumerate() {
        let mut window = adapted.window_block(block).unwrap().into_owned();
        let lin = window.adaptable_mut(&["attn", "to_q"]).unwrap();
        assert!(lin.is_adapted(), "the window copy carries the residual");
        assert_eq!(
            max_abs(&__model_proof_output!(lin.forward(&x)).unwrap(), want),
            0.0
        );
    }
    adapted.promote_streamed_blocks().unwrap();
    for (block, want) in want.iter().enumerate() {
        assert_eq!(
            max_abs(&leaf_forward(&mut adapted, block, "attn.to_q", &x), want),
            0.0
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED training::training_adapter_tests::the_adapter_adds_its_mapped_delta_in_every_residency");
}

/// The training adapter is frozen and never saved: the trainable factors install beside it on
/// the same leaves, the optimizer's `Var`s are exactly the trainable factors, a gradient flows to
/// them through base + adapter (and differs from the bare base's), and the saved adapter carries
/// only the trainable keys. Mutations: install the adapter's factors as `Var`s ⇒ the var-count
/// assert fails; skip the adapter on the trained leaf ⇒ the gradient assert fails; save the
/// adapter's tensors ⇒ the key assert fails.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn training_runs_through_the_adapter_and_saves_only_the_trainable_keys() {
    let tmp = tempfile::tempdir().unwrap();
    let (file, _) = write_adapter(tmp.path(), &[0, 1]);
    let cfg = TrainingConfig {
        rank: 4,
        alpha: 4.0,
        seed: 3,
        ..TrainingConfig::default()
    };
    let paths: Vec<String> = (0..2)
        .flat_map(|b| {
            LEAVES
                .iter()
                .map(move |(l, _)| format!("transformer_blocks.{b}.{l}"))
        })
        .collect();
    let grad_of = |with_adapter: bool| -> (Tensor, Vec<String>) {
        let mut d = dit();
        if with_adapter {
            install_training_adapter(&mut d, &file, 1.0).unwrap();
        }
        let set = install_adapters(&mut d, &paths, &cfg, &Device::Cpu).unwrap();
        assert_eq!(
            set.vars.len(),
            2 * paths.len(),
            "only the trainable A/B are Vars"
        );
        // Give B a value so the trainable residual is live, as after a first update.
        for (name, var) in set.named_vars() {
            if name.contains("lora_B") || name.contains("lora_up") {
                var.set(&(seeded(var.dims(), 5) * 0.1).unwrap()).unwrap();
            }
        }
        let x = seeded(&[6, FFN], 13);
        let y = leaf_forward(&mut d, 1, "ff.net.2", &x);
        let grads = y.sqr().unwrap().sum_all().unwrap().backward().unwrap();
        let (_, a) = set
            .named_vars()
            .into_iter()
            .find(|(n, _)| n.starts_with("transformer_blocks.1.ff.net.2") && n.contains("lora_A"))
            .unwrap();
        let g = grads.get(a.as_tensor()).unwrap().clone();
        let saved = tmp
            .path()
            .join(format!("trained-{with_adapter}.safetensors"));
        save_adapter(&set, &provenance(false), &saved).unwrap();
        let keys = candle_gen::candle_core::safetensors::load(&saved, &Device::Cpu)
            .unwrap()
            .into_keys()
            .collect();
        (g, keys)
    };
    let (adapted_grad, adapted_keys) = grad_of(true);
    let (bare_grad, bare_keys) = grad_of(false);
    assert!(
        max_abs(&adapted_grad, &bare_grad) > 1e-4,
        "the trainable factor's gradient flows through base + adapter"
    );
    let keys: BTreeSet<String> = adapted_keys.into_iter().collect();
    assert_eq!(keys, bare_keys.into_iter().collect::<BTreeSet<_>>());
    assert_eq!(keys.len(), 3 * paths.len());
    for key in &keys {
        assert!(
            paths.iter().any(|p| key.starts_with(&format!("{p}.")))
                && (key.ends_with(".lora_A.weight")
                    || key.ends_with(".lora_B.weight")
                    || key.ends_with(".alpha")),
            "{key}"
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED training::training_adapter_tests::training_runs_through_the_adapter_and_saves_only_the_trainable_keys");
}

/// A file that does not fit the DiT is a named refusal, never a partial install: a block the DiT
/// does not have, a key that names no H3 module, and a missing file.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_bad_training_adapter_is_a_named_error() {
    let tmp = tempfile::tempdir().unwrap();
    let (stray, _) = write_adapter(tmp.path(), &[0, 7]);
    let err = install_training_adapter(&mut dit(), &stray, 1.0)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("training adapter") && err.contains("blocks"),
        "{err}"
    );

    let unknown = tmp.path().join("unknown.safetensors");
    let mut tensors = HashMap::new();
    tensors.insert(
        "lora_unet_blocks_0_not_a_module.lora_down.weight".to_owned(),
        seeded(&[RANK, HIDDEN], 1),
    );
    tensors.insert(
        "lora_unet_blocks_0_not_a_module.lora_up.weight".to_owned(),
        seeded(&[HIDDEN, RANK], 2),
    );
    candle_gen::candle_core::safetensors::save(&tensors, &unknown).unwrap();
    let err = install_training_adapter(&mut dit(), &unknown, 1.0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not_a_module"), "{err}");

    let missing = tmp.path().join("missing.safetensors");
    let mut refused = dit();
    let err = install_training_adapter(&mut refused, &missing, 1.0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("missing.safetensors"), "{err}");
    // A refused install leaves the DiT bare: its projections compute exactly the fresh base.
    let x = seeded(&[3, HIDDEN], 17);
    assert_eq!(
        max_abs(
            &leaf_forward(&mut refused, 0, "attn.to_q", &x),
            &leaf_forward(&mut dit(), 0, "attn.to_q", &x)
        ),
        0.0
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED training::training_adapter_tests::a_bad_training_adapter_is_a_named_error");
}
