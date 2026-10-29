use super::*;
use crate::adapters::{AdaptableLinear, Adapter};
use crate::runtime::{AdapterKind, AdapterSpec};
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::ops::{all_close, array_eq};
use std::collections::HashMap;
use std::path::PathBuf;

/// A host whose modules live at arbitrary dotted paths — including segment names with internal
/// underscores (`feed_forward`, `to_out.0`) so the kohya flattening is genuinely ambiguous and a
/// blind `_`→`.` split would mis-route. `adaptable_paths` returns the registered paths, so it
/// exercises the real `flattened → dotted` table path.
struct MultiHost {
    mods: HashMap<String, AdaptableLinear>,
    paths: Vec<String>,
}
impl MultiHost {
    fn new(specs: &[(&str, Array)]) -> Self {
        let mut mods = HashMap::new();
        let mut paths = Vec::new();
        for (p, w) in specs {
            mods.insert((*p).to_string(), AdaptableLinear::dense(w.clone(), None));
            paths.push((*p).to_string());
        }
        Self { mods, paths }
    }
}
impl AdaptableHost for MultiHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        self.mods.get_mut(&path.join("."))
    }
    fn adaptable_paths(&self) -> Vec<String> {
        self.paths.clone()
    }
}

/// iOS: every adapter file read is checked against live headroom (its header-only floor)
/// BEFORE it is read, on each apply path; with no headroom nothing is read or applied.
#[cfg(target_os = "macos")]
#[test]
fn adapter_reads_check_their_header_floor_before_reading() {
    const CHILD: &str = "MLX_GEN_ADAPTER_LOAD_FLOOR_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "adapters::loader::tests::adapter_reads_check_their_header_floor_before_reading",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let tmp = tempfile::tempdir().unwrap();
    let path = scratch_file(&tmp, "lora.safetensors");
    let down = Array::ones::<f32>(&[2, 4]).unwrap();
    let up = Array::ones::<f32>(&[4, 2]).unwrap();
    Array::save_safetensors(
        vec![("lin.lora_A.weight", &down), ("lin.lora_B.weight", &up)],
        None,
        &path,
    )
    .unwrap();
    let spec = AdapterSpec::new(path.clone(), 1.0, AdapterKind::Lora);
    let base = Array::ones::<f32>(&[4, 4]).unwrap();
    let mut host = OneLinear {
        lin: AdaptableLinear::dense(base.clone(), None),
    };
    let _ = crate::ios_quantum::take_refusal();
    let refused = crate::ios_quantum::with_cpu_diagnostic_headroom(0, || {
        apply_adapter_specs(&mut host, std::slice::from_ref(&spec), None)
    });
    assert!(refused.is_err());
    let record = crate::ios_quantum::take_refusal().expect("typed refusal");
    assert_eq!(record.label, ADAPTER_LOAD_QUANTUM);
    // Header floor: both tensors' stored bytes — a load peaks at exactly what it stores
    // (`ios_quantum_budget::load_floor_bytes`, proven by measurement), no extra read scratch.
    assert_eq!(record.scratch_bytes, 32 + 32);
    assert!(
        host.lin.adapters().is_empty(),
        "nothing applied past the refusal"
    );
    // With headroom it applies exactly as off-device.
    crate::ios_quantum::with_cpu_diagnostic_headroom(1 << 30, || {
        apply_adapter_specs(&mut host, std::slice::from_ref(&spec), None)
    })
    .unwrap();
}

/// Minimal host with a single adaptable linear at path `["lin"]`.
struct OneLinear {
    lin: AdaptableLinear,
}
impl AdaptableHost for OneLinear {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["lin"] => Some(&mut self.lin),
            _ => None,
        }
    }
}

fn scratch_file(tmp: &tempfile::TempDir, name: &str) -> PathBuf {
    // Per-process scratch dir — a fixed `$TMPDIR` name races a second concurrent `cargo test`.
    let dir = tmp.path().join("mlx_gen_loader_test");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Mimics the Krea host's native-dialect routing (sc-8185 aliases `blocks`/`wq…`) but,
/// like every `AdaptableHost`, routes from the *bare* module path — it has no
/// `diffusion_model.` arm. So it only resolves an ai-toolkit key once the
/// `strip_common_lora_prefix` step (sc-8395) removes that namespace.
struct BareModuleHost {
    lin: AdaptableLinear,
}
impl AdaptableHost for BareModuleHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["blocks", "0", "attn", "wq"] => Some(&mut self.lin),
            _ => None,
        }
    }
    fn adaptable_paths(&self) -> Vec<String> {
        vec!["blocks.0.attn.wq".to_string()]
    }
}

#[test]
fn strip_common_lora_prefix_removes_known_namespaces() {
    assert_eq!(
        strip_common_lora_prefix("diffusion_model.blocks.0.attn.wq"),
        "blocks.0.attn.wq"
    );
    assert_eq!(strip_common_lora_prefix("transformer.x.y"), "x.y");
    // No known prefix → returned unchanged.
    assert_eq!(
        strip_common_lora_prefix("blocks.0.attn.wq"),
        "blocks.0.attn.wq"
    );
}

#[test]
fn thirdparty_lokr_resolves_diffusion_model_prefixed_keys() {
    let tmp = tempfile::tempdir().unwrap();
    // sc-8395: ostris ai-toolkit writes Krea-2 LoKr keys as
    // `diffusion_model.‹native path›.lokr_w*`. Before the prefix strip the dotted
    // fallback handed the host `["diffusion_model", …]`, which matches no arm →
    // "no target modules matched". After the strip it resolves via the bare path.
    let weight = Array::from_slice(
        &(0..16).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &[4, 4],
    );
    // Both factors full ⇒ lycoris forces scale 1; w1[2,2] ⊗ w2[2,2] = ΔW[4,4].
    let w1 = Array::from_slice(&[0.5f32, 0.6, 0.7, 0.8], &[2, 2]);
    let w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
    let path = scratch_file(&tmp, "krea_aitoolkit_lokr.safetensors");
    Array::save_safetensors(
        vec![
            ("diffusion_model.blocks.0.attn.wq.lokr_w1", &w1),
            ("diffusion_model.blocks.0.attn.wq.lokr_w2", &w2),
        ],
        None,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();
    assert!(is_lokr_keys(&w));

    let mut host = BareModuleHost {
        lin: AdaptableLinear::dense(weight, None),
    };
    let report = apply_lokr_thirdparty(&mut host, &w, 1.0).unwrap();
    assert_eq!(
        report.applied, 1,
        "the diffusion_model.-prefixed key must resolve after the strip"
    );
    assert!(report.unmatched_paths.is_empty());
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn lora_peft_transposes_and_folds_alpha() {
    let tmp = tempfile::tempdir().unwrap();
    // base [out=4, in=3]; PEFT lora_A [r=2, in=3], lora_B [out=4, r=2], alpha=4 (rank=2).
    let weight = Array::from_slice(
        &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &[4, 3],
    );
    let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
    let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
    let alpha = Array::from_slice(&[4.0f32], &[1]);

    let path = scratch_file(&tmp, "lora.safetensors");
    Array::save_safetensors(
        vec![
            ("lin.lora_A.weight", &a_raw),
            ("lin.lora_B.weight", &b_raw),
            ("lin.alpha", &alpha),
        ],
        None,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();

    let mut host = OneLinear {
        lin: AdaptableLinear::dense(weight.clone(), None),
    };
    let report = apply_lora_peft(&mut host, &w, 0.5, None).unwrap();
    assert_eq!(report.applied, 1);
    assert!(report.unmatched_paths.is_empty());

    // Reference: a = A^T [in,r], b = B^T * (alpha/rank=2.0) [r,out], scale 0.5.
    let mut expected = AdaptableLinear::dense(weight, None);
    let b_scaled = b_raw
        .t()
        .multiply(Array::from_slice(&[2.0f32], &[1]))
        .unwrap();
    expected.push(Adapter::Lora {
        a: a_raw.t(),
        b: b_scaled,
        scale: 0.5,
    });

    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let got = host.lin.forward(&x).unwrap();
    let want = expected.forward(&x).unwrap();
    assert!(all_close(&got, &want, 1e-5, 1e-5, false)
        .unwrap()
        .item::<bool>());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::lora_peft_transposes_and_folds_alpha");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn lora_peft_honors_lora_adapter_metadata_alpha() {
    let tmp = tempfile::tempdir().unwrap();
    // sc-5513: a diffusers / PEFT `save_lora_adapter` file carries NO per-target `.alpha` tensor —
    // the scaling lives in the `lora_adapter_metadata` header blob. With `lora_alpha = 16`, `r = 8`
    // the PEFT loader must fold `(16/8) = 2.0` (the metadata strength), not the pre-sc-5513
    // `alpha = rank` default (factor 1.0). Proves the blob is read and applied.
    let weight = Array::from_slice(
        &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &[4, 3],
    );
    // PEFT factors with a TRUE rank of 8 (matching the blob `r`): A [r=8, in=3], B [out=4, r=8].
    let a_raw = Array::from_slice(
        &(0..24).map(|i| i as f32 * 0.03 - 0.3).collect::<Vec<_>>(),
        &[8, 3],
    );
    let b_raw = Array::from_slice(
        &(0..32).map(|i| 0.4 - i as f32 * 0.02).collect::<Vec<_>>(),
        &[4, 8],
    );

    let path = scratch_file(&tmp, "lora_adapter_metadata.safetensors");
    let meta = HashMap::from([(
        "lora_adapter_metadata".to_string(),
        r#"{"lora_alpha": 16, "r": 8}"#.to_string(),
    )]);
    // Deliberately NO `lin.alpha` tensor — the scaling must come from the blob.
    Array::save_safetensors(
        vec![("lin.lora_A.weight", &a_raw), ("lin.lora_B.weight", &b_raw)],
        Some(&meta),
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();
    assert!(w.metadata("lora_adapter_metadata").is_some());

    let mut host = OneLinear {
        lin: AdaptableLinear::dense(weight.clone(), None),
    };
    let report = apply_lora_peft(&mut host, &w, 1.0, None).unwrap();
    assert_eq!(report.applied, 1);

    // Reference: alpha 16 over rank 8 ⇒ factor 2.0 folded into B (scale 1.0).
    let mut expected = AdaptableLinear::dense(weight.clone(), None);
    expected.push(Adapter::Lora {
        a: a_raw.t(),
        b: b_raw
            .t()
            .multiply(Array::from_slice(&[2.0f32], &[1]))
            .unwrap(),
        scale: 1.0,
    });
    // The pre-sc-5513 default (alpha = rank ⇒ factor 1.0) would diverge by a full factor of 2.
    let mut buggy = AdaptableLinear::dense(weight, None);
    buggy.push(Adapter::Lora {
        a: a_raw.t(),
        b: b_raw.t(),
        scale: 1.0,
    });

    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let got = host.lin.forward(&x).unwrap();
    let want = expected.forward(&x).unwrap();
    let old = buggy.forward(&x).unwrap();
    assert!(
        all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>(),
        "metadata-alpha fold must match (16/8)·scale"
    );
    assert!(
        !all_close(&got, &old, 1e-4, 1e-4, false)
            .unwrap()
            .item::<bool>(),
        "metadata alpha must differ from the alpha=rank default"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::lora_peft_honors_lora_adapter_metadata_alpha");
}

/// sc-5513 **live torch-PEFT A/B** (the epic 3641 / sc-3671 on-device harness; torch IS available
/// on the Mac at `~/mlx-flux-venv`). `#[ignore]` — gated on a real diffusers `save_lora_adapter`
/// file, generated with peft 0.19 + diffusers 0.37:
/// ```text
/// ~/mlx-flux-venv/bin/python - <<'PY'
/// import os, torch
/// from diffusers import UNet2DConditionModel
/// from peft import LoraConfig
/// torch.manual_seed(0)
/// unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
///     block_out_channels=(16,32), down_block_types=("CrossAttnDownBlock2D","DownBlock2D"),
///     up_block_types=("UpBlock2D","CrossAttnUpBlock2D"), cross_attention_dim=16,
///     attention_head_dim=2, norm_num_groups=4)
/// unet.add_adapter(LoraConfig(r=8, lora_alpha=16, target_modules=["to_q","to_k","to_v"],
///     alpha_pattern={"to_k":8}, rank_pattern={"to_k":16}, init_lora_weights=False))
/// unet.save_lora_adapter("/tmp/sc5513_adapter")
/// PY
/// SC5513_PEFT_ADAPTER=/tmp/sc5513_adapter/pytorch_lora_weights.safetensors \
///   cargo test -p mlx-gen peft_lora_adapter_metadata_ab -- --ignored --nocapture
/// ```
/// peft's authoritative per-module scaling (`mod.scaling['default']`) is then `to_q`/`to_v` = 16/8 =
/// 2.0 and `to_k` = 8/16 = 0.5 (the override is deliberately discriminating). The Rust core loader
/// must install each residual at exactly that scaling — proving the `lora_adapter_metadata` blob is
/// honored on a genuine torch file (which carries NO per-target `.alpha` tensor, the bug's premise).
#[test]
#[ignore = "needs a diffusers save_lora_adapter file via SC5513_PEFT_ADAPTER (see doc comment)"]
fn peft_lora_adapter_metadata_ab() {
    let Ok(path) = std::env::var("SC5513_PEFT_ADAPTER") else {
        eprintln!("SC5513_PEFT_ADAPTER unset — skipping live torch A/B");
        crate::model_inference_device::record_model_inference_prerequisite_missing("required work unavailable in peft_lora_adapter_metadata_ab; see preceding skip diagnostic");
        return;
    };
    let w = Weights::from_file(&path).unwrap();
    // The whole premise: a real diffusers `save_lora_adapter` file ships NO per-target `.alpha`
    // tensor — the scaling lives in the `lora_adapter_metadata` blob.
    assert!(
        !w.keys().any(|k| k.ends_with(".alpha")),
        "diffusers save_lora_adapter must not ship a per-target .alpha tensor"
    );
    assert!(w.metadata(wmeta::LORA_ADAPTER_METADATA_KEY).is_some());

    let block = "down_blocks.0.attentions.0.transformer_blocks.0.attn1";
    // peft ground truth (confirmed independently via `mod.scaling`): global 2.0, `to_k` override 0.5.
    for (leaf, peft_scale) in [("to_q", 2.0f32), ("to_k", 0.5f32)] {
        let module = format!("{block}.{leaf}");
        let a_raw = w
            .require(&format!("{module}.lora_A.weight"))
            .unwrap()
            .clone();
        let b_raw = w
            .require(&format!("{module}.lora_B.weight"))
            .unwrap()
            .clone();
        let out = b_raw.shape()[0];
        let inp = a_raw.shape()[1];
        // Zero base so the forward IS the pure (scaled) residual.
        let zero = Array::from_slice(&vec![0f32; (out * inp) as usize], &[out, inp]);
        let mut host = MultiHost::new(&[(module.as_str(), zero.clone())]);
        let report = apply_lora_peft(&mut host, &w, 1.0, None).unwrap();
        assert!(report.applied >= 1, "{leaf}: not applied");

        // Reference at peft's ground-truth scaling: residual = (x·Aᵀ·Bᵀ)·peft_scale.
        let mut expect = AdaptableLinear::dense(zero, None);
        expect.push(Adapter::Lora {
            a: a_raw.t(),
            b: b_raw
                .t()
                .multiply(Array::from_slice(&[peft_scale], &[1]))
                .unwrap(),
            scale: 1.0,
        });
        let x = Array::from_slice(
            &(0..inp).map(|i| (i as f32 * 0.3).sin()).collect::<Vec<_>>(),
            &[1, inp],
        );
        let segs: Vec<&str> = module.split('.').collect();
        let got = host.adaptable_mut(&segs).unwrap().forward(&x).unwrap();
        let want = expect.forward(&x).unwrap();
        assert!(
            all_close(&got, &want, 1e-4, 1e-4, false)
                .unwrap()
                .item::<bool>(),
            "{leaf}: Rust apply diverged from peft scaling {peft_scale}"
        );
        println!("OK {leaf}: Rust apply matches peft scaling {peft_scale}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::peft_lora_adapter_metadata_ab"
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn lora_bf16_scalar_alpha_reads_without_panic() {
    let tmp = tempfile::tempdir().unwrap();
    // sc-2657: real kohya/BFL FLUX LoRAs ship `alpha` as a **bf16 scalar of shape []**. The alpha
    // read used `as_slice::<f32>()`, which `unwrap`s a dtype mismatch and would panic on bf16 — a
    // latent bug masked by every prior test synthesizing f32 alpha. The fix casts to f32 first.
    // Here a bf16 `[]`-shaped alpha must load AND fold identically to its f32 `[1]`-shaped twin.
    let weight = Array::from_slice(
        &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &[4, 3],
    );
    let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
    let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
    // alpha = 16 (exactly representable in bf16), stored bf16 and 0-d — like the real file.
    let alpha_bf16 = Array::from_slice(&[16.0f32], &[1])
        .reshape(&[] as &[i32])
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    let path = scratch_file(&tmp, "lora_bf16_alpha.safetensors");
    Array::save_safetensors(
        vec![
            ("lin.lora_A.weight", &a_raw),
            ("lin.lora_B.weight", &b_raw),
            ("lin.alpha", &alpha_bf16),
        ],
        None,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();

    let mut host = OneLinear {
        lin: AdaptableLinear::dense(weight.clone(), None),
    };
    let report = apply_lora_peft(&mut host, &w, 0.5, None).unwrap();
    assert_eq!(report.applied, 1, "bf16 alpha LoRA should apply, not panic");

    // Reference: identical fold with alpha=16, rank=2 → factor 8.
    let mut expected = AdaptableLinear::dense(weight, None);
    expected.push(Adapter::Lora {
        a: a_raw.t(),
        b: b_raw
            .t()
            .multiply(Array::from_slice(&[8.0f32], &[1]))
            .unwrap(),
        scale: 0.5,
    });
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let got = host.lin.forward(&x).unwrap();
    let want = expected.forward(&x).unwrap();
    assert!(all_close(&got, &want, 1e-5, 1e-5, false)
        .unwrap()
        .item::<bool>());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::lora_bf16_scalar_alpha_reads_without_panic");
}

#[test]
fn scalar_alpha_empty_tensor_errors_not_panic() {
    // sc-3959 added the no-panic guard for a malformed third-party adapter with a zero-length
    // `.alpha` (before it, `as_slice::<f32>()` panicked on the size-0 array, aborting the worker).
    // sc-5281 then tightened it from a silent `Ok(None)` to a typed error: the callers only reach
    // `scalar_alpha` for a *present* alpha key, so present-but-empty is a malformed file — returning
    // `None` would be indistinguishable from "no alpha" and silently mis-scale the adapter.
    let empty = Array::from_slice(&[] as &[f32], &[0]);
    assert_eq!(empty.size(), 0);
    let err = scalar_alpha(&empty)
        .expect_err("size-0 alpha must be a typed error, not Ok")
        .to_string();
    assert!(err.contains("empty"), "got: {err}");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn lora_peft_folds_bare_alpha_under_a_prefix() {
    let tmp = tempfile::tempdir().unwrap();
    // Prefixed `lora_A/B` (`transformer.lin.lora_{A,B}.weight`) + a BARE `lin.alpha` — the
    // fork's Qwen convention (bare-only alpha patterns). The bare alpha must NOT be dropped:
    // the residual folds alpha/rank into B exactly as the all-bare case does. (sc-2528 review.)
    let weight = Array::from_slice(
        &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &[4, 3],
    );
    let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r=2, in=3]
    let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
    let alpha = Array::from_slice(&[4.0f32], &[1]); // rank=2 -> factor 2

    let path = scratch_file(&tmp, "lora_prefixed_bare_alpha.safetensors");
    Array::save_safetensors(
        vec![
            ("transformer.lin.lora_A.weight", &a_raw),
            ("transformer.lin.lora_B.weight", &b_raw),
            ("lin.alpha", &alpha), // BARE — no `transformer.` prefix
        ],
        None,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();

    let mut host = OneLinear {
        lin: AdaptableLinear::dense(weight.clone(), None),
    };
    let report = apply_lora_peft(&mut host, &w, 0.5, Some("transformer.")).unwrap();
    assert_eq!(report.applied, 1);
    assert!(report.unmatched_paths.is_empty());

    // Reference: B scaled by alpha/rank = 2 (the bare alpha was honored).
    let mut expected = AdaptableLinear::dense(weight, None);
    let b_scaled = b_raw
        .t()
        .multiply(Array::from_slice(&[2.0f32], &[1]))
        .unwrap();
    expected.push(Adapter::Lora {
        a: a_raw.t(),
        b: b_scaled,
        scale: 0.5,
    });
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let got = host.lin.forward(&x).unwrap();
    let want = expected.forward(&x).unwrap();
    assert!(
        all_close(&got, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>(),
        "bare alpha under a prefix was dropped or mis-folded"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::lora_peft_folds_bare_alpha_under_a_prefix");
}

#[test]
fn lora_peft_conflicting_alpha_errors() {
    let tmp = tempfile::tempdir().unwrap();
    // A prefixed alpha and a bare alpha that disagree for the same path -> hard error, no
    // silent pick.
    let weight = Array::from_slice(
        &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &[4, 3],
    );
    let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
    let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
    let path = scratch_file(&tmp, "lora_conflicting_alpha.safetensors");
    Array::save_safetensors(
        vec![
            ("transformer.lin.lora_A.weight", &a_raw),
            ("transformer.lin.lora_B.weight", &b_raw),
            ("transformer.lin.alpha", &Array::from_slice(&[4.0f32], &[1])),
            ("lin.alpha", &Array::from_slice(&[8.0f32], &[1])), // disagrees
        ],
        None,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();
    let mut host = OneLinear {
        lin: AdaptableLinear::dense(weight, None),
    };
    assert!(apply_lora_peft(&mut host, &w, 1.0, Some("transformer.")).is_err());
}

#[test]
fn unmatched_paths_are_reported_not_dropped() {
    let tmp = tempfile::tempdir().unwrap();
    // A LoKr file targeting a path the host doesn't have -> applied 0, path reported.
    let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
    let mut meta = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("alpha".to_string(), "1.0".to_string());
    meta.insert("rank".to_string(), "1".to_string());
    let path = scratch_file(&tmp, "lokr_miss.safetensors");
    Array::save_safetensors(
        vec![
            ("missing.path.lokr_w1", &dummy),
            ("missing.path.lokr_w2", &dummy),
        ],
        Some(&meta),
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();
    assert!(is_lokr(&w));

    let mut host = OneLinear {
        lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
    };
    let report = apply_lokr(&mut host, &w, 1.0).unwrap();
    assert_eq!(report.applied, 0);
    assert_eq!(report.unmatched_paths, vec!["missing.path".to_string()]);
}

// ---- BFL/ComfyUI fused→split LyCORIS (sc-8345) ------------------------------------------------

/// A host with a BFL surface: the three split attention projections (`to_q/to_k/to_v`, the fused-
/// qkv destinations) plus a rename destination (`to_out`), and a `bfl_targets()` mapping the fused
/// `diffusion_model.double_blocks.0.img_attn.{qkv,proj}` BFL names onto them — the minimal shape of
/// a FLUX `Flux2Transformer` for the LyCORIS fused→split path.
struct BflHost {
    mods: HashMap<String, AdaptableLinear>,
}
impl BflHost {
    fn new() -> Self {
        let mut mods = HashMap::new();
        // qkv splits: each [out=2, in=3] → the fused source is [6, 3].
        for dst in ["to_q", "to_k", "to_v"] {
            mods.insert(
                format!("transformer_blocks.0.attn.{dst}"),
                AdaptableLinear::dense(Array::from_slice(&[0.0f32; 6], &[2, 3]), None),
            );
        }
        // rename dest: [out=4, in=4].
        mods.insert(
            "transformer_blocks.0.attn.to_out".to_string(),
            AdaptableLinear::dense(Array::from_slice(&[0.0f32; 16], &[4, 4]), None),
        );
        Self { mods }
    }
}
impl AdaptableHost for BflHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        self.mods.get_mut(&path.join("."))
    }
    fn adaptable_paths(&self) -> Vec<String> {
        self.mods.keys().cloned().collect()
    }
    fn bfl_targets(&self) -> Vec<BflTarget> {
        let block_keys = |module: &str| {
            (
                vec![
                    format!("diffusion_model.{module}.lora_B.weight"),
                    format!("diffusion_model.{module}.lora_up.weight"),
                ],
                vec![
                    format!("diffusion_model.{module}.lora_A.weight"),
                    format!("diffusion_model.{module}.lora_down.weight"),
                ],
                vec![format!("diffusion_model.{module}.alpha")],
            )
        };
        let mut out = Vec::new();
        let (up, down, alpha) = block_keys("double_blocks.0.img_attn.qkv");
        for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
            out.push(BflTarget {
                target_path: format!("transformer_blocks.0.attn.{dst}"),
                up_keys: up.clone(),
                down_keys: down.clone(),
                alpha_keys: alpha.clone(),
                up_slice: Some(LoraRowSlice::Chunk {
                    n: 3,
                    index: idx as i32,
                }),
                down_slice: Some(LoraRowSlice::ChunkIfDivisible {
                    n: 3,
                    index: idx as i32,
                }),
            });
        }
        let (up, down, alpha) = block_keys("double_blocks.0.img_attn.proj");
        out.push(BflTarget {
            target_path: "transformer_blocks.0.attn.to_out".to_string(),
            up_keys: up,
            down_keys: down,
            alpha_keys: alpha,
            up_slice: None,
            down_slice: None,
        });
        out
    }
}

/// A `networkType=lokr` file in BFL/ComfyUI fused naming (`diffusion_model.…img_attn.qkv` fused,
/// `…img_attn.proj` renamed) must apply onto a FLUX-shaped host: the fused qkv delta is rebuilt at
/// `[6,3]` and row-sliced into `to_q/to_k/to_v`, and the proj rename lands whole on `to_out`. Before
/// sc-8345 every target surfaced as unmatched (the strict apply errored). Exercises the full
/// `apply_adapters_strict` dispatch, not just the leaf applier.
#[test]
fn bfl_named_lokr_fused_qkv_and_rename_resolve() {
    let tmp = tempfile::tempdir().unwrap();
    // Fused qkv LoKr: kron(w1[3,1], w2[2,3]) → [6,3].
    let qkv_w1 = Array::from_slice(&[1.0f32, 0.5, -0.25], &[3, 1]);
    let qkv_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    // Proj rename LoKr: kron(w1[2,2], w2[2,2]) → [4,4].
    let proj_w1 = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
    let proj_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
    let mut meta = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("alpha".to_string(), "1.0".to_string());
    meta.insert("rank".to_string(), "1".to_string());
    let path = scratch_file(&tmp, "bfl_lokr_fused.safetensors");
    Array::save_safetensors(
        vec![
            (
                "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1",
                &qkv_w1,
            ),
            (
                "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2",
                &qkv_w2,
            ),
            (
                "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1",
                &proj_w1,
            ),
            (
                "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2",
                &proj_w2,
            ),
        ],
        Some(&meta),
        &path,
    )
    .unwrap();

    let mut host = BflHost::new();
    let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lokr);
    let report =
        apply_adapters_strict(&mut host, std::slice::from_ref(&spec), "flux2_klein_9b").unwrap();
    assert_eq!(report.applied, 4);
    assert!(report.unmatched_paths.is_empty());

    // The fused qkv delta, reconstructed independently at the fused shape, row-sliced into thirds.
    let full = reconstruct_lokr_delta(
        1.0,
        1.0,
        &[6, 3],
        Some(&qkv_w1),
        None,
        None,
        Some(&qkv_w2),
        None,
        None,
        Dtype::Bfloat16,
    )
    .unwrap();
    for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
        let lin = host
            .adaptable_mut(&["transformer_blocks", "0", "attn", dst])
            .unwrap();
        let Adapter::Lokr { delta, scale } = &lin.adapters()[0] else {
            panic!("expected a LoKr adapter on {dst}");
        };
        assert_eq!(*scale, 1.0);
        let start = idx as i32 * 2;
        let want = full.try_index((start..start + 2, ..)).unwrap();
        assert!(
            all_close(delta, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>(),
            "qkv split {dst} delta mismatch"
        );
    }

    // Proj rename lands whole on to_out.
    let proj_full = reconstruct_lokr_delta(
        1.0,
        1.0,
        &[4, 4],
        Some(&proj_w1),
        None,
        None,
        Some(&proj_w2),
        None,
        None,
        Dtype::Bfloat16,
    )
    .unwrap();
    let lin = host
        .adaptable_mut(&["transformer_blocks", "0", "attn", "to_out"])
        .unwrap();
    let Adapter::Lokr { delta, .. } = &lin.adapters()[0] else {
        panic!("expected a LoKr adapter on to_out");
    };
    assert!(all_close(delta, &proj_full, 1e-5, 1e-5, false)
        .unwrap()
        .item::<bool>());
}

/// A bare diffusers-named LoKr (split `to_q` directly, no BFL fused name) on the SAME BFL host must
/// still route through the ordinary `apply_lokr` path, NOT the fused→split one — `is_bfl_lycoris`
/// keys only off the BFL spellings, so non-BFL LyCORIS is untouched by sc-8345.
#[test]
fn bare_diffusers_lokr_on_bfl_host_stays_on_plain_path() {
    let tmp = tempfile::tempdir().unwrap();
    // kron(w1[2,1], w2[1,3]) → [2,3], the shape of the split to_q.
    let w1 = Array::from_slice(&[1.0f32, 0.5], &[2, 1]);
    let w2 = Array::from_slice(&[0.1f32, 0.2, 0.3], &[1, 3]);
    let mut meta = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("alpha".to_string(), "1.0".to_string());
    meta.insert("rank".to_string(), "1".to_string());
    let path = scratch_file(&tmp, "bare_lokr_on_bfl_host.safetensors");
    Array::save_safetensors(
        vec![
            ("transformer_blocks.0.attn.to_q.lokr_w1", &w1),
            ("transformer_blocks.0.attn.to_q.lokr_w2", &w2),
        ],
        Some(&meta),
        &path,
    )
    .unwrap();
    let mut host = BflHost::new();
    let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lokr);
    let report = apply_adapter_specs_autoprefix(&mut host, std::slice::from_ref(&spec)).unwrap();
    assert_eq!(report.applied, 1);
    assert!(report.unmatched_paths.is_empty());
    let lin = host
        .adaptable_mut(&["transformer_blocks", "0", "attn", "to_q"])
        .unwrap();
    assert!(matches!(lin.adapters()[0], Adapter::Lokr { .. }));
}

/// A third-party LyCORIS LoKr (no `networkType` stamp — detected by `lokr_*` keys) in BFL fused
/// naming routes through the fused→split applier too (sc-8345). Both-full factors ⇒ lycoris scale 1.
#[test]
fn bfl_named_thirdparty_lokr_fused_qkv_resolves() {
    let tmp = tempfile::tempdir().unwrap();
    // kron(w1[3,1], w2[2,3]) → [6,3]; both factors full ⇒ scale 1.0.
    let qkv_w1 = Array::from_slice(&[1.0f32, 0.5, -0.25], &[3, 1]);
    let qkv_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let proj_w1 = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
    let proj_w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
    let path = scratch_file(&tmp, "bfl_tp_lokr_fused.safetensors");
    // NO `networkType` metadata → is_lokr() false, is_lokr_keys() true (the third-party path).
    Array::save_safetensors(
        vec![
            (
                "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1",
                &qkv_w1,
            ),
            (
                "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2",
                &qkv_w2,
            ),
            (
                "diffusion_model.double_blocks.0.img_attn.proj.lokr_w1",
                &proj_w1,
            ),
            (
                "diffusion_model.double_blocks.0.img_attn.proj.lokr_w2",
                &proj_w2,
            ),
        ],
        None,
        &path,
    )
    .unwrap();
    let mut host = BflHost::new();
    let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora); // kind irrelevant — keys route it
    let report =
        apply_adapters_strict(&mut host, std::slice::from_ref(&spec), "flux2_klein_9b").unwrap();
    assert_eq!(report.applied, 4);
    assert!(report.unmatched_paths.is_empty());

    let full = reconstruct_lokr_delta_scaled(
        1.0,
        &[6, 3],
        Some(&qkv_w1),
        None,
        None,
        Some(&qkv_w2),
        None,
        None,
        None,
        Dtype::Bfloat16,
    )
    .unwrap();
    for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
        let lin = host
            .adaptable_mut(&["transformer_blocks", "0", "attn", dst])
            .unwrap();
        let Adapter::Lokr { delta, .. } = &lin.adapters()[0] else {
            panic!("expected a LoKr adapter on {dst}");
        };
        let start = idx as i32 * 2;
        let want = full.try_index((start..start + 2, ..)).unwrap();
        assert!(all_close(delta, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }
}

/// A third-party LyCORIS LoHa (`hada_*` keys) in BFL fused naming routes through the fused→split
/// applier (sc-8345); the Hadamard delta is rebuilt at the fused shape, then row-sliced.
#[test]
fn bfl_named_loha_fused_qkv_resolves() {
    let tmp = tempfile::tempdir().unwrap();
    // (w1_a@w1_b) ⊙ (w2_a@w2_b) at [6,3], rank r=1 ⇒ scale 1.0.
    let w1_a = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[6, 1]);
    let w1_b = Array::from_slice(&[1.0f32, -1.0, 0.5], &[1, 3]);
    let w2_a = Array::from_slice(&[0.6f32, 0.5, 0.4, 0.3, 0.2, 0.1], &[6, 1]);
    let w2_b = Array::from_slice(&[0.2f32, 0.4, -0.2], &[1, 3]);
    let path = scratch_file(&tmp, "bfl_loha_fused.safetensors");
    Array::save_safetensors(
        vec![
            (
                "diffusion_model.double_blocks.0.img_attn.qkv.hada_w1_a",
                &w1_a,
            ),
            (
                "diffusion_model.double_blocks.0.img_attn.qkv.hada_w1_b",
                &w1_b,
            ),
            (
                "diffusion_model.double_blocks.0.img_attn.qkv.hada_w2_a",
                &w2_a,
            ),
            (
                "diffusion_model.double_blocks.0.img_attn.qkv.hada_w2_b",
                &w2_b,
            ),
        ],
        None,
        &path,
    )
    .unwrap();
    let mut host = BflHost::new();
    let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora);
    let report =
        apply_adapters_strict(&mut host, std::slice::from_ref(&spec), "flux2_klein_9b").unwrap();
    // Only the fused qkv is present → the three splits; proj/to_out untouched (no factors for it).
    assert_eq!(report.applied, 3);
    assert!(report.unmatched_paths.is_empty());

    let full = reconstruct_loha_delta(
        1.0,
        &[6, 3],
        &w1_a,
        &w1_b,
        &w2_a,
        &w2_b,
        None,
        None,
        Dtype::Bfloat16,
    )
    .unwrap();
    for (idx, dst) in ["to_q", "to_k", "to_v"].iter().enumerate() {
        let lin = host
            .adaptable_mut(&["transformer_blocks", "0", "attn", dst])
            .unwrap();
        let Adapter::Lokr { delta, .. } = &lin.adapters()[0] else {
            panic!("expected an installed delta on {dst}");
        };
        let start = idx as i32 * 2;
        let want = full.try_index((start..start + 2, ..)).unwrap();
        assert!(all_close(delta, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }
}

/// The load-time connector stacks a mixed LoRA + LoKr spec list and is equivalent to calling
/// the underlying loaders directly, in order.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn apply_specs_stacks_mixed_lora_and_lokr() {
    let tmp = tempfile::tempdir().unwrap();
    // base [out=4, in=2].
    let base_vals: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
    let weight = Array::from_slice(&base_vals, &[4, 2]);

    // PEFT LoRA file targeting ["lin"]: lora_A [r=2, in=2], lora_B [out=4, r=2].
    let a_raw = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2], &[2, 2]);
    let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
    let lora_path = scratch_file(&tmp, "specs_lora.safetensors");
    Array::save_safetensors(
        vec![("lin.lora_A.weight", &a_raw), ("lin.lora_B.weight", &b_raw)],
        None,
        &lora_path,
    )
    .unwrap();

    // LoKr file targeting ["lin"]: kron(w1[2,1], w2[2,2]) -> [4,2]; alpha==rank -> factor 1.
    let w1 = Array::from_slice(&[1.0f32, 0.5], &[2, 1]);
    let w2 = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
    let mut meta = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("alpha".to_string(), "1.0".to_string());
    meta.insert("rank".to_string(), "1".to_string());
    let lokr_path = scratch_file(&tmp, "specs_lokr.safetensors");
    Array::save_safetensors(
        vec![("lin.lokr_w1", &w1), ("lin.lokr_w2", &w2)],
        Some(&meta),
        &lokr_path,
    )
    .unwrap();

    let specs = vec![
        AdapterSpec {
            path: lora_path.clone(),
            scale: 0.5,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        },
        AdapterSpec {
            path: lokr_path.clone(),
            scale: 1.0,
            kind: AdapterKind::Lokr,
            pass_scales: None,
            moe_expert: None,
        },
    ];

    let mut via_specs = OneLinear {
        lin: AdaptableLinear::dense(weight.clone(), None),
    };
    let report = apply_adapter_specs(&mut via_specs, &specs, None).unwrap();
    assert_eq!(report.applied, 2);
    assert!(report.unmatched_paths.is_empty());

    // Reference: the same files through the underlying loaders directly, in order.
    let mut via_loaders = OneLinear {
        lin: AdaptableLinear::dense(weight, None),
    };
    apply_lora_peft(
        &mut via_loaders,
        &Weights::from_file(&lora_path).unwrap(),
        0.5,
        None,
    )
    .unwrap();
    apply_lokr(
        &mut via_loaders,
        &Weights::from_file(&lokr_path).unwrap(),
        1.0,
    )
    .unwrap();

    let x = Array::from_slice(&[1.0f32, -2.0], &[1, 2]);
    let got = via_specs.lin.forward(&x).unwrap();
    let want = via_loaders.lin.forward(&x).unwrap();
    assert!(all_close(&got, &want, 1e-5, 1e-5, false)
        .unwrap()
        .item::<bool>());

    // Both adapters actually moved the output off the bare base.
    let base = AdaptableLinear::dense(Array::from_slice(&base_vals, &[4, 2]), None)
        .forward(&x)
        .unwrap();
    assert!(!all_close(&got, &base, 1e-5, 1e-5, false)
        .unwrap()
        .item::<bool>());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::apply_specs_stacks_mixed_lora_and_lokr");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn apply_specs_empty_is_noop() {
    let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let mut host = OneLinear {
        lin: AdaptableLinear::dense(weight.clone(), None),
    };
    let report = apply_adapter_specs(&mut host, &[], None).unwrap();
    assert_eq!(report, ApplyReport::default());

    let x = Array::from_slice(&[1.0f32, -1.0], &[1, 2]);
    let got = host.lin.forward(&x).unwrap();
    let want = AdaptableLinear::dense(weight, None).forward(&x).unwrap();
    assert!(all_close(&got, &want, 1e-6, 1e-6, false)
        .unwrap()
        .item::<bool>());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::apply_specs_empty_is_noop");
}

#[test]
fn apply_specs_reports_unmatched_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
    let mut meta = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("alpha".to_string(), "1.0".to_string());
    meta.insert("rank".to_string(), "1".to_string());
    let path = scratch_file(&tmp, "specs_miss.safetensors");
    Array::save_safetensors(
        vec![("nope.here.lokr_w1", &dummy), ("nope.here.lokr_w2", &dummy)],
        Some(&meta),
        &path,
    )
    .unwrap();

    let specs = vec![AdapterSpec {
        path,
        scale: 1.0,
        kind: AdapterKind::Lokr,
        pass_scales: None,
        moe_expert: None,
    }];
    let mut host = OneLinear {
        lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
    };
    let report = apply_adapter_specs(&mut host, &specs, None).unwrap();
    assert_eq!(report.applied, 0);
    assert_eq!(report.unmatched_paths, vec!["nope.here".to_string()]);
}

#[test]
fn apply_specs_kind_metadata_mismatch_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
    let mut meta = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    let path = scratch_file(&tmp, "specs_mismatch.safetensors");
    Array::save_safetensors(vec![("lin.lokr_w1", &dummy)], Some(&meta), &path).unwrap();

    // Declared Lora but the file's metadata says LoKr -> a loud error, not a silent no-op.
    let specs = vec![AdapterSpec {
        path,
        scale: 1.0,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    }];
    let mut host = OneLinear {
        lin: AdaptableLinear::dense(Array::from_slice(&[1.0f32], &[1, 1]), None),
    };
    assert!(apply_adapter_specs(&mut host, &specs, None).is_err());
}

#[test]
fn detect_lora_prefix_variants() {
    let tmp = tempfile::tempdir().unwrap();
    let a = Array::from_slice(&[0.0f32], &[1, 1]);
    let bare = scratch_file(&tmp, "detect_bare.safetensors");
    Array::save_safetensors(vec![("lin.lora_A.weight", &a)], None, &bare).unwrap();
    assert_eq!(
        detect_lora_prefix(&Weights::from_file(&bare).unwrap()),
        None
    );

    let tf = scratch_file(&tmp, "detect_tf.safetensors");
    Array::save_safetensors(vec![("transformer.lin.lora_A.weight", &a)], None, &tf).unwrap();
    assert_eq!(
        detect_lora_prefix(&Weights::from_file(&tf).unwrap()),
        Some("transformer.")
    );

    let dm = scratch_file(&tmp, "detect_dm.safetensors");
    Array::save_safetensors(vec![("diffusion_model.lin.lora_A.weight", &a)], None, &dm).unwrap();
    assert_eq!(
        detect_lora_prefix(&Weights::from_file(&dm).unwrap()),
        Some("diffusion_model.")
    );
}

#[test]
fn autoprefix_strips_detected_prefix_and_applies() {
    let tmp = tempfile::tempdir().unwrap();
    // base [out=2, in=2]; a `transformer.`-prefixed peft LoRA on path ["lin"].
    let weight = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
    let a = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2], &[2, 2]); // [r=2, in=2]
    let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]); // [out=2, r=2]
    let path = scratch_file(&tmp, "autoprefix_lora.safetensors");
    Array::save_safetensors(
        vec![
            ("transformer.lin.lora_A.weight", &a),
            ("transformer.lin.lora_B.weight", &b),
        ],
        None,
        &path,
    )
    .unwrap();

    let mut host = OneLinear {
        lin: AdaptableLinear::dense(weight, None),
    };
    let specs = vec![AdapterSpec {
        path,
        scale: 1.0,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    }];
    let report = apply_adapter_specs_autoprefix(&mut host, &specs).unwrap();
    assert_eq!(
        report.applied, 1,
        "transformer.-prefixed key should resolve to lin"
    );
    assert!(report.unmatched_paths.is_empty());

    // Strict wrapper: a bare-but-unmatched target errors rather than silently dropping.
    let miss = scratch_file(&tmp, "autoprefix_miss.safetensors");
    Array::save_safetensors(
        vec![
            ("transformer.nope.lora_A.weight", &a),
            ("transformer.nope.lora_B.weight", &b),
        ],
        None,
        &miss,
    )
    .unwrap();
    let mut host2 = OneLinear {
        lin: AdaptableLinear::dense(Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]), None),
    };
    let specs2 = vec![AdapterSpec {
        path: miss,
        scale: 1.0,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    }];
    assert!(apply_adapters_strict(&mut host2, &specs2, "test").is_err());
}

/// sc-2909: a diffusers/ComfyUI LoRA spelled with `lora_down`/`lora_up` factor suffixes on a
/// **dotted, un-prefixed** path (the lightx2v Qwen-Image-Lightning format) routes through the
/// PEFT loader (no `lora_unet_` prefix → not kohya) and installs the BYTE-IDENTICAL adapter to
/// its `lora_A`/`lora_B` twin — and `apply_adapter_specs_autoprefix` resolves it end-to-end.
#[test]
fn diffusers_lora_down_up_equals_peft_ab() {
    let tmp = tempfile::tempdir().unwrap();
    let weight = Array::from_slice(
        &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &[4, 3],
    );
    let a_raw = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r, in]
    let b_raw = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]); // [out, r]
    let alpha = Array::from_slice(&[4.0f32], &[1]);

    // down==A, up==B, bare alpha, no namespace prefix — exactly the lightx2v Lightning spelling.
    let down_path = scratch_file(&tmp, "diffusers_down_up.safetensors");
    Array::save_safetensors(
        vec![
            ("lin.lora_down.weight", &a_raw),
            ("lin.lora_up.weight", &b_raw),
            ("lin.alpha", &alpha),
        ],
        None,
        &down_path,
    )
    .unwrap();
    // Detected as un-prefixed (not kohya, not BFL) and resolved through the strict seam.
    let w = Weights::from_file(&down_path).unwrap();
    assert!(!is_kohya(&w), "dotted-path lora_down is NOT kohya");
    assert_eq!(detect_lora_prefix(&w), None, "no namespace prefix");

    let mut via_down = OneLinear {
        lin: AdaptableLinear::dense(weight.clone(), None),
    };
    let report = apply_adapter_specs_autoprefix(
        &mut via_down,
        &[AdapterSpec::new(down_path, 0.5, AdapterKind::Lora)],
    )
    .unwrap();
    assert_eq!(report.applied, 1, "lora_down/up resolved to lin");
    assert!(report.unmatched_paths.is_empty());

    // The `lora_A`/`lora_B` twin must install the identical adapter.
    let ab_path = scratch_file(&tmp, "diffusers_ab_twin.safetensors");
    Array::save_safetensors(
        vec![
            ("lin.lora_A.weight", &a_raw),
            ("lin.lora_B.weight", &b_raw),
            ("lin.alpha", &alpha),
        ],
        None,
        &ab_path,
    )
    .unwrap();
    let mut via_ab = OneLinear {
        lin: AdaptableLinear::dense(weight, None),
    };
    apply_lora_peft(
        &mut via_ab,
        &Weights::from_file(&ab_path).unwrap(),
        0.5,
        None,
    )
    .unwrap();

    let pull = |h: &mut OneLinear| match h.adaptable_mut(&["lin"]).unwrap().adapters() {
        [Adapter::Lora { a, b, scale }] => (a.clone(), b.clone(), *scale),
        _ => panic!("expected one LoRA"),
    };
    let (da, db, ds) = pull(&mut via_down);
    let (pa, pb, ps) = pull(&mut via_ab);
    assert_eq!(ds, ps);
    assert!(
        array_eq(&da, &pa, false).unwrap().item::<bool>()
            && array_eq(&db, &pb, false).unwrap().item::<bool>(),
        "lora_down/up and lora_A/B installed different adapters"
    );
}

// ---- kohya `lora_unet_` LoRA (sc-2618) ----

/// Two modules whose flattened kohya stems are ambiguous under a blind `_`→`.` split: the
/// segment `to_out.0` and the segment name `feed_forward` both contain the separator char.
fn kohya_two_module_host() -> MultiHost {
    let w_out = Array::from_slice(
        &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &[4, 3],
    );
    let w_ff = Array::from_slice(
        &(0..15).map(|i| i as f32 * 0.07).collect::<Vec<_>>(),
        &[5, 3],
    );
    MultiHost::new(&[
        ("blocks.0.attn.to_out.0", w_out),
        ("blocks.0.feed_forward.w1", w_ff),
    ])
}

/// The same (down, up, alpha) factors written in BOTH conventions and applied through the
/// provider seam must yield byte-identical adapters — a kohya file is interchangeable with its
/// PEFT twin. This is the sc-2618 gate at the core level (no model weights needed).
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn kohya_equiv_to_peft_bit_exact() {
    let tmp = tempfile::tempdir().unwrap();
    // out=4/in=3 and out=5/in=3, rank=2; alpha=4 (≠ rank → exercises the alpha/rank fold).
    let a_out = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]); // [r,in]
    let b_out = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]); // [out,r]
    let a_ff = Array::from_slice(&[0.05f32, -0.15, 0.2, 0.3, -0.25, 0.1], &[2, 3]);
    let b_ff = Array::from_slice(
        &[0.2f32, -0.2, 0.1, 0.3, -0.1, 0.4, 0.15, -0.35, 0.05, 0.25],
        &[5, 2],
    );
    let alpha = Array::from_slice(&[4.0f32], &[1]);

    let kohya_path = scratch_file(&tmp, "equiv_kohya.safetensors");
    Array::save_safetensors(
        vec![
            ("lora_unet_blocks_0_attn_to_out_0.lora_down.weight", &a_out),
            ("lora_unet_blocks_0_attn_to_out_0.lora_up.weight", &b_out),
            ("lora_unet_blocks_0_attn_to_out_0.alpha", &alpha),
            ("lora_unet_blocks_0_feed_forward_w1.lora_down.weight", &a_ff),
            ("lora_unet_blocks_0_feed_forward_w1.lora_up.weight", &b_ff),
            ("lora_unet_blocks_0_feed_forward_w1.alpha", &alpha),
        ],
        None as Option<&HashMap<String, String>>,
        &kohya_path,
    )
    .unwrap();

    let peft_path = scratch_file(&tmp, "equiv_peft.safetensors");
    Array::save_safetensors(
        vec![
            ("transformer.blocks.0.attn.to_out.0.lora_A.weight", &a_out),
            ("transformer.blocks.0.attn.to_out.0.lora_B.weight", &b_out),
            ("transformer.blocks.0.attn.to_out.0.alpha", &alpha),
            ("transformer.blocks.0.feed_forward.w1.lora_A.weight", &a_ff),
            ("transformer.blocks.0.feed_forward.w1.lora_B.weight", &b_ff),
            ("transformer.blocks.0.feed_forward.w1.alpha", &alpha),
        ],
        None as Option<&HashMap<String, String>>,
        &peft_path,
    )
    .unwrap();

    let mut via_kohya = kohya_two_module_host();
    let rep_k = apply_adapters_strict(
        &mut via_kohya,
        &[AdapterSpec {
            path: kohya_path,
            scale: 0.75,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
        "test",
    )
    .unwrap();
    assert_eq!(rep_k.applied, 2, "both kohya modules resolve");

    let mut via_peft = kohya_two_module_host();
    apply_adapters_strict(
        &mut via_peft,
        &[AdapterSpec {
            path: peft_path,
            scale: 0.75,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
        "test",
    )
    .unwrap();

    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    for p in ["blocks.0.attn.to_out.0", "blocks.0.feed_forward.w1"] {
        let gk = via_kohya.mods.get(p).unwrap().forward(&x).unwrap();
        let gp = via_peft.mods.get(p).unwrap().forward(&x).unwrap();
        assert!(
            array_eq(&gk, &gp, false).unwrap().item::<bool>(),
            "kohya and peft adapters diverged at {p}"
        );
        // And both actually moved off the bare base.
        let base = AdaptableLinear::dense(
            via_kohya
                .mods
                .get(p)
                .unwrap()
                .dense_weight()
                .unwrap()
                .0
                .clone(),
            None,
        )
        .forward(&x)
        .unwrap();
        assert!(
            !array_eq(&gk, &base, false).unwrap().item::<bool>(),
            "adapter at {p} was a no-op"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::kohya_equiv_to_peft_bit_exact"
    );
}

/// The flattened stem `blocks_0_feed_forward_w1` must resolve to `blocks.0.feed_forward.w1`
/// (the table), NOT the blind split `blocks.0.feed.forward.w1` — proving the disambiguation does
/// real work.
#[test]
fn kohya_table_disambiguates_underscore_segment_names() {
    let mut host = kohya_two_module_host();
    // The blind `_`→`.` split target does not exist; the correct dotted path does.
    assert!(host
        .adaptable_mut(&["blocks", "0", "feed", "forward", "w1"])
        .is_none());
    assert!(host
        .adaptable_mut(&["blocks", "0", "feed_forward", "w1"])
        .is_some());

    let table = kohya_table(&host.adaptable_paths());
    assert_eq!(
        table.get("blocks_0_feed_forward_w1").map(String::as_str),
        Some("blocks.0.feed_forward.w1")
    );
    assert_eq!(
        table.get("blocks_0_attn_to_out_0").map(String::as_str),
        Some("blocks.0.attn.to_out.0")
    );
}

/// A `lora_unet_` key whose stem is off-surface (e.g. FLUX.2 BFL `double_blocks_*`, sc-2743) is
/// surfaced in `unmatched_paths` and fails the strict policy — loud, never silently dropped.
#[test]
fn kohya_offsurface_stem_surfaced_and_strict_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
    let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
    let path = scratch_file(&tmp, "kohya_offsurface.safetensors");
    Array::save_safetensors(
        vec![
            (
                "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
                &a,
            ),
            ("lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight", &b),
        ],
        None as Option<&HashMap<String, String>>,
        &path,
    )
    .unwrap();

    let mut host = kohya_two_module_host();
    let table = kohya_table(&host.adaptable_paths());
    let report =
        apply_lora_kohya(&mut host, &Weights::from_file(&path).unwrap(), 1.0, &table).unwrap();
    assert_eq!(report.applied, 0);
    assert_eq!(
        report.unmatched_paths,
        vec!["double_blocks_0_img_attn_qkv".to_string()]
    );

    // Through the strict provider seam it is a hard error.
    let mut host2 = kohya_two_module_host();
    assert!(apply_adapters_strict(
        &mut host2,
        &[AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
        "test",
    )
    .is_err());
}

/// A kohya adapter at `scale = 0` is a bit-exact no-op (the scale-0 invariant), and `is_kohya`
/// detects the format.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn kohya_scale_zero_is_bit_exact_noop() {
    let tmp = tempfile::tempdir().unwrap();
    let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, -0.3], &[2, 3]);
    let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75, 0.1, 0.2, -0.3, 0.4], &[4, 2]);
    let path = scratch_file(&tmp, "kohya_scale0.safetensors");
    Array::save_safetensors(
        vec![
            ("lora_unet_blocks_0_attn_to_out_0.lora_down.weight", &a),
            ("lora_unet_blocks_0_attn_to_out_0.lora_up.weight", &b),
        ],
        None as Option<&HashMap<String, String>>,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();
    assert!(is_kohya(&w));

    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let mut host = kohya_two_module_host();
    let base = host
        .mods
        .get("blocks.0.attn.to_out.0")
        .unwrap()
        .forward(&x)
        .unwrap();
    let table = kohya_table(&host.adaptable_paths());
    apply_lora_kohya(&mut host, &w, 0.0, &table).unwrap();
    let out = host
        .mods
        .get("blocks.0.attn.to_out.0")
        .unwrap()
        .forward(&x)
        .unwrap();
    assert!(array_eq(&out, &base, false).unwrap().item::<bool>());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::kohya_scale_zero_is_bit_exact_noop");
}

// ---- BFL / ComfyUI fused→split LoRA (sc-2743) ----

/// The [`LoraRowSlice`] variants are byte-faithful to the fork's `LoraTransforms`. Expected values
/// are pinned to the mflux venv (`LoraTransforms.split_*` on the same inputs, sc-2743): up always
/// slices, down is shared unless the rank is divisible, and the qkv-mlp `dims` slice matches the
/// `[3072,3072,3072,12288]` boundaries.
#[test]
fn lora_row_slice_matches_fork_transforms() {
    // arange(6,2): split_q_up = rows[0:2], split_v_up = rows[4:6].
    let t6 = Array::from_slice(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[6, 2]);
    let q_up = LoraRowSlice::Chunk { n: 3, index: 0 }.apply(&t6).unwrap();
    assert_eq!(q_up.shape(), &[2, 2]);
    assert_eq!(q_up.as_slice::<f32>(), &[0.0, 1.0, 2.0, 3.0]);
    let v_up = LoraRowSlice::Chunk { n: 3, index: 2 }.apply(&t6).unwrap();
    assert_eq!(v_up.as_slice::<f32>(), &[8.0, 9.0, 10.0, 11.0]);

    // down: ChunkIfDivisible — whole when rank%3!=0 (the usual fused-qkv LoRA), sliced when ==0.
    let d4 = Array::from_slice(&(0..8).map(|i| i as f32).collect::<Vec<_>>(), &[4, 2]);
    let d4q = LoraRowSlice::ChunkIfDivisible { n: 3, index: 0 }
        .apply(&d4)
        .unwrap();
    assert_eq!(d4q.shape(), &[4, 2], "rank 4 not ÷3 → shared whole");
    assert_eq!(d4q.as_slice::<f32>(), d4.as_slice::<f32>());
    let d6q = LoraRowSlice::ChunkIfDivisible { n: 3, index: 0 }
        .apply(&t6)
        .unwrap();
    assert_eq!(
        d6q.as_slice::<f32>(),
        &[0.0, 1.0, 2.0, 3.0],
        "rank 6 ÷3 → sliced"
    );

    // sc-9099: `Chunk` ALWAYS slices (vs `ChunkIfDivisible`'s fall-through), so a non-divisible
    // rows count must be rejected instead of silently truncating the fused qkv delta.
    let err = LoraRowSlice::Chunk { n: 3, index: 0 }
        .apply(&d4)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not divisible"), "{err}");

    // qkv-mlp up `dims` (FLUX.1 `linear1`): q = rows[0:3072], mlp = rows[9216:21504].
    let dims = vec![3072, 3072, 3072, 12288];
    let total: i32 = dims.iter().sum();
    let big = Array::from_slice(
        &(0..total).map(|i| i as f32).collect::<Vec<_>>(),
        &[total, 1],
    );
    let q = LoraRowSlice::Dims {
        dims: dims.clone(),
        index: 0,
    }
    .apply(&big)
    .unwrap();
    assert_eq!(q.shape(), &[3072, 1]);
    assert_eq!(q.as_slice::<f32>()[0], 0.0);
    let mlp = LoraRowSlice::Dims {
        dims: dims.clone(),
        index: 3,
    }
    .apply(&big)
    .unwrap();
    assert_eq!(mlp.shape(), &[12288, 1]);
    assert_eq!(mlp.as_slice::<f32>()[0], 9216.0);
}

/// A host with three separate per-head linears at `blk.attn.to_{q,k,v}` (`[inner,in]` each).
fn three_qkv_host(inner: i32, inp: i32) -> MultiHost {
    let zeros = || Array::from_slice(&vec![0.0f32; (inner * inp) as usize], &[inner, inp]);
    MultiHost::new(&[
        ("blk.attn.to_q", zeros()),
        ("blk.attn.to_k", zeros()),
        ("blk.attn.to_v", zeros()),
    ])
}

/// The sc-2743 gate at the core level: a BFL *fused* qkv LoRA, split via [`apply_lora_bfl`],
/// installs the BYTE-IDENTICAL adapter at each of `to_q/to_k/to_v` as the equivalent *diffusers
/// split-target* LoRA (the fork-verified PEFT path). The fused up `[3·inner, r]` is row-sliced into
/// per-head `[inner, r]`; the down `[r, in]` (rank not ÷3) is shared. No model weights needed.
#[test]
fn bfl_fused_qkv_equals_diffusers_split() {
    let tmp = tempfile::tempdir().unwrap();
    let (inner, inp, r) = (4i32, 3i32, 2i32);
    // Per-head up factors, then the fused up = their dim-0 concat (row-major, so flat concat).
    let bq: Vec<f32> = (0..inner * r)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let bk: Vec<f32> = (0..inner * r)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.02)
        .collect();
    let bv: Vec<f32> = (0..inner * r)
        .map(|i| ((i % 3) as f32 - 1.0) * 0.03)
        .collect();
    let mut fused = Vec::new();
    fused.extend_from_slice(&bq);
    fused.extend_from_slice(&bk);
    fused.extend_from_slice(&bv);
    let b_fused = Array::from_slice(&fused, &[3 * inner, r]);
    let b_q = Array::from_slice(&bq, &[inner, r]);
    let b_k = Array::from_slice(&bk, &[inner, r]);
    let b_v = Array::from_slice(&bv, &[inner, r]);
    // Shared down [r, in] (rank 2 not ÷3 → shared across q/k/v) + alpha ≠ rank.
    let a = Array::from_slice(
        &(0..r * inp)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
            .collect::<Vec<_>>(),
        &[r, inp],
    );
    let alpha = Array::from_slice(&[4.0f32], &[1]);

    // BFL file: one fused qkv linear (kohya `lora_unet_` spelling).
    let up_key = "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight";
    let down_key = "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight";
    let alpha_key = "lora_unet_double_blocks_0_img_attn_qkv.alpha";
    let bfl_path = scratch_file(&tmp, "bfl_qkv.safetensors");
    Array::save_safetensors(
        vec![(up_key, &b_fused), (down_key, &a), (alpha_key, &alpha)],
        None as Option<&HashMap<String, String>>,
        &bfl_path,
    )
    .unwrap();
    let wb = Weights::from_file(&bfl_path).unwrap();

    let mk = |idx: i32, tgt: &str| BflTarget {
        target_path: tgt.to_string(),
        up_keys: vec![up_key.to_string()],
        down_keys: vec![down_key.to_string()],
        alpha_keys: vec![alpha_key.to_string()],
        up_slice: Some(LoraRowSlice::Chunk { n: 3, index: idx }),
        down_slice: Some(LoraRowSlice::ChunkIfDivisible { n: 3, index: idx }),
    };
    let targets = vec![
        mk(0, "blk.attn.to_q"),
        mk(1, "blk.attn.to_k"),
        mk(2, "blk.attn.to_v"),
    ];

    let mut host_bfl = three_qkv_host(inner, inp);
    let rep = apply_lora_bfl(&mut host_bfl, &wb, 0.7, &targets).unwrap();
    assert_eq!(rep.applied, 3, "all three split targets installed");
    assert!(rep.unmatched_paths.is_empty());

    // Equivalent diffusers split-target file: per-head up, SHARED down, same alpha.
    let peft_path = scratch_file(&tmp, "bfl_split_peft.safetensors");
    Array::save_safetensors(
        vec![
            ("transformer.blk.attn.to_q.lora_B.weight", &b_q),
            ("transformer.blk.attn.to_q.lora_A.weight", &a),
            ("transformer.blk.attn.to_q.alpha", &alpha),
            ("transformer.blk.attn.to_k.lora_B.weight", &b_k),
            ("transformer.blk.attn.to_k.lora_A.weight", &a),
            ("transformer.blk.attn.to_k.alpha", &alpha),
            ("transformer.blk.attn.to_v.lora_B.weight", &b_v),
            ("transformer.blk.attn.to_v.lora_A.weight", &a),
            ("transformer.blk.attn.to_v.alpha", &alpha),
        ],
        None as Option<&HashMap<String, String>>,
        &peft_path,
    )
    .unwrap();
    let wp = Weights::from_file(&peft_path).unwrap();
    let mut host_peft = three_qkv_host(inner, inp);
    apply_lora_peft(&mut host_peft, &wp, 0.7, Some("transformer.")).unwrap();

    for p in ["blk.attn.to_q", "blk.attn.to_k", "blk.attn.to_v"] {
        let pull = |h: &MultiHost| match h.mods.get(p).unwrap().adapters() {
            [Adapter::Lora { a, b, scale }] => (a.clone(), b.clone(), *scale),
            _ => panic!("expected one LoRA at {p}"),
        };
        let (ba, bb, bs) = pull(&host_bfl);
        let (pa, pb, ps) = pull(&host_peft);
        assert_eq!(bs, ps, "scale differs at {p}");
        assert!(
            array_eq(&ba, &pa, false).unwrap().item::<bool>()
                && array_eq(&bb, &pb, false).unwrap().item::<bool>(),
            "BFL split and diffusers split installed different adapters at {p}"
        );
    }
}

/// `is_bfl` detects a BFL file; an off-surface adapter-shaped key is surfaced (not dropped) while a
/// bundled text-encoder key is ignored; and a scale-0 BFL adapter is a bit-exact no-op.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn bfl_detection_unmatched_and_scale_zero() {
    let tmp = tempfile::tempdir().unwrap();
    let up = Array::from_slice(
        &(0..8).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
        &[4, 2],
    );
    let down = Array::from_slice(&[0.1f32, 0.2, -0.1, -0.2, 0.3, -0.3], &[2, 3]);
    let targets = vec![BflTarget {
        target_path: "blk.attn.to_out".to_string(),
        up_keys: vec!["lora_unet_double_blocks_0_img_attn_proj.lora_up.weight".to_string()],
        down_keys: vec!["lora_unet_double_blocks_0_img_attn_proj.lora_down.weight".to_string()],
        alpha_keys: vec![],
        up_slice: None,
        down_slice: None,
    }];

    let path = scratch_file(&tmp, "bfl_detect.safetensors");
    Array::save_safetensors(
        vec![
            (
                "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight",
                &up,
            ),
            (
                "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
                &down,
            ),
            // off-surface BFL key (no target) → surfaced, not silently dropped.
            (
                "lora_unet_double_blocks_9_img_attn_proj.lora_up.weight",
                &up,
            ),
            (
                "lora_unet_double_blocks_9_img_attn_proj.lora_down.weight",
                &down,
            ),
            // bundled text-encoder key → ignored (out of denoiser namespace).
            ("lora_te_text_model_layer_0.lora_up.weight", &up),
        ],
        None as Option<&HashMap<String, String>>,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();
    assert!(is_bfl(&w, &targets), "a BFL source key marks the file BFL");

    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let mut host = MultiHost::new(&[(
        "blk.attn.to_out",
        Array::from_slice(
            &(0..12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            &[4, 3],
        ),
    )]);
    let base = host
        .mods
        .get("blk.attn.to_out")
        .unwrap()
        .forward(&x)
        .unwrap();

    // scale 0 → bit-exact no-op; the off-surface block-9 key is surfaced, the TE key ignored.
    let rep = apply_lora_bfl(&mut host, &w, 0.0, &targets).unwrap();
    assert_eq!(rep.applied, 1, "the on-surface proj target installed");
    // Both block-9 keys (up + down) are surfaced (sorted: down < up); the `lora_te_` key ignored.
    assert_eq!(
        rep.unmatched_paths,
        vec![
            "lora_unet_double_blocks_9_img_attn_proj.lora_down.weight".to_string(),
            "lora_unet_double_blocks_9_img_attn_proj.lora_up.weight".to_string(),
        ],
        "off-surface BFL keys surfaced; TE key ignored"
    );
    let out = host
        .mods
        .get("blk.attn.to_out")
        .unwrap()
        .forward(&x)
        .unwrap();
    assert!(
        array_eq(&out, &base, false).unwrap().item::<bool>(),
        "scale-0 BFL adapter must be a bit-exact no-op"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::bfl_detection_unmatched_and_scale_zero");
}

/// sc-3642: a third-party (non-peft / lycoris) LoKr reconstructs the SAME per-module delta the
/// `lycoris` library produces. Fixtures (real lycoris adapters + ground-truth deltas) come from
/// `tools/sc3642_lokr_reference.py` via `~/mlx-flux-venv` — the on-device A/B. Covers the four
/// shapes: full-w1 + decomposed-w2, both-decomposed, both-full (scale forced to 1), and conv
/// `lokr_t2` tucker.
#[test]
fn thirdparty_lokr_matches_lycoris_reference() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3642_lokr");
    for name in [
        "linear_w1full_w2lr",
        "linear_bothlr",
        "linear_bothfull",
        "conv_tucker",
    ] {
        let w = Weights::from_file(dir.join(format!("{name}.safetensors"))).unwrap();
        let exp = Weights::from_file(dir.join(format!("{name}.expected.safetensors"))).unwrap();
        assert!(is_lokr_keys(&w), "{name}: not detected as LoKr by keys");
        assert!(
            !is_lokr(&w),
            "{name}: a third-party file has no networkType metadata"
        );

        // Reconstruct the flattened→dotted table from the expected (dotted) module paths.
        let table: BTreeMap<String, String> = exp
            .keys()
            .map(|d| (d.replace('.', "_"), d.to_string()))
            .collect();
        let groups = parse_lokr_thirdparty(&w).unwrap();
        assert!(!groups.is_empty(), "{name}: parsed no LoKr modules");
        for (raw, g) in &groups {
            let dotted = resolve_lokr_path(raw, &table)
                .unwrap_or_else(|| panic!("{name}: cannot resolve raw key {raw:?}"));
            let want = exp.require(dotted).unwrap();
            // Reconstruct in f32 (lycoris computes f32) and compare to the ground truth.
            let got = g.delta(want.shape(), Dtype::Float32).unwrap();
            assert_eq!(
                got.shape(),
                want.shape(),
                "{name}/{dotted}: reconstructed delta shape mismatch"
            );
            assert!(
                all_close(&got, want, 1e-4, 1e-5, false)
                    .unwrap()
                    .item::<bool>(),
                "{name}/{dotted}: reconstructed LoKr delta diverged from the lycoris reference"
            );
        }
    }
}

/// sc-3642: the third-party LoKr installs through the autoprefix dispatch even when the caller
/// labels the spec `AdapterKind::Lora` (a third-party file carries no `networkType` to tell the
/// caller otherwise) — detection-by-keys routes it to `apply_lokr_thirdparty`, resolving the
/// `lycoris_`-prefixed flattened key to the host's dotted module.
#[test]
fn thirdparty_lokr_routes_and_installs_via_autoprefix() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3642_lokr");
    let exp = Weights::from_file(dir.join("linear_w1full_w2lr.expected.safetensors")).unwrap();
    // Host with the one Linear the fixture targets (dotted path "proj"), sized to the delta.
    let delta_shape = exp.require("proj").unwrap().shape().to_vec();
    let base = Array::zeros::<f32>(&delta_shape).unwrap();
    let mut host = MultiHost::new(&[("proj", base)]);
    let spec = AdapterSpec::new(
        dir.join("linear_w1full_w2lr.safetensors"),
        1.0,
        AdapterKind::Lora, // deliberately mislabeled — detection must override
    );
    let report = apply_adapter_specs_autoprefix(&mut host, std::slice::from_ref(&spec)).unwrap();
    assert_eq!(report.applied, 1, "third-party LoKr was not installed");
    assert!(
        report.unmatched_paths.is_empty(),
        "unexpected unmatched: {:?}",
        report.unmatched_paths
    );
}

/// sc-3643: a third-party (non-peft / lycoris) LoHa reconstructs the SAME per-module delta the
/// `lycoris` library produces. Fixtures from `tools/sc3643_loha_reference.py` via `~/mlx-flux-venv`.
/// Covers linear, conv (kernel folded into the factors), and conv `hada_t1/t2` tucker.
#[test]
fn thirdparty_loha_matches_lycoris_reference() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3643_loha");
    for name in ["linear", "conv_notucker", "conv_tucker"] {
        let w = Weights::from_file(dir.join(format!("{name}.safetensors"))).unwrap();
        let exp = Weights::from_file(dir.join(format!("{name}.expected.safetensors"))).unwrap();
        assert!(is_loha_keys(&w), "{name}: not detected as LoHa by keys");
        assert!(!is_lokr_keys(&w), "{name}: must not look like LoKr");
        assert!(
            !is_lokr(&w),
            "{name}: a third-party file has no networkType metadata"
        );

        let table: BTreeMap<String, String> = exp
            .keys()
            .map(|d| (d.replace('.', "_"), d.to_string()))
            .collect();
        let groups = parse_loha_thirdparty(&w).unwrap();
        assert!(!groups.is_empty(), "{name}: parsed no LoHa modules");
        for (raw, g) in &groups {
            let dotted = resolve_lokr_path(raw, &table)
                .unwrap_or_else(|| panic!("{name}: cannot resolve raw key {raw:?}"));
            let want = exp.require(dotted).unwrap();
            let got = g.delta(want.shape(), Dtype::Float32).unwrap();
            assert_eq!(
                got.shape(),
                want.shape(),
                "{name}/{dotted}: reconstructed delta shape mismatch"
            );
            assert!(
                all_close(&got, want, 1e-4, 1e-5, false)
                    .unwrap()
                    .item::<bool>(),
                "{name}/{dotted}: reconstructed LoHa delta diverged from the lycoris reference"
            );
        }
    }
}

/// sc-3643: a third-party LoHa installs through the autoprefix dispatch even when the caller
/// labels the spec `AdapterKind::Lora` — detection-by-keys routes it to `apply_loha_thirdparty`.
#[test]
fn thirdparty_loha_routes_and_installs_via_autoprefix() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3643_loha");
    let exp = Weights::from_file(dir.join("linear.expected.safetensors")).unwrap();
    let delta_shape = exp.require("proj").unwrap().shape().to_vec();
    let base = Array::zeros::<f32>(&delta_shape).unwrap();
    let mut host = MultiHost::new(&[("proj", base)]);
    let spec = AdapterSpec::new(dir.join("linear.safetensors"), 1.0, AdapterKind::Lora);
    let report = apply_adapter_specs_autoprefix(&mut host, std::slice::from_ref(&spec)).unwrap();
    assert_eq!(report.applied, 1, "third-party LoHa was not installed");
    assert!(
        report.unmatched_paths.is_empty(),
        "unexpected unmatched: {:?}",
        report.unmatched_paths
    );
}

/// sc-9099: the fixed-prefix `apply_adapter_specs` must route a keys-only third-party LoKr
/// declared `AdapterKind::Lokr` to `apply_lokr_thirdparty` — the peft applier's global-metadata
/// parse ignores the per-module `.alpha` (and a tucker `lokr_t2`), so it would install the
/// delta at the wrong scale while reporting success.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn fixed_prefix_lokr_kind_routes_keys_only_file_to_thirdparty() {
    let tmp = tempfile::tempdir().unwrap();
    // Keys-only LoKr (no `networkType` stamp): w1 full [2,2], w2 decomposed [2,r=2]·[2,2], and
    // a per-module alpha=1 over rank 2 ⇒ lycoris scale 0.5 — discriminating, because the peft
    // parse drops the `.alpha` tensor and defaults alpha=rank ⇒ scale 1.
    let base = Array::from_slice(
        &(0..16).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
        &[4, 4],
    );
    let w1 = Array::from_slice(&[0.5f32, 0.6, 0.7, 0.8], &[2, 2]);
    let w2_a = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
    let w2_b = Array::from_slice(&[0.4f32, 0.3, 0.2, 0.1], &[2, 2]);
    let alpha = Array::from_slice(&[1.0f32], &[1]);
    let path = scratch_file(&tmp, "f012_keysonly_lokr.safetensors");
    Array::save_safetensors(
        vec![
            ("lin.lokr_w1", &w1),
            ("lin.lokr_w2_a", &w2_a),
            ("lin.lokr_w2_b", &w2_b),
            ("lin.alpha", &alpha),
        ],
        None as Option<&HashMap<String, String>>,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();
    assert!(
        !is_lokr(&w) && is_lokr_keys(&w),
        "premise: a keys-only third-party file"
    );

    // Reference: the third-party applier called directly (per-module alpha honored).
    let mut want = OneLinear {
        lin: AdaptableLinear::dense(base.clone(), None),
    };
    apply_lokr_thirdparty(&mut want, &w, 1.0).unwrap();

    let mut got = OneLinear {
        lin: AdaptableLinear::dense(base.clone(), None),
    };
    let report = apply_adapter_specs(
        &mut got,
        &[AdapterSpec::new(path, 1.0, AdapterKind::Lokr)],
        None,
    )
    .unwrap();
    assert_eq!(report.applied, 1);
    assert!(report.unmatched_paths.is_empty());

    let x = Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25], &[1, 4]);
    let got_y = got.lin.forward(&x).unwrap();
    let want_y = want.lin.forward(&x).unwrap();
    assert!(
        array_eq(&got_y, &want_y, false).unwrap().item::<bool>(),
        "fixed-prefix Lokr dispatch must install the identical third-party delta"
    );
    // The pre-sc-9099 route: the peft applier drops the per-module alpha (scale 1, not 0.5).
    let mut peft = OneLinear {
        lin: AdaptableLinear::dense(base, None),
    };
    apply_lokr(&mut peft, &w, 1.0).unwrap();
    let peft_y = peft.lin.forward(&x).unwrap();
    assert!(
        !all_close(&got_y, &peft_y, 1e-4, 1e-4, false)
            .unwrap()
            .item::<bool>(),
        "the peft-path result must differ (it mis-scales) for this test to discriminate"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::loader::tests::fixed_prefix_lokr_kind_routes_keys_only_file_to_thirdparty");
}

// ---- sc-9099 format classification -----------------------------------------------------------

/// Save a tiny safetensors of `keys` (dummy `[1,1]` tensors — classification reads only
/// keys/metadata) and load it back as `Weights`.
fn classify_weights(
    tmp: &tempfile::TempDir,
    name: &str,
    keys: &[&str],
    meta: Option<&HashMap<String, String>>,
) -> Weights {
    let dummy = Array::from_slice(&[1.0f32], &[1, 1]);
    let entries: Vec<(&str, &Array)> = keys.iter().map(|k| (*k, &dummy)).collect();
    let path = scratch_file(tmp, name);
    Array::save_safetensors(entries, meta, &path).unwrap();
    Weights::from_file(&path).unwrap()
}

/// sc-9099: `classify_adapter_format` is the single routing truth — one minimal key-set per
/// [`AdapterFormat`] variant, plus the precedence corners (LyCORIS keys beat kohya's shared
/// `lora_unet_` prefix; BFL naming beats kohya; the `networkType=lokr` stamp gates `lokr_*`
/// keys; a BFL-named LyCORIS file on a host without a BFL surface stays third-party).
#[test]
fn classify_adapter_format_per_format() {
    let tmp = tempfile::tempdir().unwrap();
    let lokr_meta: HashMap<String, String> = [
        ("networkType".to_string(), "lokr".to_string()),
        ("alpha".to_string(), "1.0".to_string()),
        ("rank".to_string(), "1".to_string()),
    ]
    .into_iter()
    .collect();
    // A host without a BFL surface (the common case) and the FLUX-shaped `BflHost` whose
    // `bfl_targets()` names `diffusion_model.double_blocks.…`. Tables are per-host.
    let plain = MultiHost::new(&[(
        "blocks.0.attn.to_q",
        Array::from_slice(&[0.0f32; 4], &[2, 2]),
    )]);
    let flux = BflHost::new();
    let mut tp = HostTables::default();
    let mut tf = HostTables::default();
    let bfl_qkv = "diffusion_model.double_blocks.0.img_attn.qkv";

    // Third-party LoKr: `lokr_*` keys, no stamp — regardless of the declared kind.
    let w = classify_weights(
        &tmp,
        "cls_tp_lokr.safetensors",
        &["lin.lokr_w1", "lin.lokr_w2"],
        None,
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
        AdapterFormat::ThirdpartyLokr
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lokr, &plain, &mut tp),
        AdapterFormat::ThirdpartyLokr
    );

    // Precedence: a kohya-flattened LoKr also carries `lora_unet_`, but LyCORIS keys win
    // (is_kohya would claim it and apply nothing).
    let w = classify_weights(
        &tmp,
        "cls_tp_lokr_kohya.safetensors",
        &[
            "lora_unet_blocks_0_attn_to_q.lokr_w1",
            "lora_unet_blocks_0_attn_to_q.lokr_w2",
        ],
        None,
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
        AdapterFormat::ThirdpartyLokr
    );

    // Third-party LoKr in BFL/ComfyUI fused naming — only on a host with a BFL surface.
    let w = classify_weights(
        &tmp,
        "cls_tp_lokr_bfl.safetensors",
        &[&format!("{bfl_qkv}.lokr_w1"), &format!("{bfl_qkv}.lokr_w2")],
        None,
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lokr, &flux, &mut tf),
        AdapterFormat::ThirdpartyLokrBfl
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lokr, &plain, &mut tp),
        AdapterFormat::ThirdpartyLokr,
        "no BFL surface ⇒ the same file stays on the plain third-party path"
    );

    // Third-party LoHa (`hada_*` keys) + its BFL twin.
    let w = classify_weights(
        &tmp,
        "cls_tp_loha.safetensors",
        &[
            "lin.hada_w1_a",
            "lin.hada_w1_b",
            "lin.hada_w2_a",
            "lin.hada_w2_b",
        ],
        None,
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
        AdapterFormat::ThirdpartyLoha
    );
    let w = classify_weights(
        &tmp,
        "cls_tp_loha_bfl.safetensors",
        &[
            &format!("{bfl_qkv}.hada_w1_a"),
            &format!("{bfl_qkv}.hada_w1_b"),
        ],
        None,
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lora, &flux, &mut tf),
        AdapterFormat::ThirdpartyLohaBfl
    );

    // BFL fused→split LoRA — detected before kohya despite sharing no `lora_unet_` here
    // (the `diffusion_model.` spelling), and before the PEFT fallback.
    let w = classify_weights(
        &tmp,
        "cls_bfl_lora.safetensors",
        &[
            &format!("{bfl_qkv}.lora_up.weight"),
            &format!("{bfl_qkv}.lora_down.weight"),
        ],
        None,
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lora, &flux, &mut tf),
        AdapterFormat::BflLora
    );

    // kohya-flattened LoRA.
    let w = classify_weights(
        &tmp,
        "cls_kohya.safetensors",
        &[
            "lora_unet_blocks_0_attn_to_q.lora_down.weight",
            "lora_unet_blocks_0_attn_to_q.lora_up.weight",
        ],
        None,
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
        AdapterFormat::KohyaLora
    );

    // peft LoKr: the `networkType=lokr` stamp gates the `lokr_*` keys off the third-party path.
    let w = classify_weights(
        &tmp,
        "cls_peft_lokr.safetensors",
        &["lin.lokr_w1", "lin.lokr_w2"],
        Some(&lokr_meta),
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lokr, &plain, &mut tp),
        AdapterFormat::PeftLokr
    );

    // peft LoKr in BFL/ComfyUI fused naming (sc-8345).
    let w = classify_weights(
        &tmp,
        "cls_peft_lokr_bfl.safetensors",
        &[&format!("{bfl_qkv}.lokr_w1"), &format!("{bfl_qkv}.lokr_w2")],
        Some(&lokr_meta),
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lokr, &flux, &mut tf),
        AdapterFormat::PeftLokrBfl
    );

    // PEFT/diffusers LoRA — the common fallback.
    let w = classify_weights(
        &tmp,
        "cls_peft_lora.safetensors",
        &[
            "transformer.lin.lora_A.weight",
            "transformer.lin.lora_B.weight",
        ],
        None,
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
        AdapterFormat::PeftLora
    );

    // Declared `Lora` against `networkType=lokr` metadata — the caller error.
    let w = classify_weights(
        &tmp,
        "cls_mismatch.safetensors",
        &["lin.lokr_w1", "lin.lokr_w2"],
        Some(&lokr_meta),
    );
    assert_eq!(
        classify_adapter_format(&w, AdapterKind::Lora, &plain, &mut tp),
        AdapterFormat::LoraKindMismatch
    );
}

// ---- Packed (Q4/Q8) base: structured LoKr, never a materialized delta (sc-10578) -------------

/// `[out, in] = [64, 64]` — `in` is exactly one group at the default `group_size = 64`, so the
/// base quantizes cleanly and `base_shape()` recovers `[64, 64]` off the `[64, 1]` scales grid.
const PK_OUT: i32 = 64;
const PK_IN: i32 = 64;

/// A deterministic base weight whose entries are multiples of `1/16` — exact in bf16, so the
/// `PARITY-BF16` downcast inside `quantize()` introduces no rounding of its own.
fn pk_base_weight() -> Array {
    let v: Vec<f32> = (0..(PK_OUT * PK_IN) as usize)
        .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
        .collect();
    Array::from_slice(&v, &[PK_OUT, PK_IN])
}

/// A peft LoKr over `lin`: `w1 [8,8] ⊗ w2 [8,8] → ΔW [64,64]`, `alpha/rank = 8/4 = 2.0`.
/// Factor entries are ±powers of two so `kron` and both scale multiplies stay exact in bf16 —
/// any residual mismatch is then a real disagreement, not float noise.
fn pk_lokr_file(tmp: &tempfile::TempDir, name: &str) -> Weights {
    let w1: Vec<f32> = (0..64)
        .map(|i| if i % 2 == 0 { 0.5 } else { -0.25 })
        .collect();
    let w2: Vec<f32> = (0..64)
        .map(|i| if i % 3 == 0 { 0.25 } else { -0.5 })
        .collect();
    let mut meta = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("rank".to_string(), "4".to_string());
    meta.insert("alpha".to_string(), "8.0".to_string());
    let path = scratch_file(tmp, name);
    Array::save_safetensors(
        vec![
            ("lin.lokr_w1", &Array::from_slice(&w1, &[8, 8])),
            ("lin.lokr_w2", &Array::from_slice(&w2, &[8, 8])),
        ],
        Some(&meta),
        &path,
    )
    .unwrap();
    Weights::from_file(&path).unwrap()
}

/// `ΔW = kron(w1, w2)` at the full `alpha/rank · strength` scale, applied as `x·ΔWᵀ` in f32 —
/// an INDEPENDENT reference, computed from the LoKr definition rather than by re-running either
/// install path. Both the dense (materialized) and packed (structured) residuals must match it.
fn pk_reference_residual(w: &Weights, x: &Array, full_scale: f32) -> Array {
    use mlx_rs::ops::{kron, matmul, multiply};
    let w1 = w
        .require("lin.lokr_w1")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let w2 = w
        .require("lin.lokr_w2")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let delta = multiply(kron(&w1, &w2).unwrap(), Array::from_f32(full_scale)).unwrap();
    matmul(x, delta.t()).unwrap()
}

fn pk_x() -> Array {
    let v: Vec<f32> = (0..(2 * PK_IN) as usize)
        .map(|i| ((i % 9) as f32 - 4.0) / 8.0)
        .collect();
    Array::from_slice(&v, &[2, PK_IN])
}

/// The load-bearing test. On a packed base a LoKr must install as the STRUCTURED Kronecker form
/// (no `[out,in]` delta ever allocated) and must reproduce the same residual as the dense,
/// materialized path. `alpha/rank = 2.0` and `strength = 2.5` are both ≠ 1 and differ from each
/// other, so a scale threaded through the wrong path cannot coincidentally pass.
#[test]
fn lokr_on_packed_base_installs_structured_and_matches_dense() {
    let tmp = tempfile::tempdir().unwrap();
    let w = pk_lokr_file(&tmp, "sc10578_packed_lokr.safetensors");
    let x = pk_x();
    let strength = 2.5f32;
    let reference = pk_reference_residual(&w, &x, 2.0 * strength);

    // Dense base → materialized `Adapter::Lokr` (unchanged pre-sc-10578 behavior).
    let mut dense = OneLinear {
        lin: AdaptableLinear::dense(pk_base_weight(), None),
    };
    apply_lokr(&mut dense, &w, strength).unwrap();
    let dense_adapter = &dense.lin.adapters()[0];
    assert!(
        matches!(dense_adapter, Adapter::Lokr { .. }),
        "a DENSE base must keep the materialized delta — the other families' fork-parity \
             goldens depend on it"
    );
    let dense_residual = dense_adapter.residual(&x).unwrap();

    // Packed base → structured `Adapter::LokrStructured`.
    for bits in [4, 8] {
        let mut packed = OneLinear {
            lin: AdaptableLinear::dense(pk_base_weight(), None),
        };
        packed.lin.quantize(bits, Some(64)).unwrap();
        assert!(packed.lin.is_quantized());
        assert_eq!(packed.lin.base_shape(), vec![PK_OUT, PK_IN]);

        apply_lokr(&mut packed, &w, strength).unwrap();
        let adapter = &packed.lin.adapters()[0];
        let factors = match adapter {
            Adapter::LokrStructured { factors } => factors,
            Adapter::Lokr { .. } => panic!(
                "q{bits}: a packed base installed a MATERIALIZED delta — this is the memory \
                     regression sc-10578 exists to prevent"
            ),
            _ => panic!("q{bits}: unexpected adapter variant"),
        };
        // The whole point: only the small factors are held, never `[out, in]`.
        assert_eq!(factors.w1.shape(), &[8, 8]);
        assert_eq!(factors.w2.shape(), &[8, 8]);
        assert!(
            (factors.w1.size() + factors.w2.size()) * 16 < (PK_OUT * PK_IN) as usize,
            "q{bits}: structured factors must be far smaller than the [out,in] delta"
        );

        let structured_residual = adapter.residual(&x).unwrap();
        assert!(
            all_close(&structured_residual, &reference, 1e-3, 1e-4, None)
                .unwrap()
                .item::<bool>(),
            "q{bits}: structured residual must match the independent kron reference"
        );
        assert!(
            all_close(&structured_residual, &dense_residual, 1e-3, 1e-4, None)
                .unwrap()
                .item::<bool>(),
            "q{bits}: structured and materialized residuals must agree"
        );
    }
}

/// MUTATION CHECK for the test above. `LokrFile::factors` bakes the FULL `alpha/rank · strength`
/// into `w2` because the structured residual carries no separate scale, whereas `delta` bakes only
/// `alpha/rank` and leaves `strength` to `Adapter::Lokr::scale`. Dropping `strength` here is the
/// exact silent mis-scale that distinction invites — confirm the assertion above would catch it.
#[test]
fn structured_lokr_residual_is_sensitive_to_the_strength_factor() {
    let tmp = tempfile::tempdir().unwrap();
    let w = pk_lokr_file(&tmp, "sc10578_packed_lokr_mut.safetensors");
    let x = pk_x();
    let strength = 2.5f32;
    let reference = pk_reference_residual(&w, &x, 2.0 * strength);

    let file = parse_lokr(&w).unwrap();
    let group = file.groups.get("lin").unwrap();

    // Correct: full scale = (alpha/rank) · strength.
    let good = file
        .factors(group, strength, &[PK_OUT, PK_IN], Dtype::Bfloat16)
        .unwrap()
        .expect("a linear LoKr is deferrable");
    assert!(
        all_close(good.residual(&x).unwrap(), &reference, 1e-3, 1e-4, None)
            .unwrap()
            .item::<bool>()
    );

    // Mutated: `strength` forgotten (scale = alpha/rank only). Must FAIL the same assertion —
    // otherwise the test above is incapable of catching a dropped strength.
    let bad = file
        .factors(group, 1.0, &[PK_OUT, PK_IN], Dtype::Bfloat16)
        .unwrap()
        .expect("a linear LoKr is deferrable");
    assert!(
        !all_close(bad.residual(&x).unwrap(), &reference, 1e-3, 1e-4, None)
            .unwrap()
            .item::<bool>(),
        "dropping `strength` must change the residual — otherwise the scale assertions are vacuous"
    );
}

/// LoHa has no deferred form at all (Hadamard, not Kronecker), so on a packed base it falls back to
/// the materialized `[out,in]` delta — the pre-sc-10578 behavior, preserved deliberately. Erroring
/// instead (my first cut) would break every third-party LoHa on the DEFAULT tier of every MLX model:
/// adapters install AFTER `.quantize()`, and the worker defaults `spec.quantize` to `Some(Q8)`. The
/// fallback is correct — `Adapter::Lokr`'s residual never reads the base — just memory-hungry.
/// Whether a packed base should refuse a multi-GB delta is sc-10678.
#[test]
fn loha_on_packed_base_falls_back_to_materialization() {
    let tmp = tempfile::tempdir().unwrap();
    let r = 4;
    let a: Vec<f32> = (0..(PK_OUT * r) as usize)
        .map(|i| (i % 5) as f32 * 0.1)
        .collect();
    let b: Vec<f32> = (0..(r * PK_IN) as usize)
        .map(|i| (i % 7) as f32 * 0.1)
        .collect();
    let (wa, wb) = (
        Array::from_slice(&a, &[PK_OUT, r]),
        Array::from_slice(&b, &[r, PK_IN]),
    );
    let path = scratch_file(&tmp, "sc10578_packed_loha.safetensors");
    Array::save_safetensors(
        vec![
            ("lin.hada_w1_a", &wa),
            ("lin.hada_w1_b", &wb),
            ("lin.hada_w2_a", &wa),
            ("lin.hada_w2_b", &wb),
        ],
        None,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();
    assert!(is_loha_keys(&w));

    // Dense base: still works, materialized as before.
    let mut dense = OneLinear {
        lin: AdaptableLinear::dense(pk_base_weight(), None),
    };
    assert_eq!(
        apply_loha_thirdparty(&mut dense, &w, 1.0).unwrap().applied,
        1
    );

    // Packed base: falls back to the materialized delta — installed, NOT rejected.
    let mut packed = OneLinear {
        lin: AdaptableLinear::dense(pk_base_weight(), None),
    };
    packed.lin.quantize(8, Some(64)).unwrap();
    assert_eq!(
        apply_loha_thirdparty(&mut packed, &w, 1.0).unwrap().applied,
        1,
        "a LoHa must still install on a packed base — erroring here would break every \
             third-party LoHa on the default tier of every MLX model (adapters install AFTER \
             .quantize(), and the worker defaults spec.quantize to Some(Q8))"
    );
    match packed.lin.adapters() {
        [Adapter::Lokr { delta, .. }] => assert_eq!(delta.shape(), &[PK_OUT, PK_IN]),
        _ => panic!("LoHa on a packed base must install the materialized delta"),
    }
    // And it must be CORRECT, not merely present: the residual is base-independent, so the packed
    // and dense installs produce the same one. (Catches a fallback that silently mis-scales.)
    let x = pk_x();
    assert!(
        all_close(
            packed.lin.adapters()[0].residual(&x).unwrap(),
            dense.lin.adapters()[0].residual(&x).unwrap(),
            1e-5,
            1e-6,
            None
        )
        .unwrap()
        .item::<bool>(),
        "the packed fallback residual must equal the dense one"
    );
    assert!(
        packed.lin.is_quantized(),
        "the fallback must not dequantize the base"
    );
}

/// sc-10678 pre-flight guard — the projection is exact (`out·in·2` bytes for the bf16 delta) and the
/// budget check is a pure comparison, so both are deterministically testable with **no** global MLX
/// memory state (the `install_lycoris_groups` wiring reads `get_active_memory()`/`get_memory_limit()`
/// and hands the values to these; the fits-path accumulation is exercised by
/// `loha_on_packed_base_falls_back_to_materialization`, whose tiny delta stays under any real budget).
#[test]
fn projected_delta_bytes_is_out_times_in_times_bf16() {
    assert_eq!(
        projected_delta_bytes(&[2048, 2048], Dtype::Bfloat16),
        2048 * 2048 * 2
    );
    assert_eq!(
        projected_delta_bytes(&[64, 128], Dtype::Bfloat16),
        64 * 128 * 2
    );
    assert_eq!(
        projected_delta_bytes(&[64, 128], Dtype::Float32),
        64 * 128 * 4
    );
    // A whole DiT's worth of non-deferrable deltas (Anima: 448 targets ≈ 2048² each) ≈ 3.75 GiB —
    // the memory-lie that motivated the guard (a ~1 GB q4 DiT would balloon past bf16).
    let whole = 448 * projected_delta_bytes(&[2048, 2048], Dtype::Bfloat16);
    let gib = whole as f64 / (1024.0 * 1024.0 * 1024.0);
    assert!((3.5..4.0).contains(&gib), "448×2048² bf16 ≈ {gib:.2} GiB");
}

#[test]
fn materialization_exceeds_budget_respects_headroom_and_unlimited() {
    let gib = 1024usize * 1024 * 1024;
    // 4 GiB of new deltas on a 1 GiB resident set, 16 GiB limit → 5 GiB < 16·0.85=13.6 → fits.
    assert!(!materialization_exceeds_budget(gib, 4 * gib, 16 * gib));
    // Same deltas against a 4 GiB limit → 5 GiB > 4·0.85=3.4 → refuse.
    assert!(materialization_exceeds_budget(gib, 4 * gib, 4 * gib));
    // The 0.85 headroom bites between 8 and 9 GiB against a 10 GiB limit (safe = 8.5) — dropping the
    // headroom (using the raw limit) would flip the second assert, so this pins it.
    assert!(
        !materialization_exceeds_budget(0, 8 * gib, 10 * gib),
        "8 < 8.5 fits"
    );
    assert!(
        materialization_exceeds_budget(0, 9 * gib, 10 * gib),
        "9 > 8.5 refuses"
    );
    // limit == 0 ⇒ MLX has no cap set (unlimited) ⇒ never guard, even for a huge projection.
    assert!(!materialization_exceeds_budget(gib, 100 * gib, 0));
    // Nothing resident + nothing projected → never exceeds. NOTE the plain-LoRA / all-deferred /
    // dense case (`projected == 0`) is short-circuited in `install_lycoris_groups` BEFORE this is
    // called (`if projected_materialize > 0`), so the guard never fires on the resident set alone —
    // this pure fn, given a resident set already over budget, WOULD return true, which is why the
    // `projected > 0` gate lives at the call site, not here.
    assert!(!materialization_exceeds_budget(0, 0, gib));
}

/// sc-10678 — the over-budget REFUSAL path, end to end through `apply_loha_thirdparty`. `#[ignore]`d
/// because it mutates the PROCESS-GLOBAL MLX memory limit (`set_memory_limit`), which would clobber
/// concurrent allocations in the default multi-threaded runner. Run it ALONE:
/// `cargo test -p mlx-gen --lib loha_on_packed_over_budget_is_refused -- --ignored --exact`.
/// Sets the limit below the resident set so any materialization exceeds it, applies a LoHa to a
/// packed base, and asserts a catchable, ACTIONABLE error (not a mid-run SIGKILL), leaving the base
/// untouched — then restores the limit. The guard fires in pass 1 (shape-only, no allocation), so the
/// tiny limit never spuriously errors before the refusal.
#[test]
#[ignore = "mutates the process-global MLX memory limit; run alone"]
fn loha_on_packed_over_budget_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let r = 4;
    let a: Vec<f32> = (0..(PK_OUT * r) as usize)
        .map(|i| (i % 5) as f32 * 0.1)
        .collect();
    let b: Vec<f32> = (0..(r * PK_IN) as usize)
        .map(|i| (i % 7) as f32 * 0.1)
        .collect();
    let (wa, wb) = (
        Array::from_slice(&a, &[PK_OUT, r]),
        Array::from_slice(&b, &[r, PK_IN]),
    );
    let path = scratch_file(&tmp, "sc10678_over_budget_loha.safetensors");
    Array::save_safetensors(
        vec![
            ("lin.hada_w1_a", &wa),
            ("lin.hada_w1_b", &wb),
            ("lin.hada_w2_a", &wa),
            ("lin.hada_w2_b", &wb),
        ],
        None,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();

    let mut packed = OneLinear {
        lin: AdaptableLinear::dense(pk_base_weight(), None),
    };
    packed.lin.quantize(8, Some(64)).unwrap();

    // Force any materialization over budget: limit = 1 byte ⇒ safe = 0. Save + restore.
    let prev = mlx_rs::memory::set_memory_limit(1);
    let result = apply_loha_thirdparty(&mut packed, &w, 1.0);
    mlx_rs::memory::set_memory_limit(prev);

    let err =
        result.expect_err("a LoHa over budget on a packed base must be refused, not materialized");
    let msg = err.to_string();
    assert!(
            msg.contains("quantized") && msg.contains("plain LoRA/LoKr"),
            "the refusal must be actionable (name the quantized-tier issue + the plain-LoRA escape): {msg}"
        );
    // A refused install leaks no partial adapter, and never dequantizes the base.
    assert!(
        packed.lin.adapters().is_empty(),
        "a refused install must not leave a partial adapter"
    );
    assert!(packed.lin.is_quantized());
}

/// A tucker/CP `lokr_t2` has no 2-D matrix form, so `build_lokr_factors` returns `None` and the
/// packed base falls back to the materialized path — **exactly what a dense base does**. Here that
/// fallback then fails, because a conv-shaped `ΔW` cannot reshape onto a Linear `[64,64]` target
/// (lycoris only emits `lokr_t2` for conv layers). The point of the test is that packed and dense
/// behave IDENTICALLY: sc-10578 introduced no new failure mode for non-deferrable modules.
#[test]
fn tucker_lokr_on_packed_base_matches_dense_behavior() {
    let tmp = tempfile::tempdir().unwrap();
    let t2 = Array::from_slice(&[0.1f32; 2 * 2 * 3 * 3], &[2, 2, 3, 3]);
    let w2a = Array::from_slice(&[0.2f32; 2 * 8], &[2, 8]);
    let w2b = Array::from_slice(&[0.3f32; 2 * 8], &[2, 8]);
    let w1 = Array::from_slice(&[0.4f32; 8 * 8], &[8, 8]);
    let path = scratch_file(&tmp, "sc10578_packed_tucker_lokr.safetensors");
    Array::save_safetensors(
        vec![
            ("lin.lokr_w1", &w1),
            ("lin.lokr_t2", &t2),
            ("lin.lokr_w2_a", &w2a),
            ("lin.lokr_w2_b", &w2b),
        ],
        None,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();

    // Dense base: the materialized path rejects a conv delta on a Linear target.
    let mut dense = OneLinear {
        lin: AdaptableLinear::dense(pk_base_weight(), None),
    };
    let dense_err = apply_lokr_thirdparty(&mut dense, &w, 1.0)
        .expect_err("a conv/tucker delta cannot reshape onto a Linear base")
        .to_string();

    // Packed base: falls back to that same path, so it fails the same way — not with a new,
    // sc-10578-specific rejection.
    let mut packed = OneLinear {
        lin: AdaptableLinear::dense(pk_base_weight(), None),
    };
    packed.lin.quantize(4, Some(64)).unwrap();
    let e = apply_lokr_thirdparty(&mut packed, &w, 1.0)
        .expect_err("a tucker LoKr on a packed base fails via the materialized fallback")
        .to_string();
    assert_eq!(
        e, dense_err,
        "packed must fail identically to dense — sc-10578 must add no new failure mode"
    );
    assert!(
        packed.lin.adapters().is_empty(),
        "nothing installed on failure"
    );
}

/// Write a third-party LoHa file targeting `lin` at `[PK_OUT, PK_IN]` (rank 4).
fn write_loha_lin(tmp: &tempfile::TempDir, name: &str) -> Weights {
    let r = 4;
    let a: Vec<f32> = (0..(PK_OUT * r) as usize)
        .map(|i| (i % 5) as f32 * 0.1)
        .collect();
    let b: Vec<f32> = (0..(r * PK_IN) as usize)
        .map(|i| (i % 7) as f32 * 0.1)
        .collect();
    let (wa, wb) = (
        Array::from_slice(&a, &[PK_OUT, r]),
        Array::from_slice(&b, &[r, PK_IN]),
    );
    let path = scratch_file(tmp, name);
    Array::save_safetensors(
        vec![
            ("lin.hada_w1_a", &wa),
            ("lin.hada_w1_b", &wb),
            ("lin.hada_w2_a", &wa),
            ("lin.hada_w2_b", &wb),
        ],
        None,
        &path,
    )
    .unwrap();
    Weights::from_file(&path).unwrap()
}

#[test]
#[ignore = "mutates the process-global MLX memory limit; run alone"]
fn dense_base_loha_over_budget_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    // sc-11129: a DENSE-base LoHa materializes the same `[out,in]` bf16 delta as a stacked
    // residual, so it must be included in the sc-10678 budget projection — the OOM the packed guard
    // prevents is reachable on the dense tier too. Before the fix a dense group added 0 to the
    // projection and this install went through unchecked.
    let w = write_loha_lin(&tmp, "f011_dense_over_budget_loha.safetensors");
    let mut dense = OneLinear {
        lin: AdaptableLinear::dense(pk_base_weight(), None),
    };
    let prev = mlx_rs::memory::set_memory_limit(1);
    let result = apply_loha_thirdparty(&mut dense, &w, 1.0);
    mlx_rs::memory::set_memory_limit(prev);
    result.expect_err("a dense LoHa over budget must now be refused up front (sc-11129)");
    assert!(
        dense.lin.adapters().is_empty(),
        "a refused install must leave no partial adapter on the dense base"
    );
}

#[test]
fn dense_base_loha_within_budget_still_installs() {
    let tmp = tempfile::tempdir().unwrap();
    // sc-11129 regression: counting the dense materialization must only refuse GENUINELY over-budget
    // runs — a LoHa that fits still installs on the dense base (the default, unlimited-limit path).
    let w = write_loha_lin(&tmp, "f011_dense_ok_loha.safetensors");
    let mut dense = OneLinear {
        lin: AdaptableLinear::dense(pk_base_weight(), None),
    };
    assert_eq!(
        apply_loha_thirdparty(&mut dense, &w, 1.0).unwrap().applied,
        1,
        "a within-budget dense LoHa must still install"
    );
}

#[test]
fn lycoris_pass2_resolution_miss_is_surfaced_not_dropped() {
    let tmp = tempfile::tempdir().unwrap();
    // sc-11129: if a module resolves in pass 1 but vanishes by pass 2 (a lazy/offloaded
    // host, epic 10834), the plan must be SURFACED in `unmatched_paths`, never silently dropped —
    // honoring `install_lycoris_groups`'s contract. Modeled with a host whose `adaptable_mut`
    // returns the linear only on its FIRST call (pass 1) and `None` after (pass 2).
    struct VanishingHost {
        lin: AdaptableLinear,
        calls: usize,
    }
    impl AdaptableHost for VanishingHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            if path != ["lin"] {
                return None;
            }
            self.calls += 1;
            if self.calls == 1 {
                Some(&mut self.lin)
            } else {
                None
            }
        }
    }
    let w = pk_lokr_file(&tmp, "f012_vanishing_lokr.safetensors");
    let mut host = VanishingHost {
        lin: AdaptableLinear::dense(pk_base_weight(), None),
        calls: 0,
    };
    let report = apply_lokr(&mut host, &w, 1.0).unwrap();
    assert_eq!(report.applied, 0, "the vanished target installs nothing");
    assert_eq!(
        report.unmatched_paths,
        vec!["lin".to_string()],
        "a pass-2 resolution miss must be surfaced in unmatched_paths"
    );
}

#[test]
#[allow(clippy::type_complexity)]
fn bfl_lycoris_fused_mismatched_in_dim_errors() {
    // sc-11129: a fused BFL destination set that does not share one input dim must fail with
    // a NAMED validation error (module + dims), not an opaque downstream kron/reshape error. Drive
    // `install_bfl_lycoris` directly with a host whose two fused destinations disagree on in_dim.
    struct MismatchHost {
        q: AdaptableLinear,
        k: AdaptableLinear,
    }
    impl AdaptableHost for MismatchHost {
        fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
            match path {
                ["attn", "to_q"] => Some(&mut self.q),
                ["attn", "to_k"] => Some(&mut self.k),
                _ => None,
            }
        }
        fn adaptable_paths(&self) -> Vec<String> {
            vec!["attn.to_q".to_string(), "attn.to_k".to_string()]
        }
        fn bfl_targets(&self) -> Vec<BflTarget> {
            let up = vec!["diffusion_model.blk.qkv.lora_B.weight".to_string()];
            let down = vec!["diffusion_model.blk.qkv.lora_A.weight".to_string()];
            let alpha = vec!["diffusion_model.blk.qkv.alpha".to_string()];
            ["to_q", "to_k"]
                .iter()
                .enumerate()
                .map(|(idx, dst)| BflTarget {
                    target_path: format!("attn.{dst}"),
                    up_keys: up.clone(),
                    down_keys: down.clone(),
                    alpha_keys: alpha.clone(),
                    up_slice: Some(LoraRowSlice::Chunk {
                        n: 2,
                        index: idx as i32,
                    }),
                    down_slice: None,
                })
                .collect()
        }
    }
    // to_q in=3, to_k in=4 — the fused qkv cannot share one input dim.
    let mut host = MismatchHost {
        q: AdaptableLinear::dense(Array::from_slice(&[0.0f32; 6], &[2, 3]), None),
        k: AdaptableLinear::dense(Array::from_slice(&[0.0f32; 8], &[2, 4]), None),
    };
    let targets = host.bfl_targets();
    let map = bfl_lycoris_module_map(&targets);
    let groups: Vec<(String, fn(&[i32]) -> Result<Array>)> = vec![(
        "diffusion_model.blk.qkv".to_string(),
        // Never reached — the in_dim guard fires in pass 1, before any reconstruction.
        (|_: &[i32]| unreachable!("reconstruct must not run when the in_dim guard fires"))
            as fn(&[i32]) -> Result<Array>,
    )];
    let err = install_bfl_lycoris(&mut host, &map, groups, 1.0)
        .expect_err("mismatched fused in-dims must be a named validation error");
    let msg = err.to_string();
    assert!(
        msg.contains("in_dim") && msg.contains("diffusion_model.blk.qkv"),
        "the error must name the module and the mismatched dims: {msg}"
    );
}

#[test]
#[ignore = "mutates the process-global MLX memory limit; run alone"]
#[allow(clippy::type_complexity)]
fn bfl_lycoris_over_budget_is_refused() {
    // sc-11129: the BFL fused→split path materializes the whole fused delta with no
    // deferred form, so it must apply the sc-10678 memory guard like its plain-path twin — refusing
    // UP FRONT rather than OOMing the worker mid-reconstruct. Drive `install_bfl_lycoris` with a
    // 1-byte memory limit so any projected materialization is over budget.
    let mut host = BflHost::new();
    let targets = host.bfl_targets();
    let map = bfl_lycoris_module_map(&targets);
    let groups: Vec<(String, fn(&[i32]) -> Result<Array>)> = vec![(
        "diffusion_model.double_blocks.0.img_attn.qkv".to_string(),
        // Never reached — the budget guard fires in pass 1, before any reconstruction.
        (|_: &[i32]| unreachable!("reconstruct must not run when the budget guard fires"))
            as fn(&[i32]) -> Result<Array>,
    )];
    let prev = mlx_rs::memory::set_memory_limit(1);
    let result = install_bfl_lycoris(&mut host, &map, groups, 1.0);
    mlx_rs::memory::set_memory_limit(prev);
    let err =
        result.expect_err("a BFL LyCORIS materialization over budget must be refused (sc-11129)");
    assert!(
        err.to_string().contains("budget"),
        "the refusal must be the actionable budget error: {err}"
    );
    // Nothing installed on refusal.
    for dst in ["to_q", "to_k", "to_v"] {
        assert!(
            host.adaptable_mut(&["transformer_blocks", "0", "attn", dst])
                .unwrap()
                .adapters()
                .is_empty(),
            "a refused BFL install must leave no partial adapter on {dst}"
        );
    }
}

#[test]
fn bfl_lora_conflicting_alpha_errors() {
    let tmp = tempfile::tempdir().unwrap();
    // sc-11129: two `.alpha` spellings mapping to one BFL target must be a hard conflict
    // error (mirroring `apply_lora_peft`), not a nondeterministic HashMap-order last-wins scale.
    let a = Array::from_slice(&[0.1f32, 0.2, 0.3], &[1, 3]);
    let b = Array::from_slice(&[0.4f32, 0.5], &[2, 1]);
    let alpha4 = Array::from_slice(&[4.0f32], &[1]);
    let alpha8 = Array::from_slice(&[8.0f32], &[1]);
    let path = scratch_file(&tmp, "f014_bfl_alpha_conflict.safetensors");
    Array::save_safetensors(
        vec![
            ("diffusion_model.m.lora_A.weight", &a),
            ("diffusion_model.m.lora_B.weight", &b),
            ("diffusion_model.m.alpha", &alpha4),
            ("diffusion_model.m2.alpha", &alpha8),
        ],
        None,
        &path,
    )
    .unwrap();
    let w = Weights::from_file(&path).unwrap();

    // One target whose two alpha_keys both resolve to it — the conflict shape.
    let targets = vec![BflTarget {
        target_path: "attn.to_q".to_string(),
        up_keys: vec!["diffusion_model.m.lora_B.weight".to_string()],
        down_keys: vec!["diffusion_model.m.lora_A.weight".to_string()],
        alpha_keys: vec![
            "diffusion_model.m.alpha".to_string(),
            "diffusion_model.m2.alpha".to_string(),
        ],
        up_slice: None,
        down_slice: None,
    }];
    let mut host = MultiHost::new(&[("attn.to_q", Array::from_slice(&[0.0f32; 6], &[2, 3]))]);
    let err = apply_lora_bfl(&mut host, &w, 1.0, &targets)
        .expect_err("conflicting alphas for one BFL target must error");
    assert!(
        err.to_string().contains("alpha conflict"),
        "expected a named alpha-conflict error, got: {err}"
    );
}

// ---- ComfyUI/lightx2v diff-patch fold (sc-13825, MLX parity for candle sc-13726) -------------

/// A host with one biased dense `AdaptableLinear` at the Krea projector path, carrying the
/// `txtfusion`→`text_fusion` alias the real DiT uses — so a `diffusion_model.txtfusion.projector.*`
/// diff-patch key resolves after [`strip_common_lora_prefix`], exactly as on the DiT.
struct DiffHost {
    lin: AdaptableLinear,
}
impl AdaptableHost for DiffHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["text_fusion" | "txtfusion", "projector"] => Some(&mut self.lin),
            _ => None,
        }
    }
    fn adaptable_paths(&self) -> Vec<String> {
        vec!["text_fusion.projector".to_string()]
    }
}

fn save_one(tmp: &tempfile::TempDir, name: &str, entries: Vec<(&str, &Array)>) -> PathBuf {
    let path = scratch_file(tmp, name);
    Array::save_safetensors(entries, None, &path).unwrap();
    path
}

#[test]
fn has_diff_patch_keys_detects_diff_and_diff_b() {
    let tmp = tempfile::tempdir().unwrap();
    let dp = save_one(
        &tmp,
        "dp_detect.safetensors",
        vec![(
            "diffusion_model.txtfusion.projector.diff",
            &Array::from_slice(&[0.1f32; 3], &[1, 3]),
        )],
    );
    assert!(has_diff_patch_keys(&Weights::from_file(&dp).unwrap()));

    // A plain low-rank file carries no `.diff`/`.diff_b` → not a diff-patch file.
    let plain = save_one(
        &tmp,
        "dp_plain.safetensors",
        vec![
            (
                "diffusion_model.txtfusion.projector.lora_down.weight",
                &Array::from_slice(&[0.1f32; 3], &[1, 3]),
            ),
            (
                "diffusion_model.txtfusion.projector.lora_up.weight",
                &Array::from_slice(&[0.1f32; 1], &[1, 1]),
            ),
        ],
    );
    assert!(!has_diff_patch_keys(&Weights::from_file(&plain).unwrap()));

    // The header-only twin (sc-13884 multi-phase guard) makes the SAME diff-patch-vs-low-rank
    // distinction from just the key names — no tensor load: a `.diff`/`.diff_b` marks a diff-patch
    // (rejected from multi-phase), a low-rank `lora_*` file does not (the toggleable turbo-LoRA case).
    assert!(has_diff_patch_key_names(
        ["diffusion_model.txtfusion.projector.diff"].into_iter()
    ));
    assert!(has_diff_patch_key_names(
        ["some.module.diff_b", "other.lora_down.weight"].into_iter()
    ));
    assert!(!has_diff_patch_key_names(
        [
            "blk.0.attn.to_q.lora_down.weight",
            "blk.0.attn.to_q.lora_up.weight"
        ]
        .into_iter()
    ));
    assert!(!has_diff_patch_key_names(std::iter::empty()));
}

/// The community filter-bypass: a single `diffusion_model.txtfusion.projector.diff` folds
/// `W += scale·δ` into the dense projector base, resolving through the prefix-strip + `txtfusion`
/// alias. Untargeted state stays put.
#[test]
fn fold_diff_patch_folds_projector_weight_delta() {
    let tmp = tempfile::tempdir().unwrap();
    let base_w = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]);
    let delta = Array::from_slice(&[10.0f32, 20.0, 30.0], &[1, 3]);
    let dp = save_one(
        &tmp,
        "dp_proj.safetensors",
        vec![("diffusion_model.txtfusion.projector.diff", &delta)],
    );

    let mut host = DiffHost {
        lin: AdaptableLinear::dense(base_w, None),
    };
    let report =
        fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 0.5, AdapterKind::Lora)])
            .unwrap();
    assert_eq!(report.folded, 1, "the projector diff must fold");
    assert!(report.skipped.is_empty());
    assert!(report.unmatched.is_empty());

    // W + 0.5·δ = [1+5, 2+10, 3+15].
    let (w, _) = host.lin.dense_weight().unwrap();
    let want = Array::from_slice(&[6.0f32, 12.0, 18.0], &[1, 3]);
    assert!(all_close(w, &want, 1e-5, 1e-5, false)
        .unwrap()
        .item::<bool>());
}

/// A `.diff_b` bias delta folds into the base bias alongside the `.diff` weight delta — the channel
/// low-rank adapters cannot express. Both fold at `scale`, counted as two.
#[test]
fn fold_diff_patch_folds_weight_and_bias_delta() {
    let tmp = tempfile::tempdir().unwrap();
    let dw = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let db = Array::from_slice(&[5.0f32, 6.0], &[2]);
    let dp = save_one(
        &tmp,
        "dp_wb.safetensors",
        vec![
            ("diffusion_model.txtfusion.projector.diff", &dw),
            ("diffusion_model.txtfusion.projector.diff_b", &db),
        ],
    );
    let mut host = DiffHost {
        lin: AdaptableLinear::dense(
            Array::from_slice(&[0.0f32; 4], &[2, 2]),
            Some(Array::from_slice(&[0.0f32; 2], &[2])),
        ),
    };
    let report =
        fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 1.0, AdapterKind::Lora)])
            .unwrap();
    assert_eq!(report.folded, 2, "weight + bias delta both fold");

    let (w, b) = host.lin.dense_weight().unwrap();
    assert!(all_close(w, &dw, 1e-5, 1e-5, false).unwrap().item::<bool>());
    assert!(all_close(b.unwrap(), &db, 1e-5, 1e-5, false)
        .unwrap()
        .item::<bool>());
}

/// Module-coupled shape-aware skip: a `.diff` whose shape ≠ the base is skipped as a whole module —
/// its coupled `.diff_b` dropped too, never a half-patch — surfaced, never folded; base untouched.
#[test]
fn fold_diff_patch_shape_mismatch_skips_whole_module() {
    let tmp = tempfile::tempdir().unwrap();
    let base_w = Array::from_slice(&[0.0f32; 4], &[2, 2]);
    let base_b = Array::from_slice(&[0.0f32; 2], &[2]);
    let dp = save_one(
        &tmp,
        "dp_mm.safetensors",
        vec![
            // [3,3] cannot fold into the [2,2] base; its coupled [2] bias must drop with it.
            (
                "diffusion_model.txtfusion.projector.diff",
                &Array::from_slice(&[1.0f32; 9], &[3, 3]),
            ),
            (
                "diffusion_model.txtfusion.projector.diff_b",
                &Array::from_slice(&[9.0f32; 2], &[2]),
            ),
        ],
    );
    let mut host = DiffHost {
        lin: AdaptableLinear::dense(base_w.clone(), Some(base_b.clone())),
    };
    let report =
        fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 1.0, AdapterKind::Lora)])
            .unwrap();
    assert_eq!(report.folded, 0, "shape-mismatched module folds nothing");
    assert_eq!(
        report.skipped.len(),
        2,
        "weight + coupled bias both surfaced as skipped"
    );

    let (w, b) = host.lin.dense_weight().unwrap();
    assert!(
        array_eq(w, &base_w, false).unwrap().item::<bool>(),
        "weight untouched"
    );
    assert!(
        array_eq(b.unwrap(), &base_b, false).unwrap().item::<bool>(),
        "bias untouched"
    );
}

/// The strict diff-patch entry tolerates a **diff-only** file (the low-rank pass matches nothing,
/// but the fold pre-applied it — so no "matched nothing" error), and still errors when a diff-patch
/// file resolves to no module at all (folds nothing AND matches no low-rank target).
#[test]
fn strict_with_diff_patch_tolerates_diff_only_and_errors_on_all_unmatched() {
    let tmp = tempfile::tempdir().unwrap();
    // (a) diff-only, targets the projector → folds; applied counts the fold.
    let ok = save_one(
        &tmp,
        "dp_only.safetensors",
        vec![(
            "diffusion_model.txtfusion.projector.diff",
            &Array::from_slice(&[1.0f32; 3], &[1, 3]),
        )],
    );
    let mut host = DiffHost {
        lin: AdaptableLinear::dense(Array::from_slice(&[0.0f32; 3], &[1, 3]), None),
    };
    let report = apply_adapters_strict_with_diff_patch(
        &mut host,
        &[AdapterSpec::new(ok, 1.0, AdapterKind::Lora)],
        "krea_2",
    )
    .unwrap();
    assert_eq!(
        report.applied, 1,
        "the folded diff-patch counts toward applied"
    );

    // (b) diff-patch stem resolves to no module → folds nothing, no low-rank target → error.
    let bad = save_one(
        &tmp,
        "dp_bad.safetensors",
        vec![(
            "diffusion_model.blocks.99.unknown.diff",
            &Array::from_slice(&[0.1f32; 3], &[1, 3]),
        )],
    );
    let mut host2 = DiffHost {
        lin: AdaptableLinear::dense(Array::from_slice(&[0.0f32; 3], &[1, 3]), None),
    };
    let err = apply_adapters_strict_with_diff_patch(
        &mut host2,
        &[AdapterSpec::new(bad, 1.0, AdapterKind::Lora)],
        "krea_2",
    )
    .expect_err("an all-unmatched diff-patch file must still error");
    assert!(
        err.to_string().contains("no target modules matched"),
        "expected the no-match error, got: {err}"
    );
}
// ---- sc-15326: tier-independent diff-patch channels + no-silent-drop reporting ----------------

/// A host with one `[64,64]` biased Linear (quantizable — `in % group_size == 0`) plus an RMSNorm
/// gain and an affine LayerNorm's weight+bias, exposed on the two DIFFERENT diff-patch surfaces:
/// the Linear through `adaptable_mut`, the norms through `diff_patch_param_mut`. Mirrors the shape
/// of the real Krea Realtime / Wan host, whose lightx2v step-distill deltas split the same way.
struct TierHost {
    lin: AdaptableLinear,
    norm_q: Array,
    norm3_w: Array,
    norm3_b: Array,
}
impl TierHost {
    fn new() -> Self {
        Self {
            lin: AdaptableLinear::dense(
                Array::from_slice(&[0.25f32; 64 * 64], &[64, 64]),
                Some(Array::from_slice(&[1.0f32; 64], &[64])),
            ),
            norm_q: Array::from_slice(&[1.0f32; 8], &[8]),
            norm3_w: Array::from_slice(&[1.0f32; 8], &[8]),
            norm3_b: Array::from_slice(&[0.0f32; 8], &[8]),
        }
    }
}
impl AdaptableHost for TierHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["blocks", "0", "self_attn", "q"] => Some(&mut self.lin),
            _ => None,
        }
    }
    fn adaptable_paths(&self) -> Vec<String> {
        vec!["blocks.0.self_attn.q".to_string()]
    }
    fn diff_patch_param_mut(&mut self, path: &[&str], part: DiffPatchPart) -> Option<&mut Array> {
        match (path, part) {
            (["blocks", "0", "self_attn", "norm_q"], DiffPatchPart::Weight) => {
                Some(&mut self.norm_q)
            }
            (["blocks", "0", "norm3"], DiffPatchPart::Weight) => Some(&mut self.norm3_w),
            (["blocks", "0", "norm3"], DiffPatchPart::Bias) => Some(&mut self.norm3_b),
            _ => None,
        }
    }
}

/// **The sc-15326 decision, gated.** A `.diff_b` bias delta must fold IDENTICALLY over a packed Q4
/// base and a dense one: `QuantizedLinear` packs only the weight, so `b += scale·δ_b` is the same
/// arithmetic either way. This is what makes applying a lightx2v step-distill LoRA's 407 Linear
/// bias deltas tier-independent — without it, switching Krea Realtime to the diff-patch installer
/// would have folded 7 of 647 keys at the default Q4 against 407 at bf16, i.e. the same LoRA
/// rendering differently per tier on a product where the quant tier is a *creative* choice.
///
/// Discriminating: it compares the two tiers' resulting biases element-wise, so a tier-gated bias
/// fold (the pre-sc-15326 behaviour) fails on the fold COUNT and a wrongly-scaled one fails on the
/// values.
#[test]
fn diff_patch_bias_delta_folds_identically_on_a_packed_and_a_dense_base() {
    let tmp = tempfile::tempdir().unwrap();
    let db = Array::from_slice(&(0..64).map(|i| i as f32).collect::<Vec<_>>(), &[64]);
    let dp = save_one(
        &tmp,
        "dp_tier_bias.safetensors",
        vec![("diffusion_model.blocks.0.self_attn.q.diff_b", &db)],
    );
    let spec = |p: &PathBuf| vec![AdapterSpec::new(p.clone(), 0.5, AdapterKind::Lora)];

    let mut dense = TierHost::new();
    let r_dense = fold_diff_patch_adapters(&mut dense, &spec(&dp)).unwrap();

    let mut packed = TierHost::new();
    packed.lin.quantize(4, None).unwrap();
    assert!(
        packed.lin.is_quantized(),
        "the Q4 arm must actually be packed"
    );
    let r_packed = fold_diff_patch_adapters(&mut packed, &spec(&dp)).unwrap();

    assert_eq!(r_dense.folded, 1, "the bias delta folds on the dense base");
    assert_eq!(
        r_packed.folded, 1,
        "the bias delta must ALSO fold on the packed base — a QuantizedLinear keeps its bias dense"
    );
    assert!(
        r_packed.skipped.is_empty() && r_packed.unmatched.is_empty(),
        "nothing may be skipped on the packed tier: {r_packed:?}"
    );

    // b + 0.5·δ_b, bit-for-bit the same on both tiers.
    let want = Array::from_slice(
        &(0..64).map(|i| 1.0 + 0.5 * i as f32).collect::<Vec<_>>(),
        &[64],
    );
    for (label, b) in [
        ("dense", dense.lin.bias().unwrap()),
        ("packed", packed.lin.bias().unwrap()),
    ] {
        assert!(
            all_close(b, &want, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>(),
            "{label} bias is not b + 0.5·δ_b"
        );
    }
}

/// The one channel that genuinely CANNOT be tier-independent: a full-rank **weight** `.diff` aimed
/// at a packed Linear. It is skipped as a whole module (its coupled `.diff_b` dropped with it,
/// never a half-patch) and surfaced — this fold does not unpack a base the user deliberately chose.
/// The real lightx2v T2V file carries zero such keys, which is why the tier stays undivided in
/// practice.
#[test]
fn diff_patch_weight_delta_on_a_packed_linear_skips_the_whole_module() {
    let tmp = tempfile::tempdir().unwrap();
    let dw = Array::from_slice(&[0.5f32; 64 * 64], &[64, 64]);
    let db = Array::from_slice(&[7.0f32; 64], &[64]);
    let dp = save_one(
        &tmp,
        "dp_tier_weight.safetensors",
        vec![
            ("diffusion_model.blocks.0.self_attn.q.diff", &dw),
            ("diffusion_model.blocks.0.self_attn.q.diff_b", &db),
        ],
    );
    let mut host = TierHost::new();
    host.lin.quantize(4, None).unwrap();
    let before = host.lin.bias().unwrap().clone();
    let report =
        fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 1.0, AdapterKind::Lora)])
            .unwrap();
    assert_eq!(
        report.folded, 0,
        "a weight .diff cannot fold into a packed base"
    );
    assert_eq!(
        report.skipped.len(),
        2,
        "weight + coupled bias both surfaced as skipped, never half-applied: {report:?}"
    );
    assert!(
        array_eq(host.lin.bias().unwrap(), &before, false)
            .unwrap()
            .item::<bool>(),
        "the coupled bias must NOT be folded when its weight delta is skipped"
    );
}

/// The norm surface (sc-15326): `.diff` on an RMSNorm gain and `.diff`+`.diff_b` on an affine
/// LayerNorm fold through `AdaptableHost::diff_patch_param_mut` — the 200 `.diff` + 40
/// `norm3.diff_b` deltas a lightx2v step-distill file carries, which are not `AdaptableLinear`s at
/// any surface width and were dropped without a word before this existed.
#[test]
fn diff_patch_norm_deltas_fold_through_the_param_surface() {
    let tmp = tempfile::tempdir().unwrap();
    let d_nq = Array::from_slice(&[0.5f32; 8], &[8]);
    let d_n3w = Array::from_slice(&[2.0f32; 8], &[8]);
    let d_n3b = Array::from_slice(&[3.0f32; 8], &[8]);
    let dp = save_one(
        &tmp,
        "dp_norms.safetensors",
        vec![
            ("diffusion_model.blocks.0.self_attn.norm_q.diff", &d_nq),
            ("diffusion_model.blocks.0.norm3.diff", &d_n3w),
            ("diffusion_model.blocks.0.norm3.diff_b", &d_n3b),
        ],
    );
    let mut host = TierHost::new();
    let report =
        fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 2.0, AdapterKind::Lora)])
            .unwrap();
    assert_eq!(report.folded, 3, "all three norm deltas fold: {report:?}");
    assert!(report.skipped.is_empty() && report.unmatched.is_empty());

    for (label, got, want) in [
        ("norm_q", &host.norm_q, 1.0 + 2.0 * 0.5),
        ("norm3.weight", &host.norm3_w, 1.0 + 2.0 * 2.0),
        ("norm3.bias", &host.norm3_b, 0.0 + 2.0 * 3.0),
    ] {
        let expect = Array::from_slice(&[want; 8], &[8]);
        assert!(
            all_close(got, &expect, 1e-5, 1e-5, false)
                .unwrap()
                .item::<bool>(),
            "{label} is not p + 2.0·δ"
        );
    }
}

/// The norm path honors the same two rules the Linear path does: a shape-mismatched delta skips the
/// WHOLE module (coupled bias dropped with it, never a half-patch), and a stem the host exposes on
/// neither half is reported unmatched rather than dropped.
#[test]
fn diff_patch_norm_shape_mismatch_skips_whole_module_and_unknown_is_unmatched() {
    let tmp = tempfile::tempdir().unwrap();
    let dp = save_one(
        &tmp,
        "dp_norm_bad.safetensors",
        vec![
            // [4] cannot fold into the [8] norm3; its coupled [8] bias must drop with it.
            (
                "diffusion_model.blocks.0.norm3.diff",
                &Array::from_slice(&[1.0f32; 4], &[4]),
            ),
            (
                "diffusion_model.blocks.0.norm3.diff_b",
                &Array::from_slice(&[1.0f32; 8], &[8]),
            ),
            (
                "diffusion_model.blocks.0.nowhere.diff",
                &Array::from_slice(&[1.0f32; 8], &[8]),
            ),
        ],
    );
    let mut host = TierHost::new();
    let (w0, b0) = (host.norm3_w.clone(), host.norm3_b.clone());
    let report =
        fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 1.0, AdapterKind::Lora)])
            .unwrap();
    assert_eq!(report.folded, 0, "nothing may fold: {report:?}");
    assert_eq!(
        report.skipped.len(),
        2,
        "norm3 weight + coupled bias skipped"
    );
    assert_eq!(
        report.unmatched,
        vec!["blocks.0.nowhere".to_string()],
        "an unexposed stem is unmatched, not dropped"
    );
    assert!(array_eq(&host.norm3_w, &w0, false).unwrap().item::<bool>());
    assert!(
        array_eq(&host.norm3_b, &b0, false).unwrap().item::<bool>(),
        "the coupled bias must not be folded when its weight delta is skipped"
    );
}

/// sc-15326 review — on the norm path, a `.diff_b` whose target has **no bias channel** must take
/// only itself out, not the module's perfectly valid weight `.diff`.
///
/// Every qk-RMSNorm is weight-only, so a lightning / step-distill variant carrying `norm_q.diff_b`
/// next to `norm_q.diff` would otherwise lose all 200 valid norm weight deltas. This is exactly the
/// Linear path's rule ("a `.diff_b` with no shape-matching base bias is surfaced but does not undo
/// the weight fold"), which the norm path claimed to mirror and did not. A **shape** mismatch still
/// skips the whole module — that split is the point, and the sibling test above pins the other half.
#[test]
fn diff_patch_norm_bias_with_no_host_part_does_not_undo_the_weight_fold() {
    let tmp = tempfile::tempdir().unwrap();
    let dp = save_one(
        &tmp,
        "dp_norm_qk_bias.safetensors",
        vec![
            (
                "diffusion_model.blocks.0.self_attn.norm_q.diff",
                &Array::from_slice(&[0.5f32; 8], &[8]),
            ),
            // An RMSNorm has no β. Correctly shaped, simply nowhere to land.
            (
                "diffusion_model.blocks.0.self_attn.norm_q.diff_b",
                &Array::from_slice(&[9.0f32; 8], &[8]),
            ),
        ],
    );
    let mut host = TierHost::new();
    let report =
        fold_diff_patch_adapters(&mut host, &[AdapterSpec::new(dp, 2.0, AdapterKind::Lora)])
            .unwrap();
    assert_eq!(
        report.folded, 1,
        "the weight .diff must still fold: {report:?}"
    );
    assert_eq!(
        report.skipped,
        vec!["blocks.0.self_attn.norm_q.bias".to_string()],
        "only the bias half is surfaced, and the weight stem is NOT: {report:?}"
    );
    assert!(report.unmatched.is_empty(), "{report:?}");
    let expect = Array::from_slice(&[1.0 + 2.0 * 0.5f32; 8], &[8]);
    assert!(
        all_close(&host.norm_q, &expect, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>(),
        "norm_q must carry w + 2.0·δ, not the pre-fold value"
    );
}

/// sc-15265's invariant, held on the diff-patch path too: **installing an adapter at scale 0 is
/// byte-identical to never installing one.** The fold resolves and shape-checks as usual — so the
/// skipped/unmatched surfaces stay truthful and the zero-match guard still sees the targets — but
/// writes nothing, and the resolved deltas land on `disabled` rather than `folded`.
///
/// Also pins the consequence: a **pure** diff-patch file at scale 0 folds nothing, so without
/// counting `disabled` toward the guard relaxation, disabling it would turn into a spurious
/// "no target modules matched" error.
///
/// Discriminating on two axes, because array equality alone is **not** enough — `\u{3b4} \u{b7} 0.0` is
/// exact zero for finite deltas, so a fold that runs anyway leaves the arrays bit-identical. So this
/// asserts the report split (`folded == 0`, `disabled == 3`) AND carries one **non-finite** delta
/// entry: `Inf \u{b7} 0.0 = NaN`, which a still-running fold would write into the host at precisely the
/// setting the user chose to turn the adapter OFF. That is sc-15265's own stated rationale for
/// skipping the residual outright rather than trusting the multiply.
#[test]
fn diff_patch_at_scale_zero_writes_nothing_and_still_clears_the_zero_match_guard() {
    let tmp = tempfile::tempdir().unwrap();
    let mut nq_delta = [0.5f32; 8];
    nq_delta[3] = f32::INFINITY;
    let dp = save_one(
        &tmp,
        "dp_scale_zero.safetensors",
        vec![
            (
                "diffusion_model.blocks.0.self_attn.norm_q.diff",
                &Array::from_slice(&nq_delta, &[8]),
            ),
            (
                "diffusion_model.blocks.0.norm3.diff_b",
                &Array::from_slice(&[3.0f32; 8], &[8]),
            ),
            // A Linear bias delta — the other channel that folds on every tier.
            (
                "diffusion_model.blocks.0.self_attn.q.diff_b",
                &Array::from_slice(&[7.0f32; 64], &[64]),
            ),
        ],
    );
    // The fold itself: everything resolves, nothing is written.
    let mut probe = TierHost::new();
    let dp_report = fold_diff_patch_adapters(
        &mut probe,
        &[AdapterSpec::new(dp.clone(), 0.0, AdapterKind::Lora)],
    )
    .unwrap();
    assert_eq!(
        (dp_report.folded, dp_report.disabled),
        (0, 3),
        "a scale-0 spec resolves its targets but folds none of them: {dp_report:?}"
    );
    assert!(
        dp_report.skipped.is_empty() && dp_report.unmatched.is_empty(),
        "resolution/shape surfacing stays truthful at scale 0: {dp_report:?}"
    );

    let mut host = TierHost::new();
    let (nq0, n3b0) = (host.norm_q.clone(), host.norm3_b.clone());
    let b0 = host.lin.bias().unwrap().clone();

    let specs = [AdapterSpec::new(dp, 0.0, AdapterKind::Lora)];
    let report = apply_adapters_strict_with_diff_patch(&mut host, &specs, "tier_host")
        .expect("a disabled pure diff-patch file must install, not read as `matched nothing`");
    assert_eq!(
        report.applied, 3,
        "all three targets resolved and were accepted: {report:?}"
    );
    assert!(report.diff_patch_unapplied.is_empty(), "{report:?}");

    for (label, got, want) in [
        ("norm_q", &host.norm_q, &nq0),
        ("norm3.bias", &host.norm3_b, &n3b0),
        ("q.bias", host.lin.bias().unwrap(), &b0),
    ] {
        assert!(
            array_eq(got, want, false).unwrap().item::<bool>(),
            "{label} moved at scale 0 — a disabled adapter must be byte-identical to none"
        );
    }
}

/// **No silent drop, and it reaches the caller.** Everything the diff-patch pass could not land —
/// skipped AND unmatched — comes back on `ApplyReport::diff_patch_unapplied`, so a provider can
/// stamp a partial install into user-visible provenance instead of it living in an engine `stderr`
/// line nobody reads. A clean file leaves the list empty.
#[test]
fn strict_with_diff_patch_reports_every_unapplied_delta_on_the_report() {
    let tmp = tempfile::tempdir().unwrap();
    // A file that lands its norm delta but cannot land a packed-Linear weight delta or an
    // out-of-surface stem.
    let mixed = save_one(
        &tmp,
        "dp_report.safetensors",
        vec![
            (
                "diffusion_model.blocks.0.self_attn.norm_q.diff",
                &Array::from_slice(&[0.5f32; 8], &[8]),
            ),
            (
                "diffusion_model.blocks.0.self_attn.q.diff",
                &Array::from_slice(&[0.5f32; 64 * 64], &[64, 64]),
            ),
            (
                "diffusion_model.blocks.0.nowhere.diff",
                &Array::from_slice(&[1.0f32; 8], &[8]),
            ),
        ],
    );
    let mut host = TierHost::new();
    host.lin.quantize(4, None).unwrap();
    let report = apply_adapters_strict_with_diff_patch(
        &mut host,
        &[AdapterSpec::new(mixed, 1.0, AdapterKind::Lora)],
        "krea_realtime_14b",
    )
    .unwrap();
    assert_eq!(report.applied, 1, "only the norm delta landed");
    let mut unapplied = report.diff_patch_unapplied.clone();
    unapplied.sort();
    assert_eq!(
        unapplied,
        vec![
            "blocks.0.nowhere".to_string(),
            "blocks.0.self_attn.q".to_string()
        ],
        "both the packed-weight skip and the unmatched stem must be reported to the caller"
    );

    // A file whose every delta lands reports NOTHING unapplied — the list is a real signal, not
    // noise present on every render.
    let clean = save_one(
        &tmp,
        "dp_report_clean.safetensors",
        vec![(
            "diffusion_model.blocks.0.norm3.diff",
            &Array::from_slice(&[0.5f32; 8], &[8]),
        )],
    );
    let mut host2 = TierHost::new();
    let clean_report = apply_adapters_strict_with_diff_patch(
        &mut host2,
        &[AdapterSpec::new(clean, 1.0, AdapterKind::Lora)],
        "krea_realtime_14b",
    )
    .unwrap();
    assert!(
        clean_report.diff_patch_unapplied.is_empty(),
        "a fully-applied file must report nothing unapplied: {clean_report:?}"
    );
}
