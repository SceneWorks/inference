use super::*;
use mlx_rs::ops::{all_close, array_eq};
use std::path::PathBuf;

/// sc-4986 — retire the "does a lightx2v Lightning LoRA actually load through mlx-gen?" risk by
/// running the **real** distill-LoRA file's keys through the genuine [`normalize_wan_key`] and
/// asserting every module resolves to a valid Wan DiT target (`blocks.N.{self,cross}_attn.{q,k,v,o}`
/// or `blocks.N.ffn.{fc1,fc2}`). `#[ignore]` — point `WAN_LIGHTNING_LORA` at a staged lightx2v
/// Wan2.2 Lightning LoRA `.safetensors` (an explicit passed-in path; inference never self-fetches,
/// epic 13657):
/// ```text
/// WAN_LIGHTNING_LORA="/snapshots/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/\
/// high_noise_model.safetensors" \
///   cargo test -p mlx-gen-wan lightning_lora_keys_normalize -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs a downloaded lightx2v Wan2.2 Lightning LoRA (WAN_LIGHTNING_LORA)"]
fn lightning_lora_keys_normalize_to_wan_dit_targets() {
    let path = match std::env::var_os("WAN_LIGHTNING_LORA") {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("skip: set WAN_LIGHTNING_LORA to a downloaded lightx2v Lightning LoRA");
            return;
        }
    };
    let lw = Weights::from_file(&path).expect("read LoRA safetensors");

    // Collapse factor keys (.lora_down/.lora_up/.alpha/.lora_A/.lora_B) to distinct module paths,
    // exactly as the merge loop keys its parts, then normalize each through the real mapper.
    let mut modules = std::collections::BTreeSet::new();
    for key in lw.keys() {
        if let Some((stem, _role)) = SUFFIXES
            .iter()
            .find_map(|(suf, r)| key.strip_suffix(suf).map(|s| (s, *r)))
        {
            modules.insert(normalize_wan_key(stem));
        }
    }
    assert!(!modules.is_empty(), "no LoRA factor keys found in {path:?}");

    // Every normalized module must hit the native converted-Wan DiT namespace. Anything else
    // would fold onto nothing (silent no-op) — the exact failure we are de-risking.
    let valid = |m: &str| -> bool {
        let Some(rest) = m.strip_prefix("blocks.") else {
            // a handful of non-block targets the distill LoRA may also touch
            return matches!(m, "head.head" | "patch_embedding_proj")
                || m.starts_with("text_embedding_")
                || m.starts_with("time_embedding_")
                || m == "time_projection";
        };
        let Some((_n, tail)) = rest.split_once('.') else {
            return false;
        };
        matches!(
            tail,
            "self_attn.q"
                | "self_attn.k"
                | "self_attn.v"
                | "self_attn.o"
                | "cross_attn.q"
                | "cross_attn.k"
                | "cross_attn.v"
                | "cross_attn.o"
                | "cross_attn.k_img"
                | "cross_attn.v_img"
                | "ffn.fc1"
                | "ffn.fc2"
        )
    };
    let bad: Vec<&String> = modules.iter().filter(|m| !valid(m)).collect();
    println!(
        "[lightning lora] {} distinct modules; {} resolve to valid Wan DiT targets, {} unmatched",
        modules.len(),
        modules.len() - bad.len(),
        bad.len()
    );
    if !bad.is_empty() {
        println!("[lightning lora] UNMATCHED (would fold onto nothing): {bad:?}");
    }
    // Sample the resolved targets for the log.
    for m in modules.iter().take(4) {
        println!("[lightning lora]   e.g. {m}");
    }
    assert!(
        bad.is_empty(),
        "{} Lightning LoRA module(s) normalize to non-DiT targets and would silently no-op",
        bad.len()
    );
}

#[test]
fn normalize_strips_prefix_and_renames() {
    // attn q/k/v/o pass through (already checkpoint naming).
    assert_eq!(
        normalize_wan_key("diffusion_model.blocks.0.self_attn.q"),
        "blocks.0.self_attn.q"
    );
    assert_eq!(
        normalize_wan_key("diffusion_model.blocks.7.cross_attn.o"),
        "blocks.7.cross_attn.o"
    );
    // ffn.0/.2 → fc1/fc2.
    assert_eq!(
        normalize_wan_key("diffusion_model.blocks.0.ffn.0"),
        "blocks.0.ffn.fc1"
    );
    assert_eq!(
        normalize_wan_key("diffusion_model.blocks.3.ffn.2"),
        "blocks.3.ffn.fc2"
    );
    // global renames + other prefixes.
    assert_eq!(
        normalize_wan_key("model.diffusion_model.text_embedding.0"),
        "text_embedding_0"
    );
    assert_eq!(
        normalize_wan_key("base_model.model.text_embedding.2"),
        "text_embedding_1"
    );
    assert_eq!(normalize_wan_key("time_embedding.0"), "time_embedding_0");
    assert_eq!(
        normalize_wan_key("diffusion_model.time_projection.1"),
        "time_projection"
    );
    assert_eq!(
        normalize_wan_key("diffusion_model.patch_embedding"),
        "patch_embedding_proj"
    );
}

#[test]
fn normalize_matches_reference_golden() {
    // Parity vs the reference `_normalize_wan_lora_key` over every real lauren MoE LoRA module
    // stem (400) + synthetic global / alternate-prefix spellings, resolved against the real
    // converted A14B weight-key set (tools/dump_lora_fixtures.py). This is the load-bearing
    // piece of the merge — the Wan key→module map must be byte-identical to the reference's.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/wan_lora_keys.json"
    );
    let text = std::fs::read_to_string(path).expect("read wan_lora_keys.json fixture");
    let map: BTreeMap<String, String> = serde_json::from_str(&text).expect("parse fixture");
    assert!(
        map.len() >= 400,
        "fixture should cover the full real LoRA surface (got {})",
        map.len()
    );
    for (raw, expected) in &map {
        assert_eq!(
            &normalize_wan_key(raw),
            expected,
            "normalize_wan_key({raw}) must match the reference _normalize_wan_lora_key"
        );
    }
}

#[test]
fn normalize_vace_passes_diffusers_and_renames_native() {
    // Diffusers names (the host layout) pass through after prefix strip — incl. vace_blocks.
    for (raw, want) in [
        ("diffusion_model.blocks.0.attn1.to_q", "blocks.0.attn1.to_q"),
        (
            "diffusion_model.blocks.3.attn2.to_out.0",
            "blocks.3.attn2.to_out.0",
        ),
        (
            "diffusion_model.blocks.0.ffn.net.0.proj",
            "blocks.0.ffn.net.0.proj",
        ),
        ("blocks.2.ffn.net.2", "blocks.2.ffn.net.2"),
        (
            "diffusion_model.vace_blocks.0.attn1.to_v",
            "vace_blocks.0.attn1.to_v",
        ),
        (
            "diffusion_model.vace_blocks.0.proj_in",
            "vace_blocks.0.proj_in",
        ),
        (
            "diffusion_model.vace_blocks.7.proj_out",
            "vace_blocks.7.proj_out",
        ),
    ] {
        assert_eq!(normalize_vace_key(raw), want, "passthrough {raw}");
    }
    // Native Wan spellings (musubi / diffusion-pipe) → diffusers (the diffusers loader's map).
    for (raw, want) in [
        (
            "diffusion_model.blocks.0.self_attn.q",
            "blocks.0.attn1.to_q",
        ),
        (
            "diffusion_model.blocks.5.self_attn.o",
            "blocks.5.attn1.to_out.0",
        ),
        (
            "diffusion_model.blocks.0.cross_attn.k",
            "blocks.0.attn2.to_k",
        ),
        ("diffusion_model.blocks.0.ffn.0", "blocks.0.ffn.net.0.proj"),
        ("diffusion_model.blocks.2.ffn.2", "blocks.2.ffn.net.2"),
        // VACE block native hint projections → proj_in/proj_out (diffusers leaves these alone;
        // we complete the map for native-trained VACE LoRAs).
        (
            "diffusion_model.vace_blocks.0.before_proj",
            "vace_blocks.0.proj_in",
        ),
        (
            "diffusion_model.vace_blocks.3.after_proj",
            "vace_blocks.3.proj_out",
        ),
        (
            "diffusion_model.vace_blocks.1.self_attn.v",
            "vace_blocks.1.attn1.to_v",
        ),
        // Globals + alternate prefixes.
        (
            "diffusion_model.time_projection.1",
            "condition_embedder.time_proj",
        ),
        ("model.diffusion_model.head.head", "proj_out"),
        (
            "base_model.model.text_embedding.0",
            "condition_embedder.text_embedder.linear_1",
        ),
        (
            "diffusion_model.time_embedding.2",
            "condition_embedder.time_embedder.linear_2",
        ),
    ] {
        assert_eq!(normalize_vace_key(raw), want, "rename {raw}");
    }
}

#[test]
fn normalize_vace_matches_reference_golden() {
    // sc-3439 parity gate for the diffusers-named VACE key→module map. The fixture
    // (tools/dump_wanvace_lora_keys.py) takes the base-block + global native→diffusers mappings
    // **authoritatively from the diffusers loader** (`_convert_non_diffusers_wan_lora_to_diffusers`)
    // and the vace_blocks + diffusers-passthrough entries from the shared rename rule, every target
    // verified to be a real module in the cached `Wan2.1-VACE-1.3B-diffusers` checkpoint. The Rust
    // `normalize_vace_key` must reproduce each mapping — the load-bearing piece of the VACE merge.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/wanvace_lora_keys.json"
    );
    let text = std::fs::read_to_string(path).expect("read wanvace_lora_keys.json fixture");
    let map: BTreeMap<String, String> = serde_json::from_str(&text).expect("parse fixture");
    assert!(
        map.len() >= 80,
        "fixture should cover the VACE LoRA surface (got {})",
        map.len()
    );
    for (raw, expected) in &map {
        assert_eq!(
            &normalize_vace_key(raw),
            expected,
            "normalize_vace_key({raw}) must match the diffusers VACE key map"
        );
    }
}

fn scratch_file(tmp: &tempfile::TempDir, name: &str) -> PathBuf {
    // Per-process scratch dir — a fixed `$TMPDIR` name races a second concurrent `cargo test`.
    let dir = tmp.path().join("mlx_gen_wan_adapters_test");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// Write a PEFT LoRA file (`diffusion_model.‹stem›.lora_A/B.weight`) for the given stems, with
/// A `[rank,in]`, B `[out,rank]`, no alpha (→ scale = 1). Values are deterministic per stem.
fn write_lora(
    tmp: &tempfile::TempDir,
    name: &str,
    stems: &[(&str, i32, i32)],
    rank: i32,
    seed: f32,
) -> PathBuf {
    let mut entries: Vec<(String, Array)> = Vec::new();
    for (stem, out, inp) in stems {
        let a = Array::from_slice(
            &(0..rank * inp)
                .map(|i| (i as f32 * 0.001 + seed).sin() * 0.02)
                .collect::<Vec<_>>(),
            &[rank, *inp],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let b = Array::from_slice(
            &(0..out * rank)
                .map(|i| (i as f32 * 0.0007 + seed).cos() * 0.02)
                .collect::<Vec<_>>(),
            &[*out, rank],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        entries.push((format!("diffusion_model.{stem}.lora_A.weight"), a));
        entries.push((format!("diffusion_model.{stem}.lora_B.weight"), b));
    }
    let path = scratch_file(tmp, name);
    let refs: Vec<(&str, &Array)> = entries.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, &path).unwrap();
    path
}

/// A synthetic expert weight map with the two module weights the test LoRA targets, bf16.
fn synthetic_weights(tmp: &tempfile::TempDir) -> Weights {
    let path = scratch_file(tmp, "base.safetensors");
    let q = Array::from_slice(
        &(0..16 * 8)
            .map(|i| i as f32 * 0.01 - 0.3)
            .collect::<Vec<_>>(),
        &[16, 8],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    let fc1 = Array::from_slice(
        &(0..24 * 8)
            .map(|i| i as f32 * 0.005 - 0.2)
            .collect::<Vec<_>>(),
        &[24, 8],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    Array::save_safetensors(
        vec![
            ("blocks.0.self_attn.q.weight", &q),
            ("blocks.0.ffn.fc1.weight", &fc1),
        ],
        None,
        &path,
    )
    .unwrap();
    Weights::from_file(&path).unwrap()
}

fn spec(path: PathBuf, scale: f32, expert: Option<MoeExpert>) -> AdapterSpec {
    AdapterSpec {
        path,
        scale,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: expert,
    }
}

/// Write a peft LoKr file for `blocks.0.self_attn.q` ([16,8] = kron(w1[4,2], w2[4,4])) with the
/// given `alpha`/`rank` in metadata (sc-10044 additive tests). Deterministic factor values.
fn write_lokr(tmp: &tempfile::TempDir, name: &str, alpha: f32, rank: f32) -> PathBuf {
    use std::collections::HashMap;
    let w1 = Array::from_slice(
        &(0..8)
            .map(|i| (i as f32 * 0.03).sin() * 0.1)
            .collect::<Vec<_>>(),
        &[4, 2],
    );
    let w2 = Array::from_slice(
        &(0..16)
            .map(|i| (i as f32 * 0.05).cos() * 0.1)
            .collect::<Vec<_>>(),
        &[4, 4],
    );
    let meta = HashMap::from([
        ("networkType".to_string(), "lokr".to_string()),
        ("alpha".to_string(), alpha.to_string()),
        ("rank".to_string(), rank.to_string()),
    ]);
    let path = scratch_file(tmp, name);
    Array::save_safetensors(
        vec![
            ("blocks.0.self_attn.q.lokr_w1", &w1),
            ("blocks.0.self_attn.q.lokr_w2", &w2),
        ],
        Some(&meta),
        &path,
    )
    .unwrap();
    path
}

/// A LoKr-kind [`AdapterSpec`] (the [`spec`] helper defaults to `AdapterKind::Lora`).
fn lokr_spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec {
        kind: AdapterKind::Lokr,
        ..spec(path, scale, None)
    }
}

#[test]
fn merge_folds_delta_bit_exact() {
    let tmp = tempfile::tempdir().unwrap();
    // Reference merge: W += (B·A)·(alpha/rank·strength).astype(W.dtype), at the factor dtype.
    let lora = write_lora(
        &tmp,
        "merge.safetensors",
        &[("blocks.0.self_attn.q", 16, 8), ("blocks.0.ffn.0", 24, 8)],
        4,
        0.1,
    );
    let mut w = synthetic_weights(&tmp);
    let report =
        merge_wan_adapters(&mut w, &[spec(lora.clone(), 1.0, None)], MoeExpert::High).unwrap();
    assert_eq!(report.applied, 2);
    assert!(report.skipped.is_empty());

    // Hand-compute the expected merge for the q weight.
    let lw = Weights::from_file(&lora).unwrap();
    let base = synthetic_weights(&tmp);
    let q_base = base.require("blocks.0.self_attn.q.weight").unwrap();
    let a = lw
        .require("diffusion_model.blocks.0.self_attn.q.lora_A.weight")
        .unwrap();
    let b = lw
        .require("diffusion_model.blocks.0.self_attn.q.lora_B.weight")
        .unwrap();
    let delta = matmul(b, a).unwrap();
    let want = add(q_base, delta.as_dtype(q_base.dtype()).unwrap()).unwrap();
    let got = w.require("blocks.0.self_attn.q.weight").unwrap();
    assert!(
        array_eq(got, &want, false).unwrap().item::<bool>(),
        "merged q weight must be bit-exact to W + (B·A).astype(W.dtype)"
    );
    // And the ffn key was the renamed target (ffn.0 → ffn.fc1).
    assert!(w.get("blocks.0.ffn.fc1.weight").is_some());
}

#[test]
fn merge_honors_lora_adapter_metadata_alpha() {
    let tmp = tempfile::tempdir().unwrap();
    // sc-5513: a diffusers / PEFT `save_lora_adapter` LoRA carries NO per-target `.alpha` tensor —
    // the scaling lives in the `lora_adapter_metadata` blob. With `lora_alpha = 16`, `r = 8` (the
    // factor's true rank) the Wan merge must fold `(16/8) = 2.0`, not the pre-sc-5513 `alpha = rank`
    // default (factor 1.0).
    use std::collections::HashMap;
    let rank = 8;
    // One target, factor rank 8 (= the blob `r`): A [8, 8], B [16, 8] for the [16,8] base q weight.
    let a = Array::from_slice(
        &(0..rank * 8)
            .map(|i| (i as f32 * 0.001 + 0.1).sin() * 0.02)
            .collect::<Vec<_>>(),
        &[rank, 8],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    let b = Array::from_slice(
        &(0..16 * rank)
            .map(|i| (i as f32 * 0.0007 + 0.1).cos() * 0.02)
            .collect::<Vec<_>>(),
        &[16, rank],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    let path = scratch_file(&tmp, "merge_meta_alpha.safetensors");
    // Deliberately NO `.alpha` tensor — the scaling must come from the blob.
    let meta = HashMap::from([(
        "lora_adapter_metadata".to_string(),
        r#"{"lora_alpha": 16, "r": 8}"#.to_string(),
    )]);
    Array::save_safetensors(
        vec![
            ("diffusion_model.blocks.0.self_attn.q.lora_A.weight", &a),
            ("diffusion_model.blocks.0.self_attn.q.lora_B.weight", &b),
        ],
        Some(&meta),
        &path,
    )
    .unwrap();

    let mut w = synthetic_weights(&tmp);
    let report = merge_wan_adapters(&mut w, &[spec(path, 1.0, None)], MoeExpert::High).unwrap();
    assert_eq!(report.applied, 1);

    // Reference: W += (B·A)·(alpha/rank = 2.0), folded at the factor dtype like the merge does.
    let base = synthetic_weights(&tmp);
    let q_base = base.require("blocks.0.self_attn.q.weight").unwrap();
    let delta = matmul(&b, &a).unwrap();
    let two = scalar(2.0f32).as_dtype(delta.dtype()).unwrap();
    let want = add(
        q_base,
        multiply(&delta, &two)
            .unwrap()
            .as_dtype(q_base.dtype())
            .unwrap(),
    )
    .unwrap();
    let got = w.require("blocks.0.self_attn.q.weight").unwrap();
    assert!(
        array_eq(got, &want, false).unwrap().item::<bool>(),
        "metadata-alpha merge must fold (16/8)·strength = 2.0"
    );
    // The pre-sc-5513 default (alpha = rank = 8 ⇒ factor 1.0) would diverge by a full factor of 2.
    let one_want = add(q_base, delta.as_dtype(q_base.dtype()).unwrap()).unwrap();
    assert!(
        !array_eq(got, &one_want, false).unwrap().item::<bool>(),
        "metadata alpha must differ from the alpha=rank default"
    );
}

#[test]
fn scale_zero_is_bit_exact_noop() {
    let tmp = tempfile::tempdir().unwrap();
    let lora = write_lora(
        &tmp,
        "zero.safetensors",
        &[("blocks.0.self_attn.q", 16, 8)],
        4,
        0.3,
    );
    let base = synthetic_weights(&tmp);
    let mut w = synthetic_weights(&tmp);
    let report = merge_wan_adapters(&mut w, &[spec(lora, 0.0, None)], MoeExpert::Low).unwrap();
    assert_eq!(report.applied, 1); // still "applied" (folded a zero delta), like the reference.
    let got = w.require("blocks.0.self_attn.q.weight").unwrap();
    let unchanged = base.require("blocks.0.self_attn.q.weight").unwrap();
    assert!(
        array_eq(got, unchanged, false).unwrap().item::<bool>(),
        "strength 0 must leave the weight bit-identical"
    );
}

#[test]
fn high_low_filter_selects_shared_plus_expert() {
    let tmp = tempfile::tempdir().unwrap();
    let shared = write_lora(
        &tmp,
        "shared.safetensors",
        &[("blocks.0.self_attn.q", 16, 8)],
        4,
        0.2,
    );
    let high_only = write_lora(
        &tmp,
        "highonly.safetensors",
        &[("blocks.0.ffn.0", 24, 8)],
        4,
        0.5,
    );

    // Building the LOW expert: the shared file applies, the high-only file does NOT.
    let mut low = synthetic_weights(&tmp);
    let low_rep = merge_wan_adapters(
        &mut low,
        &[
            spec(shared.clone(), 1.0, None),
            spec(high_only.clone(), 1.0, Some(MoeExpert::High)),
        ],
        MoeExpert::Low,
    )
    .unwrap();
    assert_eq!(low_rep.applicable, 1, "only the shared spec applies to low");
    assert_eq!(low_rep.applied, 1);

    // Building the HIGH expert: both the shared and the high-only file apply.
    let mut high = synthetic_weights(&tmp);
    let high_rep = merge_wan_adapters(
        &mut high,
        &[
            spec(shared, 1.0, None),
            spec(high_only, 1.0, Some(MoeExpert::High)),
        ],
        MoeExpert::High,
    )
    .unwrap();
    assert_eq!(high_rep.applicable, 2, "shared + high-only apply to high");
    assert_eq!(high_rep.applied, 2);

    // The two experts' q weights differ from the bare base (visible effect) and the high expert's
    // ffn was merged while the low expert's was not.
    let base = synthetic_weights(&tmp);
    let q_base = base.require("blocks.0.self_attn.q.weight").unwrap();
    let q_low = low.require("blocks.0.self_attn.q.weight").unwrap();
    assert!(!array_eq(q_low, q_base, false).unwrap().item::<bool>());
    let fc1_base = base.require("blocks.0.ffn.fc1.weight").unwrap();
    let fc1_low = low.require("blocks.0.ffn.fc1.weight").unwrap();
    let fc1_high = high.require("blocks.0.ffn.fc1.weight").unwrap();
    assert!(
        array_eq(fc1_low, fc1_base, false).unwrap().item::<bool>(),
        "low expert's ffn must be untouched (high-only LoRA)"
    );
    assert!(!array_eq(fc1_high, fc1_base, false).unwrap().item::<bool>());
}

#[test]
fn accumulates_multiple_specs_on_one_module() {
    let tmp = tempfile::tempdir().unwrap();
    // Two shared LoRAs on the same module accumulate (W + d1 + d2), order-preserving.
    let l1 = write_lora(
        &tmp,
        "acc1.safetensors",
        &[("blocks.0.self_attn.q", 16, 8)],
        4,
        0.1,
    );
    let l2 = write_lora(
        &tmp,
        "acc2.safetensors",
        &[("blocks.0.self_attn.q", 16, 8)],
        4,
        0.9,
    );
    let mut w = synthetic_weights(&tmp);
    merge_wan_adapters(
        &mut w,
        &[spec(l1.clone(), 1.0, None), spec(l2.clone(), 1.0, None)],
        MoeExpert::High,
    )
    .unwrap();

    let base = synthetic_weights(&tmp);
    let mut want = base.require("blocks.0.self_attn.q.weight").unwrap().clone();
    for lpath in [&l1, &l2] {
        let lw = Weights::from_file(lpath).unwrap();
        let a = lw
            .require("diffusion_model.blocks.0.self_attn.q.lora_A.weight")
            .unwrap();
        let b = lw
            .require("diffusion_model.blocks.0.self_attn.q.lora_B.weight")
            .unwrap();
        let delta = matmul(b, a).unwrap();
        want = add(&want, delta.as_dtype(want.dtype()).unwrap()).unwrap();
    }
    let got = w.require("blocks.0.self_attn.q.weight").unwrap();
    assert!(
        all_close(got, &want, 1e-6, 1e-6, false)
            .unwrap()
            .item::<bool>(),
        "two stacked LoRAs must equal W + d1 + d2 in order"
    );
}

#[test]
fn lokr_merge_matches_reconstruct_and_scale_zero_is_noop() {
    let tmp = tempfile::tempdir().unwrap();
    // sc-2393: LoKr merges through the same in-place fold. `blocks.0.self_attn.q` is [16,8] =
    // kron(w1[4,2], w2[4,4]); the merged weight must equal W + (reconstruct·scale).astype(W.dtype),
    // and scale 0 must be a bit-exact no-op.
    use mlx_gen::adapters::reconstruct_lokr_delta;
    use std::collections::HashMap;

    let w1 = Array::from_slice(
        &(0..8)
            .map(|i| (i as f32 * 0.03).sin() * 0.1)
            .collect::<Vec<_>>(),
        &[4, 2],
    );
    let w2 = Array::from_slice(
        &(0..16)
            .map(|i| (i as f32 * 0.05).cos() * 0.1)
            .collect::<Vec<_>>(),
        &[4, 4],
    );
    let (alpha, rank) = (4.0f32, 4.0f32); // alpha/rank = 1.0
    let mut meta = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("alpha".to_string(), alpha.to_string());
    meta.insert("rank".to_string(), rank.to_string());
    let lokr_path = scratch_file(&tmp, "lokr.safetensors");
    Array::save_safetensors(
        vec![
            ("blocks.0.self_attn.q.lokr_w1", &w1),
            ("blocks.0.self_attn.q.lokr_w2", &w2),
        ],
        Some(&meta),
        &lokr_path,
    )
    .unwrap();

    let scale = 0.5f32;
    let mut w = synthetic_weights(&tmp);
    let report = merge_wan_adapters(
        &mut w,
        &[AdapterSpec {
            kind: AdapterKind::Lokr,
            ..spec(lokr_path, scale, None)
        }],
        MoeExpert::High,
    )
    .unwrap();
    assert_eq!(report.applied, 1);
    assert!(report.skipped.is_empty());

    let base = synthetic_weights(&tmp);
    let q_base = base.require("blocks.0.self_attn.q.weight").unwrap();
    let delta = reconstruct_lokr_delta(
        alpha,
        rank,
        q_base.shape(),
        Some(&w1),
        None,
        None,
        Some(&w2),
        None,
        None,
        Dtype::Float32,
    )
    .unwrap();
    let delta = multiply(&delta, scalar(scale).as_dtype(delta.dtype()).unwrap()).unwrap();
    let want = add(q_base, delta.as_dtype(q_base.dtype()).unwrap()).unwrap();
    let got = w.require("blocks.0.self_attn.q.weight").unwrap();
    assert!(
        array_eq(got, &want, false).unwrap().item::<bool>(),
        "merged LoKr weight must be bit-exact to W + (reconstruct·scale).astype(W.dtype)"
    );

    // scale 0 → the merged weight is bit-identical to the base.
    let mut w0 = synthetic_weights(&tmp);
    merge_wan_adapters(
        &mut w0,
        &[AdapterSpec {
            kind: AdapterKind::Lokr,
            ..spec(scratch_file(&tmp, "lokr.safetensors"), 0.0, None)
        }],
        MoeExpert::High,
    )
    .unwrap();
    assert!(
        array_eq(
            w0.require("blocks.0.self_attn.q.weight").unwrap(),
            q_base,
            false
        )
        .unwrap()
        .item::<bool>(),
        "scale-0 LoKr merge must be a bit-exact no-op"
    );
}

/// sc-3671: a third-party (non-peft / lycoris) LoKr **and** LoHa file merges into the Wan weight
/// map via the same `merge_wan_adapters` path (detected by keys), reconstructing the lycoris
/// reference delta. Base weight = 0 so the merged weight equals `ΔW` exactly (the fixtures from
/// `<repo>/tests/fixtures`, generated through `~/mlx-flux-venv`). The fixture module "proj" stands
/// in for a Wan checkpoint module; `wan_module_table` resolves the `lycoris_proj` key to it.
#[test]
fn thirdparty_lycoris_merges_against_reference() {
    let tmp = tempfile::tempdir().unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    for (dir, stem) in [
        ("sc3642_lokr", "linear_w1full_w2lr"),
        ("sc3643_loha", "linear"),
    ] {
        let base = root.join("tests/fixtures").join(dir);
        let exp = Weights::from_file(base.join(format!("{stem}.expected.safetensors"))).unwrap();
        let want = exp.require("proj").unwrap();
        // Base weight map: a single f32 zero "proj.weight" of the delta's shape.
        let zero = Array::zeros::<f32>(want.shape()).unwrap();
        let base_path = scratch_file(&tmp, &format!("wan_tp_base_{stem}.safetensors"));
        Array::save_safetensors(vec![("proj.weight", &zero)], None, &base_path).unwrap();
        let mut w = Weights::from_file(&base_path).unwrap();

        let report = merge_wan_adapters(
            &mut w,
            &[spec(base.join(format!("{stem}.safetensors")), 1.0, None)],
            MoeExpert::High,
        )
        .unwrap();
        assert_eq!(report.applied, 1, "{stem}: third-party file did not merge");
        assert!(
            report.skipped.is_empty(),
            "{stem}: unexpected skip {:?}",
            report.skipped
        );

        let got = w.require("proj.weight").unwrap();
        assert!(
            all_close(got, want, 1e-4, 1e-5, false)
                .unwrap()
                .item::<bool>(),
            "{stem}: Wan third-party merge diverged from the lycoris reference"
        );
    }
}

// ====================================================================================
// sc-3439 — VACE diffusers-named merge (`merge_vace_adapters`). Same merge math + format
// dispatch as the native Wan path, on the diffusers `attn1/attn2.to_*` / `ffn.net.*` /
// `vace_blocks.*` host. Bit-exact vs the in-test hand-computed `W + (B·A)` / `W + reconstruct`.
// ====================================================================================

/// A synthetic VACE (diffusers-layout) weight map: a base-block attn projection, a base-block
/// FFN proj, and a vace-block hint projection — bf16, the modules the VACE LoRA tests target.
fn synthetic_vace_weights(tmp: &tempfile::TempDir) -> Weights {
    let path = scratch_file(tmp, "vace_base.safetensors");
    let mk = |n: i32, scale: f32, bias: f32| {
        Array::from_slice(
            &(0..n).map(|i| i as f32 * scale - bias).collect::<Vec<_>>(),
            &[n / 8, 8],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
    };
    let q = mk(16 * 8, 0.01, 0.3); // attn1.to_q [16,8]
    let fc1 = mk(24 * 8, 0.005, 0.2); // ffn.net.0.proj [24,8]
    let pin = mk(16 * 8, 0.007, 0.25); // vace_blocks.0.proj_in [16,8]
    Array::save_safetensors(
        vec![
            ("blocks.0.attn1.to_q.weight", &q),
            ("blocks.0.ffn.net.0.proj.weight", &fc1),
            ("vace_blocks.0.proj_in.weight", &pin),
        ],
        None,
        &path,
    )
    .unwrap();
    Weights::from_file(&path).unwrap()
}

#[test]
fn merge_vace_folds_diffusers_named_delta_bit_exact() {
    let tmp = tempfile::tempdir().unwrap();
    // A diffusers-named LoRA (the host layout) folds W += B·A on the matching VACE modules,
    // including a vace_blocks Linear. Bit-exact to the hand-computed merge.
    let lora = write_lora(
        &tmp,
        "vace_diff.safetensors",
        &[
            ("blocks.0.attn1.to_q", 16, 8),
            ("blocks.0.ffn.net.0.proj", 24, 8),
            ("vace_blocks.0.proj_in", 16, 8),
        ],
        4,
        0.2,
    );
    let mut w = synthetic_vace_weights(&tmp);
    let report = merge_vace_adapters(&mut w, &[spec(lora.clone(), 1.0, None)]).unwrap();
    assert_eq!(report.applied, 3);
    assert!(report.skipped.is_empty());

    let lw = Weights::from_file(&lora).unwrap();
    let base = synthetic_vace_weights(&tmp);
    for stem in [
        "blocks.0.attn1.to_q",
        "blocks.0.ffn.net.0.proj",
        "vace_blocks.0.proj_in",
    ] {
        let wkey = format!("{stem}.weight");
        let a = lw
            .require(&format!("diffusion_model.{stem}.lora_A.weight"))
            .unwrap();
        let b = lw
            .require(&format!("diffusion_model.{stem}.lora_B.weight"))
            .unwrap();
        let delta = matmul(b, a).unwrap();
        let want = add(
            base.require(&wkey).unwrap(),
            delta.as_dtype(Dtype::Bfloat16).unwrap(),
        )
        .unwrap();
        let got = w.require(&wkey).unwrap();
        assert!(
            array_eq(got, &want, false).unwrap().item::<bool>(),
            "{stem}: merged weight must be bit-exact to W + (B·A).astype(W.dtype)"
        );
    }
}

#[test]
fn merge_vace_renames_native_named_lora_to_diffusers_host() {
    let tmp = tempfile::tempdir().unwrap();
    // A native-Wan-named LoRA (self_attn.q / ffn.0 — what musubi / diffusion-pipe emit) resolves
    // onto the diffusers host modules (attn1.to_q / ffn.net.0.proj) and folds there.
    let lora = write_lora(
        &tmp,
        "vace_native.safetensors",
        &[("blocks.0.self_attn.q", 16, 8), ("blocks.0.ffn.0", 24, 8)],
        4,
        0.4,
    );
    let mut w = synthetic_vace_weights(&tmp);
    let report = merge_vace_adapters(&mut w, &[spec(lora.clone(), 1.0, None)]).unwrap();
    assert_eq!(
        report.applied, 2,
        "native names must resolve to the diffusers host"
    );
    assert!(report.skipped.is_empty());

    let base = synthetic_vace_weights(&tmp);
    // The diffusers host keys moved; the native key names are absent (they were renamed).
    let q = w.require("blocks.0.attn1.to_q.weight").unwrap();
    let q_base = base.require("blocks.0.attn1.to_q.weight").unwrap();
    assert!(!array_eq(q, q_base, false).unwrap().item::<bool>());
    assert!(w.get("blocks.0.self_attn.q.weight").is_none());
    assert!(w.get("blocks.0.ffn.net.0.proj.weight").is_some());
}

#[test]
fn merge_vace_lokr_matches_reconstruct_and_scale_zero_is_noop() {
    let tmp = tempfile::tempdir().unwrap();
    // sc-2393 LoKr on the diffusers host: `blocks.0.attn1.to_q` is [16,8] = kron(w1[4,2],w2[4,4]).
    // Merged weight must equal W + (reconstruct·scale).astype(W.dtype); scale 0 is a bit-exact no-op.
    use mlx_gen::adapters::reconstruct_lokr_delta;
    use std::collections::HashMap;

    let w1 = Array::from_slice(
        &(0..8)
            .map(|i| (i as f32 * 0.03).sin() * 0.1)
            .collect::<Vec<_>>(),
        &[4, 2],
    );
    let w2 = Array::from_slice(
        &(0..16)
            .map(|i| (i as f32 * 0.05).cos() * 0.1)
            .collect::<Vec<_>>(),
        &[4, 4],
    );
    let (alpha, rank) = (4.0f32, 4.0f32);
    let mut meta = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("alpha".to_string(), alpha.to_string());
    meta.insert("rank".to_string(), rank.to_string());
    let lokr_path = scratch_file(&tmp, "vace_lokr.safetensors");
    Array::save_safetensors(
        vec![
            ("blocks.0.attn1.to_q.lokr_w1", &w1),
            ("blocks.0.attn1.to_q.lokr_w2", &w2),
        ],
        Some(&meta),
        &lokr_path,
    )
    .unwrap();

    let scale = 0.5f32;
    let mut w = synthetic_vace_weights(&tmp);
    let report = merge_vace_adapters(
        &mut w,
        &[AdapterSpec {
            kind: AdapterKind::Lokr,
            ..spec(lokr_path.clone(), scale, None)
        }],
    )
    .unwrap();
    assert_eq!(report.applied, 1);
    assert!(report.skipped.is_empty());

    let base = synthetic_vace_weights(&tmp);
    let q_base = base.require("blocks.0.attn1.to_q.weight").unwrap();
    let delta = reconstruct_lokr_delta(
        alpha,
        rank,
        q_base.shape(),
        Some(&w1),
        None,
        None,
        Some(&w2),
        None,
        None,
        Dtype::Float32,
    )
    .unwrap();
    let delta = multiply(&delta, scalar(scale).as_dtype(delta.dtype()).unwrap()).unwrap();
    let want = add(q_base, delta.as_dtype(q_base.dtype()).unwrap()).unwrap();
    let got = w.require("blocks.0.attn1.to_q.weight").unwrap();
    assert!(
        array_eq(got, &want, false).unwrap().item::<bool>(),
        "merged VACE LoKr weight must be bit-exact to W + (reconstruct·scale).astype(W.dtype)"
    );

    // scale 0 → bit-exact no-op.
    let mut w0 = synthetic_vace_weights(&tmp);
    merge_vace_adapters(
        &mut w0,
        &[AdapterSpec {
            kind: AdapterKind::Lokr,
            ..spec(lokr_path, 0.0, None)
        }],
    )
    .unwrap();
    assert!(
        array_eq(
            w0.require("blocks.0.attn1.to_q.weight").unwrap(),
            q_base,
            false
        )
        .unwrap()
        .item::<bool>(),
        "scale-0 VACE LoKr merge must be a bit-exact no-op"
    );
}

#[test]
fn merge_vace_reports_skipped_target_never_fatal() {
    let tmp = tempfile::tempdir().unwrap();
    // A LoRA module absent from the checkpoint is surfaced (skipped), never fatal — and a module
    // that IS present still merges in the same file.
    let lora = write_lora(
        &tmp,
        "vace_skip.safetensors",
        &[
            ("blocks.0.attn1.to_q", 16, 8),
            ("blocks.99.attn1.to_q", 16, 8),
        ],
        4,
        0.1,
    );
    let mut w = synthetic_vace_weights(&tmp);
    let report = merge_vace_adapters(&mut w, &[spec(lora, 1.0, None)]).unwrap();
    assert_eq!(report.applied, 1);
    assert_eq!(report.skipped, vec!["blocks.99.attn1.to_q".to_string()]);
}

// ---- Additive (unmerged) install (sc-10044) --------------------------------------------------

use mlx_gen::adapters::AdaptableLinear;

/// A minimal [`AdaptableHost`] mirroring the native Wan DiT's adaptable surface for one block:
/// `blocks.0.self_attn.{q,k,v,o}`, `blocks.0.cross_attn.{q,k,v,o}`, `blocks.0.ffn.{fc1,fc2}`. It
/// carries the two linears the test LoRA files target (`self_attn.q` and `ffn.fc1`) as real
/// [`AdaptableLinear`]s, so [`apply_wan_adapters_additive`] exercises the genuine install path
/// (normalize + factor convention + residual stacking) without the full ~7GB `WanTransformer`.
struct TinyWanHost {
    q: AdaptableLinear,
    fc1: AdaptableLinear,
}

impl AdaptableHost for TinyWanHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["blocks", "0", "self_attn", "q"] => Some(&mut self.q),
            ["blocks", "0", "ffn", "fc1"] => Some(&mut self.fc1),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        vec![
            "blocks.0.self_attn.q".to_string(),
            "blocks.0.ffn.fc1".to_string(),
        ]
    }
}

impl TinyWanHost {
    /// Build from the shared `synthetic_weights()` map (same `[16,8]` q + `[24,8]` fc1 the fold
    /// tests use), so an additive install and a folded merge start from the identical base.
    fn from_synthetic(tmp: &tempfile::TempDir) -> Self {
        let w = synthetic_weights(tmp);
        Self {
            q: AdaptableLinear::dense(
                w.require("blocks.0.self_attn.q.weight").unwrap().clone(),
                None,
            ),
            fc1: AdaptableLinear::dense(
                w.require("blocks.0.ffn.fc1.weight").unwrap().clone(),
                None,
            ),
        }
    }
}

/// f32 activations `[n, 8]` for the parity forwards (f32 in makes `addmm == matmul+add` bit-exact,
/// so the only tolerance is the low-rank residual matmul).
fn acts(n: i32) -> Array {
    Array::from_slice(
        &(0..n * 8)
            .map(|i| (i as f32 * 0.017).sin() * 0.5)
            .collect::<Vec<_>>(),
        &[n, 8],
    )
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn additive_lora_matches_folded_merge_dense() {
    let tmp = tempfile::tempdir().unwrap();
    // ACCEPTANCE: the additive residual on a DENSE base == the folded merge, same output within
    // Metal matmul tolerance. Fold into a `Weights` map via `merge_wan_adapters`, build a dense
    // linear from the merged weight, and compare its forward to the additive-installed host's.
    let lora = write_lora(
        &tmp,
        "additive_parity.safetensors",
        &[("blocks.0.self_attn.q", 16, 8), ("blocks.0.ffn.0", 24, 8)],
        4,
        0.13,
    );
    let x = acts(3);

    // Folded reference.
    let mut w = synthetic_weights(&tmp);
    let fold_report =
        merge_wan_adapters(&mut w, &[spec(lora.clone(), 0.75, None)], MoeExpert::High).unwrap();
    assert_eq!(fold_report.applied, 2);
    let folded_q = AdaptableLinear::dense(
        w.require("blocks.0.self_attn.q.weight").unwrap().clone(),
        None,
    );
    let folded_fc1 =
        AdaptableLinear::dense(w.require("blocks.0.ffn.fc1.weight").unwrap().clone(), None);

    // Additive.
    let mut host = TinyWanHost::from_synthetic(&tmp);
    let add_report =
        apply_wan_adapters_additive(&mut host, &[spec(lora, 0.75, None)], MoeExpert::High).unwrap();
    assert_eq!(add_report.applied, 2, "both q and fc1 installed additively");
    assert!(add_report.skipped.is_empty());

    for (folded, additive, name) in [
        (&folded_q, &host.q, "self_attn.q"),
        (&folded_fc1, &host.fc1, "ffn.fc1"),
    ] {
        let want = folded.forward(&x).unwrap();
        let got = additive.forward(&x).unwrap();
        assert!(
            all_close(&got, &want, 2e-2, 2e-2, false)
                .unwrap()
                .item::<bool>(),
            "additive residual must match the folded merge on {name} (dense)"
        );
        // Non-degenerate: the adapter actually changed the output vs the bare base.
        let bare = AdaptableLinear::dense(
            synthetic_weights(&tmp)
                .require(&format!(
                    "blocks.0.{}.weight",
                    if name == "self_attn.q" {
                        "self_attn.q"
                    } else {
                        "ffn.fc1"
                    }
                ))
                .unwrap()
                .clone(),
            None,
        )
        .forward(&x)
        .unwrap();
        assert!(
            !array_eq(&got, &bare, false).unwrap().item::<bool>(),
            "additive adapter must change {name}'s output (non-degenerate)"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED adapters::tests::additive_lora_matches_folded_merge_dense"
    );
}

/// A quant-sized host: `blocks.0.self_attn.q` is `[128, 64]` so its `in = 64` is a multiple of the
/// group size (64) and the base can be packed to Q4/Q8. Only the `q` target is populated.
struct QuantWanHost {
    q: AdaptableLinear,
}
impl AdaptableHost for QuantWanHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["blocks", "0", "self_attn", "q"] => Some(&mut self.q),
            _ => None,
        }
    }
    fn adaptable_paths(&self) -> Vec<String> {
        vec!["blocks.0.self_attn.q".to_string()]
    }
}
impl QuantWanHost {
    fn new() -> Self {
        let w = Array::from_slice(
            &(0..128 * 64)
                .map(|i| (i as f32 * 0.0007).sin() * 0.05)
                .collect::<Vec<_>>(),
            &[128, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        Self {
            q: AdaptableLinear::dense(w, None),
        }
    }
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn additive_lora_on_quantized_base_no_error_and_nondegenerate() {
    let tmp = tempfile::tempdir().unwrap();
    // ACCEPTANCE: a LoRA applies on a Q4/Q8 PACKED base with NO error (the old model.rs:614
    // rejection) and produces finite, non-zero output that differs from the no-adapter forward.
    // The base stays packed (never dequantized). The target is [128,64] so it is quantizable.
    let lora = write_lora(
        &tmp,
        "additive_quant.safetensors",
        &[("blocks.0.self_attn.q", 128, 64)],
        8,
        0.21,
    );
    let x = Array::from_slice(
        &(0..3 * 64)
            .map(|i| (i as f32 * 0.017).sin() * 0.5)
            .collect::<Vec<_>>(),
        &[3, 64],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();

    for bits in [8, 4] {
        let mut host = QuantWanHost::new();
        // Pack the base to Q{bits} FIRST — so the additive install writes onto a packed snapshot.
        host.q.quantize(bits, None).unwrap();
        assert!(host.q.is_quantized(), "base must be packed before install");
        let baseline = host.q.forward(&x).unwrap();

        let report = apply_wan_adapters_additive(
            &mut host,
            &[spec(lora.clone(), 0.9, None)],
            MoeExpert::High,
        )
        .unwrap();
        assert_eq!(
            report.applied, 1,
            "q installed additively on the packed base (Q{bits})"
        );
        assert!(
            host.q.is_quantized(),
            "base must STAY packed after install (Q{bits}) — no dequant of W"
        );

        let got = host.q.forward(&x).unwrap();
        // Finite + non-zero.
        let g = got.as_dtype(Dtype::Float32).unwrap();
        let vals = g.as_slice::<f32>();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "Q{bits} additive output must be finite"
        );
        assert!(
            vals.iter().any(|v| v.abs() > 1e-6),
            "Q{bits} additive output must be non-zero"
        );
        // Differs from the no-adapter forward (the residual actually fired).
        assert!(
            !array_eq(&got, &baseline, false).unwrap().item::<bool>(),
            "Q{bits} additive adapter must change the packed base's output"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::tests::additive_lora_on_quantized_base_no_error_and_nondegenerate");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn additive_no_adapter_is_byte_identical() {
    let tmp = tempfile::tempdir().unwrap();
    // ACCEPTANCE: with NO adapters installed the forward is byte-identical to the bare base — the
    // additive path adds nothing when empty (no regression to the no-adapter path).
    let x = acts(3);
    let host = TinyWanHost::from_synthetic(&tmp);
    let bare_q = AdaptableLinear::dense(
        synthetic_weights(&tmp)
            .require("blocks.0.self_attn.q.weight")
            .unwrap()
            .clone(),
        None,
    );
    let bare_out = bare_q.forward(&x).unwrap();
    let host_out = host.q.forward(&x).unwrap();
    assert!(
        array_eq(host_out, &bare_out, false).unwrap().item::<bool>(),
        "an un-adapted additive host must be byte-identical to the bare base"
    );

    // And an EMPTY spec list is a no-op install (applied 0, output unchanged).
    let mut host2 = TinyWanHost::from_synthetic(&tmp);
    let report = apply_wan_adapters_additive(&mut host2, &[], MoeExpert::High).unwrap();
    assert_eq!(report.applied, 0);
    let host2_out = host2.q.forward(&x).unwrap();
    assert!(
        array_eq(host2_out, &bare_out, false)
            .unwrap()
            .item::<bool>(),
        "empty additive install must leave the forward byte-identical"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED adapters::tests::additive_no_adapter_is_byte_identical"
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn additive_scale_zero_is_bit_exact_noop() {
    let tmp = tempfile::tempdir().unwrap();
    // A strength-0 additive LoRA is a bit-exact no-op (the residual is `0·…`), mirroring the
    // fold path's `scale_zero_is_bit_exact_noop`.
    let lora = write_lora(
        &tmp,
        "additive_zero.safetensors",
        &[("blocks.0.self_attn.q", 16, 8)],
        4,
        0.3,
    );
    let x = acts(3);
    let bare = TinyWanHost::from_synthetic(&tmp);
    let want = bare.q.forward(&x).unwrap();

    let mut host = TinyWanHost::from_synthetic(&tmp);
    let report =
        apply_wan_adapters_additive(&mut host, &[spec(lora, 0.0, None)], MoeExpert::High).unwrap();
    assert_eq!(report.applied, 1);
    let got = host.q.forward(&x).unwrap();
    assert!(
        array_eq(got, &want, false).unwrap().item::<bool>(),
        "strength 0 must leave the additive forward bit-identical"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED adapters::tests::additive_scale_zero_is_bit_exact_noop"
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn additive_lokr_matches_folded_merge_dense() {
    let tmp = tempfile::tempdir().unwrap();
    // The LoKr additive residual on a dense base matches the folded LoKr merge (same reconstructed
    // ΔW, one as `W+=δ`, the other as `x·δᵀ`). Reuses the fold tests' `write_lokr` helper.
    let lokr = write_lokr(&tmp, "additive_lokr.safetensors", 8.0, 4.0);
    let x = acts(3);

    // Folded reference: merge onto the q weight, forward the merged dense linear.
    let mut w = synthetic_weights(&tmp);
    let fold =
        merge_wan_adapters(&mut w, &[lokr_spec(lokr.clone(), 0.6)], MoeExpert::High).unwrap();
    assert_eq!(fold.applied, 1);
    let folded_q = AdaptableLinear::dense(
        w.require("blocks.0.self_attn.q.weight").unwrap().clone(),
        None,
    );

    // Additive.
    let mut host = TinyWanHost::from_synthetic(&tmp);
    let add_report =
        apply_wan_adapters_additive(&mut host, &[lokr_spec(lokr, 0.6)], MoeExpert::High).unwrap();
    assert_eq!(add_report.applied, 1);

    let want = folded_q.forward(&x).unwrap();
    let got = host.q.forward(&x).unwrap();
    assert!(
        all_close(&got, &want, 2e-2, 2e-2, false)
            .unwrap()
            .item::<bool>(),
        "additive LoKr residual must match the folded LoKr merge (dense)"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED adapters::tests::additive_lokr_matches_folded_merge_dense"
    );
}

// ---- Structured (deferred) LoKr on a PACKED base (sc-10050) ----------------------------------

/// Write a peft LoKr for `blocks.0.self_attn.q` sized to a **quantizable** `[128,64]` base
/// (`in = 64` is a multiple of the group size): `[128,64] = kron(w1[16,8], w2[8,8])`. `alpha`/`rank`
/// in metadata. Deterministic factor values.
fn write_lokr_quant(tmp: &tempfile::TempDir, name: &str, alpha: f32, rank: f32) -> PathBuf {
    use std::collections::HashMap;
    let w1 = Array::from_slice(
        &(0..16 * 8)
            .map(|i| (i as f32 * 0.013).sin() * 0.08)
            .collect::<Vec<_>>(),
        &[16, 8],
    );
    let w2 = Array::from_slice(
        &(0..8 * 8)
            .map(|i| (i as f32 * 0.021).cos() * 0.08)
            .collect::<Vec<_>>(),
        &[8, 8],
    );
    let meta = HashMap::from([
        ("networkType".to_string(), "lokr".to_string()),
        ("alpha".to_string(), alpha.to_string()),
        ("rank".to_string(), rank.to_string()),
    ]);
    let path = scratch_file(tmp, name);
    Array::save_safetensors(
        vec![
            ("blocks.0.self_attn.q.lokr_w1", &w1),
            ("blocks.0.self_attn.q.lokr_w2", &w2),
        ],
        Some(&meta),
        &path,
    )
    .unwrap();
    path
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn structured_lokr_on_quantized_matches_dense_and_stays_packed() {
    let tmp = tempfile::tempdir().unwrap();
    // ACCEPTANCE (sc-10050): a peft LoKr applies on a Q4/Q8 PACKED base with NO error, NO full
    // out×in delta materialized, the base STAYS packed, and its LoKr CONTRIBUTION (packed-with-LoKr
    // minus packed-baseline) matches the dense LoKr residual within tolerance. Comparing the
    // residual contribution — not the full output — isolates the adapter from the (much larger)
    // base-quantization error, which is what the vec-trick vs the materialized delta must agree on.
    // `[128,64]` target so `in` is a group-size multiple.
    let lokr = write_lokr_quant(&tmp, "structured_quant_lokr.safetensors", 8.0, 4.0);
    let x = Array::from_slice(
        &(0..3 * 64)
            .map(|i| (i as f32 * 0.017).sin() * 0.5)
            .collect::<Vec<_>>(),
        &[3, 64],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();

    let dense_w = Array::from_slice(
        &(0..128 * 64)
            .map(|i| (i as f32 * 0.0007).sin() * 0.05)
            .collect::<Vec<_>>(),
        &[128, 64],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();

    // Dense reference LoKr residual = (dense base + materialized LoKr) − (dense base). This is the
    // established materialized-delta path (the `install_one_lokr_additive` dense branch).
    let dense_base = AdaptableLinear::dense(dense_w.clone(), None);
    let dense_baseline = dense_base.forward(&x).unwrap();
    let mut dense_host = QuantWanHost {
        q: AdaptableLinear::dense(dense_w.clone(), None),
    };
    apply_wan_adapters_additive(
        &mut dense_host,
        &[lokr_spec(lokr.clone(), 0.6)],
        MoeExpert::High,
    )
    .unwrap();
    let dense_with_lokr = dense_host.q.forward(&x).unwrap();
    let dense_residual = mlx_rs::ops::subtract(&dense_with_lokr, &dense_baseline).unwrap();

    for bits in [8, 4] {
        let mut host = QuantWanHost {
            q: AdaptableLinear::dense(dense_w.clone(), None),
        };
        host.q.quantize(bits, None).unwrap();
        assert!(host.q.is_quantized(), "base must be packed before install");
        let baseline = host.q.forward(&x).unwrap();

        let report = apply_wan_adapters_additive(
            &mut host,
            &[lokr_spec(lokr.clone(), 0.6)],
            MoeExpert::High,
        )
        .unwrap();
        assert_eq!(report.applied, 1, "LoKr installed structurally on Q{bits}");
        assert!(report.skipped.is_empty());
        assert!(
            host.q.is_quantized(),
            "base must STAY packed after structured LoKr install (Q{bits})"
        );

        // The installed adapter is the STRUCTURED variant whose factors are the SMALL Kronecker
        // matrices — never an [out=128, in=64] delta. Assert the memory/no-materialization property
        // structurally: the only stored tensors are [16,8] (w1) and [8,8] (w2).
        match host.q.adapters() {
            [Adapter::LokrStructured { factors }] => {
                assert_eq!(
                    factors.w1.shape(),
                    &[16, 8],
                    "w1 must be the small [a,c] factor, not [out,in]"
                );
                assert_eq!(
                    factors.w2.shape(),
                    &[8, 8],
                    "w2 must be the small [b,d] factor, not [out,in]"
                );
                // Neither factor is the [out,in]=[128,64] delta.
                assert_ne!(factors.w1.shape(), &[128, 64]);
                assert_ne!(factors.w2.shape(), &[128, 64]);
            }
            other => panic!(
                "expected a single structured LoKr on the packed base, got {} adapter(s)",
                other.len()
            ),
        }

        let got = host.q.forward(&x).unwrap();
        // Finite + non-degenerate (differs from the no-adapter forward).
        let g = got.as_dtype(Dtype::Float32).unwrap();
        assert!(
            g.as_slice::<f32>().iter().all(|v| v.is_finite()),
            "Q{bits} structured LoKr output must be finite"
        );
        assert!(
            !array_eq(&got, &baseline, false).unwrap().item::<bool>(),
            "Q{bits} structured LoKr must change the packed base's output"
        );
        // The structured LoKr CONTRIBUTION matches the dense materialized-delta residual. The
        // residual runs over the packed base's bf16 activation stream (same as the dense path here),
        // so the only difference is the vec-trick vs the materialized delta — Metal matmul tolerance.
        let quant_residual = mlx_rs::ops::subtract(&got, &baseline).unwrap();
        // The vec-trick (two small bf16 matmuls) vs the materialized delta (one bf16 matmul) round
        // differently; the residual itself is bf16, so allow the bf16-residual reduced-precision band.
        assert!(
            all_close(&quant_residual, &dense_residual, 5e-2, 8e-3, false)
                .unwrap()
                .item::<bool>(),
            "Q{bits} structured LoKr contribution must match the dense materialized-delta residual"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED adapters::tests::structured_lokr_on_quantized_matches_dense_and_stays_packed");
}

// ---- Packed-snapshot routing guard (sc-10045 / narrowed by sc-10050) ------------------------

#[test]
fn reject_loha_on_packed_passes_plain_lora_and_lokr() {
    let tmp = tempfile::tempdir().unwrap();
    // Plain-LoRA AND LoKr spec lists are allowed on a packed tier (sc-10050): LoRA installs
    // additively, LoKr via the structured deferred-Kronecker path — the guard is a no-op for both.
    let l1 = write_lora(
        &tmp,
        "packroute_lora1.safetensors",
        &[("blocks.0.self_attn.q", 16, 8)],
        4,
        0.1,
    );
    let l2 = write_lora(
        &tmp,
        "packroute_lora2.safetensors",
        &[("blocks.0.ffn.0", 24, 8)],
        4,
        0.4,
    );
    reject_loha_on_packed(
        "wan2_2_t2v_14b",
        &[spec(l1, 1.0, None), spec(l2, 1.0, Some(MoeExpert::High))],
    )
    .expect("plain LoRA files must pass the packed guard");
    // A peft LoKr file (networkType=lokr stamp) now ALSO passes — it installs via the structured
    // vec-trick (sc-10050), no longer rejected.
    let lokr = write_lokr(&tmp, "packroute_lokr.safetensors", 8.0, 4.0);
    reject_loha_on_packed("wan2_2_t2v_14b", &[lokr_spec(lokr, 0.6)])
        .expect("LoKr must pass the packed guard now that sc-10050 applies it structurally");
    // Third-party LoKr (no networkType stamp — detected by keys) likewise passes.
    let tp_lokr = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests/fixtures/sc3642_lokr/linear_w1full_w2lr.safetensors");
    reject_loha_on_packed("wan2_2_i2v_14b", &[spec(tp_lokr, 1.0, None)])
        .expect("third-party LoKr must pass the packed guard (structural apply, sc-10050)");
}

#[test]
fn reject_loha_on_packed_errors_on_thirdparty_loha() {
    // sc-10051 (Option A): the committed third-party LyCORIS LoHa fixture (no networkType stamp —
    // detected by keys) is rejected on a packed tier with a CLEAR, ACTIONABLE, TYPED error. It
    // must NOT panic, NOT succeed, and NOT silently materialize the dense delta. Declared as LoRA
    // kind so the guard catches it by KEYS.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests/fixtures/sc3643_loha/linear.safetensors");
    let err = reject_loha_on_packed("wan2_2_i2v_14b", &[spec(path, 1.0, None)])
        .expect_err("third-party LoHa must be rejected on a packed tier");
    // Distinct typed variant the worker matches on (bridges 1:1 to gen_core::Error::Unsupported),
    // NOT a generic Msg — so the worker can surface steer-to-bf16 guidance, not an opaque failure.
    assert!(
        matches!(err, Error::Unsupported(_)),
        "LoHa-on-packed must be a typed Unsupported error, got: {err:?}"
    );
    let msg = err.to_string();
    // Actionable: names the family, the mechanism (Hadamard), and the fix (bf16 tier), + story.
    assert!(
        msg.contains("bf16") && msg.contains("sc-10051"),
        "error must point at bf16 + the LoHa deferral story (sc-10051), got: {msg}"
    );
    assert!(msg.contains("LoHa"), "error must name LoHa, got: {msg}");
    assert!(
        msg.contains("Hadamard"),
        "error must explain the Hadamard mechanism, got: {msg}"
    );
    assert!(
        msg.contains("bf16 tier"),
        "error must steer the user to the bf16 tier, got: {msg}"
    );
    assert!(
        !msg.contains("not yet wired"),
        "must NOT be the old generic 'not yet wired' message, got: {msg}"
    );
}

#[test]
fn reject_loha_on_packed_unsupported_bridges_to_gen_core() {
    // The typed Unsupported must survive the mlx-gen → gen-core seam 1:1 (epic 3720, F-008) so the
    // worker's Unsupported classification (not a generic backend failure) fires with the full
    // steer-to-bf16 text intact.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests/fixtures/sc3643_loha/linear.safetensors");
    let err = reject_loha_on_packed("wan2_2_t2v_14b", &[spec(path, 1.0, None)])
        .expect_err("LoHa on a packed tier must error");
    let bridged: mlx_gen::gen_core::Error = err.into();
    match bridged {
        mlx_gen::gen_core::Error::Unsupported(s) => {
            assert!(
                s.contains("bf16 tier") && s.contains("LoHa"),
                "bridged gen-core Unsupported lost the actionable text: {s}"
            );
        }
        other => panic!("LoHa-on-packed degraded across the seam to {other:?}"),
    }
}

#[test]
fn additive_skips_target_absent_from_host() {
    let tmp = tempfile::tempdir().unwrap();
    // A LoRA target outside the host's adaptable surface (a non-block / far-block module) is
    // surfaced (skipped), never fatal — mirroring the fold path's skip reporting.
    let lora = write_lora(
        &tmp,
        "additive_skip.safetensors",
        &[
            ("blocks.0.self_attn.q", 16, 8),
            ("blocks.99.self_attn.q", 16, 8),
        ],
        4,
        0.1,
    );
    let mut host = TinyWanHost::from_synthetic(&tmp);
    let report =
        apply_wan_adapters_additive(&mut host, &[spec(lora, 1.0, None)], MoeExpert::High).unwrap();
    assert_eq!(report.applied, 1, "the present target installs");
    assert_eq!(
        report.skipped,
        vec!["blocks.99.self_attn.q".to_string()],
        "the absent target is surfaced, not fatal"
    );
}

#[test]
fn read_alpha_empty_tensor_errors_not_panic() {
    // F-059 (sc-11129): a size-0 `.alpha` from a malformed/truncated adapter file must produce a
    // TYPED error, not the `as_slice()[0]` panic that would abort the worker on any Wan entry.
    // (safetensors cannot serialize an empty array, so exercise `read_alpha` directly on the
    // in-memory empty tensor — the exact value a truncated file would deserialize to.)
    let empty = Array::from_slice::<f32>(&[], &[0]);
    let err = read_alpha(&empty).expect_err("an empty .alpha must error, not panic");
    assert!(
        matches!(err, Error::Msg(_)),
        "expected a typed Msg error, got {err:?}"
    );
    // Happy-path regression: a well-formed bf16 scalar still reads its value.
    let good = Array::from_slice(&[8.0f32], &[1])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    assert_eq!(read_alpha(&good).unwrap(), 8.0);
}

/// Write a PEFT LoRA file for `blocks.0.self_attn.q` whose factors are **1-D** (non-2-D) — a
/// malformed adapter that would otherwise derive a degenerate rank and merge garbage (sc-11129,
/// F-058). A true 0-size `lora_A` leading dim triggers the same guard but is not serializable.
fn write_lora_non_2d(tmp: &tempfile::TempDir, name: &str) -> PathBuf {
    let a = Array::from_slice(&[0.1f32; 8], &[8])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let b = Array::from_slice(&[0.1f32; 16], &[16])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let path = scratch_file(tmp, name);
    Array::save_safetensors(
        vec![
            ("diffusion_model.blocks.0.self_attn.q.lora_A.weight", &a),
            ("diffusion_model.blocks.0.self_attn.q.lora_B.weight", &b),
        ],
        None,
        &path,
    )
    .unwrap();
    path
}

#[test]
fn fold_path_rejects_degenerate_factors_instead_of_nan_merge() {
    let tmp = tempfile::tempdir().unwrap();
    // F-058 (sc-11129): the fold path must reject degenerate factors with a typed error, exactly
    // like the additive twin's guard (`install_one_lora_additive`) — rather than deriving a
    // NaN/degenerate `alpha/rank` and silently replacing the whole base weight. Before the fix the
    // fold path had no such guard; the additive twin did (fixes-don't-travel).
    let lora = write_lora_non_2d(&tmp, "f058_non2d.safetensors");
    let mut w = synthetic_weights(&tmp);
    let err = merge_wan_adapters(&mut w, &[spec(lora, 1.0, None)], MoeExpert::High)
        .expect_err("degenerate factors must be rejected on the fold path");
    match err {
        Error::Msg(s) => assert!(
            s.contains("non-2-D") || s.contains("zero rank"),
            "unexpected error text: {s}"
        ),
        other => panic!("expected a typed Msg error, got {other:?}"),
    }
    // The base weight is untouched (not corrupted) — the guard fired before the merge.
    let got = w.require("blocks.0.self_attn.q.weight").unwrap();
    assert!(
        !got.as_dtype(Dtype::Float32)
            .unwrap()
            .sum(None)
            .unwrap()
            .item::<f32>()
            .is_nan(),
        "the base weight must not be NaN-poisoned when the guard rejects"
    );
}

/// Write a synthetic third-party LyCORIS **LoHa** for `blocks.0.self_attn.q` at `[128,64]`
/// (`(w1_a·w1_b) ⊙ (w2_a·w2_b)`, rank 4) — the shape `QuantWanHost` carries (sc-11129, F-060).
fn write_thirdparty_loha_q(tmp: &tempfile::TempDir, name: &str) -> PathBuf {
    let r = 4;
    let wa = Array::from_slice(
        &(0..128 * r)
            .map(|i| (i as f32 * 0.001).sin() * 0.05)
            .collect::<Vec<_>>(),
        &[128, r],
    );
    let wb = Array::from_slice(
        &(0..r * 64)
            .map(|i| (i as f32 * 0.002).cos() * 0.05)
            .collect::<Vec<_>>(),
        &[r, 64],
    );
    let path = scratch_file(tmp, name);
    Array::save_safetensors(
        vec![
            ("blocks.0.self_attn.q.hada_w1_a", &wa),
            ("blocks.0.self_attn.q.hada_w1_b", &wb),
            ("blocks.0.self_attn.q.hada_w2_a", &wa),
            ("blocks.0.self_attn.q.hada_w2_b", &wb),
        ],
        None,
        &path,
    )
    .unwrap();
    path
}

#[test]
fn loha_additive_installer_rejects_packed_base_per_target() {
    let tmp = tempfile::tempdir().unwrap();
    // F-060 (sc-11129): the LoHa additive installer must refuse a materialization on a PACKED base
    // per target (not rely solely on the call-site file-level `reject_loha_on_packed`). Install a
    // third-party LoHa directly via `apply_wan_adapters_additive` onto a packed q — it must return
    // the typed `Error::Unsupported`, and on a DENSE q the same file installs fine (guard parity).
    let loha = write_thirdparty_loha_q(&tmp, "f060_loha.safetensors");

    let mut packed = QuantWanHost::new();
    packed.q.quantize(8, Some(64)).unwrap();
    assert!(
        packed.q.is_quantized(),
        "base must be packed before install"
    );
    let err = apply_wan_adapters_additive(
        &mut packed,
        &[spec(loha.clone(), 1.0, None)],
        MoeExpert::High,
    )
    .expect_err("a LoHa on a packed base must be rejected per target");
    assert!(
        matches!(err, Error::Unsupported(_)),
        "LoHa-on-packed must be a typed Unsupported error, got {err:?}"
    );

    // Guard parity: on a DENSE base the identical file installs (materialized), never rejected.
    let mut dense = QuantWanHost::new();
    let report = apply_wan_adapters_additive(&mut dense, &[spec(loha, 1.0, None)], MoeExpert::High)
        .expect("a LoHa on a dense base must still install");
    assert_eq!(report.applied, 1, "the LoHa materializes on a dense base");
}
