//! sc-21485 — FLUX.2 Klein **universal NVFP4 single-file import** real-weight gates (epic 11037).
//!
//! Both rows are `#[ignore]`d: they need the real pinned artifact
//! (`wikeeyang/Flux2-Klein-9B-True-V2@9c9fe988… / Flux2-Klein-9B-True-v2-nvfp4mixed.safetensors`,
//! SHA-256 `32ab8333…`), a resident `black-forest-labs/FLUX.2-klein-9B` diffusers snapshot for the
//! TE / VAE / tokenizer, and a CUDA device. Inference never self-fetches or derives a cache
//! location (epic 13657) — every path arrives via env:
//!
//! ```sh
//! CANDLE_FLUX2_SNAPSHOT=E:\huggingface\hub\models--black-forest-labs--FLUX.2-klein-9B\snapshots\<rev> \
//! CANDLE_FLUX2_TRUE_V2_NVFP4_FILE=E:\huggingface\hub\models--wikeeyang--Flux2-Klein-9B-True-V2\snapshots\<rev>\Flux2-Klein-9B-True-v2-nvfp4mixed.safetensors \
//!   cargo test -p candle-gen-flux2 --release --features cuda --test integration \
//!     klein_single_file_real_weights:: -- --ignored --nocapture --test-threads 1
//! ```
//!
//! * [`linked_and_managed_copies_share_one_plan_and_one_render`] is the story's headline AC: the
//!   HF-cache ("managed") path and a plain-file ("linked") copy of the SAME artifact compile the
//!   same semantic plan and, through the registered `flux2_klein_9b` provider (NOT `flux2_dev`),
//!   render byte-identical fixed-seed images.
//! * [`adapter_inheritance_composes_over_the_packed_projections`] proves LoRA inheritance over the
//!   NVFP4-packed rows: a synthetic LoRA targeting packed projections loads through the same route,
//!   changes the fixed-seed output, and never silently no-ops.

// The whole module is cuda-gated (sc-11045 fix round, MAJOR 8): every test needs a CUDA device,
// and the CPU `--all-targets` lane must not even compile the render harness.
#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use candle_gen::gen_core::checkpoint_codec::NVFP4_CODEC;
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, Progress,
    WeightsSource, BASE_SNAPSHOT_COMPONENT,
};
use candle_gen::logical_weights::{plan_logical_weights, CandleCodecResidency};
use candle_gen_flux2::config::Flux2Variant;
use candle_gen_flux2::single_file::plan_semantic_summary;
use candle_gen_flux2::Flux2BflToDiffusersMapping;

fn base_snapshot() -> PathBuf {
    PathBuf::from(std::env::var("CANDLE_FLUX2_SNAPSHOT").expect(
        "set CANDLE_FLUX2_SNAPSHOT to a black-forest-labs/FLUX.2-klein-9B diffusers snapshot dir",
    ))
}

fn managed_nvfp4_file() -> PathBuf {
    PathBuf::from(std::env::var("CANDLE_FLUX2_TRUE_V2_NVFP4_FILE").expect(
        "set CANDLE_FLUX2_TRUE_V2_NVFP4_FILE to the wikeeyang \
         Flux2-Klein-9B-True-v2-nvfp4mixed.safetensors file (the managed HF-cache copy)",
    ))
}

/// Stage a "linked" copy of the managed artifact under a **TempDir guard** (sc-11045 fix round,
/// MAJOR 8 — no raw `temp_dir` litter, and cleanup survives a panicking assertion): a hardlink in
/// a temp dir beside the managed copy (same volume — byte-identical by construction, no 5.6 GB
/// duplication; and NOT a junction/symlink, which candle refuses, os error 448), else a copy.
fn stage_linked_copy(managed: &Path) -> (tempfile::TempDir, PathBuf) {
    let stage = tempfile::Builder::new()
        .prefix("sc21485-linked-")
        .tempdir_in(managed.parent().expect("the managed artifact has a parent"))
        .expect("temp dir beside the managed copy");
    let linked = stage
        .path()
        .join("sc21485-linked-Flux2-Klein-9B-True-v2-nvfp4mixed.safetensors");
    if std::fs::hard_link(managed, &linked).is_err() {
        std::fs::copy(managed, &linked).expect("copy the linked stand-in");
    }
    (stage, linked)
}

fn spec_for(file: &Path, adapters: Vec<AdapterSpec>) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::File(file.to_path_buf()))
        .with_component(BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base_snapshot()))
        .with_adapters(adapters);
    spec.prepare_file_sources().expect("prepare file pins");
    spec
}

fn render(file: &Path, adapters: Vec<AdapterSpec>) -> Vec<u8> {
    let spec = spec_for(file, adapters);
    let registry = candle_gen_flux2::provider_registry().expect("registry");
    // Through the registered provider — the identity half of the AC: the single file loads as
    // `flux2_klein_9b`, never as `flux2_dev`.
    let gen = registry
        .load("flux2_klein_9b", &spec)
        .expect("klein single-file load");
    assert_eq!(gen.descriptor().id, "flux2_klein_9b");
    let req = GenerationRequest {
        prompt: "a photo of a rusty robot holding a lit candle, dramatic lighting".into(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(4242),
        steps: Some(4),
        ..Default::default()
    };
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            eprintln!("[sc-21485] step {current}/{total}");
        }
    };
    let output = gen.generate(&req, &mut on_progress).expect("render");

    // sc-11045 fix round (MAJOR 8c / BLOCKER 1): the facts cross the worker boundary — read off
    // the SAME `Box<dyn Generator>` a worker holds, i.e. through the trait surface, after the DiT
    // materialized. This is the assertion the shadowed inherent method used to defeat.
    let facts = gen
        .checkpoint_weight_facts()
        .expect("a klein single-file load publishes its checkpoint facts to the trait surface");
    assert!(
        facts.source().declares(NVFP4_CODEC.codec_id),
        "the pinned artifact stores nvfp4-v1"
    );
    assert!(
        facts.is_complete(),
        "the render materialized the whole DiT, so the receipt must cover the plan"
    );
    assert!(facts.resident_bytes() > 0);
    let device = candle_gen::default_device().expect("device");
    if CandleCodecResidency::probe(&device).nvfp4_native {
        assert!(
            facts.executes_natively(NVFP4_CODEC.codec_id),
            "on sm_120 the mixed policy must keep the compute bulk genuinely packed"
        );
        assert!(
            !facts.receipt().demotions.is_empty(),
            "the W4A16 outlier class must be demoted in the receipt, never labelled native"
        );
    } else {
        assert!(
            !facts.executes_natively(NVFP4_CODEC.codec_id),
            "below the sm_120 floor nothing may be labelled native"
        );
    }

    let GenerationOutput::Images(images) = output else {
        panic!("expected images");
    };
    assert_eq!(images.len(), 1);
    images.into_iter().next().unwrap().pixels
}

#[test]
#[ignore = "needs the pinned NVFP4 klein single file + base snapshot + CUDA (env-driven)"]
fn linked_and_managed_copies_share_one_plan_and_one_render() {
    let managed = managed_nvfp4_file();
    let (_stage, linked) = stage_linked_copy(&managed);

    // Same SEMANTIC plan, independent of which path the bytes load from: mapping id plus the
    // (codec, residency) → logical-key rows, compiled under the same policy the provider loads
    // with (the probed floors, fp8 leg masked — MAJOR 10).
    let device = candle_gen::default_device().expect("device");
    let residency = CandleCodecResidency::probe(&device).with_dense_fp8();
    let cfg = Flux2Variant::Klein9b.config();
    let mapping = Flux2BflToDiffusersMapping::new(&cfg);
    let plan_managed = plan_logical_weights(&managed, &mapping, &residency).expect("managed plan");
    let plan_linked = plan_logical_weights(&linked, &mapping, &residency).expect("linked plan");
    let summary = plan_semantic_summary(&plan_managed);
    assert_eq!(summary, plan_semantic_summary(&plan_linked));
    let rows: Vec<(&str, &str, usize)> = summary
        .1
        .iter()
        .map(|(codec, mode, keys)| (codec.as_str(), mode.as_str(), keys.len()))
        .collect();
    // The pinned artifact really exercises the packed lane: it must plan NVFP4 rows…
    assert!(
        rows.iter().any(|(codec, _, _)| *codec == "nvfp4-v1"),
        "the pinned artifact must plan nvfp4-v1 rows; got {rows:?}"
    );
    // …and on this box's sm_120 device those rows must be priced **Packed** (sc-11045 fix round,
    // MAJOR 8c: the residency MODE, not merely the codec's presence — a dense-only pricing of the
    // same artifact would satisfy the weaker check while proving nothing about the packed lane).
    if CandleCodecResidency::probe(&device).nvfp4_native {
        assert!(
            rows.iter()
                .any(|(codec, mode, keys)| *codec == "nvfp4-v1" && *mode == "Packed" && *keys > 0),
            "on sm_120 the pinned artifact must price packed nvfp4-v1 rows; got {rows:?}"
        );
    }

    let a = render(&managed, Vec::new());
    let b = render(&linked, Vec::new());
    assert_eq!(
        a, b,
        "linked vs managed fixed-seed renders must be byte-identical"
    );
}

/// Deterministic tiny LoRA over two NVFP4-packed projections (a double-block `to_q` and the fused
/// single-block `to_qkv_mlp_proj`), written on the fly. Non-trivial magnitudes so the render must
/// visibly change; rank 4.
fn write_synthetic_lora(dir: &std::path::Path) -> PathBuf {
    use std::collections::BTreeMap;
    let inner = 4096usize;
    let single_out = 3 * inner + 2 * 12288;
    let rank = 4usize;
    let f32_bytes =
        |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let ramp = |n: usize, scale: f32| -> Vec<f32> {
        (0..n).map(|i| ((i % 17) as f32 - 8.0) * scale).collect()
    };
    let payloads: Vec<(String, Vec<usize>, Vec<u8>)> = vec![
        (
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight".into(),
            vec![rank, inner],
            f32_bytes(&ramp(rank * inner, 0.02)),
        ),
        (
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight".into(),
            vec![inner, rank],
            f32_bytes(&ramp(inner * rank, 0.02)),
        ),
        (
            "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight".into(),
            vec![rank, inner],
            f32_bytes(&ramp(rank * inner, 0.02)),
        ),
        (
            "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.weight".into(),
            vec![single_out, rank],
            f32_bytes(&ramp(single_out * rank, 0.02)),
        ),
    ];
    let tensors: BTreeMap<&str, ::safetensors::tensor::TensorView<'_>> = payloads
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                ::safetensors::tensor::TensorView::new(
                    ::safetensors::Dtype::F32,
                    shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        })
        .collect();
    let path = dir.join("sc21485-synthetic-lora.safetensors");
    ::safetensors::serialize_to_file(tensors, None, &path).unwrap();
    path
}

/// Deterministic tiny **LoKr** over the same NVFP4-packed double-block `to_q` (AC3 names LoRA and
/// LoKr; both ride the one `AdaptLinear` additive seam, sc-11091 / sc-21483).
///
/// `to_q` is `[4096, 4096]`, so the Kronecker factorisation is `out = 64·64`, `in = 64·64` — two
/// `[64, 64]` full factors, no low-rank split. PEFT-stamped metadata (`networkType=lokr`) is what
/// the loader's declared-kind check reads, so it is written here rather than inferred.
fn write_synthetic_lokr(dir: &std::path::Path) -> PathBuf {
    use std::collections::BTreeMap;
    let side = 64usize;
    let f32_bytes =
        |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let ramp = |n: usize, scale: f32| -> Vec<f32> {
        (0..n).map(|i| ((i % 13) as f32 - 6.0) * scale).collect()
    };
    let payloads: Vec<(String, Vec<usize>, Vec<u8>)> = vec![
        (
            "transformer.transformer_blocks.0.attn.to_q.lokr_w1".into(),
            vec![side, side],
            f32_bytes(&ramp(side * side, 0.05)),
        ),
        (
            "transformer.transformer_blocks.0.attn.to_q.lokr_w2".into(),
            vec![side, side],
            f32_bytes(&ramp(side * side, 0.05)),
        ),
    ];
    let tensors: BTreeMap<&str, ::safetensors::tensor::TensorView<'_>> = payloads
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                ::safetensors::tensor::TensorView::new(
                    ::safetensors::Dtype::F32,
                    shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        })
        .collect();
    let metadata = std::collections::HashMap::from([
        ("networkType".to_string(), "lokr".to_string()),
        ("rank".to_string(), "1".to_string()),
        ("alpha".to_string(), "1".to_string()),
    ]);
    let path = dir.join("sc21485-synthetic-lokr.safetensors");
    ::safetensors::serialize_to_file(tensors, Some(metadata), &path).unwrap();
    path
}

#[test]
#[ignore = "needs the pinned NVFP4 klein single file + base snapshot + CUDA (env-driven)"]
fn adapter_inheritance_composes_over_the_packed_projections() {
    let managed = managed_nvfp4_file();
    let tmp = tempfile::tempdir().expect("temp dir");
    let lora = write_synthetic_lora(tmp.path());
    let lokr = write_synthetic_lokr(tmp.path());

    let base = render(&managed, Vec::new());
    let adapted = render(
        &managed,
        vec![AdapterSpec {
            path: lora,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    );
    assert_eq!(base.len(), adapted.len());
    assert_ne!(
        base, adapted,
        "a LoRA over the packed projections must change the fixed-seed render \
         (an unchanged render is the silent-un-adapted class, epic E6)"
    );

    // AC3's other half. The LoKr residual is the deferred Kronecker-vector identity — it never
    // materializes the `[out, in]` delta — so "it loaded" is NOT evidence it applied; only a
    // changed fixed-seed render is.
    let lokr_adapted = render(
        &managed,
        vec![AdapterSpec {
            path: lokr,
            scale: 1.0,
            kind: AdapterKind::Lokr,
            pass_scales: None,
            moe_expert: None,
        }],
    );
    assert_eq!(base.len(), lokr_adapted.len());
    assert_ne!(
        base, lokr_adapted,
        "a LoKr over the packed projections must change the fixed-seed render"
    );
    assert_ne!(
        adapted, lokr_adapted,
        "the LoRA and LoKr arms must not collapse onto one render — that would mean neither \
         residual is actually keyed to its own factors"
    );
}
