//! Tier-snapshot tests on a synthetic YuE2 (sc-22995) — no real weights, so they run in every CI
//! lane (the audio family runs `--lib`).
//!
//! The fixture is a complete synthetic "original": the tiny real-architecture MoT with NAR heads
//! ([`crate::nar::synthetic`]) saved as BF16, a `config.json`, licence/notice/model-card files and
//! a tokenizer file, all pinned by a synthetic [`Component`] pair exactly like the real inventory.
//! Every test drives the production entry points ([`convert_from`], [`verify_tier_from`],
//! [`crate::model::open_tier`] → [`Yue2Nar::from_opened`]); only the source components differ from
//! the pinned YuE2-3B.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_audio::candle_core::quantized::GgmlDType;
use candle_audio::candle_core::DType;

use super::*;
use crate::inventory::{Component, ComponentId, FileRole, PinnedFile, UpstreamRepo};
use crate::model::synthetic as mot;
use crate::nar::{synthetic as heads, Yue2Nar};
use crate::precision::{Backend, TensorClass};

const REPO: UpstreamRepo = UpstreamRepo {
    id: "m-a-p/YuE2-Synthetic-Tier",
    revision: "fedcba9876543210fedcba9876543210fedcba98",
    gated: false,
    card_license: "cc-by-nc-4.0",
};

fn sha(bytes: &[u8]) -> String {
    crate::engine::hex(&Sha256::digest(bytes))
}

fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

struct Source {
    tmp: tempfile::TempDir,
    dirs: SnapshotDirs,
    source: TierSource,
    /// The BF16 values of every tensor (what the original holds), for reference models.
    tensors: HashMap<String, Tensor>,
}

impl Source {
    fn out(&self, name: &str) -> PathBuf {
        self.tmp.path().join(name)
    }
}

fn config_json() -> String {
    let c = mot::config();
    json!({
        "model_type": "yue2",
        "tie_word_embeddings": false,
        "latent_type": "vae",
        "hidden_size": c.hidden_size,
        "num_hidden_layers": c.num_hidden_layers,
        "num_attention_heads": c.num_attention_heads,
        "num_key_value_heads": c.num_key_value_heads,
        "head_dim": c.head_dim,
        "intermediate_size": c.intermediate_size,
        "vocab_size": c.vocab_size,
        "rms_norm_eps": c.rms_norm_eps,
        "rope_theta": c.rope_theta,
        "max_position_embeddings": c.max_position_embeddings,
        "latent_dim": crate::latent::LATENT_CHANNELS,
        "max_latent_frames": heads::MAX_LATENT_FRAMES,
        "timestep_shift": 1.0,
    })
    .to_string()
}

fn pin(path: &'static str, bytes: &[u8], role: FileRole) -> PinnedFile {
    PinnedFile {
        path,
        bytes: bytes.len() as u64,
        sha256: leak_str(sha(bytes)),
        role,
    }
}

fn manifest(key: &str, files: &[PinnedFile], native: &PinnedFile, extra: Value) -> &'static str {
    let source_files: Map<String, Value> = files
        .iter()
        .map(|f| {
            (
                f.path.to_string(),
                json!({"bytes": f.bytes, "sha256": f.sha256}),
            )
        })
        .collect();
    let mut m = json!({
        "schema": 1,
        "component": key,
        "source": {"repo": REPO.id, "revision": REPO.revision, "files": source_files},
        "conversion": {"kind": "identity"},
        "native": {"file": native.path, "bytes": native.bytes, "sha256": native.sha256},
    });
    for (k, v) in extra.as_object().unwrap() {
        m[k] = v.clone();
    }
    leak_str(m.to_string())
}

/// The synthetic original, provisioned under [`REPO`].
fn source() -> Source {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("YuE2-Synthetic");
    std::fs::create_dir_all(dir.join("licenses")).unwrap();
    let tensors: HashMap<String, Tensor> = heads::all_tensors()
        .into_iter()
        .map(|(n, t)| (n, t.to_dtype(DType::BF16).unwrap()))
        .collect();
    let weights_path = dir.join("model.safetensors");
    candle_core::safetensors::save(&tensors, &weights_path).unwrap();
    let weights = std::fs::read(&weights_path).unwrap();
    let config = config_json().into_bytes();
    let license = b"Attribution-NonCommercial 4.0 International (synthetic fixture)\n".to_vec();
    let card = b"---\nlicense: cc-by-nc-4.0\n---\nsynthetic\n".to_vec();
    let notice = b"MIT (synthetic third-party notice)\n".to_vec();
    let tiktoken = std::fs::read(crate::test_fixtures::dir().join("synthetic.tiktoken")).unwrap();
    for (rel, bytes) in [
        ("config.json", &config),
        ("LICENSE", &license),
        ("README.md", &card),
        ("licenses/NOTICE.txt", &notice),
        ("qwen.tiktoken", &tiktoken),
    ] {
        std::fs::write(dir.join(rel), bytes).unwrap();
    }
    let lm_files = vec![
        pin("config.json", &config, FileRole::Config),
        pin("model.safetensors", &weights, FileRole::Weights),
        pin("README.md", &card, FileRole::ModelCard),
        pin("LICENSE", &license, FileRole::License),
        pin("licenses/NOTICE.txt", &notice, FileRole::License),
    ];
    let mut rows: Vec<Value> = tensors
        .iter()
        .map(|(name, t)| {
            json!({
                "name": name,
                "dtype": "BF16",
                "shape": t.dims(),
                "sha256": tensor_digest(t).unwrap(),
            })
        })
        .collect();
    rows.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    let lm_manifest = manifest(
        "yue2_synthetic_tier",
        &lm_files,
        &lm_files[1],
        json!({"tensors": rows}),
    );
    let tok_files = vec![pin("qwen.tiktoken", &tiktoken, FileRole::Tokenizer)];
    let lines = tiktoken
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .count();
    let tok_manifest = manifest(
        "yue2_synthetic_tiktoken",
        &tok_files,
        &tok_files[0],
        json!({"tokenizer": {"ordinary_tokens": lines}}),
    );
    let lm: &'static Component = Box::leak(Box::new(Component {
        id: ComponentId::Lm,
        key: "yue2_synthetic_tier",
        repo: REPO,
        files: Box::leak(lm_files.into_boxed_slice()),
        manifest_json: lm_manifest,
    }));
    let tok: &'static Component = Box::leak(Box::new(Component {
        id: ComponentId::QwenTiktoken,
        key: "yue2_synthetic_tiktoken",
        repo: REPO,
        files: Box::leak(tok_files.into_boxed_slice()),
        manifest_json: tok_manifest,
    }));
    Source {
        dirs: SnapshotDirs::new().with(REPO.id, dir),
        tmp,
        source: TierSource {
            lm,
            tok,
            pins: None,
        },
        tensors,
    }
}

fn convert_to(src: &Source, tier: Tier, name: &str) -> PathBuf {
    let out = src.out(name);
    convert_from(src.source, &src.dirs, tier, &out).unwrap_or_else(|e| panic!("{tier}: {e}"));
    out
}

fn load(src: &Source, dir: &Path) -> Yue2Nar {
    let verified = verify_tier_from(src.source, dir).unwrap();
    let opened = crate::model::open_tier(&verified, DType::F32, &Device::Cpu).unwrap();
    Yue2Nar::from_opened(opened).unwrap()
}

/// The BF16 original as an F32 model — the reference every tier is compared with.
fn original(src: &Source) -> Yue2Nar {
    let vb = candle_nn::VarBuilder::from_tensors(src.tensors.clone(), DType::F32, &Device::Cpu);
    Yue2Nar::from_var_builder(mot::config(), heads::nar_config(1.0), vb, "original".into()).unwrap()
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

/// The same original and conversion code give byte-identical tiers (weights **and** manifest),
/// the tier verifies, every copied file is the original's byte for byte, every BF16 tensor keeps
/// its pinned digest, and every matmul weight — and only the matmul weights — is GGML at the
/// tier's block type (Q4_K where the input width is 256-aligned, Q4_0 otherwise).
///
/// Mutation run: a wall-clock field in the manifest fails the reproducibility check. A tier that
/// kept `lm_head` or a NAR head dense, or a Q4 that never took the Q4_K path, fails the storage
/// assertions.
#[test]
fn conversion_is_deterministic_and_verifies() {
    let src = source();
    for tier in [Tier::Q8, Tier::Q4] {
        let a = convert_to(&src, tier, &format!("{tier}-a"));
        let b = convert_to(&src, tier, &format!("{tier}-b"));
        for file in [WEIGHTS_FILE, TIER_MANIFEST] {
            assert_eq!(
                std::fs::read(a.join(file)).unwrap(),
                std::fs::read(b.join(file)).unwrap(),
                "{tier}: {file} is not reproducible"
            );
        }
        let v = verify_tier_from(src.source, &a).unwrap();
        assert_eq!(v.tier(), tier);
        let src_dir = src.dirs.snapshot_dir(&REPO).unwrap();
        for pinned in src.source.copied_files() {
            assert_eq!(
                std::fs::read(a.join(pinned.path)).unwrap(),
                std::fs::read(src_dir.join(pinned.path)).unwrap(),
                "{}",
                pinned.path
            );
        }
        let mut seen_q4k = false;
        for t in v.plan() {
            let want = match (tier, t.class.follows_tier()) {
                (_, false) => Storage::Bf16,
                (Tier::Q8, true) => Storage::Ggml(GgmlDType::Q8_0),
                (_, true) if t.logical_shape[1] % 256 == 0 => {
                    seen_q4k = true;
                    Storage::Ggml(GgmlDType::Q4K)
                }
                (_, true) => Storage::Ggml(GgmlDType::Q4_0),
            };
            assert_eq!(t.storage, want, "{tier} {}", t.name);
        }
        assert_eq!(
            seen_q4k,
            tier == Tier::Q4,
            "time_embedder.mlp.0 is 256 wide"
        );
        let pinned: HashMap<String, String> = src
            .source
            .lm
            .conversion_manifest()
            .unwrap()
            .tensors
            .into_iter()
            .map(|t| (t.name, t.sha256))
            .collect();
        for e in v.manifest()["tensors"].as_array().unwrap() {
            let name = e["name"].as_str().unwrap();
            let original = pinned.get(name).map(String::as_str);
            if e["storage"] == "bf16" {
                assert_eq!(e["sha256"].as_str(), original, "{name}");
            } else {
                assert_ne!(e["sha256"].as_str(), original, "{name}");
            }
        }
    }
}

/// Relative L2 and top-1 agreement of teacher-forced logits (a prefill and six decode steps).
fn logit_fidelity(tier: &Yue2Nar, reference: &Yue2Nar) -> (f64, f64) {
    let seq: Vec<u32> = vec![
        151643, 40, 1234, 99, 151847, 5, 777, 151848, 151851, 160000, 170000, 152000,
    ];
    let rows = |m: &Yue2Nar| -> Vec<Vec<f32>> {
        let mut cache = m.lm().new_cache(seq.len()).unwrap();
        let mut out = vec![m
            .lm()
            .prefill(&seq[..6], &mut cache, || Ok(()))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()];
        for &t in &seq[6..] {
            out.push(m.lm().decode(t, &mut cache).unwrap().to_vec1().unwrap());
        }
        out
    };
    let (a, b) = (rows(tier), rows(reference));
    let (mut num, mut den, mut agree) = (0f64, 0f64, 0usize);
    let argmax = |r: &[f32]| {
        r.iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .unwrap()
            .0
    };
    for (x, y) in a.iter().zip(&b) {
        for (p, q) in x.iter().zip(y) {
            num += ((p - q) as f64).powi(2);
            den += (*q as f64).powi(2);
        }
        agree += usize::from(argmax(x) == argmax(y));
    }
    ((num / den).sqrt(), agree as f64 / a.len() as f64)
}

fn latent_fidelity(tier: &mut Yue2Nar, reference: &mut Yue2Nar) -> f64 {
    let noise = crate::nar::SongNoise::seeded(7, 24);
    let codes: Vec<u32> = (0..24).map(|i| (i * 977) % 32768).collect();
    let request = crate::nar::SynthesisRequest {
        prefix: &[151643, 40, 1234, 99, 151847, 5],
        codes: &codes,
        noise: &noise,
        steps: 3,
        context: crate::protocol::CONTEXT,
    };
    let run = |m: &mut Yue2Nar| {
        crate::nar::synthesize(
            m,
            &request,
            &crate::nar::NarOptions::default(),
            crate::nar::SynthesisHooks {
                cancelled: &|| false,
                observer: &mut (),
            },
        )
        .unwrap()
        .latents
        .values()
        .to_vec()
    };
    let (x, y) = (run(tier), run(reference));
    (x.iter()
        .zip(&y)
        .map(|(p, q)| ((p - q) as f64).powi(2))
        .sum::<f64>()
        / y.iter().map(|q| (*q as f64).powi(2)).sum::<f64>())
    .sqrt()
}

/// A tier loads through the verified loader at exactly its precision map — every AR and NAR
/// projection, `lm_head` and every NAR head GGML; the embedding, norms and position table dense —
/// its measured residency equals the weights-free price of its plan, and its outputs stay close to
/// the original (q8 closer than q4).
///
/// Bounds: see `SYNTHETIC_BOUNDS`. A loader that read a GGML tensor densely, a transposed or
/// mis-blocked conversion, or a tier-unaware loader fails the labels or the bounds.
#[test]
fn a_tier_loads_at_its_precision_map_and_stays_close_to_the_original() {
    let src = source();
    let reference = original(&src);
    let mut previous = (0.0, 0.0);
    for (tier, (logit_bound, latent_bound)) in
        [Tier::Q8, Tier::Q4].into_iter().zip(SYNTHETIC_BOUNDS)
    {
        let dir = convert_to(&src, tier, tier.name());
        let mut model = load(&src, &dir);
        assert_eq!(model.lm().tier(), tier);
        let plan = plan_from(src.source, tier).unwrap();
        let storage: HashMap<&str, Storage> =
            plan.iter().map(|t| (t.name.as_str(), t.storage)).collect();
        for (i, layer) in model.lm().layers().iter().enumerate() {
            for (path, prefix) in [
                (&layer.ar, ["self_attn", "mlp"]),
                (layer.nar.as_ref().unwrap(), ["nar_self_attn", "nar_mlp"]),
            ] {
                let names = ["q_proj", "k_proj", "v_proj", "o_proj"]
                    .map(|p| format!("model.layers.{i}.{}.{p}.weight", prefix[0]))
                    .into_iter()
                    .chain(
                        ["gate_proj", "up_proj", "down_proj"]
                            .map(|p| format!("model.layers.{i}.{}.{p}.weight", prefix[1])),
                    );
                for (p, name) in path.projections().into_iter().zip(names) {
                    assert_eq!(p.label(), storage[name.as_str()].label(), "{tier} {name}");
                }
            }
        }
        assert_eq!(
            model.lm().lm_head().label(),
            storage["lm_head.weight"].label()
        );
        assert_eq!(
            model.weight_residency(),
            crate::precision::weight_residency(&plan, Backend::Cpu, DType::F32, false),
            "{tier}: measured residency vs the plan's price"
        );
        assert!(plan
            .iter()
            .filter(|t| matches!(
                t.class,
                TensorClass::TokenEmbedding | TensorClass::Norm | TensorClass::LatentPositions
            ))
            .all(|t| t.storage == Storage::Bf16));

        let (rel, top1) = logit_fidelity(&model, &reference);
        let latent_rel = latent_fidelity(&mut model, &mut original(&src));
        println!(
            "{tier}: logits rel L2 {rel:.3e}, top-1 {top1:.2}; latents rel L2 {latent_rel:.3e}"
        );
        assert!(rel < logit_bound, "{tier}: logits rel L2 {rel}");
        assert!(
            latent_rel < latent_bound,
            "{tier}: latents rel L2 {latent_rel}"
        );
        assert!(
            rel > previous.0 && latent_rel > previous.1,
            "q4 is coarser than q8"
        );
        previous = (rel, latent_rel);
    }
}

/// `(logits rel L2, latents rel L2)` bounds of the synthetic q8 and q4 tiers against the BF16
/// original, CPU F32 compute. Measured 2026-09-26 (Apple M-series CPU): q8 1.8e-2 / 7.7e-3, q4
/// 1.5e-1 / 9.7e-2 — the hashed synthetic weights are uniform (no outlier structure) and only 32
/// wide, and Candle's CPU quantized matmul also quantizes the activation to 8-bit blocks, so these
/// are far coarser than the real checkpoint's (see [`crate::precision`]). Bounded at ≈3× the
/// measurement for another CPU's quantized dot kernel. Mutation run: dropping the bias of the GGML
/// NAR-head projections moves the q8 latents to 8.5e-2 (over 3× its bound).
const SYNTHETIC_BOUNDS: [(f64, f64); 2] = [(5e-2, 2.5e-2), (4.5e-1, 3e-1)];

fn expect_refused(src: &Source, dir: &Path, what: &str) -> AssetError {
    match verify_tier_from(src.source, dir) {
        Ok(_) => panic!("{what}: a tampered tier verified"),
        Err(e) => e,
    }
}

fn edit_manifest(dir: &Path, edit: impl FnOnce(&mut Value)) {
    let path = dir.join(TIER_MANIFEST);
    let mut m: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    edit(&mut m);
    std::fs::write(&path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
}

/// Every way a tier can be wrong is refused: a changed copied file (checked against the inventory
/// pins, not the manifest), a changed weights byte, a stray weights file, and a manifest that lies
/// about the source, the conversion, the tier, a tensor's storage, or a BF16 tensor's digest.
///
/// Mutations run: skipping the copied-file pin check, and skipping the pinned-digest check of BF16
/// tensors, each fails this test.
#[test]
fn a_tampered_tier_is_refused() {
    let src = source();
    let good = convert_to(&src, Tier::Q8, "good");
    let fresh = |name: &str| {
        let d = src.out(name);
        copy_dir(&good, &d);
        assert!(verify_tier_from(src.source, &d).is_ok());
        d
    };

    let d = fresh("license");
    std::fs::write(d.join("LICENSE"), b"relicensed\n").unwrap();
    assert!(matches!(
        expect_refused(&src, &d, "license"),
        AssetError::SizeMismatch { .. } | AssetError::HashMismatch { .. }
    ));

    let d = fresh("weights");
    let mut bytes = std::fs::read(d.join(WEIGHTS_FILE)).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(d.join(WEIGHTS_FILE), &bytes).unwrap();
    assert!(matches!(
        expect_refused(&src, &d, "weights"),
        AssetError::HashMismatch { .. }
    ));

    let d = fresh("stray");
    std::fs::write(d.join("extra.safetensors"), b"x").unwrap();
    assert!(matches!(
        expect_refused(&src, &d, "stray"),
        AssetError::UnexpectedWeightFile { .. }
    ));

    let d = fresh("source");
    edit_manifest(&d, |m| m["source"]["revision"] = json!("0".repeat(40)));
    assert!(matches!(
        expect_refused(&src, &d, "source"),
        AssetError::Manifest { .. }
    ));

    let d = fresh("conversion");
    edit_manifest(&d, |m| m["conversion"]["id"] = json!("yue2-ggml-tier-v0"));
    let e = expect_refused(&src, &d, "conversion");
    assert!(e.to_string().contains("re-derive"), "{e}");

    let d = fresh("tier");
    edit_manifest(&d, |m| m["tier"] = json!("q4"));
    assert!(matches!(
        expect_refused(&src, &d, "tier"),
        AssetError::Manifest { .. }
    ));

    let d = fresh("bf16-tier");
    edit_manifest(&d, |m| m["tier"] = json!("bf16"));
    assert!(expect_refused(&src, &d, "bf16-tier")
        .to_string()
        .contains("not a derived tier"));

    let d = fresh("storage");
    edit_manifest(&d, |m| {
        let row = m["tensors"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|r| r["name"] == "lm_head.weight")
            .unwrap();
        row["storage"] = json!("bf16");
    });
    assert!(matches!(
        expect_refused(&src, &d, "storage"),
        AssetError::Manifest { .. }
    ));

    let d = fresh("digest");
    edit_manifest(&d, |m| {
        let row = m["tensors"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|r| r["name"] == "model.embed_tokens.weight")
            .unwrap();
        row["sha256"] = json!("0".repeat(64));
    });
    assert!(expect_refused(&src, &d, "digest")
        .to_string()
        .contains("not the pinned original"));
}

/// A tier must be derived: bf16 is refused by name, a non-empty destination is refused, and a
/// failed conversion (here: the original is not provisioned) leaves neither the destination nor a
/// working directory behind.
#[test]
fn conversion_refusals_leave_nothing_behind() {
    let src = source();
    let e = convert_from(src.source, &src.dirs, Tier::Bf16, &src.out("bf16")).unwrap_err();
    assert!(matches!(e, RunError::Invalid(_)), "{e}");
    let busy = src.out("busy");
    std::fs::create_dir_all(&busy).unwrap();
    std::fs::write(busy.join("x"), b"x").unwrap();
    assert!(matches!(
        convert_from(src.source, &src.dirs, Tier::Q8, &busy),
        Err(RunError::Exists(_))
    ));
    let missing = src.out("missing");
    assert!(convert_from(src.source, &SnapshotDirs::new(), Tier::Q8, &missing).is_err());
    assert!(!missing.exists() && !partial_dir(&missing).exists());
}

/// An asserted tier is checked against the staged snapshot before anything is verified or read:
/// `q8` over the BF16 original names the local derivation, and a mismatched derived tier is
/// refused too.
#[test]
fn an_asserted_tier_must_be_the_staged_one() {
    let tmp = tempfile::tempdir().unwrap();
    let dirs = SnapshotDirs::new().with(ComponentId::Lm.component().repo.id, tmp.path());
    let err = crate::model::open_verified(&dirs, DType::F32, &Device::Cpu, Some(Tier::Q8))
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, gen_core::Error::Unsupported(_)), "{err}");
    assert!(
        err.to_string().contains("derive the tier snapshot"),
        "{err}"
    );
    // Without an assertion the original is verified (here: found incomplete), not refused by tier.
    let err = crate::model::open_verified(&dirs, DType::F32, &Device::Cpu, None)
        .map(|_| ())
        .unwrap_err();
    assert!(!matches!(err, gen_core::Error::Unsupported(_)), "{err}");

    let src = source();
    let dir = convert_to(&src, Tier::Q8, "q8");
    let staged = verify_tier_from(src.source, &dir).unwrap().tier();
    assert!(crate::model::check_tier(Some(Tier::Q4), staged, &dir).is_err());
    assert!(crate::model::check_tier(Some(Tier::Q8), staged, &dir).is_ok());
    assert!(crate::model::check_tier(None, staged, &dir).is_ok());
}

/// Deriving a tier is authorized (local noncommercial experimentation, the basis every conversion
/// checks); rehosting one is redistribution of the MoT and its tokenizer, which has no recorded
/// basis — and the tier's manifest says so.
#[test]
fn rehosting_a_tier_is_not_authorized() {
    use crate::license::{authorize, IntendedUse};
    let tier_components = [ComponentId::Lm, ComponentId::QwenTiktoken];
    assert!(authorize(&tier_components, IntendedUse::NoncommercialExperimentation).is_ok());
    let refused = authorize(&tier_components, IntendedUse::Redistribution).unwrap_err();
    let why = refused.to_string();
    assert!(
        why.contains("Lm:") && why.contains("QwenTiktoken:"),
        "{why}"
    );
    let src = source();
    let dir = convert_to(&src, Tier::Q8, "q8");
    let note = read_manifest(&dir).unwrap()["license"]["note"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(note.contains("not authorized"), "{note}");
}

/// A tier must reproduce its source's pinned conversion output: the synthetic source's own
/// outputs verify once pinned, a different pinned digest is a hash mismatch, and a source that
/// pins no output for the tier refuses it. The pinned YuE2-3B original pins both derived tiers.
///
/// Mutation run: skipping the pin comparison fails the `wrong` case.
#[test]
fn pinned_outputs_are_enforced() {
    let src = source();
    let dir = convert_to(&src, Tier::Q8, "q8");
    let rec = &read_manifest(&dir).unwrap()["files"][WEIGHTS_FILE];
    let (sha, bytes) = (
        leak_str(rec["sha256"].as_str().unwrap().to_string()),
        rec["bytes"].as_u64().unwrap(),
    );
    let with = |pins: Vec<TierPin>| TierSource {
        pins: Some(Box::leak(pins.into_boxed_slice())),
        ..src.source
    };
    let right = with(vec![TierPin {
        tier: Tier::Q8,
        bytes,
        sha256: sha,
    }]);
    assert!(verify_tier_from(right, &dir).is_ok());
    let wrong = with(vec![TierPin {
        tier: Tier::Q8,
        bytes,
        sha256: leak_str("0".repeat(64)),
    }]);
    assert!(matches!(
        verify_tier_from(wrong, &dir),
        Err(AssetError::HashMismatch { .. })
    ));
    let none = with(vec![TierPin {
        tier: Tier::Q4,
        bytes,
        sha256: sha,
    }]);
    assert!(verify_tier_from(none, &dir)
        .unwrap_err()
        .to_string()
        .contains("no pinned q8 output"));
    let pinned: Vec<Tier> = TIER_PINS.iter().map(|p| p.tier).collect();
    assert_eq!(pinned, [Tier::Q8, Tier::Q4]);
    assert_eq!(TierSource::pinned().pins, Some(TIER_PINS));
}

/// A saved closure carries a derived tier byte for byte, and the copy verifies.
#[test]
fn a_tier_is_saved_into_a_closure_byte_for_byte() {
    let src = source();
    let dir = convert_to(&src, Tier::Q4, "q4");
    let root = src.out("closure/YuE2-3B");
    let identity = crate::closure::copy_tier_from(src.source, &dir, &root).unwrap();
    assert_eq!(identity["tier"], "q4");
    let copy = verify_tier_from(src.source, &root).unwrap();
    assert_eq!(
        copy.weights_sha256(),
        verify_tier_from(src.source, &dir).unwrap().weights_sha256()
    );
    std::fs::write(root.join("README.md"), b"changed").unwrap();
    assert!(verify_tier_from(src.source, &root).is_err());
}

fn set_tree_read_only(dir: &Path) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            set_tree_read_only(&path);
        } else {
            let mut perms = std::fs::metadata(&path).unwrap().permissions();
            perms.set_readonly(true);
            std::fs::set_permissions(&path, perms).unwrap();
        }
    }
}

/// Saving a closure whose MoT is a derived tier — every file read-only, as a hub cache holds it —
/// copies the tier once (the tokenizer included, verified by the tier's own check against its
/// pin), never resolves or copies the MoT or `qwen.tiktoken` components beside it, syncs the
/// read-only copies, and the saved tier verifies.
///
/// Mutations run: resolving the `QwenTiktoken` component again when a tier is staged fails this
/// test (the resolver refuses it), and so did the write-handle `sync_file` this story first shipped
/// (the read-only copies could not be synced).
#[test]
fn a_closure_with_a_read_only_tier_copies_the_tokenizer_once() {
    use crate::inventory::VaeVariant;
    use crate::snapshot::verify_component;

    let src = source();
    let dir = convert_to(&src, Tier::Q8, "q8");
    set_tree_read_only(&dir);
    let vae_dir = src.out("vae");
    let vae = crate::snapshot::tests::synthetic_snapshot(
        &vae_dir,
        ComponentId::VaeStandard,
        "yue2_synth_tier_vae",
        UpstreamRepo {
            id: "m-a-p/YuE2-Synthetic-Tier-Vae",
            ..REPO
        },
        &[],
    );
    let resolve = |id: ComponentId| match id {
        ComponentId::VaeStandard => verify_component(vae, &vae_dir),
        other => panic!("{other:?} must not be resolved: the staged tier carries it"),
    };
    let dest = src.out("saved");
    let metadata = crate::closure::save_resolved(
        &resolve,
        Some((&dir, src.source)),
        &[VaeVariant::Standard],
        &crate::protocol::GenerationConfig::default(),
        &dest,
    )
    .unwrap();
    let saved = dest.join("YuE2-Synthetic-Tier");
    assert_eq!(
        verify_tier_from(src.source, &saved).unwrap().tier(),
        Tier::Q8
    );
    assert_eq!(
        metadata["source_weights"]["yue2_synthetic_tier"]["tier"]["tier"],
        "q8"
    );
    assert!(metadata["source_weights"]
        .get("yue2_synthetic_tiktoken")
        .is_some());
}
