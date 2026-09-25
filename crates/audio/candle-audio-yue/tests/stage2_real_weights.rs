//! Real-weight stage-2 parity (sc-19381, epic sc-19373 R9) against
//! `tests/fixtures/stage2_real_reference.json` — upstream YuE-v1's own `stage2_inference` run on
//! `m-a-p/YuE-s2-1B-general` (bf16 weights, f32 compute, CPU) over xcodec-encoded codebook-0
//! inputs (producer: `scripts/reference/yue_stage2_reference.py real`; see the fixture README).
//!
//! Gated like the other real-weight tests: `#[ignore]`d in ordinary runs, and under `--ignored`
//! the snapshot variable is **required** (unset panics; it never passes silently).
//!
//! ```text
//! YUE_S2_SNAPSHOT=/path/yue-s2-1b-general-candle \
//!   cargo test --release -p candle-audio-yue --test stage2_real_weights -- --ignored --nocapture
//! ```
//!
//! `YUE_S2_SNAPSHOT` is the tiered SceneWorks rehost root (`bf16/`, `q8/`, `q4/`). Measured peak
//! RSS on an M-series Mac: the exact-parity test ~15 GB (a 2-row, 2702-position f32 KV cache
//! beside the f32 decoder), the all-tier test ~22 GB.

use candle_audio_yue::candle_audio::candle_core::Device;
use candle_audio_yue::config::{Assets, Tier};
use candle_audio_yue::gen_core::CancelFlag;
use candle_audio_yue::snapshot::resolve_tier_dir;
use candle_audio_yue::stage2::{
    self, characterise, upsample_with, CandleStage2Lm, CHUNK_FRAMES, DEFAULT_BATCH_SIZE,
};
use candle_llm::{CausalLm, ModelConfig};

struct Case {
    name: String,
    cb0: Vec<u32>,
    codebooks: Vec<Vec<u32>>,
    repaired: usize,
}

fn golden() -> Vec<Case> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stage2_real_reference.json");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(v["checkpoint"]["compute_dtype"], "float32");
    let u32s = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_u64().unwrap() as u32)
            .collect()
    };
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            name: c["name"].as_str().unwrap().into(),
            cb0: u32s(&c["cb0"]),
            codebooks: c["codebooks"]
                .as_array()
                .unwrap()
                .iter()
                .map(u32s)
                .collect(),
            repaired: c["repaired_codes"].as_u64().unwrap() as usize,
        })
        .collect()
}

fn snapshot_root() -> std::path::PathBuf {
    std::env::var_os("YUE_S2_SNAPSHOT")
        .expect("set YUE_S2_SNAPSHOT to the tiered yue-s2-1b-general-candle snapshot root")
        .into()
}

/// The dense tier on an explicit device, at candle-llm's compute dtype for it (f32 on the CPU).
fn dense_lm(device: &Device) -> CandleStage2Lm {
    let (dir, tier) = resolve_tier_dir(&snapshot_root(), Some(Tier::Bf16), "stage-2").unwrap();
    assert_eq!(tier, Tier::Bf16);
    let weights = candle_llm::primitives::Weights::from_dir(&dir, device).unwrap();
    let cfg = ModelConfig::from_dir(&dir).unwrap();
    CandleStage2Lm::from_model(CausalLm::from_weights(&weights, "", cfg).unwrap())
}

/// Frame and codebook of the first difference, for a readable failure.
fn first_difference(got: &[Vec<u32>], want: &[Vec<u32>]) -> Option<(usize, usize, u32, u32)> {
    (0..want[0].len()).find_map(|t| {
        (0..want.len())
            .find_map(|k| (got[k][t] != want[k][t]).then(|| (t, k, got[k][t], want[k][t])))
    })
}

/// **AC1** — the dense tier on the CPU reproduces every code of all 8 codebooks of the upstream
/// reference, and the same number of repaired codes.
#[test]
#[ignore = "needs the YuE stage-2 snapshot via YUE_S2_SNAPSHOT"]
fn stage2_bf16_cpu_matches_the_upstream_reference_exactly() {
    let mut lm = dense_lm(&Device::Cpu);
    for case in golden() {
        let (grid, repaired) =
            upsample_with(&mut lm, &case.cb0, DEFAULT_BATCH_SIZE, &CancelFlag::new()).unwrap();
        grid.check_against(&case.cb0).unwrap();
        if let Some((t, k, got, want)) = first_difference(&grid.codebooks, &case.codebooks) {
            panic!(
                "{}: frame {t} codebook {k}: got {got}, reference {want}",
                case.name
            );
        }
        assert_eq!(repaired, case.repaired, "{}", case.name);
        eprintln!(
            "{}: {} frames x 8 codebooks identical",
            case.name,
            case.cb0.len()
        );
    }
}

/// The production loader over every published tier (`stage2::load`: tier resolution →
/// `LlamaProvider::load`), each characterised against the dense CPU reference: the quantized
/// projections move the logits, so their residual picks are reported, not asserted equal.
#[test]
#[ignore = "needs the YuE stage-2 snapshot via YUE_S2_SNAPSHOT"]
fn stage2_every_tier_loads_through_production_and_is_characterised() {
    let root = snapshot_root();
    let assets = Assets {
        stage1: root.join("absent-stage1"),
        stage2: root.clone(),
        xcodec: root.join("absent-xcodec"),
    };
    // The single-chunk cases only: this test characterises three tiers, and the 650-frame case's
    // batched schedule is already pinned exactly by the dense parity test above.
    let cases: Vec<Case> = golden()
        .into_iter()
        .filter(|c| c.cb0.len() <= CHUNK_FRAMES)
        .collect();
    let mut reference = dense_lm(&Device::Cpu);
    for tier in [Tier::Bf16, Tier::Q8, Tier::Q4] {
        let mut stage2 = stage2::load(&assets, Some(tier)).unwrap();
        for case in &cases {
            let grid = stage2.upsample(&case.cb0, &CancelFlag::new()).unwrap();
            grid.check_against(&case.cb0).unwrap();
            let same: usize = (1..8)
                .map(|k| {
                    grid.codebooks[k]
                        .iter()
                        .zip(&case.codebooks[k])
                        .filter(|(a, b)| a == b)
                        .count()
                })
                .sum();
            eprintln!(
                "{tier:?} {}: {same} of {} residual codes equal the reference",
                case.name,
                7 * case.cb0.len()
            );
        }
        drop(stage2);
        let (dir, _) = resolve_tier_dir(&root, Some(tier), "stage-2").unwrap();
        let mut lm = CandleStage2Lm::load(&dir).unwrap();
        let d = characterise(&mut reference, &mut lm, &cases[0].cb0).unwrap();
        eprintln!(
            "{tier:?} vs dense f32 (teacher-forced on the reference stream): {} of {} picks \
             differ; max |Δlogit| {:.3} at scale {:.3}",
            d.mismatches.len(),
            d.positions,
            d.max_abs_logit_delta,
            d.max_abs_logit
        );
        assert_eq!(d.positions, 7 * cases[0].cb0.len());
    }
}

/// Bound on `max |Δlogit| / max |logit|` for the dense 1B on Metal (bf16 compute) against the CPU
/// (f32). PROVISIONAL until the first Metal run (sc-19381): set from the measured value with
/// headroom. For scale, the CPU q8 tier measured 0.028 against the same reference.
#[cfg(feature = "metal")]
const METAL_MAX_REL_LOGIT_DELTA: f32 = 0.05;
/// Bound on the fraction of residual picks that flip on Metal, teacher-forced on the reference
/// stream (so flips cannot cascade). PROVISIONAL until the first Metal run (sc-19381).
#[cfg(feature = "metal")]
const METAL_MAX_FLIP_FRACTION: f64 = 0.05;

/// **AC3 on real weights** — the dense tier on Metal (bf16 compute) against the CPU f32
/// reference, teacher-forced along the reference stream, per golden case (a case longer than one
/// chunk contributes its first full 300-frame chunk — the full per-chunk context). The flips and
/// the logit noise are reported and bounded by [`METAL_MAX_REL_LOGIT_DELTA`] /
/// [`METAL_MAX_FLIP_FRACTION`]; the CPU stream itself must still be the golden.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the YuE stage-2 snapshot via YUE_S2_SNAPSHOT and a Metal device"]
fn stage2_metal_bf16_divergence_is_characterised_on_real_weights() {
    let metal = Device::new_metal(0).expect("the metal feature requires a Metal device");
    let mut reference = dense_lm(&Device::Cpu);
    let mut gpu = dense_lm(&metal);
    for case in golden() {
        let frames = case.cb0.len().min(CHUNK_FRAMES);
        let d = characterise(&mut reference, &mut gpu, &case.cb0[..frames]).unwrap();
        eprintln!(
            "{} ({frames} frames): metal bf16 flips {} of {} residual picks ({:.4}); max |Δlogit| \
             {:.3} at scale {:.3} (relative {:.4}); flip margins {:?}",
            case.name,
            d.mismatches.len(),
            d.positions,
            d.flip_fraction(),
            d.max_abs_logit_delta,
            d.max_abs_logit,
            d.relative_logit_delta(),
            d.mismatches
                .iter()
                .map(|m| m.reference_margin)
                .collect::<Vec<_>>()
        );
        if case.repaired == 0 {
            let want: Vec<u32> = (0..frames)
                .flat_map(|t| (0..8).map(move |k| (t, k)))
                .map(|(t, k)| 45_334 + 1_024 * k as u32 + case.codebooks[k][t])
                .collect();
            assert_eq!(
                d.reference_ids, want,
                "{}: the CPU stream is the golden",
                case.name
            );
        }
        assert!(
            d.relative_logit_delta() <= METAL_MAX_REL_LOGIT_DELTA,
            "{}: relative logit delta {} exceeds {METAL_MAX_REL_LOGIT_DELTA}",
            case.name,
            d.relative_logit_delta()
        );
        assert!(
            d.flip_fraction() <= METAL_MAX_FLIP_FRACTION,
            "{}: flip fraction {} exceeds {METAL_MAX_FLIP_FRACTION}",
            case.name,
            d.flip_fraction()
        );
    }
}
